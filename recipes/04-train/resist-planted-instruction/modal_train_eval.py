"""Generate from the base, train the LoRA, run the three-arm eval. On Modal.

Nothing here deploys an endpoint. vLLM starts inside the container on
127.0.0.1 with a per-run key and stops when the function returns. Generation
and the eval sit on an H100 because the workload is decode throughput and
the bigger card is cheaper per generated token; the LoRA sits on an L40S.

    modal run modal_train_eval.py::run_generate --repeats 2 --seed 11 --budget 1500
    modal run modal_train_eval.py::run_train --rows-file out/sft_rows.jsonl
    modal run modal_train_eval.py::run_train --rows-file out/sft_rows_random.jsonl \\
        --name planted-instruction-random-control --epochs 1.0608
    modal run modal_train_eval.py::run_eval --repeats 4
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

app = modal.App("resist-planted-instruction")

BASE_MODEL = "Qwen/Qwen3-4B"
RUN_NAME = "planted-instruction-v1"
CONTROL_NAME = "planted-instruction-random-control"
MAX_MODEL_LEN = 32768

TRAIN_WORLD = {"n": 700, "seed": 1717, "prefix": 40000, "attack_share": 0.8}
HOLDOUT_WORLD = {"n": 120, "seed": 9091, "prefix": 58000, "attack_share": 0.8}

SERVE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.0", "transformers==4.54.0", "requests>=2.25", "whileai==0.62")
    .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_USE_V1": "1", "ZP_CONTEXT_TOKENS": "32768"})
    .add_local_dir(str(HERE), remote_path="/root/recipe")
)

TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

adapters = modal.Volume.from_name("resist-planted-adapters", create_if_missing=True)
hf_cache = modal.Volume.from_name("resist-planted-hf-cache", create_if_missing=True)
VOL = "/vol"


def _recipe():
    import sys

    sys.path.insert(0, "/root/recipe")
    from rubric import criterion_failures, make_grader
    from world import POLICY, TEACHER_SCAFFOLD, TOOLS, build_world, make_execute

    import whileai.simulations as wai

    return (
        wai,
        criterion_failures,
        make_grader,
        POLICY,
        TEACHER_SCAFFOLD,
        TOOLS,
        build_world,
        make_execute,
    )


def _start_vllm(lora_modules: dict[str, str], rank: int):
    """One vLLM process serving the base and every adapter passed in."""
    import subprocess
    import sys
    import time

    import requests

    key = "recipe-local-key"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        BASE_MODEL,
        "--served-model-name",
        BASE_MODEL,
        "--port",
        "8000",
        "--api-key",
        key,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--gpu-memory-utilization",
        "0.88",
        "--disable-log-requests",
    ]
    if lora_modules:
        cmd += [
            "--enable-lora",
            "--max-lora-rank",
            str(rank),
            "--max-loras",
            str(len(lora_modules)),
            "--lora-modules",
            *[f"{name}={path}" for name, path in lora_modules.items()],
        ]
    proc = subprocess.Popen(cmd)
    for _ in range(200):
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with {proc.returncode} before serving")
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=3).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        proc.kill()
        raise RuntimeError("vllm did not come up")
    os.environ["VLLM_API_KEY"] = key
    return proc, "http://127.0.0.1:8000/v1"


def _stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill()


def _arm(model_name: str, url: str, world: dict, repeats: int, seed: int, budget: float) -> dict:
    """One arm over the pinned tasks, with a top-up pass for any prompt the
    engine dropped: the same arm, the same process, only the missing ones."""
    wai, criterion_failures, make_grader, POLICY, _, TOOLS, _, make_execute = _recipe()
    tasks = [{"prompt": sc["opener"], "scenario_id": sid} for sid, sc in world["scenarios"].items()]

    def roll(task_list: list[dict], run_seed: int, tb: float) -> list[dict]:
        data = wai.simulate(
            tools=TOOLS,
            system_prompt=POLICY,
            backend=f"vllm:{model_name}@{url}",
            # The customer is the same model on every arm. Left to default the
            # policy under test voices its own customer and the arms face
            # different environments.
            user_model=f"vllm:{BASE_MODEL}@{url}",
            simulator=f"vllm:{BASE_MODEL}@{url}",
            execute=make_execute(world),
            tasks=task_list,
            repeats=repeats,
            avg_turns=6.0,
            max_turns=30,
            concurrency=32,
            seed=run_seed,
            time_budget=tb,
            grade=False,
        )
        return list(data.trajectories)

    rows = roll(tasks, seed, budget)
    topups = []
    for attempt in range(3):
        seen = {str(r.get("scenario_id") or "") for r in rows}
        missing = [t for t in tasks if t["scenario_id"] not in seen]
        if not missing:
            break
        got = roll(missing, seed + 100 + attempt, 600.0)
        topups.append({"missing": [t["scenario_id"] for t in missing], "recovered_rows": len(got)})
        rows += got
    scored = wai.run_judge(rows, make_grader(world["scenarios"]), version="code@recipe")
    return {
        "model": model_name,
        "rows_jsonl": "\n".join(json.dumps(r, default=str) for r in scored.rows),
        "n_rows": len(scored.rows),
        "n_graded": sum(1 for r in scored.rows if isinstance(r.get("reward"), int | float)),
        "pass_at": scored.pass_at.to_dict(),
        "criterion_failures": criterion_failures(scored.rows),
        "topups": topups,
        "warnings": data_warnings(scored),
    }


def data_warnings(scored) -> list[str]:
    return [str(w) for w in (getattr(scored, "warnings", None) or [])]


@app.function(
    image=SERVE_IMAGE,
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=60,
)
def generate(repeats: int, seed: int, budget: int, time_budget: float) -> dict:
    """Rejection sampling from the base itself (ch. 9): sample, keep what the
    grader passes. The selection happens in run.py, on your machine."""
    wai, criterion_failures, make_grader, POLICY, SCAFFOLD, TOOLS, build_world, make_execute = (
        _recipe()
    )
    world = build_world(**TRAIN_WORLD)
    tasks = [{"prompt": sc["opener"], "scenario_id": sid} for sid, sc in world["scenarios"].items()]
    proc, url = _start_vllm({}, 16)
    try:
        data = wai.simulate(
            tools=TOOLS,
            system_prompt=POLICY,
            backend=f"vllm:{BASE_MODEL}@{url}",
            user_model=f"vllm:{BASE_MODEL}@{url}",
            simulator=f"vllm:{BASE_MODEL}@{url}",
            execute=make_execute(world),
            scaffold=SCAFFOLD,  # generation-only; never enters the exported policy
            tasks=tasks,
            repeats=repeats,
            budget=budget,
            avg_turns=6.0,
            max_turns=30,
            concurrency=32,
            seed=seed,
            time_budget=time_budget,
            grade=False,
        )
        scored = wai.run_judge(
            data.trajectories, make_grader(world["scenarios"]), version="code@recipe"
        )
    finally:
        _stop(proc)
    return {
        "rows_jsonl": "\n".join(json.dumps(r, default=str) for r in scored.rows),
        "n_rows": len(scored.rows),
        "pass_at": scored.pass_at.to_dict(),
        "criterion_failures": criterion_failures(scored.rows),
        "stopped_because": data.stopped_because,
    }


@app.function(
    image=TRAIN_IMAGE,
    gpu="L40S",
    timeout=3 * 60 * 60,
    volumes={VOL: adapters, "/root/.cache/huggingface": hf_cache},
    scaledown_window=60,
)
def train_lora(job: dict, rows: list[dict]) -> dict:
    """LoRA SFT with the loss on the assistant turn only.

    Each row is one unrolled turn: `messages` ending at the assistant turn
    that carries loss, plus the tool schemas. The prompt is rendered with the
    model's own chat template so training sees the string vLLM will build,
    and TRL masks it, which keeps the system prompt, every user turn and every
    tool result out of the loss (ch. 4; ch. 13 on never training on tool
    output).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    name = job["name"]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    rank = int(job.get("rank", 16))
    lora = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    samples, bad_prefix, skipped = [], 0, 0
    for row in rows:
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2 or msgs[-1].get("role") != "assistant":
            skipped += 1
            continue
        fixed = []
        for m in msgs:
            m = dict(m)
            m["content"] = m.get("content") or ""
            if not m.get("tool_calls"):
                m.pop("tool_calls", None)
            fixed.append(m)
        tools = row.get("tools")
        prefix = tok.apply_chat_template(
            fixed[:-1], tools=tools, tokenize=False, add_generation_prompt=True
        )
        full = tok.apply_chat_template(fixed, tools=tools, tokenize=False)
        if not full.startswith(prefix):
            bad_prefix += 1
            continue
        samples.append({"prompt": prefix, "completion": full[len(prefix) :]})
    print(f"rendered {len(samples)} samples; bad_prefix={bad_prefix} skipped={skipped}")
    ds = Dataset.from_list(samples)
    lr = float(job.get("lr", 2e-5))  # explicit: 2e-4 was measured as catastrophic
    epochs = float(job.get("epochs", 1.0))
    # batch 1 x accumulation 8: the loss upcasts a (batch, seq, vocab) logits
    # tensor, and batch 2 at 4096 tokens does not fit a 44 GiB card.
    bs, accum = 1, 8
    steps = max(1, int(len(ds) * epochs / (bs * accum)))
    print(f"samples={len(ds)} epochs={epochs} lr={lr:g} -> ~{steps} optimizer steps")
    cfg = SFTConfig(
        output_dir=f"{VOL}/{name}/checkpoints",
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        bf16=True,
        packing=False,
        max_length=int(job.get("max_len", 4096)),
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=17,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora
    )
    trainer.train()
    adapter_dir = f"{VOL}/{name}/adapter"
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    adapters.commit()
    return {
        "adapter": adapter_dir,
        "samples": len(ds),
        "steps": steps,
        "lr": lr,
        "epochs": epochs,
        "rank": rank,
        "loss": [h for h in trainer.state.log_history if "loss" in h],
    }


@app.function(
    image=SERVE_IMAGE,
    gpu="H100",
    timeout=4 * 60 * 60,
    volumes={VOL: adapters, "/root/.cache/huggingface": hf_cache},
    scaledown_window=60,
)
def evaluate(job: dict, repeats: int, time_budget: float) -> dict:
    """Base, the random-selection control and the trained adapter over the
    same pinned tasks from ONE vLLM process, so only the weights differ."""
    _, _, _, _, _, _, build_world, _ = _recipe()
    world = build_world(**HOLDOUT_WORLD)
    modules = {}
    for label, name in (("trained", job["name"]), ("random", job.get("control_name") or "")):
        path = f"{VOL}/{name}/adapter"
        if name and os.path.isdir(path):
            modules[label] = path
    proc, url = _start_vllm(modules, int(job.get("rank", 16)))
    out: dict = {"n_tasks": len(world["scenarios"]), "adapters": sorted(modules)}
    try:
        out["base"] = _arm(BASE_MODEL, url, world, repeats, 21, time_budget)
        for label in ("random", "trained"):
            if label in modules:
                out[label] = _arm(label, url, world, repeats, 21, time_budget)
    finally:
        _stop(proc)
    return out


def _write(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=1, default=str))
    print("wrote", OUT / name)


@app.local_entrypoint()
def run_generate(repeats: int = 2, seed: int = 11, budget: int = 1500, time_budget: float = 2700.0):
    out = generate.remote(repeats, seed, budget, time_budget)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pool_seed{seed}.jsonl").write_text(out.pop("rows_jsonl", ""))
    _write(f"generate_seed{seed}.json", out)
    print(json.dumps(out, indent=1, default=str)[:3000])


@app.local_entrypoint()
def run_train(
    rows_file: str, epochs: float = 1.0, lr: float = 2e-5, rank: int = 16, name: str = RUN_NAME
):
    rows = [json.loads(line) for line in Path(rows_file).read_text().splitlines() if line.strip()]
    out = train_lora.remote({"name": name, "epochs": epochs, "lr": lr, "rank": rank}, rows)
    _write(f"train_{name}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "loss"}, indent=1))


@app.local_entrypoint()
def run_eval(repeats: int = 4, time_budget: float = 5400.0, rank: int = 16):
    out = evaluate.remote(
        {"name": RUN_NAME, "control_name": CONTROL_NAME, "rank": rank}, repeats, time_budget
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for side in ("base", "random", "trained"):
        if side in out:
            (OUT / f"eval_{side}.jsonl").write_text(out[side].pop("rows_jsonl", ""))
    _write("eval.json", out)
    print(json.dumps(out, indent=1, default=str)[:4000])
    print("\nNext: python analyse.py --out out")
