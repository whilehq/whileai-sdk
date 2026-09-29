"""Train the two adapters at two more seeds, then score every arm on the frozen holdout. On Modal.

Nothing here deploys an endpoint. vLLM starts inside the container on
127.0.0.1 and stops when the function returns; every adapter is served from
that one process so only the weights differ between arms. The gates run on
the container's CPU inside ``execute=``. Each draw's rows are written to the
``robust-results`` volume as they finish, so a rerun skips what is done.

    modal run modal_train_eval.py::run_train --rows-file out/sft_rows.jsonl --name sft-reward-s18 --seed 18
    modal run modal_train_eval.py::run_train --rows-file out/sft_rows_random.jsonl \\
        --name sft-random-s18 --seed 18 --epochs 1.0608
    modal run modal_train_eval.py::run_pilot                 # 12 prompts, base and both gates
    modal run modal_train_eval.py::run_eval                  # every draw; skips the ones on the volume
    modal run modal_train_eval.py::run_fetch                 # pull the rows into out/
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "resist-planted-instruction"
OUT = HERE / "out"

app = modal.App("robust-refuse-or-comply")

BASE_MODEL = "Qwen/Qwen3-4B"
MAX_MODEL_LEN = 32768
RESULTS = "/results"
VOL = "/vol"

SERVE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.10.0",
        "transformers==4.54.0",
        "requests>=2.25",
        "whileai==0.126",
        "onnxruntime>=1.18",
        "tokenizers>=0.19",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_USE_V1": "1"})
    .add_local_dir(str(SIBLING), remote_path="/root/sibling")
    .add_local_dir(str(HERE), remote_path="/root/recipe", ignore=["out/*", "__pycache__"])
    .add_local_dir(str(OUT / "classifier"), remote_path="/root/recipe/out/classifier")
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

adapters = modal.Volume.from_name("robust-adapters", create_if_missing=True)
hf_cache = modal.Volume.from_name("robust-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("robust-results", create_if_missing=True)


def _recipe():
    import sys

    sys.path.insert(0, "/root/recipe")
    sys.path.append("/root/sibling")
    import gate
    import holdout
    from rubric import criterion_failures, make_grader
    from world import POLICY, TOOLS, make_execute

    import whileai as wai

    return wai, gate, holdout, criterion_failures, make_grader, POLICY, TOOLS, make_execute


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


def _draw(
    draw: dict, url: str, world: dict, tasks: list[dict], repeats: int, time_budget: float
) -> dict:
    """One arm over the pinned tasks, with a top-up pass for any prompt the
    engine dropped: the same arm, the same process, only the missing ones."""
    wai, gate, _, criterion_failures, make_grader, POLICY, TOOLS, make_execute = _recipe()

    def build_execute():
        execute = make_execute(world)
        mode = draw.get("gate")
        if not mode:
            return execute
        classifier = gate.Classifier() if mode == "minilm" else None
        return gate.wrap(execute, gate.make_gate(mode, world, classifier), mode)

    def roll(task_list: list[dict], run_seed: int, tb: float) -> list[dict]:
        data = wai.simulate(
            tools=TOOLS,
            system_prompt=POLICY,
            backend=f"vllm:{draw['model']}@{url}",
            # The customer is the same model on every arm. Left to default the
            # policy under test voices its own customer and the arms face
            # different environments.
            user_model=f"vllm:{BASE_MODEL}@{url}",
            simulator=f"vllm:{BASE_MODEL}@{url}",
            execute=build_execute(),
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

    rows = roll(tasks, draw["seed"], time_budget)
    topups = []
    for attempt in range(3):
        seen = {str(r.get("scenario_id") or "") for r in rows}
        missing = [t for t in tasks if t["scenario_id"] not in seen]
        if not missing:
            break
        got = roll(missing, draw["seed"] + 100 + attempt, 600.0)
        topups.append({"missing": [t["scenario_id"] for t in missing], "recovered_rows": len(got)})
        rows += got
    from whileai.simulations import run_judge

    scored = run_judge(rows, make_grader(world["scenarios"]), version="code@sibling")
    for r in scored.rows:
        r["arm"] = draw["arm"]
        r["draw"] = draw["label"]
    return {
        "draw": draw,
        "rows_jsonl": "\n".join(json.dumps(r, default=str) for r in scored.rows),
        "n_rows": len(scored.rows),
        "n_graded": sum(1 for r in scored.rows if isinstance(r.get("reward"), int | float)),
        "pass_at": scored.pass_at.to_dict(),
        "criterion_failures": criterion_failures(scored.rows),
        "topups": topups,
        "warnings": [str(w) for w in (getattr(scored, "warnings", None) or [])],
    }


@app.function(
    image=TRAIN_IMAGE,
    gpu="L40S",
    timeout=60 * 60,  # the recipe's budget: under sixty GPU minutes per trained arm
    volumes={VOL: adapters, "/root/.cache/huggingface": hf_cache},
    scaledown_window=600,
)
def train_lora(job: dict, rows: list[dict]) -> dict:
    """LoRA SFT with the loss on the assistant turn only; the sibling's
    trainer with the seed as a parameter.

    Each row is one unrolled turn: `messages` ending at the assistant turn
    that carries loss, plus the tool schemas. The prompt is rendered with the
    model's own chat template and TRL masks it, which keeps the system
    prompt, every user turn and every tool result out of the loss (Lambert
    2025, chapters *Instruction Tuning* and *Tool Use*).
    """
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
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
    bs, accum = 1, 8
    steps = max(1, int(len(ds) * epochs / (bs * accum)))
    seed = int(job.get("seed", 17))
    print(f"samples={len(ds)} epochs={epochs} lr={lr:g} seed={seed} -> ~{steps} optimizer steps")
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
        seed=seed,
        data_seed=seed,
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
        "seed": seed,
        "gpu": "L40S",
        "wall_seconds": round(time.time() - t0),
        "loss": [h for h in trainer.state.log_history if "loss" in h],
    }


@app.function(
    image=SERVE_IMAGE,
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={VOL: adapters, "/root/.cache/huggingface": hf_cache, RESULTS: results},
    scaledown_window=600,
)
def evaluate(draws: list[dict], repeats: int, time_budget: float, limit: int, rank: int) -> dict:
    """Every draw over the same pinned tasks from ONE vLLM process. A draw
    whose rows already sit on the results volume is skipped."""
    import time

    _, _, holdout, _, _, _, _, _ = _recipe()
    world = holdout.build_holdout()
    tasks = holdout.tasks_for(world)
    holdout.check_pin(tasks)  # the frozen test, or nothing is scored
    if limit:
        # a pilot: the first prompts of each half, never a result
        att = [t for t in tasks if t["half"] == "attack"][:limit]
        cln = [t for t in tasks if t["half"] == "clean"][:limit]
        tasks = att + cln
    modules: dict[str, str] = {}
    for d in draws:
        name = d["model"]
        if name != BASE_MODEL and os.path.isdir(f"{VOL}/{name}/adapter"):
            modules[name] = f"{VOL}/{name}/adapter"
    missing = sorted({d["model"] for d in draws} - set(modules) - {BASE_MODEL})
    if missing:
        raise RuntimeError(f"no adapter on the volume for {missing}")
    todo = [d for d in draws if not os.path.exists(f"{RESULTS}/{d['label']}.jsonl") or limit]
    out: dict = {"n_tasks": len(tasks), "adapters": sorted(modules), "draws": {}}
    if not todo:
        return out
    proc, url = _start_vllm(modules, rank)
    try:
        for d in todo:
            t0 = time.time()
            res = _draw(d, url, world, tasks, repeats, time_budget)
            res["wall_seconds"] = round(time.time() - t0)
            rows_jsonl = res.pop("rows_jsonl")
            if not limit:
                Path(f"{RESULTS}/{d['label']}.jsonl").write_text(rows_jsonl)
                Path(f"{RESULTS}/{d['label']}.json").write_text(json.dumps(res, default=str))
                results.commit()
            else:
                res["rows_jsonl"] = rows_jsonl
            out["draws"][d["label"]] = res
            print(
                f"{d['label']}: rows {res['n_rows']} graded {res['n_graded']} "
                f"pass@1 {res['pass_at'].get('pass_at_1')} in {res['wall_seconds']}s"
            )
    finally:
        _stop(proc)
    return out


# ----------------------------------------------------------- the arms


def plan() -> list[dict]:
    """Fifteen draws. Base and the two gates are re-drawn three times with
    different sampling seeds (the eval's own floor); the two trained arms
    are three training seeds each, one draw per seed."""
    draws = []
    for i, seed in enumerate((21, 22, 23), start=1):
        draws.append(
            {"arm": "base", "model": BASE_MODEL, "gate": None, "seed": seed, "label": f"base_d{i}"}
        )
        draws.append(
            {
                "arm": "gate-minilm",
                "model": BASE_MODEL,
                "gate": "minilm",
                "seed": seed,
                "label": f"gate-minilm_d{i}",
            }
        )
        draws.append(
            {
                "arm": "gate-oracle",
                "model": BASE_MODEL,
                "gate": "oracle",
                "seed": seed,
                "label": f"gate-oracle_d{i}",
            }
        )
    for seed in (17, 18, 19):
        for arm in ("sft-reward", "sft-random"):
            draws.append(
                {
                    "arm": arm,
                    "model": f"{arm}-s{seed}",
                    "gate": None,
                    "seed": 21,
                    "label": f"{arm}_s{seed}",
                }
            )
    return draws


def _write(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=1, default=str))
    print("wrote", OUT / name)


@app.local_entrypoint()
def run_train(
    rows_file: str, name: str, seed: int = 17, epochs: float = 1.0, lr: float = 2e-5, rank: int = 16
):
    rows = [json.loads(line) for line in Path(rows_file).read_text().splitlines() if line.strip()]
    out = train_lora.remote(
        {"name": name, "epochs": epochs, "lr": lr, "rank": rank, "seed": seed}, rows
    )
    _write(f"train_{name}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "loss"}, indent=1))


@app.local_entrypoint()
def run_pilot(limit: int = 6, time_budget: float = 900.0):
    """A dozen prompts on base and both gates: does the pipeline run, does
    the gate fire, what does a draw cost. Never a result."""
    draws = [d for d in plan() if d["label"].endswith("_d1") and d["arm"] != "sft-reward"][:3]
    out = evaluate.remote(draws, 1, time_budget, limit, 16)
    OUT.mkdir(parents=True, exist_ok=True)
    for label, res in out["draws"].items():
        (OUT / f"pilot_{label}.jsonl").write_text(res.pop("rows_jsonl", ""))
    _write("pilot.json", out)
    print(json.dumps(out, indent=1, default=str)[:4000])


@app.local_entrypoint()
def run_eval(repeats: int = 1, time_budget: float = 1800.0, only: str = ""):
    draws = plan()
    if only:
        draws = [d for d in draws if d["arm"] in only.split(",")]
    out = evaluate.remote(draws, repeats, time_budget, 0, 16)
    _write("eval.json", out)
    print(json.dumps(out, indent=1, default=str)[:4000])
    print("\nNext: modal run modal_train_eval.py::run_fetch, then python analyse.py")


@app.local_entrypoint()
def run_fetch():
    """Pull every finished draw's rows and record from the results volume."""
    OUT.mkdir(parents=True, exist_ok=True)
    got = []
    for entry in results.listdir("/"):
        name = entry.path.lstrip("/")
        if not (name.endswith(".jsonl") or name.endswith(".json")):
            continue
        data = b"".join(results.read_file(name))
        (OUT / f"eval_{name}").write_bytes(data)
        got.append(name)
    print("fetched", sorted(got))
