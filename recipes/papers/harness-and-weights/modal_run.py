"""The Modal side of the recipe: the harness search on the base weights,
and one GRPO arm trained under a harness and evaluated under it.

The stack is the zero-rl-format-reward recipe's, pinned the same way
(CUDA 12.8, vLLM 0.29.0, TRL 1.13.0, PEFT 0.21.0, Python 3.12), on one
H100. Two memory lessons from that recipe are kept in the config below:
half a group per forward with twice the accumulation, and
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` in the image.

``recipe.py`` imports this module only on the paid paths (``--smoke`` and
the full run), so the dry run, the tests and CI never need ``modal``.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any

import modal

from whileai.config import requirement

HERE = Path(__file__).resolve().parent
DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "H100")
VOLUME_ROOT = "/vol"
# VLLM_MAX_LEN = 6144: prompt (skills text plus task, under 1.5k tokens), up
# to three assistant turns of MAX_COMPLETION, and two tool results; the
# engine the trainer shares serves the multi-turn eval after training. TRL
# 1.13 has no prompt truncation knob; the longest prompt here is well
# inside this window.
VLLM_MAX_LEN = 6144
# EVAL_TEMPERATURE = 1.0: the eval samples the way the trainer samples
# (temperature 1.0, top-p 1.0), as the sibling recipes do.
EVAL_TEMPERATURE = 1.0

app = modal.App("whileai-recipe-harness-and-weights")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "vllm==0.29.0",
        "trl==1.13.0",
        "peft==0.21.0",
        "datasets>=4.7.0",
        requirement(),
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(str(HERE / "tasks.py"), "/root/tasks.py")
    .add_local_file(str(HERE / "harnesses.py"), "/root/harnesses.py")
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)


def _pins() -> dict[str, str]:
    return {name: version(name) for name in ("torch", "transformers", "trl", "peft", "vllm")}


def _chat(llm: Any, tokenizer: Any, *, max_tokens: int, seed: int, temperature: float):
    """A ``harnesses.play`` chat over a vLLM engine: the chat template the
    trainer applies, one seeded sampling parameter set per conversation."""
    import harnesses
    from vllm import SamplingParams

    def chat(convos: list[dict[str, Any]]) -> list[str]:
        prompts = [
            tokenizer.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True)
            for c in convos
        ]
        params = [
            SamplingParams(
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
                seed=harnesses.shuffle_seed(
                    seed, c["task"]["scenario_id"], c["rollout_index"], c["turn"]
                ),
            )
            for c in convos
        ]
        outs = llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    return chat


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=2 * 60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
)
def search(
    specs: list[dict[str, Any]],
    all_tasks: list[dict[str, Any]],
    holdout_ids: list[str],
    *,
    k: int,
    seed: int,
    eval_runs: int,
    max_tokens: int,
    base_model: str,
) -> dict[str, Any]:
    """Every candidate harness on the base weights over every task, graded;
    then the baseline harness on the holdout ``eval_runs - 1`` more times
    for the noise floor. Returns rows by harness label, the base re-runs,
    GPU minutes and the pins."""
    from transformers import AutoTokenizer
    from vllm import LLM

    sys.path.insert(0, "/root")
    import harnesses

    started = time.time()
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=VLLM_MAX_LEN,
        seed=seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    rows_by_label: dict[str, list[dict]] = {}
    for spec in specs:
        chat = _chat(llm, tokenizer, max_tokens=max_tokens, seed=seed, temperature=EVAL_TEMPERATURE)
        rows = harnesses.play(chat, spec, all_tasks, k=k, seed=seed, model="base")
        rows_by_label[spec["label"]] = harnesses.grade(rows)
        passed = sum(1 for r in rows_by_label[spec["label"]] if r.get("reward") == 1)
        print(f"search: {spec['label']} {passed}/{len(rows)} rollouts passed")
    hold_set = set(holdout_ids)
    hold = [t for t in all_tasks if t["scenario_id"] in hold_set]
    base_runs = [[r for r in rows_by_label[specs[0]["label"]] if r["scenario_id"] in hold_set]]
    for i in range(1, eval_runs):
        chat = _chat(
            llm,
            tokenizer,
            max_tokens=max_tokens,
            seed=seed + 1000 + i,
            temperature=EVAL_TEMPERATURE,
        )
        rows = harnesses.play(chat, specs[0], hold, k=k, seed=seed + 1000 + i, model="base")
        base_runs.append(harnesses.grade(rows))
    return {
        "rows": rows_by_label,
        "base_runs": base_runs,
        "gpu_minutes": (time.time() - started) / 60.0,
        "pins": _pins(),
    }


def _reward_fn(setup: str, recorder: list[dict]):
    """The TRL reward: ``CodeExec`` on the hidden tests, in parallel. The
    last batch is kept in ``recorder`` for ``hack_scan``."""
    from whileai.simulations.verify import CodeExec

    verifier = CodeExec(setup=setup, timeout=10.0)

    def reward(completions, prompts, tests, task_id, **kwargs) -> list[float]:
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        rows = [
            {
                "prompt": str(p),
                "final_text": t,
                "privileged": {"tests": ts},
                "scenario_id": tid,
                "rollout_index": 0,
            }
            for p, t, ts, tid in zip(prompts, texts, tests, task_id)
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            outs = list(pool.map(verifier, rows))
        seen: dict[str, int] = {}
        recorder.clear()
        scores = []
        for row, out in zip(rows, outs):
            r = float(out.get("reward") or 0.0)
            scores.append(r)
            row["reward"] = r
            row["rollout_index"] = seen.get(row["scenario_id"], 0)
            seen[row["scenario_id"]] = row["rollout_index"] + 1
            recorder.append(row)
        return scores

    reward.__name__ = "code_exec"
    return reward


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=3 * 60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
)
def train_arm(
    arm: str,
    spec: dict[str, Any],
    train_tasks: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    run_name: str,
    *,
    base_model: str,
    steps: int,
    num_generations: int,
    prompts_per_step: int,
    learning_rate: float,
    beta: float,
    max_completion_length: int,
    lora_rank: int,
    k: int,
    seed: int,
) -> dict[str, Any]:
    """One arm: GRPO under ``spec``'s instructions as the system prompt, the
    reward ``CodeExec`` on the hidden tests, then the trained weights played
    under the same harness (tool loop and all) on the holdout."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    import harnesses
    import tasks as task_mod

    import whileai.simulations as wai

    started = time.time()
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": spec["instructions"]},
                    {"role": "user", "content": t["prompt"]},
                ],
                "tests": t["privileged"]["tests"],
                "task_id": t["scenario_id"],
            }
            for t in train_tasks
        ]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=num_generations,
        # Half a group per forward, twice the accumulation: the same
        # rollouts a step, and the loss forward never holds a whole group's
        # fp32 logits at once (the zero-rl recipe's OOM fix: 4 and 12).
        per_device_train_batch_size=num_generations // 2,
        gradient_accumulation_steps=2 * prompts_per_step,
        generation_batch_size=num_generations * prompts_per_step,
        learning_rate=learning_rate,
        beta=beta,
        epsilon=0.2,
        loss_type="dapo",
        temperature=1.0,
        top_p=1.0,
        max_completion_length=max_completion_length,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.40,
        vllm_max_model_length=VLLM_MAX_LEN,
        bf16=True,
        gradient_checkpointing=True,
        model_init_kwargs={"dtype": "bfloat16"},
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
    )
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    last_batch: list[dict] = []
    set_seed(seed)
    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=[_reward_fn(task_mod.SETUP, last_batch)],
        args=grpo,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()
    trainer.vllm_generation.sync_weights()
    llm = trainer.vllm_generation.llm
    tokenizer = trainer.processing_class
    chat = _chat(
        llm, tokenizer, max_tokens=max_completion_length, seed=2000, temperature=EVAL_TEMPERATURE
    )
    after_rows = harnesses.grade(
        harnesses.play(chat, spec, holdout, k=k, seed=2000, model="trained")
    )
    after = wai.pass_at(after_rows)
    print(f"{arm}: {after}")
    scan = wai.hack_scan(last_batch) if last_batch else {}
    top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
    hack_scan_top = top.get("name", "") if isinstance(top, dict) else str(top)
    print(f"{arm} hack scan: top feature {hack_scan_top or 'none above the floor'}")
    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()
    return {
        "arm": arm,
        "rows": after_rows,
        "gpu_minutes": (time.time() - started) / 60.0,
        "steps": steps,
        "hack_scan_top": hack_scan_top,
        "pins": _pins(),
        "adapter": f"whileai-recipe-runs:/{run_name}/adapter",
    }
