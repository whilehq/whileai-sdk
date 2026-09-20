"""Zero RL on a base model: a rigid format reward costs accuracy.

    python recipe.py                      # both arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the two rewards, offline, no GPU

SimpleRL-Zoo (Zeng et al., arXiv:2503.18892) runs GRPO straight on base
models with nothing but a rule reward, and reports one design choice that
matters most for small models: do not make the reward depend on format. The
DeepSeek-R1 style reward pays a correct answer only if it sits inside
``\\boxed{}`` and punishes anything else, and section 3.1 of the paper shows
that this "penalizes many correct explorations" a base model makes before it
has learned the format, and later shrinks the chain of thought. Their default
is +1 for a correct answer read leniently, 0 otherwise, no format term.

Two arms, one change:

  baseline  strict: +1 if the last ``\\boxed{}`` holds the gold answer,
            -1 if there is no ``\\boxed{}`` at all, 0 otherwise (the paper's
            "-1 if they fail to adhere to the required format", section 3.1)
  recipe    lenient: +1 if the answer read from ``\\boxed{}``, "the answer
            is", or the last line equals the gold, else 0 (the paper's
            default, appendix B.2)

Both arms are scored on the same holdout with the same lenient reader, so
the target metric is one number and the arms differ only in what the policy
was paid for during training.

Shape of the run:
  1. data():      MATH train, levels 3-5 (the paper's "Hard" tier), MATH-500 held out
  2. run_arm():   TRL GRPOTrainer + LoRA, vLLM colocated for rollouts, on Modal
  3. evaluate():  same holdout, k samples per task, graded by MathEqual
  4. results.json + the paired delta (wai.delta_report) on the run page
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
METRIC = "pass@1"
BOOK = "Reasoning Reasoning"  # zero RL on a base model with a verifiable reward
# The baseline trains on the strict reward and is scored on the lenient one;
# delta_report(proxy=) names that gap so an over-optimized verdict can fire.
PROXY = "strict_reward"
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)
NO_BOX_PENALTY = -1.0  # section 3.1: "a reward of -1 if they fail to adhere to the required format"

# The prompt SimpleRL-Zoo uses for the Qwen family (their Figure 10 "simple"
# variant): a plain question, one instruction line, no chat template. A base
# model has no chat turns to speak of, and the paper found complex prompts
# destabilise weak instruction followers.
PROMPT = (
    "Question: {question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "Answer:"
)

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


# --------------------------------------------------------------------------
# The two rewards. Pure functions, no torch: `--selftest` runs them on
# hand-written replies, and the Modal container imports this same file.
# --------------------------------------------------------------------------


def prompt_for(question: str) -> str:
    return PROMPT.format(question=question)


def gold_of(task: dict) -> str:
    """MATH-500 ships a clean `answer`; the MATH train split ships only the
    worked `solution`, whose last \\boxed{} is the gold."""
    if task.get("answer"):
        return str(task["answer"]).strip()
    boxes = _BOXED.findall(task.get("solution") or "")
    return boxes[-1].strip() if boxes else ""


def _equal(candidate_answer: str, gold: str) -> bool:
    """One equality rule for both rewards and the eval: sympy with a numeric
    and normalised-string fallback (whileai's MathEqual), fed an answer span
    that has already been extracted, so the reader is the only difference."""
    from whileai.simulations.verify import MathEqual

    row = {
        "prompt": "",
        "final_text": f"\\boxed{{{candidate_answer}}}",
        "privileged": {"reference": gold},
        "scenario_id": "x",
        "rollout_index": 0,
    }
    return MathEqual()(row).get("reward") == 1


def lenient_reward(text: str, gold: str) -> float:
    """The paper's default (appendix B.2): +1 if the answer, read from
    \\boxed{}, "the answer is", or the last line, equals the gold; else 0."""
    from whileai.simulations.verify.math import extract_answer

    got = extract_answer(text)
    if not got or not gold:
        return 0.0
    return 1.0 if _equal(got, gold) else 0.0


def strict_reward(text: str, gold: str) -> float:
    """THE ONE CHANGE, in the other direction: the R1-style format reward
    the paper argues against (section 3.1). +1 only when the last \\boxed{}
    holds the gold; NO_BOX_PENALTY when there is no \\boxed{} at all; 0 for a
    boxed wrong answer."""
    boxes = _BOXED.findall(text or "")
    if not boxes:
        return NO_BOX_PENALTY
    if not gold:
        return 0.0
    return 1.0 if _equal(boxes[-1].strip(), gold) else 0.0


def outcome_of(text: str, gold: str) -> float:
    """The target metric for both arms: the lenient read, so a correct answer
    counts however it was written. The eval never applies the penalty."""
    return lenient_reward(text, gold)


def make_reward(strict: bool, recorder: list[dict]):
    """Build the TRL reward function for one arm.

    `recorder` is refilled with the batch it just graded, so after training
    `hack_scan` can be run on the last one (Lambert 2025, chapter Over-optimization) without keeping
    every step in memory.
    """
    score = strict_reward if strict else lenient_reward

    def reward(completions, prompts, gold, **kwargs) -> list[float]:
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        rewards = [score(t, g) for t, g in zip(texts, gold)]
        seen: dict[str, int] = {}
        recorder.clear()
        for key, text, r in zip(prompts, texts, rewards):
            recorder.append(
                {
                    "prompt": key,
                    "final_text": text,
                    "reward": r,
                    "scenario_id": key,
                    "rollout_index": seen.get(key, 0),
                }
            )
            seen[key] = seen.get(key, 0) + 1
        return rewards

    reward.__name__ = "math_strict" if strict else "math_lenient"
    return reward


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def boxed_share(rows: list[dict]) -> float:
    """How often a reply carries a \\boxed{} at all: the format the strict arm
    is paid for, and the thing the paper says it over-learns."""
    if not rows:
        return 0.0
    return statistics.fmean(1.0 if _BOXED.search(r.get("final_text") or "") else 0.0 for r in rows)


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `delta_report` read: binary
    `reward`, one row per sample, grouped by task. `strict_reward` rides
    along in `markers` so `delta_report(proxy=)` can compare the two."""
    rows: list[dict] = []
    for task, texts in zip(holdout, replies):
        gold = gold_of(task)
        for i, text in enumerate(texts):
            rows.append(
                {
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": outcome_of(text, gold),
                    "markers": {"strict_reward": strict_reward(text, gold)},
                    "scenario_id": task["scenario_id"],
                    "rollout_index": i,
                    "privileged": {"reference": gold},
                }
            )
    return rows


# --------------------------------------------------------------------------
# Modal. The stack is newer than the sibling recipes because Qwen3.5 needs
# transformers >= 4.57 and its linear-attention layers need the fla kernels;
# vLLM is what makes 1.5k-token rollouts affordable on one GPU.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "H100")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-zero-rl-format-reward")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "vllm==0.29.0",
        "trl==1.13.0",
        "peft==0.21.0",
        "flash-linear-attention==0.5.2",
        "datasets>=4.7.0",
        "whileai",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _platform(arm: str, strict: bool, holdout: list[dict], config: dict):
    """Open the run on the platform's Runs page. Returns (tracked, run) or
    (None, None) when there is no key or the platform is unreachable."""
    if not os.environ.get("WHILEAI_API_KEY"):
        return None, None
    try:
        from whileai.platform import track

        tracked = track(
            config.get("agent") or "zero-rl-qwen3.5-4b",
            model=config["base_model"],
            harness={"label": "recipe zero-rl-format-reward", "instructions": PROMPT},
        )
        tracked.behavior(
            "math500",
            n=len(holdout),
            contamination=0,
            reward_is_judge=False,
            test_version=f"math500-{len(holdout)}",
            description="pass@1 on MATH-500, lenient MathEqual read, k=4 samples a task",
        )
        prun = tracked.run(
            config.get("version") or ("strict" if strict else "lenient"),
            base=config["base_model"],
            method="grpo",
            targets=["math500"],
            trained_on=["MATH train lv3-5"],
            gpu=config["gpu"],
        )
        print(f"platform: {prun.url}")
        return tracked, prun
    except Exception as exc:  # the platform is not the experiment
        print(f"platform: skipped ({type(exc).__name__}: {exc})")
        return None, None


def _record(config: dict, holdout: int, pins: dict | None = None) -> dict:
    """The run's scientific record for the platform (`RunRecord`): the data,
    the optimizer knobs, the eval setup and the pins, as the Runs page draws
    them. Plain dicts so the recipe runs on a whileai older than the model."""
    return {
        "data": {
            "train": "DigitalLearningGmbH/MATH-lighteval train, levels 3-5",
            "n_train": config["train_prompts"],
            "difficulty": "hard (MATH lv3-5)",
            "holdout": "HuggingFaceH4/MATH-500",
            "n_holdout": holdout,
        },
        "optimizer": {
            "loss_type": config["loss_type"],
            "lr": config["learning_rate"],
            "beta": config["beta"],
            "epsilon": config["epsilon"],
            "num_generations": config["num_generations"],
            "prompts_per_step": config["prompts_per_step"],
            "max_completion_tokens": config["max_completion_length"],
            "temperature": config["temperature"],
            "top_p": 1.0,
            "lora_rank": config["lora_rank"],
            "seed": config.get("seed", 17),
        },
        "eval": {"metric": METRIC, "k": 4, "reader": "lenient MathEqual"},
        "provenance": {
            "pins": pins or {},
            "recipe": "recipes/papers/zero-rl-format-reward",
            "paper": "2503.18892",
            "reward": config["reward"],
        },
    }


def _post_score(prun, rows: list[dict], steps: int, gpu_minutes: float, record=None) -> None:
    """Score one version on `math500` in points (0-100) with the half-width of
    the 95% interval, then close the run with what it cost and its record."""
    try:
        import whileai.simulations as wai

        p = wai.pass_at(rows)
        lo, hi = p.ci95 or (p.pass_at_1, p.pass_at_1)
        n_tasks = len({r["scenario_id"] for r in rows})
        prun.score("math500", round(100 * p.pass_at_1, 1), ci=round(50 * (hi - lo), 1), n=n_tasks)
        usd = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 4.00) * gpu_minutes / 60
        closing = {
            "hours": round(gpu_minutes / 60, 3),
            "gpu": DEFAULT_GPU,
            "cost_usd": round(usd, 2),
            "steps": steps,
            "summary": {"boxed_share": boxed_share(rows), "length_chars": mean_length(rows)},
        }
        try:
            prun.finish("evaluated", record=record, **closing)
        except TypeError:  # whileai before RunRecord: the record is not sent
            prun.finish("evaluated", **closing)
    except Exception as exc:
        print(f"platform: score skipped ({type(exc).__name__}: {exc})")


def _platform_callback(prun):
    """``wai.TrainerCallback`` with one more key: TRL logs the share of
    rollouts that hit ``max_completion_length`` as
    ``completions/clipped_ratio``, and the platform's Rollouts tile reads it
    as ``clip_ratio`` (the length-cap curve; Lambert 2025, chapter Reasoning on length
    growth). The SDK maps TRL's ``clip_ratio/*`` (the policy-ratio clip),
    not this key, so the recipe folds it into the same point."""
    import whileai.simulations as wai

    class Callback(wai.TrainerCallback):
        def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
            if isinstance(logs, dict) and "completions/clipped_ratio" in logs:
                logs = {**logs, "clip_ratio": logs["completions/clipped_ratio"]}
            return super().on_log(args, state, control, logs=logs, **kwargs)

    return Callback(prun, finish=False)


def _sample_vllm(llm, prompts: list[str], *, n: int, max_tokens: int, seed: int) -> list[list[str]]:
    """`n` replies per prompt from the colocated vLLM engine, sampled the way
    the trainer samples (temperature 1.0, top-p 1.0: the paper's setting)."""
    from vllm import SamplingParams

    params = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=max_tokens, seed=seed)
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [[o.text for o in out.outputs] for out in outs]


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    strict: bool,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 30,
    num_generations: int = 8,
    prompts_per_step: int = 6,
    learning_rate: float = 5e-5,
    beta: float = 1e-4,
    max_completion_length: int = 1536,
    lora_rank: int = 32,
    eval_samples: int = 4,
    eval_base: bool = False,
    seed: int = 17,
    loss_type: str = "dapo",
    agent: str = "zero-rl-qwen3.5-4b",
    platform_version: str | None = None,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again."""
    import time

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        EVAL_RUNS,
        boxed_share,
        gold_of,
        graded_rows,
        make_reward,
        mean_length,
        prompt_for,
    )

    import whileai.simulations as wai

    started = time.time()
    config = {
        "arm": arm,
        "reward": "strict boxed, -1 without a box" if strict else "lenient, correctness only",
        "base_model": base_model,
        "steps": steps,
        "num_generations": num_generations,
        "prompts_per_step": prompts_per_step,
        "learning_rate": learning_rate,
        "beta": beta,
        "epsilon": 0.2,
        "loss_type": loss_type,
        "temperature": 1.0,
        "seed": seed,
        "agent": agent,
        "version": platform_version or ("strict" if strict else "lenient"),
        "lora_rank": lora_rank,
        "max_completion_length": max_completion_length,
        "train_prompts": len(train_tasks),
        "holdout": len(holdout),
        "gpu": DEFAULT_GPU,
        "generation": "vllm colocate",
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        run = wai.training_run(
            run_name,
            base_model=base_model,
            trainer="trl-grpo-lora-vllm",
            total_steps=steps,
            config=config,
        )
        print(f"dashboard: {run.url}")
    # The Runs page on the platform: one tracked agent, one version per arm,
    # `math500` as the behavior. A platform outage must never cost the GPU
    # run, so every call here is best effort.
    tracked, prun = _platform(arm, strict, holdout, config)

    dataset = Dataset.from_list(
        [{"prompt": prompt_for(t["question"]), "gold": gold_of(t)} for t in train_tasks]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=num_generations,
        # Half a group per forward, twice the accumulation: the same 48
        # rollouts a step, but the loss forward materialises the full
        # 248k-vocabulary logits in fp32 and a whole group of 8 at 1.5k
        # tokens ran the H100 out of memory.
        per_device_train_batch_size=num_generations // 2,
        gradient_accumulation_steps=2 * prompts_per_step,
        generation_batch_size=num_generations * prompts_per_step,
        learning_rate=learning_rate,
        # The paper's GRPO: clip 0.2, KL coefficient 1e-4 for models up to
        # 14B, temperature 1.0, and the length-normalisation term removed
        # (appendix B.5). TRL's `dapo` loss is that token-level objective.
        beta=beta,
        epsilon=0.2,
        loss_type=loss_type,
        temperature=1.0,
        top_p=1.0,
        max_completion_length=max_completion_length,
        # Rollouts come from a vLLM engine sharing the GPU; TRL merges the
        # adapter into it before every generation step.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.40,
        vllm_max_model_length=768 + max_completion_length,
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
        # Every linear layer, because 24 of Qwen3.5's 32 blocks are gated
        # DeltaNet whose projections are not named q/k/v/o.
        target_modules="all-linear",
    )
    last_batch: list[dict] = []
    # Seed before the trainer builds the adapter so both arms draw the same
    # LoRA init whatever ran in this process before (the sibling adaptive-clip
    # recipe learned this the expensive way).
    set_seed(seed)
    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=[make_reward(strict, last_batch)],
        args=grpo,
        train_dataset=dataset,
        peft_config=lora,
    )
    llm = trainer.vllm_generation.llm
    questions = [prompt_for(t["question"]) for t in holdout]

    # The base is evaluated EVAL_RUNS times, not once. The spread across those
    # re-runs is the eval's own noise, and a delta smaller than it is not a
    # result (Lambert 2025, chapter Evaluation). Only the first arm pays for this. Before the
    # first step the adapter's B matrix is zero, so the engine holds the base.
    base_runs = []
    if eval_base:
        for i in range(EVAL_RUNS):
            replies = _sample_vllm(
                llm, questions, n=eval_samples, max_tokens=max_completion_length, seed=1000 + i
            )
            rows = graded_rows(holdout, replies)
            base_runs.append(rows)
            print(
                f"base run {i + 1}/{EVAL_RUNS}: {wai.pass_at(rows)} boxed {boxed_share(rows):.2f}"
            )
        if tracked is not None:
            _post_score(tracked.run("base", method="none", base=base_model), base_runs[0], 0, 0.0)
            if len(base_runs) >= 2:  # the behavior's re-run floor, measured, not typed by hand
                tracked.noise_floor("math500", *base_runs)

    if run is not None:
        trainer.add_callback(wai.TrainerCallback(run, finish=False))
    if prun is not None:
        trainer.add_callback(_platform_callback(prun))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise

    # The engine last saw the weights at the start of the final step; push the
    # trained adapter once more so the eval samples from what was trained.
    trainer.vllm_generation.sync_weights()
    replies = _sample_vllm(
        llm, questions, n=eval_samples, max_tokens=max_completion_length, seed=2000
    )
    after_rows = graded_rows(holdout, replies)
    after = wai.pass_at(after_rows)
    print(f"{arm}: {after} boxed {boxed_share(after_rows):.2f}")

    # What the reward actually paid for in the last training batch (chapter Over-optimization).
    scan = wai.hack_scan(last_batch) if last_batch else {}
    hack_top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
    hack_scan_top = hack_top.get("name", "") if isinstance(hack_top, dict) else str(hack_top)
    print(f"{arm} hack scan: top feature {hack_scan_top or 'none above the floor'}")

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    trainer.processing_class.save_pretrained(adapter_dir)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    pins = {
        "torch": version("torch"),
        "transformers": version("transformers"),
        "trl": version("trl"),
        "peft": version("peft"),
        "vllm": version("vllm"),
    }
    summary = {
        "arm": arm,
        "strict_reward": strict,
        "pass_at_1": after.pass_at_1,
        "boxed_share_after": boxed_share(after_rows),
        "length_after_chars": mean_length(after_rows),
        "gpu_minutes": gpu_minutes,
        "steps": steps,
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    if prun is not None:
        _post_score(
            prun, after_rows, steps, gpu_minutes, record=_record(config, len(holdout), pins)
        )
        summary["platform_url"] = prun.url
    return {
        "arm": arm,
        "base_runs": base_runs,
        "after_rows": after_rows,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
        "length_after": mean_length(after_rows),
        "boxed_after": boxed_share(after_rows),
        "hack_scan_top": hack_scan_top,
        "run_url": summary.get("run_url", ""),
        "pins": pins,
    }


# --------------------------------------------------------------------------
# Local: data, orchestration, results.json.
# --------------------------------------------------------------------------


def data(seed: int, n_train: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    """MATH train at levels 3 to 5 (the paper's "Hard" tier, the one it pairs
    with Qwen-class base models), MATH-500 held out. Disjoint by construction:
    MATH-500 is drawn from the MATH test split."""
    from datasets import load_dataset

    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    train = train.filter(lambda r: r["level"] in ("Level 3", "Level 4", "Level 5"))
    train = train.shuffle(seed=seed)
    test = load_dataset("HuggingFaceH4/MATH-500", split="test").shuffle(seed=seed)
    train_tasks = [
        {
            "question": r["problem"],
            "solution": r["solution"],
            "level": r["level"],
            "scenario_id": f"train-{i}",
        }
        for i, r in enumerate(train.select(range(n_train)))
    ]
    holdout = [
        {
            "question": r["problem"],
            "answer": r["answer"],
            "level": str(r["level"]),
            "scenario_id": f"math500-{r['unique_id']}",
        }
        for r in test.select(range(n_holdout))
    ]
    return train_tasks, holdout


def summarize(rows: list[dict]) -> dict:
    import whileai.simulations as wai

    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
    }


def selftest() -> None:
    """The two rewards on hand-written replies, on the CPU. No GPU, no key,
    no model download."""
    gold = "\\frac{1}{2}"
    boxed_right = "so x = 1/2, that is \\boxed{\\frac{1}{2}}"
    boxed_wrong = "so \\boxed{2}"
    unboxed_right = "Therefore the answer is 1/2"
    unboxed_wrong = "I think it is 3"

    # Lenient: correctness only, however it was written.
    assert lenient_reward(boxed_right, gold) == 1.0
    assert lenient_reward(unboxed_right, gold) == 1.0
    assert lenient_reward(boxed_wrong, gold) == 0.0
    assert lenient_reward(unboxed_wrong, gold) == 0.0

    # Strict: the box is the contract. A correct answer outside it is punished
    # exactly as hard as no answer at all, which is the paper's complaint.
    assert strict_reward(boxed_right, gold) == 1.0
    assert strict_reward(boxed_wrong, gold) == 0.0
    assert strict_reward(unboxed_right, gold) == NO_BOX_PENALTY
    assert strict_reward(unboxed_wrong, gold) == NO_BOX_PENALTY

    # The eval reads leniently for both arms: one target metric.
    assert outcome_of(unboxed_right, gold) == 1.0

    # The train split's gold comes from the solution's last box.
    assert gold_of({"solution": "First \\boxed{3} was wrong; finally \\boxed{7}."}) == "7"
    assert gold_of({"answer": "\\left( 3, \\frac{\\pi}{2} \\right)"}).startswith("\\left")

    rows = [{"final_text": boxed_right}, {"final_text": unboxed_right}]
    assert boxed_share(rows) == 0.5
    print("lenient: 1/2 in a box 1.0, 1/2 in a sentence 1.0, wrong 0.0")
    print(f"strict:  1/2 in a box 1.0, 1/2 in a sentence {NO_BOX_PENALTY}, boxed wrong 0.0")
    print(f"prompt:\n{prompt_for('What is 1 + 1?')}")
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--n-holdout", type=int, default=160)
    ap.add_argument("--generations", type=int, default=8, help="rollouts per prompt")
    ap.add_argument("--prompts-per-step", type=int, default=6)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument(
        "--beta", type=float, default=1e-4, help="KL coefficient, the paper's for <=14B"
    )
    ap.add_argument("--selftest", action="store_true", help="the two rewards, offline")
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="take an arm from .cache/<arm>.json when it is there instead of training it again",
    )
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai.simulations as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    # MATH-500 is a subset of the MATH test split, so this should drop nothing
    # from the train split. It runs anyway, and the count goes in the Checks
    # table, because "should" is not a measurement (Lambert 2025, chapter Evaluation).
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]
    strict = {"baseline": True, "recipe": False}

    results = json.loads((HERE / "results.json").read_text())
    results.update(
        {
            "recipe": HERE.name,
            "title": "Zero RL on a base model: a rigid format reward costs accuracy",
            "paper": "https://arxiv.org/abs/2503.18892",
            "book": BOOK,
            "base_model": BASE_MODEL,
            "metric": METRIC,
            "n_holdout": len(holdout),
            "k": args.k,
            "gpu": DEFAULT_GPU,
            "whileai": version("whileai"),
        }
    )
    results.setdefault("arms", {})
    checks = results.setdefault("checks", {})
    checks["decontaminated_dropped"] = int(decon.get("n_contaminated", 0))
    checks["seed"] = args.seed
    checks.setdefault("length_after", {})
    checks.setdefault("boxed_after", {})
    arm_rows: dict[str, list[dict]] = {}
    run_std = 0.0
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 4.00)
    gpu_minutes = 0.0
    run_url = ""

    # Every arm's rows land in .cache/<arm>.json the moment it returns, so a
    # crash in the second arm never costs the first, and `--reuse` rebuilds
    # the delta from disk.
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)
    with modal.enable_output(), app.run():
        for i, arm in enumerate(arms):
            cached = cache / f"{arm}.json"
            if args.reuse and cached.exists():
                out = json.loads(cached.read_text())
                print(f"{arm}: reused {cached}")
            else:
                out = run_arm.remote(
                    arm,
                    strict[arm],
                    train_tasks,
                    holdout,
                    f"zero-rl-{arm}-{date.today().isoformat()}",
                    steps=args.steps,
                    num_generations=args.generations,
                    prompts_per_step=args.prompts_per_step,
                    learning_rate=args.lr,
                    beta=args.beta,
                    max_completion_length=args.max_completion,
                    eval_samples=args.k,
                    eval_base=(i == 0),
                )
                cached.write_text(json.dumps(out))
            gpu_minutes += out["gpu_minutes"]
            run_url = out["run_url"] or run_url
            checks["pins"] = out["pins"]
            if out["base_runs"]:
                base_runs = out["base_runs"]
                noise = wai.eval_variance(*base_runs)
                run_std = float(noise["run_std"])
                print(f"eval noise over {len(base_runs)} base runs: run_std {run_std:.4f}")
                arm_rows["base"] = base_runs[0]
                results["arms"]["base"] = {
                    **summarize(base_runs[0]),
                    "steps": 0,
                    "gpu_minutes": 0,
                }
                checks["run_std"] = run_std
                checks["run_std_runs"] = int(noise["n_runs"])
                checks["length_before"] = mean_length(base_runs[0])
                checks["boxed_before"] = boxed_share(base_runs[0])
            arm_rows[arm] = out["after_rows"]
            results["arms"][arm] = {
                **summarize(out["after_rows"]),
                "steps": out["steps"],
                "gpu_minutes": round(out["gpu_minutes"], 1),
            }
            checks["length_after"][arm] = out["length_after"]
            checks["boxed_after"][arm] = out["boxed_after"]
            checks["hack_scan_top"] = out["hack_scan_top"]

    if "baseline" in arm_rows and "recipe" in arm_rows:
        d = wai.delta_report(
            arm_rows["baseline"],
            arm_rows["recipe"],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks.get("run_std_runs") or EVAL_RUNS),
            markers=[PROXY],
            proxy=PROXY,
        )
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": "moved" if d["target_verdict"] == "moved" else "flat",
        }
        checks["over_optimized"] = bool(d.get("over_optimized"))
        results["verified"] = date.today().isoformat()
        results.pop("partial_run", None)
        print(wai.format_delta_report(d))
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"
        print(f"one arm only ({', '.join(arms)}): delta and verified left as they were")

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
