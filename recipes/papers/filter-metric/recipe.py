"""Filter metric: which signal decides a group is worth training on.

    python recipe.py                      # both arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --arm base --base-runs 10   # the noise floor only: no training
    python recipe.py --selftest           # the filter, offline, no GPU and no key

GRPO throws away a group of rollouts when the group has no contrast: every
rollout scores the same, so every advantage is zero and there is nothing to
learn. Trainers let you pick which number that "same" is read off. The paper
says that choice is not cosmetic.

The reward here is shaped: a 0/1 outcome minus a small length penalty,
``r = outcome - 0.30 * min(len/512, 1)``. Take a group where every rollout is
wrong. On the outcome the group is flat, so it is dropped. On the shaped score
it is not flat at all, because the answers differ in length. It survives the
filter, and GRPO's divide-by-group-std blows those length crumbs up into
full-size advantages. The model is then taught, at full strength, that the
shortest wrong answer is the good one. The paper calls these phantom
advantages.

The two arms differ in one line: which list the "is this group flat?" test
reads. ``--filter-metric score`` reads the shaped score (the baseline, and the
setting the paper warns about). ``--filter-metric outcome`` reads the binary
outcome (the recipe, and what DAPO's dynamic sampling actually means).

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out
  2. run_arm():   TRL GRPOTrainer + LoRA on Modal, one arm per call
  3. evaluate():  same holdout, k samples per task, graded by MathEqual
  4. results.json + the paired delta (wai.delta_report) on the run page
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import modal

from whileai.config import provenance

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"

# chapter Over-optimization is the chapter this recipe lives in: the shaped score is a proxy, and
# the paper's failure is that proxy being optimized while the target does not
# move. The filter itself belongs to chapter Reinforcement Learning (policy gradients, group baselines).
BOOK = "Over-optimization Over-optimization"
# The training reward is not the target metric, so it is named as the proxy and
# `delta_report` gets to call over-optimization when the two come apart.
PROXY = "marker:shaped_reward"
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

# The shaped reward, straight from the paper: outcome minus a length penalty
# normalized at 512 characters. LAMBDA is the "phantom strength" knob; the
# paper sweeps 0.1/0.3/0.5 and reports the collapse at 0.30.
LAMBDA = 0.30
LENGTH_NORM = 512

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."


# --------------------------------------------------------------------------
# The reward and the filter. Pure functions, no torch: `--selftest` runs them
# on hand-written groups, and the Modal container imports this same file.
# --------------------------------------------------------------------------


def messages_for(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def gold_of(answer: str) -> str:
    """GSM8K ships the worked solution then `#### 72`. The gold is the tail."""
    return answer.split("####")[-1].strip().replace(",", "")


def outcome_of(text: str, gold: str) -> float:
    """1.0 when the final number matches the gold, else 0.0. A program, not a
    judge: `MathEqual` is sympy with a numeric and string fallback."""
    from whileai.simulations.verify import MathEqual

    row = {
        "prompt": "",
        "final_text": text,
        "privileged": {"reference": gold},
        "scenario_id": "x",
        "rollout_index": 0,
    }
    return 1.0 if MathEqual()(row).get("reward") == 1 else 0.0


def shaped_of(text: str, outcome: float) -> float:
    """The composite the trainer optimizes. Among wrong answers the shortest
    one scores highest, which is the shortcut the paper is about."""
    return outcome - LAMBDA * min(len(text) / LENGTH_NORM, 1.0)


def groups_of(prompts: list) -> list[tuple[int, int]]:
    """(start, end) for each run of identical consecutive prompts. TRL hands a
    reward function one batch with `num_generations` rollouts per prompt, laid
    out contiguously; grouping on the prompt itself does not assume that."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(prompts) + 1):
        if i == len(prompts) or prompts[i] != prompts[start]:
            bounds.append((start, i))
            start = i
    return bounds


def apply_filter(
    prompts: list, outcomes: list[float], shaped: list[float], filter_metric: str
) -> list[float]:
    """THE ONE CHANGE. Drop every group that is flat on `filter_metric`.

    "Drop" here is masking, not deletion: a group whose rewards are all equal
    gets a zero advantage from GRPO, so it contributes nothing to the update.
    The paper's DAPO arm physically deletes the group and refills the batch,
    which also changes how many fresh prompts a step sees; masking reproduces
    the advantage-level effect on a fixed batch, not the refill dynamics.
    """
    if filter_metric not in ("score", "outcome", "none"):
        raise ValueError(f"filter_metric must be score, outcome or none: {filter_metric!r}")
    rewards = list(shaped)
    if filter_metric == "none":
        return rewards
    read = shaped if filter_metric == "score" else outcomes
    for start, end in groups_of(prompts):
        window = read[start:end]
        if len(set(window)) == 1:  # flat on the filter metric -> no contrast
            flat = sum(shaped[start:end]) / (end - start)
            for i in range(start, end):
                rewards[i] = flat
    return rewards


def make_reward(filter_metric: str, sink: list | None = None):
    """The reward function TRL calls, closed over the arm's filter metric.

    `sink` keeps the most recent batch in graded-row shape, so `hack_scan` can
    read the last training batch after `trainer.train()` returns (chapter Over-optimization)."""

    def reward(completions, prompts, answer, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        keys = [json.dumps(p, sort_keys=True, default=str) for p in prompts]
        outcomes = [outcome_of(t, gold_of(a)) for t, a in zip(texts, answer)]
        shaped = [shaped_of(t, o) for t, o in zip(texts, outcomes)]
        rewards = apply_filter(keys, outcomes, shaped, filter_metric)
        if sink is not None:
            sink.clear()
            seen: dict[str, int] = {}
            for key, text, out, sc in zip(keys, texts, outcomes, shaped):
                seen[key] = seen.get(key, -1) + 1
                sink.append(
                    {
                        "prompt": key,
                        "final_text": text,
                        "reward": out,
                        "scenario_id": key,
                        "rollout_index": seen[key],
                        "markers": {"shaped_reward": sc},
                    }
                )
        return rewards

    reward.__name__ = f"shaped_reward_filter_{filter_metric}"
    return reward


def mean_length(rows: list[dict]) -> float:
    """Mean completion length, the chapter Over-optimization tell that a length term is winning."""
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def top_hack_feature(rows: list[dict]) -> str:
    """What the reward actually paid for in the last training batch (chapter Over-optimization)."""
    import whileai.simulations as wai

    try:
        scan = wai.hack_scan(rows, endorsed=[PROXY])
    except Exception as exc:  # a batch too small to rank is not a failed run
        return f"unavailable: {type(exc).__name__}"
    features = scan.get("features") or []
    return str(features[0]["name"]) if features else "none above the noise floor"


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `delta_report` read: binary
    `reward`, one row per sample, grouped by task."""
    rows: list[dict] = []
    for task, texts in zip(holdout, replies):
        gold = gold_of(task["answer"])
        for i, text in enumerate(texts):
            outcome = outcome_of(text, gold)
            rows.append(
                {
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": outcome,
                    # The training reward rides along as a marker so
                    # `delta_report(proxy=)` can compare it to the target.
                    "markers": {"shaped_reward": shaped_of(text, outcome)},
                    "scenario_id": task["scenario_id"],
                    "rollout_index": i,
                    "privileged": {"reference": gold},
                }
            )
    return rows


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/04-train/grpo/train_modal.py.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-filter-metric")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "whileai",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _sample(model, tokenizer, questions, *, n, max_new_tokens, batch=8):
    """`n` replies per question, batched, sampled the way the trainer samples."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    model.eval()
    tokenizer.padding_side = "left"
    out: list[list[str]] = []
    texts = [
        tokenizer.apply_chat_template(messages_for(q), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append(decoded[i * n : (i + 1) * n])
    model.train()
    return out


def _base_runs(model, tokenizer, holdout, runs, eval_samples, max_new_tokens) -> list[list[dict]]:
    """`runs` evaluations of the same untrained model on the same holdout:
    the spread between them is the eval's own noise (Lambert 2025, chapter Evaluation)."""
    sys.path.insert(0, "/root")
    from recipe_mod import graded_rows

    import whileai.simulations as wai

    questions = [t["question"] for t in holdout]
    out: list[list[dict]] = []
    for i in range(runs):
        replies = _sample(
            model, tokenizer, questions, n=eval_samples, max_new_tokens=max_new_tokens
        )
        out.append(graded_rows(holdout, replies))
        print(f"base run {i + 1}/{runs}: {wai.pass_at(out[-1])}")
    return out


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def eval_base(
    holdout: list[dict],
    runs: int,
    base_model: str = BASE_MODEL,
    eval_samples: int = 4,
    max_completion_length: int = 256,
) -> dict:
    """The noise floor on its own: the untrained base evaluated `runs` times,
    no training. `--arm base --base-runs 10` uses this to refresh `run_std`
    without re-running either arm."""
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    base_runs = _base_runs(model, tokenizer, holdout, runs, eval_samples, max_completion_length)
    return {"base_runs": base_runs, "gpu_minutes": (time.time() - started) / 60.0}


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    filter_metric: str,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 40,
    num_generations: int = 5,
    prompts_per_step: int = 8,
    learning_rate: float = 1e-4,
    beta: float = 0.04,
    max_completion_length: int = 256,
    lora_rank: int = 32,
    eval_samples: int = 4,
    eval_base: bool = False,
    eval_runs: int = EVAL_RUNS,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        LAMBDA,
        graded_rows,
        make_reward,
        mean_length,
        messages_for,
        top_hack_feature,
    )

    import whileai.simulations as wai

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    config = {
        "arm": arm,
        "filter_metric": filter_metric,
        "base_model": base_model,
        "steps": steps,
        "num_generations": num_generations,
        "prompts_per_step": prompts_per_step,
        "learning_rate": learning_rate,
        "beta": beta,
        "lora_rank": lora_rank,
        "lambda": LAMBDA,
        "max_completion_length": max_completion_length,
        "train_prompts": len(train_tasks),
        "holdout": len(holdout),
        "gpu": DEFAULT_GPU,
        "reward": "outcome(MathEqual) - 0.30 * min(len/512, 1)",
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        run = wai.training_run(
            run_name,
            base_model=base_model,
            trainer="trl-grpo-lora",
            total_steps=steps,
            config=config,
        )
        print(f"dashboard: {run.url}")

    questions = [t["question"] for t in holdout]
    base_runs: list[list[dict]] = []
    if eval_base:
        # Re-runs of the same eval on the same untrained model: the spread
        # between them is the noise floor any delta has to clear.
        base_runs = _base_runs(
            model, tokenizer, holdout, eval_runs, eval_samples, max_completion_length
        )

    dataset = Dataset.from_list(
        [{"prompt": messages_for(t["question"]), "answer": t["answer"]} for t in train_tasks]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=prompts_per_step,
        learning_rate=learning_rate,
        beta=beta,
        max_completion_length=max_completion_length,
        max_prompt_length=512,
        temperature=0.9,
        bf16=True,
        # Off on purpose: this trainer generates during training, and
        # checkpointing corrupts Qwen generation on these pins.
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=17,
    )
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
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
    last_batch: list[dict] = []
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward(filter_metric, sink=last_batch)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    if run is not None:
        trainer.add_callback(wai.TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise

    replies = _sample(
        trainer.model, tokenizer, questions, n=eval_samples, max_new_tokens=max_completion_length
    )
    after_rows = graded_rows(holdout, replies)
    after = wai.pass_at(after_rows)
    print(f"{arm}: {after}")

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()

    hack_top = top_hack_feature(last_batch)
    print(f"hack scan, last training batch: top feature {hack_top}")

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
        "filter_metric": filter_metric,
        "pass_at_1": after.pass_at_1,
        "mean_length": mean_length(after_rows),
        "hack_scan_top": hack_top,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    return {
        "arm": arm,
        "base_runs": base_runs,
        "after_rows": after_rows,
        "hack_scan_top": hack_top,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
        "run_url": summary.get("run_url", ""),
    }


# --------------------------------------------------------------------------
# Local: data, orchestration, results.json.
# --------------------------------------------------------------------------


def data(seed: int, n_train: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    """GSM8K. Train prompts from the train split, holdout from the test split,
    so the two are disjoint by construction, not by a shuffle."""
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    train_tasks = [
        {"question": r["question"], "answer": r["answer"], "scenario_id": f"train-{i}"}
        for i, r in enumerate(train.select(range(n_train)))
    ]
    holdout = [
        {"question": r["question"], "answer": r["answer"], "scenario_id": f"test-{i}"}
        for i, r in enumerate(test.select(range(n_holdout)))
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
    """The one change, on hand-written groups. No GPU, no key, no model."""
    short_wrong, long_wrong = "7", "7" + " because " * 40
    # A group where every rollout is wrong, and they differ only in length.
    prompts = ["p"] * 4
    outcomes = [0.0, 0.0, 0.0, 0.0]
    shaped = [
        shaped_of(short_wrong, 0.0),
        shaped_of(long_wrong, 0.0),
        shaped_of(short_wrong * 3, 0.0),
        shaped_of(long_wrong, 0.0),
    ]
    by_score = apply_filter(prompts, outcomes, shaped, "score")
    by_outcome = apply_filter(prompts, outcomes, shaped, "outcome")
    assert len(set(by_score)) > 1, "score filter should keep this all-wrong group"
    assert len(set(by_outcome)) == 1, "outcome filter should flatten this all-wrong group"
    print("all-wrong group, lengths differ:")
    print(f"  filter=score   rewards {[round(r, 3) for r in by_score]}  -> trains on it")
    print(f"  filter=outcome rewards {[round(r, 3) for r in by_outcome]}  -> advantage 0")

    # A group that really does split: one right, three wrong. Both filters keep it.
    outcomes = [1.0, 0.0, 0.0, 0.0]
    shaped = [shaped_of(short_wrong, o) for o in outcomes]
    assert len(set(apply_filter(prompts, outcomes, shaped, "score"))) > 1
    assert len(set(apply_filter(prompts, outcomes, shaped, "outcome"))) > 1
    print("mixed group: both filters keep it")

    # Grouping does not assume the batch layout.
    assert groups_of(["a", "a", "b", "b", "b"]) == [(0, 2), (2, 5)]

    # The grader is a program, not a judge.
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("grader: MathEqual reads \\boxed{} and the last number")

    selftest_science_bar()
    print("selftest ok")


def selftest_science_bar() -> None:
    """The chapter Evaluation and chapter Over-optimization plumbing, on synthetic rows. This does not check
    the recipe's numbers -- there are none until it runs -- only that every
    check is wired to something real and reads the field it thinks it reads."""
    import random

    import whileai.simulations as wai

    rng = random.Random(0)

    def fake(p: float, n: int = 40, k: int = 4) -> list[dict]:
        holdout = [
            {"question": f"q{t}", "answer": f"#### {t}", "scenario_id": f"test-{t}"}
            for t in range(n)
        ]
        replies = [
            [f"\\boxed{{{t if rng.random() < p else t + 1}}}" + "." * rng.randint(0, 200)] * 1 * k
            for t in range(n)
        ]
        return graded_rows(holdout, replies)

    # chapter Evaluation, decontaminate: the holdout must be handed over prompt-keyed, or
    # `against=` reads nothing and the check silently passes. Assert it bites.
    train = [{"question": "shared prompt", "answer": "#### 1", "scenario_id": "train-0"}]
    kept, report = wai.decontaminate(
        train,
        against=[{"prompt": "shared prompt", "answer": "#### 1"}],
        fields=("question",),
    )
    assert report["n_contaminated"] == 1 and not kept, "decontaminate is not reading the prompt"
    print(f"decontaminate: catches a shared prompt ({report['n_contaminated']} dropped)")

    # chapter Evaluation, eval noise: three re-runs of the same model give a run_std.
    runs = [fake(0.35) for _ in range(EVAL_RUNS)]
    noise = wai.eval_variance(*runs)
    assert "run_std" in noise and noise["n_runs"] == EVAL_RUNS
    print(f"eval_variance: {EVAL_RUNS} re-runs -> run_std {float(noise['run_std']):.4f}")

    # chapter Over-optimization, proxy vs target: rows carry the shaped reward as a marker, so a
    # proxy that climbs while the target sits still is an over-optimized verdict.
    d = wai.delta_report(
        runs[0],
        fake(0.36),
        target="pass_at_1",
        run_std=float(noise["run_std"]),
        run_std_runs=int(noise["n_runs"]),
        proxy=PROXY,
    )
    assert "over_optimized" in d, "delta_report is not running the proxy check"
    assert d["run_std_df"] == EVAL_RUNS - 1, "the band is not carrying run_std's degrees of freedom"
    assert runs[0][0]["markers"].get("shaped_reward") is not None, "rows lost the proxy marker"
    print(
        f"delta_report(proxy={PROXY}): verdict {d['target_verdict']}, "
        f"over_optimized {d['over_optimized']}"
    )

    # chapter Over-optimization, hack scan: the top feature comes back named, not as a crash.
    top = top_hack_feature(runs[0])
    assert isinstance(top, str) and top
    print(f"hack_scan: top feature {top}")
    print(f"length: mean completion {mean_length(runs[0]):.1f} chars")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both", "base"], default="both")
    ap.add_argument(
        "--base-runs",
        type=int,
        default=EVAL_RUNS,
        help="re-runs of the untrained base that set the noise floor; with --arm base, "
        "only these run (no training) and the verdict is re-read against the new band",
    )
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=120)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--selftest", action="store_true", help="the filter, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai.simulations as wai
    from whileai.simulations.score.stats import noise_band

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    # chapter Evaluation: drop any train prompt that is a holdout prompt. `against=` reads
    # the eval texts from `prompt`/`answer`, so the holdout is handed over
    # prompt-keyed; passing it question-keyed silently finds nothing.
    train_tasks, decon = wai.decontaminate(
        train_tasks,
        against=[{"prompt": t["question"], "answer": t["answer"]} for t in holdout],
        fields=("question",),
    )
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train prompts dropped")

    arms = {"both": ["baseline", "recipe"], "base": []}.get(args.arm, [args.arm])
    filters = {"baseline": "score", "recipe": "outcome"}
    if args.base_runs < 2:  # two runs before a spread exists
        ap.error("--base-runs is the number of re-runs a run_std comes from, at least 2")

    # Start from what is already on disk, so `--arm recipe` refreshes one arm
    # instead of wiping the other one and the delta. Only a both-arm run moves
    # `verified` and the delta; a one-arm run says so and leaves them alone.
    results = json.loads((HERE / "results.json").read_text())
    results.update(
        {
            "recipe": HERE.name,
            "title": "Filter metric: phantom advantages under a shaped reward",
            "paper": "https://arxiv.org/abs/2609.13866",
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
    checks.update(
        {
            "decontaminated_dropped": int(decon.get("n_contaminated", 0)),
            "seed": args.seed,
            "pins": "torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0",
        }
    )
    checks.setdefault("length_after", {})
    arm_rows: dict[str, list[dict]] = {}
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""

    def record_base(base_runs: list[list[dict]]) -> None:
        # chapter Evaluation: the floor is the sample std over the re-runs, and it is an
        # estimate from `n_runs` draws, so the band on it is the t quantile
        # at df = n_runs - 1, not 1.96. Both numbers go to results.json.
        arm_rows["base"] = base_runs[0]
        noise = wai.eval_variance(*base_runs)
        run_std = float(noise["run_std"])
        checks["run_std"] = run_std
        checks["run_std_runs"] = int(noise["n_runs"])
        checks["length_before"] = mean_length(base_runs[0])
        band = noise_band(run_std, df=int(noise["n_runs"]) - 1)
        print(
            f"eval noise over {noise['n_runs']} base re-runs: run_std {run_std:.4f}, "
            f"band {band:.4f} (t at df={noise['n_runs'] - 1} x sqrt(2) x run_std)"
        )
        results["arms"]["base"] = {**summarize(base_runs[0]), "steps": 0, "gpu_minutes": 0}

    with modal.enable_output(), app.run():
        if args.arm == "base":
            out = eval_base.remote(holdout, args.base_runs, eval_samples=args.k)
            gpu_minutes += out["gpu_minutes"]
            record_base(out["base_runs"])
        for i, arm in enumerate(arms):
            out = run_arm.remote(
                arm,
                filters[arm],
                train_tasks,
                holdout,
                f"filter-metric-{arm}-{date.today().isoformat()}",
                steps=args.steps,
                prompts_per_step=args.prompts_per_step,
                eval_samples=args.k,
                eval_base=(i == 0),
                eval_runs=args.base_runs,
            )
            gpu_minutes += out["gpu_minutes"]
            run_url = out["run_url"] or run_url
            if out["base_runs"]:
                record_base(out["base_runs"])
            arm_rows[arm] = out["after_rows"]
            checks["length_after"][arm] = mean_length(out["after_rows"])
            checks["hack_scan_top"] = out["hack_scan_top"]
            results["arms"][arm] = {
                **summarize(out["after_rows"]),
                "steps": out["steps"],
                "gpu_minutes": round(out["gpu_minutes"], 1),
            }

    if "baseline" in arm_rows and "recipe" in arm_rows:
        d = wai.delta_report(
            arm_rows["baseline"],
            arm_rows["recipe"],
            target="pass_at_1",
            run_std=float(checks.get("run_std") or 0.0),
            run_std_runs=int(checks.get("run_std_runs") or EVAL_RUNS),
            # one training seed per arm: the report says unresolved (#356)
            train_runs={"before": [arm_rows["baseline"]], "after": [arm_rows["recipe"]]},
            proxy=PROXY,
        )
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "noise_band": d["noise_band"],
            "verdict": (
                "unresolved"
                if d["target_verdict"] == "unresolved"
                else "moved"
                if d["target_verdict"] == "moved"
                else "flat"
            ),
        }
        checks["train_seeds"] = {"baseline": 1, "recipe": 1}
        checks["over_optimized"] = bool(d["over_optimized"])
        results["verified"] = date.today().isoformat()
        results.pop("partial_run", None)
        print(wai.format_delta_report(d))
    elif args.arm == "base" and "delta" in results:
        # No arm was re-run, so the delta and its interval stand; only the
        # band they are read against moved. Same rule as check.py: moved
        # needs the interval off zero, the delta over the band, and a clean
        # proxy check.
        delta = results["delta"]
        lo, hi = delta.get("ci", [0.0, 0.0])
        band = noise_band(checks["run_std"], df=checks["run_std_runs"] - 1)
        clears = abs(float(delta["recipe_vs_baseline"])) >= band
        moved = clears and not (lo <= 0.0 <= hi) and not checks.get("over_optimized")
        delta["noise_band"] = band
        # and, as in check.py, one training seed per arm resolves nothing (#356)
        seeds = checks.get("train_seeds") or {"baseline": 1, "recipe": 1}
        resolved = min(seeds.values()) >= wai.simulations.defaults.MIN_TRAIN_SEEDS
        delta["verdict"] = ("moved" if moved else "flat") if resolved else "unresolved"
        checks["run_std_verified"] = date.today().isoformat()
        print(
            f"base re-evaluated {checks['run_std_runs']} times: delta "
            f"{float(delta['recipe_vs_baseline']):+.3f} [{lo:+.3f}, {hi:+.3f}] against band "
            f"{band:.3f} -> {delta['verdict']}"
        )
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"
        print(f"one arm only ({', '.join(arms)}): delta and verified left as they were")

    usd = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    if args.arm == "base":
        # The arms' cost stands; the re-evaluation adds to it.
        usd = round(float(results.get("usd") or 0.0) + usd, 2)
    else:
        results["run_url"] = run_url
    results["usd"] = usd
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
