"""Endpoint SFT: keep the two ends of the reasoning trace, drop the middle.

    python recipe.py --selftest   # the truncation rule, offline: no GPU, no key, no network
    python recipe.py --plan       # the same rule on the real dataset, CPU only: picks n, prints what it drops
    python recipe.py              # both arms on Modal, writes results.json
    python recipe.py --arm recipe --steps 150

Standard reasoning SFT trains on the whole machine-written trajectory. The
paper's attention and ablation studies say the middle of that trajectory is
weakly attended and causally cheap: cut it out and the answer barely changes.
So it proposes E-SFT, which trains on the endpoints instead.

Split the trace into steps at blank lines, keep the first ``n`` steps and the
last ``n`` steps, drop the ``total - 2n`` steps in between, and pick ``n`` so
that roughly 20% of the dataset's trace tokens go away (the paper's §C.3
heuristic: n=100 on s1K-1.1 at 234 steps average, n=200 on OpenThoughts3 at
461). That is the one change. Same problems, same model, same learning rate,
same number of optimizer steps, same prompt masking; only the target text is
shorter.

Shape of the run:
  1. data():     OpenR1-Math-220k traces to train on, MATH-500 held out,
                 decontaminated against it before anything is trained (chapter Evaluation)
  2. run_arm():  TRL SFTTrainer + LoRA on Modal, one arm per call
  3. evaluate(): the same 64 held-out problems, k samples each, graded by
                 MathEqual against the public gold answer
  4. results.json + the paired delta (wai.delta_report) and the run page
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Callable
from datetime import date
from importlib.metadata import version
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Instruction Tuning Instruction tuning"  # what SFT learns from, and the prompt mask
# SFT has no per-rollout reward to point at, but it does have a form it
# teaches: a closed <think> block ending in \boxed{}. That form is what the
# likelihood objective credits whether or not the answer is right, so it is
# the proxy, and pass@1 is the target (Lambert 2025, chapter Over-optimization).
PROXY = "marker:trace_form"
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

# The paper's own heuristic: choose the retained step count so that about this
# much of the dataset's trace tokens is removed. §C.3, and Figure B says
# performance is insensitive to the exact value inside a broad band.
TARGET_DROP = 0.20

SYSTEM = "Solve the problem. Put the final answer in \\boxed{}."


# --------------------------------------------------------------------------
# The one change. Pure text functions, no torch and no network, so `--selftest`
# runs them here and the Modal container imports this same file.
# --------------------------------------------------------------------------


def messages_for(problem: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": problem}]


def split_steps(trace: str) -> list[str]:
    """Reasoning steps are the blank-line-separated blocks of the trace.

    The paper splits trajectories at double newlines and checks in §B.2 that
    those positions are semantic breakpoints. Empty blocks are dropped so that
    a run of blank lines does not count as several steps.
    """
    return [s for s in trace.split("\n\n") if s.strip()]


def endpoint_trace(trace: str, n: int) -> str:
    """Keep the first ``n`` steps and the last ``n`` steps; drop the middle.

    A trace with 2n steps or fewer comes back untouched, which is the paper's
    rule: there is no middle to remove. The kept steps are rejoined with the
    same blank line they were split on and nothing is inserted at the seam --
    no ellipsis, no marker. The paper does not describe one, and a marker
    would be a second change on top of the one being tested.
    """
    steps = split_steps(trace)
    if n <= 0 or len(steps) <= 2 * n:
        return "\n\n".join(steps)
    return "\n\n".join(steps[:n] + steps[-n:])


def step_lengths(traces: list[str], length: Callable[[str], int]) -> list[list[int]]:
    """Token count of every step of every trace, measured once.

    ``length`` counts tokens (the base model's tokenizer on the GPU and in
    ``--plan``, ``str.split`` in the offline selftest). Everything below reads
    these counts instead of re-tokenizing, which is what keeps the search for
    ``n`` down to seconds: the trace text itself is never tokenized again.
    """
    return [[length(s) for s in split_steps(t)] for t in traces]


def drop_fraction(lengths: list[list[int]], n: int) -> float:
    """Share of the dataset's step tokens that keeping ``n`` at each end drops."""
    total = sum(sum(steps) for steps in lengths)
    if total <= 0:
        return 0.0
    kept = sum(
        sum(steps) if n <= 0 or len(steps) <= 2 * n else sum(steps[:n]) + sum(steps[-n:])
        for steps in lengths
    )
    return (total - kept) / total


def choose_n(
    lengths: list[list[int]], target: float = TARGET_DROP, max_n: int | None = None
) -> tuple[int, float]:
    """The retained step count whose token drop lands closest to ``target``.

    One number for the whole dataset, not one per trace: that is how §C.3 sets
    it (n=100 for s1K-1.1 at 234 steps average, n=200 for OpenThoughts3 at 461,
    both about 20% of tokens). Larger ``n`` keeps more, so the drop falls as
    ``n`` rises; the search walks ``n`` up, stops at the first value at or
    below the target, and returns whichever of the two neighbours is nearer.
    """
    ceiling = max_n or max((len(steps) for steps in lengths), default=1)
    previous: tuple[int, float] | None = None
    for n in range(1, ceiling + 1):
        drop = drop_fraction(lengths, n)
        if drop <= target:
            if previous is None:
                return n, drop
            return (n, drop) if abs(drop - target) <= abs(previous[1] - target) else previous
        previous = (n, drop)
    return previous or (ceiling, 0.0)


def trace_form(text: str) -> int:
    """1 when the completion has the shape SFT was shown: a closed thinking
    block and a boxed answer at the end. The proxy (chapter Over-optimization) -- it says the
    model learned the form, not that it got the answer right."""
    return int(
        "<think>" in text and "</think>" in text and "\\boxed{" in text.split("</think>")[-1]
    )


# --------------------------------------------------------------------------
# Grading. A program against the public gold answer, never a judge.
# --------------------------------------------------------------------------


def outcome_of(text: str, gold: str) -> float:
    """1.0 when the final answer matches the MATH-500 gold, else 0.0.
    `MathEqual` is Math-Verify."""
    from whileai.simulations.verify import MathEqual

    row = {
        "prompt": "",
        "final_text": text,
        "privileged": {"reference": gold},
        "scenario_id": "x",
        "rollout_index": 0,
    }
    return 1.0 if MathEqual()(row).get("reward") == 1 else 0.0


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at`, `delta_report` and `hack_scan` read:
    binary `reward`, one row per sample, grouped by task, with the proxy
    marker beside it."""
    rows: list[dict] = []
    for task, texts in zip(holdout, replies):
        gold = task["answer"]
        for i, text in enumerate(texts):
            rows.append(
                {
                    "prompt": task["problem"],
                    "final_text": text,
                    "reward": outcome_of(text, gold),
                    "markers": {"trace_form": trace_form(text)},
                    "scenario_id": task["scenario_id"],
                    "rollout_index": i,
                    "privileged": {"reference": gold},
                }
            )
    return rows


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


# --------------------------------------------------------------------------
# Data. Streamed, so the container pulls a few hundred rows and not 220k.
# --------------------------------------------------------------------------


def usable_trace(row: dict) -> str | None:
    """The first generation the dataset's own math verifier marked correct.

    OpenR1-Math-220k ships several R1 trajectories per problem with a parallel
    list of booleans from math-verify. Training on a trajectory that ends in
    the wrong answer would teach the wrong thing, in both arms equally, so
    take a checked one or skip the row.
    """
    generations = row.get("generations") or []
    correct = row.get("correctness_math_verify") or []
    for text, ok in zip(generations, correct):
        if ok and isinstance(text, str) and text.strip():
            return text
    return None


def load_traces(
    n_train: int,
    min_tokens: int,
    max_tokens: int,
    min_steps: int,
    length: Callable[[str], int],
    scan_limit: int = 20000,
) -> list[dict]:
    """Train rows: problem, a verified reasoning trace, the gold answer.

    The window keeps traces that fit the training context whole, so the
    baseline arm is never truncated from the right by the trainer -- if it
    were, "full trace" would silently mean "the trace minus its answer" and
    the comparison would be against a straw man. ``min_steps`` keeps traces
    with a middle worth removing.
    """
    from datasets import load_dataset

    stream = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)
    rows: list[dict] = []
    for seen, row in enumerate(stream):
        if seen >= scan_limit or len(rows) >= n_train:
            break
        trace = usable_trace(row)
        if trace is None:
            continue
        if len(split_steps(trace)) < min_steps:
            continue
        if not min_tokens <= length(trace) <= max_tokens:
            continue
        rows.append(
            {
                "prompt": row["problem"],
                "trace": trace,
                "answer": row["answer"],
                "scenario_id": f"train-{len(rows)}",
            }
        )
    return rows


def load_holdout(n_holdout: int, seed: int) -> list[dict]:
    """MATH-500: 500 public problems with gold answers, a different corpus
    from the training traces and held out by construction."""
    from datasets import load_dataset

    test = load_dataset("HuggingFaceH4/MATH-500", split="test").shuffle(seed=seed)
    return [
        {
            "problem": r["problem"],
            "answer": r["answer"],
            "prompt": r["problem"],
            "scenario_id": r["unique_id"],
        }
        for r in test.select(range(min(n_holdout, len(test))))
    ]


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/04-train/grpo/train_modal.py.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-endpoint-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        requirement(),
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _sample(model, tokenizer, problems, *, n, max_new_tokens, batch=8):
    """`n` replies per problem, batched, at the paper's eval temperature."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    model.eval()
    tokenizer.padding_side = "left"
    out: list[list[str]] = []
    texts = [
        tokenizer.apply_chat_template(messages_for(p), tokenize=False, add_generation_prompt=True)
        for p in problems
    ]
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.6,
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


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    endpoint: bool,
    train_rows: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    epochs: float = 1.0,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_length: int = 4096,
    lora_rank: int = 32,
    eval_samples: int = 4,
    eval_new_tokens: int = 1024,
    eval_base: bool = False,
    steps: int = 0,
) -> dict:
    """One arm: optionally eval the base model EVAL_RUNS times, train, eval."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        EVAL_RUNS,
        TARGET_DROP,
        choose_n,
        endpoint_trace,
        graded_rows,
        mean_length,
        messages_for,
        split_steps,
        step_lengths,
    )

    import whileai.simulations as wai

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    def token_length(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    # THE ONE CHANGE. Both arms see the same problems and the same number of
    # examples; the recipe arm's target text loses its middle steps. `n` is
    # chosen on this dataset, by tokens, the way §C.3 chooses it.
    traces = [r["trace"] for r in train_rows]
    n_steps, dropped = (0, 0.0)
    if endpoint:
        n_steps, dropped = choose_n(step_lengths(traces, token_length), TARGET_DROP)
        print(f"E-SFT: keeping the first and last {n_steps} steps, {dropped:.1%} of tokens dropped")
    targets = [endpoint_trace(t, n_steps) if endpoint else t for t in traces]
    train_tokens = sum(token_length(t) for t in targets)
    mean_steps = statistics.fmean(len(split_steps(t)) for t in traces)

    config = {
        "arm": arm,
        "endpoint_sft": endpoint,
        "retained_steps_each_end": n_steps,
        "token_drop": round(dropped, 4),
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "lora_rank": lora_rank,
        "train_rows": len(train_rows),
        "train_target_tokens": train_tokens,
        "mean_steps_per_trace": round(mean_steps, 1),
        "holdout": len(holdout),
        "gpu": DEFAULT_GPU,
        "grader": "MathEqual against the MATH-500 gold answer",
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        run = wai.training_run(
            run_name,
            base_model=base_model,
            trainer="trl-sft-lora",
            config=config,
        )
        print(f"dashboard: {run.url}")

    problems = [t["problem"] for t in holdout]
    # The base is evaluated EVAL_RUNS times, not once. The spread across those
    # re-runs is the eval's own noise, and a delta smaller than it is not a
    # result (Lambert 2025, chapter Evaluation). Only the first arm pays for this.
    base_runs = []
    if eval_base:
        for i in range(EVAL_RUNS):
            replies = _sample(
                model, tokenizer, problems, n=eval_samples, max_new_tokens=eval_new_tokens
            )
            rows = graded_rows(holdout, replies)
            base_runs.append(rows)
            print(f"base run {i + 1}/{EVAL_RUNS}: {wai.pass_at(rows)}")

    # A prompt-completion dataset, so TRL masks the prompt and takes the loss
    # on the trace alone -- the book's instruction-tuning rule (chapter Instruction Tuning): the
    # model is not learning to predict the question.
    dataset = Dataset.from_list(
        [
            {
                "prompt": messages_for(row["prompt"]),
                "completion": [{"role": "assistant", "content": target}],
            }
            for row, target in zip(train_rows, targets)
        ]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    sft = SFTConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        num_train_epochs=epochs,
        max_steps=steps or -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=max_length,
        packing=False,
        completion_only_loss=True,
        bf16=True,
        # On, unlike the GRPO recipes next door: this trainer never generates
        # while it trains, so the Qwen3 generation corruption that rules it
        # out there cannot happen here, and 4k-token sequences need the memory.
        gradient_checkpointing=True,
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
    trainer = SFTTrainer(
        model=model,
        args=sft,
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
    trained_steps = int(trainer.state.global_step)

    replies = _sample(
        trainer.model, tokenizer, problems, n=eval_samples, max_new_tokens=eval_new_tokens
    )
    after_rows = graded_rows(holdout, replies)
    after = wai.pass_at(after_rows)
    print(f"{arm}: {after}")

    # What separates a rewarded rollout from an unrewarded one (chapter Over-optimization). SFT
    # has no per-rollout training reward to scan, so this runs on the arm's
    # graded holdout rollouts: the features that track being marked right.
    # Nothing is endorsed -- the reward is the answer being correct, and any
    # surface feature that predicts it is the thing to be suspicious of.
    scan = wai.hack_scan(after_rows) if after_rows else {}
    top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
    hack_scan_top = top.get("name", "") if isinstance(top, dict) else str(top)
    print(f"{arm} hack scan: top feature {hack_scan_top or 'none above the floor'}")

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
        "endpoint_sft": endpoint,
        "pass_at_1": after.pass_at_1,
        "gpu_minutes": gpu_minutes,
        "steps": trained_steps,
        "retained_steps_each_end": n_steps,
        "token_drop": round(dropped, 4),
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    return {
        "arm": arm,
        "base_runs": base_runs,
        "after_rows": after_rows,
        "gpu_minutes": gpu_minutes,
        "steps": trained_steps,
        "length_after": mean_length(after_rows),
        "hack_scan_top": hack_scan_top,
        "retained_steps_each_end": n_steps,
        "token_drop": round(dropped, 4),
        "train_target_tokens": train_tokens,
        "run_url": summary.get("run_url", ""),
    }


# --------------------------------------------------------------------------
# Local: orchestration, the plan, the selftest, results.json.
# --------------------------------------------------------------------------


def summarize(rows: list[dict]) -> dict:
    import whileai.simulations as wai

    p = wai.pass_at(rows)
    return {"score": p.pass_at_1, "ci": list(p.ci95 or (0.0, 0.0)), "pass_at_k": p.pass_at_k}


def words(text: str) -> int:
    """A tokenizer-free stand-in for token count, for the offline selftest."""
    return len(text.split())


def tokenizer_length(model: str = BASE_MODEL) -> Callable[[str], int]:
    """Token count under the base model's tokenizer.

    The train-set window and `n` are both measured in tokens, and they have to
    be measured with the same tokenizer the trainer will use, or "3500 tokens"
    means one thing here and another on the GPU. Loading a tokenizer needs no
    torch and no GPU.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    def length(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return length


def selftest() -> None:
    """The truncation rule on hand-written traces. No GPU, no key, no network."""
    trace = "\n\n".join(f"step {i}" for i in range(10))
    assert split_steps(trace) == [f"step {i}" for i in range(10)]
    assert split_steps("a\n\n\n\nb") == ["a", "b"], "blank runs are one break, not two"

    kept = endpoint_trace(trace, 2)
    assert split_steps(kept) == ["step 0", "step 1", "step 8", "step 9"], kept
    assert endpoint_trace(trace, 5) == trace, "2n = total: nothing to remove"
    assert endpoint_trace(trace, 9) == trace, "n past the end leaves the trace alone"
    assert endpoint_trace(trace, 0) == trace

    # The ends are what carry the thinking tags and the answer, so both
    # survive the cut. This is the property the recipe rests on.
    r1 = "<think>\n\nfirst\n\n" + "\n\n".join(f"middle {i}" for i in range(40))
    r1 += "\n\nlast\n\n</think>\n\nThe answer is \\boxed{42}."
    cut = endpoint_trace(r1, 2)
    assert cut.startswith("<think>") and "</think>" in cut and "\\boxed{42}" in cut
    assert "middle 20" not in cut
    assert trace_form(cut) == 1 and trace_form("no tags here") == 0

    # choose_n walks n up until the drop falls to the target and takes the
    # nearer neighbour. Uniform steps make the arithmetic checkable by hand:
    # 20 steps of one word each, so keeping 2n of them drops (20 - 2n) / 20.
    flat = step_lengths(["\n\n".join(["w"] * 20)] * 5, words)
    assert flat[0] == [1] * 20
    n, drop = choose_n(flat, 0.20)
    assert (n, round(drop, 3)) == (8, 0.2), (n, drop)
    assert drop_fraction(flat, 10) == 0.0, "keeping everything drops nothing"
    assert drop_fraction(flat, 5) == 0.5
    # Monotone in n: more retained steps can never drop more tokens.
    drops = [drop_fraction(flat, i) for i in range(1, 11)]
    assert drops == sorted(drops, reverse=True)

    # Mixed lengths: the drop is by token mass, not by trace count, so one
    # long trace pulls the choice the way §C.3's dataset-wide rule intends,
    # and the short traces come out untouched at the n it picks.
    mixed_traces = ["\n\n".join(["w"] * 100)] + ["\n\n".join(["w"] * 10)] * 4
    mixed = step_lengths(mixed_traces, words)
    n_mixed, drop_mixed = choose_n(mixed, 0.20)
    assert 0.15 <= drop_mixed <= 0.25, (n_mixed, drop_mixed)
    assert endpoint_trace(mixed_traces[1], n_mixed) == mixed_traces[1]
    print(f"uniform traces: n={n}, drop={drop:.1%}; mixed: n={n_mixed}, drop={drop_mixed:.1%}")

    # The grader is a program, not a judge.
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("grader: MathEqual, decided by Math-Verify")
    print("selftest ok")


def plan(args) -> None:
    """Apply the rule to the real dataset on the CPU: what n comes out, how
    much it drops, what the arms will actually train on. No GPU, no training.
    """
    token_length = tokenizer_length()
    rows = load_traces(args.n_train, args.min_tokens, args.max_tokens, args.min_steps, token_length)
    traces = [r["trace"] for r in rows]
    steps = [len(split_steps(t)) for t in traces]
    tokens = [token_length(t) for t in traces]
    print(f"{len(rows)} traces inside {args.min_tokens}-{args.max_tokens} tokens")
    print(
        f"steps per trace: min {min(steps)}, median {statistics.median(steps):.0f}, "
        f"mean {statistics.fmean(steps):.1f}, max {max(steps)}"
    )
    print(
        f"tokens per trace: median {statistics.median(tokens):.0f}, "
        f"mean {statistics.fmean(tokens):.0f}, total {sum(tokens)}"
    )
    lengths = step_lengths(traces, token_length)
    n, drop = choose_n(lengths, TARGET_DROP)
    print(f"chosen n (steps kept at each end): {n} -> {drop:.1%} of trace tokens dropped")
    for probe in (n - 2, n, n + 2, n + 5):
        if probe > 0:
            print(f"  n={probe:3d}: {drop_fraction(lengths, probe):.1%} dropped")
    cut = [endpoint_trace(t, n) for t in traces]
    untouched = sum(1 for a, b in zip(traces, cut) if a == b)
    print(f"traces with no middle to remove (<= 2n steps): {untouched} of {len(traces)}")
    print(
        f"baseline trains on {sum(tokens)} target tokens, "
        f"recipe on {sum(token_length(t) for t in cut)}"
    )


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-holdout", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0, help="cap optimizer steps (0 = a full epoch)")
    ap.add_argument("--batch-size", type=int, default=8, help="sequences per optimizer step")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--min-tokens", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--min-steps", type=int, default=24)
    ap.add_argument("--eval-new-tokens", type=int, default=1024)
    ap.add_argument("--selftest", action="store_true", help="the truncation rule, offline")
    ap.add_argument("--plan", action="store_true", help="the rule on the real data, CPU only")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.plan:
        plan(args)
        return

    import whileai.simulations as wai

    train_rows = load_traces(
        args.n_train, args.min_tokens, args.max_tokens, args.min_steps, tokenizer_length()
    )
    holdout = load_holdout(args.n_holdout, args.seed)
    # OpenR1-Math-220k is built from NuminaMath, which draws on the MATH
    # training set, and MATH-500 is a slice of the MATH test set. Nothing says
    # a problem cannot appear in both, so this is measured, not assumed
    # (Lambert 2025, chapter Evaluation), and the count goes in the Checks table.
    train_rows, decon = wai.decontaminate(train_rows, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]
    endpoint = {"baseline": False, "recipe": True}

    # Start from what is already on disk, so `--arm recipe` refreshes one arm
    # instead of wiping the other one and the delta. Only a both-arm run moves
    # `verified` and the delta; a one-arm run says so and leaves them alone.
    results = json.loads((HERE / "results.json").read_text())
    results.update(
        {
            "recipe": HERE.name,
            "title": "Endpoint SFT: keep the two ends of the reasoning trace, drop the middle",
            "paper": "https://arxiv.org/abs/2609.07103",
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
    arm_rows: dict[str, list[dict]] = {}
    run_std = 0.0
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""

    with modal.enable_output(), app.run():
        for i, arm in enumerate(arms):
            out = run_arm.remote(
                arm,
                endpoint[arm],
                train_rows,
                holdout,
                f"endpoint-sft-{arm}-{date.today().isoformat()}",
                epochs=args.epochs,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                eval_samples=args.k,
                eval_new_tokens=args.eval_new_tokens,
                eval_base=(i == 0),
            )
            gpu_minutes += out["gpu_minutes"]
            run_url = out["run_url"] or run_url
            if out["base_runs"]:
                noise = wai.eval_variance(*out["base_runs"])
                run_std = float(noise["run_std"])
                print(f"eval noise over {len(out['base_runs'])} base runs: run_std {run_std:.4f}")
                arm_rows["base"] = out["base_runs"][0]
                results["arms"]["base"] = {
                    **summarize(out["base_runs"][0]),
                    "steps": 0,
                    "gpu_minutes": 0,
                }
                checks["run_std"] = run_std
                checks["run_std_runs"] = int(noise["n_runs"])
                checks["length_before"] = mean_length(out["base_runs"][0])
            arm_rows[arm] = out["after_rows"]
            results["arms"][arm] = {
                **summarize(out["after_rows"]),
                "steps": out["steps"],
                "gpu_minutes": round(out["gpu_minutes"], 1),
            }
            checks["length_after"][arm] = out["length_after"]
            checks["hack_scan_top"] = out["hack_scan_top"]
            if arm == "recipe":
                checks["retained_steps_each_end"] = out["retained_steps_each_end"]
                checks["token_drop"] = out["token_drop"]

    if "baseline" in arm_rows and "recipe" in arm_rows:
        # run_std makes "moved" mean bigger than the eval's own re-run noise;
        # proxy names what SFT actually optimizes, the shape of the trace, so
        # a run that only taught the form shows up as over-optimized (chapter Over-optimization).
        d = wai.delta_report(
            arm_rows["baseline"],
            arm_rows["recipe"],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks.get("run_std_runs") or EVAL_RUNS),
            # one training seed per arm: the report says unresolved (#356)
            train_runs={"before": [arm_rows["baseline"]], "after": [arm_rows["recipe"]]},
            proxy=PROXY,
        )
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": (
                "unresolved"
                if d["target_verdict"] == "unresolved"
                else "moved"
                if d["target_verdict"] == "moved"
                else "flat"
            ),
        }
        checks["train_seeds"] = {"baseline": 1, "recipe": 1}
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
