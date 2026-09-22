"""Both levers on a support agent's lookup behaviour, on one L40S.

    modal run --detach support_modal.py

Three stages, three containers, one volume:

1. ``search``  -- the base weights under each candidate harness on the train
   split; the gate picks the searched harness on the holdout-free evidence.
   Then the base under the deployed harness three times (the noise floor)
   and under the searched harness once (the ``harness`` cell, no GPU
   training at all).
2. ``train``   -- two LoRA SFT adapters on the same rows, same base, same
   steps: one under the deployed harness (``weights``, the busy engineer's
   default), one under the searched harness (``both``, the method).
3. ``evaluate`` -- the two trained cells on the same frozen holdout.

Nothing here is hosted by anyone but me: my Modal workspace, my volume, an
open base from the Hub, and a grader that is a program.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASE_MODEL = os.environ.get("SUPPORT_BASE", "Qwen/Qwen3-4B")
GPU = os.environ.get("SUPPORT_GPU", "L40S")
VOL = "/vol"

app = modal.App("support-lookup-levers")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.29.0", "trl==1.13.0", "peft==0.21.0", "datasets>=4.7.0", requirement())
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VLLM_USE_V1": "1",
        }
    )
    .add_local_file(str(HERE / "support.py"), "/root/support.py")
    .add_local_file(str(HERE / "split.json"), "/root/split.json")
)

runs = modal.Volume.from_name("support-lookup-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)

HOLDOUT_N = int(os.environ.get("SUPPORT_HOLDOUT", "320"))
TRAIN_GATE_N = int(os.environ.get("SUPPORT_GATE", "160"))
K = int(os.environ.get("SUPPORT_K", "2"))
MAX_NEW = 160


def _load_split():
    d = json.load(open("/root/split.json"))
    return d["train"], d["holdout"]


def _stratified(tasks: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Keep every tier and both targets represented: a behaviour test has to
    be able to fail, which sampling traffic proportionally does not give you."""
    import random

    rng = random.Random(seed)
    buckets: dict[tuple, list] = {}
    for t in tasks:
        buckets.setdefault((t["target"], t["tier"]), []).append(t)
    for v in buckets.values():
        rng.shuffle(v)
    out, i = [], 0
    keys = sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(out) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out


def _play(llm, tok, tasks, harness, *, k, seed, lora=None, max_new=MAX_NEW):
    """Play one cell. Every row is stamped with its harness and model so
    ``wai.harness.attribute`` can read the grid later."""
    import support
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    prompts, index = [], []
    for t in tasks:
        text = support.render(t, harness, tok)
        for j in range(k):
            prompts.append(text)
            index.append((t, j))
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_new, seed=seed, n=1)
    kw = {}
    if lora:
        kw["lora_request"] = LoRARequest("arm", 1, lora)
    outs = llm.generate(prompts, sp, **kw)

    rows = []
    for (t, _j), o in zip(index, outs):
        text = o.outputs[0].text
        g = support.grade(t, text)
        rows.append(
            {
                "task_id": t["task_id"],
                "prompt": t["prompt"],
                "reward": g["reward"],
                "markers": {k2: v for k2, v in g["markers"].items() if v is not None},
                "target": t["target"],
                "tier": t["tier"],
                "domain": t["domain"],
                "ask_family": t["ask_family"],
                "final_text": text[:2000],
                "harness": {"label": harness, "model": "trained" if lora else "base"},
            }
        )
    return rows


def _mean(rows, marker=None):
    vals = (
        [r["markers"][marker] for r in rows if marker in r["markers"]]
        if marker
        else [r["reward"] for r in rows]
    )
    return sum(vals) / len(vals) if vals else float("nan")


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
    timeout=60 * 90,
)
def search():
    from transformers import AutoTokenizer
    from vllm import LLM

    import whileai as wai

    sys.path.insert(0, "/root")
    import support

    train, holdout = _load_split()
    gate_tasks = _stratified(train, TRAIN_GATE_N, seed=1)
    hold = _stratified(holdout, HOLDOUT_N, seed=2)
    Path(f"{VOL}/holdout.json").write_text(json.dumps(hold))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=32,
        seed=0,
    )

    out = {"candidates": {}, "cells": {}}
    print("=== search: every candidate harness on the train split, base weights")
    for label in support.HARNESSES:
        rows = _play(llm, tok, gate_tasks, label, k=K, seed=11)
        out["candidates"][label] = {
            "right_first_action": _mean(rows, "right_first_action"),
            "looks_up": _mean(rows, "looks_up_when_it_should"),
            "stays_in_scope": _mean(rows, "stays_in_scope"),
            "called_a_tool": _mean(rows, "called_a_tool"),
        }
        Path(f"{VOL}/gate_{label}.json").write_text(json.dumps(rows))
        print(f"  {label:22s} {out['candidates'][label]}")

    # the gate: a candidate has to clear the deployed harness on the train
    # split by an interval that excludes zero, or the deployed one stays
    base_rows = json.loads(Path(f"{VOL}/gate_00_deployed.json").read_text())
    picked, best = "00_deployed", None
    for label in support.HARNESSES:
        if label == "00_deployed":
            continue
        cand = json.loads(Path(f"{VOL}/gate_{label}.json").read_text())
        rep = wai.compare(base_rows, cand, target="marker:right_first_action")
        m = rep["metrics"]["marker:right_first_action"]
        print(f"  gate {label}: delta {m['delta']:+.3f} {m['ci95']} -> {m['verdict']}")
        out["candidates"][label]["gate"] = {
            "delta": m["delta"],
            "ci95": list(m["ci95"]),
            "verdict": m["verdict"],
        }
        if m["ci95"][0] > 0 and (best is None or m["delta"] > best):
            picked, best = label, m["delta"]
    out["searched_harness"] = picked
    print(f"=== searched harness: {picked}")

    print("=== base under the deployed harness, three times (the noise floor)")
    for i, seed in enumerate([101, 202, 303]):
        rows = _play(llm, tok, hold, "00_deployed", k=K, seed=seed)
        Path(f"{VOL}/cell_neither_run{i}.json").write_text(json.dumps(rows))
        print(f"  run {i}: right_first_action {_mean(rows, 'right_first_action'):.3f}")

    print("=== base under the searched harness (the harness lever, no training)")
    rows = _play(llm, tok, hold, picked, k=K, seed=101)
    Path(f"{VOL}/cell_harness.json").write_text(json.dumps(rows))
    print(f"  harness cell: right_first_action {_mean(rows, 'right_first_action'):.3f}")

    Path(f"{VOL}/search.json").write_text(json.dumps(out))
    runs.commit()
    return out


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
    timeout=60 * 90,
)
def train(harness: str, arm: str, steps: int = 0, seed: int = 17):
    """LoRA SFT on the reference agent's first action, rendered under
    ``harness``. Same rows, same base, same steps for both arms."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sys.path.insert(0, "/root")
    import support

    train_rows, _ = _load_split()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    recs = []
    for t in train_rows:
        prompt = support.render(t, harness, tok)
        recs.append({"prompt": prompt, "completion": support.target_completion(t) + tok.eos_token})
    ds = Dataset.from_list(recs)
    print(f"[{arm}] {len(recs)} training rows under {harness}")

    out_dir = f"{VOL}/{arm}"
    wanted = {
        "output_dir": out_dir,
        "num_train_epochs": 1,
        "max_steps": steps or -1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 5,
        "save_strategy": "no",
        "bf16": True,
        "max_length": 6144,
        "completion_only_loss": True,
        "report_to": [],
        "seed": seed,
        # The recipes' "checkpointing OFF" rule is scoped to trainers that
        # GENERATE during training, where it corrupts Qwen3 generation. SFT
        # never generates, and a 4B model with 5k-token prompts OOMs a 48GB
        # card without it. Off here costs the run; on here costs nothing.
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    # TRL renames these between versions; keep what this one takes rather
    # than dying on a keyword three minutes into a paid container.
    allowed = {f.name for f in dataclasses.fields(SFTConfig)}
    dropped = sorted(set(wanted) - allowed)
    if dropped:
        print(f"[{arm}] SFTConfig does not take {dropped}; dropped")
    cfg = SFTConfig(**{k: v for k, v in wanted.items() if k in allowed})

    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = SFTTrainer(model=BASE_MODEL, args=cfg, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.model.save_pretrained(f"{out_dir}/adapter")
    tok.save_pretrained(f"{out_dir}/adapter")
    hist = [h for h in trainer.state.log_history if "loss" in h]
    Path(f"{VOL}/{arm}_train.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "harness": harness,
                "rows": len(recs),
                "steps": trainer.state.global_step,
                "loss": hist,
            }
        )
    )
    runs.commit()
    del trainer
    torch.cuda.empty_cache()
    print(f"[{arm}] adapter at {out_dir}/adapter, {trainer_steps(hist)} logged points")
    return f"{out_dir}/adapter"


def trainer_steps(hist):
    return len(hist)


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
    timeout=60 * 90,
)
def evaluate(searched: str):
    from transformers import AutoTokenizer
    from vllm import LLM

    sys.path.insert(0, "/root")

    hold = json.loads(Path(f"{VOL}/holdout.json").read_text())
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=32,
        seed=0,
    )
    cells = [
        ("harness", searched, None),
        ("weights", "00_deployed", f"{VOL}/weights/adapter"),
        ("both", searched, f"{VOL}/both/adapter"),
    ]
    for arm, harness, lora in cells:
        rows = _play(llm, tok, hold, harness, k=K, seed=101, lora=lora)
        Path(f"{VOL}/cell_{arm}.json").write_text(json.dumps(rows))
        print(
            f"  {arm:8s} harness={harness:14s} model={'trained' if lora else 'base':7s} "
            f"right_first_action {_mean(rows, 'right_first_action'):.3f}  "
            f"stays_in_scope {_mean(rows, 'stays_in_scope'):.3f}"
        )
    runs.commit()
    return "ok"


# When no candidate clears the gate, the gate's answer is "keep what you
# serve" -- a result, and the one this run got. But one harness level makes
# the 2x2 a 2x1, so the second level is the best candidate on the train
# split, and every table says the gate did not pass. This is the fallback
# the harness-and-weights recipe documents.
FALLBACK = os.environ.get("SUPPORT_FALLBACK_HARNESS", "01_skills")


@app.local_entrypoint()
def main(stage: str = "all"):
    print(provenance(), file=sys.stderr)
    searched = os.environ.get("SUPPORT_SEARCHED", "")
    if stage in ("all", "search"):
        out = search.remote()
        searched = out["searched_harness"]
        if searched == "00_deployed":
            print(f"gate cleared nothing; second level falls back to {FALLBACK}")
            searched = FALLBACK
    searched = searched or FALLBACK
    if stage in ("all", "arms"):
        # both arms at once: two L40S for ten minutes beats one for twenty
        a = train.spawn("00_deployed", "weights")
        b = train.spawn(searched, "both")
        print("weights:", a.get())
        print("both:   ", b.get())
        evaluate.remote(searched)
    print("done; second harness level:", searched)
