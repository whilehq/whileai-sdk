"""Lesson 7 step 2: LoRA SFT on one A10G from the rows the lesson exported, with the before and after.

    modal run recipes/04-train/sft/train_modal.py --data train.jsonl --steps 10   # the wiring: under a minute of training
    modal run recipes/04-train/sft/train_modal.py --data train.jsonl              # 40 steps, about ten minutes end to end

What happens:

1. ``train.jsonl`` is what ``select(mode="sft").export("train.jsonl")`` wrote:
   one chat per line, the system prompt in, the tool schema on, the tool
   calls in the wire format. ``holdout.jsonl`` next to it carries the
   held-out tasks (lesson 5's split), one stand-in row per ask, and is the
   only thing either model is scored on.
2. The base model answers every held-out ask four times, three passes with
   three seeds: that is the noise floor (``eval_variance``). Each answer is
   two turns: the call, then the reply to what the fake world returned.
3. TRL's ``SFTTrainer`` with a PEFT LoRA adapter trains on the export for
   ``--steps`` steps. The chat template is applied here, with the tool
   schema, because ``SFTTrainer`` does not read the export's ``tools`` column.
4. The adapter answers the same asks once more, on the first base seed, so
   before and after are paired. Every row comes back to your laptop as
   ``holdout_rows.jsonl``; the judge, the noise floor and the paired
   comparison run there (``wiring.report``), which is lesson 7 step 3.

The adapter lands on the ``whileai-sft-runs`` volume under the run name.
One A10G is about $1.10 an hour; the default run is about ten minutes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wiring import (
    BASE_MODEL,
    BASE_PASSES,
    BASE_SEEDS,
    STEPS,
    load_export,
    load_holdout,
    plan,
    report,
)

VOLUME_ROOT = "/vol"
#: one A10G: the whole run, model load included, is about ten minutes on it.
GPU = "A10G"

app = modal.App("whileai-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    # The container imports this module too, so the plan and the loaders ride along.
    .add_local_python_source("wiring")
)

runs_volume = modal.Volume.from_name("whileai-sft-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse_turn(text: str) -> tuple[str, dict | None]:
    """What the model said and, when it emitted a well-formed ``<tool_call>``
    block, the call as ``{"name", "arguments"}``; a malformed block is text."""
    m = TOOL_CALL.search(text or "")
    if not m:
        return (text or "").strip(), None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return (text or "").strip(), None
    if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
        return (text or "").strip(), None
    outside = (text[: m.start()] + text[m.end() :]).strip()
    return outside, {"name": str(call.get("name")), "arguments": call["arguments"]}


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
)
def train(
    rows: list[dict],
    tools: list[dict],
    system_prompt: str,
    holdout: list[dict],
    plan: dict,
    run_name: str,
    base_model: str = BASE_MODEL,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    timing: dict[str, float] = {}
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    timing["load"] = round(time.time() - t0, 1)

    texts = [tok.apply_chat_template(r["messages"], tools=tools, tokenize=False) for r in rows]
    longest = max(len(tok(t)["input_ids"]) for t in texts)
    print(
        f"{len(rows)} training rows rendered; the longest is {longest} tokens (max_length {plan['sft']['max_length']})",
        flush=True,
    )
    print(
        f"{len(holdout)} held-out asks over {len({t['scenario_id'] for t in holdout})} tasks",
        flush=True,
    )

    sampling = plan["sampling"]

    def generate(prompts: list[str], n: int, seed: int) -> list[str]:
        torch.manual_seed(seed)
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                max_new_tokens=sampling["max_new_tokens"],
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id,
            )
        return tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def evaluate(tag: str, run: int, seed: int) -> list[dict]:
        """Every held-out ask, ``samples`` times, two turns each: the call,
        then the reply to what the world returned. Both arms go through this
        one function; the only difference is the adapter."""
        model.eval()
        n = sampling["samples"]
        out_rows: list[dict] = []
        for start in range(0, len(holdout), sampling["batch"]):
            chunk = holdout[start : start + sampling["batch"]]
            firsts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": t["prompt"]},
                    ],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in chunk
            ]
            replies = generate(firsts, n, seed + start)
            pending: list[tuple[int, list[dict]]] = []
            for i, task in enumerate(chunk):
                for k in range(n):
                    said, call = _parse_turn(replies[i * n + k])
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task["prompt"]},
                    ]
                    if call is None:
                        messages.append({"role": "assistant", "content": said})
                    else:
                        messages.append(
                            {"role": "assistant", "content": said, "tool_calls": [call]}
                        )
                        answer = (
                            task["tool_result"]
                            if call["name"] == "get_order"
                            else json.dumps({"error": "unknown tool"})
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "name": call["name"],
                                "content": answer or json.dumps({"status": "not_found"}),
                            }
                        )
                    row = {
                        "scenario_id": task["scenario_id"],
                        "prompt": task["prompt"],
                        "rollout_index": k,
                        "messages": messages,
                        "model_version": tag,
                        "policy_version": f"{base_model}-{tag}",
                        "run": run,
                        "seed": seed,
                    }
                    out_rows.append(row)
                    if call is not None:
                        pending.append((len(out_rows) - 1, messages))
            # Turn two, for every sample that called a tool: the reply.
            for p_start in range(0, len(pending), sampling["batch"] * n):
                batch = pending[p_start : p_start + sampling["batch"] * n]
                seconds = [
                    tok.apply_chat_template(
                        msgs, tools=tools, tokenize=False, add_generation_prompt=True
                    )
                    for _, msgs in batch
                ]
                finals = generate(seconds, 1, seed + start + p_start + 1)
                for (idx, _), text in zip(batch, finals):
                    said, _ = _parse_turn(text)
                    out_rows[idx]["messages"].append({"role": "assistant", "content": said})
        model.train()
        return out_rows

    all_rows: list[dict] = []
    for run, seed in enumerate(BASE_SEEDS[:BASE_PASSES], start=1):
        t = time.time()
        all_rows += evaluate("base", run, seed)
        timing[f"base_{run}"] = round(time.time() - t, 1)
        print(
            f"base pass {run} (seed {seed}): {len(holdout) * sampling['samples']} rows in {timing[f'base_{run}']}s",
            flush=True,
        )

    t = time.time()
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list([{"text": t} for t in texts]),
        peft_config=LoraConfig(**plan["lora"]),
        args=SFTConfig(output_dir=os.path.join(out_dir, "checkpoints"), **plan["sft"]),
        processing_class=tok,
    )
    result = trainer.train()
    timing["train"] = round(time.time() - t, 1)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(
        f"trained {plan['sft']['max_steps']} steps in {timing['train']}s: loss {losses[0]:.3f} -> {losses[-1]:.3f}",
        flush=True,
    )
    model = trainer.model
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    runs_volume.commit()

    t = time.time()
    all_rows += evaluate("trained", 1, BASE_SEEDS[0])
    timing["trained_1"] = round(time.time() - t, 1)
    print(f"trained pass (seed {BASE_SEEDS[0]}): {timing['trained_1']}s", flush=True)
    timing["total"] = round(time.time() - t0, 1)
    return {
        "rows": all_rows,
        "timing": timing,
        "gpu": GPU,
        "adapter": f"whileai-sft-runs:/{run_name}/adapter",
        "train_loss": [round(x, 4) for x in losses],
        "train_runtime": round(result.metrics.get("train_runtime", 0.0), 1),
        "longest_tokens": longest,
    }


@app.local_entrypoint()
def main(
    data: str = "train.jsonl",
    holdout: str = "holdout.jsonl",
    steps: int = STEPS,
    run_name: str = "lesson7-sft",
    base_model: str = BASE_MODEL,
    out: str = "holdout_rows.jsonl",
):
    try:
        from whileai.config import provenance

        print(provenance(), file=sys.stderr)
    except ImportError:
        pass  # `modal` as a tool runs this in its own environment; the container needs no whileai
    export = load_export(data)
    tasks = load_holdout(holdout)
    the_plan = plan(len(export["rows"]), steps)
    print(
        f"{len(export['rows'])} rows from {data}, {len(tasks)} held-out asks from {holdout}; "
        f"{steps} steps is {the_plan['epochs']} passes over the rows"
    )
    result = train.remote(
        rows=export["rows"],
        tools=export["tools"],
        system_prompt=export["system_prompt"],
        holdout=tasks,
        plan=the_plan,
        run_name=run_name,
        base_model=base_model,
    )
    with open(out, "w", encoding="utf-8") as fh:
        for row in result["rows"]:
            fh.write(json.dumps(row) + "\n")
    timing = result["timing"]
    print(
        f"{result['gpu']}: {timing['total']}s total, training {timing['train']}s; adapter at {result['adapter']}"
    )
    print(f"wrote {len(result['rows'])} rows to {out}")
    try:
        print(report(result["rows"]))
    except ImportError:
        # `modal` installed as a tool runs this entrypoint in its own
        # environment, where whileai may not be; the rows are on disk.
        print(f"whileai is not importable here; run: python wiring.py --report {out}")
