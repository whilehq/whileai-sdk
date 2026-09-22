"""Does honouring the loss mask still lose when you give it four times the rows?

Three LoRA SFT arms on the same graded rows, the same base, the same
hyperparameters. The only difference is the file `wai.export(format="trl")`
wrote, and every one of the three is a first-class export this release:

  assistant  mask_mode="assistant"           messages rows, TRL trains every token
  final      mask_mode="final"               prompt/completion, last assistant turn
  unroll     mask_mode="final", unroll=True  every assistant turn, its own row

Two training seeds per arm. Base evaluated three times first for the noise
floor. Everything runs on my own Modal account; nothing touches While hosting.

    modal run train_modal.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
APP = "wai-mask-turns"
ARMS = ("assistant", "final", "unroll")
TRAIN_SEEDS = (0, 1)
N_EVAL = 120

app = modal.App(APP)

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
    .add_local_file(str(DATA / "train.assistant.jsonl"), "/root/train.assistant.jsonl")
    .add_local_file(str(DATA / "train.final.jsonl"), "/root/train.final.jsonl")
    .add_local_file(str(DATA / "train.unroll.jsonl"), "/root/train.unroll.jsonl")
    .add_local_file(str(DATA / "holdout.jsonl"), "/root/holdout.jsonl")
    .add_local_file(str(DATA / "eval_context.json"), "/root/eval_context.json")
)

hf_cache = modal.Volume.from_name("wai-mask-turns-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("wai-mask-turns-out", create_if_missing=True)

TOOL_NAMES = {"lookup_order", "refund_order", "order_status"}
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


ID_RE = re.compile(r"ORD-\d+")


def gold_of(row: dict) -> dict | None:
    """The first tool call in the trace: which tool, which order id."""
    for m in row.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            call = m["tool_calls"][0]
            fn = call.get("function", call)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return {"tool": fn.get("name"), "order_id": (args or {}).get("order_id")}
    return None


def _metrics(texts: list[str], golds: list[dict]) -> list[dict]:
    """Per-reply metric row. Kept here so base and arms score identically.

    `tool_call` says the model called a tool. It does not say it called the
    right one, and it does not say the id was real - the policy is "Look the
    order up before you answer. Never invent an order id." So the three
    task metrics below grade the call against the trace's own gold call.
    """
    out = []
    for t, gold in zip(texts, golds):
        body = CALL_RE.sub("", t).strip()
        calls = []
        for m in CALL_RE.finditer(t):
            try:
                obj = json.loads(m.group(1))
                args = obj.get("arguments") or {}
                calls.append((obj.get("name"), args.get("order_id")))
            except Exception:
                calls.append((None, None))
        named = [c for c in calls if c[0] in TOOL_NAMES]
        ask_ids = set(ID_RE.findall(gold["ask"]))
        used_ids = [c[1] for c in named if c[1]]
        out.append(
            {
                "chars": len(body),
                "stub": 1 if len(body) < 60 else 0,
                "tool_call": 1 if named else 0,
                # called the tool the trace called
                "right_tool": 1 if any(c[0] == gold["tool"] for c in named) else 0,
                # called it with the id the trace used
                "right_id": 1
                if any(c[0] == gold["tool"] and c[1] == gold["order_id"] for c in named)
                else 0,
                # every id it used appears in the ask: "never invent an order id"
                "real_id": 1 if (used_ids and all(i in ask_ids for i in used_ids)) else 0,
            }
        )
    return out


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 150,
    volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol},
)
def run_all() -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    hold = [json.loads(x) for x in open("/root/holdout.jsonl")]
    # Deterministic eval subset: one row per distinct ask, first N by sorted
    # id, and only rows whose trace actually calls a tool - otherwise
    # right_tool/right_id have no gold to grade against.
    seen, eval_rows, golds = set(), [], []
    for r in sorted(hold, key=lambda r: (str(r.get("scenario_id")), str(r.get("prompt")))):
        ask, gold = r.get("prompt"), gold_of(r)
        if ask and gold and gold["tool"] and ask not in seen:
            seen.add(ask)
            eval_rows.append(r)
            golds.append({**gold, "ask": ask})
        if len(eval_rows) >= N_EVAL:
            break
    ctx = json.load(open("/root/eval_context.json"))
    system, tools = ctx["system_prompt"], ctx["tools"]
    print(f"eval prompts: {len(eval_rows)}", flush=True)

    def generate(model, seed: int) -> list[str]:
        set_seed(seed)
        model.eval()
        texts, bs = [], 16
        chats = [
            tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in eval_rows
        ]
        for i in range(0, len(chats), bs):
            enc = tok(chats[i : i + bs], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=160,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            for j in range(out.shape[0]):
                texts.append(
                    tok.decode(out[j][enc["input_ids"].shape[1] :], skip_special_tokens=True)
                )
        return texts

    def fresh_base():
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    results = {"base": [], "arms": {}, "train": {}}

    base = fresh_base()
    for s in (0, 1, 2):
        results["base"].append(_metrics(generate(base, 1000 + s), golds))
        print(f"base pass {s} done", flush=True)
    del base
    torch.cuda.empty_cache()

    for arm in ARMS:
        ds = load_dataset("json", data_files=f"/root/train.{arm}.jsonl", split="train")
        # keep "tools": TRL renders the schemas into the system turn, and the
        # eval prompt renders them too. Dropping it here would train without
        # tool schemas and evaluate with them.
        keep = ({"messages"} if arm == "assistant" else {"prompt", "completion"}) | {"tools"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        for seed in TRAIN_SEEDS:
            set_seed(seed)
            cfg = SFTConfig(
                output_dir=f"/out/{arm}-s{seed}",
                num_train_epochs=2,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                logging_steps=10,
                max_length=1024,
                bf16=True,
                gradient_checkpointing=False,
                report_to=[],
                save_strategy="no",
                seed=seed,
            )
            trainer = SFTTrainer(
                model=BASE_MODEL,
                args=cfg,
                train_dataset=ds,
                processing_class=tok,
                peft_config=LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                ),
            )
            # what TRL actually supervises, straight off its own collator
            batch = trainer.data_collator(
                [trainer.train_dataset[i] for i in range(min(8, len(ds)))]
            )
            sup = int((batch["labels"] != -100).sum())
            tot = int(batch["labels"].numel())
            results["train"][f"{arm}-s{seed}"] = {
                "rows": len(ds),
                "supervised": sup,
                "of": tot,
                "frac": round(sup / tot, 4),
            }
            print(f"{arm}-s{seed}: TRL supervises {sup}/{tot} = {sup / tot:.4f}", flush=True)
            trainer.train()
            texts = generate(trainer.model, 2000 + seed)
            results["arms"].setdefault(arm, {})[f"s{seed}"] = _metrics(texts, golds)
            results["arms"][arm].setdefault("sample", []).append(texts[0][:300])
            del trainer
            torch.cuda.empty_cache()
            print(f"{arm}-s{seed} trained + evaluated", flush=True)

    results["eval_prompts"] = [r["prompt"] for r in eval_rows]
    results["golds"] = golds
    with open("/out/results_raw.json", "w") as fh:
        json.dump(results, fh)
    out_vol.commit()
    return results


@app.local_entrypoint()
def main():
    print(provenance(), file=sys.stderr)
    res = run_all.remote()
    (DATA / "results_raw.json").write_text(json.dumps(res))
    print("supervised fractions, off TRL's own collator:")
    for k, v in res["train"].items():
        print(f"  {k:14s} rows={v['rows']:4d}  {v['supervised']:6d}/{v['of']:6d} = {v['frac']}")
