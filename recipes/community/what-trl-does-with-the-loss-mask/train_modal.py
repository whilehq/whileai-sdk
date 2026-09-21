"""Does the loss_mask that wai.export writes, and TRL drops, change the model?

Two LoRA SFT arms on identical rows and identical hyperparameters. The only
difference is which tokens carry a label:

  as_exported  - every token supervised, which is what TRL 0.19.1's
                 SFTTrainer does with this file (measured on CPU: 0.964)
  mask_honored - the export's own loss_mask applied: assistant turns only

Base is evaluated three times first for the noise floor. Everything runs on
my own Modal account; nothing touches While hosting.

    modal run train_modal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
APP = "wai-seat3-lossmask"

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
    .add_local_file(str(HERE / "train.trl.jsonl"), "/root/train.trl.jsonl")
    .add_local_file(str(HERE / "holdout.jsonl"), "/root/holdout.jsonl")
)

hf_cache = modal.Volume.from_name("wai-seat3-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("wai-seat3-runs", create_if_missing=True)


def _tokenize(rows, tok, max_len: int, honor_mask: bool):
    """Render each row's messages and build labels.

    This is the part the SDK does not own: turning the exported per-message
    loss_mask into per-token labels. ~25 lines.
    """
    out = []
    for row in rows:
        msgs, mask = row["messages"], row["loss_mask"]
        tools = row.get("tools")
        ids, labels = [], []
        prev = 0
        for i in range(len(msgs)):
            text = tok.apply_chat_template(
                msgs[: i + 1], tools=tools, tokenize=False, add_generation_prompt=False
            )
            piece = tok(text, add_special_tokens=False)["input_ids"]
            seg = piece[prev:]
            prev = len(piece)
            ids.extend(seg)
            train_on = (mask[i] == 1) if honor_mask else True
            labels.extend(seg if train_on else [-100] * len(seg))
        out.append({"input_ids": ids[:max_len], "labels": labels[:max_len]})
    return out


def _collate(batch, pad_id):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        labs.append(b["labels"] + [-100] * k)
        att.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labs),
        "attention_mask": torch.tensor(att),
    }


def _generate(model, tok, prompts, *, temperature: float, seed: int, max_new: int = 96):
    import torch

    torch.manual_seed(seed)
    model.eval()
    tok.padding_side = "left"
    outs = []
    for start in range(0, len(prompts), 16):
        chunk = prompts[start : start + 16]
        texts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=temperature or None,
                top_p=0.95,
                max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        plen = enc["input_ids"].shape[1]
        outs.extend(tok.batch_decode(gen[:, plen:], skip_special_tokens=True))
    model.train()
    return outs


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 75,
    volumes={"/root/.cache/huggingface": hf_cache, "/vol": out_vol},
)
def run(steps_epochs: int = 3, lr: float = 1e-4, seed: int = 0) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_rows = [json.loads(x) for x in open("/root/train.trl.jsonl")]
    hold_rows = [json.loads(x) for x in open("/root/holdout.jsonl")]
    print(f"{len(train_rows)} train rows, {len(hold_rows)} holdout rows")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    SYS = train_rows[0]["messages"][0]["content"]
    hold_prompts, hold_meta = [], []
    for r in hold_rows:
        ask = r.get("prompt") or r.get("prompt_text")
        if not ask:
            continue
        hold_prompts.append([{"role": "system", "content": SYS}, {"role": "user", "content": ask}])
        hold_meta.append({"scenario_id": r.get("scenario_id"), "prompt": ask})
    print(f"{len(hold_prompts)} holdout prompts")

    # supervised fraction, both ways, so the manipulation is on the record
    frac = {}
    for honor in (False, True):
        t = _tokenize(train_rows, tok, 1024, honor)
        sup = sum(sum(1 for x in r["labels"] if x != -100) for r in t)
        tot = sum(len(r["labels"]) for r in t)
        frac["mask_honored" if honor else "as_exported"] = round(sup / tot, 4)
    print("supervised fraction:", frac)

    results = {"base_evals": [], "arms": {}, "supervised_fraction": frac}

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    # noise floor: three sampled passes over the same holdout
    for s in (0, 1, 2):
        gens = _generate(base, tok, hold_prompts, temperature=0.7, seed=s)
        results["base_evals"].append(gens)
        print(f"base eval seed {s}: {len(gens)} replies")

    del base
    torch.cuda.empty_cache()

    for arm, honor in (("as_exported", False), ("mask_honored", True)):
        print(f"===== arm {arm} =====")
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()
        data = _tokenize(train_rows, tok, 1024, honor)
        args = TrainingArguments(
            output_dir=f"/vol/{arm}",
            num_train_epochs=steps_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=lr,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            seed=seed,
            gradient_checkpointing=False,
        )
        tr = Trainer(
            model=model,
            args=args,
            train_dataset=data,
            data_collator=lambda b: _collate(b, tok.pad_token_id or tok.eos_token_id),
        )
        hist = tr.train()
        print(arm, "final loss", hist.training_loss)
        gens = _generate(model, tok, hold_prompts, temperature=0.7, seed=0)
        results["arms"][arm] = {"gens": gens, "train_loss": hist.training_loss}
        model.save_pretrained(f"/vol/{arm}/adapter")
        del model, tr
        torch.cuda.empty_cache()

    results["holdout_meta"] = hold_meta
    Path("/vol/results.json").write_text(json.dumps(results))
    out_vol.commit()
    return results


@app.local_entrypoint()
def main() -> None:
    print(provenance(), file=sys.stderr)
    res = run.remote()
    Path(HERE / "raw_results.json").write_text(json.dumps(res))
    print("supervised fraction:", res["supervised_fraction"])
    print("wrote raw_results.json")
