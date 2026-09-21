"""Score the pool once, then train the three selector arms on your own Modal.

One scoring pass (inference, no training) measures each row's response loss
under the untrained base. The three arms then spend the *same token budget*
on different rows, which is the only thing that differs between them.

    modal run --detach recipes/community/identity-spec-no-unasked-maker-aas/train_modal.py

Adapters land on the volume ``identity-aas-runs`` under ``/<selector>/adapter``.
Run ``python run.py prep`` first; this reads ``out/pool.jsonl``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen3-1.7B"
HERE = Path(__file__).parent
SELECTORS = ("random", "loss", "aas")
BUDGET_FRACTION = 0.15
MAX_LEN = 2048

app = modal.App("identity-aas-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "whileai==0.110",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
)

dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)
runs = modal.Volume.from_name("identity-aas-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("identity-aas-hf-cache", create_if_missing=True)
VOL = "/vol"


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
)
def score(rows: list[dict], base_model: str = BASE_MODEL) -> list[dict]:
    """Per-row response loss under the untrained base: the online scorer.

    The score is the mean NLL of the final assistant turn, which is the part
    the row actually teaches. ``n_tokens`` is the whole rendered
    conversation, because that is what the row costs the trainer.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    meta = []
    for n, row in enumerate(rows):
        msgs = row["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(full, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids
        n_prefix = len(tok(prefix, truncation=True, max_length=MAX_LEN).input_ids)
        ids = ids.to("cuda")
        if ids.shape[1] - n_prefix < 1:
            loss = 0.0
        else:
            labels = ids.clone()
            labels[:, :n_prefix] = -100
            with torch.no_grad():
                loss = float(model(ids, labels=labels).loss)
        meta.append(
            {
                "idx": row["idx"],
                "kind": row["kind"],
                "loss": loss,
                "n_tokens": int(ids.shape[1]),
            }
        )
        if n % 500 == 0:
            print(f"scored {n}/{len(rows)}", flush=True)

    Path(f"{VOL}/pool_meta.json").write_text(json.dumps(meta))
    runs.commit()
    print(f"scored {len(meta)} rows")
    return meta


@app.function(
    image=image,
    gpu="L40S",
    timeout=2 * 60 * 60,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def train(
    selector: str,
    rows: list[dict],
    base_model: str = BASE_MODEL,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(base_model)

    def to_text(row: dict) -> dict:
        return {
            "text": tok.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    ds = Dataset.from_list(rows).map(to_text, remove_columns=list(rows[0].keys()))
    print(f"[{selector}] rows {len(ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    out_dir = f"{VOL}/{selector}"
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
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
        ),
        args=SFTConfig(
            output_dir=f"{out_dir}/checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            max_length=MAX_LEN,
            packing=False,
            # Off deliberately: this stack corrupts Qwen3 generation with
            # checkpointing on, and the adapter is generated from later.
            gradient_checkpointing=False,
            seed=0,
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(f"{out_dir}/adapter")
    tok.save_pretrained(f"{out_dir}/adapter")
    Path(f"{out_dir}/manifest.json").write_text(
        json.dumps(
            {
                "selector": selector,
                "rows": len(rows),
                "tokens": sum(r.get("n_tokens", 0) for r in rows),
                "base_model": base_model,
                "epochs": epochs,
                "learning_rate": learning_rate,
            },
            indent=2,
        )
    )
    runs.commit()
    print(f"[{selector}] adapter -> {out_dir}/adapter")
    return f"{out_dir}/adapter"


@app.local_entrypoint()
def main() -> None:
    import sys

    sys.path.insert(0, str(HERE))
    import arm_selectors as sel

    out = HERE / "out"
    pool = [json.loads(line) for line in (out / "pool.jsonl").read_text().splitlines()]
    print(f"pool {len(pool)} rows -> scoring under {BASE_MODEL}")

    meta_path = out / "pool_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"reusing {meta_path}")
    else:
        meta = score.remote(pool)
        meta_path.write_text(json.dumps(meta))

    budget = int(sum(r["n_tokens"] for r in meta) * BUDGET_FRACTION)
    by_idx = {r["idx"]: r for r in pool}
    jobs, manifest = [], {"budget_tokens": budget, "arms": {}}
    for name in SELECTORS:
        kept = sel.select(meta, name, budget)
        manifest["arms"][name] = sel.audit(kept, meta)
        rows = [{"messages": by_idx[k["idx"]]["messages"], "n_tokens": k["n_tokens"]} for k in kept]
        jobs.append((name, rows))
        print(
            f"{name:<8} rows {len(rows):>4} tokens {manifest['arms'][name]['tokens']:>8,} "
            f"identity-token-share {manifest['arms'][name]['identity_token_share']:.1%}"
        )
    (out / "selection.json").write_text(json.dumps(manifest, indent=2))

    for path in train.starmap(jobs):
        print("done:", path)
