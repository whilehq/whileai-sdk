"""Fine-tune a tiny encoder as a binary injection classifier on one L40S.

    modal run recipes/04-train/prompt-injection-classifier/train_modal.py --train out/train.jsonl --seeds 1,2,3

``wai.train`` is ``sft | grpo | dpo | rm`` on decoder LMs; there is no
sequence-classification path, so this is recipe-local
``AutoModelForSequenceClassification`` on the export, mirroring
``recipes/04-train/sft/train_modal.py`` in shape. One seed is about four
minutes on an L40S ($1.95/h); the model and tokenizer come back to the caller
as bytes and are written under ``out/<run>/``.

Defaults (Devlin et al. 2019 fine-tuning recipe: lr 2e-5 to 5e-5, 2 to 4
epochs, batch 32; the exact values here are convention, untested on this
task beyond the runs in results.json).
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
GPU = "L40S"
BASE_MODEL = "nreimers/MiniLM-L6-H384-uncased"  # MIT; 22M params, 6 layers, hidden 384
EPOCHS = 3
LR = 5e-5
BATCH = 32
MAX_LENGTH = 512

app = modal.App("pinj-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "scikit-learn==1.7.0",
        "sentencepiece==0.2.0",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
)
hf_cache = modal.Volume.from_name("pinj-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 40,
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=60,
)
def train(
    rows: list[dict], seed: int, base_model: str = BASE_MODEL, epochs: int = EPOCHS, lr: float = LR
) -> dict:
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "BENIGN", 1: "INJECTION"},
        label2id={"BENIGN": 0, "INJECTION": 1},
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n_val = max(200, len(rows) // 20)
    val_rows = [rows[i] for i in idx[:n_val]]
    tr_rows = [rows[i] for i in idx[n_val:]]

    def enc(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds_tr = Dataset.from_list([{"text": r["text"], "label": int(r["label"])} for r in tr_rows]).map(
        enc, batched=True
    )
    ds_va = Dataset.from_list(
        [{"text": r["text"], "label": int(r["label"])} for r in val_rows]
    ).map(enc, batched=True)

    def compute_metrics(p):
        pred = p.predictions.argmax(-1)
        return {"accuracy": float((pred == p.label_ids).mean())}

    args = TrainingArguments(
        output_dir="/tmp/run",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=64,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        bf16=True,
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    ev = trainer.evaluate()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    out = "/tmp/model"
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=".")
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n)
    record = {
        "seed": seed,
        "base_model": base_model,
        "epochs": epochs,
        "lr": lr,
        "batch": BATCH,
        "max_length": MAX_LENGTH,
        "train_rows": len(tr_rows),
        "val_rows": len(val_rows),
        "val_accuracy": ev.get("eval_accuracy"),
        "val_loss": ev.get("eval_loss"),
        "train_loss_first_last": [round(losses[0], 4), round(losses[-1], 4)] if losses else None,
        "params_total": n_params,
        "params_non_embedding": n_params - n_emb,
        "seconds": round(time.time() - t0, 1),
        "gpu": GPU,
        "torch": torch.__version__,
    }
    # plain JSON plus bytes: the caller's environment need not have torch
    return {**json.loads(json.dumps(record, default=float)), "tar": buf.getvalue()}


@app.local_entrypoint()
def main(
    train_file: str = str(HERE / "out/train.jsonl"),
    seeds: str = "1",
    out: str = str(HERE / "out"),
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
):
    rows = [json.loads(line) for line in Path(train_file).read_text().splitlines() if line.strip()]
    seed_list = [int(s) for s in seeds.split(",")]
    print(
        f"{len(rows)} rows, seeds {seed_list}, {base_model}, {epochs} epochs, {GPU}",
        file=sys.stderr,
    )
    results = list(train.starmap([(rows, s, base_model, epochs) for s in seed_list]))
    tag = base_model.split("/")[-1].lower()
    for r in results:
        run_dir = Path(out) / f"{tag}-seed{r['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(r.pop("tar")), mode="r:gz") as tar:
            tar.extractall(run_dir)
        (run_dir / "train_record.json").write_text(json.dumps(r, indent=1))
        print(
            f"seed {r['seed']}: val acc {r['val_accuracy']:.4f}, loss {r['train_loss_first_last']}, "
            f"{r['seconds']}s, {r['params_non_embedding'] / 1e6:.1f}M non-embedding params -> {run_dir}"
        )
