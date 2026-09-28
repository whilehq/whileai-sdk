"""Fine-tune a tiny encoder as a binary injection classifier on one L40S.

    modal run recipes/04-train/prompt-injection-classifier/train_modal.py --train-file out/train.jsonl --seeds 1,2,3
    modal run ... --loss twin --margin 2.0          # pairwise margin on the matched twins
    modal run ... --weight-key weight               # per-row weights (hard-negative mining)
    modal run ... --soft-key soft                   # distillation on a teacher's probability

``wai.train`` is ``sft | grpo | dpo | rm`` on decoder LMs; there is no
sequence-classification path, so this is recipe-local
``AutoModelForSequenceClassification`` on the export, mirroring
``recipes/04-train/sft/train_modal.py`` in shape. One seed is under a minute
on an L40S ($1.95/h); the model and tokenizer come back as bytes and are
written under ``out/<run>/``.

Losses:

* ``ce``: cross-entropy on the label (Devlin et al. 2019 fine-tuning recipe).
* ``twin``: cross-entropy plus a hinge on every matched pair,
  ``max(0, margin - (s_pos - s_twin))`` with ``s`` the injection logit minus
  the benign logit. The classifier analogue of a pairwise preference loss
  (Lambert 2025, chapter Direct Alignment; Bradley-Terry in chapter Reward
  Modeling): the pair differs only in the insert, so the margin is paid for
  by the insert and nothing else. ``lambda`` 1.0 and ``margin`` 2.0 are
  convention, untested beyond the runs in results.json.
* ``soft``: cross-entropy against a teacher probability (Hinton et al. 2015,
  arXiv:1503.02531; Lambert 2025, chapter Synthetic Data and Distillation).

Defaults: lr 5e-5, 3 epochs, batch 32 (Devlin et al. 2019); convention,
untested on this task beyond results.json.
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
    rows: list[dict],
    seed: int,
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
    lr: float = LR,
    loss: str = "ce",
    margin: float = 2.0,
    lam: float = 1.0,
    weight_key: str | None = None,
    soft_key: str | None = None,
) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
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
    ).cuda()

    # one item per pair (both members), or per unpaired row
    by_pair: dict[str, dict] = {}
    items: list[dict] = []
    for r in rows:
        p = r.get("pair")
        if loss == "twin" and p:
            slot = by_pair.setdefault(p, {})
            slot["pos" if int(r["label"]) == 1 else "neg"] = r
        else:
            items.append({"a": r})
    for slot in by_pair.values():
        if "pos" in slot and "neg" in slot:
            items.append({"a": slot["pos"], "b": slot["neg"]})
        else:
            items.extend({"a": r} for r in slot.values())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_val = max(200, len(items) // 20)
    val_items = [items[i] for i in idx[:n_val]]
    tr_items = [items[i] for i in idx[n_val:]]

    def fields(r: dict) -> tuple[int, float, float]:
        w = float(r.get(weight_key, 1.0)) if weight_key else 1.0
        s = float(r[soft_key]) if soft_key and r.get(soft_key) is not None else float(r["label"])
        return int(r["label"]), w, s

    class Items(Dataset):
        def __init__(self, xs):
            self.xs = xs

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, i):
            return self.xs[i]

    def collate(batch):
        texts, labels, weights, softs, pair_a, pair_b = [], [], [], [], [], []
        for it in batch:
            for key in ("a", "b"):
                r = it.get(key)
                if r is None:
                    continue
                if key == "b":
                    pair_a.append(len(texts) - 1)
                    pair_b.append(len(texts))
                texts.append(r["text"])
                lab, w, s = fields(r)
                labels.append(lab)
                weights.append(w)
                softs.append(s)
        enc = tok(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        enc["weights"] = torch.tensor(weights)
        enc["softs"] = torch.tensor(softs)
        enc["pair_a"] = torch.tensor(pair_a, dtype=torch.long)
        enc["pair_b"] = torch.tensor(pair_b, dtype=torch.long)
        return enc

    def step_loss(batch):
        labels = batch.pop("labels").cuda()
        weights = batch.pop("weights").cuda()
        softs = batch.pop("softs").cuda()
        pa = batch.pop("pair_a").cuda()
        pb = batch.pop("pair_b").cuda()
        logits = model(**{k: v.cuda() for k, v in batch.items()}).logits.float()
        if soft_key:
            target = torch.stack([1 - softs, softs], dim=1)
            ce = -(target * F.log_softmax(logits, -1)).sum(-1)
        else:
            ce = F.cross_entropy(logits, labels, reduction="none")
        total = (ce * weights).sum() / weights.sum()
        hinge = torch.tensor(0.0, device=logits.device)
        if loss == "twin" and len(pa):
            s = logits[:, 1] - logits[:, 0]
            hinge = F.relu(margin - (s[pa] - s[pb])).mean()
            total = total + lam * hinge
        return total, logits, labels, hinge

    per_device = BATCH // 2 if loss == "twin" else BATCH  # a pair is two sequences
    dl = DataLoader(Items(tr_items), batch_size=per_device, shuffle=True, collate_fn=collate)
    dl_val = DataLoader(Items(val_items), batch_size=64, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        for batch in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total, _, _, _ = step_loss(batch)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(float(total))
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for batch in dl_val:
            _, logits, labels, _ = step_loss(batch)
            correct += int((logits.argmax(-1) == labels).sum())
            n += len(labels)
    out = "/tmp/model"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=".")
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for name, p in model.named_parameters() if "embeddings" in name)
    record = {
        "seed": seed,
        "base_model": base_model,
        "epochs": epochs,
        "lr": lr,
        "batch": BATCH,
        "max_length": MAX_LENGTH,
        "loss": loss,
        "margin": margin if loss == "twin" else None,
        "lambda": lam if loss == "twin" else None,
        "weight_key": weight_key,
        "soft_key": soft_key,
        "train_items": len(tr_items),
        "train_rows": len(rows),
        "val_items": len(val_items),
        "val_accuracy": correct / max(1, n),
        "train_loss_first_last": [round(losses[0], 4), round(float(np.mean(losses[-20:])), 4)]
        if losses
        else None,
        "steps": steps,
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
    loss: str = "ce",
    margin: float = 2.0,
    lam: float = 1.0,
    weight_key: str = "",
    soft_key: str = "",
    tag: str = "",
):
    rows = [json.loads(line) for line in Path(train_file).read_text().splitlines() if line.strip()]
    seed_list = [int(s) for s in seeds.split(",")]
    print(
        f"{len(rows)} rows, seeds {seed_list}, {base_model}, {epochs} epochs, loss {loss}, {GPU}",
        file=sys.stderr,
    )
    args = [
        (rows, s, base_model, epochs, LR, loss, margin, lam, weight_key or None, soft_key or None)
        for s in seed_list
    ]
    results = list(train.starmap(args))
    name = tag or base_model.split("/")[-1].lower()
    for r in results:
        run_dir = Path(out) / f"{name}-seed{r['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(r.pop("tar")), mode="r:gz") as tar:
            tar.extractall(run_dir)
        (run_dir / "train_record.json").write_text(json.dumps(r, indent=1))
        print(
            f"seed {r['seed']}: val acc {r['val_accuracy']:.4f}, loss {r['train_loss_first_last']}, "
            f"{r['seconds']}s, {r['params_non_embedding'] / 1e6:.1f}M non-embedding params -> {run_dir}"
        )
