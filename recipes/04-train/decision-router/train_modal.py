"""Train a small decision model that says, for one question, how likely each of twelve models is to get it right.

Three shapes, all trained on the same table as part 1 (recipes/04-train/model-router):

- `pointer`: the shape the outside evidence points to for TypeSafe's Jev [1, 2].
  A small causal LM (Qwen3-0.6B, LoRA) reads the question once, then the
  candidate models as a list, one reserved marker token after each name.
  A linear head reads the hidden state at every marker and gives that model
  a logit. The list is shuffled on every pass, so no position is learned;
  later names attend to earlier ones, so the options are read together [2].
- `encoder`: the fast baseline the open clones use [3]. ModernBERT-base
  reads the question; a fixed twelve-way head gives every model a logit.
- `pointer-kind`: the pointer plus a prior from the kind of question (Jev's
  answer through the training table, run.py builds it); the head starts at
  zero and learns a correction on top of the prior.

Both minimise log loss on "did model m answer correctly" (a proper scoring
rule, so the probabilities are pushed toward honest [4]), keep the epoch
with the best val loss, then fit one temperature on val [5]. No text is
generated at any point.

Run (from the repo root):

    modal run recipes/04-train/decision-router/train_modal.py --arm pointer --seed 0

`run.py --train` launches every arm and seed and collects the predictions.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time

import modal

from whileai.config import provenance, requirement

POINTER_BASE = "Qwen/Qwen3-0.6B"
ENCODER_BASE = "answerdotai/ModernBERT-base"
MARKER = "<|box_end|>"  # a reserved Qwen3 token: question text cannot forge an option boundary
MAX_QUESTION_TOKENS = 1024  # longer questions keep their head; convention, untested
EPOCHS = 4
BATCH = 16
MICRO = 4  # BATCH is reached by accumulating micro-batches of this size: 16 at once ran out of memory on an L40S
LORA_RANK = 16  # convention for a sub-1B LoRA, untested here
LORA_ALPHA = 32
LR = {"pointer": 2e-4, "encoder": 5e-5}  # LoRA vs full fine-tune conventions, untested here
HEAD_LR = 1e-3
PRIOR_EPS = 0.02  # a prior of 0 or 1 is clipped before the logit; convention, untested
EVAL_ORDERS = 4  # pointer: average logits over this many shuffled lists at val/test
LATENCY_REPS = 50

app = modal.App("decision-router-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.7.1",
    "transformers==4.54.0",
    "peft==0.16.0",
    "numpy<2.3",
    requirement(),
)
hf_cache = modal.Volume.from_name("decision-router-hf-cache", create_if_missing=True)


def batches(idx, size, rng=None):
    idx = list(idx)
    if rng:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        yield idx[i : i + size]


@app.function(
    image=image,
    gpu="L40S",
    timeout=2 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def train(arm: str, seed: int, models: list[str], rows: list[dict]) -> dict:
    """rows: {"query", "labels" (len(models) floats) or None, "split": fit|val|test}.
    Returns val and test probabilities in `models` order, the val loss per epoch,
    the fitted temperature and the measured latency."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = "cuda"
    K = len(models)
    pointer = arm.startswith("pointer")  # "pointer" or "pointer-kind"
    base = POINTER_BASE if pointer else ENCODER_BASE
    tok = AutoTokenizer.from_pretrained(base)
    # The LM's frozen weights in bf16; anything trained (LoRA, the encoder, the heads) in fp32 under autocast.
    dtype = torch.bfloat16 if pointer else torch.float32
    lm = AutoModel.from_pretrained(base, torch_dtype=dtype).to(dev)
    hidden = lm.config.hidden_size

    if pointer:
        from peft import LoraConfig, get_peft_model

        lm = get_peft_model(
            lm,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.0,
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
        )
        for prm in lm.parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
        head = torch.nn.Linear(hidden, 1).to(dev)
        marker_id = tok.convert_tokens_to_ids(MARKER)
        name_ids = [tok(f"\n- {m}", add_special_tokens=False)["input_ids"] for m in models]
        q_ids = [
            tok(
                "Question:\n" + r["query"].replace(MARKER, ""),
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_QUESTION_TOKENS,
            )["input_ids"]
            for r in rows
        ]
        tail = tok("\n\nWhich of these models will answer it correctly?", add_special_tokens=False)[
            "input_ids"
        ]

        def encode(ix, perms):
            seqs, marks = [], []
            for i, perm in zip(ix, perms):
                s = q_ids[i] + tail
                pos = []
                for j in perm:
                    s = s + name_ids[j] + [marker_id]
                    pos.append(len(s) - 1)
                seqs.append(s)
                marks.append(pos)
            n = max(map(len, seqs))
            ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), n), dtype=torch.long)
            for b, s in enumerate(seqs):
                ids[b, : len(s)] = torch.tensor(s)
                att[b, : len(s)] = 1
            return ids.to(dev), att.to(dev), torch.tensor(marks, device=dev)

        def logits_for(ix, perms):
            ids, att, marks = encode(ix, perms)
            h = lm(input_ids=ids, attention_mask=att).last_hidden_state
            g = h[torch.arange(len(ix), device=dev)[:, None], marks]  # (B, K, H) in list order
            z_list = head(g.float()).squeeze(-1)
            z = torch.empty_like(z_list)
            p = torch.tensor(perms, device=dev)
            z.scatter_(1, p, z_list)  # back to `models` order
            return z

        params = [
            {"params": [p for p in lm.parameters() if p.requires_grad], "lr": LR["pointer"]},
            {"params": head.parameters(), "lr": HEAD_LR},
        ]
        if arm == "pointer-kind":
            # The kind as a prior: each row carries every model's training accuracy on the
            # kinds Jev says the question could be. The model starts at that prior (head at
            # zero, weight one) and learns a correction on top of it.
            prior = torch.tensor([r["prior"] for r in rows], dtype=torch.float32, device=dev)
            prior_logit = torch.logit(prior.clamp(PRIOR_EPS, 1 - PRIOR_EPS))
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
            prior_w = torch.nn.Parameter(torch.ones(1, device=dev))
            params.append({"params": [prior_w], "lr": HEAD_LR})
            text_logits = logits_for

            def logits_for(ix, perms):
                return text_logits(ix, perms) + prior_w * prior_logit[ix]

    else:
        head = torch.nn.Linear(hidden, K).to(dev)
        texts = [r["query"] for r in rows]

        def logits_for(ix, perms=None):
            enc = tok(
                [texts[i] for i in ix],
                truncation=True,
                max_length=MAX_QUESTION_TOKENS,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            h = lm(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            return head((h * m).sum(1) / m.sum(1))

        params = [
            {"params": lm.parameters(), "lr": LR["encoder"]},
            {"params": head.parameters(), "lr": HEAD_LR},
        ]

    split = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("fit", "val", "test")}
    Y = torch.tensor([r["labels"] or [0.0] * K for r in rows], dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    steps = EPOCHS * math.ceil(len(split["fit"]) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, (steps - s) / steps)
    )

    def perms_for(n, orders_rng):
        out = []
        for _ in range(n):
            p = list(range(K))
            orders_rng.shuffle(p)
            out.append(p)
        return out

    @torch.no_grad()
    def predict(ix):
        lm.eval()
        zs = []
        for b in batches(ix, 8):
            if pointer:
                orng = random.Random(1234)  # the same shuffled lists for every epoch and seed
                z = (
                    sum(logits_for(b, perms_for(len(b), orng)) for _ in range(EVAL_ORDERS))
                    / EVAL_ORDERS
                )
            else:
                z = logits_for(b)
            zs.append(z.float().cpu())
        lm.train()
        return torch.cat(zs)

    history, best = [], None
    lm.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        for b in batches(split["fit"], BATCH, rng):
            for mb in batches(b, MICRO):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = logits_for(mb, perms_for(len(mb), rng)) if pointer else logits_for(mb)
                loss = F.binary_cross_entropy_with_logits(z.float(), Y[mb]) * len(mb) / len(b)
                loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zv = predict(split["val"])
        vloss = F.binary_cross_entropy_with_logits(zv, Y[split["val"]].cpu()).item()
        history.append(
            {"epoch": epoch + 1, "val_log_loss": vloss, "minutes": (time.time() - t0) / 60}
        )
        print(json.dumps(history[-1]), flush=True)
        if best is None or vloss < best[0]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zt = predict(split["test"])
            best = (vloss, epoch + 1, zv.clone(), zt)

    # One temperature on val: z / T minimising log loss.
    _, best_epoch, zv, zt = best
    yv = Y[split["val"]].cpu()
    logt = torch.zeros(1, requires_grad=True)
    topt = torch.optim.LBFGS([logt], lr=0.1, max_iter=100)

    def closure():
        topt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(zv / logt.exp(), yv)
        loss.backward()
        return loss

    topt.step(closure)
    T = float(logt.exp())

    # Latency: one question at a time, the median question length.
    med = sorted(split["test"], key=lambda i: len(rows[i]["query"]))[len(split["test"]) // 2]
    lm.eval()
    times = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(LATENCY_REPS):
            torch.cuda.synchronize()
            s = time.perf_counter()
            logits_for([med], [list(range(K))]) if pointer else logits_for([med])
            torch.cuda.synchronize()
            times.append(time.perf_counter() - s)
    return {
        "arm": arm,
        "seed": seed,
        "base": base,
        "best_epoch": best_epoch,
        "history": history,
        "temperature": T,
        "val": torch.sigmoid(zv / T).numpy().tolist(),
        "test": torch.sigmoid(zt / T).numpy().tolist(),
        "latency_ms_median": 1000 * float(np.median(times[5:])),
        "gpu": torch.cuda.get_device_name(0),
    }


@app.local_entrypoint()
def main(arm: str = "pointer", seed: int = 0, data: str = "out/train_rows.json", out: str = "out"):
    print(provenance(), file=sys.stderr)
    payload = json.loads(open(data, encoding="utf-8").read())
    res = train.remote(arm, seed, payload["models"], payload["rows"])
    path = f"{out}/{arm}-s{seed}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    print(f"wrote {path}: best epoch {res['best_epoch']}, T={res['temperature']:.2f}")
