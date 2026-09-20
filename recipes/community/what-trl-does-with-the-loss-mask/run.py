"""Build the graded set, export it for TRL, and show what TRL supervises.

Offline: no model key, no GPU, no account. The GPU half is train_modal.py.

    python run.py                 # build + export + the supervised-token count
    python run.py --dry-run       # build + export only, no tokenizer download
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import whileai as wai

HERE = Path(__file__).resolve().parent

POLICY = (
    "You are a support agent for an online store. Look the order up before "
    "you answer. Never invent an order id. If the customer gives no id, ask "
    "for it. Answer in at most three sentences, no preamble."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "Refund an order that has been looked up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "order_status",
            "description": "Current shipping status for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]


def tokenize(rows, tok, max_len: int, honor_mask: bool):
    """Per-message loss_mask -> per-token labels.

    The SDK writes the mask; nothing in TRL reads it. This is the bridge,
    and it is the code this recipe exists to point at.
    """
    out = []
    for row in rows:
        msgs, mask, tools = row["messages"], row["loss_mask"], row.get("tools")
        ids, labels, prev = [], [], 0
        for i in range(len(msgs)):
            text = tok.apply_chat_template(
                msgs[: i + 1], tools=tools, tokenize=False, add_generation_prompt=False
            )
            piece = tok(text, add_special_tokens=False)["input_ids"]
            seg = piece[prev:]
            prev = len(piece)
            ids.extend(seg)
            labels.extend(seg if (honor_mask is False or mask[i] == 1) else [-100] * len(seg))
        out.append({"input_ids": ids[:max_len], "labels": labels[:max_len]})
    return out


def build(seed: int, out_dir: Path) -> dict:
    agent = wai.seeded_agent(TOOLS, rate=0.35, seed=seed)
    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        budget=600,
        grade=True,
        reproducible=True,
        seed=seed,
    )
    rows = list(data.trajectories)
    scen = sorted({r.get("scenario_id") for r in rows})
    cut = int(len(scen) * 0.65)
    train_ids = set(scen[:cut])
    train_rows = [r for r in rows if r.get("scenario_id") in train_ids]
    hold_rows = [r for r in rows if r.get("scenario_id") not in train_ids]

    sel = wai.select(train_rows, mode="sft", target=400)
    kept = list(sel)  # Selection is a list subclass; it also has .export()
    print(sel)

    dec = wai.decontaminate(kept, against=hold_rows)
    report = dec[1] if isinstance(dec, tuple) else dec
    print("decontaminate:", {k: report[k] for k in sorted(report) if k.startswith("n_")})

    rep = wai.export(
        kept,
        str(out_dir / "train.trl.jsonl"),
        system_prompt=POLICY,
        tools=TOOLS,
        format="trl",
    )
    print("export:", {k: rep[k] for k in ("n", "n_written", "mask_mode", "format")})
    with open(out_dir / "holdout.jsonl", "w") as fh:
        for r in hold_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(kept)} train rows, {len(hold_rows)} holdout rows")
    return {"train": len(kept), "holdout": len(hold_rows), "mask_mode": rep["mask_mode"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dry-run", action="store_true", help="skip the tokenizer download")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = build(args.seed, out_dir)

    if args.dry_run:
        print("--dry-run: stopping before the tokenizer")
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(x) for x in open(out_dir / "train.trl.jsonl")]
    for honor in (False, True):
        t = tokenize(rows, tok, 1024, honor)
        sup = sum(sum(1 for x in r["labels"] if x != -100) for r in t)
        tot = sum(len(r["labels"]) for r in t)
        label = "loss_mask honoured" if honor else "as TRL trains it"
        print(f"{label:22s}: {sup}/{tot} tokens supervised = {sup / tot:.4f}")
    print(f"\nexport reported mask_mode={info['mask_mode']!r}")


if __name__ == "__main__":
    main()
