"""Build the tool-using set, export it three ways, and count what each one supervises.

The three exports are the three things `wai.export(format="trl")` can hand
TRL this release. They differ in which assistant turns carry loss:

    assistant  mask_mode="assistant"           messages rows, every token
    final      mask_mode="final"               prompt/completion, last turn only
    unroll     mask_mode="final", unroll=True   one row per assistant turn

Offline: no model key, no GPU, no account. The GPU half is train_modal.py.

    python run.py                 # build + export + the tool-call count
    python run.py --dry-run       # same, fewer seeds, nothing written outside --out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import whileai as wai
from whileai.config import provenance

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

ARMS = {
    "assistant": {"mask_mode": "assistant"},
    "final": {"mask_mode": "final"},
    "unroll": {"mask_mode": "final", "unroll": True},
}


def pool(seeds: int) -> list[dict]:
    """One `simulate` per seed, concatenated.

    A single call caps out: `budget=2000` returns the same 156 rows as
    `budget=600`. Different seeds re-draw the *asks* from the same scenario
    templates, so pooling is how you get past the ceiling.
    """
    rows = []
    for s in range(seeds):
        agent = wai.seeded_agent(TOOLS, rate=0.35, seed=s)
        data = wai.simulate(
            agent,
            tools=TOOLS,
            system_prompt=POLICY,
            simulator=False,
            budget=600,
            grade="conduct",
            reproducible=True,
            seed=s,
        )
        for row in data.trajectories:
            row = dict(row)
            row["pool_seed"] = s
            rows.append(row)
    return rows


def tool_call_census(path: Path) -> dict:
    """How many tool calls does mask_mode="final" ever put in a completion?

    `final` supervises the last assistant turn. In a tool-using trace the
    turn that *calls* the tool is never last - the tool answers and the
    agent summarises - so the answer is usually zero, and the export
    report's `masked_messages` does not say so.
    """
    total = final = 0
    for line in open(path, encoding="utf-8"):
        turns = [m for m in json.loads(line)["messages"] if m.get("role") == "assistant"]
        for i, m in enumerate(turns):
            if m.get("tool_calls"):
                total += 1
                final += i == len(turns) - 1
    return {"tool_calls": total, "supervised_by_final": final}


def build(seeds: int, out_dir: Path) -> dict:
    rows = pool(seeds)
    print(f"pooled {len(rows)} rows from {seeds} seeds")

    # Split by scenario_id: the template id, stable across seeds, so no
    # template straddles train and holdout.
    scen = sorted({r.get("scenario_id") for r in rows})
    train_ids = set(scen[: int(len(scen) * 0.65)])
    train_rows = [r for r in rows if r.get("scenario_id") in train_ids]
    hold_rows = [r for r in rows if r.get("scenario_id") not in train_ids]
    print(
        f"{len(train_ids)}/{len(scen)} templates train -> "
        f"{len(train_rows)} train rows, {len(hold_rows)} holdout rows"
    )

    sel = wai.select(train_rows, mode="sft", target=900)
    print(sel)

    dec = wai.decontaminate(list(sel), against=hold_rows)
    clean, report = dec if isinstance(dec, tuple) else (dec, {})
    clean = list(clean)
    counts = {k: report[k] for k in sorted(report) if k.startswith("n_")}
    print("decontaminate:", counts)
    print(f"APPLIED: {len(list(sel))} -> {len(clean)} rows")

    reports = {}
    for name, kwargs in ARMS.items():
        rep = wai.export(
            clean,
            str(out_dir / f"train.{name}.jsonl"),
            system_prompt=POLICY,
            tools=TOOLS,
            format="trl",
            **kwargs,
        )
        reports[name] = {
            k: rep.get(k)
            for k in ("n", "n_written", "trained_messages", "masked_messages", "warnings")
        }
        print(
            f"{name:10s} n={rep['n_written']:4d} trained_messages={rep['trained_messages']:5d} "
            f"masked_messages={rep['masked_messages']:5d} warnings={rep.get('warnings')}"
        )

    census = tool_call_census(out_dir / "train.assistant.jsonl")
    print(f"\ntool calls in the set              : {census['tool_calls']}")
    print(f"supervised by mask_mode='final'    : {census['supervised_by_final']}")
    print(
        f"never supervised by mask_mode='final': "
        f"{census['tool_calls'] - census['supervised_by_final']}"
    )

    with open(out_dir / "holdout.jsonl", "w", encoding="utf-8") as fh:
        for r in hold_rows:
            fh.write(json.dumps(r) + "\n")
    with open(out_dir / "eval_context.json", "w", encoding="utf-8") as fh:
        json.dump({"system_prompt": POLICY, "tools": TOOLS}, fh, indent=1)

    return {
        "pooled": len(rows),
        "train_rows": len(train_rows),
        "holdout": len(hold_rows),
        "selected": len(list(sel)),
        "clean": len(clean),
        "decontaminate": counts,
        "exports": reports,
        "tool_call_census": census,
    }


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=6, help="how many simulate() seeds to pool")
    ap.add_argument("--limit", type=int, default=None, help="alias for --seeds, for smoke runs")
    ap.add_argument("--dry-run", action="store_true", help="two seeds, offline; no key, no GPU")
    ap.add_argument("--out", default=str(HERE / "data"), help="where the jsonl goes")
    args = ap.parse_args()

    seeds = args.limit or (2 if args.dry_run else args.seeds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = build(seeds, out_dir)
    (out_dir / "prep.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"\nwrote {out_dir}/prep.json")


if __name__ == "__main__":
    main()
