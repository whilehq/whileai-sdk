"""Read any row file the SDK ever wrote; stamp it, split it, report on it.

    python migrate.py                 # simulate 24 rows offline, then migrate them
    python migrate.py rows.jsonl      # migrate an existing file
    python migrate.py rows.jsonl --out some/dir

Writes ``<out>/rows.v1.jsonl`` (every row re-stamped as schema version 1),
``<out>/tasks.jsonl`` (situations only, no model output), and prints a shape
and validation report. No key needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import whileai.simulations as wai
from whileai.config import provenance
from whileai.simulations import schema

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
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
            "name": "create_refund",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]
POLICY = (
    "Look up an order before refunding it. If a lookup fails, say so "
    "and stop. Never invent an order id."
)


def scripted_agent(message: str) -> dict:
    """A scripted agent with bad habits: it refunds without looking up when
    the amount is large, invents an id when the lookup fails, and gets
    careless on every second rollout of the same prompt. That last one is
    what gives a task both a passing and a failing rollout, which is where
    preference pairs come from."""
    from whileai.simulations.generate.agents import current_rollout

    raw = int((re.search(r"(\d+)", message) or [0, "40"])[1])
    amount = raw % 200
    if getattr(current_rollout, "rollout_index", 0) % 2 == 1:
        amount = 150  # careless repeat: refund first, look up never
    # Ground the order id in the prompt, the way a well-behaved agent would.
    token = re.search(r"[A-Za-z]+[-_]?\d+", message)
    order = token.group(0) if token else f"ord_{amount}"
    if amount > 120:
        return {
            "steps": [
                {
                    "tool": "create_refund",
                    "arguments": {"order_id": f"acct_{raw * 7}", "amount": amount},
                    "result": {"status": "created", "id": f"re_{amount}"},
                }
            ],
            "final_text": f"Refunded ${amount}.",
        }
    if amount % 7 == 0:
        return {
            "steps": [
                {
                    "tool": "lookup_order",
                    "arguments": {"order_id": order},
                    "result": {"status": "not_found"},
                }
            ],
            "final_text": f"Refunded ${amount} successfully.",
        }
    return {
        "steps": [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": order},
                "result": {"status": "ok", "order_id": order},
            },
            {
                "tool": "create_refund",
                "arguments": {"order_id": order, "amount": amount},
                "result": {"status": "created", "id": f"re_{amount}"},
            },
        ],
        "final_text": f"Refunded ${amount}.",
    }


def simulate_rows(n: int = 24, seed: int = 0) -> list[dict]:
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=n,
        seed=seed,
        grade=True,
        concurrency=4,
        simulator=False,
        time_budget=None,
        mode="rl",
        repeats=2,
        advanced={"per_round": 8, "mutate_failures": False},
    )
    return [schema.to_row(*schema.from_row(r)) for r in data.trajectories]


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def migrate(rows: list[dict], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    shapes: Counter[str] = Counter()
    problems: Counter[str] = Counter()
    v1_rows: list[dict] = []
    tasks: dict[str, dict] = {}
    for row in rows:
        shapes[schema.detect_shape(row)] += 1
        for problem in schema.validate(row):
            problems[problem] += 1
        if not isinstance(row, dict):
            continue
        task, rollout, judgments, markers = schema.from_row(row)
        v1_rows.append(schema.to_row(task, rollout, judgments, markers))
        # One task per situation, and never a rollout inside it.
        tasks.setdefault(task.task_id, schema.as_dict(task))
    with open(out / "rows.v1.jsonl", "w", encoding="utf-8") as fh:
        for row in v1_rows:
            fh.write(json.dumps(row, default=str) + "\n")
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for task in tasks.values():
            fh.write(json.dumps(task, default=str) + "\n")
    schema.check(v1_rows, where="rows.v1.jsonl")
    return {
        "rows": len(v1_rows),
        "tasks": len(tasks),
        "shapes": dict(shapes),
        "problems": dict(problems),
        "out": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", nargs="?", help="JSONL to migrate; omit to simulate")
    parser.add_argument("--out", default="out")
    parser.add_argument("--rows", type=int, default=24)
    args = parser.parse_args(argv)
    rows = read_rows(Path(args.path)) if args.path else simulate_rows(args.rows)
    report = migrate(rows, Path(args.out))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
