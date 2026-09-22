"""pass@1, pass^k and pass@k for one agent, offline, in seconds.

    python measure.py                      # 12 asks x 8 repeats with a scripted agent
    python measure.py graded.jsonl         # any graded row file (reward 0/1, grouped by task)
    python measure.py --asks 40 --k 16     # bigger grid

Three numbers off the same graded groups, one job each: pass@1 is what
production sees, pass^k is how often the agent is right every time, and
pass@k minus pass@1 is what a grouped RL update has to learn from. No key
needed for the scripted run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import whileai.simulations as wai
from whileai.config import provenance

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
    """A scripted agent whose reliability depends on the ask. Large amounts
    make it skip the lookup; a failed lookup makes it claim success; and on
    some asks it gets careless every third repeat. Some asks it always
    gets right, some never, and some only sometimes. That spread is what
    pass@1, pass^k and pass@k tell apart."""
    from whileai.simulations.generate.agents import current_rollout

    token = re.search(r"[A-Za-z]+[-_]?\d+", message)
    if token is None:
        # No order id in the ask: the honest move is to ask for one.
        return {"steps": [], "final_text": "Which order id should I look at?"}
    order = token.group(0)
    raw = int((re.search(r"(\d+)", message) or [0, "40"])[1])
    amount = raw % 200
    careless_ask = raw % 3 == 0
    if careless_ask and getattr(current_rollout, "rollout_index", 0) % 3 == 1:
        amount = 150  # careless repeat: refund first, look up never
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


def simulate_rows(
    asks: int = 12,
    k: int = 8,
    seed: int = 0,
    *,
    concurrency: int = 4,
    seeds: list[str] | None = None,
) -> list[dict]:
    """``asks`` situations, ``k`` repeats each, graded by the built-in
    conduct grader. Offline: template writer, scripted agent.
    ``reproducible=True`` makes the seed decide which asks are drawn at
    any concurrency; without it completion order steers later draws and
    the same seed prints different numbers run to run. ``seeds`` are asks
    kept as drawn, so a run can pin a situation the writer might not draw."""
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        situations=asks,
        repeats=k,
        budget=asks * k,
        seed=seed,
        seeds=seeds,
        grade="conduct",
        concurrency=concurrency,
        reproducible=True,
        simulator=False,
        time_budget=None,
        mode="rl",
    )
    return list(data.rows())


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def histogram(per_task: dict[str, float]) -> dict[str, int]:
    """Where each ask sits: never, sometimes, always. The mean hides this."""
    out = {"never (p=0)": 0, "sometimes (0<p<1)": 0, "always (p=1)": 0}
    for p in per_task.values():
        if p == 0:
            out["never (p=0)"] += 1
        elif p == 1:
            out["always (p=1)"] += 1
        else:
            out["sometimes (0<p<1)"] += 1
    return out


def report(rows: list[dict]) -> dict:
    rates = wai.pass_at(rows)
    signal = wai.group_signal(rows)
    out = {
        "summary": str(rates),  # the headline line, interval included
        "pass_at": rates.to_dict(),
        "histogram": histogram(rates.per_task),
        "n_mixed": signal["n_mixed"],
        "mixed_rate": signal["mixed_rate"],
        "verdict": [],
    }
    if rates.pass_at_1 is None:
        out["verdict"].append("no binary rewards: grade the rows first")
        return out
    out["verdict"].append(f"production sees pass@1 = {rates.pass_at_1:.0%}")
    if rates.pass_at_k is None:
        out["verdict"].append(rates.note)
        return out
    assert rates.pass_pow_k is not None and rates.headroom is not None
    gap = rates.pass_at_1 - rates.pass_pow_k
    if gap > 0.005:
        out["verdict"].append(
            f"the agent is right every time on pass^{rates.k} = {rates.pass_pow_k:.0%} of asks; "
            f"the {gap:.0%} gap to pass@1 is inconsistency, not inability"
        )
    else:
        out["verdict"].append(
            f"the agent is right every time on pass^{rates.k} = {rates.pass_pow_k:.0%} of asks, "
            "the same as pass@1: every ask it passes, it passes on every try"
        )
    if rates.headroom >= 0.1:
        out["verdict"].append(
            f"RL headroom {rates.headroom:.0%}: {signal['n_mixed']} mixed asks carry gradient; "
            "select_for_rl keeps whole groups"
        )
    else:
        out["verdict"].append(
            f"RL headroom {rates.headroom:.0%}: almost nothing a grouped update can learn here; "
            "harder cells or a stricter judge before buying more rollouts"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", nargs="?", help="graded JSONL; omit to simulate offline")
    parser.add_argument("--asks", type=int, default=12)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4, help="rollouts in flight")
    args = parser.parse_args(argv)

    rows = (
        load_rows(Path(args.rows))
        if args.rows
        else simulate_rows(args.asks, args.k, args.seed, concurrency=args.concurrency)
    )
    out = report(rows)
    print(out["summary"])
    for bucket, count in out["histogram"].items():
        print(f"  {bucket:<20} {count}")
    for line in out["verdict"]:
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
