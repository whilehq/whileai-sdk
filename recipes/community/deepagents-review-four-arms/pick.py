"""While's pick of the training traces: correct AND lean, spread over ways of being right.

smithtune's own pick is "the traces a council of judges keeps against a rubric".
This is the other pick, made by `wai.simulations.optimize(mode="sft")` over the same pool:
a trace qualifies only if its verdict matched the hidden tests and it used at
most BUDGET model calls, and among those `select_for_sft` round-robins over
tool-call signatures so one habit does not fill the set.

    python pick.py --target 150

The pick is written back to LangSmith as feedback `while_pick = 1` on each
chosen run, so smithtune pulls it with an ordinary filter:
    smithtune dataset pull ... --filter 'and(eq(feedback_key, "while_pick"), eq(feedback_score, 1))'

Experiment B of the rerun (PREREGISTRATION.md) picks over the original traces
plus the aimed rollouts, with the same rule and its own feedback key:
    python pick.py --arms base,b-aimed --key while_pick_b --out while_pick_b.json --target 1000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import report
from post import BUDGET

import whileai as wai
from whileai.config import provenance

HERE = pathlib.Path(__file__).resolve().parent
PICK = HERE / "while_pick.json"


def tool_sequences(client, rows: list[dict]) -> dict[str, list[str]]:
    """Tool names per run. New rows carry them; older ones are read back from
    LangSmith in batches of 100 (one list call, not one read per run)."""
    out = {r["run_id"]: r["tools"] for r in rows if "tools" in r}
    missing = [r["run_id"] for r in rows if "tools" not in r]
    for i in range(0, len(missing), 100):
        batch = missing[i : i + 100]
        names: dict[str, list[tuple]] = {rid: [] for rid in batch}
        for tool in client.list_runs(
            trace_id=None,
            run_type="tool",
            parent_run_id=None,
            filter=f"in(trace_id, {json.dumps(batch)})".replace('"', "'"),
            project_name="deepagents-review",
            select=["name", "trace_id", "start_time"],
        ):
            names[str(tool.trace_id)].append((tool.start_time, tool.name))
        out.update({rid: [n for _, n in sorted(v)] for rid, v in names.items()})
    return out


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--target", type=int, default=150, help="traces to keep; match smithtune's count"
    )
    p.add_argument("--dry-run", action="store_true", help="pick, but write no feedback")
    p.add_argument("--arms", default="base", help="comma-separated row sets to pool")
    p.add_argument("--key", default="while_pick", help="the LangSmith feedback key to write")
    p.add_argument("--out", default=PICK.name, help="where to write the picked run ids")
    args = p.parse_args(argv)

    from langsmith import Client

    ls = Client()
    # The same pool smithtune pulls: finished, un-nudged reviews (a nudged
    # review spans two traces and smithtune's thread check rejects it).
    pool = [
        r
        for arm in args.arms.split(",")
        for r in report.load(arm, "train")
        if r["run_id"] and not r["error"] and not r.get("nudged")
    ]
    tools = tool_sequences(ls, pool)
    rows = []
    for r in pool:
        rows.append(
            {
                "task_id": r["task_id"],
                "scenario_id": r["task_id"],
                "prompt": r["task_id"],
                "steps": [{"tool": t} for t in tools.get(r["run_id"], [])],
                "final_text": f"VERDICT: {r['verdict']}",
                # 1 only when right AND within budget: a correct review that took
                # 20 calls is a demonstration of taking 20 calls.
                "reward": int(r["correct"] and r["model_calls"] <= BUDGET),
                "run_id": r["run_id"],
                "model_calls": r["model_calls"],
            }
        )

    picked, rep = wai.simulations.optimize(
        rows, mode="sft", target=args.target, output=str(HERE / ".cache" / "while_pick.sft.jsonl")
    )
    correct = sum(r["correct"] for r in pool)
    print(
        f"pool {len(pool)} traces, {correct} correct, "
        f"{sum(r['reward'] for r in rows)} correct within {BUDGET} calls; kept {len(picked)}",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {k: v for k, v in rep.items() if k in ("mode", "kept", "dropped", "notes")},
            indent=2,
            default=str,
        )
    )

    (HERE / args.out).write_text(
        json.dumps({"budget": BUDGET, "run_ids": [r["run_id"] for r in picked]}, indent=2)
    )
    if not args.dry_run:
        for r in picked:
            ls.create_feedback(r["run_id"], key=args.key, score=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
