"""Every arm against the stock base on one split: accuracy and calls, paired by task.

    python report.py                 # holdout
    python report.py --split dev     # the harness search's own split

Reads rows/<arm>.<split>.jsonl. Needs no key: it is arithmetic on saved rows.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import whileai.simulations as wai
from whileai.config import provenance

HERE = pathlib.Path(__file__).resolve().parent
ROWS = HERE / "rows"
MAX_CALLS = 40  # agent.review's max_steps


def load(arm: str, split: str) -> list[dict]:
    """Rows in the shape wai's paired stats read: an int reward and a calls marker."""
    import data

    path = ROWS / f"{arm}.{split}.jsonl"
    live = {t["id"] for t in data.load(split)}  # minus decontaminated and unavailable tasks
    last: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        last[r["task_id"]] = r  # a retried task's last row wins
    # An infrastructure failure (network, credits, tarball) is not a review the
    # model got wrong: leave it out, rerun it, and let the pairing see the gap.
    # Running out of steps is the model's doing and scores as wrong.
    # A review that ran out of steps used every call it was allowed; its row
    # says 0 only because the run raised before returning its messages.
    for r in last.values():
        if (r["error"] or "").startswith("GraphRecursionError"):
            r["model_calls"] = MAX_CALLS
    return [
        {**r, "reward": int(r["correct"]), "markers": {"model_calls": r["model_calls"]}}
        for tid, r in last.items()
        if tid in live and (not r["error"] or r["error"].startswith("GraphRecursionError"))
    ]


def arms(split: str) -> list[str]:
    return sorted(p.name.removesuffix(f".{split}.jsonl") for p in ROWS.glob(f"*.{split}.jsonl"))


def summary(rows: list[dict]) -> str:
    acc = sum(r["reward"] for r in rows) / len(rows)
    calls = sum(r["model_calls"] for r in rows) / len(rows)
    missing = sum(r["verdict"] is None for r in rows)
    return f"{acc:6.1%} correct  {calls:5.1f} calls  {missing:3d} no verdict  n={len(rows)}"


def delta(a: list[dict], b: list[dict], metric: str, scale: float = 1.0, unit: str = "") -> str:
    c = wai.compare_runs(a, b, metric=metric)
    lo, hi = c["ci95"] or (float("nan"), float("nan"))
    return f"{c['delta'] * scale:+.1f}{unit} [{lo * scale:+.1f}, {hi * scale:+.1f}] {c['verdict']}"


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="holdout")
    p.add_argument("--baseline", default="base")
    args = p.parse_args(argv)

    base = load(args.baseline, args.split)
    print(f"{args.split}: every arm vs {args.baseline}, paired by task, 95% bootstrap interval\n")
    for arm in arms(args.split):
        rows = load(arm, args.split)
        if not rows:
            print(f"{arm:28s} no scored reviews yet")
            continue
        line = f"{arm:28s} {summary(rows)}"
        if arm != args.baseline:
            line += (
                f"\n{'':28s} accuracy {delta(base, rows, 'pass_at_1', 100, ' pts')}"
                f"\n{'':28s} calls    {delta(base, rows, 'marker:model_calls')}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
