"""Lever 1, the harness: score every Deep Agents profile in candidates/ on dev, pick one.

No weights change. Each candidate is a `HarnessProfile` (prompt suffix, tool
set, tool descriptions); each runs in its own process over the dev split,
repos the holdout never sees. The pick is the most accurate candidate whose
calls did not go up; a tie on accuracy goes to fewer calls.

    python harness.py            # score all candidates on dev, write pick.json
    python harness.py --limit 40 # a cheaper search on the first 40 dev tasks

Then the pick plays the holdout like any other arm:
    python collect.py --arm harness --split holdout --profile candidates/<pick>.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import report

import whileai as wai
from whileai.config import provenance

HERE = pathlib.Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args(argv)

    files = sorted(CANDIDATES.glob("[0-9]*.py"))
    for f in files:
        cmd = [
            sys.executable,
            "-W",
            "ignore",
            "collect.py",
            "--arm",
            f"h-{f.stem}",
            "--split",
            "dev",
            "--profile",
            str(f),
            "--workers",
            str(args.workers),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd, cwd=HERE, check=True)

    stock = report.load(f"h-{files[0].stem}", "dev")
    scored = []
    for f in files:
        rows = report.load(f"h-{f.stem}", "dev")
        acc = wai.simulations.compare_runs(stock, rows)
        calls = wai.simulations.compare_runs(stock, rows, metric="marker:model_calls")
        scored.append(
            {
                "candidate": f.name,
                "accuracy": acc["mean_b"],
                "calls": calls["mean_b"],
                "accuracy_vs_stock": acc,
                "calls_vs_stock": calls,
            }
        )
        print(f"{f.name:28s} {report.summary(rows)}")

    # Calls going up is a cost, not a harness win: only keep candidates whose
    # calls interval does not say "more calls than stock".
    eligible = [s for s in scored if s["calls_vs_stock"]["verdict"] != "b_better"] or scored
    pick = max(eligible, key=lambda s: (round(s["accuracy"], 3), -s["calls"]))
    (HERE / "pick.json").write_text(
        json.dumps({"pick": pick["candidate"], "scored": scored}, indent=2)
    )
    print(f"\npick: {pick['candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
