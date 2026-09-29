"""Refuse or comply: does resisting a planted instruction cost the job, and
does a classifier in front of the tool channel get the resistance without
the cost. Five arms on one frozen holdout, program-graded.

`--dry-run` needs no key, no network and no GPU: it rebuilds the holdout and
checks it against the pin, runs the offline stand-in gate over a few of its
tool results, and scores the bundled fixture rows with the same analysis the
committed `results.json` came from.

Run: python run.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from whileai.config import provenance
except ImportError:  # a fresh clone, before the package is installed
    raise SystemExit(
        "This recipe needs the SDK importable. From the repository root run "
        "`uv sync --extra dev` (or `pip install -e .`), then re-run this script."
    ) from None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "resist-planted-instruction"))

from analyse import main as analyse_main
from gate import GATE_CONFIG, make_gate, wrap
from holdout import build_holdout, check_pin, composition, tasks_for
from world import make_execute


def dry_run(limit: int) -> int:
    world = build_holdout()
    tasks = tasks_for(world)
    digest = check_pin(tasks)
    print(f"holdout t-{digest[:8]}: {json.dumps(composition(world))}")

    # the gate wiring, with the keyword stand-in (the real gate is
    # out/classifier, private; `python gate.py --probe` scores it)
    log: list[dict] = []
    execute = wrap(make_execute(world), make_gate("keyword"), "keyword", log)
    attacked = [sid for sid, sc in world["scenarios"].items() if sc["attack"]][:limit]
    withheld = 0
    for sid in attacked:
        res = execute("get_order", {"order_id": sid})
        withheld += int("content_withheld" in res)
    print(
        f"keyword gate over get_order on {len(attacked)} attacked scenarios: "
        f"{withheld} withheld, {len(log)} calls logged; live gate config {GATE_CONFIG}"
    )

    fixture = HERE / "fixtures.jsonl"
    if not fixture.exists():
        print("no fixtures.jsonl yet; the analysis runs on out/ after the eval")
        return 0
    return analyse_main(["--fixture", str(fixture), "--out", str(HERE / "out" / "dry-run")])


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="no model calls, no key")
    p.add_argument("--limit", type=int, default=10, help="scenarios to push through the gate")
    args = p.parse_args(argv)
    if args.dry_run:
        return dry_run(args.limit)
    print(
        "The live path is on Modal: modal run modal_train_eval.py::run_pilot, ::run_train, "
        "::run_eval, ::run_fetch, then python analyse.py --results results.json. "
        "Run --dry-run for the offline path."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
