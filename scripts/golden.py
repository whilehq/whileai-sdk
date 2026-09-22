#!/usr/bin/env python3
"""Golden-output harness: prove a change to the engine leaves simulate() alone.

Runs a fixed set of offline ``simulate()`` configurations at
``concurrency=1`` with fixed seeds, scrubs the keys that cannot be
reproducible (wall-clock timings and per-invocation ids), and writes one
JSON file per configuration. Run it before a change and after, then diff
the two directories.

    python scripts/golden.py capture /tmp/golden/before
    # ... make the change ...
    python scripts/golden.py capture /tmp/golden/after
    python scripts/golden.py diff /tmp/golden/before /tmp/golden/after

``diff`` exits 0 when every configuration is byte-identical and 1 when any
key differs, naming the configuration and the dotted path of each change,
so it drops straight into a shell ``&&`` chain. ``capture`` exits non-zero
if any configuration raises.

Offline only: the scripted agent from ``tests.helpers`` answers every
rollout, the mock world answers every tool call, and no configuration
needs an API key or a GPU. Nothing here is imported by the package; this
is a maintenance helper that lives outside the wheel.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers import GITHUB_SPEC, LINEAR_SPEC, simulate_offline  # noqa: E402

# Wall-clock keys. Same regex as tests/api/test_reproducibility.py; keep the
# two in step, because a key that is reproducible there must be pinned here.
TIMING = re.compile(r"(seconds|elapsed|rate|_s$|_at$|per_second)")

# Per-invocation identities, not seeded output. ``run_judge`` stamps
# ``lineage.scoring_run_id`` with a fresh ``uuid4`` on every call, so a
# ``grader=`` run carries an id that differs between two identical runs by
# design. Scrubbed here for the same reason timing is: a key that cannot be
# reproducible must not be compared, or every diff is noise.
IDENTITY = frozenset({"scoring_run_id", "prior_scoring_run_id"})

# Rows shaped like production traces, for the trace-steered configurations.
TRACES = [
    {
        "prompt": f"refund order 8{i}",
        "reward": 0,
        "final_text": "Refunded.",
        "steps": [
            {
                "tool": "create_refund",
                "arguments": {"order_id": f"8{i}"},
                "result": {"status": "timeout"},
            }
        ],
    }
    for i in range(5)
]


def _tool_count_grader(row: dict) -> int:
    """A grader that is a pure function of the row, so it stays seeded."""
    return 1 if len(row.get("steps") or ()) >= 2 else 0


# Each entry is (name, kwargs for tests.helpers.simulate_offline). Every one
# runs serially on a fixed seed; the point is breadth of code path, not size.
CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("explore_small", dict(budget=12, per_round=12)),
    ("explore_graded", dict(budget=12, per_round=12, grade="conduct")),
    ("explore_seed_7", dict(budget=12, per_round=12, seed=7)),
    ("explore_wide", dict(budget=24, per_round=24)),
    ("explore_mutating", dict(budget=16, per_round=16, mutate_failures=True)),
    ("grader_callable", dict(budget=12, per_round=12, grader=_tool_count_grader)),
    ("traces_steered", dict(budget=24, per_round=40, traces=TRACES)),
    ("traces_steered_graded", dict(budget=24, per_round=40, traces=TRACES, grade="conduct")),
    ("rl_repeats", dict(budget=24, per_round=24, mode="rl", repeats=3)),
    ("rl_phrasings", dict(budget=24, per_round=24, mode="rl", phrasings=3, repeats=2)),
    ("unique_situations", dict(budget=12, per_round=12, unique=True)),
    ("spec_github", dict(budget=12, per_round=12, spec=str(GITHUB_SPEC))),
    ("spec_linear", dict(budget=12, per_round=12, spec=str(LINEAR_SPEC))),
]


def _comparable(key: Any) -> bool:
    name = str(key)
    return not TIMING.search(name) and name not in IDENTITY


def scrub(obj: Any) -> Any:
    """Drop every timing and per-run identity key, recursively."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if _comparable(k)}
    if isinstance(obj, (list, tuple)):
        return [scrub(x) for x in obj]
    return obj


def snapshot(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Run one configuration and return its scrubbed, comparable output."""
    data = simulate_offline(concurrency=1, **kwargs)
    return scrub(
        {
            "rows": data.trajectories,
            "search": data.search,
            "coverage": data.coverage,
            "coverage_curve": data.coverage_curve,
            "arm_yield": data.arm_yield,
            "arm_weights": data.arm_weights,
            "allocator": data.allocator,
            "stages": data.stages,
            "degraded": data.degraded,
            "stopped_because": data.stopped_because,
            "unique_prompts": data.unique_prompts,
            "unique_behavior_signatures": data.unique_behavior_signatures,
            "semantic_duplicate_rate": data.semantic_duplicate_rate,
            "mode": data.mode,
            "repeat_policy": data.repeat_policy,
            "n_situations": data.n_situations,
            "requests_per_situation": data.requests_per_situation,
            "rollouts_per_request": data.rollouts_per_request,
            "unique_situations": data.unique_situations,
            "budget": data.budget,
            "metadata": data.metadata,
        }
    )


def capture(out_dir: Path, only: str | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, kwargs in CONFIGS:
        if only and only != name:
            continue
        try:
            payload = snapshot(kwargs)
        except Exception as exc:  # the harness reports every config, never aborts
            failures += 1
            print(f"{name}: RAISED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        rows = len(payload["rows"])
        print(f"{name}: {rows} rows -> {path}")
    return 1 if failures else 0


def _walk(before: Any, after: Any, path: str, report: Callable[[str], None]) -> None:
    if type(before) is not type(after):
        report(f"{path}: {type(before).__name__} -> {type(after).__name__}")
        return
    if isinstance(before, dict):
        for key in sorted(set(before) | set(after)):
            if key not in before:
                report(f"{path}.{key}: added")
            elif key not in after:
                report(f"{path}.{key}: removed")
            else:
                _walk(before[key], after[key], f"{path}.{key}", report)
        return
    if isinstance(before, list):
        if len(before) != len(after):
            report(f"{path}: length {len(before)} -> {len(after)}")
        for i in range(min(len(before), len(after))):
            _walk(before[i], after[i], f"{path}[{i}]", report)
        return
    if before != after:
        report(f"{path}: {before!r} -> {after!r}")


def diff(before_dir: Path, after_dir: Path, limit: int = 20) -> int:
    names = sorted(
        {p.stem for p in before_dir.glob("*.json")} | {p.stem for p in after_dir.glob("*.json")}
    )
    if not names:
        print(f"no snapshots in {before_dir} or {after_dir}", file=sys.stderr)
        return 1
    changed = 0
    for name in names:
        before_path, after_path = before_dir / f"{name}.json", after_dir / f"{name}.json"
        if not before_path.exists() or not after_path.exists():
            missing = before_path if not before_path.exists() else after_path
            print(f"{name}: MISSING {missing}")
            changed += 1
            continue
        before, after = json.loads(before_path.read_text()), json.loads(after_path.read_text())
        lines: list[str] = []
        _walk(before, after, name, lines.append)
        if lines:
            changed += 1
            print(f"{name}: {len(lines)} differences")
            for line in lines[:limit]:
                print(f"  {line}")
            if len(lines) > limit:
                print(f"  ... and {len(lines) - limit} more")
        else:
            print(f"{name}: identical")
    if changed:
        print(f"\n{changed} of {len(names)} configurations changed", file=sys.stderr)
        return 1
    print(f"\nall {len(names)} configurations identical")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="run every configuration and write snapshots")
    cap.add_argument("out_dir", type=Path)
    cap.add_argument("--only", help="run just this configuration")
    dif = sub.add_parser("diff", help="compare two snapshot directories")
    dif.add_argument("before_dir", type=Path)
    dif.add_argument("after_dir", type=Path)
    dif.add_argument("--limit", type=int, default=20, help="differences printed per config")
    sub.add_parser("list", help="print the configuration names")
    args = parser.parse_args(argv)
    if args.command == "capture":
        return capture(args.out_dir, args.only)
    if args.command == "diff":
        return diff(args.before_dir, args.after_dir, args.limit)
    for name, _ in CONFIGS:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
