"""Pre-flight for a swarm: does the fitness it climbs predict the pass it needs?

Regrades saved rollouts with a finer fitness (the share of FIT generated
tests passed, every test run, no early exit) and a target the fitness never
saw (the private tests plus TARGET more generated tests). Prints
P(target pass | fitness bucket). A swarm can only work on a hill: the
probability has to rise with the bucket, not jump from zero to one at 1.0.

Run: python calibrate.py --out out-27b            # the 27B base rollouts
     python calibrate.py --out out-27b --arms     # the arms' samples too
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R

FIT = 40  # generated tests that make the fine fitness
TARGET = 20  # generated tests, beyond FIT, that join the private tests as the target


def load_full_tasks(seed: int) -> dict[str, tuple[list, list]]:
    """Per task id: (fitness tests, target tests) from the raw parquet."""
    import pyarrow.parquet as pq

    out = {}
    for split in ("test", "valid"):
        for r in pq.read_table(R.RAW / f"{split}.parquet").to_pylist():
            tid = f"cc-{split}-{r['name'].split('.')[0].replace(' ', '_')}"
            gen = list(zip(r["generated_tests"]["input"], r["generated_tests"]["output"]))
            priv = list(zip(r["private_tests"]["input"], r["private_tests"]["output"]))
            out[tid] = (gen[:FIT], priv + gen[FIT : FIT + TARGET])
    return out


def run_all(code: str, tests: list[tuple[str, str]]) -> int:
    """Number of tests passed, every test run."""
    passed = 0
    for t in tests:
        p, _ = R.run_tests(code, [t])
        passed += p
    return passed


def grade_one(item: tuple[str, str, list, list]) -> dict:
    tid, text, fit_tests, target_tests = item
    code = R.extract_code(text)
    fit_pass = run_all(code, fit_tests) if fit_tests else 0
    # the target is all-or-nothing, so it can stop at the first failure
    _passed, fail = R.run_tests(code, target_tests)
    return {
        "task_id": tid,
        "fitness": fit_pass / len(fit_tests) if fit_tests else 0.0,
        "fit_passed": fit_pass,
        "fit_total": len(fit_tests),
        "target_pass": fail is None,
        "target_total": len(target_tests),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="out-27b")
    p.add_argument("--arms", action="store_true", help="include the arms' samples")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    out_dir = R.HERE / args.out
    base = json.loads((out_dir / "base.json").read_text(encoding="utf-8"))
    tasks = load_full_tasks(0)
    all_fail = [t for t, b in base.items() if not any(s["grade"]["correct"] for s in b)]
    items = []
    for t in all_fail:
        fit_tests, target_tests = tasks[t]
        if len(fit_tests) < FIT or len(target_tests) < 5:
            continue
        for s in base[t]:
            items.append((t, s["text"], fit_tests, target_tests))
        if args.arms:
            for arm in R.ARMS:
                path = out_dir / f"arm-{arm}.json"
                if path.exists():
                    res = json.loads(path.read_text(encoding="utf-8"))
                    for s in res.get(t, {}).get("samples", []):
                        items.append((t, s["text"], fit_tests, target_tests))
    print(f"{len(all_fail)} all-fail tasks, {len(items)} samples to regrade", file=sys.stderr)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, row in enumerate(ex.map(grade_one, items), 1):
            rows.append(row)
            if n % 100 == 0:
                print(f"  {n}/{len(items)}", file=sys.stderr)
    (out_dir / "calibration.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    edges = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1.0]
    for r in rows:
        f = r["fitness"]
        if f == 0:
            k = "0"
        elif f >= 1.0:
            k = "1.0"
        else:
            lo = max(e for e in edges if e <= f)
            hi = min(e for e in edges if e > f)
            k = f"{lo:.2f}-{hi:.2f}"
        buckets[k][1] += 1
        buckets[k][0] += int(r["target_pass"])
    print(f"\nP(target pass | fitness over {FIT} generated tests), {len(rows)} samples")
    for k in sorted(buckets, key=lambda x: (x != "0", x)):
        c, n = buckets[k]
        print(f"  fitness {k:>10}: {c:4d}/{n:<5d} = {c / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
