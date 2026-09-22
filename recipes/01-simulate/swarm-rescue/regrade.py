"""Regrade a run's saved samples in place, from their text.

For a run whose grader broke mid-way (every sample "error: exit 3221225794")
or whose grading rule changed. Only the arms whose samples do not depend on
earlier grades are safe to regrade: resample and the noise re-runs draw
independent samples; solo, ring and star condition each round on the
grades of the last, so a broken grade there means a re-run.

Run: python regrade.py --out out-mbpp --tasks mbpp-bundle --files arm-resample noise-0 noise-1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", choices=["code-contests", "mbpp-bundle"], default="code-contests")
    p.add_argument("--bundle", type=int, default=R.BUNDLE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--files", nargs="+", default=["arm-resample", "noise-0", "noise-1"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    out_dir = R.HERE / args.out
    if args.tasks == "mbpp-bundle":
        tasks = {t.id: t for t in R.load_mbpp_bundles(None, args.seed, args.bundle)}
    else:
        tasks = {t.id: t for t in R.load_tasks(None, args.seed)}
    for name in args.files:
        path = out_dir / f"{name}.json"
        res = json.loads(path.read_text(encoding="utf-8"))
        items = [(tid, s) for tid, v in res.items() for s in v["samples"]]
        print(f"{name}: {len(items)} samples", file=sys.stderr)

        def one(item: tuple[str, dict]) -> None:
            tid, s = item
            s["grade"] = R.grade(s["text"], tasks[tid])

        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for n, _ in enumerate(ex.map(one, items), 1):
                if n % 500 == 0:
                    print(f"  {n}/{len(items)}", file=sys.stderr)
        for v in res.values():
            hits = [s["round"] for s in v["samples"] if s["grade"]["correct"]]
            v["rescued"] = bool(hits)
            v["round"] = min(hits) if hits else None
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        rescued = sum(v["rescued"] for v in res.values())
        print(f"{name}: rescued {rescued} of {len(res)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
