"""Experiment B's two draws of 400 new review tasks: aimed (While) and uniform (smithtune).

The rules are PREREGISTRATION.md's, fixed before any new rollout. Both draw
from the training-split tasks no original trace touched. The aimed draw puts
every task in a bucket from features known before a rollout (verdict label,
whether the patch changes non-test Python, files touched, lines changed),
weights each bucket by how often the stock agent failed there (wrong, or more
than BUDGET model calls) on the original traces, and draws in proportion.

    python aim.py            # writes aim.json: both task lists and the bucket table

    python collect.py --arm b-random --split train --tasks aim.json:random   # LANGSMITH_PROJECT=deepagents-review-b-random
    python collect.py --arm b-aimed  --split train --tasks aim.json:aimed    # LANGSMITH_PROJECT=deepagents-review-b-aimed
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys
from collections import Counter

import data
import report
from post import BUDGET

from whileai.config import provenance
from whileai.simulations.defaults import laplace

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "aim.json"
DRAW = 400
SEED = 0


def bucket(task: dict) -> str:
    """Features of the task alone, readable before the agent sees it."""
    files = re.findall(r"^diff --git a/(\S+)", task["patch"], flags=re.M)
    code = any(f.endswith(".py") and "test" not in f.lower() for f in files)
    changed = sum(
        1
        for line in task["patch"].splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )
    size = "s" if changed <= 10 else "m" if changed <= 40 else "l"
    return f"{task['label']}|{'code' if code else 'no-code'}|{'1f' if len(files) == 1 else 'nf'}|{size}"


def main() -> int:
    print(provenance(), file=sys.stderr)
    tasks = {t["id"]: t for t in data.load("train")}
    seen = [r for r in report.load("base", "train") if r["task_id"] in tasks]
    seen_ids = {
        json.loads(line)["task_id"]
        for line in (report.ROWS / "base.train.jsonl").read_text(encoding="utf-8").splitlines()
    }
    unused = sorted(tid for tid in tasks if tid not in seen_ids)

    fails, n = Counter(), Counter()
    for r in seen:
        b = bucket(tasks[r["task_id"]])
        n[b] += 1
        fails[b] += int(not (r["correct"] and r["model_calls"] <= BUDGET))
    weight = {b: laplace(fails[b], n[b]) for b in n}
    prior = laplace(0, 0)  # a bucket the traces never saw

    rng = random.Random(SEED)
    uniform = sorted(rng.sample(unused, DRAW))

    rng = random.Random(SEED)
    pool = list(unused)
    aimed = []
    for _ in range(DRAW):  # without replacement, proportional to bucket weight
        w = [weight.get(bucket(tasks[t]), prior) for t in pool]
        aimed.append(pool.pop(rng.choices(range(len(pool)), weights=w)[0]))
    aimed.sort()

    table = {
        b: {"traces": n[b], "failed": fails[b], "weight": round(weight[b], 4)} for b in sorted(n)
    }
    OUT.write_text(
        json.dumps(
            {
                "rule": "PREREGISTRATION.md, Experiment B",
                "unused": len(unused),
                "draw": DRAW,
                "seed": SEED,
                "buckets": table,
                "random": uniform,
                "aimed": aimed,
                "aimed_buckets": Counter(bucket(tasks[t]) for t in aimed),
                "random_buckets": Counter(bucket(tasks[t]) for t in uniform),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(unused)} unused tasks; wrote {DRAW} random and {DRAW} aimed to {OUT.name}")
    for b, row in table.items():
        print(f"  {b:28s} {row['failed']:3d}/{row['traces']:<3d} weight {row['weight']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
