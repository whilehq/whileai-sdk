"""Grade rows sampled from the Bedrock import with the text-to-SQL verifier and
pair them, task by task, with the published rows of the same adapter served on
vLLM and of its base.

    python compare.py ../../04-train/text-to-sql/raw/<bedrock>.jsonl
    python compare.py --check-published        # regrade the published rows locally first

Needs the text-to-SQL recipe's Postgres (T2S_PG_DSN, or the default local
cluster). The two published files ship in rows/; a missing one is pulled from
the `while-ai/text-to-sql-shop` dataset on the Hub.

Numbers: pass@1 is the mean over tasks of the per-task pass rate (k samples);
pass@k is the share of tasks with at least one correct sample; intervals are a
percentile bootstrap over tasks (BOOTSTRAP resamples, seed 0); a paired delta
bootstraps the per-task difference, so the same tasks sit on both sides.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
T2S = HERE.parents[1] / "04-train" / "text-to-sql"
sys.path.insert(0, str(T2S))
from sql_verifier import load_tasks, verdict

HUB = "https://huggingface.co/datasets/while-ai/text-to-sql-shop/resolve/main/data"
PUBLISHED = {
    "vllm r1": "eval-nemotron-8b-r1.jsonl",
    "vllm base": "eval-nemotron-8b-base.jsonl",
}
# BOOTSTRAP = 2000 resamples: the interval endpoints move by under 0.005
# between seeds at 140 tasks (convention, checked once).
BOOTSTRAP = 2000
ROWS = HERE / "rows"


def fetch(name: str) -> Path:
    path = ROWS / name
    if not path.exists():
        ROWS.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f"{HUB}/{name}", path)
    return path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def per_task(rows: list[dict], id_key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        out[r[id_key]].append(float(r["reward"] or 0))
    return out


def boot(values: list[float], seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    draws = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return mean, draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP) - 1]


def summarize(name: str, tasks: dict[str, list[float]], extra: dict | None = None) -> None:
    p1 = [sum(v) / len(v) for v in tasks.values()]
    pk = [1.0 if any(v) else 0.0 for v in tasks.values()]
    m, lo, hi = boot(p1)
    line = (
        f"{name:<12} tasks {len(tasks):>3}  pass@1 {m:.3f} ({lo:.3f}..{hi:.3f})"
        f"  pass@k {sum(pk) / len(pk):.3f}"
    )
    if extra:
        line += "  " + "  ".join(f"{k} {v:.3f}" for k, v in extra.items())
    print(line)


def paired(a: dict[str, list[float]], b: dict[str, list[float]], label: str) -> None:
    ids = sorted(set(a) & set(b))
    d = [sum(a[i]) / len(a[i]) - sum(b[i]) / len(b[i]) for i in ids]
    m, lo, hi = boot(d)
    sign = "excludes zero" if lo > 0 or hi < 0 else "covers zero"
    print(f"{label:<28} {m:+.3f} ({lo:+.3f}..{hi:+.3f}) over {len(ids)} tasks, {sign}")


def by_difficulty(name: str, rows: list[dict], id_key: str) -> None:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r.get("difficulty") or "?"][r[id_key]].append(float(r["reward"] or 0))
    parts = []
    for diff in ("easy", "medium", "hard"):
        t = groups.get(diff)
        if t:
            rate = sum(sum(v) / len(v) for v in t.values()) / len(t)
            parts.append(f"{diff} {rate:.2f} (n={len(t)})")
    print(f"  {name:<12} " + "  ".join(parts))


def grade(path: Path) -> list[dict]:
    """Rows from rollout.py, graded with the recipe's verifier."""
    gold = {t["id"]: t["sql"] for t in load_tasks()}
    graded = []
    for r in read(path):
        ref = (r.get("privileged") or {}).get("reference") or gold[r["scenario_id"]]
        correct, executes, reason = verdict(r.get("final_text") or "", ref)
        graded.append({**r, "reward": correct, "executes": executes, "reason": reason})
    return graded


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "rows", nargs="?", help="raw/<name>.jsonl written by rollout.py for the Bedrock ARN"
    )
    ap.add_argument(
        "--check-published",
        action="store_true",
        help="regrade the published vLLM rows with the local verifier and report agreement",
    )
    args = ap.parse_args()
    if not args.rows and not args.check_published:
        ap.error("give the rollout file, or --check-published")
    if args.check_published:
        gold = {t["id"]: t["sql"] for t in load_tasks()}
        for name, fname in PUBLISHED.items():
            rows = read(fetch(fname))
            agree = sum(
                int(verdict(r["reply"], gold[r["task_id"]])[0] == int(float(r["reward"])))
                for r in rows
            )
            print(
                f"{name}: local verifier agrees with the published reward on {agree}/{len(rows)} rows"
            )
        return

    bedrock = grade(Path(args.rows))
    ids = {r["scenario_id"] for r in bedrock}
    pub = {k: [r for r in read(fetch(f)) if r["task_id"] in ids] for k, f in PUBLISHED.items()}
    k = len(bedrock) // max(len(ids), 1)
    print(f"Bedrock rows: {len(bedrock)}  ({len(ids)} tasks, k={k})\n")
    summarize(
        "bedrock r1",
        per_task(bedrock, "scenario_id"),
        {
            "executes": sum(r["executes"] for r in bedrock) / len(bedrock),
            "no sql": sum(r["reason"].startswith("no sql") for r in bedrock) / len(bedrock),
            "truncated": sum(bool(r.get("truncated")) for r in bedrock) / len(bedrock),
        },
    )
    for name, rows in pub.items():
        summarize(name, per_task(rows, "task_id"))
    print()
    b = per_task(bedrock, "scenario_id")
    paired(b, per_task(pub["vllm r1"], "task_id"), "bedrock r1 - vllm r1")
    paired(b, per_task(pub["vllm base"], "task_id"), "bedrock r1 - vllm base")
    paired(
        per_task(pub["vllm r1"], "task_id"),
        per_task(pub["vllm base"], "task_id"),
        "vllm r1 - vllm base",
    )
    print("\nby difficulty (pass@1):")
    by_difficulty("bedrock r1", bedrock, "scenario_id")
    for name, rows in pub.items():
        by_difficulty(name, rows, "task_id")
    reasons: dict[str, int] = defaultdict(int)
    for r in bedrock:
        reasons[r["reason"].split(":")[0].split(" (")[0]] += 1
    print("\nbedrock outcomes:", dict(sorted(reasons.items(), key=lambda kv: -kv[1])))
    lat = sorted(r["latency_s"] for r in bedrock if r.get("latency_s"))
    if lat:
        print(
            f"latency s: median {lat[len(lat) // 2]:.1f}  p90 {lat[int(0.9 * len(lat))]:.1f}  max {lat[-1]:.1f}"
        )


if __name__ == "__main__":
    main()
