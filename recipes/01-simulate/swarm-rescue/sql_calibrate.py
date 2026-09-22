"""Pre-flight on text-to-SQL: is row overlap a hill?

Samples G replies per task from the hosted model on the text-to-SQL task set
(recipes/04-train/text-to-sql), grades each with the recipe's verifier, and
scores a graded fitness: the Jaccard overlap between the candidate's result
rows and the gold's. Prints P(exact match | fitness bucket) over the tasks
where the base fails every time, the population a swarm would work on. A
swarm needs that probability to rise with the bucket.

Run: python sql_calibrate.py --limit 600         # hosted Qwen3-4B, ~30 minutes
     python sql_calibrate.py --reuse             # the curve again from the saved rows
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04-train" / "text-to-sql"))
import run as R
import sql_verifier as V

G = 8
SQL_TOKENS = 512


def cell_f1(got: list[tuple], want: list[tuple]) -> float:
    """F1 over the multiset of cell values: credit for the right entities or
    numbers in the wrong shape, none for a query that runs but is unrelated."""
    from collections import Counter

    a, b = Counter(str(v) for t in got for v in t), Counter(str(v) for t in want for v in t)
    hit = sum((a & b).values())
    if not hit:
        return 0.0
    prec, rec = hit / sum(a.values()), hit / sum(b.values())
    return 2 * prec * rec / (prec + rec)


def fitness(text: str, gold_sql: str) -> dict:
    """Cell-level F1 against the gold result (and row Jaccard beside it),
    0 when the query does not run."""
    sql = V.extract_sql(text)
    if not sql:
        return {"fitness": 0.0, "executes": 0, "correct": 0, "why": "no sql"}
    try:
        got = V.norm_rows(V.run_sql(sql))
    except Exception as exc:
        return {"fitness": 0.0, "executes": 0, "correct": 0, "why": type(exc).__name__}
    want = V.gold_rows(gold_sql)
    a, b = {V._key(t) for t in got}, {V._key(t) for t in want}
    jac = len(a & b) / len(a | b) if (a | b) else 1.0
    correct = int(V.equivalent(got, want, ordered=V.has_order_by(gold_sql)))
    f1 = 1.0 if correct else cell_f1(got, want)
    return {
        "fitness": round(f1, 4),
        "jaccard": round(jac, 4),
        "executes": 1,
        "correct": correct,
        "why": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--out", default="out-sql")
    args = p.parse_args(argv)
    out_dir = R.HERE / args.out
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "base.jsonl"
    tasks = V.load_tasks()
    random.Random(args.seed).shuffle(tasks)
    tasks = tasks[: args.limit]
    if not (args.reuse and path.exists()):
        system = V.system_prompt()
        model = R.Model(R.HOSTED_URL, R.HOSTED_MODEL, R.resolve_api_key(), max_tokens=SQL_TOKENS)

        def one(task: dict) -> list[dict]:
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": task["question"]},
            ]
            rows = []
            for i in range(G):
                text = model.chat(msgs, seed=args.seed * 7919 + i)
                rows.append(
                    {
                        "task": task["id"],
                        "difficulty": task.get("difficulty"),
                        "text": text,
                        **fitness(text, task["sql"]),
                    }
                )
            return rows

        with (
            path.open("w", encoding="utf-8") as fh,
            cf.ThreadPoolExecutor(max_workers=args.workers) as ex,
        ):
            for n, rows in enumerate(ex.map(one, tasks), 1):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                if n % 25 == 0:
                    print(f"  {n}/{len(tasks)} tasks, calls {model.calls}", file=sys.stderr)
        print(
            f"calls {model.calls}, tokens {model.prompt_tokens}+{model.completion_tokens}",
            file=sys.stderr,
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if args.reuse:
        # Regrade the saved replies: the fitness definition may have changed.
        gold = {t["id"]: t["sql"] for t in V.load_tasks()}
        rows = [{**r, **fitness(r["text"], gold[r["task"]])} for r in rows]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    all_fail = {t for t, rs in by_task.items() if not any(r["correct"] for r in rs)}
    near = {t for t in all_fail if max(r["fitness"] for r in by_task[t]) > 0}
    print(
        f"\n{len(by_task)} tasks, {G} samples each; all-fail {len(all_fail)}, near-miss {len(near)}"
    )
    edges = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999]

    def curve(subset: set[str] | None, title: str) -> None:
        b: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in rows:
            if subset is not None and r["task"] not in subset:
                continue
            f = r["fitness"]
            k = "0" if f == 0 else "1.0" if f >= 1 else f"{max(e for e in edges if e <= f):.2f}+"
            b[k][1] += 1
            b[k][0] += r["correct"]
        print(f"\nP(exact match | fitness bucket), {title}")
        for k in sorted(b, key=lambda x: (x != "0", x)):
            c, n = b[k]
            print(f"  fitness {k:>6}: {c:5d}/{n:<6d} = {c / n:.3f}")

    curve(None, "all samples")
    # A swarm sees only the tasks it is given: those whose base group all failed.
    # The interesting curve is on the *arms'* samples there, which we do not have
    # yet; the base curve says whether partial overlap is ever near a match.
    fails = {t: rs for t, rs in by_task.items() if any(not r["correct"] for r in rs)}
    mixed = {t for t, rs in fails.items() if any(r["correct"] for r in rs)}
    curve(mixed, "tasks with a mixed base group (the same task has passes and fails)")
    # Task level: do the tasks whose WRONG attempts score higher get solved more
    # often? That is the closeness a swarm needs the fitness to carry.
    xs, ys = [], []
    for rs in by_task.values():
        wrong = [r["fitness"] for r in rs if not r["correct"]]
        if wrong:
            xs.append(sum(wrong) / len(wrong))
            ys.append(sum(r["correct"] for r in rs) / len(rs))
    if len(xs) > 10:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        print(
            f"\ntask level, {len(xs)} tasks: corr(mean fitness of wrong attempts, task pass rate) = "
            f"{cov / (vx * vy) if vx and vy else float('nan'):.3f}"
        )
        bins = defaultdict(lambda: [0.0, 0])
        for x, y in zip(xs, ys):
            k = "0" if x == 0 else "<0.25" if x < 0.25 else "<0.5" if x < 0.5 else "0.5+"
            bins[k][0] += y
            bins[k][1] += 1
        print(
            "  task pass rate by mean wrong-attempt fitness:",
            {k: f"{v[0] / v[1]:.2f} (n={v[1]})" for k, v in sorted(bins.items())},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
