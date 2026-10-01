"""Run one arm over a split, write rows, and score each trace in LangSmith.

    python collect.py --arm base --split train --limit 20
    python collect.py --arm base-r2 --split holdout --base-url ... --model ...

Rows land in rows/<arm>.<split>.jsonl and resume: a task already in the file is
skipped. Every root run gets LangSmith feedback `correctness` (1 or 0) and
`model_calls`, which is what `smithtune dataset pull --filter` selects on.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys

import agent
import data

HERE = pathlib.Path(__file__).resolve().parent
ROWS = HERE / "rows"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", required=True)
    p.add_argument("--split", required=True, choices=["train", "dev", "holdout", "public"])
    p.add_argument("--limit", type=int)
    p.add_argument("--tasks", help="FILE:KEY, a task-id list in a JSON file (aim.json:aimed)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--base-url", default=agent.OPENROUTER)
    p.add_argument("--model", default=agent.BASE_MODEL)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--profile", help="a candidates/*.py file; its PROFILE is the harness")
    args = p.parse_args(argv)

    if args.profile:
        # Registered per process: Deep Agents merges registrations for a key,
        # so two candidates in one process would blend into one harness.
        agent.use_profile(pathlib.Path(args.profile), args.model)

    ROWS.mkdir(exist_ok=True)
    out = ROWS / f"{args.arm}.{args.split}.jsonl"
    done = set()
    if out.exists():
        # A review that errored before the model decided (tarball, path, provider)
        # is retried; report.load keeps the last row per task.
        done = {
            r["task_id"]
            for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())
            if not r["error"] or r["error"].startswith("GraphRecursionError")
        }
    tasks = [t for t in data.load(args.split, args.limit) if t["id"] not in done]
    if args.tasks:
        path, key = args.tasks.rsplit(":", 1)
        wanted = set(json.loads(pathlib.Path(path).read_text(encoding="utf-8"))[key])
        tasks = [t for t in tasks if t["id"] in wanted]
    print(f"{args.arm}/{args.split}: {len(tasks)} to run, {len(done)} done", file=sys.stderr)

    chat = agent.model(args.base_url, args.model, os.environ[args.api_key_env])
    from langsmith import Client

    ls = Client()
    with out.open("a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(agent.review, t, chat, arm=args.arm): t for t in tasks}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            # Nothing in this loop may raise: an exception here stops the
            # writer while the pool keeps reviewing (and billing) unrecorded.
            try:
                row = fut.result()
            except Exception as e:
                t = futs[fut]
                row = {
                    "task_id": t["id"],
                    "arm": args.arm,
                    "split": t["split"],
                    "label": t["label"],
                    "verdict": None,
                    "correct": False,
                    "model_calls": 0,
                    "tool_calls": 0,
                    "nudged": False,
                    "run_id": None,
                    "error": f"{type(e).__name__}: {e}"[:300],
                }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if not row["error"]:
                try:
                    ls.create_feedback(row["run_id"], key="correctness", score=int(row["correct"]))
                    ls.create_feedback(row["run_id"], key="model_calls", score=row["model_calls"])
                except Exception as e:
                    print(f"feedback failed for {row['task_id']}: {e}", file=sys.stderr)
            print(
                f"[{i}/{len(tasks)}] {row['task_id']} {row['verdict']} "
                f"{'ok' if row['correct'] else 'x'} calls={row['model_calls']}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
