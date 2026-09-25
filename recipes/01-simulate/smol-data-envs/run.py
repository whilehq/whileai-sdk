"""SmolDataEnvs with whileai: data-analysis tasks, a program as the answer,
the dataset's exact-answer grader as the reward.

Offline, no key (canned replies over checked-in tables):

    python run.py

Live, any model the SDK can call, on the dataset's eval split:

    python run.py --agent openai:gpt-4.1-mini --split eval --limit 24 --k 4

The live run downloads the split and each task's tables from Hugging Face
(public, no token) into raw/, runs every reply's program locally, and writes
the graded rows to out/<name>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from env import (
    RAW,
    SYSTEM,
    DataEnvReward,
    fetch_tables,
    fixture_rollouts,
    fixture_tasks,
    hf_tasks,
    prompt_for,
    task_row_fields,
)

import whileai as wai
from whileai.config import provenance
from whileai.simulations.score.judging import run_judge

OUT = Path(__file__).resolve().parent / "out"


def offline_rows() -> list[dict]:
    tasks = {t["task_id"]: t for t in fixture_tasks()}
    rows = []
    for r in fixture_rollouts():
        task = tasks[r["scenario_id"]]
        rows.append({**r, "prompt": prompt_for(task), **task_row_fields(task)})
    return rows


def live_rows(args: argparse.Namespace) -> list[dict]:
    tasks = hf_tasks(args.split, limit=args.limit)
    print(f"{len(tasks)} {args.split} tasks; fetching tables into {RAW / 'tables'}", flush=True)
    tasks = [fetch_tables(t) for t in tasks]
    failed = [t["task_id"] for t in tasks if t.get("table_error")]
    if failed:
        print(f"  {len(failed)} tasks without tables; their rollouts will be ungraded", flush=True)
    data = wai.simulate(
        args.agent,
        system_prompt=SYSTEM,
        tasks=[{"prompt": prompt_for(t), "scenario_id": t["task_id"]} for t in tasks],
        repeats=args.k,
        max_turns=1,  # one reply: the program
        avg_turns=1,
        simulator=False,  # the prompts are the dataset's; no situation writer
        temperature=args.temperature,
        concurrency=args.concurrency,
        budget=len(tasks) * args.k,
        agent_max_tokens=args.max_tokens,
    )
    by_id = {t["task_id"]: t for t in tasks}
    return [
        {**r, **task_row_fields(by_id[r["scenario_id"]])}
        for r in data.trajectories
        if r.get("scenario_id") in by_id
    ]


def failure_kind(row: dict) -> str:
    """One bucket per way a rollout loses: ungraded, wrong value, which
    exception, timeout, shell, silence."""
    reason = str(row.get("reason", ""))
    if row.get("reward") is None:
        return "ungraded (environment failure)"
    if reason.endswith(": miss"):
        return "ran, wrong value"
    if reason.startswith("program failed:"):
        return "crashed: " + reason.split(":")[1].strip()
    return reason.split(":")[0]


def report(scored: list[dict], *, verbose: bool) -> None:
    if verbose:
        print("\n== each rollout")
        for r in scored:
            reward = "None" if r.get("reward") is None else r["reward"]
            print(
                f"  {r['scenario_id']:<28}{reward!s:>5}  {r.get('note', ''):<38}{r.get('reason', '')[:70]}"
            )
    else:
        print("\n== why rollouts failed")
        why = Counter(failure_kind(r) for r in scored if r.get("reward") != 1)
        for kind, n in why.most_common(8):
            print(f"  {n:>4}  {kind}")

    ungraded = sum(1 for r in scored if r.get("reward") is None)
    print(f"\n== pass@1 by difficulty ({ungraded} ungraded rollouts left out, not scored 0)")
    print(f"  {'tier':<8}{'tasks':>6}{'pass@1':>8}  {'95% CI':>14}")
    for tier in ("easy", "medium", "hard", "all"):
        sub = scored if tier == "all" else [r for r in scored if r.get("category") == tier]
        pa = wai.pass_at(sub, min_k=2)
        if not pa.n_groups:
            continue
        ci = f"{pa.ci95[0]:.2f}..{pa.ci95[1]:.2f}" if pa.ci95 else "n/a, <3 tasks"
        print(f"  {tier:<8}{pa.n_groups:>6}{pa.pass_at_1:>8.2f}  {ci:>14}")


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--agent",
        default="",
        help="SDK agent spec (openai:<model>, anthropic:<model>, vllm:<model>@<url>, ...). "
        "Omit for the offline run.",
    )
    ap.add_argument("--split", default="eval", choices=["train", "test", "eval"])
    ap.add_argument("--limit", type=int, default=24, help="first N tasks by task_id (0 = all)")
    ap.add_argument("--k", type=int, default=4, help="rollouts per task (GRPO group size)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048, help="reply budget per rollout")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args(argv)

    offline = not args.agent
    rows = offline_rows() if offline else live_rows(args)
    scored = run_judge(rows, DataEnvReward(), source="grade", concurrency=4).rows
    report(scored, verbose=offline)

    # The reward is the GRPO reward. select runs the RL gates: it drops
    # ungraded rows and groups where every rollout scored the same (no
    # gradient), and the export never carries the gold.
    picked = wai.select(scored, mode="rl")
    print(f"\n== wai.select(mode='rl')\n{picked}")

    name = "offline" if offline else args.agent.split("@")[0].replace(":", "-").replace("/", "-")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in scored:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"graded rows -> {path}")
    if picked:
        picked.export(str(OUT / f"{name}.rl.jsonl"))
        print(f"RL rows -> {OUT / f'{name}.rl.jsonl'}")
    if offline:
        print(
            "\nLive: python run.py --agent openai:gpt-4.1-mini --split eval --limit 24 --k 4"
            "\n(any OpenAI-compatible model via OPENAI_BASE_URL; needs pandas and pyarrow)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
