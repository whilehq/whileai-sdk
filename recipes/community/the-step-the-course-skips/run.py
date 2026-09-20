"""Build the rows lesson 7 exports, then prove the trained model on them.

The learn course stops at `# 2. Train. See recipes/04-train. This course
skips it.` This script does the halves either side of that line: it writes
`train.jsonl` and a locked `holdout.json` offline, and after
`train_modal.py` has run it reads `rows.json` back and prints the paired
delta with its interval and the noise floor.

Run: python recipes/community/the-step-the-course-skips/run.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
from pathlib import Path

import whileai as wai
from whileai.simulations import format_delta_report

ORD = re.compile(r"\bORD-\d{3,6}\b")
HERE = Path(__file__).parent


@wai.tool
def get_order(order_id: str) -> dict:
    """Look up an order by id."""
    ...


COMMON = dict(
    tools=[get_order],
    system_prompt="Help customers with orders.",
    simulator=False,
    mode="rl",
    repeats=4,
    repeat_policy="fixed",
    phrasings=4,
)


def call_name_and_args(call: dict):
    """A tool call has two spellings and nothing says which you are holding.

    Rollout rows (what you grade) carry ``{"name", "arguments": dict}``.
    Exported rows (what you train on) carry the OpenAI wire shape,
    ``{"function": {"name", "arguments": "<json string>"}}``. A reward that
    reads only one of them scores 0 on every row and reports it with a
    tight interval, which is indistinguishable from a hopeless agent.
    """
    fn = call.get("function") or call
    name = fn.get("name")
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return name, None
    return name, args if isinstance(args, dict) else None


def first_tool_call(messages):
    for m in messages:
        if m.get("role") != "assistant":
            continue
        calls = m.get("tool_calls") or []
        if calls:
            return calls[0]
        if (m.get("content") or "").strip():
            return None  # it spoke before it looked anything up
    return None


def looked_it_up(row) -> int:
    """1 when the agent's first move is get_order on an id from the ask.

    A program, not a judge: lesson 7's `int(not row["seeded"])` reads a
    field only `seeded_agent` writes, so it cannot score a real model.
    """
    asked = set(ORD.findall(row.get("prompt") or ""))
    call = first_tool_call(row.get("messages") or [])
    if not call:
        return 0
    name, args = call_name_and_args(call)
    if name != "get_order" or args is None:
        return 0
    return int(bool(asked) and str(args.get("order_id", "")).strip() in asked)


def judge(row):
    return {"reward": looked_it_up(row)}


def build(budget: int, holdout_tasks: int, seed: int, write: bool) -> int:
    data = wai.simulate(wai.seeded_agent([get_order]), budget=budget, seed=seed, **COMMON)
    scored = data.grade(judge=judge)
    rows = list(scored.rows)
    print("graded rows:", len(rows))
    print("pass_at:", scored.pass_at)

    # Lesson 3 says check the checker. An all-zero or all-one reward is
    # indistinguishable from a broken one, so refuse to go on.
    share = sum(1 for r in rows if r.get("reward")) / max(len(rows), 1)
    print(f"checker: reward=1 on {share:.3f} of rows")
    if share in (0.0, 1.0):
        raise SystemExit(f"reward is degenerate ({share:.3f}); the checker is wrong, not the agent")

    eligible = sorted({r["scenario_id"] for r in rows if ORD.findall(r.get("prompt") or "")})
    print(f"scenarios: {len({r['scenario_id'] for r in rows})} ({len(eligible)} name an order id)")
    holdout_tasks = min(holdout_tasks, max(len(eligible) - 1, 1))

    # Lesson 5: split by task, so all four tries of one ask land on one side.
    held_ids = set(eligible[:holdout_tasks])
    holdout = [r for r in rows if r["scenario_id"] in held_ids]

    # Lesson 6: keep the passes. Select on the ScoredData -- that is the
    # only path that carries the system prompt and the tool schema into the
    # file (#592) -- and apply the split and decontamination afterwards.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "all.jsonl"
        exported = wai.select(scored, mode="sft").export(str(staged))
        print(
            "export:",
            {k: exported[k] for k in ("n", "with_system", "with_tools", "trained_messages")},
        )
        all_rows = [json.loads(x) for x in staged.read_text().splitlines()]

    train_rows = [r for r in all_rows if r.get("scenario_id") not in held_ids]
    kept, report = wai.decontaminate(train_rows, against=holdout)
    print(
        "decontaminate:",
        {
            k: report[k]
            for k in (
                "n",
                "n_kept",
                "n_contaminated",
                "n_same_task",
                "n_exact",
                "n_near",
                "n_semantic",
            )
        },
    )

    seen, tasks = set(), []
    for r in holdout:
        if r["scenario_id"] in seen:
            continue
        seen.add(r["scenario_id"])
        tasks.append(
            {
                "scenario_id": r["scenario_id"],
                "prompt": r["prompt"],
                "order_ids": sorted(set(ORD.findall(r["prompt"] or ""))),
            }
        )

    print(f"train rows: {len(kept)} | holdout tasks: {len(tasks)}")
    if kept:
        print(
            "  rows carry tools:",
            bool(kept[0].get("tools")),
            "| system:",
            kept[0]["messages"][0]["role"] == "system",
        )
    if not write:
        print("\n--dry-run: nothing written. The real run continues with")
        print("  modal run recipes/community/the-step-the-course-skips/train_modal.py")
        return 0

    with (HERE / "train.jsonl").open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    (HERE / "holdout.json").write_text(json.dumps(tasks, indent=1))
    (HERE / "tools.json").write_text(
        json.dumps(
            {
                "tools": (kept[0].get("tools") if kept else []),
                "system_prompt": COMMON["system_prompt"],
            },
            indent=1,
        )
    )
    print("\nwrote train.jsonl, holdout.json, tools.json. Next:")
    print("  modal run recipes/community/the-step-the-course-skips/train_modal.py")
    return 0


def report(rows_path: str, out_path: str) -> int:
    data = json.loads(Path(rows_path).read_text())
    base_passes, trained = data["base"], data["trained"]

    def rate(rs):
        return sum(r["reward"] for r in rs) / max(len(rs), 1)

    # The noise floor, read before either arm.
    rates = [rate(b) for b in base_passes]
    run_std = statistics.stdev(rates)
    print("noise floor (3 base passes):", " / ".join(f"{r:.3f}" for r in rates))
    print(f"  mean {statistics.mean(rates):.3f} | run_std {run_std:.4f}")
    print()

    before = base_passes[0]  # seed 101, the pass the trained arm is paired to
    print("before:", wai.pass_at(before, k=4))
    print("after: ", wai.pass_at(trained, k=4))
    print()

    rep = wai.compare(before, trained, run_std=run_std, run_std_runs=len(rates))
    print(format_delta_report(rep))

    d = rep["metrics"]["pass_at_1"]
    lo, hi = d["ci95"]
    excludes_zero = lo > 0 or hi < 0
    Path(out_path).write_text(
        json.dumps(
            {
                "base_pass_rates": rates,
                "base_mean": statistics.mean(rates),
                "run_std": run_std,
                "before_pass_at_1": rate(before),
                "after_pass_at_1": rate(trained),
                "delta": d["delta"],
                "ci95": [lo, hi],
                "excludes_zero": excludes_zero,
                "headline_verdict": rep.get("headline_verdict"),
                "n_tasks": len({r["scenario_id"] for r in before}),
            },
            indent=1,
        )
    )
    print("\nwrote", out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", nargs="?", default="build", choices=["build", "report"])
    p.add_argument("--budget", type=int, default=2400)
    p.add_argument("--holdout-tasks", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rows", default=str(HERE / "rows.json"))
    p.add_argument("--out", default=str(HERE / "results.json"))
    p.add_argument(
        "--dry-run", action="store_true", help="no model calls, no key, no GPU, nothing written"
    )
    a = p.parse_args(argv)

    if a.step == "report" and not a.dry_run:
        return report(a.rows, a.out)
    if a.step == "report":
        print("--dry-run: report needs rows.json from the Modal run.")
        return 0
    budget = 120 if a.dry_run else a.budget
    holdout = 4 if a.dry_run else a.holdout_tasks
    return build(budget, holdout, a.seed, write=not a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
