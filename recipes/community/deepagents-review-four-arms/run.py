"""Four ways to improve a Deep Agents reviewer from its LangSmith traces, measured on repos it never saw.

The agent is stock Deep Agents on Qwen3.8-27B reviewing real patches (approve or
reject) whose hidden-test outcome is known. Its traces go to LangSmith; the arms
are a searched harness profile, smithtune SFT on smithtune's pick, the same SFT
on While's pick, and RL. While decides what the training set is, whether the
holdout is clean, and whether a gain clears the noise.

    python run.py data        # build tasks, decontaminate, drop deleted repos
    python run.py traces      # stock agent over the train split -> LangSmith
    python run.py harness     # search candidates/ on dev, write pick.json
    python run.py pick        # While's SFT pick -> LangSmith feedback while_pick=1
    python run.py holdout     # base x3 and every finished arm over the holdout
    python run.py report      # paired deltas, offline
    python run.py post        # the experiment on the While Runs page
    python run.py --dry-run   # offline: the report and pick logic on fixture rows

The SFT arms are trained with smithtune (see README: the two `smithtune` commands
differ only in the pull filter).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "rows.jsonl"


def dry_run() -> int:
    """The measurement half on eight hand-built reviews: paired deltas, the
    in-budget behavior, and While's pick rule. No key, no network, no GPU."""
    import whileai as wai

    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(
            {**r, "reward": int(r["correct"]), "markers": {"model_calls": r["model_calls"]}}
        )
    base = by_arm.pop("base")
    for arm, got in by_arm.items():
        acc = wai.simulations.compare_runs(base, got, min_paired=2)
        calls = wai.simulations.compare_runs(base, got, metric="marker:model_calls", min_paired=2)
        print(
            f"{arm}: accuracy {acc['delta']:+.2f}  calls {calls['delta']:+.1f}  (paired over {acc['n_paired']} tasks)"
        )

    budget = 8
    pool = [
        {
            **r,
            "scenario_id": r["task_id"],
            "prompt": r["task_id"],
            "steps": [{"tool": t} for t in r["tools"]],
            "final_text": f"VERDICT: {r['verdict']}",
            "reward": int(r["correct"] and r["model_calls"] <= budget),
        }
        for r in base
    ]
    picked, _ = wai.simulations.optimize(
        pool, mode="sft", target=len(pool), output=str(HERE / ".cache" / "dry.sft.jsonl")
    )
    print(
        f"While's pick: {len(picked)} of {len(pool)} base traces are correct within {budget} calls"
    )
    return 0


STEPS = {
    "data": [["data.py"]],
    "traces": [["collect.py", "--arm", "base", "--split", "train", "--limit", "400"]],
    "harness": [["harness.py"]],
    "pick": [["pick.py"]],
    "holdout": [
        ["collect.py", "--arm", a, "--split", "holdout"] for a in ("base", "base-r2", "base-r3")
    ],
    "report": [["report.py"]],
    "post": [["post.py"]],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", nargs="?", choices=list(STEPS))
    p.add_argument("--dry-run", action="store_true", help="offline, no key, no GPU")
    args = p.parse_args(argv)
    if args.dry_run or not args.step:
        (HERE / ".cache").mkdir(exist_ok=True)
        return dry_run()
    for cmd in STEPS[args.step]:
        subprocess.run([sys.executable, *cmd], cwd=HERE, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
