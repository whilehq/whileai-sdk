"""<Recipe name>: <paper> in one file.

    python recipe.py                          # both arms, writes results.json
    python recipe.py --arm recipe --steps 200 # one arm, longer

Shape of every recipe:
  1. data():      tasks + a task-disjoint holdout (public data or a seeded env in this dir),
                  decontaminated: train rows that overlap the holdout are dropped (Lambert 2025, chapter Evaluation)
  2. evaluate():  the untrained base, k samples per task, THREE times -> eval_variance run_std,
                  so a delta smaller than the eval's own noise is never called a result (chapter Evaluation)
  3. train(arm):  "baseline" or "recipe"; the recipe arm is the baseline plus ONE change
  4. evaluate():  each arm on the same holdout; delta_report with run_std and the training
                  reward named as proxy, so over-optimization is a verdict, not a vibe (chapter Over-optimization)
  5. results.json + the checks the README table reads

Training runs on Modal (TRL + LoRA, see recipes/04-train/grpo/train_modal.py)
or through the hosted trainer (wai.train(..., method=, loss_type=, beta=, ...)).
Keep the default under 60 GPU minutes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import whileai.simulations as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # recipes/papers, for cache_stamp

import cache_stamp

BASE_MODEL = "Qwen/Qwen3-4B"
METRIC = "pass@1"
BOOK = "Reinforcement Learning"  # the chapter title of Lambert 2025 this recipe tests or relies on
PROXY = None  # e.g. "marker:shaped_reward" when the training reward differs from the target
# The rule that decides `reward`, by name. It goes into results.json and,
# for a recipe that caches arms, into .cache/<arm>.json beside the rows:
# a cached row is a rollout, and a verdict is not a rollout, so `--reuse`
# re-grades the stored rollouts when this name or the whileai version
# changed rather than believing the stored reward (#737). Change the reader
# or the equality rule and change this name with it.
GRADER = "<the rule that decides reward, by name>"
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor


def data(seed: int) -> tuple[list[dict], list[dict]]:
    """Return (train_tasks, holdout_tasks). Holdout is split by task id, never by row."""
    raise NotImplementedError


def train(arm: str, tasks: list[dict], steps: int, seed: int) -> str:
    """Train one arm. Return an adapter path or a served model name."""
    raise NotImplementedError


def evaluate(model: str | None, holdout: list[dict], k: int, seed: int) -> list[dict]:
    """k samples per holdout task from `model` (None = the untrained base), graded rows.
    Every row carries `prompt`, `final_text`, `reward` (0/1 from a program or gold answer),
    and `markers` (the training reward under PROXY when it differs from the target)."""
    raise NotImplementedError


def summarize(rows: list[dict]) -> dict:
    p = wai.pass_at(rows)  # pass@1 with its task-bootstrap interval, pass@k, pass^k
    return {"score": p.pass_at_1, "ci": list(p.ci95 or (0.0, 0.0)), "pass_at_k": p.pass_at_k}


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_tasks, holdout = data(args.seed)
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)

    base_runs = [evaluate(None, holdout, args.k, args.seed + i) for i in range(EVAL_RUNS)]
    noise = wai.eval_variance(*base_runs)
    run_std = float(noise["run_std"])
    base_rows = base_runs[0]

    results: dict = {
        "recipe": HERE.name,
        "paper": "",
        "book": BOOK,
        "base_model": BASE_MODEL,
        "metric": METRIC,
        "n_holdout": len(holdout),
        "k": args.k,
        "arms": {"base": {**summarize(base_rows), "steps": 0, "gpu_minutes": 0}},
        "checks": {
            "run_std": run_std,
            "run_std_runs": int(noise["n_runs"]),
            "decontaminated_dropped": int(decon.get("n_contaminated", 0)),
            "over_optimized": False,
            "length_before": mean_length(base_rows),
            "length_after": {},
            "hack_scan_top": "",
            "seed": args.seed,
        },
        "verified": date.today().isoformat(),
        "whileai": version("whileai"),
        # Who decided every `reward` above. Belief 1: a number is a result
        # only with the versions that produced it, and the grader is one of
        # them. A results.json whose stamp is not the tree's is void until
        # it is re-graded.
        "grader": cache_stamp.stamp(GRADER),
    }
    arm_rows: dict[str, list[dict]] = {}
    for arm in ["baseline", "recipe"] if args.arm == "both" else [args.arm]:
        model = train(arm, train_tasks, args.steps, args.seed)
        arm_rows[arm] = evaluate(model, holdout, args.k, args.seed)
        results["arms"][arm] = {**summarize(arm_rows[arm]), "steps": args.steps}
        results["checks"]["length_after"][arm] = mean_length(arm_rows[arm])
    if len(arm_rows) == 2:
        d = wai.delta_report(
            arm_rows["baseline"],
            arm_rows["recipe"],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(noise["n_runs"]),
            # one training seed per arm: the report says unresolved; two or
            # more per arm (pass every seed's rows) resolve it to moved or flat (#356)
            train_runs={"before": [arm_rows["baseline"]], "after": [arm_rows["recipe"]]},
            proxy=PROXY,
        )
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": (
                "unresolved"
                if d["target_verdict"] == "unresolved"
                else "moved"
                if d["target_verdict"] == "moved"
                else "flat"
            ),
        }
        results["checks"]["train_seeds"] = {"baseline": 1, "recipe": 1}
        results["checks"]["over_optimized"] = bool(d["over_optimized"])
        print(wai.format_delta_report(d))
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
