"""GRPO group size at a fixed rollout budget: prep the sets, then measure the arms.

    python run.py prep        # build the prompt set and the holdout, offline
    python run.py analyze     # every number, from arm_g*.json
    python run.py --dry-run   # both halves on tiny synthetic arms, no GPU, no key

Between the two halves:

    modal run sweep_modal.py  # three arms on your own Modal

Every arm spends the same 48 x 16 = 768 rollouts. Group size G decides only
how those rollouts are grouped: 768/G prompt visits. So this measures group
size against prompt coverage at equal GPU cost, which is not what sweeping
``num_generations`` in a trainer usually measures -- there the generation
batch moves with G and the arms cost different amounts.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import whileai as wai
from whileai.config import provenance
from whileai.simulations import eval_variance, holdout_size

HERE = Path(__file__).resolve().parent
SEED = 0
N_HOLDOUT = 100
N_TRAIN = 384
ARMS = (2, 4, 8)

PROMPT = (
    "Solve the problem. Reason step by step, then give the final "
    "numeric answer on the last line as: #### <answer>\n\nProblem: {q}"
)


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


# --------------------------------------------------------------------------- prep


def prep() -> None:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    rng = random.Random(SEED)
    test, train = list(ds["test"]), list(ds["train"])
    rng.shuffle(test)
    rng.shuffle(train)

    holdout = [
        {
            "prompt": PROMPT.format(q=r["question"]),
            "reference": gold(r["answer"]),
            "task_id": f"ho-{i}",
        }
        for i, r in enumerate(test[:N_HOLDOUT])
    ]
    train_rows = [
        {
            "prompt": PROMPT.format(q=r["question"]),
            "reference": gold(r["answer"]),
            "task_id": f"tr-{i}",
        }
        for i, r in enumerate(train[:N_TRAIN])
    ]

    kept, report = wai.decontaminate(train_rows, holdout)
    print("decontaminate:", {k: report[k] for k in ("n", "n_kept", "n_contaminated")})
    print("notes:", report.get("notes"))

    # Size the run before spending the GPU.
    for effect in (0.05, 0.10, 0.15):
        need = holdout_size(effect, base=0.45, k=4)
        print(
            f"effect {effect:+.2f} -> {need['n_tasks']} tasks "
            f"({need['n_tasks_concentrated']} if the gain is concentrated)"
        )

    (HERE / "holdout.json").write_text(json.dumps(holdout, indent=1))
    (HERE / "train_prompts.json").write_text(json.dumps(kept, indent=1))
    print(f"wrote holdout.json ({len(holdout)}) and train_prompts.json ({len(kept)})")


# ------------------------------------------------------------------------ analyze


def build(holdout: list, completions: list) -> list:
    return wai.rows(
        [r["prompt"] for r in holdout],
        completions,
        wai.verify.MathEqual(),
        references=[r["reference"] for r in holdout],
        task_ids=[r["task_id"] for r in holdout],
    )


def analyze(arm_dir: Path = HERE, holdout_path: Path | None = None) -> dict:
    holdout = json.loads((holdout_path or (HERE / "holdout.json")).read_text())
    data = {g: json.loads((arm_dir / f"arm_g{g}.json").read_text()) for g in ARMS}

    base_rows = {g: build(holdout, data[g]["base"]) for g in ARMS}
    after_rows = {g: build(holdout, data[g]["after"]) for g in ARMS}

    print("=" * 72)
    print("NOISE FLOOR  (three base passes of the same weights, three containers)")
    print(json.dumps(eval_variance(*[base_rows[g] for g in ARMS]), indent=1, default=str))
    base_means = [wai.pass_at(base_rows[g]).pass_at_1 for g in ARMS]
    run_std = statistics.stdev(base_means)
    print(f"\nbase pass@1 per container: {[round(m, 4) for m in base_means]}")
    print(f"spread {max(base_means) - min(base_means):.4f}   sd {run_std:.4f}")

    print("\n" + "=" * 72)
    print("PER-ARM DELTA  (trained vs its own base, paired, noise floor supplied)")
    per_arm = {}
    for g in ARMS:
        rep = wai.compare(base_rows[g], after_rows[g], run_std=run_std, run_std_runs=len(ARMS))
        print(f"\n--- G={g} ({data[g]['prompt_visits']} prompt visits) ---")
        print(rep)
        per_arm[g] = rep

    print("\n" + "=" * 72)
    print("ARM vs ARM  (does group size itself move the result?)")
    pairs = {}
    for a, b in ((2, 8), (2, 4), (4, 8)):
        rep = wai.compare(after_rows[a], after_rows[b], run_std=run_std, run_std_runs=len(ARMS))
        print(f"\n--- G={a} -> G={b} ---")
        print(rep)
        pairs[f"{a}->{b}"] = rep

    print("\n" + "=" * 72)
    print("BUDGET THAT CARRIED GRADIENT")
    print(
        "zero-adv is the share of groups that were unanimous, so their advantage\n"
        "was zero and those rollouts bought nothing. It has to fall as G rises --\n"
        "iid is p^G + (1-p)^G at the arm's own mean reward p -- so the column that\n"
        "matters is observed/iid: how much worse than chance the waste actually is."
    )
    print(
        f"{'G':>3} {'visits':>7} {'zero-adv':>9} {'iid':>7} {'ratio':>6} "
        f"{'useful':>7} {'reward':>7} {'len':>7} {'clip':>6}"
    )
    mean = lambda xs: statistics.fmean(xs) if xs else float("nan")  # noqa: E731
    budget = {}
    for g in ARMS:
        curve = data[g]["curve"]

        def pick(key: str, curve: list = curve) -> list:
            return [h[key] for h in curve if key in h]

        p = mean(pick("reward"))
        zero = mean(pick("frac_reward_zero_std"))
        iid = p**g + (1 - p) ** g
        row = {
            "zero_advantage": zero,
            "zero_advantage_iid": iid,
            "ratio_observed_over_iid": zero / iid if iid else float("nan"),
            "useful_rollouts": round(data[g]["rollouts"] * (1 - zero)),
            "reward": p,
            "length": mean(pick("completions/mean_length")),
            "clipped": mean(pick("completions/clipped_ratio")),
        }
        budget[g] = row
        print(
            f"{g:>3} {data[g]['prompt_visits']:>7} {zero:>9.3f} {iid:>7.3f} "
            f"{row['ratio_observed_over_iid']:>6.1f} {row['useful_rollouts']:>7} "
            f"{p:>7.3f} {row['length']:>7.1f} {row['clipped']:>6.3f}"
        )

    def verdict_of(rep: dict) -> dict:
        m = rep["metrics"]["pass_at_1"]
        return {
            "before": m["mean_a"],
            "after": m["mean_b"],
            "delta": m["delta"],
            "ci95": m["ci95"],
            "verdict": m["verdict"],
            "within_noise": m["within_noise"],
            "headline_verdict": rep["headline_verdict"],
            "noise_band": rep["noise_band"],
            "n_paired": m["n_paired"],
        }

    summary = {
        "base_means": base_means,
        "run_std": run_std,
        "noise_band": per_arm[ARMS[0]]["noise_band"],
        "per_arm": {str(g): verdict_of(per_arm[g]) for g in ARMS},
        "arm_vs_arm": {k: verdict_of(v) for k, v in pairs.items()},
        "arms": {
            str(g): {
                "prompt_visits": data[g]["prompt_visits"],
                "rollouts": data[g]["rollouts"],
                "base": wai.pass_at(base_rows[g]).pass_at_1,
                "after": wai.pass_at(after_rows[g]).pass_at_1,
                **budget[g],
            }
            for g in ARMS
        },
    }
    (arm_dir / "results_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote results_summary.json")
    return summary


# ------------------------------------------------------------------------ dry run


def dry_run() -> None:
    """Both halves on synthetic arms: no GPU, no key, no network."""
    import tempfile

    rng = random.Random(0)
    holdout = [
        {
            "prompt": PROMPT.format(q=f"synthetic question {i}"),
            "reference": str(i),
            "task_id": f"ho-{i}",
        }
        for i in range(60)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "holdout.json").write_text(json.dumps(holdout))
        for g, p in zip(ARMS, (0.46, 0.52, 0.58)):

            def draw(row: dict, q: float) -> list[str]:
                return [
                    f"#### {row['reference']}" if rng.random() < q else "#### 99999"
                    for _ in range(4)
                ]

            (d / f"arm_g{g}.json").write_text(
                json.dumps(
                    {
                        "group_size": g,
                        "steps": 48,
                        "gen_batch": 16,
                        "rollouts": 768,
                        "prompt_visits": 768 // g,
                        "base": [draw(r, 0.45) for r in holdout],
                        "after": [draw(r, p) for r in holdout],
                        "curve": [
                            {
                                "step": s,
                                "reward": 0.5,
                                "frac_reward_zero_std": 0.5 - 0.03 * g,
                                "completions/mean_length": 250.0,
                                "completions/clipped_ratio": 0.3,
                            }
                            for s in range(1, 13)
                        ],
                    }
                )
            )
        analyze(arm_dir=d, holdout_path=d / "holdout.json")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", nargs="?", choices=("prep", "analyze"), default="analyze")
    ap.add_argument("--dry-run", action="store_true", help="synthetic arms, no GPU and no key")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
    elif args.stage == "prep":
        prep()
    else:
        analyze()


if __name__ == "__main__":
    main()
