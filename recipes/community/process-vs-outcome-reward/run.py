"""Does reward granularity survive a small-scale reproduction?

arXiv:2607.02869 (Palandye et al., July 2026) trains Qwen2.5-0.5B with GRPO
on GSM8K under five reward regimes and reports that scoring the *steps*
beats scoring the *answer* by about ten points of test accuracy: 63.73% for
process-only against 53.75% for outcome-only, from single runs with no
intervals.

This script is the measurement half of a two-arm reproduction. `prep` builds
a held-out set, decontaminates the training prompts against it and writes
`prompts.json`; `train_modal.py` runs the two arms on your own Modal;
`analyze` reads the rows back and prints the paired deltas with their 95%
intervals against the untrained base's noise floor.

    python run.py --dry-run     # offline: the reward programs, on a fixture
    python run.py prep          # writes prompts.json
    modal run train_modal.py    # both arms, ~35 min on one L40S each
    python run.py analyze       # the numbers
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import urllib.request
from pathlib import Path

from rewards import gold_final_answer, numbers_in, outcome_score, process_score

import whileai as wai
from whileai.simulations import eval_variance, holdout_size

HERE = Path(__file__).parent
GSM8K = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{}.jsonl"

N_HOLDOUT = 200
N_TRAIN = 192
K = 2  # rollouts per held-out question
SEED = 0

# One GSM8K row, verbatim, so --dry-run needs no network.
FIXTURE = {
    "question": (
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per fresh "
        "duck egg. How much in dollars does she make every day at the farmers' "
        "market?"
    ),
    "answer": (
        "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
        "She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n"
        "#### 18"
    ),
}


def fetch(split: str) -> list[dict]:
    path = HERE / f"gsm_{split}.jsonl"
    if not path.exists():
        with urllib.request.urlopen(GSM8K.format(split)) as r:
            path.write_bytes(r.read())
    return [json.loads(line) for line in path.open()]


def as_row(split: str, i: int, r: dict) -> dict:
    return {
        "id": f"{split}-{i}",
        "task_id": f"{split}-{i}",
        "question": r["question"],
        "prompt": r["question"],
        "gold_answer": r["answer"],
        "gold_final": gold_final_answer(r["answer"]),
    }


def dry_run() -> None:
    """The two reward programs, on one row, with no network and no key.

    The point of the fixture is that the two rewards must come apart: a
    chain with the right steps and the wrong answer scores 1.0 and 0.0, and
    a chain with the right answer and no steps does not score 1.0 on
    process. If they ever agree everywhere, the paper has no question.
    """
    q, gold = FIXTURE["question"], FIXTURE["answer"]
    final = gold_final_answer(gold)
    chain = re.sub(r"<<[^<>]*>>", "", gold.split("####")[0]).strip()

    cases = [
        ("the gold chain", f"{chain}\n#### {final}"),
        ("right steps, wrong answer", f"{chain}\n#### 12345"),
        ("right answer, no work", f"#### {final}"),
        (
            "padded chain, right answer",
            "\n".join(f"step {i}: {i}" for i in range(12)) + f"\n#### {final}",
        ),
    ]
    print(f"process/outcome on one GSM8K row (gold final = {final})\n")
    print(f"  {'completion':<28} {'R_process':>10} {'R_outcome':>10}")
    for name, completion in cases:
        print(
            f"  {name:<28} {process_score(completion, gold, q):>10.3f} "
            f"{outcome_score(completion, final):>10.3f}"
        )
    print(
        f"\nsizing: {holdout_size(0.10, base=0.35, k=K)['n_tasks']} paired tasks "
        f"for a 10-point gain at k={K}; this recipe uses {N_HOLDOUT}."
    )


def prep() -> None:
    rng = random.Random(SEED)
    test, train_pool = fetch("test"), fetch("train")

    holdout = [as_row("test", i, test[i]) for i in rng.sample(range(len(test)), N_HOLDOUT)]
    train = [
        as_row("train", i, train_pool[i]) for i in rng.sample(range(len(train_pool)), N_TRAIN + 32)
    ]

    print(holdout_size(0.10, base=0.35, k=K))
    clean, report = wai.decontaminate(train, against=holdout)
    print(
        f"decontaminate: {report['n_contaminated']} of {len(train)} dropped; "
        f"rules_skipped={report.get('rules_skipped')}"
    )

    clean = list(clean)[:N_TRAIN]
    assert len(clean) == N_TRAIN, len(clean)
    (HERE / "prompts.json").write_text(
        json.dumps(
            {
                "holdout": holdout,
                "train": clean,
                "seed": SEED,
                "decontamination": {k: v for k, v in report.items() if k != "offenders"},
            }
        )
    )
    print(f"wrote prompts.json: {len(holdout)} held out, {len(clean)} train")


def _mean(rows: list[dict], key: str = "reward") -> float:
    return sum(r[key] for r in rows) / len(rows)


def _metrics(rep: dict) -> dict:
    return {
        name: {k: m[k] for k in ("before", "after", "delta", "ci95", "verdict") if k in m}
        for name, m in rep["metrics"].items()
    }


#: Two-sided 95% t quantiles, for a library older than the #616 fix.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262}


def applied_band(ev: dict) -> float:
    """The band a one-run-per-side delta has to clear.

    ``run_std`` estimated from n re-runs is an estimate, not the eval's
    exact spread, so the quantile is Student's t at n - 1 degrees of
    freedom rather than 1.96: 4.30 from three re-runs. Since #616,
    ``eval_variance`` carries exactly that band and says so with
    ``noise_band_df``, and this reads it off the report. Older installs
    return the 1.96 form under the same key, so compute it there instead
    rather than judge the run against a band 2.2x too permissive.
    """
    if ev.get("noise_band_df") is not None:
        return float(ev["noise_band"])
    q = _T975.get(max(1, int(ev.get("n_runs", 3)) - 1), 1.96)
    return q * float(ev["run_std"]) * math.sqrt(2.0)


def _hack_scan(rows: list[dict]) -> dict:
    """Does the reward track the behavior, or a shortcut?

    A process reward invites one obvious cheat: emit many numbers and hope
    some match a gold step. ``numbers_emitted`` puts that cheat on the
    feature list explicitly, beside the built-in length and truncation
    features, so the scan can rank it.
    """
    rep = wai.hack_scan(
        rows,
        reward="reward",
        endorsed=["marker:process"],
        features={"numbers_emitted": lambda r: float(len(numbers_in(r["final_text"])))},
    )
    return {
        "regime": rep["regime"],
        "tau": round(rep["tau"], 4),
        "top_feature": rep["top_feature"],
        "endorsed_on_top": rep["endorsed_on_top"],
        "integrity": rep["integrity"],
        "warnings": list(rep.get("warnings", [])),
    }


def _dynamics(d: dict) -> dict:
    """What the arm's own training log says about how it spent its rollouts.

    ``frac_reward_zero_std`` is the share of GRPO groups where all rollouts
    scored the same, so the group-relative advantage is zero and the step
    learns nothing from them. It is the mechanism a dense reward is supposed
    to fix, and it is measured here rather than argued.
    """
    logs = [x for x in d["train_log"] if "frac_reward_zero_std" in x]
    n = len(logs) or 1
    return {
        "steps_logged": len(logs),
        "frac_reward_zero_std": round(sum(x["frac_reward_zero_std"] for x in logs) / n, 4),
        "clipped_ratio": round(sum(x["completions/clipped_ratio"] for x in logs) / n, 4),
        "reward_first": round(logs[0]["reward"], 4) if logs else None,
        "reward_last": round(logs[-1]["reward"], 4) if logs else None,
        "kl_last": round(logs[-1]["kl"], 5) if logs else None,
        "base_reply_chars": round(_mean(d["base_passes"][0], "n_chars"), 1),
        "trained_reply_chars": round(_mean(d["trained_pass"], "n_chars"), 1),
    }


def analyze(out: Path) -> None:
    arms = {}
    for arm in ("process", "outcome"):
        d = json.loads((out / f"{arm}.json").read_text())
        for rows in d["base_passes"] + [d["trained_pass"]]:
            for r in rows:
                r["markers"] = {"process": r["process"]}
        arms[arm] = d

    base = arms["process"]["base_passes"]
    ev = eval_variance(*base)
    means = [round(v, 4) for v in ev["means"].values()]
    band_applied = applied_band(ev)
    df = ev.get("noise_band_df", len(base) - 1)
    print("NOISE FLOOR  untrained base, three passes")
    print(f"  run means  {means}   stability {ev['stability']}")
    print(f"  run_std    {ev['run_std']:.4f}")
    print(f"  the bar    {band_applied:.4f}  (t(df={df}) x sqrt(2) x run_std)")
    other = [round(_mean(p), 4) for p in arms["outcome"]["base_passes"]]
    print(f"  the same three seeds on the other container: {other}")

    run_std = ev.get("run_std_by_metric") or ev["run_std"]
    report = {
        "paper": "https://arxiv.org/abs/2607.02869",
        "base_model": arms["process"]["base_model"],
        "gpu": arms["process"]["gpu"],
        "n_train": arms["process"]["n_train"],
        "n_holdout": arms["process"]["n_holdout"],
        "k": K,
        "noise_floor": {
            "run_means": means,
            "run_std": ev["run_std"],
            "noise_band_applied": round(band_applied, 4),
            "noise_band_df": ev.get("noise_band_df"),
            "stability": ev["stability"],
            "run_std_by_metric": ev.get("run_std_by_metric"),
            "replication_other_container": other,
        },
        "arms": {},
    }

    for arm, d in arms.items():
        print(f"\n=== {arm}-only reward: untrained base -> trained ===")
        rep = wai.compare(
            d["base_passes"][0],
            d["trained_pass"],
            target="pass_at_1",
            proxy="marker:process",
            run_std=run_std,
            run_std_runs=len(base),
        )
        print(rep)
        report["arms"][arm] = {
            "base_pass_at_1": round(_mean(d["base_passes"][0]), 4),
            "trained_pass_at_1": round(_mean(d["trained_pass"]), 4),
            "base_process": round(_mean(d["base_passes"][0], "process"), 4),
            "trained_process": round(_mean(d["trained_pass"], "process"), 4),
            "headline_verdict": rep["headline_verdict"],
            "metrics": _metrics(rep),
            "warnings": list(rep.get("warnings", [])),
            "over_optimized": rep.get("over_optimized"),
            "training": _dynamics(d),
            "hack_scan": _hack_scan(d["trained_pass"]),
        }

    print("\n=== how the two rewards spent the same rollout budget ===")
    print(f"  {'':18} {'process':>10} {'outcome':>10}")
    for label, key in (
        ("dead groups", "frac_reward_zero_std"),
        ("hit token cap", "clipped_ratio"),
        ("KL at the end", "kl_last"),
        ("reply chars after", "trained_reply_chars"),
    ):
        p_, o_ = report["arms"]["process"]["training"], report["arms"]["outcome"]["training"]
        print(f"  {label:18} {p_[key]:>10.3f} {o_[key]:>10.3f}")

    print("\n=== the paper's claim: outcome-trained -> process-trained ===")
    head = wai.compare(
        arms["outcome"]["trained_pass"],
        arms["process"]["trained_pass"],
        target="pass_at_1",
        proxy="marker:process",
        run_std=run_std,
        run_std_runs=len(base),
    )
    print(head)
    report["process_vs_outcome"] = {
        "outcome_pass_at_1": round(_mean(arms["outcome"]["trained_pass"]), 4),
        "process_pass_at_1": round(_mean(arms["process"]["trained_pass"]), 4),
        "headline_verdict": head["headline_verdict"],
        "metrics": _metrics(head),
        "warnings": list(head.get("warnings", [])),
    }

    (HERE / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nwrote results.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", nargs="?", default="prep", choices=["prep", "analyze"])
    p.add_argument("--dry-run", action="store_true", help="offline: no key, no GPU, no download")
    p.add_argument("--out", default="out", help="where train_modal.py's arm json landed")
    args = p.parse_args()

    if args.dry_run:
        dry_run()
    elif args.stage == "prep":
        prep()
    else:
        analyze(Path(args.out))


if __name__ == "__main__":
    main()
