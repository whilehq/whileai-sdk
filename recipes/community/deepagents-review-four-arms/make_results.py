"""Write results.json: every arm on both benchmarks, the with-vs-without comparison, and the noise floor.

    python make_results.py

Offline: arithmetic on rows/. The PR, the blog post and the charts all read this file.
"""

from __future__ import annotations

import json
import pathlib
import sys

import report

import whileai as wai
from whileai.config import provenance

HERE = pathlib.Path(__file__).resolve().parent
ARMS = {
    "srv-base": "base (thinking off)",
    "srv-base-r2": "base (thinking off), run 2",
    "srv-base-r3": "base (thinking off), run 3",
    "srv-harness": "tuned harness, no training",
    "srv-sft-without": "SFT on smithtune's pick",
    "srv-sft-with": "SFT on While's pick",
    "base": "base (thinking on, OpenRouter)",
    "base-r2": "base (thinking on, OpenRouter), run 2",
    "base-r3": "base (thinking on, OpenRouter), run 3",
}
SPLITS = {"holdout": "held-out repos (246 reviews)", "public": "SWE-bench Verified (250 reviews)"}


def cmp(a, b, metric):
    c = wai.simulations.compare_runs(a, b, metric=metric)
    return {
        "delta": round(c["delta"], 4),
        "ci95": [round(x, 4) for x in c["ci95"]],
        "verdict": c["verdict"],
        "n_paired": c["n_paired"],
    }


SEEDS = (42, 43, 44)
# PREREGISTRATION.md: two experiments, two schedules. "registered" is the
# early-stopping schedule as first registered; "fixed" is amendment 1 (2 epochs).
RERUN = {"registered": "", "fixed": "-e2"}


def rerun(split: str) -> dict:
    """The fair rerun: each arm's three seeds pooled, paired by review."""
    out: dict = {}
    for schedule, tag in RERUN.items():
        for exp in ("a", "b"):
            arms = {}
            for side in ("without", "with"):
                names = [f"srv-{exp}-{side}{tag}-s{s}" for s in SEEDS]
                if not all((report.ROWS / f"{n}.{split}.jsonl").exists() for n in names):
                    break
                per = [report.load(n, split) for n in names]
                arms[side] = {
                    "rows": [r for rows in per for r in rows],
                    "seeds": [round(sum(r["reward"] for r in rows) / len(rows), 4) for rows in per],
                    "no_verdict": [sum(r["verdict"] is None for r in rows) for rows in per],
                }
            if len(arms) < 2:
                continue
            a, b = arms["without"], arms["with"]
            out.setdefault(schedule, {})[exp] = {
                side: {
                    "accuracy": round(sum(r["reward"] for r in v["rows"]) / len(v["rows"]), 4),
                    "calls": round(sum(r["model_calls"] for r in v["rows"]) / len(v["rows"]), 2),
                    "seed_accuracy": v["seeds"],
                    "seed_no_verdict": v["no_verdict"],
                }
                for side, v in arms.items()
            } | {
                "with_vs_without": {
                    "accuracy": cmp(a["rows"], b["rows"], "pass_at_1"),
                    "calls": cmp(a["rows"], b["rows"], "marker:model_calls"),
                },
                # Of the nine seed pairs, how many have the with-While seed ahead.
                "seed_pairs_with_ahead": sum(x > y for x in b["seeds"] for y in a["seeds"]),
            }
    return out


def main() -> int:
    print(provenance(), file=sys.stderr)
    out = {"splits": SPLITS, "arms": {}, "with_vs_without": {}, "vs_base": {}, "noise": {}}
    for split in SPLITS:
        out["arms"][split] = {}
        for arm, label in ARMS.items():
            path = report.ROWS / f"{arm}.{split}.jsonl"
            if not path.exists():
                continue
            rows = report.load(arm, split)
            out["arms"][split][arm] = {
                "label": label,
                "n": len(rows),
                "accuracy": round(sum(r["reward"] for r in rows) / len(rows), 4),
                "calls": round(sum(r["model_calls"] for r in rows) / len(rows), 2),
                "no_verdict": sum(r["verdict"] is None for r in rows),
            }
        load = lambda a, s=split: report.load(a, s)  # noqa: E731
        out["with_vs_without"][split] = {
            "accuracy": cmp(load("srv-sft-without"), load("srv-sft-with"), "pass_at_1"),
            "calls": cmp(load("srv-sft-without"), load("srv-sft-with"), "marker:model_calls"),
        }
        out["vs_base"][split] = {
            a: {
                "accuracy": cmp(load("srv-base"), load(a), "pass_at_1"),
                "calls": cmp(load("srv-base"), load(a), "marker:model_calls"),
            }
            for a in ("srv-harness", "srv-sft-without", "srv-sft-with")
        }
        passes = [load(a) for a in ("srv-base", "srv-base-r2", "srv-base-r3")]
        ev = wai.eval_variance(*passes)
        out["noise"][split] = {"run_std": ev["run_std"], "noise_band": ev["noise_band"]}
    out["rerun"] = {split: rerun(split) for split in SPLITS}
    out["rerun_selection"] = json.loads((HERE / "keep" / "summary.json").read_text())
    tok = {
        a: json.loads((HERE / ".cache" / "st" / a / "summary.json").read_text())
        for a in ("sft-with", "sft-without")
    }
    out["training"] = {
        "sft-with": {
            "traces": 173,
            "tokens_per_epoch": 3619086,
            "best_epoch": 2,
            **tok["sft-with"],
        },
        "sft-without": {
            "traces": 258,
            "tokens_per_epoch": 10221451,
            "best_epoch": 2,
            **tok["sft-without"],
        },
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("with_vs_without", "noise")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
