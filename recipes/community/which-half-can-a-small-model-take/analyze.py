"""Grade the GPU's generations with the same program that graded the incumbent.

    python analyze.py

Three questions, one held-out set:
  1. Did SFT move the small model at all?          base -> trained
  2. Does the trained model match the incumbent?   incumbent -> trained
  3. Is the boring/hard split real?                the same, `by="slice"`
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

from grader import flaws

import whileai as wai
import whileai.simulations as wsim
from whileai.config import provenance

HERE = pathlib.Path(__file__).parent
OUT = HERE / "out"
# L40S list price on Modal, $/hour, for the cost line.
GPU_USD_PER_HOUR = 1.95


def as_rows(records: list[dict], holdout: dict[tuple, dict], model: str) -> list[dict]:
    """Generated text -> graded rows the measurement calls accept."""
    rows = []
    for rec in records:
        key = (rec["scenario_id"], rec["rollout_index"])
        src = holdout[key]
        row = {
            "task_id": rec["scenario_id"],
            "scenario_id": rec["scenario_id"],
            "rollout_index": rec["rollout_index"],
            "slice": rec["slice"],
            "prompt": src["prompt"],
            "steps": src["steps"],
            "final_text": rec["text"],
            "model_version": model,
        }
        f = flaws(row)
        row["reward"] = int(not f)
        row["grader_reason"] = ",".join(f) or "clean"
        rows.append(row)
    return rows


def slice_of(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r["slice"] == name]


def main() -> None:
    print(provenance(), file=sys.stderr)
    payload = json.loads((OUT / "generations.json").read_text())
    holdout_list = json.loads((OUT / "holdout.json").read_text())
    holdout = {(r["scenario_id"], r["rollout_index"]): r for r in holdout_list}

    # The incumbent's own rows, already graded by the same program in prep.
    incumbent = [
        {
            "task_id": r["scenario_id"],
            "scenario_id": r["scenario_id"],
            "rollout_index": r["rollout_index"],
            "slice": r["slice"],
            "prompt": r["prompt"],
            "steps": r["steps"],
            "final_text": r["incumbent_text"],
            "reward": r["incumbent_reward"],
            "model_version": "incumbent",
        }
        for r in holdout_list
    ]

    gens = payload["generations"]
    by_pass: dict[str, list[dict]] = {}
    for g in gens:
        by_pass.setdefault(g["pass"], []).append(g)
    base_runs = [as_rows(by_pass[f"base{i}"], holdout, f"base{i}") for i in range(3)]
    trained = as_rows(by_pass["trained"], holdout, "trained")

    report: dict = {
        "base_model": payload["base_model"],
        "gpu": payload["gpu"],
        "seconds": payload["seconds"],
        "usd": round(payload["seconds"] / 3600 * GPU_USD_PER_HOUR, 2),
        "train_rows": payload["train_rows"],
        "epochs": payload["epochs"],
        "lr": payload["lr"],
        "slices": {},
    }

    print(f"\n{'=' * 64}\nNOISE FLOOR: three passes of identical untrained weights\n{'=' * 64}")
    for name in ("boring", "hard"):
        runs = [slice_of(b, name) for b in base_runs]
        var = wsim.eval_variance(*runs)
        means = var.get("run_means") or var.get("means")
        print(
            f"{name:7s} base passes {means} run_std {var.get('run_std'):.4f} "
            f"band {var.get('noise_band'):.4f}"
        )
        report["slices"].setdefault(name, {})["noise"] = {
            "run_means": means,
            "run_std": var.get("run_std"),
            "noise_band": var.get("noise_band"),
        }

    print(f"\n{'=' * 64}\nQ1  Did SFT move the small model?   base pass 0 -> trained\n{'=' * 64}")
    for name in ("boring", "hard"):
        n = report["slices"][name]["noise"]
        d = wai.compare(
            slice_of(base_runs[0], name),
            slice_of(trained, name),
            run_std=n["run_std"],
            run_std_runs=3,
        )
        print(f"\n--- {name} ---")
        print(d)
        report["slices"][name]["sft_delta"] = summarize(d)

    print(
        f"\n{'=' * 64}\nQ2  Does the trained model match the incumbent?  incumbent -> trained\n{'=' * 64}"
    )
    for name in ("boring", "hard"):
        n = report["slices"][name]["noise"]
        d = wai.compare(
            slice_of(incumbent, name),
            slice_of(trained, name),
            run_std=n["run_std"],
            run_std_runs=3,
        )
        print(f"\n--- {name} ---")
        print(d)
        report["slices"][name]["vs_incumbent"] = summarize(d)

    print(f"\n{'=' * 64}\nPASS RATES\n{'=' * 64}")
    for name in ("boring", "hard"):
        row = {
            "incumbent": mean_reward(slice_of(incumbent, name)),
            "base": statistics.mean(mean_reward(slice_of(b, name)) for b in base_runs),
            "trained": mean_reward(slice_of(trained, name)),
        }
        report["slices"][name]["pass_rate"] = row
        print(
            f"{name:7s} incumbent {row['incumbent']:.3f}  base {row['base']:.3f}  "
            f"trained {row['trained']:.3f}"
        )

    # What the small model gets wrong, by flaw, so the failure has a name.
    for name in ("boring", "hard"):
        counts: dict[str, int] = {}
        for r in slice_of(trained, name):
            if r["reward"] == 0:
                counts[r["grader_reason"]] = counts.get(r["grader_reason"], 0) + 1
        report["slices"][name]["trained_flaws"] = dict(
            sorted(counts.items(), key=lambda kv: -kv[1])
        )
    print(
        "\ntrained model's flaws:",
        json.dumps({k: v["trained_flaws"] for k, v in report["slices"].items()}, indent=1),
    )

    (OUT / "results.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nGPU: {payload['seconds']:.0f}s on {payload['gpu']} = ${report['usd']}")
    print("wrote out/results.json")


def mean_reward(rows: list[dict]) -> float:
    return sum(r["reward"] for r in rows) / max(1, len(rows))


def summarize(d) -> dict:
    m = (d.get("metrics") or {}).get("pass_at_1") or {}
    return {
        "headline_verdict": d.get("headline_verdict"),
        "mean_before": m.get("mean_a"),
        "mean_after": m.get("mean_b"),
        "delta": m.get("delta"),
        "ci95": m.get("ci95"),
        "noise_band": m.get("noise_band"),
        "within_noise": m.get("within_noise"),
        "verdict": m.get("verdict"),
        "n_paired": m.get("n_paired"),
    }


if __name__ == "__main__":
    main()
