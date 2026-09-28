"""Collect every measured number into results.json, with intervals, seeds and versions.

    python collect.py

Reads ``out/scores_*.json``, ``out/*/train_record.json``, ``out/onnx-seed1/export.json``
and ``out/route.json``. Nothing here computes a new number.
"""

from __future__ import annotations

import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
from data import HELDOUT_CARRIERS, HELDOUT_FAMILIES, read_jsonl
from metrics import by_slice

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
KEEP = (
    "n",
    "n_pos",
    "n_neg",
    "threshold",
    "auroc",
    "auroc_ci95",
    "recall_at_1pct_fpr",
    "recall_at_1pct_fpr_ci95",
    "precision",
    "recall",
    "recall_ci95",
    "f1",
    "fpr",
    "fpr_ci95",
    "accuracy",
)


def trim(per_slice: dict) -> dict:
    return {s: {k: v for k, v in r.items() if k in KEEP} for s, r in per_slice.items()}


def main() -> None:
    rows = read_jsonl(HERE / "test.jsonl")
    arms: dict = {}
    base = json.loads((OUT / "scores_protectai.json").read_text())
    arms["protectai-v2-184M"] = {
        "model": base["model"],
        "threshold": base["threshold"],
        "threshold_rule": "shipped 0.5",
        **trim(base["per_slice"]),
    }
    seeds = sorted(OUT.glob("scores_minilm_seed*.json"))
    per_seed = []
    for p in seeds:
        r = json.loads(p.read_text())
        seed = int(p.stem.rsplit("seed", 1)[1])
        rec = json.loads(
            (OUT / f"minilm-l6-h384-uncased-seed{seed}" / "train_record.json").read_text()
        )
        per_seed.append(
            {"seed": seed, "threshold": r["threshold"], "train": rec, **trim(r["per_slice"])}
        )
    if per_seed:
        first = per_seed[0]
        arms["minilm-l6-22M"] = {
            "model": first["train"]["base_model"],
            "threshold": first["threshold"],
            "threshold_rule": "1% FPR on the in-distribution benign carriers",
            **{k: first[k] for k in first if k not in ("seed", "threshold", "train")},
            "seeds": per_seed,
        }
        # across seeds: mean and sd of the two headline numbers, the noise floor
        for key in ("indirect_heldout", "notinject", "deepset"):
            vals = [s[key].get("auroc", s[key].get("fpr")) for s in per_seed]
            arms["minilm-l6-22M"].setdefault("seed_spread", {})[key] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n_seeds": len(vals),
            }
    export = OUT / "onnx-seed1" / "export.json"
    latency: dict = {}
    if export.exists():
        e = json.loads(export.read_text())
        latency = {
            "int8_single_thread": e["latency_single_thread"],
            "int8_mb": e["int8_mb"],
            "fp32_mb": e["fp32_mb"],
            "cpu": e["cpu"],
            "machine": e["machine"],
        }
        if "test_scores" in e:
            thr = arms["minilm-l6-22M"]["threshold"] if "minilm-l6-22M" in arms else 0.5
            arms["minilm-l6-22M-int8-onnx"] = {
                "model": e["onnx_int8"],
                "threshold": thr,
                "threshold_rule": "seed 1's torch threshold",
                **trim(by_slice(rows, e["test_scores"], thr)),
            }
    route = json.loads((OUT / "route.json").read_text()) if (OUT / "route.json").exists() else None
    a, b = arms.get("minilm-l6-22M"), arms["protectai-v2-184M"]
    verdict = "unmeasured"
    if a:
        lo_a, hi_b = a["indirect_heldout"]["auroc_ci95"][0], b["indirect_heldout"]["auroc_ci95"][1]
        verdict = "moved" if lo_a > hi_b else "unresolved"
    result = {
        "recipe": "prompt-injection-classifier",
        "date": date.today().isoformat(),
        "claim": "a 22M encoder trained on planted indirect injections beats the accessible 184M baseline on held-out indirect injection at lower latency",
        "verdict": verdict,
        "verdict_rule": "moved when the 95% row-bootstrap interval of the recipe's indirect held-out AUROC lies above the baseline's; seeds reported apart",
        "test": {
            "rows": len(rows),
            "sha256_rows": (HERE / "test.sha256").read_text().strip(),
            "heldout_families": HELDOUT_FAMILIES,
            "heldout_carriers": HELDOUT_CARRIERS,
        },
        "arms": arms,
        "latency": latency,
        "route": route,
        "versions": {"python": platform.python_version(), "platform": platform.platform()},
        "checks": {
            "train_seeds": {"minilm-l6-22M": len(per_seed), "protectai-v2-184M": 0},
            "bootstrap": "percentile, 1000 resamples over rows, seed 0",
        },
    }
    try:
        import onnxruntime
        import torch
        import transformers

        result["versions"].update(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "onnxruntime": onnxruntime.__version__,
            }
        )
    except ImportError:
        pass
    (HERE / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"results.json: verdict {verdict}; arms {list(arms)}")


if __name__ == "__main__":
    main()
