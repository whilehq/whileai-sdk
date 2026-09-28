"""Collect every measured number into results.json, with intervals, seeds and versions.

    python collect.py

Reads ``out/rounds.json`` (one entry per round), every ``out/scores_*`` file
those rounds name, the seeds' ``train_record.json``, ``out/onnx-*/export.json``,
``out/shortcut_probe.json``, ``out/sdk_measure.json`` and ``out/scores_byte-*.json``.
Nothing here computes a new number.
"""

from __future__ import annotations

import json
import math
import platform
from datetime import date
from pathlib import Path

from data import HELDOUT_CARRIERS, HELDOUT_FAMILIES, read_jsonl

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
HEADLINE = (
    "agentdojo_docs",
    "llmail_inject",
    "multilingual_direct",
    "llm_heldout_domain",
    "hard",
    "paste",
    "deepset",
    "sim_tool",
    "notinject",
    "indirect_heldout_family",
)


def trim(per_slice: dict) -> dict:
    return {s: {k: v for k, v in r.items() if k in KEEP} for s, r in per_slice.items()}


def correctness(rows: list[dict], scores: list[float], thr: float, name: str) -> dict:
    idx = [i for i, r in enumerate(rows) if r["slice"] == name]
    ok = [(scores[i] > thr) == bool(rows[i]["label"]) for i in idx]
    k, n = sum(ok), len(ok)
    z = 1.96
    p = k / n if n else float("nan")
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        if n
        else float("nan")
    )
    return {"points": round(100 * p, 1), "ci95_half": round(100 * half, 1), "n": n}


def load_scores(rd: dict, seed: int | None) -> dict[str, dict]:
    """{test: score json} for one round; seed None is the round's seed-1 files."""
    files = {
        "test": rd["scores"],
        "test_hard": rd["hard_scores"],
        "test_paste": rd["paste_scores"],
        "test_llm": rd.get("llm_scores", ""),
        "test_external": rd.get("external_scores", ""),
    }
    out = {}
    for t, f in files.items():
        if seed is not None and seed != 1 and "seed1" in f:
            f = f.replace("seed1", f"seed{seed}")
        p = OUT / f
        if f and p.exists():
            out[t] = json.loads(p.read_text())
    return out


def main() -> None:
    tests = {
        t: read_jsonl(HERE / f"{t}.jsonl")
        for t in ("test", "test_hard", "test_paste", "test_llm", "test_external")
    }
    rounds = json.loads((OUT / "rounds.json").read_text())
    arms: dict = {}
    for rd in rounds:
        arm: dict = {
            "model": rd["base"],
            "method": rd["method"],
            "trained_on": rd.get("trained_on"),
            "n_train": rd.get("n_train"),
            "seeds": [],
        }
        seeds = [None] if rd["method"] == "eval" else [1, 2, 3]
        for s in seeds:
            sc = load_scores(rd, s)
            if "test" not in sc:
                continue
            entry: dict = {"seed": s, "threshold": sc["test"]["threshold"]}
            for t, j in sc.items():
                entry[t] = trim(j["per_slice"])
                for name in HEADLINE:
                    if any(r["slice"] == name for r in tests[t]):
                        entry.setdefault("points", {})[name] = correctness(
                            tests[t], j["scores"], j["threshold"], name
                        )
            if "probe" in sc["test"]:
                entry["probe"] = sc["test"]["probe"]
            tag = rd.get("tag")
            rec = OUT / f"{tag}-seed{s}" / "train_record.json" if tag and s else None
            if rec is None and tag and s:
                rec = OUT / tag / f"minilm-l6-h384-uncased-seed{s}" / "train_record.json"
            if rec and rec.exists():
                entry["train"] = json.loads(rec.read_text())
            arm["seeds"].append(entry)
        if arm["seeds"]:
            first = arm["seeds"][0]
            arm["headline_points"] = first.get("points", {})
            if len(arm["seeds"]) > 1:
                spread = {}
                for name in HEADLINE:
                    vals = [
                        e["points"][name]["points"]
                        for e in arm["seeds"]
                        if name in e.get("points", {})
                    ]
                    if len(vals) > 1:
                        m = sum(vals) / len(vals)
                        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
                        spread[name] = {
                            "mean": round(m, 1),
                            "sd": round(sd, 2),
                            "n_seeds": len(vals),
                        }
                arm["seed_spread_points"] = spread
        arm["note"] = rd.get("note")
        arms[rd["version"]] = arm
    latency: dict = {}
    for export in sorted(OUT.glob("onnx-*/export.json")):
        e = json.loads(export.read_text())
        latency[export.parent.name] = {
            k: e[k]
            for k in ("int8_mb", "fp32_mb", "latency_single_thread", "cpu", "machine")
            if k in e
        }
        if "sliding_window_128" in e:
            latency[export.parent.name]["sliding_window_128"] = e["sliding_window_128"]
    byte = {}
    for p in sorted(OUT.glob("scores_byte-*.json")):
        b = json.loads(p.read_text())
        byte[b["tag"]] = {
            "threshold": b["threshold"],
            "latency_ms_2000_chars": b["latency_ms_2000_chars"],
            "model_kb": b["model_kb"],
            "test": trim(b["test"]["per_slice"]),
            "test_hard": trim(b["test_hard"]["per_slice"]),
        }
    probe = (
        json.loads((OUT / "shortcut_probe.json").read_text())
        if (OUT / "shortcut_probe.json").exists()
        else None
    )
    sdk = (
        json.loads((OUT / "sdk_measure.json").read_text())
        if (OUT / "sdk_measure.json").exists()
        else None
    )
    base = arms.get("protectai-v2", {}).get("headline_points", {})
    best_name, best = None, None
    for name, arm in arms.items():
        if arm["method"] == "eval" or not arm.get("headline_points"):
            continue
        h = arm["headline_points"]
        wins = sum(
            1
            for k in HEADLINE
            if k in h
            and k in base
            and h[k]["points"] - h[k]["ci95_half"] > base[k]["points"] + base[k]["ci95_half"]
        )
        score = sum(h[k]["points"] for k in HEADLINE if k in h)
        if best is None or score > best:
            best_name, best = name, score
        arm["slices_clearing_baseline"] = wins
    verdict = "unmeasured"
    if best_name:
        h, w = arms[best_name]["headline_points"], arms[best_name]["slices_clearing_baseline"]
        losses = [
            k
            for k in HEADLINE
            if k in h
            and k in base
            and h[k]["points"] + h[k]["ci95_half"] < base[k]["points"] - base[k]["ci95_half"]
        ]
        verdict = f"{best_name}: clears the baseline on {w} of {len(HEADLINE)} headline slices; loses on {losses or 'none'}"
    result = {
        "recipe": "prompt-injection-classifier",
        "date": date.today().isoformat(),
        "claim": "a 22M encoder trained on planted indirect injections with matched twins beats the accessible 184M baseline on indirect and pasted injection at CPU latency; direct attacks are where it still loses",
        "verdict": verdict,
        "verdict_rule": "a slice is cleared when the Wilson 95% intervals of correctness at the run's threshold are apart; three seeds per round, spread reported",
        "tests": {
            t: {"rows": len(rows), "sha256_rows": (HERE / f"{t}.sha256").read_text().strip()}
            for t, rows in tests.items()
        },
        "heldout_families": HELDOUT_FAMILIES,
        "heldout_carriers": HELDOUT_CARRIERS,
        "arms": arms,
        "byte_stage": byte,
        "latency": latency,
        "shortcut_probe": {
            k: {"threshold": v["threshold"], "per_slice": trim(v["per_slice"])}
            for k, v in probe.items()
            if isinstance(v, dict)
        }
        | {"train_file": probe.get("train_file")}
        if probe
        else None,
        "sdk_measure": {
            k: sdk[k]
            for k in ("decontaminate", "eval_variance", "holdout_size", "route")
            if k in sdk
        }
        if sdk
        else None,
        "versions": {"python": platform.python_version(), "platform": platform.platform()},
        "checks": {
            "train_seeds": {k: len(v["seeds"]) for k, v in arms.items()},
            "bootstrap": "percentile, 1000 resamples over rows, seed 0; Wilson for correctness",
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
    print(f"results.json: {verdict}")


if __name__ == "__main__":
    main()
