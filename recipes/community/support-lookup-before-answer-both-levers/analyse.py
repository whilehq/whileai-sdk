"""Read the four cells off the volume and say which lever moved the behaviour.

Paired on one frozen holdout, per-metric noise floor from three base
re-runs, the capability twin as a must-not-regress guard, and the 2x2
attribution the method is actually about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import whileai as wai
from whileai.config import provenance

OUT = Path("out")

# What the Modal image pinned; read off the training container's own report.
PINS = {"vllm": "0.29.0", "trl": "1.13.0", "peft": "0.21.0"}


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    print(provenance(), file=sys.stderr)
    search = load("search.json")
    searched = search["searched_harness"]
    neither = [load(f"cell_neither_run{i}.json") for i in range(3)]
    harness_cell = load("cell_harness.json")
    weights = load("cell_weights.json")
    both = load("cell_both.json")

    print("=" * 78)
    print(f"searched harness: {searched}")
    for label, c in search["candidates"].items():
        g = c.get("gate")
        gs = f"  gate {g['delta']:+.3f} {g['ci95']} {g['verdict']}" if g else ""
        print(
            f"  {label:22s} right_first_action {c['right_first_action']:.3f}"
            f"  looks_up {c['looks_up']:.3f}  stays_in_scope {c['stays_in_scope']:.3f}"
            f"  called_a_tool {c['called_a_tool']:.3f}{gs}"
        )

    # ---- noise floor: three re-runs of the base under the deployed harness.
    # One scalar applied to every metric prints a band against whatever scale
    # happens to be there, so each metric gets its own floor.
    print("\n" + "=" * 78)
    print("noise floor: the base under the deployed harness, three times")
    ev = wai.eval_variance(*neither, metric="marker:right_first_action")
    run_std = dict(ev["run_std_by_metric"])
    print(ev)
    print("per-metric run_std:", {k: round(v, 4) for k, v in run_std.items()})
    variance = {
        "means": dict(ev["means"]),
        "run_std_by_metric": run_std,
        "noise_band": ev["noise_band"],
        "stability": ev["stability"],
        "tasks_in_every_run": ev["tasks_in_every_run"],
    }

    base = neither[0]
    results = {
        "searched_harness": searched,
        "candidates": search["candidates"],
        "run_std": dict(run_std),
        "eval_variance": variance,
        "cells": {},
        "deltas": {},
    }

    def cell(name, rows):
        results["cells"][name] = {
            "n": len(rows),
            "right_first_action": sum(r["markers"]["right_first_action"] for r in rows) / len(rows),
            "looks_up_when_it_should": _m(rows, "looks_up_when_it_should"),
            "stays_in_scope": _m(rows, "stays_in_scope"),
            "called_a_tool": _m(rows, "called_a_tool"),
            "args_match": _m(rows, "args_match"),
        }
        print(
            f"  {name:8s} n={len(rows):4d} "
            f"right_first_action {results['cells'][name]['right_first_action']:.3f}  "
            f"looks_up {results['cells'][name]['looks_up_when_it_should']:.3f}  "
            f"stays_in_scope {results['cells'][name]['stays_in_scope']:.3f}  "
            f"called_a_tool {results['cells'][name]['called_a_tool']:.3f}"
        )

    print("\n" + "=" * 78)
    print("the four cells on the same frozen holdout")
    for nm, rows in [
        ("neither", base),
        ("harness", harness_cell),
        ("weights", weights),
        ("both", both),
    ]:
        cell(nm, rows)

    # ---- the comparisons that matter
    print("\n" + "=" * 78)
    pairs = [
        ("harness vs neither", base, harness_cell),
        ("weights vs neither", base, weights),
        ("both vs neither", base, both),
        ("both vs weights  (method vs baseline)", weights, both),
        ("both vs harness  (did the weights add anything)", harness_cell, both),
    ]
    for name, before, after in pairs:
        rep = wai.compare(
            before,
            after,
            target="marker:right_first_action",
            must_not_regress=["stays_in_scope"],
            by="target",
            run_std=run_std,
            run_std_runs=3,
        )
        m = rep["metrics"]["marker:right_first_action"]
        print(f"\n--- {name}")
        print(rep)
        results["deltas"][name] = {
            "delta": m["delta"],
            "ci95": list(m["ci95"]),
            "verdict": m["verdict"],
            "headline": rep["headline_verdict"],
            "groups": {
                g: {"delta": v.get("delta"), "ci95": list(v.get("ci95", []))}
                for g, v in (rep.get("groups") or {}).items()
            },
        }

    # ---- which lever: the 2x2 the method is about
    print("\n" + "=" * 78)
    print("attribution: harness x model over the same holdout tasks")
    grid = base + harness_cell + weights + both
    att = wai.harness.attribute(grid, metric="marker:right_first_action")
    print(att)
    results["attribution"] = json.loads(json.dumps(att, default=str))

    # ---- what is the reward paying for
    print("\n" + "=" * 78)
    for nm, rows in [("weights", weights), ("both", both)]:
        try:
            hs = wai.hack_scan(rows)
            print(f"hack scan {nm}: {hs}")
            results.setdefault("hack_scan", {})[nm] = json.loads(json.dumps(hs, default=str))
        except Exception as e:
            print(f"hack scan {nm}: not run ({type(e).__name__}: {e})")

    # ---- the recipe's results.json, in the papers template's shape
    split_meta = (
        json.loads(Path("split.json").read_text())["decontam"]
        if Path("split.json").exists()
        else {}
    )
    results["decontamination"] = split_meta
    results["n_train"] = split_meta.get("kept")
    results["pins"] = PINS
    d = results["deltas"]
    out = {
        "recipe": "support-lookup-before-answer-both-levers",
        "title": "Both levers on a support agent's lookup behaviour",
        "method_recipe": "recipes/papers/harness-and-weights",
        "paper": "https://arxiv.org/abs/2607.03935",
        "book": "Evaluation",
        "base_model": "Qwen/Qwen3-4B",
        "metric": "marker:right_first_action",
        "n_holdout": results["cells"]["neither"]["n"],
        "k": 2,
        "gpu": "L40S",
        "whileai": wai.__version__,
        "searched_harness": searched,
        "gate_passed": False,
        "arms": {
            "base": {
                "score": results["cells"]["neither"]["right_first_action"],
                "stays_in_scope": results["cells"]["neither"]["stays_in_scope"],
                "steps": 0,
            },
            "harness": {
                "score": results["cells"]["harness"]["right_first_action"],
                "stays_in_scope": results["cells"]["harness"]["stays_in_scope"],
                "steps": 0,
            },
            "baseline": {
                "score": results["cells"]["weights"]["right_first_action"],
                "stays_in_scope": results["cells"]["weights"]["stays_in_scope"],
            },
            "recipe": {
                "score": results["cells"]["both"]["right_first_action"],
                "stays_in_scope": results["cells"]["both"]["stays_in_scope"],
            },
        },
        "delta": {
            "recipe_vs_baseline": d["both vs weights  (method vs baseline)"]["delta"],
            "ci": d["both vs weights  (method vs baseline)"]["ci95"],
            "verdict": "unresolved",
            "why_unresolved": "one training seed per arm",
        },
        "checks": {
            "run_std": results["run_std"],
            "run_std_runs": 3,
            "noise_band": variance["noise_band"],
            "stability": variance["stability"],
            "decontaminate": split_meta,
            "reward_is_judge": False,
        },
        "full": results,
    }
    Path("results.json").write_text(json.dumps(out, indent=2, default=str))
    Path("results_raw.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nwrote results.json and results_raw.json")


def _m(rows, marker):
    v = [r["markers"][marker] for r in rows if marker in r["markers"]]
    return sum(v) / len(v) if v else float("nan")


if __name__ == "__main__":
    main()
