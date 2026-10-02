"""Phase 0 analysis, exactly as pre-registered. Reads out/<model>/cells/*.json.

    python analyze.py            # prints the tables, writes results.json
"""

import json
import sys
from pathlib import Path

import whileai as wai
from whileai.simulations.score.stats import compare_runs

HERE = Path(__file__).parent
TRAINED = ["opencode", "claude-code", "codex", "mini-swe-agent"]
UNSEEN = ["pi", "gemini-cli", "qwen-coder", "vibe", "openhands-sdk", "terminus-2"]
MODELS = ["base", "oc-rl", "mh-rl", "oc-sft", "mh-sft"]


def load(root: Path) -> dict[str, list[dict]]:
    """One row per graded cell, in whileai's row shape."""
    out: dict[str, list[dict]] = {}
    for model in MODELS:
        rows = []
        for f in sorted((root / model / "cells").glob("*.json")):
            c = json.loads(f.read_text())
            rows.append(
                {
                    "task_id": c["task"],
                    "reward": c.get("correctness"),
                    "harness": {"label": c["harness"], "model": model},
                    "difficulty": c["difficulty"],
                    "tool_calls": c.get("tool_calls"),
                    "error": c.get("error"),
                }
            )
        if rows:
            out[model] = rows
    return out


def graded(rows, harnesses):
    return [r for r in rows if r["harness"]["label"] in harnesses and r["reward"] in (0, 1)]


def main(root: Path = HERE / "out") -> dict:
    data = load(root)
    if not data:
        sys.exit(f"no cells under {root}")
    report: dict = {"coverage": {}, "pass": {}, "vs_base": {}, "h1": None}

    # Coverage and per-harness pass rate with a task-resampled interval.
    for model, rows in data.items():
        report["coverage"][model] = {
            h: f"{len(graded(rows, [h]))}/{sum(r['harness']['label'] == h for r in rows)}"
            for h in TRAINED + UNSEEN
        }
        report["pass"][model] = {}
        for h in TRAINED + UNSEEN:
            g = graded(rows, [h])
            if g:
                p = wai.pass_at(g, min_k=1)
                report["pass"][model][h] = {"pass": p.pass_at_1, "ci95": p.ci95}

    # H1: mh-rl minus oc-rl on the six unseen harnesses pooled, paired by task.
    # Rows pooled per task average the 6 harnesses x 3 samples, so each harness weighs the same.
    if "mh-rl" in data and "oc-rl" in data:
        report["h1"] = compare_runs(graded(data["oc-rl"], UNSEEN), graded(data["mh-rl"], UNSEEN))
        report["h1_trained"] = compare_runs(
            graded(data["oc-rl"], TRAINED), graded(data["mh-rl"], TRAINED)
        )

    # Each model minus base, pooled over unseen and over trained harnesses.
    if "base" in data:
        for model in data:
            if model != "base":
                report["vs_base"][model] = {
                    "unseen": compare_runs(graded(data["base"], UNSEEN), graded(data[model], UNSEEN)),
                    "trained": compare_runs(graded(data["base"], TRAINED), graded(data[model], TRAINED)),
                }

    # Harness vs model share of the spread, over every complete cell.
    try:
        rows = [r for m in data.values() for r in graded(m, TRAINED + UNSEEN)]
        report["attribution"] = str(wai.harness.attribute(rows))
    except ValueError as exc:
        report["attribution"] = f"not computed: {exc}"

    (HERE / "results.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out")
