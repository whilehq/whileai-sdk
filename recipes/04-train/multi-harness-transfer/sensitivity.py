"""Declared after the primary result (amendment 4): per-harness gaps, and the unseen pool without Gemini CLI."""

import json
from pathlib import Path

from whileai.simulations.score.stats import compare_runs

from analyze import TRAINED, UNSEEN, graded, load

HERE = Path(__file__).parent
data = load(HERE / "out")


def fmt(c):
    lo, hi = c["ci95"]
    return {"delta": round(c["delta"] * 100, 1), "ci95": [round(lo * 100, 1), round(hi * 100, 1)], "verdict": c["verdict"], "oc": round(c["mean_a"] * 100, 1), "mh": round(c["mean_b"] * 100, 1)}


out = {"per_harness": {}, "unseen_without_gemini": None}
for h in TRAINED + UNSEEN:
    out["per_harness"][h] = fmt(compare_runs(graded(data["oc-rl"], [h]), graded(data["mh-rl"], [h])))
five = [h for h in UNSEEN if h != "gemini-cli"]
out["unseen_without_gemini"] = fmt(compare_runs(graded(data["oc-rl"], five), graded(data["mh-rl"], five)))
for m in ("oc-rl", "mh-rl"):
    out[f"{m}_minus_base_unseen_without_gemini"] = fmt(compare_runs(graded(data["base"], five), graded(data[m], five)))
(HERE / "sensitivity.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=1))
