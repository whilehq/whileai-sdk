"""Paired bootstrap over held-out prompts: which mask_mode trains the policy?

python analyze.py data/results_raw.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

from whileai.config import provenance

METRICS = ("stub", "chars", "tool_call", "right_tool", "right_id", "real_id")
ARMS = ("assistant", "final", "unroll")
B = 2000
ID = re.compile(r"ORD-\d+")

# "Never invent an order id" can only be graded on an ask that contains one.
# 24 of our 120 held-out asks mention no id at all while their gold trace
# still calls with one (it comes from the scenario's world state, not the
# text), so scoring those as "invented" is a defect in the metric, not in
# the model. real_id is therefore reported over the asks that name an id.
CONDITIONAL = {"real_id"}


def arm_series(passes: list[list[dict]], metric: str) -> list[float]:
    """Mean over training seeds, per prompt. Keeps the pairing by prompt."""
    n = len(passes[0])
    return [st.mean(p[i][metric] for p in passes) for i in range(n)]


def paired(a: list[float], b: list[float], seed: int = 0, keep: list[bool] | None = None) -> dict:
    """Bootstrap the paired difference a-b over prompts.

    `keep` restricts the pairing to a subset of prompts (see CONDITIONAL).
    """
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    if keep is not None:
        d = [v for v, k in zip(d, keep) if k]
    n = len(d)
    boots = sorted(st.mean(rng.choices(d, k=n)) for _ in range(B))
    return {
        "delta": round(st.mean(d), 4),
        "lo": round(boots[int(0.025 * B)], 4),
        "hi": round(boots[int(0.975 * B)], 4),
        "n": n,
    }


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", nargs="?", default="data/results_raw.json")
    ap.add_argument("--out", default="data/results.json")
    args = ap.parse_args()

    res = json.loads(Path(args.raw).read_text())
    base_passes = res["base"]
    keep = [bool(ID.findall(g["ask"])) for g in res["golds"]]
    out: dict = {
        "noise_floor": {},
        "arms": {},
        "contrasts": {},
        "train": res["train"],
        "conditional": {
            "metrics": sorted(CONDITIONAL),
            "prompts_scored": sum(keep),
            "of": len(keep),
            "why": "asks that name no order id have no gold to grade 'never invent one' against",
        },
    }

    def mask_for(m: str) -> list[bool] | None:
        return keep if m in CONDITIONAL else None

    def mean_of(series: list[float], m: str) -> float:
        k = mask_for(m)
        vals = [v for v, on in zip(series, k) if on] if k else series
        return round(st.mean(vals), 4)

    for m in METRICS:
        means = [mean_of(arm_series([p], m), m) for p in base_passes]
        out["noise_floor"][m] = {
            "passes": means,
            "band": round(max(means) - min(means), 4),
            "mean": round(st.mean(means), 4),
        }

    base_series = {m: arm_series(base_passes, m) for m in METRICS}
    arm_series_by = {}
    for arm in ARMS:
        keys = sorted(k for k in res["arms"][arm] if k[:1] == "s" and k[1:].isdigit())
        passes = [res["arms"][arm][k] for k in keys]
        arm_series_by[arm] = {m: arm_series(passes, m) for m in METRICS}
        out["arms"][arm] = {
            m: {
                "mean": mean_of(arm_series_by[arm][m], m),
                "per_seed": [mean_of(arm_series([p], m), m) for p in passes],
            }
            for m in METRICS
        }

    def verdict(c: dict, band: float) -> str:
        if c["lo"] <= 0 <= c["hi"]:
            return "flat"
        return "moved" if abs(c["delta"]) > band else "flat (inside noise band)"

    for name, x, y in (
        ("final_minus_assistant", "final", "assistant"),
        ("unroll_minus_assistant", "unroll", "assistant"),
        ("unroll_minus_final", "unroll", "final"),
    ):
        out["contrasts"][name] = {}
        for m in METRICS:
            c = paired(arm_series_by[x][m], arm_series_by[y][m], keep=mask_for(m))
            c["verdict"] = verdict(c, out["noise_floor"][m]["band"])
            out["contrasts"][name][m] = c
    for arm in ARMS:
        out["contrasts"][f"{arm}_minus_base"] = {}
        for m in METRICS:
            c = paired(arm_series_by[arm][m], base_series[m], keep=mask_for(m))
            c["verdict"] = verdict(c, out["noise_floor"][m]["band"])
            out["contrasts"][f"{arm}_minus_base"][m] = c

    Path(args.out).write_text(json.dumps(out, indent=1))

    print("noise floor (3 base passes)")
    for m in METRICS:
        nf = out["noise_floor"][m]
        print(f"  {m:10s} {nf['passes']}  band {nf['band']}")
    print("\narm means (over 2 training seeds)")
    for arm in ARMS:
        row = "  ".join(f"{m}={out['arms'][arm][m]['mean']}" for m in METRICS)
        print(f"  {arm:10s} {row}")
    print("\npaired contrasts, 95% bootstrap over prompts")
    for name, c in out["contrasts"].items():
        print(f"  {name}")
        for m in METRICS:
            v = c[m]
            print(f"    {m:10s} {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]  {v['verdict']}")


if __name__ == "__main__":
    main()
