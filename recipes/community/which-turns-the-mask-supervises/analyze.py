"""Paired bootstrap over held-out prompts: does honouring the mask still lose?

python analyze.py data/results_raw.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

from whileai.config import provenance

METRICS = ("stub", "chars", "tool_call")
ARMS = ("assistant", "final", "unroll")
B = 2000


def arm_series(passes: list[list[dict]], metric: str) -> list[float]:
    """Mean over training seeds, per prompt. Keeps the pairing by prompt."""
    n = len(passes[0])
    return [st.mean(p[i][metric] for p in passes) for i in range(n)]


def paired(a: list[float], b: list[float], seed: int = 0) -> dict:
    """Bootstrap the paired difference a-b over prompts."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    boots = sorted(st.mean(rng.choices(d, k=n)) for _ in range(B))
    return {
        "delta": round(st.mean(d), 4),
        "lo": round(boots[int(0.025 * B)], 4),
        "hi": round(boots[int(0.975 * B)], 4),
    }


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", nargs="?", default="data/results_raw.json")
    ap.add_argument("--out", default="data/results.json")
    args = ap.parse_args()

    res = json.loads(Path(args.raw).read_text())
    base_passes = res["base"]
    out: dict = {"noise_floor": {}, "arms": {}, "contrasts": {}, "train": res["train"]}

    for m in METRICS:
        means = [round(st.mean(r[m] for r in p), 4) for p in base_passes]
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
                "mean": round(st.mean(arm_series_by[arm][m]), 4),
                "per_seed": [round(st.mean(r[m] for r in p), 4) for p in passes],
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
            c = paired(arm_series_by[x][m], arm_series_by[y][m])
            c["verdict"] = verdict(c, out["noise_floor"][m]["band"])
            out["contrasts"][name][m] = c
    for arm in ARMS:
        out["contrasts"][f"{arm}_minus_base"] = {}
        for m in METRICS:
            c = paired(arm_series_by[arm][m], base_series[m])
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
