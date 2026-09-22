"""Fan the seeds out over CPU containers on your own Modal.

The measurement is a text rule over offline rollouts, so it needs no GPU and no
key -- but more seeds make the marker-vs-gold intervals tighter, and each seed
is independent. One container per seed.

    modal run markers_modal.py                 # 12 seeds
    modal run markers_modal.py --seeds 24      # tighter intervals
"""

from __future__ import annotations

import json
import sys

import modal

WHILEAI = "whileai==0.116"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(WHILEAI)
    .add_local_file(__file__.replace("markers_modal.py", "run.py"), "/root/run.py")
)
app = modal.App("wai-marker-trust")


@app.function(image=image, cpu=1, timeout=900)
def one_seed(seed: int, budget: int, phrasings: int) -> dict:
    sys.path.insert(0, "/root")
    import run as recipe

    grid = recipe.task_grid(budget=budget, phrasings=phrasings)
    before = recipe.build(rate=0.50, seed=seed, tasks=grid)
    after = recipe.build(rate=0.15, seed=seed, tasks=grid)
    green = recipe.green_dashboard(before)
    gold = sum(r["gold_clean"] for r in after) / len(after) - sum(
        r["gold_clean"] for r in before
    ) / len(before)
    marker = sum(r["marker_clean"] for r in after) / len(after) - sum(
        r["marker_clean"] for r in before
    ) / len(before)
    return {
        "seed": seed,
        "n_rows": len(before),
        "detection": recipe.detection(before),
        "green_dashboard": green,
        "gold_delta": round(gold, 4),
        "marker_delta": round(marker, 4),
        "recovered": round(marker / gold, 4) if gold else None,
    }


@app.local_entrypoint()
def main(seeds: int = 12, budget: int = 600, phrasings: int = 6, out: str = "seeds.json"):
    from whileai.config import provenance

    print(provenance(), file=sys.stderr)
    args = [(s, budget, phrasings) for s in range(seeds)]
    results = list(one_seed.starmap(args))
    results.sort(key=lambda r: r["seed"])

    green = [
        r["green_dashboard"]["share_of_green_rows_carrying_a_planted_failure"] for r in results
    ]
    recovered = [r["recovered"] for r in results if r["recovered"] is not None]
    print(f"\n{len(results)} seeds, {sum(r['n_rows'] for r in results)} rows")
    print(
        "share of marker-clean rows carrying a planted failure: "
        f"{min(green):.4f} to {max(green):.4f}"
    )
    print(
        "share of the real improvement the marker dashboard recovers: "
        f"{min(recovered):.4f} to {max(recovered):.4f}"
    )
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(f"wrote {out}", file=sys.stderr)
