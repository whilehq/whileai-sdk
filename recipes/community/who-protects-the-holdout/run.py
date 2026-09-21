"""Who protects a held-out set from contamination: the ids, or the text?

``decontaminate()`` applies four rules in order, and the first of them --
``same_task`` -- compares ``scenario_id`` / ``task_id`` rather than words. Rows
with no recorded id skip that rule, which the docstring says plainly. What
nobody had measured is how much of the protection that first rule is carrying,
and therefore how much is left when the evaluation set is one you did not write
with ``simulate()`` -- GSM8K, a Hugging Face set, a customer's logged traces.

The answer is most of it. On a ``simulate()``-native holdout the ids catch
98-100% of contaminated rows. Strip the ids and the same call on the same rows
catches 18-30%.

    python run.py                # the regime comparison, three seed pairs
    python run.py --dry-run      # one seed pair, smaller budget
    python run.py --json out.json

Offline: no key, no model, no GPU. Under a minute.

The GPU half of this study -- what the surviving contamination does to a
measured held-out number -- is ``inflation_modal.py`` in this directory.
"""

from __future__ import annotations

import argparse
import json
import sys

import whileai as wai
from whileai.config import provenance

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]

# Train seed vs holdout seed. Different seeds, same generator and same briefs:
# the holdout is a re-run, which is the case the decontaminate docstring names
# as the one word overlap cannot see.
SEED_PAIRS = ((0, 1), (2, 3), (4, 5))


def rows_for(seed: int, budget: int) -> list[dict]:
    agent = wai.seeded_agent(tools=TOOLS, seed=0)
    return wai.simulate(agent=agent, tools=TOOLS, budget=budget, seed=seed, simulator=False).rows()


def without_ids(rows: list[dict]) -> list[dict]:
    """What every evaluation set that simulate() did not write looks like."""
    return [{k: v for k, v in r.items() if k not in ("scenario_id", "task_id")} for r in rows]


def proportion(k: int, n: int) -> list[float]:
    """A proportion with a 95% interval.

    ``pass_at`` groups by task, so rows sharing a task id collapse into one
    group and ``ci95`` comes back ``None`` with nothing said about why. One
    group per row is what makes it a binomial interval over rows.
    """
    rows = [{"reward": 1.0 if i < k else 0.0, "task_id": f"t{i}"} for i in range(n)]
    ci = wai.pass_at(rows, k=1).ci95
    return [round(x, 3) for x in ci]


def measure(train: list[dict], holdout: list[dict]) -> dict:
    n = len(train)
    _, native = wai.decontaminate(train, against=holdout)
    _, idless = wai.decontaminate(train, against=without_ids(holdout))

    def pack(rep: dict) -> dict:
        caught = rep["n_contaminated"]
        return {
            "caught": caught,
            "n_train": n,
            "rate": round(caught / n, 4),
            "ci95": proportion(caught, n),
            "same_task": rep["n_same_task"],
            "exact": rep["n_exact"],
            "near": rep["n_near"],
            "notes": rep["notes"],
        }

    return {"native": pack(native), "id_less": pack(idless)}


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=120, help="rollouts per simulate run")
    ap.add_argument("--dry-run", action="store_true", help="one seed pair, small budget")
    ap.add_argument("--json", metavar="PATH", help="write the numbers here")
    args = ap.parse_args(argv)

    pairs = SEED_PAIRS[:1] if args.dry_run else SEED_PAIRS
    budget = 20 if args.dry_run else args.budget

    results = {"whileai": wai.__version__, "budget": budget, "trials": {}}
    for train_seed, holdout_seed in pairs:
        train = rows_for(train_seed, budget)
        holdout = rows_for(holdout_seed, budget)
        trial = measure(train, holdout)
        trial["train_seed"], trial["holdout_seed"] = train_seed, holdout_seed
        trial["n_holdout"] = len(holdout)
        results["trials"][f"{train_seed}v{holdout_seed}"] = trial

        print(
            f"\ntrain seed {train_seed} vs holdout seed {holdout_seed} "
            f"(train n={trial['native']['n_train']}, holdout n={len(holdout)})"
        )
        for regime in ("native", "id_less"):
            r = trial[regime]
            print(
                f"  {regime:8s} caught {r['caught']:3d}/{r['n_train']}  "
                f"rate {r['rate']:.3f} {r['ci95']}  "
                f"same_task={r['same_task']:3d} near={r['near']}  notes={r['notes']}"
            )

    natives = [t["native"]["rate"] for t in results["trials"].values()]
    idless = [t["id_less"]["rate"] for t in results["trials"].values()]
    print(f"\nnative holdout  : {min(natives):.3f}-{max(natives):.3f} of contaminated rows caught")
    print(f"id-less holdout : {min(idless):.3f}-{max(idless):.3f} of contaminated rows caught")
    print("\nThe rows are identical. Only the evaluation set's ids changed.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
