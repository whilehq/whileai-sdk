"""The same safety suite on a real model.

    python live.py                                   # Ollama, llama3.1:8b, no key
    python live.py --agent ollama:qwen3.8:27b --k 4
    python live.py --agent openai:gpt-4.1-mini       # needs OPENAI_API_KEY
    python live.py --json live.json

Everything ``run.py`` does to a scripted agent, done to a model behind an
OpenAI-compatible endpoint: the suite goes in as ``seeds=``, the model
gets the six tools and the policy, ``execute=world`` answers its tool
calls from the same reviews and intent data the scripted agents read (so
the planted reviews reach the model as tool results), every ask is rolled
``k`` times, and the same six-marker judge grades the trajectories.
``fault_rate=0`` so no tool call is broken on purpose; this is a safety
run, not a robustness run.

The output is the per-category table, one example per failure class, and
pass^k over every attack class together. Write it with ``--json`` and the
file is a ``before`` for ``run.compare`` once you have a candidate fix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import by_category, failures, format_categories, grade
from suite import ATTACKS, CATEGORIES, SEEDS, SYSTEM_PROMPT, TOOLS, world

import whileai.simulations as wai
from whileai.config import provenance

DEFAULT_AGENT = "ollama:llama3.1:8b"


GRID_SITUATIONS = 8  # situations the simulator writes on top of the suite


def simulate_live(
    agent: str, *, k: int = 4, seed: int = 0, grid: int = GRID_SITUATIONS, concurrency: int = 2
) -> wai.SimulationData:
    """The suite plus ``grid`` of the simulator's own situations, ``k``
    repeats each. The grid slice is not optional: the engine keeps a few
    slots for its own coverage, and with ``situations=len(SEEDS)`` it
    drops seeds to make room. With eight on top every ask in the suite is
    rolled and the grid fills the rest."""
    n = len(SEEDS) + grid
    return wai.simulate(
        agent=agent,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        seeds=SEEDS,
        execute=world,
        simulator=False,  # the suite and the template grid; no writer model
        mode="rl",
        repeats=k,
        repeat_policy="fixed",
        reproducible=True,
        seed=seed,
        situations=n,
        budget=n * k,
        fault_rate=0,
        concurrency=concurrency,
    )


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--agent", default=DEFAULT_AGENT, help="ollama:<model>, openai:<model>, vllm:<model>@<url>"
    )
    ap.add_argument("--k", type=int, default=4, help="repeats per ask")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--grid",
        type=int,
        default=GRID_SITUATIONS,
        help="simulator situations on top of the suite (keep at least a few, see simulate_live)",
    )
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--json", help="write the graded rows and the tables to this path")
    args = ap.parse_args(argv)

    print(
        f"== suite: {len(ATTACKS)} asks in {len(CATEGORIES)} categories, k={args.k}, agent={args.agent}"
    )
    t0 = time.monotonic()
    data = simulate_live(
        args.agent, k=args.k, seed=args.seed, grid=args.grid, concurrency=args.concurrency
    )
    rows = grade(data, args.agent)
    elapsed = time.monotonic() - t0
    errors = sum(1 for r in rows if r.get("error"))
    print(
        f"   {len(rows)} rows in {elapsed:.0f}s ({data.stopped_because}); {errors} rollouts raised"
    )

    print(f"\n== {args.agent}, by category")
    table = by_category(rows)
    print(format_categories(table))
    attacks = [r for r in rows if r.get("category") not in ("benign", "grid")]
    p = wai.pass_at(attacks, min_k=2)
    pk = f"{p.pass_pow_k:.2f}" if p.pass_pow_k is not None else "n/a"
    print(f"  all attack classes: pass@1 {p.pass_at_1:.2f}, pass^k {pk} over {p.n_groups} asks")
    benign = [r for r in rows if r.get("category") == "benign"]
    if benign:
        ref = wai.refusal_report(benign)
        print(f"  refusal on benign asks: {ref['refusal_rate']:.0%}")
    print("  where it failed, one example per failure class:")
    print("\n".join(failures(rows, limit=8)) or "  (none)")

    if args.json:
        out: dict[str, Any] = {
            "agent": args.agent,
            "k": args.k,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "by_category": table,
            "attacks": p.__dict__,
            "rows": rows,
        }
        Path(args.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
