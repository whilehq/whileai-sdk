"""Generate an RL dataset: k rollouts per prompt, uniform groups.

GRPO scores a rollout against the other rollouts of the same prompt, so the
unit of training is a group, not a row. Two things have to be true:

  1. every prompt gets the same k, so groups are comparable
  2. rewards vary *within* a group, or the advantage is zero and the prompt
     teaches nothing

`mode="rl"` sets the topology. Pass `situations=` explicitly: without it the
generator seeds probes from the spec's `situations` list and can starve before
it reaches the row cap.
"""

import argparse
import os
import sys

import whileai.simulations as wai
from whileai.config import provenance


def main() -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(os.path.dirname(__file__), "spec.json"))
    ap.add_argument("--situations", type=int, default=100, help="distinct prompts (N)")
    ap.add_argument("--k", type=int, default=8, help="rollouts per prompt (group size)")
    ap.add_argument("--fault-rate", type=float, default=0.15)
    ap.add_argument("--out", default="data/rl.jsonl")
    args = ap.parse_args()

    if not os.environ.get("VLLM_API_KEY"):
        print(
            "VLLM_API_KEY is unset. The user writer and the rollout agent both need it.",
            file=sys.stderr,
        )
        return 2

    data = wai.simulate(
        spec=args.spec,
        mode="rl",
        situations=args.situations,
        rollouts_per_request=args.k,
        budget=args.situations * args.k,
        time_budget=None,  # the 60s default silently truncates the run
        fault_rate=args.fault_rate,
        grade=True,
        output=args.out,
    )
    data.save(args.out, meta=True)
    print(
        f"rows={len(data.rows())} prompts={data.unique_prompts} "
        f"k={args.k} rate={data.rows_per_second * 60:.0f}/min "
        f"stopped={data.stopped_because}"
    )
    print(f"wrote {args.out} (+ .meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
