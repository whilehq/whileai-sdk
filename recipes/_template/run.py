"""<One line: what this recipe produces.>

<Two or three lines on the one idea it teaches. Shorter than the code.>

Run: python recipes/<step>/<name>/run.py
"""

from __future__ import annotations

import argparse
import sys

import whileai.simulations as wai
from whileai.config import provenance

# Rows as they come off a rollout: `final_text` is what the policy said, the
# gold lives in `privileged` (the training export never projects it), and
# `scenario_id` + `rollout_index` are what group repeats of one prompt.
ROWS = [
    {
        "prompt": "2+2?",
        "final_text": r"\boxed{4}",
        "privileged": {"reference": "4"},
        "scenario_id": "m1",
        "rollout_index": 0,
    },
    {
        "prompt": "2+2?",
        "final_text": "five",
        "privileged": {"reference": "4"},
        "scenario_id": "m1",
        "rollout_index": 1,
    },
]


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Every recipe takes one of these, and it is what CI runs: a path through
    # the whole script that needs no key, no network and no GPU.
    p.add_argument("--limit", type=int, default=len(ROWS), help="rows to grade")
    p.add_argument("--dry-run", action="store_true", help="no model calls, no key")
    args = p.parse_args(argv)

    rows = ROWS[: args.limit]

    # Grade. A verifier is a checker, not a judge: no model call, so this half
    # runs offline. Swap in `wai.rubric_judge(...)` when the reward needs one,
    # and keep it behind `--dry-run`.
    scored = wai.run_judge(rows, wai.verify.MathEqual(), source="grade")

    # Report a paired number with its interval, never a mean alone.
    print(wai.pass_at(scored.rows))

    # End on the next command the reader can run themselves.
    if not args.dry_run:
        print("\nNext: wai.push_rows(scored.rows, '<name>-v1', gate=True, mode='rl')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
