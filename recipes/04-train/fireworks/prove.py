"""Base against trained on the same held-out tasks, both served by Fireworks.

Needs ``FIREWORKS_API_KEY`` (both arms) and a judge: ``--judge`` names it
(``openai:gpt-4.1-mini`` on ``OPENAI_API_KEY``, ``anthropic:...``), or leave
it out for the judge While hosts on ``WHILEAI_API_KEY``. The judge is never
either arm.

    python prove.py --base accounts/fireworks/models/qwen3-4b \
        --tuned accounts/<ACCOUNT_ID>/models/refunds-dpo --judge openai:gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import os
import sys

from export_fireworks import BASE_MODEL, POLICY, RUBRIC, TOOLS

import whileai as wai

# Four rollouts per task is what the paired comparison needs to tell a
# model change from sampling noise on a few dozen tasks; 32 tasks is the
# floor at which the interval stops spanning zero for a five-point gain.
ROLLOUTS = 4
TASKS = 32


def main() -> None:
    from whileai.config import provenance

    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE_MODEL, help="the base Fireworks model id")
    ap.add_argument("--tuned", default=None, help="the deployed trained model id")
    ap.add_argument("--judge", default=None, help="judge spec; default is the judge While hosts")
    ap.add_argument("--tasks", type=int, default=TASKS)
    args = ap.parse_args()
    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit("prove.py runs both arms on Fireworks: set FIREWORKS_API_KEY first")
    if not args.tuned:
        ap.error("--tuned names the deployed trained model, accounts/<ACCOUNT_ID>/models/<name>")
    judge = wai.Judge(rubric=RUBRIC, model=args.judge) if args.judge else wai.Judge(rubric=RUBRIC)

    before = wai.simulate(
        wai.Fireworks(args.base),
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        mode="rl",
        repeats=ROLLOUTS,
        budget=args.tasks * ROLLOUTS,
    )
    after = wai.simulate(wai.Fireworks(args.tuned), tasks=before)  # same tasks, same k
    before_scored, after_scored = before.grade(judge), after.grade(judge)
    print(wai.compare(before_scored.rows, after_scored.rows))


if __name__ == "__main__":
    main()
