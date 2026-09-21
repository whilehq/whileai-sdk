"""Put the two arms on the platform, so the numbers are somewhere a person reads.

Reads results.json (written by `run.py analyze`) and posts one tracked agent
with one behavior and three runs: the untrained base, the outcome-only arm
and the process-only arm. Every score carries its interval and the eval's
own re-run band, because the platform's verdict() refuses a score without
them.

    python report_platform.py             # needs WHILEAI_API_KEY
    python report_platform.py --offline   # prints the calls, touches nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from whileai.config import provenance
from whileai.platform import Behavior, Frontier, Harness, track

HERE = Path(__file__).parent
BEHAVIOR = "gsm8k-final-answer"


def half_width(ci: object) -> float | None:
    """The platform wants the half-width; compare() gives [lo, hi]."""
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return None
    return round((float(ci[1]) - float(ci[0])) / 2 * 100, 2)


def printing_transport(method: str, path: str, body: object) -> dict:
    print(f"{method} {path} {json.dumps(body)[:160]}")
    if not isinstance(body, dict):  # /evals and /live post lists
        return {"ok": True}
    return {"id": body.get("id", "")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default="gsm8k-reward-granularity")
    ap.add_argument("--offline", action="store_true", help="print the calls, touch nothing")
    args = ap.parse_args()

    r = json.loads((HERE / "results.json").read_text())
    floor = r["noise_floor"]
    n = r["n_holdout"]

    agent = track(
        args.agent,
        model=r["base_model"],
        harness=Harness(
            label="gsm8k-cot-v1",
            instructions=(
                "Solve the grade-school math problem. Show your work as short "
                "numbered steps, one per line. End with #### <number>."
            ),
            tools=[],
        ),
        frontier=Frontier(name="paper reported", score=64, cost_per_1k=0.0, p50_s=0.0),
        transport=printing_transport if args.offline else None,
    )
    agent.behavior(
        Behavior(
            name=BEHAVIOR,
            test_version="v1",
            n=n,
            # the band the verdict was judged against. results.json has no
            # "noise_band": it records noise_band_eval_variance (the 1.96
            # form eval_variance returned before #616) beside this one.
            noise_floor=round(floor["noise_band_applied"] * 100, 2),
            contamination=0,
            reward_is_judge=False,
            description=(
                "GSM8K final-answer accuracy on 200 held-out questions, two "
                "sampled rollouts each, decontaminated against the training "
                "prompts. Reproduction of arXiv:2607.02869."
            ),
        )
    )

    base_ci = half_width(r["arms"]["outcome"]["metrics"].get("pass_at_1", {}).get("ci95"))
    runs = [
        ("base", "none", r["arms"]["outcome"]["base_pass_at_1"], base_ci, None),
        (
            "outcome-only",
            "GRPO",
            r["arms"]["outcome"]["trained_pass_at_1"],
            half_width(r["arms"]["outcome"]["metrics"]["pass_at_1"]["ci95"]),
            r["arms"]["outcome"]["headline_verdict"],
        ),
        (
            "process-only",
            "GRPO",
            r["arms"]["process"]["trained_pass_at_1"],
            half_width(r["arms"]["process"]["metrics"]["pass_at_1"]["ci95"]),
            r["arms"]["process"]["headline_verdict"],
        ),
    ]

    for version, method, score, ci, verdict in runs:
        with agent.run(
            version,
            id=f"{args.agent}-{version}",
            method=method,
            targets=[BEHAVIOR] if method != "none" else [],
            gpu=f"1x{r['gpu']}" if method != "none" else None,
        ) as run:
            run.score(
                BEHAVIOR,
                round(score * 100, 2),
                ci=ci,
                n=n,
                test_version="v1",
                notes=verdict or "three passes, the noise floor",
            )
            run.finish()

    # The dashboard scores candidates against the served version, so serve
    # the untrained base: that makes "did training help" the question it
    # answers, which is the question this recipe asked.
    agent.promote("base")

    if args.offline:
        # verdict() reads the dashboard back, which a printing transport
        # cannot invent; the calls above are what --offline is for.
        print("\noffline: posted nothing. Re-run without --offline to read the verdict.")
    else:
        print(agent.verdict())


if __name__ == "__main__":
    print(provenance(), file=sys.stderr)
    main()
