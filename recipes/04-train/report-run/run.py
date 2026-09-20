"""One full loop through whileai.platform for one agent.

Registers the agent, declares behaviors, reports five versions (a base and
four trained ones) with curves and scores on every behavior, marks one
served, posts live traffic, prints the verdict. ``--offline`` prints the
calls instead of making them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta

from whileai.config import provenance
from whileai.platform import Behavior, Frontier, Harness, Judge, track

SCORES = {
    "base": {
        "refunds": (62, 3.1),
        "tone": (66, 3.2),
        "handoff": (70, 3.0),
        "policy": (88, 2.1),
        "length": (84, 2.6),
    },
    "v1": {
        "refunds": (68, 3.0),
        "tone": (67, 3.2),
        "handoff": (71, 3.0),
        "policy": (88, 2.1),
        "length": (83, 2.6),
    },
    "v2": {
        "refunds": (74, 2.9),
        "tone": (69, 3.1),
        "handoff": (72, 2.9),
        "policy": (89, 2.0),
        "length": (82, 2.6),
    },
    "v3": {
        "refunds": (78, 2.8),
        "tone": (70, 3.1),
        "handoff": (73, 2.9),
        "policy": (89, 2.0),
        "length": (80, 2.7),
    },
    "v4": {
        "refunds": (83, 2.7),
        "tone": (72, 3.0),
        "handoff": (74, 2.9),
        "policy": (89, 2.0),
        "length": (76, 2.8),
    },
}
BEHAVIORS = {
    "refunds": "Refund email replies, scored by the judge against the policy",
    "tone": "Spec-adherence on tone",
    "handoff": "Hands off to a human when it should",
    "policy": "No policy violations",
    "length": "Reply length stays in band",
}


def printing_transport(method: str, path: str, body=None):
    """The offline transport: print the call, answer like the API."""
    body_id = path.split("/")[2] if path.startswith("/agents/") else "agent"
    shown = json.dumps(body)[:90] if body is not None else ""
    print(f"{method:5} {path} {shown}")
    if path == "/runs":
        return {"id": f"{body['agent']}-{body['version']}", "version": body["version"]}
    if path.endswith("/evals"):
        return {"evals": body}
    if "/dashboard" in path:
        # The same rule the platform applies: the difference interval is
        # delta +- sqrt(ci_a^2 + ci_b^2), and the served version is v3.
        cand, serv = "v4", "v3"
        (c_score, c_ci), (s_score, s_ci) = SCORES[cand]["refunds"], SCORES[serv]["refunds"]
        delta = round(c_score - s_score, 1)
        lower = sum(1 for name in SCORES[cand] if SCORES[cand][name][0] < SCORES[serv][name][0])
        return {
            "agent": {"id": body_id, "name": body_id},
            "behavior": {
                "name": "refunds",
                "n": 240,
                "judge": {"agreement": 0.86, "humanN": 60},
                "noiseFloor": 2.4,
                "rewardIsJudge": False,
                "contamination": 0,
            },
            "verdict": {
                "candidate": cand,
                "serving": serv,
                "delta": delta,
                "excludesZero": abs(delta) > math.sqrt(c_ci**2 + s_ci**2),
                "regressions": lower,
            },
        }
    if not isinstance(body, dict):
        return {"ok": True}
    return {"id": body.get("id", "")}


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="refund-bot")
    ap.add_argument("--offline", action="store_true", help="print the calls, touch nothing")
    args = ap.parse_args()

    agent = track(
        args.agent,
        model="Qwen/Qwen3-4B",
        harness=Harness(
            label="h2",
            instructions="You handle refund emails for a store. Be brief and follow policy.",
            tools=["lookup_order", "issue_refund", "handoff"],
        ),
        frontier=Frontier(name="Sonnet 5", score=81, cost_per_1k=18.0, p50_s=2.1),
        transport=printing_transport if args.offline else None,
    )
    agent.behavior(
        Behavior(
            name="refunds",
            test_version="v2",
            n=240,
            judge=Judge(name="phi-4 vs spec", agreement=0.86, human_n=60, length_bias=0.08),
            noise_floor=2.4,
            contamination=0,
            reward_is_judge=False,
            description=BEHAVIORS["refunds"],
        )
    )
    for name, desc in BEHAVIORS.items():
        if name != "refunds":
            agent.behavior(Behavior(name=name, test_version="v1", n=120, description=desc))

    for version, by_behavior in SCORES.items():
        trained = version != "base"
        with agent.run(
            version,
            id=f"{args.agent}-{version}",
            method="GRPO" if trained else "none",
            targets=["refunds"] if trained else [],
            trained_on=["refunds-grpo", "character-sft"] if trained else [],
            gpu="1xH100" if trained else None,
            flush_every=100,
        ) as run:
            if trained:
                for i in range(31):
                    run.log(
                        i * 10,
                        reward=0.31 + 0.40 * (1 - math.exp(-i / 9)),
                        kl=0.005 + 0.035 * i / 30,
                    )
            for behavior, (score, ci) in by_behavior.items():
                run.score(
                    behavior,
                    score,
                    ci=ci,
                    n=240 if behavior == "refunds" else 120,
                    test_version="v2" if behavior == "refunds" else "v1",
                )
            run.finish(hours=2.1 if trained else None, cost_usd=31 if trained else None)

    agent.promote("v3")
    today = date.today()
    flagged_pct = [6.2, 5.9, 6.4, 6.1, 5.8, 6.0, 6.3, 5.1, 4.8, 4.5, 4.3, 4.2, 4.0, 4.1]
    for i, pct in enumerate(flagged_pct):
        day = today - timedelta(days=13 - i)
        replies = 2400 + (i * 37) % 300
        agent.live(
            day.isoformat(),
            version="v2" if i < 7 else "v3",
            replies=replies,
            flagged=round(replies * pct / 100),
            p50_s=0.7 if i < 7 else 0.6,
            cost_usd=round(replies * 0.0007, 3),
        )

    print(agent.verdict())


if __name__ == "__main__":
    main()
