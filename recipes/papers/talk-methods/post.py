"""Put this recipe on while.ai/platform/experiments as the agent `talk-methods`.

    python post.py --question   # the card: question, hypothesis, method, measure, decide
    python post.py              # plus one run per method, a chart, read from results.json

Needs WHILEAI_API_KEY. The training curves are already on the platform
(recipe.py posts them); this is the Experiments card that reads them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from whileai.config import provenance
from whileai.platform import Harness, track

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recipe import BASE_MODEL, SYSTEM

BEHAVIOR = "split_facts_correct"
# (version, results.json arm, one line on what the method does)
CONDITIONS = [
    ("untrained", "base", "no training; two copies chat, A B A B"),
    ("grpo", "baseline", "every turn paid team reward minus the group mean"),
    ("reinforce-ada", "recipe", "replay a problem until its chats disagree; pool baseline"),
    ("raft", "raft", "train only on chats the team solved outright"),
    ("mixed-partners", "mixed", "GRPO; in half the chats B is the untrained base"),
]
CAPTION = (
    "Team solves it, out of 100, on 300 held-out split-fact problems: with a trained "
    "partner, and with the untrained base as partner. Mean over three training seeds."
)


def figure(r: dict) -> dict:
    names, paired, alone = [], [], []
    for version, key, _ in CONDITIONS:
        a = r["arms"].get(key)
        if not a:
            continue
        names.append(version)
        paired.append(round(100 * a.get("pooled_score", a["score"]), 1))
        cross = r["arms"].get(f"{key}_with_untrained_partner")
        alone.append(round(100 * _mean(cross["per_seed"]), 1) if cross else None)
    data = [{"type": "bar", "name": "Trained partner", "x": names, "y": paired}]
    if any(v is not None for v in alone):
        data.append({"type": "bar", "name": "Untrained partner", "x": names, "y": alone})
    return {
        "data": data,
        "layout": {"barmode": "group", "yaxis": {"title": "points out of 100", "range": [0, 60]}},
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main(question_only: bool) -> None:
    print(provenance(), file=sys.stderr)
    tracked = track(
        "talk-methods",
        model=BASE_MODEL,
        harness=Harness(label="chat@qwen2.5-1.5b", instructions=SYSTEM, model=BASE_MODEL),
    )
    tracked.experiment(
        question="Which training method teaches two model copies to talk best: GRPO, Reinforce-Ada, fine-tuning on winning chats (RAFT), or GRPO with mixed partners?",
        hypothesis="An untrained pair solves 11 in 100, so most GRPO groups teach nothing; Reinforce-Ada recovers them and learns faster. Mixed partners should carry over to an untrained partner, which plain GRPO did not.",
        method="The talk-to-solve task: each copy sees half the facts of a GSM8K problem, they chat A B A B, every turn is paid the team's outcome. Qwen2.5-1.5B-Instruct, TRL GRPO machinery + LoRA, 80 steps, 8 problems x 4 chats, 3 seeds per method. Only the credit changes.",
        measure="Team solves it on 300 held-out problems, points out of 100 with a 95% interval, paired delta against GRPO across three seeds. Also with the untrained base as partner, and the talk evals (sends own facts, uses partner's).",
        decide="A method beats GRPO if its delta clears the interval and the re-run noise band. Mixed partners earns its keep if it keeps GRPO's paired score and beats it with an untrained partner.",
    )
    tracked.behavior(
        BEHAVIOR,
        graded_by="program",
        reward_is_judge=False,
        test_version="gsm8k-split-s0-300",
        n=300,
        description="Both copies box the GSM8K gold, each having seen half the facts.",
        rubric="Math-Verify equality against the gold after ####, per copy. No judge.",
    )
    print("posted the question: https://while.ai/platform/experiments")
    if question_only:
        return

    r = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    runs = {}
    for version, key, what in CONDITIONS:
        a = r["arms"].get(key)
        if not a:
            continue
        run = tracked.run(
            version,
            harness=Harness(label=f"{version}@qwen2.5-1.5b", instructions=SYSTEM, model=BASE_MODEL),
            method="grpo" if a.get("steps") else "eval",
        )
        lo, hi = a["ci"]
        run.score(BEHAVIOR, a["score"], fraction=True, ci=(hi - lo) / 2, n=r["n_holdout"])
        t = a.get("talk") or {}
        seeds = a.get("per_seed")
        moved = f"Moved: {100 * a['score']:.0f} points [{100 * lo:.0f}, {100 * hi:.0f}]"
        if seeds:
            moved += f"; per seed {', '.join(f'{100 * s:.0f}' for s in seeds)}"
        cross = r["arms"].get(f"{key}_with_untrained_partner")
        if cross:
            moved += "; with an untrained partner " + ", ".join(
                f"{100 * s:.0f}" for s in cross["per_seed"]
            )
        d = (r.get("deltas") or {}).get(key)
        if d:
            lo_d, hi_d = d["ci"]
            moved += (
                f"; vs GRPO {100 * d['recipe_vs_baseline']:+.1f} "
                f"[{100 * lo_d:+.1f}, {100 * hi_d:+.1f}] {d['verdict']}"
            )
        run.note(
            "\n".join(
                [
                    f"Changed: {what}",
                    moved,
                    f"Why: sent {100 * t.get('shared', 0):.0f}% of own numbers, answer used "
                    f"{100 * t.get('used', 0):.0f}% of partner's, both agree "
                    f"{100 * t.get('agree', 0):.0f}%",
                    "Learned: see the recipe README",
                    f"Reproduce: python recipes/papers/talk-methods/recipe.py --arm {key}",
                ]
            )
        )
        runs[key] = run
        run.finish(steps=a.get("steps") or None, say=False)
    tracked.figure("methods", figure(r), caption=CAPTION, run=runs.get("recipe"))
    print("posted", ", ".join(runs))


if __name__ == "__main__":
    main(question_only="--question" in sys.argv[1:])
