"""Put this recipe on while.ai/platform/experiments as the agent `team-talk`.

    python post.py --question   # the card: question, hypothesis, method, measure, decide
    python post.py              # plus one run per condition, read from results.json

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
from recipe import BASE_MODEL, SYSTEM, SYSTEM_TEAM

BEHAVIOR = "gsm8k_correct"
# (version, results.json arm, prompt, one line on what changed)
CONDITIONS = [
    ("base", "base", SYSTEM, "no training, plain prompt"),
    ("base-team-prompt", "base_team_prompt", SYSTEM_TEAM, "no training, team prompt"),
    ("grpo", "baseline", SYSTEM, "GRPO 40 steps, plain prompt"),
    ("grpo-team", "recipe", SYSTEM_TEAM, "GRPO 40 steps, team prompt in training and eval"),
    (
        "grpo-team-plain-eval",
        "recipe_plain_prompt",
        SYSTEM,
        "trained with the team prompt, evaluated without it",
    ),
]


def main(question_only: bool) -> None:
    print(provenance(), file=sys.stderr)
    tracked = track(
        "team-talk",
        model=BASE_MODEL,
        harness=Harness(label="plain@qwen2.5-1.5b", instructions=SYSTEM, model=BASE_MODEL),
    )
    tracked.experiment(
        question="Does training a model to reason as a team talking to itself (Solver, Checker, Lead in one reply) beat plain GRPO, and does it keep talking once the team prompt is gone?",
        hypothesis="A Checker turn catches arithmetic slips a monologue keeps, so the team arm ends higher; RL on accuracy alone keeps the talk because the talk pays (Kim et al. 2601.10825).",
        method="Qwen2.5-1.5B-Instruct, TRL GRPO + LoRA, 40 steps, 12 prompts x 4 rollouts, 512-token cap in both arms. One change: the prompt. Reward is the answer only, never the talk. Seeds 17 and 18.",
        measure="Correct answers on 120 held-out GSM8K test problems, 4 samples each, points out of 100 with a 95% interval; paired delta across both seeds. Talk rate = replies with 3+ turns, 2+ speakers, a Checker turn.",
        decide="The team arm wins if its delta over GRPO clears the interval and the re-run noise band. It is in the weights if the plain-prompt eval still beats GRPO and still talks.",
    )
    tracked.behavior(
        BEHAVIOR,
        graded_by="program",
        reward_is_judge=False,
        test_version="gsm8k-test-s0-120",
        n=120,
        description="Final boxed number equals the GSM8K gold, on 120 held-out test problems.",
        rubric="Math-Verify equality against the gold number after '####'. No judge.",
    )
    print("posted the question: https://while.ai/platform/experiments")
    if question_only:
        return

    r = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    arms, checks = r["arms"], r["checks"]
    noise = checks.get("run_std")
    for version, key, prompt, changed in CONDITIONS:
        a = arms.get(key)
        if not a:
            continue
        run = tracked.run(
            version,
            harness=Harness(
                label=f"{version}@qwen2.5-1.5b",
                instructions=prompt,
                model=BASE_MODEL,
            ),
            method="grpo" if a.get("steps") else "eval",
        )
        lo, hi = a["ci"]
        run.score(
            BEHAVIOR,
            a["score"],
            fraction=True,
            ci=(hi - lo) / 2,
            n=r["n_holdout"],
        )
        seeds = a.get("per_seed")
        moved = f"Moved: {100 * a['score']:.0f} points [{100 * lo:.0f}, {100 * hi:.0f}]"
        if seeds:
            moved += f"; per seed {', '.join(f'{100 * s:.0f}' for s in seeds)}"
        run.note(
            "\n".join(
                [
                    f"Changed: {changed}",
                    moved,
                    f"Why: talk rate {100 * a.get('talk_rate', 0):.0f}%, {a.get('turns', 0)} turns, {a.get('chars', 0)} chars per reply",
                    "Learned: see the recipe README",
                    "Reproduce: python recipes/papers/team-talk/recipe.py --train-seeds 17 18",
                ]
            )
        )
        run.finish(steps=a.get("steps") or None, say=False)
    d, p = r.get("delta", {}), r.get("delta_plain_prompt", {})
    print(
        f"posted {len(CONDITIONS)} runs; team vs grpo {100 * d.get('recipe_vs_baseline', 0):+.1f} "
        f"({d.get('verdict')}), plain prompt {100 * p.get('recipe_vs_baseline', 0):+.1f} "
        f"({p.get('verdict')}); noise run_std {noise}"
    )


if __name__ == "__main__":
    main(question_only="--question" in sys.argv[1:])
