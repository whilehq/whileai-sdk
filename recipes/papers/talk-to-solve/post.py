"""Put this recipe on while.ai/platform/experiments as the agent `talk-to-solve`.

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
from recipe import BASE_MODEL, SYSTEM

BEHAVIOR = "split_facts_correct"
# (version, results.json arm, one line on what changed)
CONDITIONS = [
    ("base-no-channel", "base_channel_off", "no training; messages never delivered"),
    ("base-chat", "base", "no training; two copies chat, A B A B"),
    ("trained-no-channel", "baseline", "GRPO 80 steps on every turn; messages never delivered"),
    ("trained-chat", "recipe", "GRPO 80 steps on every turn; messages delivered"),
    (
        "trained-a-untrained-b",
        "recipe_with_untrained_partner",
        "trained A chats with the untrained base as B",
    ),
]


# The talk evals, one bar group each, in points out of 100.
MEASURES = [
    ("solved", "Team solves it"),
    ("shared", "Sends facts only it has"),
    ("used", "Answer uses partner's facts"),
    ("agree", "Both copies agree"),
    ("asked", "First message asks"),
]
BARS = [
    ("base", "Before training"),
    ("recipe", "Trained, channel on"),
    ("baseline", "Trained, channel off"),
    ("recipe_with_untrained_partner", "Trained A + untrained B"),
]
CAPTION = (
    "Talk evals on 300 held-out split-fact problems, first training seed. "
    "Trained copies send what only they know and answer from what their partner sent."
)


def talk_figure(r: dict) -> dict:
    data = []
    for key, name in BARS:
        a = r["arms"].get(key)
        if not a:
            continue
        t = a.get("talk") or {}
        vals = [100 * a["score"]] + [100 * t.get(m, 0) for m, _ in MEASURES[1:]]
        data.append(
            {
                "type": "bar",
                "name": name,
                "x": [label for _, label in MEASURES],
                "y": [round(v, 1) for v in vals],
            }
        )
    return {
        "data": data,
        "layout": {"barmode": "group", "yaxis": {"title": "points out of 100", "range": [0, 100]}},
    }


def example(c: dict) -> str:
    """One chat, trimmed, as the note's last block."""
    task = c["task"]
    out = [
        "",
        f"Example (gold {task['gold']}). A sees: {' '.join(task['facts_a'])} "
        f"B sees: {' '.join(task['facts_b'])} Asked: {task['ask']}",
    ]
    for who, text in c["chat"]:
        flat = " ".join(text.split())
        out.append(f"- {who}: {flat[:350]}{'...' if len(flat) > 350 else ''}")
    return "\n".join(out)


def main(question_only: bool) -> None:
    print(provenance(), file=sys.stderr)
    tracked = track(
        "talk-to-solve",
        model=BASE_MODEL,
        harness=Harness(label="chat@qwen2.5-1.5b", instructions=SYSTEM, model=BASE_MODEL),
    )
    tracked.experiment(
        question="Can RL teach two copies of a model to talk to each other: share what only they know, ask for what they are missing, and solve a problem neither can solve alone?",
        hypothesis="Each copy sees half the facts of a GSM8K problem, so the answer needs the other's facts. Paying every message the team's outcome teaches the copies to send their numbers and use their partner's; the same training with the channel cut cannot.",
        method="Qwen2.5-1.5B-Instruct, TRL GRPO + LoRA, 80 steps, 8 problems x 4 chats. A and B take turns A B A B; each copy's last message boxes an answer. Every message is paid the mean of both answers. One change: messages delivered, or replaced by '(no message)'. Seeds 17, 18, 19.",
        measure="Both copies' answers on 300 held-out GSM8K test problems split the same way, points out of 100 with a 95% interval. Also: the share of its private numbers a copy sends, the share of its partner's numbers its answer uses, first messages that ask, message length. And a trained A with an untrained B.",
        decide="Talking was learned if the chat arm beats the no-channel arm past the noise, sends and uses more of the private numbers than the untrained chat, and keeps some of it with an untrained partner.",
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
    for version, key, changed in CONDITIONS:
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
        lines = [
            f"Changed: {changed}",
            moved,
            f"Why: sent {100 * t.get('shared', 0):.0f}% of own numbers, answer used "
            f"{100 * t.get('used', 0):.0f}% of partner's, asked {100 * t.get('asked', 0):.0f}%, "
            f"both copies agree {100 * t.get('agree', 0):.0f}%, "
            f"{t.get('msg_chars', 0)} chars a message",
            "Learned: see the recipe README",
            "Reproduce: python recipes/papers/talk-to-solve/recipe.py",
        ]
        if key in ("base", "recipe"):
            lines.append(example(r["samples"][key][0]))
        run.note("\n".join(lines))
        runs[key] = run
        run.finish(steps=a.get("steps") or None, say=False)
    tracked.figure("talk-before-after", talk_figure(r), caption=CAPTION, run=runs.get("recipe"))
    d = r.get("delta", {})
    print(
        f"posted; chat vs no channel {100 * d.get('recipe_vs_baseline', 0):+.1f} ({d.get('verdict')})"
    )


if __name__ == "__main__":
    main(question_only="--question" in sys.argv[1:])
