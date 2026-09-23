"""Put this recipe on while.ai/platform/experiments as the agent `message-board`.

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
from recipe import BASE_MODEL, READ_SYSTEM

BEHAVIOR = "gsm8k_correct"
# (version, results.json arm, one line on what changed)
CONDITIONS = [
    ("base-own-note", "base", "no training; each reader sees only its own note"),
    ("base-shared-board", "base_shared_board", "no training; each reader sees all four notes"),
    ("grpo-own-note", "baseline", "GRPO 40 steps on the reader, own note only"),
    ("grpo-shared-board", "recipe", "GRPO 40 steps on the reader, all four notes"),
]


def main(question_only: bool) -> None:
    print(provenance(), file=sys.stderr)
    tracked = track(
        "message-board",
        model=BASE_MODEL,
        harness=Harness(label="board@qwen2.5-1.5b", instructions=READ_SYSTEM, model=BASE_MODEL),
    )
    tracked.experiment(
        question="Can GRPO teach copies of a model to use a shared message board: post a note, read the others' notes, and answer better than a copy that only rereads its own?",
        hypothesis="Talk inside one reply was trained away because nothing read it (team-talk). A board has readers, so notes pay: readers trained on the shared board beat readers trained on their own note, and beat a plain vote over the board.",
        method="Qwen2.5-1.5B-Instruct, TRL GRPO + LoRA, 40 steps, 12 problems x 4 copies. Round one: 4 notes per problem from the current model. Round two: each copy reads its board and answers; GRPO trains round two. One change: the board shows all 4 notes or only the copy's own. Seeds 17 and 18.",
        measure="Correct answers on 120 held-out GSM8K test problems, 4 readers each, points out of 100 with a 95% interval; paired delta across both seeds. Also: vote over the board, and the share of readers rescued (own note wrong, answer right) or misled.",
        decide="The shared board wins if its delta over the own-note arm clears the interval and the re-run noise band, and it beats the vote. If readers only copy the majority, the board is a vote, not a conversation.",
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
    for version, key, changed in CONDITIONS:
        a = r["arms"].get(key)
        if not a:
            continue
        run = tracked.run(
            version,
            harness=Harness(
                label=f"{version}@qwen2.5-1.5b", instructions=READ_SYSTEM, model=BASE_MODEL
            ),
            method="grpo" if a.get("steps") else "eval",
        )
        lo, hi = a["ci"]
        run.score(BEHAVIOR, a["score"], fraction=True, ci=(hi - lo) / 2, n=r["n_holdout"])
        b = a.get("board") or {}
        seeds = a.get("per_seed")
        moved = f"Moved: {100 * a['score']:.0f} points [{100 * lo:.0f}, {100 * hi:.0f}]"
        if seeds:
            moved += f"; per seed {', '.join(f'{100 * s:.0f}' for s in seeds)}"
        run.note(
            "\n".join(
                [
                    f"Changed: {changed}",
                    moved,
                    f"Why: notes right {100 * b.get('note_acc', 0):.0f}%, vote over the board {100 * b.get('vote_acc', 0):.0f}%, "
                    f"readers rescued {100 * b.get('rescued', 0):.0f}%, misled {100 * b.get('misled', 0):.0f}%",
                    "Learned: see the recipe README",
                    "Reproduce: python recipes/papers/message-board/recipe.py --train-seeds 17 18",
                ]
            )
        )
        run.finish(steps=a.get("steps") or None, say=False)
    d = r.get("delta", {})
    print(f"posted; shared vs own {100 * d.get('recipe_vs_baseline', 0):+.1f} ({d.get('verdict')})")


if __name__ == "__main__":
    main(question_only="--question" in sys.argv[1:])
