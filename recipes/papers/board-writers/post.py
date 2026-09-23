"""Put this recipe on while.ai/platform/experiments as the agent `board-writers`.

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

BEHAVIOR = "math_correct"
# (version, results.json arm, one line on what changed)
CONDITIONS = [
    ("base", "base", "no training; four notes, each reader sees two on a ring"),
    (
        "readers-only",
        "baseline",
        "GRPO 80 steps; readers paid, notes in the batch at zero advantage",
    ),
    (
        "readers-and-writers",
        "recipe",
        "GRPO 80 steps; readers paid, each note paid its two readers' mean",
    ),
]


def main(question_only: bool) -> None:
    print(provenance(), file=sys.stderr)
    tracked = track(
        "board-writers",
        model=BASE_MODEL,
        harness=Harness(label="ring@qwen2.5-1.5b", instructions=READ_SYSTEM, model=BASE_MODEL),
    )
    tracked.experiment(
        question="If each note on the board is paid by how its readers did, do the notes get useful and the team get more problems right?",
        hypothesis="In message-board the notes were never trained: long, cut off before the answer, barely better than rereading your own. Paying a note its readers' reward makes notes short and answer-bearing, and the readers score higher.",
        method="Qwen2.5-1.5B-Instruct, TRL GRPO + LoRA, 80 steps, 12 MATH problems (levels 3-5) x 4 copies. Each copy posts a note; reader j reads notes j and j+1 (a ring, so every note has two readers). One change: notes train on their readers' mean reward, or sit in the batch at zero advantage. Seeds 17, 18, 19.",
        measure="Correct answers on 300 held-out MATH-500 problems (levels 3-5), 4 readers each, points out of 100 with a 95% interval; paired delta across three seeds. Also: notes that carry a boxed answer, note length, vote over the board.",
        decide="Writers win if the delta clears the interval and the re-run noise band. If notes get shorter and carry answers but the readers do not move, the board was not the bottleneck.",
    )
    tracked.behavior(
        BEHAVIOR,
        graded_by="program",
        reward_is_judge=False,
        test_version="math500-l3to5-s0-300",
        n=300,
        description="Final boxed answer equals the MATH-500 answer, levels 3 to 5.",
        rubric="Math-Verify equality against the MATH-500 answer. No judge.",
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
                    f"Why: notes with an answer {100 * b.get('note_boxed', 0):.0f}%, "
                    f"{b.get('note_chars', 0)} chars; notes right {100 * b.get('note_acc', 0):.0f}%, "
                    f"vote {100 * b.get('vote_acc', 0):.0f}%, rescued {100 * b.get('rescued', 0):.0f}%",
                    "Learned: see the recipe README",
                    "Reproduce: python recipes/papers/board-writers/recipe.py",
                ]
            )
        )
        run.finish(steps=a.get("steps") or None, say=False)
    d = r.get("delta", {})
    print(
        f"posted; writers vs readers-only {100 * d.get('recipe_vs_baseline', 0):+.1f} ({d.get('verdict')})"
    )


if __name__ == "__main__":
    main(question_only="--question" in sys.argv[1:])
