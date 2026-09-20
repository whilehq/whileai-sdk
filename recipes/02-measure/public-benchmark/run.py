"""A public benchmark into the measurement: 200 GSM8K test questions, a model's
answers, and every number with its interval. Offline, no key, seconds.

    python run.py                    # 200 questions x 5 answers, both arms
    python run.py --limit 40 --k 5   # a smoke run
    python run.py --json out.json    # every number as one file

The road, in the order the output prints:

1. **Rows.** ``wai.rows(questions, answers, MathEqual(), references=gold)``:
   the questions and the model's answers become the rows every measurement
   reads. The reward is a program against the GSM8K gold, not a judge.
2. **pass@1 with its interval**, pass^k and pass@k, over tasks.
3. **The eval's own noise.** The same model scored three times
   (``eval_variance``): a before/after delta inside ``noise_band`` is what
   re-running the eval does on its own.
4. **How many tasks a holdout needs** to prove a five-point gain
   (``holdout_size``), read off the before arm.
5. **The groups that carry gradient.** ``select(mode="rl", band=(0.2, 0.8))``
   drops the questions the model always or never solves.
6. **Is the change real.** ``compare(before, after, run_std=...)``.

The model is a seeded stand-in: each question has a difficulty drawn from
its own text, each arm a skill, and the answer is the gold or a wrong number
at that rate, wrapped in a short chain of thought. Swap ``answer()`` for a
call to your model and nothing else changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import whileai as wai
from whileai.config import provenance
from whileai.simulations import eval_variance, holdout_size

HERE = Path(__file__).resolve().parent
# The first 200 rows of the GSM8K test split (Cobbe et al. 2021,
# arXiv:2110.14168; github.com/openai/grade-school-math, MIT licence), as
# ``{"id", "question", "answer"}``. The answer is the worked solution and
# ends on ``#### <number>``, the gold.
DATA = HERE / "gsm8k_test_200.jsonl"

# The stand-in model. SKILL is the arm's chance of the gold on a question of
# middle difficulty; the tuned arm is five points better, the gain the recipe
# then has to prove. Both are round numbers for a demo, not measurements.
SKILL = {"before": 0.55, "after": 0.60}
# EFFECT = 0.05: the gain in pass rate the holdout is sized to prove (the
# five points ``SKILL`` plants).
EFFECT = 0.05
# BAND = (0.2, 0.8): keep questions the model passes 20 to 80% of the time
# (Lambert 2025, chapter Reasoning; DAPO arXiv:2503.14476 drops accuracy 0
# and 1 groups). The SDK's own default; named here so the README can point
# at it.
BAND = (0.2, 0.8)
# RUNS = 3: the fewest re-runs ``eval_variance`` reads a standard deviation
# from (two degrees of freedom).
RUNS = 3


def load(limit: int | None) -> tuple[list[str], list[str], list[str]]:
    ids, questions, gold = [], [], []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.append(row["id"])
        questions.append(row["question"])
        gold.append(row["answer"].split("####")[-1].strip().replace(",", ""))
        if limit and len(ids) >= limit:
            break
    return ids, questions, gold


def difficulty(question: str) -> float:
    """A stable number in [0, 1] from the question text: how much harder
    than average this question is for the stand-in. Some questions land
    near 0 or 1, so the model always or never solves them, which is what
    ``select`` has to drop."""
    digest = hashlib.sha256(question.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def answer(question: str, gold: str, *, arm: str, seed: int, draw: int) -> str:
    """One completion: a two-line chain of thought ending on the answer."""
    rng = random.Random(f"{arm}:{seed}:{draw}:{question}")
    p = min(1.0, max(0.0, SKILL[arm] + 0.9 * (0.5 - difficulty(question))))
    right = rng.random() < p
    number = gold if right else str(int(float(gold)) + rng.choice([-3, -1, 1, 2, 10]))
    # Each draw reads differently: ``select`` drops a repeated reply as a
    # duplicate, the way it would a sampler stuck at temperature 0.
    return f"Attempt {draw + 1}: work through it step by step.\nThe answer is {number}"


def arm(questions: list[str], gold: list[str], ids: list[str], *, name: str, seed: int, k: int):
    completions = [
        [answer(q, g, arm=name, seed=seed, draw=j) for j in range(k)]
        for q, g in zip(questions, gold)
    ]
    return wai.rows(questions, completions, wai.verify.MathEqual(), references=gold, task_ids=ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="questions to use (all 200)")
    parser.add_argument("--k", type=int, default=5, help="answers per question")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="write every number here")
    args = parser.parse_args(argv)
    print(provenance(), file=sys.stderr)

    ids, questions, gold = load(args.limit)
    print(f"{len(questions)} GSM8K test questions, {args.k} answers each, reward MathEqual\n")

    # 1. rows, 2. pass@1
    before = arm(questions, gold, ids, name="before", seed=args.seed, k=args.k)
    passed = wai.pass_at(before)
    print("before:", passed)

    # 3. the eval's own noise: the same model scored three times
    reruns = [
        arm(questions, gold, ids, name="before", seed=args.seed + i, k=args.k) for i in range(RUNS)
    ]
    noise = eval_variance(*reruns)
    print(
        f"eval noise over {RUNS} passes: run_std {noise['run_std']:.3f}, "
        f"a delta under {noise['noise_band']:.3f} is re-run noise"
    )

    # 4. how many tasks prove a five-point gain
    size = holdout_size(EFFECT, before=before)
    print(
        f"holdout for a {EFFECT:.0%} gain: {size['n_tasks']} tasks "
        f"(base {size['base']:.2f}, k {size['k']}, sd from {size['sd_source']})"
    )

    # 5. the groups that carry gradient
    picked = wai.select(before, mode="rl", band=BAND)
    print(picked)

    # 6. is the change real
    after = arm(questions, gold, ids, name="after", seed=args.seed, k=args.k)
    report = wai.compare(before, after, run_std=noise["run_std"], run_std_runs=RUNS)
    print()
    print(report)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "n_questions": len(questions),
                    "k": args.k,
                    "pass_at": passed.to_dict(),
                    "eval_variance": noise,
                    "holdout_size": size,
                    "select": picked.report,
                    "compare": dict(report),
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
