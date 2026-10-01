"""Did my prompt rewrite really help? Both prompts on one local model, one verdict.

The old prompt and the new one each answer the same 24 questions four
times, on the same sampling seeds, and ``wai.harness.compare`` hands the
two sets of graded replies to ``wai.compare``: the gain, its 95% interval
over tasks, and PASS only when the interval supports it.

Offline by default: ``scripted_model`` below stands in for a small local
model and is planted to do better under the new prompt. ``--model
ollama:qwen3:4b-instruct`` runs the same check on a real one, free, on
this machine.

Run: python recipes/02-measure/before-and-after/run.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import whileai as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
BEFORE = (HERE / "before.txt").read_text(encoding="utf-8").strip()
AFTER = (HERE / "after.txt").read_text(encoding="utf-8").strip()
TASKS = HERE / "tasks.jsonl"

# The stand-in's chance of the right number under each prompt, on a
# question of middle difficulty. Round numbers for a demo: the rewrite is
# planted to help by 30 points, so the check has a real gain to find.
SKILL = {BEFORE: 0.5, AFTER: 0.8}


def scripted_model(messages: list[dict], seed: int) -> str:
    """A fake small model: ``(messages, seed) -> reply``, the shape
    ``model=`` takes. Right at the rate its system prompt earns, else a
    near miss; the same question and seed give the same reply."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    question = messages[-1]["content"]
    a, b = (int(x) for x in question.rstrip("?").split("is ")[-1].split(" * "))
    hard = random.Random(f"difficulty:{question}").random()
    p = min(1.0, max(0.0, SKILL.get(system, 0.5) + 0.4 * (0.5 - hard)))
    rng = random.Random(f"{question}:{seed}")
    number = a * b if rng.random() < p else a * b + rng.choice([-10, -1, 1, 2, 10])
    return f"{a} times {b}: multiply the tens, then the ones.\nThe answer is {number}"


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model", default=None, help="ollama:qwen3:4b-instruct or any spec (default: scripted)"
    )
    p.add_argument("--k", type=int, default=4, help="replies per task per arm")
    p.add_argument("--seed", type=int, default=0, help="same seed, same output")
    p.add_argument("--same", action="store_true", help="old prompt on both arms: the null check")
    args = p.parse_args(argv)

    report = wai.harness.compare(
        BEFORE,
        BEFORE if args.same else AFTER,
        tasks=TASKS,
        reward=wai.verify.Numeric(),
        model=args.model or scripted_model,
        k=args.k,
        seed=args.seed,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
