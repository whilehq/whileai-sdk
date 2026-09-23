"""Build the pinned task files: MATH problems and their broken twins.

    python build_data.py            # writes data/train.jsonl and data/holdout.jsonl

A broken twin is the same problem with one edit that leaves it without a
unique answer (a contradicting condition, or a missing one). One model
writes the edit, a second model checks it, and only twins the checker calls
ill-posed are kept, so the gold for a twin is ``\\boxed{NONE}``. The edit is
LLM-made, which is why the files are committed: a rerun of the recipe reads
the same tasks, and this script is how they were made, not a step a rerun
repeats. Keys: ``OPENROUTER_API_KEY`` (your own).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import random
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://openrouter.ai/api/v1/chat/completions"
EDITOR = "deepseek/deepseek-v4-pro"
CHECKER = "z-ai/glm-5.3"
SEED = 0
N_TRAIN, N_HOLDOUT = 320, 120  # source problems
TWIN_SHARE = (
    0.85  # a twin survives the checker ~half the time, so ~40% of each split ends up broken
)

BREAK = """Here is a math problem with a unique answer ({gold}).

{problem}

Rewrite it with ONE small edit so that it has NO valid unique answer: either add a condition that contradicts the others,
or remove a condition so the answer is not determined. Keep the style and length; the flaw should not be announced and
should take some work to notice. Reply with JSON only: {{"problem": "...", "flaw": "one sentence"}}"""

CHECK = """Is this math problem well-posed with a single answer? Work it out carefully.

{problem}

Reply at the end with JSON only: {{"well_posed": true|false, "why": "one sentence"}}"""


def last_json(text: str) -> dict | None:
    dec, best = json.JSONDecoder(), None
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    best = obj
            except ValueError:
                pass
    return best


def chat(model: str, prompt: str) -> dict | None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "reasoning": {"effort": "medium"},
    }
    req = urllib.request.Request(
        URL,
        json.dumps(body).encode(),
        {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            msg = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]
            return last_json(msg.get("content") or "")
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def twin(task: dict) -> dict | None:
    edit = chat(EDITOR, BREAK.format(problem=task["problem"], gold=task["gold"]))
    if not edit or not edit.get("problem"):
        return None
    check = chat(CHECKER, CHECK.format(problem=edit["problem"]))
    if not check or check.get("well_posed") is not False:
        return None
    return {
        "task_id": task["task_id"] + "-broken",
        "source": task["task_id"],
        "kind": "broken",
        "problem": edit["problem"],
        "gold": "NONE",
        "flaw": edit.get("flaw", ""),
        "check": check.get("why", ""),
    }


def math_tasks() -> tuple[list[dict], list[dict]]:
    """MATH train levels 2-4 for training, MATH-500 levels 2-4 held out (MATH test split, so disjoint)."""
    from datasets import load_dataset

    from whileai.simulations.verify.math import (
        extract_answer,
    )  # the gold is the solution's last box

    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    train = train.filter(lambda r: r["level"] in ("Level 2", "Level 3", "Level 4")).shuffle(
        seed=SEED
    )
    test = (
        load_dataset("HuggingFaceH4/MATH-500", split="test")
        .filter(lambda r: r["level"] in (2, 3, 4))
        .shuffle(seed=SEED)
    )
    tr = [
        {
            "task_id": f"math-train-{i}",
            "kind": "solvable",
            "problem": r["problem"],
            "gold": extract_answer(r["solution"]) or "",
        }
        for i, r in enumerate(train.select(range(N_TRAIN)))
    ]
    ho = [
        {
            "task_id": f"math500-{r['unique_id'].replace('/', '-').removesuffix('.json')}",
            "kind": "solvable",
            "problem": r["problem"],
            "gold": r["answer"],
        }
        for r in test.select(range(N_HOLDOUT))
    ]
    return [t for t in tr if t["gold"]], ho


def main() -> None:
    train, holdout = math_tasks()
    with cf.ThreadPoolExecutor(24) as ex:
        twins = [t for t in ex.map(twin, train + holdout) if t]
    by_src = {t["source"]: t for t in twins}
    rnd = random.Random(SEED)
    for name, tasks in (("train", train), ("holdout", holdout)):
        # every source contributes either itself or its twin, never both, so no
        # solvable problem is the answer key to a broken one in the same split
        rows = []
        for t in tasks:
            tw = by_src.get(t["task_id"])
            rows.append(tw if tw and rnd.random() < TWIN_SHARE else t)
        rnd.shuffle(rows)
        (HERE / "data" / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf8"
        )
        print(name, len(rows), "broken", sum(r["kind"] == "broken" for r in rows))


if __name__ == "__main__":
    main()
