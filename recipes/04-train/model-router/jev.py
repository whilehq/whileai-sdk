"""Ask TypeSafe's Jev to route the val and held-out questions, and cache its answers for run.py.

Jev is a decision model: it reads a piece of state plus typed questions and
returns a probability for every answer you name, in about a twentieth of a
second, generating no text. That is the shape a router wants as you type.
Each question gets one request with two questions in it:

- `task`: which of the twelve kinds of question this is. run.py turns the
  probabilities into each model's expected accuracy and cost with the
  training table, the way a task-type router such as OpenRouter's works.
- `model`: which of the twelve models will answer it correctly, given each
  model's training accuracy on every kind of question. run.py weighs the
  probabilities against cost with the same knob as the other routers.

Jev cannot be trained: the training table reaches it only as the text of
the `model` criteria. Training questions are never sent.

Run: python jev.py   (after prepare.py; needs TYPESAFE_API_KEY; ~2,200 requests, cents)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from prepare import DATASETS, MODELS
from run import load, split

HERE = Path(__file__).resolve().parent
URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
MAX_CHARS = (
    4000  # of the question sent as state; Jev bills input only, at $0.042 per million tokens
)
WORKERS = 8  # Jev's stated limit is 1,200 requests a minute
TRIES = 4

# The twelve kinds of question, in plain words: the task labels Jev picks from.
TASKS = {
    "aime": "A competition math problem whose answer is a single integer.",
    "arc-agi": "A grid puzzle given as arrays of numbers: infer the rule that maps inputs to outputs.",
    "arenahard_coding": "An open-ended programming request: write, fix or explain code.",
    "arenahard_creative_writing": "A creative writing request: a story, poem, script or other prose.",
    "arenahard_math": "An open-ended math request: a derivation, proof or worked explanation.",
    "gpqa": "A graduate-level multiple-choice question in physics, chemistry or biology.",
    "hle": "An expert-level question at the frontier of some field, far harder than an exam.",
    "livecodebench": "A competitive programming problem with an input and output format.",
    "livemathbench": "A recent contest math problem asking for a number or expression.",
    "mmlupro": "A multiple-choice knowledge question with up to ten options, any subject.",
    "simpleqa": "A short factual question with one specific answer: a name, date or number.",
    "swe-bench": "A bug report from a real GitHub repository that asks for a code patch.",
}
assert list(TASKS) == [d for d, _ in DATASETS]

TASK_INSTRUCTIONS = "What kind of question is this?"
MODEL_INSTRUCTIONS = (
    "Which model is most likely to answer this question correctly? Each option "
    "lists the model's accuracy on earlier questions of each kind."
)


def model_criteria(rows, fit) -> dict[str, str]:
    """One line per model: its accuracy on each kind of question, from the fit split only."""
    out = {}
    for m in MODELS:
        parts = []
        for ds, desc in TASKS.items():
            s = [r["score"][m] for r, f in zip(rows, fit) if f and r["dataset"] == ds]
            parts.append(f"{desc.split(':')[0].rstrip('.').lower()}: {sum(s) / len(s):.0%}")
        out[m] = f"{m}. Accuracy by kind of question: " + "; ".join(parts) + "."
    return out


def ask(key: str, query: str, criteria: dict) -> dict:
    body = {
        "state": {"question": query[:MAX_CHARS]},
        "model": JEV_MODEL,
        "questions": {
            "task": {"type": "choice", "instructions": TASK_INSTRUCTIONS, "criteria": TASKS},
            "model": {"type": "choice", "instructions": MODEL_INSTRUCTIONS, "criteria": criteria},
        },
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(TRIES):
        t0 = time.perf_counter()
        resp = requests.post(URL, headers=headers, json=body, timeout=60)
        if resp.status_code == 200:
            reply = resp.json()
            reply["seconds"] = time.perf_counter() - t0
            return reply
        if resp.status_code not in (429, 500, 502, 503, 504) or attempt == TRIES - 1:
            raise RuntimeError(f"TypeSafe {resp.status_code}: {resp.text[:300]}")
        time.sleep(2**attempt)
    raise AssertionError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(HERE / "out"))
    p.add_argument("--limit", type=int, default=0, help="ask about the first N questions only")
    args = p.parse_args(argv)
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        sys.exit("set TYPESAFE_API_KEY (console.typesafe.ai/settings/keys)")
    out = Path(args.out)
    rows, _ = load(out)
    queries = {}
    for line in (out / "queries.jsonl").open(encoding="utf-8"):
        q = json.loads(line)
        queries[q["id"]] = q["query"]
    fit, val, test = split(rows)
    criteria = model_criteria(rows, fit)
    cache_path = out / "jev.jsonl"
    done = set()
    if cache_path.exists():
        done = {json.loads(line)["id"] for line in cache_path.open(encoding="utf-8")}
    todo = [r["id"] for r, v, t in zip(rows, val, test) if (v or t) and r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} cached, asking Jev about {len(todo)} questions", file=sys.stderr)

    def one(i):
        reply = ask(key, queries[i], criteria)
        a = reply["answers"]
        return {
            "id": i,
            "jev": reply.get("model"),
            "task": a["task"]["probabilities"],
            "model": a["model"]["probabilities"],
            "input_tokens": (reply.get("usage") or {}).get("input_tokens"),
            "seconds": reply["seconds"],
        }

    with cache_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(WORKERS) as pool:
        for n, rec in enumerate(pool.map(one, todo), 1):
            fh.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}", file=sys.stderr)
    print(f"wrote {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
