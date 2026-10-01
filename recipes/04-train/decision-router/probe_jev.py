"""Probe TypeSafe's Jev from outside: what its answers say about how it is built.

Seven small experiments, about 150 requests and a few cents: the same request
repeated (is it sampled?), the options shuffled (does order matter?), labels
swapped against descriptions (which does it read?), a statement against its
negation (one pass per question?), and latency against the number of
questions, the length of the state and the number of options (what is shared?).
README.md, "What Jev is", reads the results.

Run: python probe_jev.py   (needs TYPESAFE_API_KEY; writes out/probe.json)
"""

from __future__ import annotations

import json
import os
import random
import statistics as st
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
KEY = os.environ.get("TYPESAFE_API_KEY", "")
URL = "https://api.typesafe.ai/v1/systemone"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def call(state, questions):
    t0 = time.perf_counter()
    r = requests.post(
        URL,
        headers=H,
        json={"state": state, "model": "jev-latest", "questions": questions},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    return j, time.perf_counter() - t0


Q = "A train leaves at 3pm going 60 mph. Another leaves at 4pm going 80 mph on the same track. When does the second catch the first?"
KINDS = {
    "math": "A math word problem.",
    "code": "A programming request.",
    "trivia": "A short factual question.",
    "writing": "A creative writing request.",
}
out = {}

# 1. Sampling noise: identical request 12 times.
reps = []
for _ in range(12):
    j, s = call(
        {"question": Q},
        {
            "k": {
                "type": "choice",
                "instructions": "What kind of question is this?",
                "criteria": KINDS,
            },
            "n": {"type": "noul", "instructions": "This question needs arithmetic."},
        },
    )
    reps.append((j["answers"]["k"]["probabilities"], j["answers"]["n"]["noul"], s))
out["repeat_choice"] = [r[0] for r in reps]
out["repeat_noul"] = [r[1] for r in reps]

# 2. Order bias: an ambiguous question, options shuffled 12 times; track P(each label).
AMB = "Write a short poem about how quicksort works, with the code in a comment."
order = []
for i in range(12):
    items = list(KINDS.items())
    random.Random(i).shuffle(items)
    j, _ = call(
        {"question": AMB},
        {
            "k": {
                "type": "choice",
                "instructions": "What kind of question is this?",
                "criteria": dict(items),
            }
        },
    )
    p = j["answers"]["k"]["probabilities"]
    order.append({"first": items[0][0], "last": items[-1][0], "p": p})
out["order"] = order

# 3. Label vs description: same descriptions, opaque labels.
opaque = {f"opt{i}": d for i, d in enumerate(KINDS.values())}
j, _ = call(
    {"question": Q},
    {"k": {"type": "choice", "instructions": "What kind of question is this?", "criteria": opaque}},
)
out["opaque_labels"] = j["answers"]["k"]["probabilities"]
swapped = dict(
    zip(
        KINDS,
        [
            "A creative writing request.",
            "A short factual question.",
            "A programming request.",
            "A math word problem.",
        ],
    )
)
j, _ = call(
    {"question": Q},
    {
        "k": {
            "type": "choice",
            "instructions": "What kind of question is this?",
            "criteria": swapped,
        }
    },
)
out["labels_swapped_vs_descriptions"] = j["answers"]["k"]["probabilities"]

# 4. Negation symmetry.
neg = []
for stmt, nstmt in [
    ("This question needs arithmetic.", "This question does not need arithmetic."),
    ("This question is about trains.", "This question is not about trains."),
    ("This question is hard for a 10 year old.", "This question is easy for a 10 year old."),
]:
    j, _ = call(
        {"question": Q},
        {"a": {"type": "noul", "instructions": stmt}, "b": {"type": "noul", "instructions": nstmt}},
    )
    neg.append((stmt, j["answers"]["a"]["noul"], j["answers"]["b"]["noul"]))
out["negation"] = neg


def timed(state, questions, n=5):
    ts = []
    for _ in range(n):
        j, s = call(state, questions)
        ts.append(s)
    return round(st.median(ts), 3), j.get("usage")


# 5. Latency vs number of questions.
nq = {}
for k in (1, 4, 16, 64):
    qs = {
        f"q{i}": {"type": "noul", "instructions": f"This question mentions the number {i}."}
        for i in range(k)
    }
    nq[k] = timed({"question": Q}, qs)
out["latency_vs_questions"] = nq

# 6. Latency vs state length.
filler = "The quick brown fox jumps over the lazy dog. " * 2000
sl = {}
for chars in (200, 2000, 20000, 80000):
    sl[chars] = timed(
        {"question": Q, "notes": filler[:chars]},
        {"n": {"type": "noul", "instructions": "This question needs arithmetic."}},
    )
out["latency_vs_state_chars"] = sl

# 7. Latency vs number of options.
no = {}
for k in (2, 12, 64, 250):
    crit = {f"o{i}": f"Option number {i}." for i in range(k)}
    no[k] = timed(
        {"question": Q},
        {
            "k": {
                "type": "choice",
                "instructions": "Pick the option whose number equals the train speed of the first train.",
                "criteria": crit,
            }
        },
    )
out["latency_vs_options"] = no

(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
