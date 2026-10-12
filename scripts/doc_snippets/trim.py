"""The reader's side of docs/trim.md.

The page trains a model once per subset size, which takes GPUs. This file
stands in for that: `scored` is 16,000 graded SFT rows, and `train_and_eval`
returns per-item scores from a known learning curve (the untuned model at
0.30 rising to 0.70, rectified form, arXiv:2402.02314), so the page's own
`trim` calls run for real and the printed verdict can be checked against a
known answer (N* about 3,700 rows at min_gain 0.01).
"""

import random

base_model = "untuned"


def _row(i):
    ask = f"Question {i}: total sales for region r{i % 37} in month m{i % 12}?"
    say = f"The total for r{i % 37} in m{i % 12} is {i * 7} dollars."
    return {
        "prompt": ask,
        "final_text": say,
        "messages": [{"role": "user", "content": ask}, {"role": "assistant", "content": say}],
        "reward": 1.0,
    }


scored = [_row(i) for i in range(16000)]

_rng = random.Random(0)
_DIFFICULTY = [_rng.random() for _ in range(1000)]


def _p(n):
    return 0.30 + 0.40 * n / (200 + n)


def evaluate(model):
    return [1.0 if u < 0.30 else 0.0 for u in _DIFFICULTY]


def train_and_eval(rows, seed):
    p = _p(len(rows)) + random.Random(seed * 100003 + len(rows)).gauss(0, 0.005)
    return [1.0 if u < p else 0.0 for u in _DIFFICULTY]
