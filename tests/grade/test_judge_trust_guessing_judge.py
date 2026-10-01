"""A guessing judge is told it is guessing, not that it reads length.

External case study (gentlyventures.com/casestudies/whileai, whileai
0.126): three scripted judges against 200 human labels. Honest kappa
1.00; length-biased kappa 0.44 with a 68-point short/long gap; coin flip
kappa 0.04. The length warning fired on the coin flip too, which points
the user at the wrong fix. Scripted judges, fixed seeds, no network.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from whileai.simulations.score.judge_trust import judge_trust

N_ROWS = 200
REPLY = "Done: the invoice total is 42."
DETAIL = " Line item checked against the ledger."
# Replies run 30 to about 450 characters; a length-biased judge passes
# anything over LENGTH_CUT, near the median.
LENGTH_CUT = 240


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _labeled(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(N_ROWS):
        final = REPLY + DETAIL * rng.randint(0, 11)
        rows.append(
            {
                "prompt": f"case {seed}-{i}",
                "final_text": final,
                "messages": [
                    {"role": "user", "content": f"case {seed}-{i}"},
                    {"role": "assistant", "content": final},
                ],
                "steps": [],
                "gold_reward": int(rng.random() < 0.5),
                "gold_kind": "human",
            }
        )
    return rows


def _judges(rows: list[dict], seed: int) -> dict:
    gold = {r["prompt"]: r["gold_reward"] for r in rows}

    def honest(row):
        return {"score": gold[row["prompt"]], "reason": "matches the label"}

    def length_biased(row):
        # Seven rows in ten are decided by length alone, the rest honestly.
        if _h(str(seed), row["prompt"]) % 10 < 7:
            return {"score": int(len(row["final_text"]) > LENGTH_CUT), "reason": "long"}
        return {"score": gold[row["prompt"]], "reason": "matches the label"}

    def coin_flip(row):
        # A fair coin per (row, text): stable on an identical re-judge,
        # independent of length and of the label.
        return {"score": _h(str(seed), row["prompt"], row["final_text"]) % 2, "reason": "coin"}

    return {"honest": honest, "length_biased": length_biased, "coin_flip": coin_flip}


def _report(seed: int, name: str):
    rows = _labeled(seed)
    judge = _judges(rows, seed)[name]
    for r in rows:
        r["reward"] = judge(r)["score"]
    return judge_trust(rows, judge, concurrency=1)


def _length_warnings(report) -> list[str]:
    return [w for w in report["warnings"] if "length" in w]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_honest_judge_is_clean(seed):
    report = _report(seed, "honest")
    assert report["agreement"]["kappa"] == pytest.approx(1.0)
    assert report["warnings"] == [] and report["ok"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_length_biased_judge_gets_the_length_warning(seed):
    report = _report(seed, "length_biased")
    assert report["length_sensitivity"]["max_gap"] > 0.5
    assert report["length_sensitivity"]["flagged"]
    assert _length_warnings(report), report["warnings"]
    assert not report["ok"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coin_flip_judge_is_called_guessing_not_length_biased(seed):
    report = _report(seed, "coin_flip")
    assert abs(report["agreement"]["kappa"]) < 0.2
    assert not report["ok"]
    assert any("guessing" in w for w in report["warnings"]), report["warnings"]
    assert _length_warnings(report) == [], report["warnings"]
    assert not report["length_sensitivity"]["flagged"]
    assert not report["perturbation"]["flagged_length"]
