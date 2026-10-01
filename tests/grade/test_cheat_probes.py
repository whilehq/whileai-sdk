"""Cheat probes: degenerate agents scored by the user's scorer (cheat_probes).

One planted scorer per hole: each pays exactly one kind of no-work reply,
so the probe that plays that agent must flag and the others must not. A
strict exact-match scorer flags nothing. No network, fixed seeds.
"""

from __future__ import annotations

import re

import pytest

import whileai as wai
from whileai.simulations.score.cheat_probes import CHEAT_PROBES, cheat_probes
from whileai.simulations.score.judging import run_judge

# Forty cases with distinct answers and no "?" in the prompt, so only the
# ask probe carries a question mark.
CASES = [{"prompt": f"Compute {i} plus {i}.", "reference": str(2 * i)} for i in range(40)]


def asks_question(row: dict) -> float:
    """Gameable: credit for any answer containing a question mark."""
    return float("?" in row["final_text"])


def rewards_refusal(row: dict) -> float:
    """Gameable: credit for declining."""
    return float(bool(re.search(r"\b(sorry|can't|cannot)\b", row["final_text"], re.I)))


def exact_match(row: dict) -> float:
    """Strict: the reply is the reference and nothing else."""
    return float(row["final_text"].strip() == str(row["reference"]))


def test_question_mark_scorer_flags_the_ask_probe():
    report = cheat_probes(CASES, asks_question)
    assert report["flagged"] == ["ask"]
    ask = report["probes"]["ask"]
    assert ask["mean"] == 1.0 and ask["n"] == 40
    assert ask["ci95"][0] >= 0.9  # Wilson on 40 of 40
    assert report["ok"] is False
    assert any(w.startswith("scorer pays a degenerate agent: ask") for w in report["warnings"])
    assert "FLAGGED" in str(report)


def test_refusal_scorer_flags_the_refuse_probe():
    report = cheat_probes(CASES, rewards_refusal)
    assert report["flagged"] == ["refuse"]
    assert report["probes"]["refuse"]["mean"] == 1.0


def test_exact_match_scorer_flags_nothing():
    report = cheat_probes(CASES, exact_match)
    assert report["flagged"] == []
    assert report["ok"] is True
    # every answer is distinct, so "always the most common answer" is right once
    assert report["probes"]["constant"]["mean"] == pytest.approx(1 / 40)
    assert all(p["ci95"][0] < 0.1 for p in report["probes"].values())


def test_constant_answer_flags_on_a_lopsided_answer_set():
    lopsided = [
        {"prompt": f"Is item {i} in stock.", "answer": "yes" if i % 4 else "no"} for i in range(40)
    ]
    report = cheat_probes(lopsided, lambda r: float(r["final_text"] == r["answer"]))
    assert report["flagged"] == ["constant"]
    assert report["probes"]["constant"]["answer_share"] == 0.75
    assert "'yes' is the reference on 30 of 40 cases" in " ".join(report["warnings"])


def test_same_seed_same_report_and_seed_moves_the_sample():
    many = [{"prompt": f"Compute {i} plus one.", "reference": str(i + 1)} for i in range(100)]
    a = cheat_probes(many, asks_question, sample=40, seed=7)
    b = cheat_probes(many, asks_question, sample=40, seed=7)
    assert a == b and str(a) == str(b)
    c = cheat_probes(many, asks_question, sample=40, seed=8)
    assert a["probes"]["echo"]["reply"] == "Compute 41 plus one."
    assert c["probes"]["echo"]["reply"] == "Compute 29 plus one."


def test_agent_score_comes_from_graded_rows_and_the_method_matches():
    scored = run_judge(
        [{**c, "final_text": c["reference"]} for c in CASES], exact_match, concurrency=1
    )
    report = scored.cheat_probes(exact_match)
    assert report["agent"]["mean"] == 1.0
    assert report["probes"]["empty"]["vs_agent"] == -1.0
    assert report == cheat_probes(scored.rows, exact_match)


def test_small_sample_over_the_flag_is_a_note_not_a_flag():
    # 1 of 4 cases pays the ask probe: 0.25 is over 0.10, the interval is not
    few = CASES[:4]
    report = cheat_probes(few, lambda r: float("?" in r["final_text"] and r["reference"] == "0"))
    assert report["flagged"] == []
    assert any(n.startswith("ask: 0.25") for n in report["notes"])


def test_two_argument_scorer_and_unknown_probe():
    report = cheat_probes(
        CASES, lambda prompt, completion: float(completion == ""), probes=["empty"]
    )
    assert list(report["probes"]) == ["empty"] and report["flagged"] == ["empty"]
    with pytest.raises(ValueError, match="choose from"):
        cheat_probes(CASES, exact_match, probes=["bogus"])
    assert set(CHEAT_PROBES) == {"refuse", "ask", "empty", "echo", "filler", "constant"}


def test_constant_probe_skips_without_references():
    report = cheat_probes([{"prompt": "hi there"}] * 5, asks_question)
    assert "no reference answers" in report["probes"]["constant"]["skipped"]


def test_resolves_from_the_one_import():
    assert wai.cheat_probes is cheat_probes
