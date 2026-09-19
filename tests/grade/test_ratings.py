"""run_judge(scale=(lo, hi)): a rating judge, reward normalized, rating kept
(Lambert 2025, chapter Preference Data, ratings vs rankings)."""

from __future__ import annotations

import pytest

from whileai.simulations.score.judging import evaluate, normalize_judge_result, run_judge


def _row(i):
    return {"prompt": f"ask {i}", "final_text": "ok", "steps": [], "messages": []}


def test_normalize_with_a_scale_maps_ratings_and_keeps_them():
    out = normalize_judge_result(4, scale=(1, 5))
    assert out["reward"] == 0.75 and out["judge_status"] == "ok"
    assert out["judge_meta"] == {"rating": 4.0, "scale": [1.0, 5.0]}
    assert normalize_judge_result(5, scale=(1, 5))["reward"] == 1
    assert normalize_judge_result(1, scale=(1, 5))["reward"] == 0
    assert normalize_judge_result(7, scale=(0, 10))["reward"] == 0.7
    dict_form = normalize_judge_result(
        {"rating": 3, "reason": "fine", "markers": {"tone": 1.0}}, scale=(1, 5)
    )
    assert dict_form["reward"] == 0.5 and dict_form["reason"] == "fine"
    assert dict_form["judge_meta"]["markers"] == {"tone": 1.0}
    assert dict_form["judge_meta"]["rating"] == 3.0
    assert normalize_judge_result({"score": 2}, scale=(1, 5))["reward"] == 0.25


def test_out_of_scale_and_missing_stay_contract_breaks():
    bad = normalize_judge_result(0, scale=(1, 5))
    assert bad["reward"] is None and bad["judge_status"] == "invalid_result"
    assert bad["judge_meta"]["rating_out_of_scale"] == 0.0
    assert normalize_judge_result("four", scale=(1, 5))["judge_status"] == "invalid_result"
    assert normalize_judge_result(True, scale=(1, 5))["judge_status"] == "invalid_result"
    assert normalize_judge_result({"reason": "x"}, scale=(1, 5))["judge_status"] == "missing_reward"
    # without a scale a 4 is still a contract break, as before
    assert normalize_judge_result(4)["judge_status"] == "invalid_result"
    with pytest.raises(ValueError, match="hi > lo"):
        normalize_judge_result(3, scale=(5, 1))


def test_run_judge_and_evaluate_thread_the_scale():
    ratings = {"ask 0": 5, "ask 1": 3, "ask 2": 1, "ask 3": 9}

    def judge(row):
        return {"rating": ratings[row["prompt"]], "reason": "r"}

    scored = run_judge([_row(i) for i in range(4)], judge, scale=(1, 5), concurrency=2)
    rewards = [r["reward"] for r in scored.rows]
    assert rewards[:3] == [1, 0.5, 0] and rewards[3] is None
    assert scored.rows[1]["judge_meta"]["rating"] == 3.0
    assert scored.rows[3]["judge_status"] == "invalid_result"
    held = evaluate([_row(0)], judge, scale=(1, 5))
    assert held.rows[0]["reward"] == 1 and held.rows[0]["lineage"]["source"] == "eval"
