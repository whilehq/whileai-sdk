"""#257: how many tasks prove a gain, checked against the test the
verdict actually runs. #254: the next round trains on the band."""

from __future__ import annotations

import random

import pytest

import whileai.simulations as wai
from whileai.simulations.ingest import platform
from whileai.simulations.score.delta import delta_report
from whileai.simulations.score.stats import compare_runs, detectable_effect, holdout_size


def test_the_recipes_numbers_come_out():
    """140 tasks at k=4 around 0.6 gave a +-0.06 band in the text-to-sql
    recipe; a 3-point gain needs about a thousand tasks."""
    need = holdout_size(0.03, base=0.6, k=4)
    assert 900 <= need["n_tasks"] <= 1200
    assert need["k"] == 4 and need["power"] == 0.8
    band = holdout_size(0.03, base=0.6, k=4)  # half_width is reported at n_tasks
    assert band["half_width"] < 0.03
    at_140 = detectable_effect(140, base=0.6, k=4)
    assert 0.07 <= at_140 <= 0.10  # a 3-point gain cannot be proven there
    # the band the recipe saw, from the same model: 1.96 * sd / sqrt(140)
    sd = holdout_size(0.0001, base=0.6, k=4)["task_std"]
    assert 0.05 <= 1.96 * sd / 140**0.5 <= 0.065


def test_more_rollouts_or_a_bigger_effect_need_fewer_tasks():
    assert holdout_size(0.05, k=8)["n_tasks"] < holdout_size(0.05, k=4)["n_tasks"]
    assert holdout_size(0.10)["n_tasks"] < holdout_size(0.05)["n_tasks"]
    assert holdout_size(0.05, power=0.9)["n_tasks"] > holdout_size(0.05, power=0.8)["n_tasks"]
    with pytest.raises(ValueError, match="between 0 and 1"):
        holdout_size(0)
    assert detectable_effect(1) is None


def _paired_rows(n_tasks, k, p_before, p_after, rng):
    before, after = [], []
    for t in range(n_tasks):
        for i in range(k):
            before.append(
                {
                    "scenario_id": f"t{t}",
                    "rollout_index": i,
                    "prompt": f"p{t}",
                    "final_text": f"x{i}",
                    "reward": 1 if rng.random() < p_before else 0,
                    "judge_status": "ok",
                }
            )
            after.append(
                {
                    "scenario_id": f"t{t}",
                    "rollout_index": i,
                    "prompt": f"p{t}",
                    "final_text": f"x{i}",
                    "reward": 1 if rng.random() < p_after else 0,
                    "judge_status": "ok",
                }
            )
    return before, after


def test_holdout_size_predicts_the_power_of_compare_runs():
    """At the size it returns, a real gain of that size is proven about
    80% of the time by the paired bootstrap ``delta_report`` uses."""
    need = holdout_size(0.15, base=0.5, k=4, power=0.8)["n_tasks"]
    rng = random.Random(7)
    wins = 0
    reps = 40
    for r in range(reps):
        before, after = _paired_rows(need, 4, 0.5, 0.65, rng)
        out = compare_runs(before, after, n_boot=200, seed=r)
        wins += out["verdict"] == "b_better"
    assert 0.6 <= wins / reps <= 0.95, wins / reps


def test_holdout_size_reads_base_and_k_off_rows():
    rng = random.Random(1)
    before, _ = _paired_rows(30, 4, 0.6, 0.6, rng)
    need = holdout_size(0.05, rows=before)
    assert need["k"] == 4 and 0.4 <= need["base"] <= 0.8
    with pytest.raises(ValueError, match="grade them first"):
        holdout_size(0.05, rows=[{"prompt": "p"}])


def test_delta_report_says_what_the_holdout_can_prove():
    rng = random.Random(3)
    before, _ = _paired_rows(40, 4, 0.6, 0.6, rng)
    after = [dict(r) for r in before]  # the same policy twice: nothing moved
    report = delta_report(before, after, target="pass_at_1", n_boot=200)
    assert report["target_verdict"] == "no_change_detected"
    assert report["detectable_effect"] and report["detectable_effect"] > 0.1
    joined = " ".join(report["warnings"])
    assert "40 paired tasks at k=4 can prove a gain of about" in joined
    assert "holdout_size" in joined
    if report["target_delta"] > 0:
        assert report["tasks_needed"] and "you need about" in joined


def test_push_warns_when_the_holdout_is_too_small(monkeypatch):
    calls = []

    def fake_call(method, path, api_key=None, body=None, **kw):
        calls.append((method, path))
        if path == "/datasets":
            return {"datasetId": "ds_x", "uploadUrl": "https://u"}
        return {"datasetId": "ds_x", "rows": 8}

    monkeypatch.setattr(platform, "_call", fake_call)
    rng = random.Random(2)
    rows, _ = _paired_rows(12, 4, 0.6, 0.6, rng)
    with pytest.warns(UserWarning, match=r"holdout has 12 tasks at k=4; proving a 5% gain"):
        platform.push_rows(rows, "hold-v1", purpose="holdout", api_key="k")
    rows, _ = _paired_rows(12, 4, 0.6, 0.6, rng)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        platform.push_rows(rows, "train-v1", purpose="train", api_key="k")


# ------------------------------------------------------------ #254


def _prior(rates, k=4):
    rows = []
    for t, rate in enumerate(rates):
        passes = round(rate * k)
        for i in range(k):
            rows.append(
                {
                    "scenario_id": f"t{t}",
                    "rollout_index": i,
                    "prompt": f"p{t}",
                    "final_text": f"x{i}",
                    "reward": 1 if i < passes else 0,
                    "judge_status": "ok",
                    "policy_version": "m@r1",
                    "steps": [{"tool": "t", "arguments": {}}],
                }
            )
    return rows


def test_next_round_keeps_the_band_and_counts_the_rest():
    prior = _prior([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    plan = wai.next_round(prior)
    assert plan["kept"] == 3 and plan["dropped_solved"] == 2 and plan["dropped_unsolved"] == 1
    assert plan["unknown"] == 0 and plan["from_policy"] == ["m@r1"]
    kept = {r["scenario_id"] for r in plan["tasks"]}
    assert kept == {"t1", "t2", "t3"}
    assert all(r["calibration"]["pass_rate"] in (0.25, 0.5, 0.75) for r in plan["tasks"])
    assert len(plan["prompt_set_sha"]) == 16
    # a candidate the prior never saw is unknown and kept
    plan = wai.next_round(prior, tasks=["t1", "t5", "brand new prompt"])
    assert plan["kept"] == 1 and plan["dropped_solved"] == 1 and plan["unknown"] == 1
    assert plan["tasks"][-1] == {"prompt": "brand new prompt"}


def test_select_for_rl_prior_drops_solved_and_unsolved_tasks():
    prior = _prior([0.0, 0.25, 0.5, 0.75, 1.0])
    # this round's rollouts: every task mixed, so the band alone keeps all
    rows = _prior([0.5, 0.5, 0.5, 0.5, 0.5])
    _plain, plain_report = wai.select_for_rl(rows, target=100)
    assert plain_report["prior"] is None and plain_report["groups_selected"] == 5
    picked, report = wai.select_for_rl(rows, target=100, prior=prior)
    assert report["prior"]["dropped_solved"] == 1 and report["prior"]["dropped_unsolved"] == 1
    assert report["prior"]["rows_dropped"] == 8 and report["groups_selected"] == 3
    assert {r["scenario_id"] for r in picked} == {"t1", "t2", "t3"}
