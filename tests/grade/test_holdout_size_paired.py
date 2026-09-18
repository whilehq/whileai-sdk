"""#292: the binomial model under-sizes when the gain is concentrated on a
few tasks and over-sizes when tasks differ in difficulty, because it has
no covariance term. Measure the paired sd off both arms, or take it as
given, and say what the model assumes when it is used."""

from __future__ import annotations

import random

import pytest

from whileai.simulations.score.stats import holdout_size


def _arm(rates, k, rng=None):
    """Graded rows, one task per rate. Deterministic pass counts when
    ``rng`` is None (round(rate * k) passes), Bernoulli draws otherwise."""
    rows = []
    for t, rate in enumerate(rates):
        passes = round(rate * k)
        for i in range(k):
            hit = (rng.random() < rate) if rng is not None else (i < passes)
            rows.append(
                {
                    "scenario_id": f"t{t}",
                    "rollout_index": i,
                    "prompt": f"p{t}",
                    "final_text": f"x{i}",
                    "reward": 1 if hit else 0,
                    "judge_status": "ok",
                }
            )
    return rows


def test_the_model_path_prints_the_concentrated_number_beside_its_own():
    """The issue's lane: base 0 -> 0.127 at k=4. The model says 14; the
    measured sd (0.333, 19 of 150 tasks carrying the gain) says 54."""
    need = holdout_size(0.127, base=0.0, k=4)
    assert need["n_tasks"] == 14 and need["sd_source"] == "model"
    assert need["n_tasks_concentrated"] == 54
    assert need["task_std"] == pytest.approx(0.167, abs=0.002)
    joined = " ".join(need["notes"])
    assert "spread evenly" in joined and "independent draws" in joined
    assert "54 tasks" in joined and "before= and after=" in joined and "task_std=" in joined
    # the default answer did not move
    same = holdout_size(0.03, base=0.6, k=4)
    assert 900 <= same["n_tasks"] <= 1200 and same["n_tasks_concentrated"] > same["n_tasks"]


def test_a_measured_sd_replaces_the_model():
    given = holdout_size(0.127, task_std=0.333)
    assert given["n_tasks"] == 54 and given["sd_source"] == "given"
    assert given["task_std"] == 0.333 and given["n_tasks_concentrated"] is None
    assert "binomial model was not used" in " ".join(given["notes"])
    # the spread-out lane from the issue: model 0.294 says more than the measured 0.233
    assert (
        holdout_size(0.012, task_std=0.233)["n_tasks"]
        < holdout_size(0.012, base=0.771, k=4)["n_tasks"]
    )
    with pytest.raises(ValueError, match="above 0"):
        holdout_size(0.05, task_std=0)


def test_both_arms_measure_the_paired_sd_with_its_covariance():
    # concentrated: 150 tasks at 0, 19 of them go to 1 after (the issue's shape)
    before = _arm([0.0] * 150, 4)
    after = _arm([1.0] * 19 + [0.0] * 131, 4)
    model = holdout_size(0.127, base=0.0, k=4)
    measured = holdout_size(0.127, before=before, after=after)
    assert measured["sd_source"] == "rows" and measured["n_paired"] == 150
    assert measured["k"] == 4 and measured["base"] == 0.0
    assert 0.30 <= measured["task_std"] <= 0.36  # 0.333 on the lane
    assert measured["n_tasks"] >= 3 * model["n_tasks"]  # 54 vs 14
    assert measured["n_tasks_concentrated"] is None
    assert "no model" in " ".join(measured["notes"])

    # spread out: tasks differ in difficulty, the gain is uniform. Pairing
    # removes the between-task variance the independent-arms model keeps.
    rng = random.Random(11)
    rates = [0.2, 0.8] * 400  # between-task sd 0.3
    before = _arm(rates, 4, rng)
    after = _arm([r + 0.05 for r in rates], 4, rng)
    model = holdout_size(0.05, base=0.5, k=4)
    measured = holdout_size(0.05, before=before, after=after)
    assert measured["task_std"] < model["task_std"]
    assert 1.1 <= model["task_std"] / measured["task_std"] <= 1.5  # 1.25 here, the issue saw 1.30
    assert measured["n_tasks"] < model["n_tasks"]


def test_both_arms_need_shared_graded_tasks():
    before = _arm([0.5] * 10, 4)
    other = [dict(r, scenario_id=f"u{i}") for i, r in enumerate(_arm([0.5] * 10, 4))]
    with pytest.raises(ValueError, match="simulate\\(tasks=base\\)"):
        holdout_size(0.05, before=before, after=other)
    with pytest.raises(ValueError, match="needs before="):
        holdout_size(0.05, after=before)


def test_before_rows_alone_report_the_difficulty_spread_and_the_ratio():
    flat = holdout_size(0.05, before=_arm([0.5] * 20, 4))
    assert flat["base_spread"] == 0.0 and flat["sd_source"] == "model"
    assert not any("spread sd" in n for n in flat["notes"])
    # [0.2, 0.8]: p 0.5, Var(p_i) 0.09, so the model asks 1 / (1 - 0.36) = 1.56x
    mixed = holdout_size(0.05, before=_arm([0.2, 0.8] * 10, 5))
    assert mixed["base_spread"] == pytest.approx(0.308, abs=0.01)
    joined = " ".join(mixed["notes"])
    assert "asks for 1.56x the tasks pairing needs" in joined and "Var(p_i)" in joined
    assert mixed["n_tasks_concentrated"] is not None
    # [0.3, 0.7]: Var 0.04 -> 1 / (1 - 0.16) = 1.19x
    assert "1.19x" in " ".join(holdout_size(0.05, before=_arm([0.3, 0.7] * 10, 10))["notes"])
    # every task a sure pass or fail: the ratio has no finite value, say so
    sure = holdout_size(0.05, before=_arm([0.0, 1.0] * 10, 4))
    assert "pairing removes all of it" in " ".join(sure["notes"])
    # the old name still works and is the same argument
    assert holdout_size(0.05, rows=_arm([0.2, 0.8] * 10, 5)) == mixed


def test_the_result_has_the_same_keys_on_every_path():
    keys = {
        "n_tasks",
        "effect",
        "base",
        "k",
        "power",
        "alpha",
        "task_std",
        "sd_source",
        "half_width",
        "n_tasks_concentrated",
        "base_spread",
        "n_paired",
        "notes",
    }
    before = _arm([0.5] * 10, 4)
    model = holdout_size(0.05, base=0.6, k=4)
    given = holdout_size(0.05, task_std=0.38)
    measured = holdout_size(0.05, before=before, after=_arm([0.75] * 10, 4))
    for result in (model, given, measured):
        assert set(result) == keys and isinstance(result["notes"], list)
    assert model["n_paired"] is None and model["base_spread"] is None
    assert given["n_paired"] is None and given["n_tasks_concentrated"] is None
    assert measured["n_tasks_concentrated"] is None and measured["n_paired"] == 10
