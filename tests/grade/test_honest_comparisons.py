"""#392: a saturated baseline must not size a holdout (#375 shipped on main in 0.81).


#392: ``holdout_size(effect, before=rows)`` on a baseline every task passes
returned ``n_tasks=2, task_std=0.0`` with no warning: the binomial model at
``p = 1`` and the sizing formula's floor, not evidence. A saturated
baseline now answers with the model at the default base, says ``saturated``,
and warns with the ceiling and the fix.
"""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations import defaults
from whileai.simulations.score.stats import MIN_HOLDOUT_TASKS, holdout_size

# ------------------------------------------------------------------ #392


def _graded(rate: float, n_tasks: int = 10, k: int = 6) -> list[dict]:
    rows = []
    for t in range(n_tasks):
        passes = round(rate * k)
        for i in range(k):
            rows.append(
                {
                    "scenario_id": f"q{t}",
                    "prompt": f"q {t}",
                    "reward": 1.0 if i < passes else 0.0,
                    "rollout_index": i,
                }
            )
    return rows


def test_the_issue_repro_no_longer_answers_two():
    rows = [
        {"scenario_id": f"q{t}", "prompt": f"q {t}", "reward": 1.0, "rollout_index": k}
        for t in range(10)
        for k in range(6)
    ]
    hs = wai.holdout_size(0.05, before=rows)
    assert hs["saturated"] is True
    assert hs["base"] == 1.0 and hs["k"] == 6
    assert hs["sd_source"] == "model"
    # the model's answer at the default base, the same as with no rows
    assert hs["n_tasks"] == wai.holdout_size(0.05, base=defaults.BASE_PASS_RATE, k=6)["n_tasks"]
    assert hs["n_tasks"] > MIN_HOLDOUT_TASKS and hs["task_std"] > 0 and hs["half_width"] > 0
    [line] = hs["warnings"]
    assert line.startswith("CEILING:")
    assert "pass 1.00 of tasks, at or above the ceiling 0.90" in line
    assert f"({MIN_HOLDOUT_TASKS} tasks), which is the model collapsing, not evidence" in line
    assert f"default base {defaults.BASE_PASS_RATE:.2f} with these rows' k=6" in line
    assert "harder situations" in line and "20%-80% difficulty band" in line
    assert "hard_share" in line and "Lambert 2025, chapter Reasoning" in line


def test_both_arms_saturated_measure_a_zero_sd_and_get_the_same_answer():
    before = _graded(1.0)
    after = _graded(1.0)
    hs = holdout_size(0.05, before=before, after=after)
    assert hs["saturated"] is True and hs["sd_source"] == "model"
    assert hs["n_tasks"] == wai.holdout_size(0.05, base=defaults.BASE_PASS_RATE, k=6)["n_tasks"]
    assert hs["n_paired"] is None
    assert "task_std 0.000 measured" in " ".join(hs["notes"])
    assert hs["warnings"] and "at or above the ceiling" in hs["warnings"][0]


def test_a_floor_baseline_with_zero_measured_sd_is_also_saturated():
    """Every task fails on both arms: the base is 0, the measured paired sd
    is 0, and the answer would be the floor again."""
    hs = holdout_size(0.05, before=_graded(0.0), after=_graded(0.0))
    assert hs["saturated"] is True
    [line] = hs["warnings"]
    assert line.startswith("DEGENERATE: the paired difference is the same on every one of the 10")
    assert "(task_std 0.000 measured is 0) at a base of 0.00" in line
    assert hs["n_tasks"] > MIN_HOLDOUT_TASKS


def test_a_baseline_inside_the_band_is_unchanged():
    before = _graded(0.5)
    hs = holdout_size(0.05, before=before)
    assert hs["saturated"] is False and hs["warnings"] == []
    assert hs["base"] == 0.5 and hs["sd_source"] == "model"
    assert hs["n_tasks"] == holdout_size(0.05, base=0.5, k=6)["n_tasks"]
    after = _graded(0.5)
    for row in after:  # a gain that varies by task, so the paired sd is real
        if row["scenario_id"] in {"q0", "q1", "q2"} and row["reward"] == 0.0:
            row["reward"] = 1.0
    measured = holdout_size(0.05, before=before, after=after)
    assert measured["saturated"] is False and measured["sd_source"] == "rows"
    assert measured["task_std"] > 0 and measured["warnings"] == []


def test_the_ceiling_is_a_knob():
    rows = _graded(0.9, k=10)
    assert holdout_size(0.05, before=rows)["saturated"] is True
    relaxed = holdout_size(0.05, before=rows, ceiling_pass_rate=0.95)
    assert relaxed["saturated"] is False and relaxed["base"] == pytest.approx(0.9)
    # a given task_std is the caller's measurement; the rows do not override it
    given = holdout_size(0.05, before=_graded(1.0), task_std=0.38)
    assert given["saturated"] is False and given["sd_source"] == "given"
