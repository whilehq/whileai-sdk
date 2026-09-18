"""#303: the two arms of a delta_report at different k (rows per task) get a
warning that names both, and balance_rollouts=True trims them to the same k."""

from __future__ import annotations

import random

from whileai.simulations.score.delta import delta_report, format_delta_report


def _rows(n_tasks: int, k_per_task, p: float, rng: random.Random) -> list[dict]:
    rows = []
    for t in range(n_tasks):
        k = k_per_task(t)
        for i in range(k):
            rows.append(
                {
                    "scenario_id": f"t{t}",
                    "rollout_index": i,
                    "prompt": f"p{t}",
                    "final_text": f"x{i}",
                    "reward": 1 if rng.random() < p else 0,
                    "judge_status": "ok",
                }
            )
    return rows


def _arms():
    rng = random.Random(7)
    before = _rows(10, lambda t: 4, 0.5, rng)
    # the after arm lost rollouts on four tasks, the way a cold endpoint or
    # a leaky reply format loses them: k=4 on six tasks, k=2 on the rest
    after = _rows(10, lambda t: 4 if t < 6 else 2, 0.6, rng)
    return before, after


def test_different_k_on_the_two_sides_is_warned_next_to_the_sizing_line():
    before, after = _arms()
    report = delta_report(before, after, target="pass_at_1")
    assert report["config"]["before"]["k"] == 4 and report["config"]["after"]["k"] == 2
    note = next(w for w in report["warnings"] if w.startswith("before has k="))
    assert "k=4" in note and "k=2" in note
    assert "4 of 10 paired tasks on the after side have fewer than 4 rows" in note
    assert "precision issue, not a bias" in note and "rows lost for a reason" in note
    assert "re-running the after side" in note and "balance_rollouts=True only" in note
    assert report["balanced"] is None
    # the sizing line is the neighbour, still quoting the before side's k
    sizing = [w for w in report["warnings"] if "paired tasks at k=4" in w]
    assert sizing, report["warnings"]


def test_equal_k_does_not_warn():
    rng = random.Random(3)
    before = _rows(10, lambda t: 4, 0.5, rng)
    after = _rows(10, lambda t: 4, 0.6, rng)
    report = delta_report(before, after)
    assert not any(w.startswith("before has k=") for w in report["warnings"])


def test_balance_rollouts_trims_the_larger_side_to_the_same_k():
    before, after = _arms()
    report = delta_report(before, after, target="pass_at_1", balance_rollouts=True)
    assert report["balanced"] == {
        "rows_dropped": {"before": 8, "after": 0},
        "tasks_trimmed": 4,
    }
    assert report["config"]["before"]["k"] == report["config"]["after"]["k"] == 2
    assert not any(w.startswith("before has k=") for w in report["warnings"])
    assert any("dropped 8 before rows and 0 after rows on 4 tasks" in w for w in report["warnings"])
    assert "balanced: dropped 8 before rows and 0 after rows on 4 tasks" in format_delta_report(
        report
    )
    assert report["n_paired_tasks"] == 10
    # the inputs are not mutated
    assert len(before) == 40 and len(after) == 32


def test_balance_is_deterministic_by_seed():
    before, after = _arms()
    one = delta_report(before, after, balance_rollouts=True, seed=1)
    two = delta_report(before, after, balance_rollouts=True, seed=1)
    assert one["metrics"]["pass_at_1"] == two["metrics"]["pass_at_1"]
    other = delta_report(before, after, balance_rollouts=True, seed=2)
    # a different seed keeps different rows of the trimmed tasks, so the
    # before mean can move; the row counts cannot
    assert other["balanced"] == one["balanced"]
