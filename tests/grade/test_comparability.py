"""delta_report names the ways two arms differ in more than the weights
(who was graded, who played the user), the re-run band on a delta is the
band on a difference of two draws, and the multiplicity note is an upper
bound (rlhf-book ch. 16, appendix C)."""

from __future__ import annotations

import math
import random

from whileai.simulations.score.delta import (
    GRADED_SHARE_FAIL_GAP,
    GRADED_SHARE_WARN_GAP,
    delta_report,
    format_delta_report,
)
from whileai.simulations.score.passat import run_config
from whileai.simulations.score.stats import NOISE_SIGMAS, eval_variance


def _rows(
    pass_rates: dict[str, float],
    k: int = 4,
    *,
    ungraded_every: int | None = None,
    partial: bool = False,
    stamp: dict | None = None,
) -> list[dict]:
    rows = []
    for t, (task, p) in enumerate(pass_rates.items()):
        passes = round(p * k)
        for i in range(k):
            reward: float | None = 1.0 if i < passes else 0.0
            if partial:
                reward = 0.75 if i < passes else 0.25
            if ungraded_every and (t * k + i) % ungraded_every == 0:
                reward = None
            row = {
                "prompt": task,
                "reward": reward,
                "markers": {"h": 1.0 if i < passes else 0.0},
                **(stamp or {}),
            }
            rows.append(row)
    return rows


def test_partial_rewards_are_graded_rows():
    # a rubric score is a verdict the judge reached; pass@1 (binary) does
    # not count it, but the row did not leave the denominator
    before = _rows({f"t{i}": 0.5 for i in range(30)}, partial=True)
    after = _rows({f"t{i}": 0.5 for i in range(30)}, partial=True)
    assert run_config(before)["graded_share"] == 1.0
    report = delta_report(before, after, n_boot=100)
    assert report["ok"] is True
    assert not any("NOT COMPARABLE" in w or "were graded against" in w for w in report["warnings"])
    # a missing or boolean reward is the ungraded case
    assert run_config([{"reward": None}, {"reward": True}, {"reward": "x"}])["graded_share"] == 0.0
    assert run_config([{"reward": 0.75}, {"reward": None}])["graded_share"] == 0.5


def test_graded_share_gap_warns_at_two_points_and_fails_at_ten():
    base = {f"t{i}": 0.5 for i in range(34)}  # 136 rows
    before = _rows(base)
    # 3 rows ungraded on one side: 2.2 points, a warning that bounds the bias
    after = _rows(base)
    for row in after[:3]:
        row["reward"] = None
    report = delta_report(before, after, n_boot=100)
    gap = 1.0 - run_config(after)["graded_share"]
    assert GRADED_SHARE_WARN_GAP < gap < GRADED_SHARE_FAIL_GAP
    assert report["ok"] is True
    note = [w for w in report["warnings"] if "were graded against" in w]
    assert len(note) == 1 and "NOT COMPARABLE" not in note[0]
    assert "up to 2.2%" in note[0] and "re-grade" in note[0]
    # 2 rows is 1.5 points: nothing
    after = _rows(base)
    for row in after[:2]:
        row["reward"] = None
    quiet = delta_report(before, after, n_boot=100)
    assert not any("were graded against" in w for w in quiet["warnings"])
    # 16 rows is 11.8 points, the lane that motivated the check: fails
    after = _rows(base)
    for row in after[:16]:
        row["reward"] = None
    loud = delta_report(before, after, n_boot=100)
    assert loud["ok"] is False
    assert any(w.startswith("ARMS NOT COMPARABLE: 100.0% of before rows") for w in loud["warnings"])
    assert "FAIL" in format_delta_report(loud)


def test_noise_band_is_the_band_on_a_difference_everywhere():
    assert 2 * math.sqrt(2) == NOISE_SIGMAS
    base = {f"t{i}": 0.5 for i in range(10)}
    variance = eval_variance(_rows(base), _rows({**base, "t0": 0.75}), _rows({**base, "t1": 0.25}))
    assert variance["run_std"] == 0.025
    assert variance["noise_band"] == round(NOISE_SIGMAS * 0.025, 4) == 0.0707
    # a delta of 0.06 is over 2 x run_std (0.05) and under the band (0.0707):
    # a re-run draw, not a change
    before = _rows({f"t{i}": 0.25 for i in range(50)})
    after = _rows({f"t{i}": 0.5 if i < 12 else 0.25 for i in range(50)})  # +0.25 on 12: +0.06
    delta = delta_report(before, after, target="pass_at_1", run_std=0.025, n_boot=200)
    assert 0.05 < abs(delta["metrics"]["pass_at_1"]["delta"]) < 0.0707
    assert delta["target_verdict"] == "within_eval_noise"
    assert any("2.83 x run_std = 0.071" in w for w in delta["warnings"])
    assert "a delta under 0.071 is noise" in format_delta_report(delta)
    # and outside the band it is a change
    assert (
        delta_report(before, after, target="pass_at_1", run_std=0.015, n_boot=200)["target_verdict"]
        == "moved"
    )


def test_null_deltas_clear_the_band_about_five_percent_of_the_time():
    # the reason for sqrt(2): with the old 2 x run_std band about 15% of
    # pure-noise deltas read as real; with the band on a difference, about 5%
    rng = random.Random(0)
    run_std = 0.02
    old = new = 0
    n = 4000
    for _ in range(n):
        delta = rng.gauss(0, run_std) - rng.gauss(0, run_std)
        old += abs(delta) >= 2 * run_std
        new += abs(delta) >= NOISE_SIGMAS * run_std
    assert 0.12 < old / n < 0.19
    assert 0.03 < new / n < 0.07


def test_family_error_is_labelled_an_upper_bound():
    base = {f"t{i}": 0.25 for i in range(40)}
    before = _rows(base)
    after = _rows({f"t{i}": 0.75 for i in range(40)})
    for rows in (before, after):
        for row in rows:
            row["markers"].update({"a": 1.0, "b": 1.0, "c": 1.0, "d": row["markers"]["h"]})
    report = delta_report(before, after, n_boot=100)
    assert report["n_metrics"] == 6 and report["family_error"] == round(1 - 0.95**6, 4)
    note = [w for w in report["warnings"] if "tested at 95%" in w]
    assert len(note) == 1 and "up to about a 26%" in note[0] and "upper bound" in note[0]


def test_user_and_writer_model_that_moved_with_the_arm_fail_the_report():
    base = {f"t{i}": 0.5 for i in range(20)}
    before = _rows(base, stamp={"user_model": "sim-a", "writer_model": "w"})
    after = _rows(base, stamp={"user_model": "sim-b", "writer_model": "w"})
    report = delta_report(before, after, n_boot=100)
    assert report["ok"] is False
    assert report["config"]["before"]["user_model"] == "sim-a"
    assert any(
        w.startswith("ARMS NOT COMPARABLE: user_model was 'sim-a' before and 'sim-b' after")
        for w in report["warnings"]
    )
    same = delta_report(before, _rows(base, stamp={"user_model": "sim-a"}), n_boot=100)
    assert same["ok"] is True and not any("NOT COMPARABLE" in w for w in same["warnings"])
