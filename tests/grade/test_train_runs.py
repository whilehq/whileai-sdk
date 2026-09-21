"""#616: ``eval_variance``'s band is the band ``compare`` applies. #356: a
delta between two trained models carries a between-seed term, and one
training seed per arm is never "moved"."""

from __future__ import annotations

import math

import pytest

from whileai.simulations.defaults import MIN_TRAIN_SEEDS
from whileai.simulations.score.delta import (
    UNRESOLVED_LINE,
    delta_report,
    format_delta_report,
    headline_word,
)
from whileai.simulations.score.stats import _t_quantile, eval_variance, noise_band

K = 4


def _rows(rates: dict[str, float]) -> list[dict]:
    """``K`` rollouts per task; a task at rate ``p`` passes ``round(p * K)``."""
    out = []
    for task, p in rates.items():
        passes = round(p * K)
        for i in range(K):
            out.append(
                {
                    "task_id": task,
                    "reward": 1.0 if i < passes else 0.0,
                    "messages": [],
                    "markers": {},
                }
            )
    return out


def _flat(rate: float, n: int = 40) -> list[dict]:
    return _rows({f"t{i}": rate for i in range(n)})


def test_eval_variance_band_is_the_band_compare_applies():
    base = {f"t{i}": 0.5 for i in range(10)}
    ev = eval_variance(_rows(base), _rows({**base, "t0": 0.75}), _rows({**base, "t1": 0.25}))
    assert ev["n_runs"] == 3 and ev["run_std"] == 0.025
    # the t quantile at df = runs - 1, not 1.96: 4.30 x sqrt(2) x run_std
    assert ev["noise_band_df"] == 2
    assert ev["noise_band"] == round(noise_band(0.025, df=2), 4) == 0.1521
    assert ev["noise_band"] == pytest.approx(_t_quantile(2) * math.sqrt(2) * 0.025, abs=5e-5)
    assert ev["noise_band"] > round(noise_band(0.025), 4)  # the old 1.96 band, 0.0693
    # and it is the number compare(run_std=, run_std_runs=) prints and applies
    rep = delta_report(
        _flat(0.25), _flat(0.5), target="pass_at_1", run_std=ev["run_std"], run_std_runs=3
    )
    assert rep["noise_band"] == pytest.approx(ev["noise_band"], abs=5e-5)
    assert f"a delta under {ev['noise_band']:.3f} is noise" in format_delta_report(rep)
    assert "t(df=2)=4.30" in rep["noise_rule"]


def test_two_seeds_per_arm_widen_the_interval_by_the_known_spread():
    # before seeds at 0.25 and 0.50, after seeds at 0.50 and 0.75: every
    # task moves the same, so the task interval has zero width and the
    # widened interval is the between-seed term alone
    b1, b2, a1, a2 = _flat(0.25), _flat(0.5), _flat(0.5), _flat(0.75)
    rep = delta_report(
        b1,
        a1,
        target="pass_at_1",
        run_std=0.01,
        run_std_runs=3,
        train_runs={"before": [b1, b2], "after": [a1, a2]},
        n_boot=200,
    )
    seed_std = math.sqrt(((0.25 - 0.375) ** 2 + (0.5 - 0.375) ** 2) / 1)  # sd of two seeds
    assert rep["train_runs"] == {"before": 2, "after": 2}
    assert rep["train_std"]["before"] == pytest.approx(seed_std)
    assert rep["train_std"]["after"] == pytest.approx(seed_std)
    assert rep["train_df"] == 2
    assert rep["train_delta"] == pytest.approx(0.625 - 0.375)
    half = _t_quantile(2) * math.sqrt(seed_std**2 / 2 + seed_std**2 / 2)
    lo, hi = rep["train_ci95"]
    assert lo == pytest.approx(0.25 - half) and hi == pytest.approx(0.25 + half)
    # this seed pair moved +0.25 and clears the floor, but across seeds the
    # interval covers zero: the change does not survive the seed spread
    assert rep["metrics"]["pass_at_1"]["delta"] == pytest.approx(0.25)
    assert rep["target_verdict"] == "no_change_detected" and rep["ok"]
    assert any("does not survive the seed spread" in w for w in rep["warnings"])
    text = format_delta_report(rep)
    assert "training seeds: 2 seeds before, 2 seeds after" in text
    assert f"between-seed std {seed_std:.3f} before, {seed_std:.3f} after" in text
    assert f"interval widened to {lo:+.3f}..{hi:+.3f}" in text
    assert "t(df=2)=4.30 x sqrt(seed_std_before^2/2 + seed_std_after^2/2)" in text


def test_two_seeds_that_agree_keep_moved():
    b1, a1 = _flat(0.25), _flat(0.5)
    rep = delta_report(
        b1,
        a1,
        target="pass_at_1",
        run_std=0.01,
        run_std_runs=3,
        train_runs={"before": [b1, _flat(0.25)], "after": [a1, _flat(0.5)]},
        n_boot=200,
    )
    assert rep["train_std"] == {"before": 0.0, "after": 0.0}
    assert rep["train_ci95"] == pytest.approx((0.25, 0.25))
    assert rep["target_verdict"] == "moved" and headline_word(rep) == "PASS"
    # an untrained before arm has no between-seed term and no seed count
    single = delta_report(
        b1, a1, target="pass_at_1", run_std=0.01, run_std_runs=3, train_runs=[a1, _flat(0.5)]
    )
    assert single["train_runs"] == {"before": None, "after": 2}
    assert single["train_std"]["before"] is None and single["train_df"] == 1
    assert single["target_verdict"] == "moved"
    assert "training seeds: untrained before, 2 seeds after" in format_delta_report(single)


def test_one_training_seed_per_arm_is_unresolved():
    assert MIN_TRAIN_SEEDS == 2
    b1, a1 = _flat(0.25), _flat(0.5)
    rep = delta_report(
        b1, a1, target="pass_at_1", run_std=0.01, run_std_runs=3, train_runs=[a1], n_boot=200
    )
    assert rep["target_verdict"] == rep["headline_verdict"] == "unresolved"
    assert rep["ok"] and headline_word(rep) == "UNRESOLVED (one training seed per arm)"
    assert rep["train_runs"] == {"before": None, "after": 1}
    assert rep["train_ci95"] is None and rep["train_df"] is None
    assert any(w.startswith("UNRESOLVED: " + UNRESOLVED_LINE) for w in rep["warnings"])
    text = format_delta_report(rep)
    # the interval and the floor lines still print
    assert "pass_at_1: unresolved (+0.250, 95% +0.250..+0.250, 40 paired tasks)" in text
    assert "eval noise: run_std 0.010, a delta under" in text
    assert f"training seeds: untrained before, 1 seed after; unresolved: {UNRESOLVED_LINE}" in text
    # one seed on either trained arm is enough to leave it unresolved
    both = delta_report(
        b1, a1, target="pass_at_1", train_runs={"before": [b1], "after": [a1, _flat(0.5)]}
    )
    assert both["target_verdict"] == "unresolved" and both["train_runs"] == {
        "before": 1,
        "after": 2,
    }
    # the headline without a target follows the same rule
    assert delta_report(b1, a1, train_runs=[a1])["headline_verdict"] == "unresolved"
    # a wrong-way delta on one seed is unresolved in words and still fails the gate
    wrong = delta_report(a1, b1, target="pass_at_1", run_std=0.01, run_std_runs=3, train_runs=[b1])
    assert wrong["target_verdict"] == "unresolved" and not wrong["ok"]
    assert headline_word(wrong) == "FAIL"


def test_train_runs_absent_says_nothing_about_seeds():
    rep = delta_report(_flat(0.25), _flat(0.5), target="pass_at_1", n_boot=200)
    assert rep["train_runs"] is None and rep["train_ci95"] is None
    assert rep["target_verdict"] == "moved_unreplicated"
    assert "training seeds" not in format_delta_report(rep)


def test_train_runs_names_the_shape_it_wants():
    b1, a1 = _flat(0.25), _flat(0.5)
    with pytest.raises(ValueError, match="'before' and 'after'"):
        delta_report(b1, a1, train_runs={"after": [a1]})
    with pytest.raises(ValueError, match="is empty"):
        delta_report(b1, a1, train_runs=[])
    with pytest.raises(ValueError, match="list of rows"):
        delta_report(b1, a1, train_runs=a1)
    with pytest.raises(ValueError, match="no trained arm"):
        delta_report(b1, a1, train_runs={"before": None, "after": None})
