"""delta_report names the ways two arms differ in more than the weights
(who was graded, who played the user) under one prefix and one key, the
re-run band is ``noise_band`` everywhere (1.96 or the t quantile, times
``run_std * sqrt(1/n_a + 1/n_b)``), and the multiplicity note is an upper
bound (rlhf-book ch. 16, appendix C)."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from whileai.simulations.score.delta import delta_report, format_delta_report
from whileai.simulations.score.passat import run_config
from whileai.simulations.score.stats import _t975, eval_variance, noise_band


def _rows(
    pass_rates: dict[str, float],
    k: int = 4,
    *,
    partial: bool = False,
    stamp: dict | None = None,
    eval_run: int | None = None,
) -> list[dict]:
    rows = []
    for task, p in pass_rates.items():
        passes = round(p * k)
        for i in range(k):
            reward: float | None = 1.0 if i < passes else 0.0
            if partial:
                reward = 0.75 if i < passes else 0.25
            row = {
                "prompt": task,
                "reward": reward,
                "markers": {"h": 1.0 if i < passes else 0.0},
                **(stamp or {}),
            }
            if eval_run is not None:
                row["lineage"] = {"eval_run": eval_run}
            rows.append(row)
    return rows


def _drop(rows: list[dict], *, reward: float, share: float) -> None:
    """Ungrade a share of the rows that scored ``reward`` (selection, not
    noise: the judge lost the failures, or the passes)."""
    hits = [r for r in rows if r["reward"] == reward]
    for r in hits[: round(share * len(rows))]:
        r["reward"] = None


def test_partial_rewards_are_graded_rows():
    # a rubric score is a verdict the judge reached; pass@1 (binary) does
    # not count it, but the row did not leave the denominator
    before = _rows({f"t{i}": 0.5 for i in range(30)}, partial=True)
    after = _rows({f"t{i}": 0.5 for i in range(30)}, partial=True)
    assert run_config(before)["graded_share"] == 1.0
    report = delta_report(before, after, n_boot=100)
    assert report["ok"] is True and report["not_comparable"] == []
    assert not any("NOT COMPARABLE" in w or "carry a verdict" in w for w in report["warnings"])
    # a missing or boolean reward is the ungraded case
    assert run_config([{"reward": None}, {"reward": True}, {"reward": "x"}])["graded_share"] == 0.0
    assert run_config([{"reward": 0.75}, {"reward": None}])["graded_share"] == 0.5


def test_graded_share_bound_is_per_side_and_adds_so_a_zero_gap_can_still_fail():
    # the audit's case: before lost 10% of its rows and they were failures,
    # after lost 10% and they were passes. The shares match to the digit,
    # so a gap rule sees nothing; the survivors' rates moved 0.11 apart on
    # selection alone, and the bound (0.1/0.9 per side, added) says so
    base = {f"t{i}": 0.5 for i in range(40)}  # 160 rows
    before, after = _rows(base), _rows(base)
    _drop(before, reward=0.0, share=0.10)
    _drop(after, reward=1.0, share=0.10)
    cfg_a, cfg_b = run_config(before), run_config(after)
    assert cfg_a["graded_share"] == cfg_b["graded_share"] == 0.9
    report = delta_report(before, after, target="pass_at_1", n_boot=100)
    assert report["target_verdict"] == "moved_the_wrong_way"
    assert report["ok"] is False and report["not_comparable"] == ["graded_share"]
    note = [w for w in report["warnings"] if w.startswith("NOT COMPARABLE: 90.0% of before rows")]
    assert len(note) == 1
    assert "up to 22.2%" in note[0] and "covers the whole" in note[0] and "re-grade" in note[0]
    assert "FAIL" in format_delta_report(report)
    assert "graded: 90.0% before, 90.0% after (selection can move the delta up to 22.2%)" in (
        format_delta_report(report)
    )


def test_graded_share_bound_warns_over_the_band_and_fails_over_the_delta():
    base = {f"t{i}": 0.25 for i in range(34)}  # 136 rows
    before = _rows(base)
    after = _rows({f"t{i}": 0.75 for i in range(34)})  # a real +0.5
    for row in after[:3]:  # 2.2% dropped: bound 2.3 points
        row["reward"] = None
    # the bound is under the re-run band: nothing to say beyond the rates
    quiet = delta_report(before, after, run_std=0.05, n_boot=100)  # band 0.139
    assert quiet["ok"] is True and quiet["not_comparable"] == []
    assert not any("carry a verdict" in w for w in quiet["warnings"])
    # over the band, under the delta: a warning that states the bound
    warned = delta_report(before, after, run_std=0.005, n_boot=100)  # band 0.014
    assert warned["ok"] is True and warned["not_comparable"] == []
    note = [w for w in warned["warnings"] if "carry a verdict" in w]
    assert len(note) == 1 and "NOT COMPARABLE" not in note[0]
    assert "up to 2.3%" in note[0] and "more than the re-run band (0.014)" in note[0]
    # with no run_std the interval's half-width is the bar: 0.044 here,
    # over the 2.3-point bound, so nothing is said
    unreplicated = delta_report(before, after, n_boot=100)
    assert not any("carry a verdict" in w for w in unreplicated["warnings"])
    assert (
        unreplicated["metrics"]["pass_at_1"]["ci95"][1]
        - unreplicated["metrics"]["pass_at_1"]["ci95"][0]
    ) / 2 > 0.023
    # the lane that motivated the check: 11.8% of one side dropped, and the
    # delta it left is smaller than what selection can do
    flat = _rows(base)
    for row in flat[:16]:
        row["reward"] = None
    loud = delta_report(before, flat, run_std=0.01, n_boot=100)
    assert loud["ok"] is False and loud["not_comparable"] == ["graded_share"]
    assert any(w.startswith("NOT COMPARABLE: 100.0% of before rows") for w in loud["warnings"])


def test_noise_band_is_one_function_everywhere():
    # one run per side, run_std taken as the eval's spread: 1.96 x sqrt(2)
    assert noise_band(0.025) == pytest.approx(1.96 * math.sqrt(2) * 0.025)
    # three runs per side, run_std estimated from them: t at df=4, sqrt(2/3)
    assert _t975(4) == 2.776 and _t975(1) == 12.706
    assert noise_band(0.025, 3, 3, df=4) == pytest.approx(2.776 * 0.025 * math.sqrt(2 / 3))
    # past the table the expansion is within a thousandth of the quantile
    assert _t975(30) == pytest.approx(2.042, abs=1e-3)
    assert _t975(60) == pytest.approx(2.000, abs=2e-3)
    assert _t975(120) == pytest.approx(1.980, abs=2e-3)
    with pytest.raises(ValueError, match="run counts"):
        noise_band(0.1, 0, 1)
    base = {f"t{i}": 0.5 for i in range(10)}
    variance = eval_variance(_rows(base), _rows({**base, "t0": 0.75}), _rows({**base, "t1": 0.25}))
    assert variance["run_std"] == 0.025
    assert variance["noise_band"] == round(noise_band(0.025), 4) == 0.0693
    # a delta of 0.06 is over 2 x run_std (0.05) and under the band (0.069):
    # a re-run draw, not a change
    before = _rows({f"t{i}": 0.25 for i in range(50)})
    after = _rows({f"t{i}": 0.5 if i < 12 else 0.25 for i in range(50)})  # +0.25 on 12: +0.06
    delta = delta_report(before, after, target="pass_at_1", run_std=0.025, n_boot=200)
    assert 0.05 < abs(delta["metrics"]["pass_at_1"]["delta"]) < 0.0693
    assert delta["target_verdict"] == "within_eval_noise"
    assert delta["noise_band"] == pytest.approx(0.0693, abs=1e-4)
    assert delta["noise_rule"] == "1.96 x run_std x sqrt(1/1 + 1/1)"
    assert delta["metrics"]["pass_at_1"]["noise_band"] == delta["noise_band"]
    assert any("1.96 x run_std x sqrt(1/1 + 1/1) = 0.069" in w for w in delta["warnings"])
    assert "a delta under 0.069 is noise (1.96 x run_std x sqrt(1/1 + 1/1); run_std given)" in (
        format_delta_report(delta)
    )
    # and outside the band it is a change
    assert (
        delta_report(before, after, target="pass_at_1", run_std=0.015, n_boot=200)["target_verdict"]
        == "moved"
    )


def test_three_runs_per_side_use_the_t_band_on_the_pooled_std():
    rng = random.Random(3)

    def _runs(centre: float) -> list[dict]:
        return [
            row
            for run in range(3)
            for row in _rows(
                {f"t{i}": centre + rng.choice([-0.25, 0.0, 0.25]) for i in range(30)},
                eval_run=run,
            )
        ]

    before, after = _runs(0.5), _runs(0.5)
    report = delta_report(before, after, target="pass_at_1", n_boot=100)
    assert report["run_std_source"] == "eval_run" and report["eval_runs"] == {
        "before": 3,
        "after": 3,
    }
    assert report["noise_rule"] == "t(df=4)=2.78 x run_std x sqrt(1/3 + 1/3)"
    assert report["noise_band"] == pytest.approx(noise_band(report["run_std"], 3, 3, df=4))
    assert report["noise_band"] < 2.83 * report["run_std"]  # narrower than the flat band
    assert "t(df=4)=2.78 x run_std x sqrt(1/3 + 1/3); 3 eval runs before, 3 after" in (
        format_delta_report(report)
    )


def test_null_deltas_clear_the_band_about_five_percent_of_the_time():
    # pure noise on both paths: with a known run_std and one run per side,
    # and with three runs per side and run_std pooled from them (df=4).
    # The flat 2 x run_std band let about 15% through on the first path.
    rng = random.Random(0)
    sigma = 0.02
    n = 4000
    old = single = triple = 0
    for _ in range(n):
        delta = rng.gauss(0, sigma) - rng.gauss(0, sigma)
        old += abs(delta) >= 2 * sigma
        single += abs(delta) >= noise_band(sigma)
        a = [rng.gauss(0, sigma) for _ in range(3)]
        b = [rng.gauss(0, sigma) for _ in range(3)]
        pooled = math.sqrt((statistics.variance(a) + statistics.variance(b)) / 2)
        triple += abs(statistics.mean(b) - statistics.mean(a)) >= noise_band(pooled, 3, 3, df=4)
    assert 0.12 < old / n < 0.19
    assert 0.03 < single / n < 0.08
    assert 0.03 < triple / n < 0.08


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
    assert "family error: 6 metrics at 95%, up to 26% chance that one clears zero on luck" in (
        format_delta_report(report)
    )


def test_user_and_writer_model_that_moved_with_the_arm_fail_the_report():
    base = {f"t{i}": 0.5 for i in range(20)}
    before = _rows(base, stamp={"user_model": "sim-a", "writer_model": "w"})
    after = _rows(base, stamp={"user_model": "sim-b", "writer_model": "w"})
    report = delta_report(before, after, n_boot=100)
    assert report["ok"] is False and report["not_comparable"] == ["user_model"]
    assert report["config"]["before"]["user_model"] == "sim-a"
    assert any(
        w.startswith("NOT COMPARABLE: user_model was 'sim-a' before and 'sim-b' after")
        for w in report["warnings"]
    )
    same = delta_report(before, _rows(base, stamp={"user_model": "sim-a"}), n_boot=100)
    assert same["ok"] is True and same["not_comparable"] == []
    assert not any("NOT COMPARABLE" in w for w in same["warnings"])


def test_own_model_line_fires_only_for_one_served_name_under_two_policies():
    base = {f"t{i}": 0.5 for i in range(20)}

    def _arm(policy: str) -> list[dict]:
        return _rows(base, stamp={"user_model": "qwen", "policy_version": policy})

    # the default before/after: base and adapter named apart, the user on
    # the served model both times. Nothing to say.
    plain = delta_report(_arm("qwen-base@abc"), _arm("qwen-sft@abc"), n_boot=100)
    assert plain["config"]["before"]["policy_version"] == "qwen-base@abc"
    assert not any("agent's own served model" in w for w in plain["warnings"])
    # one served name, two policy stamps: if those were two sets of weights
    # the user moved with them, and the line says so
    shared = delta_report(_arm("qwen@abc"), _arm("qwen@def"), n_boot=100)
    note = [w for w in shared["warnings"] if "agent's own served model ('qwen')" in w]
    assert len(note) == 1 and "pin user_model=" in note[0]
    assert shared["ok"] is True and shared["not_comparable"] == []
    # identical stamps are the model-to-itself case, not this one
    same = delta_report(_arm("qwen@abc"), _arm("qwen@abc"), n_boot=100)
    assert not any("agent's own served model" in w for w in same["warnings"])
    assert any("compares a model to itself" in w for w in same["warnings"])
