"""Run-to-run noise and whether the verdict repeated, beside every verdict.

An external case study on 0.126 said what made a 0.46 -> 0.76 gain
believable was that "identical re-runs differed by only 0.01 to 0.02",
and asked for that next to every verdict so one lucky run never reads as
a win. ``delta_report`` now carries ``repeats`` and prints a
``repeated:`` line. Fixed rows, fixed seeds, no network.
"""

from __future__ import annotations

import pytest

from whileai.simulations.score.delta import delta_report, format_delta_report

K = 4
TASKS = 40
BOOT = 500


def _run(rate: float, run: int | None, lift: float = 0.0) -> list[dict]:
    """One eval run: ``TASKS`` tasks at ``rate`` with ``K`` rollouts each,
    the first four tasks lifted by ``lift``; ``run`` is the
    ``lineage.eval_run`` stamp (``None``: a single run with no lineage)."""
    out = []
    for t in range(TASKS):
        passes = round((rate + (lift if t < 4 else 0.0)) * K)  # four lifted tasks
        for i in range(K):
            row: dict = {
                "task_id": f"t{t}",
                "reward": 1.0 if i < passes else 0.0,
                "messages": [],
                "markers": {},
            }
            if run is not None:
                row["lineage"] = {"eval_run": run}
            out.append(row)
    return out


def _runs(rates: list[float], lifts: list[float]) -> list[dict]:
    return [r for i, (rate, lift) in enumerate(zip(rates, lifts)) for r in _run(rate, i, lift)]


def test_three_tight_repeats_show_the_spread_and_agree():
    before = _runs([0.5] * 3, [0.0, 0.25, 0.0])
    after = _runs([0.75] * 3, [0.0, 0.0, 0.25])
    rep = delta_report(before, after, target="pass_at_1", n_boot=BOOT, seed=0)
    r = rep["repeats"]
    assert r["n_runs"] == {"before": 3, "after": 3}
    assert r["pairing"] == "run i before vs run i after"
    assert r["run_tags"] == ["up", "up", "up"] and r["pooled_tag"] == "up"
    assert (r["agree"], r["n"], r["note"]) == (3, 3, "3/3 runs agree")
    assert r["spread"] == {"before": pytest.approx(0.025), "after": pytest.approx(0.025)}
    assert r["range"]["after"] == pytest.approx((0.75, 0.775))
    assert format_delta_report(rep) == "\n".join(
        [
            "pass_at_1: moved (+0.250, 95% +0.250..+0.250, 40 paired tasks)",
            "PASS",
            "eval noise: run_std 0.014, a delta under 0.033 is noise (t(df=4)=2.78 x run_std x "
            "sqrt(1/3 + 1/3); 3 eval runs before, 3 after)",
            "repeated: 3/3 runs agree on pass_at_1 (each run alone: up, up, up; pooled: up); "
            "run-to-run range 0.025 before (3 runs, 0.500..0.525), 0.025 after (3 runs, "
            "0.750..0.775)",
            "answered: 0.0% before, 0.0% after",
            "  pass_at_1                    0.508 -> 0.758  +0.250 [+0.250..+0.250]  up  "
            "(40 paired)  noise<0.033",
        ]
    )


def test_repeats_that_disagree_say_so():
    before = _runs([0.5] * 3, [0.0] * 3)
    after = _runs([0.75, 0.5, 0.25], [0.0] * 3)
    rep = delta_report(before, after, target="pass_at_1", n_boot=BOOT, seed=0)
    r = rep["repeats"]
    assert r["run_tags"] == ["up", "flat", "DOWN"] and r["pooled_tag"] == "flat"
    assert (r["agree"], r["note"]) == (1, "1/3 runs agree")
    assert r["spread"]["after"] == pytest.approx(0.5)
    line = next(x for x in str(rep).splitlines() if x.startswith("repeated:"))
    assert line == (
        "repeated: 1/3 runs agree on pass_at_1 (each run alone: up, flat, DOWN; pooled: flat); "
        "run-to-run range 0.000 before (3 runs, 0.500..0.500), 0.500 after (3 runs, "
        "0.250..0.750)"
    )


def test_a_pooled_gain_two_runs_reach_is_not_three_of_three():
    before = _runs([0.5] * 3, [0.0] * 3)
    after = _runs([0.75, 0.75, 0.5], [0.0] * 3)
    rep = delta_report(before, after, target="pass_at_1", n_boot=BOOT, seed=0)
    assert rep["repeats"]["run_tags"] == ["up", "up", "flat"]
    assert rep["repeats"]["note"] == "1/3 runs agree"  # the pooled delta is inside the band


def test_one_run_says_noise_unknown():
    rep = delta_report(_run(0.5, None), _run(0.75, None), target="pass_at_1", n_boot=BOOT, seed=0)
    r = rep["repeats"]
    assert r["n_runs"] == {"before": 1, "after": 1}
    assert (r["agree"], r["n"], r["note"]) == (None, 1, "1 run, noise unknown")
    assert r["spread"] == {"before": None, "after": None}
    assert format_delta_report(rep) == "\n".join(
        [
            "pass_at_1: moved_unreplicated (+0.250, 95% +0.250..+0.250, 40 paired tasks)",
            "INCONCLUSIVE (1 eval run a side, rerun to confirm)",
            "repeated: 1 run, noise unknown (simulate(tasks=..., runs=3) shows whether the "
            "verdict repeats)",
            "answered: 0.0% before, 0.0% after",
            "  pass_at_1                    0.500 -> 0.750  +0.250 [+0.250..+0.250]  up  "
            "(40 paired)",
            "! One eval run on each side, so this could be noise. Run each side three times "
            "with simulate(tasks=..., runs=3) and the report will say.",
        ]
    )


def test_no_target_prints_the_pooled_interval_on_the_line():
    rep = delta_report(_run(0.5, None), _run(0.75, None), n_boot=BOOT, seed=0)
    assert "repeated: 1 run, noise unknown (pass_at_1 +0.250, 95% +0.250..+0.250; " in str(rep)


def test_one_run_before_three_after_compares_each_after_run():
    before = _run(0.5, None)
    after = _runs([0.75] * 3, [0.0, 0.0, 0.25])
    rep = delta_report(before, after, target="pass_at_1", n_boot=BOOT, seed=0)
    r = rep["repeats"]
    assert r["pairing"] == "each after run vs all before runs"
    assert r["n"] == 3 and r["note"] == "3/3 runs agree"
    assert r["spread"]["before"] is None
    assert "run-to-run range 1 run before, 0.025 after (3 runs, 0.750..0.775)" in str(rep)


def test_training_seeds_count_as_repeats():
    seeds_b = [_run(0.5, None, lift) for lift in (0.0, 0.25, 0.0)]
    seeds_a = [_run(0.75, None, lift) for lift in (0.0, 0.0, 0.25)]
    rep = delta_report(
        [r for s in seeds_b for r in s],
        [r for s in seeds_a for r in s],
        target="pass_at_1",
        train_runs={"before": seeds_b, "after": seeds_a},
        n_boot=BOOT,
        seed=0,
    )
    assert rep["repeats"]["n_runs"] == {"before": 3, "after": 3}
    assert rep["repeats"]["note"] == "3/3 runs agree"


def test_same_seed_same_report():
    before = _runs([0.5] * 3, [0.0, 0.25, 0.0])
    after = _runs([0.75] * 3, [0.0, 0.0, 0.25])
    one = delta_report(before, after, n_boot=BOOT, seed=7)
    two = delta_report(before, after, n_boot=BOOT, seed=7)
    assert one["repeats"] == two["repeats"] and str(one) == str(two)


def test_one_run_with_a_given_floor_names_where_the_noise_came_from():
    rep = delta_report(
        _run(0.5, None),
        _run(0.75, None),
        target="pass_at_1",
        run_std=0.02,
        run_std_runs=3,
        n_boot=BOOT,
        seed=0,
    )
    assert rep["repeats"]["note"] == "1 run, noise from the given run_std"
    assert "repeated: 1 run, noise from the given run_std (simulate(" in str(rep)
