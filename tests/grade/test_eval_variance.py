"""Evaluation re-run variance (rlhf-book ch. 16) and the noise band on
delta_report."""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations.score.delta import delta_report, format_delta_report
from whileai.simulations.score.stats import eval_variance


def _rows(pass_rates: dict[str, float], k: int = 4, run_id: str | None = None) -> list[dict]:
    rows = []
    for task, p in pass_rates.items():
        passes = round(p * k)
        for i in range(k):
            row = {
                "prompt": task,
                "reward": 1 if i < passes else 0,
                "final_text": "ok",
                "steps": [],
                "messages": [],
                "markers": {"honest": 1.0 if i < passes else 0.0},
            }
            if run_id:
                row["lineage"] = {"scoring_run_id": run_id, "source": "eval"}
            rows.append(row)
    return rows


def test_eval_variance_over_separate_runs():
    base = {f"t{i}": 0.5 for i in range(10)}
    run1 = _rows(base)
    run2 = _rows({**base, "t0": 0.75})
    run3 = _rows({**base, "t1": 0.25})
    report = eval_variance(run1, run2, run3)
    assert report["n_runs"] == 3 and report["metric"] == "pass_at_1"
    assert report["means"] == {"run_1": 0.5, "run_2": 0.525, "run_3": 0.475}
    assert report["mean"] == 0.5 and report["run_std"] == 0.025
    assert report["noise_band"] == 0.05 and report["run_std_points"] == 2.5
    assert report["stability"] == "high_variance"
    assert report["tasks_in_every_run"] == 10 and report["notes"] == []


def test_eval_variance_splits_one_list_by_scoring_run_id_and_by_key():
    base = {f"t{i}": 0.5 for i in range(8)}
    rows = _rows(base, run_id="score_a") + _rows(base, run_id="score_b")
    rows += _rows({"t0": 1.0})  # no run id: left out, noted
    report = eval_variance(rows)
    assert report["n_runs"] == 2 and set(report["means"]) == {"score_a", "score_b"}
    assert report["run_std"] == 0.0 and report["stability"] == "very_stable"
    assert any("no run id" in n for n in report["notes"])
    assert any("two is a difference" in n for n in report["notes"])

    for row in rows:
        row["seed"] = 1 if row.get("lineage", {}).get("scoring_run_id") == "score_a" else 2
    assert eval_variance(rows, by="seed")["n_runs"] == 2
    assert eval_variance(rows, metric="marker:honest")["metric"] == "marker:honest"


def test_eval_variance_with_one_run_has_no_std():
    report = eval_variance(_rows({"t0": 0.5, "t1": 0.5}))
    assert report["n_runs"] == 0 and report["run_std"] is None  # no run ids on the rows
    report = eval_variance(_rows({"t0": 0.5}, run_id="x"))
    assert report["n_runs"] == 1 and report["run_std"] is None and report["stability"] is None


def test_delta_report_refuses_a_verdict_inside_the_noise_band():
    before = _rows({f"t{i}": 0.25 for i in range(12)})
    after = _rows({f"t{i}": 0.5 for i in range(12)})
    loud = delta_report(before, after, target="pass_at_1")
    assert loud["target_verdict"] == "moved_unreplicated" and loud["within_noise"] == []
    assert loud["metrics"]["pass_at_1"]["within_noise"] is False
    # a 0.25 delta inside a 2 x 0.2 band is what re-running the eval does
    quiet = delta_report(before, after, target="pass_at_1", run_std=0.2)
    assert quiet["target_verdict"] == "within_eval_noise" and quiet["ok"] is True
    assert quiet["within_noise"] == ["pass_at_1", "marker:honest"]
    assert quiet["improved"] == [] and quiet["run_std"] == 0.2
    assert any("re-run band" in w for w in quiet["warnings"])
    text = format_delta_report(quiet)
    assert text.startswith("pass_at_1: within_eval_noise") and "noise" in text

    # a regression inside the band is not a regression either
    for r in before:
        r["markers"]["polite"] = 1.0
    for i, r in enumerate(after):
        r["markers"]["polite"] = 0.0 if i % 4 == 0 else 1.0
    strict = delta_report(before, after, target="pass_at_1", must_not_regress=["polite"])
    assert strict["regressions"] == ["marker:polite"] and strict["ok"] is False
    lenient = delta_report(
        before, after, target="pass_at_1", must_not_regress=["polite"], run_std=0.2
    )
    assert lenient["regressions"] == [] and lenient["ok"] is True
    assert "marker:polite" in lenient["within_noise"]


def test_per_metric_run_std_does_not_apply_pass_at_1_noise_to_markers():
    runs = []
    marker_rates = (0.25, 0.5, 0.75)
    for marker_rate in marker_rates:
        rows = _rows({f"t{i}": 0.5 for i in range(8)})
        for task_i in range(8):
            value = 1.0 if task_i < round(marker_rate * 8) else 0.0
            for row in rows[task_i * 4 : (task_i + 1) * 4]:
                row["markers"]["honest"] = value
        runs.append(rows)

    variance = eval_variance(*runs)
    floors = variance["run_std_by_metric"]
    assert floors["pass_at_1"] == 0.0
    assert floors["marker:honest"] > floors["pass_at_1"]

    before = runs[0]
    after = runs[1]
    report = delta_report(before, after, must_not_regress=["honest"], run_std=floors, n_boot=200)
    assert report["metrics"]["marker:honest"]["run_std"] == floors["marker:honest"]
    assert report["metrics"]["pass_at_1"]["run_std"] == floors["pass_at_1"]


def test_per_metric_run_std_does_not_fall_back_for_missing_marker_floor():
    before = _rows({f"t{i}": 0.5 for i in range(8)})
    after = _rows({f"t{i}": 0.5 for i in range(8)})
    report = delta_report(before, after, run_std={"pass_at_1": 0.2}, n_boot=100)
    marker = report["metrics"]["marker:honest"]
    assert marker["run_std"] is None
    assert marker["noise_note"] == "no_replicate_floor"


def test_public_surface():
    assert "eval_variance" in wai.__all__ and callable(wai.eval_variance)
    with pytest.raises(ValueError, match="at least one"):
        wai.eval_variance()


# --- #31: the wrong-shape argument must name the next action ---


def _one_run(n=6, run="r1"):
    return [
        {"task_id": f"t{i}", "reward": i % 2, "lineage": {"scoring_run_id": run}} for i in range(n)
    ]


def test_a_simulation_data_argument_names_the_attribute_to_pass():
    """``simulate()`` returns a SimulationData; Python's bare "object is not
    iterable" never said that ``.trajectories`` is one attribute away."""

    class FakeSimulationData:
        def __init__(self, rows):
            self.trajectories = rows

    data = FakeSimulationData(_one_run())
    with pytest.raises(TypeError) as exc:
        eval_variance(data, data, data)
    msg = str(exc.value)
    assert "argument 1" in msg and "FakeSimulationData" in msg
    assert ".trajectories" in msg
    # the message is runnable as written
    assert "eval_variance(a.trajectories, b.trajectories, c.trajectories)" in msg


def test_the_single_run_path_reports_the_same_way():
    class FakeSimulationData:
        def __init__(self, rows):
            self.trajectories = rows

    with pytest.raises(TypeError, match=r"\.trajectories"):
        eval_variance(FakeSimulationData(_one_run()))


def test_a_container_whose_rows_live_on_rows_is_named_correctly():
    class FakeScored:
        def __init__(self, rows):
            self.rows = rows

    with pytest.raises(TypeError, match=r"Pass its \.rows"):
        eval_variance(FakeScored(_one_run()))


def test_a_report_dict_is_told_to_pass_row_lists():
    with pytest.raises(TypeError, match="each argument is one re-run's rows"):
        eval_variance({"run_1": 0.5, "run_2": 0.6})


def test_an_argument_with_no_rows_anywhere_still_explains_the_shape():
    with pytest.raises(TypeError, match="which is not a list of rows"):
        eval_variance(42)


def test_an_iterable_that_is_not_a_list_still_works():
    """The guard must not narrow what already worked: any iterable of rows."""
    runs = [_one_run(run=f"r{i}") for i in range(3)]
    from_lists = eval_variance(*runs)
    from_iters = eval_variance(*(iter(r) for r in runs))
    assert from_iters["run_std"] == from_lists["run_std"]
    assert from_iters["means"] == from_lists["means"]


def test_non_dict_entries_are_still_skipped_not_fatal():
    runs = [[*_one_run(run=f"r{i}"), None, "junk"] for i in range(3)]
    assert eval_variance(*runs)["n_runs"] == 3
