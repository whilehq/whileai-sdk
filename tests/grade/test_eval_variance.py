"""Evaluation re-run variance (rlhf-book ch. 16) and the noise band on
delta_report."""

from __future__ import annotations

import random

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
    assert report["noise_band"] == 0.0693 and report["run_std_points"] == 2.5  # 1.96*sqrt(2)*0.025
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


def _issue_300_rows(seed: int, applicable: int = 8) -> list[dict]:
    """The generator from #300: 30 tasks x 4 rollouts, pass rate 0.6, and a
    policy marker that applies to ``applicable`` of the tasks at rate 0.3.
    Every call is the same model; two seeds are a model against itself."""
    rng = random.Random(seed)
    rows = []
    for t in range(30):
        for k in range(4):
            row = {
                "prompt": f"task {t}",
                "task_key": f"t{t}",
                "rollout_index": k,
                "reward": 1.0 if rng.random() < 0.60 else 0.0,
                "markers": {},
            }
            if t < applicable:
                row["markers"]["policy_marker"] = 1.0 if rng.random() < 0.30 else 0.0
            rows.append(row)
    return rows


def test_eval_variance_gives_each_marker_its_own_floor_and_the_scalar_matches():
    variance = eval_variance(_issue_300_rows(11), _issue_300_rows(22), _issue_300_rows(33))
    floors = variance["run_std_by_metric"]
    assert set(floors) == {"pass_at_1", "marker:policy_marker"}
    # the scalar and the mapping are the same number for the same metric
    assert variance["run_std"] == floors["pass_at_1"] == 0.0293
    # a marker on 8 of 30 tasks is several times noisier than pass@1
    assert floors["marker:policy_marker"] == 0.0786
    assert floors["marker:policy_marker"] > 2.5 * floors["pass_at_1"]
    assert (
        eval_variance(_issue_300_rows(1), _issue_300_rows(2))["run_std_by_metric"][
            "marker:policy_marker"
        ]
        == eval_variance(_issue_300_rows(1), _issue_300_rows(2), metric="marker:policy_marker")[
            "run_std"
        ]
    )


@pytest.mark.parametrize("seeds", [(108, 109), (110, 113), (119, 121)])
def test_a_model_against_itself_is_within_its_marker_noise_not_slipped(seeds):
    # #300: judged against pass@1's floor a re-run draw of the marker reads as
    # a regression; against the marker's own floor it is noise.
    floors = eval_variance(_issue_300_rows(11), _issue_300_rows(22), _issue_300_rows(33))
    before, after = (_issue_300_rows(s) for s in seeds)
    old = delta_report(before, after, run_std=floors["run_std"], n_boot=300)
    assert "marker:policy_marker" in old["slipped"]
    assert old["metrics"]["marker:policy_marker"]["within_noise"] is False
    assert any("marker:policy_marker dropped" in w for w in old["warnings"])

    new = delta_report(before, after, run_std=floors["run_std_by_metric"], n_boot=300)
    marker = new["metrics"]["marker:policy_marker"]
    assert marker["within_noise"] is True and "marker:policy_marker" in new["within_noise"]
    assert new["slipped"] == [] and new["regressions"] == [] and new["ok"] is True
    assert marker["run_std"] == 0.0786 and new["metrics"]["pass_at_1"]["run_std"] == 0.0293
    assert new["run_std"] == 0.0293 and new["run_std_source"] == "given"
    assert new["run_std_by_metric"] == {"pass_at_1": 0.0293, "marker:policy_marker": 0.0786}
    assert not any("dropped" in w for w in new["warnings"])
    # the guard reads the same floor
    guarded = delta_report(
        before, after, must_not_regress=["policy_marker"], run_std=floors["run_std_by_metric"]
    )
    assert guarded["regressions"] == [] and guarded["ok"] is True
    text = format_delta_report(new)
    assert (
        "eval noise: run_std 0.029, a delta under 0.081 is noise "
        "(1.96 x run_std x sqrt(1/1 + 1/1); run_std given, per metric"
    ) in text
    # each metric's own band: 1.96 x sqrt(2) x its floor
    assert "noise<0.218" in text and "noise<0.081" in text


def test_a_scalar_run_std_still_applies_one_floor_to_every_metric():
    before, after = _issue_300_rows(108), _issue_300_rows(109)
    report = delta_report(before, after, run_std=0.1, n_boot=100)
    assert report["run_std"] == 0.1 and report["replicated"] is True
    assert report["run_std_by_metric"] == {"pass_at_1": 0.1, "marker:policy_marker": 0.1}
    assert report["within_noise"] == ["pass_at_1", "marker:policy_marker"]
    assert "per metric" not in format_delta_report(report)


def test_a_missing_or_none_floor_is_said_not_borrowed():
    before = _rows({f"t{i}": 0.5 for i in range(8)})
    after = _rows({f"t{i}": 0.5 for i in range(8)})
    for run_std in ({"pass_at_1": 0.2}, {"pass_at_1": 0.2, "honest": None}):
        report = delta_report(before, after, run_std=run_std, n_boot=100)
        marker = report["metrics"]["marker:honest"]
        assert marker["run_std"] is None and marker["within_noise"] is False
        assert marker["noise_note"] == "no_replicate_floor"
        assert "noise_note" not in report["metrics"]["pass_at_1"]
        assert report["run_std_by_metric"] == {"pass_at_1": 0.2, "marker:honest": None}
        # the token stays on the metric; the warning is plain English
        note = [w for w in report["warnings"] if "no re-run floor for marker:honest" in w]
        assert len(note) == 1 and "no_replicate_floor" not in note[0]
        assert "run_std_by_metric" in note[0]
        assert "marker:honest" in format_delta_report(report).split("no_replicate_floor")[0]
    # eval_variance under two runs hands back None per metric; the same note
    single = eval_variance(_rows({"t0": 0.5}, run_id="x"))["run_std_by_metric"]
    assert single == {"pass_at_1": None, "marker:honest": None}
    report = delta_report(before, after, target="honest", run_std=single, n_boot=100)
    assert report["replicated"] is False and report["run_std"] is None
    assert report["metrics"]["marker:honest"]["noise_note"] == "no_replicate_floor"
    assert not any(w.startswith("One eval run") for w in report["warnings"])
    assert any("no re-run floor for" in w and "marker:honest" in w for w in report["warnings"])
    assert "no run_std for the headline metric" in format_delta_report(report)
    # no run_std at all: single-run honesty, no note (#240)
    bare = delta_report(before, after, n_boot=100)
    assert bare["replicated"] is False and "noise_note" not in bare["metrics"]["marker:honest"]
    assert bare["run_std_by_metric"] == {"pass_at_1": None, "marker:honest": None}


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
