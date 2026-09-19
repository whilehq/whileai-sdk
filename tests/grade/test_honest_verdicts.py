"""delta_report says ``moved`` only when the eval was repeated (Lambert
2025, chapter Evaluation and its evaluation-variance appendix), and
flags a ceiling the training cannot show a gain
over."""

from __future__ import annotations

import random

from whileai.simulations.score.delta import delta_report, format_delta_report
from whileai.simulations.training import TrainingRun, _holdout_without_rows


def _rows(pass_rates: dict[str, float], k: int = 4, eval_run: int | None = None) -> list[dict]:
    rows = []
    for task, p in pass_rates.items():
        passes = round(p * k)
        for i in range(k):
            row = {"prompt": task, "reward": 1 if i < passes else 0, "markers": {"h": 1.0}}
            if eval_run is not None:
                row["lineage"] = {"eval_run": eval_run, "scoring_run_id": "score_x"}
            rows.append(row)
    return rows


def _repeated(rates_of, runs: int = 3) -> list[dict]:
    return [row for run in range(runs) for row in _rows(rates_of(run), eval_run=run)]


def test_single_run_sides_move_unreplicated_with_the_fix_in_the_warning():
    before = _rows({f"t{i}": 0.25 for i in range(30)})
    after = _rows({f"t{i}": 0.5 for i in range(30)})
    report = delta_report(before, after, target="pass_at_1")
    assert report["target_verdict"] == "moved_unreplicated" and report["ok"] is True
    assert report["replicated"] is False and report["run_std"] is None
    assert report["eval_runs"] == {"before": 0, "after": 0}
    assert report["warnings"] == [
        "One eval run on each side, so this could be noise. Run each side three times with "
        "simulate(tasks=..., runs=3) and the report will say."
    ]
    text = format_delta_report(report)
    assert text.startswith("pass_at_1: moved_unreplicated") and "could be noise" in text
    # one side repeated is still unreplicated, and the warning names the side
    after3 = _repeated(lambda run: {f"t{i}": 0.5 for i in range(30)})
    one_side = delta_report(before, after3, target="pass_at_1")
    assert one_side["target_verdict"] == "moved_unreplicated"
    assert one_side["warnings"][0].startswith("One eval run on the before side")
    # a run_std given by hand counts as replicated, as before
    given = delta_report(before, after, target="pass_at_1", run_std=0.01)
    assert given["target_verdict"] == "moved" and given["run_std_source"] == "given"


def test_three_runs_per_side_compute_run_std_and_say_moved():
    rng = random.Random(1)
    before = _repeated(lambda run: {f"t{i}": rng.choice([0.25, 0.5]) for i in range(30)})
    after = _repeated(lambda run: {f"t{i}": rng.choice([0.5, 0.75]) for i in range(30)})
    report = delta_report(before, after, target="pass_at_1")
    assert report["target_verdict"] == "moved" and report["replicated"] is True
    assert report["run_std_source"] == "eval_run" and report["eval_runs"] == {
        "before": 3,
        "after": 3,
    }
    assert 0 < report["run_std"] < 0.05
    assert not any("could be noise" in w for w in report["warnings"])
    assert "eval noise: run_std" in format_delta_report(report)


def test_three_runs_per_side_inside_the_band_is_within_eval_noise():
    # the eval swings by 0.1 between runs; a 0.05 shift is inside that
    before = _repeated(lambda run: {f"t{i}": [0.25, 0.5, 0.25][run] for i in range(30)})
    after = _repeated(lambda run: {f"t{i}": [0.5, 0.25, 0.5][run] for i in range(30)})
    report = delta_report(before, after, target="pass_at_1")
    assert report["run_std_source"] == "eval_run" and report["run_std"] > 0.1
    assert report["target_verdict"] == "within_eval_noise" and report["ok"] is True
    assert "pass_at_1" in report["within_noise"]
    assert any("re-run band" in w for w in report["warnings"])


def test_two_runs_per_side_is_used_but_called_rough():
    before = _repeated(lambda run: {f"t{i}": [0.25, 0.3][run] for i in range(30)}, runs=2)
    after = _repeated(lambda run: {f"t{i}": [0.75, 0.7][run] for i in range(30)}, runs=2)
    report = delta_report(before, after, target="pass_at_1")
    assert report["target_verdict"] == "moved" and report["eval_runs"] == {"before": 2, "after": 2}
    assert any("Two eval runs" in w for w in report["warnings"])


def test_ceiling_fires_on_a_before_side_that_already_passes():
    easy = _rows({f"t{i}": 1.0 if i < 28 else 0.5 for i in range(30)})
    report = delta_report(easy, easy, target="pass_at_1")
    assert report["ceiling"] is True
    assert any(
        w.startswith("The before run already passes 0.97 of tasks") for w in report["warnings"]
    )
    assert "CEILING" in format_delta_report(report)
    # too few paired tasks with room, most already solved every time
    mostly = _rows({f"t{i}": 1.0 if i < 45 else 0.0 for i in range(60)})
    report = delta_report(mostly, mostly, target="pass_at_1")
    assert report["ceiling"] is True
    assert any("45 of 60 paired tasks every time" in w for w in report["warnings"])
    # room to move: no flag
    hard = _rows({f"t{i}": 0.25 for i in range(30)})
    assert delta_report(hard, hard, target="pass_at_1")["ceiling"] is False


def test_training_holdout_block_carries_the_uncertainty_and_the_verdict():
    run = TrainingRun("run_1", name="n", transport=lambda *a, **k: {})
    before = _rows({f"t{i}": 0.25 for i in range(30)})
    after = _rows({f"t{i}": 0.5 for i in range(30)})
    run.delta(before, after, target="marker:h")
    block = run.holdout_summary
    assert block["before"]["n_tasks"] == 30 and block["before"]["k"] == 4
    assert len(block["before"]["ci95"]) == 2 and block["after"]["pass"] == 0.5
    assert block["verdict"] == "moved_unreplicated" and block["note"] is None
    assert run._summary["holdout"] == block
    # two floats from the platform: no interval, and it says so
    bare = _holdout_without_rows(0.17, 0.29)
    assert bare["before"] == {"pass": 0.17, "n_tasks": None, "k": None, "ci95": None}
    assert bare["verdict"] is None and bare["note"].startswith("No interval")
