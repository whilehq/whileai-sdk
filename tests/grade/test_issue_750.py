"""#750: the re-run band counts the row sets ``train_runs`` names as eval
draws, so three seeds a side get ``sqrt(1/3 + 1/3)``, not ``sqrt(2)``."""

from __future__ import annotations

import math

import pytest

from whileai.simulations.score.delta import delta_report, format_delta_report
from whileai.simulations.score.stats import noise_band

K = 4


def _rows(rates: dict[str, float]) -> list[dict]:
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


def _seeds(centre: float, n: int = 3) -> list[list[dict]]:
    """``n`` training seeds at ``centre``; seed ``s`` passes one more
    rollout on its first ``s`` tasks, a small between-seed spread."""
    return [
        _rows({f"t{i}": centre + (0.25 if i < s else 0.0) for i in range(40)}) for s in range(n)
    ]


def test_three_seeds_a_side_get_the_three_run_band():
    before, after = _seeds(0.25), _seeds(0.5)
    flat_b = [r for run in before for r in run]
    flat_a = [r for run in after for r in run]
    rep = delta_report(
        flat_b,
        flat_a,
        target="pass_at_1",
        run_std=0.02,
        run_std_runs=3,
        train_runs={"before": before, "after": after},
        n_boot=200,
    )
    assert rep["eval_runs"] == {"before": 3, "after": 3}
    assert rep["noise_rule"] == "t(df=2)=4.30 x run_std x sqrt(1/3 + 1/3)"
    assert rep["noise_band"] == pytest.approx(noise_band(0.02, 3, 3, df=2))
    assert rep["noise_band"] == pytest.approx(noise_band(0.02, 1, 1, df=2) / math.sqrt(3))
    assert "sqrt(1/3 + 1/3); run_std given from 3 re-runs" in format_delta_report(rep)


def test_seeds_on_one_arm_count_on_that_side_only():
    after = _seeds(0.5)
    rep = delta_report(
        _rows({f"t{i}": 0.25 for i in range(40)}),
        after[1],
        target="pass_at_1",
        run_std=0.02,
        train_runs=after,
        n_boot=200,
    )
    assert rep["eval_runs"] == {"before": 0, "after": 3}  # no lineage on the before rows
    assert rep["noise_rule"] == "1.96 x run_std x sqrt(1/1 + 1/3)"
    assert rep["noise_band"] == pytest.approx(noise_band(0.02, 1, 3))


def test_seeds_without_a_floor_say_what_to_pass():
    before, after = _seeds(0.25), _seeds(0.5)
    rep = delta_report(
        before[1],
        after[1],
        target="pass_at_1",
        train_runs={"before": before, "after": after},
        n_boot=200,
    )
    assert rep["eval_runs"] == {"before": 3, "after": 3} and rep["run_std"] is None
    assert rep["target_verdict"] == "moved_unreplicated"
    assert not any("One eval run" in w for w in rep["warnings"])
    assert any("3 runs before and 3 after but no re-run floor" in w for w in rep["warnings"])
