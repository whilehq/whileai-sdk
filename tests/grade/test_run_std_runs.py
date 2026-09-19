"""A ``run_std`` handed to ``delta_report`` carries where it came from.
``run_std_runs=`` makes the re-run band the two-sided t quantile at ``runs -
1`` degrees of freedom; a bare ``run_std=`` keeps 1.96 and names the fix
(Lambert 2025, evaluation-variance appendix). ``recipes/papers/check.py``
holds its stdlib copy of ``noise_band`` to the same numbers."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from whileai.simulations.score.delta import delta_report, format_delta_report
from whileai.simulations.score.stats import noise_band

REPO = Path(__file__).resolve().parents[2]
SQRT2 = math.sqrt(2.0)


def _rows(pass_rates: dict[str, float], k: int = 4) -> list[dict]:
    rows = []
    for task, p in pass_rates.items():
        passes = round(p * k)
        for i in range(k):
            rows.append({"prompt": task, "reward": 1 if i < passes else 0, "markers": {"h": 1.0}})
    return rows


def _sides() -> tuple[list[dict], list[dict]]:
    # a +0.25 delta over 40 paired tasks: the interval is well off zero, so
    # the verdict turns on the re-run band alone
    return _rows({f"t{i}": 0.25 for i in range(40)}), _rows({f"t{i}": 0.5 for i in range(40)})


def test_run_std_runs_reads_the_band_at_the_t_quantile():
    before, after = _sides()
    # 0.07 as an exact spread: band 1.96 x sqrt(2) x 0.07 = 0.194, and 0.25 clears it
    exact = delta_report(before, after, target="pass_at_1", run_std=0.07)
    assert exact["target_verdict"] == "moved"
    assert exact["run_std_runs"] is None and exact["run_std_df"] is None
    assert exact["noise_rule"] == "1.96 x run_std x sqrt(1/1 + 1/1)"
    assert exact["noise_band"] == pytest.approx(1.96 * 0.07 * SQRT2)
    # the same 0.07 estimated from three re-runs: t(df=2) = 4.30, band 0.426,
    # and the same delta is inside the eval's own noise
    honest = delta_report(before, after, target="pass_at_1", run_std=0.07, run_std_runs=3)
    assert honest["target_verdict"] == "within_eval_noise"
    assert honest["run_std_runs"] == 3 and honest["run_std_df"] == 2
    assert honest["run_std_source"] == "given" and honest["replicated"] is True
    assert honest["noise_band"] == pytest.approx(noise_band(0.07, df=2))
    assert honest["noise_band"] == pytest.approx(4.303 * 0.07 * SQRT2, rel=1e-3)
    assert honest["noise_rule"] == "t(df=2)=4.30 x run_std x sqrt(1/1 + 1/1)"
    assert honest["metrics"]["marker:h"]["noise_band"] == honest["noise_band"]
    assert not any("run_std_runs" in w for w in honest["warnings"])
    text = format_delta_report(honest)
    assert "run_std given from 3 re-runs" in text and "t(df=2)=4.30" in text
    # ten re-runs: t(df=9) = 2.26, band 0.224, and 0.25 clears it again
    ten = delta_report(before, after, target="pass_at_1", run_std=0.07, run_std_runs=10)
    assert ten["target_verdict"] == "moved" and ten["run_std_df"] == 9
    assert ten["noise_band"] == pytest.approx(2.262 * 0.07 * SQRT2, rel=1e-3)
    # the interval keeps its own level; only the band's quantile moved
    assert ten["target_ci95"] == honest["target_ci95"] == exact["target_ci95"]


def test_a_bare_run_std_keeps_z_and_names_the_fix():
    before, after = _sides()
    report = delta_report(before, after, target="pass_at_1", run_std=0.07)
    fix = [w for w in report["warnings"] if "run_std_runs" in w]
    assert len(fix) == 1
    assert fix[0].startswith("run_std was given as a number, so the band uses 1.96")
    assert "pass run_std_runs=<how many> (eval_variance(...)['n_runs'])" in fix[0]
    assert "3 re-runs is 4.30, not 1.96" in fix[0]
    # a per-metric mapping is a given floor too
    mapped = delta_report(before, after, run_std={"pass_at_1": 0.07, "h": 0.07})
    assert sum("run_std_runs" in w for w in mapped["warnings"]) == 1
    # no floor at all: the single-run warning, not this one
    bare = delta_report(before, after, target="pass_at_1")
    assert bare["run_std_runs"] is None and bare["run_std_df"] is None
    assert not any("run_std_runs" in w for w in bare["warnings"])
    assert "run_std given" not in format_delta_report(bare)


def test_run_std_runs_is_checked_and_two_is_called_rough():
    before, after = _sides()
    with pytest.raises(ValueError, match="pass run_std with it"):
        delta_report(before, after, run_std_runs=3)
    with pytest.raises(ValueError, match="at least 2"):
        delta_report(before, after, run_std=0.07, run_std_runs=1)
    two = delta_report(before, after, target="pass_at_1", run_std=0.07, run_std_runs=2)
    assert two["run_std_df"] == 1 and two["noise_rule"].startswith("t(df=1)=12.71")
    assert any(w.startswith("run_std came from 2 re-runs, a difference") for w in two["warnings"])
    assert not any("run_std was given as a number" in w for w in two["warnings"])


def test_papers_check_noise_band_mirrors_the_package():
    spec = importlib.util.spec_from_file_location(
        "papers_check", REPO / "recipes" / "papers" / "check.py"
    )
    assert spec is not None and spec.loader is not None
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    for df in (None, 1, 2, 4, 9, 29, 30, 45, 200):
        assert check.noise_band(0.0169, df=df) == pytest.approx(
            noise_band(0.0169, df=df), rel=1e-9
        ), df
    assert check.noise_band(0.02, 3, 3, df=4) == pytest.approx(noise_band(0.02, 3, 3, df=4))
    # filter-metric's numbers: +0.067 with run_std 0.0169 from three base
    # re-runs cleared the 1.96 band (0.047) and does not clear the t band (0.103)
    assert check.noise_band(0.0169) < 0.0667 < check.noise_band(0.0169, df=2)
    assert check.noise_band(0.0169, df=2) == pytest.approx(0.1028, abs=5e-4)
