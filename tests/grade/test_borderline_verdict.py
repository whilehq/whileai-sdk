"""A gain seen on one eval run is not printed as a bare ``PASS``.

External case study on whileai 0.126 (gentlyventures.com/casestudies/whileai):
"A borderline single run prints PASS next to a warning." The headline
said ``PASS`` while the warning below it said the result could be noise.
One run per side has no run-to-run spread, so the headline now reads
``INCONCLUSIVE`` and names why; a gain repeated over three runs a side,
clear of the re-run band, still prints ``PASS``.
"""

from __future__ import annotations

import random

from whileai.simulations.score.delta import delta_report, format_delta_report, headline_word


def _rows(rates: dict[str, float], *, k: int = 4, run: int | None = None) -> list[dict]:
    rows = []
    for task, p in rates.items():
        passes = round(p * k)
        for i in range(k):
            row = {"prompt": task, "reward": 1.0 if i < passes else 0.0}
            if run is not None:
                row["lineage"] = {"eval_run": run}
            rows.append(row)
    return rows


def _rates(seed: int, base: float) -> dict[str, float]:
    rng = random.Random(seed)
    return {f"t{i}": min(1.0, max(0.0, base + rng.choice([-0.25, 0.0, 0.25]))) for i in range(30)}


def test_a_single_run_gain_is_inconclusive_not_pass():
    before = _rows(_rates(7, 0.5))
    after = _rows(_rates(8, 0.75))
    report = delta_report(before, after, n_boot=200, seed=0, target="pass_at_1")
    assert report["headline_verdict"] == "moved_unreplicated"
    assert any("One eval run on each side" in w for w in report["warnings"])
    lines = format_delta_report(report).splitlines()
    assert lines[0].startswith("pass_at_1: moved_unreplicated (+")
    # the headline under the target line: not a bare PASS, and it says why
    assert "PASS" not in lines[1]
    assert lines[1] == "INCONCLUSIVE (1 eval run a side, rerun to confirm)"
    # the gate is unchanged: a one-run gain is not a regression
    assert report["ok"] is True
    # no target: the headline is pass@1's verdict, same reading
    plain = delta_report(before, after, n_boot=200, seed=0)
    assert headline_word(plain) == "INCONCLUSIVE (1 eval run a side, rerun to confirm)"


def test_a_repeated_clear_gain_still_prints_pass():
    before, after = [], []
    for run in range(3):
        before += _rows(_rates(10 + run, 0.25), run=run)
        after += _rows(_rates(20 + run, 0.75), run=run)
    report = delta_report(before, after, n_boot=200, seed=0)
    assert report["headline_verdict"] == "moved"
    assert format_delta_report(report).splitlines()[0] == "PASS"
    assert headline_word(report) == "PASS"
