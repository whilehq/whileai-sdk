"""The science gate in ``recipes/papers/check.py``, driven red on purpose.

``CONSTITUTION.md`` says this is where belief 1 is enforced: the check
"refuses 'moved' without an interval that excludes zero, three base re-runs, a
clean holdout, and a proxy-vs-target verdict". Until #809 no recipe had ever
reached that branch -- all nine were ``unresolved`` at one training seed per
arm -- so the four criteria were a claim about code nobody had run. A gate that
has never fired has not been shown to work.

So each criterion gets a recipe that violates it and nothing else, and the
assertion is that the gate refuses it and names the fix. Three of the six cases
found a real hole:

* three base re-runs: the check asked for 2, not 3, so a ``moved`` verdict on a
  ``run_std`` from two re-runs passed the file the constitution says refuses it.
* a clean holdout: there was no holdout gate at all.
  ``checks.decontaminated_dropped`` was required to be present and never read,
  so ``null`` -- contamination never looked for -- passed.
* a proxy-vs-target verdict: ``if r["checks"]["over_optimized"]:`` refuses
  ``true`` and waves through ``null``, which is the absence of the verdict the
  constitution asks for, not a clean one.

``test_a_clean_moved_recipe_passes`` is the other half: a gate that refuses
everything is not a gate either.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "recipes" / "papers"


def _check_module() -> Any:
    spec = importlib.util.spec_from_file_location("papers_check_gate", PAPERS / "check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _check_module()

#: A recipe that clears every criterion: the interval excludes zero, the delta
#: (0.10) is over the band that three re-runs of a 0.01 run_std make
#: (t(df=2)=4.30 x 0.01 x sqrt(2) = 0.061), the train split was decontaminated
#: against the holdout, and the proxy-vs-target check came back clean.
CLEAN: dict[str, Any] = {
    "recipe": "gate-fixture",
    "title": "A recipe that claims a result",
    "paper": "https://arxiv.org/abs/2609.00444",
    "book": "Evaluation",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "metric": "pass@1",
    "n_holdout": 120,
    "k": 4,
    "arms": {
        "base": {"score": 0.30, "ci": [0.24, 0.36], "steps": 0},
        "baseline": {"score": 0.40, "ci": [0.34, 0.46], "steps": 40},
        "recipe": {"score": 0.50, "ci": [0.44, 0.56], "steps": 40},
    },
    "delta": {"recipe_vs_baseline": 0.10, "ci": [0.05, 0.15], "verdict": "moved"},
    "checks": {
        "run_std": 0.01,
        "run_std_runs": 3,
        "train_seeds": {"baseline": 3, "recipe": 3},
        "decontaminated_dropped": 0,
        "over_optimized": False,
        "length_before": 700.0,
        "length_after": {"baseline": 500.0, "recipe": 520.0},
        "hack_scan_top": "n:digits",
        "seed": 0,
    },
    "gpu": "L40S",
    "usd": 0.68,
    "verified": "2026-09-18",
    "whileai": "0.123",
}

README = """# A recipe that claims a result

**Paper:** a paper. https://arxiv.org/abs/2609.00444
**Book:** the chapter.
**Claim:** the claim.
**The change:** the change.

## Recipe

## Run

## Result

## Checks

## Climb

## Learned
"""


def _recipe(tmp_path: Path, results: dict, *, entry: str = "recipe.py") -> Path:
    d = tmp_path / results["recipe"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text(README, encoding="utf-8")
    (d / entry).write_text("# the recipe\n", encoding="utf-8")
    (d / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return d


def _refused(tmp_path: Path, results: dict, capsys: pytest.CaptureFixture[str]) -> str:
    """Run the gate on a recipe that must not pass, and return what it said."""
    with pytest.raises(SystemExit) as exit_info:
        check.check_recipe(_recipe(tmp_path, results))
    assert exit_info.value.code == 1
    said = capsys.readouterr().out
    assert said.startswith("FAIL "), said
    return said


def _with(**checks: Any) -> dict:
    """The clean recipe with one criterion broken."""
    results = copy.deepcopy(CLEAN)
    results["checks"].update(checks)
    return results


# --------------------------------------------------------------- (a) the interval


def test_moved_with_an_interval_covering_zero_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = copy.deepcopy(CLEAN)
    results["delta"]["ci"] = [-0.02, 0.22]
    said = _refused(tmp_path, results, capsys)
    assert "interval" in said and "covers zero" in said
    assert "say flat" in said


# --------------------------------------------------------------- (b) the noise band


def test_moved_under_the_re_run_band_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The interval excludes zero and the delta is still inside the noise the
    eval makes on its own re-runs: t(df=2) = 4.30 x 0.05 x sqrt(2) = 0.30."""
    results = _with(run_std=0.05)
    results["delta"] = {"recipe_vs_baseline": 0.02, "ci": [0.01, 0.03], "verdict": "moved"}
    said = _refused(tmp_path, results, capsys)
    assert "|delta| 0.020" in said and "0.304" in said
    assert "t(df=2)=4.30" in said


# --------------------------------------------------------------- (c) three base re-runs


def test_moved_on_fewer_than_three_base_re_runs_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A REAL HOLE until #809: the check asked for ``run_std_runs >= 2``, and
    the constitution's row asks for three. ``run_std`` is small enough here
    that the band at df=1 (12.71 x 0.001 x sqrt(2) = 0.018) does not bind, so
    this criterion is the only one that can refuse the recipe."""
    said = _refused(tmp_path, _with(run_std_runs=2, run_std=0.001), capsys)
    assert "2 base re-run(s)" in said and f"the bar is {check.MIN_BASE_RERUNS}" in said
    assert "checks.run_std_runs" in said and "or say flat" in said
    # and it is the constitution's number, not a number this file invented
    assert check.MIN_BASE_RERUNS == 3


# --------------------------------------------------------------- (d) a clean holdout


@pytest.mark.parametrize("dropped", [None, -1, True, "0", 1.5])
def test_moved_on_a_holdout_that_was_never_decontaminated_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], dropped: Any
) -> None:
    """A REAL HOLE until #809: there was no holdout gate at all.
    ``checks.decontaminated_dropped`` was required to exist and never read, so
    ``null`` -- nobody looked for overlap -- claimed a clean holdout."""
    said = _refused(tmp_path, _with(decontaminated_dropped=dropped), capsys)
    assert "decontaminated_dropped" in said and "not known clean" in said
    assert "wai.decontaminate(train, holdout)" in said


def test_a_decontamination_that_dropped_rows_is_a_clean_holdout(tmp_path: Path) -> None:
    """A positive count is overlap found and removed, which is the check doing
    its job; only a missing count is a holdout nobody checked."""
    assert check.check_recipe(_recipe(tmp_path, _with(decontaminated_dropped=7)))


# --------------------------------------------------------- (e) a proxy-vs-target verdict


def test_moved_with_an_over_optimized_proxy_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    said = _refused(tmp_path, _with(over_optimized=True), capsys)
    assert "over-optimized" in said


def test_moved_with_no_proxy_verdict_at_all_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A REAL HOLE until #809: ``if r["checks"]["over_optimized"]:`` refuses
    ``true`` and waves ``null`` through, and ``null`` is the absence of the
    verdict the constitution asks for, not a clean one."""
    said = _refused(tmp_path, _with(over_optimized=None), capsys)
    assert "over_optimized" in said and "no proxy-vs-target verdict" in said
    assert "Over-Optimization" in said


# --------------------------------------------------------------- (f) the clean case


def test_a_clean_moved_recipe_passes(tmp_path: Path) -> None:
    """The other half of the gate: it has to let a real result through, and
    say nothing about criteria it did not skip."""
    d = _recipe(tmp_path, copy.deepcopy(CLEAN))
    results = check.check_recipe(d)
    assert results["delta"]["verdict"] == "moved"
    assert check.skipped_note(results) == ""
    assert "moved, 3 seeds per arm" in check.row(d, results)


# ------------------------------------------- the no-trained-arm path does not bypass it


def test_a_recipe_with_no_trained_arm_still_faces_the_science_half(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``train_seeds: null`` says "nothing was trained here", so the seed rule
    has nothing to count. It is not a way around the four criteria (#809)."""
    results = _with(train_seeds=None)
    results["delta"]["ci"] = [-0.02, 0.22]
    said = _refused(tmp_path, results, capsys)
    assert "covers zero" in said
    # and the same recipe, unresolved, is accepted with run.py as its entry point
    ok = _with(train_seeds=None)
    ok["delta"]["verdict"] = "unresolved"
    assert check.check_recipe(_recipe(tmp_path / "step", ok, entry="run.py"))
    assert check.arms_read(ok) == "no trained arm"


def test_train_seeds_must_be_the_dict_or_an_explicit_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    said = _refused(tmp_path, _with(train_seeds={"baseline": 3}), capsys)
    assert "checks.train_seeds" in said and "null when the recipe has no trained arm" in said


# ------------------------------------------------------------------- meta-harness itself


def test_meta_harness_is_inside_the_gate_now() -> None:
    """#809: the recipe carrying the harness-optimization headline was the one
    recipe ``recipe_dirs()`` skipped, because the filter was the shape
    (``run.py``) rather than the claim (``results.json``)."""
    assert PAPERS / "meta-harness" in check.recipe_dirs()
    results = check.check_recipe(PAPERS / "meta-harness")
    assert results["checks"]["train_seeds"] is None, "the recipe trains nothing"
    # three of the four criteria are met and the fourth is not measured, so the
    # verdict is not "moved" and the note says which criterion is missing.
    assert results["delta"]["verdict"] == "unresolved"
    assert results["delta"]["ci"] == [0.27, 0.47] and results["checks"]["run_std_runs"] == 4
    assert results["checks"]["over_optimized"] is None
    note = check.skipped_note(results)
    assert "no trained arm" in note and "a proxy-vs-target verdict" in note
    assert "moved is not available" in note


def test_every_recipe_that_claims_a_number_is_in_the_table() -> None:
    """The filter is the claim, not the shape: one ``results.json``, one row."""
    claims = {d.name for d in PAPERS.iterdir() if d.is_dir() and (d / "results.json").exists()}
    listed = {d.name for d in check.recipe_dirs()}
    assert claims - listed == {"_template"}, claims - listed
