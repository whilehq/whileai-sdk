"""``train`` reads the set's profile before the GPU is spent (#396, #397).

SFT on a set with failing rows is refused: the hosted trainer clones every
row, so the model learns the failure (#396 measured tool use 0.99 -> 0.49).
A grouped method with no mixed task is refused, and with few is warned,
naming the count used against the count given, the reason per dropped
class, and the knob (#397 trained on 6 of 84 rows and said nothing).
"""

from __future__ import annotations

import warnings

import pytest

import whileai.simulations as wai
from whileai.simulations import defaults
from whileai.simulations.training import (
    CHECK_MODES,
    TrainingSelectionError,
    selection_report,
    train,
)

# wai.profile("ds_68c3bac292fe6222") as #396 / #397 printed it: 84 rows,
# 12 passes, 21 tasks of 4, 6 of them mixed.
ISSUE_PROFILE = {
    "rows": 84,
    "graded": 84,
    "pass_rate": 0.143,
    "split": {"pass": 12, "fail": 72, "ungraded": 0},
    "tasks": 21,
    "rows_per_task": 4.0,
    "tasks_with_repeats": 21,
    "mixed_tasks": 6,
    "mixed_fraction": 0.286,
    "support": 0.238,
    "per_task": [],
}
UNANIMOUS = {**ISSUE_PROFILE, "mixed_tasks": 0, "mixed_fraction": 0.0, "support": 0.0}


def _per_task(mixed: int, all_pass: int, all_fail: int, singles: int = 0, k: int = 4) -> list:
    rows = []
    for i in range(mixed):
        rows.append({"task": f"m{i}", "n": k, "graded": k, "pass_rate": 0.5})
    for i in range(all_pass):
        rows.append({"task": f"p{i}", "n": k, "graded": k, "pass_rate": 1})
    for i in range(all_fail):
        rows.append({"task": f"f{i}", "n": k, "graded": k, "pass_rate": 0})
    for i in range(singles):
        rows.append({"task": f"s{i}", "n": 1, "graded": 1, "pass_rate": 1})
    return rows


class Gate:
    """Profile route plus the train route; ``posted`` is the train body or
    None when the run never started."""

    def __init__(self, profile, *, profile_error: Exception | None = None):
        self.profile = profile
        self.profile_error = profile_error
        self.reads = 0
        self.posted: dict | None = None

    def __call__(self, method, path, api_key=None, body=None, **kw):
        if method == "GET" and path.endswith("/profile"):
            self.reads += 1
            if self.profile_error is not None:
                raise self.profile_error
            return {"profile": self.profile} if self.profile is not None else {}
        if method == "POST" and path.endswith("/train"):
            self.posted = body
            return {"training": {"runId": "run_s1", "method": body["method"], "status": "running"}}
        raise AssertionError(f"unexpected {method} {path}")


def _quiet(**kw):
    """train() with the served-base notice out of the way."""
    kw.setdefault("base_model", "Qwen/Qwen3-4B")
    return train("ds_68c3bac292fe6222", **kw)


# ------------------------------------------------------------- #396: sft


def test_sft_on_failing_rows_is_refused_before_any_post():
    gate = Gate(ISSUE_PROFILE)
    with pytest.raises(TrainingSelectionError) as err:
        _quiet(method="sft", epochs=2, transport=gate)
    text = str(err.value)
    assert gate.posted is None, "the run must not start"
    assert "all 84 rows, 72 of which fail" in text
    assert "scored.passes()" in text and 'check="warn"' in text
    assert "Lambert 2025, chapter Rejection Sampling" in text
    assert f"reward under {defaults.PASS_THRESHOLD:g}" in text


def test_sft_check_warn_says_the_same_and_starts_the_run():
    gate = Gate(ISSUE_PROFILE)
    with pytest.warns(UserWarning, match="72 of which fail") as caught:
        run = _quiet(method="sft", epochs=2, check="warn", transport=gate)
    assert gate.posted == {"method": "sft", "epochs": 2.0, "base": "Qwen/Qwen3-4B"}
    assert run.selection["refuse"] and run.selection["used"] == 84
    assert not any("could not read" in str(w.message) for w in caught)


def test_sft_on_passes_only_or_ungraded_rows_is_quiet():
    for split in ({"pass": 84, "fail": 0, "ungraded": 0}, {"pass": 0, "fail": 0, "ungraded": 84}):
        gate = Gate({**ISSUE_PROFILE, "split": split, "mixed_tasks": 0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run = _quiet(method="sft", transport=gate)
        assert gate.posted is not None
        assert run.selection == {
            "method": "sft",
            "given": 84,
            "used": 84,
            "dropped": {},
            "refuse": [],
            "warn": [],
        }


# ------------------------------------------------------------ #397: grpo


def test_grpo_with_no_mixed_task_is_refused_with_the_reasons():
    gate = Gate({**UNANIMOUS, "per_task": _per_task(0, 3, 18)})
    with pytest.raises(TrainingSelectionError) as err:
        _quiet(method="grpo", steps=20, transport=gate)
    text = str(err.value)
    assert gate.posted is None
    assert "0 of 21 tasks (0 of 84 rows)" in text
    assert "3 tasks all pass, 18 tasks all fail" in text
    assert "Shao et al. 2024, arXiv:2402.03300" in text and "2503.14476" in text
    assert "profile('ds_68c3bac292fe6222')['mixed_tasks']" in text
    assert 'check="warn"' in text


def test_grpo_on_few_mixed_tasks_warns_with_used_vs_given_and_the_floor():
    gate = Gate(ISSUE_PROFILE)
    with pytest.warns(UserWarning) as caught:
        run = _quiet(method="grpo", steps=20, generations=4, seed=11, transport=gate)
    (line,) = [str(w.message) for w in caught if "will use" in str(w.message)]
    assert line.startswith("grpo on ds_68c3bac292fe6222 will use 6 of 21 tasks of 84 rows: ")
    assert "15 tasks unanimous (all pass or all fail)" in line
    assert (
        f"under min_mixed_tasks ({defaults.TRAIN_MIN_MIXED_TASKS}, TRAIN_MIN_MIXED_TASKS)" in line
    )
    assert "20 steps is 3.3 passes over them" in line
    assert "train(min_mixed_tasks=) moves the floor" in line
    assert "mixed_tasks" in line
    assert gate.posted is not None, "a warning, not a stop"
    assert run.selection["given"] == 21 and run.selection["used"] == 6
    assert run.selection["dropped"] == {"tasks unanimous (all pass or all fail)": 15}


def test_the_floor_is_a_knob_and_a_complete_per_task_table_names_each_class():
    profile = {**ISSUE_PROFILE, "per_task": _per_task(6, 1, 14), "tasks_with_repeats": 21}
    gate = Gate(profile)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = _quiet(method="grpo", steps=20, min_mixed_tasks=6, transport=gate)
    lines = [str(w.message) for w in caught if "will use" in str(w.message)]
    assert lines == [
        "grpo on ds_68c3bac292fe6222 will use 6 of 21 tasks (24 of 84 rows): "
        "1 tasks all pass, 14 tasks all fail."
    ], "at the floor only the used-vs-given line is said"
    assert run.selection["used_rows"] == 24
    assert run.selection["dropped"] == {"tasks all pass": 1, "tasks all fail": 14}
    with pytest.raises(ValueError, match="min_mixed_tasks"):
        _quiet(method="grpo", min_mixed_tasks=0, transport=Gate(profile))


def test_dpo_and_rm_use_the_pair_rule():
    for method in ("dpo", "rm"):
        gate = Gate(UNANIMOUS)
        with pytest.raises(TrainingSelectionError, match="chapter Direct Alignment"):
            _quiet(method=method, transport=gate)
        assert gate.posted is None


def test_enough_mixed_tasks_is_silent():
    full = {
        **ISSUE_PROFILE,
        "rows": 160,
        "split": {"pass": 80, "fail": 80, "ungraded": 0},
        "tasks": 40,
        "tasks_with_repeats": 40,
        "mixed_tasks": 40,
    }
    gate = Gate(full)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = _quiet(method="grpo", transport=gate)
    assert run.selection["warn"] == [] and run.selection["refuse"] == []


# ---------------------------------------------------------- the check knob


def test_check_off_does_not_read_the_profile():
    gate = Gate(ISSUE_PROFILE)
    run = _quiet(method="sft", check="off", transport=gate)
    assert gate.reads == 0 and gate.posted is not None and run.selection is None
    with pytest.raises(ValueError, match="require, warn, off"):
        _quiet(method="sft", check="loud", transport=Gate(ISSUE_PROFILE))
    assert CHECK_MODES == ("require", "warn", "off")


def test_an_unreadable_profile_is_said_and_does_not_stop_the_run():
    for gate in (Gate(None), Gate(None, profile_error=RuntimeError("503"))):
        with pytest.warns(UserWarning, match="could not read wai.profile") as caught:
            run = _quiet(method="grpo", transport=gate)
        assert gate.posted is not None and run.selection is None
        assert 'check="off"' in str(caught[0].message)


def test_selection_report_is_pure_and_public():
    report = selection_report(ISSUE_PROFILE, method="grpo", steps=20)
    assert report["given"] == 21 and report["used"] == 6 and report["refuse"] == []
    assert report["warn"] and "6 of 21 tasks" in report["warn"][0]
    report = selection_report(ISSUE_PROFILE, method="sft")
    assert report["used"] == 84 and len(report["refuse"]) == 1
    assert wai.training.selection_report is selection_report  # one dot down (style.md)
    assert wai.training.TrainingSelectionError is TrainingSelectionError
    assert issubclass(TrainingSelectionError, ValueError)
