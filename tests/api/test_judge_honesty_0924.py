"""Three SDK calls let a caller believe their judge was validated when
nothing measured it (2026-09-24 dogfooding). Each test here reproduces the
lie and pins the honest behavior instead.

1. ``platform.Judge(agreement=, human_n=)`` are declared fields; nothing
   computed them, but the Runs page judge check read them as if something
   had. ``verified`` (default ``False``) says so plainly.
2. ``judge_trust`` grades the ``reward`` already on the row against gold,
   not a fresh call to ``judge=``. A chance-level judge with no
   ``judge_name`` stamp on its rows could clear the default perturbation
   sample by luck and print PASS at 100% agreement.
3. ``check_spread`` (``GroupwiseGrading``) scores variance only; "spread:
   yes" read as "this grader is trustworthy" when it only means the
   scores differ.
4. ``attach_labels(kind=...)`` used to default silently to ``"human"``.
"""

from __future__ import annotations

import warnings

import pytest

from whileai.groupwise import GroupwiseGrading, spread
from whileai.platform import Behavior, Judge, eval_checks
from whileai.simulations.score.judge_trust import judge_trust
from whileai.simulations.score.labels import attach_labels

# --------------------------------------------------- 1. platform.Judge


def test_declared_agreement_does_not_pass_the_judge_check_on_its_own_say_so():
    """Typing two numbers into Judge() used to read as measured: the Runs
    page judge check scored it ``ok`` whatever the numbers were attached
    to. Declaring them is fine; the check just cannot certify them."""
    declared = Judge(agreement=0.98, human_n=200)
    b = Behavior(name="refunds", test_version="v2", n=240, judge=declared)
    health = eval_checks(b, [])
    (check,) = [c for c in health.checks if c.key == "judge"]
    assert check.ok is False, "an unverified pair must not score a green check"
    assert "declared" in check.value and "unverified" in check.value
    assert "0.98" in check.value and "200" in check.value  # the numbers still show
    assert "verified" in check.action

    # the same numbers, marked as actually measured, do pass
    measured = Judge(agreement=0.98, human_n=200, verified=True)
    b2 = Behavior(name="refunds", test_version="v2", n=240, judge=measured)
    (check2,) = [c for c in eval_checks(b2, []).checks if c.key == "judge"]
    assert check2.ok is True


# --------------------------------------------------- 2. judge_trust


def _rows_reward_equals_gold(majority: int, minority: int) -> list[dict]:
    """``reward`` matches ``gold_reward`` exactly on every row and no row
    carries a ``judge_name``: nothing shows this number came from any
    judge at all, chance-level or otherwise."""
    rows = [
        {
            "task_id": f"a{i}",
            "prompt": f"p{i}",
            "final_text": f"x{i}",
            "reward": 0,
            "gold_reward": 0,
            "gold_kind": "human",
        }
        for i in range(majority)
    ]
    rows += [
        {
            "task_id": f"b{i}",
            "prompt": f"q{i}",
            "final_text": f"y{i}",
            "reward": 1,
            "gold_reward": 1,
            "gold_kind": "human",
        }
        for i in range(minority)
    ]
    return rows


class _AlwaysFailJudge:
    """A constant, gold-blind judge: chance-level in the sense that it
    never looks at the row. If it were actually run, its agreement with
    the 5 minority-class rows above would be 0%."""

    __name__ = "always_fail_judge"

    def __call__(self, row: dict) -> dict:
        return {"reward": 0, "reason": "always fail"}


def test_judge_trust_does_not_pass_a_never_called_judge_on_a_lucky_sample():
    """The default 40-row perturbation sample can, by chance, miss every
    one of the 5 rows a chance-level judge would actually disagree on
    (seed=0 here reproduces exactly that on origin/main): agreement and
    kappa come from the ``reward`` field, never from calling ``judge``,
    so the report used to say PASS at 100% for a judge that was never
    run for real."""
    rows = _rows_reward_equals_gold(majority=95, minority=5)
    report = judge_trust(rows, _AlwaysFailJudge(), seed=0)  # default sample=40
    assert report["ok"] is False, "an unstamped, never-verified reward must not pass"
    assert any(w.startswith("Judge agreement is unverified") for w in report["warnings"]), (
        "the report must say the number is not a proven output of the passed judge"
    )

    # once the rows are honestly stamped by the judge that produced the
    # reward, the same check stays quiet (no false accusation)
    stamped = [{**r, "judge_name": "always_fail_judge"} for r in rows]
    clean = judge_trust(stamped, _AlwaysFailJudge(), seed=0)
    assert not any(w.startswith("Judge agreement is unverified") for w in clean["warnings"])


# --------------------------------------------------- 3. check_spread


def test_check_spread_states_what_it_does_not_establish():
    """A position-biased judge and a discriminating program grader both
    print 'spread: yes' (0.211 vs 0.099, 2026-09-21 text-to-SQL lesson);
    variance alone says nothing about accuracy or bias, and the report
    must say so, not just the headline verdict."""
    report = spread([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])  # ample spread, ok=True
    assert report["ok"] is True
    text = str(report)
    assert "spread: yes" in text
    assert "does not establish" in text
    assert "biased" in text
    assert "judge_trust" in text or "compare_judges" in text

    gar = GroupwiseGrading(grader=lambda group: {"ranking": list(range(len(group)))})
    rows = [
        {"scenario_id": "s0", "reward": 1, "final_text": "a"},
        {"scenario_id": "s0", "reward": 1, "final_text": "b"},
    ]
    ok_report = gar.check_spread(rows, strict=False, min_std=0.0)
    assert "does not establish" in str(ok_report)


# --------------------------------------------------- 4. attach_labels(kind=)


def test_attach_labels_warns_loudly_when_kind_is_left_out():
    """Omitting kind= used to default to "human" with nothing said: a
    model's labels recorded that way read as a person's forever after.
    It still defaults to "human" for compatibility, but never silently."""
    rows = [{"prompt": "p0", "scenario_id": "s0", "rollout_index": 0, "final_text": "x"}]
    with pytest.warns(UserWarning, match="kind was not given"):
        _, report = attach_labels(rows, {"s0#0": 1}, annotator="model-run")
    assert any("kind was not given" in w for w in report["warnings"])
    assert rows[0]["gold_kind"] == "human"  # unchanged default, just no longer silent

    # an explicit kind, human or otherwise, never warns
    rows2 = [{"prompt": "p0", "scenario_id": "s0", "rollout_index": 0, "final_text": "x"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, report2 = attach_labels(rows2, {"s0#0": 1}, kind="model", annotator="opus-4.5")
    assert not report2.get("warnings")
    assert rows2[0]["gold_kind"] == "model"
