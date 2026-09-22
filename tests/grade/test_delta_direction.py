"""Direction: the metric a run set out to *reduce* is a win, not a fault.

#638: a voice run cut replies from 198.119 to 150.619 words and
``delta_report`` reported ``-47.500 ... DOWN``, warned ``! marker:words
dropped -47.500``, and under ``must_not_regress`` returned
``headline_verdict: moved_the_wrong_way, ok: False``. Every production
agent metric an operator cares about moves that way: reply length,
tokens, cost, latency, turns, retries, escalations.

These tests pin the wrong output, not just the right one. The two
``truncated`` tests take no new argument, so they fail on the old
behaviour with a plain assertion (a rise in a cut-completion marker read
as an improvement); the ``lower_is_better=`` tests pin the shape the
reporter asked for. Every one of them also checks the effect size
survives: the delta and the interval stay the raw signed change, so
"47.5 words shorter, 95% [-50.2, -44.9]" is still the sentence the report
prints.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from whileai.simulations.score.delta import delta_report, format_delta_report

# The reporter's own numbers (#638).
BEFORE_WORDS = 198.119
AFTER_WORDS = 150.619
DELTA_WORDS = -47.500

N_TASKS = 40
K = 2


def _spread(t: int) -> float:
    """Task-to-task spread on the before arm, zero-mean over the 40 tasks,
    so the printed mean is exactly the reporter's number."""
    return (t % 8 - 3.5) * 10.0


def _jitter(t: int) -> float:
    """Per-task noise on the after arm, also zero-mean: it gives the paired
    difference a spread to bootstrap without moving the delta."""
    return ((t % 5) - 2) * 6.0


def _arms(markers: Callable[[int, str], dict[str, float]]) -> tuple[list[dict], list[dict]]:
    """Two row sets on the same 40 tasks, ``K`` rollouts each, with the same
    rewards on both sides (so pass@1 is flat and off the ceiling) and
    whatever ``markers`` says per task and side."""
    before: list[dict] = []
    after: list[dict] = []
    for t in range(N_TASKS):
        for i in range(K):
            for side, rows in (("before", before), ("after", after)):
                rows.append(
                    {
                        "task_id": f"t{t}",
                        "prompt": f"ask {t}",
                        "tier": "even" if t % 2 == 0 else "odd",
                        "reward": 1 if (t + i) % 2 == 0 else 0,
                        "final_text": "reply",
                        "markers": markers(t, side),
                    }
                )
    return before, after


def _words(t: int, side: str) -> dict[str, float]:
    """The run whose whole purpose was to make this number smaller."""
    w = BEFORE_WORDS + _spread(t)
    return {"words": w if side == "before" else w + DELTA_WORDS + _jitter(t)}


def _words_and_helpfulness(t: int, side: str) -> dict[str, float]:
    """``words`` falls (the win) and ``helpfulness`` falls too (a real
    slip): the second is the negative control."""
    out = _words(t, side)
    out["helpfulness"] = 1.0 if side == "before" else (0.0 if t % 4 else 1.0)
    return out


def _words_rise(t: int, side: str) -> dict[str, float]:
    """The same run, failed: replies got longer."""
    w = BEFORE_WORDS + _spread(t)
    return {"words": w if side == "before" else w - DELTA_WORDS + _jitter(t)}


def _truncated(before_rate: float, after_rate: float) -> Callable[[int, str], dict[str, float]]:
    """A ``truncated`` marker at a given share per side, spread over tasks."""

    def markers(t: int, side: str) -> dict[str, float]:
        rate = before_rate if side == "before" else after_rate
        return {"truncated": 1.0 if (t % 10) < rate * 10 else 0.0}

    return markers


def _line(report: dict, metric: str) -> str:
    """The metric's own row of the printed table."""
    return next(
        line for line in format_delta_report(report).splitlines() if metric in line and "->" in line
    )


def test_a_cut_in_reply_length_is_not_reported_as_a_regression() -> None:
    """The pin for #638, with the reporter's numbers. On the old behaviour
    this run read ``DOWN``, warned ``marker:words dropped``, and came back
    ``moved_the_wrong_way`` with ``ok: False``."""
    before, after = _arms(_words)
    report = delta_report(
        before,
        after,
        target="marker:words",
        must_not_regress=["words"],
        lower_is_better=["words"],
        n_boot=400,
        seed=3,
    )
    words = report["metrics"]["marker:words"]

    # The effect size the operator asked for, unflipped: 47.5 words
    # shorter, with the interval on the raw change.
    assert words["mean_a"] == pytest.approx(BEFORE_WORDS, abs=1e-6)
    assert words["mean_b"] == pytest.approx(AFTER_WORDS, abs=1e-6)
    assert words["delta"] == pytest.approx(DELTA_WORDS, abs=1e-6)
    lo, hi = words["ci95"]
    assert lo < DELTA_WORDS < hi < 0.0
    assert report["target_delta"] == pytest.approx(DELTA_WORDS, abs=1e-6)

    # The specific wrong output is gone, on every surface.
    assert report["headline_verdict"] != "moved_the_wrong_way"
    assert report["headline_verdict"] in {"moved", "moved_unreplicated"}
    assert report["ok"] is True
    assert report["target_verdict"] != "moved_the_wrong_way"
    assert report["regressions"] == []
    assert report["slipped"] == []
    assert "marker:words" in report["improved"]
    assert "marker:words" in report["lower_is_better"]
    assert not [w for w in report["warnings"] if w.startswith("marker:words dropped")]
    assert not [w for w in report["warnings"] if w.startswith("REGRESSION marker:words")]

    text = format_delta_report(report)
    line = _line(report, "marker:words")
    assert "DOWN" not in line
    assert "lower is better" in line
    assert "-47.500" in line  # the raw number, not a sign-flipped one
    assert "FAIL" not in text.splitlines()
    assert "lower is better: marker:words" in text


def test_a_marker_not_named_lower_is_better_still_warns_on_a_drop() -> None:
    """The negative control: direction is opt-in per metric, so the marker
    nobody claimed is still higher-is-better and still slips."""
    before, after = _arms(_words_and_helpfulness)
    report = delta_report(
        before,
        after,
        target="marker:words",
        lower_is_better=["words"],
        n_boot=400,
        seed=3,
    )
    assert report["slipped"] == ["marker:helpfulness"]
    assert "marker:helpfulness" not in report["lower_is_better"]
    assert [w for w in report["warnings"] if w.startswith("marker:helpfulness dropped")]
    assert "DOWN" in _line(report, "marker:helpfulness")
    # ...while the metric that was claimed is still the win.
    assert "marker:words" in report["improved"]


def test_a_rise_in_a_lower_is_better_marker_warns_and_fails_the_guard() -> None:
    """The same run in the other direction: replies got longer, which is
    the regression now, and ``must_not_regress`` catches it."""
    before, after = _arms(_words_rise)
    report = delta_report(
        before,
        after,
        target="marker:words",
        must_not_regress=["words"],
        lower_is_better=["words"],
        n_boot=400,
        seed=3,
    )
    words = report["metrics"]["marker:words"]
    assert words["delta"] == pytest.approx(-DELTA_WORDS, abs=1e-6)  # raw, still positive
    assert report["ok"] is False
    assert report["headline_verdict"] == "moved_the_wrong_way"
    assert report["regressions"] == ["marker:words"]
    assert report["improved"] == []
    assert [
        w
        for w in report["warnings"]
        if w.startswith("REGRESSION marker:words") and "lower is better" in w
    ]
    assert "UP" in _line(report, "marker:words")


def test_a_rise_in_a_lower_is_better_marker_warns_without_the_guard() -> None:
    """Not named in ``must_not_regress``: still a warning, and the verb is
    the raw move, so the word and the number agree."""
    before, after = _arms(_words_rise)
    report = delta_report(before, after, lower_is_better=["words"], n_boot=400, seed=3)
    assert report["slipped"] == ["marker:words"]
    assert [w for w in report["warnings"] if w.startswith("marker:words rose +47.500")]


def test_truncated_going_up_is_the_regression_with_nothing_asked_for() -> None:
    """The built-in set. A completion the token cap cut is a failed
    completion on any run, so a rise in ``marker:truncated`` is the
    regression without anyone naming a direction. Before this, the rise
    read as an improvement and the guard passed."""
    before, after = _arms(_truncated(0.1, 0.6))
    report = delta_report(
        before, after, target="pass_at_1", must_not_regress=["truncated"], n_boot=400, seed=3
    )
    truncated = report["metrics"]["marker:truncated"]
    assert truncated["delta"] > 0  # the raw change is still a rise
    assert report["ok"] is False
    assert report["regressions"] == ["marker:truncated"]
    assert "marker:truncated" not in report["improved"]
    assert "marker:truncated" in report["lower_is_better"]


def test_truncated_going_down_is_not_a_slip() -> None:
    """The other half of the built-in set: fewer cut completions warns
    about nothing. Before this it read ``marker:truncated dropped``."""
    before, after = _arms(_truncated(0.6, 0.1))
    report = delta_report(before, after, target="pass_at_1", n_boot=400, seed=3)
    assert report["slipped"] == []
    assert report["improved"] == ["marker:truncated"]
    assert not [w for w in report["warnings"] if "marker:truncated dropped" in w]


def test_a_mapping_turns_a_built_in_default_back_the_other_way() -> None:
    """Rows can carry a name at the other polarity, so the built-in set is
    tunable from the call (CONSTITUTION belief 3)."""
    before, after = _arms(_truncated(0.6, 0.1))
    report = delta_report(
        before,
        after,
        target="pass_at_1",
        lower_is_better={"truncated": False},
        n_boot=400,
        seed=3,
    )
    assert report["lower_is_better"] == []
    assert report["slipped"] == ["marker:truncated"]


def test_names_work_with_or_without_the_marker_prefix() -> None:
    """The reporter wrote ``"words"``; the metric key is
    ``"marker:words"``. Both are the same ask."""
    before, after = _arms(_words)
    bare = delta_report(before, after, target="marker:words", lower_is_better=["words"], seed=3)
    prefixed = delta_report(
        before, after, target="marker:words", lower_is_better=["marker:words"], seed=3
    )
    assert bare["lower_is_better"] == prefixed["lower_is_better"] == ["marker:words"]
    assert bare["headline_verdict"] == prefixed["headline_verdict"]


def test_an_unmeasured_direction_raises_and_names_the_fix() -> None:
    """A direction that matched nothing would leave the report reading
    backwards in silence, so it raises instead, and the message says what
    was measured and how to fix it (style.md rule 10)."""
    before, after = _arms(_words)
    with pytest.raises(ValueError) as caught:
        delta_report(before, after, lower_is_better=["latency_ms"])
    message = str(caught.value)
    assert "latency_ms" in message
    assert "marker:words" in message  # what was measured
    assert "mark_rows" in message  # the fix


def test_direction_reaches_the_by_split() -> None:
    """A ``by=`` group is the target measured again, so it reads the
    target's direction: the group whose replies got *longer* is the one
    that moved the wrong way."""

    def markers(t: int, side: str) -> dict[str, float]:
        w = BEFORE_WORDS + _spread(t)
        if side == "before":
            return {"words": w}
        # even tasks fall (the win), odd tasks rise (the slip)
        step = DELTA_WORDS if t % 2 == 0 else -DELTA_WORDS
        return {"words": w + step + _jitter(t)}

    before, after = _arms(markers)
    report = delta_report(
        before,
        after,
        target="marker:words",
        by="tier",
        lower_is_better=["words"],
        n_boot=400,
        seed=3,
    )
    assert report["groups_down"] == ["odd"]
    assert report["groups"]["even"]["gain_verdict"] == "b_better"
    assert report["groups"]["odd"]["gain_verdict"] == "a_better"
    assert report["groups"]["odd"]["delta"] > 0  # raw, unflipped
    assert [w for w in report["warnings"] if "moved the wrong way for odd" in w]
    assert not [w for w in report["warnings"] if "moved the wrong way for even" in w]
    assert "lower is better" in _line(report, "odd")


def test_direction_reaches_the_over_optimization_check() -> None:
    """``proxy`` is the metric the run trained on. A down-is-the-win proxy
    that fell while the target did not follow is over-optimization, the
    same as a higher-is-better proxy that rose."""

    def markers(t: int, side: str) -> dict[str, float]:
        w = BEFORE_WORDS + _spread(t)
        return {
            "words": w if side == "before" else w + DELTA_WORDS + _jitter(t),
            "helpfulness": 1.0 if t % 2 else 0.0,  # flat: the target does not follow
        }

    before, after = _arms(markers)
    report = delta_report(
        before,
        after,
        target="marker:helpfulness",
        proxy="marker:words",
        lower_is_better=["words"],
        n_boot=400,
        seed=3,
    )
    assert report["proxy_verdict"] == "moved"
    assert report["over_optimized"] is True
    assert report["ok"] is False
    assert report["proxy_delta"] == pytest.approx(DELTA_WORDS, abs=1e-6)
    assert [
        w
        for w in report["warnings"]
        if w.startswith("OVER-OPTIMIZED: marker:words down -47.500 (lower is better)")
    ]


def test_the_built_in_set_is_the_stock_presence_markers_and_truncated() -> None:
    """The default set is small, named and sourced (CONSTITUTION belief 3):
    the five presence markers ``score.markers`` stamps at 1.0-is-worse, and
    the cut-completion marker."""
    from whileai.simulations.score.delta import LOWER_IS_BETTER_MARKERS
    from whileai.simulations.score.markers import STOCK_MARKERS

    assert set(LOWER_IS_BETTER_MARKERS) == set(STOCK_MARKERS) | {"truncated"}
