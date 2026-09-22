"""#668: a label the ingest path cannot read is not an ungraded row.

``_binary_reward`` returned ``None`` for everything it could not read, and
``None`` is indistinguishable from "never graded". Two classes of customer
lost their labels with no warning anywhere:

* the column is named ``score`` / ``label`` / ``grade`` / ``rating`` /
  ``pass`` - 60 traces carrying ``score`` reported
  ``graded: 0, ungraded: 60, warnings: []``;
* the reward is fractional - ``0.75`` is what a multi-criterion rubric
  judge emits, and every such row read as unlabelled.

The judge label IS the reward, and a reward pipeline that discards labels
quietly cannot be audited (Lambert 2025, chapters Reward Models and
Evaluation). The SDK still picks no binarisation threshold and guesses no
column name: it stops being silent, and the caller names both.

Offline: no model, no keys.
"""

from __future__ import annotations

import pytest

from whileai.simulations.ingest import traces as ingest_traces
from whileai.simulations.ingest.traces import (
    _binary_reward,
    format_trace_report,
    load_traces,
    trace_report,
)


def _rows(n: int = 60, **label) -> list[dict]:
    """``n`` clean traces, every tool call a success, labelled by ``label``."""
    return [
        {
            "prompt": f"cancel order {i}",
            "steps": [
                {
                    "tool": "cancel_order",
                    "arguments": {"order_id": f"ord_{i}"},
                    "result": {"status": "ok", "order_id": f"ord_{i}"},
                }
            ],
            "final_text": "Sure, cancelled.",
            **label,
        }
        for i in range(n)
    ]


def test_sixty_rows_carrying_score_are_no_longer_silently_ungraded():
    """The measured line from the issue: graded 0, ungraded 60, warnings []."""
    report = trace_report(_rows(60, score=0.0))
    assert report["graded"] == 0, "nothing is guessed: score is not read without being named"
    assert report["ungraded"] == 60
    # The part that was missing entirely. A column the report can see and
    # does not read has to be said out loud.
    assert report["warnings"], "60 graded traces reported with warnings: [] is the defect"
    assert report["unread_numeric_columns"] == {"score": 60}
    assert "score" in report["warnings"][0]
    assert "reward_key" in report["warnings"][0], "the warning names the call that fixes it"


def test_naming_the_column_recovers_every_label():
    report = trace_report(_rows(60, score=0.0), reward_key="score")
    assert (report["graded"], report["fails"], report["ungraded"]) == (60, 60, 0)
    assert report["unread_numeric_columns"] == {}
    assert report["warnings"] == []


def test_a_fractional_reward_is_not_the_same_silence_as_never_graded():
    """0.75 was found, parsed as a float, and thrown away as unlabelled."""
    fractional = trace_report(_rows(12, reward=0.75))
    never = trace_report(_rows(12))
    assert fractional["non_binary_labels"] == 12
    assert fractional["ungraded"] == 0, "a value was there; that is not 'no column'"
    assert never["ungraded"] == 12
    assert never["non_binary_labels"] == 0
    # Both used to be one number, so the two reports were identical here.
    assert (fractional["ungraded"], fractional["non_binary_labels"]) != (
        never["ungraded"],
        never["non_binary_labels"],
    )
    assert "0.75" in fractional["warnings"][0]
    assert "threshold" in fractional["warnings"][0]


def test_the_warning_names_the_numeric_columns_actually_on_the_rows():
    rows = _rows(6, score=0.5, rubric_mean=0.75, latency_ms=120)
    report = trace_report(rows)
    assert report["unread_numeric_columns"] == {"latency_ms": 6, "rubric_mean": 6, "score": 6}
    note = report["warnings"][0]
    for column in ("score", "rubric_mean", "latency_ms"):
        assert column in note, f"{column} is on every row and the report never mentioned it"


def test_the_threshold_is_the_callers_and_binarises_when_given():
    rows = _rows(4, score=0.75) + _rows(4, score=0.25)
    assert trace_report(rows, reward_key="score")["non_binary_labels"] == 8
    banded = trace_report(rows, reward_key="score", threshold=0.5)
    assert (banded["graded"], banded["passes"], banded["fails"]) == (8, 4, 4)
    assert banded["non_binary_labels"] == 0
    # and the rows themselves come back labelled, ready for simulate(traces=)
    loaded = load_traces(rows, reward_key="score", threshold=0.5)
    assert sorted(r["reward"] for r in loaded) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_binary_reward_reads_the_issues_table():
    """reward=0.0 -> 0 ; reward=1.0 -> 1 ; reward=0.75 -> None ; score=0.0 -> None."""
    assert _binary_reward({"reward": 0.0}) == 0
    assert _binary_reward({"reward": 1.0}) == 1
    assert _binary_reward({"reward": 0.75}) is None, "no threshold means no opinion"
    assert _binary_reward({"score": 0.0}) is None, "a column is never guessed"
    # ...and both silences have a caller-supplied way out.
    assert _binary_reward({"score": 0.0}, reward_key="score") == 0
    assert _binary_reward({"reward": 0.75}, threshold=0.5) == 1
    assert _binary_reward({"reward": 0.25}, threshold=0.5) == 0


def test_a_model_specific_key_keeps_working_for_one_release_and_says_the_new_name(monkeypatch):
    """CONSTITUTION belief 8: never big-bang. qwen_reward still steers, and
    now it is disclosed instead of being the only answer."""
    # The warning is said once per run (style rule 10), so this test owns
    # the flag rather than depending on being the first to trip it.
    monkeypatch.setattr(ingest_traces, "_legacy_key_warned", False)
    rows = _rows(5, qwen_reward=0)
    with pytest.warns(DeprecationWarning, match="reward_key='qwen_reward'"):
        report = trace_report(rows)
    assert report["advisory_labels"] == 5
    assert report["ungraded"] == 0
    assert any("qwen_reward" in note for note in report["warnings"])
    # naming it makes it an ordinary label column like any other
    named = trace_report(rows, reward_key="qwen_reward")
    assert (named["graded"], named["fails"], named["advisory_labels"]) == (5, 5, 0)


def test_the_printed_report_carries_the_warnings():
    text = format_trace_report(trace_report(_rows(60, score=0.0)))
    assert "warning:" in text
    assert "score" in text
    fractional = format_trace_report(trace_report(_rows(3, reward=0.75)))
    assert "labelled but not 0/1 3" in fractional
