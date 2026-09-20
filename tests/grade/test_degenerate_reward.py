"""A reward that scores every row the same says so: on the pass_at line, in
the graded run's warnings, and through the log. A confident
``pass@1 0.00 [0.00..0.00]`` on a tool-calling agent is believable, and it
sent a researcher to the agent when the checker was reading the wrong tool
call shape (#594)."""

from __future__ import annotations

import logging

import whileai as wai
import whileai.simulations as sim
from tests.helpers import simulate_offline
from whileai.simulations.score.passat import degenerate_note

SENTENCE = "check the judge before reading this number"


def _rows(rewards):
    return [{"task_id": f"t{i // 4}", "reward": r} for i, r in enumerate(rewards)]


def test_all_zero_pass_at_carries_the_note():
    rates = wai.pass_at(_rows([0] * 16))
    assert rates.pass_at_1 == 0.0
    assert rates.note.startswith("every row scored 0 (16 of 16); " + SENTENCE)
    assert SENTENCE in str(rates)
    assert rates.to_dict()["note"] == rates.note


def test_all_one_pass_at_carries_the_note():
    rates = wai.pass_at(_rows([1] * 16))
    assert rates.pass_at_1 == 1.0
    assert rates.note.startswith("every row scored 1 (16 of 16); " + SENTENCE)


def test_one_partial_value_on_every_row_is_named_too():
    rates = wai.pass_at(_rows([0.5] * 8))
    assert rates.pass_at_1 is None
    assert "every row scored 0.50 (8 of 8)" in rates.note
    assert "grade first" in rates.note


def test_mixed_rewards_do_not_warn():
    rates = wai.pass_at(_rows([1, 0, 1, 1] * 4))
    assert SENTENCE not in rates.note
    assert SENTENCE not in str(rates)
    assert sim.coverage_warnings(_rows([1, 0, 1, 1] * 4)) == []
    assert degenerate_note(_rows([1, 0])) == ""
    # one graded row cannot be unanimous, and ungraded rows are not counted
    assert degenerate_note(_rows([0])) == ""
    assert degenerate_note([{"task_id": "a"}, {"task_id": "b"}]) == ""


def test_coverage_warnings_name_a_unanimous_run():
    notes = sim.coverage_warnings(_rows([0] * 4))
    assert notes == ["every row scored 0 (4 of 4); " + SENTENCE]
    notes = sim.coverage_warnings(_rows([1.0] * 4))
    assert notes == ["every row scored 1 (4 of 4); " + SENTENCE]


def test_grade_judge_path_warns_and_does_not_raise(caplog):
    data = simulate_offline(budget=8)
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        scored = data.grade(judge=lambda row: 0.0)
    assert any(SENTENCE in note for note in scored.warnings)
    assert any(SENTENCE in rec.getMessage() for rec in caplog.records)
    assert SENTENCE in str(wai.pass_at(scored.rows))


def test_grade_grader_path_warns_and_does_not_raise(caplog):
    data = simulate_offline(budget=8)
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        data.grade(lambda row: 1.0)
    assert any(SENTENCE in note for note in data.warnings)
    assert any(SENTENCE in rec.getMessage() for rec in caplog.records)
    assert "every row scored 1" in str(data.pass_at)


def test_grade_with_a_reward_that_varies_stays_quiet(caplog):
    data = simulate_offline(budget=8)
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        scored = data.grade(judge=lambda row: float(len(row.get("steps") or []) > 1))
    assert {r["reward"] for r in scored.rows} == {0.0, 1.0}
    assert not any(SENTENCE in note for note in scored.warnings)
    assert not any(SENTENCE in rec.getMessage() for rec in caplog.records)
