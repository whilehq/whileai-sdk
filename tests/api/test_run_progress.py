"""A long run says where it is, and the hosted writer is a named value.

A tester ran 64 rollouts through the hosted writer, saw nothing for eight
minutes, and nearly killed the process. These pin the progress line, how
often it may appear, and ``simulator="hosted"`` as the spelled-out
default, so the way back from the offline writer is a value and not
"delete the argument".
"""

from __future__ import annotations

import logging

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, offline, scripted_agent
from whileai.simulations.defaults import RunKnobs
from whileai.simulations.run import engine
from whileai.simulations.run.config import resolve_run_config, writer_spec_for


class _FakeRun:
    """Just the state ``Run._note_progress`` reads, with a fake clock."""

    def __init__(self, cap: int = 64) -> None:
        self.now = 0.0
        self.started = 0.0
        self.c = type("Cfg", (), {"cap": cap})()
        self.data = type("Data", (), {"trajectories": []})()
        self.generated_pool: list[str] = []
        self.progress_on = cap >= engine.PROGRESS_MIN_BUDGET
        self.progress_every_s = RunKnobs().progress_every_s
        self.progress_every_rows = RunKnobs().progress_every_rows
        self.progress_rows = 0
        self.progress_at = 0.0
        self.progress_clock = lambda: self.now
        # the #470 counters the line and the callback read
        self.on_progress = None
        self.inflight: dict = {}
        self.landed = 0
        self.resumed = 0
        self.timed_out = 0
        self.rerolled_by: dict[str, int] = {}
        self.lost_by: dict[str, int] = {}
        self.cap_lifted = {"lifted": False, "lost": 0}

    note = engine.Run._note_progress
    _progress = engine.Run._progress

    def tick(self, seconds: float = 0.0, rows: int = 0, situations: int = 0) -> None:
        self.now += seconds
        self.data.trajectories.extend({"row": 1} for _ in range(rows))
        self.landed += rows
        self.generated_pool.extend("ask" for _ in range(situations))


def _lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "whileai.simulations"]


# ------------------------------------------------------------ the line


def test_progress_line_reads_like_the_example():
    assert (
        engine.progress_line(12, 64, 3, 100.0)
        == "12/64 rollouts, 3 situations written, 1m40s elapsed, ~7m left"
    )


def test_no_estimate_before_five_rollouts():
    line = engine.progress_line(4, 64, 2, 20.0)
    assert line == "4/64 rollouts, 2 situations written, 20s elapsed"
    assert "left" not in line


def test_the_last_line_has_no_estimate_to_make():
    assert engine.progress_line(64, 64, 9, 480.0) == (
        "64/64 rollouts, 9 situations written, 8m elapsed"
    )


def test_spans_are_short_words():
    assert engine._clock_text(0) == "0s"
    assert engine._clock_text(45) == "45s"
    assert engine._clock_text(60) == "1m"
    assert engine._clock_text(100) == "1m40s"
    assert engine._clock_text(3600) == "1h"
    assert engine._clock_text(3900) == "1h5m"
    # the estimate rounds: nobody reads a forecast to the second
    assert engine._left_text(29.4) == "29s"
    assert engine._left_text(310) == "5m"
    assert engine._left_text(0.1) == "1s"


# ------------------------------------------------------- how often it says it


def test_a_line_every_ten_seconds_even_with_no_new_rows(caplog):
    run = _FakeRun()
    with caplog.at_level(logging.INFO, logger="whileai.simulations"):
        run.tick(seconds=9.0)
        run.note()
        assert not _lines(caplog)
        run.tick(seconds=1.5)
        run.note()
    assert len(_lines(caplog)) == 1
    assert "0/64 rollouts" in _lines(caplog)[0]


def test_a_line_every_ten_rollouts_even_in_the_same_second(caplog):
    run = _FakeRun()
    with caplog.at_level(logging.INFO, logger="whileai.simulations"):
        run.tick(seconds=1.0, rows=9, situations=2)
        run.note()
        assert not _lines(caplog)
        run.tick(seconds=0.2, rows=1)
        run.note()
    assert _lines(caplog) == ["10/64 rollouts, 2 situations written, 1s elapsed, ~6s left"]


def test_a_small_budget_stays_quiet(caplog):
    run = _FakeRun(cap=8)
    with caplog.at_level(logging.INFO, logger="whileai.simulations"):
        run.tick(seconds=60.0, rows=8)
        run.note(force=True)
    assert not _lines(caplog)


# --------------------------------------------------------------- end to end


def test_an_offline_run_logs_progress_and_no_print(capsys, caplog):
    with caplog.at_level(logging.INFO, logger="whileai.simulations"):
        data = wai.simulate(scripted_agent, **offline(budget=12, tools=TOOLS, policy=POLICY))
    assert len(data.trajectories) == 12
    progress = [line for line in _lines(caplog) if "rollouts," in line]
    assert progress, _lines(caplog)
    assert progress[-1].startswith("12/12 rollouts, ")
    # the offline writer is not the hosted one, and nothing is printed
    assert not [line for line in _lines(caplog) if "hosted writer" in line]
    assert capsys.readouterr().out == ""


# ------------------------------------------------------- simulator="hosted"


def test_hosted_is_the_default_written_out():
    assert writer_spec_for(None, "hosted") is None
    assert writer_spec_for(None, None) is None
    # and it inherits a string agent's backend, exactly as the default does
    assert writer_spec_for("openai:gpt-4.1-mini", "hosted") == "openai:gpt-4.1-mini"
    assert writer_spec_for(None, "Hosted ") is None
    # anything else is still taken literally
    assert writer_spec_for(None, "openai:gpt-4.1-mini") == "openai:gpt-4.1-mini"
    assert writer_spec_for(None, False) is False


def test_hosted_reaches_the_config_as_none():
    hosted = resolve_run_config(
        None, tools=TOOLS, system_prompt=POLICY, passed={"simulator": "hosted"}
    )
    assert hosted.simulator is None
    off = resolve_run_config(None, tools=TOOLS, system_prompt=POLICY, passed={"simulator": False})
    assert off.simulator is False
