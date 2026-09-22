"""The trial warning lands before a hosted run spends the allowance, not after."""

from __future__ import annotations

import json
import logging

import pytest

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, scripted_agent
from whileai import auth

LINE = (
    "trial key: the hosted writer covers about 12 situations a day (25k input tokens); "
    "simulator=False writes them offline with no quota; sign in once at "
    f"{auth.SIGN_IN_URL} to lift it"
)


def _save(home, **fields) -> None:
    home.mkdir(parents=True, exist_ok=True)
    payload = {"api_key": "zp_" + "b" * 48, "api_url": auth.DEFAULT_API_URL}
    payload.update(fields)
    (home / "credentials.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "whileai-home"
    monkeypatch.setenv("WHILEAI_HOME", str(path))
    return path


def _run_against_blocked_writer() -> None:
    """Start a hosted run whose writer conftest has blocked.

    The note under test is logged before the writer's first call. What
    happens after is a race between the one-second clock and the engine's
    empty-writer guard: the clock wins and the run returns empty, or eight
    empty rounds win first and the engine raises "hosted Qwen produced no
    situations". Both are the right answer to a blocked writer, so the
    tests read the note off the log and accept either ending.
    """
    try:
        wai.simulate(
            scripted_agent,
            tools=TOOLS,
            system_prompt=POLICY,
            budget=2,
            seed=0,
            grade=False,
            time_budget=1,
        )
    except RuntimeError as err:
        assert "produced no situations" in str(err)


def test_note_reads_the_recorded_tier(home):
    _save(home, tier="trial", daily_input_tokens=25000, expires_at="2026-09-21T00:00:00.000Z")
    assert auth.trial_prerun_note() == LINE


def test_a_full_key_says_nothing(home):
    _save(home, tier="full")
    assert auth.trial_prerun_note() is None


def test_a_key_from_the_environment_says_nothing(home, monkeypatch):
    # the file describes the saved account, not whatever key the variable
    # holds, so its tier must not be read onto that key
    _save(home, tier="trial", daily_input_tokens=25000)
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_env")
    assert auth.trial_prerun_note() is None


def test_allowance_from_the_file_sets_the_count(home):
    _save(home, tier="trial", daily_input_tokens=100_000)
    note = auth.trial_prerun_note()
    assert note and "about 50 situations a day (100k input tokens)" in note


def test_the_run_warns_before_the_hosted_writer_starts(home, caplog):
    _save(home, tier="trial", daily_input_tokens=25000)
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        _run_against_blocked_writer()
    assert LINE in caplog.text


def test_the_offline_writer_has_no_quota_to_warn_about(home):
    _save(home, tier="trial", daily_input_tokens=25000)
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        budget=4,
        seed=0,
        simulator=False,
        grade=False,
        time_budget=None,
        advanced={"per_round": 8, "mutate_failures": False},
    )
    assert data.trajectories
    assert not [w for w in data.warnings if "trial key" in w]


def test_a_full_key_runs_without_the_note(home, caplog):
    _save(home, tier="full")
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        _run_against_blocked_writer()
    assert "trial key" not in caplog.text


def test_vllm_api_key_no_longer_diverts_the_writer_off_the_account(home, monkeypatch, caplog):
    # Until 2026-09-21 VLLM_API_KEY routed the writer to a shared, unmetered
    # pool, so the trial allowance said nothing. That pool is gone and
    # While's hosts take a zp_ key only, so the writer stays on the account
    # route and the trial note has to fire even with VLLM_API_KEY set.
    _save(home, tier="trial", daily_input_tokens=25000)
    monkeypatch.setenv("VLLM_API_KEY", "pool-key")
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        _run_against_blocked_writer()
    assert "trial key" in caplog.text
