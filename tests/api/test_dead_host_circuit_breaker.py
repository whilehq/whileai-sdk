"""A permanently dead host stops the run; a busy one still gets retried (#816).

A stopped Modal app answers 404 forever. Before this the run read as a
stall -- ``0/1000 rollouts, 0 situations written, 7m14s elapsed``, no
error, no non-zero counter, no mention of a 404 anywhere -- because a
retry loop against a dead host is indistinguishable from a slow one.

The negative half is the point of the positive half: a breaker that trips
on a 503 or a timeout is worse than no breaker, because it turns a cold
container into a failed run. Both halves are pinned here.

Naming the dead app in the error text is PR #752's half (``dead_app_error``,
``public_llm_error``). This file tests only the breaker, and covers both
message shapes so it keeps working whichever lands first.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS
from whileai.simulations.run.engine import DEAD_HOST_STRIKES, Run, _dead_host

DEAD_HOST = "example--stopped-vllm-serve.modal.run"
#: What ``complete()`` raises today for a status it has no branch for.
MAIN_404 = f"{DEAD_HOST} returned 404: modal-http: invalid function call"
#: What ``dead_app_error`` raises once PR #752 lands. Same host, same rule.
PR752_404 = f"{DEAD_HOST} is not deployed: Modal has no app 'stopped-vllm-serve'."
#: Measured on origin/main at 5ec23e0 with these same two runs: calls made
#: to a host that answered 404 every time, before the run gave up. Neither
#: run named the host as the cause; the agent one raised nothing at all.
MAIN_AGENT_CALLS = 80
MAIN_WRITER_CALLS = 816


def test_dead_host_reads_the_host_out_of_either_message_shape():
    assert _dead_host(f"<agent error: RuntimeError: {MAIN_404}>") == (DEAD_HOST, "answered 404")
    assert _dead_host(PR752_404) == (DEAD_HOST, "is not deployed")
    assert _dead_host(f"{DEAD_HOST} returned 410: gone") == (DEAD_HOST, "answered 410")


def test_transient_failures_are_not_permanent_ones():
    """The negative, at the unit. 5xx, 429 and timeouts belong to the
    transient ladder; a breaker that reads them is a worse bug than the
    one it fixes."""
    assert _dead_host(f"{DEAD_HOST} returned 503: unavailable") is None
    assert _dead_host(f"{DEAD_HOST} returned 500: internal error") is None
    assert _dead_host(f"{DEAD_HOST} returned 429: rate limited") is None
    assert _dead_host(f"{DEAD_HOST} returned a transient error after retries") is None
    assert _dead_host("<agent error: TimeoutError: timed out>") is None
    assert _dead_host("") is None


def _counting_agent(message: str, calls: list):
    calls.append(message)
    raise RuntimeError(message)


@pytest.mark.parametrize("error", [MAIN_404, PR752_404], ids=["main", "pr752"])
def test_a_dead_agent_host_trips_the_breaker_and_names_the_host(error):
    """The run stops with the host in the message instead of retrying it."""
    calls: list = []

    def agent(_prompt, **_kw):
        return _counting_agent(error, calls)

    t0 = time.monotonic()
    with pytest.raises(RuntimeError) as caught:
        wai.simulate(
            agent,
            tools=TOOLS,
            system_prompt=POLICY,
            simulator=False,
            budget=40,
            time_budget=60,
            advanced={"concurrency": 1},
        )

    message = str(caught.value)
    assert DEAD_HOST in message, "the run has to say which host is gone"
    assert "stopped instead of retrying" in message
    # the fix, named from the call that can change it (style rule 10)
    assert "wai.configure(" in message or "WHILEAI_AGENT" in message
    # main made MAIN_AGENT_CALLS against this host and still raised nothing
    assert len(calls) <= DEAD_HOST_STRIKES + 1, f"kept calling a dead host: {len(calls)} times"
    assert len(calls) < MAIN_AGENT_CALLS
    assert time.monotonic() - t0 < 20.0


def test_a_busy_host_is_still_retried():
    """The negative, end to end. A 503 is a cold container, not a dead app.

    The run must keep going past ``DEAD_HOST_STRIKES`` calls and must not
    stop with the breaker's message. It ends the way a failing agent has
    always ended -- ``agent_failed``, no exception -- and that is the
    behaviour this change must leave alone.
    """
    calls: list = []

    def agent(_prompt, **_kw):
        return _counting_agent(f"{DEAD_HOST} returned a transient error after retries", calls)

    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        budget=40,
        time_budget=30,
        advanced={"concurrency": 1},
    )

    assert len(calls) > DEAD_HOST_STRIKES, "a transient failure must not trip the breaker"
    assert data.stopped_because != "agent_host_dead"
    assert not data.trajectories


def test_a_dead_writer_host_trips_the_breaker(monkeypatch):
    """The path #816 actually walked: the writer, not the agent.

    main does stop here eventually, on the empty-rounds counter, and the
    404 rides along inside the writer's last error. What it does not do is
    say that the host is dead: it blames the writer for producing no
    situations, which is the sentence that sent the reporter looking at
    the writer prompt instead of at a stopped Modal app. The wording is
    the pin.
    """
    calls: list = []

    def dead(*_a, **_kw):
        calls.append(1)
        raise RuntimeError(MAIN_404)

    monkeypatch.setenv("VLLM_API_KEY", "not-a-key-offline-test")
    monkeypatch.setattr("whileai.simulations.generate.generator.complete", dead)
    monkeypatch.setattr("whileai.simulations.generate.agents.complete", dead)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError) as caught:
        wai.simulate(
            tools=TOOLS,
            system_prompt=POLICY,
            budget=20,
            time_budget=60,
            advanced={"concurrency": 1},
        )
    message = str(caught.value)
    assert DEAD_HOST in message
    assert "stopped instead of retrying" in message
    assert "produced no situations" not in message
    # a wave is several calls, so this counts waves only loosely; what it
    # pins is the order of magnitude against main's MAIN_WRITER_CALLS
    assert len(calls) < MAIN_WRITER_CALLS // 10, f"kept calling a dead writer: {len(calls)}"
    assert time.monotonic() - t0 < 20.0


def test_a_host_that_recovers_keeps_no_strikes():
    """Strikes are consecutive. One 404 during a redeploy is not a dead app.

    This is why the threshold is not one: a host swapping its route
    mid-deploy answers a single 404 and then serves. It is also why the
    count is per host -- a dead judge must not condemn a live agent.
    """
    calls: list = []

    def agent(message: str) -> dict:
        calls.append(message)
        if len(calls) % DEAD_HOST_STRIKES != 0:
            raise RuntimeError(MAIN_404)
        return {"final_text": "Done.", "steps": [{"role": "assistant", "text": "Done."}]}

    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        budget=4,
        time_budget=30,
        advanced={"concurrency": 1},
    )
    assert data.trajectories, "a host that answers between 404s must not trip the breaker"
    assert data.stopped_because != "agent_host_dead"


def test_strikes_are_counted_per_host():
    """One 404 each from three different hosts is three live hosts."""
    run = SimpleNamespace(dead_host_hits={}, dead_host_reason={}, dead_host_side={})
    for host in ("a.modal.run", "b.modal.run", "c.modal.run"):
        assert Run._strike_dead_host(run, f"{host} returned 404: gone", "agent") is None
    assert run.dead_host_hits == {"a.modal.run": 1, "b.modal.run": 1, "c.modal.run": 1}

    for _ in range(DEAD_HOST_STRIKES - 2):
        assert Run._strike_dead_host(run, "a.modal.run returned 404: gone", "agent") is None
    note = Run._strike_dead_host(run, "a.modal.run returned 404: gone", "agent")
    assert note and "a.modal.run" in note and "b.modal.run" not in note
