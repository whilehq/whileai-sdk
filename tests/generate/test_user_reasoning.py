"""#284: the simulated user's reasoning never becomes user speech, an
unclosed block is dropped whole, and ``local_model(thinking=False)``
reaches the user simulator."""

from __future__ import annotations

import threading

import whileai.simulations as wai
from tests.generate.test_logprobs import POLICY, TOOLS
from whileai.simulations.generate.agents import (
    _MIN_SPOKEN,
    _user_followup,
    local_model,
    split_reasoning,
    user_sim_system,
)

REASONING = (
    "<think> okay, the user is confused because the customer says they saw the order "
    "confirm on their dashboard on 2023-05-15, but the system shows 2023-05-14, so the "
    "right move is to point at the dashboard date and ask for a recheck </think>"
)
SPOKEN = "the dashboard shows 2023-05-15 for order 4412, can you check that date again"
CUT_OFF = "<think> okay, the user says they saw the order confirm on 2023-05-15 but the system"


def _stats() -> dict:
    return {"n": 0, "sum": 0, "lock": threading.Lock()}


def _is_user_sim_call(messages: list[dict], kwargs: dict) -> bool:
    return (
        kwargs.get("tools") is None
        and bool(messages)
        and messages[0].get("role") == "system"
        and messages[0].get("content") == user_sim_system(TOOLS)
    )


def test_split_reasoning_counts_words_not_markup():
    assert split_reasoning(f"{REASONING} {SPOKEN}") == (SPOKEN, 1, False)
    # an unclosed block is the cap landing inside the reasoning: nothing
    # after the tag is a reply, so it all goes
    assert split_reasoning(f"first line. {CUT_OFF}") == ("first line. ", 0, True)
    assert split_reasoning(CUT_OFF) == ("", 0, True)
    # an empty block is what a reasoning-suppressed model prints; markup only
    assert split_reasoning("<think> </think> yes") == ("yes", 0, False)
    assert split_reasoning("order 4821") == ("order 4821", 0, False)


def _followup(monkeypatch, replies: list[str], *, agent_text: str, stats: dict | None = None):
    calls: list[dict] = []

    def fake_complete(_url, _model, messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return {"content": replies[min(len(calls) - 1, len(replies) - 1)]}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    text = _user_followup(
        "http://127.0.0.1:9",
        "m",
        "where is order 4412",
        agent_text,
        api_key=None,
        timeout=30,
        tools=TOOLS,
        turn_stats=stats,
    )
    return text, calls


def test_closed_reasoning_is_stripped_before_it_becomes_a_user_turn(monkeypatch):
    stats = _stats()
    text, calls = _followup(
        monkeypatch, [f"{REASONING} {SPOKEN}"], agent_text="Which date do you see?", stats=stats
    )
    assert "<think>" not in text and "okay, the user" not in text
    assert "2023-05-15" in text
    assert len(calls) == 1
    assert stats["user_think_stripped"] == 1 and stats.get("user_think_unclosed", 0) == 0


def test_unclosed_tail_is_dropped_and_the_turn_retried(monkeypatch):
    stats = _stats()
    text, calls = _followup(monkeypatch, [CUT_OFF, SPOKEN], agent_text="Ok, noted.", stats=stats)
    assert text and "<think>" not in text and "okay, the user" not in text
    assert "2023-05-15" in text
    # one retry for the reasoning-only turn, on a prompt that was not
    # going to retry at all
    assert len(calls) == 2
    assert stats["user_think_stripped"] == 1 and stats["user_think_unclosed"] == 1


def test_reasoning_with_no_spoken_line_is_no_followup_not_a_fragment(monkeypatch):
    stats = _stats()
    text, calls = _followup(
        monkeypatch, [f"{REASONING} ok", CUT_OFF], agent_text="Ok, noted.", stats=stats
    )
    assert text == ""
    assert len(calls) == 2
    assert len("ok") < _MIN_SPOKEN
    assert stats["user_think_stripped"] == 2 and stats["user_think_unclosed"] == 1


def test_the_floor_applies_only_to_turns_that_carried_reasoning(monkeypatch):
    asked = "Shall I proceed with the refund for order 4412? (yes/no)"
    assert _followup(monkeypatch, ["yes"], agent_text=asked)[0] == "yes"
    # the empty block a reasoning-suppressed model prints is not reasoning
    assert _followup(monkeypatch, ["<think> </think> yes"], agent_text=asked)[0] == "yes"
    assert _followup(monkeypatch, ["order 4821"], agent_text="Which order?")[0] == "order 4821"


def _local(monkeypatch, **kw):
    seen: list[dict] = []

    def fake_complete(_url, _model, messages, **kwargs):
        seen.append({"user_sim": _is_user_sim_call(messages, kwargs), "extra": kwargs.get("extra")})
        if kwargs.get("tools"):
            n_user = sum(1 for m in messages if m.get("role") == "user")
            if n_user < 2:
                return {"content": "Which order do you mean?", "_finish_reason": "stop"}
            return {"content": "Order 4412 shipped yesterday.", "_finish_reason": "stop"}
        return {"content": "order 4412, the one from tuesday"}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    agent = local_model(
        "http://127.0.0.1:9",
        "agent-model",
        tools=TOOLS,
        system=POLICY,
        max_turns=4,
        min_user_turns=2,
        **kw,
    )
    agent("where is order 4412")
    return seen


def test_thinking_false_reaches_the_user_simulator(monkeypatch):
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    seen = _local(monkeypatch, thinking=False)
    user_calls = [c for c in seen if c["user_sim"]]
    assert user_calls, "the agent asked a question, so the user simulator was called"
    assert all(c["extra"] == off for c in user_calls)
    assert all(c["extra"] == off for c in seen if not c["user_sim"])
    # a user model on another endpoint keeps that server's default
    seen = _local(monkeypatch, thinking=False, user_model="vllm:user-model@http://127.0.0.1:8")
    user_calls = [c for c in seen if c["user_sim"]]
    assert user_calls and all(c["extra"] is None for c in user_calls)
    # with nothing asked, nothing is sent, as before
    seen = _local(monkeypatch)
    assert all(c["extra"] is None for c in seen)


def test_run_counts_stripped_user_turns_per_arm_and_names_the_fix(monkeypatch):
    calls = {"user_sim": 0}

    def fake_complete(_url, _model, messages, **kwargs):
        if kwargs.get("tools"):
            n_user = sum(1 for m in messages if m.get("role") == "user")
            if n_user < 2:
                return {"content": "Which order do you mean?", "_finish_reason": "stop"}
            return {"content": "Order 4412 shipped yesterday.", "_finish_reason": "stop"}
        if _is_user_sim_call(messages, kwargs):
            # the first try is cut off inside the block; the retry speaks
            calls["user_sim"] += 1
            return {"content": CUT_OFF if calls["user_sim"] == 1 else f"{REASONING} {SPOKEN}"}
        return {"content": ""}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 4
    )
    data = wai.simulate(
        tools=TOOLS,
        policy=POLICY,
        extra_situations=["where is order 4412"],
        budget=1,
        grade=False,
        concurrency=1,
        simulator=False,
        backend="vllm:fake@http://127.0.0.1:9",
        seed=0,
        time_budget=None,
        max_turns=4,
        avg_turns=2,
        advanced={"per_round": 2, "mutate_failures": False},
    )
    assert data.trajectories
    for row in data.trajectories:
        for step in row["steps"]:
            assert "<think>" not in str(step.get("user") or "")
        for message in row.get("messages") or []:
            if message.get("role") == "user":
                assert "<think>" not in str(message.get("content") or "")
    assert data.search["user_think_stripped"] == 2
    assert data.search["user_think_unclosed"] == 1
    note = next(w for w in data.warnings if "simulated-user turns came back as <think>" in w)
    assert "thinking=False" in note and "user_model=" in note
    assert f"({data.search['user_think_unclosed']} cut off" in note
