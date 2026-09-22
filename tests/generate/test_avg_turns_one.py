"""``avg_turns=1`` is one user line and one reply, the same as ``max_turns=1``.

Before this, #587 guarded ``max_turns=1`` in ``_want_followup`` but the
budget ``avg_turns`` draws through ``sample_turn_budget`` never went under
2, so a model-backed agent at ``avg_turns=1`` got two user messages and two
agent calls on 12 of 12 prompts, and a reply ending in "?" earned a
follow-up the agent model wrote itself. A callable agent takes one message
whatever the knob says, which is why the skill and the sweep looked fine.
"""

from __future__ import annotations

import whileai.simulations as wai
from tests.helpers import TOOLS, offline
from whileai.simulations.generate.agents import local_model
from whileai.simulations.generate.diversity import sample_turn_budget

BACKEND = "vllm:agent-model@http://127.0.0.1:9"
QUESTIONS = ["Which order was that, and what did you pay?", "Was it the blue lamp or the desk?"]


def _asking_agent(calls: dict):
    """The agent asks a fresh question every turn (a repeat would trip the
    echo guard and hide the second turn); the user-sim always has an answer."""

    def fake_complete(_url, _model, messages, **kwargs):
        if kwargs.get("tools"):
            calls["agent"] = calls.get("agent", 0) + 1
            asked = sum(1 for m in messages if m.get("role") == "assistant")
            return {"content": QUESTIONS[asked % len(QUESTIONS)]}
        calls["user"] = calls.get("user", 0) + 1
        return {"content": "ORD-7, the blue lamp, twenty dollars"}

    return fake_complete


def _user_lines(row: dict) -> int:
    return 1 + sum(1 for s in row["steps"] if isinstance(s, dict) and s.get("user"))


def test_sample_turn_budget_at_or_under_one_is_one_and_above_one_keeps_the_floor():
    keys = [f"prompt {i}" for i in range(50)]
    assert {sample_turn_budget(0, k, 12, avg_turns=1) for k in keys} == {1}
    assert {sample_turn_budget(0, k, 12, avg_turns=0.5) for k in keys} == {1}
    # Above 1 the floor stays: user, agent is the shortest thread.
    assert min(sample_turn_budget(0, k, 12, avg_turns=2) for k in keys) == 2
    assert min(sample_turn_budget(0, k, 12, avg_turns=1.5) for k in keys) == 2


def test_local_model_at_avg_turns_one_never_writes_a_second_user_line(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr("whileai.simulations.generate.agents.complete", _asking_agent(calls))
    agent = local_model("http://example", "m", tools=TOOLS, avg_turns=1)
    rows = [agent(f"refund order ORD-{i}, it arrived broken") for i in range(12)]
    assert [_user_lines(r) for r in rows] == [1] * 12
    assert calls == {"agent": 12}, calls
    assert all(r["final_text"] == QUESTIONS[0] for r in rows)


def test_simulate_with_a_backend_at_avg_turns_one_is_single_turn(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr("whileai.simulations.generate.agents.complete", _asking_agent(calls))
    data = wai.simulate(backend=BACKEND, budget=12, avg_turns=1, **offline())
    rows = list(data.trajectories)
    assert len(rows) == 12
    assert [_user_lines(r) for r in rows] == [1] * 12
    assert calls == {"agent": 12}, calls
