"""What a first run with your own agent hits: the writer's key, messages, pairs."""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, scripted_agent


def test_callable_agent_without_key_is_told_about_the_offline_writer(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as err:
        wai.simulate(scripted_agent, tools=TOOLS, system_prompt=POLICY, budget=2)
    text = str(err.value)
    assert "VLLM_API_KEY" in text
    assert "simulator=False" in text
    assert "openai:" in text


def test_rows_carry_messages_in_memory():
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
    for row in data.trajectories:
        messages = row["messages"]
        assert messages and messages[0]["role"] == "user"
        assert messages[0]["content"] == row["prompt"]
        assert any(m["role"] == "assistant" for m in messages)


def test_export_preference_on_plain_rows_names_the_pair_builder(tmp_path):
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        budget=4,
        seed=0,
        simulator=False,
        grade="conduct",
        time_budget=None,
        advanced={"per_round": 8, "mutate_failures": False},
    )
    with pytest.raises(ValueError, match="build_preference_pairs"):
        wai.export_preference(data.trajectories, str(tmp_path / "pref.jsonl"))
