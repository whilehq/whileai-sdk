"""Every row says which model wrote the prompt, played the user, and graded
it, and a run whose agent did those jobs itself says so out loud."""

from __future__ import annotations

import json

import pytest

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, scripted_agent, simulate_offline
from whileai.simulations import schema
from whileai.simulations.data import export_row
from whileai.simulations.export import training_rows
from whileai.simulations.generate.agents import local_model


def test_offline_rows_and_metadata_say_who_wrote_what(tmp_path):
    data = simulate_offline(budget=4, repeats=1)
    assert data.trajectories
    for row in data.trajectories:
        # no model wrote offline: the built-in template writer did
        assert row["writer_model"] == "template"
        assert row["model_version"] == "scripted_agent"
        # a callable agent takes one message; nobody played the user
        assert row["user_model"] is None
    assert data.writer_model == "template"
    assert data.user_model is None
    assert data.metadata["writer_model"] == "template"
    assert data.metadata["user_model"] is None
    assert data.metadata["judge_model"] is None
    assert "same_model" not in data.degraded
    # the one note an offline run without seeds= always carries
    assert [w for w in data.warnings if "generic template" not in w] == []

    path = str(tmp_path / "run.jsonl")
    data.save(path, meta=True)
    with open(path) as fh:
        on_disk = json.loads(fh.readline())
    assert on_disk["writer_model"] == "template"
    assert "user_model" not in on_disk  # None never ships
    with open(str(tmp_path / "run.meta.json")) as fh:
        meta = json.load(fh)
    assert meta["writer_model"] == "template"
    assert meta["user_model"] is None
    assert [w for w in meta["warnings"] if "generic template" not in w] == []


def test_provenance_survives_export_and_the_typed_round_trip():
    row = {
        "prompt": "refund order ORD-1",
        "steps": [],
        "final_text": "done",
        "scenario_id": "sc-1",
        "model_version": "agent-model",
        "policy_version": "agent-model@abc",
        "writer_model": "writer-model",
        "user_model": "user-model",
        "reward": 1,
    }
    out = export_row(dict(row))
    assert out["writer_model"] == "writer-model"
    assert out["user_model"] == "user-model"
    back = schema.to_row(*schema.from_row(schema.stamp(dict(row))))
    assert back["writer_model"] == "writer-model"
    assert back["user_model"] == "user-model"
    trained = training_rows([dict(row)], system_prompt=POLICY, tools=TOOLS)
    assert trained and trained[0]["writer_model"] == "writer-model"
    assert trained[0]["user_model"] == "user-model"


def test_judge_model_is_read_off_judge_meta():
    data = simulate_offline(budget=2, repeats=1)
    data.trajectories[0]["judge_meta"] = {"model": "microsoft/phi-4", "version": "x@y"}
    assert data.judge_model == "microsoft/phi-4"
    assert data.metadata["judge_model"] == "microsoft/phi-4"


def test_user_model_reaches_the_user_simulator(monkeypatch):
    """simulate(user_model=) lands on the rollout loop, and the loop sends
    the user's turns to that model, not the agent's."""
    seen: list[dict] = []

    def fake_local(url, model, **kwargs):
        seen.append({"url": url, "model": model, "user_model": kwargs.get("user_model")})

        def agent(message):
            return {"steps": [], "final_text": "ok"}

        return agent

    monkeypatch.setattr("whileai.simulations.run.engine.local_model", fake_local)
    data = wai.simulate(
        tools=TOOLS,
        policy=POLICY,
        backend="vllm:agent-model@http://127.0.0.1:9",
        user_model="vllm:user-model@http://127.0.0.1:8",
        budget=2,
        repeats=1,
        grade=False,
        simulator=False,
        seed=0,
        time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert seen and seen[0]["user_model"] == "vllm:user-model@http://127.0.0.1:8"
    assert data.user_model == "user-model"
    assert all(t["user_model"] == "user-model" for t in data.trajectories)

    calls: list[tuple[str, str, str]] = []

    def fake_complete(url, model, messages, **kwargs):
        calls.append((url, model, messages[0]["role"] + ":" + messages[0]["content"][:40]))
        return {"content": "The order number is ORD-77."}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    agent = local_model(
        "http://127.0.0.1:9",
        "agent-model",
        tools=TOOLS,
        system=POLICY,
        user_model="vllm:user-model@http://127.0.0.1:8",
        max_turns=4,
        min_user_turns=2,
    )
    agent("Refund order ORD-1 please.")
    agent_calls = [c for c in calls if c[1] == "agent-model"]
    user_calls = [c for c in calls if c[1] == "user-model"]
    assert agent_calls and all(c[0] == "http://127.0.0.1:9" for c in agent_calls)
    assert user_calls and all(c[0] == "http://127.0.0.1:8" for c in user_calls)


def test_user_model_must_be_a_spec_string():
    with pytest.raises(TypeError, match="user_model="):
        simulate_offline(user_model=3)


def test_same_model_is_said_out_loud(monkeypatch):
    """A bring-your-own agent spec is the writer too by default, and plays
    the user: the run flags it and names the calls that separate them."""

    def fake_local(url, model, **kwargs):
        def agent(message):
            return {"steps": [], "final_text": "ok"}

        return agent

    def fake_writer(_url, _model, _messages, **kwargs):
        idx = len(written)
        written.append(idx)
        return {"content": json.dumps([{"region_id": None, "message": f"check order {idx}"}])}

    written: list[int] = []
    monkeypatch.setattr("whileai.simulations.generate.adapters.local_model", fake_local)
    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_writer)
    data = wai.simulate(
        agent="vllm:one-model@http://127.0.0.1:9",
        tools=TOOLS,
        policy=POLICY,
        budget=3,
        repeats=1,
        grade=False,
        seed=0,
        time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert data.writer_model == "one-model"
    assert data.user_model == "one-model"
    assert "same_model" in data.degraded
    # the run also notes that the fake agent never called a tool; find ours
    same = [w for w in data.warnings if "wrote the situations" in w]
    assert len(same) == 1
    note = same[0]
    assert "wrote the situations" in note and "played the user" in note
    assert "simulator=" in note and "user_model=" in note

    # a separate user model clears that half of the note
    data2 = wai.simulate(
        agent="vllm:one-model@http://127.0.0.1:9",
        tools=TOOLS,
        policy=POLICY,
        user_model="vllm:other-model@http://127.0.0.1:8",
        budget=3,
        repeats=1,
        grade=False,
        seed=0,
        time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert data2.user_model == "other-model"
    assert "same_model" in data2.degraded
    note2 = next(w for w in data2.warnings if "wrote the situations" in w)
    assert "played the user" not in note2
    assert "user_model=" not in note2


def test_offline_callable_agent_is_never_flagged():
    data = simulate_offline(agent=scripted_agent, budget=2, repeats=1)
    assert "same_model" not in data.degraded
