"""Sampling facts on every row: policy_version, sampling, token_logprobs
(async RL, Noukhovitch et al. 2024, arXiv:2410.18252), and staleness_report."""

from __future__ import annotations

import hashlib

import pytest

import whileai.simulations as wai
from tests.generate.test_logprobs import POLICY, TOOLS, _simulate
from tests.helpers import simulate_offline
from whileai.simulations import schema
from whileai.simulations.export import training_rows
from whileai.simulations.generate.agents import LOCAL_MODEL_TEMPERATURE, reply_budget
from whileai.simulations.score.logprobs import staleness_report


def test_model_backed_rows_carry_policy_version_sampling_and_token_logprobs(monkeypatch):
    data, _calls = _simulate(monkeypatch, logprobs="tokens")
    row = data.trajectories[0]
    assert row["sampling"] == {
        "temperature": LOCAL_MODEL_TEMPERATURE,
        "max_tokens": reply_budget(),
        "model": "fake",
    }
    assert row["policy_version"].startswith(row["model_version"] + "@")
    assert len(row["policy_version"].split("@")[-1]) == 16
    # the fake backend returns summed logprobs only; the tokens list stays absent
    assert "token_logprobs" not in row
    exported = data.rows()[0]
    assert exported["sampling"] == row["sampling"]
    assert exported["policy_version"] == row["policy_version"]
    trained = training_rows(data)[0]
    assert trained["policy_version"] == row["policy_version"]
    assert trained["sampling"] == row["sampling"]
    assert schema.validate(exported) == []


def test_temperature_and_reply_budget_knobs_are_recorded(monkeypatch):
    data, _ = _simulate(monkeypatch, temperature=0.3, agent_max_tokens=4096)
    row = data.trajectories[0]
    assert row["sampling"] == {"temperature": 0.3, "max_tokens": 4096, "model": "fake"}


def test_callable_agent_rows_say_sampling_is_unknown_unless_told():
    data = simulate_offline(tools=TOOLS, policy=POLICY, budget=4, per_round=4, concurrency=1)
    row = data.trajectories[0]
    assert "sampling" in row and row["sampling"] is None
    assert row["policy_version"].startswith(row["model_version"] + "@")
    # the exported row drops the None the way it drops every absent fact
    assert "sampling" not in data.rows()[0]

    told = {"temperature": 0.7, "max_tokens": 1024, "model": "my-agent"}
    data = simulate_offline(
        tools=TOOLS, policy=POLICY, budget=4, per_round=4, concurrency=1, sampling=told
    )
    assert data.trajectories[0]["sampling"] == told
    assert data.rows()[0]["sampling"] == told
    assert training_rows(data)[0]["sampling"] == told


def test_sampling_knob_must_be_a_dict():
    with pytest.raises(ValueError, match="sampling= is a dict"):
        simulate_offline(tools=TOOLS, policy=POLICY, budget=1, sampling="0.7")


def _tokens_complete_factory(calls: dict):
    """A backend that returns per-token logprobs on every turn."""

    def fake_complete(_url, _model, messages, **kwargs):
        calls["n"] = calls.get("n", 0) + 1
        want = kwargs.get("logprobs") == "tokens"
        if kwargs.get("tools") and calls["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"order_id": "4412"}'},
                    }
                ],
                **({"_logprobs": {"sum": -1.0, "n": 2, "tokens": [-0.4, -0.6]}} if want else {}),
                "_finish_reason": "tool_calls",
            }
        if kwargs.get("tools"):
            return {
                "content": "Order 4412 shipped yesterday.",
                **({"_logprobs": {"sum": -0.5, "n": 1, "tokens": [-0.5]}} if want else {}),
                "_finish_reason": "stop",
            }
        return {"content": ""}

    return fake_complete


def test_token_logprobs_roll_up_from_steps_in_order(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.complete", _tokens_complete_factory(calls)
    )
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
        logprobs="tokens",
        advanced={"per_round": 2, "mutate_failures": False},
    )
    row = data.trajectories[0]
    assert row["token_logprobs"] == [-0.4, -0.6, -0.5]
    assert row["logprob"] == -1.5 and row["n_tokens"] == 3
    assert row["sampling"]["model"] == "fake"
    exported = data.rows()[0]
    assert exported["token_logprobs"] == [-0.4, -0.6, -0.5]
    assert training_rows(data)[0]["token_logprobs"] == [-0.4, -0.6, -0.5]
    assert wai.staleness_report([exported])["token_logprob_coverage"] == 1.0


def test_policy_version_round_trips_through_the_schema():
    row = {
        "prompt": "where is order 4412",
        "steps": [],
        "final_text": "shipped",
        "scenario_id": "s1",
        "model_version": "fake",
        "policy_version": "fake@" + hashlib.sha256(b"p").hexdigest()[:16],
        "sampling": {"temperature": 0.8, "max_tokens": 768, "model": "fake"},
        "token_logprobs": [-0.1, -0.2],
        "logprob": -0.3,
        "n_tokens": 2,
    }
    task, rollout, _, _ = schema.from_row(row)
    assert rollout.policy.version == row["policy_version"]
    assert rollout.extra["sampling"] == row["sampling"]
    assert rollout.extra["token_logprobs"] == [-0.1, -0.2]
    back = schema.to_row(task, rollout)
    assert back["policy_version"] == row["policy_version"]
    assert back["sampling"] == row["sampling"] and back["token_logprobs"] == [-0.1, -0.2]
    assert schema.validate(row) == []


def test_staleness_report_counts_versions_stale_rows_and_coverage():
    def r(model, version, logprob=None, sampling=None, tokens=None):
        row = {"prompt": "p", "model_version": model, "policy_version": version}
        if logprob is not None:
            row["logprob"] = logprob
        if sampling is not None:
            row["sampling"] = sampling
        if tokens is not None:
            row["token_logprobs"] = tokens
        return row

    rows = [
        r("qwen3", "qwen3@aaa", -1.0, {"temperature": 0.8}, [-0.5, -0.5]),
        r("qwen3", "qwen3@aaa", -2.0, {"temperature": 0.8}),
        r("qwen2.5", "qwen2.5@bbb"),
        {"prompt": "legacy", "model_version": "qwen3"},
    ]
    report = staleness_report(rows, base_model="qwen3")
    assert report["n"] == 4
    assert report["versions"] == {"qwen3@aaa": 2, "qwen2.5@bbb": 1, "qwen3": 1}
    assert report["models"] == {"qwen3": 3, "qwen2.5": 1}
    assert report["stale"] == 1 and report["stale_share"] == 0.25
    assert report["sampling_coverage"] == 0.5 and report["logprob_coverage"] == 0.5
    assert report["token_logprob_coverage"] == 0.25
    assert report["temperatures"] == {"0.8": 2}
    joined = " ".join(report["warnings"])
    assert "3 policy versions" in joined and "off-policy for it" in joined
    assert "carry no logprob" in joined and "do not say how they were sampled" in joined

    clean = staleness_report(rows[:2])
    assert clean["warnings"] == [] and clean["stale"] is None
    assert wai.staleness_report([])["n"] == 0


def test_every_row_says_how_it_finished(monkeypatch):
    """#253: finish_reason on the row. The fake backend's final turn is a
    length cut, so the model-backed row reads ``length``; a callable agent
    that finishes on its own reads ``stop``; one that raises reads ``error``."""
    data, _ = _simulate(monkeypatch)
    row = data.trajectories[0]
    assert row["finish_reason"] == "length"
    assert data.rows()[0]["finish_reason"] == "length"
    assert schema.validate(data.rows()[0]) == []

    honest = simulate_offline(tools=TOOLS, policy=POLICY, budget=4, per_round=4, concurrency=1)
    assert {r["finish_reason"] for r in honest.trajectories} == {"stop"}
    assert wai.pass_at(honest.trajectories).config["truncated_share"] == 0.0

    # The engine re-rolls a raised rollout and drops it if it keeps failing,
    # so the error and tool cases are checked at the function.
    from whileai.simulations.run.engine import _finish_reason

    assert _finish_reason({}, [], "<agent error: RuntimeError: boom>") == "error"
    assert _finish_reason({}, [{"tool": "get_order", "args": {}}], "") == "tool"
    assert _finish_reason({}, [{"tool": "get_order", "truncated": True}], "cut") == "length"
    assert _finish_reason({}, [], "fine.") == "stop"
    assert _finish_reason({"finish_reason": "bogus"}, [], "fine.") == "stop"

    def told(prompt, **_):
        return {"steps": [], "final_text": "done", "finish_reason": "tool"}

    said = simulate_offline(told, tools=TOOLS, policy=POLICY, budget=2, per_round=2, concurrency=1)
    assert {r["finish_reason"] for r in said.trajectories} == {"tool"}


def test_pass_at_and_delta_report_carry_the_truncated_share():
    from whileai.simulations.score.delta import delta_report

    def rows(n, cut, reward):
        out = []
        for i in range(n):
            out.append(
                {
                    "scenario_id": f"s{i}",
                    "rollout_index": 0,
                    "prompt": f"p{i}",
                    "final_text": "x",
                    "reward": reward,
                    "judge_status": "ok",
                    "finish_reason": "length" if i < cut else "stop",
                }
            )
        return out

    before = rows(20, 8, 1)
    after = rows(20, 0, 1)
    assert wai.pass_at(before).config["truncated_share"] == 0.4
    assert "40% of rows cut by the token cap" in str(wai.pass_at(before))
    assert "cut by the token cap" not in str(wai.pass_at(after))
    report = delta_report(before, after)
    assert any("cut 40% of before rows and 0% of after rows" in w for w in report["warnings"])
    # rows that never said how they finished do not pretend
    old = [{k: v for k, v in r.items() if k != "finish_reason"} for r in before]
    assert wai.pass_at(old).config["truncated_share"] is None
