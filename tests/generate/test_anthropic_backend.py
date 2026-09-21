"""``anthropic:<model>`` is a backend spec like any other. No live calls."""

from __future__ import annotations

import json

import pytest

import whileai.simulations as wai
from whileai.simulations.defaults import TRANSIENT_TRIES
from whileai.simulations.generate import agents
from whileai.simulations.generate import anthropic_backend as ab

# Captured at import, before the conftest fixture replaces it with a blocker.
_REAL_COMPLETE = agents.complete


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.delenv("WHILEAI_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.fixture(autouse=True)
def _reset_temperature_memo():
    """The 'this model rejects temperature' memo is process-global; keep it
    from leaking between tests."""
    ab._NO_TEMPERATURE.clear()
    yield
    ab._NO_TEMPERATURE.clear()


@pytest.fixture(autouse=True)
def _only_anthropic_calls(monkeypatch):
    """The conftest blocks every model call. Let the Anthropic ones through:
    requests.post is monkeypatched in each test, so nothing leaves the box."""

    def gate(base_url, model, messages, **kw):
        if not ab.is_anthropic_url(base_url):
            raise OSError("hosted simulator disabled in unit tests")
        return _REAL_COMPLETE(base_url, model, messages, **kw)

    monkeypatch.setattr(agents, "complete", gate)
    monkeypatch.setattr("whileai.simulations.generate.generator.complete", gate)


class _Response:
    def __init__(self, payload, status=200, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


def _message(*, text="", tool_use=None, stop_reason="end_turn"):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for call in tool_use or []:
        content.append({"type": "tool_use", **call})
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-fake",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def _record(monkeypatch, responses):
    """Monkeypatch requests.post to answer from a queue. Returns the bodies."""
    sent = []
    queue = list(responses)

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append({"url": url, "headers": dict(headers or {}), "body": json})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(ab.requests, "post", fake_post)
    return sent


# --- spec parsing -----------------------------------------------------------


def test_spec_resolves_to_the_messages_api_and_the_plain_model_name():
    assert agents.parse_backend_spec("anthropic:claude-sonnet-5") == (
        ab.ANTHROPIC_BASE_URL,
        "claude-sonnet-5",
    )
    assert agents.parse_backend_spec("anthropic:claude-haiku-4-5-20251001")[1] == (
        "claude-haiku-4-5-20251001"
    )
    # a bare spec still names a model, so writer_model is never empty
    assert agents.parse_backend_spec("anthropic:")[1] == ab.DEFAULT_MODEL


def test_the_unsupported_spec_error_offers_anthropic():
    with pytest.raises(ValueError, match="anthropic:<model>"):
        agents.parse_backend_spec("mistral:big")


def test_recorded_model_names_are_the_plain_model_name(monkeypatch):
    from whileai.simulations.run.config import _model_version_tag

    assert _model_version_tag("anthropic:claude-sonnet-5", {}) == "claude-sonnet-5"


# --- tool definitions -------------------------------------------------------


def test_tool_definitions_convert_from_both_openai_shapes():
    schema = {"type": "object", "properties": {"order_id": {"type": "string"}}}
    enveloped = {
        "type": "function",
        "function": {"name": "lookup_order", "description": "Find one", "parameters": schema},
    }
    bare = {
        "name": "refund",
        "description": "Refund it",
        "parameters": schema,
        "returns": {"ok": True},
        "mock": {},
    }
    assert ab.wire_tools([enveloped, bare]) == [
        {"name": "lookup_order", "description": "Find one", "input_schema": schema},
        {"name": "refund", "description": "Refund it", "input_schema": schema},
    ]


def test_a_tool_with_no_parameters_still_gets_an_object_schema():
    assert ab.wire_tools([{"name": "ping"}]) == [
        {"name": "ping", "input_schema": {"type": "object", "properties": {}}}
    ]


# --- message translation ----------------------------------------------------


def test_system_turns_leave_the_message_list():
    system, messages = ab.split_system(
        [
            {"role": "system", "content": "Be honest."},
            {"role": "user", "content": "where is my order"},
        ]
    )
    assert system == "Be honest."
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "where is my order"}]}
    ]


def test_consecutive_tool_results_merge_into_one_user_turn():
    _, messages = ab.split_system(
        [
            {"role": "user", "content": "refund it"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "toolu_a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"id": "A1"}'},
                    },
                    {
                        "id": "toolu_b",
                        "type": "function",
                        "function": {"name": "refund", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_a", "content": '{"status": "ok"}'},
            {"role": "tool", "tool_call_id": "toolu_b", "content": '{"status": "done"}'},
        ]
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assistant = messages[1]["content"]
    assert assistant[0] == {"type": "text", "text": "checking"}
    assert assistant[1] == {
        "type": "tool_use",
        "id": "toolu_a",
        "name": "lookup",
        "input": {"id": "A1"},
    }
    assert [b["tool_use_id"] for b in messages[2]["content"]] == ["toolu_a", "toolu_b"]
    assert {b["type"] for b in messages[2]["content"]} == {"tool_result"}


def test_an_agent_opened_thread_gets_a_user_turn_first():
    _, messages = ab.split_system(
        [
            {"role": "system", "content": "Greet first."},
            {"role": "assistant", "content": "Hi, how can I help?"},
            {"role": "user", "content": "refund please"},
        ]
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0]["text"] == "Hi, how can I help?"


# --- the call itself --------------------------------------------------------


def test_the_reply_comes_back_in_the_openai_message_shape(monkeypatch):
    sent = _record(monkeypatch, [_Response(_message(text="all set"))])
    reply = agents.complete(
        ab.ANTHROPIC_BASE_URL,
        "claude-fake",
        [{"role": "system", "content": "Be honest."}, {"role": "user", "content": "hi"}],
        max_tokens=300,
        temperature=1.05,
    )
    assert reply["role"] == "assistant"
    assert reply["content"] == "all set"
    assert reply["_finish_reason"] == "stop"
    assert reply["_usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert "_logprobs" not in reply
    call = sent[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["headers"]["x-api-key"] == "sk-ant-test"
    assert call["headers"]["content-type"] == "application/json"
    assert call["body"]["system"] == "Be honest."
    assert call["body"]["max_tokens"] == 300
    # this API caps temperature at 1; the writer goes above it
    assert call["body"]["temperature"] == 1.0


def test_stop_reason_max_tokens_becomes_the_engines_truncated_marker(monkeypatch):
    _record(
        monkeypatch,
        [_Response(_message(text="I checked the order and then", stop_reason="max_tokens"))],
    )
    reply = agents.complete(
        ab.ANTHROPIC_BASE_URL, "claude-fake", [{"role": "user", "content": "hi"}]
    )
    assert reply["_finish_reason"] == "length"
    assert agents._turn_meta(reply)["truncated"] is True


def test_n_above_one_becomes_that_many_calls_on_all(monkeypatch):
    sent = _record(
        monkeypatch,
        [_Response(_message(text="one")), _Response(_message(text="two"))],
    )
    reply = agents.complete(
        ab.ANTHROPIC_BASE_URL, "claude-fake", [{"role": "user", "content": "hi"}], n=2
    )
    assert len(sent) == 2
    assert [m["content"] for m in reply["_all"]] == ["one", "two"]


# --- errors -----------------------------------------------------------------


def test_a_missing_key_names_the_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        agents.complete(ab.ANTHROPIC_BASE_URL, "claude-fake", [{"role": "user", "content": "hi"}])
    assert "ANTHROPIC_API_KEY" in (agents.missing_hosted_key(ab.ANTHROPIC_BASE_URL) or "")


def test_the_override_env_var_wins(monkeypatch):
    monkeypatch.setenv("WHILEAI_ANTHROPIC_API_KEY", "sk-ant-override")
    sent = _record(monkeypatch, [_Response(_message(text="ok"))])
    agents.complete(ab.ANTHROPIC_BASE_URL, "claude-fake", [{"role": "user", "content": "hi"}])
    assert sent[0]["headers"]["x-api-key"] == "sk-ant-override"


def test_a_4xx_raises_with_the_api_message_and_the_model(monkeypatch):
    _record(
        monkeypatch,
        [
            _Response(
                {"type": "error", "error": {"type": "not_found_error", "message": "model: nope"}},
                status=404,
            )
        ],
    )
    with pytest.raises(RuntimeError) as err:
        agents.complete(ab.ANTHROPIC_BASE_URL, "nope", [{"role": "user", "content": "hi"}])
    assert "nope" in str(err.value) and "model: nope" in str(err.value)


def test_a_temperature_400_drops_the_field_retries_and_remembers(monkeypatch):
    """Reasoning models (claude-sonnet-5, opus-5, ...) 400 on ``temperature``.
    The call drops it and retries once, mirroring the max_tokens walk, then
    omits it up front for that model."""
    error = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "temperature is deprecated for this model",
        },
    }
    sent: list[dict] = []
    queue = [_Response(error, status=400), _Response(_message(text="ok"))]

    def fake_post(url, headers=None, json=None, timeout=None):
        # copy: _one_call mutates the one payload dict across retries
        sent.append(dict(json))
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(ab.requests, "post", fake_post)
    reply = agents.complete(
        ab.ANTHROPIC_BASE_URL,
        "claude-sonnet-5",
        [{"role": "user", "content": "hi"}],
        temperature=0.7,
    )
    assert reply["content"] == "ok"
    assert len(sent) == 2
    assert sent[0]["temperature"] == 0.7  # first attempt carried it
    assert "temperature" not in sent[1]  # the retry dropped it
    assert "claude-sonnet-5" in ab._NO_TEMPERATURE

    # a later call to the same model skips temperature up front: one attempt
    again: list[dict] = []

    def fake_post_again(url, headers=None, json=None, timeout=None):
        again.append(dict(json))
        return _Response(_message(text="again"))

    monkeypatch.setattr(ab.requests, "post", fake_post_again)
    agents.complete(
        ab.ANTHROPIC_BASE_URL,
        "claude-sonnet-5",
        [{"role": "user", "content": "hi"}],
        temperature=0.7,
    )
    assert len(again) == 1
    assert "temperature" not in again[0]


def test_a_rate_limit_retries_then_raises(monkeypatch):
    monkeypatch.setattr(ab.time, "sleep", lambda _s: None)
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        return _Response({"error": {"message": "slow down"}}, status=429)

    monkeypatch.setattr(ab.requests, "post", fake_post)
    with pytest.raises(RuntimeError, match="rate-limited"):
        agents.complete(ab.ANTHROPIC_BASE_URL, "claude-fake", [{"role": "user", "content": "hi"}])
    assert len(calls) == TRANSIENT_TRIES + 1


# --- the loops that consume it ----------------------------------------------


def test_a_tool_use_round_trip_through_the_model_backed_agent_loop(monkeypatch):
    tools = [
        {
            "name": "lookup_order",
            "description": "Find one order",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ]
    turns = [
        _Response(
            _message(
                text="checking that now",
                tool_use=[{"id": "toolu_01", "name": "lookup_order", "input": {"order_id": "A-1"}}],
                stop_reason="tool_use",
            )
        ),
        _Response(_message(text="Your order A-1 shipped yesterday.")),
    ]
    sent = _record(monkeypatch, turns)
    agent = agents.local_model(
        ab.ANTHROPIC_BASE_URL,
        "claude-fake",
        tools=tools,
        system="Be honest.",
        max_turns=2,
        min_user_turns=1,
    )
    out = agent("where is order A-1")
    step = next(s for s in out["steps"] if s.get("tool"))
    assert step["tool"] == "lookup_order"
    assert step["arguments"] == {"order_id": "A-1"}
    assert step["text"] == "checking that now"

    # the second request carries the result back as a tool_result block
    second = sent[1]["body"]["messages"]
    results = [
        block
        for message in second
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "toolu_01"
    assert json.loads(results[0]["content"]) == step["result"]
    # the assistant turn went back as text plus tool_use, with the same id
    uses = [
        block
        for message in second
        for block in message["content"]
        if block.get("type") == "tool_use"
    ]
    assert [u["id"] for u in uses] == ["toolu_01"]
    assert sent[1]["body"]["tools"] == [
        {
            "name": "lookup_order",
            "description": "Find one order",
            "input_schema": tools[0]["parameters"],
        }
    ]


def test_the_situation_writer_runs_on_an_anthropic_spec(monkeypatch, tmp_path):
    """simulate(simulator="anthropic:fake") end to end against a fake reply."""
    situations = json.dumps(
        [
            {"message": "my refund never arrived"},
            {"message": "cancel order A-2 before it ships"},
        ]
    )
    _record(monkeypatch, [_Response(_message(text=situations))])
    data = wai.simulate(
        agent=lambda message: {"steps": [], "final_text": "on it"},
        tools=[{"name": "lookup_order", "parameters": {"type": "object", "properties": {}}}],
        system_prompt="Refund only with a receipt.",
        simulator="anthropic:fake",
        budget=2,
        grade=False,
        output=str(tmp_path / "rows.jsonl"),
    )
    assert data.writer_model == "fake"
    assert data.trajectories
