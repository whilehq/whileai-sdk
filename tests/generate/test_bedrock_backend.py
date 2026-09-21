"""``bedrock:<model-id>[@<region>]`` is a backend spec like any other. No live calls."""

from __future__ import annotations

import json
import json as _json

import pytest

import whileai as wai
from whileai.simulations.defaults import TRANSIENT_TRIES
from whileai.simulations.generate import agents
from whileai.simulations.generate import bedrock_backend as bb

# Captured at import, before the conftest fixture replaces it with a blocker.
_REAL_COMPLETE = agents.complete


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (bb.OVERRIDE_ENV, bb.KEY_ENV, *bb.REGION_ENVS, "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bb, "_clients", {})


@pytest.fixture
def bearer(monkeypatch):
    monkeypatch.setenv(bb.KEY_ENV, "bedrock-api-key-test")


@pytest.fixture(autouse=True)
def _only_bedrock_calls(monkeypatch):
    """The conftest blocks every model call. Let the Bedrock ones through:
    the transports are monkeypatched in each test, so nothing leaves the box."""

    def gate(base_url, model, messages, **kw):
        if not bb.is_bedrock_url(base_url):
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


def _converse(*, text="", tool_use=None, stop_reason="end_turn"):
    content = []
    if text:
        content.append({"text": text})
    for call in tool_use or []:
        content.append({"toolUse": call})
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
        "metrics": {"latencyMs": 3},
    }


def _error(name, message):
    return {"__type": f"com.amazon.bedrock#{name}", "message": message}


def _record(monkeypatch, responses):
    """Monkeypatch requests.post to answer from a queue. Returns the calls."""
    sent = []
    queue = list(responses)

    def fake_post(url, headers=None, json=None, timeout=None):
        # a copy: the backend walks maxTokens down on the same dict
        sent.append(
            {"url": url, "headers": dict(headers or {}), "body": _json.loads(_json.dumps(json))}
        )
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(bb.requests, "post", fake_post)
    monkeypatch.setattr(bb.time, "sleep", lambda s: None)
    return sent


class _FakeClient:
    """Stands in for ``boto3.client("bedrock-runtime")``."""

    def __init__(self, responses):
        self.queue = list(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        out = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(out, Exception):
            raise out
        return out


class _ClientError(Exception):
    def __init__(self, code, message, status):
        super().__init__(f"{code}: {message}")
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


def _signed(monkeypatch, responses):
    client = _FakeClient(responses)
    made = []

    def make(region, timeout):
        made.append((region, timeout))
        return client

    monkeypatch.setattr(bb, "_make_client", make)
    monkeypatch.setattr(bb, "has_aws_credentials", lambda: True)
    client.made = made  # type: ignore[attr-defined]
    return client


# --- spec parsing -----------------------------------------------------------


def test_spec_resolves_to_the_regional_runtime_and_the_plain_model_id():
    assert agents.parse_backend_spec("bedrock:us.anthropic.claude-sonnet-5@us-west-2") == (
        "https://bedrock-runtime.us-west-2.amazonaws.com",
        "us.anthropic.claude-sonnet-5",
    )
    # model ids carry ``:``; ARNs carry ``:`` and ``/``; neither carries ``@``
    url, model = agents.parse_backend_spec(
        "bedrock:arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123def456"
    )
    assert model == "arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123def456"
    assert bb.region_of(url) == bb.DEFAULT_REGION
    assert agents.parse_backend_spec("bedrock:anthropic.claude-haiku-4-5-20251001-v1:0")[1] == (
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    # a bare spec still names a model, so writer_model is never empty
    assert agents.parse_backend_spec("bedrock:")[1] == bb.DEFAULT_MODEL


def test_the_region_comes_from_the_environment_unless_pinned(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    assert bb.region_of(agents.parse_backend_spec("bedrock:m")[0]) == "eu-central-1"
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    assert bb.region_of(agents.parse_backend_spec("bedrock:m")[0]) == "us-east-2"
    assert bb.region_of(agents.parse_backend_spec("bedrock:m@ap-south-1")[0]) == "ap-south-1"


def test_the_unsupported_spec_error_offers_bedrock():
    with pytest.raises(ValueError, match=r"bedrock:<model-id>\[@<region>\]"):
        agents.parse_backend_spec("mistral:big")


def test_recorded_model_names_are_the_plain_model_id():
    from whileai.simulations.run.config import _model_version_tag

    assert _model_version_tag("bedrock:us.anthropic.claude-sonnet-5@us-west-2", {}) == (
        "us.anthropic.claude-sonnet-5"
    )


def test_is_bedrock_url_is_the_runtime_host_only():
    assert bb.is_bedrock_url("https://bedrock-runtime.us-east-1.amazonaws.com")
    assert bb.is_bedrock_url("bedrock-runtime.eu-central-1.amazonaws.com/")
    assert not bb.is_bedrock_url("https://bedrock.us-east-1.amazonaws.com")
    assert not bb.is_bedrock_url("https://api.anthropic.com/v1")
    assert not bb.is_bedrock_url(None)


# --- the backend object -----------------------------------------------------


def test_backend_object_repr_names_the_region_and_the_credential_route():
    plain = wai.models.Bedrock("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert plain.spec == "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert repr(plain) == (
        "Bedrock(model='us.anthropic.claude-haiku-4-5-20251001-v1:0', "
        "key=AWS_BEARER_TOKEN_BEDROCK or AWS credentials)"
    )
    pinned = wai.models.Bedrock("m", region="us-west-2", api_key="secret-token")
    assert pinned.spec == "bedrock:m@us-west-2"
    assert repr(pinned) == "Bedrock(model='m', region='us-west-2', key=given)"
    assert "secret-token" not in repr(pinned)


def test_configure_keeps_the_bedrock_key_for_every_call_to_that_provider(monkeypatch):
    monkeypatch.setattr(bb, "has_aws_credentials", lambda: False)
    wai.configure(judge=wai.models.Bedrock("m", region="us-west-2", api_key="tok-from-object"))
    try:
        assert wai.settings.judge == "bedrock:m@us-west-2"
        url, _ = agents.parse_backend_spec(wai.settings.judge)
        assert agents.resolve_completion_key(url) == "tok-from-object"
        assert agents.missing_hosted_key(url) is None
    finally:
        wai.configure(judge=wai.Hosted())
        wai.settings.keys.pop("bedrock", None)


# --- request translation ----------------------------------------------------


def test_tool_definitions_become_tool_specs_from_both_openai_shapes():
    schema = {"type": "object", "properties": {"order_id": {"type": "string"}}}
    enveloped = {
        "type": "function",
        "function": {"name": "lookup_order", "description": "Find one", "parameters": schema},
    }
    bare = {"name": "refund", "description": "Refund it", "parameters": schema, "mock": {}}
    no_args = {"name": "ping"}
    config = bb.wire_tool_config([enveloped, bare, no_args])
    assert config == {
        "tools": [
            {
                "toolSpec": {
                    "name": "lookup_order",
                    "description": "Find one",
                    "inputSchema": {"json": schema},
                }
            },
            {
                "toolSpec": {
                    "name": "refund",
                    "description": "Refund it",
                    "inputSchema": {"json": schema},
                }
            },
            {
                "toolSpec": {
                    "name": "ping",
                    "inputSchema": {"json": {"type": "object", "properties": {}}},
                }
            },
        ]
    }
    assert bb.wire_tool_config(None) is None
    assert bb.wire_tool_config([]) is None


def test_history_becomes_system_blocks_and_converse_turns():
    history = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Where is order 7?"},
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_order", "arguments": '{"order_id": "7"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "shipped"}'},
        {"role": "tool", "tool_call_id": "call_1", "content": ""},
        {"role": "assistant", "content": "It shipped."},
    ]
    system, turns = bb.wire_messages(history)
    assert system == [{"text": "Be brief."}]
    assert turns == [
        {"role": "user", "content": [{"text": "Where is order 7?"}]},
        {
            "role": "assistant",
            "content": [
                {"text": "Let me look."},
                {
                    "toolUse": {
                        "toolUseId": "call_1",
                        "name": "get_order",
                        "input": {"order_id": "7"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "call_1",
                        "content": [{"text": '{"status": "shipped"}'}],
                    }
                },
                {"toolResult": {"toolUseId": "call_1", "content": [{"text": "(empty)"}]}},
            ],
        },
        {"role": "assistant", "content": [{"text": "It shipped."}]},
    ]


def test_request_body_clamps_temperature_and_routes_extras():
    body = bb.build_request(
        [{"role": "user", "content": "hi"}],
        tools=None,
        temperature=1.05,
        max_tokens=300,
        extra={"top_p": 0.9, "stop": "END", "top_k": 40, "chat_template_kwargs": {"x": 1}},
    )
    assert body["inferenceConfig"] == {
        "maxTokens": 300,
        "temperature": 1.0,
        "topP": 0.9,
        "stopSequences": ["END"],
    }
    assert body["additionalModelRequestFields"] == {"top_k": 40}
    assert "system" not in body and "toolConfig" not in body
    assert "chat_template_kwargs" not in json.dumps(body)


def test_converse_url_escapes_an_arn():
    arn = "arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123def456"
    url = bb.converse_url("https://bedrock-runtime.us-east-1.amazonaws.com", arn)
    assert url == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
        "arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Aimported-model%2Fabc123def456/converse"
    )


# --- reply translation ------------------------------------------------------


def test_reply_carries_text_tool_calls_finish_reason_and_usage():
    reply = bb.reply_from_response(
        _converse(
            text="Looking.",
            tool_use=[{"toolUseId": "tooluse_1", "name": "get_order", "input": {"order_id": "7"}}],
            stop_reason="tool_use",
        )
    )
    assert reply["role"] == "assistant"
    assert reply["content"] == "Looking."
    assert reply["tool_calls"] == [
        {
            "id": "tooluse_1",
            "type": "function",
            "function": {"name": "get_order", "arguments": '{"order_id": "7"}'},
        }
    ]
    assert reply["_finish_reason"] == "tool_calls"
    assert reply["_usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert "_logprobs" not in reply


@pytest.mark.parametrize(
    ("stop", "finish"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("guardrail_intervened", "content_filter"),
        ("content_filtered", "content_filter"),
    ],
)
def test_stop_reasons_map_to_the_engine_words(stop, finish):
    assert bb.reply_from_response(_converse(text="x", stop_reason=stop))["_finish_reason"] == finish


# --- the bearer transport ---------------------------------------------------


def test_bearer_path_posts_to_the_converse_route_with_the_token(monkeypatch, bearer):
    sent = _record(monkeypatch, [_Response(_converse(text="hello"))])
    url, model = agents.parse_backend_spec("bedrock:us.anthropic.claude-sonnet-5@us-west-2")
    reply = agents.complete(url, model, [{"role": "user", "content": "hi"}], max_tokens=400)
    assert reply["content"] == "hello"
    assert sent[0]["url"] == (
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/"
        "us.anthropic.claude-sonnet-5/converse"
    )
    assert sent[0]["headers"]["Authorization"] == "Bearer bedrock-api-key-test"
    assert sent[0]["body"]["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert sent[0]["body"]["inferenceConfig"]["maxTokens"] == 400


def test_an_explicit_key_beats_the_environment(monkeypatch, bearer):
    sent = _record(monkeypatch, [_Response(_converse(text="ok"))])
    url, model = agents.parse_backend_spec("bedrock:m")
    agents.complete(url, model, [{"role": "user", "content": "hi"}], api_key="explicit")
    assert sent[0]["headers"]["Authorization"] == "Bearer explicit"


def test_the_override_variable_beats_the_aws_one(monkeypatch, bearer):
    monkeypatch.setenv(bb.OVERRIDE_ENV, "override-token")
    assert bb.resolve_key() == "override-token"


def test_n_samples_become_n_calls_and_land_on_all(monkeypatch, bearer):
    sent = _record(monkeypatch, [_Response(_converse(text="a")), _Response(_converse(text="b"))])
    url, model = agents.parse_backend_spec("bedrock:m")
    reply = agents.complete(url, model, [{"role": "user", "content": "hi"}], n=2)
    assert len(sent) == 2
    assert reply["content"] == "a"
    assert [r["content"] for r in reply["_all"]] == ["a", "b"]
    assert all(not k.startswith("_") for r in reply["_all"] for k in r)


def test_throttling_is_retried_then_named(monkeypatch, bearer):
    sent = _record(
        monkeypatch,
        [_Response(_error("ThrottlingException", "Too many requests"), status=429)],
    )
    url, model = agents.parse_backend_spec("bedrock:m")
    with pytest.raises(RuntimeError, match=r"rate-limited m \(429\) after \d+ tries"):
        agents.complete(url, model, [{"role": "user", "content": "hi"}])
    assert len(sent) == TRANSIENT_TRIES + 1


def test_model_not_ready_gets_its_own_sentence(monkeypatch, bearer):
    _record(
        monkeypatch,
        [_Response(_error("ModelNotReadyException", "Model is being restored"), status=429)],
    )
    url, model = agents.parse_backend_spec("bedrock:arn:aws:bedrock:us-east-1:1:imported-model/x")
    with pytest.raises(RuntimeError, match="still restoring the imported model"):
        agents.complete(url, model, [{"role": "user", "content": "hi"}])


def test_access_denied_and_not_found_name_the_fix(monkeypatch, bearer):
    _record(monkeypatch, [_Response(_error("AccessDeniedException", "no access"), status=403)])
    url, model = agents.parse_backend_spec("bedrock:m")
    with pytest.raises(RuntimeError, match=r"refused the credentials.*access to this model"):
        agents.complete(url, model, [{"role": "user", "content": "hi"}])
    _record(monkeypatch, [_Response(_error("ResourceNotFoundException", "gone"), status=404)])
    with pytest.raises(RuntimeError, match=r"no model 'm' in this region.*@<region>"):
        agents.complete(url, model, [{"role": "user", "content": "hi"}])


def test_a_max_tokens_400_walks_the_budget_down(monkeypatch, bearer):
    sent = _record(
        monkeypatch,
        [
            _Response(_error("ValidationException", "maxTokens too large"), status=400),
            _Response(_converse(text="fits")),
        ],
    )
    url, model = agents.parse_backend_spec("bedrock:m")
    reply = agents.complete(url, model, [{"role": "user", "content": "hi"}], max_tokens=2048)
    assert reply["content"] == "fits"
    assert sent[0]["body"]["inferenceConfig"]["maxTokens"] == 2048
    assert sent[1]["body"]["inferenceConfig"]["maxTokens"] == 1024


# --- the signed transport ---------------------------------------------------


def test_without_a_token_the_call_is_signed_by_boto3_in_the_urls_region(monkeypatch):
    client = _signed(monkeypatch, [_converse(text="signed")])
    url, model = agents.parse_backend_spec("bedrock:m@eu-central-1")
    reply = agents.complete(url, model, [{"role": "user", "content": "hi"}], timeout=30)
    assert reply["content"] == "signed"
    assert client.made == [("eu-central-1", 30.0)]
    call = client.calls[0]
    assert call["modelId"] == "m"
    assert call["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    # the client is built once per region and reused
    agents.complete(url, model, [{"role": "user", "content": "again"}], timeout=30)
    assert len(client.made) == 1


def test_boto3_errors_map_to_the_same_sentences(monkeypatch):
    _signed(monkeypatch, [_ClientError("AccessDeniedException", "denied", 403)])
    url, model = agents.parse_backend_spec("bedrock:m")
    with pytest.raises(RuntimeError, match="refused the credentials"):
        agents.complete(url, model, [{"role": "user", "content": "hi"}])


def test_no_token_and_no_credentials_is_one_sentence(monkeypatch):
    monkeypatch.setattr(bb, "has_aws_credentials", lambda: False)
    url, model = agents.parse_backend_spec("bedrock:m")
    assert agents.missing_hosted_key(url) == bb.MISSING_BEDROCK_CREDENTIALS
    with pytest.raises(RuntimeError, match="No credentials for Amazon Bedrock"):
        agents.complete(url, model, [{"role": "user", "content": "hi"}])


def test_missing_boto3_says_what_to_install(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_boto(name, *args, **kwargs):
        if name.startswith(("boto3", "botocore")):
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_boto)
    assert bb.has_aws_credentials() is False
    with pytest.raises(ImportError, match=r'pip install "whileai\[bedrock\]"'):
        bb._make_client("us-east-1", 10.0)


# --- the judge --------------------------------------------------------------


def test_judge_spec_and_key_resolution_accept_bedrock(monkeypatch, bearer):
    from whileai.simulations.score.grade_llm import judge_spec
    from whileai.simulations.score.llm_judge import resolve_judge_key

    assert judge_spec(spec="bedrock:m@us-west-2") == "bedrock:m@us-west-2"
    assert resolve_judge_key(backend_spec="bedrock:m") == "bedrock-api-key-test"
    monkeypatch.delenv(bb.KEY_ENV)
    assert resolve_judge_key(backend_spec="bedrock:m") is None
