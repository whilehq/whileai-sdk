"""``typesafe:<model>`` is a judge-only backend spec. No live calls."""

from __future__ import annotations

import json

import pytest

from whileai.simulations.defaults import TRANSIENT_TRIES
from whileai.simulations.generate import agents
from whileai.simulations.generate import typesafe_backend as tb
from whileai.simulations.run.config import resolve_run_config

# Captured at import, before the conftest fixture replaces it with a blocker.
_REAL_COMPLETE = agents.complete

TOOLS = [{"name": "lookup_order", "description": "Find one", "parameters": {"type": "object"}}]


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.delenv("WHILEAI_TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.setattr(tb.time, "sleep", lambda *_a, **_k: None)


class _Response:
    def __init__(self, payload, status=200, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload

    @property
    def text(self):
        return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)


def _answers(**answers):
    return {
        "model": "jev-latest",
        "answers": answers,
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }


def _record(monkeypatch, responses):
    """Monkeypatch requests.post to answer from a queue. Returns the requests."""
    sent = []
    queue = list(responses)

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append({"url": url, "headers": dict(headers or {}), "body": json, "timeout": timeout})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(tb.requests, "post", fake_post)
    return sent


# --- spec parsing -----------------------------------------------------------


def test_spec_resolves_to_the_api_root_and_the_plain_model_name():
    assert agents.parse_backend_spec("typesafe:jev-latest") == (tb.TYPESAFE_BASE_URL, "jev-latest")
    # a bare spec still names a model, so judge_meta.model is never empty
    assert agents.parse_backend_spec("typesafe:") == (tb.TYPESAFE_BASE_URL, tb.DEFAULT_MODEL)


def test_the_unsupported_spec_error_offers_typesafe_as_judge_only():
    with pytest.raises(ValueError, match=r"typesafe:<model> \(judge only\)"):
        agents.parse_backend_spec("mistral:big")


def test_the_base_url_env_var_points_the_spec_at_a_gateway(monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://gateway.example.com/typesafe/")
    url, _ = agents.parse_backend_spec("typesafe:jev-latest")
    assert url == "https://gateway.example.com/typesafe"
    assert tb.is_typesafe_url(url)
    assert tb.is_typesafe_url(tb.TYPESAFE_BASE_URL)
    assert not tb.is_typesafe_url("https://api.openai.com/v1")


def test_is_decision_spec_and_the_refusal_name_the_fix():
    assert tb.is_decision_spec("typesafe:jev-latest")
    assert not tb.is_decision_spec("openai:gpt-4o-mini")
    assert tb.no_chat_error("openai:gpt-4o-mini") is None
    text = tb.no_chat_error("typesafe:jev-latest")
    assert 'data.grade(spec="typesafe:jev-latest")' in text
    assert "agent=" in text and "simulator=" in text and "user_model=" in text


# --- keys -------------------------------------------------------------------


def test_a_missing_key_names_the_env_var_and_the_waitlist(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    err = tb.missing_key()
    assert "TYPESAFE_API_KEY" in err and "console.typesafe.ai" in err and "waitlist" in err
    assert agents.missing_hosted_key(tb.TYPESAFE_BASE_URL) == err
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        tb.decide(tb.TYPESAFE_BASE_URL, "jev-latest", "s", {"q": {"type": "noul"}})


def test_the_override_env_var_wins(monkeypatch):
    monkeypatch.setenv("WHILEAI_TYPESAFE_API_KEY", "ts-override")
    assert tb.resolve_key() == "ts-override"
    assert agents.resolve_completion_key(tb.TYPESAFE_BASE_URL) == "ts-override"
    assert agents.missing_hosted_key(tb.TYPESAFE_BASE_URL) is None


# --- chat roles refuse it ---------------------------------------------------


def test_complete_refuses_a_decision_model():
    with pytest.raises(ValueError, match="answers typed questions, not chat"):
        _REAL_COMPLETE(tb.TYPESAFE_BASE_URL, "jev-latest", [{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("role", ["agent", "simulator", "user_model"])
def test_simulate_config_refuses_a_decision_model_in_a_chat_role(role):
    kwargs = {"tools": TOOLS, "system_prompt": "Be honest."}
    agent = "typesafe:jev-latest" if role == "agent" else "openai:gpt-4o-mini"
    if role == "simulator":
        kwargs["advanced"] = {"simulator": "typesafe:jev-latest"}
    if role == "user_model":
        kwargs["advanced"] = {"user_model": "typesafe:jev-latest"}
    with pytest.raises(ValueError, match=rf"{role}= typesafe:jev-latest answers typed questions"):
        resolve_run_config(agent, **kwargs)


# --- the wire ---------------------------------------------------------------


def test_decide_sends_state_model_and_questions_with_a_bearer_key(monkeypatch):
    sent = _record(
        monkeypatch,
        [
            _Response(
                _answers(
                    ok={"type": "noul", "noul": 0.93},
                    future={"type": "rank", "ranking": ["a"]},
                )
            )
        ],
    )
    questions = {"ok": {"type": "noul", "instructions": "It worked."}}
    out = tb.decide(tb.TYPESAFE_BASE_URL, "jev-latest", {"ticket": "hi"}, questions, timeout=7)
    assert len(sent) == 1
    assert sent[0]["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent[0]["headers"]["Authorization"] == "Bearer ts-test"
    assert sent[0]["headers"]["Content-Type"] == "application/json"
    assert sent[0]["timeout"] == 7
    assert sent[0]["body"] == {
        "state": {"ticket": "hi"},
        "model": "jev-latest",
        "questions": questions,
    }
    # an answer type this module does not know is dropped, as the SDK drops it
    assert out == {
        "model": "jev-latest",
        "answers": {"ok": {"type": "noul", "noul": 0.93}},
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }


def test_decide_needs_a_question():
    with pytest.raises(ValueError, match="at least one question"):
        tb.decide(tb.TYPESAFE_BASE_URL, "jev-latest", "s", {})


def test_a_401_names_the_key_fix(monkeypatch):
    _record(monkeypatch, [_Response({"error": "invalid api key"}, status=401)])
    with pytest.raises(RuntimeError, match=r"401.*TYPESAFE_API_KEY"):
        tb.decide(tb.TYPESAFE_BASE_URL, "jev-latest", "s", {"q": {"type": "noul"}})


def test_a_422_raises_with_the_validation_detail(monkeypatch):
    body = {"detail": [{"loc": ["body", "questions", "q"], "msg": "Field required", "type": "x"}]}
    _record(monkeypatch, [_Response(body, status=422)])
    with pytest.raises(RuntimeError, match=r"TypeSafe 422 for jev-latest: questions.q: Field"):
        tb.decide(tb.TYPESAFE_BASE_URL, "jev-latest", "s", {"q": {"type": "noul"}})


def test_a_rate_limit_waits_retry_after_then_succeeds(monkeypatch):
    waits = []
    monkeypatch.setattr(tb.time, "sleep", waits.append)
    sent = _record(
        monkeypatch,
        [
            _Response({"error": "slow down"}, status=429, headers={"retry-after-ms": "50"}),
            _Response(_answers(q={"type": "noul", "noul": 0.5})),
        ],
    )
    out = tb.decide(tb.TYPESAFE_BASE_URL, "jev-latest", "s", {"q": {"type": "noul"}})
    assert len(sent) == 2
    assert waits == [0.05]
    assert out["answers"]["q"]["noul"] == 0.5


def test_a_5xx_retries_transient_tries_then_raises(monkeypatch):
    sent = _record(monkeypatch, [_Response("bad gateway", status=502)] * (TRANSIENT_TRIES + 2))
    with pytest.raises(RuntimeError, match="TypeSafe 502 for jev-latest: bad gateway"):
        tb.decide(tb.TYPESAFE_BASE_URL, "jev-latest", "s", {"q": {"type": "noul"}})
    assert len(sent) == TRANSIENT_TRIES + 1


def test_a_reply_without_answers_raises(monkeypatch):
    _record(monkeypatch, [_Response({"model": "jev-latest"})])
    with pytest.raises(RuntimeError, match="no answers object"):
        tb.decide(tb.TYPESAFE_BASE_URL, "jev-latest", "s", {"q": {"type": "noul"}})


def test_list_models_reads_the_names(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen["url"], seen["headers"] = url, dict(headers or {})
        return _Response(
            {"models": [{"name": "jev-latest", "description": "", "release_date": ""}]}
        )

    monkeypatch.setattr(tb.requests, "get", fake_get)
    assert tb.list_models() == ["jev-latest"]
    assert seen["url"] == "https://api.typesafe.ai/v1/models"
    assert seen["headers"]["Authorization"] == "Bearer ts-test"


def test_list_models_401_names_the_key_fix(monkeypatch):
    monkeypatch.setattr(tb.requests, "get", lambda *a, **k: _Response({"error": "no"}, status=401))
    with pytest.raises(RuntimeError, match=r"401.*TYPESAFE_API_KEY"):
        tb.list_models()


# --- answer readers ---------------------------------------------------------


def test_noul_probability_is_a_float_in_range_or_none():
    assert tb.noul_probability({"type": "noul", "noul": 0.25}) == 0.25
    assert tb.noul_probability({"type": "noul", "noul": 1.2}) is None
    assert tb.noul_probability({"type": "noul", "noul": True}) is None
    assert tb.noul_probability({"type": "choice", "choice": "a"}) is None
    assert tb.noul_probability(None) is None


def test_choice_of_takes_the_label_or_the_argmax():
    answer = {
        "type": "choice",
        "choice": "b",
        "confidence": 0.7,
        "probabilities": {"a": 0.3, "b": 0.7},
    }
    assert tb.choice_of(answer) == ("b", {"a": 0.3, "b": 0.7})
    assert tb.choice_of({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}}) == (
        "b",
        {"a": 0.3, "b": 0.7},
    )
    assert tb.choice_of({"type": "noul", "noul": 0.5}) == (None, {})


def test_score_of_reads_integer_levels_and_computes_a_missing_expectation():
    answer = {"type": "score", "score": 1.6, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}
    assert tb.score_of(answer) == (1.6, {0: 0.1, 1: 0.2, 2: 0.7})
    expected, probs = tb.score_of({"type": "score", "probabilities": {"0": 0.5, "2": 0.5}})
    assert expected == 1.0 and probs == {0: 0.5, 2: 0.5}
    assert tb.score_of({"type": "score"}) == (None, {})
