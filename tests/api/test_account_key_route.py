"""No VLLM_API_KEY, an account key: hosted runs go to the account endpoints.

`wai signup` and `wai login` are enough for a hosted run. The
zeroproof-serve proxy behind ACCOUNT_AGENT / ACCOUNT_JUDGE takes the zp_ key,
enforces the daily allowance with 429 and meters on the server. VLLM_API_KEY
still wins and goes to the shared pool.
"""

from __future__ import annotations

import json

import pytest

from whileai.simulations.generate import agents
from whileai.simulations.generate.agents import (
    ACCOUNT_AGENT,
    ACCOUNT_JUDGE,
    DEFAULT_AGENT,
    DEFAULT_JUDGE,
    MISSING_HOSTED_KEY,
    _quota_error,
    _request_extras,
    default_agent_spec,
    default_judge_spec,
    default_simulator_spec,
    missing_hosted_key,
    resolve_completion_key,
)
from whileai.simulations.run.engine import _auth_error, _stop_reason

ACCOUNT_URL = "https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1"
POOL_URL = "https://zeroproofai--stressd-vllm-serve.modal.run/v1"


@pytest.fixture
def saved_key(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.setattr("whileai.auth.stored_api_key", lambda: "zp_saved")


def test_account_key_alone_selects_the_account_endpoints(saved_key):
    assert default_agent_spec() == ACCOUNT_AGENT
    assert default_simulator_spec() == ACCOUNT_AGENT
    assert default_judge_spec() == ACCOUNT_JUDGE
    assert resolve_completion_key(ACCOUNT_URL) == "zp_saved"
    assert missing_hosted_key() is None


def test_vllm_key_still_wins_and_goes_to_the_shared_pool(saved_key, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "pool-key")
    assert default_agent_spec() == DEFAULT_AGENT
    assert default_judge_spec() == DEFAULT_JUDGE
    assert resolve_completion_key(POOL_URL) == "pool-key"
    # an explicit account URL keeps the account key even with the pool key set
    assert resolve_completion_key(ACCOUNT_URL) == "zp_saved"


def test_no_key_at_all_names_both_ways_in(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.setattr("whileai.auth.stored_api_key", lambda: None)
    assert default_agent_spec() == DEFAULT_AGENT
    msg = missing_hosted_key()
    assert msg == MISSING_HOSTED_KEY
    assert "wai login" in msg and "VLLM_API_KEY" in msg and "\n" not in msg


def test_explicit_env_specs_are_untouched(saved_key, monkeypatch):
    monkeypatch.setenv("WHILEAI_AGENT", "openai:gpt-4.1-mini")
    monkeypatch.setenv("WHILEAI_JUDGE", "vllm:x@http://localhost:8000/v1")
    assert default_agent_spec() == "openai:gpt-4.1-mini"
    assert default_judge_spec() == "vllm:x@http://localhost:8000/v1"


def test_account_qwen_is_asked_not_to_think():
    assert _request_extras(ACCOUNT_URL, "Qwen/Qwen3-4B") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert _request_extras(POOL_URL, "Qwen/Qwen3-4B-Instruct-2507") == {}
    assert _request_extras(ACCOUNT_URL, "microsoft/phi-4") == {}


def test_quota_429_is_named_and_not_retried():
    body = json.dumps(
        {
            "error": {
                "message": "daily account quota exceeded (25100/25000 input, 900/50000 output tokens); resets at midnight UTC.",
                "type": "rate_limit_error",
            }
        }
    )
    msg = _quota_error(429, body)
    assert msg and msg.startswith("Hosted model daily quota exceeded")
    assert _quota_error(429, "slow down") is None
    assert _quota_error(503, body) is None
    assert not agents._transient_http(429, body)
    # the engine treats it like a rejected key: stop, do not spend the clock
    assert _auth_error(f"RuntimeError: {msg}") == msg
    assert _stop_reason("writer", msg) == "writer_quota_exceeded"
    assert _stop_reason("agent", "Hosted Qwen rejected the API key (401).") == "agent_auth_failed"


def test_account_calls_are_metered_by_the_proxy_not_the_client(saved_key):
    assert agents._client_metered(ACCOUNT_URL) is False
    assert agents._client_metered(POOL_URL) is True
    assert agents._client_metered("https://api.openai.com/v1") is False
    assert agents._client_metered("http://localhost:11434/v1") is False


def test_a_modal_303_on_cold_start_is_followed_to_the_result(monkeypatch):
    """Modal answers a request past 150 s with a 303 to a result URL that
    blocks until done. The client must follow it or it reads an empty body."""
    import http.client

    calls = []

    class _Resp:
        def __init__(self, status, body, location=None):
            self.status, self._body, self._loc = status, body, location

        def read(self):
            return self._body

        def getheader(self, name):
            return self._loc if name == "Location" else None

    class _Conn:
        def __init__(self, host, port=None, timeout=None):
            self.host = host

        def request(self, method, path, body=None, headers=None):
            calls.append((method, self.host, path, dict(headers or {})))

        def getresponse(self):
            # first poll: still pending, redirected again; second: the result
            if len(calls) == 1:
                return _Resp(303, b"", "https://x.modal.run/v1/chat/completions?__modal_result=2")
            return _Resp(200, b'{"ok": true}')

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", _Conn)
    first = _Resp(303, b"", "https://x.modal.run/v1/chat/completions?__modal_result=1")
    status, body = agents._follow_redirects(
        first, b"", {"Authorization": "Bearer zp_k", "Content-Type": "application/json"}, 5
    )
    assert (status, body) == (200, b'{"ok": true}')
    assert [c[0] for c in calls] == ["GET", "GET"]
    assert calls[0][2] == "/v1/chat/completions?__modal_result=1"
    assert calls[0][3] == {"Authorization": "Bearer zp_k"}

    plain = _Resp(200, b"{}")
    assert agents._follow_redirects(plain, b"{}", {}, 5) == (200, b"{}")


def test_status_names_the_hosted_route(saved_key, monkeypatch):
    from whileai import auth

    monkeypatch.setattr(auth, "account", lambda key: {"tier": "full"})
    # status() reads the credentials file itself, not stored_api_key()
    monkeypatch.setattr(auth, "_read", lambda path: {"api_key": "zp_saved", "name": "t"})
    assert auth.status()["hosted_route"] == "account endpoints (this key)"
    monkeypatch.setenv("VLLM_API_KEY", "pool-key")
    assert auth.status()["hosted_route"] == "shared pool (VLLM_API_KEY)"
    monkeypatch.delenv("VLLM_API_KEY")
    monkeypatch.setattr("whileai.auth.stored_api_key", lambda: None)
    monkeypatch.setattr(auth, "_read", lambda path: {})
    assert auth.status()["hosted_route"] is None
