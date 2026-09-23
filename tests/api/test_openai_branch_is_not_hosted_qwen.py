"""The ``openai:`` branch of ``complete()`` stops assuming hosted Qwen (#755).

Three defects, one root cause: the ``openai:`` branch inherited the
hosted-Qwen assumptions that the ``anthropic:`` and ``bedrock:`` branches
return above. Each test here pins the measured failure from the issue, so
each one goes red on ``origin/main``:

1. a 15,000-character prompt to a 272k-context model arrived as **7,500**
   characters with ``max_tokens`` **1522** instead of 4096, and nothing in
   the return value said so;
2. a 400 naming ``max_completion_tokens`` was answered by halving
   ``max_tokens`` four times -- **5 paid requests**, budgets
   ``[4018, 2009, 1004, 502, 256]``, and the retry could never succeed
   because the parameter name was wrong, not its value;
3. an api.openai.com failure was reported as "hosted Qwen".

Every request is answered from this file. ``http.client.HTTPSConnection``
is the only wire ``complete()`` has, so spying on it costs no network and
no tokens.
"""

from __future__ import annotations

import http.client
import json

import pytest

from whileai._env import PLATFORM_MODAL_PREFIX
from whileai.simulations.generate import agents

# Captured at import, before the autouse fixture in tests/conftest.py swaps
# the module attribute for its offline stub: this is the real call.
_complete = agents.complete
_parse = agents.parse_backend_spec

#: The reproducer's prompt: 3000 words of five characters (#755).
PROMPT_CHARS = 15_000
#: What main sent instead, and the reply budget it sent with it.
MAIN_SENT_CHARS = 7_500
MAIN_SENT_MAX_TOKENS = 1522
#: What main spent on a GPT-5 class model before giving up.
MAIN_REQUESTS = 5
MAIN_BUDGET_LADDER = [4018, 2009, 1004, 502, 256]

_WRONG_KEY_NAME = json.dumps(
    {
        "error": {
            "message": (
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."
            )
        }
    }
)
_OK = json.dumps(
    {"choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}]}
)


class _Reply:
    """The two methods ``complete()`` reads off an HTTPResponse."""

    def __init__(self, status: int, body: str) -> None:
        self.status, self.reason, self._body = status, "", body.encode()

    def read(self) -> bytes:
        return self._body

    def getheader(self, *_a, **_k):
        return None

    def getheaders(self):
        return []


def _spy(monkeypatch, answers):
    """Record every request body; answer it without touching the network.

    ``answers`` is either a list of ``(status, body)`` read in order, the
    last one repeating, or a callable that reads the request payload and
    returns one pair -- which is how a server that objects to a *key*
    rather than to a value is modelled.
    """
    sent: list[dict] = []

    def request(_self, _method, _url, body=None, headers=None, **_kw):
        sent.append(json.loads(body))

    def getresponse(_self):
        if callable(answers):
            status, body = answers(sent[-1])
        else:
            status, body = answers[min(len(sent) - 1, len(answers) - 1)]
        return _Reply(status, body)

    monkeypatch.setattr(http.client.HTTPSConnection, "request", request, raising=False)
    monkeypatch.setattr(http.client.HTTPSConnection, "getresponse", getresponse, raising=False)
    monkeypatch.setattr(http.client.HTTPSConnection, "close", lambda _self: None, raising=False)
    return sent


@pytest.fixture
def openai_key(monkeypatch):
    # a literal, never a real key: nothing here reaches api.openai.com
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-key-offline-test")
    monkeypatch.delenv("ZP_CONTEXT_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_long_prompt_reaches_an_openai_model_whole(monkeypatch, openai_key):
    """#755.1 -- the silent one. Pins 7500 chars and 1522 max_tokens."""
    sent = _spy(monkeypatch, [(200, _OK)])
    url, model = _parse("openai:gpt-4.1-mini")
    reply = _complete(
        url,
        model,
        [{"role": "user", "content": "word " * 3000}],
        max_tokens=4096,
        timeout=60,
    )

    wire = sent[0]
    assert len(wire["messages"][0]["content"]) == PROMPT_CHARS
    assert len(wire["messages"][0]["content"]) != MAIN_SENT_CHARS
    assert wire["max_tokens"] == 4096
    assert wire["max_tokens"] != MAIN_SENT_MAX_TOKENS
    # nothing was cut, so nothing is reported as cut
    assert "_prompt_truncated" not in reply


def test_a_squeeze_that_does_happen_is_on_the_reply(monkeypatch, openai_key, caplog):
    """The fix stays visible where the squeeze is still right to run.

    A caller who names the window with ZP_CONTEXT_TOKENS gets the squeeze,
    and gets told: ``_prompt_truncated`` on the reply, ``prompt_truncated``
    on the step meta, and one warning naming the fix. Without this half the
    change would trade a silent cut for a silent no-cut.
    """
    monkeypatch.setenv("ZP_CONTEXT_TOKENS", "4096")
    monkeypatch.setattr(agents, "_SQUEEZE_WARNED", set())
    sent = _spy(monkeypatch, [(200, _OK)])
    url, model = _parse("openai:gpt-4.1-mini")
    with caplog.at_level("WARNING", logger="whileai.simulations"):
        reply = _complete(
            url,
            model,
            [{"role": "user", "content": "word " * 3000}],
            max_tokens=4096,
            timeout=60,
        )

    assert len(sent[0]["messages"][0]["content"]) == MAIN_SENT_CHARS
    assert sent[0]["max_tokens"] == MAIN_SENT_MAX_TOKENS
    cut = reply["_prompt_truncated"]
    assert cut["prompt_chars"] == PROMPT_CHARS
    assert cut["prompt_chars_sent"] == MAIN_SENT_CHARS
    assert cut["max_tokens_asked"] == 4096
    assert cut["max_tokens_sent"] == MAIN_SENT_MAX_TOKENS
    assert agents._turn_meta(reply)["prompt_truncated"] == cut
    assert any("cut to fit" in r.message for r in caplog.records)


def test_max_completion_tokens_is_renamed_once_not_halved_four_times(monkeypatch, openai_key):
    """#755.2 -- the expensive one. Pins 5 requests and the halving ladder.

    The server here behaves the way every GPT-5 class model does: it
    refuses ``max_tokens`` at any value and answers only once the other
    spelling arrives. On main that is unreachable, so the run pays for
    five requests and fails.
    """

    def openai_reasoning_model(payload: dict) -> tuple[int, str]:
        if "max_completion_tokens" in payload:
            return 200, _OK
        return 400, _WRONG_KEY_NAME

    sent = _spy(monkeypatch, openai_reasoning_model)
    url, model = _parse("openai:gpt-5.4-mini")
    _complete(url, model, [{"role": "user", "content": "What is 2+2?"}], max_tokens=4096)

    assert len(sent) == 2, "one rename and one retry, not a ladder"
    assert len(sent) != MAIN_REQUESTS
    assert [p.get("max_tokens") for p in sent] != MAIN_BUDGET_LADDER
    assert sent[0]["max_tokens"] == 4096 and "max_completion_tokens" not in sent[0]
    # the older spelling goes first: every vLLM and Ollama endpoint takes it
    assert sent[1]["max_completion_tokens"] == 4096 and "max_tokens" not in sent[1]


def test_a_renamed_budget_is_still_halved_when_the_value_is_what_is_wrong(monkeypatch, openai_key):
    """The halving ladder survives the rename: it just follows the key.

    A model that renamed the key and then objects to its value must still
    get a smaller one, or the rename would have retired a working retry.
    """
    sent = _spy(monkeypatch, [(400, _WRONG_KEY_NAME), (400, "max_completion_tokens too large")])
    url, model = _parse("openai:gpt-5.4-mini")
    with pytest.raises(RuntimeError):
        _complete(url, model, [{"role": "user", "content": "hi"}], max_tokens=4096)

    budgets = [p.get("max_completion_tokens") for p in sent if "max_completion_tokens" in p]
    assert budgets[:3] == [4096, 2048, 1024]


def test_an_openai_failure_does_not_name_qwen(monkeypatch, openai_key):
    """#755.3 -- the misleading one. A 401 names the host that answered."""
    _spy(monkeypatch, [(401, '{"error": {"message": "Incorrect API key provided"}}')])
    url, model = _parse("openai:gpt-4.1-mini")
    with pytest.raises(RuntimeError) as caught:
        _complete(url, model, [{"role": "user", "content": "hi"}], max_tokens=64)

    message = str(caught.value)
    assert "qwen" not in message.lower()
    assert "api.openai.com" in message
    # the mark whileai/simulations/run/engine.py::_auth_error reads
    assert "rejected the API key" in message


def test_an_openai_400_does_not_name_qwen(monkeypatch, openai_key):
    """The other two hardcoded branches: the bare 400 and the context 400."""
    _spy(monkeypatch, [(400, '{"error": {"message": "something about the schema"}}')])
    url, model = _parse("openai:gpt-4.1-mini")
    with pytest.raises(RuntimeError) as caught:
        _complete(url, model, [{"role": "user", "content": "hi"}], max_tokens=64)
    assert "qwen" not in str(caught.value).lower()
    assert "api.openai.com" in str(caught.value)

    _spy(monkeypatch, [(400, '{"error": {"message": "maximum context length exceeded"}}')])
    with pytest.raises(RuntimeError) as caught:
        _complete(url, model, [{"role": "user", "content": "hi"}], max_tokens=64)
    assert "qwen" not in str(caught.value).lower()
    assert "api.openai.com" in str(caught.value)


def test_a_hosted_endpoint_keeps_its_window(monkeypatch):
    """The squeeze is not gone: While's own 4k endpoints still get it.

    ``_window_is_known`` is the whole rule, so it is pinned on both sides.
    """
    monkeypatch.delenv("ZP_CONTEXT_TOKENS", raising=False)
    ours = f"https://{PLATFORM_MODAL_PREFIX}qwen3-4b.modal.run/v1"
    assert agents._window_is_known(ours) is True
    assert agents._window_is_known("https://api.openai.com/v1") is False
    # a vLLM someone runs themselves is "anything else", even on Modal: the
    # 4096 fallback is a guess about another operator's server (#755)
    assert agents._window_is_known("https://mylab--my-qwen-serve.modal.run/v1") is False
    monkeypatch.setenv("ZP_CONTEXT_TOKENS", "272000")
    assert agents._window_is_known("https://api.openai.com/v1") is True


def test_a_truncated_prompt_reaches_degraded_and_the_run_warning():
    """The last hop: the step flag becomes a run-level finding.

    ``degraded`` is where a caller looks for "this run is not what you
    asked for", and a cut prompt belongs there for the same reason
    ``no_tool_calls`` does: the rows still grade.
    """
    import whileai.simulations as wai
    from tests.helpers import POLICY, TOOLS

    cut = {
        "context_tokens": 4096,
        "prompt_chars": PROMPT_CHARS,
        "prompt_chars_sent": MAIN_SENT_CHARS,
        "max_tokens_asked": 4096,
        "max_tokens_sent": MAIN_SENT_MAX_TOKENS,
    }

    def agent(message: str) -> dict:
        return {
            "final_text": "Done.",
            "steps": [{"role": "assistant", "text": "Done.", "prompt_truncated": cut}],
        }

    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        budget=2,
        time_budget=30,
        advanced={"concurrency": 1},
    )
    assert "prompt_truncated" in data.degraded
    assert data.search["prompt_truncated_rows"] == len(data.trajectories)
    assert any(str(PROMPT_CHARS) in w for w in data.warnings)
