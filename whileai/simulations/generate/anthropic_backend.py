"""Anthropic Messages API behind the same reply shape as an OpenAI backend.

``anthropic:<model>`` is a backend spec like ``openai:<model>``, so it works
anywhere one is accepted: the situation writer (``simulator=``), the simulated
user (``user_model=``), a model-backed agent (``agent="anthropic:..."``) and
the judge (``grade(spec=...)``). ``parse_backend_spec`` maps the spec to this
module's base URL and ``complete()`` in ``agents`` hands those calls here.

The translation happens at the boundary, so nothing downstream learns a second
message format:

* the OpenAI history (``system``/``user``/``assistant`` with ``tool_calls``,
  ``tool`` with ``tool_call_id``) becomes an Anthropic ``system`` string plus
  ``user``/``assistant`` turns with ``tool_use`` and ``tool_result`` blocks;
* OpenAI tool definitions, enveloped or bare, become ``input_schema`` tools;
* the reply comes back as an OpenAI assistant message: ``content``,
  ``tool_calls`` with ``id``/``name``/JSON ``arguments``, ``_finish_reason``
  (``stop_reason`` ``max_tokens`` becomes ``length``, which is how the engine
  marks a truncated turn) and ``_usage``.

Anthropic returns no log-probabilities, so ``_logprobs`` is never set and the
engine's existing "no logprobs" path applies. No new dependency: the calls go
out over ``requests``, which the package already depends on.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any
from urllib.parse import urlparse

import requests

from whileai._env import getenv

from ..defaults import MIN_REPLY_TOKENS, TRANSIENT_BACKOFF_S, TRANSIENT_TRIES

# The spec carries only the model name, so the base URL is fixed and doubles
# as the marker that routes a call here.
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_HOST = "api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-4-5"
KEY_ENV = "ANTHROPIC_API_KEY"
OVERRIDE_ENV = "WHILEAI_ANTHROPIC_API_KEY"

MISSING_ANTHROPIC_KEY = (
    f"No Anthropic API key: set {KEY_ENV} (or {OVERRIDE_ENV} to override it) "
    "to a key from console.anthropic.com."
)

# Transient retry count and backoff: TRANSIENT_TRIES / TRANSIENT_BACKOFF_S
# in ``simulations.defaults``, shared with the OpenAI-compatible path.

# Body fields the Messages API accepts. Anything else a caller put in
# ``extra=`` is vLLM-only (chat_template_kwargs) and is dropped rather than
# sent, because an unknown field is a 400 here.
_PASSTHROUGH = (
    "top_p",
    "top_k",
    "stop_sequences",
    "thinking",
    "tool_choice",
    "metadata",
)

_STOP_REASONS = {
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
}

# Models that answered 400 "temperature is deprecated for this model": the
# reasoning family (claude-sonnet-5, opus-5/4.8/4.7, fable-5) dropped sampling
# and rejects the field. Remembered per process so the retry in ``_one_call``
# fires at most once per model, not on every call the way an opt-in param
# would. No static model list to keep current: a model teaches us on its first
# 400. ``_NO_TEMPERATURE`` is reset only by restarting the process.
_NO_TEMPERATURE: set[str] = set()


def is_anthropic_url(base_url: str | None) -> bool:
    """True when this base URL is the Anthropic Messages API."""
    if not base_url:
        return False
    raw = str(base_url)
    if "://" not in raw:
        raw = "https://" + raw
    return (urlparse(raw).hostname or "").lower() == ANTHROPIC_HOST


def resolve_key(api_key: str | None = None) -> str:
    """The key for this endpoint: an explicit one, else the override, else
    ``ANTHROPIC_API_KEY``. Never logged or returned in an error message."""
    if api_key:
        return str(api_key).strip()
    override = str(getenv("ANTHROPIC_API_KEY") or "").strip()
    if override:
        return override
    return str(os.environ.get(KEY_ENV) or "").strip()


def missing_key(api_key: str | None = None) -> str | None:
    """The one-sentence auth error, or None when a key is present."""
    return None if resolve_key(api_key) else MISSING_ANTHROPIC_KEY


def wire_tools(tools: list[dict] | None) -> list[dict]:
    """Anthropic tool definitions from OpenAI ones.

    Both shapes are accepted: the envelope ``{"type": "function",
    "function": {name, description, parameters}}`` and the bare
    ``{name, description, parameters}`` the SDK's own specs use. Local-only
    keys (``returns``, ``mock``, ``kind``, ``drafted``) stay off the wire.
    """
    out: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str((fn or {}).get("name") or "").strip()
        if not name:
            continue
        schema = (fn or {}).get("parameters")
        if not isinstance(schema, dict) or not schema:
            # Anthropic requires an object schema even for a no-argument tool.
            schema = {"type": "object", "properties": {}}
        entry: dict[str, Any] = {"name": name, "input_schema": schema}
        description = (fn or {}).get("description")
        if description:
            entry["description"] = str(description)
        out.append(entry)
    return out


def _text_blocks(content: Any) -> list[dict]:
    """Text blocks for one message's content. Empty text is dropped: the
    API rejects an empty text block."""
    if isinstance(content, list):
        blocks = []
        for item in content:
            if isinstance(item, dict) and item.get("type"):
                blocks.append(item)
            elif str(item or "").strip():
                blocks.append({"type": "text", "text": str(item)})
        return blocks
    text = str(content or "")
    return [{"type": "text", "text": text}] if text.strip() else []


def _tool_use_blocks(tool_calls: Any) -> list[dict]:
    blocks: list[dict] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        raw_fn = call.get("function")
        fn: dict = raw_fn if isinstance(raw_fn, dict) else {}
        name = str(fn.get("name") or call.get("name") or "")
        if not name:
            continue
        raw = fn.get("arguments", call.get("arguments"))
        if isinstance(raw, str):
            try:
                arguments = json.loads(raw or "{}")
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(raw, dict):
            arguments = raw
        else:
            arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or f"toolu_{len(blocks)}"),
                "name": name,
                "input": arguments,
            }
        )
    return blocks


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return (system prompt, Anthropic messages) for an OpenAI history.

    System turns are joined into the ``system`` string, because Anthropic has
    no system role. Consecutive ``tool`` results are merged into one user
    message: the API wants every ``tool_use`` in a turn answered by
    ``tool_result`` blocks in the single user message that follows it.
    """
    system_parts: list[str] = []
    out: list[dict] = []
    pending: list[dict] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for raw in messages or []:
        message = dict(raw) if isinstance(raw, dict) else {}
        role = str(message.get("role") or "")
        if role == "system":
            flush()
            text = str(message.get("content") or "")
            if text.strip():
                system_parts.append(text)
            continue
        if role == "tool":
            pending.append(
                {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id") or ""),
                    "content": str(message.get("content") or ""),
                }
            )
            continue
        flush()
        if role == "assistant":
            blocks = _text_blocks(message.get("content")) + _tool_use_blocks(
                message.get("tool_calls")
            )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue
        blocks = _text_blocks(message.get("content"))
        if blocks:
            out.append({"role": "user", "content": blocks})
    flush()
    if out and out[0].get("role") == "assistant":
        # An agent-opens thread (opening_rate>0) starts on the assistant, which
        # this API refuses. The opener stays where it is behind one synthetic
        # user line; it is request-only and never reaches a row's steps.
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(hi)"}]})
    return "\n\n".join(system_parts), out


def reply_from_response(data: Mapping[str, Any]) -> dict:
    """An OpenAI-shaped assistant message from one Messages API response."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("type") or "")
        if kind == "text":
            text = str(block.get("text") or "")
            if text:
                text_parts.append(text)
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    text = "\n\n".join(text_parts)
    reply: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        reply["tool_calls"] = tool_calls
    stop = str(data.get("stop_reason") or "")
    if stop:
        reply["_finish_reason"] = _STOP_REASONS.get(stop, stop)
    usage = data.get("usage")
    if isinstance(usage, dict):
        reply["_usage"] = {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        }
    return reply


def _error_message(status: int, body: str) -> str:
    """The API's own error text, or the raw body when it is not JSON."""
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return str(body or "")[:300]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:300]
    return str(body or "")[:300]


def _raise_for_status(status: int, body: str, model: str, *, tries: int) -> None:
    detail = _error_message(status, body)
    if status in {401, 403}:
        raise RuntimeError(f"Anthropic rejected the API key ({status}): {detail} Check {KEY_ENV}.")
    if status == HTTPStatus.NOT_FOUND:
        raise RuntimeError(
            f"Anthropic has no model {model!r} (404): {detail} "
            "Use a model id from console.anthropic.com."
        )
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        raise RuntimeError(
            f"Anthropic rate-limited {model} (429) after {tries} tries: {detail} "
            "Lower concurrency= or wait for the limit to reset."
        )
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        raise RuntimeError(
            f"Anthropic returned {status} for {model} after {tries} tries: {detail} Retry later."
        )
    raise RuntimeError(f"Anthropic rejected the request for {model} ({status}): {detail}")


def _retry_after(headers: Any, attempt: int) -> float:
    """Server-named delay when there is one, else the OpenAI path's backoff."""
    try:
        value = float(str((headers or {}).get("retry-after") or "").strip())
    except (TypeError, ValueError):
        value = 0.0
    return min(30.0, value) if value > 0 else TRANSIENT_BACKOFF_S * (2**attempt)


def _one_call(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    model: str,
    timeout: float,
) -> dict:
    """One Messages call, with the transient retries and the 400 max_tokens
    walk-down the OpenAI-compatible path also does."""
    transient = 0
    for _ in range(8):
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = int(response.status_code)
        if status < HTTPStatus.BAD_REQUEST:
            return reply_from_response(response.json())
        body = response.text or ""
        if (
            status == HTTPStatus.BAD_REQUEST
            and "max_tokens" in body
            and int(payload.get("max_tokens") or 0) > MIN_REPLY_TOKENS
        ):
            payload["max_tokens"] = max(MIN_REPLY_TOKENS, int(payload["max_tokens"]) // 2)
            continue
        if (
            status == HTTPStatus.BAD_REQUEST
            and "temperature" in payload
            and "temperature" in body.lower()
        ):
            # Reasoning models 400 with "temperature is deprecated for this
            # model". Drop the field, note the model so ``complete()`` omits it
            # next time, and retry once. The ``in payload`` guard keeps a 400
            # that merely mentions temperature for another reason from looping.
            payload.pop("temperature", None)
            _NO_TEMPERATURE.add(model)
            continue
        if (
            status == HTTPStatus.TOO_MANY_REQUESTS or status >= HTTPStatus.INTERNAL_SERVER_ERROR
        ) and transient < TRANSIENT_TRIES:
            time.sleep(_retry_after(response.headers, transient))
            transient += 1
            continue
        _raise_for_status(status, body, model, tries=transient + 1)
    raise RuntimeError(
        f"Anthropic never accepted the request for {model}; "
        "lower agent_max_tokens= or shorten the system prompt."
    )


def complete(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 60,
    n: int = 1,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """POST /v1/messages and return the reply in the OpenAI message shape.

    ``n>1`` has no Messages API equivalent, so it becomes that many calls; the
    first is the return value and the rest land on ``_all``, which is what the
    situation writer reads. ``logprobs`` is not a parameter: Anthropic does not
    return log-probabilities, so ``_logprobs`` is absent and the engine's
    existing "no logprobs" path applies.
    """
    auth_err = missing_key(api_key)
    if auth_err:
        raise RuntimeError(auth_err)
    url = str(base_url or ANTHROPIC_BASE_URL).rstrip("/")
    if "://" not in url:
        url = "https://" + url
    if not url.endswith("/messages"):
        url = url + "/messages"
    system, wire_messages = split_system([dict(m) for m in messages])
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max(1, int(max_tokens)),
        "messages": wire_messages,
    }
    if model not in _NO_TEMPERATURE:
        # The OpenAI-compatible backends take temperatures above 1 (the writer
        # goes to 1.05); this API caps at 1 and 400s above it. Newer reasoning
        # models drop sampling and 400 on ``temperature`` at any value; once one
        # does, ``_one_call`` records it in ``_NO_TEMPERATURE`` and the field is
        # omitted here on the next call.
        payload["temperature"] = max(0.0, min(1.0, float(temperature)))
    if system:
        payload["system"] = system
    wired = wire_tools(tools)
    if wired:
        payload["tools"] = wired
    for key in _PASSTHROUGH:
        if extra and key in extra:
            payload[key] = extra[key]
    headers = {
        "x-api-key": resolve_key(api_key),
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    samples = max(1, min(8, int(n)))
    replies = [
        _one_call(url, dict(payload), headers, model=model, timeout=timeout) for _ in range(samples)
    ]
    first = replies[0]
    if len(replies) > 1:
        first["_all"] = [
            {k: v for k, v in r.items() if not str(k).startswith("_")} for r in replies
        ]
    return first
