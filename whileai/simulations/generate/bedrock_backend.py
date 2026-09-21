"""Amazon Bedrock behind the same reply shape as an OpenAI backend.

``bedrock:<model-id>[@<region>]`` is a backend spec like ``anthropic:<model>``,
so it works anywhere one is accepted: the agent under test, the situation
writer, the simulated user and the judge. The model id is what Bedrock calls
it: a foundation model (``anthropic.claude-haiku-4-5-20251001-v1:0``), a
cross-region inference profile (``us.anthropic.claude-sonnet-5``) or the ARN
of a model you imported yourself (``arn:aws:bedrock:...:imported-model/...``),
which is how a trained adapter, merged into its base, is served on AWS.

Every call goes to the Converse API, the one request shape Bedrock offers
for every chat model it hosts. The translation happens at the boundary, so
nothing downstream learns a second message format:

* the OpenAI history becomes a Converse ``system`` list plus ``user`` /
  ``assistant`` turns of ``text``, ``toolUse`` and ``toolResult`` blocks,
  through the same ``split_system`` the Anthropic backend uses;
* OpenAI tool definitions become ``toolSpec`` entries with the JSON schema
  under ``inputSchema.json``;
* the reply comes back as an OpenAI assistant message: ``content``,
  ``tool_calls`` with ``id``/``name``/JSON ``arguments``, ``_finish_reason``
  (``max_tokens`` becomes ``length``, which is how the engine marks a
  truncated turn) and ``_usage``.

Two ways in, both the user's own (BYOK):

* a Bedrock API key, ``api_key=`` on ``wai.Bedrock`` or
  ``AWS_BEARER_TOKEN_BEDROCK`` in the environment: sent as a bearer token
  over ``requests``, so nothing new is installed;
* AWS credentials (``aws configure``, ``AWS_PROFILE``, an instance role):
  signed by ``boto3``, which ``pip install "whileai[bedrock]"`` adds.

Bedrock returns no log-probabilities through Converse, so ``_logprobs`` is
never set and the engine's existing "no logprobs" path applies.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any
from urllib.parse import quote, urlparse

import requests

from whileai._env import getenv

from ..defaults import MAX_SAMPLES_PER_CALL, MIN_REPLY_TOKENS, TRANSIENT_BACKOFF_S, TRANSIENT_TRIES
from .anthropic_backend import split_system, wire_tools

# The runtime host is ``bedrock-runtime.<region>.amazonaws.com``; the region
# rides in the URL, so one URL is enough to say where a call goes and which
# regional endpoint signs it.
HOST_PREFIX = "bedrock-runtime."
HOST_SUFFIX = ".amazonaws.com"
# DEFAULT_REGION = us-east-1: the first region every Bedrock model launches
# in and one of the four Custom Model Import runs in (AWS docs,
# model-customization-import-model); AWS_REGION / AWS_DEFAULT_REGION win
# when set, as they do for every AWS tool.
DEFAULT_REGION = "us-east-1"
REGION_ENVS = ("AWS_REGION", "AWS_DEFAULT_REGION")
# DEFAULT_MODEL: the cross-region profile for the same Haiku the
# ``anthropic:`` spec defaults to, so a bare ``bedrock:`` reaches the same
# family on the same account tier; the ``us.`` profile is what on-demand
# Anthropic models are invoked through in the US regions.
DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
KEY_ENV = "AWS_BEARER_TOKEN_BEDROCK"
OVERRIDE_ENV = "WHILEAI_AWS_BEARER_TOKEN_BEDROCK"
INSTALL_HINT = 'pip install "whileai[bedrock]"'

MISSING_BEDROCK_CREDENTIALS = (
    f"No credentials for Amazon Bedrock: set {KEY_ENV} to a Bedrock API key "
    f"(or {OVERRIDE_ENV} to override it), or configure AWS credentials "
    f"(aws configure, AWS_PROFILE) and install boto3 with {INSTALL_HINT}."
)

# Converse ``inferenceConfig`` fields, from the names the OpenAI-compatible
# callers already pass in ``extra=``. Anything else the model itself accepts
# rides in ``additionalModelRequestFields``; unknown vLLM-only keys are
# dropped rather than sent, because an unknown field is a 400 here.
_INFERENCE_FIELDS = {"top_p": "topP", "stop_sequences": "stopSequences", "stop": "stopSequences"}
_MODEL_FIELDS = ("top_k", "thinking")

_STOP_REASONS = {
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
    "malformed_model_output": "stop",
    "malformed_tool_use": "stop",
}

# Converse errors that are worth a retry, by the exception name Bedrock
# puts in the body. ModelNotReadyException is an imported model being
# restored after idling; the first call after a pause starts that.
_TRANSIENT_ERRORS = {
    "ThrottlingException",
    "ModelNotReadyException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
}


def region_from_env() -> str:
    """The region the AWS tools would use, else ``DEFAULT_REGION``."""
    for name in REGION_ENVS:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return DEFAULT_REGION


def base_url(region: str | None = None) -> str:
    """The Converse endpoint for ``region`` (the environment's when unset)."""
    return f"https://{HOST_PREFIX}{region or region_from_env()}{HOST_SUFFIX}"


def parse_spec_rest(rest: str) -> tuple[str, str]:
    """``(base_url, model)`` for the part after ``bedrock:``.

    ``<model>@<region>`` pins a region; ``<model>`` alone reads the
    environment. The split is on the last ``@`` because a model id or ARN
    never contains one while it does contain ``:`` and ``/``.
    """
    text = str(rest or "").strip()
    model, sep, region = text.rpartition("@")
    if not sep:
        model, region = text, ""
    return base_url(region.strip() or None), model or DEFAULT_MODEL


def is_bedrock_url(url: str | None) -> bool:
    """True when this base URL is a Bedrock runtime endpoint."""
    if not url:
        return False
    raw = str(url)
    if "://" not in raw:
        raw = "https://" + raw
    host = (urlparse(raw).hostname or "").lower()
    return host.startswith(HOST_PREFIX) and host.endswith(HOST_SUFFIX)


def region_of(url: str) -> str:
    """The region a Bedrock runtime URL names."""
    raw = str(url)
    if "://" not in raw:
        raw = "https://" + raw
    host = (urlparse(raw).hostname or "").lower()
    return host[len(HOST_PREFIX) : -len(HOST_SUFFIX)] if is_bedrock_url(raw) else ""


def resolve_key(api_key: str | None = None) -> str:
    """The bearer token for this endpoint: an explicit one, else the
    override, else ``AWS_BEARER_TOKEN_BEDROCK``. Empty means "sign with AWS
    credentials instead". Never logged or returned in an error message."""
    if api_key:
        return str(api_key).strip()
    override = str(getenv(KEY_ENV) or "").strip()
    if override:
        return override
    return str(os.environ.get(KEY_ENV) or "").strip()


def has_aws_credentials() -> bool:
    """True when boto3 is installed and its credential chain finds something."""
    try:
        import botocore.session
    except ImportError:
        return False
    try:
        return botocore.session.get_session().get_credentials() is not None
    except Exception:  # a malformed profile or config file
        return False


def missing_key(api_key: str | None = None) -> str | None:
    """The one-sentence auth error, or None when a token or AWS credentials
    are present."""
    if resolve_key(api_key) or has_aws_credentials():
        return None
    return MISSING_BEDROCK_CREDENTIALS


# --- request translation ----------------------------------------------------


def _content_blocks(blocks: list[dict]) -> list[dict]:
    """Converse content blocks from the Anthropic-shaped ones ``split_system`` builds."""
    out: list[dict] = []
    for block in blocks:
        kind = str(block.get("type") or "")
        if kind == "text":
            text = str(block.get("text") or "")
            if text.strip():
                out.append({"text": text})
        elif kind == "tool_use":
            out.append(
                {
                    "toolUse": {
                        "toolUseId": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                    }
                }
            )
        elif kind == "tool_result":
            content = str(block.get("content") or "")
            out.append(
                {
                    "toolResult": {
                        "toolUseId": str(block.get("tool_use_id") or ""),
                        # Converse wants a non-empty block; an empty tool result
                        # is still a result, so it is spelled out.
                        "content": [{"text": content if content.strip() else "(empty)"}],
                    }
                }
            )
    return out


def wire_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (``system`` blocks, Converse ``messages``) for an OpenAI history."""
    system, turns = split_system(messages)
    wired = []
    for turn in turns:
        content = _content_blocks(list(turn.get("content") or []))
        if content:
            wired.append({"role": turn["role"], "content": content})
    return ([{"text": system}] if system.strip() else []), wired


def wire_tool_config(tools: list[dict] | None) -> dict | None:
    """Converse ``toolConfig`` from OpenAI tool definitions, or None without any."""
    specs = []
    for tool in wire_tools(tools):
        spec: dict[str, Any] = {
            "name": tool["name"],
            "inputSchema": {"json": tool["input_schema"]},
        }
        if tool.get("description"):
            spec["description"] = tool["description"]
        specs.append({"toolSpec": spec})
    return {"tools": specs} if specs else None


def build_request(
    messages: list[dict],
    *,
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The Converse request body (everything but ``modelId``)."""
    system, wired = wire_messages([dict(m) for m in messages])
    inference: dict[str, Any] = {
        "maxTokens": max(1, int(max_tokens)),
        # The OpenAI-compatible backends take temperatures above 1 (the writer
        # goes to 1.05); Converse caps at 1 for every model family and 400s above.
        "temperature": max(0.0, min(1.0, float(temperature))),
    }
    body: dict[str, Any] = {"messages": wired, "inferenceConfig": inference}
    if system:
        body["system"] = system
    tool_config = wire_tool_config(tools)
    if tool_config:
        body["toolConfig"] = tool_config
    if extra:
        for key, field in _INFERENCE_FIELDS.items():
            if key in extra and extra[key] is not None:
                value = extra[key]
                inference[field] = (
                    [str(value)] if field == "stopSequences" and isinstance(value, str) else value
                )
        model_fields = {k: extra[k] for k in _MODEL_FIELDS if k in extra and extra[k] is not None}
        if model_fields:
            body["additionalModelRequestFields"] = model_fields
    return body


# --- reply translation ------------------------------------------------------


def reply_from_response(data: Mapping[str, Any]) -> dict:
    """An OpenAI-shaped assistant message from one Converse response."""
    message = (data.get("output") or {}).get("message") or {}
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text = str(block.get("text") or "")
            if text:
                text_parts.append(text)
        elif isinstance(block.get("toolUse"), dict):
            use = block["toolUse"]
            tool_calls.append(
                {
                    "id": str(use.get("toolUseId") or ""),
                    "type": "function",
                    "function": {
                        "name": str(use.get("name") or ""),
                        "arguments": json.dumps(use.get("input") or {}),
                    },
                }
            )
    text = "\n\n".join(text_parts)
    reply: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        reply["tool_calls"] = tool_calls
    stop = str(data.get("stopReason") or "")
    if stop:
        reply["_finish_reason"] = _STOP_REASONS.get(stop, stop)
    usage = data.get("usage")
    if isinstance(usage, dict):
        reply["_usage"] = {
            "input_tokens": int(usage.get("inputTokens") or 0),
            "output_tokens": int(usage.get("outputTokens") or 0),
        }
    return reply


# --- errors -----------------------------------------------------------------


def _error_parts(body: str) -> tuple[str, str]:
    """(exception name, message) from a Bedrock error body."""
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return "", str(body or "")[:300]
    if not isinstance(payload, dict):
        return "", str(body or "")[:300]
    name = str(payload.get("__type") or payload.get("code") or "").rpartition("#")[2]
    message = str(payload.get("message") or payload.get("Message") or body or "")[:300]
    return name, message


def _raise_for_status(status: int, name: str, detail: str, model: str, *, tries: int) -> None:
    where = f"{name}: " if name else ""
    if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
        raise RuntimeError(
            f"Bedrock refused the credentials for {model} ({status}): {where}{detail} "
            f"Check {KEY_ENV} or the AWS profile, and that the account has access to "
            "this model in this region."
        )
    if status == HTTPStatus.NOT_FOUND:
        raise RuntimeError(
            f"Bedrock has no model {model!r} in this region ({status}): {where}{detail} "
            "Use a model id from `aws bedrock list-foundation-models`, an inference "
            "profile, or the ARN of an imported model, and pin the region with @<region>."
        )
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        if name == "ModelNotReadyException":
            raise RuntimeError(
                f"Bedrock is still restoring the imported model {model} after {tries} tries: "
                f"{detail} An idle import is unloaded and the first call brings it back; "
                "wait a few minutes and call again."
            )
        raise RuntimeError(
            f"Bedrock rate-limited {model} ({status}) after {tries} tries: {where}{detail} "
            "Lower concurrency= or ask AWS for a higher quota."
        )
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        raise RuntimeError(
            f"Bedrock returned {status} for {model} after {tries} tries: {where}{detail} Retry later."
        )
    raise RuntimeError(f"Bedrock rejected the request for {model} ({status}): {where}{detail}")


def _retry_after(headers: Any, attempt: int) -> float:
    """Server-named delay when there is one, else the OpenAI path's backoff."""
    try:
        value = float(str((headers or {}).get("retry-after") or "").strip())
    except (TypeError, ValueError):
        value = 0.0
    return min(30.0, value) if value > 0 else TRANSIENT_BACKOFF_S * (2**attempt)


# --- the two transports -----------------------------------------------------


def converse_url(url: str, model: str) -> str:
    """``POST /model/{modelId}/converse``; ARNs carry ``:`` and ``/``, so the id is escaped."""
    root = str(url).rstrip("/")
    if "://" not in root:
        root = "https://" + root
    return f"{root}/model/{quote(model, safe='')}/converse"


def _one_bearer_call(
    url: str, body: dict[str, Any], token: str, *, model: str, timeout: float
) -> dict:
    """One Converse call over HTTPS with a Bedrock API key, with the transient
    retries and the 400 max_tokens walk-down the other backends do."""
    headers = {"Authorization": f"Bearer {token}", "content-type": "application/json"}
    transient = 0
    for _ in range(8):
        response = requests.post(url, headers=headers, json=body, timeout=timeout)
        status = int(response.status_code)
        if status < HTTPStatus.BAD_REQUEST:
            return reply_from_response(response.json())
        name, detail = _error_parts(response.text or "")
        if (
            status == HTTPStatus.BAD_REQUEST
            and "maxTokens" in detail
            and int(body["inferenceConfig"].get("maxTokens") or 0) > MIN_REPLY_TOKENS
        ):
            body["inferenceConfig"]["maxTokens"] = max(
                MIN_REPLY_TOKENS, int(body["inferenceConfig"]["maxTokens"]) // 2
            )
            continue
        if (
            name in _TRANSIENT_ERRORS
            or status == HTTPStatus.TOO_MANY_REQUESTS
            or status >= HTTPStatus.INTERNAL_SERVER_ERROR
        ) and transient < TRANSIENT_TRIES:
            time.sleep(_retry_after(response.headers, transient))
            transient += 1
            continue
        _raise_for_status(status, name, detail, model, tries=transient + 1)
    raise RuntimeError(
        f"Bedrock never accepted the request for {model}; "
        "lower agent_max_tokens= or shorten the system prompt."
    )


_clients: dict[tuple[str, float], Any] = {}
_clients_lock = threading.Lock()


def _make_client(region: str, timeout: float) -> Any:
    """A ``bedrock-runtime`` client that signs with the AWS credential chain.
    Tests replace this function; nothing else constructs a client."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError(
            f"Signing Bedrock calls with AWS credentials needs boto3: {INSTALL_HINT} "
            f"(or set {KEY_ENV} to a Bedrock API key, which needs nothing installed). "
            f"Underlying: {exc}"
        ) from exc
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            # botocore's own retry loop covers throttling and ModelNotReady;
            # one more attempt than the requests path's TRANSIENT_TRIES, so
            # both paths give up after the same number of tries.
            retries={"max_attempts": TRANSIENT_TRIES + 1, "mode": "adaptive"},
            read_timeout=timeout,
            connect_timeout=timeout,
        ),
    )


def _client(region: str, timeout: float) -> Any:
    key = (region, float(timeout))
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = _make_client(region, timeout)
        return client


def _one_signed_call(region: str, body: dict[str, Any], *, model: str, timeout: float) -> dict:
    """One Converse call through boto3, errors mapped to the same sentences."""
    client = _client(region, timeout)
    try:
        return reply_from_response(client.converse(modelId=model, **body))
    except Exception as exc:
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            raise
        error = response.get("Error") or {}
        status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
        name = str(error.get("Code") or type(exc).__name__)
        detail = str(error.get("Message") or exc)[:300]
        if (
            status == HTTPStatus.BAD_REQUEST
            and "maxTokens" in detail
            and int(body["inferenceConfig"].get("maxTokens") or 0) > MIN_REPLY_TOKENS
        ):
            body["inferenceConfig"]["maxTokens"] = max(
                MIN_REPLY_TOKENS, int(body["inferenceConfig"]["maxTokens"]) // 2
            )
            return _one_signed_call(region, body, model=model, timeout=timeout)
        _raise_for_status(
            status or HTTPStatus.BAD_REQUEST, name, detail, model, tries=TRANSIENT_TRIES + 1
        )
    raise AssertionError("unreachable")  # pragma: no cover


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
    """Converse with ``model`` and return the reply in the OpenAI message shape.

    A bearer token (``api_key=`` or ``AWS_BEARER_TOKEN_BEDROCK``) goes over
    ``requests``; without one the call is signed by boto3 from the AWS
    credential chain. ``n>1`` has no Converse equivalent, so it becomes that
    many calls; the first is the return value and the rest land on ``_all``,
    which is what the situation writer reads.
    """
    auth_err = missing_key(api_key)
    if auth_err:
        raise RuntimeError(auth_err)
    body = build_request(
        messages, tools=tools, temperature=temperature, max_tokens=max_tokens, extra=extra
    )
    token = resolve_key(api_key)
    region = region_of(base_url) or region_from_env()
    url = converse_url(base_url, model)
    samples = max(1, min(MAX_SAMPLES_PER_CALL, int(n)))
    replies = []
    for _ in range(samples):
        request = json.loads(json.dumps(body))  # each call walks its own maxTokens down
        if token:
            replies.append(_one_bearer_call(url, request, token, model=model, timeout=timeout))
        else:
            replies.append(_one_signed_call(region, request, model=model, timeout=timeout))
    first = replies[0]
    if len(replies) > 1:
        first["_all"] = [
            {k: v for k, v in r.items() if not str(k).startswith("_")} for r in replies
        ]
    return first


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REGION",
    "KEY_ENV",
    "MISSING_BEDROCK_CREDENTIALS",
    "base_url",
    "build_request",
    "complete",
    "converse_url",
    "has_aws_credentials",
    "is_bedrock_url",
    "missing_key",
    "parse_spec_rest",
    "region_from_env",
    "region_of",
    "reply_from_response",
    "resolve_key",
    "wire_messages",
    "wire_tool_config",
]
