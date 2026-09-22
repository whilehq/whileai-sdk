"""OpenAI-compatible chat loop for hosted and local simulation backends."""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from http import HTTPStatus
from typing import Any
from urllib.parse import urlparse

from whileai._env import getenv, is_platform_host
from whileai.auth import SIGN_IN_URL
from whileai.config import SPEC_FORMS

from ..defaults import (
    CHARS_PER_TOKEN,
    LOCAL_MODEL_TEMPERATURE,
    MAX_SAMPLES_PER_CALL,
    MIN_REPLY_TOKENS,
    TEXT_HEURISTICS,
    TRANSIENT_BACKOFF_S,
    TRANSIENT_TRIES,
)
from ..text import split_reasoning
from ..tools import schemas as _tool_schemas
from ..world.sandbox import MockEnvironment, WorldOptions
from .anthropic_backend import ANTHROPIC_BASE_URL, is_anthropic_url
from .anthropic_backend import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from .anthropic_backend import complete as anthropic_complete
from .anthropic_backend import missing_key as missing_anthropic_key
from .anthropic_backend import resolve_key as anthropic_key
from .bedrock_backend import complete as bedrock_complete
from .bedrock_backend import is_bedrock_url
from .bedrock_backend import missing_key as missing_bedrock_key
from .bedrock_backend import parse_spec_rest as bedrock_spec_rest
from .bedrock_backend import resolve_key as bedrock_key
from .diversity import DEFAULT_AVG_TURNS, running_turn_mean, sample_turn_budget
from .typesafe_backend import DEFAULT_MODEL as TYPESAFE_DEFAULT_MODEL
from .typesafe_backend import base_url as typesafe_base_url
from .typesafe_backend import is_typesafe_url, no_chat_error
from .typesafe_backend import missing_key as missing_typesafe_key
from .typesafe_backend import resolve_key as typesafe_key
from .usage_meter import report_usage

log = logging.getLogger("whileai.simulations")

DEFAULT_AGENT = "vllm:Qwen/Qwen3-4B@https://zeroproofai--whileai-serve-qwen3-4b.modal.run/v1"
DEFAULT_SIMULATOR = DEFAULT_AGENT
# Judge and policy are the same family for now: Qwen3-8B grades Qwen3-4B.
# A judge grading its own writing prefers it (self-preference bias,
# Panickssery et al. 2024, arXiv:2404.13076), and same-family is a weaker
# version of that, so the default is a floor and not a recommendation:
# bring your own judge, and audit it (``wai.judge_agreement``). The 8B runs
# as the qwen3_8b function of whileai-serve, behind the same token gate as
# the agent: one route, one key, one app.
DEFAULT_JUDGE = "vllm:Qwen/Qwen3-8B@https://zeroproofai--whileai-serve-qwen3-8b.modal.run/v1"
# The account route. These two endpoints sit behind the whileai-serve
# proxy (backend/modal/serve.py on the platform), which takes the account's
# own zp_ key, refuses an exhausted daily allowance with 429, and records
# every token on the account's usage. A signup or a login is enough, and
# is now the only way in: the token gate rejects any key without the zp_
# prefix, so VLLM_API_KEY no longer reaches While's hosts at all. These are
# the same two endpoints as DEFAULT_AGENT and DEFAULT_JUDGE; the pair of
# names is kept because callers import both.
ACCOUNT_AGENT = "vllm:Qwen/Qwen3-4B@https://zeroproofai--whileai-serve-qwen3-4b.modal.run/v1"
ACCOUNT_JUDGE = "vllm:Qwen/Qwen3-8B@https://zeroproofai--whileai-serve-qwen3-8b.modal.run/v1"
#: The account-route host prefixes: the renamed apps, and the pre-rename
#: zeroproof-serve-* hosts kept for one release (CONSTITUTION.md: never
#: big-bang).
_ACCOUNT_HOST_PREFIXES = ("zeroproofai--whileai-serve-", "zeroproofai--zeroproof-serve-")
_tls = threading.local()


class _CurrentRollout(threading.local):
    # identity of the rollout running on this thread, set by simulate()
    # before each call so an execute= world knows which run it answers
    prompt: str = ""
    rollout_index: int | None = None
    seed: int | None = None
    # the row's scheduled faults and world, so a callable agent's world
    # (``wai.world``) answers the way the hosted agent's would
    faults: dict | None = None
    world_state: str = ""
    tools: list | None = None
    # the teacher's block for this row; ``seeded_agent`` quotes it on
    # purpose so ``leak_report`` has something to catch
    privileged: dict | None = None


current_rollout = _CurrentRollout()


def parse_backend_spec(spec: str) -> tuple[str, str]:
    """Return (base_url, model) for ollama:/vllm:/openai:/anthropic:/fireworks:/
    bedrock:/typesafe: specs. ``typesafe:`` is judge-only: ``complete()`` refuses it and says
    where it goes."""
    kind, _, rest = str(spec).partition(":")
    if kind == "ollama":
        return "http://localhost:11434/v1", rest or "llama3.1:8b"
    if kind == "vllm":
        model, _, url = rest.partition("@")
        if not url:
            raise ValueError("vllm spec must be vllm:<model>@<base_url>")
        return url, model
    if kind == "openai":
        return (
            os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            rest or "gpt-4o-mini",
        )
    if kind == "anthropic":
        # The Messages API, on ANTHROPIC_API_KEY. The URL is fixed, so the
        # spec is just the model name and every caller (writer, user model,
        # agent, judge) records that name the way the other backends do.
        return ANTHROPIC_BASE_URL, rest or ANTHROPIC_DEFAULT_MODEL
    if kind == "fireworks":
        # Fireworks serves open models behind an OpenAI-compatible URL, so
        # the spec is the model id as Fireworks names it
        # (``accounts/fireworks/models/<name>``) and the key is
        # FIREWORKS_API_KEY; OPENAI_API_KEY is never sent there.
        return FIREWORKS_BASE_URL, rest or FIREWORKS_DEFAULT_MODEL
    if kind == "bedrock":
        # Amazon Bedrock's Converse API, on a Bedrock API key or the AWS
        # credential chain. ``<model-id>@<region>`` pins the region; the
        # URL carries it, so the spec is otherwise just the model id (or
        # the ARN of an imported model).
        return bedrock_spec_rest(rest)
    if kind == "typesafe":
        # TypeSafe's Jev, on TYPESAFE_API_KEY: typed decisions with
        # probabilities, so a judge spec only. The URL is the API root
        # (TYPESAFE_BASE_URL overrides it) and the spec is the model name.
        return typesafe_base_url(), rest or TYPESAFE_DEFAULT_MODEL
    # One vocabulary: ``SPEC_FORMS`` in ``whileai.config`` names the forms,
    # the branches above read them, and ``tests/api/test_facade.py`` pins
    # the two together.
    raise ValueError(f"unsupported backend spec {spec!r}; use {', '.join(SPEC_FORMS.values())}")


def _settings():
    """The settings ``wai.configure`` / ``wai.context`` hold. Imported late:
    ``whileai.config`` is dependency-free, this module is not."""
    from ...config import current

    return current()


def _account_key() -> str:
    """The account's zp_ key: ``wai.configure(api_key=)``, else
    WHILEAI_API_KEY, else what `wai login` saved."""
    from ...auth import resolve_api_key

    return str(resolve_api_key() or "").strip()


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_HOST = "api.fireworks.ai"
FIREWORKS_DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"


def is_fireworks_url(base_url: str | None) -> bool:
    """True when this base URL is Fireworks' OpenAI-compatible API."""
    if not base_url:
        return False
    raw = base_url if "://" in str(base_url) else "https://" + str(base_url)
    return (urlparse(raw).hostname or "").lower() == FIREWORKS_HOST


def _account_url(base_url: str | None) -> bool:
    """True for the whileai-serve endpoints (and the pre-rename hosts, for one release), which take the account key."""
    if not base_url:
        return False
    raw = base_url if "://" in str(base_url) else "https://" + str(base_url)
    host = (urlparse(raw).hostname or "").lower()
    return host.startswith(_ACCOUNT_HOST_PREFIXES)


def _account_route() -> bool:
    """Use the account endpoints: an account key exists.

    Until 2026-09-21 a VLLM_API_KEY sent hosted calls to a shared, unmetered
    vLLM pool instead, and that pool is gone. While's own hosts take a
    ``zp_`` key and nothing else (the platform's token_gate rejects any
    other prefix), so VLLM_API_KEY now means only "my own endpoint".
    """
    return bool(_account_key())


def default_agent_spec() -> str:
    """Tool-using rollout model. ``wai.configure(agent=)`` if set; else
    WHILEAI_AGENT; else the hosted endpoint on the account key."""
    return (
        _settings().agent
        or getenv("AGENT")
        or (ACCOUNT_AGENT if _account_route() else DEFAULT_AGENT)
    )


def default_judge_spec() -> str:
    """Grader model. ``wai.configure(judge=)`` if set; else WHILEAI_JUDGE;
    else hosted Qwen3-8B on the same route as the agent. Never the policy
    checkpoint by default: see DEFAULT_JUDGE."""
    return (
        _settings().judge
        or getenv("JUDGE")
        or (ACCOUNT_JUDGE if _account_route() else DEFAULT_JUDGE)
    )


def default_simulator_spec() -> str:
    """User-message writer. ``wai.configure(simulator=)`` if set; else
    WHILEAI_SURROGATE; else the same hosted Qwen as the agent."""
    return (
        _settings().simulator
        or getenv("SURROGATE")
        or (ACCOUNT_AGENT if _account_route() else DEFAULT_SIMULATOR)
    )


#: Working context estimate for the rollout backend. Sized to hosted Qwen
#: by default; a bigger-window backend sets ZP_CONTEXT_TOKENS and every
#: derived budget (turn caps, shrink threshold) scales with it.
# CONTEXT_FLOOR_TOKENS = 2048: the smallest window the loop is sized for;
# below it one system prompt plus one tool schema leaves no room for a
# reply (convention, untested).
CONTEXT_FLOOR_TOKENS = 2048
CONTEXT_TOKENS = max(CONTEXT_FLOOR_TOKENS, int(os.environ.get("ZP_CONTEXT_TOKENS") or 4096))
_CONTEXT_TOKENS = CONTEXT_TOKENS
# LARGE_CONTEXT_TOKENS = 8192: above this window the loop assumes a
# long-horizon backend and raises the turn cap and the reply budget
# (convention: the hosted 4k Qwen is below it, every 16k+ server above).
LARGE_CONTEXT_TOKENS = 8192
#: The turn cap is what the window can hold: the reserved head (system
#: prompt, 64 tokens per tool schema up to 1536) comes off, and each
#: user+agent exchange is budgeted at 128 tokens (convention, untested;
#: a measured per-turn mean would replace it).
TURN_CAP_RESERVED_TOKENS = 2048
TURN_CAP_TOOL_TOKENS = 64
TURN_CAP_TOOLS_MAX_TOKENS = 1536
TURN_CAP_TOKENS_PER_TURN = 128
# TURN_CAP_SMALL = 40 / TURN_CAP_LARGE = 120: ceilings on the derived cap
# for a small and a large window; TURN_CAP_MIN = 8 is the floor
# (convention, untested).
TURN_CAP_SMALL = 40
TURN_CAP_LARGE = 120
TURN_CAP_MIN = 8


def default_max_turns(context_tokens: int | None = None, *, n_tools: int = 0) -> int:
    """Conversation cap. Scales with the context window; small windows
    reach ``TURN_CAP_SMALL`` turns, large ones (ZP_CONTEXT_TOKENS) go
    long-horizon."""
    ctx = int(context_tokens if context_tokens is not None else CONTEXT_TOKENS)
    reserved = TURN_CAP_RESERVED_TOKENS + min(
        TURN_CAP_TOOLS_MAX_TOKENS, max(0, int(n_tools)) * TURN_CAP_TOOL_TOKENS
    )
    ceiling = TURN_CAP_SMALL if ctx <= LARGE_CONTEXT_TOKENS else TURN_CAP_LARGE
    return max(TURN_CAP_MIN, min(ceiling, max(1, ctx - reserved) // TURN_CAP_TOKENS_PER_TURN))


def _models_url(base_url: str | None = None) -> tuple[str, Any] | None:
    url = base_url
    if not url:
        try:
            url, _ = parse_backend_spec(default_agent_spec())
        except ValueError:
            return None
    raw = url if "://" in str(url) else "https://" + str(url)
    return raw, urlparse(raw)


def _models_path(parsed) -> str:
    path = parsed.path.rstrip("/")
    if path.endswith("/models"):
        get_path = path
    elif path.endswith("/v1"):
        get_path = path + "/models"
    else:
        get_path = (path or "") + "/v1/models"
    if not get_path.startswith("/"):
        get_path = "/" + get_path
    return get_path


def ping_hosted(base_url: str | None = None, *, timeout: float = 3.0) -> bool:
    """True if hosted Qwen answers GET /v1/models. 5xx and connection errors are down."""
    parsed_pair = _models_url(base_url)
    if not parsed_pair:
        return False
    raw, parsed = parsed_pair
    key = resolve_completion_key(raw)
    if missing_hosted_key(raw, key):
        return False
    headers = {"Connection": "keep-alive"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        conn: http.client.HTTPConnection
        if (parsed.scheme or "https") == "https":
            conn = http.client.HTTPSConnection(
                parsed.hostname or "", parsed.port or 443, timeout=timeout
            )
        else:
            conn = http.client.HTTPConnection(
                parsed.hostname or "", parsed.port or 80, timeout=timeout
            )
        conn.request("GET", _models_path(parsed), headers=headers)
        resp = conn.getresponse()
        status = int(getattr(resp, "status", 200) or 200)
        resp.read()
        conn.close()
    except Exception:
        return False
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR or status == HTTPStatus.NOT_FOUND:
        return False
    return HTTPStatus.OK <= status < HTTPStatus.INTERNAL_SERVER_ERROR


def touch_hosted(base_url: str | None = None, *, timeout: float = 5.0) -> None:
    """Best-effort GET /v1/models. Keeps the reserved replica awake."""
    ping_hosted(base_url, timeout=timeout)


USER_TURN_MARK = "\n<USER_TURN>\n"


def split_user_turns(message: str) -> list[str]:
    """Split a writer prompt on USER_TURN into separate user lines."""
    turns = [part.strip() for part in str(message).split(USER_TURN_MARK) if part.strip()]
    return turns or [str(message)]


def _hosted_qwen_url(base_url: str | None) -> bool:
    url = base_url
    if not url:
        try:
            url, _ = parse_backend_spec(default_agent_spec())
        except ValueError:
            return False
    raw = url if "://" in str(url) else "https://" + str(url)
    host = (urlparse(raw).hostname or "").lower()
    return host.endswith("modal.run") or is_platform_host(raw)


def _configured_key(base_url: str | None) -> str | None:
    """The key a backend object registered for this URL's provider, if any
    (``wai.OpenAI("gpt-4.1-mini", api_key=...)`` keeps its key for every
    OpenAI call in the process). ``None`` falls through to the environment."""
    keys = _settings().keys
    if not keys:
        return None
    if is_anthropic_url(base_url):
        return keys.get("anthropic")
    if is_bedrock_url(base_url):
        return keys.get("bedrock")
    if is_typesafe_url(base_url):
        return keys.get("typesafe")
    if is_fireworks_url(base_url):
        return keys.get("fireworks")
    if _account_url(base_url) or not base_url:
        return None
    if _hosted_qwen_url(base_url):
        return keys.get("vllm")
    if _local_url(base_url):
        return keys.get("vllm")
    return keys.get("openai") or keys.get("vllm")


def resolve_completion_key(base_url: str | None = None, api_key: str | None = None) -> str:
    """Key for an OpenAI-compatible completion URL.

    Hosted Qwen on *.modal.run uses VLLM_API_KEY (or an explicit api_key).
    OPENAI_API_KEY is not a fallback there. Other URLs still accept either.
    """
    if api_key:
        return str(api_key).strip()
    configured = _configured_key(base_url)
    if configured:
        return configured
    if is_anthropic_url(base_url):
        return anthropic_key()
    if is_bedrock_url(base_url):
        # empty means "sign with AWS credentials"; missing_hosted_key decides
        return bedrock_key()
    if is_typesafe_url(base_url):
        return typesafe_key()
    if is_fireworks_url(base_url):
        return str(os.environ.get("FIREWORKS_API_KEY") or "").strip()
    vllm = str(os.environ.get("VLLM_API_KEY") or "").strip()
    if not base_url:
        # no URL means the default agent, whichever route that resolves to
        try:
            base_url, _ = parse_backend_spec(default_agent_spec())
        except ValueError:
            base_url = None
    if _account_url(base_url):
        # the proxy only knows zp_ keys; the shared-pool key is not one
        return _account_key() or vllm
    if _hosted_qwen_url(base_url):
        return vllm
    return vllm or str(os.environ.get("OPENAI_API_KEY") or "").strip()


def _local_url(base_url: str | None) -> bool:
    """A loopback or plain-http endpoint, which needs no key (ollama, local vLLM)."""
    if not base_url:
        return False
    raw = base_url if "://" in str(base_url) else "https://" + str(base_url)
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "http"
        or host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        or host.endswith(".local")
    )


def missing_hosted_key(base_url: str | None = None, api_key: str | None = None) -> str | None:
    """One-sentence auth error, or None if a key is present or not needed.

    Hosted Qwen wants VLLM_API_KEY. Any other https endpoint wants
    OPENAI_API_KEY (or an explicit api_key). Loopback and plain-http
    endpoints run without one. Without this a bring-your-own run with no
    key spent its whole time budget on 401s and returned nothing.
    """
    if is_anthropic_url(base_url):
        return missing_anthropic_key(api_key)
    if is_bedrock_url(base_url):
        # a key given on wai.models.Bedrock(api_key=) counts as present
        return missing_bedrock_key(api_key or _configured_key(base_url))
    if is_typesafe_url(base_url):
        return missing_typesafe_key(api_key)
    key = resolve_completion_key(base_url, api_key)
    if key:
        return None
    if _hosted_qwen_url(base_url):
        return MISSING_HOSTED_KEY
    if base_url and not _local_url(base_url):
        raw = base_url if "://" in str(base_url) else "https://" + str(base_url)
        host = urlparse(raw).hostname or str(base_url)
        return (
            f"No API key for {host}: set OPENAI_API_KEY "
            "(and OPENAI_BASE_URL for a non-OpenAI endpoint)."
        )
    return None


MISSING_HOSTED_KEY = (
    "Hosted models need an account key: run `wai login` (or `wai signup "
    "--email you@example.com`). VLLM_API_KEY reaches your own vLLM endpoint, "
    "not While's: pass it with agent='vllm:<model>@<your-url>'."
)
QUOTA_MARK = "quota exceeded"
#: What to do about a spent daily allowance. A trial day is about 12
#: hosted situations (``auth.trial_note``), which one real run spends, so
#: the error names the writer that has no allowance and the sign-in that
#: lifts the limit instead of leaving the run dead with a number.
QUOTA_FIX = (
    "simulate(..., simulator=False) writes the situations offline with no quota and no "
    f"network; signing in once at {SIGN_IN_URL} lifts a trial key's daily limit."
)


def _quota_error(status: int, body: str) -> str | None:
    """The proxy's 429 for a spent daily allowance, or None. Not transient:
    every later call today answers the same, so the run stops instead of
    retrying into the clock. Carries ``QUOTA_FIX``, the two ways on."""
    if int(status) != HTTPStatus.TOO_MANY_REQUESTS:
        return None
    text = str(body or "")
    if QUOTA_MARK not in text.lower():
        return None
    try:
        msg = json.loads(text).get("error", {}).get("message") or text
    except (ValueError, AttributeError):
        msg = text
    head = f"Hosted model daily {msg[msg.lower().find('quota') :]}".rstrip(". ")
    return f"{head}. {QUOTA_FIX}"


def _client_metered(base_url: str | None) -> bool:
    """Whether the client reports this call's tokens: the shared pool yes,
    the account proxy no (it meters on the server), anything else no."""
    return _hosted_qwen_url(base_url) and not _account_url(base_url)


_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
# _REDIRECT_HOPS = 8: Modal 303s again every 150 s of a long call, so eight
# hops is twenty minutes of cold start before the call is given up
# (convention, untested).
_REDIRECT_HOPS = 8


def _follow_redirects(
    resp: http.client.HTTPResponse, raw: bytes, headers: dict, timeout: float
) -> tuple[int, bytes]:
    """Follow a 3xx to its Location with GET and return the final (status, body).

    Modal answers a web request that runs past 150 seconds with a 303 to a
    result URL that blocks until the work is done, and may 303 again after
    another 150 seconds. A cold start of a scale-to-zero judge or policy is
    longer than that, so without this the first call read the redirect's
    empty body as the reply and the whole run graded as unreachable.
    """
    status, body = int(resp.status), raw
    location = resp.getheader("Location") if status in _REDIRECT_STATUSES else None
    hops = 0
    while location and hops < _REDIRECT_HOPS:
        hops += 1
        target = urlparse(location)
        if (target.scheme or "https") == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                target.hostname or "", target.port or 443, timeout=timeout
            )
        else:
            conn = http.client.HTTPConnection(
                target.hostname or "", target.port or 80, timeout=timeout
            )
        try:
            path = (target.path or "/") + (f"?{target.query}" if target.query else "")
            get_headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
            conn.request("GET", path, headers=get_headers)
            nxt = conn.getresponse()
            body = nxt.read()
            status = int(nxt.status)
            location = nxt.getheader("Location") if status in _REDIRECT_STATUSES else None
        finally:
            with contextlib.suppress(Exception):
                conn.close()
    return status, body


def _request_extras(base_url: str | None, model: str) -> dict[str, Any]:
    """Per-endpoint request fields. The account Qwen is the thinking base
    (`Qwen/Qwen3-4B`): without this it reasons before every reply, which
    the writer's JSON parse and the rollout's turn cap were not built for."""
    if _account_url(base_url) and str(model).startswith("Qwen/Qwen3") and "Instruct" not in model:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


HOSTED_DROPPED = (
    "Hosted Qwen dropped an in-flight request. "
    "Lower concurrency or wait for the other simulate to finish."
)
_TRANSIENT_RETRY = "retry_transient"
_TRANSIENT_STATUSES = {500, 502, 503, 504}
# _REQUEST_ATTEMPTS = 9: one first try, up to five shape retries (rename
# the reply-budget key, halve it, shrink the input, drop n, drop logprobs)
# and TRANSIENT_TRIES transient retries; the loop bound is their sum
# (structural, not a knob).
_REQUEST_ATTEMPTS = 1 + 5 + TRANSIENT_TRIES


def _is_lost_track(text: str) -> bool:
    low = str(text or "").lower()
    if "lost track of input" in low or "internalfailure" in low:
        return True
    if "modal-http" in low and ("500" in low or "internal error" in low):
        return True
    return "returned 500" in low and "modal.run" in low


def _transient_http(status: int, body: str) -> bool:
    if int(status) in _TRANSIENT_STATUSES:
        return True
    return _is_lost_track(body)


def _wire_tools(tools: list[dict] | None) -> list[dict]:
    """OpenAI wire shape for the tools array. Bare specs get the function
    envelope; local-only keys such as returns and mock stay off the wire."""
    out: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            if "kind" in tool or "drafted" in tool:
                tool = {k: v for k, v in tool.items() if k not in ("kind", "drafted")}
            out.append(tool)
            continue
        fn = {k: tool[k] for k in ("name", "description", "parameters") if k in tool}
        out.append({"type": "function", "function": fn})
    return out


def public_llm_error(exc: BaseException | str | None) -> str:
    """Studio/JSONL-safe message. Strip Modal internals from dropped requests."""
    text = str(exc or "").strip()
    if _is_lost_track(text) or str(exc) == _TRANSIENT_RETRY:
        return HOSTED_DROPPED
    return text


def _estimate_tokens(messages: list[dict], tools: list[dict] | None) -> int:
    blob = json.dumps(messages, default=str, separators=(",", ":"))
    if tools:
        blob += json.dumps(tools, default=str, separators=(",", ":"))
    return max(1, (len(blob) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def _last_user_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


# SHRINK_USER_MIN_CHARS = 800: a last user message shorter than this is
# never cut to make room; longer ones are halved, never below half of
# this (convention, untested: a user line under 800 chars is speech, one
# over it is usually a pasted document).
SHRINK_USER_MIN_CHARS = 800


def _shrink_last_user(messages: list[dict], *, frac: float = 0.5) -> bool:
    """Halve a long last user message. Short agent turns stay intact."""
    i = _last_user_index(messages)
    if i < 0:
        return False
    content = str(messages[i].get("content") or "")
    if len(content) < SHRINK_USER_MIN_CHARS:
        return False
    floor = SHRINK_USER_MIN_CHARS // 2
    messages[i] = {**messages[i], "content": content[: max(floor, int(len(content) * frac))]}
    return True


def _last_user_chars(messages: list[dict]) -> int:
    """Length of the message ``_shrink_last_user`` would cut, or 0."""
    i = _last_user_index(messages)
    return len(str(messages[i].get("content") or "")) if i >= 0 else 0


def _context_tokens_declared() -> bool:
    """Did the caller name the window, rather than inherit the fallback?"""
    return bool(str(os.environ.get("ZP_CONTEXT_TOKENS") or "").strip())


def _window_is_known(base_url: str | None) -> bool:
    """Does ``CONTEXT_TOKENS`` describe *this* endpoint's context window?

    Two ways it can: the caller set ZP_CONTEXT_TOKENS for the backend they
    are pointing at, or the call goes to a While-hosted endpoint, which is
    what the 4096 fallback was measured against. For anything else --
    api.openai.com, Fireworks, a vLLM someone runs themselves -- 4096 is a
    guess about another operator's server, and acting on it cut a 15,000
    character prompt to 7,500 and a 4096-token reply budget to 1522 against
    a 272k-context model, with nothing in the return value saying so
    (#755). Where the window is unknown the prompt goes out as written and
    the 400 ladder below shrinks it only if the server actually objects.
    """
    return _context_tokens_declared() or _hosted_qwen_url(base_url)


#: One warning per endpoint per process when the squeeze fires: a line
#: repeated once per rollout is noise the reader stops seeing (style rule
#: 10, "no warning is emitted twice for the same cause in one run").
_SQUEEZE_WARNED: set[str] = set()


def _warn_squeezed(host: str, dropped: int, before: int, want: int, asked: int) -> None:
    note = (
        f"{host}: the prompt was cut to fit a {CONTEXT_TOKENS}-token window before it was "
        f"sent -- {dropped} of {before} characters of the last user message dropped"
        + (f", and max_tokens {asked} lowered to {want}" if want < asked else "")
        + ". The reply is about a shorter question than the one that was asked. Set "
        "ZP_CONTEXT_TOKENS to this endpoint's real window, or shorten the prompt. The "
        "reply carries `_prompt_truncated` with the same numbers."
    )
    if host not in _SQUEEZE_WARNED:
        _SQUEEZE_WARNED.add(host)
        log.warning(note)


# LENGTH_CUT_MIN_CHARS = 40: a token-capped reply is cut back to its last
# sentence only when that leaves more than this; a shorter stub is left
# for the junk gate (convention, untested).
LENGTH_CUT_MIN_CHARS = 40


def _trim_length_cut(choice: dict) -> None:
    """Cut a token-capped reply back to its last complete sentence.

    finish_reason "length" means the server stopped mid-thought; the
    dangling fragment otherwise ships as a truncated final_text (measured
    20/160 rows in an rl run). Tool-call replies are left alone. If no
    sentence boundary exists the text stays and the junk gate decides.
    """
    if not isinstance(choice, dict) or choice.get("finish_reason") != "length":
        return
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("tool_calls"):
        return
    text = str(message.get("content") or "")
    if text.lstrip()[:1] in ("{", "["):
        # a json reply has no sentences; cutting at the last period
        # left the shape writer 147 chars of an 11-tool answer
        return
    cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if cut > LENGTH_CUT_MIN_CHARS:
        message["content"] = text[: cut + 1]


def _logprob_summary(choice: dict, *, tokens: bool) -> dict | None:
    """Sum and count of the sampled tokens' log-probabilities, when the
    server returned them. ``tokens=True`` keeps the per-token list too."""
    lp = choice.get("logprobs") if isinstance(choice, dict) else None
    content = lp.get("content") if isinstance(lp, dict) else None
    if not isinstance(content, list) or not content:
        return None
    values = [
        float(t["logprob"])
        for t in content
        if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))
    ]
    if not values:
        return None
    out: dict[str, Any] = {"sum": round(sum(values), 6), "n": len(values)}
    if tokens:
        out["tokens"] = [round(v, 6) for v in values]
    return out


def _turn_meta(reply: dict) -> dict:
    """Step fields an agent turn carries: logprob, n_tokens, truncated, tokens used."""
    meta: dict[str, Any] = {}
    lp = reply.get("_logprobs") if isinstance(reply, dict) else None
    if isinstance(lp, dict):
        meta["logprob"] = lp["sum"]
        meta["n_tokens"] = lp["n"]
        if lp.get("tokens"):
            meta["token_logprobs"] = list(lp["tokens"])
    if isinstance(reply, dict) and reply.get("_finish_reason") == "length":
        meta["truncated"] = True
    cut = reply.get("_prompt_truncated") if isinstance(reply, dict) else None
    if isinstance(cut, dict):
        # The input side of the same fact. "truncated" says the reply was
        # cut off; this says the question was, before it was sent (#755).
        meta["prompt_truncated"] = dict(cut)
    usage = reply.get("_usage") if isinstance(reply, dict) else None
    if isinstance(usage, dict):
        meta["input_tokens"] = int(usage.get("input_tokens") or 0)
        meta["output_tokens"] = int(usage.get("output_tokens") or 0)

    return meta


def _thread_connection(parsed: Any, conn_key: tuple, timeout: float) -> http.client.HTTPConnection:
    """One keep-alive connection per thread. A connection to a different
    host is closed before it is replaced, not dropped: a run that
    alternated hosts leaked one socket per rollout and printed a
    ResourceWarning for each."""
    conn: http.client.HTTPConnection | None = getattr(_tls, "conn", None)
    if getattr(_tls, "conn_key", None) == conn_key and conn is not None:
        return conn
    if conn is not None:
        with contextlib.suppress(Exception):
            conn.close()
    if (parsed.scheme or "https") == "https":
        conn = http.client.HTTPSConnection(
            parsed.hostname or "", parsed.port or 443, timeout=timeout
        )
    else:
        conn = http.client.HTTPConnection(parsed.hostname or "", parsed.port or 80, timeout=timeout)
    _tls.conn, _tls.conn_key = conn, conn_key
    return conn


# MIN_REPLY_TOKENS lives in defaults.py: both HTTP backends read it.
# CONTEXT_MARGIN_TOKENS = 64: slack between the estimated input and the
# window, for chat-template tokens the estimate does not see
# (convention, untested).
CONTEXT_MARGIN_TOKENS = 64
#: COMPLETE_TEMPERATURE = 0.7 / COMPLETE_MAX_TOKENS = 1024 / COMPLETE_TIMEOUT_S
#: = 60: what a bare ``complete()`` call uses when the caller names
#: nothing; every caller in this package names its own (convention, the
#: OpenAI client defaults).
COMPLETE_TEMPERATURE = 0.7
COMPLETE_MAX_TOKENS = 1024
COMPLETE_TIMEOUT_S = 60.0


def complete(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    api_key: str | None = None,
    temperature: float = COMPLETE_TEMPERATURE,
    max_tokens: int = COMPLETE_MAX_TOKENS,
    timeout: float = COMPLETE_TIMEOUT_S,
    n: int = 1,
    logprobs: bool | str = False,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """POST /chat/completions. Reuses a thread-local keep-alive connection.

    ``n>1`` asks vLLM for several samples on one prefill. The first choice is
    the return value; extra choices are on ``_all`` when the server honors ``n``.
    ``logprobs=True`` asks for the sampled tokens' log-probabilities; the
    reply then carries ``_logprobs`` (sum, n, and with ``"tokens"`` the
    per-token list). ``_finish_reason`` is always set from the first choice.
    A server that rejects ``logprobs`` gets the request again without it.

    The prompt is squeezed to fit ``CONTEXT_TOKENS`` only on an endpoint
    whose window that number describes (``_window_is_known``). When it is
    squeezed the reply carries ``_prompt_truncated`` -- the window, the
    characters asked and sent, the reply budget asked and sent -- and one
    warning per endpoint says so, because a shorter question silently
    answered moves an estimate instead of widening it (#755).

    A 400 naming ``max_completion_tokens`` renames the reply-budget key
    once and retries; the older spelling goes first because every vLLM,
    Ollama and pre-GPT-5 endpoint takes it.

    An ``anthropic:`` spec goes to the Messages API instead, translated to
    and from this same shape by ``anthropic_backend``; a ``bedrock:`` spec
    goes to Amazon Bedrock's Converse API the same way (``bedrock_backend``).
    Neither returns log-probabilities, so ``logprobs`` yields no
    ``_logprobs`` there.
    """
    if is_typesafe_url(base_url):
        # A decision model has no chat completion. The judges route to
        # score.decision_judge before they get here; a writer, agent or
        # simulated user on this spec is a configuration error, named.
        raise ValueError(no_chat_error(f"typesafe:{model}"))
    if is_anthropic_url(base_url):
        # The Messages API, translated at the boundary. It runs before the
        # context squeeze below because that budget is sized to hosted Qwen's
        # 4k window, not to a 200k one; the openai: path reaches the squeeze
        # and is now gated on _window_is_known for the same reason (#755).
        # report_usage stays out of it because a bring-your-own model is the
        # customer's own bill.
        reply = anthropic_complete(
            base_url,
            model,
            messages,
            tools=tools,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            n=n,
            extra=extra,
        )
        _trim_length_cut({"finish_reason": reply.get("_finish_reason"), "message": reply})
        return reply
    if is_bedrock_url(base_url):
        # Converse, translated at the boundary; same reasons as above.
        reply = bedrock_complete(
            base_url,
            model,
            messages,
            tools=tools,
            api_key=api_key or _configured_key(base_url),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            n=n,
            extra=extra,
        )
        _trim_length_cut({"finish_reason": reply.get("_finish_reason"), "message": reply})
        return reply
    key = resolve_completion_key(base_url, api_key)
    auth_err = missing_hosted_key(base_url, key)
    if auth_err:
        raise RuntimeError(auth_err)
    raw_url = base_url.rstrip("/")
    if "://" not in raw_url:
        raw_url = "https://" + raw_url
    parsed = urlparse(raw_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        post_path = path
    elif path.endswith("/v1"):
        post_path = path + "/chat/completions"
    else:
        post_path = (path or "") + "/v1/chat/completions"
    if not post_path.startswith("/"):
        post_path = "/" + post_path
    messages = [dict(m) for m in messages]
    asked = max(MIN_REPLY_TOKENS, int(max_tokens))
    truncated: dict[str, Any] | None = None
    if _window_is_known(base_url):
        before = _last_user_chars(messages)
        room = _CONTEXT_TOKENS - _estimate_tokens(messages, tools) - CONTEXT_MARGIN_TOKENS
        while room < MIN_REPLY_TOKENS and _shrink_last_user(messages):
            room = _CONTEXT_TOKENS - _estimate_tokens(messages, tools) - CONTEXT_MARGIN_TOKENS
        want = max(MIN_REPLY_TOKENS, min(int(max_tokens), max(MIN_REPLY_TOKENS, room)))
        after = _last_user_chars(messages)
        if after < before or want < asked:
            # Visible, three ways: this dict rides back on the reply, the
            # step meta carries the flag, and the log line names the fix.
            truncated = {
                "context_tokens": _CONTEXT_TOKENS,
                "prompt_chars": before,
                "prompt_chars_sent": after,
                "max_tokens_asked": asked,
                "max_tokens_sent": want,
            }
            _warn_squeezed(parsed.hostname or base_url, before - after, before, want, asked)
    else:
        # The window is someone else's to know. Send what the caller wrote.
        want = asked
    samples = max(1, min(MAX_SAMPLES_PER_CALL, int(n)))
    #: OpenAI's reasoning models renamed this key. The first request always
    #: sends ``max_tokens`` because that is what every vLLM, Ollama and
    #: pre-GPT-5 endpoint takes; a 400 that names the other spelling
    #: renames it once, below, and this remembers which one is on the wire.
    budget_key = "max_tokens"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": want,
    }
    if samples > 1:
        payload["n"] = samples
    if logprobs:
        payload["logprobs"] = True
    if tools:
        payload["tools"] = _wire_tools(tools)
    payload.update(_request_extras(base_url, model))
    if extra:
        payload.update(dict(extra))
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    conn_key = (parsed.scheme or "https", parsed.hostname, parsed.port, timeout)
    last_err: Exception | None = None
    transient = 0
    for _ in range(_REQUEST_ATTEMPTS):
        payload["messages"] = messages
        body = json.dumps(payload, separators=(",", ":")).encode()
        conn = _thread_connection(parsed, conn_key, timeout)
        try:
            conn.request("POST", post_path, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            status, raw = _follow_redirects(resp, raw, headers, timeout)
            if status >= HTTPStatus.BAD_REQUEST:
                err = raw[:400].decode("utf-8", "replace")
                if (
                    status == HTTPStatus.BAD_REQUEST
                    and "max_completion_tokens" in err
                    and budget_key == "max_tokens"
                ):
                    # The name is wrong, not the value. Every GPT-5 class
                    # model rejects max_tokens and asks for this spelling;
                    # halving a rejected key four times bought five paid
                    # requests and still failed (#755). Renaming needs no
                    # table of model ids, so it also fits the next one.
                    budget_key = "max_completion_tokens"
                    payload[budget_key] = payload.pop("max_tokens")
                    raise RuntimeError("retry_rename_budget")
                if (
                    status == HTTPStatus.BAD_REQUEST
                    and budget_key in err
                    and int(payload[budget_key]) > MIN_REPLY_TOKENS
                ):
                    payload[budget_key] = max(MIN_REPLY_TOKENS, int(payload[budget_key]) // 2)
                    raise RuntimeError("retry_max_tokens")
                if status == HTTPStatus.BAD_REQUEST and _shrink_last_user(messages):
                    room = (
                        _CONTEXT_TOKENS - _estimate_tokens(messages, tools) - CONTEXT_MARGIN_TOKENS
                    )
                    payload[budget_key] = max(
                        MIN_REPLY_TOKENS,
                        min(int(payload[budget_key]), max(MIN_REPLY_TOKENS, room)),
                    )
                    raise RuntimeError("retry_shrink_input")
                if status == HTTPStatus.BAD_REQUEST and payload.get("n"):
                    payload.pop("n", None)
                    raise RuntimeError("retry_drop_n")
                if (
                    status == HTTPStatus.BAD_REQUEST
                    and payload.get("logprobs")
                    and "logprob" in err.lower()
                ):
                    payload.pop("logprobs", None)
                    raise RuntimeError("retry_drop_logprobs")
                if status in {401, 403}:
                    # The host, not "hosted Qwen": an api.openai.com 401 sent
                    # the reader to look at a Modal endpoint they were not
                    # using (#755). The wording after the host is unchanged,
                    # because _auth_error matches on "rejected the API key".
                    raise RuntimeError(
                        MISSING_HOSTED_KEY
                        if not key
                        else f"{parsed.hostname} rejected the API key ({status})."
                    )
                quota = _quota_error(status, err)
                if quota:
                    raise RuntimeError(quota)
                if status == HTTPStatus.BAD_REQUEST:
                    if "context" in err.lower() or "input tokens" in err.lower():
                        window = (
                            f"the {_CONTEXT_TOKENS}-token window this run is sized for"
                            if _window_is_known(base_url)
                            else "the model's context window"
                        )
                        raise RuntimeError(
                            f"{parsed.hostname} rejected the prompt ({status}): it exceeded "
                            f"{window}. Shorten the prompt, or set ZP_CONTEXT_TOKENS to this "
                            "endpoint's real window so the run budgets against it."
                        )
                    raise RuntimeError(f"{parsed.hostname} rejected the request (400): {err[:200]}")
                if _transient_http(status, err):
                    raise RuntimeError(_TRANSIENT_RETRY)
                raise RuntimeError(f"{parsed.hostname} returned {status}: {err}")
            data = json.loads(raw)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"{parsed.hostname} returned no choices")
            for c in choices:
                _trim_length_cut(c)
            first = dict(choices[0].get("message") or {})
            extras = [dict(c.get("message") or {}) for c in choices]
            if len(extras) > 1:
                first["_all"] = extras
            if choices[0].get("finish_reason"):
                first["_finish_reason"] = str(choices[0]["finish_reason"])
            usage = data.get("usage")
            if isinstance(usage, dict) and (
                usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None
            ):
                first["_usage"] = {
                    "input_tokens": int(usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                }

            if payload.get("logprobs"):
                summary = _logprob_summary(choices[0], tokens=logprobs == "tokens")
                if summary:
                    first["_logprobs"] = summary
            if truncated:
                first["_prompt_truncated"] = dict(truncated)
            # the account proxy meters on the server; only the shared pool
            # needs the client to report what it used
            report_usage(first, hosted=_client_metered(base_url))
            return first
        except Exception as exc:
            last_err = exc
            with contextlib.suppress(Exception):
                conn.close()
            _tls.conn = None
            kind = str(exc)
            if kind in {
                "retry_rename_budget",
                "retry_max_tokens",
                "retry_shrink_input",
                "retry_drop_n",
                "retry_drop_logprobs",
            }:
                continue
            if kind == _TRANSIENT_RETRY and transient < TRANSIENT_TRIES:
                transient += 1
                time.sleep(TRANSIENT_BACKOFF_S * (2 ** (transient - 1)))
                continue
            break
    if last_err is not None and str(last_err) == _TRANSIENT_RETRY:
        if _hosted_qwen_url(base_url):
            raise RuntimeError(HOSTED_DROPPED)
        raise RuntimeError(f"{parsed.hostname} returned a transient error after retries")
    if last_err is not None:
        mapped = public_llm_error(last_err)
        if mapped != str(last_err).strip():
            raise RuntimeError(mapped)
    raise last_err  # type: ignore[misc]


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_text_tool_calls(text: str) -> list[dict]:
    """Turn Qwen/Hermes ``<tool_call>`` markup into OpenAI tool_calls."""
    if not text or "<tool_call>" not in text:
        return []
    chunks = _TOOL_CALL_BLOCK.findall(text)
    if not chunks:
        chunks = [part.strip() for part in text.split("<tool_call>")[1:] if part.strip()]
    calls = []
    for index, chunk in enumerate(chunks):
        cleaned = re.sub(r",\s*,", ",", chunk.strip())
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get("arguments") or payload.get("args") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not name:
            continue
        calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": str(name), "arguments": json.dumps(arguments)},
            }
        )
    return calls


def _strip_tool_markup(text: str) -> str:
    if not text or "<tool_call>" not in text:
        return (text or "").strip()
    cleaned = _TOOL_CALL_BLOCK.sub("", text)
    cleaned = re.sub(r"</?tool_call>", "", cleaned)
    return cleaned.strip()


# MIN_SPOKEN_CHARS = 25: a simulated-user turn that carried reasoning and
# left fewer spoken characters than this was reasoning with no spoken
# line: the writer thought and never typed, so the turn is retried, never
# emitted as a fragment. The floor applies only to turns that carried
# reasoning; a bare ``yes`` or ``order 4821`` with no ``<think>`` is a
# real user turn and passes untouched (#284) (convention, untested: 25
# chars is about the shortest full user sentence seen in source traces).
MIN_SPOKEN_CHARS = 25


def _strip_think(text: str) -> str:
    """Drop a thinking model's reasoning markup. A closed block goes whole;
    an unclosed ``<think>`` (the token cap landed inside it) goes to the
    end. What is left is the reply, which is what a grader, a marker and
    the next turn's history should see (#264)."""
    return split_reasoning(text)[0]


def _note_user_turn(turn_stats: dict | None, closed: int, unclosed: bool) -> None:
    """One simulated-user reply came back: count the turn, and count it
    again as stripped when it carried reasoning, and as unclosed when the
    reasoning was cut off. The run reports counts and shares per arm
    (#284: 18 unclosed on one arm, 0 on the other)."""
    if not turn_stats:
        return
    lock = turn_stats.get("lock")
    with lock if lock is not None else contextlib.nullcontext():
        turn_stats["user_turns"] = turn_stats.get("user_turns", 0) + 1
        if closed or unclosed:
            turn_stats["user_think_stripped"] = turn_stats.get("user_think_stripped", 0) + 1
        if unclosed:
            turn_stats["user_think_unclosed"] = turn_stats.get("user_think_unclosed", 0) + 1


def _spoken_text(reply: dict) -> str:
    return _strip_tool_markup(_strip_think(str(reply.get("content") or "")))


def _calls_from_reply(reply: dict) -> tuple[list[dict], dict]:
    native = reply.get("tool_calls") or []
    if native:
        # The reply goes back to the server as the assistant turn; its
        # private fields (_logprobs, _finish_reason, _all) must not.
        return native, {k: v for k, v in reply.items() if not str(k).startswith("_")}
    content = reply.get("content") or ""
    parsed = parse_text_tool_calls(content)
    if not parsed:
        return [], reply
    spoken = _strip_tool_markup(content)
    return parsed, {"role": "assistant", "content": spoken or None, "tool_calls": parsed}


def _has_tool_steps(steps: list) -> bool:
    return any(isinstance(s, dict) and s.get("tool") for s in steps)


_PREAMBLE = re.compile(
    r"^(sure|ok|okay|got it|let me|i will|i'll|one (sec|moment)|hang on)\b", re.I
)
_STOCK_CLOSE = re.compile(
    r"(\s*let me know if you need (anything|any(thing)? else)!?\s*)+$",
    re.I,
)


# PREAMBLE_MAX_WORDS = 24: an agent line that opens like "let me" and is
# no longer than this is a preamble before a tool call, so the loop lets
# the agent speak once more instead of handing the turn to the user
# (convention, untested).
PREAMBLE_MAX_WORDS = 24


def _looks_unfinished(text: str, steps: list) -> bool:
    words = (text or "").strip().split()
    if not words:
        return True
    head = " ".join(words[:8])
    return bool(_PREAMBLE.match(head) and len(words) <= PREAMBLE_MAX_WORDS)


def _last_agent_utterance(steps: list) -> tuple[str, int]:
    """Last spoken agent line, plus the index of the last tool step (-1 if none)."""
    last_tool = -1
    before = ""
    after = ""
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("tool"):
            last_tool = i
        text = str(step.get("text") or "")
        if not text:
            continue
        if last_tool >= 0 and i > last_tool:
            after = text
        else:
            before = text
    if last_tool >= 0:
        return after, last_tool
    return before, last_tool


def _finish_on_agent(steps: list, final_text: str) -> dict:
    """Normal rows end on the agent's last utterance, not a dangling user line.

    After tools succeed, ``final_text`` is the post-tool utterance. An earlier
    clarifying question is not reused as the closer.
    """
    while (
        steps
        and isinstance(steps[-1], dict)
        and steps[-1].get("user")
        and not steps[-1].get("text")
        and not steps[-1].get("tool")
    ):
        steps.pop()
    spoken, last_tool = _last_agent_utterance(steps)
    passed = str(final_text or "")
    if spoken:
        return {"steps": steps, "final_text": spoken}
    if last_tool < 0:
        return {"steps": steps, "final_text": passed}
    earlier = {
        str(s.get("text") or "")
        for s in steps
        if isinstance(s, dict) and str(s.get("text") or "").strip()
    }
    if passed and passed not in earlier:
        spoken = passed
    return {"steps": steps, "final_text": spoken}


def _agent_asked(text: str) -> bool:
    """True when the agent actually asked the human something."""
    t = str(text or "").strip()
    return bool(t) and ("?" in t)


#: ``simulate(patience=)``: how long the simulated person keeps answering
#: the agent's questions. The first question is always attempted; from the
#: second on the person may walk away at these odds, per level. ``endless``
#: is what every run did before the knob existed: every question answered
#: until the depth cap, so whether a thread ended was decided by the turn
#: budget and never by what the agent said, and no rubric criterion about
#: asking could fail (#289).
PATIENCE_LEVELS = ("short", "normal", "endless")
# PATIENCE_HAZARDS: per level, (second, later) = the chance the person
# leaves at the agent's second question and at every question after it.
# normal = (0.35, 0.60) and short = (0.60, 0.90) are a convention, not a
# measurement: the 63% of asked source threads that ended with the person
# walking away (#289) says the shape is common, not what the per-question
# hazard is. The literature gives a direction, not a number: benchmark
# users never leave (tau-bench 2406.12045 and tau2-bench 2506.07982 stop
# only on goal or out-of-scope), simulated users are more cooperative than
# humans (2601.17087: questions in 18.8% of simulated user turns against
# 9.8% for humans, politeness markers in 39.2% against 19.9%), and an
# impatient persona drops task success from 0.675 to 0.15-0.20
# (2605.12894, tau-trait). To ground a level, fit a Kaplan-Meier hazard
# per question index on production traces (of the threads still answering
# at question k, the share that leave at k) and pass it as
# ``patience={"second": p, "later": q}``.
PATIENCE_HAZARDS: dict[str, tuple[float, float]] = {
    "short": (0.6, 0.9),
    "normal": (0.35, 0.6),
    "endless": (0.0, 0.0),
}
Patience = str | Mapping[str, float] | tuple[float, float] | list[float]


def patience_hazards(patience: Patience | None) -> tuple[float, float]:
    """``(second, later)`` for a patience level name, or for a custom table
    given as ``{"second": p, "later": q}`` or ``(p, q)``, each in 0..1.
    ``None`` is ``normal``. A name outside ``PATIENCE_LEVELS`` or a chance
    outside 0..1 is refused with the fix."""
    if patience is None:
        return PATIENCE_HAZARDS["normal"]
    if isinstance(patience, str):
        level = patience.strip().lower() or "normal"
        if level not in PATIENCE_HAZARDS:
            raise ValueError(
                f"patience={patience!r} is not a level; use one of "
                + ", ".join(repr(p) for p in PATIENCE_LEVELS)
                + ', or a table {"second": p, "later": q} of walk-away chances'
            )
        return PATIENCE_HAZARDS[level]
    raw_second: Any
    raw_later: Any
    if isinstance(patience, Mapping):
        raw_second, raw_later = patience.get("second"), patience.get("later")
    else:
        items = tuple(patience)
        raw_second, raw_later = (items[0], items[1]) if len(items) == 2 else (None, None)  # noqa: PLR2004  # a patience table is a pair (second, later)
    try:
        if raw_second is None or raw_later is None:
            raise TypeError("patience table is missing a value")
        second, later = float(raw_second), float(raw_later)
    except (TypeError, ValueError):
        raise ValueError(
            'patience= takes a level name or a table {"second": p, "later": q}: the chance '
            "the person leaves at the agent's second question and at every later one"
        ) from None
    if not (0.0 <= second <= 1.0 and 0.0 <= later <= 1.0):
        raise ValueError(
            f"patience table {patience!r} has a chance outside 0..1; "
            "second and later are probabilities of leaving"
        )
    return second, later


def patience_may_leave(patience: Patience | None) -> bool:
    """True when this patience can end a thread: any non-zero hazard."""
    return any(h > 0 for h in patience_hazards(patience))


#: What ``_user_followup`` returns when the person gives up on the question.
USER_LEFT = "[leaves]"
_LEAVES = re.compile(r"\W*leaves\W*", re.I)


def _draw(message: str, turn_i: int, salt: str) -> float:
    """A uniform draw in [0, 1), fixed by the message and turn so a seeded
    run reproduces exactly."""
    digest = hashlib.sha256(f"{message}:{turn_i}:{salt}".encode()).hexdigest()
    return int(digest[:8], 16) / float(1 << 32)


def _walk_away_hazard(patience: Patience | None, questions: int) -> float:
    """Chance the person leaves instead of answering, given how many
    questions the agent already asked on this thread (not counting this
    one). Zero on the first question: the person always tries once."""
    second, later = patience_hazards(patience)
    if questions <= 0:
        return 0.0
    return second if questions == 1 else later


def _user_turn_cap(budget: int) -> int:
    """How many times the person speaks at most on a thread of ``budget``
    turns: half of it, because turns alternate user, agent, user, agent
    (structural, not a knob; ``avg_turns`` sets the budget). The budget's
    rule, kept apart from the person's patience: a thread the cap ends is
    not one the person left."""
    return max(2, int(budget) // 2)


def _user_walks_away(
    message: str,
    turn_i: int,
    *,
    questions: int = 0,
    patience: Patience | None = "normal",
) -> bool:
    """True when the person gives up on the agent's question rather than
    answer it (#289): the patience hazard at this question index, and
    nothing about the budget. Deterministic in the message and turn, so a
    seeded run reproduces."""
    hazard = _walk_away_hazard(patience, questions)
    if hazard <= 0:
        return False
    return _draw(message, turn_i, "abandon") < hazard


def _user_left(text: str) -> bool:
    """The user-sim's answer was the leave mark and nothing else."""
    body = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip()
    return bool(body) and bool(_LEAVES.fullmatch(body))


def ended_on_question(rows) -> dict:
    """``{"share", "n", "user_left"}``: of ``n`` rows, the share that ended
    on the agent's question, and how many of those the person left
    (``ended_by="user_left"``; the rest hit the turn budget).

    Report this next to any criterion about how the agent asks. Before #289
    it was structurally zero, so a lane could not tell "the agent never asked
    badly" from "the loop cannot produce that ending". The engine writes it
    to ``search["ended_on_question"]`` on every run.
    """
    total = 0
    ended = 0
    left = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        total += 1
        turns = row.get("messages") or row.get("trajectory") or []
        last_agent = ""
        for turn in turns:
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                last_agent = str(turn.get("content") or "")
        if not turns:
            last_agent = str(row.get("final_text") or "")
        if last_agent and _agent_asked(last_agent):
            ended += 1
        if row.get("ended_by") == "user_left":
            left += 1
    return {
        "share": (ended / total if total else 0.0),
        "n": total,
        "user_left": left,
    }


#: A thread budget under four turns has no room for a follow-up: the
#: opener, the agent's reply, one more user line and the agent's answer to
#: it (structural, not a tuning).
_FOLLOWUP_MIN_BUDGET = 4


def _want_followup(
    message: str,
    turn_i: int,
    *,
    user_turns: int = 1,
    budget: int = 6,
    agent_text: str = "",
    questions: int = 0,
    patience: Patience | None = "normal",
) -> bool:
    """True when the human would naturally speak again.

    A question or refusal earns an answer while the turn budget has
    room: the depth cap is ``budget // 2`` user turns (avg_turns=12
    allows 6, avg_turns=4 allows 2). Otherwise the thread continues with
    probability ``1 - 1/cap``, which makes the mean depth track the cap.
    A budget under 4 still ends on the agent.

    Before this, any reply that was not a question, a refusal, or a
    recognised success ended the thread, and a completed action only
    continued on a fixed coin flip. "Your reservation has been cancelled"
    matches none of those, so threads died at one user turn and the mean
    sat near 1.5 whatever ``avg_turns`` said: measured 1.54 at avg_turns=6
    and 1.52 at avg_turns=10 on the same spec. That silently caps every
    behaviour that needs three turns to happen at all. Confirm-before-
    acting is the clearest case: the user asks, the agent names the action
    and asks, the user says yes, the agent acts. At 1.5 user turns most
    rollouts never reach the write, so the rule is never exercised and the
    training set cannot demonstrate it.

    A question no longer earns an answer unconditionally. ``questions`` is
    how many the agent already asked on this thread; from the second one
    the person may walk away at the odds ``patience`` sets (#289). Before
    this, whether a thread ended was decided by the turn budget and never
    by what the agent said, so no rubric criterion about asking could
    fail: in source traces 63% of asked threads ended with the person
    walking away, in generated data 19-22%, all of those depth-cap cuts.
    """
    if int(budget) <= 1:
        # ``max_turns=1``: one user line, one reply, whatever the reply
        # says. Before this the question branch below ran first and
        # ``_user_turn_cap`` never goes under 2, so any reply holding a
        # "?" (a model reasoning in plain text does that on 15-65% of
        # rows) earned a follow-up written by the agent model itself, and
        # that second reply was what the grader scored (#586).
        return False
    cap = _user_turn_cap(budget)
    if int(user_turns) >= cap:
        return False
    text = str(agent_text or "")
    if _agent_asked(text):
        # Asking is free in simulation and costly in reality. This used to be
        # an unconditional True (#289); now the person always tries the first
        # question and from the second on may walk away at the odds
        # ``patience`` sets. Short threads (budget under 4) never reach a
        # second question, so the old contract holds there: with room for
        # one exchange, abandoning would leave the question as the whole
        # rollout and fill the set with stubs.
        return not _user_walks_away(message, turn_i, questions=questions, patience=patience)
    if _AGENT_REFUSAL.search(text):
        return True
    if int(budget) < _FOLLOWUP_MIN_BUDGET:
        # A short thread still ends on the agent: with room for one user line
        # there is nothing a second one could be for.
        return False
    # Geometric with p = 1 - 1/cap: a thread of cap turns in expectation,
    # deterministic in the message and turn so a seeded run reproduces
    # (structural: the mean follows avg_turns, which is the knob; human
    # threads with an assistant average 7 to 8 turns on SimulatorArena,
    # 2510.05444, the default avg_turns sits near that).
    return _draw(message, turn_i, "react") < (1.0 - 1.0 / float(cap))


# USER_TURN_TEMPERATURE = 0.475: sampling temperature of a simulated
# user's follow-up line (the opener's diversity comes from the writer's
# band, WRITER_TEMP_LO..HI). The midpoint of the 0.40-0.55 band the code
# used to name and never sample. Low on purpose: a follow-up must stay
# consistent with the thread and passes an accept gate with at most three
# tries, so a hotter draw costs retries. Benchmarks disagree with each
# other: tau-bench runs its user at 1.0 (2406.12045), tau2-bench at 0
# (2506.07982), SimulatorArena at 0.7 (2510.05444); none measured the
# effect. Convention, untested; ``local_model(user_temperature=)`` moves it.
USER_TURN_TEMPERATURE = 0.475
# HUMAN_TOOL_TEMPERATURE = 0.9: temperature of the answer a ``kind="human"``
# tool gets. Hotter than a follow-up because it is one shot, no accept
# gate, and the stance (mistaken, unsure, adversarial) has to come through
# in the wording; 78% plain-yes answers were measured at the old fixed
# posture, not at a temperature (convention, untested; the same
# ``user_temperature`` overrides it).
HUMAN_TOOL_TEMPERATURE = 0.9
# USER_TURN_MAX_TOKENS = 180 / USER_RETRY_MAX_TOKENS = 120: reply budgets
# of the first and the retried follow-up prompt; a user line is one short
# line, and the retry prompt asks for less (convention, untested).
USER_TURN_MAX_TOKENS = 180
USER_RETRY_MAX_TOKENS = 120
# HUMAN_TOOL_MAX_TOKENS = 120: "one or two short lines" (convention).
HUMAN_TOOL_MAX_TOKENS = 120
# OPENER_MAX_TOKENS = 120: the agent's greeting when it opens (convention).
OPENER_MAX_TOKENS = 120
#: USER_TURN_WAIT_S = 5 / USER_TURN_WAIT_ASKED_S = 8 / USER_TURN_WAIT_CAP_S =
#: 30: seconds a follow-up may take, half the rollout timeout clamped to
#: this band. Floor, not ceiling: a 5 s wait on a busy endpoint silently
#: killed every follow-up and collapsed whole datasets to single-turn; an
#: answer to a question gets the longer floor (measured on the hosted pool,
#: the cap is convention).
USER_TURN_WAIT_S = 5.0
USER_TURN_WAIT_ASKED_S = 8.0
USER_TURN_WAIT_CAP_S = 30.0
# USER_TRACE_CHARS = 2400: how much of the thread the user simulator sees,
# newest turns first (about 800 tokens of a 4k window; convention).
USER_TRACE_CHARS = 2400
# USER_TURN_MAX_CHARS = 2000: a follow-up longer than this is a document,
# not a line, and is dropped (convention, untested).
USER_TURN_MAX_CHARS = 2000
# DETAIL_HINTS_MAX = 8 / TOOL_WORLD_MAX = 12: how many parameter names and
# tool names the user-sim prompt lists, so a 40-tool agent does not fill
# the prompt with its schema (convention).
DETAIL_HINTS_MAX = 8
TOOL_WORLD_MAX = 12
_USER_SIM_SYSTEM = (
    "You write only the human's next spoken line. You are not the assistant. "
    "Ordinary speech. No emojis, no em dashes, no thanks, no you're welcome. "
    "Stay in the same world as the opening line and the tools on this thread. "
    "If the agent asked a question, answer it with a concrete detail a person "
    "here would know ({hints}whatever this thread is actually about). "
    "{leave_note}"
    "{code_note}"
    "If they already acted, react: push back, correct them, or ask for the next thing. "
    "Do not acknowledge. Do not repeat their question. Do not describe a persona."
)
_CODE_PARAM = re.compile(
    r"\b(repo|repository|branch|commit|pr|pull_request|issue|path|file|diff)\b", re.I
)


def _detail_hints(tools: list | None) -> list[str]:
    """What a person on this thread would know, read off the agent's own
    tool parameters: ``order_id`` becomes "order id". Nothing from any
    other agent's world."""
    out: list[str] = []
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        params = (fn or {}).get("parameters") or {}
        for key in (params.get("properties") or {}) if isinstance(params, dict) else {}:
            label = re.sub(r"[_\-]+", " ", str(key)).strip().lower()
            if label and label not in out and len(label) <= TEXT_HEURISTICS.detail_label_max_chars:
                out.append(label)
    return out[:DETAIL_HINTS_MAX]


_LEAVE_NOTE = (
    "Only when their question asks for something this person could not know, "
    "or asks again what was already answered on this thread, the person may "
    "give up instead: then write exactly [leaves] and nothing else. "
)


def user_sim_system(tools: list | None = None, may_leave: bool = False) -> str:
    """The user simulator's instructions for this agent's world.
    ``may_leave`` adds the one way out: a question this person cannot or
    would not answer may be met with the leave mark (#289)."""
    hints = _detail_hints(tools)
    names = " ".join(_tool_world(tools).split(",")) + " " + " ".join(hints)
    code_note = (
        "Do not invent a repo, pull request, issue, or branch unless this "
        "thread is already about those. "
        if _CODE_PARAM.search(names)
        else ""
    )
    return _USER_SIM_SYSTEM.format(
        hints=(", ".join(hints) + ", ") if hints else "",
        leave_note=_LEAVE_NOTE if may_leave else "",
        code_note=code_note,
    )


_EMOJI = re.compile("[\U0001f300-\U0001faff\U00002700-\U000027bf\U0001f600-\U0001f64f]+")
_STOCK_HIT = re.compile(
    r"(i('d| would) be happy to help|of course!|have a great day|"
    r"feel free to (reach out|ask)|let me know if you need|"
    r"i understand your (frustration|concern)|how can i (help|assist) you today)",
    re.I,
)
_MD_MARK = re.compile(r"^#{1,6}\s+|\*\*(.+?)\*\*", re.M)


def _scrub_ai_traces(text: str) -> str:
    """User-side only. Agent voice is left alone."""
    out = _EMOJI.sub("", str(text or ""))
    out = out.replace("\u2014", ", ").replace("\u2013", ", ")
    out = _STOCK_CLOSE.sub("", out)
    out = _STOCK_HIT.sub("", out)
    out = _MD_MARK.sub(lambda m: m.group(1) or "", out)
    return re.sub(r"[ \t]+\n", "\n", re.sub(r" {2,}", " ", out)).strip()


def _render_user_trace(
    messages: list[dict] | None, steps: list[dict] | None, *, limit: int = USER_TRACE_CHARS
) -> str:
    """Plain-text script. Never packed as chat roles."""
    lines: list[str] = []
    if messages:
        for item in messages:
            role = str(item.get("role") or "")
            if role == "system":
                continue
            if role == "user":
                lines.append(f"You said: {item.get('content') or ''}")
                continue
            if role == "assistant":
                spoken = str(item.get("content") or "").strip()
                if spoken:
                    lines.append(f"The agent replied: {spoken}")
                for call in item.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    lines.append(f"The agent used {fn.get('name')}({fn.get('arguments') or '{}'})")
                continue
            if role == "tool":
                name = item.get("name") or "tool"
                lines.append(f"Result ({name}): {str(item.get('content') or '')[:300]}")
    elif steps:
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("user"):
                lines.append(f"You said: {step['user']}")
            if step.get("tool"):
                args = step.get("arguments") or {}
                lines.append(f"The agent used {step.get('tool')}({args})")
                lines.append(f"Result ({step.get('tool')}): {str(step.get('result') or '')[:300]}")
            elif step.get("text"):
                lines.append(f"The agent replied: {step['text']}")
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    # Keep the newest turns: a follow-up must see what was just said,
    # not the opening of a long thread.
    tail = text[-limit:]
    cut = tail.find("\n")
    return tail[cut + 1 :] if 0 <= cut < len(tail) // 4 else tail


_RUBBER_STAMP = re.compile(
    r"^(thanks|thank you|thx|ok|okay|great|perfect|awesome|got it|"
    r"sounds good|cool|you're welcome|you are welcome)[!., ]*$",
    re.I,
)
_THANKS_WORDS = {"thanks", "thank", "thx", "appreciate", "welcome", "glad"}
_ID_FOLLOW = re.compile(
    r"[#/]|\b(pr|issue|repo|branch|sku|store|order|asin|cart)\s*#?\d+"
    r"|\b[\w.-]+/[\w.-]+\b",
    re.I,
)
_CODING_MARK = re.compile(
    r"\b(repo|repository|pull request|\bpr\b|issue\s*#|git branch|"
    r"branch feature/)\b",
    re.I,
)
_CONFIRM_ONLY = re.compile(
    r"^(the )?.{0,40}(successfully|correctly|all good|no issues)\.?$",
    re.I,
)
_AGENT_QUESTION = re.compile(r"\?\s*$|\b(which|what|who|where|can you|could you)\b", re.I)
_AGENT_REFUSAL = re.compile(
    r"\b(can't|cannot|won't|unable|not allowed|against (the )?(policy|rules?))\b",
    re.I,
)


_TEXTURE_NOTES = {
    "lowercase": "Keep every letter small. Do not mention how you type.",
    "no_punctuation": "Leave out end marks. Never write the word punctuation.",
    "abbreviations": "Use short forms like u, pls, thx, rn.",
    "typo": "Let a typo or two through. Never point them out.",
    "clipped": "Write in short clipped fragments.",
    "run_on": "Run your thoughts together in one long sentence.",
    "standard": "Capitalize normally and end sentences with the usual marks.",
}
_TONE_NOTES = {
    "impatient": "You are impatient and want this done now.",
    "frustrated": "You are frustrated until this is actually solved.",
    "chatty": "You chat a little and add a bit of context.",
    "polite": "Stay polite, but do not just thank them.",
    "curt": "Answer curtly, a few words.",
    "sarcastic": "You are sarcastic and dry. Never explain the sarcasm.",
}


def _persona_from_tags(tags: dict | None) -> str:
    """Typing and mood notes from the situation's own tags, so the person
    the writer drew on turn one is the same person on turn five."""
    if not isinstance(tags, dict):
        return ""
    notes = [
        _TEXTURE_NOTES.get(str(tags.get("texture") or ""), ""),
        _TONE_NOTES.get(str(tags.get("tone") or ""), ""),
    ]
    return " ".join(n for n in notes if n)


def _persona_notes(prior: str) -> str:
    """Backstage typing notes inferred from the opening line. Never copy these."""
    text = str(prior or "")
    notes: list[str] = []
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) < LOWERCASE_UPPER_SHARE:
        notes.append("Keep every letter small. Do not mention how you type.")
    elif text[:1].isupper():
        notes.append("Capitalize normally, the way the opening line does.")
    if not re.search(r"[.!?]", text):
        notes.append("Leave out end marks. Never write the word punctuation.")
    elif re.search(r"[.!?]\s*$", text):
        notes.append("End sentences with the usual marks.")
    if re.search(r"\b(again|seriously|come on|still not|ridiculous)\b", text, re.I):
        notes.append("You are still frustrated until this is actually solved.")
    elif re.search(r"\b(asap|right now|waiting)\b", text, re.I):
        notes.append("You are still in a hurry.")
    elif (
        re.search(r"\b(please|thanks|thank you)\b", text, re.I)
        and len(text) > TEXT_HEURISTICS.polite_filler_min_chars
    ):
        notes.append("Stay polite, but do not just thank them.")
    return " ".join(notes)


# The user-side scrubbers and gates below are English lexicons and
# overlap rules. Fixed on purpose: they describe the hosted writer's
# habits (stock closers, thanks-only turns, echoes), not the customer's
# domain, and a run in another language sets ``user_model=`` to a writer
# that does not produce them. The thresholds are conventions, untested.
# LOWERCASE_UPPER_SHARE = 0.08: an opener with fewer capitals than this
# share of its letters is typed lowercase, and the persona keeps it so.
LOWERCASE_UPPER_SHARE = 0.08
# THANKS_SHARE = 0.25: a follow-up whose words are a quarter thanks and
# carries no new ask is a rubber stamp, not a turn.
THANKS_SHARE = 0.25
# ECHO_OVERLAP = 0.70: a user line sharing this share of its words with
# the agent's line is a restatement, not an answer (identifiers exempt).
ECHO_OVERLAP = 0.70
# REPEAT_OVERLAP = 0.78: a user line this close to an earlier user line is
# the same line again (Self-Instruct drops a new instruction above ROUGE-L
# 0.7 against the pool, 2212.10560; a word-set Jaccard-style overlap is a
# coarser measure, so the bar sits a little higher).
REPEAT_OVERLAP = 0.78


def _mostly_thanks(text: str) -> bool:
    words = re.findall(r"[a-z']+", str(text or "").lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _THANKS_WORDS)
    if hits / len(words) < THANKS_SHARE:
        return False
    return not re.search(r"\b(also|but|instead|wrong|pr|issue|repo|#\d+)\b", text, re.I)


def _off_world_coding(text: str, prior: str, agent_text: str) -> bool:
    """Git identifiers do not belong on a shopping or orders thread."""
    blob = f"{prior} {agent_text}"
    return bool(_CODING_MARK.search(text) and not _CODING_MARK.search(blob))


_ASKS_CONFIRM = re.compile(
    r"\byes/no\b|\(yes/no\)|\bplease confirm\b|\bconfirm (?:if|whether|that)\b|"
    r"\bshall i (?:proceed|go ahead)\b|\bwould you like (?:me )?to proceed\b|"
    r"\bdo you want me to\b[^.!\n]{0,60}\?|\bproceed with (?:the|this)\b"
    r"[^.!\n]{0,60}\?",
    re.I,
)


def _accept_followup(text: str, prior: str, agent_text: str) -> bool:
    from .generator import usable_user_message

    if not text or text == prior or len(text) > USER_TURN_MAX_CHARS:
        return False
    # A person never types call syntax: name_with_underscores( or a JSON dump.
    if re.search(r"\b\w+_\w+\s*\(", text) or text.count('"') >= TEXT_HEURISTICS.code_quote_marks:
        return False
    # A bare yes is filler everywhere except where the agent asked for
    # exactly that. Confirm-then-execute data depends on it, so it wins
    # over the echo, stamp, and length gates below.
    if (
        _ASKS_CONFIRM.search(str(agent_text or ""))
        and len(text) <= TEXT_HEURISTICS.confirm_reply_max_chars
        and re.match(
            r"^(yes|yep|yeah|sure|ok(ay)?|confirmed|do it|"
            r"go ahead|no\b)",
            text,
            re.I,
        )
    ):
        return True
    if _RUBBER_STAMP.match(text) or _CONFIRM_ONLY.match(text):
        return False
    if _mostly_thanks(text):
        return False
    if _off_world_coding(text, prior, agent_text):
        return False
    if _ID_FOLLOW.search(text) and len(text) >= TEXT_HEURISTICS.id_reply_min_chars:
        return True
    if _echoes_agent(text, agent_text):
        return False
    if len(text) < TEXT_HEURISTICS.followup_min_chars:
        return False
    return usable_user_message(text)


def _repeats_user_history(text: str, messages: list[dict] | None) -> bool:
    current = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    if len(current) < TEXT_HEURISTICS.history_min_words:
        return False
    for message in messages or []:
        if message.get("role") != "user":
            continue
        prior = set(re.findall(r"[a-z0-9]+", str(message.get("content") or "").lower()))
        if len(prior) < TEXT_HEURISTICS.history_min_words:
            continue
        overlap = len(current & prior) / min(len(current), len(prior))
        if overlap >= REPEAT_OVERLAP:
            return True
    return False


def _tool_world(tools: list | None) -> str:
    names: list[str] = []
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = str((fn or {}).get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return ", ".join(names[:TOOL_WORLD_MAX])


def _user_followup(
    base_url: str,
    model: str,
    prior: str,
    agent_text: str,
    *,
    api_key: str | None,
    timeout: float,
    messages: list[dict] | None = None,
    steps: list[dict] | None = None,
    want: str = "",
    tools: list | None = None,
    force: bool = False,
    persona_tags: dict | None = None,
    extra: Mapping[str, Any] | None = None,
    turn_stats: dict | None = None,
    may_leave: bool = False,
    temperature: float = USER_TURN_TEMPERATURE,
) -> str:
    """The simulated user's next line, or ``""`` when the writer produced
    none worth keeping. ``extra`` carries the agent's request fields
    (``thinking=False`` on ``local_model``) so the user model is asked not
    to reason either; whatever it still emits as ``<think>`` is stripped
    before the words become a user turn, an unclosed block is dropped
    whole, and a turn that was reasoning with no spoken line is retried
    rather than emitted as a fragment (#284)."""
    from .generator import _realize_typed_message, _strip_directive_phrases, clean_user_message

    trace = _render_user_trace(messages, steps)
    if not trace:
        trace = f"You said: {prior[:500]}\nThe agent replied: {(agent_text or '')[:500]}"
    opening = str(want or prior).strip()
    # explicit tags win over notes inferred from the opening line
    persona = _persona_from_tags(persona_tags) or _persona_notes(opening)
    need = opening[:180] if opening else "this handled"
    who = f"You are the human who needs {need}."
    if persona:
        who = f"{who} {persona}"
    world = _tool_world(tools)
    if world:
        who = f"{who} This agent's tools are: {world}."
    asked = _agent_asked(agent_text)
    nudge = ""
    if asked:
        nudge = (
            " They asked you a question. Answer it with one concrete detail "
            "from this same world. One short line. If they asked you to "
            "confirm an action, a plain yes go ahead, or a no with a reason, "
            "is a real answer."
        )
    elif force:
        nudge = (
            " The matter is not resolved yet. Add exactly one NEW relevant "
            "fact, correction, constraint, observable problem, or related next "
            "request. Never repeat an earlier request. Never explain the "
            "assistant's tools or policies. Speak only as the human. Do not "
            "thank them or end the conversation."
        )
    body = f"This is what's been said so far:\n{trace}\n\nYour turn to respond. {who}{nudge}"
    retry_body = (
        f"This is what's been said so far:\n{trace}\n\n"
        + (
            "Answer their last question with one concrete detail from this same world. "
            if asked
            else "Continue with a different, concrete fact, correction, constraint, "
            "or related next request that has not appeared earlier. "
        )
        + "Never restate the request or describe tool limitations. One short "
        "line. No thanks. No git repo unless this thread is already about git."
    )
    tags: dict = {}
    if persona and "every letter small" in persona:
        tags["texture"] = "lowercase"
    if persona and "Leave out end marks" in persona:
        tags["texture"] = "no_punctuation"
    if persona and "Capitalize normally" in persona and "texture" not in tags:
        tags["texture"] = "standard"
    # Floor, not ceiling: see USER_TURN_WAIT_S.
    wait = max(
        USER_TURN_WAIT_ASKED_S if asked else USER_TURN_WAIT_S,
        min(USER_TURN_WAIT_CAP_S, float(timeout or USER_TURN_WAIT_CAP_S) / 2),
    )
    attempts = list(
        (body, retry_body, retry_body) if force else ((body, retry_body) if asked else (body,))
    )
    retried_for_reasoning = False
    attempt = 0
    while attempt < len(attempts):
        content = attempts[attempt]
        attempt += 1
        try:
            reply = complete(
                base_url,
                model,
                [
                    {"role": "system", "content": user_sim_system(tools, may_leave=may_leave)},
                    {"role": "user", "content": content},
                ],
                tools=None,
                api_key=api_key,
                temperature=float(temperature),
                max_tokens=USER_RETRY_MAX_TOKENS if attempt > 1 else USER_TURN_MAX_TOKENS,
                timeout=wait,
                extra=extra,
            )
        except Exception:
            continue
        spoken, closed, unclosed = split_reasoning(reply.get("content") or "")
        _note_user_turn(turn_stats, closed, unclosed)
        # A leave mark is an answer of its own kind: read it before the
        # reasoning-only retry rule, or a person who thinks then leaves is
        # retried instead of gone (#289).
        if may_leave and _user_left(reply.get("content") or ""):
            return USER_LEFT
        if (closed or unclosed) and len(spoken.strip()) < MIN_SPOKEN_CHARS:
            # Reasoning and no spoken line. One more try with the short
            # prompt; never a fragment, never empty user speech.
            if not retried_for_reasoning:
                retried_for_reasoning = True
                attempts.append(retry_body)
            continue
        text = _scrub_ai_traces(
            _realize_typed_message(_strip_directive_phrases(clean_user_message(spoken)), tags)
        )
        if _accept_followup(text, prior, agent_text) and not _repeats_user_history(text, messages):
            return text
    return ""


def _echoes_agent(user: str, agent: str) -> bool:
    """True when the user turn is mostly a restatement of the assistant.

    Shared identifiers (order ids, ticket numbers) are normal ping-pong,
    not an echo. Compare letter tokens only, and require a high overlap.
    """
    if re.search(r"\b\w*\d\w*\b", user or ""):
        # Carries an identifier: that is the answer, not an echo.
        return False
    u = set(re.findall(r"[a-z]{3,}", (user or "").lower()))
    a = set(re.findall(r"[a-z]{3,}", (agent or "").lower()))
    if len(u) < TEXT_HEURISTICS.echo_min_words or not a:
        # Too few words to call a restatement; short replies are answers.
        return False
    return (len(u & a) / len(u)) >= ECHO_OVERLAP


#: Generation-side cue only; never enters the exported conversation.
_OPENING_CUE = (
    "(You are starting this conversation. Greet the user in one "
    "short line and offer help. Do not mention this instruction.)"
)


def human_tool_names(tools: list[dict] | None) -> set[str]:
    """Tools declared kind='human': their result is the person's answer,
    voiced by the user simulator, never a mock-environment payload."""
    out: set[str] = set()
    for tool in tools or []:
        if isinstance(tool, dict) and tool.get("kind") == "human":
            name = (tool.get("function") or {}).get("name") or tool.get("name")
            if name:
                out.add(str(name))
    return out


def _human_answer(
    base_url: str,
    model: str,
    *,
    want: str,
    question: str,
    api_key: str | None,
    timeout: float,
    stance: str = "",
    extra: Mapping[str, Any] | None = None,
    turn_stats: dict | None = None,
    temperature: float = HUMAN_TOOL_TEMPERATURE,
) -> str:
    """The simulated user answers the agent's question, in character.

    The stance travels with the answer. Without it the person defaults to
    approving whatever they first asked for, which makes the question
    decorative: measured over one 405-row set, 78% of answers were a plain
    yes and 3% redirected.
    """
    posture = {
        "mistaken": "You were wrong about a detail in your first message. "
        "Correct it now rather than confirming.",
        "unsure": "You are not certain. Say what you do not know instead of approving.",
        "hurried": "You are in a rush. Answer in a few words.",
        "retry": "This has failed before. Say what went wrong last time.",
        "contradicts_earlier": "You have changed your mind since your first message. Say so.",
        "ambiguous": "Your first message could be read two ways. Say which you meant.",
        "exploratory": "You are still deciding. Ask something back or hold off.",
        "adversarial": "You want it done anyway and you push back.",
    }.get(str(stance or "").lower(), "")
    msgs = [
        {
            "role": "system",
            "content": (
                "You are the person the assistant is helping. Answer its "
                "question in one or two short lines, in plain chat style. "
                "Approve, refuse, correct a wrong assumption, or give the "
                "detail asked for, whichever your situation actually calls "
                "for. Never mention being simulated." + (" " + posture if posture else "")
            ),
        },
        {
            "role": "user",
            "content": (
                f"What you originally asked the assistant for:\n{want}\n\n"
                f"The assistant now asks you:\n{question}\n\nYour reply:"
            ),
        },
    ]
    reply = complete(
        base_url,
        model,
        msgs,
        api_key=api_key,
        temperature=float(temperature),
        timeout=timeout,
        max_tokens=HUMAN_TOOL_MAX_TOKENS,
        extra=extra,
    )
    spoken, closed, unclosed = split_reasoning(reply.get("content") or "")
    _note_user_turn(turn_stats, closed, unclosed)
    if (closed or unclosed) and len(spoken.strip()) < MIN_SPOKEN_CHARS:
        return ""
    return _strip_tool_markup(spoken).strip()


def _answer_tool_call(env: Any, execute: Callable | None, tool: str, arguments: dict) -> dict:
    """The world's answer to one tool call.

    With ``execute`` the caller's own world answers: their repo, their
    database, their tools, whatever they are. Scheduled faults still
    apply first, so the row's ``faults`` stay truthful. Without it the
    mock world answers, which fits record-shaped tools and not code.
    ``current_rollout`` (thread-local) names the rollout being answered.
    """
    if execute is None:
        return env.call(tool, arguments)
    fault = env._fault_for(tool, arguments)
    if fault is not None:
        return fault
    try:
        result = execute(tool, arguments)
    except Exception as exc:
        return {"status": "error", "reason": public_llm_error(exc)}
    if isinstance(result, dict):
        return result
    return {"status": "ok", "result": result}


# LOCAL_MODEL_TEMPERATURE lives in defaults.py (the monitor samples at
# the same value); it is re-exported here for the callers that read it.
#: Seconds one completion may take. A served model that scaled to zero
#: takes two to three minutes to answer its first request (113 s measured
#: on the account's own endpoint, #302; the hosted judge is the same
#: shape), and the old 60 s dropped every rollout of the first pass and
#: returned an empty run that looked finished.
LOCAL_MODEL_TIMEOUT = 300.0


# REPLY_TOKENS = 768 / REPLY_TOKENS_LARGE = 2048: an agent reply's token
# budget on a small and a large context window. 768 is what a 4k window
# leaves after the history and tools; a coding agent's diff does not fit
# in it, and a reasoning model's thinking does not fit in 2048, which is
# what ``agent_max_tokens`` is for (convention, sized to hosted Qwen).
REPLY_TOKENS = 768
REPLY_TOKENS_LARGE = 2048


def reply_budget(max_tokens: int | None = None) -> int:
    """Tokens one agent reply may use: ``simulate(agent_max_tokens=)`` when
    set, else ``REPLY_TOKENS``, or ``REPLY_TOKENS_LARGE`` above an 8k
    context (``ZP_CONTEXT_TOKENS``)."""
    if max_tokens:
        return int(max_tokens)
    return REPLY_TOKENS if CONTEXT_TOKENS <= LARGE_CONTEXT_TOKENS else REPLY_TOKENS_LARGE


def local_model(
    base_url: str,
    model: str,
    *,
    tools: list[dict],
    system: str = "",
    api_key: str | None = None,
    max_turns: int | None = None,
    avg_turns: float = DEFAULT_AVG_TURNS,
    min_user_turns: int = 1,
    turn_stats: dict | None = None,
    temperature: float = LOCAL_MODEL_TEMPERATURE,
    logprobs: bool | str = False,
    fault_plans: dict | None = None,
    result_shapes: dict | None = None,
    opening_rate: float = 0.0,
    human_tools: set | None = None,
    execute: Callable | None = None,
    timeout: float = LOCAL_MODEL_TIMEOUT,
    max_tokens: int | None = None,
    user_model: str | None = None,
    thinking: bool | None = None,
    patience: Patience | None = "normal",
    user_temperature: float | None = None,
    world_options: WorldOptions | Mapping[str, Any] | None = None,
) -> Callable:
    """Build an agent that talks to any OpenAI-compatible endpoint for ``simulate(agent=...)``.

    Reach for it when the policy under test is a served model: a trained
    adapter behind vLLM, a local server, any chat endpoint. It returns a
    callable that plays the multi-turn agent (tool calls, the simulated
    user, faults) against ``model`` at ``base_url`` with ``tools`` and the
    ``system`` prompt, and every rollout comes back as a row.

    * ``base_url``, ``model``, ``api_key``: where the model is served and
      what to call it.
    * ``thinking``: for reasoning bases such as Qwen3. ``False`` sends
      ``chat_template_kwargs={"enable_thinking": False}`` so the reply is
      the answer, not the reasoning, the way the hosted Qwen path already
      does; ``True`` asks for it; ``None`` (the default) sends nothing and
      leaves the server's default. The same field goes to the simulated
      user when the agent's own model plays it (the default) or
      ``user_model`` sits on the same endpoint, so the customer is asked
      not to reason either; a ``user_model`` on another endpoint keeps
      that server's default. Either way ``<think>`` markup never reaches
      ``step["text"]``, ``final_text``, or a user turn (``step["user"]``
      and the ``messages`` history): what the user model still emits as
      reasoning is stripped before it becomes speech, and a turn that was
      reasoning with no spoken line is retried, then dropped. The run
      reports those under ``search["user_think"]``: ``user_turns``,
      ``stripped`` and ``unclosed`` as counts, ``stripped_share`` and
      ``unclosed_share`` as shares of the user turns, zeros when none.
    * ``result_shapes``: pins what a tool returns, as
      ``{tool_name: example result dict}``. The sandbox fills the example
      on every call instead
      of inventing a record, so a policy branch that only exists for some
      tool results (a credit over $200 must be escalated) is reached on
      purpose rather than by luck. Field names and free text stay as
      written; ids, dates and people are re-drawn per call, and a number
      moves by up to about a third of itself (``900.0`` lands in roughly
      600 to 1200, ``90.0`` in 60 to 120), so pick a template value whose
      whole range sits on the side of the threshold you want. An argument
      that shares a key with the template is echoed back (``invoice_id``
      in, same ``invoice_id`` out). To measure a branch, run the same
      pinned tasks under two shapes, one per side of the rule. Without it
      the situation writer drafts an example per tool
      (``write_result_shapes``) and the branch is exercised at random.
    * ``fault_plans``: schedules faults per ask, as
      ``{message: {tool_name: {"mode": "timeout", "rate": 1.0}}}``, keyed
      by the exact user message, with ``mode`` one of ``timeout``,
      ``malformed``, ``stale``
      or ``permission_denied`` and ``rate`` the chance the fault fires on
      a call. The plan may also carry ``world_state``, ``stance``,
      ``tone`` and ``texture``, which are popped off and shape the world
      and the simulated user for that ask. ``simulate()`` writes these
      itself from ``fault_rate=``; pass your own only to replay a known
      plan (``tasks=`` does this for you).
    * ``timeout``: seconds per completion, ``LOCAL_MODEL_TIMEOUT`` (300)
      by default: a served model that scaled to zero takes two to three
      minutes to answer its first request, and a timeout under that drops
      every rollout of the first pass. When a call still times out the
      run says so in ``data.warnings`` with the fix (raise ``timeout=``,
      or send one throwaway request first so the endpoint is warm).
    * ``patience``: a level name (``PATIENCE_LEVELS``, ``"normal"`` by
      default) or a table ``{"second": p, "later": q}``: the chance the
      person leaves at the agent's second question and at every later
      one, fitted from your own traces (see ``PATIENCE_HAZARDS``).
    * ``user_model`` and ``user_temperature``: the simulated person's
      model (the agent's own by default) and the sampling temperature of
      every simulated-user line, follow-ups (``USER_TURN_TEMPERATURE``)
      and human-tool answers (``HUMAN_TOOL_TEMPERATURE``) alike; ``None``
      keeps those two defaults.
    * ``world_options``: the mock world's dials (a ``WorldOptions`` or the
      same fields as a dict: fault modes, hit counts, name pools, ...);
      ``simulate(advanced={"world": {...}})`` lands here. ``None`` is the
      defaults in ``defaults.py``.
    * ``execute``: your own world ``(tool, arguments) -> result`` in place
      of the mock one. ``max_turns`` / ``avg_turns`` (12.0) cap and shape
      the conversation length; ``avg_turns=1`` is one user line and one
      reply, and the follow-up branch never runs; ``temperature`` (0.8),
      ``max_tokens`` and
      ``logprobs`` are the agent's own sampling, recorded on every row.

    ```python
    agent = wai.local_model("http://localhost:8000/v1", "my-adapter",
                            tools=TOOLS, system=POLICY, thinking=False)
    data = wai.simulate(agent, tools=TOOLS, system_prompt=POLICY, budget=100)
    ```
    """
    tools = _tool_schemas(tools) or []
    hazards = patience_hazards(patience)
    world_opts = WorldOptions.coerce(world_options)
    may_leave = patience_may_leave(patience)
    followup_temp = USER_TURN_TEMPERATURE if user_temperature is None else float(user_temperature)
    human_temp = HUMAN_TOOL_TEMPERATURE if user_temperature is None else float(user_temperature)
    local = threading.local()
    plans = fault_plans if fault_plans is not None else {}
    extras: dict[str, Any] | None = (
        None if thinking is None else {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    )
    # The simulated user's model. None means the agent's own model plays
    # the user (the default); a backend spec moves that role to another
    # model, with the key resolved for that endpoint.
    if user_model:
        user_url, user_name = parse_backend_spec(user_model)
        user_key: str | None = None
    else:
        user_url, user_name, user_key = base_url, model, api_key
    # thinking= reaches the user simulator on the agent's own endpoint.
    # Another server has its own template fields, so it keeps its default.
    user_extras = extras if str(user_url).rstrip("/") == str(base_url).rstrip("/") else None
    shapes = result_shapes if result_shapes is not None else {}
    cap = default_max_turns(n_tools=len(tools)) if max_turns is None else max(1, int(max_turns))
    if avg_turns is not None and float(avg_turns) <= 1:
        # ``avg_turns=1`` is one user line and one reply: the same path
        # ``max_turns=1`` takes, so the sampler's 1 is not lifted back to
        # 2 by the ``min_user_turns`` floor below (#587 guarded the
        # follow-up on ``max_turns`` only; ``avg_turns=1`` still ran two
        # turns on every prompt of a model-backed agent).
        cap = 1
    min_users = max(1, min(int(min_user_turns), max(1, cap // 2)))
    human_names = set(human_tools or set()) | human_tool_names(tools)
    policy_text = str(system or "").strip()

    def agent(message: str) -> dict:
        # New chat every call. Prior turns, tools, and follow-ups do not carry over.
        for attr in ("messages", "steps", "history"):
            if hasattr(local, attr):
                delattr(local, attr)
        plan = dict(plans.get(message) or {})
        world = str(plan.pop("world_state", "") or "")
        stance = str(plan.pop("stance", "") or "")
        persona_tags = {k: plan.pop(k) for k in ("tone", "texture") if plan.get(k)}
        local.env = MockEnvironment(
            tools, faults=plan, world_state=world, result_shapes=shapes, options=world_opts
        )
        turns = split_user_turns(message)
        messages = ([{"role": "system", "content": policy_text}] if policy_text else []) + [
            {"role": "user", "content": turns[0]},
        ]
        # Conversation topology is a map axis, never a hardcoded frame:
        # a deterministic per-situation draw decides whether the agent
        # opens (deployments like tau2 greet first) or the user does.
        opener_text = ""
        if opening_rate > 0:
            draw = int(hashlib.sha256(f"opening:{message}".encode()).hexdigest(), 16) % 10**6
            if draw < float(opening_rate) * 10**6:
                cue = [*list(messages[:-1]), {"role": "user", "content": _OPENING_CUE}]
                greet = complete(
                    base_url,
                    model,
                    cue,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=timeout,
                    max_tokens=OPENER_MAX_TOKENS,
                    extra=extras,
                )
                opener_text = (_spoken_text(greet) or "").strip()
                if opener_text:
                    messages.insert(
                        len(messages) - 1, {"role": "assistant", "content": opener_text}
                    )

        def _done(done_steps: list, final: str, ended_by: str = "") -> dict:
            out = _finish_on_agent(done_steps, final)
            if opener_text:
                out["opener"] = opener_text
                out["opening"] = "agent"
            if ended_by:
                out["ended_by"] = ended_by
            return out

        steps: list[dict] = []
        user_turn = 0
        n_user = 1
        last_user = turns[0]
        final_text = ""
        budget = sample_turn_budget(
            0, message, cap, avg_turns=avg_turns, running_mean=running_turn_mean(turn_stats)
        )
        budget = min(cap, max(budget, min_users * 2))
        remaining = budget
        turn_i = 0
        closing_bonus = False
        while remaining > 0:
            remaining -= 1
            turn_i += 1
            reply = complete(
                base_url,
                model,
                messages,
                tools=tools,
                api_key=api_key,
                temperature=temperature,
                timeout=timeout,
                max_tokens=reply_budget(max_tokens),
                logprobs=logprobs,
                extra=extras,
            )
            calls, assistant = _calls_from_reply(reply)
            # One agent turn, one set of sampling facts, on its first step.
            turn_meta = _turn_meta(reply)
            spoken = (_spoken_text(reply) or "").strip()
            if spoken:
                final_text = spoken
            if not spoken and not calls:
                if remaining > 0:
                    continue
                return _done(steps, final_text)
            if calls:
                messages.append(assistant)
                attached = False
                for call in calls:
                    fn = call.get("function", {})
                    try:
                        arguments = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    if fn.get("name", "") in human_names:
                        # A human tool's result is the person's answer,
                        # voiced by the user simulator - never a mock
                        # payload. It also counts as a user turn.
                        answer = _human_answer(
                            user_url,
                            user_name,
                            want=turns[0],
                            question=str(
                                arguments.get("question") or arguments.get("summary") or ""
                            ),
                            api_key=user_key,
                            timeout=timeout,
                            stance=stance,
                            extra=user_extras,
                            turn_stats=turn_stats,
                            temperature=human_temp,
                        )
                        # an empty answer used to default to "go ahead", which
                        # silently taught the agent that asking always clears
                        result = {"answer": answer or "(no reply yet)"}
                        n_user += 1
                        step = {
                            "tool": fn.get("name", ""),
                            "arguments": arguments,
                            "result": result,
                        }
                        if spoken and not attached:
                            step["text"] = spoken
                            attached = True
                        if turn_meta:
                            step.update(turn_meta)
                            turn_meta = {}
                        steps.append(step)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id", ""),
                                "content": json.dumps(result),
                            }
                        )
                        continue
                    result = _answer_tool_call(local.env, execute, fn.get("name", ""), arguments)
                    step = {"tool": fn.get("name", ""), "arguments": arguments, "result": result}
                    if spoken and not attached:
                        step["text"] = spoken
                        attached = True
                    if turn_meta:
                        step.update(turn_meta)
                        turn_meta = {}
                    steps.append(step)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": json.dumps(result),
                        }
                    )
                if remaining <= 0 and not closing_bonus:
                    remaining = 1
                    closing_bonus = True
                continue
            if spoken:
                prev = ""
                for s in reversed(steps):
                    if isinstance(s, dict) and str(s.get("text") or "").strip():
                        prev = str(s["text"]).strip()
                        break
                if prev and spoken.strip() == prev and n_user >= 2:  # noqa: PLR2004  # the second user turn is where an echo can start
                    return _done(steps, prev)
            messages.append({"role": "assistant", "content": spoken})
            if spoken:
                steps.append({"text": spoken, **turn_meta})
                turn_meta = {}
            # After the agent speaks, the user replies only if the thread
            # is still open. Do not stack assistant-only variants of the
            # same line. One extra agent beat is allowed only for a short
            # preamble before the first tool call.
            if not _has_tool_steps(steps) and _looks_unfinished(spoken, steps) and remaining > 0:
                continue
            room = remaining > 0
            if room and user_turn + 1 < len(turns):
                user_turn += 1
                n_user += 1
                last_user = turns[user_turn]
                steps.append({"user": last_user})
                messages.append({"role": "user", "content": last_user})
                continue
            need_first = n_user < 2  # noqa: PLR2004  # the first user turn is structural
            force_followup = n_user < min_users
            # Questions the agent already asked on this thread, not counting
            # this one: the person's patience runs out with them (#289).
            prior_questions = sum(
                1 for s in steps if isinstance(s, dict) and _agent_asked(str(s.get("text") or ""))
            ) - (1 if _agent_asked(spoken) else 0)
            if (room or need_first) and (
                force_followup
                or _want_followup(
                    message,
                    turn_i,
                    user_turns=n_user,
                    budget=budget,
                    agent_text=spoken,
                    questions=prior_questions,
                    patience=hazards,
                )
            ):
                follow = _user_followup(
                    user_url,
                    user_name,
                    last_user,
                    spoken,
                    api_key=user_key,
                    timeout=timeout,
                    messages=messages,
                    steps=steps,
                    want=turns[0],
                    persona_tags=persona_tags,
                    tools=tools,
                    force=force_followup,
                    extra=user_extras,
                    turn_stats=turn_stats,
                    may_leave=may_leave,
                    temperature=followup_temp,
                )
                if follow == USER_LEFT:
                    # The person could not or would not answer this one.
                    return _done(steps, spoken or final_text, ended_by="user_left")
                if follow:
                    last_user = follow
                    n_user += 1
                    steps.append({"user": follow})
                    messages.append({"role": "user", "content": follow})
                    if remaining <= 0:
                        remaining = 1
                    continue
                if turn_stats is not None:
                    # Wanted a follow-up, writer produced none. Many of
                    # these means the dataset is going single-turn.
                    with turn_stats["lock"]:
                        turn_stats["followup_misses"] = turn_stats.get("followup_misses", 0) + 1
                return _done(steps, spoken or final_text)
            # No follow-up. Under the cap and asked, that was the person's
            # patience; at the cap it was the budget, and the row says so
            # by carrying no ended_by.
            left = (
                (room or need_first)
                and not force_followup
                and n_user < _user_turn_cap(budget)
                and _agent_asked(spoken)
                and _user_walks_away(message, turn_i, questions=prior_questions, patience=hazards)
            )
            return _done(steps, spoken or final_text, ended_by="user_left" if left else "")
        return _done(steps, final_text)

    agent.__name__ = f"local_model[{model}]"
    # How every reply was sampled, as the engine stamps it on the row.
    agent.sampling = {  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        "temperature": float(temperature),
        "max_tokens": reply_budget(max_tokens),
        "model": model,
    }
    agent.fault_plans = plans  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    agent.system = policy_text  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    agent.policy = policy_text  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return agent


def hosted_model(
    tools: list[dict], system: str = "", fault_plans: dict | None = None, **kwargs
) -> Callable:
    """The default simulation brain: hosted Qwen wearing these tools."""
    url, model = parse_backend_spec(default_agent_spec())
    return local_model(url, model, tools=tools, system=system, fault_plans=fault_plans, **kwargs)


__all__ = [
    "ACCOUNT_AGENT",
    "ACCOUNT_JUDGE",
    "COMPLETE_MAX_TOKENS",
    "COMPLETE_TEMPERATURE",
    "COMPLETE_TIMEOUT_S",
    "CONTEXT_FLOOR_TOKENS",
    "CONTEXT_MARGIN_TOKENS",
    "CONTEXT_TOKENS",
    "DEFAULT_AGENT",
    "DEFAULT_JUDGE",
    "DEFAULT_SIMULATOR",
    "DETAIL_HINTS_MAX",
    "ECHO_OVERLAP",
    "HOSTED_DROPPED",
    "HUMAN_TOOL_MAX_TOKENS",
    "HUMAN_TOOL_TEMPERATURE",
    "LARGE_CONTEXT_TOKENS",
    "LENGTH_CUT_MIN_CHARS",
    "LOCAL_MODEL_TEMPERATURE",
    "LOCAL_MODEL_TIMEOUT",
    "LOWERCASE_UPPER_SHARE",
    "MIN_REPLY_TOKENS",
    "MIN_SPOKEN_CHARS",
    "MISSING_HOSTED_KEY",
    "OPENER_MAX_TOKENS",
    "PATIENCE_HAZARDS",
    "PATIENCE_LEVELS",
    "PREAMBLE_MAX_WORDS",
    "QUOTA_FIX",
    "QUOTA_MARK",
    "REPEAT_OVERLAP",
    "REPLY_TOKENS",
    "REPLY_TOKENS_LARGE",
    "SHRINK_USER_MIN_CHARS",
    "THANKS_SHARE",
    "TOOL_WORLD_MAX",
    "TURN_CAP_LARGE",
    "TURN_CAP_MIN",
    "TURN_CAP_RESERVED_TOKENS",
    "TURN_CAP_SMALL",
    "TURN_CAP_TOKENS_PER_TURN",
    "TURN_CAP_TOOLS_MAX_TOKENS",
    "TURN_CAP_TOOL_TOKENS",
    "USER_LEFT",
    "USER_RETRY_MAX_TOKENS",
    "USER_TRACE_CHARS",
    "USER_TURN_MARK",
    "USER_TURN_MAX_CHARS",
    "USER_TURN_MAX_TOKENS",
    "USER_TURN_TEMPERATURE",
    "USER_TURN_WAIT_ASKED_S",
    "USER_TURN_WAIT_CAP_S",
    "USER_TURN_WAIT_S",
    "Patience",
    "complete",
    "current_rollout",
    "default_agent_spec",
    "default_judge_spec",
    "default_max_turns",
    "default_simulator_spec",
    "ended_on_question",
    "hosted_model",
    "human_tool_names",
    "local_model",
    "missing_hosted_key",
    "parse_backend_spec",
    "parse_text_tool_calls",
    "patience_hazards",
    "patience_may_leave",
    "ping_hosted",
    "public_llm_error",
    "reply_budget",
    "resolve_completion_key",
    "split_user_turns",
    "touch_hosted",
    "user_sim_system",
]
