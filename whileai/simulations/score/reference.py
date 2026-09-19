"""Score rollouts under a reference model, so ``mean_kl`` has its other side.

``simulate(logprobs=True)`` records what the sampling policy thought of its
own tokens (``row["logprob"]``, ``row["n_tokens"]``). The KL penalty an RL
update pays (Lambert 2025, chapters Direct Alignment and Regularization) and
the importance ratio an off-policy update forms (Noukhovitch et al. 2024,
arXiv:2410.18252) both need the same tokens scored under a second model: the
reference the policy is being kept close to, or the policy version about to be
trained. ``reference_logprobs`` does that scoring against any
OpenAI-compatible endpoint that returns prompt logprobs (vLLM does), and
stamps ``ref_logprob`` / ``ref_n_tokens`` / ``ref_model`` on each row.
``mean_kl(rows)`` then reads them.

How a turn is scored. The conversation up to the turn is sent once with
the generation prompt appended, only to learn how many tokens the prefix
is; then the conversation including the turn is sent with
``prompt_logprobs``, and the log-probabilities of the tokens after the
prefix are summed. That is exactly the span the policy generated (its
reply plus the end-of-turn token), rendered by the reference's own chat
template, so no tokenizer is needed on this side. When the reference
shares the policy's tokenizer the token counts match ``n_tokens``; the
report says how far apart they are, which is the check that the two
sides are comparable at all.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Mapping, Sequence
from http import HTTPStatus
from typing import Any

from whileai._env import getenv

from ..defaults import MESSAGE_EXAMPLES
from ..generate.agents import parse_backend_spec, resolve_completion_key

REF_KEYS = ("ref_logprob", "ref_n_tokens", "ref_model")


def _post(url: str, key: str, body: dict, timeout: float) -> dict:
    import requests

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    res = requests.post(url, headers=headers, json=body, timeout=timeout)
    if res.status_code >= HTTPStatus.BAD_REQUEST:
        raise RuntimeError(f"{url} -> {res.status_code}: {res.text[:300]}")
    return res.json()


def _chat_url(base_url: str) -> str:
    raw = base_url.rstrip("/")
    if "://" not in raw:
        raw = "https://" + raw
    if raw.endswith("/chat/completions"):
        return raw
    if raw.endswith("/v1"):
        return raw + "/chat/completions"
    return raw + "/v1/chat/completions"


def _candidate(entry: Any, token_id: Any) -> Mapping[str, Any] | None:
    """The scored token's record from one ``prompt_logprobs`` position.

    vLLM returns, per position, a mapping from token id to ``{logprob,
    rank, decoded_token}`` holding the actual token plus the top
    candidate. With ``return_token_ids`` the actual token is looked up by
    id; without ids it is the candidate with the worse rank (the top
    candidate is rank 1, and when the actual token is the top one there
    is only one record). The first position has no entry.
    """
    if not isinstance(entry, Mapping) or not entry:
        return None
    if token_id is not None:
        hit = entry.get(str(token_id))
        if hit is None:
            hit = entry.get(token_id)
        return hit if isinstance(hit, Mapping) else None
    records = [v for v in entry.values() if isinstance(v, Mapping)]
    if not records:
        return None
    return max(records, key=lambda r: int(r.get("rank") or 0))


def _actual_logprob(entry: Any, token_id: Any) -> float | None:
    hit = _candidate(entry, token_id)
    if hit is not None and isinstance(hit.get("logprob"), (int, float)):
        return float(hit["logprob"])
    return None


def _decoded(entry: Any, token_id: Any) -> str:
    hit = _candidate(entry, token_id)
    return str(hit.get("decoded_token") or "") if hit is not None else ""


def score_turns(
    messages: Sequence[dict],
    *,
    post: Callable[[dict], dict],
    model: str,
    tools: Sequence[dict] | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[float, int, int]:
    """(summed logprob, token count, turns scored) over every assistant turn.

    ``post`` sends one chat-completions body and returns the parsed reply.
    """
    base: dict[str, Any] = {"model": model, "max_tokens": 1, "temperature": 0}
    if tools:
        # The template renders the tool block the policy saw; "none" keeps a
        # server without a tool-call parser from refusing the request, since
        # nothing is generated here anyway.
        base["tools"] = list(tools)
        base["tool_choice"] = "none"
    if chat_template_kwargs:
        base["chat_template_kwargs"] = dict(chat_template_kwargs)
    total = 0.0
    count = 0
    turns = 0
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        prefix = post({**base, "messages": list(messages[:i]), "add_generation_prompt": True})
        prefix_len = int(((prefix.get("usage") or {}).get("prompt_tokens")) or 0)
        full = post(
            {
                **base,
                "messages": list(messages[: i + 1]),
                "add_generation_prompt": False,
                "prompt_logprobs": 1,
                "return_token_ids": True,
            }
        )
        entries = full.get("prompt_logprobs")
        if not isinstance(entries, list) or prefix_len <= 0:
            raise RuntimeError(
                "the endpoint returned no prompt_logprobs; the reference must be an "
                "OpenAI-compatible server that supports prompt_logprobs (vLLM does)"
            )
        raw_ids = full.get("prompt_token_ids")
        ids: list[Any] = (
            list(raw_ids) if isinstance(raw_ids, list) and len(raw_ids) == len(entries) else []
        )
        if not ids:
            ids = [None] * len(entries)
        tail = list(range(prefix_len, len(entries)))
        # The template closes the turn and adds a newline after the end
        # token; the policy stopped at the end token, so the newline is not
        # a generated token.
        while tail and _decoded(entries[tail[-1]], ids[tail[-1]]).strip() == "":
            tail.pop()
        for pos in tail:
            value = _actual_logprob(entries[pos], ids[pos])
            if value is not None:
                total += value
                count += 1
        turns += 1
    return total, count, turns


def reference_logprobs(
    source,
    ref: str,
    *,
    system_prompt: str | None = None,
    tools: Sequence[dict] | None = None,
    api_key: str | None = None,
    concurrency: int = 4,
    timeout: float = 600.0,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    transport: Callable[[dict], dict] | None = None,
) -> dict[str, Any]:
    """Stamp ``ref_logprob`` on every row: the reference's summed logprob
    over the tokens the policy generated. Rows are modified in place;
    the report says what was scored.

    ``ref`` is a backend spec, ``vllm:<model>@<base_url>``; on the
    platform's serving endpoint ``<model>`` is the base by its own name
    (``Qwen/Qwen3-4B``: the reference of an SFT/GRPO/DPO run), a hosted
    model's name, or ``run:<runId>`` for a finished run's adapter, with
    ``WHILEAI_API_KEY`` as the key. ``source`` is a ``SimulationData``
    (system prompt and tools come from its profile), a row list, or a
    JSONL path; pass ``system_prompt=``/``tools=`` for the last two so
    the reference sees the prompt the policy saw. ``chat_template_kwargs``
    must match what the policy sampled with (``{"enable_thinking":
    False}`` for Qwen3).

    Report: ``n_rows``, ``n_skipped`` (no assistant turn or a failed
    call, with ``errors``), ``n_tokens``, ``model``, and
    ``token_count_gap`` (mean |ref_n_tokens - n_tokens| over rows that
    carry ``n_tokens``): near zero when the reference shares the policy's
    tokenizer, which is when ``mean_kl`` is a KL and not a length
    artifact.
    """
    from ..export import _convert_messages, _resolve

    rows, system, resolved_tools, _ = _resolve(source)
    if system_prompt is not None:
        system = str(system_prompt)
    if tools is not None:
        resolved_tools = list(tools)
    base_url, model = parse_backend_spec(ref)
    if transport is None:
        key = (
            str(api_key or "").strip()
            or (getenv("API_KEY", "").strip() if "zeroproof" in base_url else "")
            or resolve_completion_key(base_url, api_key)
        )
        url = _chat_url(base_url)

        def transport(body: dict) -> dict:
            return _post(url, key, body, timeout)

    from whileai.simulations import conversation

    def one(row: dict) -> tuple[float, int, int]:
        messages = row.get("messages") or conversation(row)
        converted = _convert_messages(messages, system=system, strip_think=True)
        return score_turns(
            converted,
            post=transport,
            model=model,
            tools=resolved_tools,
            chat_template_kwargs=chat_template_kwargs,
        )

    scored = skipped = tokens = 0
    errors: list[str] = []
    gaps: list[int] = []
    targets = [r for r in rows if isinstance(r, dict)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        futures = {pool.submit(one, row): row for row in targets}
        for future, row in futures.items():
            try:
                total, count, turns = future.result()
            except Exception as exc:  # one bad row must not lose the batch
                skipped += 1
                if len(errors) < MESSAGE_EXAMPLES:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                continue
            if turns == 0:
                skipped += 1
                continue
            row["ref_logprob"] = round(total, 6)
            row["ref_n_tokens"] = count
            row["ref_model"] = model
            scored += 1
            tokens += count
            n = row.get("n_tokens")
            if isinstance(n, int) and not isinstance(n, bool) and n > 0:
                gaps.append(abs(count - n))
    return {
        "n_rows": scored,
        "n_skipped": skipped,
        "n_tokens": tokens,
        "model": model,
        "token_count_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
        "errors": errors,
    }


__all__ = ["REF_KEYS", "reference_logprobs", "score_turns"]
