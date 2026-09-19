"""OTLP/HTTP JSON out, named measurements out. Stdlib only, no OTel SDK.

Two endpoints on the While gate, and this file is the whole client:

    POST $WHILEAI_API_URL/v1/traces    the run: spans, prompts, tool calls, tokens
    POST $WHILEAI_API_URL/v1/scores    opinions about a run already sent

The split matters. A judge answers after the turn it is judging has closed:
scoring costs a model call, nobody wants it in the request path, and a human
disagreeing with the judge answers a day later. So a trace is not finished when
its spans stop arriving; it keeps accepting opinions about itself, keyed by
name, and re-sending a name is a correction rather than a duplicate.

The envelope is built by hand rather than through the OTel SDK because seeing
it is the point. Every field below is one an ordinary exporter would send; the
`zeroproof.` attributes are the only additions, and each is optional.

Three rules the store cares about, learned the hard way:

  * JSON, not protobuf. `OTEL_EXPORTER_OTLP_PROTOCOL=http/json`; protobuf is a
    415 by design.
  * A span must never travel twice. The store hashes a batch to dedupe an
    exporter retry, so a resent identical batch is an overwrite, but the same
    span inside a *different* batch is a second part and the per-trace sums
    count it twice. This file sends one batch per trace, once.
  * `startedMs` is the producer's clock and is what the platform's time filter
    reads, so a backdated span lands in the day it says it did.
"""

from __future__ import annotations

import json
import os
import secrets
import urllib.error
import urllib.request

#: The public gate. This used to have no default, on the grounds that the API
#: base was infrastructure that could move; it is now a hostname we own, which
#: is the whole point of naming it, so the example asks for one less thing.
#: Override with ``WHILEAI_API_URL`` or ``--gate`` for a staging gate.
DEFAULT_API_URL = "https://api.withwhile.com"
API_URL_ENV = "WHILEAI_API_URL"
API_KEY_ENV = "WHILEAI_API_KEY"

#: One attribute, clipped. Keeps a batch under the store's 8 MB limit.
MAX_ATTR_BYTES = 8000
SEND_TIMEOUT_S = 30


class GateError(RuntimeError):
    pass


def trace_id() -> str:
    return secrets.token_hex(16)


def span_id() -> str:
    return secrets.token_hex(8)


def _nano(epoch_ms: float) -> str:
    return str(int(epoch_ms) * 1_000_000)


def _clip(text: str) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= MAX_ATTR_BYTES:
        return text
    return raw[:MAX_ATTR_BYTES].decode("utf-8", "ignore") + "\n[truncated]"


def _attr(key: str, value: object) -> dict:
    """One attribute in the OTLP AnyValue shape."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, (int, float)):
        return {"key": key, "value": {"doubleValue": float(value)}}
    if isinstance(value, str):
        return {"key": key, "value": {"stringValue": _clip(value)}}
    return {"key": key, "value": {"stringValue": _clip(json.dumps(value, default=str))}}


class Trace:
    """The spans of one agent run, and the export request that carries them."""

    def __init__(
        self,
        *,
        agent: str,
        prompt: str,
        started_ms: float,
        session_id: str = "",
        scenario_id: str = "",
        attributes: dict[str, object] | None = None,
    ) -> None:
        self.trace_id = trace_id()
        self.root_span_id = span_id()
        self.agent = agent
        self.prompt = prompt
        self.started_ms = started_ms
        self.session_id = session_id
        self.scenario_id = scenario_id
        self.attributes = dict(attributes or {})
        self.spans: list[dict] = []

    def llm(
        self,
        *,
        model: str,
        input_messages: list[dict],
        output_text: str,
        tool_calls: list[dict],
        started_ms: float,
        duration_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        """One model call, in the GenAI semantic conventions."""
        if tool_calls:
            output = [
                {
                    "role": "assistant",
                    "content": output_text,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in tool_calls
                    ],
                }
            ]
        else:
            output = [{"role": "assistant", "content": output_text}]

        attrs: dict[str, object] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": "openai-compatible",
            "gen_ai.input.messages": json.dumps(input_messages, default=str),
            "gen_ai.output.messages": json.dumps(output, default=str),
        }
        if input_tokens is not None:
            attrs["gen_ai.usage.input_tokens"] = input_tokens
        if output_tokens is not None:
            attrs["gen_ai.usage.output_tokens"] = output_tokens

        self._add(f"chat {model}", started_ms, duration_ms, attrs, error)

    def tool(
        self,
        *,
        name: str,
        arguments: str,
        result: str,
        failed: bool,
        started_ms: float,
        duration_ms: float,
    ) -> None:
        """One tool call. `gen_ai.tool.status` is how a failure travels."""
        self._add(
            f"execute_tool {name}",
            started_ms,
            duration_ms,
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.arguments": arguments,
                "gen_ai.tool.call.result": result,
                "gen_ai.tool.status": "error" if failed else "success",
            },
            "tool failed" if failed else None,
        )

    def _add(
        self,
        name: str,
        started_ms: float,
        duration_ms: float,
        attrs: dict[str, object],
        error: str | None,
    ) -> None:
        end_ms = started_ms + duration_ms
        self.spans.append(
            {
                "traceId": self.trace_id,
                "spanId": span_id(),
                "parentSpanId": self.root_span_id,
                "name": name,
                "kind": 1,
                "startTimeUnixNano": _nano(started_ms),
                "endTimeUnixNano": _nano(end_ms),
                "attributes": [_attr(k, v) for k, v in attrs.items()],
                "events": (
                    [
                        {
                            "name": "exception",
                            "timeUnixNano": _nano(end_ms),
                            "attributes": [_attr("exception.message", error)],
                        }
                    ]
                    if error
                    else []
                ),
                "status": {"code": 2 if error else 1},
            }
        )

    def body(
        self,
        resource: dict[str, str],
        *,
        final_text: str,
        ended_ms: float,
        outcome: str = "complete",
        measurements: list[tuple[str, float, str]] | None = None,
        evidence: dict[str, str] | None = None,
        reward: float | None = None,
    ) -> dict:
        """The export request for the whole run: root span plus every child.

        Measurements ride on the root rather than going out as a second request.
        They are known the moment the turn ends and need no trace to have landed
        first, which is the only reason the judge needs a separate path at all.

        The answer goes in because half of what is worth measuring is the gap
        between what the turn claimed and what it did.
        """
        attrs: list[dict] = [
            _attr("gen_ai.operation.name", "invoke_agent"),
            _attr("gen_ai.agent.name", self.agent),
            # Raw text: the store wraps a non-JSON string itself, so wrapping
            # here would double-wrap it.
            _attr("gen_ai.input.messages", self.prompt),
            _attr(
                "gen_ai.output.messages", json.dumps([{"role": "assistant", "content": final_text}])
            ),
        ]
        if self.session_id:
            attrs.append(_attr("gen_ai.conversation.id", self.session_id))
        if self.scenario_id:
            # Ties every attempt at the same task together. Without it each run
            # is its own group of one, and a group of one has no variance to
            # learn from.
            attrs.append(_attr("zeroproof.scenario_id", self.scenario_id))
        if reward is not None:
            attrs.append(_attr("whileai.reward", reward))
        for key, value in self.attributes.items():
            attrs.append(_attr(key, value))

        for name, value, description in measurements or []:
            attrs.append(_attr(f"zeroproof.scores.{name}", value))
            if description:
                attrs.append(_attr(f"zeroproof.describe.{name}", description))

        # Why each flag fired. Evidence is not a measurement and is capped
        # separately, so the receipts cost no measurement names.
        for name, text in (evidence or {}).items():
            attrs.append(_attr(f"zeroproof.evidence.{name}", text))

        root = {
            "traceId": self.trace_id,
            "spanId": self.root_span_id,
            "parentSpanId": "",
            "name": f"invoke_agent {self.agent}",
            "kind": 1,
            "startTimeUnixNano": _nano(self.started_ms),
            "endTimeUnixNano": _nano(ended_ms),
            "attributes": attrs,
            "events": [],
            # A turn that died is an error on the run itself, not only on
            # whichever child span happened to throw.
            "status": {"code": 2 if outcome == "error" else 1},
        }

        return {
            "resourceSpans": [
                {
                    "resource": {"attributes": [_attr(k, v) for k, v in resource.items()]},
                    "scopeSpans": [
                        {
                            "scope": {"name": "whileai-agent-behavior"},
                            "spans": [root, *self.spans],
                        }
                    ],
                }
            ]
        }


class Client:
    """The two POSTs, with the API key on every one of them."""

    def __init__(self, api_key: str | None = None, gate: str | None = None) -> None:
        base = gate or os.environ.get(API_URL_ENV, "") or DEFAULT_API_URL
        self.gate = base.rstrip("/")
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        if not self.api_key:
            raise GateError(
                f"No API key. Set {API_KEY_ENV} or pass --api-key. "
                "Get one at https://app.withwhile.com/platform"
            )

    def send_trace(self, body: dict) -> dict:
        return self._post("/v1/traces", body)

    def send_scores(self, trace_id_: str, scores: list[dict]) -> dict:
        return self._post("/v1/scores", {"traceId": trace_id_, "scores": scores})

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.gate + path,
            data=json.dumps(body, default=str).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-Api-Key": self.api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=SEND_TIMEOUT_S) as response:
                payload = response.read()
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")[:400]
            raise GateError(f"POST {path} -> {err.code}: {detail}") from None
        except urllib.error.URLError as err:
            raise GateError(f"POST {path} failed: {err.reason}") from None
        return json.loads(payload) if payload else {}


__all__ = ["API_KEY_ENV", "API_URL_ENV", "Client", "GateError", "Trace", "span_id", "trace_id"]
