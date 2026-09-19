"""
OTLP trace ingest for the While token gate.

Push agent traces so they land as a dataset on your account. There is one
call and one response: POST the OTLP/HTTP JSON batch with your zp_ key, get
back a datasetId. No presigned URL, no finalize step, nothing to poll.

Batches append. Everything an exporter sends under one dataset name on one
UTC day belongs to one dataset, so a five-second flush interval does not turn
into thousands of datasets.

Two ways in, both dependency-light:

1. Live exporter. Point any OpenTelemetry SDK at the gate. This is the least
   invasive option: no While code in your app at all, just environment.

       import os
       from whileai.ingest import otel_env
       os.environ.update(otel_env("zp_...", dataset="prod-refunds"))

2. Local file. Replay an OTLP batch you already have on disk (JSON, gzip or
   raw).

       from whileai.ingest import ingest_traces
       print(ingest_traces("zp_...", "traces.json")["datasetId"])

3. Rows you already scored. Most harnesses are not OTel-instrumented: they
   call a model k times a prompt, score each answer, and hold a list. Send
   that list; the envelope is written for you.

       from whileai.ingest import send_runs
       send_runs(rows, agent="refund-triage")

Auth is the X-Api-Key header, because exporters cannot carry a Clerk JWT.
"""

import gzip
import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping
from typing import Any

import requests

from whileai._env import getenv

# Where the gate lives (api.withwhile.com; the older api.zeroproofai.com
# still answers). Override with WHILEAI_TRACE_URL to point at a different
# deployment.
_DEFAULT_TRACE_URL = "https://api.withwhile.com"
_GZIP_MAGIC = b"\x1f\x8b"

# The resource attribute that names the dataset. The gate reads
# `zeroproof.dataset` and nothing else, so sending only the whileai spelling
# lands every batch in a dataset called `traces` whatever you asked for, with
# a 202 that says so too late to notice. Both are written: the second costs
# one attribute and means the rename needs no release.
_DATASET_KEYS = ("zeroproof.dataset", "whileai.dataset")


class WhileIngestError(Exception):
    """Raised when the gate rejects a trace batch."""


def _base(base_url: str | None = None) -> str:
    url = base_url or getenv("TRACE_URL") or _DEFAULT_TRACE_URL
    return url.rstrip("/")


def _traces_endpoint(base_url: str | None = None) -> str:
    return _base(base_url) + "/v1/traces"


def otel_env(api_key: str, dataset: str = "traces", base_url: str | None = None) -> dict[str, str]:
    """
    Environment for an OpenTelemetry OTLP/HTTP exporter.

    The exporter sends the batch body itself and forwards
    ``OTEL_EXPORTER_OTLP_HEADERS`` as request headers, so the key reaches the
    gate. ``http/json`` is required: the gate parses the OTLP JSON wire format
    and answers protobuf batches with a 415.
    """
    return {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": _traces_endpoint(base_url),
        "OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=" + api_key,
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
        # Resource attribute the gate reads to name the dataset.
        "OTEL_RESOURCE_ATTRIBUTES": ",".join(k + "=" + dataset for k in _DATASET_KEYS),
    }


def _check_nanos(body: bytes) -> None:
    """Reject span times that are not nanoseconds.

    The store divides by 1e6 without a unit guard, so a batch sent in
    milliseconds or seconds lands near 1970 and disappears from every bounded
    time window: the upload succeeds and the traces are simply never seen.
    Better to fail here, where the sender can still fix it.
    """
    if body[:2] == _GZIP_MAGIC:
        return
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return
    for resource in payload.get("resourceSpans") or []:
        for scope in resource.get("scopeSpans") or []:
            for span in scope.get("spans") or []:
                raw = span.get("startTimeUnixNano")
                if raw in (None, "", 0, "0"):
                    continue
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    continue
                # 1e15 ns is 1970-01-12; any real timestamp is far above it
                if 0 < value < 10**15:
                    raise WhileIngestError(
                        f"span {span.get('name') or 'unnamed'!r} has "
                        f"startTimeUnixNano={value}, which is not nanoseconds. "
                        "Multiply by 1e6 for milliseconds or 1e9 for seconds; "
                        "as sent, these traces would be stored near 1970 and "
                        "hidden from every time window."
                    )


def send_traces(
    api_key: str,
    body: bytes,
    base_url: str | None = None,
    timeout: int = 60,
) -> dict:
    """POST one OTLP/HTTP JSON batch (raw or gzipped) and return the 202 body."""
    _check_nanos(body)
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
    if body[:2] == _GZIP_MAGIC:
        headers["Content-Encoding"] = "gzip"
    res = requests.post(_traces_endpoint(base_url), data=body, headers=headers, timeout=timeout)
    if res.status_code >= 300:
        raise WhileIngestError(f"ingest failed: HTTP {res.status_code} {res.text[:400]}")
    return res.json()


def ingest_traces(
    api_key: str,
    file: str,
    dataset: str | None = None,
    base_url: str | None = None,
) -> dict:
    """
    Push a local OTLP batch file end to end and return ``{datasetId, dataset,
    rows}``.

    ``dataset`` overrides the dataset name by setting the dataset resource
    attribute on every resourceSpan, which requires reading the batch; leave
    it unset to send the bytes untouched.
    """
    with open(file, "rb") as fh:
        body = fh.read()
    if not body:
        raise WhileIngestError("empty file: " + file)

    if dataset is not None:
        raw = gzip.decompress(body) if body[:2] == _GZIP_MAGIC else body
        batch = json.loads(raw.decode("utf-8"))
        for resource_span in batch.get("resourceSpans", []):
            resource = resource_span.setdefault("resource", {})
            attributes = [
                a for a in resource.get("attributes", []) if a.get("key") not in _DATASET_KEYS
            ]
            attributes += [{"key": k, "value": {"stringValue": dataset}} for k in _DATASET_KEYS]
            resource["attributes"] = attributes
        body = json.dumps(batch).encode("utf-8")

    return send_traces(api_key, body, base_url=base_url)


# The row keys a scored run arrives under. `pull()` hands back
# scenario_id/prompt/final_text/reward, so what comes off the platform goes
# straight back in; the rest are what harnesses in the wild call the same
# field. `info` is read as a fallback because that is where `pull()` keeps
# the model and the duration.
_PROMPT_KEYS = ("prompt", "input", "question")
_COMPLETION_KEYS = ("final_text", "completion", "output", "response", "answer")
_REWARD_KEYS = ("reward", "score")
_SCENARIO_KEYS = ("scenario_id", "scenario", "case", "task_id")
_MODEL_KEYS = ("model", "model_version")


def _attr(key: str, value: object) -> dict:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, (int, float)):
        return {"key": key, "value": {"doubleValue": float(value)}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _field(row: Mapping, info: Mapping, keys: tuple[str, ...]) -> Any:
    for source in (row, info):
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return None


def send_runs(
    rows: Iterable[Mapping],
    *,
    agent: str,
    api_key: str | None = None,
    dataset: str | None = None,
    pass_at: float = 1.0,
    model: str | None = None,
    base_url: str | None = None,
    timeout: int = 60,
) -> dict:
    """
    Send scored runs — the rows your eval loop already has — as traces.

    The inverse of ``rows_from_otel``: you hand over
    ``{scenario_id, prompt, final_text, reward}`` and the OTLP envelope is
    written for you, so an agent that emits no OpenTelemetry still lands on
    the traces page.

        whileai.send_runs(rows, agent="refund-triage")
        wai.cuts("refund-triage")   # what is worth training on

    Runs of one prompt must share a ``scenario_id`` to be grouped, and a
    grouped prompt the agent passes some of the time is what training data
    is made of. Rows without one are grouped by their prompt text. ``reward``
    is scored against ``pass_at`` (1.0 by default: a reward of 1 is a pass);
    rows without a reward land ungraded, which is honest and cuts nothing.
    One span per run — tool steps are not sent.
    """
    from .auth import resolve_api_key

    key = resolve_api_key(api_key)
    if not key:
        raise WhileIngestError(
            "no API key: pass api_key=, set WHILEAI_API_KEY, or run `whileai login`"
        )
    name = str(agent or "").strip()
    if not name:
        raise WhileIngestError("agent= is required: it is how the traces page finds these runs")
    runs = list(rows or [])
    if not runs:
        raise WhileIngestError("no rows to send")

    now = time.time_ns()
    spans = []
    for i, row in enumerate(runs):
        if not isinstance(row, Mapping):
            raise WhileIngestError(
                f"row {i} is a {type(row).__name__}, not a dict of "
                "{scenario_id, prompt, final_text, reward}"
            )
        raw = row.get("info")
        info: Mapping = raw if isinstance(raw, Mapping) else {}
        prompt = _field(row, info, _PROMPT_KEYS)
        if prompt is None:
            raise WhileIngestError(
                f"row {i} has no prompt (keys: {', '.join(map(str, row)) or 'none'})"
            )
        scenario = _field(row, info, _SCENARIO_KEYS)
        if scenario is None:
            scenario = "p-" + hashlib.sha1(str(prompt).encode("utf-8")).hexdigest()[:12]
        attributes = [
            _attr("gen_ai.agent.name", name),
            _attr("test.case.name", scenario),
            _attr("gen_ai.prompt", prompt),
        ]
        completion = _field(row, info, _COMPLETION_KEYS)
        if completion is not None:
            attributes.append(_attr("gen_ai.completion", completion))
        used = _field(row, info, _MODEL_KEYS) or model
        if used:
            attributes.append(_attr("gen_ai.request.model", used))
        reward = _field(row, info, _REWARD_KEYS)
        if reward is not None:
            try:
                score = float(reward)
            except (TypeError, ValueError) as exc:
                raise WhileIngestError(
                    f"row {i} has a reward that is not a number: {reward!r}"
                ) from exc
            # Without the bar a score is stored but never judged, so no prompt
            # is ever worth training on and every cut comes back empty.
            attributes.append(_attr("zeroproof.score", score))
            attributes.append(_attr("zeroproof.score.pass_at", float(pass_at)))
        # Milliseconds apart, oldest first, so the runs table keeps the order
        # the loop produced them in.
        start = now - (len(runs) - i) * 1_000_000
        held = _field(row, info, ("duration_ms",)) or 0
        spans.append(
            {
                "traceId": os.urandom(16).hex(),
                "spanId": os.urandom(8).hex(),
                "name": "agent.run",
                "kind": 1,
                "startTimeUnixNano": str(start),
                "endTimeUnixNano": str(start + int(float(held) * 1_000_000)),
                "attributes": attributes,
            }
        )

    tag = str(dataset or name)
    batch = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attr("service.name", name),
                        *[_attr(k, tag) for k in _DATASET_KEYS],
                    ]
                },
                "scopeSpans": [{"scope": {"name": "whileai.send_runs"}, "spans": spans}],
            }
        ]
    }
    body = json.dumps(batch).encode("utf-8")
    if len(body) > 1_000_000:
        body = gzip.compress(body)
    return send_traces(key, body, base_url=base_url, timeout=timeout)


def list_traces(api_key: str | None = None, base_url: str | None = None, timeout: int = 30) -> dict:
    """
    What this key's account has ingested: one entry per dataset name per day,
    with row counts and sizes, plus the account totals.

        for t in list_traces()["traces"]:
            print(t["name"], t["rows"], t["sizeBytes"])

    ``api_key`` resolves like every other platform call: the argument, then
    ``WHILEAI_API_KEY``, then the key ``whileai login`` saved.
    """
    from .auth import resolve_api_key

    key = resolve_api_key(api_key)
    if not key:
        raise WhileIngestError(
            "no API key: pass api_key=, set WHILEAI_API_KEY, or run `whileai login`"
        )
    res = requests.get(
        _base(base_url) + "/traces",
        headers={"X-Api-Key": key},
        timeout=timeout,
    )
    if res.status_code >= 300:
        raise WhileIngestError(f"list failed: HTTP {res.status_code} {res.text[:400]}")
    return res.json()


# The name this exception had before the package was renamed.
ZeroProofIngestError = WhileIngestError
