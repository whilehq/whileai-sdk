"""whileai.ingest: the exporter env, the nanosecond guard, gzip, the dataset override."""

from __future__ import annotations

import gzip
import json

import pytest

from whileai import ingest


class Reply:
    def __init__(self, status=202, body=None, text=""):
        self.status_code = status
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def _batch(start="1700000000000000000"):
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "app"}}]
                },
                "scopeSpans": [{"spans": [{"name": "llm.call", "startTimeUnixNano": start}]}],
            }
        ]
    }


def _capture_post(monkeypatch, reply):
    seen = {}

    def post(url, data=None, headers=None, timeout=None):
        seen.update(url=url, data=data, headers=headers, timeout=timeout)
        return reply

    monkeypatch.setattr(ingest.requests, "post", post)
    return seen


def test_otel_env_points_the_exporter_at_the_gate_with_the_key_as_a_header(monkeypatch):
    monkeypatch.delenv("WHILEAI_TRACE_URL", raising=False)
    env = ingest.otel_env("zp_abc", dataset="prod-refunds")
    assert env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == "https://api.withwhile.com/v1/traces"
    assert env["OTEL_EXPORTER_OTLP_HEADERS"] == "x-api-key=zp_abc"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/json"
    # The gate names the dataset from zeroproof.dataset only: without it the
    # batch lands in a dataset called `traces` and the 202 says so.
    assert "zeroproof.dataset=prod-refunds" in env["OTEL_RESOURCE_ATTRIBUTES"].split(",")


def test_otel_env_honors_the_base_url_override_and_the_env_var(monkeypatch):
    env = ingest.otel_env("zp_abc", base_url="http://localhost:8080/")
    assert env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == "http://localhost:8080/v1/traces"
    monkeypatch.setenv("WHILEAI_TRACE_URL", "http://gate.test")
    assert ingest.otel_env("zp_abc")["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == (
        "http://gate.test/v1/traces"
    )


def test_send_traces_posts_the_key_and_returns_the_gate_body(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_1", "rows": 1}))
    body = json.dumps(_batch()).encode()
    out = ingest.send_traces("zp_abc", body, base_url="http://gate.test")
    assert out == {"datasetId": "ds_1", "rows": 1}
    assert seen["url"] == "http://gate.test/v1/traces"
    assert seen["headers"]["X-Api-Key"] == "zp_abc"
    assert seen["headers"]["Content-Type"] == "application/json"
    assert "Content-Encoding" not in seen["headers"]
    assert seen["data"] is body


def test_gzipped_batches_go_up_as_is_with_the_encoding_header(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_1"}))
    body = gzip.compress(json.dumps(_batch()).encode())
    ingest.send_traces("zp_abc", body)
    assert seen["headers"]["Content-Encoding"] == "gzip"
    assert seen["data"] is body


def test_millisecond_timestamps_are_refused_before_anything_is_uploaded(monkeypatch):
    def post(*a, **k):
        raise AssertionError("a batch with bad units must never reach the gate")

    monkeypatch.setattr(ingest.requests, "post", post)
    body = json.dumps(_batch(start="1700000000000")).encode()
    with pytest.raises(ingest.WhileIngestError, match="not nanoseconds"):
        ingest.send_traces("zp_abc", body)


def test_gate_rejection_becomes_an_ingest_error_with_the_status(monkeypatch):
    monkeypatch.setattr(
        ingest.requests, "post", lambda *a, **k: Reply(415, text="protobuf not accepted")
    )
    with pytest.raises(ingest.WhileIngestError, match="HTTP 415 protobuf not accepted"):
        ingest.send_traces("zp_abc", json.dumps(_batch()).encode())


def test_ingest_traces_renames_the_dataset_on_every_resource(tmp_path, monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_9"}))
    batch = _batch()
    batch["resourceSpans"][0]["resource"]["attributes"].append(
        {"key": "whileai.dataset", "value": {"stringValue": "old-name"}}
    )
    path = tmp_path / "traces.json.gz"
    path.write_bytes(gzip.compress(json.dumps(batch).encode()))
    out = ingest.ingest_traces("zp_abc", str(path), dataset="new-name")
    assert out["datasetId"] == "ds_9"
    sent = json.loads(seen["data"])
    attrs = sent["resourceSpans"][0]["resource"]["attributes"]
    for key in ("zeroproof.dataset", "whileai.dataset"):
        names = [a["value"]["stringValue"] for a in attrs if a["key"] == key]
        assert names == ["new-name"], f"{key}: the old name is replaced, not doubled"
    assert any(a["key"] == "service.name" for a in attrs), "other attributes survive"


def test_ingest_traces_sends_bytes_untouched_without_a_dataset(tmp_path, monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_9"}))
    raw = json.dumps(_batch()).encode()
    path = tmp_path / "traces.json"
    path.write_bytes(raw)
    ingest.ingest_traces("zp_abc", str(path))
    assert seen["data"] == raw


def test_ingest_traces_refuses_an_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.requests, "post", lambda *a, **k: Reply(202, {}))
    path = tmp_path / "empty.json"
    path.write_bytes(b"")
    with pytest.raises(ingest.WhileIngestError, match="empty file"):
        ingest.ingest_traces("zp_abc", str(path))


def test_list_traces_reads_with_the_key_and_surfaces_a_rejection(monkeypatch):
    seen = {}

    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return Reply(200, {"traces": [{"name": "prod", "rows": 3}]})

    monkeypatch.setattr(ingest.requests, "get", get)
    out = ingest.list_traces("zp_abc", base_url="http://gate.test")
    assert out["traces"][0]["rows"] == 3
    assert seen["url"] == "http://gate.test/traces"
    assert seen["headers"] == {"X-Api-Key": "zp_abc"}
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: Reply(401, text="bad key"))
    with pytest.raises(ingest.WhileIngestError, match="HTTP 401"):
        ingest.list_traces("zp_abc")


# ------------------------------------------------------------------ send_runs


def _spans(seen):
    sent = json.loads(seen["data"])
    return sent["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attrs(span):
    out = {}
    for a in span["attributes"]:
        out[a["key"]] = next(iter(a["value"].values()))
    return out


def test_send_runs_turns_scored_rows_into_a_batch_the_gate_can_cut(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_1", "rows": 2}))
    rows = [
        {"scenario_id": "case-1", "prompt": "refund 8812", "final_text": "done", "reward": 1.0},
        {"scenario_id": "case-1", "prompt": "refund 8812", "final_text": "no", "reward": 0.0},
    ]
    out = ingest.send_runs(rows, agent="refund-triage", api_key="zp_abc")
    assert out["datasetId"] == "ds_1"
    first = _attrs(_spans(seen)[0])
    # The four attributes a cut needs: the agent, the group, the score, the bar.
    assert first["gen_ai.agent.name"] == "refund-triage"
    assert first["test.case.name"] == "case-1"
    assert first["zeroproof.score"] == 1.0
    assert first["zeroproof.score.pass_at"] == 1.0
    assert first["gen_ai.prompt"] == "refund 8812"
    assert first["gen_ai.completion"] == "done"
    # Nanoseconds, in the order the loop produced them.
    starts = [int(s["startTimeUnixNano"]) for s in _spans(seen)]
    assert starts[0] < starts[1] and starts[0] > 10**18


def test_send_runs_groups_repeats_of_one_prompt_without_a_scenario_id(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_2"}))
    rows = [
        {"prompt": "same task", "final_text": "a", "reward": 1},
        {"prompt": "same task", "final_text": "b", "reward": 0},
        {"prompt": "other task", "final_text": "c", "reward": 1},
    ]
    ingest.send_runs(rows, agent="a", api_key="zp_abc")
    cases = [_attrs(s)["test.case.name"] for s in _spans(seen)]
    assert cases[0] == cases[1] != cases[2], "same prompt is one group, not three groups of one"


def test_send_runs_takes_what_pull_hands_back(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_3"}))
    ingest.send_runs(
        [
            {
                "scenario_id": "c1",
                "prompt": "p",
                "final_text": "f",
                "reward": 1,
                "info": {"model": "qwen3-4b", "duration_ms": 500},
            }
        ],
        agent="a",
        api_key="zp_abc",
    )
    span = _spans(seen)[0]
    assert _attrs(span)["gen_ai.request.model"] == "qwen3-4b"
    assert int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"]) == 500_000_000


def test_send_runs_leaves_an_unscored_row_ungraded(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_4"}))
    ingest.send_runs([{"prompt": "p", "final_text": "f"}], agent="a", api_key="zp_abc")
    attrs = _attrs(_spans(seen)[0])
    assert "zeroproof.score" not in attrs and "zeroproof.score.pass_at" not in attrs


def test_send_runs_names_the_dataset_under_both_keys(monkeypatch):
    seen = _capture_post(monkeypatch, Reply(202, {"datasetId": "ds_5"}))
    ingest.send_runs([{"prompt": "p", "reward": 1}], agent="a", dataset="week-1", api_key="zp_abc")
    attrs = json.loads(seen["data"])["resourceSpans"][0]["resource"]["attributes"]
    named = {a["key"]: a["value"]["stringValue"] for a in attrs}
    assert named["zeroproof.dataset"] == named["whileai.dataset"] == "week-1"


def test_send_runs_says_which_row_is_wrong(monkeypatch):
    monkeypatch.setattr(ingest.requests, "post", lambda *a, **k: Reply(202, {}))
    with pytest.raises(ingest.WhileIngestError, match="row 1 has no prompt"):
        ingest.send_runs([{"prompt": "p"}, {"final_text": "f"}], agent="a", api_key="zp_abc")
    with pytest.raises(ingest.WhileIngestError, match="not a number"):
        ingest.send_runs([{"prompt": "p", "reward": "good"}], agent="a", api_key="zp_abc")
    with pytest.raises(ingest.WhileIngestError, match="no rows"):
        ingest.send_runs([], agent="a", api_key="zp_abc")
    with pytest.raises(ingest.WhileIngestError, match="agent="):
        ingest.send_runs([{"prompt": "p"}], agent="", api_key="zp_abc")
