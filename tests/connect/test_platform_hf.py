"""Hugging Face, both directions, against a recorded ``_call``.

hf_publish / hf_publish_run post the body and poll the row until ``hf``
settles; import_hf posts the split and polls the dataset until it is
ready or failed. Nothing here touches the network.
"""

from __future__ import annotations

import pytest

from whileai.simulations.ingest import platform
from whileai.simulations.ingest.platform import PlatformError


class Recorder:
    def __init__(self, replies):
        self.calls = []
        self.replies = list(replies)

    def __call__(self, method, path, api_key, body=None, *, raw_url=None, public=False, **kw):
        self.calls.append((method, path, body))
        return self.replies.pop(0) if self.replies else {}


def test_hf_status_reads_me(monkeypatch):
    rec = Recorder(
        [
            {
                "connected": True,
                "username": "jacob",
                "namespaces": ["jacob", "while-ai"],
                "scopes": [],
            }
        ]
    )
    monkeypatch.setattr(platform, "_call", rec)
    out = platform.hf_status(api_key="zp_x")
    assert out["namespaces"] == ["jacob", "while-ai"]
    assert rec.calls == [("GET", "/hf/me", None)]


def test_hf_publish_posts_then_waits_for_done(monkeypatch):
    rec = Recorder(
        [
            {"datasetId": "ds_1", "hf": {"status": "pushing", "repo": "jacob/x"}},
            {"datasetId": "ds_1", "hf": {"status": "pushing", "repo": "jacob/x"}},
            {
                "datasetId": "ds_1",
                "hf": {"status": "done", "repo": "jacob/x", "commit": "abc", "tag": "zp-ds_1"},
            },
        ]
    )
    monkeypatch.setattr(platform, "_call", rec)
    monkeypatch.setattr(platform.time, "sleep", lambda s: None)
    out = platform.hf_publish("ds_1", namespace="jacob", repo="x", private=True, api_key="zp_x")
    assert out["commit"] == "abc"
    assert rec.calls[0] == (
        "POST",
        "/datasets/ds_1/hf-publish",
        {"private": True, "namespace": "jacob", "repo": "x"},
    )
    assert rec.calls[1][1] == "/datasets/ds_1"


def test_hf_publish_without_wait_returns_the_stamp(monkeypatch):
    rec = Recorder([{"datasetId": "ds_1", "hf": {"status": "pushing", "repo": "jacob/x"}}])
    monkeypatch.setattr(platform, "_call", rec)
    assert platform.hf_publish("ds_1", wait=False, api_key="zp_x")["status"] == "pushing"
    assert len(rec.calls) == 1


def test_hf_publish_error_raises(monkeypatch):
    rec = Recorder(
        [
            {"hf": {"status": "pushing"}},
            {"hf": {"status": "error", "error": "LFS upload failed"}},
        ]
    )
    monkeypatch.setattr(platform, "_call", rec)
    monkeypatch.setattr(platform.time, "sleep", lambda s: None)
    with pytest.raises(PlatformError, match="LFS upload failed"):
        platform.hf_publish("ds_1", api_key="zp_x")


def test_hf_publish_run_defaults_private_and_polls_the_run(monkeypatch):
    rec = Recorder(
        [
            {"runId": "run_1", "hf": {"status": "pushing"}},
            {
                "runId": "run_1",
                "hf": {
                    "status": "done",
                    "repo": "jacob/m",
                    "url": "https://huggingface.co/jacob/m",
                },
            },
        ]
    )
    monkeypatch.setattr(platform, "_call", rec)
    monkeypatch.setattr(platform.time, "sleep", lambda s: None)
    out = platform.hf_publish_run("run_1", api_key="zp_x")
    assert out["repo"] == "jacob/m"
    assert rec.calls[0] == ("POST", "/runs/run_1/hf-publish", {"private": True})
    assert rec.calls[1] == ("GET", "/runs/run_1", None)


def test_import_hf_posts_and_waits_until_ready(monkeypatch):
    rec = Recorder(
        [
            {"datasetId": "ds_7", "status": "importing"},
            {"datasetId": "ds_7", "status": "importing"},
            {"datasetId": "ds_7", "status": "ready", "rows": 1066},
        ]
    )
    monkeypatch.setattr(platform, "_call", rec)
    monkeypatch.setattr(platform.time, "sleep", lambda s: None)
    row = platform.import_hf(
        "rotten_tomatoes", split="test", purpose="eval", name="rt test", api_key="zp_x"
    )
    assert row["status"] == "ready"
    method, path, body = rec.calls[0]
    assert (method, path) == ("POST", "/datasets/import-hf")
    assert body == {
        "repo": "rotten_tomatoes",
        "split": "test",
        "purpose": "eval",
        "max_rows": 100_000,
        "name": "rt test",
    }
    assert rec.calls[1] == ("GET", "/datasets/ds_7", None)


def test_import_hf_failed_raises_with_the_host_error(monkeypatch):
    rec = Recorder(
        [
            {"datasetId": "ds_7", "status": "importing"},
            {
                "datasetId": "ds_7",
                "status": "failed",
                "hfImport": {"error": "the split has no rows"},
            },
        ]
    )
    monkeypatch.setattr(platform, "_call", rec)
    monkeypatch.setattr(platform.time, "sleep", lambda s: None)
    with pytest.raises(PlatformError, match="the split has no rows"):
        platform.import_hf("a/b", api_key="zp_x")


def test_import_hf_without_wait_returns_the_importing_row(monkeypatch):
    rec = Recorder([{"datasetId": "ds_7", "status": "importing"}])
    monkeypatch.setattr(platform, "_call", rec)
    assert platform.import_hf("a/b", wait=False, api_key="zp_x")["status"] == "importing"
    assert len(rec.calls) == 1
