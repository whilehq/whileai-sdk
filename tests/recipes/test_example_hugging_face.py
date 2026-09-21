"""The Hugging Face example, with the platform stubbed.

Every Hub call the SDK makes is a request to the While API (the platform
holds the Hub token), so the whole script can run against a local HTTP stub
that records what it was asked and answers what the platform would. That
checks the three call sites, the request bodies, the printed numbers, and
the two fail-fast paths (no key, no connected account).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "recipes" / "05-export" / "hugging-face" / "roundtrip.py"
KEY = "zp_test_key"

IMPORTED = {"datasetId": "ds_import1", "status": "ready", "rows": 1066}
PROFILE = {"rows": 1066, "tasks": 1, "graded": 0, "pass_rate": None, "support": None}
PUSHED = {
    "status": "done",
    "repo": "alice/my-set",
    "url": "https://huggingface.co/datasets/alice/my-set",
    "commit": "abcdef0123456789",
    "tag": "zp-ds_0123",
    "split": "train",
    "previous": {"datasetId": "ds_0001", "rows": 10, "pass_rate": 0.5},
}
PUSHED_RUN = {
    "status": "done",
    "repo": "alice/my-lora",
    "url": "https://huggingface.co/alice/my-lora",
    "commit": "0123456789abcdef",
    "tag": "zp-run_0123",
}


@dataclass
class Stub:
    url: str = ""
    connected: bool = True
    requests: list[tuple[str, str, dict | None]] = field(default_factory=list)
    keys: list[str | None] = field(default_factory=list)

    def route(self, method: str, path: str) -> tuple[int, dict]:
        if (method, path) == ("POST", "/datasets/import-hf"):
            return 200, IMPORTED
        if (method, path) == ("GET", "/datasets/ds_import1/profile"):
            return 200, {"profile": PROFILE}
        if (method, path) == ("DELETE", "/datasets/ds_import1"):
            return 200, {"deleted": True}
        if (method, path) == ("GET", "/hf/me"):
            if not self.connected:
                return 200, {"connected": False}
            return 200, {
                "connected": True,
                "username": "alice",
                "namespaces": ["alice"],
                "scopes": ["write-repos"],
            }
        if (method, path) == ("POST", "/datasets/ds_0123/hf-publish"):
            return 200, {"hf": {"status": "pushing"}}
        if (method, path) == ("GET", "/datasets/ds_0123"):
            return 200, {"datasetId": "ds_0123", "hf": PUSHED}
        if (method, path) == ("POST", "/runs/run_0123/hf-publish"):
            return 200, {"hf": {"status": "pushing"}}
        if (method, path) == ("GET", "/runs/run_0123"):
            return 200, {"runId": "run_0123", "hf": PUSHED_RUN}
        return 404, {"error": f"no route for {method} {path}"}


@pytest.fixture
def stub():
    state = Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _serve(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            path = self.path.split("?")[0]
            state.requests.append((self.command, path, body))
            state.keys.append(self.headers.get("X-Api-Key"))
            if self.headers.get("X-Api-Key") != KEY:
                status, payload = 401, {"error": "bad key"}
            else:
                status, payload = state.route(self.command, path)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_DELETE = _serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _env(tmp_path: Path, api_url: str | None, key: str | None) -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "OPENAI_API_KEY",
        "WHILEAI_API_KEY",
        "WHILEAI_DELEGATED_CREDENTIAL",
        "WHILEAI_API_URL",
        "VLLM_API_KEY",
        "HF_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    env["WHILEAI_HOME"] = str(tmp_path / "home")  # never a real stored login
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    if api_url:
        env["WHILEAI_API_URL"] = api_url
    if key:
        env["WHILEAI_API_KEY"] = key
    return env


def _run(tmp_path: Path, stub: Stub | None, *args: str, key: str | None = KEY):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(tmp_path),
        env=_env(tmp_path, stub.url if stub else None, key),
        timeout=120,
    )


def _calls(stub: Stub) -> list[tuple[str, str]]:
    return [(method, path) for method, path, _ in stub.requests]


def test_import_half_imports_profiles_and_deletes(stub, tmp_path):
    out = _run(tmp_path, stub)
    assert out.returncode == 0, out.stderr[-3000:]
    assert _calls(stub) == [
        ("POST", "/datasets/import-hf"),
        ("GET", "/datasets/ds_import1/profile"),
        ("DELETE", "/datasets/ds_import1"),
    ]
    assert stub.requests[0][2] == {
        "repo": "cornell-movie-review-data/rotten_tomatoes",
        "split": "test",
        "purpose": "eval",
        "max_rows": 100000,
        "name": "rotten_tomatoes:test (example)",
    }
    assert set(stub.keys) == {KEY}
    lines = out.stdout.splitlines()
    assert lines[0] == (
        "imported cornell-movie-review-data/rotten_tomatoes:test -> ds_import1 (1066 rows)"
    )
    assert lines[1] == "  rows 1066 · prompts 1 · graded 0 · pass – · support –"
    assert lines[2].startswith("  (no reward field")
    assert lines[3] == "  deleted"


def test_keep_and_a_different_split_leave_the_import_on_the_account(stub, tmp_path):
    out = _run(tmp_path, stub, "--repo-in", "tatsu-lab/alpaca", "--split", "train", "--keep")
    assert out.returncode == 0, out.stderr[-3000:]
    assert ("DELETE", "/datasets/ds_import1") not in _calls(stub)
    body = stub.requests[0][2]
    assert (body["repo"], body["split"], body["name"]) == (
        "tatsu-lab/alpaca",
        "train",
        "alpaca:train (example)",
    )
    assert "  kept as ds_import1" in out.stdout.splitlines()


def test_push_prints_the_tag_and_the_load_dataset_call(stub, tmp_path):
    out = _run(tmp_path, stub, "--push", "ds_0123", "--repo", "my-set", "--private")
    assert out.returncode == 0, out.stderr[-3000:]
    assert _calls(stub)[3:] == [
        ("GET", "/hf/me"),
        ("POST", "/datasets/ds_0123/hf-publish"),
        ("GET", "/datasets/ds_0123"),
    ]
    assert stub.requests[4][2] == {"private": True, "repo": "my-set"}
    text = out.stdout
    assert "pushing ds_0123 as alice ..." in text
    assert "  https://huggingface.co/datasets/alice/my-set" in text
    assert "  split train · commit abcdef0 · tag zp-ds_0123" in text
    assert "  replaced ds_0001 (rows 10, pass 50%)" in text
    assert 'load_dataset("alice/my-set", split="train", revision="zp-ds_0123")' in text


def test_push_run_pushes_the_adapter_as_a_model_repo(stub, tmp_path):
    out = _run(tmp_path, stub, "--push-run", "run_0123", "--repo", "my-lora")
    assert out.returncode == 0, out.stderr[-3000:]
    assert _calls(stub)[3:] == [
        ("GET", "/hf/me"),
        ("POST", "/runs/run_0123/hf-publish"),
        ("GET", "/runs/run_0123"),
    ]
    assert stub.requests[4][2] == {"private": True, "repo": "my-lora"}, "a checkpoint stays private"
    text = out.stdout
    assert "pushing adapter of run_0123 as alice ..." in text
    assert "  https://huggingface.co/alice/my-lora" in text
    assert "  commit 0123456 · tag zp-run_0123" in text
    assert 'PeftModel.from_pretrained(base, "alice/my-lora", revision="zp-run_0123")' in text


def test_push_without_a_connected_account_stops_before_pushing(stub, tmp_path):
    stub.connected = False
    out = _run(tmp_path, stub, "--push", "ds_0123")
    assert out.returncode != 0
    assert "Connect a Hugging Face account first" in out.stderr
    assert _calls(stub)[-1] == ("GET", "/hf/me")
    assert not any(path.endswith("/hf-publish") for _, path in _calls(stub))


def test_without_a_key_it_names_the_env_var_and_makes_no_request(stub, tmp_path):
    out = _run(tmp_path, stub, key=None)
    assert out.returncode != 0
    assert "WHILEAI_API_KEY" in out.stderr
    assert "wai login" in out.stderr
    assert stub.requests == []
