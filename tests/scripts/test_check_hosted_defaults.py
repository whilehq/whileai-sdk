"""The hosted-defaults gate fails on a host whose app is not deployed.

A gate that has never fired has not been shown to work, so this pins the
failure, not just the pass. The case is the real one: `stressd-vllm` was
stopped on 2026-09-21 and three defaults went on naming it into release
0.120, where a VLLM_API_KEY user met a 404 on their first `simulate()`.

Everything here is offline. The check itself never calls Modal (that is
what the committed manifest is for) and neither does this test, so no
scale-to-zero container is woken to run CI.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "check_hosted_defaults.py"
MANIFEST = ROOT / "scripts" / "hosted_defaults_manifest.json"


def _mod():
    spec = importlib.util.spec_from_file_location("check_hosted_defaults", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


APPS = ["whileai-serve", "whileai-judge", "zeroproof-serve", "t2s-serve-qwen-qwen3-4b"]


@pytest.mark.parametrize(
    "host,app,fn",
    [
        ("zeroproofai--whileai-serve-qwen3-4b.modal.run", "whileai-serve", "qwen3-4b"),
        ("zeroproofai--whileai-serve-qwen3-8b.modal.run", "whileai-serve", "qwen3-8b"),
        ("zeroproofai--whileai-judge-serve.modal.run", "whileai-judge", "serve"),
        ("zeroproofai--zeroproof-serve-qwen3-4b.modal.run", "zeroproof-serve", "qwen3-4b"),
        # the app that was stopped: no deployed app prefixes it
        ("zeroproofai--stressd-vllm-serve.modal.run", None, "stressd-vllm-serve"),
    ],
)
def test_split_finds_the_longest_deployed_app_that_prefixes_the_host(host, app, fn):
    assert _mod()._split(host, APPS) == (app, fn)


def test_a_hyphenated_app_name_does_not_swallow_a_shorter_one():
    # "whileai-serve" must not claim a host belonging to "whileai-serve-extra"
    apps = ["whileai-serve", "whileai-serve-extra"]
    assert _mod()._split("zeroproofai--whileai-serve-extra-v1.modal.run", apps) == (
        "whileai-serve-extra",
        "v1",
    )


def test_a_non_modal_host_is_out_of_scope():
    # a user's own endpoint is not ours to verify, and must not be reported
    assert _mod()._split("my-vllm.internal.example.com", APPS) == (None, "")


def test_host_is_read_out_of_a_vllm_spec():
    m = _mod()
    spec = "vllm:Qwen/Qwen3-4B@https://zeroproofai--whileai-serve-qwen3-4b.modal.run/v1"
    assert m._host(spec) == "zeroproofai--whileai-serve-qwen3-4b.modal.run"


def test_the_committed_manifest_lists_apps_and_passes_the_check():
    m = _mod()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["apps"], "manifest carries no deployed apps; run --refresh"
    # the apps come from Modal, never from the defaults being checked
    assert "whileai-serve" in manifest["apps"]
    assert m.check() == 0


def test_the_gate_fails_when_a_default_names_a_stopped_app(monkeypatch):
    m = _mod()
    monkeypatch.setattr(
        m,
        "_defaults",
        lambda: {
            "DEFAULT_AGENT": (
                "vllm:Qwen/Qwen3-4B-Instruct-2507@"
                "https://zeroproofai--stressd-vllm-serve.modal.run/v1"
            )
        },
    )
    assert m.check() == 1
