"""``wai.hub.push`` and ``export(push_to=)``: the Hub with your own token (#506).

``huggingface_hub`` is stubbed, so these check the contract rather than the
network: the token comes from ``token=`` else ``HF_TOKEN``; repos are
private unless said otherwise; a file goes to a dataset repo and an
adapter directory to a model repo; and nothing on this path talks to the
While platform (``urllib.request.urlopen``, the only transport the platform
client uses, is armed to fail).
"""

from __future__ import annotations

import json
import sys
import types
import urllib.request

import pytest

import whileai as wai
from whileai.hub import INSTALL_HINT, push, repo_type_of
from whileai.simulations.export import export_training

POLICY = "Read before you write."


class _Commit:
    oid = "abc1234def"


class _FakeApi:
    instances: list[_FakeApi] = []

    def __init__(self, token=None):
        self.token = token
        self.calls: list[tuple[str, dict]] = []
        _FakeApi.instances.append(self)

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))
        prefix = "datasets/" if kwargs.get("repo_type") == "dataset" else ""
        return f"https://huggingface.co/{prefix}{repo_id}"

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))
        return _Commit()

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return _Commit()


@pytest.fixture
def hub(monkeypatch):
    """A fake ``huggingface_hub`` on the import path, and the platform
    transport armed: any request would fail the test."""
    _FakeApi.instances.clear()
    module = types.ModuleType("huggingface_hub")
    module.HfApi = _FakeApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def no_platform(*args, **kwargs):
        raise AssertionError(f"a request left for the platform: {args[0]!r}")

    monkeypatch.setattr(urllib.request, "urlopen", no_platform)
    return _FakeApi


def _last(hub) -> _FakeApi:
    assert hub.instances, "HfApi was never constructed"
    return hub.instances[-1]


def test_token_comes_from_hf_token_when_not_passed(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    f = tmp_path / "train.jsonl"
    f.write_text('{"messages": []}\n')
    report = push(f, "me/my-set")
    api = _last(hub)
    assert api.token == "hf_from_env"
    assert api.calls[0] == (
        "create_repo",
        {"repo_id": "me/my-set", "repo_type": "dataset", "private": True, "exist_ok": True},
    )
    name, kwargs = api.calls[1]
    assert name == "upload_file"
    assert kwargs["path_in_repo"] == "train.jsonl" and kwargs["repo_type"] == "dataset"
    assert kwargs["path_or_fileobj"] == str(f)
    assert report == {
        "repo_id": "me/my-set",
        "repo_type": "dataset",
        "url": "https://huggingface.co/datasets/me/my-set",
        "commit": "abc1234def",
        "files": ["train.jsonl"],
        "private": True,
    }


def test_token_argument_wins_over_the_environment(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    f = tmp_path / "train.jsonl"
    f.write_text("{}\n")
    push(f, "me/my-set", token="hf_passed")
    assert _last(hub).token == "hf_passed"


def test_no_token_anywhere_defers_to_the_cached_login(hub, tmp_path):
    """``HfApi(token=None)`` reads ``hf auth login``'s cache; the SDK does
    not second-guess it or invent an error of its own."""
    f = tmp_path / "train.jsonl"
    f.write_text("{}\n")
    push(f, "me/my-set")
    assert _last(hub).token is None


def test_private_is_the_default_and_can_be_turned_off(hub, tmp_path):
    f = tmp_path / "train.jsonl"
    f.write_text("{}\n")
    assert push(f, "me/a")["private"] is True
    assert push(f, "me/b", private=False)["private"] is False
    assert _last(hub).calls[0][1]["private"] is False


def test_an_adapter_directory_goes_to_a_model_repo(hub, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"\0")
    assert repo_type_of(adapter) == "model"
    report = push(adapter, "me/my-lora", token="hf_x")
    api = _last(hub)
    assert api.calls[0][1]["repo_type"] == "model"
    name, kwargs = api.calls[1]
    assert name == "upload_folder" and kwargs["folder_path"] == str(adapter)
    assert report["url"] == "https://huggingface.co/me/my-lora"
    assert report["files"] == ["adapter_config.json", "adapter_model.safetensors"]


def test_rows_are_written_as_train_jsonl_and_pushed(hub, tmp_path):
    rows = [{"messages": [{"role": "user", "content": "hi"}]}]
    seen: dict = {}

    def upload_file(self, **kwargs):
        seen["body"] = open(kwargs["path_or_fileobj"], encoding="utf-8").read()
        self.calls.append(("upload_file", kwargs))
        return _Commit()

    hub.upload_file = upload_file
    report = push(rows, "me/my-set", token="hf_x")
    assert report["files"] == ["train.jsonl"] and report["repo_type"] == "dataset"
    assert json.loads(seen["body"]) == rows[0]


def test_export_push_to_pushes_the_written_file(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    row = {
        "prompt": "hi",
        "reward": 1,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    out = tmp_path / "train.jsonl"
    report = export_training(
        [row], str(out), system_prompt=POLICY, format="trl", push_to="me/my-set"
    )
    api = _last(hub)
    assert api.token == "hf_from_env"
    assert api.calls[1][1]["path_or_fileobj"] == report["path"] == str(out)
    assert report["hub"]["repo_id"] == "me/my-set" and report["hub"]["private"] is True
    assert report["hub"]["url"] == "https://huggingface.co/datasets/me/my-set"


def test_export_push_to_needs_a_file_to_push(hub):
    with pytest.raises(ValueError, match="push_to needs an output path"):
        export_training([], None, push_to="me/my-set")
    assert not hub.instances


def test_hub_is_reachable_from_the_front_door(hub, tmp_path):
    f = tmp_path / "train.jsonl"
    f.write_text("{}\n")
    assert wai.hub.push is push
    assert "hub" in dir(wai) and "hub" not in wai.__all__, "one dot down, not on the front door"


def test_missing_huggingface_hub_names_the_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    f = tmp_path / "train.jsonl"
    f.write_text("{}\n")
    with pytest.raises(ImportError, match=r"whileai\[hf\]") as err:
        push(f, "me/my-set", token="hf_x")
    assert str(err.value) == INSTALL_HINT


def test_repo_id_and_missing_source_are_refused_before_any_call(hub, tmp_path):
    with pytest.raises(ValueError, match="namespace/name"):
        push(tmp_path, "my-set")
    with pytest.raises(FileNotFoundError):
        push(tmp_path / "nope.jsonl", "me/my-set")
    assert all(not api.calls for api in hub.instances)
