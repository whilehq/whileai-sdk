"""The hosted loop, offline: the CLI, the state file, and every step with
the platform calls mocked. The ``data`` step runs for real up to the push
(template writer, scripted agent, judge, split), since none of that needs
a key."""

from __future__ import annotations

import argparse
import json
import re
from types import SimpleNamespace

import pytest
import requests
from example_helpers import EXAMPLES, load_script

import whileai.simulations as wai

README = EXAMPLES / "04-train/hosted-loop" / "README.md"


@pytest.fixture
def hl(monkeypatch, tmp_path):
    mod = load_script("hosted_loop_run", EXAMPLES / "04-train/hosted-loop" / "run.py")
    monkeypatch.setattr(mod, "STATE", tmp_path / "hosted-loop.json")
    monkeypatch.setattr(mod, "resolve_api_key", lambda explicit=None: "zp_test_key")
    return mod


def _args(**over) -> argparse.Namespace:
    base = {
        "step": "all",
        "name": "hl",
        "method": "sft",
        "base": "Qwen/Qwen3-4B",
        "epochs": 1.0,
        "steps": 20,
        "budget": 24,
        "seed": 1,
        "timeout": 1800.0,
    }
    base.update(over)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------- the CLI


def test_help_lists_every_flag_the_readme_uses(hl, capsys):
    with pytest.raises(SystemExit) as exc:
        hl.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    documented = set()
    for line in README.read_text(encoding="utf-8").splitlines():
        if "run.py" in line:
            documented.update(re.findall(r"--[a-z][a-z-]*", line))
    documented.update(re.findall(r"^\| `(--[a-z-]+)` \|", README.read_text(encoding="utf-8"), re.M))
    assert documented >= {"--method", "--epochs"}
    for flag in sorted(documented):
        assert flag in out, f"README uses {flag}, --help does not list it"
    for step in ("data", "train", "serve", "call", "models"):
        assert step in out


def test_readme_defaults_are_the_parser_defaults(hl, capsys):
    seen = {}
    for name in hl.STEPS:
        seen[name] = None
    for name in hl.STEPS:

        def record(args, _name=name):
            seen[_name] = args

        hl.STEPS[name] = record
    assert hl.main([]) == 0
    args = seen["data"]
    table = dict(re.findall(r"^\| `--([a-z-]+)` \| ([^|]+) \|", README.read_text("utf-8"), re.M))
    assert table, "the README has a flags table"
    for flag, documented in table.items():
        actual = getattr(args, flag.replace("-", "_"))
        documented = documented.strip()
        try:
            assert float(documented) == float(actual), (flag, documented, actual)
        except (TypeError, ValueError):
            assert str(actual) == documented, (flag, documented, actual)


def test_all_runs_the_four_steps_in_order_and_skips_models(hl, monkeypatch):
    order: list[str] = []
    for name in hl.STEPS:
        monkeypatch.setitem(hl.STEPS, name, lambda args, _n=name: order.append(_n))
    assert hl.main([]) == 0
    assert order == ["data", "train", "serve", "call"]
    order.clear()
    assert hl.main(["models"]) == 0
    assert order == ["models"]
    order.clear()
    assert hl.main(["train", "--method", "grpo", "--steps", "10"]) == 0
    assert order == ["train"]


def test_unknown_step_is_rejected(hl):
    with pytest.raises(SystemExit) as exc:
        hl.main(["bogus"])
    assert exc.value.code == 2


def test_without_a_key_it_names_login_and_the_env_var(hl, monkeypatch):
    monkeypatch.setattr(hl, "resolve_api_key", lambda explicit=None: None)
    with pytest.raises(SystemExit) as exc:
        hl.main(["models"])
    message = str(exc.value.code)
    assert "whileai login" in message and "WHILEAI_API_KEY" in message and "http" in message


# ------------------------------------------------------------ state file


def test_state_accumulates_across_saves_and_need_names_the_missing_key(hl):
    assert hl.load() == {}
    hl.save(train_id="ds_1")
    hl.save(holdout_id="ds_2")
    assert hl.load() == {"train_id": "ds_1", "holdout_id": "ds_2"}
    assert json.loads(hl.STATE.read_text())["train_id"] == "ds_1"
    hl.need(hl.load(), "train_id", "holdout_id")
    with pytest.raises(SystemExit) as exc:
        hl.need(hl.load(), "train_id", "run_id", "endpoint")
    message = str(exc.value.code)
    assert "run_id, endpoint" in message and "hosted-loop.json" in message


# ------------------------------------------------------ agent and judge


def test_scripted_agent_has_contrast_and_the_judge_reads_it(hl):
    asked = hl.scripted_agent("I want a refund please")
    assert asked["steps"] == [] and "order id" in asked["final_text"]
    assert hl.judge(asked) == 1
    verdicts = set()
    for i in range(30):
        row = hl.scripted_agent(f"please refund order {10000 + i}")
        tools = [s["tool"] for s in row["steps"]]
        assert tools in (["lookup_order", "create_refund"], ["create_refund"])
        assert row["steps"][-1]["arguments"]["order_id"] == str(10000 + i)
        verdicts.add(hl.judge(row))
    assert verdicts == {0, 1}, "the graded set needs passes and fails to learn from"


# ----------------------------------------------------------------- steps


def test_data_step_simulates_grades_splits_and_pushes(hl, monkeypatch, capsys):
    pushed: list[tuple[str, list[dict], dict]] = []

    def fake_push(rows, name, **kw):
        pushed.append((name, rows, kw))
        return {"datasetId": f"ds_{name}", "gate": {"ok": True, "warnings": ["w1"]}}

    monkeypatch.setattr(wai, "push_rows", fake_push)
    hl.step_data(_args(budget=24))
    assert [p[0] for p in pushed] == ["hl-train", "hl-holdout"]
    (_, train, train_kw), (_, held, held_kw) = pushed
    assert train_kw["purpose"] == "train" and train_kw["gate"] is True
    assert train_kw["mode"] == "rl" and train_kw["agent"] == "hl"
    assert held_kw["purpose"] == "holdout" and held_kw["parent"] == "ds_hl-train"
    assert len(train) + len(held) == 24
    # split_pseudo_production moves whole tasks: every row of a prompt lands on one side.
    assert {r["prompt"] for r in train} & {r["prompt"] for r in held} == set()
    assert len(held) >= round(0.25 * 24), "the holdout overshoots the fraction, never undershoots"
    assert all("reward" in r and "messages" in r for r in train + held)
    assert hl.load() == {"train_id": "ds_hl-train", "holdout_id": "ds_hl-holdout"}
    out = capsys.readouterr().out
    assert "pass@1" in out and "gate: w1" in out


class _FakeRun:
    def __init__(self, status="done", error=None):
        self.run_id = "run_1"
        self.url = "https://withwhile.com/platform/training/run_1"
        self.method = "sft"
        self.adapter = "volume whileai-train-runs:/run_1/adapter"
        self.training = {"before": 5.0, "after": 4.1, "holdoutRows": 72}
        self.error = error
        self._status = status
        self.waited: dict = {}

    def wait(self, *, timeout=None, poll=15.0):
        self.waited = {"timeout": timeout, "poll": poll}
        return self._status


def test_train_step_passes_the_method_knobs_and_waits(hl, monkeypatch, capsys):
    hl.save(train_id="ds_t", holdout_id="ds_h")
    calls: list[tuple] = []
    run = _FakeRun()

    def fake_train(dataset, **kw):
        calls.append((dataset, kw))
        return run

    monkeypatch.setattr(wai, "train", fake_train)
    hl.step_train(_args(method="sft", epochs=2.0, steps=99, timeout=60.0))
    dataset, kw = calls[-1]
    assert dataset == "ds_t" and kw["holdout"] == "ds_h" and kw["base_model"] == "Qwen/Qwen3-4B"
    assert kw["method"] == "sft" and kw["epochs"] == 2.0 and kw["steps"] is None
    assert run.waited == {"timeout": 60.0, "poll": 20}
    assert hl.load()["run_id"] == "run_1"
    out = capsys.readouterr().out
    assert "5.0 -> 4.1 on 72 held-out rows" in out and run.adapter in out

    hl.step_train(_args(method="grpo", epochs=2.0, steps=10))
    _, kw = calls[-1]
    assert kw["method"] == "grpo" and kw["steps"] == 10 and kw["epochs"] is None


def test_train_step_needs_the_data_step_and_reports_a_failed_run(hl, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        hl.step_train(_args())
    assert "train_id" in str(exc.value.code)
    hl.save(train_id="ds_t", holdout_id="ds_h")
    monkeypatch.setattr(wai, "train", lambda *a, **k: _FakeRun("failed", error="OOM"))
    with pytest.raises(SystemExit) as exc:
        hl.step_train(_args())
    assert "failed" in str(exc.value.code) and "OOM" in str(exc.value.code)


def test_serve_step_saves_the_endpoint_and_model_name(hl, monkeypatch, capsys):
    hl.save(run_id="run_1")
    seen = {}

    def fake_serve(name, run=None, **kw):
        seen.update(name=name, run=run)
        return {
            "name": "hl",
            "version": 1,
            "baseModel": "Qwen/Qwen3-4B",
            "endpoint": "https://example.modal.run/v1",
        }

    monkeypatch.setattr(wai, "serve", fake_serve)
    hl.step_serve(_args())
    assert seen == {"name": "hl", "run": "run_1"}
    assert hl.load()["endpoint"] == "https://example.modal.run/v1"
    assert hl.load()["model_name"] == "hl"
    assert "endpoint https://example.modal.run/v1" in capsys.readouterr().out


def test_call_step_posts_a_chat_completion_with_the_resolved_key(hl, monkeypatch, capsys):
    hl.save(endpoint="https://example.modal.run/v1/", model_name="hl")
    seen = {}

    def fake_post(url, **kw):
        seen.update(url=url, **kw)
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": " Looking up 88213. "}}]},
        )

    monkeypatch.setattr(hl.requests, "post", fake_post)
    hl.step_call(_args())
    assert seen["url"] == "https://example.modal.run/v1/chat/completions"
    assert seen["headers"] == {"Authorization": "Bearer zp_test_key"}
    body = seen["json"]
    assert body["model"] == "hl" and body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["content"] == hl.POLICY
    assert seen["timeout"] >= 600, "the first call after idle pays a cold start"
    out = capsys.readouterr().out
    assert "HTTP 200" in out and "Looking up 88213." in out


def _fake_clock(monkeypatch, hl):
    """A ``time`` whose ``sleep`` advances ``time`` so the window runs out
    without waiting; returns the recorded sleeps."""
    slept: list[float] = []
    now = [0.0]

    def sleep(s):
        slept.append(s)
        now[0] += s

    monkeypatch.setattr(hl, "time", SimpleNamespace(time=lambda: now[0], sleep=sleep))
    return slept


def _response(status: int, content: str = "ok"):
    def raise_for_status():
        if status >= 400:
            raise requests.HTTPError(f"{status} Server Error")

    return SimpleNamespace(
        status_code=status,
        raise_for_status=raise_for_status,
        json=lambda: {"choices": [{"message": {"content": content}}]},
    )


def test_call_step_retries_a_502_from_the_cold_container(hl, monkeypatch, capsys):
    """Issue #445: a 502/503/504 while the container wakes is retried inside
    the same window the README documents, not raised as a traceback."""
    hl.save(endpoint="https://example.modal.run/v1", model_name="hl")
    slept = _fake_clock(monkeypatch, hl)
    replies = iter([_response(502), _response(503), _response(200, "Refunded order 88213.")])
    posts: list[float] = []

    def fake_post(url, **kw):
        posts.append(kw["timeout"])
        return next(replies)

    monkeypatch.setattr(hl.requests, "post", fake_post)
    hl.step_call(_args())
    assert len(posts) == 3, "two warm-up statuses, then the reply"
    assert slept == [hl.WARMUP_RETRY_S, min(2 * hl.WARMUP_RETRY_S, hl.WARMUP_RETRY_MAX_S)]
    assert posts[0] == hl.CALL_WINDOW_S and posts[-1] < posts[0], "retries wait out the same window"
    out = capsys.readouterr().out
    assert "HTTP 502" in out and "HTTP 503" in out and "still starting" in out
    assert "HTTP 200" in out and "Refunded order 88213." in out


def test_call_step_gives_up_at_the_deadline_with_what_to_do(hl, monkeypatch):
    hl.save(endpoint="https://example.modal.run/v1", model_name="hl")
    slept = _fake_clock(monkeypatch, hl)
    monkeypatch.setattr(hl.requests, "post", lambda url, **kw: _response(502))
    with pytest.raises(SystemExit) as exc:
        hl.step_call(_args())
    message = str(exc.value.code)
    assert "HTTP 502" in message and "still starting" in message
    assert "python run.py call" in message, "the error says what to do next"
    assert slept and sum(slept) <= hl.CALL_WINDOW_S, "every retry stays inside the window"
    assert max(slept) == hl.WARMUP_RETRY_MAX_S, "the gap backs off to the cap"


def test_call_step_raises_other_errors_at_once(hl, monkeypatch):
    hl.save(endpoint="https://example.modal.run/v1", model_name="hl")
    slept = _fake_clock(monkeypatch, hl)
    monkeypatch.setattr(hl.requests, "post", lambda url, **kw: _response(401))
    with pytest.raises(requests.HTTPError):
        hl.step_call(_args())
    assert slept == [], "a 401 is not a cold start"


def test_models_step_lists_what_the_account_hosts(hl, monkeypatch, capsys):
    monkeypatch.setattr(
        wai,
        "models",
        lambda: [{"name": "hl", "version": 2, "baseModel": "Qwen/Qwen3-4B", "adapterRunId": "r"}],
    )
    hl.step_models(_args())
    out = capsys.readouterr().out
    assert "hl" in out and "v2" in out and "Qwen/Qwen3-4B" in out and "run r" in out
