"""``wai.harness.compare`` and ``wai compare``: the before-and-after prompt check.

Every test runs offline on a scripted model with a fixed seed: no network,
no Ollama, no key. The three the feature promises: a rewrite that truly
helps reads PASS, identical arms read NO DIFFERENCE, and the same seed
prints the same report byte for byte.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import whileai as wai
from whileai import before_after
from whileai.cli import main

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "recipes" / "02-measure" / "before-and-after"


def _check(after: str = before_after.DEMO_AFTER, **kw):
    return wai.harness.compare(
        before_after.DEMO_BEFORE,
        after,
        tasks=before_after.demo_tasks(),
        reward=wai.verify.Numeric(),
        model=before_after.demo_model,
        **kw,
    )


def test_a_rewrite_that_helps_reads_pass():
    report = _check()
    metric = report["compare"]["metrics"]["pass_at_1"]
    assert report["verdict"] == "PASS"
    assert report["ok"] is True
    assert metric["delta"] > 0
    assert metric["ci95"][0] > 0  # the 95% range excludes zero
    assert metric["n_paired"] == before_after.DEMO_TASKS
    assert report["after"]["pass_at_1"] > report["before"]["pass_at_1"]
    text = str(report)
    assert "seed 0" in text
    assert "PASS" in text


def test_identical_arms_read_no_difference():
    report = _check(after=before_after.DEMO_BEFORE)
    metric = report["compare"]["metrics"]["pass_at_1"]
    assert report["verdict"] == "NO DIFFERENCE"
    assert metric["delta"] == 0
    assert report["before"]["hash"] == report["after"]["hash"]


def test_same_seed_same_output_and_another_seed_another_draw():
    first, second = _check(seed=7), _check(seed=7)
    assert str(first) == str(second)
    assert json.dumps(first, default=str) == json.dumps(second, default=str)
    # rows carry a fresh lineage run id each call; the replies and rewards repeat
    replies = [[(r["final_text"], r["reward"]) for r in x.after_rows] for x in (first, second)]
    assert replies[0] == replies[1]
    assert str(_check(seed=8)) != str(first)


def test_both_arms_draw_on_the_same_seeds_through_a_spec_string(monkeypatch):
    seen: list[tuple[str, str, int]] = []

    def fake_complete(base_url, model, messages, **kw):
        seen.append((messages[0]["content"], messages[-1]["content"], kw["extra"]["seed"]))
        assert kw["temperature"] == before_after.TEMPERATURE
        return {"content": "The answer is 4"}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    report = wai.harness.compare(
        "old", "new", ["What is 2 + 2?"], lambda p, c: 1.0, model="ollama:qwen3:4b-instruct", k=3
    )
    old = [(q, s) for system, q, s in seen if system == "old"]
    new = [(q, s) for system, q, s in seen if system == "new"]
    assert len(old) == 3 and old == new  # common random numbers: draw j shares its seed
    assert report["model"] == "ollama:qwen3:4b-instruct"


def test_two_harnesses_compare_two_configurations():
    small = wai.Harness(before_after.demo_model, instructions=before_after.DEMO_BEFORE, label="v1")
    report = wai.harness.compare(
        small,
        wai.Harness(instructions=before_after.DEMO_AFTER, label="v2"),
        tasks=before_after.demo_tasks(8),
        reward=wai.verify.Numeric(),
        model=before_after.demo_model,
    )
    assert (report["before"]["label"], report["after"]["label"]) == ("v1", "v2")
    assert report.after_rows[0]["harness"]["label"] == "v2"


def test_no_model_names_the_fix():
    with pytest.raises(ValueError, match=r"model=wai.Ollama"):
        wai.harness.compare("a", "b", ["q"], lambda p, c: 1.0)


def test_cli_demo_prints_the_python_report(capsys):
    assert main(["compare", "--demo"]) == 0
    assert capsys.readouterr().out.strip() == str(_check()).strip()


def test_cli_files_and_json(capsys):
    code = main(
        [
            "compare",
            "--model",
            "unused:model",
            "--before",
            str(RECIPE / "before.txt"),
            "--after",
            str(RECIPE / "after.txt"),
            "--tasks",
            str(RECIPE / "tasks.jsonl"),
            "--json",
        ]
    )
    # "unused:" is no provider, so the run fails with the reason, exit 1
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_cli_without_model_or_demo_exits_1(capsys):
    assert main(["compare", "--tasks", "x.jsonl"]) == 1
    assert "--demo" in capsys.readouterr().err


def test_recipe_runs_offline_both_ways(capsys):
    spec = importlib.util.spec_from_file_location("before_after_recipe", RECIPE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("before_after_recipe", module)
    spec.loader.exec_module(module)
    assert module.main([]) == 0
    helped = capsys.readouterr().out
    assert module.main(["--same"]) == 0
    same = capsys.readouterr().out
    assert "\nPASS\n" in helped
    assert "\nNO DIFFERENCE\n" in same
