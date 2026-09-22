"""The Prime Intellect example, minus the model: an ``rl.jsonl`` built
offline from the shipped ``spec.json`` with a scripted agent, then the two
scripts that need no key (``diagnose.py``, ``export_prompts.py``) run on it
and on hand-made sets that exercise every verdict.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import whileai.simulations as wai
from whileai.simulations.generate.agents import current_rollout

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "recipes" / "03-select" / "prime-intellect-rl"
SPEC = EXAMPLE / "spec.json"
INFO_FIELDS = {
    "scenario_id",
    "world_state",
    "stance",
    "tier",
    "faults",
    "tool_known",
    "intent_known",
}


def _env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "WHILEAI_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLE / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd),
        env=_env(),
        timeout=300,
    )


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _stat(stdout: str, key: str) -> str:
    for line in stdout.splitlines():
        if line.startswith(key + " "):
            return line[len(key) :].strip()
    raise AssertionError(f"{key} missing from report:\n{stdout}")


def _coding_agent(message: str) -> dict:
    """Every second rollout of a prompt answers in two tokens with no tool
    call, the policy the README says a process reward pays for."""
    if getattr(current_rollout, "rollout_index", 0) % 2 == 1:
        return {"steps": [], "final_text": "No."}
    return {
        "steps": [
            {
                "tool": "read_file",
                "arguments": {"path": "backtest.py"},
                "result": {"content": "import pandas as pd"},
            },
            {
                "tool": "run_tests",
                "arguments": {},
                "result": {"exit_code": 0, "output": "3 passed"},
            },
        ],
        "final_text": "Read backtest.py and ran the tests: exit code 0, 3 passed.",
    }


@pytest.fixture(scope="module")
def spec() -> dict:
    with open(SPEC, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def rl_file(tmp_path_factory) -> Path:
    """What generate.py writes, from the same spec, with no model."""
    data = wai.simulate(
        _coding_agent,
        spec=str(SPEC),
        mode="rl",
        situations=6,
        rollouts_per_request=4,
        budget=24,
        time_budget=None,
        grade="conduct",
        simulator=False,
        seed=0,
        concurrency=4,
    )
    out = tmp_path_factory.mktemp("pi") / "data" / "rl.jsonl"
    data.save(str(out), meta=True)
    return out


def test_spec_is_a_complete_tool_and_policy_spec(spec):
    assert spec["name"] == "quant-coding-agent"
    assert len(spec["policy"]) >= 5 and all(isinstance(p, str) for p in spec["policy"])
    names = [t["function"]["name"] for t in spec["tools"]]
    assert len(names) == len(set(names)) == 10
    for tool in spec["tools"]:
        assert tool["type"] == "function"
        assert tool["function"]["parameters"]["type"] == "object"
        assert tool["function"]["description"]
    assert len(spec["situations"]) == 7


def test_offline_rl_run_gives_every_prompt_the_same_k(rl_file):
    rows = _read(rl_file)
    assert len(rows) == 24
    groups = collections.Counter(r["prompt"] for r in rows)
    assert set(groups.values()) == {4}, groups
    assert len(groups) == 6
    for row in rows:
        assert row["reward"] is not None
        assert row["scenario_id"]
        assert isinstance(row["steps"], list)
    assert rl_file.with_name("rl.meta.json").exists()


def test_diagnose_reports_the_four_numbers_on_the_offline_run(rl_file, tmp_path):
    out = _run("diagnose.py", str(rl_file), cwd=tmp_path)
    for key in (
        "rollouts",
        "prompts",
        "group_sizes",
        "uniform_groups",
        "live_groups",
        "live_fraction",
        "mean_within_group_std",
        "mean_reward",
        "effort_correlation",
    ):
        _stat(out.stdout, key)
    assert _stat(out.stdout, "uniform_groups") == "True"
    assert _stat(out.stdout, "group_sizes") == "{4: 6}"
    # The verdict follows the live fraction, whatever the grader made of it.
    live = float(_stat(out.stdout, "live_fraction"))
    assert out.returncode == (0 if live >= 0.5 else 1), out.stdout
    assert ("PASS" in out.stdout) == (out.returncode == 0)


def test_export_drops_seed_probes_and_keeps_one_phrasing_per_situation(rl_file, spec, tmp_path):
    dest = tmp_path / "new" / "dir" / "prompts.jsonl"  # directory does not exist yet
    out = _run("export_prompts.py", str(rl_file), "--out", str(dest), cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    rows = _read(dest)
    seeds = {s.lower() for s in spec["situations"]}
    assert "seed probes dropped:" in out.stdout
    dropped = int(out.stdout.split("seed probes dropped:")[1].split()[0])
    assert dropped >= 1, out.stdout
    assert len(rows) + dropped == 6
    for row in rows:
        assert set(row) == {"prompt", "example_id", "info"}
        assert row["prompt"].lower() not in seeds
        assert set(row["info"]) <= INFO_FIELDS
        assert row["info"]["scenario_id"] == row["example_id"]
    assert len({r["example_id"] for r in rows}) == len(rows)


def test_export_keep_seeds_keeps_every_situation(rl_file, tmp_path):
    dest = tmp_path / "prompts.jsonl"
    out = _run("export_prompts.py", str(rl_file), "--out", str(dest), "--keep-seeds", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "seed probes dropped: 0" in out.stdout
    assert {r["prompt"] for r in _read(dest)} == {r["prompt"] for r in _read(rl_file)}


def test_export_keeps_every_phrasing_only_when_asked(tmp_path):
    rows = [
        {"prompt": f"phrasing {i} of s1", "scenario_id": "s1", "tier": "ordinary"} for i in range(3)
    ] + [{"prompt": "the one phrasing of s2", "scenario_id": "s2", "faults": {"x": 1}}]
    src = _write(tmp_path / "rl.jsonl", rows)
    one = tmp_path / "one.jsonl"
    every = tmp_path / "every.jsonl"
    assert _run("export_prompts.py", str(src), "--out", str(one), cwd=tmp_path).returncode == 0
    assert (
        _run(
            "export_prompts.py", str(src), "--out", str(every), "--keep-phrasings", cwd=tmp_path
        ).returncode
        == 0
    )
    assert [r["example_id"] for r in _read(one)] == ["s1", "s2"]
    assert [r["example_id"] for r in _read(every)] == ["s1", "s1", "s1", "s2"]
    assert _read(one)[1]["info"] == {"scenario_id": "s2", "faults": {"x": 1}}


def test_export_rows_without_a_scenario_id_stay_separate_tasks(tmp_path):
    rows = [{"prompt": f"foreign prompt {i}"} for i in range(3)]
    src = _write(tmp_path / "foreign.jsonl", rows)
    dest = tmp_path / "prompts.jsonl"
    out = _run("export_prompts.py", str(src), "--out", str(dest), cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    exported = _read(dest)
    assert len(exported) == 3
    assert len({r["example_id"] for r in exported}) == 3
    assert all(r["example_id"].startswith("prompt_") for r in exported)


def _group(prompt: str, rewards: list[float], calls: list[int]) -> list[dict]:
    return [
        {
            "prompt": prompt,
            "reward": reward,
            "steps": [{"tool": "read_file", "arguments": {}}] * n,
            "scenario_id": prompt,
            "rollout_index": i,
        }
        for i, (reward, n) in enumerate(zip(rewards, calls))
    ]


def test_diagnose_passes_a_uniform_live_set_with_positive_effort(tmp_path):
    rows = []
    for i in range(4):
        rows += _group(f"p{i}", [1.0, 0.0, 1.0, 0.0], [3, 0, 2, 1])
    src = _write(tmp_path / "live.jsonl", rows)
    out = _run("diagnose.py", str(src), cwd=tmp_path)
    assert out.returncode == 0, out.stdout
    assert _stat(out.stdout, "uniform_groups") == "True"
    assert _stat(out.stdout, "live_fraction") == "1.0"
    assert _stat(out.stdout, "mean_within_group_std") == "0.5"
    assert float(_stat(out.stdout, "effort_correlation")) > 0
    assert "PASS  groups are uniform and carry gradient" in out.stdout
    assert "WARN" not in out.stdout


def test_diagnose_fails_dead_groups_and_warns_when_effort_is_punished(tmp_path):
    # Three groups of four: one live, two dead (all-max, all-zero). Reward
    # falls as tool calls rise, which is the README's failure mode.
    rows = (
        _group("live", [1.0, 0.0, 1.0, 0.0], [0, 3, 0, 4])
        + _group("all_max", [1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0])
        + _group("all_zero", [0.0, 0.0, 0.0, 0.0], [5, 5, 5, 5])
    )
    src = _write(tmp_path / "dead.jsonl", rows)
    out = _run("diagnose.py", str(src), cwd=tmp_path)
    assert out.returncode == 1, out.stdout
    assert _stat(out.stdout, "live_groups") == "1"
    assert _stat(out.stdout, "dead_all_max") == "1"
    assert _stat(out.stdout, "dead_all_zero") == "1"
    assert "FAIL  only 33% of groups carry gradient" in out.stdout
    assert "WARN  reward is anti-correlated with tool use" in out.stdout
    # The bar is a flag: at --min-live 0.3 the same file passes the gate.
    relaxed = _run("diagnose.py", str(src), "--min-live", "0.3", cwd=tmp_path)
    assert relaxed.returncode == 0, relaxed.stdout


def test_diagnose_fails_groups_of_unequal_size(tmp_path):
    rows = _group("a", [1.0, 0.0, 1.0, 0.0], [1, 1, 1, 1]) + _group("b", [1.0, 0.0], [1, 1])
    src = _write(tmp_path / "ragged.jsonl", rows)
    out = _run("diagnose.py", str(src), cwd=tmp_path)
    assert out.returncode == 1
    assert _stat(out.stdout, "group_sizes") == "{4: 1, 2: 1}"
    assert "FAIL  group sizes are not uniform" in out.stdout


def test_diagnose_ignores_unscored_rows(tmp_path):
    rows = [*_group("a", [1.0, 0.0], [1, 1]), {"prompt": "a", "reward": None, "steps": []}]
    src = _write(tmp_path / "partial.jsonl", rows)
    out = _run("diagnose.py", str(src), cwd=tmp_path)
    assert _stat(out.stdout, "rollouts") == "2"
    assert _stat(out.stdout, "group_sizes") == "{2: 1}"


def test_generate_without_the_key_names_it_and_exits_2(tmp_path):
    out = _run("generate.py", "--situations", "1", "--k", "1", cwd=tmp_path)
    assert out.returncode == 2
    assert "VLLM_API_KEY" in out.stderr
    assert not (tmp_path / "data").exists()
