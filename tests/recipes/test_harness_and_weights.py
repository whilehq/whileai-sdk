"""The harness-and-weights recipe runs its dry run offline end to end on a
temporary copy: the task build, a disjoint split, one ledger line per
candidate, the gate, the 2x2 grid with the weights arms marked not run, and
``out/dry_run.json``. The copy carries ``meta-harness`` too, because the
recipe imports that loop rather than forking it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "recipes" / "papers"
NAME = "harness-and-weights"


def _offline_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("WHILEAI_")}
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


@pytest.fixture(scope="module")
def copy(tmp_path_factory) -> Path:
    papers = tmp_path_factory.mktemp("papers")
    # results.json is the full live run's; the copy starts without it so the
    # last test can assert that a dry run never writes one.
    ignore = shutil.ignore_patterns("out", "__pycache__", ".cache", "results.json")
    shutil.copytree(PAPERS / "meta-harness", papers / "meta-harness", ignore=ignore)
    shutil.copytree(PAPERS / NAME, papers / NAME, ignore=ignore)
    return papers / NAME


@pytest.fixture(scope="module")
def dry_run(copy: Path) -> tuple[str, Path]:
    proc = subprocess.run(
        [sys.executable, "recipe.py", "--dry-run", "--limit", "16", "--k", "2"],
        cwd=copy,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout, copy / "out"


def test_tasks_build_is_large_and_split_is_disjoint(copy: Path):
    sys.path.insert(0, str(copy))
    try:
        import importlib

        tasks = importlib.import_module("tasks")
        tasks = importlib.reload(tasks)
    finally:
        sys.path.remove(str(copy))
    built = tasks.build()
    assert len(built) >= 120
    assert len({t["scenario_id"] for t in built}) == len(built)
    for t in built:
        assert "tests" in t["privileged"] and "assert" in t["privileged"]["tests"]
        # the answer is in the tests, never in the prompt
        expected = t["privileged"]["tests"].split("(")[-1].split(")")[0]
        assert expected not in t["prompt"]
    train, hold = tasks.split(built)
    assert {t["scenario_id"] for t in train}.isdisjoint({t["scenario_id"] for t in hold})
    assert {t["family"] for t in train}.isdisjoint({t["family"] for t in hold})
    assert len(train) + len(hold) == len(built)
    train_t, hold_t = tasks.split(built, by="task")
    assert {t["scenario_id"] for t in train_t}.isdisjoint({t["scenario_id"] for t in hold_t})


def test_ledger_has_one_line_per_candidate(copy: Path, dry_run):
    _, out = dry_run
    files = sorted(p.name for p in (copy / "candidates").glob("*.py"))
    lines = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines() if line]
    assert [e["candidate"] for e in lines] == files
    for entry in lines:
        assert entry["model"] == "base"
        assert entry["train"]["ci95"] and entry["holdout"]["ci95"], "intervals, never a mean alone"
        assert (out / entry["worst"]).exists() and (out / entry["rows"]).exists()
    assert lines[0]["tool_calls"] == 0, "the baseline has no tool"
    assert lines[-1]["tool_calls"] > 0, "the tool candidate called run_python"
    assert (out / "tasks.jsonl").exists() and (out / "proposal.md").exists()


def test_dry_run_json_marks_the_weights_arms_not_run(dry_run):
    stdout, out = dry_run
    dry = json.loads((out / "dry_run.json").read_text(encoding="utf-8"))
    assert set(dry) >= {
        "tasks",
        "candidates",
        "searched_harness",
        "gate_passed",
        "grid",
        "attribution",
        "pairs",
    }
    assert dry["tasks"]["n"] >= 120 and dry["tasks"]["decontaminated_dropped"] == 0
    assert dry["tasks"]["train"] + dry["tasks"]["holdout"] == dry["tasks"]["n"]
    for arm in ("weights", "both"):
        assert dry["grid"][arm]["status"] == "not run (dry run: no GPU)"
    for arm in ("neither", "harness"):
        assert dry["grid"][arm]["ci"] and dry["grid"][arm]["model"] == "base"
    assert "skipped" in dry["attribution"], "two cells cannot be attributed; the hole is named"
    assert "attribution skipped:" in stdout
    assert dry["pairs"]["both_vs_weights"] == {"status": "not run"}
    assert "n_paired" in dry["pairs"]["harness_vs_neither"]
    assert not (out.parent / "results.json").exists(), "a dry run claims no number"
