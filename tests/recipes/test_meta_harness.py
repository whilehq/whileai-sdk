"""The Meta-Harness recipe runs offline end to end on a temporary copy: one
ledger line per candidate, a proposal that names every candidate, a
selection with the gate's checks, and a README whose flags table and quoted
candidate match the code."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.recipes.example_helpers import load_script, readme_defaults

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "recipes" / "papers" / "meta-harness"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _offline_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("WHILEAI_")}
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@pytest.fixture(scope="module")
def copy(tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("meta-harness") / "meta-harness"
    shutil.copytree(RECIPE, dest, ignore=shutil.ignore_patterns("out", "__pycache__"))
    return dest


@pytest.fixture(scope="module")
def dry_run(copy: Path) -> tuple[str, Path]:
    proc = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--dry-run",
            "--propose",
            "--select",
            "--budget",
            "12",
            "--k",
            "2",
        ],
        cwd=copy,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.stdout, copy / "out"


def test_ledger_has_one_line_per_candidate(copy: Path, dry_run):
    _, out = dry_run
    files = sorted(p.name for p in (copy / "candidates").glob("*.py"))
    lines = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines() if line]
    assert [e["candidate"] for e in lines] == files
    for entry in lines:
        assert re.fullmatch(r"[0-9a-f]{12}", entry["fingerprint"])
        assert entry["model"] == "scripted"
        assert entry["train"]["ci95"] and entry["holdout"]["ci95"], "intervals, never a mean alone"
        assert entry["train"]["n_tasks"] + entry["holdout"]["n_tasks"] == entry["n_tasks"]
        assert (out / entry["worst"]).exists() and (out / entry["rows"]).exists()
        assert "scripted-b" in entry["held_out_models"]


def test_proposal_names_every_candidate_and_the_next_file(copy: Path, dry_run):
    _, out = dry_run
    proposal = (out / "proposal.md").read_text(encoding="utf-8")
    files = sorted(p.name for p in (copy / "candidates").glob("*.py"))
    for name in files:
        assert f"## {name}" in proposal
        assert (copy / "candidates" / name).read_text(encoding="utf-8").rstrip() in proposal
    assert f"write candidates/{len(files):02d}_<name>.py" in proposal
    assert "Worst rows:" in proposal
    assert "arXiv:2603.28052" in proposal


def test_selection_carries_the_gate(dry_run):
    stdout, out = dry_run
    selected = json.loads((out / "selected.json").read_text(encoding="utf-8"))
    assert selected["baseline"] == "00_baseline.py"
    assert set(selected["checks"]) == {"scripted", "scripted-b"}
    for check in selected["checks"].values():
        lo, hi = check["ci95"]
        assert lo <= check["delta"] <= hi
        assert check["clears"] == (lo > 0)
    assert "attribution" in selected and selected["attribution"]["verdict"] in (
        "harness",
        "model",
        "unresolved",
    )
    assert "select:" in stdout and "attribution on pass_at_1" in stdout


def test_frozen_tasks_replay_across_runs(copy: Path, dry_run):
    _, out = dry_run
    before = (out / "ledger.jsonl").read_text(encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "run.py", "--dry-run", "--budget", "12", "--k", "2"],
        cwd=copy,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "ledger.jsonl").read_text(encoding="utf-8") == before, (
        "a second run replays out/tasks.jsonl and reproduces every number"
    )


def test_scripted_models_without_dry_run_name_the_fix(copy: Path):
    proc = subprocess.run(
        [sys.executable, "run.py"],
        cwd=copy,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2
    assert "--dry-run" in proc.stderr and "provider:model" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_readme_flags_match_the_parser():
    sys.path.insert(0, str(RECIPE))
    run = load_script("meta_harness_run", RECIPE / "run.py")
    defaults = {a.dest: a.default for a in run.build_parser()._actions if a.dest != "help"}
    documented = readme_defaults(RECIPE / "README.md")
    assert set(documented) == set(defaults), set(documented) ^ set(defaults)
    for name, text in documented.items():
        actual = defaults[name]
        if text == "off":
            assert actual is False, name
        else:
            assert str(actual) == text, (name, text, actual)


def test_readme_quotes_a_candidate_verbatim():
    text = (RECIPE / "README.md").read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    assert blocks, "the README shows one candidate file"
    candidate = (RECIPE / "candidates" / "01_no_filler.py").read_text(encoding="utf-8")
    assert blocks[0].strip() == candidate.strip()
