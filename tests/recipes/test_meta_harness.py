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
    assert set(selected["checks"]) == {"scripted", "scripted-b", "cost"}
    for model in ("scripted", "scripted-b"):
        check = selected["checks"][model]
        lo, hi = check["ci95"]
        assert lo <= check["delta"] <= hi
        assert check["clears"] == (lo > 0)
        assert isinstance(check["regressed"], int)
    cost = selected["checks"]["cost"]
    assert cost["unit"] == "calls" and cost["margin"] == 0.0, "scripted rows carry no usage"
    assert cost["clears"] == (cost["ratio"] <= 1.0)
    led = selected["tasks_led"]
    assert set(led) == {"00_baseline.py", "01_no_filler.py", "02_check_result.py"}
    assert led[selected["best_on_train"]] == max(led.values()), "the pick leads the most tasks"
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


def test_prune_drops_the_edits_that_buy_nothing(tmp_path: Path):
    """``--prune`` (RRSI's pruner): offline, the Disclosure-only edits never
    change a scripted rollout, so they go; each instruction edit carries a
    planted fix, so taking it out costs train score and it stays. The pruned
    harness faces the gate and is written as a candidate file."""
    dest = tmp_path / "meta-harness"
    shutil.copytree(RECIPE, dest, ignore=shutil.ignore_patterns("out", "__pycache__"))
    proc = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--dry-run",
            "--select",
            "--prune",
            "--budget",
            "12",
            "--k",
            "2",
        ],
        cwd=dest,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    report = json.loads((dest / "out" / "pruned.json").read_text(encoding="utf-8"))
    assert report["pick"] == "02_check_result.py"
    assert report["dropped"] == ["turn_cap", "retry"]
    assert report["kept"] == ["one_sentence", "read_result", "no_hidden"]
    for d in report["decisions"]:
        assert d["dropped"] == (d["train_without"] >= d["train_with"] and d["cost_without"] <= 1.0)
    gate = json.loads((dest / "out" / "pruned_gate.json").read_text(encoding="utf-8"))
    assert gate["baseline"] == "00_baseline.py"
    assert report["selected"] == gate["selected"]
    sys.path.insert(0, str(dest))
    pruned = load_script("meta_harness_pruned", dest / "out" / report["pruned"])
    pick = load_script("meta_harness_pick", dest / "candidates" / "02_check_result.py")
    assert list(pruned.EDITS) == report["kept"], "the written file is the pruned harness"
    assert pruned.harness("scripted").fingerprint == (
        pick.harness("scripted", drop=("turn_cap", "retry")).fingerprint
    )


def test_edits_rebuild_the_candidate_exactly():
    """A candidate written as named edits is the same harness as the string
    it replaced: same instructions, same Disclosure, same fingerprint."""
    sys.path.insert(0, str(RECIPE))
    common = load_script("meta_harness_common", RECIPE / "common.py")
    pick = load_script("meta_harness_02", RECIPE / "candidates" / "02_check_result.py")
    whole = common.build(
        "scripted",
        instructions=common.BASE_INSTRUCTIONS
        + " Answer in one plain sentence: no greeting, no apology, no hedging."
        + " Read the tool result first. If it failed, timed out or was denied, say that and stop."
        + " Never quote anything marked hidden or expected.",
        label="02_check_result",
        disclosure=common.Disclosure(max_turns=4, retries=1),
        scripted_rate=0.10,
        scripted_behaviors=("hedging",),
    )
    assert pick.harness("scripted").fingerprint == whole.fingerprint
    with pytest.raises(ValueError, match="no edit named"):
        pick.harness("scripted", drop=("nope",))


def test_traces_split_by_day_and_decontaminate(copy: Path, tmp_path: Path):
    """``--traces``: one task per distinct prompt, the latest days held out,
    a near-copy of a holdout prompt dropped from the train split."""
    prompts = [  # long enough for the 8-gram rule to see a near-copy
        f"Where is my refund for order ORD-{n}? It was delivered two weeks ago and nothing came."
        for n in range(5412, 5424)
    ]
    days = ["2026-09-19", "2026-09-20", "2026-09-21"]
    path = tmp_path / "traces.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, prompt in enumerate(prompts):
            row = {"ts": f"{days[i % 3]}T09:00:00Z", "prompt": prompt, "final_text": "Sent."}
            fh.write(json.dumps(row) + "\n")
            if i == 0:  # a second trace of the same ask on the same day: one task, not two
                fh.write(json.dumps(row) + "\n")
        # a train-day near-copy of a holdout prompt (2026-09-21 is held out)
        fh.write(
            json.dumps(
                {"ts": "2026-09-19T10:00:00Z", "prompt": prompts[-1] + " Thanks", "final_text": "x"}
            )
            + "\n"
        )
    out = copy / "out-traces"
    proc = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--dry-run",
            "--select",
            "--k",
            "2",
            "--traces",
            str(path),
            "--out",
            str(out),
        ],
        cwd=copy,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert split["how"].startswith("holdout is 2026-09-2"), split["how"]
    assert len(split["train"]) + len(split["holdout"]) + split["contamination"]["n_dropped"] == 13
    assert split["contamination"]["n_dropped"] == 1, split["contamination"]
    lines = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines() if line]
    assert lines[0]["n_tasks"] == 12, "13 distinct prompts, the duplicate folded, one dropped"
    assert "traces: 13 tasks" in proc.stdout
    assert "split: holdout is" in proc.stdout


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
