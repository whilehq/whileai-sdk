"""Every paper recipe under ``recipes/papers/`` keeps the contract, its table
row is current, and it still compiles."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "recipes" / "papers"


def test_papers_check_passes() -> None:
    out = subprocess.run(
        [sys.executable, str(PAPERS / "check.py")], capture_output=True, text=True, cwd=REPO
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_every_paper_recipe_compiles() -> None:
    scripts = sorted(PAPERS.glob("*/recipe.py"))
    assert scripts, "no recipe.py under recipes/papers/ (the template counts)"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)


def test_check_says_which_gates_a_recipe_did_not_reach() -> None:
    """The interval, noise band and proxy gates live inside ``verdict ==
    "moved"``; a recipe that never reached them must not print the same
    plain ``ok`` as one that passed them (#673)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("papers_check", PAPERS / "check.py")
    assert spec is not None and spec.loader is not None
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    unresolved = {
        "delta": {"verdict": "unresolved"},
        "checks": {"train_seeds": {"baseline": 1, "recipe": 1}, "over_optimized": False},
    }
    note = check.skipped_note(unresolved)
    assert "verdict unresolved at 1 training seed(s) per arm" in note
    assert "interval, noise band and proxy check not enforced" in note
    assert "OVER-OPTIMIZED" not in note
    # over-optimization is reported at every verdict, not only on "moved"
    hidden = {**unresolved, "checks": {**unresolved["checks"], "over_optimized": True}}
    assert "proxy-vs-target says OVER-OPTIMIZED" in check.skipped_note(hidden)
    moved = {
        "delta": {"verdict": "moved"},
        "checks": {"train_seeds": {"baseline": 3, "recipe": 3}, "over_optimized": False},
    }
    assert check.skipped_note(moved) == ""
    # and the script's own lines carry the note for every recipe that skipped
    out = subprocess.run(
        [sys.executable, str(PAPERS / "check.py")], capture_output=True, text=True, cwd=REPO
    )
    assert out.returncode == 0, out.stdout + out.stderr
    for line in out.stdout.splitlines():
        if line.startswith("ok   ") and "paper recipe(s)" not in line:
            assert "not enforced" in line or "OVER-OPTIMIZED" in line or "(" not in line, line
