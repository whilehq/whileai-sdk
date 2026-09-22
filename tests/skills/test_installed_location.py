"""A skill check has to run where ``wai init`` puts it.

``tests/skills/test_skills.py`` runs every ``check.py`` from
``skills/<name>/`` in this checkout. That is not the path a user takes:
``wai init`` installs the same file at ``<your repo>/.claude/skills/<name>/``,
with no ``recipes/``, no ``tests/`` and no repository root above it, because
neither is in the wheel. ``skills/harness-search/check.py`` resolved its recipe
as ``Path(__file__).resolve().parents[2]`` and crashed there with a bare
``FileNotFoundError`` (#806), and nothing caught it, because the one path a
user takes was the one path nothing tested.

So: install every skill the way ``wai init`` installs it, into a directory
outside this checkout, and run it there. A check either passes, or exits
``init_repo.NEEDS_CHECKOUT`` with a message naming the command that fixes it
(``docs/reference/style.md``, rule 10). A traceback is neither.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from whileai import init_repo

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / "skills"
TESTED = sorted(p.parent for p in SKILLS.glob("*/check.py"))
CLONE = "git clone https://github.com/whilehq/whileai-sdk"


def _fetch(url: str) -> str | None:
    """No network in tests: serve the files from this checkout."""
    name, file = url.rsplit("/", 2)[1:]
    path = SKILLS / name / file
    return path.read_text(encoding="utf-8") if path.exists() else None


def _install(root: Path, name: str) -> Path:
    """What ``wai init --skill <name>`` leaves behind, and where it leaves it."""
    landed = init_repo.install_skills(root, [name], fetch=_fetch)
    assert landed[name], f"{name}: install_skills wrote nothing"
    script = root / ".claude" / "skills" / name / "check.py"
    assert script.exists(), f"{name}: no check.py under .claude/skills/"
    return script


def _run(script: Path, home: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("WHILEAI_")}
    env["WHILEAI_HOME"] = str(home)  # no saved credentials either
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("skill", TESTED, ids=[p.name for p in TESTED])
def test_check_runs_where_wai_init_installs_it(skill: Path, tmp_path: Path) -> None:
    script = _install(tmp_path / "userproj", skill.name)
    # The user's repo is not this checkout, and nothing above it is either.
    assert not any((p / "recipes" / "papers").exists() for p in script.parents), (
        "this test only means something when the installed copy has no source checkout above it"
    )
    proc = _run(script, tmp_path / "home")
    said = proc.stdout + proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr, (
        f"{skill.name}/check.py crashed from the installed location instead of naming the fix"
        f"\n--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
    )
    assert proc.returncode in (0, init_repo.NEEDS_CHECKOUT), (
        f"{skill.name}/check.py exited {proc.returncode} from the installed location"
        f"\n--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
    )
    if proc.returncode == init_repo.NEEDS_CHECKOUT:
        assert CLONE in said, f"{skill.name}/check.py stopped without naming the clone command"
        assert f"skills/{skill.name}/check.py" in said, (
            f"{skill.name}/check.py stopped without naming the command to run in the clone"
        )


@pytest.mark.parametrize("name", init_repo.DEFAULT_SKILLS, ids=list(init_repo.DEFAULT_SKILLS))
def test_the_default_skills_pass_from_the_installed_location(name: str, tmp_path: Path) -> None:
    """``wai init`` with no ``--skill`` installs these three; all three run
    where it puts them, and that is what ``wai init`` reports."""
    if not (SKILLS / name / "check.py").exists():
        pytest.skip(f"{name} is prose-only")
    proc = _run(_install(tmp_path / "userproj", name), tmp_path / "home")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]


def test_harness_search_names_the_clone_instead_of_a_filenotfounderror(tmp_path: Path) -> None:
    """#806: ``wai init --skill harness-search`` then running the check gave a
    ``copytree`` traceback naming ``.claude/recipes/papers/meta-harness``, a
    path that is never created. It now says which clone to run."""
    script = _install(tmp_path / "userproj", "harness-search")
    proc = _run(script, tmp_path / "home")
    assert proc.returncode == init_repo.NEEDS_CHECKOUT, proc.stdout + proc.stderr
    assert "FileNotFoundError" not in proc.stderr
    assert "recipes/papers/meta-harness" in proc.stderr
    assert CLONE in proc.stderr
    assert "uv run python skills/harness-search/check.py" in proc.stderr
    # and wai init reads that exit code as "cannot run here", not as a failure
    rc, tail = init_repo.run_check(tmp_path / "userproj", "harness-search")
    assert rc == init_repo.NEEDS_CHECKOUT
    assert CLONE in tail or "checkout" in tail, tail


def test_the_installed_check_is_the_checked_in_one(tmp_path: Path) -> None:
    """The file a user runs is the file CI runs; no install-time rewriting."""
    for name in ("harness-search", init_repo.CHECK_SKILL):
        script = _install(tmp_path / name, name)
        assert script.read_text(encoding="utf-8") == (SKILLS / name / "check.py").read_text(
            encoding="utf-8"
        )
