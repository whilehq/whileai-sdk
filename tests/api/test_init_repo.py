"""``whileai init`` leaves a repo a coding agent can work in: an AGENTS.md
block, a CLAUDE.md include, the tested skills under .claude/skills/."""

from __future__ import annotations

import io
from pathlib import Path

from whileai import __version__, init_repo
from whileai.cli import main

REPO = Path(__file__).resolve().parents[2]


def _fetch(url: str) -> str | None:
    """No network in tests: serve the files from this checkout."""
    name, file = url.rsplit("/", 2)[1:]
    path = REPO / "skills" / name / file
    return path.read_text(encoding="utf-8") if path.exists() else None


def test_init_writes_block_include_and_skills(tmp_path: Path):
    out = io.StringIO()
    rc = init_repo.init(tmp_path, check=False, fetch=_fetch, out=out)
    assert rc == 0, out.getvalue()
    agents = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert agents.count(init_repo.MARK_START) == 1 and init_repo.MARK_END in agents
    assert f"v{__version__}" in agents
    assert "strengthen-your-evals/SKILL.md" in agents and "tasks=" in agents
    assert len(init_repo.agents_block().splitlines()) <= 20
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"
    for name in init_repo.DEFAULT_SKILLS:
        assert (tmp_path / ".claude" / "skills" / name / "SKILL.md").exists()
    assert (tmp_path / ".claude" / "skills" / "strengthen-your-evals" / "check.py").exists()
    s = init_repo.status(tmp_path)
    assert s["agents_md"] and s["agents_md_version"] == __version__ and not s["stale"]
    assert s["claude_md_includes"] and "strengthen-your-evals" in s["skills"]


def test_init_replaces_the_block_and_keeps_the_rest(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text(
        "# Our agents\n\nKeep tests green.\n\n<!-- whileai:start v0.1 -->\nold\n<!-- whileai:end -->\n",
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text("Read AGENTS.md first.\n", encoding="utf-8")
    assert init_repo.status(tmp_path)["stale"]
    init_repo.init(tmp_path, check=False, fetch=_fetch, out=io.StringIO())
    agents = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert agents.startswith("# Our agents\n\nKeep tests green.")
    assert agents.count("<!-- whileai:start") == 1 and "\nold\n" not in agents
    claude = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert claude.startswith("Read AGENTS.md first.") and claude.rstrip().endswith("@AGENTS.md")
    init_repo.init(tmp_path, check=False, fetch=_fetch, out=io.StringIO())
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").count("@AGENTS.md") == 1


def test_init_says_when_a_skill_cannot_be_fetched(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(init_repo, "_local_skills_dir", lambda: None)
    out = io.StringIO()
    rc = init_repo.init(
        tmp_path, skills=["no-such-skill"], check=False, fetch=lambda _u: None, out=out
    )
    assert rc == 1 and "could not fetch skills/no-such-skill" in out.getvalue()


def test_cli_init_and_status(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(init_repo, "_http_fetch", _fetch)
    assert main(["init", "--dir", str(tmp_path), "--no-check"]) == 0
    captured = capsys.readouterr().out
    assert (
        "wrote AGENTS.md" in captured
        and "installed .claude/skills/strengthen-your-evals/" in captured
    )
    monkeypatch.chdir(tmp_path)
    assert main(["status"]) == 0
    assert '"agents_md": true' in capsys.readouterr().out


def test_run_check_runs_the_installed_skill_from_a_relative_root(tmp_path: Path, monkeypatch):
    """The 0.99 wheel resolved a relative script path against the skill dir and
    found nothing; the check must run from ``whileai init`` in any cwd."""
    init_repo.install_skills(tmp_path, ["strengthen-your-evals"], fetch=_fetch)
    monkeypatch.chdir(tmp_path.parent)
    rc, tail = init_repo.run_check(Path(tmp_path.name))
    assert rc == 0, tail
    assert "ok:" in tail
