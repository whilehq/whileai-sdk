"""``whileai init``: make a repository dumb simple for a coding agent.

One command after ``pip install whileai``. It writes a short block into
``AGENTS.md`` (Codex, Cursor and Copilot read it), makes sure ``CLAUDE.md``
includes it (Claude Code reads that), installs the tested skills under
``.claude/skills/``, and runs the evals skill's ``check.py`` so the agent
sees the verdict line before it touches the user's code. After that "make
my evals better" in any coding agent lands on the tested playbook with no
URL to remember. Run it again after an upgrade: the block is replaced in
place, never duplicated.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from . import __version__

MARK_START = "<!-- whileai:start"
MARK_END = "<!-- whileai:end -->"
BLOCK = re.compile(re.escape(MARK_START) + r".*?" + re.escape(MARK_END) + r"\n?", re.DOTALL)
INCLUDE = "@AGENTS.md"

#: The skills a repo gets by default: the evals playbook and the simulate loop.
DEFAULT_SKILLS: tuple[str, ...] = (
    "strengthen-your-evals",
    "manage-experiments",
    "whileai-simulations",
)
SKILL_FILES: tuple[str, ...] = ("SKILL.md", "check.py")
RAW_URL = "https://raw.githubusercontent.com/whilehq/whileai-sdk/main/skills/{name}/{file}"
CHECK_SKILL = "strengthen-your-evals"
HTTP_OK = 200

Fetch = Callable[[str], str | None]


def agents_block(version: str = __version__) -> str:
    """The block a coding agent reads. Under twenty lines on purpose."""
    return f"""{MARK_START} v{version} -->
## Evals and post-training with whileai

whileai (`pip install whileai`, `import whileai.simulations as wai`) builds evals that can fail and proves every change on a held-out set with a 95% interval. Docs: https://docs.withwhile.com. Runs: https://withwhile.com/platform/runs.

When asked to build, improve or check evals for an agent, follow `.claude/skills/strengthen-your-evals/SKILL.md` top to bottom. Its `check.py` is the tested copy of every code block; run it first (`python .claude/skills/strengthen-your-evals/check.py`).

Rules:
- The agent is a callable `message -> {{"steps": [...], "final_text": ...}}` that runs its own tools. `whileai init-evals` writes that wrapper for a Python bot.
- Write the held-out asks once, then replay them with `tasks=` for every version. Never hand-pick the asks the agent failed.
- Hand labels come from a person. Ask before labelling; never invent `gold_reward`.
- You know this repo best: name the agent after the product, behaviors as the policy doc phrases them, versions as the team ships them (tag, PR, date, prompt label), the test by its content hash. https://docs.withwhile.com/platform/naming
- Before a second version, a sweep or a replicate, follow `.claude/skills/manage-experiments/SKILL.md`: post the question first, name arms in words with numbers in the record, seeds as replicates, points not fractions, failed rows and a note per score, then read the account back with `readback(tracked)`.
- Score every behavior and report: `track(...)`, `tracked.behavior(...)`, `tracked.run(version, method="eval", harness=Harness(...))`, `run.score(...)`, `run.finish(...)`, `print(tracked.verdict())`.
- A difference is a result only when its interval excludes zero and clears the noise floor. Otherwise say "about the same".
- `WHILEAI_API_KEY` (`whileai signup --email you@example.com`) is needed only for the report; everything else runs offline with no key.
{MARK_END}
"""


def write_agents_md(root: Path, version: str = __version__) -> Path:
    """Create ``AGENTS.md`` or replace the whileai block inside it."""
    path = root / "AGENTS.md"
    block = agents_block(version)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if BLOCK.search(text):
            text = BLOCK.sub(lambda _m: block, text, count=1)
        else:
            text = text.rstrip("\n") + "\n\n" + block if text.strip() else block
    else:
        text = "# Agents\n\n" + block
    path.write_text(text, encoding="utf-8")
    return path


def ensure_claude_md(root: Path) -> Path:
    """Claude Code reads ``CLAUDE.md``; one include line points it at ``AGENTS.md``."""
    path = root / "CLAUDE.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if INCLUDE not in text:
            path.write_text(text.rstrip("\n") + f"\n\n{INCLUDE}\n", encoding="utf-8")
    else:
        path.write_text(f"{INCLUDE}\n", encoding="utf-8")
    return path


def _local_skills_dir() -> Path | None:
    """A source checkout has ``skills/`` next to the package; the wheel does not."""
    candidate = Path(__file__).resolve().parents[1] / "skills"
    return candidate if (candidate / CHECK_SKILL / "SKILL.md").exists() else None


def _http_fetch(url: str) -> str | None:
    try:
        import requests

        reply = requests.get(url, timeout=20)
    except Exception:  # offline is a normal state
        return None
    return reply.text if reply.status_code == HTTP_OK else None


def install_skills(
    root: Path, names: Iterable[str], fetch: Fetch | None = None
) -> dict[str, list[Path]]:
    """Copy each skill's ``SKILL.md`` and ``check.py`` under ``.claude/skills/``.

    A source checkout is read directly; otherwise the files come from the
    repository's ``main`` branch. Returns what landed per skill; a skill
    with no file could not be reached."""
    fetch = fetch or _http_fetch
    local = _local_skills_dir()
    out: dict[str, list[Path]] = {}
    for name in names:
        written: list[Path] = []
        for file in SKILL_FILES:
            text: str | None = None
            if local is not None and (local / name / file).exists():
                text = (local / name / file).read_text(encoding="utf-8")
            if text is None:
                text = fetch(RAW_URL.format(name=name, file=file))
            if text is None:
                continue
            target = root / ".claude" / "skills" / name / file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            written.append(target)
        out[name] = written
    return out


def run_check(root: Path, name: str = CHECK_SKILL, timeout: float = 180) -> tuple[int, str]:
    """Run a skill's ``check.py`` offline and return (exit code, last lines)."""
    script = (Path(root) / ".claude" / "skills" / name / "check.py").resolve()
    if not script.exists():
        return 1, f"{script} not found"
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=None,
        )
    except subprocess.TimeoutExpired:
        return 1, f"check.py did not finish in {timeout:.0f}s"
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    tail = "\n".join(lines[-3:]) if lines else (proc.stderr or "")[-400:]
    return proc.returncode, tail


def status(root: Path | str = ".") -> dict[str, Any]:
    """What ``whileai init`` left in this repo, for ``whileai status``."""
    root = Path(root)
    agents = root / "AGENTS.md"
    text = agents.read_text(encoding="utf-8") if agents.exists() else ""
    m = re.search(re.escape(MARK_START) + r" v([^\s]+) -->", text)
    version = m.group(1) if m else None
    skills_dir = root / ".claude" / "skills"
    skills = (
        sorted(p.parent.name for p in skills_dir.glob("*/SKILL.md")) if skills_dir.exists() else []
    )
    claude = root / "CLAUDE.md"
    return {
        "agents_md": version is not None,
        "agents_md_version": version,
        "stale": version is not None and version != __version__,
        "claude_md_includes": claude.exists() and INCLUDE in claude.read_text(encoding="utf-8"),
        "skills": skills,
    }


def init(
    root: Path | str = ".",
    *,
    skills: Iterable[str] = DEFAULT_SKILLS,
    check: bool = True,
    fetch: Fetch | None = None,
    out: Any = None,
) -> int:
    """Write the block, the include and the skills; run the evals check.

    Returns 0 when everything landed and the check passed, 1 otherwise.
    """
    out = out or sys.stdout
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    agents = write_agents_md(root)
    claude = ensure_claude_md(root)
    print(
        f"wrote {agents.name} (whileai block v{__version__}); {claude.name} includes it", file=out
    )
    landed = install_skills(root, skills, fetch=fetch)
    code = 0
    for name, files in landed.items():
        if files:
            print(
                f"installed .claude/skills/{name}/ ({', '.join(f.name for f in files)})", file=out
            )
        else:
            code = 1
            print(
                f"could not fetch skills/{name} (offline?): read it at "
                f"https://github.com/whilehq/whileai-sdk/blob/main/skills/{name}/SKILL.md",
                file=out,
            )
    if check and landed.get(CHECK_SKILL):
        rc, tail = run_check(root, CHECK_SKILL)
        print(
            f"check .claude/skills/{CHECK_SKILL}/check.py: {'ok' if rc == 0 else f'exit {rc}'}",
            file=out,
        )
        if tail:
            print("  " + tail.replace("\n", "\n  "), file=out)
        code = code or rc
    print(
        "next: tell your coding agent 'use whileai to build me better evals for my agent'",
        file=out,
    )
    return code
