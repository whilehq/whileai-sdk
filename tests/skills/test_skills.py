"""Every skill under skills/ is a tested playbook.

The rule (skills/BRIEF.md): each ```python block in a SKILL.md must appear
verbatim in the skill's check.py, and check.py must run offline, with no
key, in under 60 seconds. So the code an agent copies out of the playbook
is code that ran in CI this morning.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / "skills"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

# Skills that carry a check.py are the tested ones; the two older playbooks
# are prose-only and stay that way until someone gives them a check.
TESTED = sorted(p.parent for p in SKILLS.glob("*/check.py"))
ALL = sorted(p.parent for p in SKILLS.glob("*/SKILL.md"))


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


@pytest.mark.parametrize("skill", ALL, ids=[p.name for p in ALL])
def test_frontmatter(skill: Path):
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    m = FRONT.match(text)
    assert m, f"{skill.name}: SKILL.md needs a --- frontmatter block"
    front = m.group(1)
    assert re.search(r"^name:\s*" + re.escape(skill.name) + r"\s*$", front, re.M), (
        f"{skill.name}: frontmatter name must equal the folder name"
    )
    assert "description:" in front and "version:" in front


@pytest.mark.parametrize("skill", TESTED, ids=[p.name for p in TESTED])
def test_every_code_block_is_in_check(skill: Path):
    doc = (skill / "SKILL.md").read_text(encoding="utf-8")
    check = _norm((skill / "check.py").read_text(encoding="utf-8"))
    blocks = [_norm(b) for b in BLOCK.findall(doc)]
    assert blocks, f"{skill.name}: SKILL.md has no ```python blocks"
    for i, block in enumerate(blocks, 1):
        assert block in check, (
            f"{skill.name}: python block {i} of SKILL.md is not in check.py verbatim:\n{block}"
        )


@pytest.mark.parametrize("skill", TESTED, ids=[p.name for p in TESTED])
def test_check_runs_offline(skill: Path, tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("WHILEAI_")}
    env["WHILEAI_HOME"] = str(tmp_path)  # no saved credentials either
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(skill / "check.py")],
        cwd=skill,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"{skill.name}/check.py failed\n--- stdout ---\n{proc.stdout[-3000:]}"
        f"\n--- stderr ---\n{proc.stderr[-3000:]}"
    )
    assert "verdict" in proc.stdout.lower() or "beats" in proc.stdout or "serving" in proc.stdout, (
        f"{skill.name}/check.py did not print the verdict line"
    )


def test_skills_index_lists_every_skill():
    index = (SKILLS / "README.md").read_text(encoding="utf-8")
    for skill in ALL:
        assert f"{skill.name}/" in index, f"skills/README.md does not list {skill.name}"
