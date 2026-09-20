"""The README is a front page, not a paper (CONSTITUTION.md, belief 6).

Pins the skeleton the most-used Python repos share and the README adopted
on 2026-09-19 (#566): the loop as five lines before the first heading, the
sections in one order, and a prose budget so explanation moves to code
blocks or to ``docs/`` instead of growing here.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

# The section order. Every H2 in the README, in this order, nothing else.
SECTIONS = [
    "Install",
    "Quick start",
    "With your agent",
    "The loop, call by call",
    "Why the numbers hold",
    "Recipes",
    "Platform",
    "Documentation",
    "Development",
    "Cite",
    "License",
]

# The loop, one line per step, the first thing after the badges and link bar.
STEPS = ["Simulate", "Grade", "Measure", "Select", "Train"]

# Words of prose outside code fences, HTML, tables and the collapsed
# references. The 2026-09-19 rewrite landed at about 800.
PROSE_BUDGET = 900


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _headings(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"^## (.+)$", text, re.M)]


def _prose_words(text: str) -> int:
    words = 0
    in_fence = in_details = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if line.startswith("<details"):
            in_details = True
        if line.startswith("</details"):
            in_details = False
            continue
        if in_fence or in_details:
            continue
        if line.startswith(("<", "|", "#")):
            continue
        words += len(line.split())
    return words


def test_sections_in_order() -> None:
    assert _headings(_text()) == SECTIONS, (
        "README H2s drifted from the skeleton in CONSTITUTION.md belief 6; "
        "add depth under an existing section or in docs/, not a new heading"
    )


def test_loop_is_the_first_thing() -> None:
    text = _text()
    top = text.split("\n## ", 1)[0]
    bullets = re.findall(r"^- \*\*(\w+)\.\*\*", top, re.M)
    assert bullets == STEPS, f"the five-line loop must open the page; found {bullets}"
    # Nothing in front of the list but the hero, badges and link bar.
    before = top.split("- **Simulate.**", 1)[0]
    prose = [
        ln
        for ln in before.splitlines()
        if ln.strip() and not ln.startswith("<") and not ln.strip().startswith("<")
    ]
    assert prose == [], f"no hook or paragraph before the loop list: {prose}"


def test_prose_budget() -> None:
    n = _prose_words(_text())
    assert n <= PROSE_BUDGET, (
        f"README prose is {n} words, over the {PROSE_BUDGET} budget; "
        "move the explanation into a code block or docs/"
    )


def test_quick_start_is_offline_and_shows_its_output() -> None:
    text = _text()
    section = text.split("## Quick start", 1)[1].split("\n## ", 1)[0]
    assert "seeded_agent" in section and "simulator=False" in section
    assert "pass@1" in section.split("```\n", 1)[1], "the printed output follows the block"


def test_references_are_collapsed_and_numbered() -> None:
    text = _text()
    assert "<details>" in text and "<summary><b>References</b></summary>" in text
    refs = re.findall(r"^(\d+)\. ", text.split("<details>", 1)[1], re.M)
    assert [int(r) for r in refs] == list(range(1, len(refs) + 1))
    assert len(refs) >= 24, "docstrings cite these by number; never renumber or drop one"
