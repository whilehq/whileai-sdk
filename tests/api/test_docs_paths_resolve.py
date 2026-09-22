"""Every ``wai.<path>`` a hand-written docs page spells has to resolve from
``import pytest

import whileai as wai``.

#735 and #780 are the same defect twice: a page names a path that raises
``AttributeError`` when a reader types it. ``docs/reference/what-to-run.md``
wrote ``wai.score.eval_power`` (``whileai.score`` does not exist) and
``docs/reference/five-calls.md`` opened with ``import whileai as wai`` and
then called ``wai.delta_report``, which lives one dot down. Jacob's ask on
#780: "One check that every ``wai.<path>`` a docs page spells resolves would
catch all four."

Most of the remaining misses are #818 and #823, not typos: the measurement
half of the loop (``grade``, ``delta_report``, ``train``, ``serve`` and
about forty more) is reachable at ``wai.simulations.*`` only, and the pages
write the front-door spelling the library does not have yet. That is a
design decision, not something a docs PR should paper over, so this is a
ratchet rather than a hard gate: the count may fall and never rise. Lower
``PIN`` when a page is fixed or a name joins the front door.

``docs/api/`` is generated from docstrings by ``scripts/gen_api_docs.py``;
fixing those means fixing the docstring, so they are counted separately and
not pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import whileai as wai

ROOT = Path(__file__).resolve().parents[2]

# Hand-written docs pages citing a ``wai.<path>`` that does not resolve.
# A ceiling, never a target. See the module docstring.
PIN = 220

# ``wai.<name>`` inside an illustrative signature, not a real call.
PLACEHOLDERS = {"name", "X", "data"}

_PATH = re.compile(r"\bwai\.([A-Za-z_][A-Za-z0-9_.]*)")


def _resolves(dotted: str) -> bool:
    obj: object = wai
    for part in dotted.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return False
    return True


def _hand_written_pages() -> list[Path]:
    docs = ROOT / "docs"
    pages = sorted(p for p in docs.rglob("*.md")) + sorted(p for p in docs.rglob("*.mdx"))
    return [p for p in pages if "api" not in p.relative_to(docs).parts[:1]]


def _misses(pages: list[Path]) -> list[str]:
    out: list[str] = []
    for page in pages:
        for i, line in enumerate(page.read_text(encoding="utf-8").split("\n"), 1):
            for match in _PATH.finditer(line):
                dotted = match.group(1).rstrip(".")
                if not dotted or dotted in PLACEHOLDERS:
                    continue
                if not _resolves(dotted):
                    out.append(f"{page.relative_to(ROOT)}:{i}: wai.{dotted}")
    return out


def test_the_four_paths_from_735_and_780_resolve() -> None:
    """The named sites, pinned so the specific bug cannot come back."""
    for dotted in (
        "compare",
        "simulations.recommend",
        "simulations.score.eval_power",
        "simulations.delta_report",
        "simulations.run_judge",
        "simulations.export_dataset",
        "noise_band",
    ):
        assert _resolves(dotted), f"docs spell wai.{dotted}; it has to resolve"

    for page, forbidden in (
        ("docs/reference/what-to-run.md", ("wai.score.eval_power", "wai.recommend(")),
        ("docs/reference/five-calls.md", ("import whileai.simulations as wai",)),
    ):
        text = (ROOT / page).read_text(encoding="utf-8")
        for bad in forbidden:
            assert bad not in text, f"{page} still writes {bad}"


def test_unresolvable_docs_paths_only_shrink() -> None:
    misses = _misses(_hand_written_pages())
    assert len(misses) <= PIN, (
        f"{len(misses)} hand-written docs citations of a wai.<path> that does not resolve, "
        f"pin is {PIN}. New ones:\n" + "\n".join(misses[:40])
    )
    if len(misses) < PIN:
        pytest.fail(
            f"{len(misses)} < pin {PIN}. Good: lower PIN to {len(misses)} in this PR "
            "so the ratchet holds the gain.",
            pytrace=False,
        )
