"""The alias is the mark: `import whileai as wai` is the only import shape.

"while" is heard as "whale", the mark is wai the whale, and the alias is
how a reader, a search engine and an answer engine find this package
(``docs/reference/style.md`` rule 1). This file pins the places that say
so, the way ``test_style_ratchet.py`` pins the retired shapes: the
booleans hold, and ``LEGACY_IMPORTS`` is a ceiling that falls as the
examples migrate and never rises.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import whileai

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")

# The definition sentence, reused verbatim wherever a person or a machine
# asks what wai is.
DEFINITION = "wai is While's whale and the alias of the whileai SDK: `import whileai as wai`."

# The ownership sentence. It lives in the first README paragraph once, and
# nowhere in code: a docstring carries mechanism and a citation, never a
# thesis (CONSTITUTION.md belief 6).
OWNERSHIP = (
    "You own the model, the data and the weights: the datasets are built from your "
    "production traces, the model is an open model post-trained with SFT and RL, and "
    "the trained weights are yours to download and serve anywhere."
)

# Import shapes that are not the alias. A line that is exactly ``import
# whileai``, a ``from whileai import x``, and the legacy engine path bound
# to the alias name.
LEGACY_PATTERNS = (
    re.compile(r"^import whileai$", re.M),
    re.compile(r"from whileai import ", re.M),
    re.compile(r"import whileai\.simulations as wai", re.M),
)

# Today's count over the scanned trees, less the exemptions below. Lower it
# when a PR migrates examples; never raise it. The leverage is `whileai/`: two
# generators write the legacy shape into the user's own repo
# (`init_repo.py`'s AGENTS.md block and `templates/evals.py`'s run.py), so
# their coding agent reads it forever. See the Style log, #444.
LEGACY_IMPORTS = 98

SCANNED = ("README.md", "whileai", "recipes", "skills", "docs")
SUFFIXES = {".py", ".md", ".mdx"}
# The standard has to spell the shapes it retires, the way this file does.
EXEMPT = ("docs/api/", "docs/reference/style.md")


def _scanned_files() -> list[Path]:
    out: list[Path] = []
    for name in SCANNED:
        path = ROOT / name
        if path.is_file():
            out.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if child.suffix not in SUFFIXES or not child.is_file():
                continue
            rel = child.relative_to(ROOT).as_posix()
            if rel.startswith(EXEMPT):  # generated pages, and the guide itself
                continue
            out.append(child)
    return out


def _legacy_hits() -> list[str]:
    hits: list[str] = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8")
        for pattern in LEGACY_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{line}")
    return sorted(hits)


def test_one_import_shape_ratchet() -> None:
    """Rule 1: `import whileai as wai` is the only shape an example shows."""
    hits = _legacy_hits()
    assert len(hits) <= LEGACY_IMPORTS, (
        f"{len(hits)} imports are not `import whileai as wai`, ceiling is "
        f"{LEGACY_IMPORTS}: {hits[:10]}"
    )
    if len(hits) < LEGACY_IMPORTS:
        pytest.fail(
            f"{len(hits)} < ceiling {LEGACY_IMPORTS}. Good: lower LEGACY_IMPORTS to "
            f"{len(hits)} in this PR so the gain holds.",
            pytrace=False,
        )


def test_readme_never_shows_another_shape() -> None:
    """The front page is the one document every reader sees."""
    assert [h for h in _legacy_hits() if h.startswith("README.md")] == []


def test_banner_alt_names_the_mark() -> None:
    """The alt text is what a reader without images, and a crawler, gets."""
    assert 'alt="wai, the While whale. Models improve while they work."' in README


def _flat(text: str) -> str:
    """One line, so a sentence matches however markdown wrapped it."""
    return " ".join(text.split())


def test_first_paragraph_defines_the_alias_and_the_ownership() -> None:
    """The first prose paragraph, which is the PyPI page's opening too."""
    first = README.split("- **Train.**", 1)[1].split("\n## ", 1)[0].strip()
    first = _flat(first.split("\n\n", 1)[1])  # past the last loop bullet
    assert first.startswith(DEFINITION), first[:120]
    assert _flat(README).count(OWNERSHIP) == 1, "the ownership sentence, verbatim, once"
    assert OWNERSHIP in first


def test_the_thesis_never_enters_code() -> None:
    """A docstring carries mechanism and a citation, never a thesis."""
    for path in sorted((ROOT / "whileai").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "trained weights are yours" not in text, path


def test_module_docstring_opens_with_the_package_and_uses_the_alias() -> None:
    doc = whileai.__doc__ or ""
    assert doc.lstrip().startswith("whileai:"), doc[:60]
    assert "import whileai as wai" in doc
    assert "wai." in doc.split("import whileai as wai", 1)[1]


def test_packaging_metadata_carries_the_alias() -> None:
    """What PyPI and an answer engine read when nobody clicks through."""
    if sys.version_info < (3, 11):
        pytest.skip("tomllib is 3.11+")
    import tomllib

    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert meta["description"].startswith("whileai (import whileai as wai):")
    for keyword in ("wai", "whileai", "post-training", "open-weights", "byok"):
        assert keyword in meta["keywords"], keyword


def test_nothing_else_is_named_wai() -> None:
    """The alias is lowercase and belongs to the import, not to a module,
    a class, a CLI command or a flag."""
    for path in sorted((ROOT / "whileai").rglob("*.py")):
        assert path.stem != "wai", path
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(Wai|WAI)\b", text), path
        assert not re.search(r"^(class|def) wai\b", text, re.M), path
