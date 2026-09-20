"""The version a program reads is the version a human reads (#612).

``wai.__version__`` comes back from the installed metadata, and PEP 440
normalizes that string: ``1.07`` in ``pyproject.toml`` installs as ``1.7``,
which no ``CHANGELOG.md`` heading ever says. So the counter is written in
its normal form everywhere (``0.99`` then ``0.100``), the top changelog
heading is the pyproject version, and the installed build reports exactly
that string. ``release.py --dry-run`` and the publish gate refuse a padded
or rolled-over counter before it can ship.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
HEADING = re.compile(r"^## (\d+\.\d+) \(\d{4}-\d{2}-\d{2}\)$", re.MULTILINE)


def _release_script():
    path = ROOT / ".github" / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("release_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pyproject_version() -> str:
    m = re.search(
        r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M
    )
    assert m, "pyproject.toml has no version line"
    return m.group(1)


def _top_changelog_heading() -> str:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    after_unreleased = text.split("## Unreleased\n", 1)[1]
    m = HEADING.search(after_unreleased)
    assert m, "CHANGELOG.md has no `## x.y (date)` heading under `## Unreleased`"
    return m.group(1)


def test_pyproject_version_is_its_own_normal_form() -> None:
    v = _pyproject_version()
    assert Version(v).public == v, f"pyproject version {v!r} normalizes to {Version(v).public!r}"
    # Belief 10: the counter is 0.N, never padded and never rolled over.
    assert _release_script().not_plain(v) is None, _release_script().not_plain(v)


def test_top_changelog_heading_is_the_pyproject_version() -> None:
    top = _top_changelog_heading()
    assert top == _pyproject_version()
    assert Version(top).public == top


def test_installed_build_reports_the_changelog_heading() -> None:
    try:
        installed = importlib.metadata.version("whileai")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("whileai is not installed; run `uv sync` first")
    import whileai as wai

    top = _top_changelog_heading()
    assert installed == Version(top).public == top, (
        f"pip reports {installed!r}; CHANGELOG.md says {top!r}. Re-sync (`uv sync`) if you just "
        f"changed the version; otherwise the counter is padded or rolled over"
    )
    assert wai.__version__ == installed


@pytest.mark.parametrize("version", ["0.99", "0.100", "0.1000", "0.9"])
def test_release_script_accepts_a_plain_counter(version: str) -> None:
    assert _release_script().not_plain(version) is None
    assert Version(version).public == version


@pytest.mark.parametrize(
    ("version", "why"),
    [
        ("1.07", "zero-padded"),
        ("0.099", "zero-padded"),
        ("1.0", "rolled the major"),
        ("1.10", "rolled the major"),
        ("0.99.1", "not MAJOR.N"),
        ("0.100rc1", "not MAJOR.N"),
    ],
)
def test_release_script_refuses_a_padded_or_rolled_counter(version: str, why: str) -> None:
    reason = _release_script().not_plain(version)
    assert reason and why in reason, (version, reason)


def test_release_script_refuses_to_cut_from_a_padded_version(tmp_path: Path) -> None:
    padded = tmp_path / "pyproject.toml"
    padded.write_text('[project]\nname = "whileai"\nversion = "1.07"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _release_script().read_version(padded)
    assert "zero-padded" in str(exc.value)


def test_next_version_never_rolls_over_or_pads() -> None:
    script = _release_script()
    assert script.next_version("0.99") == "0.100"
    assert script.next_version("0.109") == "0.110"
    assert script.next_version("0.999") == "0.1000"
