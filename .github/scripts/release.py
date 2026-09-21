"""Cut the next release from whatever is under ``## Unreleased``.

One command does the whole bump so two people (or two agents) cannot do it
two different ways:

    uv run python .github/scripts/release.py            # bump, relock
    uv run python .github/scripts/release.py --dry-run  # say what would happen

What it does, in order:

1. Reads the version from ``pyproject.toml`` and steps the counter by one
   (``0.80`` -> ``0.81``, ``0.99`` -> ``0.100``), the same rule
   ``check_version.py`` enforces at publish time.
2. In ``CHANGELOG.md`` renames ``## Unreleased`` to ``## <version> (<date>)``
   and puts a fresh, empty ``## Unreleased`` above it. That empty header is
   the whole fix for release collisions: a PR that adds its entry under
   ``## Unreleased`` still lands under ``## Unreleased`` when it merges a
   minute after a cut, instead of sliding under a version that shipped
   without it.
3. Runs ``uv lock`` so ``uv.lock`` agrees.

It refuses to cut when ``## Unreleased`` is missing or has no entries
(exit code 3), so a run with nothing to ship is a no-op, not an empty
release. It also refuses (exit code 1, ``--dry-run`` included) when the
version in ``pyproject.toml`` is not a plain ``0.N``: a zero-padded segment
(``1.07``) is ``1.7`` once PEP 440 normalizes it, so the string the package
reports would not be the string ``CHANGELOG.md`` uses (#612). The release workflow (``.github/workflows/release.yml``) runs
this on main and serializes concurrent runs; that workflow is how a
release is cut, not a hand edit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
UNRELEASED = "## Unreleased"
NOTHING_TO_SHIP = 3

VERSION_LINE = re.compile(r'^version = "(\d+)\.(\d+)"$', re.MULTILINE)


def next_version(current: str) -> str:
    """The counter goes up by one and never rolls over: 0.99 -> 0.100 -> 0.101.

    Mirrors ``check_version.next_allowed``. The major stays 0; no zero
    padding, because PEP 440 drops it (``1.07`` is ``1.7`` on PyPI, which is
    how 2026-09-20 shipped 1.0..1.9 instead of 0.100..0.109).
    """
    major, minor = (int(p) for p in current.split("."))
    return f"{major}.{minor + 1}"


def not_plain(version: str) -> str | None:
    """Why ``version`` is not a plain ``0.N`` counter, or None when it is.

    The string in ``pyproject.toml`` must survive PEP 440 unchanged, or
    ``wai.__version__`` (read back from the installed metadata) stops
    matching the ``CHANGELOG.md`` heading: ``1.07`` installs as ``1.7``.
    A two-part version with no zero padding is its own normal form, so
    the check needs no ``packaging`` import here; the test in
    ``tests/api/test_version_string.py`` cross-checks it against
    ``packaging.version.Version``.
    """
    parts = version.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return f"{version!r} is not MAJOR.N"
    padded = [p for p in parts if p != str(int(p))]
    if padded:
        return (
            f"{version!r} has a zero-padded segment ({', '.join(padded)}); PEP 440 reads it as "
            f"{'.'.join(str(int(p)) for p in parts)!r}, so the installed version would not be "
            f"the CHANGELOG.md heading. The counter is 0.N with no padding (0.99 then 0.100)."
        )
    if parts[0] != "0":
        return f"{version!r} rolled the major; the counter stays 0.N (0.99 then 0.100)"
    return None


def read_version(path: Path) -> str:
    m = VERSION_LINE.search(path.read_text(encoding="utf-8"))
    if not m:
        sys.exit(f'{path}: no `version = "0.N"` line')
    version = f"{m.group(1)}.{m.group(2)}"
    reason = not_plain(version)
    if reason:
        sys.exit(f"{path}: {reason}")
    return version


def unreleased_entries(text: str) -> str | None:
    """The body under ``## Unreleased``, or None when the header is missing."""
    start = text.find(f"{UNRELEASED}\n")
    if start < 0:
        return None
    body_start = start + len(UNRELEASED) + 1
    nxt = text.find("\n## ", body_start)
    return text[body_start : nxt + 1 if nxt >= 0 else len(text)]


def cut_changelog(text: str, version: str, today: str) -> str:
    body = unreleased_entries(text)
    if body is None:
        sys.exit(f"CHANGELOG.md has no `{UNRELEASED}` section; add one with the entries to ship")
    if not any(line.startswith("- ") for line in body.splitlines()):
        print(f"{UNRELEASED} has no entries; nothing to ship")
        sys.exit(NOTHING_TO_SHIP)
    header = f"## {version} ({today})"
    return text.replace(f"{UNRELEASED}\n", f"{UNRELEASED}\n\n{header}\n", 1)


def bump(path: Path, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    text, n = VERSION_LINE.subn(f'version = "{version}"', text, count=1)
    if n != 1:
        sys.exit(f"{path}: no version line to bump")
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    ap.add_argument("--no-lock", action="store_true", help="skip `uv lock` (tests)")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="header date (tests)")
    args = ap.parse_args()

    current = read_version(PYPROJECT)
    version = next_version(current)
    changelog = cut_changelog(CHANGELOG.read_text(encoding="utf-8"), version, args.date)
    entries = unreleased_entries(CHANGELOG.read_text(encoding="utf-8")) or ""
    shipped = [line[2:60] for line in entries.splitlines() if line.startswith("- ")]

    print(f"{current} -> {version} with {len(shipped)} entr{'y' if len(shipped) == 1 else 'ies'}:")
    for line in shipped:
        print(f"  - {line}")
    if args.dry_run:
        return 0

    CHANGELOG.write_text(changelog, encoding="utf-8", newline="\n")
    bump(PYPROJECT, version)
    if not args.no_lock:
        subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    print(f"version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
