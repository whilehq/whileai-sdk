"""Cut the next release from whatever is under ``## Unreleased``.

One command does the whole bump so two people (or two agents) cannot do it
two different ways:

    uv run python .github/scripts/release.py            # bump, relock
    uv run python .github/scripts/release.py --dry-run  # say what would happen

What it does, in order:

1. Reads the version from ``pyproject.toml`` and steps the counter by one
   (``0.80`` -> ``0.81``, ``0.99`` -> ``1.00``), the same rule
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
release. The release workflow (``.github/workflows/release.yml``) runs
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
    how 2026-09-20 shipped 1.0..1.8 instead of 0.100..0.108).
    """
    major, minor = (int(p) for p in current.split("."))
    return f"{major}.{minor + 1}"


def read_version(path: Path) -> str:
    m = VERSION_LINE.search(path.read_text(encoding="utf-8"))
    if not m:
        sys.exit(f'{path}: no `version = "0.N"` line')
    return f"{m.group(1)}.{m.group(2)}"


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
