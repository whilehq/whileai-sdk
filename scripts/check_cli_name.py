"""The command is ``wai``; docs and code never tell anyone to type ``whileai <command>``.

``wai`` and ``whileai`` run the same entry point. ``wai`` is the one we
write: three letters, one token, the same name as the import alias, and a
coding agent types it many times a session (2026-09-21). This check fails
on any tracked text file that spells a CLI invocation the long way, so a
new page, skill, recipe or error message cannot drift back. The package
name stays ``whileai`` in ``pip install whileai``, ``uv add whileai`` and
``import whileai``; those are not commands and do not match.

Run: ``uv run python scripts/check_cli_name.py`` (CI runs it in lint).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VERBS = (
    "login|signup|status|logout|agents|agent|runs|verdict|promote|live|keys|"
    "archive|init-evals|init|purge"
)
LONG_FORM = re.compile(rf"(?<![\w./-])whileai ({VERBS})\b")

SUFFIXES = {".py", ".md", ".mdx", ".toml", ".json", ".yml", ".yaml", ".txt"}

# History is not new code, and this check names the pattern it forbids.
SKIP = ("CHANGELOG.md", "uv.lock", "scripts/check_cli_name.py")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def main() -> int:
    bad: list[str] = []
    for path in tracked_files():
        if path in SKIP or Path(path).suffix not in SUFFIXES:
            continue
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = LONG_FORM.search(line)
            if m:
                bad.append(f"{path}:{i}: {m.group(0)!r} -> 'wai {m.group(1)}'")
    if bad:
        print("The command is `wai`. Spell these that way:")
        print("\n".join(bad))
        return 1
    print("check_cli_name: every CLI mention says wai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
