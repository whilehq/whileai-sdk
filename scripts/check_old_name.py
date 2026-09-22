"""The old name stays out of new code.

ZeroProof became While on 2026-09-16 and the cutover finished on 2026-09-19:
the package is ``whileai``, the hosts are ``while.ai``, the variables
are ``WHILEAI_*``. What is left of the old name is wire protocol and
infrastructure that cannot change without breaking users (Modal app
hostnames, the ``zp_`` key prefix, ``zeroproof.*`` span attribute keys,
volume and table names) plus the history in ``CHANGELOG.md``.

This script pins today's count of the old name per file in
``scripts/old_name_baseline.json``. A file may lose mentions, never gain
them, and a file not in the baseline may not mention the old name at all.
``--update`` rewrites the baseline after a PR that removes mentions;
raising a count needs a sentence in the PR body saying why.

Run: ``uv run python scripts/check_old_name.py`` (CI runs it in lint).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "old_name_baseline.json"

# One regex for every spelling. Case-insensitive so "Zero Proof", "ZeroProof",
# "zeroproofai" and "ZEROPROOF_" all count.
OLD_NAME = re.compile(r"zero[ -]?proof|\bzps\b", re.IGNORECASE)

# History, the rename tool, and this check itself are not new code.
SKIP = (
    "CHANGELOG.md",
    "uv.lock",
    "scripts/rebrand.py",
    "scripts/check_old_name.py",
    "scripts/old_name_baseline.json",
)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def count(path: str) -> int:
    if path in SKIP:
        return 0
    try:
        text = (ROOT / path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0
    return len(OLD_NAME.findall(text))


def main(argv: list[str]) -> int:
    counts = {p: n for p in tracked_files() if (n := count(p))}
    if "--update" in argv:
        BASELINE.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline: {len(counts)} files, {sum(counts.values())} mentions")
        return 0
    baseline: dict[str, int] = json.loads(BASELINE.read_text(encoding="utf-8"))
    bad = []
    for path, n in sorted(counts.items()):
        allowed = baseline.get(path, 0)
        if n > allowed:
            bad.append(f"{path}: {n} mentions of the old name, baseline allows {allowed}")
    if bad:
        print("\n".join(bad))
        print(
            "\nThe package is whileai, the hosts are while.ai, the variables are"
            " WHILEAI_*. Use those. If the mention is wire protocol or infrastructure"
            " that cannot change, say so in the PR body and run"
            " `uv run python scripts/check_old_name.py --update`."
        )
        return 1
    fell = sum(baseline.get(p, 0) - counts.get(p, 0) for p in baseline)
    if fell:
        print(f"{fell} fewer mentions than the baseline; run with --update to lower it")
    print(f"ok: {sum(counts.values())} mentions of the old name in {len(counts)} files, none new")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
