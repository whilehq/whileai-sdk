"""What a cached arm may carry: the rollouts, never the verdict.

A paper recipe caches each arm's rows the moment they land, so a crash in
the second arm never costs the first and ``--reuse`` can rebuild the delta
from disk. The rows carry ``reward``, so until this file existed the cache
carried the grader's verdict too, and a change to the verifier invalidated
nothing: whilehq/whileai-sdk#737 measured 402 of 5,760 cached
``zero-rl-format-reward`` rollouts changing verdict when ``MathEqual``
moved to Math-Verify (788c553), moving every arm 2 to 4 points, while
``--reuse`` reproduced the old numbers exactly. The run was repeatable and
wrong.

The rule, stated in ``recipes/papers/README.md``:

    a cached row is a rollout, and a verdict is not a rollout.

The rollouts are the expensive part and they stay good. The verdict is
cheap. So :func:`write` stamps the file with the grader that decided it and
the whileai the verifier came from, and :func:`read` recomputes every
verdict when that stamp is not the one in the tree, or refuses the reuse
and says what changed when it cannot recompute.

    from cache_stamp import StaleCache, read, write

    write(cache / "recipe.json", out, grader=GRADER)
    out, note = read(cache / "recipe.json", grader=GRADER, regrade=regrade)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

#: Where the grader stamp sits in a cache file, beside the rows rather than
#: inside them: the rows are the rollouts, this is who judged them.
STAMP_KEY = "grader_stamp"


class StaleCache(RuntimeError):
    """A cached file's verdicts came from a grader that is no longer here,
    and re-grading them was not possible. The rollouts may still be good;
    the numbers on them are not."""


def _whileai_version() -> str:
    try:
        return version("whileai")
    except PackageNotFoundError:  # a checkout with no metadata installed
        return "unknown"


def stamp(grader: str) -> dict[str, str]:
    """Who decided the verdicts in a cache file: the grader's own name and
    the whileai release its verifier came from. Both move the verdict, so
    both are part of the stamp (CONSTITUTION.md, belief 1: a number is a
    result only with the versions that produced it)."""
    return {"grader": grader, "whileai": _whileai_version()}


def describe(found: dict[str, str] | None, want: dict[str, str]) -> str:
    """One line naming what changed between the stamp on disk and the tree."""
    if not found:
        return (
            f"no grader stamp: written before the verdicts were stamped, "
            f"so it is not known whether {want['grader']} decided them"
        )
    parts = [
        f"{key} {found.get(key, '?')!r} -> {want[key]!r}"
        for key in want
        if found.get(key) != want[key]
    ]
    return "; ".join(parts) if parts else "stamp matches"


def write(path: Path, payload: dict[str, Any], *, grader: str) -> dict[str, Any]:
    """Write one arm's cache with the grader stamped beside the rows."""
    stamped = {**payload, STAMP_KEY: stamp(grader)}
    path.write_text(json.dumps(stamped), encoding="utf-8")
    return stamped


def read(
    path: Path,
    *,
    grader: str,
    regrade: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    """One arm back from ``path``, with its verdicts good for the grader in
    the tree.

    Returns ``(payload, note)``. ``note`` is empty when the stamp matched
    and the stored rewards were reused as they are; otherwise it says what
    changed and what re-grading moved.

    Raises :class:`StaleCache` when the stamp does not match and no
    ``regrade`` was given, or when ``regrade`` cannot decide a row. The
    refusal is the point: returning the old grader's verdicts silently is
    the bug this module exists for.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    found = payload.get(STAMP_KEY)
    want = stamp(grader)
    if found == want:
        return payload, ""
    what = describe(found if isinstance(found, dict) else None, want)
    if regrade is None:
        raise StaleCache(
            f"{path.name} was graded by a rule that is not in this tree ({what}). "
            f"A cached row is a rollout, and a verdict is not a rollout: re-grade "
            f"the stored rollouts, or delete {path.name} and roll them again."
        )
    fresh = regrade(payload)
    fresh[STAMP_KEY] = want
    path.write_text(json.dumps(fresh), encoding="utf-8")
    return fresh, f"re-graded ({what})"


def moved(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, int]:
    """How many verdicts a re-grade moved, and in which direction: the table
    #737 reports, so a recipe that re-grades prints it rather than asserting
    the rollouts were unaffected."""
    counts = {"rows": len(after), "unchanged": 0, "now_correct": 0, "now_wrong": 0}
    for old, new in zip(before, after):
        was, is_ = float(old.get("reward") or 0.0), float(new.get("reward") or 0.0)
        if was == is_:
            counts["unchanged"] += 1
        elif is_ > was:
            counts["now_correct"] += 1
        else:
            counts["now_wrong"] += 1
    counts["moved"] = counts["now_correct"] + counts["now_wrong"]
    return counts


__all__ = ["STAMP_KEY", "StaleCache", "describe", "moved", "read", "stamp", "write"]
