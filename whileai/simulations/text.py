"""Reasoning markup, split from the words a model actually said.

Stdlib only, no HTTP and no threads, so the scoring side can read a
reply's spoken text without importing the agent runtime.
"""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>\s*", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.S | re.I)


def split_reasoning(text: str) -> tuple[str, int, bool]:
    """A thinking model's output as ``(spoken, closed_blocks, unclosed)``.

    Closed ``<think>...</think>`` blocks are removed whole; the ones with
    words in them are counted (an empty ``<think> </think>``, what a
    reasoning-suppressed model prints, is markup and not reasoning). An
    unclosed ``<think>`` means the token cap landed inside the reasoning,
    so everything from it to the end is dropped and ``unclosed`` is True:
    the writer was cut off, and what follows the tag is not a reply.
    """
    raw = str(text or "")
    closed = 0

    def _drop(match: re.Match[str]) -> str:
        nonlocal closed
        if match.group(1).strip():
            closed += 1
        return ""

    spoken = _THINK_BLOCK.sub(_drop, raw)
    spoken, opened = _THINK_OPEN.subn("", spoken)
    return spoken, closed, bool(opened)


__all__ = ["split_reasoning"]
