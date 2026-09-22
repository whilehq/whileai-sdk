"""Candidate 00: the baseline. Bare instructions, no skills text, no tool,
one turn. Every other candidate, and the `weights` arm, is measured against
this harness."""

from __future__ import annotations

from harnesses import build

import whileai as wai

SKILLS: str | None = None
TOOL = False

# Offline stand-in: how often the scripted agent solves a task under this
# harness. The live run ignores it; a real model brings its own rate.
SCRIPTED_RATE = 0.35


def harness(model: str) -> wai.Harness:
    return build(model, skills=SKILLS, tool=TOOL, label="00_baseline", scripted_rate=SCRIPTED_RATE)
