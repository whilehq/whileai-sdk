"""Candidate 00: the baseline. The instructions as the team wrote them, no
turn cap, no retry. Every other candidate is measured against this one."""

from __future__ import annotations

from common import BASE_INSTRUCTIONS, build

import whileai as wai
from whileai.harness import Disclosure

INSTRUCTIONS = BASE_INSTRUCTIONS
DISCLOSURE = Disclosure()

# Offline stand-in: how often the scripted agent gets a rollout wrong, and in
# which ways (None: every planted kind). The live run ignores these two lines;
# a real model brings its own mistakes.
SCRIPTED_RATE = 0.45
SCRIPTED_BEHAVIORS: tuple[str, ...] | None = None


def harness(model: str) -> wai.Harness:
    return build(
        model,
        instructions=INSTRUCTIONS,
        label="00_baseline",
        disclosure=DISCLOSURE,
        scripted_rate=SCRIPTED_RATE,
        scripted_behaviors=SCRIPTED_BEHAVIORS,
    )
