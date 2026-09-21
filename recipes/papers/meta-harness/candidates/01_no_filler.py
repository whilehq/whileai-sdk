"""Candidate 01: cut the filler. The baseline's worst rows open with a
greeting, an apology or a hedge before the answer; this candidate tells the
model to answer in one plain sentence and caps the loop at four turns."""

from __future__ import annotations

from common import BASE_INSTRUCTIONS, build

import whileai as wai
from whileai.harness import Disclosure

INSTRUCTIONS = (
    BASE_INSTRUCTIONS + " Answer in one plain sentence: no greeting, no apology, no hedging."
)
DISCLOSURE = Disclosure(max_turns=4)

# Offline stand-in: the filler kinds are gone, the tool-result mistakes stay.
SCRIPTED_RATE = 0.35
SCRIPTED_BEHAVIORS: tuple[str, ...] | None = ("ignore_fault", "leak", "hedging")


def harness(model: str) -> wai.Harness:
    return build(
        model,
        instructions=INSTRUCTIONS,
        label="01_no_filler",
        disclosure=DISCLOSURE,
        scripted_rate=SCRIPTED_RATE,
        scripted_behaviors=SCRIPTED_BEHAVIORS,
    )
