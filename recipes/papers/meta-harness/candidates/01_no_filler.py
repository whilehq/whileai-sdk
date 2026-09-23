"""Candidate 01: cut the filler. The baseline's worst rows open with a
greeting, an apology or a hedge before the answer; this candidate tells the
model to answer in one plain sentence and caps the loop at four turns."""

from __future__ import annotations

from common import Edit, from_edits

import whileai as wai

# Each change to the baseline, named, so --prune can take it back out.
EDITS = {
    "one_sentence": Edit(
        instructions="Answer in one plain sentence: no greeting, no apology, no hedging.",
        fixes=("sycophancy", "apology", "boilerplate"),
    ),
    "turn_cap": Edit(disclosure={"max_turns": 4}),
}

# Offline stand-in: the filler kinds are gone, the tool-result mistakes stay.
SCRIPTED_RATE = 0.35
SCRIPTED_BEHAVIORS: tuple[str, ...] = ("ignore_fault", "leak", "hedging")


def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:
    return from_edits(
        model,
        EDITS,
        label="01_no_filler",
        scripted_rate=SCRIPTED_RATE,
        scripted_behaviors=SCRIPTED_BEHAVIORS,
        drop=drop,
    )
