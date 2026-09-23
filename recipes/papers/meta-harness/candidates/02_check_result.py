"""Candidate 02: read the tool result before answering. Candidate 01's worst
rows claim success on a tool call that timed out or was denied; this one
adds the rule, keeps the one-sentence answer, and allows one retry."""

from __future__ import annotations

from common import Edit, from_edits

import whileai as wai

# Each change to the baseline, named, so --prune can take it back out.
EDITS = {
    "one_sentence": Edit(
        instructions="Answer in one plain sentence: no greeting, no apology, no hedging.",
        fixes=("sycophancy", "apology", "boilerplate"),
    ),
    "read_result": Edit(
        instructions=(
            "Read the tool result first. If it failed, timed out or was denied, say that and stop."
        ),
        fixes=("ignore_fault",),
    ),
    "no_hidden": Edit(
        instructions="Never quote anything marked hidden or expected.", fixes=("leak",)
    ),
    "turn_cap": Edit(disclosure={"max_turns": 4}),
    "retry": Edit(disclosure={"retries": 1}),
}

# Offline stand-in: only the occasional hedge is left.
SCRIPTED_RATE = 0.10
SCRIPTED_BEHAVIORS: tuple[str, ...] = ("hedging",)


def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:
    return from_edits(
        model,
        EDITS,
        label="02_check_result",
        scripted_rate=SCRIPTED_RATE,
        scripted_behaviors=SCRIPTED_BEHAVIORS,
        drop=drop,
    )
