"""Candidate 00, the starter harness (v0). A plain, reasonable extraction
prompt: what the job is, how to use the code tool (the document is the
variable DOC), the output format (one JSON object, ISO dates, plain numbers,
null for a field the document does not give). Four turns, no retry, no
validator. Nothing is held back to make the climb look better."""

from __future__ import annotations

from extract_harness import BASE_INSTRUCTIONS, build

import whileai as wai

WHY = "the starting point every later candidate is scored against"

INSTRUCTIONS = BASE_INSTRUCTIONS


def harness(model: str) -> wai.Harness:
    return build(model, instructions=INSTRUCTIONS, label="00_baseline")
