"""Candidate 01, the placebo. The baseline's instructions reworded sentence
by sentence, with no rule added or removed: same tool, same format, same
turn cap. Any gap between this and 00 is what rewording alone does, and a
later candidate has to beat that too (manage-experiments; meta-harness)."""

from __future__ import annotations

from extract_harness import build

import whileai as wai

WHY = "control: measures what a rewording with no new rule does"

INSTRUCTIONS = """detailed thinking off
Your job is to pull structured fields out of business documents. Each request states the kind of document, lists the fields wanted with a one-line description of each, and gives the document's text; that text can carry OCR mistakes.

A Python tool is available. To call it, send a reply that is a single ```python code block and nothing more. The code runs in a sandbox (Python 3 standard library only, network off, 10 seconds at most) with the document's text in the string variable DOC, and anything it prints is returned to you in the next message. You can call the tool at most 3 times.

To finish, send one ```json block holding a single JSON object keyed by exactly the field names asked for. Dates go in YYYY-MM-DD form and amounts of money as bare numbers (such as 1234.5). Put null for any field the document does not provide."""


def harness(model: str) -> wai.Harness:
    return build(model, instructions=INSTRUCTIONS, label="01_placebo")
