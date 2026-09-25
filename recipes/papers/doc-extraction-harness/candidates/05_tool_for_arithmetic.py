"""Candidate 05: the tool is for arithmetic, not for reading. On 03's train
rows the lowest-scoring rollouts are still the 49 of 404 with no JSON, and
34 of those are a first turn cut at 1,024 tokens: a ```python block that
opens with DOC = \"\"\" and retypes the document, rule or no rule (27 of them
bank statements, the longest documents). The model reaches for code to read
text it can already see. One rule on top of 03: the fields are read off the
document in the message directly; the tool is only for arithmetic the
document does not print (a sum, a date plus N days, a line count), and a
document that prints the number needs no code at all."""

from __future__ import annotations

from extract_harness import Edit, from_edits

import whileai as wai

WHY = (
    "03's train rows: 49 of 404 rollouts end with no JSON, 34 of them a first turn cut at the "
    "token cap that opens a triple-quoted DOC = and retypes the document (27 bank statements)"
)

EDITS = {
    "doc_is_defined": Edit(
        instructions=(
            "DOC already holds the full document text inside the sandbox. Never paste or retype "
            "the document into your code; read it from DOC (for example DOC.splitlines())."
        )
    ),
    "answer_after_errors": Edit(
        instructions=(
            "If the tool errors, prints nothing, or you have no calls left, do not explain, "
            "apologize or describe a fix: read the document text in the message yourself and reply "
            "with the ```json block. Every value the document shows goes in the block; null is only "
            "for a field the document truly does not give, never a placeholder for a value your "
            "code failed to extract."
        )
    ),
    "tool_for_arithmetic": Edit(
        instructions=(
            "Read the fields directly off the document in the message; you can see it, so no code "
            "is needed to find a name, a number, a date or an id. Use the tool only for arithmetic "
            "the document does not print (adding line amounts, a date plus N days, counting lines "
            "across pages), with a few short statements over DOC. If the document prints the "
            "number, copy it and skip the tool. A first reply that is the ```json block is the "
            "best reply."
        )
    ),
}


def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:
    return from_edits(model, EDITS, label="05_tool_for_arithmetic", drop=drop)
