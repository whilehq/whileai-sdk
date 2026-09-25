"""Candidate 07: one retry with the validator's reason. 05 leads the train
set (50 of 101 documents) and its lowest rows are still the 58 of 404 with
no JSON: 26 are a first turn cut at the cap, now a 1,024-token chain of
print(DOC.splitlines()[n]...) statements instead of a paste, and 32 stop on
prose ("Sure, I'll provide the JSON block... the error occurred because")
with no object in it. Rules about this are heard about six times in seven.
One loop change instead of another rule: a final reply with no parseable
JSON object, or one whose dates are not YYYY-MM-DD or whose money is not a
number, is sent back once with the reason (``extract_harness.agent``:
retries=1, validate=True); the reply to that is final."""

from __future__ import annotations

from extract_harness import Edit, from_edits

import whileai as wai

WHY = (
    "05's train rows: 58 of 404 rollouts end with no JSON (26 a first turn cut at the cap, 32 "
    "prose after a tool error); a rule reduced them, a retry with the reason is the loop's own fix"
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
    "retry_once": Edit(retries=1, validate=True),
}


def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:
    return from_edits(model, EDITS, label="07_retry_once", drop=drop)
