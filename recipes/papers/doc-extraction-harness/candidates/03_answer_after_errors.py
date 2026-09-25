"""Candidate 03: answer after a tool error. On 02's train rows, 62 of 404
rollouts end with no JSON at all and another handful with an all-null
"placeholder" object: after the code errors (a bad regex, an undefined
function, empty output) the model writes prose about fixing the code, or
hands back nulls "because the function was not defined", instead of reading
the document it was given. 28 of the 62 are bank statements. One rule: a tool
failure is not an answer; read the document yourself and reply with the JSON
block, and null is never a placeholder for a value the document shows."""

from __future__ import annotations

from extract_harness import Edit, from_edits

import whileai as wai

WHY = (
    "02's train rows: 62 of 404 rollouts reply with prose about a failed tool call and no JSON "
    "(28 of them bank statements); others return an all-null placeholder object after an error"
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
}


def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:
    return from_edits(model, EDITS, label="03_answer_after_errors", drop=drop)
