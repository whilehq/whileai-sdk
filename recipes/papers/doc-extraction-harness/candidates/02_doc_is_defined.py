"""Candidate 02: DOC is already defined. The baseline's worst train rows
paste the whole document into a triple-quoted DOC = \"\"\"...\"\"\" inside their
code: 414 of 475 tool calls did it, and every one of the 74 replies cut at
the 1,024-token cap was a paste that never reached the JSON. One sentence
tells the model the variable exists and must not be retyped."""

from __future__ import annotations

from extract_harness import Edit, from_edits

import whileai as wai

WHY = (
    "baseline train rows: 414 of 475 tool calls retype the document into the code, and all 74 "
    "truncated replies are such pastes"
)

EDITS = {
    "doc_is_defined": Edit(
        instructions=(
            "DOC already holds the full document text inside the sandbox. Never paste or retype "
            "the document into your code; read it from DOC (for example DOC.splitlines())."
        )
    ),
}


def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:
    return from_edits(model, EDITS, label="02_doc_is_defined", drop=drop)
