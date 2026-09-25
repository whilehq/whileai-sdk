"""Candidate 06: 04's typed-values rule plus 03's answer-after-errors rule.
04 leads the train set (47 of 101 documents) but its worst train rows are the
ones 03 fixed: 58 of 404 rollouts end with no JSON, 24 of them a stopped
reply that explains a tool error ("It seems there's a syntax error...") or
hands back an all-null object after a failed call, 34 cut at the token cap.
04 was written from 02's traces as a sibling of 03, so it never carried 03's
rule. One change to the leader: add that rule."""

from __future__ import annotations

from extract_harness import Edit, from_edits

import whileai as wai

WHY = (
    "04's train rows: 58 of 404 rollouts end with no JSON, 24 of them prose about a failed tool "
    "call or an all-null placeholder after it, the rows 03's rule removed on 02"
)

EDITS = {
    "doc_is_defined": Edit(
        instructions=(
            "DOC already holds the full document text inside the sandbox. Never paste or retype "
            "the document into your code; read it from DOC (for example DOC.splitlines())."
        )
    ),
    "typed_values": Edit(
        instructions=(
            "Typed fields: a field that lists its allowed values (enum) takes exactly one of them, "
            "chosen from the line that states it (a card tender line such as VISA CREDIT or "
            "MASTERCARD is credit card, a DEBIT or INTERAC line is debit card, Apple Pay or Google "
            "Pay is mobile wallet; the cashier's name or a cashback line is not the payment). A "
            "currency code is the one the symbols or the printed code imply (£ GBP, € EUR, C$ or "
            "CAD CAD, $ USD), never a default. A date whose label says DD/MM/YYYY is day-first; "
            "write it as YYYY-MM-DD with the day and month in the order the label states."
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
    return from_edits(model, EDITS, label="06_typed_and_answer", drop=drop)
