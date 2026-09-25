"""Candidate 04: typed values. Written from 02's train rows while 03 scored
(sibling of 03, on 02's edits). Among 02's parsed train replies the field
wrong most often is receipt.payment_method, 63 of 77: the tender line says
"VISA DEBIT XXXX9965" or "VISA CREDIT ****4896" and the reply says "cash",
"CASHBACK" or "CASHIER" (words from other lines), never one of the five
options the field lists; invoice.currency is wrong 21 of 74, "USD" for a
document priced in £ or €; day-first dates ("Invoice Date (DD/MM/YYYY):
06/02/2025") come back as the wrong month. One rule about typed fields:
an enum field takes one of its listed values read off the matching line, a
code field the code its symbols imply, a date the order its label states."""

from __future__ import annotations

from extract_harness import Edit, from_edits

import whileai as wai

WHY = (
    "02's train rows: receipt.payment_method wrong in 63 of 77 parsed replies (cash/CASHBACK/"
    "CASHIER for a VISA DEBIT or VISA CREDIT tender line), invoice.currency USD for GBP/EUR/CAD "
    "in 21 of 74, day-first dates read month-first"
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
}


def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:
    return from_edits(model, EDITS, label="04_typed_values", drop=drop)
