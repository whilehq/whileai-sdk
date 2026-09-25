"""Meta-Harness on document extraction: Nemotron-Nano-8B with a sandboxed Python tool.

    python recipe.py --selftest                                   # offline: generator, grader, tool, loop
    python recipe.py --model "vllm:nvidia/Llama-3.1-Nemotron-Nano-8B-v1@$URL"   # both arms on the holdout

Shape (recipes/papers/_template), with a harness in place of a trained arm:
  1. data():      200 generated business documents with their gold records, split 101/99 by a
                  hash of the ask id; the test version is printed so a run can prove it saw the
                  same holdout (Lambert 2025, chapter Evaluation)
  2. evaluate():  a harness on the holdout, k rollouts per document, graded by field F1
                  (SROIE/CORD; ANLS on names); the baseline three times for the noise floor
  3. arms:        "baseline" is the starting harness; "recipe" is the harness the Meta-Harness
                  search found (Lee et al. 2026, arXiv:2603.28052; the loop is
                  recipes/papers/meta-harness). No weights change: steps are 0.
  4. results.json with the paired delta and the checks the README table reads

Serve the model with vLLM on any GPU (the run below used one Modal L40S, vLLM 0.10.0,
transformers 4.55.4) and pass its OpenAI-compatible URL; VLLM_API_KEY is its key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
METRIC = "field F1"
BOOK = "Evaluation"
EVAL_RUNS = 3  # baseline re-runs that set the noise floor


# ---------------------------------------------------------------- the documents

# ---------------------------------------------------------------- the schemas

# Field -> (kind, description the model sees). The descriptions are part of
# the task, not of the harness: every candidate sends the same ones.
SCHEMAS: dict[str, dict[str, tuple[str, str]]] = {
    "invoice": {
        "invoice_number": ("id", "the invoice number as printed"),
        "invoice_date": ("date", "the date the invoice was issued"),
        "due_date": (
            "date",
            "the payment due date; if only payment terms are given (e.g. Net 30), "
            "the invoice date plus that many days; null if neither is given",
        ),
        "vendor_name": ("name", "the company that issued the invoice"),
        "customer_name": ("name", "the company billed (Bill To)"),
        "subtotal": ("money", "sum of the line items before tax and shipping"),
        "tax": ("money", "the tax amount; null if no tax is charged"),
        "total": ("money", "the invoice total, before any payment already made is subtracted"),
        "currency": ("code", "ISO 4217 code: USD, EUR, GBP or CAD"),
    },
    "receipt": {
        "merchant_name": ("name", "the store or restaurant"),
        "date": ("date", "the date of the purchase"),
        "subtotal": ("money", "the amount before tax and tip; null if not printed"),
        "tax": ("money", "the tax amount"),
        "tip": ("money", "the tip or gratuity; null if none"),
        "total": ("money", "the amount charged, including tax and tip"),
        "payment_method": (
            "enum",
            "one of: cash, credit card, debit card, gift card, mobile wallet",
        ),
        "card_last4": ("digits", "the last four digits of the card; null if not a card"),
    },
    "purchase_order": {
        "po_number": ("id", "the purchase order number"),
        "order_date": ("date", "the date the order was placed"),
        "delivery_date": ("date", "the requested delivery date; null if not given"),
        "buyer_name": ("name", "the company placing the order"),
        "supplier_name": ("name", "the company the order is sent to (vendor or supplier)"),
        "line_item_count": ("int", "how many line items the order has, across all pages"),
        "total": (
            "money",
            "the order total including shipping and tax; if not printed, the sum of the line "
            "amounts plus any shipping and tax shown",
        ),
        "currency": ("code", "ISO 4217 code: USD, EUR, GBP or CAD"),
    },
    "bank_statement": {
        "account_holder": ("name", "the account holder's name"),
        "account_last4": ("digits", "the last four digits of the account number"),
        "period_start": ("date", "first day of the statement period"),
        "period_end": ("date", "last day of the statement period"),
        "opening_balance": ("money", "balance at the start of the period"),
        "closing_balance": (
            "money",
            "balance at the end of the period; if not printed, opening balance plus deposits "
            "minus withdrawals",
        ),
        "total_deposits": ("money", "sum of every deposit (credit) in the period"),
        "total_withdrawals": (
            "money",
            "sum of every withdrawal (debit) in the period, as a positive number",
        ),
    },
    "claim_form": {
        "claimant_name": ("name", "the claimant's full name as First Last"),
        "date_of_birth": ("date", "the claimant's date of birth"),
        "policy_number": ("id", "the insurance policy number"),
        "incident_date": ("date", "the date of the incident"),
        "claim_amount": ("money", "the amount claimed"),
        "phone": ("phone", "the claimant's phone number"),
        "email": ("email", "the claimant's email; null if not given"),
    },
}

DOC_TYPES = tuple(SCHEMAS)
FIELDS = [(t, f) for t in DOC_TYPES for f in SCHEMAS[t]]

# ---------------------------------------------------------------- vocabularies

COMPANIES = [
    "Northwind Traders",
    "Blue Heron Logistics",
    "Castell & Moore LLP",
    "Keystone Fabrication",
    "Aurora Dental Group",
    "Pinecrest Hardware",
    "Meridian Office Supply",
    "Halvorsen Marine",
    "Brightline Analytics",
    "Oak & Ember Catering",
    "Quantum Circuits GmbH",
    "Saltmarsh Bakery",
    "Redwood Veterinary Clinic",
    "Tidewater Plumbing Co.",
    "Vantage Print Works",
    "Lindqvist AB",
    "Harbor Point Realty",
    "Crescent Moon Studio",
    "Granite Peak Outfitters",
    "Fairfield Electric",
    "Marlowe Textiles Ltd",
    "Sable Creek Farms",
    "Ironwood Consulting",
    "Beaumont Freight",
    "Cobalt Software Inc.",
    "Delacroix Imports",
    "Evergreen Medical Supply",
    "Foxglove Florists",
    "Juniper Legal Services",
    "Kestrel Aviation Parts",
    "Lakeshore Furniture",
    "Montague & Sons",
]
EU_COMPANIES = [
    "Quantum Circuits GmbH",
    "Lindqvist AB",
    "Delacroix Imports",
    "Brauer Maschinenbau GmbH",
    "Vermeer Logistiek BV",
    "Rossi Forniture S.r.l.",
    "Kowalski Budownictwo",
]
UK_COMPANIES = ["Marlowe Textiles Ltd", "Castell & Moore LLP", "Pemberton Joinery Ltd"]
MERCHANTS = [
    "Corner Grocer",
    "Blue Bottle Diner",
    "Hilltop Pharmacy",
    "Main St. Hardware",
    "Sunrise Cafe",
    "Green Leaf Market",
    "The Rusty Anchor",
    "QuickFuel #214",
    "Paper & Ink",
    "Luigi's Trattoria",
    "Harvest Bistro",
    "City Books",
    "Petal Pushers",
    "Midtown Deli",
    "Summit Sports",
]
FIRST = [
    "Maria",
    "James",
    "Aisha",
    "Wei",
    "Olivia",
    "Diego",
    "Priya",
    "Noah",
    "Fatima",
    "Liam",
    "Sofia",
    "Kenji",
    "Amara",
    "Lucas",
    "Hannah",
    "Omar",
    "Elena",
    "Samuel",
    "Nadia",
    "Theo",
]
LAST = [
    "Chen",
    "Okafor",
    "Martinez",
    "Nguyen",
    "Schmidt",
    "Patel",
    "O'Brien",
    "Kowalski",
    "Haddad",
    "Johansson",
    "Rossi",
    "Tanaka",
    "Mensah",
    "Dubois",
    "Fischer",
    "Ramirez",
    "Kim",
    "Novak",
]
ITEMS = [
    "Consulting hours",
    "A4 copy paper (box)",
    "Toner cartridge",
    "Steel bracket 40mm",
    "Hydraulic hose 2m",
    "Annual license",
    "Installation labor",
    "Safety gloves (pair)",
    "LED panel 60x60",
    "Oak shelving unit",
    "Cable ties (100)",
    "Freight handling",
    "Design review",
    "Server rack rail",
    "Printer maintenance",
    "Copper pipe 15mm",
    "Office chair",
    "Laptop stand",
    "Cleaning service",
    "Training session",
]
RECEIPT_ITEMS = [
    "COFFEE LG",
    "BAGEL",
    "MILK 2%",
    "BANANAS",
    "BREAD WHT",
    "EGGS DOZ",
    "SALMON FILLET",
    "PASTA",
    "HAMMER 16OZ",
    "SCREWS BOX",
    "ASPIRIN 100",
    "NOTEBOOK",
    "PEN BLK 3PK",
    "BURGER",
    "FRIES",
    "IPA DRAFT",
    "SALAD",
    "SOUP DAY",
    "TEA",
    "CROISSANT",
]
DEPOSITS = [
    "PAYROLL ACME CORP",
    "TRANSFER FROM SAVINGS",
    "MOBILE DEPOSIT",
    "REFUND AMAZON",
    "INTEREST PAID",
    "ZELLE FROM J SMITH",
    "VENMO CASHOUT",
]
WITHDRAWALS = [
    "RENT PAYMENT",
    "UTILITY ELECTRIC",
    "GROCERY STORE",
    "ATM WITHDRAWAL",
    "CARD PURCHASE GAS",
    "ONLINE TRANSFER",
    "INSURANCE PREMIUM",
    "PHONE BILL",
    "RESTAURANT",
    "STREAMING SUB",
    "CHECK #1042",
    "PHARMACY",
]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# ---------------------------------------------------------------- formatting


def cents(x: float) -> float:
    return round(x + 1e-9, 2)


def fmt_money(x: float, style: str, cur: str = "USD") -> str:
    """One printed spelling of an amount. ``eu`` swaps the separators."""
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "C$"}[cur]
    whole = f"{abs(x):,.2f}"
    if style == "eu":
        whole = whole.replace(",", "X").replace(".", ",").replace("X", ".")
    elif style == "space":
        whole = whole.replace(",", " ")
    elif style == "plain":
        whole = f"{abs(x):.2f}"
    neg = x < 0
    if style == "eu":
        s = f"{whole} {sym}"
    elif style == "code":
        s = f"{cur} {whole}"
    elif style in ("plain", "bare"):
        s = whole
    else:
        s = f"{sym}{whole}"
    return f"({s})" if neg and style == "paren" else (f"-{s}" if neg else s)


def fmt_date(d: date, style: str) -> str:
    m = MONTHS[d.month - 1]
    if style == "iso":
        return d.isoformat()
    if style == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "us_short":
        return f"{d.month}/{d.day}/{d.year % 100:02d}"
    if style == "long":
        return f"{m} {d.day}, {d.year}"
    if style == "long_ord":
        suf = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
        return f"{m} {d.day}{suf}, {d.year}"
    if style == "dmy_text":
        return f"{d.day:02d} {m[:3]} {d.year}"
    if style == "eu_dot":
        return f"{d.day:02d}.{d.month:02d}.{d.year}"
    if style == "eu_slash":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    raise ValueError(style)


OCR_SUBS = [
    ("O", "0"),
    ("o", "0"),
    ("l", "1"),
    ("I", "l"),
    ("S", "5"),
    ("e", "c"),
    ("m", "rn"),
    ("B", "8"),
    ("a", "o"),
    ("t", "f"),
]


def ocr(text: str, rng: random.Random, rate: float) -> str:
    """OCR-style damage to a label or boilerplate: character swaps, a dropped
    or doubled space. Gold values are never passed through here."""
    if rate <= 0:
        return text
    out = []
    for ch in text:
        r = rng.random()
        if r < rate:
            for a, b in OCR_SUBS:
                if ch == a:
                    ch = b
                    break
        elif r < rate * 1.3 and ch == " ":
            ch = "" if rng.random() < 0.5 else "  "
        out.append(ch)
    return "".join(out)


def rand_date(rng: random.Random, lo: date = date(2025, 1, 1), days: int = 600) -> date:
    return lo + timedelta(days=rng.randrange(days))


def person(rng: random.Random) -> tuple[str, str]:
    return rng.choice(FIRST), rng.choice(LAST)


def page_break(n: int, total: int, header: str) -> str:
    return f"\n\f---------------- Page {n} of {total} ----------------\n{header}\n"


# ---------------------------------------------------------------- generators


def _line_items(rng: random.Random, n: int) -> list[tuple[str, int, float, float]]:
    rows = []
    for name in rng.sample(ITEMS, n):
        qty = rng.choice([1, 1, 1, 2, 3, 4, 5, 10, 12, 20])
        unit = cents(rng.choice([rng.uniform(3, 60), rng.uniform(60, 400), rng.uniform(400, 2500)]))
        rows.append((name, qty, unit, cents(qty * unit)))
    return rows


def gen_invoice(rng: random.Random) -> tuple[str, dict[str, Any]]:
    region = rng.choices(["us", "eu", "uk", "ca"], [5, 2, 1, 1])[0]
    cur = {"us": "USD", "eu": "EUR", "uk": "GBP", "ca": "CAD"}[region]
    vendor = rng.choice(
        EU_COMPANIES if region == "eu" else UK_COMPANIES if region == "uk" else COMPANIES
    )
    customer = rng.choice([c for c in COMPANIES if c != vendor])
    num = rng.choice(
        [
            f"INV-{rng.randint(2025, 2026)}-{rng.randint(1, 9999):04d}",
            f"{rng.randint(10000, 99999)}",
            f"A{rng.randint(100, 999)}-{rng.randint(10, 99)}",
        ]
    )
    issued = rand_date(rng)
    terms = rng.choice([None, 15, 30, 45, 60])
    due_mode = rng.choices(["printed", "terms", "none"], [5, 3, 1])[0]
    due = issued + timedelta(days=terms or 30) if due_mode != "none" else None
    items = _line_items(rng, rng.randint(1, 6))
    subtotal = cents(sum(i[3] for i in items))
    rate = rng.choice([0.0, 0.05, 0.07, 0.0825, 0.19, 0.2]) if rng.random() < 0.85 else 0.0
    tax = cents(subtotal * rate) if rate else None
    shipping = cents(rng.uniform(8, 90)) if rng.random() < 0.3 else 0.0
    total = cents(subtotal + (tax or 0) + shipping)
    paid = cents(total * rng.choice([0.25, 0.5])) if rng.random() < 0.3 else 0.0
    prev = cents(rng.uniform(50, 3000)) if rng.random() < 0.2 else 0.0
    mstyle = "eu" if region == "eu" else rng.choice(["us", "code", "bare", "space"])
    dstyle = (
        "eu_dot"
        if region == "eu"
        else "eu_slash"
        if region == "uk"
        else rng.choice(["us", "long", "dmy_text", "iso", "us_short"])
    )
    noise = rng.choice([0.0, 0.02, 0.05])

    def L(s: str) -> str:
        return ocr(s, rng, noise)

    def M(x: float) -> str:
        return fmt_money(x, mstyle, cur)

    def D(d: date) -> str:
        return fmt_date(d, dstyle)

    date_hint = " (DD/MM/YYYY)" if dstyle == "eu_slash" else ""
    due_line = (
        f"{L('Due Date')}{date_hint}: {D(due)}"
        if due_mode == "printed" and due
        else f"{L('Terms')}: Net {terms or 30}"
        if due_mode == "terms"
        else f"{L('Terms')}: Due on receipt"
    )
    if due_mode == "none":
        due = None
    ship_date = issued + timedelta(days=rng.randint(1, 9))
    layout = rng.choice(["classic", "table", "letter", "two_col"])
    lines: list[str] = []
    if layout == "letter":
        lines += [
            vendor.upper(),
            f"{rng.randint(10, 999)} {rng.choice(['Harbor', 'Main', 'Elm'])} Street",
            f"{L('Phone')}: ({rng.randint(200, 989)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
            "",
            f"{D(issued)}",
            "",
            f"To: {customer}",
            "",
            f"Re: {L('Invoice')} {num}",
            "",
            f"Dear {customer} accounts team,",
            "",
            f"Please find below our charges for services delivered through {D(ship_date)}.",
        ]
        for name, qty, unit, amt in items:
            lines.append(f"  - {name}: {qty} x {M(unit)} = {M(amt)}")
        lines.append(
            f"The subtotal comes to {M(subtotal)}"
            + (f", plus tax of {M(tax)}" if tax else "")
            + (f" and shipping of {M(shipping)}" if shipping else "")
            + f", for a total of {M(total)}."
        )
        if paid:
            lines.append(
                f"We have received a deposit of {M(paid)}; the balance due is {M(total - paid)}."
            )
        lines.append(due_line + ".")
        lines += ["", "Kind regards,", f"{rng.choice(FIRST)} {rng.choice(LAST)}", vendor]
    else:
        head = [
            f"{vendor}",
            L("INVOICE")
            if layout != "two_col"
            else f"{L('INVOICE')}{' ' * 30}{L('Invoice #')}: {num}",
        ]
        lines += head
        if layout != "two_col":
            lines.append(f"{L('Invoice No.')}: {num}")
        lines.append(f"{L('Invoice Date')}{date_hint}: {D(issued)}")
        lines.append(f"{L('Ship Date')}: {D(ship_date)}")
        lines.append(due_line)
        if rng.random() < 0.4:
            lines.append(f"{L('Customer PO')}: PO-{rng.randint(1000, 99999)}")
        if layout == "two_col":
            lines += [
                "",
                f"{L('Bill To')}:{' ' * 26}{L('Remit To')}:",
                f"{customer:<35}{vendor}",
                f"{rng.randint(1, 999)} Commerce Way{' ' * 17}PO Box {rng.randint(100, 9999)}",
            ]
        else:
            lines += [
                "",
                f"{L('Bill To')}:",
                customer,
                f"{rng.randint(1, 999)} Commerce Way",
                "",
                f"{L('Ship To')}:",
                rng.choice(COMPANIES) if rng.random() < 0.3 else customer,
            ]
        lines.append("")
        if layout == "table":
            lines.append(
                f"| {L('Description'):<24} | {L('Qty'):>4} | {L('Unit Price'):>12} | {L('Amount'):>12} |"
            )
            lines.append("|" + "-" * 26 + "|------|" + "-" * 14 + "|" + "-" * 14 + "|")
            for name, qty, unit, amt in items:
                lines.append(f"| {name:<24} | {qty:>4} | {M(unit):>12} | {M(amt):>12} |")
        else:
            lines.append(f"{L('Item'):<26}{L('Qty'):>5}{L('Rate'):>14}{L('Amount'):>14}")
            for name, qty, unit, amt in items:
                lines.append(f"{name:<26}{qty:>5}{M(unit):>14}{M(amt):>14}")
        lines.append("")
        lines.append(f"{L('Subtotal'):>45}: {M(subtotal)}")
        if tax:
            lines.append(f"{L('Tax')} ({rate * 100:g}%):".rjust(46) + f" {M(tax)}")
        if shipping:
            lines.append(f"{L('Shipping'):>45}: {M(shipping)}")
        if prev:
            lines.append(f"{L('Previous Balance'):>45}: {M(prev)}")
        lines.append(f"{L('TOTAL'):>45}: {M(total)}")
        if paid:
            lines.append(f"{L('Amount Paid'):>45}: {M(paid)}")
            lines.append(f"{L('Balance Due'):>45}: {M(total - paid + prev)}")
        elif prev:
            lines.append(f"{L('Total Amount Due'):>45}: {M(total + prev)}")
        lines += [
            "",
            L("Thank you for your business. Please include the invoice number with payment."),
        ]
    gold = {
        "invoice_number": num,
        "invoice_date": issued.isoformat(),
        "due_date": due.isoformat() if due else None,
        "vendor_name": vendor,
        "customer_name": customer,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "currency": cur,
    }
    return "\n".join(lines), gold


def gen_receipt(rng: random.Random) -> tuple[str, dict[str, Any]]:
    merchant = rng.choice(MERCHANTS)
    when = rand_date(rng)
    items = [
        (n, cents(rng.uniform(0.99, 24.99))) for n in rng.sample(RECEIPT_ITEMS, rng.randint(1, 8))
    ]
    subtotal = cents(sum(p for _, p in items))
    tax = cents(subtotal * rng.choice([0.04, 0.06, 0.0725, 0.0875, 0.1]))
    restaurant = any(
        w in merchant for w in ("Diner", "Trattoria", "Bistro", "Anchor", "Cafe", "Deli")
    )
    tip = (
        cents(subtotal * rng.choice([0.15, 0.18, 0.2]))
        if restaurant and rng.random() < 0.7
        else None
    )
    total = cents(subtotal + tax + (tip or 0))
    method = rng.choices(
        ["credit card", "debit card", "cash", "gift card", "mobile wallet"], [5, 3, 2, 1, 1]
    )[0]
    last4 = (
        f"{rng.randint(0, 9999):04d}"
        if method in ("credit card", "debit card", "gift card")
        else None
    )
    noise = rng.choice([0.03, 0.06, 0.1])

    def L(s: str) -> str:
        return ocr(s, rng, noise)

    dstyle = rng.choice(["us", "us_short", "dmy_text", "iso"])
    W = 34
    lines = [
        merchant.upper().center(W),
        ocr(f"{rng.randint(10, 999)} MAIN ST", rng, noise).center(W),
        f"TEL {rng.randint(200, 989)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}".center(W),
        "",
        f"{fmt_date(when, dstyle)}  {rng.randint(7, 22):02d}:{rng.randint(0, 59):02d}   {L('TRANS')} #{rng.randint(1000, 99999)}",
        f"{L('CASHIER')}: {rng.choice(FIRST).upper()}    {L('REG')} {rng.randint(1, 9)}",
        "-" * W,
    ]
    for name, price in items:
        lines.append(f"{L(name):<24}{price:>10.2f}")
    lines.append("-" * W)
    print_sub = rng.random() < 0.8
    if print_sub:
        lines.append(f"{L('SUBTOTAL'):<24}{subtotal:>10.2f}")
    lines.append(f"{L('TAX'):<24}{tax:>10.2f}")
    if tip is not None:
        if print_sub:
            lines.append(f"{L('TIP'):<24}{tip:>10.2f}")
        else:
            lines.append(f"{L('GRATUITY'):<24}{tip:>10.2f}")
    lines.append(f"{L('TOTAL'):<24}{total:>10.2f}")
    if method == "cash":
        tendered = float(((int(total) // 20) + 1) * 20)
        lines += [
            f"{L('CASH TEND'):<24}{tendered:>10.2f}",
            f"{L('CHANGE DUE'):<24}{tendered - total:>10.2f}",
        ]
    elif method == "credit card":
        brand = rng.choice(["VISA", "MASTERCARD", "AMEX", "VISA CREDIT"])
        lines += [
            f"{brand} ************{last4}",
            f"{L('AUTH CODE')} {rng.randint(100000, 999999)}",
            f"{L('AMOUNT')} {total:.2f}",
        ]
    elif method == "debit card":
        lines += [
            f"{rng.choice(['DEBIT', 'VISA DEBIT', 'INTERAC DEBIT'])} XXXX{last4}",
            f"{L('CASHBACK')} 0.00",
            f"{L('AMOUNT')} {total:.2f}",
        ]
    elif method == "gift card":
        lines += [f"{L('GIFT CARD')} ...{last4}", f"{L('REMAINING BAL')} {rng.uniform(1, 80):.2f}"]
    else:
        lines += [
            f"{rng.choice(['APPLE PAY', 'GOOGLE PAY'])} {L('CONTACTLESS')}",
            f"{L('AMOUNT')} {total:.2f}",
        ]
    if not print_sub and tip is None:
        pass
    lines += [
        "",
        L("ITEMS SOLD") + f" {len(items)}",
        L("THANK YOU - COME AGAIN").center(W),
        L(f"RETURNS WITHIN {rng.choice([14, 30])} DAYS W/ RECEIPT").center(W),
    ]
    if rng.random() < 0.4:
        lines.append(L(f"You saved $ {rng.uniform(0.5, 9):.2f} today!").center(W))
    gold = {
        "merchant_name": merchant,
        "date": when.isoformat(),
        "subtotal": subtotal if print_sub else None,
        "tax": tax,
        "tip": tip,
        "total": total,
        "payment_method": method,
        "card_last4": last4,
    }
    return "\n".join(lines), gold


def gen_purchase_order(rng: random.Random) -> tuple[str, dict[str, Any]]:
    region = rng.choices(["us", "eu", "uk", "ca"], [5, 2, 1, 1])[0]
    cur = {"us": "USD", "eu": "EUR", "uk": "GBP", "ca": "CAD"}[region]
    buyer = rng.choice(COMPANIES)
    supplier = rng.choice(
        [c for c in (EU_COMPANIES if region == "eu" else COMPANIES) if c != buyer]
    )
    po = rng.choice(
        [
            f"PO-{rng.randint(10000, 99999)}",
            f"{rng.randint(4500000000, 4509999999)}",
            f"PO{rng.randint(2025, 2026)}/{rng.randint(100, 999)}",
        ]
    )
    ordered = rand_date(rng)
    delivery = ordered + timedelta(days=rng.randint(5, 45)) if rng.random() < 0.7 else None
    n_items = rng.choice([2, 3, 4, 6, 8, 11, 14, 18, 23])
    items = []
    for i in range(n_items):  # noqa: B007
        name = rng.choice(ITEMS)
        qty = rng.choice([1, 2, 5, 10, 25, 50, 100])
        unit = cents(rng.uniform(1.5, 300))
        items.append(
            (
                f"{rng.choice('ABCDEFGH')}{rng.randint(1000, 9999)}",
                name,
                qty,
                unit,
                cents(qty * unit),
            )
        )
    lines_total = cents(sum(i[4] for i in items))
    shipping = cents(rng.uniform(15, 250)) if rng.random() < 0.5 else 0.0
    tax = cents(lines_total * rng.choice([0.05, 0.08, 0.2])) if rng.random() < 0.4 else 0.0
    total = cents(lines_total + shipping + tax)
    print_total = rng.random() < 0.75
    mstyle = "eu" if region == "eu" else rng.choice(["us", "code", "bare"])
    dstyle = (
        "eu_dot"
        if region == "eu"
        else "eu_slash"
        if region == "uk"
        else rng.choice(["us", "long", "iso", "dmy_text"])
    )
    noise = rng.choice([0.0, 0.03, 0.06])

    def L(s: str) -> str:
        return ocr(s, rng, noise)

    def M(x: float) -> str:
        return fmt_money(x, mstyle, cur)

    hint = " (DD/MM/YYYY)" if dstyle == "eu_slash" else ""
    header = f"{L('PURCHASE ORDER')}  {po}"
    head = [
        buyer,
        f"{rng.randint(1, 999)} Industrial Pkwy",
        "",
        header,
        f"{L('Order Date')}{hint}: {fmt_date(ordered, dstyle)}",
    ]
    if delivery:
        head.append(f"{L('Requested Delivery')}: {fmt_date(delivery, dstyle)}")
    head.append(
        f"{L('Quote Ref')}: Q-{rng.randint(1000, 9999)}   {L('Buyer Contact')}: {' '.join(person(rng))}"
    )
    head += [
        "",
        f"{L('Vendor')}:",
        supplier,
        f"{L('Ship To')}:",
        buyer + " Receiving Dock " + str(rng.randint(1, 9)),
        "",
    ]
    col = f"{L('Line'):<5}{L('Part #'):<9}{L('Description'):<24}{L('Qty'):>5}{L('Unit'):>12}{L('Ext. Price'):>14}"
    per_page = 9
    pages = (n_items + per_page - 1) // per_page
    lines = head + [col]  # noqa: RUF005
    for i, (part, name, qty, unit, amt) in enumerate(items):
        if i and i % per_page == 0:
            page_sum = cents(sum(x[4] for x in items[i - per_page : i]))
            lines.append(f"{'':<38}{L('Page subtotal')}: {M(page_sum)}")
            lines.append(L("Continued on next page"))
            lines.append(page_break(i // per_page + 1, pages, header).rstrip("\n"))
            lines.append(col)
        lines.append(f"{i + 1:<5}{part:<9}{name[:23]:<24}{qty:>5}{M(unit):>12}{M(amt):>14}")
    lines.append("")
    if print_total:
        lines.append(f"{L('Lines total'):>50}: {M(lines_total)}")
    if shipping:
        lines.append(f"{L('Shipping'):>50}: {M(shipping)}")
    if tax:
        lines.append(f"{L('Tax'):>50}: {M(tax)}")
    if print_total:
        lines.append(f"{L('PO TOTAL'):>50}: {M(total)}")
    lines += [
        "",
        L("Terms: Net 30. Please confirm receipt of this order within 2 business days."),
        f"{L('Authorized by')}: {' '.join(person(rng))}",
    ]
    gold = {
        "po_number": po,
        "order_date": ordered.isoformat(),
        "delivery_date": delivery.isoformat() if delivery else None,
        "buyer_name": buyer,
        "supplier_name": supplier,
        "line_item_count": n_items,
        "total": total,
        "currency": cur,
    }
    return "\n".join(lines), gold


def gen_bank_statement(rng: random.Random) -> tuple[str, dict[str, Any]]:
    first, last = person(rng)
    holder = f"{first} {last}"
    acct = f"{rng.randint(0, 9999):04d}"
    y, m = rng.choice([2025, 2026]), rng.randint(1, 12)
    start = date(y, m, 1)
    end = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    opening = cents(rng.uniform(200, 12000))
    n = rng.randint(8, 30)
    txns = []
    bal = opening
    days = sorted(rng.randrange((end - start).days + 1) for _ in range(n))
    for d in days:
        if rng.random() < 0.3:
            amt = cents(rng.uniform(20, 3500))
            desc = rng.choice(DEPOSITS)
        else:
            amt = -cents(rng.uniform(4, 900))
            desc = rng.choice(WITHDRAWALS)
        bal = cents(bal + amt)
        txns.append((start + timedelta(days=d), desc, amt, bal))
    deposits = cents(sum(a for _, _, a, _ in txns if a > 0))
    withdrawals = cents(-sum(a for _, _, a, _ in txns if a < 0))
    closing = cents(opening + deposits - withdrawals)
    print_closing = rng.random() < 0.7
    print_totals = rng.random() < 0.4
    dstyle = rng.choice(["us", "dmy_text", "long", "iso"])
    tstyle = rng.choice(["us_short", "dmy_text", "iso"]) if dstyle != "long" else "us"
    noise = rng.choice([0.0, 0.03, 0.05])
    layout = rng.choice(["columns", "signed"])

    def L(s: str) -> str:
        return ocr(s, rng, noise)

    bank = rng.choice(
        ["FIRST HARBOR BANK", "CIVIC CREDIT UNION", "SUMMIT NATIONAL BANK", "PRAIRIE SAVINGS"]
    )
    header = f"{bank}  |  {L('Account ending')} {acct}"
    lines = [
        bank,
        L("Statement of Account"),
        "",
        holder.upper() if rng.random() < 0.5 else holder,
        f"{rng.randint(10, 9999)} {rng.choice(['Maple', 'Cedar', 'Lake'])} Ave",
        "",
        f"{L('Account Number')}: XXXX-XXXX-{acct}"
        if rng.random() < 0.5
        else f"{L('Checking account ending in')} {acct}",
        f"{L('Statement Period')}: {fmt_date(start, dstyle)} {rng.choice(['-', 'to', 'through'])} {fmt_date(end, dstyle)}",
        f"{L('Routing')}: {rng.randint(100000000, 999999999)}",
        "",
        f"{L('Opening Balance')}: {fmt_money(opening, 'us')}",
    ]
    if print_totals:
        lines += [
            f"{L('Total Deposits and Credits')}: {fmt_money(deposits, 'us')}",
            f"{L('Total Withdrawals and Debits')}: {fmt_money(withdrawals, 'us')}",
        ]
    if print_closing:
        lines.append(f"{L('Closing Balance')}: {fmt_money(closing, 'us')}")
    lines.append(
        f"{L('Interest rate')}: 0.{rng.randint(1, 9)}0% APY   {L('Overdraft limit')}: {fmt_money(500, 'us')}"
    )
    lines.append("")
    if layout == "columns":
        col = f"{L('Date'):<12}{L('Description'):<26}{L('Withdrawals'):>13}{L('Deposits'):>12}{L('Balance'):>12}"
    else:
        col = f"{L('Date'):<12}{L('Description'):<26}{L('Amount'):>13}{L('Balance'):>12}"
    lines.append(col)
    per_page = 12
    pages = (n + per_page - 1) // per_page
    for i, (d, desc, amt, b) in enumerate(txns):
        if i and i % per_page == 0:
            lines.append(L("Continued on next page"))
            lines.append(page_break(i // per_page + 1, pages, header).rstrip("\n"))
            lines.append(col)
        ds = fmt_date(d, tstyle)
        if layout == "columns":
            w = f"{-amt:,.2f}" if amt < 0 else ""
            dp = f"{amt:,.2f}" if amt > 0 else ""
            lines.append(f"{ds:<12}{desc:<26}{w:>13}{dp:>12}{b:>12,.2f}")
        else:
            lines.append(f"{ds:<12}{desc:<26}{amt:>13,.2f}{b:>12,.2f}")
    lines += [
        "",
        L("Pending transactions are not included in this statement."),
        f"{L('Available balance as of today')}: {fmt_money(cents(closing - rng.uniform(0, 150)), 'us')}",
    ]
    gold = {
        "account_holder": holder,
        "account_last4": acct,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "opening_balance": opening,
        "closing_balance": closing,
        "total_deposits": deposits,
        "total_withdrawals": withdrawals,
    }
    return "\n".join(lines), gold


def gen_claim_form(rng: random.Random) -> tuple[str, dict[str, Any]]:
    first, last = person(rng)
    dob = date(rng.randint(1950, 2004), rng.randint(1, 12), rng.randint(1, 28))
    incident = rand_date(rng)
    policy = rng.choice(
        [
            f"POL-{rng.randint(100000, 999999)}",
            f"HX{rng.randint(1000000, 9999999)}",
            f"{rng.randint(10, 99)}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}",
        ]
    )
    amount = cents(rng.choice([rng.uniform(80, 900), rng.uniform(900, 15000)]))
    area, ex, sub = rng.randint(201, 989), rng.randint(200, 999), rng.randint(1000, 9999)
    phone = f"{area}{ex}{sub}"
    email = (
        f"{first.lower()}.{last.lower().replace(chr(39), '')}@{rng.choice(['mail.com', 'example.org', 'inbox.net'])}"
        if rng.random() < 0.75
        else None
    )
    noise = rng.choice([0.02, 0.05, 0.08])
    dstyle = rng.choice(["us", "long", "dmy_text", "iso"])

    def L(s: str) -> str:
        return ocr(s, rng, noise)

    name_style = rng.choice(["last_first", "first_last", "split"])
    pstyle = rng.choice(
        [
            f"({area}) {ex}-{sub}",
            f"{area}-{ex}-{sub}",
            f"+1 {area} {ex} {sub}",
            f"{area}.{ex}.{sub}",
        ]
    )
    lines = [L("PROPERTY & CASUALTY CLAIM FORM"), L("Section A - Claimant Information"), ""]
    if name_style == "last_first":
        lines.append(f"{L('Name (Last, First)')}: __{last.upper()}, {first.upper()}__")
    elif name_style == "split":
        lines += [f"{L('First name')}: {first}", f"{L('Last name')}: {last}"]
    else:
        lines.append(f"{L('Full name')}: {first} {last}")
    lines += [
        f"{L('Date of Birth')}: {fmt_date(dob, dstyle)}",
        f"{L('Policy No.')}: {policy}      {L('Group No.')}: G-{rng.randint(100, 999)}",
        f"{L('Daytime phone')}: {pstyle}",
        f"{L('Alt. phone')}: ____________"
        if rng.random() < 0.5
        else f"{L('Agent phone')}: ({rng.randint(201, 989)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
        f"{L('Email')}: {email}" if email else f"{L('Email')}: ______________",
        "",
        L("Section B - Incident"),
        f"{L('Date of incident')}: {fmt_date(incident, dstyle)}",
        f"{L('Date reported')}: {fmt_date(incident + timedelta(days=rng.randint(1, 20)), dstyle)}",
        f"{L('Type')}: [x] {rng.choice(['Water damage', 'Theft', 'Collision', 'Fire', 'Storm'])}   [ ] Other",
        f"{L('Description')}: "
        + L(
            rng.choice(
                [
                    "Pipe burst in kitchen, flooring and cabinets damaged.",
                    "Bicycle and laptop taken from garage overnight.",
                    "Rear-ended at a stop light; bumper and tail lamp replaced.",
                    "Hail damaged the roof and two skylights.",
                ]
            )
        ),
        "",
        L("Section C - Amounts"),
        f"{L('Estimated repair cost')}: {fmt_money(cents(amount * rng.uniform(1.0, 1.4)), 'us')}",
        f"{L('Deductible')}: {fmt_money(rng.choice([250, 500, 1000]), 'us')}",
        f"{L('Total amount claimed')}: {fmt_money(amount, 'us')}",
        "",
        f"{L('Signature')}: ___{first[0]}. {last}___   {L('Date')}: {fmt_date(incident + timedelta(days=rng.randint(1, 30)), dstyle)}",
    ]
    gold = {
        "claimant_name": f"{first} {last}",
        "date_of_birth": dob.isoformat(),
        "policy_number": policy,
        "incident_date": incident.isoformat(),
        "claim_amount": amount,
        "phone": phone,
        "email": email,
    }
    return "\n".join(lines), gold


GENERATORS = {
    "invoice": gen_invoice,
    "receipt": gen_receipt,
    "purchase_order": gen_purchase_order,
    "bank_statement": gen_bank_statement,
    "claim_form": gen_claim_form,
}

# ---------------------------------------------------------------- the pool and the split

POOL_PER_TYPE = 40  # 200 asks, the Meta-Harness run's --budget 200 at --holdout 0.5
DATA_SEED = 20260925


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def split_of(ask_id: str) -> str:
    """Half the asks choose candidates, half decide: a hash of the id, never its position."""
    return "selection" if _h("split:" + ask_id) % 2 == 0 else "holdout"


def build(seed: int = DATA_SEED) -> list[dict[str, Any]]:
    """Every ask: id, type, split, the document text, the gold record."""
    out = []
    for t in DOC_TYPES:
        for i in range(POOL_PER_TYPE):
            ask_id = f"{t}-{i:03d}"
            rng = random.Random(_h(f"{seed}:{ask_id}"))
            text, gold = GENERATORS[t](rng)
            assert set(gold) == set(SCHEMAS[t]), (t, set(gold) ^ set(SCHEMAS[t]))
            out.append(
                {"id": ask_id, "doc_type": t, "split": split_of(ask_id), "text": text, "gold": gold}
            )
    return out


def test_version(asks: list[dict[str, Any]]) -> str:
    """The frozen set's name is its content: ids, texts and gold together."""
    blob = "\n".join(
        json.dumps([a["id"], a["text"], a["gold"]], sort_keys=True)
        for a in sorted(asks, key=lambda a: a["id"])
    )
    return "t-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------- normalization and grading

_MONTH = {m.lower()[:3]: i + 1 for i, m in enumerate(MONTHS)}


def norm_money(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "n/a"):
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[^\d.,]", "", s)
    if not s:
        return None
    if "," in s and "." in s:
        s = (
            s.replace(".", "").replace(",", ".")
            if s.rfind(",") > s.rfind(".")
            else s.replace(",", "")
        )
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = head.replace(",", "") + "." + tail if len(tail) == 2 else s.replace(",", "")
    try:
        x = float(s)
    except ValueError:
        return None
    return round(-x if neg else x, 2)


def norm_date(v: Any) -> str | None:
    """ISO only, or a spelling with the month as a word: a numeric date that
    is not ISO is ambiguous and is not guessed back into shape."""
    if v is None:
        return None
    s = str(v).strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?", s)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        m = re.fullmatch(r"([A-Za-z]{3,9})\.? (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})", s)
        if m and m[1].lower()[:3] in _MONTH:
            return date(int(m[3]), _MONTH[m[1].lower()[:3]], int(m[2])).isoformat()
        m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]{3,9})\.?,? (\d{4})", s)
        if m and m[2].lower()[:3] in _MONTH:
            return date(int(m[3]), _MONTH[m[2].lower()[:3]], int(m[1])).isoformat()
    except ValueError:
        return None
    return None


def norm_text(v: Any) -> str | None:
    if v is None:
        return None
    s = re.sub(r"[^\w\s@&]", " ", str(v).casefold())
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def norm_id(v: Any) -> str | None:
    if v is None:
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", str(v)).upper()
    return s or None


def norm_digits(v: Any, keep: int | None = None) -> str | None:
    if v is None:
        return None
    s = re.sub(r"\D", "", str(v))
    if keep:
        s = s[-keep:]
    return s or None


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a"))


def field_ok(kind: str, pred: Any, gold: Any) -> bool:
    """One field, normalized: null only matches null."""
    if gold is None:
        return _is_null(pred)
    if _is_null(pred):
        return False
    if kind == "money":
        p = norm_money(pred)
        return p is not None and abs(p - float(gold)) < 0.005
    if kind == "date":
        return norm_date(pred) == gold
    if kind == "name":
        return norm_text(pred) == norm_text(gold)
    if kind in ("id",):
        return norm_id(pred) == norm_id(gold)
    if kind == "digits":
        return norm_digits(pred) == gold
    if kind == "int":
        try:
            return int(float(str(pred).strip())) == int(gold)
        except ValueError:
            return False
    if kind == "phone":
        return norm_digits(pred, keep=10) == gold
    if kind in ("code", "enum", "email"):
        return norm_text(pred) == norm_text(gold)
    raise ValueError(kind)


def parse_answer(text: str) -> tuple[dict[str, Any] | None, str]:
    """The last JSON object in a reply, fenced or bare, and why not when none."""
    if not text:
        return None, "empty reply"
    # code is not an answer: a dict inside a ```python block does not count
    text = re.sub(r"```(?:python|py)\s*\n.*?(?:```|$)", "", text, flags=re.S)
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    candidates = [b for b in reversed(blocks) if b.strip().startswith("{")]
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for b in candidates:
        try:
            obj = json.loads(b)
            if isinstance(obj, dict):
                return obj, "ok"
        except json.JSONDecodeError:
            continue
    dec = json.JSONDecoder()
    for i in starts:
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict) and len(obj) > 1:
                return obj, "ok"
        except json.JSONDecodeError:
            continue
    return None, "no JSON object in the final reply"


# ANLS_THRESHOLD = 0.5: a name scored below this similarity counts 0, the
# DocVQA convention (Biten et al. 2019, arXiv:1907.00490).
ANLS_THRESHOLD = 0.5


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: Any, gold: Any) -> float:
    """Normalized Levenshtein similarity on case-folded, punctuation-free
    text, 0 below ``ANLS_THRESHOLD`` (Biten et al. 2019)."""
    p, g = norm_text(pred) or "", norm_text(gold) or ""
    if not p and not g:
        return 1.0
    sim = 1.0 - _lev(p, g) / max(len(p), len(g))
    return sim if sim >= ANLS_THRESHOLD else 0.0


def field_score(kind: str, pred: Any, gold: Any) -> float:
    """One non-null prediction against a non-null gold: ANLS for names,
    exact after normalization for every other type."""
    if kind == "name":
        return anls(pred, gold)
    return 1.0 if field_ok(kind, pred, gold) else 0.0


def grade(ask: dict[str, Any], answer: dict[str, Any] | None) -> dict[str, Any]:
    """Field-level precision, recall and F1, the SROIE (Huang et al. 2019)
    and CORD (Park et al. 2019) convention: a non-null prediction that
    matches a non-null gold is a true positive; a non-null prediction that is
    wrong or where the gold is null is a false positive; a non-null gold
    predicted null or wrong is a false negative (a wrong value is both). A
    name scores its ANLS s in [0, 1] and adds s to TP and 1 - s to FP and FN.
    A correct null is in neither and is reported as ``null_acc``. The
    headline marker is ``field_f1`` (1 when there is nothing to find and
    nothing was predicted). ``field_acc`` (share of fields exactly right,
    nulls included) is kept for continuity; ``reward`` is every field right,
    a conjunction, never the headline."""
    schema = SCHEMAS[ask["doc_type"]]
    ans = {str(k).strip(): v for k, v in (answer or {}).items()}
    gold = ask["gold"]
    per: dict[str, float] = {}
    wrong = []
    tp = fp = fn = 0.0
    nulls, nulls_ok, names = 0, 0, []
    for fname, (kind, _) in schema.items():
        pred, g = ans.get(fname), gold[fname]
        exact = answer is not None and field_ok(kind, pred, g)
        per[fname] = 1.0 if exact else 0.0
        if not exact:
            wrong.append(fname)
        pnull = answer is None or _is_null(pred)
        if g is None:
            nulls += 1
            if pnull:
                nulls_ok += 1
            else:
                fp += 1
            continue
        if pnull:
            fn += 1
            if kind == "name":
                names.append(0.0)
            continue
        sc = field_score(kind, pred, g)
        if kind == "name":
            names.append(sc)
        tp += sc
        fp += 1 - sc
        fn += 1 - sc
    acc = sum(per.values()) / len(per)
    f1 = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    markers = {
        "field_f1": f1,
        "field_precision": tp / (tp + fp) if tp + fp else 1.0,
        "field_recall": tp / (tp + fn) if tp + fn else 1.0,
        "field_acc": acc,
        **{f"{ask['doc_type']}.{k}": v for k, v in per.items()},
    }
    if nulls:
        markers["null_acc"] = nulls_ok / nulls
    if names:
        markers["anls_names"] = sum(names) / len(names)
    return {
        "reward": 1.0 if not wrong else 0.0,  # doc_exact, the conjunctive one
        "field_acc": acc,
        "field_f1": f1,
        "markers": markers,
        "wrong": wrong,
        "parsed": answer is not None,
    }


def task_prompt(ask: dict[str, Any]) -> str:
    """The user turn: the schema and the document. Same for every candidate."""
    schema = SCHEMAS[ask["doc_type"]]
    fields = "\n".join(f"- {k} ({kind}): {desc}" for k, (kind, desc) in schema.items())
    return (
        f"Document type: {ask['doc_type'].replace('_', ' ')}\n"
        f"Fields to extract:\n{fields}\n\n"
        f"Document:\n<<<\n{ask['text']}\n>>>"
    )


# ---------------------------------------------------------------- difficulty

_LABEL_WORDS = (
    "Invoice",
    "Date",
    "Total",
    "TOTAL",
    "Subtotal",
    "SUBTOTAL",
    "Balance",
    "Tax",
    "TAX",
    "Policy",
    "Statement",
    "Deposits",
    "Withdrawals",
    "Amount",
    "Vendor",
    "Order",
)


def knobs(ask: dict[str, Any]) -> dict[str, int]:
    """The generator's difficulty knobs, read back off the document and its
    gold: OCR damage, distractor amounts, missing fields, fields the schema
    says to compute, and layouts that spread a field (multi-page, letter,
    last-name-first, day-first dates)."""
    text, gold, t = ask["text"], ask["gold"], ask["doc_type"]
    clean_labels = sum(text.count(w) for w in _LABEL_WORDS)
    damaged = len(
        re.findall(r"\b(?:T0TAL|5UBTOTAL|lnvoice|[A-Za-z]*[0-9][A-Za-z]{2,}[a-z])\b", text)
    )
    ocr = 2 if damaged >= 4 else 1 if damaged >= 1 else 0
    distractors = sum(
        w in text
        for w in (
            "Balance Due",
            "Previous Balance",
            "Total Amount Due",
            "Amount Paid",
            "Page subtotal",
            "CHANGE DUE",
            "Agent phone",
            "Available balance",
            "Estimated repair",
            "deposit of",
            "Ship To",
            "Quote Ref",
        )
    )
    missing = sum(v is None for v in gold.values())
    computed = 0
    if t == "invoice" and gold["due_date"] and "Net " in text and "Due Date" not in text:
        computed += 1
    if t == "bank_statement":
        computed += ("Total Deposits" not in text) * 2 + ("Closing Balance" not in text)
    if t == "purchase_order":
        computed += 1 + ("PO TOTAL" not in text)  # the line count is always counted
    layout = (
        int("Page 2 of" in text)
        + int("Dear " in text)
        + int(", " in (gold.get("claimant_name") or "") or "(Last, First)" in text)
    )
    layout += int(bool(re.search(r"\b\d{2}\.\d{2}\.\d{4}\b|DD/MM/YYYY", text)))
    return {
        "ocr": ocr,
        "distractors": distractors,
        "missing": missing,
        "computed": computed,
        "layout": layout,
        "_clean_labels": clean_labels,
    }


def difficulty(ask: dict[str, Any]) -> str:
    """easy, medium or hard from the knobs' sum (thresholds: convention,
    untested; chosen so each level holds a sizeable share of the pool)."""
    k = knobs(ask)
    score = k["ocr"] + min(k["distractors"], 3) + min(k["missing"], 2) + k["computed"] + k["layout"]
    return "easy" if score <= 2 else "medium" if score <= 4 else "hard"


# ---------------------------------------------------------------- the harness

# TOOL_TIMEOUT_S = 10: wall clock for one run of the model's code; the
# harness-and-weights recipe's run_python uses the same (convention, untested).
TOOL_TIMEOUT_S = 10.0
# TOOL_CHARS = 2000: the tool output is cut here so a print loop cannot fill
# the context; the message says when it was cut (convention, untested).
TOOL_CHARS = 2000
# SAMPLING: the Nemotron-Nano-8B-v1 model card's recommended temperature 0.6
# and top_p 0.95; 1024 reply tokens a turn. The same for every candidate and
# model, so sampling is not a lever in this search.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 1024}
# REQUEST_TIMEOUT_S = 240: one reply is at most 1024 tokens, well under a
# minute at the slowest rate seen; a request that hangs past this is retried
# instead of holding a round-synchronous batch (and the GPU) idle.
REQUEST_TIMEOUT_S = 240
# SALT: moves every sampling seed, for the three-run noise floor (DOCX_SALT).
SALT = int(os.environ.get("DOCX_SALT", "0"))

RUN_PYTHON = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run Python 3 (standard library, no network, 10 s) with the document text as the "
            "string DOC; returns what it printed."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}

Chat = Callable[[list[dict[str, str]], int], tuple[str, dict[str, Any]]]

# The baseline's instructions (candidates/00_baseline.py). A candidate
# written as named edits appends each edit's text to these.
BASE_INSTRUCTIONS = """detailed thinking off
You extract structured data from business documents. The user gives you the document type, the fields to extract with a short description of each, and the document text, which may contain OCR errors.

You have a Python tool. To use it, reply with exactly one ```python code block and nothing else. It runs in a sandbox (Python 3 standard library, no network, 10 second limit) where the document text is the string variable DOC, and whatever it prints comes back to you as the next message. You may use the tool up to 3 times.

When you are done, reply with one ```json block containing a single JSON object whose keys are exactly the requested field names. Write dates as YYYY-MM-DD and money as a plain number (for example 1234.5). Use null for a field the document does not give."""

# ---------------------------------------------------------------- the code tool

_PRELUDE = """
import socket as _socket
def _no_network(*a, **k):
    raise OSError("network is disabled in this sandbox")
_socket.socket = _no_network
_socket.create_connection = _no_network
_socket.getaddrinfo = _no_network
try:
    import resource as _r
    _r.setrlimit(_r.RLIMIT_CPU, (10, 10))
except Exception:
    pass
with open("doc.txt", encoding="utf-8") as _f:
    DOC = _f.read()
del _f
"""


def run_python(code: str, doc: str) -> str:
    """The code tool: the model's code in a fresh interpreter (``-I``, an empty
    environment, a temp directory, socket calls replaced by an error, CPU
    capped), with the document as the string ``DOC``. Returns what it printed."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "doc.txt").write_text(doc, encoding="utf-8")
        Path(tmp, "snippet.py").write_text(_PRELUDE + "\n" + code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "snippet.py"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=TOOL_TIMEOUT_S,
                env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {TOOL_TIMEOUT_S:.0f}s"
    err = proc.stderr.strip()
    if err:
        err = "\n".join(err.splitlines()[-6:])  # the traceback's tail is the useful part
    text = (proc.stdout or "") + (("\n" + err) if err else "")
    text = text.strip() or f"(no output, exit {proc.returncode})"
    if len(text) > TOOL_CHARS:
        text = text[:TOOL_CHARS] + f"\n[cut to the first {TOOL_CHARS} characters]"
    return text


# ---------------------------------------------------------------- the loop

_PY = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)
_JSONFENCE = re.compile(r"```json\s*\n(.*?)```", re.S)
_DOC = re.compile(r"Document:\n<<<\n(.*)\n>>>\s*$", re.S)
_TYPE = re.compile(r"^Document type: (.+)$", re.M)
LAST_TURN = "That was your last tool call. Reply now with the ```json block."


def _allowed(desc: str) -> list[str]:
    """The values a field's description lists after "one of:" or as ISO codes."""
    m = re.search(r"one of:?\s*([^.;]+)", desc)
    if m:
        return [x.strip() for x in m.group(1).split(",") if x.strip()]
    m = re.search(r"ISO 4217 code:\s*([A-Z, ]+?)(?:\s+or\s+([A-Z]{3}))?$", desc.strip())
    if m:
        codes = [x.strip() for x in m.group(1).split(",") if x.strip()]
        if m.group(2):
            codes.append(m.group(2))
        return codes
    return []


def schema_problems(obj: dict[str, Any], doc_type: str, *, enums: bool = False) -> list[str]:
    """What a validator can see without the gold: the keys and the value
    shapes; with ``enums``, also that an enum or code field holds one of the
    values its description lists."""
    schema = SCHEMAS[doc_type]
    out = []
    missing = [k for k in schema if k not in obj]
    extra = [k for k in obj if k not in schema]
    if missing:
        out.append("missing keys: " + ", ".join(missing))
    if extra:
        out.append("unexpected keys: " + ", ".join(extra))
    for k, (kind, _) in schema.items():
        v = obj.get(k)
        if v is None:
            continue
        if kind == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
            out.append(f"{k} is not YYYY-MM-DD: {v!r}")
        if kind == "money" and (isinstance(v, bool) or not isinstance(v, (int, float))):
            out.append(f"{k} is not a plain number: {v!r}")
        if kind == "int" and (isinstance(v, bool) or not isinstance(v, int)):
            out.append(f"{k} is not an integer: {v!r}")
        if enums and kind in ("enum", "code"):
            allowed = _allowed(schema[k][1])
            if allowed and str(v).strip().lower() not in [a.lower() for a in allowed]:
                out.append(f"{k} is not one of the allowed values ({', '.join(allowed)}): {v!r}")
    return out


def agent(chat: Chat, *, instructions: str, max_turns: int, retries: int, validate: bool | str):
    """The harness's loop as a callable ``prompt -> trajectory``. A reply that
    is a ```python block (with no ```json block) runs in the tool while turns
    remain; the reply with the JSON is final. With ``retries``, an unparseable
    final (or, with ``validate``, a malformed one) is sent back with the reason."""

    seen: dict[str, int] = {}
    seen_lock = threading.Lock()

    def run(prompt: str) -> dict[str, Any]:
        doc_m, type_m = _DOC.search(prompt), _TYPE.search(prompt)
        doc = doc_m.group(1) if doc_m else prompt
        doc_type = type_m.group(1).strip().replace(" ", "_") if type_m else ""
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
        steps: list[dict[str, Any]] = []
        turns, retries_left, final, cut = 0, retries, "", False
        # the n-th time this harness plays this prompt is rollout n: each of
        # the k rollouts gets its own seed (the engine does not pass the index)
        with seen_lock:
            nth = seen.get(prompt, 0)
            seen[prompt] = nth + 1
        base = int(hashlib.sha256(f"{prompt}:{SALT}:{nth}".encode()).hexdigest()[:8], 16)
        while True:
            reply, u = chat(messages, base + 7919 * turns)
            turns += 1
            cut = u.get("finish_reason") == "length"
            steps.append(
                {
                    "model_turn": turns,
                    "input_tokens": int(u.get("prompt_tokens") or 0),
                    "output_tokens": int(u.get("completion_tokens") or 0),
                    "truncated": cut,
                }
            )
            messages.append({"role": "assistant", "content": reply})
            code = _PY.findall(reply)
            if code and not _JSONFENCE.search(reply) and turns < max_turns:
                out = run_python(code[-1], doc)
                steps.append({"tool": "run_python", "arguments": {"code": code[-1]}, "result": out})
                msg = f"run_python output:\n{out}"
                if turns == max_turns - 1:
                    msg += "\n\n" + LAST_TURN
                messages.append({"role": "user", "content": msg})
                continue
            final = reply
            answer, why = parse_answer(reply)
            problems = [why] if answer is None else []
            if answer is not None and validate and doc_type in SCHEMAS:
                problems = schema_problems(answer, doc_type, enums=validate == "schema+enum")
            if problems and retries_left > 0:
                retries_left -= 1
                steps.append({"retry": problems})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your answer cannot be accepted: "
                        + "; ".join(problems)
                        + ". Reply with only the corrected ```json block.",
                    }
                )
                continue
            break
        return {
            "steps": steps,
            "final_text": final,
            "finish_reason": "length" if cut else "stop",
        }

    return run


# ---------------------------------------------------------------- the backends


def endpoint_chat(url: str, model: str, key: str) -> Chat:
    """An OpenAI-compatible chat call against your own vLLM server."""
    import requests

    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    extra: dict[str, Any] = {}
    if "qwen3" in model.lower():
        # a hybrid-thinking checkpoint: served with thinking off, the same way
        # Nemotron is told "detailed thinking off" (a serving setting, not a lever)
        extra["chat_template_kwargs"] = {"enable_thinking": False}

    def chat(messages: list[dict[str, str]], seed: int) -> tuple[str, dict[str, Any]]:
        body = {"model": model, "messages": messages, "seed": seed, **SAMPLING, **extra}
        err = ""
        for attempt in range(6):
            try:
                r = requests.post(
                    f"{base}/chat/completions",
                    json=body,
                    timeout=REQUEST_TIMEOUT_S,
                    headers={"Authorization": f"Bearer {key}"},
                )
                if r.status_code == 200:
                    data = r.json()
                    choice = data["choices"][0]
                    u = dict(data.get("usage") or {})
                    u["finish_reason"] = choice.get("finish_reason")
                    return choice["message"].get("content") or "", u
                if r.status_code == 400:  # context overflow: an honest failed reply
                    return "", {"finish_reason": "length", "error": r.text[:200]}
                err = f"{r.status_code} {r.text[:200]}"
            except requests.RequestException as exc:
                err = str(exc)[:200]
            time.sleep(min(30, 5 * 2**attempt))
        raise RuntimeError(f"endpoint failed six times: {err}")

    return chat


_GOLD: dict[str, dict[str, Any]] = {}


def scripted_chat(name: str) -> Chat:
    """The offline stand-in. It knows the gold record for every document and
    makes the mistakes a small model makes, each at a planted rate that a rule
    in the system prompt removes, so a candidate that states the rule scores
    higher. It sends one ```python block first (so the tool path runs) and the
    JSON after. Its numbers show the mechanics, not a result."""
    if not _GOLD:
        _GOLD.update({task_prompt(a): a for a in build()})

    def draw(*parts: Any) -> float:
        key = ":".join(map(str, (name, *parts)))
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def chat(messages: list[dict[str, str]], seed: int) -> tuple[str, dict[str, Any]]:
        system, user = messages[0]["content"].lower(), messages[1]["content"]
        ask = _GOLD[user]
        turn = sum(1 for m in messages if m["role"] == "assistant")
        u: dict[str, Any] = {
            "prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
            "finish_reason": "stop",
        }
        if turn == 0:
            u["completion_tokens"] = 20
            return "```python\nprint(len(DOC.splitlines()), 'lines')\n```", u
        out = dict(ask["gold"])
        for f, (kind, _) in SCHEMAS[ask["doc_type"]].items():
            r = draw(ask["id"], f, seed, SALT)
            gold = ask["gold"][f]
            if gold is None and r < 0.5 and "never guess" not in system:
                out[f] = "unknown"
            elif kind == "money" and gold is not None and r < 0.3 and "sum" not in system:
                out[f] = round(gold * 1.1, 2)
            elif kind == "date" and gold is not None and r < 0.25 and "dd.mm" not in system:
                out[f] = gold[5:7] + "/" + gold[8:] + "/" + gold[:4]
            elif kind == "name" and r < 0.2 and "first last" not in system:
                out[f] = " ".join(reversed(str(gold).split()))
        u["completion_tokens"] = 80
        if draw(ask["id"], "parse", seed, SALT) < 0.08 and turn < 2:
            return "Here are the fields: " + json.dumps(out)[:-3], u
        return f"```json\n{json.dumps(out)}\n```", u

    return chat


def chat_for(model: str) -> Chat:
    if model.startswith("scripted"):
        return scripted_chat(model)
    if not model.startswith("vllm:") or "@" not in model:
        raise ValueError(f"model {model!r}: use vllm:<hub id>@<url> or scripted")
    name, url = model[len("vllm:") :].split("@", 1)
    key = os.environ.get("VLLM_API_KEY")
    if not key:
        raise SystemExit("set VLLM_API_KEY to the key your docx-serve app was deployed with")
    return endpoint_chat(url, name, key)


# ---------------------------------------------------------------- the two arms

RULES = [
    (
        "DOC already holds the full document text inside the sandbox. Never paste or retype "
        "the document into your code; read it from DOC (for example DOC.splitlines())."
    ),
    (
        "If the tool errors, prints nothing, or you have no calls left, do not explain, "
        "apologize or describe a fix: read the document text in the message yourself and reply "
        "with the ```json block. Every value the document shows goes in the block; null is only "
        "for a field the document truly does not give, never a placeholder for a value your "
        "code failed to extract."
    ),
    (
        "Read the fields directly off the document in the message; you can see it, so no code "
        "is needed to find a name, a number, a date or an id. Use the tool only for arithmetic "
        "the document does not print (adding line amounts, a date plus N days, counting lines "
        "across pages), with a few short statements over DOC. If the document prints the "
        "number, copy it and skip the tool. A first reply that is the ```json block is the "
        "best reply."
    ),
]

ARMS = {
    # the starting harness: four turns, no retry, no validation
    "baseline": {
        "instructions": BASE_INSTRUCTIONS,
        "max_turns": 4,
        "retries": 0,
        "validate": False,
    },
    # the harness the search picked: three rules, and one retry when the final answer
    # fails a shape check (the check never sees the gold)
    "recipe": {
        "instructions": BASE_INSTRUCTIONS + "\n\n" + "\n\n".join(RULES),
        "max_turns": 4,
        "retries": 1,
        "validate": True,
    },
}


def data(seed: int = DATA_SEED) -> tuple[list[dict], list[dict]]:
    """(search, holdout): the generated documents, split by a hash of the ask id."""
    asks = build(seed)
    search = [a for a in asks if a["split"] != "holdout"]
    holdout = [a for a in asks if a["split"] == "holdout"]
    return search, holdout


def evaluate(chat: Chat, arm: str, asks: list[dict], k: int, concurrency: int) -> list[dict]:
    """k rollouts of one harness per document, each graded against the gold record."""
    run = agent(chat, **ARMS[arm])
    jobs = [(a, i) for a in asks for i in range(k)]

    def one(job: tuple[dict, int]) -> dict:
        ask, _ = job
        traj = run(task_prompt(ask))
        answer, _why = parse_answer(traj["final_text"])
        g = grade(ask, answer)
        tokens = sum(s.get("input_tokens", 0) + s.get("output_tokens", 0) for s in traj["steps"])
        return {
            "task_id": ask["id"],
            "field_f1": g["field_f1"],
            "tokens": tokens,
            "final_text": traj["final_text"],
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, jobs))


def per_doc(rows: list[dict]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(r["field_f1"])
    return {t: statistics.fmean(v) for t, v in by.items()}


def boot(values: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def summarize(rows: list[dict]) -> dict:
    docs_mean = list(per_doc(rows).values())
    return {"score": statistics.fmean(docs_mean), "ci": boot(docs_mean), "steps": 0}


def selftest() -> None:
    search, holdout = data()
    assert len(search) + len(holdout) == 200 and len(holdout) == 99, (len(search), len(holdout))
    print("test version", test_version(holdout))
    a = holdout[0]
    assert grade(a, dict(a["gold"]))["field_f1"] == 1.0
    assert grade(a, None)["field_f1"] == 0.0
    assert parse_answer("```python\nprint({'a': 1, 'b': 2})\n```")[0] is None
    assert run_python("print(len(DOC))", "abc").strip() == "3"
    rows = evaluate(scripted_chat("scripted"), "recipe", holdout[:5], k=1, concurrency=2)
    assert len(rows) == 5 and all(0.0 <= r["field_f1"] <= 1.0 for r in rows)
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="", help="vllm:<hub id>@<url> of your vLLM server")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.model:
        raise SystemExit("pass --model vllm:<hub id>@<url>, or run --selftest offline")
    chat = chat_for(args.model)
    _search, holdout = data()
    print("holdout test version", test_version(holdout))
    base_runs = []
    for i in range(EVAL_RUNS):
        os.environ["DOCX_SALT"] = str(i)
        base_runs.append(evaluate(chat, "baseline", holdout, args.k, args.concurrency))
    os.environ["DOCX_SALT"] = "0"
    rec = evaluate(chat, "recipe", holdout, args.k, args.concurrency)
    b, r = per_doc(base_runs[0]), per_doc(rec)
    diffs = [r[t] - b[t] for t in b]
    run_std = statistics.pstdev(statistics.fmean(per_doc(x).values()) for x in base_runs)
    out = {
        "recipe": HERE.name,
        "paper": "https://arxiv.org/abs/2603.28052",
        "base_model": BASE_MODEL,
        "metric": METRIC,
        "n_holdout": len(holdout),
        "k": args.k,
        "book": BOOK,
        "arms": {"baseline": summarize(base_runs[0]), "recipe": summarize(rec)},
        "delta": {
            "recipe_vs_baseline": statistics.fmean(diffs),
            "ci": boot(diffs),
            "verdict": "unresolved",
        },
        "checks": {
            "run_std": run_std,
            "run_std_runs": EVAL_RUNS,
            "train_seeds": {"baseline": 1, "recipe": 1},
            "decontaminated_dropped": 0,
            "over_optimized": False,
            "length_before": statistics.fmean(len(x["final_text"]) for x in base_runs[0]),
            "length_after": {
                "baseline": statistics.fmean(len(x["final_text"]) for x in base_runs[0]),
                "recipe": statistics.fmean(len(x["final_text"]) for x in rec),
            },
            "hack_scan_top": "",
            "seed": DATA_SEED,
        },
        "gpu": "L40S",
        "usd": 0.0,
        "verified": time.strftime("%Y-%m-%d"),
        "whileai": "",
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        f"baseline {out['arms']['baseline']['score']:.3f} -> recipe {out['arms']['recipe']['score']:.3f}"
    )


if __name__ == "__main__":
    main()
