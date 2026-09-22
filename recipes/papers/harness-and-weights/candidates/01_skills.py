"""Candidate 01: a skills file. The baseline's worst rows mix tickers
together, use the population standard deviation, and return a numpy-style
value where a float was asked for. This candidate carries a SKILLS.md in
its instructions that says how to read the table and what to return
(Karten et al. 2026, Prime Agent: the harness carries skills the proposer
edits between runs). No tool yet."""

from __future__ import annotations

from harnesses import build

import whileai as wai

SKILLS = """
- Filter to the one ticker first: `rows = [b for b in bars if b["ticker"] == ticker]`. The
  rows are already in date order, so `closes = [b["close"] for b in rows]`.
- A simple return is `closes[i] / closes[i - 1] - 1`; a log return is `math.log(closes[i] /
  closes[i - 1])`. There is one fewer return than closes.
- "Last N" means the slice `[-N:]` of the returns or rows, not of the closes.
- Sample standard deviation divides by n - 1: `math.sqrt(sum((x - m) ** 2 for x in xs) /
  (len(xs) - 1))`.
- Cross-sectional means across tickers on one day: build `{ticker: closes}` once, then loop
  over day indices.
- Return a plain `float` (or `int` where the task says so). Do not print, do not read files.
"""
TOOL = False

# Offline stand-in: the read-the-table mistakes are gone, the arithmetic ones stay.
SCRIPTED_RATE = 0.55


def harness(model: str) -> wai.Harness:
    return build(model, skills=SKILLS, tool=TOOL, label="01_skills", scripted_rate=SCRIPTED_RATE)
