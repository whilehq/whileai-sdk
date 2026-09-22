"""Candidate 02: the skills file plus a REPL. Candidate 01's worst rows are
off-by-one on the window or the SMA alignment: the code runs, the number is
wrong. This candidate gives the model `run_python` (its code on the seeded
table, the printed result back) and three turns: run, read, answer. The
skills text grows by what the tool is for (Karten et al. 2026: a persistent
REPL plus skills the harness carries)."""

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
- Return a plain `float` (or `int` where the task says so). Do not read files.
- Test first: define the function, `print(fn(bars, ...))` with the task's arguments, and read
  the output. A traceback names the bug; a number that is not a plain float or int means the
  return type is wrong. Then send the final block without the print.
- When two windows meet (a fast and a slow average), line them up on the same day index
  before comparing; check the count on a small slice before trusting it on the table.
"""
TOOL = True

# Offline stand-in: only the occasional definition slip is left.
SCRIPTED_RATE = 0.75


def harness(model: str) -> wai.Harness:
    return build(
        model, skills=SKILLS, tool=TOOL, label="02_skills_tool", scripted_rate=SCRIPTED_RATE
    )
