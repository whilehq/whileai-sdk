"""Math verifiers: numeric closeness and symbolic equality.

The canonical verifiable reward (Lambert et al. 2024, arXiv:2411.15124;
Lambert 2025, chapters Reasoning and Tool Use).
Handles the common answer envelopes: ``\\boxed{...}``, "the answer is X",
trailing number. Symbolic equality uses sympy when installed and falls back
to normalized-string plus numeric comparison so the verifier still runs with
no extra dependency.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Verifier

#: Two floats closer than this are the same number: rounding in a printed
#: answer, not a wrong answer. (convention, untested)
NUMERIC_TOLERANCE = 1e-6

_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_IS = re.compile(r"(?:answer|result|solution)\s*(?:is|=|:)\s*\$?([^\n.]+)", re.I)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def extract_answer(text: str) -> str | None:
    """Pull the answer span from a chatty solution."""
    text = str(text or "").strip()
    boxes = _BOXED.findall(text)
    if boxes:
        return boxes[-1].strip()
    m = _ANSWER_IS.search(text)
    if m:
        return m.group(1).strip().rstrip("$").strip()
    line = (text.splitlines() or [""])[-1].strip()
    return line or None


_FRACTION = re.compile(r"(-?\d+)\s*/\s*(\d+)")
_LATEX_FRAC = re.compile(r"\\frac\{(-?\d+)\}\{(\d+)\}")


def _to_float(s: str) -> float | None:
    s = str(s).strip().replace(",", "").replace("$", "").rstrip("%")
    m = _LATEX_FRAC.search(s) or _FRACTION.search(s)
    if m:
        try:
            denom = float(m.group(2))
            if denom != 0:
                return float(m.group(1)) / denom
        except ValueError:
            pass
    nums = _NUMBER.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def _clean_expr(s: str) -> str:
    s = str(s).strip().strip("$").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s.strip()


def _sympy_equal(a: str, b: str) -> bool | None:
    try:
        from sympy.parsing.latex import parse_latex  # noqa: F401
    except Exception:
        return None
    from sympy import simplify
    from sympy.parsing.sympy_parser import parse_expr

    def parse(x: str):
        x = _clean_expr(x)
        for parser in (lambda t: parse_expr(t.replace("^", "**"), evaluate=True),):
            try:
                return parser(x)
            except Exception:
                pass
        try:
            from sympy.parsing.latex import parse_latex as pl

            return pl(x)
        except Exception:
            return None

    ea, eb = parse(a), parse(b)
    if ea is None or eb is None:
        return None
    try:
        return bool(simplify(ea - eb) == 0)
    except Exception:
        try:
            return bool(ea.equals(eb))
        except Exception:
            return None


class Numeric(Verifier):
    """The candidate's final number equals the reference number within a
    tolerance. ``rel`` gives a relative tolerance, ``tol`` absolute."""

    def __init__(
        self,
        *,
        tol: float = 1e-6,
        rel: float | None = None,
        field: str | None = None,
        name: str | None = None,
    ):
        super().__init__(field=field, name=name)
        self.tol, self.rel = tol, rel

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        if reference is None:
            return None
        want = _to_float(str(reference))
        got = _to_float(extract_answer(candidate) or candidate)
        if want is None:
            return None
        if got is None:
            return 0, "no number in answer"
        tol = self.tol if self.rel is None else max(self.tol, self.rel * abs(want))
        return (1 if abs(got - want) <= tol else 0), f"got {got}, want {want}"


class MathEqual(Verifier):
    """Symbolic equality (sympy) with a numeric and normalized-string
    fallback. Reads \\boxed{}/'answer is'/last-line from the candidate."""

    def __init__(self, *, field: str | None = None, name: str | None = None):
        super().__init__(field=field, name=name)

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        if reference is None:
            return None
        got = extract_answer(candidate)
        want = str(reference).strip()
        if got is None:
            return 0, "no answer found"
        if _clean_expr(got).replace(" ", "") == _clean_expr(want).replace(" ", ""):
            return 1, f"exact: {got!r}"
        sym = _sympy_equal(got, want)
        if sym is True:
            return 1, f"symbolically equal: {got!r} == {want!r}"
        if sym is False:
            return 0, f"not equal: {got!r} vs {want!r}"
        gf, wf = _to_float(got), _to_float(want)
        if gf is not None and wf is not None:
            return (1 if abs(gf - wf) <= NUMERIC_TOLERANCE else 0), f"numeric {gf} vs {wf}"
        return 0, f"cannot compare {got!r} to {want!r}"
