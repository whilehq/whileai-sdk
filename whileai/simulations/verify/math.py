"""Math verifiers: numeric closeness and symbolic equality.

The canonical verifiable reward (Lambert et al. 2024, arXiv:2411.15124;
Lambert 2025, chapters Reasoning and Tool Use).
``MathEqual`` decides with Math-Verify (Kydlicek et al. 2025), the verifier
behind Open R1 and lighteval; ``Numeric`` and ``extract_answer`` read the
common answer envelopes: ``\\boxed{...}``, "the answer is X", trailing number.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from .base import Verifier

#: Two floats closer than this are the same number: rounding in a printed
#: answer, not a wrong answer. (convention, untested)
NUMERIC_TOLERANCE = 1e-6

_ANSWER_IS = re.compile(r"(?:answer|result|solution)\s*(?:is|=|:)\s*\$?([^\n.]+)", re.I)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def extract_answer(text: str) -> str | None:
    """Pull the answer span from a chatty solution."""
    text = str(text or "").strip()
    boxes = _boxed(text)
    if boxes:
        return boxes[-1].strip()
    m = _ANSWER_IS.search(text)
    if m:
        return m.group(1).strip().rstrip("$").strip()
    line = (text.splitlines() or [""])[-1].strip()
    return line or None


def _boxed(text: str) -> list[str]:
    """Every ``\\boxed{...}`` body, braces balanced to any depth."""
    out, i = [], text.find("\\boxed{")
    while i != -1:
        j, depth = i + len("\\boxed"), 0
        for k in range(j, len(text)):
            depth += (text[k] == "{") - (text[k] == "}")
            if depth == 0:
                out.append(text[j + 1 : k])
                break
        i = text.find("\\boxed{", j)
    return out


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
    """The candidate's final answer equals the reference, decided by
    Math-Verify: LaTeX and expression parsing, symbolic and numeric equality,
    sets, intervals, matrices. ``pip install "whileai[math]"``. Without it the
    constructor says so instead of falling back to a weaker rule, because a
    verifiable reward has to verify (Lambert 2025, chapter Reasoning and
    Inference-Time Scaling): the string-and-number rule this replaced failed
    199 correct answers and passed 64 wrong ones in 3,840 MATH-500
    completions, a false-positive shape a policy can learn (chapter
    Over-Optimization)."""

    def __init__(self, *, field: str | None = None, name: str | None = None):
        super().__init__(field=field, name=name)
        try:
            from math_verify import parse, verify
        except ImportError as exc:
            raise ImportError('MathEqual needs Math-Verify: pip install "whileai[math]"') from exc
        self._parse, self._verify = parse, verify

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        if reference is None:
            return None
        # Math-Verify's timeouts use signal.alarm, which only the main thread
        # may set; graders run in worker threads, so there they run without one.
        main = threading.current_thread() is threading.main_thread()
        kw: dict[str, Any] = {} if main else {"parsing_timeout": None}
        gold = self._parse(f"${reference}$", **kw)
        if not gold:
            return None, f"reference {reference!r} is not a math expression"
        got = self._parse(str(candidate or ""), **kw)
        if not got:
            return 0, "no answer found"
        ok = bool(self._verify(gold, got, timeout_seconds=5 if main else None))
        return (1 if ok else 0), f"math-verify: {'equal' if ok else 'not equal'}"
