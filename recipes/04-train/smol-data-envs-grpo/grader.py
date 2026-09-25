"""The SmolDataEnvs grader, vendored so the reward runs offline.

Source: huggingface.co/datasets/FineEnvs/SmolDataEnvs, file ``grader.py`` at
revision b2bf35647e2381b1ab12c2ad7862cbbe4e8f857b (MIT). ``grade()`` and its
helpers are unchanged apart from formatting; the dataset's two stdin
command-line entry points and its tool-count bonus are left out, because the
reward here is the correct/incorrect bit alone.

No LLM judge. Four tiers, first match wins:
  1. exact (case-insensitive, whitespace collapsed)
  2. numeric within abs/rel tolerance, plus a percent <-> fraction bridge
  3. comma-separated list, order-insensitive, numeric-tolerant per element
  4. Math-Verify symbolic equivalence (skipped when math_verify is absent)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMERIC_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


@dataclass
class GradeResult:
    reward: float
    method: str


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _to_float(s: str) -> float | None:
    if not s:
        return None
    m = _NUMERIC_RE.search(str(s).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _num_close(g: float, c: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(g - c) <= abs_tol or abs(g - c) / max(abs(g), 1e-9) <= rel_tol


def _is_clean_number(s: str) -> bool:
    t = (s or "").strip().strip("%$").strip().replace(",", "")
    return bool(_NUMERIC_RE.fullmatch(t))


def _elem_match(a: str, b: str, abs_tol: float, rel_tol: float) -> bool:
    if _normalize(a) == _normalize(b):
        return True
    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        return _num_close(fa, fb, abs_tol, rel_tol)
    return False


def _list_match(gold: str, cand: str, abs_tol: float, rel_tol: float) -> bool:
    gl = [x.strip() for x in gold.split(",") if x.strip()]
    cl = [x.strip() for x in cand.split(",") if x.strip()]
    if len(gl) < 2 or len(gl) != len(cl):
        return False
    for gs, cs in ((gl, cl), (sorted(gl, key=str.lower), sorted(cl, key=str.lower))):
        if all(_elem_match(a, b, abs_tol, rel_tol) for a, b in zip(gs, cs, strict=True)):
            return True
    return False


def _math_verify_match(gold: str, candidate: str) -> bool:
    try:
        from math_verify import parse, verify

        return bool(verify(parse(gold), parse(candidate), timeout_seconds=5))
    except Exception:
        return False


def grade(
    gold: str,
    candidate: str | None,
    *,
    reward_mode: str = "",
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> GradeResult:
    if not gold or candidate is None:
        return GradeResult(0.0, "miss")

    # Tier 1: exact
    if _normalize(gold) == _normalize(candidate):
        return GradeResult(1.0, "exact")

    # Tier 2: numeric (clean-number gold) + percent/fraction bridge
    if reward_mode in ("numeric", "flexible") or _is_clean_number(gold):
        g, c = _to_float(gold), _to_float(candidate)
        if g is not None and c is not None:
            if _num_close(g, c, abs_tol, rel_tol):
                return GradeResult(1.0, "numeric")
            # percent<->fraction: one side is a fraction (<1), the other a percent (>=1)
            if (0 < abs(c) < 1 <= abs(g)) or (0 < abs(g) < 1 <= abs(c)):  # noqa: SIM102
                if _num_close(g, c * 100, abs_tol, rel_tol) or _num_close(
                    g, c / 100, abs_tol, rel_tol
                ):
                    return GradeResult(1.0, "numeric_scaled")

    # Tier 3: list (comma-separated), order-insensitive, per-element tolerant
    if reward_mode in ("list", "list_csv") or ("," in gold and "," in candidate):  # noqa: SIM102
        if _list_match(gold, candidate, abs_tol, rel_tol):
            return GradeResult(1.0, "list")

    # Tier 4: math-verify
    if _math_verify_match(gold, candidate):
        return GradeResult(1.0, "math_verify")

    return GradeResult(0.0, "miss")
