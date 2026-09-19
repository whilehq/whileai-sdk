"""Is the verifier failing answers that are right?

A verifier is a rule: execution match, exact match, a regex on the final
number. Rules fail correct answers for reasons that have nothing to do with
correctness (rounding, ordering, a column named differently, an equivalent row
set), and every such false negative halves the separation on its task: the
policy is told that a right answer was wrong. On 570 SEC XBRL tasks, fixing
the grader moved mean reward from 0.20 to 0.48 with no training at all.
Lambert 2025 (chapter Over-optimization) treats verifiers as solved: "a
scoring function that returns a positive reward when the answer is correct and
0 otherwise". This module is our own measurement of how often that function is
wrong.

``audit_grades`` samples failed rows, asks a judge whether each reply is
in fact correct given the reference, and reports the false-negative rate
with a Wilson interval and the verifier reasons that dominate. The judge
is a second opinion, not ground truth: the judge-prompt rules apply
(length must not sway it, no position bias, temperature 0 for stable
ratings; Zheng et al. 2023, arXiv:2306.05685; Lambert 2025, chapter
Reward Modeling), and a rate near the threshold deserves a check of
the judge itself on human labels (``judge_agreement``, ``judge_trust``).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from typing import Any

from ..defaults import JUDGE_CHECK_SAMPLE, PASS_THRESHOLD
from .judging import run_judge
from .stats import wilson_interval

# FN_WARN = 0.10: above this share of audited failures overturned by the
# judge, the verifier is the problem to fix first. One false negative in
# ten halves the separation on a tenth of the tasks, about the size of a
# training gain; on 570 SEC XBRL tasks the rate was far past it and the
# grader fix moved mean reward 0.20 -> 0.48 (measured; the cut is
# convention).
FN_WARN = 0.10
# MIN_AUDITED_FAILS = 20: audited failures before the false-negative rate
# is worth acting on; under it the Wilson interval is wider than 0.3
# (convention).
MIN_AUDITED_FAILS = 20
# AUDIT_EXAMPLES = 5: overturned rows shown in the report.
AUDIT_EXAMPLES = 5

#: what the judge is asked when the row has no rubric of its own
AUDIT_QUESTION = (
    "The reference answer is the answer key for this task. Is the reply's final answer "
    "correct, meaning it gives the same result as the reference? Differences in "
    "formatting, rounding, ordering (unless the task asks for an order), column or "
    "field naming, or an equivalent way of expressing the same result do not make it "
    "wrong. A missing, different, or partial result does."
)

_DIGITS = re.compile(r"\d+(\.\d+)?")


def _reason_key(reason: Any, verifier: str | None) -> str:
    """The kind of failure a verifier reason names, without its numbers:
    ``result differs: got 3 rows x 2 cols`` and ``result differs: got 10
    rows x 1 col`` are one reason."""
    text = str(reason or "").strip()
    if verifier and text.lower().startswith(verifier.lower() + ":"):
        text = text[len(verifier) + 1 :].strip()
    head = re.split(r"[:(]", text, maxsplit=1)[0].strip()
    head = _DIGITS.sub("#", head).lower()
    return head or "(no reason)"


def _binary(row: dict) -> int | None:
    value = row.get("reward")
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f in (0.0, 1.0):
        return int(f)
    return None


def _audit_row(row: dict, question: str) -> dict:
    """The row as the judge sees it: the reply and the reference, the
    audit question as the rubric when the row has none, and the
    verifier's own verdict alongside so a judge that reads it can say
    why the rule was wrong."""
    out = dict(row)
    priv = dict(row.get("privileged") or {})
    if not priv.get("rubric"):
        priv["rubric"] = question
    out["privileged"] = priv
    out["audit"] = {
        "question": question,
        "verifier": row.get("judge_name"),
        "verifier_reason": row.get("reason"),
        "verifier_reward": row.get("reward"),
    }
    for key in ("reward", "reason", "judge_status", "judge_meta", "judge_name", "failure_class"):
        out.pop(key, None)
    return out


def _rate(hits: int, n: int) -> tuple[float | None, tuple[float, float] | None]:
    if not n:
        return None, None
    return round(hits / n, 4), wilson_interval(hits, n)


def audit_grades(
    rows: Sequence[dict],
    *,
    judge: Callable[[dict], Any],
    sample: int = JUDGE_CHECK_SAMPLE,
    passes: int = 0,
    question: str = AUDIT_QUESTION,
    seed: int = 0,
    concurrency: int = 4,
    timeout: float = 120,
    judge_name: str | None = None,
    fn_warn: float = FN_WARN,
) -> dict[str, Any]:
    """Estimate the verifier's false-negative rate from a judged sample.

    ``rows`` are graded by the verifier (``run_judge(rows, verifier)`` or
    ``data.grade(judge=verifier)``): ``reward`` 0/1, ``reason`` from the
    rule. ``sample`` failed rows (reward 0, judge ok) are drawn with
    ``seed`` and each is put to ``judge`` (any judge in the ``run_judge``
    contract: ``rubric_judge()``, ``grade_llm``, your own callable) with
    the reference in place and ``question`` as the rubric when the row
    carries none. A judge reward at or above ``PASS_THRESHOLD`` (0.5) on
    a failed row is a false negative. ``passes`` samples passed rows the
    same way for the false-positive side.

    Returns ``fn_rate`` with ``fn_ci95`` (Wilson), ``estimated_wrong_fails``
    (the rate over every failed row), ``reasons`` (the verifier's failure
    kinds in the sample, each with how many the judge overturned), a few
    ``examples``, ``fp_rate`` when ``passes`` > 0, and ``warnings``. Above
    ``fn_warn`` (``FN_WARN``, 0.10) the summary says to fix the verifier
    before training;
    ``select_for_rl(audit=report)`` and ``optimize(audit=)`` carry the
    same warning into the selection.
    """
    if sample < 1:
        raise ValueError("sample is how many failed rows to put to the judge; at least 1")
    graded = [r for r in rows if isinstance(r, dict) and _binary(r) is not None]
    failed = [r for r in graded if _binary(r) == 0 and r.get("judge_status", "ok") == "ok"]
    passed = [r for r in graded if _binary(r) == 1 and r.get("judge_status", "ok") == "ok"]
    verifier = next((str(r.get("judge_name")) for r in graded if r.get("judge_name")), None)
    rng = random.Random(seed)
    take_f = failed if len(failed) <= sample else rng.sample(failed, sample)
    take_p = (
        [] if passes <= 0 else (passed if len(passed) <= passes else rng.sample(passed, passes))
    )

    def _judge(batch: list[dict]) -> list[dict]:
        if not batch:
            return []
        audit_rows = [_audit_row(r, question) for r in batch]
        scored = run_judge(
            audit_rows,
            judge,
            judge_name=judge_name,
            concurrency=concurrency,
            timeout=timeout,
            source="audit",
        )
        return list(scored.rows)

    judged_f = _judge(list(take_f))
    judged_p = _judge(list(take_p))
    judge_label = next((str(r.get("judge_name")) for r in judged_f + judged_p), judge_name)

    checked_f = [r for r in judged_f if r.get("judge_status") == "ok" and _num(r) is not None]
    overturned = [r for r in checked_f if float(_num(r) or 0.0) >= PASS_THRESHOLD]
    checked_p = [r for r in judged_p if r.get("judge_status") == "ok" and _num(r) is not None]
    fp_rows = [r for r in checked_p if float(_num(r) or 0.0) < PASS_THRESHOLD]
    errors = sum(1 for r in judged_f + judged_p if r.get("judge_status") != "ok")

    fn_rate, fn_ci = _rate(len(overturned), len(checked_f))
    fp_rate, fp_ci = _rate(len(fp_rows), len(checked_p))

    reasons: dict[str, dict[str, int]] = {}
    for r in checked_f:
        key = _reason_key((r.get("audit") or {}).get("verifier_reason"), verifier)
        slot = reasons.setdefault(key, {"n": 0, "fn": 0})
        slot["n"] += 1
        if float(_num(r) or 0.0) >= PASS_THRESHOLD:
            slot["fn"] += 1
    reasons = dict(sorted(reasons.items(), key=lambda kv: (-kv[1]["fn"], -kv[1]["n"], kv[0])))

    examples = [
        {
            "scenario_id": r.get("scenario_id"),
            "rollout_index": r.get("rollout_index"),
            "verifier_reason": (r.get("audit") or {}).get("verifier_reason"),
            "judge_reason": r.get("reason"),
        }
        for r in overturned[:AUDIT_EXAMPLES]
    ]
    estimated = round(float(fn_rate) * len(failed)) if fn_rate is not None else None

    warnings: list[str] = []
    if fn_rate is None:
        summary = (
            f"no failed row could be audited ({len(failed)} failed, {errors} judge errors); "
            "nothing to say about the verifier"
        )
        warnings.append(summary)
    else:
        lo, hi = fn_ci or (0.0, 0.0)
        summary = (
            f"the judge overturned {len(overturned)} of {len(checked_f)} audited failures: "
            f"false-negative rate {fn_rate:.0%} (95% {lo:.0%}..{hi:.0%}), about {estimated} of "
            f"{len(failed)} failed rows are right answers the verifier rejected"
        )
        if fn_rate > fn_warn:
            top = next(iter(reasons), None)
            warnings.append(
                f"VERIFIER: {summary}. Fix the verifier before training: each false negative "
                "halves the separation on its task, and a grader fix has moved mean reward "
                "0.20 -> 0.48 with no training"
                + (f". The reason the judge overturns most: {top!r}" if top else "")
                + "."
            )
        if len(checked_f) < MIN_AUDITED_FAILS:
            warnings.append(
                f"{len(checked_f)} audited failures is a small sample; the interval is wide. "
                "Raise sample= for a rate worth acting on."
            )
        if fp_rate is not None and fp_rate > fn_warn:
            warnings.append(
                f"the judge also disagreed with {len(fp_rows)} of {len(checked_p)} audited passes "
                f"(false-positive rate {fp_rate:.0%}); the rule may be too loose as well"
            )
    if errors:
        warnings.append(f"{errors} audit rows were not judged (judge error or timeout)")
    if fn_rate is not None:
        warnings.append(
            "The judge is a second opinion, not ground truth; check it on human labels "
            "(judge_agreement, judge_trust) before acting on a rate near the threshold."
        )
    return {
        "n_rows": len(graded),
        "n_failed": len(failed),
        "n_passed": len(passed),
        "n_sampled": len(take_f),
        "n_checked": len(checked_f),
        "false_negatives": len(overturned),
        "fn_rate": fn_rate,
        "fn_ci95": fn_ci,
        "estimated_wrong_fails": estimated,
        "n_passes_sampled": len(take_p),
        "n_passes_checked": len(checked_p),
        "false_positives": len(fp_rows),
        "fp_rate": fp_rate,
        "fp_ci95": fp_ci,
        "judge_errors": errors,
        "reasons": reasons,
        "examples": examples,
        "verifier": verifier,
        "judge": judge_label,
        "question": question,
        "warnings": warnings,
        "summary": summary,
    }


def _num(row: dict) -> float | None:
    value = row.get("reward")
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audit_warning(audit: dict[str, Any] | None) -> str | None:
    """The one line a selection report carries when an audit found the
    verifier wrong too often; ``None`` otherwise."""
    if not isinstance(audit, dict):
        return None
    rate = audit.get("fn_rate")
    if rate is None or float(rate) <= FN_WARN:
        return None
    ci = audit.get("fn_ci95") or (0.0, 0.0)
    return (
        f"verifier false-negative rate {float(rate):.0%} (95% {ci[0]:.0%}..{ci[1]:.0%}, "
        f"{audit.get('n_checked')} audited): right answers are being scored 0. Fix the "
        "verifier before training on this selection (audit_grades)."
    )


def format_audit(report: dict[str, Any]) -> str:
    """The block a person reads: the summary, then the reasons."""
    lines = [str(report.get("summary") or "")]
    for key, slot in (report.get("reasons") or {}).items():
        lines.append(f"  {key}: {slot['fn']} of {slot['n']} overturned")
    for ex in report.get("examples") or []:
        lines.append(
            f"  {ex.get('scenario_id')} r{ex.get('rollout_index')}: verifier said "
            f"{ex.get('verifier_reason')!r}; judge: {ex.get('judge_reason')!r}"
        )
    return "\n".join(lines)


__all__ = ["AUDIT_QUESTION", "FN_WARN", "audit_grades", "audit_warning", "format_audit"]
