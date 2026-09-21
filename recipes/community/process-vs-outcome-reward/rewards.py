"""The paper's two rewards, as programs.

arXiv:2607.02869 (Palandye et al.) defines:

    R_outcome  = 1 if |a_hat - a*| < eps else 0,  eps = 1e-5
    R_process  = correct steps / total steps,     numeric tolerance 1e-5,
                 with "a penalty when generated chains exceed 1.5x the
                 ground-truth step count"

The outcome reward is whileai's MathEqual verifier, unmodified, so the
training reward and the eval metric are the same program object.

The process reward is written here because the paper does not say how a
"step" is extracted from a free-form completion, nor how large the length
penalty is. Both choices are documented at the point they are made.
"""

from __future__ import annotations

import re

from whileai.simulations.verify import MathEqual

TOL = 1e-5

_GOLD_STEP = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

outcome_verifier = MathEqual()


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUMBER.findall(text or ""):
        v = _to_float(m)
        if v is not None:
            out.append(v)
    return out


def gold_steps(gold_answer: str, question: str) -> list[float]:
    """The values GSM8K's <<a+b=c>> calculator annotations compute.

    Values that already appear in the question are dropped: a step the
    model can copy is not a step it derived, and counting it would let
    restating the question earn process reward.
    """
    given = set(numbers_in(question))
    steps = []
    for _, rhs in _GOLD_STEP.findall(gold_answer):
        v = _to_float(rhs)
        if v is not None and v not in given:
            steps.append(v)
    return steps


def candidate_values(completion: str, question: str) -> list[float]:
    """Numbers the completion produced that the question did not give it."""
    given = set(numbers_in(question))
    return [v for v in numbers_in(completion) if v not in given]


def chain_len(completion: str) -> int:
    """How many reasoning steps the completion took.

    A GSM8K gold chain is one step per line, so a line carrying a number is
    the like-for-like unit to count against the gold step count. The final
    "#### n" line is the answer, not a step.
    """
    n = 0
    for line in (completion or "").splitlines():
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        if _NUMBER.search(line):
            n += 1
    return n


def process_score(completion: str, gold_answer: str, question: str) -> float:
    """R_process: the fraction of gold intermediate values the chain hit.

    Matching is multiset (each gold value consumes one candidate value) and
    absolute to 1e-5, as the paper states.

    The paper's length penalty has no stated magnitude. Chosen here: scale
    the score by (1.5 * gold_steps / candidate_steps) once the chain is
    longer than 1.5x, which is continuous, is 1.0 exactly at the threshold,
    and never goes below 0.
    """
    gold = gold_steps(gold_answer, question)
    if not gold:
        return 0.0
    pool = candidate_values(completion, question)
    matched = 0
    for g in gold:
        for i, c in enumerate(pool):
            if abs(c - g) < TOL:
                matched += 1
                pool.pop(i)
                break
    score = matched / len(gold)
    n_cand = chain_len(completion)
    limit = 1.5 * len(gold)
    if n_cand > limit:
        score *= limit / n_cand
    return max(0.0, min(1.0, score))


def outcome_score(completion: str, gold_final: str) -> float:
    """R_outcome: whileai's MathEqual, the same call the eval uses."""
    score, _ = outcome_verifier.check(completion, gold_final, {})
    return float(score)


def gold_final_answer(gold_answer: str) -> str:
    return gold_answer.rsplit("####", 1)[-1].strip().replace(",", "")
