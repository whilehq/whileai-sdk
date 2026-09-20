"""Verifiers: programmatic, verifiable rewards (Lambert et al. 2024,
arXiv:2411.15124; Lambert 2025, chapters Reasoning and Tool Use).

A verifier is a checker, not a judge: it reads a rollout and decides pass,
fail, or a partial score in [0, 1], with no model call. Every verifier
honors the judge contract (``callable(row) -> {"reward", "reason"}``), so it
drops into ``data.grade(judge=v)``, ``evaluate``, ``optimize`` and a gated
``push`` exactly where an LLM judge would go.

    import whileai.simulations as wai
    from whileai.simulations.verify import MathEqual, All, Regex

    v = All([MathEqual(), Regex(r"</think>")])       # right answer, and it closed its reasoning
    scored = data.grade(judge=v)                     # verifier IS the reward
    rows, _ = wai.optimize(scored, mode="rl")        # GRPO data with a verifiable reward

The gold answer is read from the row's ``privileged.reference`` (never
exported to training rows), with flat fields (``answer``, ``target``, ...)
as a fallback. Point any verifier at a different column with ``field=``.
"""

from __future__ import annotations

from .base import (
    All,
    Any_,
    FunctionVerifier,
    ToolCall,
    Verifier,
    VerifierError,
    Weighted,
    as_verifier,
    candidate_text,
    extract_json,
    reference_value,
    tool_calls,
    verifier,
)
from .code import CodeExec, extract_code
from .math import MathEqual, Numeric, extract_answer
from .structured import JSONField, JSONSchema, JSONValid
from .text import ExactMatch, Includes, MultipleChoice, Regex, normalize

# "Any" is a builtin; expose the verifier under a non-shadowing alias too.
Any = Any_

__all__ = [
    "All",
    "Any",
    "Any_",
    "CodeExec",
    "ExactMatch",
    "FunctionVerifier",
    "Includes",
    "JSONField",
    "JSONSchema",
    "JSONValid",
    "MathEqual",
    "MultipleChoice",
    "Numeric",
    "Regex",
    "ToolCall",
    "Verifier",
    "VerifierError",
    "Weighted",
    "as_verifier",
    "candidate_text",
    "extract_answer",
    "extract_code",
    "extract_json",
    "normalize",
    "reference_value",
    "tool_calls",
    "verifier",
]
