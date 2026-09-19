"""Verifiers: rewards that are programs, not judges.

Lambert et al. 2024 (Tulu 3, arXiv:2411.15124) and Lambert 2025, chapters Tool
Use and Reasoning: the reward for a verifiable task is a checker, not an
opinion. A verifier reads a rollout, decides pass or fail (or a partial score
in [0, 1]), and says why.

Every verifier honors the judge contract in
``whileai.simulations.score.judging``
(``callable(row) -> {"reward", "reason", ...}``), so a verifier drops
straight into ``data.grade(judge=...)``, ``evaluate``, ``optimize`` and a
gated ``push``. Nothing here talks to a model.

A verifier reads two things from a row:

- the *candidate*: what the policy produced. ``final_text`` by default,
  else the last assistant message.
- the *reference*: the gold the checker compares against. It lives in the
  row's ``privileged`` block (``privileged.reference``), which the training
  export deliberately never projects, so the answer key cannot leak into a
  training file. Common flat fields (``reference``, ``answer``, ``target``,
  ``solution``, ``info.answer``) are read as a fallback so a dataset that
  stores the key plainly still works.

A gold read out of the ``privileged`` block is kept out of the verdict's
``reason`` as well, which reads ``<reference>`` in its place: ``reason``
travels with the row into every training export, so a reason that quotes
the gold hands the answer to the student on exactly the rows it got wrong.
The candidate half of the comparison stays, and a gold the caller put in a
plain column of their own is quoted back as before.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from ..defaults import TEXT_HEURISTICS

# Where a verifier looks for the gold, in order. privileged.reference is the
# schema-native home (never exported to training rows).
REFERENCE_FIELDS = ("reference", "answer", "target", "solution", "label", "expected")


class VerifierError(Exception):
    """A verifier could not run at all (bad config), distinct from a fail."""


def candidate_text(row: dict) -> str:
    """What the policy produced: final_text, else the last assistant turn."""
    if not isinstance(row, dict):
        return str(row or "")
    text = row.get("final_text")
    if text:
        return str(text)
    for msg in reversed(row.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def _reference_with_source(row: dict, field: str | None = None) -> tuple[Any, bool]:
    """``(gold, came_out_of_the_privileged_block)``.

    The flag is what tells the reason scrub below whether the value is the
    teacher's answer key (never student-visible) or a plain column the
    caller put in the row themselves.
    """
    if not isinstance(row, dict):
        return None, False
    if field:
        if field in row:
            return row[field], field == "privileged"
        # dotted path, e.g. "info.answer"
        cur: Any = row
        for part in field.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        return cur, field.split(".")[0] == "privileged"
    priv = row.get("privileged")
    if isinstance(priv, dict) and priv.get("reference") is not None:
        return priv["reference"], True
    for key in REFERENCE_FIELDS:
        if row.get(key) is not None:
            return row[key], False
    for holder in ("info", "metadata", "privileged"):
        sub = row.get(holder)
        if isinstance(sub, dict):
            for key in REFERENCE_FIELDS:
                if sub.get(key) is not None:
                    return sub[key], holder == "privileged"
    return None, False


def reference_value(row: dict, field: str | None = None) -> Any:
    """The gold. An explicit ``field`` wins; then privileged.reference; then
    the flat fallback fields; then ``info``/``metadata`` sub-dicts."""
    return _reference_with_source(row, field)[0]


#: What a redacted answer key reads as in a verifier's ``reason``.
_REDACTED = "<reference>"


def _reference_spellings(reference: Any) -> list[str]:
    """Every spelling of the gold a reason might quote, longest first.

    A reason is built from the gold with an f-string, so the leak is
    usually verbatim: ``str(value)`` for text, and the float the verifier
    parsed it to for a number (``"42"`` is quoted back as ``42.0``, and
    ``\\frac{1}{2}`` as ``0.5``). ``!r`` escapes newlines, so the escaped
    form counts too. Lists and dicts are walked, since ``Includes`` and
    ``JSONField`` take collections.
    """
    from .math import _to_float  # math imports this module; keep it lazy

    found: set[str] = set()

    def walk(value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)
            return
        text = str(value).strip()
        if not text:
            return
        found.add(text)
        escaped = repr(text)[1:-1]
        if escaped != text:
            found.add(escaped)
        with contextlib.suppress(ValueError):
            found.add(str(float(text.replace(",", ""))))
        as_float = _to_float(text)
        if as_float is not None:
            found.add(str(as_float))
        # multi-line gold (code tests): a traceback echoes single lines
        for line in text.splitlines():
            line = line.strip()
            if len(line) >= TEXT_HEURISTICS.gold_line_min_chars:
                found.add(line)

    walk(reference)
    return sorted(found, key=len, reverse=True)


#: A spelling this long or longer is redacted when the reason quotes only
#: a prefix of it, the way ``ExactMatch`` quotes the first 60 characters.
_PREFIX_MIN = 16


def _spelling_pattern(spelling: str) -> re.Pattern[str]:
    """``spelling`` as a whole token: a gold of ``7`` is not the ``7`` in
    ``17`` or ``7.5``, so the candidate half of ``got 17, want 7`` stays
    readable. Edges that are not word characters need no boundary."""
    head = r"(?<!\w)(?<!\d\.)" if spelling[0].isalnum() or spelling[0] == "_" else ""
    tail = r"(?!\w)(?!\.\d)" if spelling[-1].isalnum() or spelling[-1] == "_" else ""
    return re.compile(head + re.escape(spelling) + tail)


def _redact_reference(reason: str, reference: Any) -> str:
    """Replace every spelling of ``reference`` in ``reason``.

    ``reason`` travels with the row into every training export (it is on
    ``export``'s carry list), so a reason that quotes the answer key puts
    the answer key in the file the student trains on -- for exactly the
    rows the student got wrong. Verifiers keep the candidate side of the
    comparison, which is the half that says what went wrong.
    """
    text = str(reason or "")
    if not text:
        return text
    for spelling in _reference_spellings(reference):
        text = _spelling_pattern(spelling).sub(_REDACTED, text)
        if len(spelling) < _PREFIX_MIN:
            continue
        # a reason that truncates the gold still quotes its head
        for cut in range(len(spelling) - 1, _PREFIX_MIN - 1, -1):
            prefix = spelling[:cut]
            if prefix in text:
                text = _spelling_pattern(prefix).sub(_REDACTED, text)
                break
    return text


def _result(reward: float | int | None, reason: str, **meta: Any) -> dict[str, Any]:
    # Build the verdict as a literal, never a subscript write, so this stays
    # a judge-contract return and not a direct verdict-key assignment.
    base: dict[str, Any] = {"reward": reward, "reason": reason[:400]}
    return {**base, "judge_meta": meta} if meta else base


class Verifier:
    """Base class. Subclasses implement ``check(candidate, reference, row)``
    and return a float in [0, 1] (or a bool), or a ``(score, reason)`` pair.

    Instances are callables honoring the judge contract, and carry ``name``
    and ``kind`` so the scored row's ``ScorerRef`` records what graded it.
    """

    kind: str = "rule"

    def __init__(self, *, field: str | None = None, name: str | None = None):
        self.field = field
        self.name = name or self.__class__.__name__

    # subclasses override this
    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        raise NotImplementedError

    def __call__(self, row: dict) -> dict[str, Any]:
        reference: Any = None
        privileged = False
        try:
            candidate = candidate_text(row)
            reference, privileged = _reference_with_source(row, self.field)
            outcome = self.check(candidate, reference, row)
        except VerifierError as exc:
            note = self._reason(f"{self.name}: {exc}", reference, privileged)
            return _result(
                None,
                note,
                verifier=self.name,
                error=self._reason(str(exc), reference, privileged),
            )
        except Exception as exc:
            return {
                "reward": None,
                "reason": self._reason(
                    f"{self.name}: {type(exc).__name__}: {exc}", reference, privileged
                )[:400],
                "judge_status": "error",
                "judge_meta": {"verifier": self.name},
            }
        if isinstance(outcome, tuple):
            score, reason = outcome[0], str(outcome[1]) if len(outcome) > 1 else ""
        else:
            score, reason = outcome, ""
        reason = self._reason(reason, reference, privileged)
        if isinstance(score, bool):
            score = 1 if score else 0
        if score is None:
            return _result(
                None, reason or f"{self.name}: no reference to check against", verifier=self.name
            )
        score = max(0.0, min(1.0, float(score)))
        score = int(score) if score in (0.0, 1.0) else score
        if not reason:
            reason = (
                f"{self.name}: {'pass' if score == 1 else 'fail' if score == 0 else f'{score:.2f}'}"
            )
        return _result(score, reason, verifier=self.name)

    @staticmethod
    def _reason(reason: str, reference: Any, privileged: bool) -> str:
        """A verdict's reason, with the teacher's answer key taken out of it.

        Only when the gold came out of the ``privileged`` block: a plain
        ``answer`` column is the caller's own data and quoting it back
        tells them nothing they did not already put in the row.
        """
        return _redact_reference(reason, reference) if privileged else str(reason or "")


class FunctionVerifier(Verifier):
    """Wrap a plain ``fn(candidate, reference, row) -> score`` as a Verifier."""

    def __init__(
        self,
        fn: Callable[[str, Any, dict], Any],
        *,
        name: str | None = None,
        field: str | None = None,
        kind: str = "rule",
    ):
        super().__init__(field=field, name=name or getattr(fn, "__name__", "verifier"))
        self._fn = fn
        self.kind = kind

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        return self._fn(candidate, reference, row)


def verifier(fn: Callable[[str, Any, dict], Any]) -> FunctionVerifier:
    """Decorator: turn ``fn(candidate, reference, row) -> score`` into a Verifier."""
    return FunctionVerifier(fn)


class All(Verifier):
    """Pass only if every verifier passes. Score is the min; the reason names
    the first failure. Use for a task with several hard constraints."""

    def __init__(self, verifiers: Sequence[Verifier], *, name: str = "All"):
        super().__init__(name=name)
        self.verifiers = list(verifiers)

    def __call__(self, row: dict) -> dict[str, Any]:
        scores, reasons = [], []
        for v in self.verifiers:
            r = v(row)
            if r.get("reward") is None:
                return r
            scores.append(r["reward"])
            reasons.append(r.get("reason", ""))
        worst = min(scores) if scores else 0
        idx = scores.index(worst) if scores else 0
        return _result(
            1 if worst == 1 else worst,
            reasons[idx] if scores else "All: no verifiers",
            verifier=self.name,
            parts=scores,
        )


class Any_(Verifier):
    """Pass if any verifier passes. Score is the max."""

    def __init__(self, verifiers: Sequence[Verifier], *, name: str = "Any"):
        super().__init__(name=name)
        self.verifiers = list(verifiers)

    def __call__(self, row: dict) -> dict[str, Any]:
        scores, reasons = [], []
        for v in self.verifiers:
            r = v(row)
            if r.get("reward") is None:
                continue
            scores.append(r["reward"])
            reasons.append(r.get("reason", ""))
        if not scores:
            return _result(None, f"{self.name}: no verifier could run", verifier=self.name)
        best = max(scores)
        idx = scores.index(best)
        return _result(best, reasons[idx], verifier=self.name, parts=scores)


class Weighted(Verifier):
    """Weighted sum of verifiers, normalized to [0, 1]. Use for a rubric with
    graded criteria rather than one hard pass/fail (rubrics as rewards,
    Lambert 2025, chapter Synthetic Data and Distillation)."""

    def __init__(self, pairs: Sequence[tuple[Verifier, float]], *, name: str = "Weighted"):
        super().__init__(name=name)
        self.pairs = list(pairs)

    def __call__(self, row: dict) -> dict[str, Any]:
        total_w = sum(w for _, w in self.pairs) or 1.0
        acc, detail = 0.0, {}
        unjudged: str | None = None
        for v, w in self.pairs:
            r = v(row)
            s = r.get("reward")
            detail[v.name] = s
            if s is None:
                # A part that could not run (no reference, bad config) is
                # not a 0: the contract never invents a reward. Same as All.
                unjudged = unjudged or f"{self.name}: {r.get('reason') or v.name}"
                continue
            acc += float(s) * w
        if unjudged is not None:
            return _result(None, unjudged, verifier=self.name, parts=detail)
        score = round(acc / total_w, 4)
        score = int(score) if score in (0.0, 1.0) else score
        return _result(score, f"{self.name}: {score}", verifier=self.name, parts=detail)


def as_verifier(obj: Callable[..., Any]) -> Verifier:
    """Coerce a Verifier or a plain callable into a Verifier."""
    if isinstance(obj, Verifier):
        return obj
    return FunctionVerifier(lambda c, r, row: obj(row), name=getattr(obj, "__name__", "verifier"))


def extract_json(text: str) -> Any:
    """Best-effort: parse text as JSON, or the first {...}/[...] block in it."""
    text = str(text or "").strip()
    fenced = text
    if "```" in fenced:
        import re

        m = re.search(r"```(?:json)?\s*(.+?)```", fenced, re.S)
        if m:
            fenced = m.group(1).strip()
    for candidate in (fenced, text):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    import re

    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None
