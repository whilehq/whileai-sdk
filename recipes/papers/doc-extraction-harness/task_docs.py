"""The document-extraction task as the Meta-Harness recipe reads it
(``run.py --task task_docs.py``): ``tasks()`` is the frozen set and
``judge(row)`` grades a row. Another task is another file with the same two
names; the loop does not change.

The judge has two halves on every row:

* **Quality, the headline** (``quality``): ``wai.Judge`` with a five-item
  ``Rubric`` on Claude Haiku 4.5 at temperature 0 (``JUDGE_TEMPERATURE``), a
  different family from the model under search. It sees the document and the
  final reply only (a slim row: the steps are the harness's business, the
  answer is what is judged), never the gold. ``quality`` is the rubric's
  score, the mean of the criteria met; each criterion is its own marker.
* **Exact checks, beside it**: per field, normalized match against the
  generator's gold (``docs.grade``), ``field_acc`` the share right, and
  ``reward`` = every field right (a conjunction, never the headline).

Identical answers to the same document get the same verdict at temperature
0, so verdicts are cached by (rubric version, prompt, final reply) in
``DOCX_JUDGE_CACHE``: a harness that repeats an answer does not pay twice.
``DOCX_JUDGE=scripted`` (the dry run) swaps the model for a stand-in that
derives the criteria from the program's gold, so the pipeline runs offline.

The split is declared here, from a hash of the ask id (``docs.split_of``),
and checked against frozen.json, committed before any model answered.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import docs
import whileai as wai

FROZEN = json.loads((HERE / "frozen.json").read_text(encoding="utf-8"))
ASKS = docs.build(FROZEN["data_seed"])
for _split in ("selection", "holdout"):
    _tv = docs.test_version([a for a in ASKS if a["split"] == _split])
    if _tv != FROZEN[_split]["test_version"]:
        raise SystemExit(
            f"{_split} is {_tv}, frozen.json says {FROZEN[_split]['test_version']}: "
            "the generator changed; that is a new test"
        )
BY_PROMPT = {docs.task_prompt(a): a for a in ASKS}

JUDGE_MODEL = "anthropic:claude-haiku-4-5-20251001"
# The revised rubric (v2): it judges what the reply asserts, wherever the
# values appear, and scores the wrapper only under Clean JSON. v1 (fenced JSON
# only, "invented" read as style) failed its audit on the baseline: agreement
# 0.42-0.71, kappa 0.11-0.17 against program gold on 380 distinct answers.
_ASSERTS = (
    " Judge the values the reply asserts wherever they appear (a JSON object, a code block, a list "
    "or prose); the wrapper matters only for Clean JSON."
)
RUBRIC = wai.simulations.Rubric(
    criteria=(
        wai.simulations.Criterion(
            "Complete",
            "Every requested field that the document states (or that the field description says to "
            "compute from it) has a value asserted in the reply." + _ASSERTS,
        ),
        wai.simulations.Criterion(
            "Nothing invented",
            "No asserted value lacks support in the document: a field the document does not give is "
            "null or left out, never guessed. Formatting, extra wording or an extra key is not an "
            "invented value." + _ASSERTS,
        ),
        wai.simulations.Criterion(
            "Faithful and normalized",
            "Each asserted value matches the document and is normalized as asked: the entity named as "
            "printed (the vendor, not the customer; the total, not the balance due), dates as "
            "YYYY-MM-DD read the right way round, amounts as plain numbers, currency as the ISO code "
            "the symbols imply." + _ASSERTS,
        ),
        wai.simulations.Criterion(
            "Ambiguity surfaced",
            "Where the document is ambiguous or a field is missing, the reply makes that visible "
            "(null, or a stated note) rather than silently resolving it with a guess.",
        ),
        wai.simulations.Criterion(
            "Clean JSON",
            "Judged on its own: the final reply contains one parseable JSON object with exactly the "
            "requested keys and no prose inside the object. A markdown ```json fence around the "
            "object is fine; a Python dict or a code block is not JSON.",
        ),
    ),
    domain="document extraction",
    notes="v2. Quality of one extraction; the per-field gold check is a separate program.",
)
QUALITY = [c.slug for c in RUBRIC.criteria]
# The criteria a program can check against the gold, and the one it cannot.
PROGRAM_CHECKABLE = QUALITY[:3] + QUALITY[4:]
QUALITY_ONLY = [QUALITY[3]]

_CACHE_PATH = Path(os.environ.get("DOCX_JUDGE_CACHE", HERE / "out" / "judge_cache.jsonl"))
_LOCK = threading.Lock()
_CACHE: dict[str, dict[str, Any]] = {}
_JUDGE: Any = None


def _load_cache() -> None:
    if _CACHE or not _CACHE_PATH.exists():
        return
    for line in _CACHE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            _CACHE[item["key"]] = item["verdict"]


def tasks() -> list[dict[str, Any]]:
    """One task per document; the selection split is the recipe's train split."""
    return [
        {
            "prompt": docs.task_prompt(a),
            "task_id": a["id"],
            "split": "train" if a["split"] == "selection" else "holdout",
            "scenario_dimensions": {"difficulty": docs.difficulty(a), "doc_type": a["doc_type"]},
        }
        for a in ASKS
    ]


def asserted(final: str) -> dict[str, Any] | None:
    """The values a reply asserts, wherever they sit: the strict JSON answer,
    else a dict literal in any block (for the audit labels only; the F1
    headline reads JSON)."""
    import ast
    import re

    answer, _ = docs.parse_answer(final)
    if answer is not None:
        return answer
    for m in re.finditer(r"\{", final):
        depth = 0
        for j in range(m.start(), len(final)):
            depth += {"{": 1, "}": -1}.get(final[j], 0)
            if depth == 0:
                chunk = final[m.start() : j + 1]
                for load in (json.loads, ast.literal_eval):
                    try:
                        obj = load(chunk)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and len(obj) > 1:
                        return obj
                break
    return None


def program_checks(
    ask: dict[str, Any], answer: dict[str, Any] | None, final: str
) -> dict[str, float]:
    """The rubric's program-checkable criteria, read off the gold: the labels
    the judge is audited against (``kind="program"``, never human)."""
    schema = docs.SCHEMAS[ask["doc_type"]]
    gold = ask["gold"]
    strict = answer
    answer = asserted(final)
    ans = answer or {}
    complete = answer is not None and all(
        not docs._is_null(ans.get(f)) for f in schema if gold[f] is not None
    )
    invented = answer is None or any(
        not docs._is_null(ans.get(f)) for f in schema if gold[f] is None
    )
    faithful = answer is not None and all(
        docs.field_ok(kind, ans.get(f), gold[f])
        for f, (kind, _) in schema.items()
        if gold[f] is not None and not docs._is_null(ans.get(f))
    )
    clean = strict is not None and set(strict) == set(schema)
    return {
        QUALITY[0]: float(complete),
        QUALITY[1]: float(not invented),
        QUALITY[2]: float(faithful),
        QUALITY[4]: float(clean),
    }


def _scripted_verdict(
    ask: dict[str, Any], answer: dict[str, Any] | None, final: str
) -> dict[str, Any]:
    """Offline stand-in for the model judge: the program's verdicts, with the
    ambiguity item read as 'no invented value'. Mechanics only."""
    p = program_checks(ask, answer, final)
    crit = {str(i + 1): bool(p.get(slug, p[QUALITY[1]])) for i, slug in enumerate(QUALITY)}
    return wai.simulations.score.rubric.score_with_rubric(RUBRIC, crit, reason="scripted judge")


def quality(
    row: dict[str, Any], ask: dict[str, Any], answer: dict[str, Any] | None
) -> dict[str, Any]:
    """The rubric verdict for one row, cached by content."""
    global _JUDGE
    final = str(row.get("final_text") or "")
    if os.environ.get("DOCX_JUDGE") == "scripted":
        return _scripted_verdict(ask, answer, final)
    key = hashlib.sha256(
        json.dumps([RUBRIC.version, JUDGE_MODEL, row.get("prompt"), final]).encode()
    ).hexdigest()
    with _LOCK:
        _load_cache()
        hit = _CACHE.get(key)
        if _JUDGE is None:
            _JUDGE = wai.Judge(RUBRIC, model=JUDGE_MODEL, name="haiku-4.5-extraction-quality")
    if hit is not None:
        return {**hit, "cached": True}
    slim = {"prompt": row.get("prompt"), "final_text": final, "steps": []}
    v = _JUDGE(slim)
    reason = str(v.get("reason") or "")
    if v.get("reward") is None:
        if any(code in reason for code in ("401", "402", "429", "credit", "authentication")):
            raise SystemExit(f"judge stopped: {reason[:200]}")
        return v  # an ungraded row stays ungraded; not cached
    keep = {
        k: v[k] for k in ("reward", "reason", "markers", "criteria", "rubric_version") if k in v
    }
    usage_chars = len(json.dumps(slim)) + len(RUBRIC.checklist())
    with _LOCK:
        _CACHE[key] = keep
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "verdict": keep, "chars_in": usage_chars}) + "\n")
    return keep


def judge(row: dict[str, Any]) -> dict[str, Any]:
    """The loop's judge: the program always; the quality judge too unless
    ``DOCX_JUDGE=off`` (then ``quality`` is added afterwards, off the GPU's
    critical path, by ``run.py rejudge`` with the same cache)."""
    return grade_row(row, with_quality=os.environ.get("DOCX_JUDGE") != "off")


def grade_row(row: dict[str, Any], *, with_quality: bool = True) -> dict[str, Any]:
    """The program's field F1 (the headline marker ``field_f1``), precision,
    recall, ANLS on names, null accuracy, exact ``field_acc`` and per-field
    markers; the quality judge's markers when ``with_quality``."""
    ask = BY_PROMPT.get(str(row.get("prompt") or ""))
    if ask is None:
        return {
            "reward": 0.0,
            "reason": "prompt not in the frozen set",
            "markers": {"field_acc": 0.0},
        }
    final = str(row.get("final_text") or "")
    answer, why = docs.parse_answer(final)
    g = docs.grade(ask, answer)
    q = quality(row, ask, answer) if with_quality else {}
    if answer is None:
        reason = f"{why}: every field scored wrong"
    elif not g["wrong"]:
        reason = "every field right"
    else:
        reason = "; ".join(
            f"{f}: gold {ask['gold'][f]!r}, got {answer.get(f)!r}" for f in g["wrong"]
        )
    missed = [s for s in QUALITY if (q.get("markers") or {}).get(f"rubric:{s}") == 0.0]
    reason = f"field F1 {g['field_f1']:.2f} | {reason}"
    if q:
        reason += f" | quality {q.get('reward')}" + (
            f" (missed: {', '.join(missed)})" if missed else ""
        )
    markers = dict(g["markers"])
    markers["parsed"] = 1.0 if answer is not None else 0.0
    if q.get("reward") is not None:
        markers["quality"] = float(q["reward"])
        markers.update(q.get("markers") or {})
    for slug, v in program_checks(ask, answer, final).items():
        markers[f"program:{slug}"] = v
    return {
        "reward": g["reward"],
        "reason": reason,
        "markers": markers,
        "judge_reason": q.get("reason"),
        "judge_cached": bool(q.get("cached")),
    }
