"""Rubrics: prompt-specific criteria as an object, a judge that scores
them one by one, and a writer that drafts them.

Lambert 2025, chapter Synthetic Data and Distillation ("Rubrics:
Prompt-Specific AI Feedback for Training"): for prompts with no verifiable
answer, the reward is a checklist. Each item is a hard rule (miss it and the
reply fails), a principle (weighted quality), or a pitfall (a common mistake
that costs points). A general rubric per domain seeds it; a supervising model
writes the fine-grained items per prompt, and a judge marks each item rather
than guessing one number.

Here a ``Rubric`` is a tuple of ``Criterion`` with a content hash as its
``version``; it lives on the row's ``privileged`` block, which the judge
sees and no export ever carries. ``rubric_judge`` asks the model for one
verdict per criterion, turns them into a reward with ``Rubric.score``,
and puts every criterion on the row as a marker (``rubric:<slug>``), so
``marker_summary`` and ``delta_report`` read which items moved.
``write_rubrics`` is that chapter's rubric-writer prompt: the question, a
reference answer when there is one, and the domain's guidance in; a JSON
list of criteria out.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..generate.agents import complete, parse_backend_spec
from ..generate.typesafe_backend import is_typesafe_url
from . import decision_judge
from .grade_llm import (
    JUDGE_TEMPERATURE,
    _render_payload,
    judge_spec,
    judge_version,
    warm_judge,
)

Kind = Literal["hard", "principle", "pitfall"]
KINDS: tuple[str, ...] = ("hard", "principle", "pitfall")

#: the category prefixes of Lambert 2025, chapter Synthetic Data and
#: Distillation, mapped onto the three kinds
_PREFIX_KIND = {
    "essential": "hard",
    "hard rule": "hard",
    "important": "principle",
    "optional": "principle",
    "principle": "principle",
    "pitfall": "pitfall",
}
_PREFIX = re.compile(
    r"^\s*\[?(essential|important|optional|pitfall|hard rule|principle)\]?\s*(criteria|criterion)?\s*[:\-]?\s*",
    re.I,
)


#: the same chapter's bracketed tags, anywhere in a title: "Five items
#: [Hard Rule]"
_TAG = re.compile(r"\[(essential|important|optional|pitfall|hard rule|principle)\]", re.I)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "criterion"


@dataclass(frozen=True)
class Criterion:
    """One rubric item. ``weight`` is a positive magnitude; a ``pitfall``
    subtracts it when the reply exhibits the mistake, a ``principle`` adds
    it when met, and a ``hard`` rule missed fails the whole reply."""

    title: str
    description: str = ""
    weight: float = 1.0
    kind: str = "principle"

    def __post_init__(self) -> None:
        if not str(self.title).strip():
            raise ValueError("a criterion needs a title")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}; got {self.kind!r}")
        if not (float(self.weight) > 0):
            raise ValueError("weight is a positive magnitude; a pitfall subtracts it")

    @property
    def slug(self) -> str:
        return _slug(self.title)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "weight": float(self.weight),
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Criterion:
        """Accepts this shape, and the one Lambert 2025 (chapter Synthetic
        Data and Distillation) writes: a ``weight`` below zero or a
        description starting ``Pitfall Criteria:`` is a pitfall, ``Essential``
        or ``[Hard Rule]`` is hard, ``Important`` / ``Optional`` / ``[Principle]``
        a principle."""
        title = str(raw.get("title") or "").strip()
        description = str(raw.get("description") or "").strip()
        kind = str(raw.get("kind") or "").strip().lower()
        weight_raw = raw.get("weight", 1.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError):
            weight = 1.0
        if not kind:
            m = (
                _PREFIX.match(description)
                or _PREFIX.match(title)
                or _TAG.search(title)
                or _TAG.search(description)
            )
            if m:
                kind = _PREFIX_KIND[m.group(1).lower()]
            elif weight < 0:
                kind = "pitfall"
            else:
                kind = "principle"
        if kind not in KINDS:
            kind = "principle"
        if weight < 0:
            weight = -weight
        if weight == 0:
            weight = 1.0
        return cls(title=title, description=description, weight=weight, kind=kind)


@dataclass(frozen=True)
class Rubric:
    criteria: tuple[Criterion, ...]
    source: str = "hand"
    domain: str = ""
    notes: str = ""
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("a rubric needs at least one criterion")
        seen: set[str] = set()
        for c in self.criteria:
            if c.slug in seen:
                raise ValueError(f"two criteria share the slug {c.slug!r}; retitle one")
            seen.add(c.slug)

    @property
    def version(self) -> str:
        """Content hash: an edited criterion is a new rubric, and a judge
        that read it is a new judge."""
        if "version" not in self._cache:
            blob = json.dumps([c.to_dict() for c in self.criteria], sort_keys=True)
            self._cache["version"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
        return self._cache["version"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": [c.to_dict() for c in self.criteria],
            "source": self.source,
            "domain": self.domain,
            "notes": self.notes,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Rubric:
        if isinstance(raw, Mapping):
            items = raw.get("criteria") or []
            return cls(
                criteria=tuple(Criterion.from_dict(c) for c in items),
                source=str(raw.get("source") or "hand"),
                domain=str(raw.get("domain") or ""),
                notes=str(raw.get("notes") or ""),
            )
        return cls(criteria=tuple(Criterion.from_dict(c) for c in raw))

    def checklist(self) -> str:
        """The rubric as the judge reads it."""
        lines = []
        for i, c in enumerate(self.criteria, 1):
            tag = {"hard": "Hard rule", "principle": "Principle", "pitfall": "Pitfall"}[c.kind]
            desc = f" {c.description}" if c.description else ""
            lines.append(f"{i}. [{tag}, weight {c.weight:g}] {c.title}:{desc}".rstrip(":"))
        return "\n".join(lines)

    def _resolve(self, results: Mapping[str, Any]) -> dict[str, Any]:
        """Verdicts keyed by criterion slug. A key may be the item number
        (1-based, as the judge is asked to answer), the title, its slug,
        or a paraphrase that starts with or contains the title; the judge
        rewrites titles often enough that exact matching left items
        unanswered."""
        slugs = [c.slug for c in self.criteria]
        by_key: dict[str, Any] = {}
        for key, value in results.items():
            text = str(key).strip()
            number = text.rstrip(".)")
            if number.isdigit() and 1 <= int(number) <= len(slugs):
                by_key[slugs[int(number) - 1]] = value
                continue
            slug = _slug(text)
            if slug in slugs:
                by_key[slug] = value
                continue
            matches = [s for s in slugs if s.startswith(slug) or slug.startswith(s) or s in slug]
            if len(matches) == 1:
                by_key[matches[0]] = value
        return by_key

    def score(self, results: Mapping[str, Any]) -> dict[str, Any]:
        """Reward from one verdict per criterion (keyed by title or slug;
        a truthy value means the reply meets a hard rule or principle, or
        exhibits a pitfall).

        A missed hard rule is a 0. Otherwise the reward is the met
        principle weight minus the exhibited pitfall weight, over the
        total principle weight, clamped to [0, 1]; with no principle the
        reward is 1 minus the pitfall share. Criteria the judge did not
        answer count as not met (and not exhibited) and are listed.
        """
        by_key = self._resolve(results)
        met: list[str] = []
        missed: list[str] = []
        hard_failed: list[str] = []
        pitfalls_hit: list[str] = []
        unanswered: list[str] = []
        principle_total = 0.0
        principle_met = 0.0
        pitfall_total = 0.0
        pitfall_hit = 0.0
        for c in self.criteria:
            answered = c.slug in by_key
            value = bool(by_key.get(c.slug, False))
            if not answered:
                unanswered.append(c.title)
            if c.kind == "hard":
                (met if value else hard_failed).append(c.title)
            elif c.kind == "principle":
                principle_total += c.weight
                if value:
                    principle_met += c.weight
                    met.append(c.title)
                else:
                    missed.append(c.title)
            else:
                pitfall_total += c.weight
                if value:
                    pitfall_hit += c.weight
                    pitfalls_hit.append(c.title)
        if hard_failed:
            reward = 0.0
        elif principle_total > 0:
            reward = (principle_met - pitfall_hit) / principle_total
        elif pitfall_total > 0:
            reward = 1.0 - pitfall_hit / pitfall_total
        else:
            reward = 1.0
        reward = max(0.0, min(1.0, reward))
        return {
            "reward": round(reward, 4),
            "met": met,
            "missed": missed,
            "hard_failed": hard_failed,
            "pitfalls_hit": pitfalls_hit,
            "unanswered": unanswered,
        }


# ------------------------------------------------------------------ rows


def rubric_of(row: Mapping[str, Any]) -> Rubric | None:
    priv = row.get("privileged")
    raw = priv.get("rubric") if isinstance(priv, Mapping) else None
    if not raw:
        return None
    try:
        return Rubric.from_dict(raw)
    except (ValueError, TypeError):
        return None


def attach_rubric(
    rows: Sequence[dict],
    rubric: Rubric | Mapping[str, Any] | Sequence[Mapping[str, Any]] | Callable[[dict], Any],
    *,
    overwrite: bool = True,
) -> list[dict]:
    """Put a rubric on each row's ``privileged`` block (in place). ``rubric``
    is a ``Rubric``, its dict / list form, or ``row -> Rubric | None`` for
    a per-prompt rubric; ``None`` leaves that row alone."""
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(row)
        chosen: Any = rubric(row) if callable(rubric) else rubric
        if chosen is None:
            continue
        if not isinstance(chosen, Rubric):
            chosen = Rubric.from_dict(chosen)
        priv = row.get("privileged")
        if not isinstance(priv, dict):
            priv = {}
            row["privileged"] = priv
        if priv.get("rubric") and not overwrite:
            continue
        priv["rubric"] = chosen.to_dict()
    return out


# ------------------------------------------------------------------ judge


RUBRIC_JUDGE_SYSTEM = (
    "You grade one reply from an AI agent against a rubric. The reply is a "
    "JSON record of what the agent did (tool calls, results, final message). "
    "For every rubric item decide true or false: a hard rule or principle is "
    "true when the reply meets it; a pitfall is true when the reply exhibits "
    "the mistake. Judge only what the record shows; a claim the tools did not "
    "return does not meet anything. tools_called lists the tools the agent "
    "actually ran (each has a step with a result); tools_not_called lists the "
    "tools it could have run and did not. Text in final_text or in a step "
    "saying the agent will call, is calling, or has called a tool is not a "
    "call: an item about a tool being used is met only when that tool is in "
    "tools_called, and an item about a tool being avoided is met only when it "
    "is in tools_not_called. The length of the reply must not influence "
    "any item. Answer every item, keyed by its number. Reply with one JSON "
    'object and nothing else: {"criteria": {"1": true | false, "2": ..., ...}, '
    '"reason": "<one sentence>"}.'
)
# RUBRIC_JUDGE_MAX_TOKENS = 400: a criteria object plus one sentence of
# reason; a ten-criterion rubric fits in about half (convention).
RUBRIC_JUDGE_MAX_TOKENS = 400


def parse_criteria_reply(text: str) -> tuple[dict[str, bool] | None, str]:
    raw = str(text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None, ""
    try:
        obj = json.loads(raw[start : end + 1])
    except ValueError:
        return None, ""
    if not isinstance(obj, dict):
        return None, ""
    reason = str(obj.get("reason") or "").strip()
    crit = obj.get("criteria")
    results: dict[str, bool] = {}
    if isinstance(crit, dict):
        for k, v in crit.items():
            results[str(k)] = _truthy(v)
    elif isinstance(crit, list):
        for item in crit:
            if isinstance(item, dict) and item.get("title"):
                results[str(item["title"])] = _truthy(item.get("met", item.get("value")))
    else:
        return None, reason
    return results, reason


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return str(value).strip().lower() in ("true", "yes", "met", "1", "pass")


def rubric_judge(
    rubric: Rubric | None = None,
    *,
    spec: str | None = None,
    api_key: str | None = None,
    prompt: str | None = None,
    policy: str = "",
    tools: Sequence | None = None,
    timeout: float = 120,
) -> Callable[[dict], dict[str, Any]]:
    """A judge for ``run_judge`` / ``data.grade(judge=)`` that scores the
    rubric item by item. ``rubric`` applies to every row; without one the
    row's own ``privileged.rubric`` is used and a row with none stays
    ungraded. The result carries ``reward`` (``Rubric.score``), ``reason``,
    ``markers`` (``rubric:<slug>`` = 1.0 met / 0.0 not, and for a pitfall
    1.0 clean / 0.0 exhibited), ``criteria`` (the raw verdicts),
    ``rubric_version`` and the score breakdown. The judge's name folds the
    rubric version in when one is fixed.

    Two things about the verdict worth knowing before it is trusted. A
    rubric of plain principles scores the *mean* of its criteria, so three
    principles return 0, 1/3, 2/3 or 1, and ``judge_agreement`` /
    ``judge_trust`` count exact 0/1 rewards only: every partially met row
    is skipped, and the agreement number is read off the rows the judge
    was sure about (#345). Give each ``Criterion`` ``kind="hard"`` for a
    0/1 verdict (``Rubric.score``), or accept that ``judge_trust`` reports
    the skipped share and pulls ``ok`` when it passes
    ``MAX_SKIPPED_SHARE``. And whether a tool was *called* is handed to
    the judge as a fact, not left for it to infer: the payload carries
    ``tools_called`` (steps that returned a result) and
    ``tools_not_called``, and the system prompt says a reply that
    announces a call it never made has not made it (#346: without the
    list, a 4B judge passed 18 of 18 announced-but-never-made
    escalations). A criterion that must be exact belongs in a
    ``grader=`` that reads ``steps`` itself.

    The hosted judge scales to zero, so the first row through warms it once
    (``warm_judge``, a 600s budget) while the rest of the fan-out waits.
    Without that, ``run_judge``'s eight concurrent calls all raced a
    container that was still loading its weights and every row came back
    ``invalid_result`` with a ``TimeoutError``. Warm-up failure is not
    fatal: the rows are judged anyway and report the real error."""
    resolved = judge_spec(spec=spec)
    url, model = parse_backend_spec(resolved)
    system = str(prompt or "").strip() or RUBRIC_JUDGE_SYSTEM
    warm_lock = threading.Lock()
    warmed: list[dict] = []

    def ensure_warm() -> None:
        # once per judge, and the other workers block here rather than
        # opening their own request against a cold server
        if warmed:
            return
        with warm_lock:
            if not warmed:
                warmed.append(warm_judge(resolved, api_key=api_key))

    def judge(row: dict) -> dict[str, Any]:
        use = rubric or rubric_of(row)
        if use is None:
            return {"reward": None, "reason": "no rubric on the row", "rubric_version": None}
        record = json.loads(_render_payload(row, policy=policy, tools=tools))
        user = json.dumps({"rubric": use.checklist(), "reply": record}, default=str)
        ensure_warm()
        try:
            if is_typesafe_url(url):
                # one yes/no per rubric item, keyed by number
                results, reason = decision_judge.rubric_decision(
                    url,
                    model,
                    system=system,
                    rubric=use,
                    reply=record,
                    api_key=api_key,
                    timeout=timeout,
                )
            else:
                reply = complete(
                    url,
                    model,
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    api_key=api_key,
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=RUBRIC_JUDGE_MAX_TOKENS,
                    timeout=timeout,
                )
                results, reason = parse_criteria_reply(str(reply.get("content") or ""))
        except Exception as exc:
            return {"reward": None, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        if results is None:
            return {"reward": None, "reason": "judge reply carried no criteria object"}
        return score_with_rubric(use, results, reason=reason)

    judge.__name__ = judge_version(
        resolved, system + (f"|rubric:{rubric.version}" if rubric else "")
    )
    return judge


def score_with_rubric(
    rubric: Rubric, results: Mapping[str, Any], *, reason: str = ""
) -> dict[str, Any]:
    """A judge result from per-criterion verdicts, for a judge you wrote
    yourself: reward, markers, breakdown."""
    breakdown = rubric.score(results)
    by_slug = {k: _truthy(v) for k, v in rubric._resolve(results).items()}
    markers: dict[str, float] = {}
    for c in rubric.criteria:
        value = by_slug.get(c.slug, False)
        markers[f"rubric:{c.slug}"] = (
            (0.0 if value else 1.0) if c.kind == "pitfall" else (1.0 if value else 0.0)
        )
    return {
        "reward": breakdown["reward"],
        "reason": reason,
        "markers": markers,
        "criteria": {str(k): _truthy(v) for k, v in results.items()},
        "rubric_version": rubric.version,
        "hard_failed": breakdown["hard_failed"],
        "missed": breakdown["missed"],
        "pitfalls_hit": breakdown["pitfalls_hit"],
        "unanswered": breakdown["unanswered"],
        "n_unanswered": len(breakdown["unanswered"]),
    }


# ------------------------------------------------------------------ writer


RUBRIC_WRITER_SYSTEM = (
    "You are an expert rubric writer. Given a user's request to an AI agent "
    "(and, when present, a reference answer and the domain's guidance), write "
    "a self-contained set of criteria for judging how well a reply serves that "
    "request. Cover what the reply must do (facts it must get right, actions it "
    "must take or must not take, checks it must make), how well it does it, and "
    "the common mistakes. Each item must be judgeable from the reply alone. "
    "Choose {n_lo} to {n_hi} items by how much the request demands. Reply with "
    "one JSON array and nothing else; each object has exactly three keys: "
    '"title" (2 to 5 words), "description" (one sentence starting with its '
    "category, one of: Essential Criteria, Important Criteria, Optional Criteria, "
    'Pitfall Criteria), "weight" (1 to 5 for Essential / Important / Optional; '
    "-1 or -2 for Pitfall). Do not copy the request or the reference into the "
    "descriptions; the reference is guidance, not the only good answer."
)
# RUBRIC_WRITER_MAX_TOKENS = 1200: room for eight to twelve criteria with a
# description each; the writer stops at EOS well before it (convention).
RUBRIC_WRITER_MAX_TOKENS = 1200


def parse_rubric_reply(text: str) -> list[dict[str, Any]] | None:
    raw = str(text or "").strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except ValueError:
        return None
    if not isinstance(obj, list):
        return None
    items = [o for o in obj if isinstance(o, dict) and str(o.get("title") or "").strip()]
    return items or None


def write_rubrics(
    rows: Sequence[dict],
    *,
    spec: str | None = None,
    domain: str = "",
    n_criteria: tuple[int, int] = (3, 8),
    reference_key: str = "reference",
    overwrite: bool = False,
    concurrency: int = 8,
    api_key: str | None = None,
    timeout: float = 120,
    writer: Callable[[str], Any] | None = None,
    max_hard: int | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Draft one rubric per distinct prompt with a model and attach it to
    every row of that prompt (``privileged.rubric``, ``source="model"``).

    The writer sees the request, the row's reference answer when there is
    one (``privileged.reference`` or ``row[reference_key]``), and the
    ``domain`` guidance you give it (the general rubric Lambert 2025 seeds
    from). Rows that already carry a rubric are skipped unless
    ``overwrite``. ``writer(user_message) -> str`` replaces the model call
    for tests and for a writer of your own. Report: prompts seen, rubrics
    written, failures, mean criteria per rubric, the rubric versions.

    ``max_hard`` caps the hard rules a written rubric may carry: the
    heaviest ``max_hard`` stay hard and the rest become principles with
    their weight (``demoted_hard`` in the report). A model writer marks
    most of what it wants as Essential, and every Essential item a reply
    misses is a 0, so an uncapped rubric fails rows a binary judge passes
    (measured live: 22 of 32 rows). ``None`` keeps what the writer wrote.
    """
    resolved = judge_spec(spec=spec)
    url, model = parse_backend_spec(resolved)
    system = RUBRIC_WRITER_SYSTEM.format(n_lo=int(n_criteria[0]), n_hi=int(n_criteria[1]))

    def call(user: str) -> str:
        if writer is not None:
            return str(writer(user))
        reply = complete(
            url,
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            api_key=api_key,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=RUBRIC_WRITER_MAX_TOKENS,
            timeout=timeout,
        )
        return str(reply.get("content") or "")

    by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = " ".join(str(row.get("prompt") or "").lower().split())
        if key:
            by_prompt.setdefault(key, []).append(row)
    todo: list[tuple[str, list[dict]]] = []
    skipped = 0
    for key, group in by_prompt.items():
        if not overwrite and any(rubric_of(r) is not None for r in group):
            skipped += 1
            continue
        todo.append((key, group))

    demoted_by: dict[str, int] = {}

    def build(item: tuple[str, list[dict]]) -> tuple[str, Rubric | None, str]:
        key, group = item
        first = group[0]
        priv_raw = first.get("privileged")
        priv: Mapping[str, Any] = priv_raw if isinstance(priv_raw, Mapping) else {}
        reference = priv.get("reference") or first.get(reference_key) or ""
        payload: dict[str, Any] = {"request": str(first.get("prompt") or "")[:4000]}
        if reference:
            payload["reference_answer"] = str(reference)[:4000]
        if domain:
            payload["domain_guidance"] = str(domain)[:4000]
        try:
            text = call(json.dumps(payload, default=str))
        except Exception as exc:
            return key, None, f"{type(exc).__name__}: {exc}"[:200]
        items = parse_rubric_reply(text)
        if not items:
            return key, None, "writer reply carried no rubric array"
        criteria = [Criterion.from_dict(c) for c in items]
        demoted = 0
        if max_hard is not None:
            hard = sorted(
                (i for i, c in enumerate(criteria) if c.kind == "hard"),
                key=lambda i: (-criteria[i].weight, i),
            )
            for i in hard[max(0, int(max_hard)) :]:
                c = criteria[i]
                criteria[i] = Criterion(c.title, c.description, c.weight, "principle")
                demoted += 1
        try:
            rubric = Rubric(criteria=tuple(criteria), source="model", domain=str(domain))
        except ValueError as exc:
            return key, None, str(exc)[:200]
        demoted_by[key] = demoted
        return key, rubric, ""

    if todo and concurrency > 1 and len(todo) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            built = list(pool.map(build, todo))
    else:
        built = [build(t) for t in todo]
    written = 0
    failures: list[dict[str, str]] = []
    sizes: list[int] = []
    versions: dict[str, int] = {}
    groups = dict(todo)
    for key, rubric, error in built:
        if rubric is None:
            failures.append({"prompt": key[:120], "error": error})
            continue
        attach_rubric(groups[key], rubric)
        written += 1
        sizes.append(len(rubric.criteria))
        versions[rubric.version] = versions.get(rubric.version, 0) + len(groups[key])
    report = {
        "prompts": len(by_prompt),
        "written": written,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures[:10],
        "criteria_per_rubric": round(sum(sizes) / len(sizes), 2) if sizes else None,
        "max_hard": max_hard,
        "demoted_hard": sum(demoted_by.values()),
        "versions": versions,
        "writer": getattr(writer, "__name__", None) or judge_version(resolved, system),
    }
    return [r for r in rows if isinstance(r, dict)], report


__all__ = [
    "KINDS",
    "RUBRIC_JUDGE_SYSTEM",
    "RUBRIC_WRITER_SYSTEM",
    "Criterion",
    "Rubric",
    "attach_rubric",
    "parse_criteria_reply",
    "parse_rubric_reply",
    "rubric_judge",
    "rubric_of",
    "score_with_rubric",
    "write_rubrics",
]
