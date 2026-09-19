"""Inspect the agent before spending simulation budget, and report after.

``preflight`` reads an agent spec and reports what will make simulation and
grading harder: tools without result shapes, missing descriptions or
schemas, destructive tools that deserve write-discipline scenarios, thin
policies. It reports; it never fixes.

``dataset_report`` summarizes a generated or graded pool the way a person
would ask for it: rows, usable SFT examples, distinct behaviors, cells,
failure-class mix.

``classify_failure`` maps a graded failing row onto the fixed failure
vocabulary so evaluation output stays diagnosable downstream.

``coverage_gap`` takes the asks a test suite already sends and says which
parts of the agent's policy they never reach: which rules have no ask,
which tools no ask names, and which axes of the grid a hand-written suite
cannot set at all.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..defaults import MESSAGE_EXAMPLES, RULE_AXIS_CAP_REPORT, TEXT_HEURISTICS
from ..tools import schemas as _tool_schemas

#: Untested policy rules named in the preflight summary before "and N more".
_UNTESTED_SHOWN = 3

FAILURE_CLASSES = (
    "fabrication",
    "unconfirmed_write",
    "junk_output",
    "fault_dishonesty",
    "arithmetic",
    "no_attempt",
    "incompleteness",
)

# Whole words only, with ``_`` and a case change as separators: ``cancel_order``
# and ``cancelOrder`` match, ``read_runbook`` (``book``) and ``profile``
# (``file``) do not.
_DESTRUCTIVE = re.compile(
    r"(?<![a-z])(?i:cancel|delete|remove|refund|reverse|transfer|send|update|"
    r"book|create|close|merge|pay)(?![a-z])|(?<![a-z])(?i:file_)"
)

_CLASS_HINTS = (
    (
        "unconfirmed_write",
        re.compile(
            r"without (?:confirm|verif|eligib|checking|asking)|no (?:confirm|"
            r"verif)|unconfirmed|unauthori[sz]ed|executed .*(?:request|"
            r"immediately)|phantom",
            re.IGNORECASE,
        ),
    ),
    (
        "junk_output",
        re.compile(
            r"truncat|cut off|mid-sentence|degenerate|verbatim.?repeat|raw json|"
            r"empty final|garbled",
            re.IGNORECASE,
        ),
    ),
    (
        "arithmetic",
        re.compile(
            r"arithmetic|math|miscomput|calculation|rounded? (?:wrong|incorrect)|"
            r"wrong (?:number|result|total)",
            re.IGNORECASE,
        ),
    ),
    (
        "no_attempt",
        re.compile(
            r"refused a doable|false capability|claim(?:ed|s) (?:it |to be )?"
            r"(?:cannot|unable)|denies? (?:its|the) (?:own )?(?:capab|tool)",
            re.IGNORECASE,
        ),
    ),
    (
        "fault_dishonesty",
        re.compile(
            r"claim(?:ed|s) success.*(?:fail|reject|error)|hid(?:es|ing)? the "
            r"(?:error|failure)|misreport|as if it succeeded",
            re.IGNORECASE,
        ),
    ),
    (
        "incompleteness",
        re.compile(
            r"incomplete|stonewall|withh(?:e|o)ld|never (?:address|answer|"
            r"present)|refus(?:es|ed|al).*(?:present|serve)|did not (?:finish|"
            r"complete|do)",
            re.IGNORECASE,
        ),
    ),
    (
        "fabrication",
        re.compile(
            r"invent|fabricat|halluc|made.?up|adopt(?:ed|s|ing).*(?:claim|fact|"
            r"invented)|ungrounded|dismiss(?:ed|es|ing).*(?:ok|valid|result)|"
            r"sycophan|endors",
            re.IGNORECASE,
        ),
    ),
)


def _fn(tool: dict) -> dict:
    inner = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return inner if isinstance(inner, dict) else {}


# How a policy written in English names a tool: the verb it starts with,
# or a synonym, plus the nouns in the rest of the name.
_VERB_SYNONYMS: dict[str, tuple[str, ...]] = {
    "get": (
        "get",
        "look up",
        "lookup",
        "look-up",
        "check",
        "fetch",
        "retrieve",
        "view",
        "read",
        "pull",
    ),
    "lookup": (
        "look up",
        "lookup",
        "look-up",
        "check",
        "find",
        "verify",
        "authenticate",
        "identify",
    ),
    "find": ("find", "look up", "search", "check"),
    "search": ("search", "look up", "find", "check"),
    "list": ("list", "show", "check"),
    "check": ("check", "confirm", "verify", "look up"),
    "verify": ("verify", "verification", "authenticate", "confirm", "check"),
    "create": ("create", "open", "file", "start", "request", "raise", "log", "submit"),
    "open": ("open", "create", "file", "raise"),
    "initiate": ("initiate", "start", "open", "issue", "create", "process", "request"),
    "request": ("request", "ask for", "submit", "file"),
    "issue": ("issue", "send", "grant", "give"),
    "send": ("send", "email", "mail", "forward", "notify", "message"),
    "update": ("update", "change", "modify", "edit", "set", "adjust"),
    "change": ("change", "update", "modify", "switch"),
    "set": ("set", "update", "change"),
    "cancel": ("cancel", "cancellation", "void"),
    "delete": ("delete", "remove", "erase"),
    "remove": ("remove", "delete", "drop"),
    "reset": ("reset", "change"),
    "unlock": ("unlock", "lockout", "locked"),
    "escalate": ("escalate", "escalation", "hand off", "handoff", "transfer", "human"),
    "transfer": ("transfer", "escalate", "hand off", "handoff", "human"),
}
_NAME_STOP = {"to", "a", "an", "the", "of", "for", "by", "and", "or", "in", "on"}


def _name_tokens(name: str) -> list[str]:
    """``escalate_to_human`` and ``escalateToHuman`` both split the same."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    return [t for t in re.split(r"[^a-z0-9]+", spaced) if t]


def _policy_mentions(name: str, policy: str) -> bool:
    """True when the policy names the tool, literally or in plain English.

    ``get_order`` is mentioned by "Look up the order before discussing it";
    ``escalate_to_human`` by "must be escalated to a human". The literal
    snake_case name still counts. A policy that never names a tool's nouns
    is still reported.
    """
    low = str(policy or "").lower()
    if not low.strip():
        return False
    if name.lower() in low:
        return True
    tokens = _name_tokens(name)
    if not tokens:
        return False
    verb, nouns = tokens[0], [t for t in tokens[1:] if t not in _NAME_STOP]
    if not nouns:
        return bool(re.search(rf"\b{re.escape(verb)}", low))
    nouns_seen = all(re.search(rf"\b{re.escape(_stem(n))}", low) for n in nouns)
    verb_seen = any(re.search(rf"\b{re.escape(v)}", low) for v in _VERB_SYNONYMS.get(verb, (verb,)))
    return nouns_seen and verb_seen


def _rule_cap_note(shown: int, total: int, cap: int | None, *, where: str) -> str | None:
    """The line a report carries when its rule axis is shorter than the
    policy: how many clauses are on it, how many the prompt has, and the
    knob that widens it. ``None`` when nothing was dropped."""
    if cap is None or total <= shown:
        return None
    dropped = total - shown
    return (
        f"{shown} of {total} policy clauses are on the rule axis (rule_cap={cap}); the other "
        f"{dropped} are never checked, so '{shown} of {shown} rules covered' is not the "
        f"policy. Pass rule_cap=None to {where} to check every clause, or a larger number."
    )


def preflight(
    tools: Sequence[dict], system_prompt: str = "", *, rule_cap: int | None = RULE_AXIS_CAP_REPORT
) -> dict[str, Any]:
    """Spec-quality report for an agent. Report only; nothing is changed.

    ``warnings`` is the list a developer should read before generating
    thousands of rows; ``cells`` is the covering-grid size the same way
    ``recommend`` counts it. ``rules`` is every clause of the policy
    (``rule_cap=None``, the default ``RULE_AXIS_CAP_REPORT``); a number
    keeps the first that many in document order, ``n_rules_total`` says
    how many the policy has, ``rules_truncated`` whether any were left
    off, and a ``warnings`` line names the count (#391). The generation
    grid keeps its own cap (``RULE_AXIS_CAP_GRID``); ``cells`` is counted
    on the same axis ``rules`` shows.
    """
    tools = _tool_schemas(tools) or []
    from ..generate.scenarios import build_dimensions, rule_axis, scenario_regions

    tools = list(tools or [])
    policy = str(system_prompt or "")
    per_tool: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_shapes: list[str] = []
    destructive: list[str] = []
    for tool in tools:
        fn = _fn(tool)
        name = str(fn.get("name") or "")
        raw_params = fn.get("parameters")
        params: dict = raw_params if isinstance(raw_params, dict) else {}
        properties = params.get("properties")
        issues: list[str] = []
        if not str(fn.get("description") or "").strip():
            issues.append("no_description")
        # ``properties: {}`` is a declared no-argument tool, not a missing schema.
        if not isinstance(properties, dict):
            issues.append("no_parameters_schema")
        elif properties and not params.get("required"):
            issues.append("no_required_fields")
        if not fn.get("returns"):
            issues.append("no_result_shape")
            missing_shapes.append(name)
        if _DESTRUCTIVE.search(name):
            destructive.append(name)
        per_tool.append({"name": name, "issues": issues})
    if missing_shapes:
        warnings.append(
            f"{len(missing_shapes)} of {len(tools)} tools declare no result "
            f"shape ({', '.join(missing_shapes[:MESSAGE_EXAMPLES])}"
            f"{', ...' if len(missing_shapes) > MESSAGE_EXAMPLES else ''}): grounding is "
            "harder and grounding-style scaffolds can convert fabrication "
            "into refusal instead of correct service; add a `returns` key "
            "to each tool (an example result or a JSON schema)"
        )
    for entry in per_tool:
        for issue in entry["issues"]:
            if issue in {"no_description", "no_parameters_schema"}:
                warnings.append(f"tool {entry['name']}: {issue}")
    if destructive:
        warnings.append(
            f"destructive tools present ({', '.join(destructive[:6])}): "
            "write-discipline scenarios (verify-before-write, "
            "confirm-before-destructive) deserve explicit coverage"
        )
    if not policy.strip():
        warnings.append(
            "no system prompt: rule axes of the grid will be "
            "generic and grading has no policy to hold against"
        )
    elif len(policy) < THIN_POLICY_CHARS:
        warnings.append(
            f"system prompt is {len(policy)} chars: thin policies give the "
            "grid few rules to test and graders little to enforce"
        )
    unreferenced = sorted(
        entry["name"].lower()
        for entry in per_tool
        if entry["name"] and not _policy_mentions(entry["name"], policy)
    )
    dimensions = build_dimensions(tools, policy, rule_cap=rule_cap)
    rules, n_rules_total = rule_axis(policy, cap=rule_cap)
    cap_note = _rule_cap_note(len(rules), n_rules_total, rule_cap, where="preflight")
    if cap_note:
        warnings.append(cap_note)
    cells = len(scenario_regions(tools, policy, mode="sft", dimensions=dimensions))
    return {
        "n_tools": len(tools),
        "policy_chars": len(policy),
        "cells": cells,
        # The policy branches the engine extracted: the rule axis of the
        # grid, and what ``coverage_gap`` checks a test suite against.
        "rules": [str(rule) for rule in dimensions.get("rule") or []],
        "n_rules_total": n_rules_total,
        "rules_truncated": n_rules_total > len(rules),
        "rule_cap": rule_cap,
        "tools": per_tool,
        "destructive_tools": destructive,
        "missing_result_shapes": missing_shapes,
        "tools_not_mentioned_in_policy": unreferenced,
        "warnings": warnings,
        "ok": not warnings,
    }


def classify_failure(row: dict) -> str | None:
    """Fixed-vocabulary class for a failing row, from its reason and shape.

    Returns None for passing or unlabeled rows and for failures the
    heuristics cannot place (leave those for a person, do not guess).
    """
    if not isinstance(row, dict) or row.get("reward") != 0:
        return None
    existing = str(row.get("failure_class") or "").strip()
    if existing in FAILURE_CLASSES:
        return existing
    text = " ".join(
        str(row.get(k) or "") for k in ("grader_reason", "reason", "label_reason", "grade_note")
    )
    final = str(row.get("final_text") or "").strip()
    if not final or final.lower().startswith("<agent error"):
        return "junk_output"
    for name, pattern in _CLASS_HINTS:
        if pattern.search(text):
            return name
    return None


# HARD_SHARE_FLOOR = 0.3: below this share of boundary, ambiguous and
# adversarial rows the set is easy. The generator's own default mix draws
# 40% from the hard tiers (``generate.diversity.HARD_SHARE``); 0.3 is that
# less the share the mixer measurably loses to cells it never reaches, so
# a run at the default mix does not warn about itself (#319; convention
# on the exact floor).
HARD_SHARE_FLOOR = 0.3
# HARD_SHARE_MIN_ROWS = 20: rows before the hard share is worth a warning;
# under twenty one row moves the share by five points (convention).
HARD_SHARE_MIN_ROWS = 20
# UNLABELLED_SHARE_WARN = 0.1: share of rows with no stance before the
# report says their difficulty is unknown, not ordinary (convention).
UNLABELLED_SHARE_WARN = 0.1
# THIN_POLICY_CHARS = 200: a system prompt shorter than this gives the
# grid few rules to test; 200 characters is two or three rules
# (convention, untested).
THIN_POLICY_CHARS = 200


def dataset_report(
    rows: Sequence[dict],
    *,
    tools: Sequence[dict] | None = None,
    system_prompt: str = "",
    hard_share_floor: float = HARD_SHARE_FLOOR,
) -> dict[str, Any]:
    """One report a developer reads after simulate/grade: size, signal, mix.
    ``hard_share_floor`` (``HARD_SHARE_FLOOR``, 0.3) is the share of hard-
    tier rows under which the set is called easy."""
    tools = _tool_schemas(tools)
    from ..generate.coverage import cell_key
    from ..generate.diversity import behavior_tier
    from .grading import behavior_signature

    if isinstance(rows, (str, bytes)):
        raise TypeError(
            "dataset_report takes the rows, not a dataset id: pull them first, "
            f'wai.dataset_report(wai.pull("{rows if isinstance(rows, str) else "ds_..."}"))'
        )
    given = list(rows)
    rows = [r for r in given if isinstance(r, dict)]
    if given and not rows:
        raise TypeError(
            f"dataset_report takes a sequence of row dicts; got {type(given[0]).__name__} items"
        )
    labeled = [r for r in rows if r.get("reward") in (0, 1)]
    passes = [r for r in labeled if r["reward"] == 1]
    fails = [r for r in labeled if r["reward"] == 0]
    prompts = {" ".join(str(r.get("prompt") or "").lower().split()) for r in rows}
    behaviors = {behavior_signature(r) for r in rows}
    cells = {cell_key(r) for r in rows if r.get("scenario_dimensions")}
    classes: Counter[str] = Counter()
    for r in fails:
        classes[classify_failure(r) or "unclassified"] += 1
    junk = sum(1 for r in passes if not str(r.get("final_text") or "").strip())
    # Difficulty mix. behavior_tier maps a missing stance to "ordinary", which
    # is right for sampling and wrong for a report: an unlabelled cell would
    # count as evidence the easy tier was covered. Count it separately.
    # Same direction as the run's search["tier_mix"] (hard_share, from #320,
    # which merges first): counts here, shares there.
    tiers: Counter[str] = Counter()
    tier_labeled: dict[str, list[int]] = {}
    for r in rows:
        assignment = r.get("scenario_dimensions") or {}
        has_stance = bool(assignment.get("stance") or assignment.get("user_behavior"))
        tier = behavior_tier(assignment) if has_stance else "unlabelled"
        tiers[tier] += 1
        if r.get("reward") in (0, 1):
            tier_labeled.setdefault(tier, []).append(int(r["reward"]))
    hard = sum(tiers[t] for t in ("boundary", "ambiguous", "adversarial"))
    hard_share = round(hard / len(rows), 3) if rows else None

    report: dict[str, Any] = {
        "rows": len(rows),
        "unique_prompts": len(prompts),
        "labeled": len(labeled),
        "passes": len(passes),
        "fails": len(fails),
        "fail_rate": round(len(fails) / len(labeled), 3) if labeled else None,
        "usable_sft": len(passes) - junk,
        "distinct_behaviors": len(behaviors),
        "cells_touched": len(cells),
        "failure_classes": dict(classes.most_common()),
        "tier_counts": dict(tiers.most_common()),
        "hard_share": hard_share,
        "tier_fail_rate": {
            t: round(1 - sum(v) / len(v), 3) for t, v in sorted(tier_labeled.items()) if v
        },
        # Always a list, like every other report's warnings; the tool
        # preflight's own list is preflight_warnings.
        "warnings": [],
    }
    if (
        hard_share is not None
        and hard_share < hard_share_floor
        and len(rows) >= HARD_SHARE_MIN_ROWS
    ):
        report["warnings"].append(
            f"easy set: {hard_share:.0%} of rows are boundary, ambiguous or adversarial. "
            "An ordinary ask is the one a base already passes, so a set like this reports "
            "a null whatever the policy does. Raise the share: "
            "simulate(..., hard_share=0.7) asks for 70% from the hard tiers (rows the "
            "mixer never sees keep the drawn share below the ask); "
            "dimensions={'stance': ['boundary', 'ambiguous', 'adversarial']} pins the axis "
            "to hard tiers only (difficulty filtering, Lambert 2025, chapter "
            "Reasoning)."
        )
    if tiers.get("unlabelled") and tiers["unlabelled"] / max(1, len(rows)) > UNLABELLED_SHARE_WARN:
        report["warnings"].append(
            f"{tiers['unlabelled']} rows carry no stance, so their difficulty is unknown, "
            "not ordinary. Label it with dimensions={'stance': [...]} on the run, or "
            "read hard_share as a floor."
        )
    if tools is not None:
        pre = preflight(tools, system_prompt)
        report["cells_total"] = pre["cells"]
        report["preflight_warnings"] = pre["warnings"]
    return report


def format_dataset_report(report: dict[str, Any]) -> str:
    """The report as the block a person actually reads."""
    lines = [
        f"Generated:            {report.get('rows', 0):>6}",
        f"Unique prompts:       {report.get('unique_prompts', 0):>6}",
        f"Labeled:              {report.get('labeled', 0):>6}",
        f"Usable SFT examples:  {report.get('usable_sft', 0):>6}",
        f"Failures:             {report.get('fails', 0):>6}"
        + (f"  ({report['fail_rate']:.0%})" if report.get("fail_rate") is not None else ""),
        f"Distinct behaviors:   {report.get('distinct_behaviors', 0):>6}",
    ]
    if report.get("cells_total") is not None:
        lines.append(
            f"Cells touched:        {report.get('cells_touched', 0):>6} of {report['cells_total']}"
            "  (cells of the 6-axis grid: training-data coverage, not policy coverage)"
        )
    classes = report.get("failure_classes") or {}
    if classes:
        lines.append("Top failures:")
        total = sum(classes.values()) or 1
        for name, n in list(classes.items())[:6]:
            lines.append(f"  {name:<20} {n:>4}  ({n / total:.0%})")
    if report.get("hard_share") is not None:
        lines.append(
            f"Hard tiers:           {report['hard_share']:>6.0%}"
            "  (boundary, ambiguous, adversarial rows)"
        )
    for warning in (report.get("warnings") or [])[:4]:
        lines.append(f"! {warning}")
    for warning in (report.get("preflight_warnings") or [])[:4]:
        lines.append(f"! {warning}")
    return "\n".join(lines)


# ------------------------------------------------------------ coverage_gap

#: A literal shorter than this, or with no space in it, is not an ask.
_ASK_MIN_CHARS = 15

#: A rule clause that carries one of these words, or an amount, is a
#: *branch*: it applies only to asks that match its condition, so it needs
#: an ask aimed at it. A clause without one is a standing rule that every
#: ask reaching its tool puts in play ("never invent order details"). A
#: bare number is not enough: "Today is 2026-09-17" is a fact, not a branch.
_BRANCH_WORDS = frozenset(
    {
        "only",
        "over",
        "above",
        "under",
        "below",
        "within",
        "outside",
        "beyond",
        "exceed",
        "exceeds",
        "exceeding",
        "unless",
        "except",
        "more",
        "less",
        "limit",
        "limits",
        "threshold",
        "maximum",
        "minimum",
        "least",
        "most",
        "than",
    }
)
_AMOUNT = re.compile(r"[$%]|\bpercent\b")
_GAP_WORD = re.compile(r"[a-z0-9$][a-z0-9$'-]*")
#: Function words dropped before overlapping a rule with an ask. Domain
#: nouns (``order``, ``refund``) are kept on purpose: they are the overlap.
_GAP_STOP = frozenset(
    {
        "a",
        "about",
        "all",
        "also",
        "always",
        "an",
        "and",
        "any",
        "are",
        "as",
        "ask",
        "asked",
        "asks",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "down",
        "each",
        "every",
        "for",
        "from",
        "get",
        "give",
        "given",
        "gives",
        "got",
        "had",
        "has",
        "have",
        "having",
        "hello",
        "here",
        "hey",
        "hi",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "let",
        "like",
        "made",
        "make",
        "may",
        "me",
        "might",
        "mine",
        "must",
        "my",
        "need",
        "needs",
        "never",
        "no",
        "not",
        "now",
        "of",
        "off",
        "on",
        "one",
        "onto",
        "or",
        "our",
        "out",
        "please",
        "said",
        "say",
        "says",
        "shall",
        "should",
        "so",
        "some",
        "still",
        "take",
        "takes",
        "tell",
        "than",
        "thank",
        "thanks",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "told",
        "too",
        "two",
        "up",
        "us",
        "very",
        "want",
        "wants",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "without",
        "would",
        "yet",
        "you",
        "your",
        "yours",
    }
)


def _stem(word: str) -> str:
    """``refunds`` and ``orders`` overlap ``refund`` and ``order``."""
    return (
        word[:-1] if len(word) >= TEXT_HEURISTICS.plural_min_chars and word.endswith("s") else word
    )


def _content_words(text: str) -> set[str]:
    return {
        _stem(word)
        for word in _GAP_WORD.findall(str(text or "").lower())
        if len(word) >= TEXT_HEURISTICS.gap_word_min_chars and word not in _GAP_STOP
    }


def _is_branch_rule(rule: str) -> bool:
    text = str(rule or "").lower()
    if _AMOUNT.search(text):
        return True
    return bool(_BRANCH_WORDS & set(_GAP_WORD.findall(text)))


def _ask_names_tool(ask: str, name: str) -> bool:
    """Does this ask name or imply this tool?

    Three ways, loosest last: the literal tool name, the policy-style
    "verb plus nouns" match ``preflight`` already uses, and the tool's own
    nouns on their own. The last one is there because asks are written
    from the user's side: "I want a refund for order A1001" never says
    ``issue_refund``, it says the object the tool acts on.
    """
    low = str(ask or "").lower()
    if not low.strip():
        return False
    if name.lower() in low:
        return True
    if _policy_mentions(name, ask):
        return True
    tokens = _name_tokens(name)
    nouns = [t for t in tokens[1:] if t not in _NAME_STOP] or tokens[:1]
    if not nouns:
        return False
    return all(re.search(rf"\b{re.escape(_stem(noun))}", low) for noun in nouns)


def _ask_stance(ask: str) -> str:
    """The stance axis value this ask's words show, from the engine's own
    writer checks. Only three of the ten stances leave a mark in text;
    the rest are set by the situation, not readable from the ask."""
    from ..generate.generator import _ADVERSARIAL, _FRUSTRATED, _IMPATIENT

    text = str(ask or "")
    if _ADVERSARIAL.search(text):
        return "adversarial"
    if _FRUSTRATED.search(text):
        return "retry"
    if _IMPATIENT.search(text):
        return "hurried"
    return "ordinary"


def _asks_from_python(source: str) -> list[str]:
    """String literals in a .py file that look like the asks it sends.

    A heuristic, and only a heuristic: a literal counts when it is passed
    to a call or sits in a list, tuple or set, is longer than 15
    characters and contains a space. That finds the asks in a normal test
    file (``agent("refund order A1001")``, ``PROMPTS = [...]``) and it
    will also pick up a long string that is not an ask. Read the report's
    ``asks`` list before trusting the counts.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[str] = []

    def collect(node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                text = child.value.strip()
                if len(text) > _ASK_MIN_CHARS and " " in text:
                    found.append(text)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                collect(arg)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            collect(node)
    out: list[str] = []
    seen: set[str] = set()
    for text in found:
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _normalize_asks(asks: Any) -> list[str]:
    """The asks a suite sends: prompt strings, rows, or a file of either."""
    from pathlib import Path as _Path

    if isinstance(asks, (str, _Path)):
        path = _Path(asks)
        suffix = path.suffix.lower()
        if suffix == ".py":
            return _asks_from_python(path.read_text(encoding="utf-8"))
        if suffix in {".jsonl", ".json"}:
            from .quality import load_jsonl

            return [str(r.get("prompt") or "") for r in load_jsonl(path) if isinstance(r, dict)]
        raise ValueError(
            f"asks={asks!r} is not a .py or .jsonl file: pass a list of prompt "
            "strings, a list of rows with a 'prompt' key, or a path to a .py "
            "or .jsonl file holding either"
        )
    out: list[str] = []
    for item in asks or []:
        text = str(item.get("prompt") or "") if isinstance(item, dict) else str(item or "")
        if text.strip():
            out.append(text)
    return out


def _short_rule(text: str, width: int = 62) -> str:
    one = " ".join(str(text or "").split())
    return one if len(one) <= width else one[: width - 3].rstrip(" ,;:") + "..."


def coverage_gap(
    asks: Any,
    *,
    tools: Sequence[dict],
    system_prompt: str = "",
    rows: Sequence[dict] | None = None,
    rule_cap: int | None = RULE_AXIS_CAP_REPORT,
) -> dict[str, Any]:
    """List the parts of an agent's policy that the asks you already send never reach.

    Reach for it before writing situations, with the test suite you
    already have: it says which tools and which policy rules no ask
    exercises, in the engine's own vocabulary. It returns a dict:
    ``untested_rules`` and ``untested_tools`` (the lists worth reading),
    ``rules`` (the policy clauses found), ``axes`` (each axis with the
    count per value), ``stances``, ``pressure_asks``, ``single_shot``,
    ``per_ask`` (where each ask landed), ``notes``, ``summary`` and
    ``n_asks``. ``format_coverage_gap(report)`` prints it.

    * ``asks``: what a suite asks the agent: a list of prompt strings, a
      list of rows carrying ``prompt``, or a path to a ``.py`` or
      ``.jsonl`` file holding either.
    * ``tools`` and ``system_prompt``: the agent's tool schemas and
      policy. The axes come from ``build_dimensions``, the same grid
      ``simulate`` covers: which tool, which policy rule, what stance the
      person takes, what the world looks like, what condition the tool is
      in, what happened before.
    * ``rows``: graded rollouts from a run. With them the report also
      checks the world side: rules whose rows all ended in the same tool
      fault are rules the asks reach but the fixtures never let happen
      (``rules_the_world_never_triggers``, with ``rows_per_rule`` and
      ``rules_with_no_rows``).

    Each ask is placed on the axes it touches with text heuristics, not a
    model: the tools its words name or imply, the rule clauses it shares
    words with, and the stance its words show. Two axes (``world_state``,
    ``tool_condition``) cannot be read from an ask at all: a prompt never
    says the order is missing or the tool timed out, so a hand-written
    suite leaves them at one point and ``notes`` says so.

    With ``rows`` (graded rollouts from a run) the report also checks the
    world side: rules whose rows all ended in the same tool fault are
    rules the asks reach but the fixtures never let happen.

    The rule axis is every clause of the policy (``rule_cap=None``, the
    default ``RULE_AXIS_CAP_REPORT``): a report over an existing suite
    has no grid to bound. A number keeps the first that many clauses in
    document order; ``n_rules_total`` and ``rules_truncated`` say what
    was left off and ``notes`` carries the count (#391).
    ```python
    gap = wai.coverage_gap(["Where is order 4473?", "Cancel order 9911."],
                           tools=TOOLS, system_prompt=POLICY)
    print(gap["untested_rules"], gap["untested_tools"])
    ```
    """
    tools = _tool_schemas(tools) or []
    from ..generate.scenarios import build_dimensions, rule_axis

    tool_list = list(tools or [])
    policy = str(system_prompt or "")
    ask_list = _normalize_asks(asks)
    dimensions = build_dimensions(tool_list, policy, rule_cap=rule_cap)
    _, n_rules_total = rule_axis(policy, cap=rule_cap)
    names = [n for n in (str(_fn(t).get("name") or "") for t in tool_list) if n]
    rules = [str(r) for r in dimensions.get("rule") or []]
    cap_note = _rule_cap_note(
        len(rules) if n_rules_total else 0, n_rules_total, rule_cap, where="coverage_gap"
    )
    branch = {rule: _is_branch_rule(rule) for rule in rules}
    rule_words = {rule: _content_words(rule) for rule in rules}
    rule_tools = {rule: [n for n in names if _policy_mentions(n, rule)] for rule in rules}

    touched: dict[str, Counter[str]] = {axis: Counter() for axis in dimensions}
    per_ask: list[dict[str, Any]] = []
    for ask in ask_list:
        words = _content_words(ask)
        hit_tools = [n for n in names if _ask_names_tool(ask, n)]
        hit_rules = []
        for rule in rules:
            if branch[rule]:
                reached = len(words & rule_words[rule]) >= 2  # noqa: PLR2004  # two shared words is the overlap floor (convention)
            elif rule_tools[rule]:
                reached = any(n in hit_tools for n in rule_tools[rule])
            else:
                # An unconditional rule that names no tool ("never invent
                # order details") is in play on every ask.
                reached = True
            if reached:
                hit_rules.append(rule)
        stance = _ask_stance(ask)
        values = list(hit_tools) or ["unrelated"]
        if len(hit_tools) > 1:
            values.append("multi_tool")
        for value in values:
            touched["tool"][value] += 1
        for rule in hit_rules:
            touched["rule"][rule] += 1
        touched["stance"][stance] += 1
        touched["history"]["fresh"] += 1
        per_ask.append({"ask": ask, "tools": hit_tools, "rules": hit_rules, "stance": stance})

    axes: dict[str, Any] = {}
    for axis, values in dimensions.items():
        counts = {str(v): int(touched[axis].get(str(v), 0)) for v in values}
        axes[axis] = {
            "counts": counts,
            "untouched": [value for value, n in counts.items() if not n],
        }
    untested_rules = [rule for rule in rules if not touched["rule"].get(rule)]
    untested_tools = [name for name in names if not touched["tool"].get(name)]
    repeats = Counter(" ".join(ask.lower().split()) for ask in ask_list)
    single_shot = bool(ask_list) and max(repeats.values()) == 1
    pressure = sum(1 for entry in per_ask if entry["stance"] != "ordinary")

    notes: list[str] = []
    if cap_note:
        notes.append(cap_note)
    if untested_rules:
        n = len(untested_rules)
        notes.append(
            f"{n} policy rule{'s' if n != 1 else ''} no ask reaches: write one ask "
            "per rule, or let the engine write them (simulate(seeds=asks, ...) covers "
            "the rule axis)"
        )
    notes.append(
        "world_state and tool_condition are not readable from an ask: a prompt never "
        "says the record is missing or the tool timed out, so every ask sits on one "
        "point of those two axes. Run the asks through simulate(seeds=asks, tools=..., "
        "system_prompt=...) to vary them, or add a fixture case per branch"
    )
    notes.append(
        "rules are matched on the words an ask shares with the rule, so a branch only "
        "the fixture data selects (an amount, a date) reads as untested even when an "
        "ask lands on it: confirm with rows= from a run"
    )
    if single_shot:
        notes.append(
            "every ask appears once: one rollout cannot tell a flake from a failure. "
            "Roll each ask k times (repeats=k, repeat_policy='fixed') and read pass^k"
        )
    if not pressure:
        notes.append(
            "no ask is hurried, adversarial or a retry: the suite tests the agent on a "
            "good day only. Add pressure asks, or take them from the stance axis"
        )

    of_total = f" (of {n_rules_total} in the prompt)" if cap_note else ""
    summary_bits = [
        f"{len(ask_list)} ask{'s' if len(ask_list) != 1 else ''} cover "
        f"{len(rules) - len(untested_rules)} of {len(rules)} policy rule"
        f"{'s' if len(rules) != 1 else ''}{of_total} and "
        f"{len(names) - len(untested_tools)} of {len(names)} tool"
        f"{'s' if len(names) != 1 else ''}"
    ]
    if untested_rules:
        shown = ", ".join(_short_rule(rule) for rule in untested_rules[:_UNTESTED_SHOWN])
        more = (
            f", and {len(untested_rules) - _UNTESTED_SHOWN} more"
            if len(untested_rules) > _UNTESTED_SHOWN
            else ""
        )
        summary_bits.append(f"untested: {shown}{more}")
    if untested_tools:
        summary_bits.append("no ask reaches " + ", ".join(untested_tools))
    if not pressure:
        summary_bits.append("no ask puts the agent under pressure")
    if single_shot:
        summary_bits.append("every ask runs once")

    report: dict[str, Any] = {
        "n_asks": len(ask_list),
        "asks": list(ask_list),
        "axes": axes,
        "rules": rules,
        "n_rules_total": n_rules_total,
        "rules_truncated": bool(cap_note),
        "rule_cap": rule_cap,
        "untested_rules": untested_rules,
        "untested_tools": untested_tools,
        "stances": dict(touched["stance"]),
        "pressure_asks": pressure,
        "single_shot": single_shot,
        "per_ask": per_ask,
        "notes": notes,
        "summary": "; ".join(summary_bits),
    }
    if rows is not None:
        report.update(_rule_rows_view(rules, rows))
        if report["rules_the_world_never_triggers"]:
            n_stuck = len(report["rules_the_world_never_triggers"])
            report["notes"].append(
                f"{n_stuck} rule{'s' if n_stuck != 1 else ''} whose every row ended in "
                "the same tool fault: the asks reach the rule but the world never "
                "triggers it, so add a fixture case that lets it happen"
            )
            report["summary"] += (
                "; "
                + ", ".join(
                    _short_rule(entry["rule"])
                    for entry in report["rules_the_world_never_triggers"][:2]
                )
                + " never triggered in the world"
            )
    return report


def _rule_rows_view(rules: Sequence[str], rows: Sequence[dict]) -> dict[str, Any]:
    """Per-rule row counts, and the rules the world never let happen."""
    from .grading import NO_FAULT, trace_fault

    by_rule: dict[str, list[dict]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        dims = row.get("scenario_dimensions")
        rule = str(dims.get("rule") or "") if isinstance(dims, dict) else ""
        if not rule:
            rule = str(row.get("rule") or "")
        if rule:
            by_rule.setdefault(rule, []).append(row)
    stuck: list[dict[str, Any]] = []
    for rule, group in sorted(by_rule.items()):
        outcomes = {trace_fault(row) for row in group}
        if len(outcomes) == 1:
            only = outcomes.pop()
            if only != NO_FAULT:
                stuck.append(
                    {
                        "rule": rule,
                        "rows": len(group),
                        "outcome": only,
                        "note": (
                            "the asks reach this rule but the world never triggers it: "
                            "add a fixture case"
                        ),
                    }
                )
    return {
        "rows_per_rule": {rule: len(group) for rule, group in sorted(by_rule.items())},
        "rules_with_no_rows": [rule for rule in rules if not by_rule.get(rule)],
        "rules_the_world_never_triggers": stuck,
    }


def format_coverage_gap(report: dict[str, Any]) -> str:
    """The gap report as the block a person actually reads."""
    rules = list(report.get("rules") or [])
    untested_rules = list(report.get("untested_rules") or [])
    untested_tools = list(report.get("untested_tools") or [])
    tool_counts = ((report.get("axes") or {}).get("tool") or {}).get("counts") or {}
    n_tools = sum(1 for name in tool_counts if name not in {"unrelated", "multi_tool"})
    lines = [
        str(report.get("summary") or ""),
        "",
        f"asks                  {report.get('n_asks', 0)}"
        + ("  (each one once)" if report.get("single_shot") else ""),
        f"policy rules covered  {len(rules) - len(untested_rules)} of {len(rules)}"
        + (
            f" (of {report['n_rules_total']} in the prompt; rule_cap={report.get('rule_cap')})"
            if report.get("rules_truncated")
            else ""
        ),
        f"tools covered         {n_tools - len(untested_tools)} of {n_tools}",
        "stance                "
        + (
            ", ".join(f"{name} {n}" for name, n in sorted((report.get("stances") or {}).items()))
            or "none"
        ),
    ]
    if untested_rules:
        lines.append("untested rules")
        for rule in untested_rules:
            lines.append(f"  - {_short_rule(rule, 72)}")
    if untested_tools:
        lines.append("untested tools        " + ", ".join(untested_tools))
    for entry in report.get("rules_the_world_never_triggers") or []:
        lines.append(
            f"world never triggers  {_short_rule(entry['rule'], 52)} "
            f"({entry['rows']} rows, all {entry['outcome']})"
        )
    for note in report.get("notes") or []:
        lines.append(f"! {note}")
    return "\n".join(lines)


__all__ = [
    "FAILURE_CLASSES",
    "classify_failure",
    "coverage_gap",
    "dataset_report",
    "format_coverage_gap",
    "format_dataset_report",
    "preflight",
]
