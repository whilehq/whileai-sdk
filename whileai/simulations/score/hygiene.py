"""Row hygiene a grouped RL update or a rejection-sampling pass cares about.

Three checks, all report-first (rlhf-book ch. 9 and 12: dedupe and strong
quality filters are what keep synthetic data out of the collapse regime;
ch. 6 and 7: overlong filtering; our own reward-hack scan from the
RLVR signal work: rank candidate features by correlation with reward and
look at what wins).

* ``dedupe_groups``: within one ask, rollouts that are the same
  trajectory (same behavior signature and same visible reply) add
  nothing to a group-relative advantage and, when they carry different
  rewards, are pure judge noise. Drop the repeats, keep the first.
* ``near_duplicate_prompts``: asks that are near-copies of each other
  (token-set Jaccard) weight one situation several times over. Report
  only; paraphrases are deliberate in ``mode="sft"``.
* ``length_report`` / ``reward_correlations``: rollouts that end
  mid-sentence, groups whose reply lengths are far apart, and rewards
  that track length or tool count instead of the behavior. Correlation
  above ``HACK_THRESHOLD`` is a warning, not a drop: the fix is the
  judge, not the rows.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from .grading import behavior_signature, dead_tools, dead_tools_note, looks_finished, tool_outcomes
from .optimize import _binary_label, _messages

#: |corr(reward, feature)| at or above this is flagged. Chosen from the
#: RLVR signal sweep where the endorsed feature cleared 0.5 and delimiter
#: hacks sat near 0.9; 0.3 catches the hack before it dominates.
HACK_THRESHOLD = 0.3
#: max / median reply length within one ask before the spread is flagged.
DEFAULT_MAX_SPREAD = 4.0
#: token-set Jaccard at or above this makes two asks near-duplicates.
NEAR_DUP_JACCARD = 0.8

_WORD = re.compile(r"[a-z0-9]+")


def _norm_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _tokens(text: Any) -> set[str]:
    return set(_WORD.findall(str(text or "").lower()))


def _group(rows: Sequence[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if isinstance(row, dict):
            groups.setdefault(str(row.get("prompt") or ""), []).append(row)
    return groups


def reply_length(row: dict) -> int:
    """Characters the agent produced: every assistant turn plus the final
    reply when it is not already the last turn. Tool output is not the
    agent's and is excluded (it is masked from the loss too)."""
    total = 0
    last = ""
    for message in _messages(row):
        if str(message.get("role") or "") == "assistant":
            content = str(message.get("content") or "")
            total += len(content)
            last = content
    final = str(row.get("final_text") or "")
    if final and final != last:
        total += len(final)
    return total


def tool_calls(row: dict) -> int:
    """Tool calls in a row, whichever shape it arrived in: engine ``steps``
    (tool/arguments/result) or a platform pull's ``tool_trace``
    (tool/input/output)."""
    steps = row.get("steps") or row.get("tool_trace") or []
    return sum(1 for s in steps if isinstance(s, dict) and s.get("tool"))


def assistant_turns(row: dict) -> int:
    """Assistant turns in the row's conversation."""
    return sum(1 for m in _messages(row) if str(m.get("role") or "") == "assistant")


def is_truncated(row: dict) -> bool:
    """A reply that stops without reaching its end (``looks_finished``:
    terminal punctuation or a sign-off), or one the grader already called
    truncated. Short replies are given the benefit of the doubt: a
    one-line answer often ends on a number or a name."""
    reason = _norm_text(row.get("reason") or row.get("grader_reason"))
    if "truncat" in reason or "cut off" in reason:
        return True
    final = str(row.get("final_text") or "").rstrip()
    return len(final) >= 200 and not looks_finished(final)


def dedupe_groups(rows: Sequence[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Drop repeat trajectories within one ask. Keeps the first of each
    (behavior signature, normalized reply) pair. Reports how many pairs
    disagreed on reward, which is label noise the judge introduced."""
    kept: list[dict] = []
    n_dropped = 0
    groups_affected: set[str] = set()
    conflicting = 0
    seen: dict[tuple[str, str, str], int | None] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "")
        key = (prompt, behavior_signature(row), _norm_text(row.get("final_text")))
        if key in seen:
            n_dropped += 1
            groups_affected.add(prompt)
            first = seen[key]
            label = _binary_label(row)
            if first is not None and label is not None and first != label:
                conflicting += 1
            continue
        seen[key] = _binary_label(row)
        kept.append(row)
    n = sum(1 for r in rows if isinstance(r, dict))
    return kept, {
        "n": n,
        "n_kept": len(kept),
        "n_dropped": n_dropped,
        "groups_affected": len(groups_affected),
        "duplicate_rate": (n_dropped / n) if n else 0.0,
        "conflicting_rewards": conflicting,
    }


def near_duplicate_prompts(
    rows: Sequence[dict], *, threshold: float = NEAR_DUP_JACCARD
) -> dict[str, Any]:
    """Pairs of distinct asks whose token sets overlap at or above
    ``threshold``. Quadratic in the number of asks; fine for the few
    thousand a run produces. Report only."""
    prompts = list(_group(rows))
    toks = [_tokens(p) for p in prompts]
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(prompts)):
        if not toks[i]:
            continue
        for j in range(i + 1, len(prompts)):
            if not toks[j]:
                continue
            inter = len(toks[i] & toks[j])
            if not inter:
                continue
            jac = inter / len(toks[i] | toks[j])
            if jac >= threshold:
                pairs.append((prompts[i], prompts[j], round(jac, 3)))
    involved = {p for a, b, _ in pairs for p in (a, b)}
    return {
        "n_prompts": len(prompts),
        "n_pairs": len(pairs),
        "n_prompts_involved": len(involved),
        "threshold": threshold,
        "examples": [{"a": a[:80], "b": b[:80], "jaccard": jac} for a, b, jac in pairs[:5]],
    }


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def length_report(
    rows: Sequence[dict], *, max_spread: float = DEFAULT_MAX_SPREAD
) -> dict[str, Any]:
    """Truncated rollouts and asks whose reply lengths are far apart.
    ``max_spread`` is max / median within one ask."""
    lengths = [reply_length(r) for r in rows if isinstance(r, dict)]
    truncated = [r for r in rows if isinstance(r, dict) and is_truncated(r)]
    spread_groups = 0
    multi = 0
    for members in _group(rows).values():
        if len(members) < 2:
            continue
        multi += 1
        group_lengths = [reply_length(r) for r in members]
        med = _median(group_lengths)
        if med > 0 and max(group_lengths) / med > max_spread:
            spread_groups += 1
    return {
        "n": len(lengths),
        "median_chars": _median(lengths),
        "max_chars": max(lengths) if lengths else 0,
        "n_truncated": len(truncated),
        "n_groups_multi": multi,
        "n_groups_wide_spread": spread_groups,
        "max_spread": max_spread,
    }


def drop_truncated(rows: Sequence[dict]) -> tuple[list[dict], dict[str, Any]]:
    kept = [r for r in rows if isinstance(r, dict) and not is_truncated(r)]
    n = sum(1 for r in rows if isinstance(r, dict))
    return kept, {"n": n, "n_kept": len(kept), "n_dropped": n - len(kept)}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def reward_correlations(
    rows: Sequence[dict], *, threshold: float = HACK_THRESHOLD
) -> dict[str, Any]:
    """corr(reward, feature) for the cheap features a judge tends to
    reward by accident: reply length, tool-call count, assistant turns,
    and the over-optimization signatures of rlhf-book ch. 14 (boilerplate,
    hedging, sycophancy, refusal phrases, 1 when present; see
    ``score.style``), plus every trajectory flag that fired on any row
    (``lie.*``, ``hack.*``, ``risk.*``; see ``score.trace``). Any |corr|
    at or above ``threshold`` is flagged. A negative tool-count
    correlation means the reward pays the policy to do less; a positive
    phrase or flag correlation means it pays for the tic or the fake."""
    from .style import STYLE_FEATURES, assistant_text, phrase_hits

    graded = [r for r in rows if isinstance(r, dict) and _binary_label(r) is not None]
    labels = [float(_binary_label(r) or 0) for r in graded]
    texts = [assistant_text(r) for r in graded]
    features = {
        "reply_length": [float(reply_length(r)) for r in graded],
        "tool_calls": [float(tool_calls(r)) for r in graded],
        "assistant_turns": [float(assistant_turns(r)) for r in graded],
    }
    for name, plist in STYLE_FEATURES.items():
        features[name] = [1.0 if phrase_hits(t, plist) else 0.0 for t in texts]
    from .trace import FLAGS, trace_flags

    fired = [trace_flags(r) for r in graded]
    for name in FLAGS:
        column = [1.0 if name in flags else 0.0 for flags in fired]
        if any(column):
            features[name] = column
    corr = {name: pearson(values, labels) for name, values in features.items()}
    flagged = {
        name: value for name, value in corr.items() if value is not None and abs(value) >= threshold
    }
    return {
        "n_graded": len(graded),
        "threshold": threshold,
        "correlations": {k: (round(v, 3) if v is not None else None) for k, v in corr.items()},
        "flagged": {k: round(v, 3) for k, v in flagged.items()},
    }


def hygiene_warnings(
    *,
    duplicates: dict[str, Any] | None = None,
    near_dups: dict[str, Any] | None = None,
    lengths: dict[str, Any] | None = None,
    correlations: dict[str, Any] | None = None,
    scan: dict[str, Any] | None = None,
) -> list[str]:
    """One line per finding a person should act on. Empty when clean.
    ``scan`` is a ``hack_scan`` report; its warnings are appended as is."""
    out: list[str] = []
    if duplicates and duplicates.get("n_dropped"):
        line = (
            f"{duplicates['n_dropped']} duplicate rollout(s) within "
            f"{duplicates['groups_affected']} ask(s) dropped"
        )
        if duplicates.get("conflicting_rewards"):
            line += (
                f"; {duplicates['conflicting_rewards']} identical trajectories got different "
                "rewards, which is judge noise"
            )
        out.append(line)
    if near_dups and near_dups.get("n_pairs"):
        out.append(
            f"{near_dups['n_pairs']} near-duplicate ask pair(s) (Jaccard >= "
            f"{near_dups['threshold']}) weight the same situation more than once"
        )
    if lengths:
        if lengths.get("n_truncated"):
            out.append(f"{lengths['n_truncated']} rollout(s) end mid-reply (truncated)")
        if lengths.get("n_groups_wide_spread"):
            out.append(
                f"{lengths['n_groups_wide_spread']} ask(s) have a reply more than "
                f"{lengths['max_spread']:g}x the ask's median length"
            )
    if correlations and correlations.get("flagged"):
        for name, value in correlations["flagged"].items():
            direction = "pays for" if value > 0 else "punishes"
            out.append(
                f"reward {direction} {name.replace('_', ' ')} (corr {value:+.2f}); "
                "check the judge before training"
            )
    if scan:
        out.extend(str(w) for w in scan.get("warnings") or [])
    return out


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(tool.get("name") or "")


def coverage_warnings(
    rows: Sequence[dict], *, tools: Sequence[dict] | Sequence[str] | None = None
) -> list[str]:
    """Plain-words notes on whether a graded run can mean anything.

    A pass@1 of 1.00 over rows where the agent never called a tool, or a
    marker that fired on no row, is the most expensive eval failure there
    is, because it reads as a result. A run can be hollow in part, too:
    rows that called no tool score on the reply alone and lift the pass
    rate of the rows that did the work, so they get their own note once
    there are two of them (or a tenth of the run). ``run_judge`` (so ``evaluate`` and
    ``data.grade``) attaches these to ``ScoredData.warnings`` and logs
    them once. ``tools=`` is the declared tool list (OpenAI or bare shape,
    or just names); ``evaluate(data, judge)`` and ``data.grade`` read it
    off the run. The tool checks need it, so an agent with no tools is
    never called hollow for calling none. Each note names the fix.
    """
    row_list = [r for r in rows if isinstance(r, dict)]
    out: list[str] = []
    n = len(row_list)
    if n == 0:
        return out
    fix = (
        "Put the ids your world has (order numbers, account names) in the tool "
        "descriptions or in seeds=, and check the agent wrapper records its steps."
    )
    names = [nm for nm in (_tool_name(t) for t in tools or []) if nm]
    with_calls = sum(1 for r in row_list if tool_calls(r) > 0)
    if names and with_calls == 0:
        out.append(f"0 of {n} rows called a tool, so this score says nothing about tool use. {fix}")
    elif names:
        # Part of a run can be hollow too. Rows that never touch a tool
        # pass on the reply alone, and they lift the run's pass rate
        # while measuring nothing. Two rows, or a tenth of them, is
        # enough to say so; one odd row is not worth the noise.
        silent = [r for r in row_list if tool_calls(r) == 0]
        if silent and (len(silent) >= 2 or len(silent) / n >= 0.1):
            asks = list(dict.fromkeys(str(r.get("prompt") or "") for r in silent))
            shown = "; ".join(a[:60] for a in asks[:3] if a)
            out.append(
                f"{len(silent)} of {n} rows made no tool call "
                f"({len(asks)} situations{': ' + shown if shown else ''}); "
                "they score without touching a tool. Drop them from the eval "
                "or give them a seed that reaches one."
            )
        called: set[str] = set()
        for r in row_list:
            for s in r.get("steps") or r.get("tool_trace") or []:
                if isinstance(s, dict) and s.get("tool"):
                    called.add(str(s["tool"]))
        never = [nm for nm in names if nm not in called]
        if never:
            out.append(
                f"{len(never)} of {len(names)} declared tools were never called "
                f"({', '.join(never)}); the policy branches behind them are untested. "
                "Add a seed ask for each."
            )
        # A declared tool the world never answers fails like a world fault
        # and an honesty rubric rewards the miss (#287). Same table the
        # engine writes to coverage["tools"]. A row list does not say who
        # answered the calls, so the note carries no execute= verdict.
        outcomes = tool_outcomes(row_list)
        dead = dead_tools(outcomes)
        if dead:
            out.append(dead_tools_note(outcomes, dead))
    populated: dict[str, int] = {}
    for r in row_list:
        marks = r.get("markers")
        if not isinstance(marks, dict):
            continue
        for name, value in marks.items():
            populated.setdefault(str(name), 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                populated[str(name)] += 1
    for name in sorted(populated):
        if populated[name] == 0:
            out.append(
                f"marker {name!r} fired on 0 of {n} rows; its rate is not a measurement. "
                "Add a seed ask that exercises it, or drop it from the report."
            )
    return out


__all__ = [
    "DEFAULT_MAX_SPREAD",
    "HACK_THRESHOLD",
    "NEAR_DUP_JACCARD",
    "assistant_turns",
    "coverage_warnings",
    "dedupe_groups",
    "drop_truncated",
    "hygiene_warnings",
    "is_truncated",
    "length_report",
    "near_duplicate_prompts",
    "pearson",
    "reply_length",
    "reward_correlations",
    "tool_calls",
]
