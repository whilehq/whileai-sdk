"""Concentrate a big simulated batch into the dataset post-training needs.

Cold start over-generates on purpose (a few thousand rows walks the whole
grid). The optimizer is the concentrator, and what survives depends on the
post-training target:

* ``select_for_sft``: correct, diverse demonstrations. Only 1-labeled
  rows; one per behavior shape first, so 800 rows cover 800 behaviors
  instead of 80 rephrasings of ten.
* ``select_for_rl``: whole groups with within-ask contrast. Groups are
  never split; unanimous groups are dead gradient and go first; what is
  left is ordered mixed-band first, spread across fault kinds.

The optimizer does not create diversity. If the mixed rate is low after
trimming, rerun the simulator rather than squeezing this batch harder.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import statistics
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..defaults import (
    DIFFICULTY_BAND,
    DIFFICULTY_BAND_ROLLOUTS,
    REJECTION_SAMPLING_MIN_K,
    RL_ROLLOUTS_PER_ASK,
    TRUNCATED_REPLY_CHARS,
)
from ..tools import schemas as _tool_schemas
from .grading import (
    _DEGENERATE,
    _HARNESS_LEAK,
    _INFRA_STUB,
    behavior_signature,
    looks_finished,
    trace_fault,
)
from .passat import pass_at
from .quality import _IDISH, _QUESTION_END, _STRONG_ACTION, load_jsonl, write_jsonl
from .stats import task_key

# DEFAULT_BAND = DIFFICULTY_BAND (0.2, 0.8): keep asks the policy passes
# between 20% and 80% of the time (Lambert 2025, chapter Reasoning, difficulty
# filtering from N=16 samples; DAPO arXiv:2503.14476 drops accuracy 0 and 1
# groups; Seed-Thinking, ORZ, Phi-4, INTELLECT-2, MiMo, Skywork-OR1 all report
# a form of it). A reported practice with no published ablation on the edges,
# so every selector takes ``band=``.
DEFAULT_BAND: tuple[float, float] = DIFFICULTY_BAND
# SFT_TARGET_DEFAULT = 800 and SELECTION_SURPLUS = 3: ``recommend`` sizes
# an SFT run for 800 selected rows (curated agent SFT lands at 500 to
# 2,000: FireAct 500, LIMA 1,000, AgentTuning 1,866) from three times as
# many candidates, so the selector chooses rather than keeps everything
# (the surplus is convention).
SFT_TARGET_DEFAULT = 800
SELECTION_SURPLUS = 3
# BUDGET_ROUNDING = 100: a recommended row budget is rounded up to the
# next hundred, a number a person can say (convention).
BUDGET_ROUNDING = 100
# MIXED_RATE_FLOOR = 0.02: the lowest mixed-group rate ``recommend`` will
# size for; below it the ask count explodes and the honest answer is a
# harder grid, which the reasoning says (convention).
MIXED_RATE_FLOOR = 0.02
# MIXED_RATE_DEFAULT = 0.5: the mixed-group rate assumed before one is
# measured; a struggling agent mixes on half its asks, a competent one on
# a cold-start grid measured 4% (field test, 48x8, hosted Qwen), so this
# is the optimistic end and the docstring says to probe first.
MIXED_RATE_DEFAULT = 0.5

# How asks inside the band are ordered within a fault kind: ``"spread"``
# takes them round-robin across pass rates, ``"middle"`` ranks the ones
# nearest a 50% pass rate first.
RL_ORDERS = ("spread", "middle")

DO_NOTHING = "do_nothing"
INCOMPLETE_JUNK = "incomplete_junk"
UNUSABLE_LABEL = "unusable_label"
RL_DROP_REASONS = (DO_NOTHING, INCOMPLETE_JUNK, UNUSABLE_LABEL)

# Kept-despite-junk tag: a verified failure row (reward 0 from a trusted
# grader) whose junk IS the behavior negative advantage should suppress.
KEPT_VERIFIED_ZERO = "kept_verified_zero"
_VERIFIED_SOURCES = ("claude", "gold")

_RAW_TOOL_MARKUP = re.compile(r"</?tool_call>", re.I)
_TOOL_SCHEMA_DUMP = re.compile(
    r'"name"\s*:\s*"[^"]+".{0,500}"description"\s*:'
    r'.{0,500}"parameters"\s*:',
    re.I | re.S,
)


def _messages(row: dict) -> list[dict]:
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return [m for m in msgs if isinstance(m, dict)]
    from whileai.simulations import conversation

    return conversation(row)


def _has_tool_call(row: dict) -> bool:
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            return True
    return any(msg.get("tool_calls") for msg in _messages(row))


def _situation_wants_tools(row: dict) -> bool:
    """True when the situation is an actionable tool ask, not chit-chat."""
    if row.get("tool_known"):
        return True
    if str(row.get("ask_family") or "") == "tool":
        return True
    if row.get("faults"):
        return True
    world = row.get("world_state")
    if world and world not in {"unspecified", "unknown"}:
        return True
    dims = row.get("scenario_dimensions")
    if isinstance(dims, dict) and (dims.get("intent") or dims.get("tool")):
        return True
    prompt = str(row.get("prompt") or "")
    return bool(_STRONG_ACTION.search(prompt) or _IDISH.search(prompt))


def is_do_nothing(row: dict, *, has_tools: bool = True) -> bool:
    """Actionable situation where the agent never called a tool.

    For an agent with no tools (``has_tools=False``) this is never a drop:
    declining an actionable ask in words IS that agent's correct behavior,
    and deleting those rows would strip its refusal demonstrations.
    """
    if not has_tools:
        return False
    return _situation_wants_tools(row) and not _has_tool_call(row)


def _visible_text(row: dict) -> str:
    parts = [str(row.get("final_text") or "")]
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("text"):
            parts.append(str(step["text"]))
    for msg in _messages(row):
        if msg.get("role") == "assistant":
            parts.append(str(msg.get("content") or ""))
    return "\n".join(parts)


def is_incomplete_junk(row: dict) -> bool:
    """Empty, truncated, leaked, or parser-broken traces."""
    final = str(row.get("final_text") or "").strip()
    has_tool = _has_tool_call(row)
    if not final:
        return True
    if final.lower().startswith("<agent error"):
        return True
    if not has_tool and _INFRA_STUB.search(final):
        return True
    if _DEGENERATE.search(final):
        return True
    if _HARNESS_LEAK.search(final):
        return True
    visible = _visible_text(row)
    if _RAW_TOOL_MARKUP.search(visible) or _TOOL_SCHEMA_DUMP.search(visible):
        return True
    # A reply cut at the cap is junk unless ``select_for_rl(truncated=)``
    # already claimed it (``overlong``): keep and penalize decide its fate,
    # not this gate, or the report counts a row the output never carries.
    if len(final) > TRUNCATED_REPLY_CHARS and not looks_finished(final) and not row.get("overlong"):
        return True
    messages = _messages(row)
    if not messages:
        return not has_tool
    last_role = str(messages[-1].get("role") or "")
    if last_role in {"user", "tool"}:
        return True
    return bool(
        not has_tool
        and _QUESTION_END.search(final)
        and len(str(row.get("prompt") or "").split()) <= 2  # noqa: PLR2004  # a one- or two-word prompt is a stub (convention)
    )


def _is_binary_01(value) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number in (0.0, 1.0)


def is_unusable_label(row: dict) -> bool:
    """Missing or non-binary ``reward`` (or ``qwen_reward`` if that is all there is)."""
    for key in ("reward", "qwen_reward"):
        if key in row and _is_binary_01(row.get(key)):
            return False
    return True


def _has_trainable_content(row: dict) -> bool:
    if _has_tool_call(row) or str(row.get("final_text") or "").strip():
        return True
    return any(
        m.get("role") == "assistant" and (m.get("content") or m.get("tool_calls"))
        for m in _messages(row)
    )


def is_verified_zero(row: dict) -> bool:
    """reward == 0 from a trusted grader (claude* / gold label_source)."""
    value = row.get("reward")
    if value is None or not _is_binary_01(value) or float(value) != 0.0:
        return False
    src = str(row.get("label_source") or "").lower()
    return src.startswith(_VERIFIED_SOURCES)


def drop_reason(row: dict, *, has_tools: bool = True, text_gates: bool = True) -> str | None:
    """First matching drop tag, or None to keep.

    A verified zero bypasses the behavioral gates: dropping it deletes
    the failure example the negative advantage exists to teach against
    (measured: the old gates deleted 13.2% of verified zeros, skewed
    toward incomplete and over-clarification failures). It must still
    contain something to train on.

    ``text_gates=False`` skips the two gates that read the reply
    (``is_incomplete_junk``, ``is_do_nothing``) and keeps only the label
    gate: the call ``select_for_rl`` makes when no row carries a reply,
    where an empty ``final_text`` is the shape of the data, not a finding.
    """
    if is_verified_zero(row) and _has_trainable_content(row):
        return None
    if text_gates and is_incomplete_junk(row):
        return INCOMPLETE_JUNK
    # The do-nothing gate is pre-judge hygiene. A row the judge already
    # scored is the judge's call: an ask the grid thought needed a tool
    # is often answerable from policy text, and the airline walk showed
    # this gate deleting 131 judge-passed correct answers.
    if text_gates and _binary_label(row) is None and is_do_nothing(row, has_tools=has_tools):
        return DO_NOTHING
    if is_unusable_label(row):
        return UNUSABLE_LABEL
    return None


def filter_rl_rows(
    rows: Sequence[dict], *, has_tools: bool = True, text_gates: bool = True
) -> tuple[list[dict], dict[str, Any]]:
    """Split keep/drop. Does not mutate ``rows``. ``text_gates`` as in
    ``drop_reason``."""
    kept: list[dict] = []
    counts = {DO_NOTHING: 0, INCOMPLETE_JUNK: 0, UNUSABLE_LABEL: 0}
    kept_verified = 0
    for row in rows:
        reason = drop_reason(row, has_tools=has_tools, text_gates=text_gates)
        if reason is None:
            if is_verified_zero(row) and (
                is_incomplete_junk(row) or is_do_nothing(row, has_tools=has_tools)
            ):
                kept_verified += 1
            kept.append(row)
        else:
            counts[reason] = counts.get(reason, 0) + 1
    n = len(rows)
    report = {
        "n": n,
        "n_kept": len(kept),
        "n_dropped": n - len(kept),
        "dropped": counts,
        KEPT_VERIFIED_ZERO: kept_verified,
    }
    return kept, report


def drop_privileged_leaks(rows: Sequence[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Drop rows whose reply quotes their own ``privileged`` block. Does
    not mutate ``rows``.

    The block (``reference``, ``principle``, ``hidden_state``) is what the
    grader was told and the agent was not. A reply that recites it did not
    earn its reward, and ``export_dataset`` refuses such rows under
    ``validate=True`` because the scrub removes the key and not the reply.
    So the gate runs first, in both modes, and a selection never keeps a
    row the export will refuse. The report carries ``n_checked`` (rows that
    carried the block), ``n_dropped``, ``leaked`` (up to 20 rows:
    ``scenario_id``, ``rollout_index``, ``field``, ``needle``) and
    ``checked`` (False when no row carried the block, so a zero is vacuous).
    Same rule as ``leak_report`` (``LEAK_MIN_QUOTE_CHARS``).
    """
    from .privileged import row_leak  # style imports this module; keep the cycle lazy

    kept: list[dict] = []
    leaked: list[dict[str, Any]] = []
    n_checked = 0
    for row in rows:
        found = row_leak(row) if isinstance(row, dict) else None
        if found is None:
            kept.append(row)
            continue
        n_checked += 1
        if not found:
            kept.append(row)
            continue
        leaked.append(
            {
                "scenario_id": row.get("scenario_id"),
                "rollout_index": row.get("rollout_index"),
                **found,
            }
        )
    report = {
        "checked": n_checked > 0,
        "n_checked": n_checked,
        "n_dropped": len(leaked),
        "leaked": leaked[:20],
    }
    return kept, report


def _group_label_lists(rows: Sequence[dict]) -> dict[str, list[int]]:
    """Binary labels per task (grouped by ``task_key``). Unlabeled rows skip."""
    groups: dict[str, list[int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = None
        for key in ("reward", "qwen_reward"):
            value = row.get(key)
            if value is None or isinstance(value, bool):
                continue
            if _is_binary_01(value):
                label = int(float(value))
                break
        if label is None:
            continue
        groups.setdefault(task_key(row), []).append(label)
    return groups


def _stamped_rate(row: dict) -> tuple[float | None, int | None]:
    """A per-task pass rate carried on the row itself: ``pass_rate`` at
    the top level or under ``calibration``, with ``n`` beside it when the
    row says how many rollouts it summarizes."""
    cal = row.get("calibration")
    for src in (row, cal if isinstance(cal, dict) else {}):
        rate = src.get("pass_rate")
        if rate is None or isinstance(rate, bool):
            continue
        try:
            value = float(rate)
        except (TypeError, ValueError):
            continue
        if not 0.0 <= value <= 1.0:
            continue
        n = src.get("n")
        count = (
            int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0 else None
        )
        return value, count
    return None, None


def _task_pass_rates(rows: Sequence[dict]) -> dict[str, tuple[float, int | None]]:
    """Per-task ``(pass_rate, n)`` from either shape a caller has.

    Per-rollout rows with a binary ``reward`` give the mean over the
    rollouts of one task (``task_key``). One row per task carrying
    ``pass_rate`` and ``n`` (at the top level or under ``calibration``,
    the stamp ``select_for_rl`` and ``next_round`` write) gives that rate
    as it stands, ``n`` None when the row does not say. This is what a
    trainer's state holds, and the difficulty band needs nothing more
    (Lambert 2025, chapter Reasoning; Yu et al. 2025, arXiv:2503.14476).
    A task with labelled rollouts is measured from them and its stamp is
    ignored: ``select_for_rl`` stamps its selection in place, so the
    stamp on a graded row is a past measurement, not this one.
    """
    labels = _group_label_lists(rows)
    out: dict[str, tuple[float, int | None]] = {
        key: (sum(v) / len(v), len(v)) for key, v in labels.items() if v
    }
    for row in rows:
        if not isinstance(row, dict) or task_key(row) in out:
            continue
        rate, n = _stamped_rate(row)
        if rate is not None:
            out[task_key(row)] = (rate, n)
    return out


def group_signal(
    rows: Sequence[dict], *, lo: float = DEFAULT_BAND[0], hi: float = DEFAULT_BAND[1]
) -> dict[str, Any]:
    """Within-ask contrast. Signal is a group whose k rollouts disagree.

    A grouped RL update learns from a mix of 0 and 1 on the same ask,
    ideally with pass rate p in [``lo``, ``hi``]. Unanimous groups are
    dead gradient. Groups of one rollout cannot mix and are counted
    separately, not blamed.
    """
    groups = _group_label_lists(rows)
    n_mixed = n_all_zero = n_all_one = n_single = in_band = 0
    for labels in groups.values():
        if len(labels) < 2:  # noqa: PLR2004  # a group of one carries no contrast
            n_single += 1
            continue
        p = sum(labels) / len(labels)
        if 0.0 < p < 1.0:
            n_mixed += 1
            if lo <= p <= hi:
                in_band += 1
        elif p == 0.0:
            n_all_zero += 1
        else:
            n_all_one += 1
    multi = n_mixed + n_all_zero + n_all_one
    rates = pass_at(rows)
    return {
        "n_groups": len(groups),
        "n_single": n_single,
        "n_mixed": n_mixed,
        "n_all_zero": n_all_zero,
        "n_all_one": n_all_one,
        "n_in_band": in_band,
        "mixed_rate": (n_mixed / multi) if multi else None,
        # The same groups in post-training vocabulary. pass@k - pass@1 is
        # the headroom the mixed groups carry; see score/passat.py.
        "k": rates.k,
        "pass_at_1": rates.pass_at_1,
        "pass_pow_k": rates.pass_pow_k,
        "pass_at_k": rates.pass_at_k,
        "headroom": rates.headroom,
    }


def trim_unanimous_groups(
    rows: Sequence[dict], *, min_k: int = 2
) -> tuple[list[dict], dict[str, Any]]:
    """Drop asks whose k >= ``min_k`` rollouts all landed 0 or all landed 1.

    The basic optimizer from the working decision: trim zeros and ones
    from tasks, then rerun the simulator and check the variance. Groups
    smaller than ``min_k`` (unique-situation runs) always stay; trimming
    them would gut an explore dataset, and they carry no group gradient
    either way.
    """
    groups = _group_label_lists(rows)
    dead: set[str] = set()
    for prompt, labels in groups.items():
        if len(labels) >= max(2, int(min_k)) and len(set(labels)) == 1:
            dead.add(prompt)
    kept = [row for row in rows if task_key(row) not in dead]
    report = {
        "n": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(rows) - len(kept),
        "n_groups_dropped": len(dead),
        "signal": group_signal(kept),
    }
    return kept, report


def eval_sourced(rows: Sequence[dict]) -> int:
    """Rows whose reward came from ``evaluate()`` (``lineage.source == "eval"``).

    An eval score is the number you report; training on it makes the
    held-out scorer the reward model. The count is surfaced on every
    selector report so the leak is visible before a run starts.
    """
    return sum(
        1
        for row in rows
        if isinstance(row, dict) and (row.get("lineage") or {}).get("source") == "eval"
    )


def _eval_sourced_warning(count: int, unit: str) -> str:
    return (
        f"{count} {unit} carry rewards from evaluate() (lineage.source == 'eval'); "
        "training on them turns the held-out scorer into the reward model"
    )


def _stable_key(text: str, salt: str = "") -> str:
    return hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()


def _binary_label(row: dict) -> int | None:
    for key in ("reward", "qwen_reward"):
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        if _is_binary_01(value):
            return int(float(value))
    return None


SFT_SELECTIONS = ("top_per_prompt", "random_per_prompt", "top_k_overall", "random_k_overall")
#: what select_for_rl does with a rollout cut at the token cap
TRUNCATED_POLICIES = ("drop", "keep", "penalize")
#: whether select_for_rl runs the gates that read the reply text
TEXT_GATE_MODES = ("auto", "require", "skip")


def _any_reply_text(rows: Sequence[dict]) -> bool:
    """Whether any row carries a reply the text gates could read: a
    ``final_text``, an assistant ``messages`` turn, or a tool step."""
    return any(isinstance(r, dict) and _has_trainable_content(r) for r in rows)


def _scalar_reward(row: dict) -> float | None:
    for key in ("reward", "qwen_reward"):
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return number
    return None


def select_for_sft(
    rows: Sequence[dict],
    *,
    target: int = 1000,
    select: str = "top_per_prompt",
    k: int | None = None,
    min_reward: float = 1.0,
    seed: int = 0,
) -> tuple[list[dict], dict[str, Any]]:
    """Diverse correct demonstrations, at most ``target`` rows.

    Imitation clones what it sees, so only rows whose reward reaches
    ``min_reward`` (default 1.0: judge-approved) and that are not junk
    qualify; unanimity is not a problem here. A grader with partial
    credit ranks by its score: lower ``min_reward`` to admit it.

    ``select`` is the rejection-sampling rule (Lambert 2025, chapter Rejection
    Sampling, "Scoring Completions"): ``"top_per_prompt"`` keeps each prompt's
    highest-reward completion and then round-robins across behavior signatures
    (tool sequence, argument provenance, outcome shape) so every distinct way
    of being right appears before any repeats; ``"top_k_overall"`` keeps the
    ``k`` highest-reward completions across all prompts, several per prompt
    allowed; the two ``random_*`` rules are the control that chapter asks for
    (same counts, seeded random picks) so a claimed gain from selection can be
    checked against chance. ``k`` defaults to ``target``.
    """
    if select not in SFT_SELECTIONS:
        raise ValueError(f"select must be one of {', '.join(SFT_SELECTIONS)}; got {select!r}")
    rng = random.Random(int(seed))
    scored: list[tuple[float, dict]] = []
    n_wrong = n_junk = 0
    # A pass that recites the answer key is not a demonstration; the
    # judge that read the same key could not tell (#249).
    graded = rows  # every graded rollout: the count and completions-per-prompt read it
    rows, leaks = drop_privileged_leaks(rows)
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _scalar_reward(row)
        if value is None or value < float(min_reward):
            n_wrong += 1
            continue
        # passing rows are judge-approved; only structural junk drops.
        if is_incomplete_junk(row):
            n_junk += 1
            continue
        scored.append((value, row))
    goal = max(1, int(target))
    limit = max(1, int(k)) if k else goal
    if select in ("top_k_overall", "random_k_overall"):
        if select == "top_k_overall":
            scored.sort(
                key=lambda pair: (
                    -pair[0],
                    _stable_key(str(pair[1].get("prompt") or ""), behavior_signature(pair[1])),
                )
            )
        else:
            rng.shuffle(scored)
        picked_overall = [row for _, row in scored[:limit]]
        pool = [row for _, row in scored]
        return picked_overall, _sft_report(
            graded, leaks, pool, picked_overall, n_wrong, n_junk, goal, select, limit, min_reward
        )
    by_prompt: dict[str, list[tuple[float, dict]]] = {}
    for value, row in scored:
        prompt = " ".join(str(row.get("prompt") or "").lower().split())
        by_prompt.setdefault(prompt, []).append((value, row))
    eligible: list[dict] = []
    for prompt, candidates in by_prompt.items():
        if select == "random_per_prompt":
            eligible.append(rng.choice(candidates)[1])
            continue
        candidates.sort(
            key=lambda pair: (-pair[0], _stable_key(prompt, behavior_signature(pair[1])))
        )
        eligible.append(candidates[0][1])
    buckets: dict[str, list[dict]] = {}
    for row in eligible:
        buckets.setdefault(behavior_signature(row), []).append(row)
    for sig, bucket in buckets.items():
        bucket.sort(key=lambda r: _stable_key(str(r.get("prompt") or ""), sig))
    order = sorted(buckets, key=lambda sig: (-len(buckets[sig]), sig))
    selected: list[dict] = []
    round_i = 0
    while len(selected) < goal:
        took = False
        for sig in order:
            bucket = buckets[sig]
            if round_i < len(bucket):
                selected.append(bucket[round_i])
                took = True
                if len(selected) >= goal:
                    break
        if not took:
            break
        round_i += 1
    return selected, _sft_report(
        graded, leaks, eligible, selected, n_wrong, n_junk, goal, select, None, min_reward
    )


def _sft_report(
    rows: Sequence[dict],
    leaks: dict[str, Any],
    eligible: Sequence[dict],
    selected: Sequence[dict],
    n_wrong: int,
    n_junk: int,
    goal: int,
    select: str,
    k: int | None,
    min_reward: float,
) -> dict[str, Any]:
    def _mean(items: Sequence[dict]) -> float | None:
        values = [v for v in (_scalar_reward(r) for r in items) if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    buckets = {behavior_signature(r) for r in eligible}
    report: dict[str, Any] = {
        "n": len(rows),
        "privileged_leaks_dropped": leaks["n_dropped"],
        "privileged_leaks": leaks,
        "n_eligible": len(eligible),
        "n_selected": len(selected),
        "n_not_pass": n_wrong,
        "n_junk": n_junk,
        "unique_behaviors": len(buckets),
        "behaviors_covered": len({behavior_signature(r) for r in selected}),
        "target": goal,
        "selection": select,
        "min_reward": float(min_reward),
        "reward_mean_eligible": _mean(eligible),
        "reward_mean_selected": _mean(selected),
        "eval_sourced": eval_sourced(selected),
    }
    if k is not None:
        report["k"] = k
    if report["eval_sourced"]:
        report["warning"] = _eval_sourced_warning(report["eval_sourced"], "selected row(s)")
    # Rejection sampling picks the best of N completions per prompt, and the
    # published recipes use 10 to 30 (REJECTION_SAMPLING_MIN_K, Lambert 2025,
    # chapter Rejection Sampling); fewer makes the pick biased or noisy. The
    # note reads the MEAN, not the max: the max is the most optimistic
    # statistic in the pool, and one prompt with 12 completions silenced it
    # for 500 prompts with one each (a measured pool ran mean k=1.07 and
    # produced a null). When the median prompt has one completion there is no
    # pick on most prompts, only a pass/fail filter, and ``top_per_prompt``
    # and ``random_per_prompt`` (the chance control of Lambert 2025, chapter
    # Rejection Sampling) return the same rows; ``selection_effective`` says
    # which operation ran so a card cannot claim a selection that did not
    # happen.
    per_prompt: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            key = " ".join(str(row.get("prompt") or "").lower().split())
            per_prompt[key] = per_prompt.get(key, 0) + 1
    counts = sorted(per_prompt.values())
    max_k = counts[-1] if counts else 0
    mean_k = round(sum(counts) / len(counts), 2) if counts else 0.0
    median_k = counts[len(counts) // 2] if counts else 0
    singles = sum(1 for c in counts if c <= 1)
    report["completions_per_prompt_max"] = max_k
    report["completions_per_prompt_mean"] = mean_k
    report["completions_per_prompt_median"] = median_k
    report["prompts_with_one_completion"] = singles
    if counts:
        report["selection_effective"] = "pass_filter" if median_k <= 1 else select
    if counts and mean_k < REJECTION_SAMPLING_MIN_K:
        report["note"] = (
            f"completions per prompt: mean {mean_k}, median {median_k}, max {max_k}; "
            f"{singles} of {len(counts)} prompts have one. Rejection-sampling selection "
            f"wants {REJECTION_SAMPLING_MIN_K} to 30 so the pick is not biased "
            "(Lambert 2025, chapter Rejection Sampling; Llama 3 samples 10 to 30)"
            + (
                ", and with one completion on the median prompt there is no pick at all, "
                "only a pass/fail filter: top_per_prompt and random_per_prompt return the "
                "same rows, so the random-selection control says nothing"
                if median_k <= 1
                else ""
            )
            + ". Raise repeats= if you mean to choose among completions rather than filter."
        )
    return report


def trim_out_of_band(
    rows: Sequence[dict],
    *,
    lo: float = DEFAULT_BAND[0],
    hi: float = DEFAULT_BAND[1],
    min_k: int = 2,
) -> tuple[list[dict], dict[str, Any]]:
    """Difficulty band filter. Nothing to do with topic or relevance.

    "Out of band" here means outside the *difficulty* band ``[lo, hi]``
    (default ``DEFAULT_BAND``, 0.2 to 0.8): an ask is dropped when its
    pass rate over k >= ``min_k`` rollouts is too high (the policy almost
    always solves it) or too low (it almost never does), because either
    way it carries little gradient per rollout. It does not read the
    prompt, the topic, or the tools; a perfectly on-topic ask is dropped
    for being too easy, and an off-topic one the policy passes half the
    time is kept. Junk rows are a separate filter (``is_incomplete_junk``,
    applied by ``optimize``), and nothing here filters by topic at all.

    Unanimous asks are ``trim_unanimous_groups``'s job and are left alone
    here; singles always stay.

    Rows are per-rollout rows with a binary ``reward``, or one row per
    task carrying ``pass_rate`` and ``n`` (a trainer's state, or the
    ``calibration`` stamp); ``min_k`` reads ``n``, and a rate row that
    does not say its ``n`` is taken at its word. The report's ``from_rates``
    counts the tasks measured from a carried rate rather than rollouts.
    """
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"band must satisfy 0 <= lo <= hi <= 1, got ({lo}, {hi})")
    labelled = _group_label_lists(rows)
    rates = _task_pass_rates(rows)
    too_easy: set[str] = set()
    too_hard: set[str] = set()
    from_rates = 0
    for prompt, (p, n) in rates.items():
        if prompt not in labelled:
            from_rates += 1
        if n is not None and n < max(2, int(min_k)):
            continue
        if not 0.0 < p < 1.0:
            continue
        if p > hi:
            too_easy.add(prompt)
        elif p < lo:
            too_hard.add(prompt)
    dead = too_easy | too_hard
    kept = [row for row in rows if task_key(row) not in dead]
    return kept, {
        "n": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(rows) - len(kept),
        "n_groups_dropped": len(dead),
        "too_easy": len(too_easy),
        "too_hard": len(too_hard),
        "from_rates": from_rates,
        "band": [lo, hi],
    }


def _spread_by(prompts: list[str], key) -> list[str]:
    """Round-robin over the distinct values of ``key`` (ascending), keeping
    the input order within each value: one ask per pass rate in turn."""
    levels: dict[float, list[str]] = {}
    for prompt in prompts:
        levels.setdefault(round(float(key(prompt)), 4), []).append(prompt)
    queues = [levels[value] for value in sorted(levels)]
    out: list[str] = []
    depth = 0
    while len(out) < len(prompts):
        for queue in queues:
            if depth < len(queue):
                out.append(queue[depth])
        depth += 1
    return out


def _task_keys_of(tasks: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for t in tasks:
        if isinstance(t, dict):
            out.append(task_key(t))
        else:
            out.append(task_key({"prompt": str(t)}))
    return out


def next_round(
    prior: Sequence[dict],
    *,
    tasks: Sequence[Any] | None = None,
    lo: float = DEFAULT_BAND[0],
    hi: float = DEFAULT_BAND[1],
) -> dict[str, Any]:
    """The prompt set for the next round, from the last round's graded
    rollouts.

    A round trained on the file it started from keeps paying for groups that
    give no gradient: at a 0.65 training reward about half the groups are
    all-pass or all-fail. The band is the published fix (Lambert 2025, chapter
    Reasoning: filter to the 20-80% band; Yu et al. 2025 (DAPO),
    arXiv:2503.14476: dynamic sampling drops groups with no contrast), applied
    to what the *current* policy does rather than what the base did. ``prior``
    is round N's graded rollouts (``simulate(tasks=..., repeats=k)`` on the
    round-N policy, or the trainer's own sampled rows); each task's pass rate
    over them decides: inside ``[lo, hi]`` it is kept, above ``hi`` it is
    solved and dropped, below ``lo`` it is unsolved and dropped. ``tasks``
    restricts the candidates (rows, task dicts with a ``prompt``, or prompt
    strings); a task with no prior rollouts is ``unknown`` and kept, since
    nothing says it is flat.

    ``prior`` takes either shape: per-rollout rows with a binary
    ``reward``, or one row per task carrying ``pass_rate`` and ``n`` (a
    trainer's per-task table, or the ``calibration`` stamp this function
    and ``select_for_rl`` write). No reply text is read. A ``prior`` that
    carries neither is a ``UserWarning`` and an all-unknown plan, not a
    silent empty one.

    Returns ``tasks`` (one representative row per kept task: the prior
    row, with ``calibration.pass_rate`` and the band), the counts
    ``kept``, ``dropped_solved``, ``dropped_unsolved``, ``unknown``,
    ``pass_rates`` per task, ``band``, ``from_policy`` (the policy
    versions the prior rows came from) and ``prompt_set_sha``: the
    identity of the kept set, for lineage on the run. Push the kept rows
    as the next train set with ``parent=`` the last one.
    """
    if not 0 <= lo < hi <= 1:
        raise ValueError("band is 0 <= lo < hi <= 1")
    measured = _task_pass_rates(prior)
    rates = {key: rate for key, (rate, _n) in measured.items()}
    n_prior_rows = sum(1 for r in prior if isinstance(r, dict))
    if n_prior_rows and not rates:
        warnings.warn(
            f"next_round: none of the {n_prior_rows} prior rows carries a binary reward "
            "or a pass_rate (top level or under calibration), so no task was measured "
            "and every candidate is unknown; pass graded rollouts or a per-task rate table",
            UserWarning,
            stacklevel=2,
        )
    first: dict[str, dict] = {}
    policies: set[str] = set()
    for row in prior:
        if not isinstance(row, dict):
            continue
        first.setdefault(task_key(row), row)
        if row.get("policy_version"):
            policies.add(str(row["policy_version"]))
    if tasks is None:
        candidates = list(rates)
        given: dict[str, Any] = {}
    else:
        given = {}
        for t in tasks:
            key = task_key(t) if isinstance(t, dict) else task_key({"prompt": str(t)})
            given.setdefault(key, t)
        candidates = list(given)
    kept: list[dict] = []
    solved = unsolved = unknown = 0
    for key in candidates:
        rate = rates.get(key)
        if rate is None:
            unknown += 1
            rep = given.get(key)
            rep = dict(rep) if isinstance(rep, dict) else {"prompt": str(rep)}
            kept.append(rep)
            continue
        if rate > hi:
            solved += 1
            continue
        if rate < lo:
            unsolved += 1
            continue
        report: dict[str, Any] = dict(first.get(key) or given.get(key) or {"prompt": key})
        cal = dict(report.get("calibration") or {})
        cal.update({"pass_rate": round(rate, 4), "n": measured[key][1], "band": [lo, hi]})
        report["calibration"] = cal
        kept.append(report)
    sha = hashlib.sha256("\n".join(sorted(task_key(r) for r in kept)).encode()).hexdigest()[:16]
    return {
        "tasks": kept,
        "kept": len(kept) - unknown,
        "dropped_solved": solved,
        "dropped_unsolved": unsolved,
        "unknown": unknown,
        "n_prior_tasks": len(rates),
        "pass_rates": {k: round(v, 4) for k, v in rates.items()},
        "band": [lo, hi],
        "from_policy": sorted(policies),
        "prompt_set_sha": sha,
    }


def select_for_rl(
    rows: Sequence[dict],
    *,
    target: int = 1000,
    lo: float = DEFAULT_BAND[0],
    hi: float = DEFAULT_BAND[1],
    enforce_band: bool = True,
    has_tools: bool = True,
    dedupe: bool = True,
    drop_truncated: bool = True,
    endorsed: Sequence[str] = (),
    truncated: str = "drop",
    order: str = "spread",
    prior: Sequence[dict] | None = None,
    audit: dict[str, Any] | None = None,
    text_gates: str = "auto",
) -> tuple[list[dict], dict[str, Any]]:
    """Whole mixed groups up to roughly ``target`` rows. Groups never split.

    ``text_gates`` says whether the gates that read the reply run: the
    junk and do-nothing checks and the duplicate trim, which key on
    ``final_text``. ``"auto"`` (the default) runs them when any row carries
    a reply (``final_text``, an assistant ``messages`` turn, or a tool
    step) and skips them when none does, because a trainer's state holds
    a task and a binary reward per sample and nothing else, and an empty
    reply there is the shape of the data, not a finding; the report's
    ``text_gates`` block and a ``hygiene_warnings`` line say the gates
    were skipped. ``"require"`` runs them regardless (a row with no reply
    is ``incomplete_junk``, the behavior before 0.121), ``"skip"`` never
    runs them. The label gate, the unanimous trim, the difficulty band
    and the ranking run in every mode: they read the reward alone
    (Lambert 2025, chapter Reasoning; Yu et al. 2025, arXiv:2503.14476).

    ``audit`` is an ``audit_grades`` report on these rows' verifier; when
    it found the verifier rejecting right answers more than ``FN_WARN``
    of the time, ``hygiene_warnings`` says to fix the verifier before
    training on the selection (#255).

    ``prior`` is the previous round's graded rollouts: tasks the round-N
    policy already solves (pass rate above ``hi`` on ``prior``) or never
    solves (below ``lo``) are dropped before anything else, so round N+1
    trains on what that policy gets right 20-80% of the time rather than
    on the file round 1 started from (``next_round``; Lambert 2025, chapter
    Reasoning).
    The report's ``prior`` block counts kept, dropped_solved,
    dropped_unsolved and unknown.

    ``truncated`` says what happens to a rollout cut at the token cap
    (DAPO's overlong handling, Yu et al. 2025, arXiv:2503.14476; overlong
    filtering, Lambert 2025, chapter Reasoning):
    ``"drop"`` removes it (the default; ``drop_truncated=False`` is the old
    spelling of ``"keep"``), ``"keep"`` leaves it in with ``overlong=True``
    and its own reward, riding with its ask rather than deciding it (the
    ask is unanimous, in band and ranked exactly as under ``"drop"``, so
    ``"keep"`` never returns fewer rows than ``"drop"``; a cut rollout's
    reward is not the contrast an ask is kept for), and ``"penalize"``
    keeps it as a failure that does count: reward 0,
    the judged score under ``reward_before_penalty``, so running past the
    cap is a negative signal instead of a rollout that vanished. A
    conduct-grade advisory 0.5 for truncation is unusable under ``"keep"``
    and a 0 under ``"penalize"``.

    After the row gates, duplicate and truncated rollouts (``dedupe``,
    ``truncated``), the unanimous trim, and (``enforce_band``) the difficulty
    band, remaining asks are taken round-robin across observed fault kinds, so
    the dataset keeps a grounded spread of no-fault, miss, timeout, and
    already-done situations rather than one over-represented failure. Within a
    fault kind, ``order="spread"`` (the default) takes asks round-robin across
    their pass rates, so a 25% ask, a 50% ask and a 75% ask are picked in turn
    with no preference for the middle (Lambert 2025, chapter Reasoning,
    filters to the 20-80% band and stops there; nothing in it says 50% is
    better than 30%). ``order="middle"`` is the older ranking by closeness to
    a 50% pass rate. The last group may overshoot ``target``; an RL update
    wants the complete group or none of it. ``enforce_band=False`` keeps
    out-of-band asks and only ranks them last. The report's ``hack_scan``
    block is the reward-hack scan over the selection (``hack_scan``: what
    separates reward within an ask, against a permutation floor; ``endorsed``
    names what it should be), ``correlations`` the older pooled scan. Reward
    tracking a shortcut is a judge problem, flagged in ``hygiene_warnings``,
    not pruned.

    Selected rows are stamped in place with the ``calibration`` measured
    on the rows as they arrived, before dedupe and the trims: the pass
    rate over the k repeats the grader saw is the task's difficulty, and
    re-measuring it on the survivors would report the post-dedup k under
    that name. ``publish_gate`` keeps the carried stamp. The k-way
    reliability numbers do not survive the prune, and ``hygiene_warnings``
    says so when they were available before it.
    """
    from .hack_scan import hack_scan
    from .hygiene import (
        dedupe_groups,
        hygiene_warnings,
        is_truncated,
        length_report,
        reward_correlations,
    )
    from .hygiene import drop_truncated as _drop_truncated
    from .publish_gate import carry_calibration

    if truncated not in TRUNCATED_POLICIES:
        raise ValueError(
            f"truncated must be one of {', '.join(TRUNCATED_POLICIES)}; got {truncated!r}"
        )
    if order not in RL_ORDERS:
        raise ValueError(f"order must be one of {', '.join(RL_ORDERS)}; got {order!r}")
    if text_gates not in TEXT_GATE_MODES:
        raise ValueError(
            f"text_gates must be one of {', '.join(TEXT_GATE_MODES)}; got {text_gates!r}"
        )
    if not drop_truncated and truncated == "drop":
        truncated = "keep"
    prior_report: dict[str, Any] | None = None
    if prior is not None:
        plan = next_round(prior, lo=lo, hi=hi)
        rates = plan["pass_rates"]
        before_n = len(rows)
        rows = [
            r
            for r in rows
            if not isinstance(r, dict)
            or rates.get(task_key(r)) is None
            or lo <= rates[task_key(r)] <= hi
        ]
        prior_report = {
            k: plan[k]
            for k in ("kept", "dropped_solved", "dropped_unsolved", "unknown", "from_policy")
        }
        prior_report["rows_dropped"] = before_n - len(rows)
        prior_report["prompt_set_sha"] = plan["prompt_set_sha"]
    n_in = len(rows)
    graded_rows = rows  # every graded rollout, for the calibration stamp
    rows, leak_rep = drop_privileged_leaks(rows)
    penalized = kept_overlong = 0
    if truncated != "drop":
        marked: list[dict] = []
        for row in rows:
            if isinstance(row, dict) and is_truncated(row):
                row = dict(row)
                row["overlong"] = True
                markers = dict(row.get("markers") or {})
                markers["finished"] = 0.0
                row["markers"] = markers
                if truncated == "penalize":
                    from ..schema import Judgment, ScorerRef, attach

                    row["reward_before_penalty"] = row.get("reward")
                    attach(
                        row,
                        Judgment(
                            rollout_id=str(row.get("rollout_id") or ""),
                            scorer=ScorerRef(name="overlong_penalty", kind="rule"),
                            reward=0,
                            reason="cut at the token cap; penalized (truncated='penalize')",
                            evidence={
                                "reward_before_penalty": row["reward_before_penalty"],
                                "judge_before": row.get("judge_name"),
                                "reason_before": row.get("reason"),
                            },
                        ),
                    )
                    penalized += 1
                else:
                    kept_overlong += 1
            marked.append(row)
        rows = marked

    # The duplicate trim keys on the reply too: without one, every rollout
    # of a task is the same "duplicate" and the group collapses to one row.
    gates_on = text_gates == "require" or (text_gates == "auto" and _any_reply_text(rows))
    text_gate_report = {
        "mode": text_gates,
        "applied": gates_on,
        "reason": None
        if gates_on
        else ("no row carries a reply" if text_gates == "auto" else "text_gates='skip'"),
    }
    kept, base_report = filter_rl_rows(rows, has_tools=has_tools, text_gates=gates_on)
    original_sizes: dict[str, int] = {}
    for row in kept:
        key = task_key(row)
        original_sizes[key] = original_sizes.get(key, 0) + 1
    dup_report: dict[str, Any] = {"n_dropped": 0, "groups_affected": 0, "conflicting_rewards": 0}
    if dedupe and gates_on:
        kept, dup_report = dedupe_groups(kept)
    trunc_report: dict[str, Any] = {"n_dropped": 0}
    if truncated == "drop":
        kept, trunc_report = _drop_truncated(kept)
    # Under ``"keep"`` an overlong rollout rides along with its ask; it
    # does not vote on whether the ask is unanimous, in band, or ranked
    # first. Letting it vote made ``"keep"`` return *fewer* rows than
    # ``"drop"`` (a kept pass tipped an ask over the band, and the whole
    # ask went), so the ask decisions are the ones ``"drop"`` makes and
    # the overlong rows are added back to the asks that survive (#31).
    # ``"penalize"`` is the opposite by design: the penalty is a failure
    # that counts.
    riders: dict[str, list[dict]] = {}
    if truncated == "keep":
        voting: list[dict] = []
        for row in kept:
            if row.get("overlong"):
                riders.setdefault(task_key(row), []).append(row)
            else:
                voting.append(row)
        kept = voting
    kept, trim_report = trim_unanimous_groups(kept)
    # A group that hygiene shrank to one rollout, or to rollouts that all
    # agree, has no contrast left. It is not a single (which always stays,
    # since it never had a group) but a dead group, and goes the same way.
    collapsed = {
        prompt
        for prompt, labels in _group_label_lists(kept).items()
        if original_sizes.get(prompt, 0) >= 2 and len(set(labels)) == 1  # noqa: PLR2004  # a group of one carries no contrast
    }
    if collapsed:
        kept = [row for row in kept if task_key(row) not in collapsed]
        trim_report["n_groups_dropped"] += len(collapsed)
    trim_report["collapsed_groups_dropped"] = len(collapsed)
    band_report: dict[str, Any] = {"n_groups_dropped": 0, "too_easy": 0, "too_hard": 0}
    if enforce_band:
        kept, band_report = trim_out_of_band(kept, lo=lo, hi=hi)
    groups: dict[str, list[dict]] = {}
    for row in kept:
        groups.setdefault(task_key(row), []).append(row)
    for prompt, rows_ in riders.items():
        if prompt in groups:
            groups[prompt].extend(rows_)
            kept.extend(rows_)

    def _pass_rate(prompt: str) -> float:
        labels = [
            lbl
            for lbl in (
                _binary_label(r) for r in groups[prompt] if not (riders and r.get("overlong"))
            )
            if lbl is not None
        ]
        return sum(labels) / len(labels) if labels else 0.0

    def _score(prompt: str) -> tuple:
        p = _pass_rate(prompt)
        in_band = lo <= p <= hi
        middle = abs(p - (lo + hi) / 2) if order == "middle" else 0.0
        return (0 if in_band else 1, middle, _stable_key(prompt))

    fault_buckets: dict[str, list[str]] = {}
    for prompt, members in groups.items():
        fault = trace_fault(members[0])
        fault_buckets.setdefault(fault, []).append(prompt)
    for fault in fault_buckets:
        fault_buckets[fault].sort(key=_score)
        if order == "spread":
            # In-band asks stay ahead of out-of-band ones; inside each
            # block, one ask per pass rate in turn so no rate dominates.
            ranked = fault_buckets[fault]
            in_band = [p for p in ranked if lo <= _pass_rate(p) <= hi]
            outside = [p for p in ranked if not lo <= _pass_rate(p) <= hi]
            fault_buckets[fault] = _spread_by(in_band, _pass_rate) + _spread_by(outside, _pass_rate)
    fault_order = sorted(fault_buckets, key=lambda f: (-len(fault_buckets[f]), f))
    selected: list[dict] = []
    picked_groups = 0
    goal = max(1, int(target))
    round_i = 0
    while len(selected) < goal:
        took = False
        for fault in fault_order:
            prompts = fault_buckets[fault]
            if round_i < len(prompts):
                selected.extend(groups[prompts[round_i]])
                picked_groups += 1
                took = True
                if len(selected) >= goal:
                    break
        if not took:
            break
        round_i += 1
    report = {
        "n": n_in,
        "privileged_leaks_dropped": leak_rep["n_dropped"],
        "privileged_leaks": leak_rep,
        "n_after_gates": base_report["n_kept"],
        "gates": base_report["dropped"],
        "text_gates": text_gate_report,
        "n_after_trim": len(kept),
        "unanimous_groups_dropped": trim_report["n_groups_dropped"],
        "collapsed_groups_dropped": trim_report["collapsed_groups_dropped"],
        "n_selected": len(selected),
        "groups_selected": picked_groups,
        "fault_kinds": {fault: len(prompts) for fault, prompts in fault_buckets.items()},
        "target": goal,
        "band": [lo, hi],
        "enforce_band": bool(enforce_band),
        "band_groups_dropped": band_report["n_groups_dropped"],
        "band_dropped": {"too_easy": band_report["too_easy"], "too_hard": band_report["too_hard"]},
        "prior": prior_report,
        "duplicates": dup_report,
        "truncated_dropped": trunc_report["n_dropped"],
        "truncated_policy": truncated,
        "truncated_kept": kept_overlong,
        "truncated_penalized": penalized,
        # the marked rows that made it into the selection
        "truncated_selected": sum(1 for r in selected if r.get("overlong")),
        "length": length_report(selected),
        "correlations": reward_correlations(selected),
        "hack_scan": hack_scan(selected, endorsed=endorsed),
        "signal": group_signal(selected, lo=lo, hi=hi),
        "eval_sourced": eval_sourced(selected),
        "calibration": carry_calibration(graded_rows, selected),
    }
    report["hygiene_warnings"] = hygiene_warnings(
        duplicates=dup_report,
        lengths=report["length"],
        correlations=report["correlations"],
        scan=report["hack_scan"],
    )
    from .audit import audit_warning

    audit_note = audit_warning(audit)
    if audit_note:
        report["hygiene_warnings"].append(audit_note)
    if not gates_on:
        report["hygiene_warnings"].append(
            f"The text gates (junk, do-nothing, duplicate) did not run ({text_gate_report['reason']}), "
            "so this selection is by reward and difficulty band alone; nothing checked the replies. "
            "Pass text_gates='require' to refuse rows with no reply."
        )
    report["audit"] = (
        {k: audit.get(k) for k in ("fn_rate", "fn_ci95", "n_checked", "verifier")}
        if isinstance(audit, dict)
        else None
    )
    # The band is assigned from a handful of rollouts per task, and a
    # Wilson interval on k=8 is about +/-0.3 wide: a task measured at 0.25
    # may really sit at 0.1 or 0.5. Say so once, with the measured width.
    tasks = report["calibration"].get("tasks") or []
    halves = [
        (ci[1] - ci[0]) / 2 for ci in (t.get("pass_rate_ci95") for t in tasks) if ci is not None
    ]
    if tasks and halves:
        median_n = statistics.median(t["n"] for t in tasks)
        if median_n < DIFFICULTY_BAND_ROLLOUTS:
            report["hygiene_warnings"].append(
                f"Difficulty was measured from {median_n:g} rollouts per task, so a task's "
                f"band assignment can be off by about ±{statistics.median(halves):.1f}. "
                f"Use repeats={DIFFICULTY_BAND_ROLLOUTS} for a firmer band (the count the "
                "20-80 band is measured from, Lambert 2025, chapter Reasoning)."
            )
    if report["eval_sourced"]:
        report["hygiene_warnings"].append(
            _eval_sourced_warning(report["eval_sourced"], "selected row(s)")
        )
    # An eval set that reaches this selector is a leak whether or not a
    # group survived the trims, so the input count is reported too.
    report["eval_sourced_input"] = eval_sourced(rows)
    if not selected:
        graded = sum(1 for r in rows if isinstance(r, dict) and _binary_label(r) is not None)
        if graded:
            report["hygiene_warnings"] = [
                w for w in report["hygiene_warnings"] if "grade first" not in w
            ]
            report["hygiene_warnings"].append(
                f"nothing selected: {graded} graded row(s) in, "
                f"{trim_report['n_groups_dropped']} unanimous and "
                f"{trim_report['collapsed_groups_dropped']} collapsed group(s) dropped, "
                f"{band_report['n_groups_dropped']} outside the band; no mixed group survived"
            )
        if report["eval_sourced_input"]:
            report["hygiene_warnings"].append(
                _eval_sourced_warning(report["eval_sourced_input"], "input row(s)")
            )
    # Dedupe and the trims shrink every group, so the selection can no
    # longer report the k-way reliability numbers the graded rows could:
    # pass^k and pass@k need k repeats of an ask and hygiene just removed
    # them. The published rows cannot measure their own reliability, so
    # the loss is said out loud here rather than turning up as "n/a".
    before, after = pass_at(rows), pass_at(selected)
    if before.pass_at_k is not None and after.pass_at_k is None:
        report["hygiene_warnings"].append(
            f"pass^k / pass@k do not survive the prune: the graded rows scored k="
            f"{before.k}, the selection leaves {after.k} rollout(s) per ask, so "
            "pass_at on these rows reports them as n/a. pass@1 and the carried "
            "calibration stamp still hold the graded measurement; take the k-way "
            "numbers from pass_at before select"
        )
    # A selection with no mixed group has no within-group contrast: GRPO
    # advantage is zero everywhere and the run trains nothing. That is a
    # grading or difficulty problem upstream, and it must not exit this
    # function looking like a dataset.
    if report["signal"].get("n_mixed", 0) == 0:
        report["warning"] = (
            "no_mixed_groups: every selected group is unanimous or single, "
            "so group-relative advantages are all zero. Regrade with a "
            "stricter rubric or raise difficulty (fault_rate, harder asks) "
            "before training on this."
        )
    return selected, report


def _default_output(src: str) -> str:
    path = Path(src)
    return str(path.with_name(path.stem + ".rl" + (path.suffix or ".jsonl")))


def optimize_for_rl(
    source, *, output: str | None = None, trim_unanimous: bool = True
) -> dict[str, Any]:
    """Filter a JSONL path or a row list. Writes kept rows when given a path.

    Given a path and no ``output``, kept rows land next to the source as
    ``<name>.rl.jsonl``; the source file is never overwritten. Passing
    ``output`` equal to the source is an explicit overwrite and allowed.

    Drops three row classes, then (``trim_unanimous=True``) whole asks
    whose k rollouts all landed 0 or all landed 1, which are dead
    gradient for a grouped update:

    * ``do_nothing``: the situation wanted a tool (``ask_family=tool``,
      known tool, world/fault, or an actionable opener) and the agent
      never called one. Those rows teach the policy to stop using tools.
    * ``incomplete_junk``: empty or infra stub, degenerate text, leaked
      internal test text, raw tool markup, truncated reply, or a
      conversation that ends on the user or a tool result.
    * ``unusable_label``: ``reward`` (or ``qwen_reward``) missing or not 0/1.

    An injected missed tool call is not a drop by itself. A row that
    called the tool and reacted stays if the 0/1 label is present.
    The report's ``signal`` block is the within-ask contrast after
    filtering; rerun the simulator if ``mixed_rate`` is low.
    """
    if isinstance(source, (str, Path)):
        rows = load_jsonl(source)
        src = str(source)
    else:
        rows = list(source)
        src = ""
    kept, report = filter_rl_rows(rows)
    if trim_unanimous:
        kept, trim_report = trim_unanimous_groups(kept)
        report["n_kept"] = len(kept)
        report["n_dropped"] = report["n"] - len(kept)
        report["unanimous_groups_dropped"] = trim_report["n_groups_dropped"]
        report["unanimous_rows_dropped"] = trim_report["n_dropped"]
        report["signal"] = trim_report["signal"]
    else:
        report["signal"] = group_signal(kept)
    dest = output or (_default_output(src) if src else "")
    written = None
    if dest:
        written = write_jsonl(dest, kept)
    if isinstance(source, list):
        source[:] = kept
    report["path"] = written or src
    report["n_written"] = len(kept) if dest else 0
    return report


def recommend(
    tools: Sequence[dict] | None = None,
    policy: str = "",
    *,
    system_prompt: str | None = None,
    mode: str = "sft",
    target: int | None = None,
    mixed_rate: float = MIXED_RATE_DEFAULT,
) -> dict[str, Any]:
    """How much data this agent needs, from its own grid. No guessing.

    ``system_prompt=`` is the same text under ``simulate``'s spelling;
    ``policy=`` and ``system_prompt=`` are interchangeable here as there.

    Grounded two ways: the agent's measured covering grid (every cell wants
    ``SATURATION_COPIES`` visits, and selection wants surplus to choose
    from), and published post-training practice (curated agent SFT lands at
    500 to 2,000 trajectories: FireAct 500, LIMA 1,000, AgentTuning 1,866;
    agent RL uses 8 to 16 rollouts per prompt and drops all-pass/all-fail
    groups: DAPO 2025, Skywork-OR1 2025).

    Returns the numbers plus ``simulate_kwargs`` ready to splat, and
    ``reasoning`` lines that show the arithmetic.
    """
    tools = _tool_schemas(tools)
    from ..generate.coverage import SATURATION_COPIES
    from ..generate.scenarios import scenario_regions

    if system_prompt is not None:
        if policy and policy != system_prompt:
            raise ValueError("pass policy= or system_prompt=, not both")
        policy = system_prompt
    kind = "sft" if str(mode).lower() == "sft" else "rl"
    cells = len(scenario_regions(list(tools or []), policy, mode=str(mode).lower()))
    reasoning = [f"covering grid: {cells} cells for this agent"]
    if kind == "sft":
        goal = int(target or SFT_TARGET_DEFAULT)
        by_grid = cells * SATURATION_COPIES
        raw = max(by_grid, SELECTION_SURPLUS * goal)
        raw = int(-(-raw // BUDGET_ROUNDING) * BUDGET_ROUNDING)
        reasoning += [
            f"saturation wants {SATURATION_COPIES} visits per cell = {by_grid} rows",
            f"selection wants about {SELECTION_SURPLUS}x its target of {goal} to choose from",
            f"generate {raw}, select {goal} diverse 1-labeled rows",
            "time_budget off: a sized run stops on rows, not the clock",
        ]
        return {
            "mode": "sft",
            "grid_cells": cells,
            "budget": raw,
            "optimize_target": goal,
            "simulate_kwargs": {"mode": "sft", "budget": raw, "time_budget": None},
            "reasoning": reasoning,
        }
    goal = int(target or SFT_TARGET_DEFAULT)
    k = RL_ROLLOUTS_PER_ASK
    # Whole-group selection keeps only asks whose k rollouts disagree.
    # The surviving fraction is the agent's, not ours: a struggling agent
    # mixes on half its asks; a competent agent on a cold-start grid
    # measured 4% (field test, 48x8, hosted Qwen). Probe first: 12 asks,
    # grade, group_signal, then pass the measured rate back in here.
    rate = min(1.0, max(MIXED_RATE_FLOOR, float(mixed_rate)))
    # Expected mixed rows = situations * k * rate, so situations =
    # goal / (k * rate). Rounding k * rate to an integer first (the old
    # form) collapsed to 1 below rate 1/16 and under-provisioned by 3x
    # at the 4% rate measured in the field.
    situations = max(2 * cells, math.ceil(goal / (k * rate)))
    raw = situations * k
    reasoning += [
        f"k={k} rollouts per ask; assumed mixed-group rate {rate:.0%} "
        "(measure it with a 12-ask probe and group_signal, then recompute)",
        f"{situations} asks x {k} = {raw} rows to select about {goal} mixed-group rows",
        "a low measured rate means the grid is too easy for this agent: "
        "use traces=, harder cells, or a stricter judge prompt before "
        "buying more rollouts",
        "in post-training terms the mixed rate is the pass@k - pass@1 headroom "
        "(group_signal reports both); pass@k near pass@1 means nothing to learn",
    ]
    return {
        "mode": "rl",
        "grid_cells": cells,
        "budget": raw,
        "situations": situations,
        "rollouts_per_request": k,
        "mixed_rate": rate,
        "optimize_target": goal,
        "simulate_kwargs": {
            "mode": "rl",
            "situations": situations,
            "budget": raw,
            "time_budget": None,
        },
        "reasoning": reasoning,
    }


def optimize(
    source,
    *,
    mode: str | None = None,
    target: int = 1000,
    output: str | None = None,
    band: tuple[float, float] = DEFAULT_BAND,
    enforce_band: bool = True,
    select: str = "top_per_prompt",
    min_reward: float = 1.0,
    endorsed: Sequence[str] = (),
    truncated: str = "drop",
    order: str = "spread",
    audit: dict[str, Any] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Select the rows worth training on, for SFT or RL, one call after grading.

    Reach for it once rows carry ``reward``. It returns ``(rows, report)``:
    the kept rows in training order, and a report saying which mode ran,
    what each gate dropped and why, and for RL a ``hack_scan`` of what the
    reward is actually tracking. It writes the rows to ``output`` when
    given, or to ``<name>.<mode>.jsonl`` next to a path source, and never
    overwrites the source file unless ``output`` names it explicitly.

    * ``source``: a ``SimulationData``, a row list, or a JSONL path.
    * ``mode``: ``"sft"`` or ``"rl"``. Defaults to the run's own mode for a
      ``SimulationData`` and to ``"rl"`` otherwise. SFT picks diverse
      correct demonstrations (``select_for_sft``); RL keeps whole mixed
      groups, never a split one (``select_for_rl``). Both drop a row whose
      reply quotes its own privileged context first
      (``drop_privileged_leaks``; ``privileged_leaks_dropped`` in the
      report), so ``export_dataset`` never refuses what was kept.
    * ``target``: about how many rows to keep, 1000 by default.
    * ``band``: the RL difficulty band as a pass-rate range, ``(0.2, 0.8)``
      by default: asks the policy always or never solves carry no advantage
      (Lambert 2025, chapter Reasoning, difficulty filtering at 20 to 80
      percent; DAPO's dynamic sampling, arXiv:2503.14476).
      ``enforce_band=False`` only ranks out-of-band asks last instead of
      dropping them. ``order`` is ``"spread"`` across pass rates (default) or
      ``"middle"`` first.
    * ``select`` (``"top_per_prompt"``) and ``min_reward`` (1.0): the SFT
      picker and the reward a demonstration needs, as in
      ``select_for_sft``.
    * ``endorsed``: what the reward should track, as substrings of feature
      names (``"tool:lookup_order"``), so the RL report's ``hack_scan`` can
      call a shortcut a hack.
    * ``truncated``: what happens to a rollout cut at the token cap
      (DAPO's overlong handling, Yu et al. 2025, arXiv:2503.14476): ``"drop"``
      removes it (the default), ``"keep"`` leaves it in with ``overlong=True``
      and its own reward, ``"penalize"`` keeps it as a failure that counts
      (reward 0, the judged score under ``reward_before_penalty``). A row
      counts as truncated when the engine stamped it so (``finish_reason``
      ``"length"``, or a step marked ``truncated``), when the grader's
      ``reason`` says truncated or cut off, or when the reply text stops
      without reaching its end; the stamp is read first, since the backend
      trims a capped reply to its last sentence and a re-grade overwrites
      the grader's reason (``hygiene.is_truncated``).

    ```python
    rows, report = wai.optimize(data, mode="rl", endorsed=["tool:lookup_order"])
    print(report["mode"], len(rows))
    ```
    """
    resolved = mode
    src = ""
    has_tools = True
    if hasattr(source, "trajectories"):
        rows = list(source.trajectories)
        resolved = resolved or getattr(source, "mode", None)
        profile = getattr(source, "profile", None)
        if profile is not None:
            has_tools = bool(getattr(profile, "tools", None))
    elif isinstance(source, (str, Path)):
        rows = load_jsonl(source)
        src = str(source)
    else:
        rows = list(source)
    resolved = "sft" if str(resolved or "").lower() == "sft" else "rl"
    if resolved == "sft":
        picked, report = select_for_sft(rows, target=target, select=select, min_reward=min_reward)
    else:
        picked, report = select_for_rl(
            rows,
            target=target,
            lo=float(band[0]),
            hi=float(band[1]),
            enforce_band=enforce_band,
            has_tools=has_tools,
            audit=audit,
            endorsed=endorsed,
            truncated=truncated,
            order=order,
        )
    report["mode"] = resolved
    dest = output
    if not dest and src:
        path = Path(src)
        dest = str(path.with_name(path.stem + f".{resolved}" + (path.suffix or ".jsonl")))
    if dest:
        report["path"] = write_jsonl(dest, picked)
        report["n_written"] = len(picked)
    return picked, report


optimize_rows = optimize_for_rl
