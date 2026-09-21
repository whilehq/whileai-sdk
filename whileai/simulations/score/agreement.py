"""Does the judge agree with labels you trust?

An LLM judge is a reward model, and a reward model is only as good as its
accuracy on a held-out set you labeled yourself (Lambert 2025, chapter Reward
Modeling, "Suggested Experiments": 50 to 200 pairs is enough to tune on).
Without that number a training run optimizes the judge's habits, not the
behavior. ``judge_agreement`` is that number, plus the two directions of
disagreement, which are not symmetric: a judge that passes a failure teaches
the failure (a reward hack in RL, a bad demonstration in SFT); a judge that
fails a pass only wastes a row.

The same function measures self-consistency: judge the same rows twice
and pass the second run as ``gold``. Agreement below what two humans
would reach is the ceiling on any drift alarm built on this judge. That
is a consistency number, not an accuracy one, so the report says so:
``gold_kind`` names where the labels came from and ``ok`` is false
unless they came from a person or a program (``allow_model_gold=True``
opts out).

A program's labels are a measurement. Lambert 2025, chapter Evaluation,
on verifiable rewards: a deterministic rule (execution match, a unit
test, a rule over tool calls) is the strongest grader there is, because
it cannot be talked into a pass and two runs of it agree with each
other every time, where two people do not (Zheng et al. 2023 put
human-human agreement at 81%). So ``gold_kind == "program"`` is trusted
on the same footing as ``"human"``; the report carries the kind so a
reviewer sees which one the number rests on (#343).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from ..defaults import MIN_GOLD

# MIN_GOLD = 50 (``defaults``): below it the accuracy estimate has a
# +/-0.1 Wilson error bar and the guidance of Lambert 2025, chapter Reward
# Modeling (a 50- to 200-example held-out set), is not met.
# LEAK_THRESHOLD = 0.1: a judge that passes one in ten gold failures leaks
# that many bad rows into a training set at the pass rate of the run; ten
# points is the same sensitivity as FLIP_FLAG (convention).
LEAK_THRESHOLD = 0.1
# Where a row's gold label came from: "human" from attach_labels(kind="human"),
# "program" from attach_labels(kind="program") (a verifier, a unit test, a
# rule over tool calls), "model" from a model's labels or a second judge
# pass, "unknown" when a row carries gold_reward with no record of who
# wrote it (older data).
GOLD_KIND_KEY = "gold_kind"
#: The kinds ``attach_labels(kind=)`` accepts; ``"verifier"`` is read as
#: ``"program"``. The two trusted kinds are the ones that measure the judge.
GOLD_KINDS = ("human", "program", "model")
TRUSTED_GOLD_KINDS = ("human", "program")
GOLD_KIND_ALIASES = {"verifier": "program"}
MODEL_GOLD_REASON = (
    "The gold labels came from a model, not a person, so this does not measure the "
    "judge. Label 50 rows with attach_labels(rows, labels, kind='human') and run again."
)
# Gold set by hand (``row["gold_reward"] = 1``) carries no kind, so it read as
# a model's labels and the fix, attach_labels(kind="human"), was never named
# for the case that needs it most: a person who did label the rows.
UNKNOWN_GOLD_REASON = (
    "These rows carry gold_reward with no record of who wrote it, so it does not measure "
    "the judge. If you labeled them yourself, attach the labels with "
    "attach_labels(rows, labels, kind='human'), which marks them as a person's; a model's "
    "labels stay model gold."
)


def gold_kind_reason(kind: str | None) -> str:
    """Why gold that is neither a person's nor a program's does not measure
    the judge, by kind.

    ``"unknown"`` (gold_reward written by hand, no ``gold_kind``) gets the
    sentence that names ``attach_labels(kind="human")`` as the way to say
    a person wrote them; ``"model"`` and ``"mixed"`` get the model-gold
    sentence. A trusted kind (``"human"``, ``"program"``) has no reason to
    give; the caller does not ask.
    """
    return UNKNOWN_GOLD_REASON if kind == "unknown" else MODEL_GOLD_REASON


def gold_words(kind: str | None) -> str:
    """The gold labels as a warning names them: ``human labels`` for a
    person's, ``gold labels (program)`` for any other recorded kind, so no
    line asserts a provenance the rows do not carry (#343)."""
    return "human labels" if kind == "human" else f"gold labels ({kind or 'unknown'})"


def is_trusted_gold(kind: str | None, allow_model_gold: bool = False) -> bool:
    """Whether labels of this kind measure the judge: a person's or a
    program's always, anything else only with ``allow_model_gold``."""
    return kind in TRUSTED_GOLD_KINDS or allow_model_gold


def missing_side_note(
    n_reward: int, n_gold: int, *, reward: str = "reward", gold: str = "gold_reward"
) -> str:
    """Which half of an agreement check is missing, said on its own.

    One sentence covering both halves ("no rows carry both 'reward' and a
    gold label") sent a tester to look at the labels when the rewards
    were what was missing. Each half now names its own call.
    """
    if not n_reward and not n_gold:
        return (
            f"no row has a reward and no row has a gold label: score the rows first with "
            f"run_judge(rows, judge) or evaluate(data, judge), then hand-label a sample with "
            f"attach_labels(rows, labels, kind='human'), which writes {gold!r}"
        )
    if not n_reward:
        return (
            f"no row has a reward: score them first with run_judge(rows, judge) or "
            f"evaluate(data, judge), which writes {reward!r}. Passing judge= to judge_trust "
            "only runs the perturbation probes; it does not score the rows."
        )
    if not n_gold:
        return (
            f"no row has a gold label: attach_labels(rows, labels, kind='human') writes "
            f"{gold!r} (0/1) and marks it as a person's, and {MIN_GOLD} labeled rows is the "
            "sample to aim for."
        )
    return (
        f"{n_reward} row(s) have a reward and {n_gold} have a gold label, but no row has "
        "both: label the rows that were scored (attach_labels matches on rollout_id, "
        "scenario_id plus rollout_index, or prompt plus final_text)."
    )


def gold_kind_of(kinds: Any) -> str | None:
    """One word for a set of label kinds: the kind itself when every label
    shares it (``"human"``, ``"program"``, ``"model"``, ``"unknown"``);
    when a trusted kind sits beside one other, that other kind (a person
    and a model on one row is model gold); ``"mixed"`` when there are
    several others. ``"verifier"`` reads as ``"program"``."""
    names = {GOLD_KIND_ALIASES.get(str(k or "unknown"), str(k or "unknown")) for k in kinds}
    if not names:
        return None
    if len(names) == 1:
        return next(iter(names))
    others = sorted(k for k in names if k not in TRUSTED_GOLD_KINDS)
    return others[0] if len(others) == 1 else "mixed"


def _label(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value in (0, 1):
        return int(value)
    return None


def row_key(row: dict) -> str:
    """Stable identity for matching a judged row to its gold twin."""
    if row.get("rollout_id"):
        return str(row["rollout_id"])
    if row.get("scenario_id") is not None and row.get("rollout_index") is not None:
        return f"{row['scenario_id']}#{row['rollout_index']}"
    text = f"{row.get('prompt', '')}\x1f{row.get('final_text', '')}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    judge_pass = (tp + fp) / n
    gold_pass = (tp + fn) / n
    pe = judge_pass * gold_pass + (1 - judge_pass) * (1 - gold_pass)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def judge_agreement(
    rows: Sequence[dict],
    gold: str | Sequence[dict] = "gold_reward",
    *,
    reward: str = "reward",
    allow_model_gold: bool = False,
) -> dict[str, Any]:
    """Agreement between the judge's ``reward`` and a trusted label.

    ``gold`` is either a key on the same rows (default ``gold_reward``,
    the field ``attach_labels`` fills when you hand-label a sample) or a
    second row list from another scoring pass, matched by rollout id,
    scenario id plus rollout index, or prompt plus final text. Only exact
    0/1 labels on both sides count; partial scores and unjudged rows are
    reported as skipped (``n_skipped``), not guessed. A fractional reward
    is what a ``Rubric`` of plain principles returns (the mean of its
    criteria), so a judge built that way loses every partially met row
    here; ``judge_trust`` counts that share against ``MAX_SKIPPED_SHARE``
    and says so, since the rows kept are the ones the judge was sure
    about and agreement over them reads high by construction (#345).

    Returns ``n``, ``agreement``, ``kappa`` (Cohen, chance-corrected), the
    confusion counts, ``pass_when_gold_fail`` (the leak rate: gold
    failures the judge passed) and ``fail_when_gold_pass``, both pass
    rates, ``gold_kind`` (where the labels came from: ``"human"``,
    ``"program"``, ``"model"``, ``"unknown"`` for rows with no record),
    ``ok`` (rows were compared and the labels are a person's or a
    program's), and ``warnings``. A second judge pass is model gold; rows
    with ``gold_reward`` but no ``gold_kind`` are unknown; either makes
    ``ok`` false with the reason unless ``allow_model_gold=True``. A
    program's labels (``attach_labels(kind="program")``: a verifier, a
    unit test, a rule over tool calls) are trusted like a person's, since
    a deterministic rule is at least as strong a gold as a rater (Lambert
    2025, chapter Evaluation, verifiable rewards).
    """
    judged = [r for r in rows if isinstance(r, dict)]
    pairs: list[tuple[int, int]] = []
    kinds: set[str] = set()
    skipped = 0
    unmatched = 0
    # counted so a report with nothing to compare can say which half is
    # missing, rather than naming both and leaving the reader to guess
    n_reward = 0
    n_gold = 0
    if isinstance(gold, str):
        for row in judged:
            j, g = _label(row.get(reward)), _label(row.get(gold))
            n_reward += j is not None
            n_gold += g is not None
            if j is None or g is None:
                skipped += 1
                continue
            pairs.append((j, g))
            kinds.add(str(row.get(GOLD_KIND_KEY) or "unknown"))
    else:
        kinds.add("model")
        by_key: dict[str, dict] = {}
        for row in gold:
            if isinstance(row, dict):
                by_key.setdefault(row_key(row), row)
        for row in judged:
            twin = by_key.get(row_key(row))
            if twin is None:
                unmatched += 1
                continue
            j, g = _label(row.get(reward)), _label(twin.get(reward))
            if j is None or g is None:
                skipped += 1
                continue
            pairs.append((j, g))

    tp = sum(1 for j, g in pairs if j == 1 and g == 1)
    fp = sum(1 for j, g in pairs if j == 1 and g == 0)
    fn = sum(1 for j, g in pairs if j == 0 and g == 1)
    tn = sum(1 for j, g in pairs if j == 0 and g == 0)
    n = len(pairs)
    leak = fp / (fp + tn) if (fp + tn) else None
    miss = fn / (fn + tp) if (fn + tp) else None
    warnings: list[str] = []
    if n == 0:
        warnings.append(
            missing_side_note(n_reward, n_gold, reward=reward, gold=gold)
            if isinstance(gold, str)
            else "no judged row matched a gold row by rollout id, scenario id, or prompt"
        )
    elif n < MIN_GOLD:
        warnings.append(
            f"{n} gold rows; the accuracy estimate is coarse below {MIN_GOLD} "
            "(Lambert 2025, chapter Reward Modeling, suggests 50-200)"
        )
    if leak is not None and leak >= LEAK_THRESHOLD and fp:
        warnings.append(
            f"judge passed {fp} of {fp + tn} gold failures ({leak:.0%}); those rows train "
            "the failure, not the behavior"
        )
    gold_kind = gold_kind_of(kinds) if n else None
    trusted = is_trusted_gold(gold_kind, allow_model_gold)
    if n and not trusted:
        warnings.append(gold_kind_reason(gold_kind))
    return {
        "n": n,
        "n_skipped": skipped,
        "n_unmatched": unmatched,
        "agreement": round((tp + tn) / n, 4) if n else None,
        "kappa": _kappa(tp, fp, fn, tn),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "pass_when_gold_fail": round(leak, 4) if leak is not None else None,
        "fail_when_gold_pass": round(miss, 4) if miss is not None else None,
        "judge_pass_rate": round((tp + fp) / n, 4) if n else None,
        "gold_pass_rate": round((tp + fn) / n, 4) if n else None,
        "gold_kind": gold_kind,
        "ok": bool(n) and trusted,
        "warnings": warnings,
    }


__all__ = [
    "GOLD_KINDS",
    "GOLD_KIND_ALIASES",
    "GOLD_KIND_KEY",
    "LEAK_THRESHOLD",
    "MIN_GOLD",
    "MODEL_GOLD_REASON",
    "TRUSTED_GOLD_KINDS",
    "UNKNOWN_GOLD_REASON",
    "gold_kind_of",
    "gold_kind_reason",
    "gold_words",
    "is_trusted_gold",
    "judge_agreement",
    "missing_side_note",
    "row_key",
]
