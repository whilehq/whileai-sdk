"""Human labels on rows: who said what, and do they agree.

Lambert 2025, chapter Preference Data: the trusted label a judge is checked
against is a person's, several people disagree, and the disagreement is
signal, not noise to average away. ``judge_trust`` and ``judge_agreement``
read ``gold_reward`` and ``gold_kind``, and a person's or a program's labels
count as a measurement of the judge; a model's labels are marked as such.

``attach_labels`` takes labels from a file or a list (each with a row
identity, a 0/1 label, and optionally an annotator, a note and a time),
keeps every label on the row as ``gold_labels``, and sets
``gold_reward`` to the majority (a tie leaves it unset and counts as a
disagreement). ``annotator_agreement`` reads those records back:
per-annotator counts, the share of multi-labeled rows where everyone
agreed, and Cohen's kappa for a pair.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .agreement import (
    GOLD_KIND_ALIASES,
    GOLD_KIND_KEY,
    GOLD_KINDS,
    _kappa,
    _label,
    gold_kind_of,
    row_key,
)

GOLD_KEY = "gold_reward"
LABELS_KEY = "gold_labels"


def _load(labels: Any) -> list[dict]:
    if isinstance(labels, (str, Path)):
        from .quality import load_jsonl

        return _records(load_jsonl(labels), f"attach_labels(labels={str(labels)!r})")
    if isinstance(labels, Mapping):
        return [{"key": str(k), "label": v} for k, v in labels.items()]
    return _records(list(labels), "attach_labels(labels=[...])")


def _records(items: list, where: str) -> list[dict]:
    """The label records, or a ``ValueError`` naming the first item that is
    not one. A plain ``[0, 1, 1, 0]`` used to be filtered to nothing and
    reported as zero labels, zero invalid (#685): a label with no row
    identity cannot be attached, and the loss should be loud."""
    stray = [x for x in items if not isinstance(x, dict)]
    if stray:
        raise ValueError(
            f"{where}: {type(stray[0]).__name__} {stray[0]!r} is not a label record; "
            "labels is a JSONL path, a list of dicts (each with 'label' and a row identity: "
            "'key', 'rollout_id', 'scenario_id' + 'rollout_index', or 'prompt'), or a "
            "{key: label} mapping. A bare 0/1 list cannot say which row each label belongs to."
        )
    return items


def _label_key(item: Mapping[str, Any]) -> str | None:
    """The row a label names: an explicit ``key`` / ``rollout_id``, a
    scenario id plus rollout index, or a prompt plus final text (the same
    identity ``judge_agreement`` matches on)."""
    if item.get("key"):
        return str(item["key"])
    if item.get("rollout_id") or (
        item.get("scenario_id") is not None and item.get("rollout_index") is not None
    ):
        return row_key(dict(item))
    if item.get("prompt") is not None:
        return row_key({"prompt": item.get("prompt"), "final_text": item.get("final_text", "")})
    return None


def attach_labels(
    rows: Sequence[dict],
    labels: Any,
    *,
    annotator: str | None = None,
    kind: str = "human",
    replace: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
    """Write hand labels onto rows (in place) and return ``(rows, report)``.

    ``labels`` is a JSONL path, a list of dicts, or a ``{key: label}``
    mapping. A dict label names its row by ``key`` / ``rollout_id``,
    ``scenario_id`` + ``rollout_index``, or ``prompt`` (+ ``final_text``),
    and carries ``label`` (or ``reward`` / ``gold_reward``: 0 or 1), and
    optionally ``annotator``, ``note``, ``ts``. ``annotator`` here is the
    default for labels that name none. ``kind`` is recorded on each label:
    ``"human"`` for a person's, ``"program"`` (or its alias
    ``"verifier"``) for a deterministic rule's (execution match, a unit
    test, a rule over tool calls), ``"model"`` for a stronger model's.
    Any other string raises ``ValueError`` naming the accepted set, so a
    typo cannot silently downgrade the gold (#343).

    Each row gains ``gold_labels`` (every label, appended unless
    ``replace``), ``gold_reward``, the majority of its labels, and
    ``gold_kind``: the one kind every label on the row shares, else
    ``"mixed"``. ``judge_trust`` and ``judge_agreement`` count human and
    program gold as a measurement of the judge: a deterministic rule is
    at least as strong a gold as a rater, since it cannot be argued into
    a pass and agrees with itself on every run (Lambert 2025, chapter
    Evaluation, verifiable rewards). A tie leaves both unset. Labels that
    name no row, or carry no 0/1 value, are counted and listed. A list or
    file holding anything but dicts (a bare ``[0, 1, 1, 0]``) raises
    naming the item and the accepted shapes, since a label with no row
    identity cannot be attached (#685).
    """
    kind = GOLD_KIND_ALIASES.get(kind, kind)
    if kind not in GOLD_KINDS:
        raise ValueError(
            f"attach_labels(kind={kind!r}): kind is one of {', '.join(GOLD_KINDS)} "
            "('verifier' reads as 'program')"
        )
    # every identity a label may use points at the row
    index: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        index.setdefault(row_key(row), row)
        if row.get("rollout_id"):
            index.setdefault(str(row["rollout_id"]), row)
        if row.get("scenario_id") is not None and row.get("rollout_index") is not None:
            index.setdefault(f"{row['scenario_id']}#{row['rollout_index']}", row)
        index.setdefault(
            row_key({"prompt": row.get("prompt"), "final_text": row.get("final_text", "")}), row
        )
    items = _load(labels)
    now = time.time()
    matched = 0
    unmatched: list[str] = []
    invalid = 0
    touched: dict[str, dict] = {}
    for item in items:
        key = _label_key(item)
        raw = item.get("label", item.get("reward", item.get(GOLD_KEY)))
        value = _label(raw)
        if value is None:
            invalid += 1
            continue
        target = index.get(key or "")
        if target is None:
            unmatched.append(str(key))
            continue
        own = str(item.get("kind") or kind)
        own = GOLD_KIND_ALIASES.get(own, own)
        if own not in GOLD_KINDS:
            raise ValueError(
                f"attach_labels: label {key!r} carries kind={own!r}; kind is one of "
                f"{', '.join(GOLD_KINDS)} ('verifier' reads as 'program')"
            )
        record = {
            "label": value,
            "annotator": str(item.get("annotator") or annotator or "unknown"),
            "kind": own,
            "ts": item.get("ts") or now,
        }
        if item.get("note"):
            record["note"] = str(item["note"])
        existing = target.get(LABELS_KEY)
        if not isinstance(existing, list) or (replace and key not in touched):
            existing = []
            target[LABELS_KEY] = existing
        existing.append(record)
        touched[key or ""] = target
        matched += 1
    ties = 0
    for row in touched.values():
        records = [r for r in row[LABELS_KEY] if _label(r.get("label")) is not None]
        votes = [int(r["label"]) for r in records]
        ones = sum(votes)
        zeros = len(votes) - ones
        if ones > zeros:
            row[GOLD_KEY] = 1
        elif zeros > ones:
            row[GOLD_KEY] = 0
        else:
            row.pop(GOLD_KEY, None)
            row.pop(GOLD_KIND_KEY, None)
            ties += 1
            continue
        row[GOLD_KIND_KEY] = gold_kind_of(r.get("kind") for r in records)
    report: dict[str, Any] = {
        "labels": len(items),
        "matched": matched,
        "unmatched": len(unmatched),
        "unmatched_keys": unmatched[:10],
        "invalid": invalid,
        "rows_labeled": len(touched),
        "ties": ties,
        "annotators": sorted(
            {str(r["annotator"]) for row in touched.values() for r in row[LABELS_KEY]}
        ),
    }
    if ties:
        report["warnings"] = [
            f"{ties} row(s) split evenly between annotators; gold_reward left unset there, "
            "read gold_labels and decide"
        ]
    return [r for r in rows if isinstance(r, dict)], report


def annotator_agreement(rows: Sequence[dict]) -> dict[str, Any]:
    """How the annotators on ``gold_labels`` agree with each other.

    ``per_annotator``: labels given and pass share. ``multi_labeled``: rows
    with two or more annotators; ``unanimous`` the share of those where
    every label matched; ``kappa`` Cohen's kappa when exactly two
    annotators labeled the same rows (``pair`` names them), else ``None``.
    ``disagreements`` lists the split rows (prompt, labels) so a person
    can read the ones the guideline did not settle (Lambert 2025, chapter
    Preference Data).
    """
    per: dict[str, dict[str, int]] = {}
    multi = 0
    unanimous = 0
    disagreements: list[dict[str, Any]] = []
    pair_votes: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        labels = row.get(LABELS_KEY)
        if not isinstance(labels, list) or not labels:
            continue
        by_ann: dict[str, int] = {}
        for rec in labels:
            if not isinstance(rec, dict):
                continue
            value = _label(rec.get("label"))
            if value is None:
                continue
            name = str(rec.get("annotator") or "unknown")
            by_ann[name] = value  # the annotator's latest label wins
            stats = per.setdefault(name, {"labels": 0, "passes": 0})
            stats["labels"] += 1
            stats["passes"] += value
        if len(by_ann) >= 2:  # noqa: PLR2004  # agreement needs two annotators
            multi += 1
            values = set(by_ann.values())
            if len(values) == 1:
                unanimous += 1
            else:
                disagreements.append(
                    {"prompt": str(row.get("prompt") or "")[:160], "labels": dict(by_ann)}
                )
            names = sorted(by_ann)
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    pair_votes.setdefault((a, b), []).append((by_ann[a], by_ann[b]))
    kappa = None
    pair = None
    if pair_votes:
        (a, b), votes = max(pair_votes.items(), key=lambda kv: len(kv[1]))
        tp = sum(1 for x, y in votes if x == 1 and y == 1)
        tn = sum(1 for x, y in votes if x == 0 and y == 0)
        fp = sum(1 for x, y in votes if x == 1 and y == 0)
        fn = sum(1 for x, y in votes if x == 0 and y == 1)
        kappa = _kappa(tp, fp, fn, tn)
        pair = [a, b]
    return {
        "per_annotator": {
            name: {
                "labels": s["labels"],
                "pass_share": round(s["passes"] / s["labels"], 4) if s["labels"] else None,
            }
            for name, s in sorted(per.items())
        },
        "multi_labeled": multi,
        "unanimous": round(unanimous / multi, 4) if multi else None,
        "kappa": round(kappa, 4) if kappa is not None else None,
        "pair": pair,
        "disagreements": disagreements[:20],
    }


__all__ = ["GOLD_KEY", "GOLD_KIND_KEY", "LABELS_KEY", "annotator_agreement", "attach_labels"]
