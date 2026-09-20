"""A model-written prompt set for the refund environment.

The offline template writer in ``reward.build_prompts`` gives about 117
distinct prompts at the default 200 situations and seed 0 (the README has
the exact count by Python version), so the holdout is 25 prompts and
every pass@1 interval is about a quarter wide. This module is the
model writer: for each template seed, an instruct model writes several
customer messages in the same situation (different tone, length, detail,
kind of customer), the rule's ``case_for`` checks each one still belongs
to its category (names the order id, names none, or is off topic), and
near-duplicates are dropped. Every written prompt keeps its seed's
``scenario_id``, so ``split_holdout`` never puts a paraphrase of a train
situation into the holdout.

The pure parts are here and unit-tested; ``write_prompts_modal.py`` runs
the model. ``prompts.jsonl`` next to this file is one such set, checked in
so the runs in the READMEs are reproducible.
"""

from __future__ import annotations

import json
import re
from typing import Any

from reward import case_for

WRITER_SYSTEM = (
    "You write realistic opening messages that customers of an online store send "
    "to its support assistant. Each message is the customer's first message in a "
    "conversation. Vary the customers: tone (curt, polite, angry, confused, "
    "chatty), length (one line to a short paragraph), and details (item, price, "
    "dates, what went wrong). Never write the assistant's reply."
)

ANGLES = [
    "Write {n} different messages a customer might send in this situation.",
    "Write {n} more, each from a different kind of customer: a first-time buyer, "
    "a business account, someone typing on a phone with typos, someone who already "
    "contacted support once, a non-native English speaker, someone in a hurry.",
]

_ORD = re.compile(r"\bORD[-_ ]?\d{3,6}\b", re.I)
_ARRAY = re.compile(r"\[.*\]", re.S)
_WORD = re.compile(r"[a-z0-9]+")


def category(case: dict[str, Any]) -> str:
    """``with_id``, ``no_id`` (about an order, no id) or ``off_topic``."""
    if case.get("order_id"):
        return "with_id"
    return "no_id" if case.get("in_domain") else "off_topic"


def writer_messages(seed: dict[str, Any], angle: int = 0, n: int = 6) -> list[dict[str, str]]:
    """The chat for one writer call: the seed situation, the category rule,
    and the output format."""
    cat = category(seed["case"])
    if cat == "with_id":
        rule = (
            f"Every message must include the order id {seed['case']['order_id']} "
            "exactly once, written as the customer would write it."
        )
    elif cat == "no_id":
        rule = (
            "The customer is asking about an order, refund, return, charge or "
            "delivery but does not give any order id or order number, because "
            "they do not know it or did not think to. Never include an id."
        )
    else:
        rule = (
            "The message is not about an order, refund, return, charge or "
            "delivery at all: something unrelated the customer asks the store "
            "anyway. Never include an order id."
        )
    ask = ANGLES[angle % len(ANGLES)].format(n=n)
    user = (
        f"Situation, as one customer put it: {seed['prompt']!r}\n\n"
        f"{ask}\n{rule}\n"
        f"Reply with a JSON array of exactly {n} strings and nothing else."
    )
    return [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": user}]


def parse_messages(text: str) -> list[str]:
    """The strings in the first JSON array of the reply; a fallback reads
    quoted or bulleted lines when the model wrapped the array in prose."""
    m = _ARRAY.search(text or "")
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(s).strip() for s in arr if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass
    out: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*0123456789.) ").strip()
        if len(line) > 2 and line[0] in "\"'" and line[-1] in "\"',":
            out.append(line.strip("\"',").strip())
    return out


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def keep(
    candidates: list[tuple[str, dict[str, Any]]],
    *,
    min_chars: int = 12,
    max_chars: int = 700,
    jaccard: float = 0.8,
) -> list[dict[str, Any]]:
    """Filter written messages: right length, same category as the seed (the
    id present or absent as the rule demands), no exact or near duplicate of
    an earlier keep. Each kept item is a prompt row with its seed's scenario."""
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    token_sets: list[set[str]] = []
    for text, seed in candidates:
        text = " ".join(text.split())
        if not (min_chars <= len(text) <= max_chars):
            continue
        case = case_for(text)
        want = category(seed["case"])
        if category(case) != want:
            continue
        if want == "with_id" and case["order_id"] != seed["case"]["order_id"]:
            continue
        key = _norm(text)
        if key in seen:
            continue
        toks = _tokens(text)
        if any(len(toks & t) / max(1, len(toks | t)) >= jaccard for t in token_sets):
            continue
        seen.add(key)
        token_sets.append(toks)
        kept.append(
            {
                "prompt": text,
                "case": case,
                "scenario_id": seed["scenario_id"],
                "seed": seed["prompt"],
            }
        )
    return kept


def load_prompts(path: str) -> list[dict[str, Any]]:
    """Prompt rows from a ``prompts.jsonl``; the case is re-read from the
    text so an edited file still matches the reward."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            if not prompt:
                continue
            out.append(
                {
                    "prompt": prompt,
                    "case": case_for(prompt),
                    "scenario_id": row.get("scenario_id") or prompt,
                }
            )
    return out


def split_holdout_stratified(
    items: list[dict[str, Any]], fraction: float = 0.2, *, min_scenarios: int = 2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by scenario, category by category, so the holdout holds every
    kind of prompt. A plain hash split over 67 scenarios put all of the
    no-id situations in train once, and the holdout could not see a policy
    that learned to always call the tool. Each category gives up
    ``fraction`` of its scenarios (at least ``min_scenarios`` when it has
    that many), chosen by a stable hash so the split is reproducible."""
    import hashlib

    by_cat: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for it in items:
        by_cat.setdefault(category(it["case"]), {}).setdefault(str(it["scenario_id"]), []).append(
            it
        )
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for cat_name in sorted(by_cat):
        scenarios = by_cat[cat_name]
        ranked = sorted(
            scenarios, key=lambda sid: hashlib.sha256(f"{cat_name}:{sid}".encode()).hexdigest()
        )
        n_held = max(min(min_scenarios, len(ranked) - 1), round(len(ranked) * fraction))
        n_held = min(n_held, max(0, len(ranked) - 1))
        held_ids = set(ranked[:n_held])
        for sid, rows in scenarios.items():
            (held if sid in held_ids else train).extend(rows)
    return train, held


def pass_by_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """pass@1 and the tool-call rate per prompt category over graded rows,
    the breakdown a headline pass@1 hides when one category dominates."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        k = category(case_for(str(r.get("prompt") or "")))
        a = acc.setdefault(k, [0.0, 0.0, 0.0])
        a[0] += float(r.get("reward") or 0)
        a[1] += 1.0 if "<tool_call>" in str(r.get("final_text") or "") else 0.0
        a[2] += 1
    return {
        k: {
            "pass_at_1": round(v[0] / v[2], 4),
            "tool_call_rate": round(v[1] / v[2], 4),
            "rows": int(v[2]),
        }
        for k, v in sorted(acc.items())
    }


def balance(
    items: list[dict[str, Any]], min_share: float = 0.25, *, max_repeat: int = 6
) -> list[dict[str, Any]]:
    """Repeat the prompts of any category below ``min_share`` of the list
    until it reaches that share (each prompt at most ``max_repeat`` times).

    Why frequency and not reward: a group-relative update only learns from
    a prompt when that prompt is sampled, and a preference round only pairs
    the prompts it sampled. With 10% no-id prompts the with-id rows carry
    the gradient and the policy learns "call the tool" before "unless there
    is no id"; the stratified holdout showed no-id pass@1 falling 0.95 to
    0.75 for both GRPO and DPO. Oversampling makes the minority visible to
    the update at the rate it matters, without touching the reward. Order
    is kept stable; the repeats are appended."""
    if min_share <= 0 or not items:
        return list(items)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        by_cat.setdefault(category(it["case"]), []).append(it)
    out = list(items)
    for name in sorted(by_cat):
        rows = by_cat[name]
        share = len(rows) / len(out)
        repeat = 1
        while share < min_share and repeat < max_repeat:
            out.extend(rows)
            repeat += 1
            share = len(rows) * repeat / len(out)
    return out


def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    cats = {"with_id": 0, "no_id": 0, "off_topic": 0}
    for it in items:
        cats[category(it["case"])] += 1
    return {"prompts": len(items), "scenarios": len({it["scenario_id"] for it in items}), **cats}


__all__ = [
    "ANGLES",
    "WRITER_SYSTEM",
    "balance",
    "category",
    "keep",
    "load_prompts",
    "parse_messages",
    "pass_by_category",
    "split_holdout_stratified",
    "summary",
    "writer_messages",
]
