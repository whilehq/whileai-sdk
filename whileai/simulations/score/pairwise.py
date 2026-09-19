"""Pairwise judging: which of two replies to the same request is better,
asked both ways round.

A preference pair built from two pointwise scores (``build_preference_
pairs``) has never had a judge look at the two replies side by side. Lambert
2025, chapter Reward Modeling ("Generative Reward Modeling") and chapter
Preference Data (Chatbot Arena ties) describe the pairwise form: show A and B,
ask for a winner or a tie, and ask again with the positions swapped, because a
judge that prefers whichever reply it read first has position bias, not a
preference. ``judge_pairs`` does that and writes the answer on the pair:
``pairwise`` (winner, whether the two orders agreed, the reasons) and ``tie``.
``export_preference`` drops ties by default; a trainer that learns from ties
can keep them.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from ..defaults import (
    JUDGE_MAX_TOKENS,
    JUDGE_SITUATION_CHARS,
    JUDGE_TEMPERATURE,
    POSITION_FLIP_FLAG,
)
from ..generate.agents import complete, parse_backend_spec
from ..generate.typesafe_backend import is_typesafe_url
from . import decision_judge
from .grade_llm import _render_payload, judge_spec, judge_version

PAIRWISE_SYSTEM = (
    "You compare two replies, A and B, from an AI agent to the same request. "
    "Each reply is a JSON record of what the agent did (tool calls, results, "
    "final message). Decide which reply served the user better: correct tool "
    "use, a truthful final message, no invented facts, no unnecessary steps. "
    "Length must not influence your decision; a shorter reply that does the "
    "job beats a longer one. If they are equally good or equally bad, say tie. "
    'Answer with one JSON object and nothing else: {"winner": "A" | "B" | '
    '"tie", "reason": "<one sentence>"}.'
)
# PAIRWISE_MAX_TOKENS = JUDGE_MAX_TOKENS (120): a winner and one sentence,
# the same reply shape as the pointwise judge.
PAIRWISE_MAX_TOKENS = JUDGE_MAX_TOKENS
# PREFERS_REJECTED_FLAG = 0.2: share of judged pairs where the pairwise
# judge picks the pointwise loser before the report says the two
# disagree on what good is. At one in five the pointwise ranking is no
# better than a coin on those pairs plus a margin (convention, untested).
PREFERS_REJECTED_FLAG = 0.2
# ERROR_CHARS = 200: how much of a judge exception is kept as the reason.
ERROR_CHARS = 200
_WINNER = re.compile(r'"?winner"?\s*:\s*"?(A|B|tie)"?', re.I)

Verdict = dict[str, Any]


def parse_pairwise(text: str) -> tuple[str | None, str]:
    """``("A" | "B" | "tie" | None, reason)`` from a judge reply."""
    raw = str(text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            winner = str(obj.get("winner") or "").strip().lower()
            reason = str(obj.get("reason") or "").strip()
            if winner in ("a", "b"):
                return winner.upper(), reason
            if winner == "tie":
                return "tie", reason
    m = _WINNER.search(raw)
    if m:
        w = m.group(1)
        return (w.upper() if w.lower() != "tie" else "tie"), ""
    return None, ""


def pairwise_judge(
    spec: str | None = None,
    *,
    api_key: str | None = None,
    prompt: str | None = None,
    policy: str = "",
    tools: Sequence | None = None,
    timeout: float = 120,
    max_tokens: int = PAIRWISE_MAX_TOKENS,
    request_chars: int = JUDGE_SITUATION_CHARS,
) -> Callable[[dict, dict], Verdict]:
    """A model judge for ``judge_pairs``: ``judge(a_row, b_row) ->
    {"winner": "A" | "B" | "tie" | None, "reason": str}``. ``spec`` is a
    backend spec (default the hosted judge); ``prompt`` replaces the
    pairwise system prompt. The judge's name is ``<model>@<prompt sha>``
    so a prompt edit is a new judge. ``max_tokens`` is the judge's reply
    budget and ``request_chars`` how much of the request it is shown."""
    resolved = judge_spec(spec=spec)
    url, model = parse_backend_spec(resolved)
    system = str(prompt or "").strip() or PAIRWISE_SYSTEM

    def judge(a: dict, b: dict) -> Verdict:
        request = str(a.get("prompt") or b.get("prompt") or "")[:request_chars]
        a_record = json.loads(_render_payload(a, policy=policy, tools=tools))
        b_record = json.loads(_render_payload(b, policy=policy, tools=tools))
        user = json.dumps({"request": request, "A": a_record, "B": b_record}, default=str)
        try:
            if is_typesafe_url(url):
                # one choice question, A / B / tie, with its distribution
                return decision_judge.pairwise_decision(
                    url,
                    model,
                    system=system,
                    request=request,
                    a=a_record,
                    b=b_record,
                    api_key=api_key,
                    timeout=timeout,
                )
            reply = complete(
                url,
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                api_key=api_key,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            return {"winner": None, "reason": f"{type(exc).__name__}: {exc}"[:ERROR_CHARS]}
        winner, reason = parse_pairwise(str(reply.get("content") or ""))
        return {"winner": winner, "reason": reason}

    judge.__name__ = judge_version(resolved, system)
    return judge


def _verdict(judge: Callable[[dict, dict], Any], a: dict, b: dict) -> Verdict:
    try:
        out = judge(a, b)
    except Exception as exc:
        return {"winner": None, "reason": f"{type(exc).__name__}: {exc}"[:ERROR_CHARS]}
    if isinstance(out, str):
        winner, reason = parse_pairwise(out)
        return {"winner": winner, "reason": reason}
    if isinstance(out, dict):
        winner = out.get("winner")
        if isinstance(winner, str):
            w = winner.strip().lower()
            winner = "tie" if w == "tie" else (w.upper() if w in ("a", "b") else None)
        else:
            winner = None
        return {"winner": winner, "reason": str(out.get("reason") or "")}
    return {"winner": None, "reason": "judge returned neither a dict nor a string"}


def _judge_one(judge: Callable[[dict, dict], Any], pair: dict, swap: bool) -> dict[str, Any]:
    chosen, rejected = pair["chosen"], pair["rejected"]
    first = _verdict(judge, chosen, rejected)  # A = chosen
    as_first = {"A": "chosen", "B": "rejected", "tie": "tie"}.get(first["winner"] or "")
    reasons = [first["reason"]] if first["reason"] else []
    if not swap:
        return {
            "winner": as_first,
            "position_consistent": None,
            "swapped": False,
            "reasons": reasons,
        }
    second = _verdict(judge, rejected, chosen)  # A = rejected
    as_second = {"A": "rejected", "B": "chosen", "tie": "tie"}.get(second["winner"] or "")
    if second["reason"]:
        reasons.append(second["reason"])
    if as_first is None or as_second is None:
        return {
            "winner": as_first or as_second,
            "position_consistent": None,
            "swapped": True,
            "reasons": reasons,
        }
    consistent = as_first == as_second
    # Two orders, two answers: the judge preferred a position, not a
    # reply. That is a tie on the evidence, and the flag says why.
    winner = as_first if consistent else "tie"
    return {
        "winner": winner,
        "position_consistent": consistent,
        "swapped": True,
        "reasons": reasons,
    }


def judge_pairs(
    pairs: Sequence[dict],
    judge: Callable[[dict, dict], Any] | None = None,
    *,
    spec: str | None = None,
    swap: bool = True,
    concurrency: int = 8,
    api_key: str | None = None,
    examples: int = 10,
    position_flip_flag: float = POSITION_FLIP_FLAG,
    prefers_rejected_flag: float = PREFERS_REJECTED_FLAG,
) -> tuple[list[dict], dict[str, Any]]:
    """Ask a judge which side of each pair is better, both ways round.

    ``judge(a_row, b_row)`` returns ``{"winner": "A" | "B" | "tie",
    "reason"}`` (or that JSON as a string); without one the hosted model
    judge from ``pairwise_judge(spec)`` is used. With ``swap=True`` each
    pair is judged twice with A and B exchanged; a pair the judge decides
    differently in the two orders is recorded as a tie with
    ``position_consistent=False``.

    Writes on each pair (in place, and returned): ``pairwise`` with
    ``winner`` (``"chosen"`` | ``"rejected"`` | ``"tie"`` | ``None`` when
    the judge failed), ``position_consistent``, ``reasons``, ``judge``;
    and ``tie`` (bool). Report: ``position_flip_rate`` (position bias:
    the judge's answer changed with the order), ``tie_rate``,
    ``agrees_with_scores`` (the pairwise winner is the pointwise
    ``chosen``), ``prefers_rejected`` (the two disagree outright, the
    rows a person should read), ``failed``. A ``position_flip_rate`` at
    or over ``position_flip_flag`` (``POSITION_FLIP_FLAG``, 0.2: Zheng et
    al. arXiv:2306.05685 measured 35% of GPT-4 verdicts flipping with the
    order) and a prefers-rejected share at or over
    ``prefers_rejected_flag`` each add a warning.
    """
    entries = [p for p in pairs if isinstance(p, dict)]
    bad = [
        i
        for i, p in enumerate(entries)
        if not (isinstance(p.get("chosen"), dict) and isinstance(p.get("rejected"), dict))
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} entries are not pairs (need chosen/rejected rows); first at {bad[0]}"
        )
    if judge is None:
        judge = pairwise_judge(spec, api_key=api_key)
    name = getattr(judge, "__name__", type(judge).__name__)
    if entries and concurrency > 1 and len(entries) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            results = list(pool.map(lambda p: _judge_one(judge, p, swap), entries))
    else:
        results = [_judge_one(judge, p, swap) for p in entries]
    n = len(entries)
    judged = failed = flips = ties = agree = prefers_rejected = swapped = 0
    disagreements: list[dict[str, Any]] = []
    for pair, res in zip(entries, results):
        res["judge"] = name
        pair["pairwise"] = res
        pair["tie"] = res["winner"] == "tie"
        if res["winner"] is None:
            failed += 1
            continue
        judged += 1
        if res["swapped"] and res["position_consistent"] is not None:
            swapped += 1
            if not res["position_consistent"]:
                flips += 1
        if res["winner"] == "tie":
            ties += 1
        elif res["winner"] == "chosen":
            agree += 1
        else:
            prefers_rejected += 1
            if len(disagreements) < examples:
                disagreements.append(
                    {
                        "prompt": str(pair.get("prompt") or pair["chosen"].get("prompt") or "")[
                            :160
                        ],
                        "chosen_score": pair.get("chosen_score"),
                        "rejected_score": pair.get("rejected_score"),
                        "reasons": res["reasons"],
                    }
                )
    report: dict[str, Any] = {
        "n": n,
        "n_judged": judged,
        "failed": failed,
        "judge": name,
        "swap": swap,
        "position_flip_rate": round(flips / swapped, 4) if swapped else None,
        "tie_rate": round(ties / judged, 4) if judged else None,
        "agrees_with_scores": round(agree / judged, 4) if judged else None,
        "prefers_rejected": prefers_rejected,
        "disagreements": disagreements,
        "warnings": [],
    }
    if swapped and flips / swapped >= position_flip_flag:
        report["warnings"].append(
            f"position bias: the judge changed its answer when A and B were swapped in "
            f"{flips}/{swapped} pairs; those pairs are ties, and the judge prompt needs work"
        )
    if judged and prefers_rejected / judged >= prefers_rejected_flag:
        report["warnings"].append(
            f"the pairwise judge prefers the rejected side in {prefers_rejected}/{judged} "
            "pairs; the pointwise scores and the pairwise judge disagree on what good is"
        )
    return entries, report


__all__ = [
    "PAIRWISE_SYSTEM",
    "judge_pairs",
    "pairwise_judge",
    "parse_pairwise",
]
