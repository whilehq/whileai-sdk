"""The environment for the GRPO example: one agent, one rule, a verifiable reward.

The agent is the refund assistant from the pass-at-k example. Its policy has
one testable rule: look an order up before you refund it, and never invent
an order id. That rule is the reward.

Each prompt is the first user turn. The policy answers once, and the reward
reads that single turn:

* the prompt names an order (``ORD-1234``): the right move is a
  ``lookup_order`` call with exactly that id. Refunding first, inventing an
  id, or asking for an id that is already there scores lower.
* the prompt is about an order but names none: the right move is to ask for
  the id, with no tool call. Any tool call here runs on an invented id.
* the prompt is off topic: a short reply and no tool call.

A well-formed ``<tool_call>`` block earns a small format bonus, so the policy
learns the wire format before it learns the rule. Everything here runs
without a model, so the reward is unit-tested and the prompts are built
offline by the simulator's template writer.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch an order by id before doing anything to it.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund",
            "description": "Refund an amount on an order that was looked up first.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]
POLICY = (
    "You are a refund assistant. Look up an order before refunding it. If the "
    "customer gives no order id, ask for it and call no tool. Never invent an "
    "order id. If the request is not about an order or a refund, say so briefly."
)
SYSTEM = (
    POLICY
    + "\n\nTools:\n"
    + json.dumps([t["function"] for t in TOOLS], indent=0)
    + "\n\nTo call a tool, reply with exactly one block:\n"
    '<tool_call>\n{"name": "<tool name>", "arguments": {...}}\n</tool_call>\n'
    "Otherwise reply in plain text."
)

_ORDER_ID = re.compile(r"\b(ORD[-_ ]?\d{3,6})\b", re.I)
_DOMAIN = re.compile(r"\b(order|refund|return|charge|purchase|payment|shipment|delivery)\b", re.I)
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_QUESTION = re.compile(r"\?|order (id|number)|reference", re.I)
FORMAT_BONUS = 0.2


def case_for(prompt: str) -> dict[str, Any]:
    """What the prompt asks for, read off its text: the order id it names
    (normalized), whether it is about orders at all."""
    m = _ORDER_ID.search(prompt or "")
    order_id = re.sub(r"[-_ ]", "-", m.group(1).upper()) if m else None
    if order_id and not order_id.startswith("ORD-"):
        order_id = "ORD-" + order_id[3:]
    return {
        "order_id": order_id,
        "in_domain": bool(order_id) or bool(_DOMAIN.search(prompt or "")),
    }


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """The single tool call in a reply, or None. Malformed JSON is None too."""
    m = _TOOL_CALL.search(text or "")
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    name = payload.get("name") or payload.get("tool")
    args = payload.get("arguments") or payload.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not name:
        return None
    return {"name": str(name), "arguments": args if isinstance(args, dict) else {}}


def _norm_id(value: Any) -> str | None:
    if value is None:
        return None
    m = _ORDER_ID.search(str(value))
    if not m:
        return None
    out = re.sub(r"[-_ ]", "-", m.group(1).upper())
    return out if out.startswith("ORD-") else "ORD-" + out[3:]


def score(reply: str, case: dict[str, Any]) -> float:
    """Reward in [0, 1]: the rule, plus a small format bonus, capped at 1."""
    text = reply or ""
    if not text.strip():
        return 0.0
    call = parse_tool_call(text)
    mentions_call = "<tool_call>" in text
    well_formed = call is not None or not mentions_call
    base = 0.0
    if case.get("order_id"):
        if call and call["name"] == "lookup_order":
            base = 1.0 if _norm_id(call["arguments"].get("order_id")) == case["order_id"] else 0.1
        elif (call and call["name"] == "create_refund") or call:
            base = 0.0
        elif _QUESTION.search(text):
            base = 0.3  # asked for what was already given
        else:
            base = 0.0
    elif case.get("in_domain"):
        if call:
            base = 0.0  # any call here runs on an invented id
        elif _QUESTION.search(text):
            base = 1.0
        else:
            base = 0.2
    else:
        base = 0.0 if call else (1.0 if 0 < len(text.strip()) <= 400 else 0.3)
    bonus = (
        FORMAT_BONUS if (well_formed and (call is not None or not case.get("order_id"))) else 0.0
    )
    return round(min(1.0, base + bonus), 3)


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def build_prompts(n: int = 200, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` first-turn prompts from the simulator's offline template writer,
    each with its ``case``. No model, no key.

    The same ``seed`` gives the same list, in the same order, on every
    call: ``reproducible=True`` makes ``simulate`` pick each batch of
    situations only after the last batch has landed, so which rows fall
    under ``budget`` no longer depends on thread timing at
    ``concurrency=4`` (whilehq/whileai-sdk#450: two runs at ``--seed 0``
    got 112 and 119 prompts and two different holdouts)."""
    import whileai.simulations as wai

    def silent_agent(message: str) -> dict:
        return {"steps": [], "final_text": "ok."}

    data = wai.simulate(
        silent_agent,
        tools=TOOLS,
        policy=POLICY,
        mode="explore",
        situations=n,
        budget=n,
        seed=seed,
        simulator=False,
        time_budget=None,
        concurrency=4,
        reproducible=True,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in data.rows():
        prompt = str(row.get("prompt") or "").strip()
        key = " ".join(prompt.lower().split())
        if not prompt or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "prompt": prompt,
                "case": case_for(prompt),
                "scenario_id": row.get("scenario_id") or key,
            }
        )
    return out


def split_holdout(
    items: list[dict[str, Any]], fraction: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """Deterministic split by scenario id, so a prompt is wholly train or holdout."""
    train, held = [], []
    for item in items:
        bucket = (
            int(hashlib.sha256(str(item["scenario_id"]).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        )
        (held if bucket < fraction else train).append(item)
    return train, held


def reward_rows(prompts: list[dict[str, Any]], replies: list[list[str]]) -> list[dict[str, Any]]:
    """Graded rows (one per sampled reply) in the SDK's row shape, so
    ``pass_at`` and ``delta_report`` read them: reward 1 when the reply
    satisfies the rule, the raw score as a marker."""
    rows: list[dict[str, Any]] = []
    for item, group in zip(prompts, replies):
        for i, reply in enumerate(group):
            raw = score(reply, item["case"])
            rows.append(
                {
                    "prompt": item["prompt"],
                    "scenario_id": item["scenario_id"],
                    "rollout_index": i,
                    "final_text": reply,
                    "steps": [],
                    "messages": [
                        *messages_for(item["prompt"]),
                        {"role": "assistant", "content": reply},
                    ],
                    "reward": 1 if raw >= 1.0 else 0,
                    "markers": {
                        "tool_rule": raw,
                        "well_formed": 1.0
                        if (parse_tool_call(reply) or "<tool_call>" not in reply)
                        else 0.0,
                    },
                }
            )
    return rows


def scripted_reply(case: dict[str, Any], follow: bool = True) -> str:
    """A reply a policy that follows the rule gives (``follow=True``), or the
    one mistake per case a policy that refunds first gives. What the smoke
    run samples in place of a model."""
    if not follow:
        return (
            "<tool_call>\n"
            + json.dumps(
                {"name": "create_refund", "arguments": {"order_id": "ORD-1", "amount": 20}}
            )
            + "\n</tool_call>"
        )
    if case.get("order_id"):
        return (
            "<tool_call>\n"
            + json.dumps({"name": "lookup_order", "arguments": {"order_id": case["order_id"]}})
            + "\n</tool_call>"
        )
    if case.get("in_domain"):
        return "Of course. What is the order id?"
    return "I can only help with orders and refunds."


def main(argv: list[str] | None = None) -> int:
    """The offline path: the prompt set, the split and the reward on two
    scripted policies, so the environment is checked before a GPU is paid for.

        python reward.py --n 40      # what smoke.sh runs; no key, no GPU
    """
    import argparse
    import sys

    import whileai as wai
    from whileai.config import provenance

    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--n", type=int, default=40, help="prompts from the template writer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    prompts = build_prompts(args.n, seed=args.seed)
    train, held = split_holdout(prompts)
    with_id = sum(1 for p in prompts if p["case"]["order_id"])
    print(
        f"{len(prompts)} prompts from the template writer ({with_id} naming an order id): "
        f"{len(train)} train, {len(held)} holdout"
    )
    before = reward_rows(held, [[scripted_reply(p["case"], follow=False)] for p in held])
    after = reward_rows(held, [[scripted_reply(p["case"])] for p in held])
    print(f"refund-first policy: {wai.pass_at(before)}")
    print(f"rule-following policy: {wai.pass_at(after)}")
    print(wai.compare(before, after, target="pass_at_1", must_not_regress=["well_formed"]))
    return 0


__all__ = [
    "FORMAT_BONUS",
    "POLICY",
    "SYSTEM",
    "TOOLS",
    "build_prompts",
    "case_for",
    "messages_for",
    "parse_tool_call",
    "reward_rows",
    "score",
    "scripted_reply",
    "split_holdout",
]


if __name__ == "__main__":
    raise SystemExit(main())
