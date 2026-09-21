"""Preference pairs for DPO, from the SDK's graded rows or its exported JSONL.

DPO needs a chosen and a rejected reply to the same prompt. Two supplies:

* **On-policy (default here)**: sample the base policy a few times per
  prompt, score every reply with the rule in ``reward.py``, and let
  ``wai.build_preference_pairs`` pair a pass with a fail of similar length.
  Both sides come from the policy being trained, which is where DPO works
  best (Lambert 2025, chapter Preference Data).
* **Exported**: a file written by ``wai.export_preference`` (chosen and
  rejected as full message lists) from any graded dataset, for example one
  pulled from the platform. Pass it as ``--pairs``.

Either way the trainer sees TRL's conversational shape: ``prompt`` is the
system and user turns, ``chosen`` and ``rejected`` are one assistant turn
each. Pure Python, no model, so it is checkable offline.
"""

from __future__ import annotations

import json
from typing import Any


def _render_assistant(message: dict) -> str:
    """An assistant message as the text the policy would emit: its content
    plus one ``<tool_call>`` block per structured tool call."""
    text = str(message.get("content") or "").strip()
    blocks: list[str] = []
    for call in message.get("tool_calls") or []:
        call = call if isinstance(call, dict) else {}
        fn = call["function"] if isinstance(call.get("function"), dict) else call
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        blocks.append(
            "<tool_call>\n"
            + json.dumps({"name": fn.get("name"), "arguments": args if args is not None else {}})
            + "\n</tool_call>"
        )
    return "\n".join([t for t in [text, *blocks] if t])


def first_turn(row: dict) -> str:
    """The text of a rollout's first assistant turn.

    A tool step first becomes a ``<tool_call>`` block; otherwise the first
    assistant message, or ``final_text`` for single-turn rows."""
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            args = step.get("arguments")
            if args is None:
                args = step.get("input")
            return (
                "<tool_call>\n"
                + json.dumps(
                    {"name": str(step["tool"]), "arguments": args if isinstance(args, dict) else {}}
                )
                + "\n</tool_call>"
            )
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            rendered = _render_assistant(message)
            if rendered:
                return rendered
    return str(row.get("final_text") or "").strip()


def dpo_rows(pairs: list[dict], system: str) -> list[dict[str, Any]]:
    """TRL conversational rows from ``build_preference_pairs`` output. A pair
    whose two first turns read the same is dropped: the contrast was later
    in the rollout and a first-turn trainer cannot learn it."""
    out: list[dict[str, Any]] = []
    for pair in pairs:
        prompt = str(pair.get("prompt") or "").strip()
        chosen = first_turn(pair.get("chosen") or {})
        rejected = first_turn(pair.get("rejected") or {})
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        out.append(
            {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "margin": pair.get("margin"),
            }
        )
    return out


def invented_call(prompt: str, tool: str = "lookup_order") -> str:
    """The rejected reply a no-id prompt never gets from the base policy: a
    well-formed call to ``tool`` with an order id that appears nowhere in
    the prompt, derived from the prompt so the pair is reproducible."""
    import hashlib

    digits = int(hashlib.sha256(prompt.encode()).hexdigest()[:6], 16) % 9000 + 1000
    fake = f"ORD-{digits}"
    if fake.lower() in prompt.lower():
        fake = f"ORD-{(digits + 1) % 9000 + 1000}"
    return (
        "<tool_call>\n"
        + json.dumps({"name": tool, "arguments": {"order_id": fake}})
        + "\n</tool_call>"
    )


def constructed_negatives(
    prompts: list[dict[str, Any]],
    replies: list[list[str]],
    *,
    system: str,
    max_pairs: int | None = None,
) -> list[dict[str, Any]]:
    """One pair per no-id or off-topic prompt the policy answered without a
    tool call: that reply as chosen, an invented call as rejected.

    Why: DPO learns only from prompts with a pass and a fail, and the base
    policy almost never invents an id on a no-id prompt, so those prompts
    never pair and the with-id pairs teach "call the tool" across every
    kind of prompt; a second round then makes it worse. The contrast has
    to be put on the no-id prompts themselves. The chosen side is the
    policy's own reply (on-policy), the rejected side is the one mistake
    the reward names."""
    from reward import parse_tool_call, score

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item, group in zip(prompts, replies):
        case = item["case"]
        if case.get("order_id"):
            continue
        # One pair per distinct prompt: --balance repeats prompts, and the
        # constructed side must not scale with the repeats. Uncapped, 411
        # constructed pairs against 351 sampled ones taught "never call"
        # (with-id pass@1 0.11 to 0.05); capped, they are a correction.
        key = " ".join(item["prompt"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        best = None
        for reply in group:
            passes = (
                parse_tool_call(reply) is None
                and "<tool_call>" not in reply
                and score(reply, case) >= 1.0
            )
            if passes and (best is None or len(reply) < len(best)):
                best = reply
        if best is None:
            continue
        out.append(
            {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": item["prompt"]},
                ],
                "chosen": [{"role": "assistant", "content": best.strip()}],
                "rejected": [{"role": "assistant", "content": invented_call(item["prompt"])}],
                "margin": 1.0,
                "constructed": True,
            }
        )
    if max_pairs is not None and len(out) > max_pairs:
        # A stable subset: keep every k-th by prompt order.
        step = len(out) / max_pairs
        out = [out[int(i * step)] for i in range(max_pairs)]
    return out


def sampled_pairs(
    prompts: list[dict[str, Any]],
    replies: list[list[str]],
    *,
    system: str,
    min_margin: float = 0.5,
    max_pairs_per_prompt: int = 2,
    constructed: bool = False,
    constructed_share: float = 0.3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """On-policy pairs: score the sampled replies with the rule, pair passes
    with fails per prompt (length-matched), return TRL rows and the pair
    report (how often chosen is the longer side, mean margin). With
    ``constructed=True`` every no-id or off-topic prompt the policy answered
    without a tool call also pairs that reply against an invented call."""
    from reward import reward_rows

    import whileai.simulations as wai

    rows = reward_rows(prompts, replies)
    for row in rows:
        # build_preference_pairs reads ``reward``; the raw rule score is the
        # finer signal, so a 1.0 vs 0.5 pair exists at min_margin 0.5.
        row["reward"] = float(row["markers"]["tool_rule"])
    pairs, report = wai.build_preference_pairs(
        rows, min_margin=min_margin, max_pairs_per_prompt=max_pairs_per_prompt, length_match=True
    )
    out = dpo_rows(pairs, system)
    report = dict(report)
    if constructed:
        # At most constructed_share of the on-policy pair count, so the
        # constructed side corrects the update instead of replacing it.
        cap = max(1, int(len(out) * constructed_share))
        extra = constructed_negatives(prompts, replies, system=system, max_pairs=cap)
        out.extend(extra)
        report["constructed_pairs"] = len(extra)
        report["constructed_cap"] = cap
    report["trl_rows"] = len(out)
    return out, report


def load_export(path: str, *, system: str | None = None) -> list[dict[str, Any]]:
    """Rows from a ``wai.export_preference`` JSONL. Each line carries
    ``chosen`` and ``rejected`` as full conversations; the prompt is every
    message before the first assistant turn, the sides are that turn."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sides: dict[str, str] = {}
            prompt_msgs: list[dict] | None = None
            for side in ("chosen", "rejected"):
                msgs = [m for m in entry.get(side) or [] if isinstance(m, dict)]
                idx = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), None)
                if idx is None:
                    break
                head = [
                    {"role": m["role"], "content": str(m.get("content") or "")}
                    for m in msgs[:idx]
                    if m.get("role") in ("system", "user")
                ]
                if system is not None:
                    head = [
                        {"role": "system", "content": system},
                        *[m for m in head if m["role"] != "system"],
                    ]
                if prompt_msgs is None:
                    prompt_msgs = head
                sides[side] = _render_assistant(msgs[idx])
            if len(sides) < 2 or not prompt_msgs or not sides["chosen"] or not sides["rejected"]:
                continue
            if sides["chosen"] == sides["rejected"]:
                continue
            out.append(
                {
                    "prompt": prompt_msgs,
                    "chosen": [{"role": "assistant", "content": sides["chosen"]}],
                    "rejected": [{"role": "assistant", "content": sides["rejected"]}],
                    "margin": entry.get("margin"),
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    """The offline path: prompts from the template writer, replies from a
    scripted policy that follows the rule half the time, and the pairs the
    trainer would see, with the pair report.

        python pairs.py --n 40       # what smoke.sh runs; no key, no GPU
    """
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grpo"))
    from reward import SYSTEM, build_prompts, scripted_reply, split_holdout

    from whileai.config import provenance

    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--n", type=int, default=40, help="prompts from the template writer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    train, held = split_holdout(build_prompts(args.n, seed=args.seed))
    print(f"{len(train)} train prompts, {len(held)} holdout")
    # Two passes and two fails per prompt: every prompt has contrast.
    replies = [
        [scripted_reply(p["case"]), scripted_reply(p["case"], follow=False)] * 2 for p in train
    ]
    rows, report = sampled_pairs(train, replies, system=SYSTEM, constructed=True)
    print(
        f"pairs {report['trl_rows']} from {report['prompts_with_contrast']}/{report['prompts_seen']} "
        f"prompts with contrast, {report['constructed_pairs']} constructed; "
        f"chosen longer {report['length']['chosen_longer_frac']:.2f}"
    )
    first = rows[0]
    print(f"one pair: user={first['prompt'][-1]['content']!r}")
    print(f"  chosen={first['chosen'][0]['content']!r}")
    print(f"  rejected={first['rejected'][0]['content']!r}")
    return 0


__all__ = [
    "constructed_negatives",
    "dpo_rows",
    "first_turn",
    "invented_call",
    "load_export",
    "sampled_pairs",
]


if __name__ == "__main__":
    raise SystemExit(main())
