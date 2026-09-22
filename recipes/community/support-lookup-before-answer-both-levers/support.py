"""One entry point: the harnesses, the render every stage uses, and the grader.

A harness here is the program around the weights: the system text the agent
serves under, whether the tools are in front of it, and the turn cap. The
deployed harness is the published policy as the traffic was served under it.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------- harnesses

SKILLS = """
# Before you answer

- If the customer's request is about *their* account, order, reservation, bill,
  line or plan, you do not know the answer. Look it up. Call a tool first and
  answer from what it returns.
- Identify the customer before anything else. An email, a phone number, a
  customer id, a reservation code or a full name with date of birth is enough
  to call a lookup tool. If you already have one of those, you have what you
  need: call the tool now rather than asking for it again.
- Read the tool list before deciding you cannot help. The tool whose name
  matches the noun the customer used is usually the right one.
- Do not ask a second clarifying question when a tool would answer the first.

# When not to call a tool

- If the request is outside this policy's scope, or asks for another
  customer's data, or asks you to confirm a policy the tools do not carry,
  say so plainly and do not call a tool.
"""

HARNESSES: dict[str, dict[str, Any]] = {
    # The deployed harness: the policy exactly as the published traffic ran
    # under it. This is the base every arm is measured against.
    "00_deployed": {"skills": "", "tools_in_prompt": True},
    # Candidate 01: the same policy plus a skills text saying when to look up.
    # No tool change, no turn change -- a prompt edit with a fingerprint.
    "01_skills": {"skills": SKILLS, "tools_in_prompt": True},
    # Candidate 02: the skills text plus a one-line reminder of the tool names
    # directly above the customer's turn, where the model is looking.
    "02_skills_toolnames": {"skills": SKILLS, "tools_in_prompt": True, "name_reminder": True},
}


def system_text(task: dict, harness: str) -> str:
    cfg = HARNESSES[harness]
    text = task["system"]
    if cfg["skills"]:
        text = text + "\n" + cfg["skills"]
    return text


def render(task: dict, harness: str, tokenizer) -> str:
    """The one render. Search, training and eval all call this."""
    cfg = HARNESSES[harness]
    msgs = [{"role": "system", "content": system_text(task, harness)}]
    for m in task["prefix"]:
        msgs.append({"role": m["role"], "content": m.get("content") or ""})
    if cfg.get("name_reminder"):
        names = ", ".join(t["function"]["name"] for t in task["tools"])
        msgs[-1] = dict(msgs[-1])
        msgs[-1]["content"] = (msgs[-1]["content"] or "") + f"\n\n(tools available: {names})"
    return tokenizer.apply_chat_template(
        msgs,
        tools=task["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def target_completion(task: dict) -> str:
    """What the reference agent did at this decision point, in the model's own
    output format. Training targets and the grader read the same shape."""
    if task["target"] == "CALL":
        call = {"name": task["tool_name"], "arguments": task["tool_args"] or {}}
        return "<tool_call>\n" + json.dumps(call) + "\n</tool_call>"
    return task.get("target_text") or "I'm not able to help with that request."


# ------------------------------------------------------------------ grading

_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_call(text: str) -> dict | None:
    m = _CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return {"name": None, "arguments": {}, "malformed": True}
    if not isinstance(obj, dict):
        return {"name": None, "arguments": {}, "malformed": True}
    return {
        "name": obj.get("name"),
        "arguments": obj.get("arguments") or {},
        "malformed": False,
    }


def grade(task: dict, text: str) -> dict:
    """Programmatic. No model sits in the reward path, in training or in eval.

    The behaviour and its capability twin are graded by the same rule, so
    neither can be bought by the degenerate policy that always calls a tool
    (or never does).
    """
    call = parse_call(text)
    called = call is not None
    wants_call = task["target"] == "CALL"

    right_tool = bool(called and call["name"] == task["tool_name"]) if wants_call else False
    if wants_call:
        want = task["tool_args"] or {}
        got = call["arguments"] if called else {}
        args_match = bool(
            right_tool
            and isinstance(got, dict)
            and all(
                str(got.get(k, "")).strip().lower() == str(v).strip().lower()
                for k, v in want.items()
            )
        )
        correct = right_tool
    else:
        args_match = False
        correct = not called

    return {
        "reward": 1 if correct else 0,
        "markers": {
            "right_first_action": 1.0 if correct else 0.0,
            "looks_up_when_it_should": (1.0 if right_tool else 0.0) if wants_call else None,
            "stays_in_scope": (1.0 if not called else 0.0) if not wants_call else None,
            "called_a_tool": 1.0 if called else 0.0,
            "args_match": 1.0 if args_match else 0.0,
            "malformed_call": 1.0 if (called and call.get("malformed")) else 0.0,
        },
    }
