"""The reader's side, shared by every page that has one.

The guides are written in the second person: they say `my_tools`, `POLICY`,
`my_agent` and leave them to the reader, because spelling a full agent out in
every block would bury the call each one is there to teach. This file is that
reader's side, written once: one support agent with two tools, one math agent,
a judge, a verifier, graded rows, and the files the pages read from disk.

A page fixture is then usually one line, `from _common import *`, and the
page's own blocks run for real against the installed package, so a renamed
argument or a changed signature still fails the check.

Everything here is offline. Nothing in this file needs a key.
"""

import json
from pathlib import Path

import whileai.simulations as wai
from whileai.simulations.verify import verifier

# ---------------------------------------------------------------- the agent

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id.",
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
            "name": "issue_refund",
            "description": "Refund an order in full.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]

POLICY = """You handle refunds for an online shoe store.
Look the order up before you say anything about it.
Refund an order that is 30 days old or less.
Refunds over $200 need a manager.
Tell the customer what you did."""

RUBRIC = """Doing the job means: the order was looked up before the reply
spoke about it, a refund was issued only when the policy allows one, and the
reply says what was done."""

SEEDS = [
    "I want a refund for order A1001, the shoes did not fit.",
    "What is the status of order A1002?",
    "Refund A1003 please, it arrived damaged.",
]

# The page's other spellings for the same two things.
my_tools = TOOLS
my_system_prompt = POLICY
SYSTEM_PROMPT = POLICY

MATH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]
MATH_POLICY = "Answer the arithmetic question. Put the final number on its own last line."


def my_agent(message: str) -> dict:
    """The callable contract: in with the situation, out with what it did."""
    return {
        "steps": [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": "4412"},
                "result": {"status": "shipped", "total": 89.0, "age_days": 4},
            }
        ],
        "final_text": "Order 4412 shipped yesterday, so I refunded it in full.",
    }


agent = my_agent


def my_judge(row: dict) -> dict:
    looked_up = any(s.get("tool") == "lookup_order" for s in row.get("steps") or [])
    return {
        "reward": float(looked_up and bool(row.get("final_text"))),
        "markers": {"looked_up_first": float(looked_up)},
    }


judge = my_judge


@verifier
def my_verifier(row: dict) -> float:
    """A reward that is a program, importable by name for export_environment."""
    return float(any(s.get("tool") == "lookup_order" for s in row.get("steps") or []))


# ------------------------------------------------------- rows the page uses

# Explore mode, which is what the page's `data` is unless a block says
# otherwise, and graded, because most of the page is downstream of grading.
data = wai.simulate(
    my_agent,
    tools=TOOLS,
    system_prompt=POLICY,
    simulator=False,
    budget=16,  # unique prompts: the publish gate treats a repeated prompt as RL data
    reproducible=True,
)
scored = data.grade(judge=my_judge)
rollouts = data.trajectories

# `rows` is what the export blocks write. The graded rows are enough: the
# agent above is honest, and `select` drops a reply that quotes its own
# privileged context (#471), so nothing here filters by hand.
rows = scored.rows

# A before/after pair and a set of preference pairs, which later blocks compare
# and export without building them first.
before = rows
after = rows
# Pairs need a chosen and a rejected side, so they come from the seeded agent,
# which fails on a labeled fraction of rollouts on purpose.
_mixed = wai.simulate(
    wai.seeded_agent(TOOLS),
    tools=TOOLS,
    system_prompt=POLICY,
    simulator=False,
    mode="rl",
    repeats=4,
    repeat_policy="fixed",
    budget=64,
    reproducible=True,
)
_mixed_scored = _mixed.grade(judge=lambda row: {"reward": int(not row["seeded"])})
pairs, _pair_report = wai.build_preference_pairs(_mixed_scored.rows)

# Files the page opens by name. Written into the scratch directory the check
# runs in, so the page's `open(...)` and `path=` arguments resolve.
Path("production.jsonl").write_text(
    "\n".join(
        json.dumps({"prompt": r.get("prompt"), "final_text": r.get("final_text")}) for r in rows[:8]
    )
    + "\n"
)
Path("labels.jsonl").write_text(
    "\n".join(
        json.dumps(
            {
                "scenario_id": r["scenario_id"],
                "rollout_index": r["rollout_index"],
                "label": int(bool(r.get("reward"))),
            }
        )
        for r in rows[:8]
    )
    + "\n"
)
labels = "labels.jsonl"

# The page loads the character example's constitution by its repo path. Put the
# real file where that path resolves, so the block reads the same document a
# reader would.
_const = Path(__file__).resolve().parents[2] / "recipes/03-select/character/constitution.json"
_dest = Path("recipes/03-select/character")
_dest.mkdir(parents=True, exist_ok=True)
(_dest / "constitution.json").write_text(_const.read_text(encoding="utf-8"), encoding="utf-8")


# A served model's endpoint and a pinned task set, for the before/after section.
endpoint = "http://127.0.0.1:8000/v1"
name = "my-tuned-v1"
pinned = data

# Graded traces of a deployed agent, which `traces=` reads from disk.
Path("traces.jsonl").write_text(
    "\n".join(
        json.dumps({"prompt": r.get("prompt"), "final_text": r.get("final_text"), "reward": 0})
        for r in rows[:8]
    )
    + "\n"
)


# `from _common import *` should hand over everything above, including the
# names that start a page mid-thought (`data`, `scored`, `rows`, `wai`).
__all__ = [n for n in dir() if not n.startswith("_")]
