"""Verifiable rewards, end to end, offline. No model, no key.

A verifier is a checker, not a judge: it reads a rollout and returns a
reward with no model call (RLHF book chapter Reasoning, 13). Because every verifier
honors the judge contract, it drops into `data.grade(judge=...)`,
`evaluate`, `optimize` and a gated `push` exactly where an LLM judge would.

Run: python recipes/01-simulate/verifiers/run.py
"""

from __future__ import annotations

import argparse

import whileai.simulations as wai
from whileai.simulations.score.judging import run_judge
from whileai.simulations.verify import (
    All,
    CodeExec,
    JSONSchema,
    MathEqual,
    Regex,
)

# Rows as they come off a rollout: `final_text` is what the policy said, and
# the gold lives in `privileged` (never exported into a training row).
MATH_ROWS = [
    {
        "prompt": "2+2?",
        "final_text": "<think>2+2</think>\\boxed{4}",
        "privileged": {"reference": "4"},
        "scenario_id": "m1",
        "rollout_index": 0,
    },
    {
        "prompt": "2+2?",
        "final_text": "it's 5",
        "privileged": {"reference": "4"},
        "scenario_id": "m1",
        "rollout_index": 1,
    },
    {
        "prompt": "half of 1?",
        "final_text": "\\boxed{1/2}",
        "privileged": {"reference": "0.5"},
        "scenario_id": "m2",
        "rollout_index": 0,
    },
]

CODE_ROWS = [
    {
        "prompt": "add(a,b)",
        "final_text": "```python\ndef add(a, b):\n    return a + b\n```",
        "privileged": {"tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"},
        "scenario_id": "c1",
        "rollout_index": 0,
    },
    {
        "prompt": "add(a,b)",
        "final_text": "```python\ndef add(a, b):\n    return a - b\n```",
        "privileged": {"tests": "assert add(2, 3) == 5\n"},
        "scenario_id": "c1",
        "rollout_index": 1,
    },
]

INTENT_SCHEMA = {
    "type": "object",
    "required": ["intent"],
    "properties": {"intent": {"type": "string"}},
}
JSON_ROWS = [
    {
        "prompt": "classify",
        "final_text": '{"intent": "refund"}',
        "scenario_id": "j1",
        "rollout_index": 0,
    },
    {
        "prompt": "classify",
        "final_text": "sorry, no json here",
        "scenario_id": "j1",
        "rollout_index": 1,
    },
]


def show(title, rows, verifier):
    print(f"\n== {title}: {verifier.name}")
    scored = run_judge(rows, verifier, source="grade")
    for r in scored.rows:
        print(f"  reward={r.get('reward')}  {r.get('reason', '')[:70]}")
    passed = sum(1 for r in scored.rows if r.get("reward") == 1)
    print(f"  {passed}/{len(scored.rows)} passed")
    return scored


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        description="Verifiable rewards, offline demo. No arguments."
    ).parse_args(argv)

    # 1. Math: symbolic equality with \boxed / fraction handling.
    math_scored = show("math", MATH_ROWS, MathEqual())

    # 2. Math AND format: the answer is right AND it closed its reasoning.
    show("math + format", MATH_ROWS, All([MathEqual(), Regex(r"</think>")], name="answer+format"))

    # 3. Code: run the candidate against hidden tests in a sandbox.
    show("code execution", CODE_ROWS, CodeExec())

    # 4. Structured output: valid JSON against a schema.
    show("json schema", JSON_ROWS, JSONSchema(INTENT_SCHEMA))

    # 5. Into the loop: the verifier's reward is the GRPO reward. `optimize`
    # runs the RL gates over the scored rows (reward band, unanimous groups,
    # duplicates). The answer key stays on the rows it returns, which are
    # still SDK rows; the training export is what never projects `privileged`.
    rows, report = wai.optimize(math_scored, mode="rl")
    trainer_rows = wai.training_rows(rows)
    leaked = sum(1 for r in trainer_rows if "privileged" in r)
    print(
        f"\n== optimize(mode='rl'): {len(rows)} rows from {report['groups_selected']} "
        f"prompt groups; training_rows() -> {len(trainer_rows)} rows, "
        f"{leaked} carry the answer key"
    )
    print("Push the optimized rows with wai.push_rows(rows, 'math-rl-v1', gate=True, mode='rl').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
