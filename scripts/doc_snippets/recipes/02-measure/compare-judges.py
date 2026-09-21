"""The reader's side of recipes/02-measure/compare-judges.

The page's block attaches `labels` to `scored.rows` and compares judges
against them, one of which is `my_verifier`. The rows are the recipe's own
300 labeled rollouts (rows/labeled.jsonl, read from the copy of the recipe
directory the check runs in); the labels are the answer key they carry,
keyed by rollout id; the verifier is the rule that wrote that key, read
back off the row. `scored` is those rows graded once by that rule.
"""

import json
from pathlib import Path

import whileai.simulations as _wai

_rows = [
    json.loads(line)
    for line in Path("rows/labeled.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
labels = {r["rollout_id"]: int(r["gold_reward"]) for r in _rows}


def my_verifier(row: dict) -> dict:
    return {"reward": labels[row["rollout_id"]], "reason": row.get("rule_reason", "")}


scored = _wai.evaluate(_rows, my_verifier, model="qwen3-4b")
