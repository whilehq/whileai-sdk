"""Ask ``wai.methods.route`` what it would do with the baseline's graded predictions.

    python route_probe.py --scores out/scores_protectai.json --model protectai/deberta-v3-base-prompt-injection-v2

Each test row is a task, the reward is 1 when the baseline classified it
correctly at its shipped threshold, k=1 (one deterministic rollout per task).
``route`` was written for rollouts of a decoder LM; the point of running it
is to record where its checks fit a classifier and where they do not
(``sdk_findings.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data import read_jsonl

import whileai as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default=str(HERE / "test.jsonl"))
    ap.add_argument("--out", default=str(HERE / "out/route.json"))
    a = ap.parse_args()
    rows = read_jsonl(Path(a.test))
    sc = json.loads(Path(a.scores).read_text())
    thr = sc["threshold"]
    graded = []
    for i, (r, s) in enumerate(zip(rows, sc["scores"])):
        pred = int(s > thr)
        graded.append(
            {
                "task_key": f"row-{i}",
                "scenario_id": f"row-{i}",
                "prompt": r["text"],
                "reward": float(pred == r["label"]),
                "model_version": a.model,
                "label_source": "program",
                "finish_reason": "stop",
                "rollout_index": 0,
            }
        )
    route = wai.methods.route(graded, model=a.model, size_b=0.184)
    text = str(route)
    print(text)
    record = {
        "model": a.model,
        "n_tasks": len(graded),
        "pass_rate": sum(g["reward"] for g in graded) / len(graded),
        "route": text,
    }
    for key in ("method", "pick", "reason", "need", "scores"):
        val = getattr(route, key, None)
        if val is not None:
            record[key] = val if isinstance(val, (str, int, float, list, dict)) else str(val)
    Path(a.out).write_text(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
