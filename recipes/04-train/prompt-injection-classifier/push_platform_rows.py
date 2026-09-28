"""Push the round-9 rows and the five frozen tests to the While account's Datasets page.

    python push_platform_rows.py

Rows ride as ``prompt`` (the chunk), ``reward`` (the label), ``label_source="program"``,
with the slice, family and carrier as fields. No gate: these are classifier rows, not
rollouts, and the gate's RL checks do not apply.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from data import read_jsonl

from whileai.config import provenance
from whileai.simulations.ingest.platform import push_rows

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def shape(rows: list[dict]) -> list[dict]:
    return [
        {
            "prompt": r["text"],
            "reward": float(r["label"]),
            "label_source": "program",
            "final_text": "",
            "slice": r.get("slice"),
            "family": r.get("family"),
            "carrier": r.get("carrier"),
            "source": r.get("source"),
            "pair": r.get("pair"),
        }
        for r in rows
    ]


def main() -> None:
    print(provenance(), file=sys.stderr)
    out = {}
    entry = push_rows(
        shape(read_jsonl(OUT / "train_v9.jsonl")),
        "prompt-injection-carriers-v9",
        purpose="train",
        agent="prompt-injection-classifier",
        description="round 9 training rows: matched twins on template and model-written carriers, seeded-world rows; label by program",
    )
    out["train_v9"] = entry.get("datasetId")
    for t in ("test", "test_hard", "test_paste", "test_llm", "test_external"):
        entry = push_rows(
            shape(read_jsonl(HERE / f"{t}.jsonl")),
            f"prompt-injection-{t.replace('_', '-')}",
            purpose="holdout",
            agent="prompt-injection-classifier",
            description=f"frozen {t}, sha256 {(HERE / f'{t}.sha256').read_text().strip()[:16]}",
        )
        out[t] = entry.get("datasetId")
    (OUT / "platform_datasets.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
