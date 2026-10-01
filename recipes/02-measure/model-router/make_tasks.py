"""Freeze the workload: 85 MMLU-Pro test questions from each of its 14 subjects.

Run once; the output, tasks.jsonl, is checked in so every run of the recipe
routes the same 1,190 questions. MMLU-Pro (Wang et al. 2024, arXiv:2406.01574)
is MIT-licensed; ten options per question keep a small model well off the
ceiling, which is what makes routing worth measuring.

Run: python make_tasks.py   (needs `huggingface_hub` and `pandas`)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PER_SUBJECT = 85  # 85 x 14 subjects = 1,190 questions: convention, sized for ~$6 of calls
SEED = 0


def main() -> int:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(path)
    rng = random.Random(SEED)
    out = []
    for subject in sorted(df.category.unique()):
        sub = df[df.category == subject]
        for i in sorted(rng.sample(range(len(sub)), PER_SUBJECT)):
            r = sub.iloc[i]
            out.append(
                {
                    "id": f"mmlupro-{r.question_id}",
                    "subject": subject,
                    "question": r.question,
                    "options": [str(o) for o in r.options],
                    "answer": r.answer,
                }
            )
    dest = Path(__file__).with_name("tasks.jsonl")
    dest.write_text("".join(json.dumps(t) + "\n" for t in out), encoding="utf-8")
    print(f"{len(out)} questions -> {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
