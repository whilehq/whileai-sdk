"""Hard-negative (and hard-positive) mining: weight the rows the last round got wrong, with a random control.

    python mine.py --model out/v5-twin-seed1 --train out/train_v5.jsonl --tag v6

Scores every training row with the last round's model. A row whose probability
sits on the wrong side of 0.5, or within 0.3 of it, gets ``weight`` 3.0; the
rest 1.0 (the classifier analogue of the 20 to 80% difficulty band, Lambert
2025, chapter Reasoning; rejection sampling's "keep what the model finds
hard", chapter Rejection Sampling). The control assigns the same number of
3.0 weights to rows drawn at random (the random-selection control the book's
rejection-sampling chapter asks for), so a gain from mining has to beat a gain
from merely up-weighting the same mass of rows. Writes
``out/train_<tag>.jsonl`` and ``out/train_<tag>-random.jsonl``; the
validation file is the last round's, unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data import read_jsonl, write_jsonl
from score import scores_for

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--band", type=float, default=0.3)
    ap.add_argument("--weight", type=float, default=3.0)
    a = ap.parse_args()
    rows = read_jsonl(Path(a.train))
    scores, secs = scores_for(a.model, [r["text"] for r in rows], batch=64)
    hard = [i for i, (r, s) in enumerate(zip(rows, scores)) if abs(s - r["label"]) > 0.5 - a.band]
    wrong = sum(1 for i in hard if abs(scores[i] - rows[i]["label"]) > 0.5)
    for r in rows:
        r["weight"] = 1.0
    for i in hard:
        rows[i]["weight"] = a.weight
    write_jsonl(OUT / f"train_{a.tag}.jsonl", rows)
    ctrl = [dict(r, weight=1.0) for r in rows]
    for i in random.Random(0).sample(range(len(ctrl)), len(hard)):
        ctrl[i]["weight"] = a.weight
    write_jsonl(OUT / f"train_{a.tag}-random.jsonl", ctrl)
    by_fam: dict[str, int] = {}
    by_car: dict[str, int] = {}
    for i in hard:
        by_fam[rows[i]["family"]] = by_fam.get(rows[i]["family"], 0) + 1
        by_car[rows[i]["carrier"]] = by_car.get(rows[i]["carrier"], 0) + 1
    stats = {
        "model": a.model,
        "rows": len(rows),
        "hard": len(hard),
        "wrong": wrong,
        "band": a.band,
        "weight": a.weight,
        "hard_by_family": by_fam,
        "hard_by_carrier": by_car,
        "scoring_seconds": round(secs, 1),
    }
    (OUT / f"mine_{a.tag}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
