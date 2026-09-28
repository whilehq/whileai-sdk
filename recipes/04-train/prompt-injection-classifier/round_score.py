"""Score one trained round on both frozen tests and the probe pairs, and append it to out/rounds.json.

    python round_score.py --tag v4-spml --version v4-spml-direct --note-file out/note_v4.txt --n-train 25461

Seeds 1, 2, 3 are scored on ``test.jsonl`` (threshold from the validation split
of that round's train); seed 1 is scored on ``test_hard.jsonl`` at the same
threshold. The platform reads seed 1; results.json carries every seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run directory prefix under out/, e.g. v4-spml")
    ap.add_argument("--version", required=True, help="the platform version name")
    ap.add_argument("--note-file", required=True)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument(
        "--val",
        default=None,
        help="validation file for the threshold (default out/val_<tag>.jsonl)",
    )
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    val = a.val or str(OUT / f"val_{a.tag.split('-')[0]}.jsonl")
    for s in a.seeds.split(","):
        run = OUT / f"{a.tag}-seed{s}"
        main_scores = OUT / f"scores_{a.tag}_seed{s}.json"
        subprocess.run(
            [
                PY,
                str(HERE / "score.py"),
                "--model",
                str(run),
                "--out",
                str(main_scores),
                "--calib",
                val,
            ],
            check=True,
        )
        thr = json.loads(main_scores.read_text())["threshold"]
        for name in ("hard", "paste", "llm"):
            subprocess.run(
                [
                    PY,
                    str(HERE / "score.py"),
                    "--model",
                    str(run),
                    "--test",
                    str(HERE / f"test_{name}.jsonl"),
                    "--threshold",
                    str(thr),
                    "--probe",
                    "none",
                    "--out",
                    str(OUT / f"scores_{name}_{a.tag}_seed{s}.json"),
                ],
                check=True,
            )
    rounds = json.loads((OUT / "rounds.json").read_text()) if (OUT / "rounds.json").exists() else []
    rounds = [r for r in rounds if r["version"] != a.version]
    rounds.append(
        {
            "version": a.version,
            "method": "classify",
            "base": "nreimers/MiniLM-L6-H384-uncased",
            "scores": f"scores_{a.tag}_seed1.json",
            "hard_scores": f"scores_hard_{a.tag}_seed1.json",
            "paste_scores": f"scores_paste_{a.tag}_seed1.json",
            "llm_scores": f"scores_llm_{a.tag}_seed1.json",
            "tag": a.tag,
            "n_train": a.n_train,
            "trained_on": f"planted-carriers-{a.tag}",
            "note": Path(a.note_file).read_text().strip(),
        }
    )
    (OUT / "rounds.json").write_text(json.dumps(rounds, indent=1))
    print(f"rounds.json: {[r['version'] for r in rounds]}")


if __name__ == "__main__":
    main()
