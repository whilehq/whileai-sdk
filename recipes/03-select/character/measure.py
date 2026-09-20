"""Did character training move the trait, and did anything else slip?

    python measure.py --demo                         # scripted before/after, offline
    python measure.py before.jsonl after.jsonl       # graded holdout rows from two models

``run.py`` writes ``holdout.jsonl`` (adversarial prompts and controls,
graded, with markers). Run it once against the model before training and
once after, then hand both files here. The report is ``delta_report``:
paired per-task differences with bootstrap intervals, the target marker
as the headline, and a hard fail if ``on_task`` or ``no_filler`` drops.
Character training that costs helpfulness is not character training
(Model Spec: style "enhances rather than distracts from" helpfulness).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import whileai.simulations as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
TARGET = "marker:trait"
GUARD = ("on_task", "no_filler")


def load(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def measure(before: list[dict], after: list[dict], *, seed: int = 0) -> dict:
    return wai.delta_report(before, after, target=TARGET, must_not_regress=list(GUARD), seed=seed)


def demo(*, seed: int = 0, k: int = 4) -> tuple[list[dict], list[dict]]:
    """Two scripted students on the same holdout: untrained and 'trained'."""
    sys.path.insert(0, str(HERE))
    import run as pipeline

    constitution = pipeline.load_constitution()
    tasks = [t for t in pipeline.build_tasks(constitution) if t["split"] in ("holdout", "control")]
    tasks_by_id = {t["task_id"]: t for t in tasks}

    def judge(row):
        return pipeline.reference_judge(row, tasks_by_id=tasks_by_id)

    out = []
    for after in (False, True):

        def student(task, i, _after=after):
            return pipeline.scripted_student(task, i, seed=seed, after=_after)

        rows = pipeline.sample_rows(
            tasks, k=k, seed=seed, student=student, model="scripted", workers=1
        )
        out.append(pipeline.grade(rows, judge, name="reference", version="spec-labels", workers=1))
    return out[0], out[1]


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("before", nargs="?")
    ap.add_argument("after", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.demo:
        before, after = demo(seed=args.seed)
    elif args.before and args.after:
        before, after = load(args.before), load(args.after)
    else:
        ap.error("pass before.jsonl after.jsonl, or --demo")
    rep = measure(before, after, seed=args.seed)
    print(wai.format_delta_report(rep))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
