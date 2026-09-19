"""Stage lineage: which post-training stage consumed each row (Lambert 2025,
chapter Training Overview).

Lambert 2025 lays out post-training as a pipeline of stages: instruction
tuning (SFT), reward modeling (RM), reinforcement learning (RL), and the
held-out evaluation that judges the result. Rows carry a ``purpose`` (train /
holdout / eval) and a grading ``lineage.source``, but nothing records the
*stage* a row fed. Without it you cannot audit the one mistake the pipeline
most needs caught: a prompt used to evaluate that was also used to train.

``stamp_stage(rows, "sft")`` writes ``row["stage"]``; ``stage_report(rows)``
counts rows per stage and flags any task that appears in both ``eval`` and a
training stage — the cross-stage leak. Provenance only; no reward changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# The post-training stages, plus mid-training corpora and the eval that
# judges the result. A row belongs to exactly one.
STAGES = ("sft", "rm", "rl", "eval", "mid")
_TRAIN_STAGES = ("sft", "rm", "rl", "mid")


class StageError(ValueError):
    """An unknown stage name."""


def _task_key(row: dict) -> str:
    return str((row or {}).get("scenario_id") or (row or {}).get("prompt") or "")


def stamp_stage(rows: Sequence[dict], stage: str) -> list[dict]:
    """Return copies of ``rows`` with ``row["stage"] = stage``. ``stage`` must
    be one of ``STAGES``; nothing else is touched."""
    stage = str(stage).lower()
    if stage not in STAGES:
        raise StageError(f"stage must be one of {', '.join(STAGES)}; got {stage!r}")
    return [{**row, "stage": stage} if isinstance(row, dict) else row for row in rows]


def stage_of(row: dict) -> str | None:
    """The stamped stage, or None. An eval-sourced row with no stamp reads as
    ``eval`` (its ``lineage.source``), so a held-out set is never mistaken for
    training data just because no one stamped it."""
    if not isinstance(row, dict):
        return None
    stage = row.get("stage")
    if stage:
        return str(stage).lower()
    lineage = row.get("lineage")
    if isinstance(lineage, dict) and lineage.get("source") == "eval":
        return "eval"
    if row.get("purpose") == "eval":
        return "eval"
    return None


def stage_report(rows: Sequence[dict]) -> dict[str, Any]:
    """Rows per stage, tasks per stage, and the cross-stage leaks: any task
    used both in ``eval`` and in a training stage (sft/rm/rl/mid). That leak
    means the number you report was optimized against."""
    counts: dict[str, int] = {}
    unstamped = 0
    stage_tasks: dict[str, set[str]] = {}
    for row in rows:
        stage = stage_of(row)
        if stage is None:
            unstamped += 1
            continue
        counts[stage] = counts.get(stage, 0) + 1
        stage_tasks.setdefault(stage, set()).add(_task_key(row))
    eval_tasks = stage_tasks.get("eval", set())
    train_tasks: set[str] = set()
    for st in _TRAIN_STAGES:
        train_tasks |= stage_tasks.get(st, set())
    leaks = sorted(t for t in (eval_tasks & train_tasks) if t)
    warnings = []
    if leaks:
        warnings.append(f"{len(leaks)} task(s) used in both eval and training")
    if unstamped:
        warnings.append(f"{unstamped} row(s) have no stage; stamp_stage before you rely on this")
    return {
        "counts": counts,
        "n_unstamped": unstamped,
        "tasks_per_stage": {k: len(v) for k, v in stage_tasks.items()},
        "eval_train_leaks": leaks[:50],
        "n_leaks": len(leaks),
        "warnings": warnings,
    }


def format_stages(report: dict[str, Any]) -> str:
    counts = report["counts"]
    order = [s for s in STAGES if s in counts] + [s for s in counts if s not in STAGES]
    lines = ["  " + "  ".join(f"{s}: {counts[s]}" for s in order) or "  (no stamped rows)"]
    for w in report.get("warnings", []):
        lines.append(f"  ! {w}")
    return "\n".join(lines)
