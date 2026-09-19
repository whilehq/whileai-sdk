"""Curriculum: order tasks easy-to-hard and retire the solved ones.

Lambert 2025, chapter Reasoning: a curriculum needs per-prompt difficulty.
Keep the prompts the current policy solves in the 20-80% band, hold the ones
it never solves (no gradient yet), and retire the ones it always solves (they
teach nothing). The difficulty is the task's pass rate over its k rollouts,
the same number ``group_signal`` and ``Calibration`` already carry, so this is
a projection of graded rows, not new measurement.

    rows = data.grade(judge=my_verifier).rows
    cur = curriculum(rows)
    cur["schedule"]          # trainable task ids, easy -> hard
    cur["retired"]           # solved: drop from the RL set
    cur["not_ready"]         # too hard now: hold until the policy improves
    rows2 = retire_solved(rows)   # rows minus the solved tasks
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .optimize import DEFAULT_BAND, _group_label_lists
from .stats import task_key

# The same edges select_for_rl keeps: below the floor a task is not ready,
# above solved it is retired, between them it is trainable.
DEFAULT_SOLVED = DEFAULT_BAND[1]
DEFAULT_FLOOR = DEFAULT_BAND[0]


def _task_stats(rows: Sequence[dict]) -> dict[str, dict[str, Any]]:
    """pass_rate, n and a representative prompt per task (grouped by ``task_key``)."""
    groups = _group_label_lists(rows)
    # The first prompt seen for each task: the row's wording, where the
    # key is the engine's situation id.
    prompts: dict[str, str] = {}
    for row in rows:
        if isinstance(row, dict):
            prompts.setdefault(task_key(row), str(row.get("prompt") or ""))
    stats: dict[str, dict[str, Any]] = {}
    for task, labels in groups.items():
        if not labels:
            continue
        stats[task] = {
            "task_id": task,
            "prompt": prompts.get(task, task),
            "n": len(labels),
            "pass_rate": round(sum(labels) / len(labels), 4),
        }
    return stats


def curriculum(
    rows: Sequence[dict],
    *,
    solved: float = DEFAULT_SOLVED,
    floor: float = DEFAULT_FLOOR,
    band: tuple[float, float] = DEFAULT_BAND,
    tiers: int = 3,
    min_rollouts: int = 2,
) -> dict[str, Any]:
    """Split graded tasks into a training curriculum by measured difficulty.

    A task is *solved* when its pass rate is above ``solved`` (retire it:
    an all-pass task is dead gradient). It is *not ready* when its pass
    rate is below ``floor`` (hold it: no signal until the policy can
    sometimes solve it). Everything from ``floor`` to ``solved`` inclusive
    is *trainable*, ordered easy to hard (highest pass rate first) and
    split into ``tiers`` difficulty buckets for a staged schedule. The
    defaults are the two edges of ``DEFAULT_BAND`` (20% and 80%), the
    same band ``select_for_rl`` keeps, so a task at 1 of 8 is not ready
    here and out of band there for the same reason. Tasks with fewer than
    ``min_rollouts`` graded rollouts cannot have a difficulty and are
    reported separately.

    Returns a report; nothing is mutated. ``band`` is recorded and used only
    to count how many trainable tasks sit in the reasoning-recipe 20-80%
    sweet spot, so you can see whether the set has usable signal.
    """
    stats = _task_stats(rows)
    solved_t, not_ready, trainable, thin = [], [], [], []
    for s in stats.values():
        if s["n"] < min_rollouts:
            thin.append(s)
            continue
        p = s["pass_rate"]
        if p > solved:
            solved_t.append(s)
        elif p < floor:
            not_ready.append(s)
        else:
            trainable.append(s)
    # Easy first: a curriculum ramps difficulty up, so high pass rate leads.
    trainable.sort(key=lambda s: (-s["pass_rate"], s["task_id"]))
    lo, hi = band
    in_band = sum(1 for s in trainable if lo <= s["pass_rate"] <= hi)

    buckets: list[list[dict]] = [[] for _ in range(max(1, tiers))]
    if trainable:
        step = len(trainable) / len(buckets)
        for i, s in enumerate(trainable):
            buckets[min(len(buckets) - 1, int(i / step))].append(s)

    return {
        "n_tasks": len(stats),
        "n_trainable": len(trainable),
        "n_solved": len(solved_t),
        "n_not_ready": len(not_ready),
        "n_thin": len(thin),
        "n_in_band": in_band,
        "thresholds": {
            "solved": solved,
            "floor": floor,
            "band": [lo, hi],
            "min_rollouts": min_rollouts,
        },
        "schedule": [s["task_id"] for s in trainable],
        "trainable": trainable,
        "retired": solved_t,
        "not_ready": not_ready,
        "thin": thin,
        "tiers": buckets,
    }


def retire_solved(
    rows: Sequence[dict], *, solved: float = DEFAULT_SOLVED, min_rollouts: int = 2
) -> list[dict]:
    """Return the rows with every solved task removed. A task above the
    ``solved`` pass rate teaches nothing, so its rollouts are dropped; tasks
    with too few rollouts to judge are kept."""
    stats = _task_stats(rows)
    drop = {
        s["task_id"] for s in stats.values() if s["n"] >= min_rollouts and s["pass_rate"] > solved
    }
    return [r for r in rows if task_key(r or {}) not in drop]


def format_curriculum(report: dict[str, Any]) -> str:
    """One-line-per-fact summary for a terminal."""
    t = report
    lines = [
        f"tasks {t['n_tasks']}  trainable {t['n_trainable']} "
        f"(in band {t['n_in_band']})  retired {t['n_solved']}  "
        f"not ready {t['n_not_ready']}  thin {t['n_thin']}",
    ]
    for i, bucket in enumerate(t["tiers"], 1):
        if bucket:
            rates = [s["pass_rate"] for s in bucket]
            lines.append(
                f"  tier {i}: {len(bucket)} tasks, pass {max(rates):.2f} -> {min(rates):.2f}"
            )
    return "\n".join(lines)
