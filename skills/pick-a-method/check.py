"""Offline check for skills/pick-a-method/SKILL.md.

Every ```python block in the SKILL.md appears verbatim below. No key, no
model, no GPU: the graded rows are written by hand, the teacher is a URL
that is never called, and prime_rl_config only writes TOML text.
"""

from __future__ import annotations

import whileai as wai
from whileai.simulations import pass_at
from whileai.simulations.defaults import PROVE_EFFECT

TASKSET = "refunds-v1"
MODEL = "Qwen/Qwen3-8B"
TEACHER = wai.Endpoint(url="https://teacher.example/v1", model="Qwen/Qwen3-32B")


def graded(passes_per_task: list[int], k: int = 4) -> list[dict]:
    """Rows as a grader leaves them: one per rollout, binary reward, grouped by task."""
    return [
        {"task_id": f"t{t}", "reward": 1.0 if i < n else 0.0}
        for t, n in enumerate(passes_per_task)
        for i in range(k)
    ]


def grader(rows):  # a program stands in for the LLM judge that ranks passes
    return [1.0 for _ in rows]


floor_rows = graded([0] * 42)
saturated_rows = graded([4] * 40)
mixed_rows = graded([0, 1, 2, 3, 4] * 8)
student_rows = graded([3, 3, 3, 2, 3, 3, 4, 3] * 6)  # pass@1 0.75
weak_teacher_rows = graded([2, 3, 2, 2, 3, 2, 2, 3] * 6)  # pass@1 0.59
strong_teacher_rows = graded([4] * 48)

# --- block 1 -------------------------------------------------------------


def band(rows) -> str:
    rates = list(pass_at(rows).per_task.values())
    if all(r == 0.0 for r in rates):
        return "floor"
    if all(r == 1.0 for r in rates):
        return "saturated"
    return "mixed"


assert band(floor_rows) == "floor"
assert band(saturated_rows) == "saturated"
assert band(mixed_rows) == "mixed"

# --- block 2 -------------------------------------------------------------
cfg = wai.prime_rl_config(TASKSET, wai.OPSD(privileged="reference"), model=MODEL)
print(cfg.method, cfg.command)

assert cfg.method == "opsd"
assert cfg.config["orchestrator"]["algo"]["type"] == "opsd"

# --- block 3 -------------------------------------------------------------
method = wai.GroupwiseGrading(grader=grader, mode="reward")

assert method.mode == "reward"

# --- block 4 -------------------------------------------------------------


def teacher_clears(teacher_rows, student_rows) -> bool:
    t, s = pass_at(teacher_rows), pass_at(student_rows)
    gap = t.pass_at_1 - s.pass_at_1
    apart = t.ci95[0] > s.ci95[1]  # intervals do not overlap
    print(f"teacher {t.pass_at_1:.3f} {t.ci95}  student {s.pass_at_1:.3f} {s.ci95}")
    return gap >= PROVE_EFFECT and apart


assert not teacher_clears(weak_teacher_rows, student_rows)
assert teacher_clears(strong_teacher_rows, student_rows)

# --- block 5 -------------------------------------------------------------
teacher_vocab, student_vocab = 151_936, 151_936  # each tokenizer's len(tokenizer)
assert teacher_vocab == student_vocab, "OPD needs one tokenizer: pick a same-family teacher"
cfg = wai.prime_rl_config(TASKSET, wai.OPD(TEACHER), model=MODEL)
print(cfg.method, cfg.command)

assert cfg.config["orchestrator"]["algo"]["type"] == "opd"

# --- the routing, end to end ----------------------------------------------


def route(rows, teacher_rows=None) -> str:
    where = band(rows)
    if where == "floor":
        return "OPSD"
    if where == "saturated":
        return "GroupwiseGrading"
    if teacher_rows is not None and teacher_clears(teacher_rows, rows):
        return "OPD"
    return "GRPO"


assert route(floor_rows) == "OPSD"
assert route(saturated_rows) == "GroupwiseGrading"
assert route(student_rows, weak_teacher_rows) == "GRPO"
assert route(student_rows, strong_teacher_rows) == "OPD"
print("verdict: floor -> OPSD, saturated -> GroupwiseGrading, strong teacher -> OPD, else GRPO")
