"""Offline check for skills/pick-a-method/SKILL.md.

Every ```python block in the SKILL.md appears verbatim below. No key, no
model, no GPU: the graded rows are written by hand, the teacher is a URL
that is never called, and prime_rl_config only writes TOML text.
"""

from __future__ import annotations

from collections import Counter

import whileai as wai
from whileai.simulations import pass_at
from whileai.simulations.defaults import PROVE_EFFECT

TASKSET = "refunds-v1"
MODEL = "Qwen/Qwen3-8B"
TEACHER = wai.Endpoint(url="https://teacher.example/v1", model="Qwen/Qwen3-32B")


def graded(passes_per_task: list[int], k: int = 4, model: str = MODEL) -> list[dict]:
    """Rows as a grader leaves them: one per rollout, binary reward, grouped by task."""
    return [
        {
            "task_id": f"t{t}",
            "reward": 1.0 if i < n else 0.0,
            "truncated": False,
            "model": model,
        }
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
one_per_prompt = graded([1, 0, 1, 1, 0] * 10, k=1)
teacher_made_rows = graded([0, 1, 2, 3, 4] * 8, model="Qwen/Qwen3-32B")
cut_off_rows = [dict(r, truncated=True) for r in graded([4] * 5)] + graded([0] * 5)

# --- block 1: truncation --------------------------------------------------


def truncated_passes(rows) -> int:
    return sum(1 for r in rows if r.get("truncated") and r["reward"] >= 1.0)


assert truncated_passes(mixed_rows) == 0
assert truncated_passes(cut_off_rows) == 20

# --- block 2: rollouts per prompt -----------------------------------------


def rollouts_per_task(rows) -> int:
    return min(Counter(r["task_id"] for r in rows).values())


assert rollouts_per_task(mixed_rows) == 4
assert rollouts_per_task(one_per_prompt) == 1

# --- block 3: single rollout ----------------------------------------------
batch = [
    {"reward": 1.0, "logprobs": [-0.5, -1.2, -0.3]},
    {"reward": 0.0, "logprobs": [-0.7, -0.4]},
]
update = wai.FlashReinforce().update(batch)
print(update.advantages, update.notes)
cfg = wai.prime_rl_config(TASKSET, "rae", model=MODEL)  # prime-rl's k=1 option

assert update.advantages[0][0] > 0 > update.advantages[1][0]
assert cfg.config["orchestrator"]["algo"]["type"] == "rae"
try:
    wai.prime_rl_config(TASKSET, wai.FlashReinforce(), model=MODEL)
    raise AssertionError("prime-rl must refuse a single-rollout method")
except ValueError as e:
    assert "group" in str(e)

# --- block 4: the band ----------------------------------------------------


def band(rows) -> str:
    rates = list(pass_at(rows).per_task.values())
    if all(r == 0.0 for r in rates):
        return "floor"
    if all(r == 1.0 for r in rates):
        return "saturated"
    return "mixed"


def split_share(rows) -> float:
    rates = list(pass_at(rows).per_task.values())
    return sum(1 for r in rates if 0.0 < r < 1.0) / len(rates)


assert band(floor_rows) == "floor"
assert band(saturated_rows) == "saturated"
assert band(mixed_rows) == "mixed"
assert split_share(mixed_rows) == 0.6

# --- block 5: floor -> OPSD -----------------------------------------------
cfg = wai.prime_rl_config(TASKSET, wai.OPSD(privileged="reference"), model=MODEL)
print(cfg.method, cfg.command)
print(cfg.ignored)  # knobs prime-rl will not read, and why
print(cfg.warnings)

assert cfg.method == "opsd"
assert cfg.config["orchestrator"]["algo"]["type"] == "opsd"
assert any("7B" in w for w in cfg.warnings), cfg.warnings

# --- block 6: saturated -> GroupwiseGrading --------------------------------
method = wai.GroupwiseGrading(grader=grader, mode="reward")

assert method.mode == "reward"

# --- block 7: on-policy ---------------------------------------------------


def on_policy(rows, model: str) -> bool:
    return all(r.get("model") == model for r in rows)


assert on_policy(mixed_rows, MODEL)
assert not on_policy(teacher_made_rows, MODEL)

# --- block 8: stale sampler -> Async --------------------------------------
cfg = wai.prime_rl_config(TASKSET, wai.Async("grpo", correction="icepop"), model=MODEL)
print(cfg.method, cfg.config["orchestrator"]["algo"])

assert cfg.method == "async grpo", cfg.method
try:
    wai.prime_rl_config(TASKSET, wai.Async("grpo", correction="tis"), model=MODEL)
    raise AssertionError("prime-rl has no tis correction")
except ValueError as e:
    assert "icepop" in str(e)

# --- block 9: teacher -----------------------------------------------------


def teacher_clears(teacher_rows, student_rows) -> bool:
    t, s = pass_at(teacher_rows), pass_at(student_rows)
    gap = t.pass_at_1 - s.pass_at_1
    apart = t.ci95[0] > s.ci95[1]  # intervals do not overlap
    print(f"teacher {t.pass_at_1:.3f} {t.ci95}  student {s.pass_at_1:.3f} {s.ci95}")
    return gap >= PROVE_EFFECT and apart


assert not teacher_clears(weak_teacher_rows, student_rows)
assert teacher_clears(strong_teacher_rows, student_rows)

# --- block 10: tokenizer -> OPD -------------------------------------------
teacher_vocab, student_vocab = 151_936, 151_936  # each tokenizer's len(tokenizer)
assert teacher_vocab == student_vocab, "OPD needs one tokenizer: pick a same-family teacher"
cfg = wai.prime_rl_config(TASKSET, wai.OPD(TEACHER), model=MODEL)
print(cfg.method, cfg.command)

assert cfg.config["orchestrator"]["algo"]["type"] == "opd"

# --- block 11: the routing, end to end ------------------------------------


def route(rows, model: str, teacher_rows=None) -> str:
    if truncated_passes(rows):
        return "fix the reward: a cut-off reply passes"
    if rollouts_per_task(rows) == 1:
        return "FlashReinforce / SAO / BPCO, or prime-rl rae"
    where = band(rows)
    if where == "floor":
        return "OPSD"
    if where == "saturated":
        return "GroupwiseGrading, or a harder eval"
    if teacher_rows is not None and teacher_clears(teacher_rows, rows):
        return "OPD"
    if not on_policy(rows, model):
        return "SFT on these rows, or OPD from their model"
    return "GRPO"


assert route(cut_off_rows, MODEL).startswith("fix the reward")
assert route(one_per_prompt, MODEL).startswith("FlashReinforce")
assert route(floor_rows, MODEL) == "OPSD"
assert route(saturated_rows, MODEL).startswith("GroupwiseGrading")
assert route(student_rows, MODEL, weak_teacher_rows) == "GRPO"
assert route(student_rows, MODEL, strong_teacher_rows) == "OPD"
assert route(teacher_made_rows, MODEL).startswith("SFT")
assert route(mixed_rows, MODEL) == "GRPO"
print("verdict: every route above picks the method its rows call for")
