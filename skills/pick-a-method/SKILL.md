---
name: pick-a-method
description: >
  Pick the training method from the graded rows before spending a GPU.
  Use when a coding agent has scored a model with k rollouts per task and
  must choose between SFT, GRPO, DPO and the research methods the hosted
  trainer does not run (wai.OPSD, wai.OPD, wai.GroupwiseGrading), when a
  base scores zero on every task, when every task already passes, or when
  someone proposes distilling from a bigger teacher. Not for rewriting the
  reward or the judge.
metadata:
  version: "1.0.0"
---

# Pick a method from the graded rows

The hosted trainer runs `sft`, `grpo`, `dpo` and `rm`. `whileai.methods`
ships more: `wai.OPSD`, `wai.OPD` and `wai.GroupwiseGrading`, run on your
own GPUs through `wai.prime_rl_config`. Nothing in `train` points at them,
so the per-task pass rates have to. Read them first, then pick.

`check.py` runs every block below offline. Names it defines: `floor_rows`,
`saturated_rows`, `mixed_rows`, `student_rows`, `weak_teacher_rows`,
`strong_teacher_rows` (graded rows, `k=4` per task), `grader` (a program
standing in for the judge), `TASKSET`, `MODEL`, `TEACHER` (a
`wai.Endpoint`).

## 1. Read the band per task, not the mean

**A mean hides the two cases that break a grouped method.** A task where
every rollout fails, and a task where every rollout passes, both give GRPO
zero advantage and teach it nothing (all-0/all-1 groups carry no signal,
DAPO dynamic sampling; Lambert 2025, chapter Reinforcement Learning). So
sort the tasks before you pick.

```python
def band(rows) -> str:
    rates = list(pass_at(rows).per_task.values())
    if all(r == 0.0 for r in rates):
        return "floor"
    if all(r == 1.0 for r in rates):
        return "saturated"
    return "mixed"
```

**Mixed** is the normal case: GRPO on a verifier (`grpo-verifier/`), or
pairs (`dpo-pairs/`). The other two need different methods.

## 2. Floor: every task fails every rollout

**More rollouts will not help.** Rejection sampling trains only on the
completions it keeps (Lambert 2025, chapter Rejection Sampling). With pass
rate `p`, `k` draws keep at least one with probability `1 - (1 - p) ** k`,
which is 0 at `p=0` for every `k`. Raising `repeats=` spends more for the
same zero.

**Give the model the answer instead.** OPSD runs the student twice: once
with privileged information (a reference answer, a hint, a passing
demonstration) and once without, and distills the second toward the first
(Zhao et al. 2026, arXiv:2601.18734; Lambert 2025, chapter Synthetic Data
and Distillation). It needs no passing rollout.

```python
cfg = wai.prime_rl_config(TASKSET, wai.OPSD(privileged="reference"), model=MODEL)
print(cfg.method, cfg.command)
```

**Check before you run it.** OPSD needs in-context learning strong enough
to use the hint; the papers report it fails under about 7B
(arXiv:2601.19897, arXiv:2601.20802). On a small model, run SFT on a
stronger model's completions instead (`sft-from-traces/`).

## 3. Saturated: every task passes every rollout

**Binary reward drops them all, even when the passes differ in quality.**
GRPO's group advantage is zero when every reward is 1 (Shao et al. 2024,
arXiv:2402.03300). If a judge can tell a good pass from a merely passing
one, `GroupwiseGrading` writes that into the reward (GRS, MiMo-V2.6 2026,
section 4.3.1).

```python
method = wai.GroupwiseGrading(grader=grader, mode="reward")
```

**Audit that grader first.** GroupwiseGrading trains on whatever the grader
ranks, so a biased grader's habits become the reward. Run
`audit-your-judge/` on it before the GPU.

**If no judge can tell them apart, stop.** The eval is too easy for this
model. Build a harder one (`strengthen-your-evals/`) before training.

## 4. A bigger teacher: score it before OPD

**OPD pulls the student toward the teacher.** Its per-token advantage is
`log pi_T - log pi_theta` (Lambert 2025, chapter Synthetic Data and
Distillation, eq. 10). A teacher no better than the student has nothing to
pull it toward, and the run looks the same as a good one until the eval.

**Score both on the same held-out rows first, with an audited judge**
(`audit-your-judge/`); a generous judge makes a weak teacher look strong.
The teacher has to clear
`PROVE_EFFECT` (0.05), and the two intervals must not overlap. The margin
is the package's proof bar against the run-to-run spread the book measures
(0.25 to 1.5 points; Lambert 2025, chapter Evaluation). The two-part rule
is engineering, not a result from the book.

```python
def teacher_clears(teacher_rows, student_rows) -> bool:
    t, s = pass_at(teacher_rows), pass_at(student_rows)
    gap = t.pass_at_1 - s.pass_at_1
    apart = t.ci95[0] > s.ci95[1]  # intervals do not overlap
    print(f"teacher {t.pass_at_1:.3f} {t.ci95}  student {s.pass_at_1:.3f} {s.ci95}")
    return gap >= PROVE_EFFECT and apart
```

If it does not clear, stay on GRPO or find a stronger teacher.

## 5. One tokenizer, or OPD learns nothing

**The teacher scores the student's own tokens.** Distillation needs student
and teacher to share a tokenizer, since the supervision is per token
(Lambert 2025, chapter Synthetic Data and Distillation). A mismatch does
not fail loudly; it silently drops the signal (SimCT, arXiv:2605.07711).
`prime_rl_config` does not check this. Compare `len(tokenizer)` for both
before writing the config.

```python
teacher_vocab, student_vocab = 151_936, 151_936  # each tokenizer's len(tokenizer)
assert teacher_vocab == student_vocab, "OPD needs one tokenizer: pick a same-family teacher"
cfg = wai.prime_rl_config(TASKSET, wai.OPD(TEACHER), model=MODEL)
print(cfg.method, cfg.command)
```

## 6. The routing, end to end

```python
def route(rows, teacher_rows=None) -> str:
    where = band(rows)
    if where == "floor":
        return "OPSD"
    if where == "saturated":
        return "GroupwiseGrading"
    if teacher_rows is not None and teacher_clears(teacher_rows, rows):
        return "OPD"
    return "GRPO"
```

Whichever method it names, finish the way every skill does: frozen test
first, noise floor, score every behavior, report with `track(...)`
(`skills/README.md`). This skill picks the method; the skill for that
method trains and reports it.
