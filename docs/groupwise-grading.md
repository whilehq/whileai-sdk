---
title: "Groupwise grading"
sidebarTitle: "Groupwise grading"
description: "A grader that tells passing rollouts apart: MiMo-V2.6's groupwise reward synthesis and groupwise advantage redistribution as one object, a spread check before the GPU, and the TRL reward function that runs it."
---

**What you learn:** why a binary reward lets RL pass by any means, the two places a grader can tell passes apart (the reward, or the advantage), the check that says whether your grader can tell them apart at all, and the TRL call. **Needs:** nothing to read the method; a judge key and a GRPO trainer to run it. **Takes:** five minutes to read, one grader call per mixed group to run.

- **Simulate.** Sample a group of rollouts per task. Passes differ: one looked the order up first, one refunded twice, one invented an id that happened to exist.
- **Grade.** A program says pass or fail. Every pass scores 1.0, so the policy learns to pass by any means and rollouts grow.
- **Measure.** Before training, score a sample of passes with the grader. If the scores do not vary, stop: a constant is erased by group normalization.
- **Select.** Give the grader one whole group. It ranks the passes and names the hacks; ranks become quality factors.
- **Train.** Move the positive advantage from lower- to higher-quality passes with its total conserved, then train as before.

The method is section 4.3 of the MiMo-V2.6 technical report [1]. A binary
test reward makes every passing rollout equal, so RL learns to pass by any
means: speculative compatibility branches, swallowed exceptions, relaxed
validation, an answer leaked from the environment. Without a grader,
turns and token length grow until trajectories hit the length limit and
the pass rate stalls; with one, pass-rate gains are sustained, turn counts
stay roughly flat and length grows slowly (Figure 8 of the report,
MiMo-V2.6-Flash on DeepSWE, 52 steps). `whileai` writes no loss for this.
It carries the grader and the report's knobs as one object, hands TRL a
reward function whose group normalization reproduces the redistributed
advantages, and refuses to run a grader that cannot tell the passes apart.

## What the research says

- **Two places to grade.** Groupwise Reward Synthesis (GRS, section 4.3.1)
  is offline: a grader reads a group of rollouts plus the task and writes
  task-specific rubrics (solution criteria and behavior criteria); during
  training each pass is scored against them and the reward is the product
  `R_i = R_test_i * S_sol_i * S_beh_i` (equation 2). Groupwise Advantage
  Redistribution (GAR, section 4.3.2) is online: for each mixed-outcome
  group the grader inspects every rollout jointly, ranks the passes on
  approach, precision, minimality, side effects and craftsmanship, zeroes
  a confirmed hack (an external or leaked answer) and treats it as a
  failure before the group statistics are recomputed. The report uses GRS
  on a subset of high-pass-rate tasks and GAR on everything else [1].
- **The redistribution conserves mass.** With `A_i = R_i - mean(R)` and `P`
  the passes, quality factors `f_i` in (0, 1] first downweight the
  lower-quality passes, then a common factor
  `lambda = sum_P A_j / sum_P f_j A_j` puts the removed mass back among the
  passes: `A'_i = lambda f_i A_i` for a pass, `A_i` otherwise (equation 3).
  Failures are untouched, `sum_P A'_i = sum_P A_i`, and the relative
  weights are the grader's. The report caps lambda against runaway
  amplification and then subtracts the group mean again so the group has
  zero mean; unusable grader output falls back to the original advantages
  [1].
- **A grader with no spread is a null result.** On 2026-09-21 GRS was run
  on single-turn text-to-SQL with Qwen3-4B: the rubric grader gave a mean
  of 0.93 to 1,472 passing replies, the multiplier was a constant, GRPO's
  group normalization erased it and the arm matched the plain-reward arm
  (-1.7 points, 95% -5.2 to +1.7) at $14. Two rules follow. Check the
  grader's spread on a sample of passes before the GPU is spent. And put
  the method on multi-turn agent tasks where passes differ (an extra tool
  call, a skipped verification, an invented id), not on one-line answers.

## The calls

The grader is your judge call. In advantage mode it receives one group as
a list of `{"index", "prompt", "completion", "reward", "passed"}` and
returns a best-first `ranking` over the passing indices (an inner list is
a tie), or `factors` in (0, 1] by index, plus an optional `hacks` list.
The scripted grader below ranks by how many tool calls the reply made,
fewest first, and calls a pass that names an order id no tool returned a
hack. A real one is a model reading the group.

```python
import whileai as wai


def rank_group(group):
    passes = [g for g in group if g["passed"]]
    ranked = sorted(passes, key=lambda g: g["completion"].count("<tool_call>"))
    hacks = [g["index"] for g in passes if "ORD-9999" in g["completion"]]
    return {"ranking": [g["index"] for g in ranked if g["index"] not in hacks], "hacks": hacks}


gar = wai.GroupwiseGrading(grader=rank_group)  # GAR, mode="advantage", cap 3.0, hack_zero
print(gar)
```

```text
GroupwiseGrading(mode=advantage, grader=rank_group, cap=3.0, min_factor=0.5, hack_zero=True)
```

The math is a pure function. One group of four, two passes, the second
of lower quality: the better pass gains what the worse one loses, the
failures do not move, and the group still sums to zero.

```python
from whileai.methods import redistribute

print(redistribute([1, 1, 0, 0], factors={0: 1.0, 1: 0.5}))
```

```text
[0.6666666666666666, 0.3333333333333333, -0.5, -0.5]
```

Before training, hand the object the graded rows you already have (any
rows with `scenario_id`, `reward` and `final_text`) and let it grade the
groups the way training would. No spread is a `ValueError` that says what
to change; `strict=False` returns the report instead.

```python
rows = [
    {
        "scenario_id": "refund-17",
        "reward": 1,
        "final_text": "<tool_call>lookup</tool_call><tool_call>refund</tool_call> done",
    },
    {"scenario_id": "refund-17", "reward": 1, "final_text": "<tool_call>refund</tool_call> done"},
    {"scenario_id": "refund-17", "reward": 0, "final_text": "I cannot help with that."},
    {
        "scenario_id": "refund-22",
        "reward": 1,
        "final_text": "<tool_call>lookup</tool_call><tool_call>refund</tool_call> done",
    },
    {
        "scenario_id": "refund-22",
        "reward": 1,
        "final_text": "<tool_call>lookup</tool_call><tool_call>lookup</tool_call><tool_call>refund</tool_call> done",
    },
    {"scenario_id": "refund-22", "reward": 1, "final_text": "Refunded ORD-9999."},
]
print(gar.check_spread(rows))
```

```text
spread check: 4 grader scores over 2 group(s), std 0.250, 0% within 0.05 of the median 0.75
spread: yes. The grader tells these passes apart.
```

Then the trainer. TRL's `GRPOTrainer` calls a reward function on a batch
of `num_generations` consecutive completions per prompt and forms the
advantage as `(r - mean_group) / std_group`. `trl_reward` scores the batch
with your verifier, grades each mixed group, and returns `A'_i + mean(R)`,
so TRL's mean subtraction gives equation 3 exactly. The division by the
group's standard deviation is TRL's and is the one limit: it rescales the
redistributed group by its new spread, so the ranking and the relative
weights hold but the total moved mass matches the report only under
`scale_rewards="none"` (Dr. GRPO [2]). Set it the same way in both arms of
a comparison.

```python
def verifier(prompts, completions, **kwargs):
    return [1.0 if "done" in c else 0.0 for c in completions]


reward_func = gar.trl_reward(verifier, num_generations=4)
print(reward_func(prompts=["p"] * 4, completions=["a done", "b done", "c", "d"]))
print(gar.stats)
```

```text
[1.1666666666666665, 0.8333333333333333, 0.0, 0.0]
groupwise grading (advantage): 3 groups, 2 mixed, 3 grader calls, 0 failed (fell back), 1 hacks zeroed; factors mean 0.750 std 0.250
```

Pass `reward_funcs=[reward_func]` to `GRPOTrainer` with `use_vllm=True`,
`vllm_mode="colocate"` and `scale_rewards="none"`. `gar.stats` prints
groups seen, grader calls and failures, hacks zeroed and the spread of the
factors, so the run's grader is a measurement too.

Reward mode is the same object with `mode="reward"`: the grader receives
one passing rollout and its rubric and returns
`{"solution": s, "behavior": b}`; the reward becomes
`R_test * max(floor, s) * max(floor, b)` with `floor` 0 as in equation 2.
`rubrics` is a mapping from `task_id` to rubric, or a callable that writes
one from the first group seen for a task and is cached.

```python
def score_rubric(item, rubric):
    calls = item["completion"].count("<tool_call>")
    return {"solution": 1.0, "behavior": 1.0 if calls <= 2 else 0.5}


grs = wai.GroupwiseGrading(
    grader=score_rubric, mode="reward", rubrics={"refund-17": ["look up first"]}
)
print(
    grs.shape(
        [1, 1, 0],
        ["p"] * 3,
        ["<tool_call>x</tool_call> done", "<tool_call>x</tool_call>" * 3 + " done", "no"],
        group_size=3,
        task_id=["refund-17"] * 3,
    )
)
```

```text
[1.0, 0.5, 0.0]
```

## The defaults, and where they come from

| knob | default | source |
|---|---|---|
| `mode` | `"advantage"` | the report runs GAR on every code-agent task not given rubrics (section 4.3) |
| `floor` | `0.0` | equation 2 multiplies raw scores; 0.5 was the 2026-09-21 run's choice and bounded the grader at 4x |
| `cap` | `3.0` | the report caps lambda and does not print the value (convention, untested) |
| `min_factor` | `0.5` | the worst-ranked pass keeps half its advantage; the report's rank-to-factor map is unpublished (convention, untested) |
| `hack_zero` | `True` | a confirmed hack is reset to zero and treated as a failure before group statistics are recomputed (section 4.3.2) |
| spread: `std < 0.05` or `> 90%` within `0.05` | | the text-to-SQL null result of 2026-09-21 |

Every one is a named constant in `whileai/simulations/defaults.py`
(`GROUPWISE_*`, `SPREAD_*`) and a field on the object.

## References

1. Xiaomi MiMo team. *MiMo-V2.6 Technical Report*, 2026-09-21, section 4.3 "Groupwise Agentic Grading": 4.3.1 Groupwise Reward Synthesis, 4.3.2 Groupwise Advantage Redistribution, Figure 8.
2. Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO), 2025, arXiv:2503.20783.
3. Lambert. *Reinforcement Learning from Human Feedback*, 2025, arXiv:2504.12501, chapter "Over-Optimization".
