# GRPO group size, with the budget held still

Sweep GRPO's group size (`num_generations`) across G ∈ {2, 4, 8} while every
arm spends **the same 768 rollouts**, and ask whether the knob moves held-out
GSM8K accuracy past the noise floor. Because the budget is fixed, G is not
"more sampling"; it is group size traded against prompt coverage, which is
the choice you actually face when the GPU hours are already decided.

What you will learn: how to sweep a training knob without the arms costing
different amounts, how to get a noise floor out of the sweep itself for free,
and what fraction of a fixed rollout budget each group size throws away on
groups that carry no gradient. You need a Modal account and about 30 minutes
of one L40S per three arms; `--dry-run` needs neither, and no model API key
is used anywhere in this recipe — the reward and the metric are both programs.

## Why fixed budget

`recipes/04-train/grpo/train_modal.py` sets

```text
num_generations=num_generations,
per_device_train_batch_size=num_generations,
```

so raising the group size also raises the generation batch: `G=8` costs four
times `G=2`, and an arm that wins may have won on compute. Here the
generation batch is pinned at 16 and only the grouping moves:

| G | groups per step | prompt visits per step | prompt visits total | rollouts |
|---|---|---|---|---|
| 2 | 8 | 8 | 384 | 768 |
| 4 | 4 | 4 | 192 | 768 |
| 8 | 2 | 2 | 96 | 768 |

All three arms finished within seconds of each other on identical hardware,
which is the check that the budget really was held still.

## Run it

```bash
uv add whileai modal
cd recipes/community/grpo-group-size-at-fixed-budget

python run.py --dry-run        # both halves on synthetic arms: no GPU, no key
python run.py prep             # GSM8K -> holdout.json + train_prompts.json
modal run sweep_modal.py       # three arms, ~30 min of one L40S each
python run.py analyze          # every number in this README
```

| flag | default | what it does |
|---|---|---|
| `--dry-run` | off | synthetic arms, no GPU and no key |
| `--steps` | 48 | optimizer steps per arm (the budget knob; keep it equal across arms) |
| `--holdout-n` | all | trim the holdout, for a smoke run |
| `--only` | all arms | run a single G |

Smoke first: `modal run sweep_modal.py --steps 2 --holdout-n 8 --only 4`
is about three minutes and proves the whole path.

## The setup

- **Base:** `Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16 α=32 on all attention and
  MLP projections, lr 5e-6, β(KL)=0.04, `scale_rewards=True`, temperature 0.8.
  Gradient checkpointing is **off**: this trainer generates during training.
- **Reward:** `wai.verify.MathEqual()` — a program, not a judge.
- **Held-out set:** 100 GSM8K test questions, 4 samples each, scored by the
  same verifier at the same temperature as training.
- **Decontamination:** `wai.decontaminate(train_rows, holdout)` before
  anything trains. 384 in, 384 kept, 0 contaminated — GSM8K's train and test
  splits are genuinely disjoint, so this is a clean control rather than a
  save.
- **Powered for:** `holdout_size(0.10, base=0.45, k=4)` says 98 tasks; 100 at
  k=4 detects **+0.10** at 80% power and would need 391 for +0.05. That bound
  was fixed before the GPU ran, and it is the reason a small gap here is
  reported as "not resolved" rather than "absent".

### The training reward is the eval metric, on purpose

Both are `MathEqual` on the final answer. For a knob sweep that is what you
want — any gap between arms is the knob, not reward design. It does mean this
run has no proxy-versus-target gap to detect, so it says nothing about reward
hacking; `hack_scan` has nothing to do here.

## What you get

**Held-out accuracy: flat, at every group size.** Nothing here moved, and the
run was sized in advance to say so honestly.

Noise floor first — the same untrained weights, the same 100 tasks, three
separate containers:

| base pass@1 | run_std | noise band |
|---|---|---|
| 0.4575 / 0.4375 / 0.4525 | 0.0104 | **0.063** |

Each arm against its own base, paired on the same 100 tasks:

| G | prompt visits | before | after | delta (95%) | verdict |
|---|---|---|---|---|---|
| 2 | 384 | 0.458 | 0.487 | **+0.030 [−0.018, +0.077]** | flat, inside noise |
| 4 | 192 | 0.438 | 0.460 | **+0.022 [−0.030, +0.077]** | flat, inside noise |
| 8 | 96 | 0.453 | 0.470 | **+0.018 [−0.028, +0.068]** | flat, inside noise |

And the knob question itself, arm against arm on the same holdout:

| comparison | delta (95%) | verdict |
|---|---|---|
| G=2 → G=4 | −0.028 [−0.075, +0.020] | flat |
| G=2 → G=8 | −0.018 [−0.070, +0.033] | flat |
| G=4 → G=8 | +0.010 [−0.043, +0.060] | flat |

Every interval contains zero and every delta sits under the 0.063 noise band.
**At 768 rollouts, group size did not move held-out GSM8K accuracy**, and the
intervals rule out any effect bigger than about ±0.07. They do not rule out a
smaller one: `holdout_size` says the +0.010 between G=4 and G=8 would need
about 9,800 tasks to resolve. This run was powered for +0.10 and found
nothing that large — that is the whole claim.

### The number that did move

The knob has a large, clean, monotone effect one level down — on how much of
the budget carries any gradient at all. A group whose members all pass or all
fail has zero advantage, so those rollouts bought nothing:

| G | unanimous groups | iid prediction | observed / iid | rollouts that carried gradient |
|---|---|---|---|---|
| 2 | 0.693 | 0.509 | 1.4× | 236 / 768 |
| 4 | 0.448 | 0.131 | 3.4× | 424 / 768 |
| 8 | **0.281** | 0.015 | **19.4×** | 552 / 768 |

The share of unanimous groups *has* to fall as G grows, so the raw column
proves nothing by itself. The comparison against `p^G + (1-p)^G` at each arm's
own mean reward is the result: at G=8 a homogeneous model predicts 1.5% of
groups wasted and the run wastes **28%** — nineteen times more. The gap widens
sharply with G.

The reason is prompt heterogeneity. GSM8K at this model is not 100 coin flips
at p≈0.57; it is mostly questions the model always gets right and questions it
always gets wrong, with few borderline ones. Bigger groups keep re-sampling
the saturated prompts and keep learning nothing from them.

**What that is worth knowing for:** it prices dynamic sampling. Dropping
unanimous groups and refilling (DAPO; `wai.select(mode="rl", band=(0.2, 0.8))`
is the same rule applied to rows) looks nearly pointless at G=8 under the
textbook estimate — 1.5% of the budget. On this task it reclaims 28%. If you
are choosing between "raise G" and "filter the groups", the filter is worth
about nineteen times more than the arithmetic suggests, and this is the
cheapest measurement that tells you so.

### What would have made the accuracy move

Not the knob. 48 steps at 768 rollouts is roughly one pass over 384 prompts;
the training reward drifted 0.54–0.59 across arms with no trend, and KL stayed
near zero. The arms were never going to separate on the metric at this budget.
The honest reading is that this run measures the knob's *mechanism* well and
its *outcome* not at all, and it says which is which.

## What did not work

- **`pip install whileai` inside the pinned trainer image gives whileai 0.53**,
  57 releases old, where `wai.verify` does not exist. Diagnosed and filed as
  [#661](https://github.com/whilehq/whileai-sdk/issues/661). `sweep_modal.py`
  pins `whileai==0.110` for this reason, and you should pin it too.
- **`wai.eval_variance` and `wai.holdout_size` are `AttributeError`.** The
  docs list them beside `pass_at` and `compare`; they are only at
  `whileai.simulations.*` ([#662](https://github.com/whilehq/whileai-sdk/issues/662)).
- **Nothing in the SDK sweeps a training knob.** The fan-out, the fixed-budget
  bookkeeping, the per-container base pass and the arm-vs-arm comparison in
  these two files are all hand-written
  ([#663](https://github.com/whilehq/whileai-sdk/issues/663)). The measurement
  half — `rows`, `pass_at`, `compare`, `holdout_size` — is excellent and is
  why this recipe is short.
- **34% of completions hit the 320-token cap** and a truncated answer scores
  zero. It is equal across arms so the comparison survives, but every pass
  rate here is depressed by it. Raising `max_completion_length` to 512 is the
  first thing to change.

## Cost

Three L40S containers in parallel, one per arm: base eval (~4 min), 48 steps
(16:24), trained eval (~4 min). About 25 minutes wall clock, ~75 GPU-minutes
total, **about $2.40** at Modal's posted L40S rate. No model API key is used
by this recipe at any point.

Run ids: `ap-S1wLfBnYlccNubBrzbNfw9` (the sweep),
`ap-FPEked5LpA3b0nOuH1mS7h` (the 3-minute smoke). Both stopped, 0 containers.

## Next

1. **More steps, not more arms.** The mechanism table says G=8 gets 552
   gradient-carrying rollouts per 768 against G=2's 236. If that matters it
   should show up over 300–500 steps, not 48. Same three arms, same holdout,
   `--steps 400`: about $12 and it is the run this one earns.
2. **Filter instead of enlarging.** Add a fourth arm at G=2 *with* unanimous
   groups dropped and refilled, so it spends its 768 rollouts on prompts that
   carry advantage. If the 19× number means anything, that arm should beat
   plain G=8 at the same cost — and that is a much more useful claim than
   "bigger groups are better".
3. **Raise `max_completion_length` to 512.** 30–35% of completions hit the
   320-token cap and a truncated answer scores zero, which puts a floor of
   pure length artifact under every reward here.
4. **Pick a task with borderline prompts.** The whole effect above is driven
   by GSM8K being saturated at this model size. `select(mode="rl",
   band=(0.2, 0.8))` on the base model's own rollouts would say, before any
   training, how many prompts are even eligible to teach anything — that
   check costs one eval pass and should precede a sweep like this.
