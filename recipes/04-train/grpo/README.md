# GRPO on Modal, with the dashboard watching

Group-relative RL on one testable rule, end to end: prompts from the
simulator, a reward that is a function rather than a judge, TRL's
`GRPOTrainer` with a LoRA adapter, reward and KL on the training page as it
runs, and a before/after on a holdout when it is done.

What you will learn: a reward that is a function of the first reply, how a
run reports into the platform through `wai.TrainerCallback` and
`HackMonitor`, a paired pass@1 delta with an interval, why the holdout has
to be stratified by prompt category, and how loss variants and prompt
balance change the result. You need a Modal account and one A10G (under
fifteen minutes at the default 40 steps, longer at 120);
`WHILEAI_API_KEY` is optional and only decides whether the run page is
drawn.

## Run it

```bash
uv add whileai modal
modal profile activate <your workspace>
export WHILEAI_API_KEY=...          # for the dashboard; optional
modal run recipes/04-train/grpo/train_modal.py
modal run recipes/04-train/grpo/train_modal.py --steps 80 --gpu H100 --run-name refund-grpo-v2
modal run recipes/04-train/grpo/train_modal.py --monitor-every 5 --stop-on feature   # end the run on a named hack
```

Default: 200 prompts, 20% held out by scenario, Qwen2.5-1.5B-Instruct, 40
steps of 8 generations, one A10G, under fifteen minutes. The run's URL is
printed at the start. No key means the same run with the numbers printed at
the end only.

## The environment

`reward.py` is the whole environment. The agent is the refund assistant with
one rule in its policy: look an order up before refunding it, never invent an
order id, ask when none is given. The reward reads the policy's first reply:

| prompt | right move | rule score |
|---|---|---|
| names an order (`ORD-4017`) | `lookup_order` with that exact id | 1.0 |
| names an order | refund first, another tool, or ask for the id again | 0.0 to 0.3 |
| about an order, no id | ask for the id, no tool call | 1.0 |
| about an order, no id | any tool call (the id is invented) | 0.0 |
| off topic | a short reply, no tool call | 1.0 |

A well-formed `<tool_call>` block adds 0.2 on top (so an invented call on a
no-id prompt scores 0.2, not 0.0), capped at 1.0, so the policy learns the
wire format before the rule. `pass@1` counts a reply as a pass at 1.0 only;
the raw score rides along as a marker, and so does `well_formed`, which the
delta report guards. `score` in `reward.py` is thirty lines and the tests in
`tests/recipes/test_grpo_example.py` pin every row of this table.

## Is the reward hackable?

Yes, in the ways any rule is, and the run watches for them:

- **It reads the first turn only.** Anything after the first reply or tool
  call is free. A policy that emits a correct lookup and then invents a
  refund is paid in full. The environment is one turn by design; a
  multi-turn reward is the next example, not this one.
- **The format bonus pays for the wrong move.** A well-formed block scores
  0.2 even when it is the wrong call, so "always emit a well-formed
  `lookup_order`" earns 1.0 on with-id prompts and 0.2 on the rest. That is
  the shortcut the category table catches: no-id pass@1 falling while the
  headline rises is this hack, and `--balance` is the answer.
- **Length is unpriced** except on off-topic prompts (a reply over 400
  characters scores 0.3), so a length drift is the monitor's job, not the
  reward's.

`wai.HackMonitor` samples the holdout from the live policy every
`--monitor-every` steps, logs the proxy reward and completion length beside
the training curve, and runs `hack_scan` on the last training batch: which
feature of a reply the reward is paying for, within prompt, against a
shuffled noise floor, with `endorsed=["lookup_order"]` naming what it should
be. There is no judge in this example, so no gold curve and no `divergence`
alarm; the `length` and `feature` alarms run, `--stop-on feature` (or
`length`) ends the run on one, and the summary lists every alarm. This is
the proxy-against-gold picture of over-optimization [1], drawn during
the run instead of after it.

Prompts come from `wai.simulate(simulator=False, ...)`: the template writer
needs no model and no key, and every prompt carries its `case` (the order
id it names, whether it is about orders at all) so the reward has ground
truth.

## What you see

- **During:** `reward`, `reward_std`, `kl`, `completion_length` and the
  progress bar at app.withwhile.com/platform/training, from
  `wai.TrainerCallback`.
- **After:** pass@1 before and after on the same holdout prompts, four
  samples each, with intervals; `run.delta` puts the paired comparison on
  the run page (a paired bootstrap over prompts [2, 3]) and names
  `well_formed` if it regressed. The adapter and both
  holdout row files land on the `whileai-grpo-runs` volume under the run
  name.

## Reading it as RL

This is the reasoning-model recipe at toy scale: verifiable reward,
group-relative advantage, no value model, small KL to the reference
(`beta=0.04`), LoRA so a 1.5B model trains on one GPU. Two things to try
before believing a number: `--steps 10` to check the reward curve moves at
all, and a second seed on the prompts, since 40 holdout prompts is a wide
interval. The dashboard shows both runs side by side.

## More prompts, tighter intervals

Every number from here on is from `train_modal.py` (or `../dpo/train_modal.py`
where DPO is named) with the flags quoted and the defaults otherwise:
`--seed 0`, 20% of scenarios held out, pass@1 on the holdout from four
samples per prompt before and after, and the bracket is the paired
bootstrap 95% interval from `run.delta`. The holdout split is a plain hash
by scenario on the template set and `split_holdout_stratified` once
`--prompts-file` is given (the section on categories below says why). A
GPU run is not bit-reproducible; expect the same verdict, not the same
third decimal.

The template writer gives about seventy distinct prompts, so the holdout is
fourteen and every pass@1 interval is a quarter wide. `prompts.jsonl` is a
model-written set: Qwen2.5-7B-Instruct wrote six customer messages per
template seed from two angles (plain, and six kinds of customer),
`prompts.py` kept the ones still in their seed's category (the order id
present, absent, or off topic, as `case_for` reads it) and not a
near-duplicate, and every prompt carries its seed's `scenario_id` so the
split stays by situation. 707 prompts from 67 situations (576 with an id,
74 without, 57 off topic; the seeds skew the same way).

```bash
uv run --with modal modal run recipes/04-train/grpo/write_prompts_modal.py     # rewrite the set, A10G, a few minutes
uv run --with modal modal run recipes/04-train/grpo/train_modal.py --prompts-file recipes/04-train/grpo/prompts.jsonl
uv run --with modal modal run recipes/04-train/dpo/train_modal.py  --prompts-file recipes/04-train/grpo/prompts.jsonl
```

On this set the holdout is 159 prompts from 14 situations. GRPO, 40 steps,
same flags as above: pass@1 0.17 [0.12, 0.22] to 0.29 [0.23, 0.35], paired
delta +0.12 [+0.06, +0.19], `moved`, `well_formed` flat at 1.0, reward
climbing the same way as on the template set. The interval is a third of
the width it was with fourteen holdout prompts, which is the point.

DPO on the same set (`recipes/04-train/dpo`, 60 steps, 8 samples per prompt for
the pairs): 308 pairs from 193 of 548 train prompts with contrast, pass@1
0.17 [0.13, 0.22] to 0.69 [0.63, 0.75], paired delta +0.49 [+0.37, +0.60],
`moved`, `well_formed` flat. One round of on-policy pairs beat 40 GRPO
steps: with 308 pairs the offline method sees far more contrast per step
than 8 samples a prompt give the online one. At 120 steps GRPO reads 0.18
[0.14, 0.23] to 0.85 [0.81, 0.89], +0.63 [+0.50, +0.73], `moved`, past
DPO's one round. The gap was budget, not method, and the interval is what
makes that readable; both scripts share one prompt set so the comparison
stays paired.

## Knobs

| flag | default | what it does |
|---|---|---|
| `--steps` | 40 | optimizer steps, one prompt's group each |
| `--num-generations` | 8 | completions per prompt; the group the advantage is relative to |
| `--learning-rate` | 5e-6 | LoRA learning rate |
| `--beta` | 0.04 | KL penalty to the reference (adapter off); 0 turns it off |
| `--loss-type` | bnpo | `bnpo`, `grpo`, `dr_grpo`: how the per-token loss is normalized |
| `--epsilon-high` | 0.0 | upper PPO clip; 0 keeps TRL's symmetric 0.2, DAPO uses 0.28 |
| `--no-scale-rewards` | off | do not divide the advantage by the group's reward std |
| `--mask-truncated` | off | drop completions cut at `max_completion_length` from the loss |
| `--monitor-every` | 10 | steps between hack-monitor samples of the holdout |
| `--stop-on` | | alarms that end the run: `feature`, `length`, or both comma-separated |
| `--prompts` | 200 | template situations to write prompts from (about 70 distinct) |
| `--prompts-file` | | the model-written set (`prompts.jsonl`) instead of the template writer |
| `--holdout` | 0.2 | share of scenarios held out; stratified by category on the model-written set |
| `--balance` | 0.0 | repeat minority-category train prompts up to this share |
| `--run-name` | refund-grpo-v1 | the run's name on the dashboard and its folder on the volume |
| `--base-model` | Qwen/Qwen2.5-1.5B-Instruct | any chat model TRL's `GRPOTrainer` loads |
| `--seed` | 0 | the template writer's seed |
| `--gpu` | A10G | or `ZP_GRPO_GPU`; the default run fits an A10G |

## Variants as flags

The loss variants come from the GRPO line of papers [4, 5].
TRL's default loss is `bnpo` (token-level, batch-normalized), which these
runs use. `--loss-type grpo` is the original per-sequence mean, which
favors short completions (every token of a short reply carries more of the
gradient). Dr.GRPO is `--loss-type dr_grpo --no-scale-rewards`: neither
length nor the group's reward std scales the advantage, so hard prompts
with a near-unanimous group stop getting amplified. DAPO's clip-higher and
overlong mask are `--epsilon-high 0.28 --mask-truncated`: a wider upper
clip lets a low-probability token that turned out good grow more in one
step, and the mask keeps a completion that hit `max_completion_length`
from being paid or punished for what it did not finish. DAPO's dynamic
sampling (drop groups that all pass or all fail, since their advantage is
zero) is what the platform's publish gate does to a dataset offline. `beta`
is the KL penalty to the reference policy [6]: small here because
the reference is the base model with the adapter off and the rule is close
to it. Every flag lands in the run's config on the dashboard.

Dr.GRPO at the same 120 steps and learning rate: 0.17 [0.13, 0.22] to
0.53 [0.46, 0.59], +0.34 [+0.27, +0.41], `moved`, behind the default. That
is what dropping the std scaling does at a fixed learning rate: the
advantages are smaller, so the steps are. Dr.GRPO's own recipe raises the
learning rate to compensate: at 1e-5 it reads 0.16 [0.11, 0.20] to 0.64
[0.58, 0.70], +0.46 [+0.37, +0.55], closer but still behind. Treat
`--loss-type` as a knob to tune, not a free upgrade, and let the interval
say which setting won.

## By category, and a split that was hiding one

Headline pass@1 on this set is mostly the with-id case (576 of 707
prompts). The first hash split by scenario put every no-id situation in
train, so the holdout had 612 with-id rows, 24 off-topic and no no-id at
all: a policy that learned to always call `lookup_order` would have scored
0.85 and the holdout could not have said otherwise. `prompts.py` now
splits by scenario within each category (`split_holdout_stratified`), and
both scripts print and return `by_category_before` / `by_category_after`:
pass@1 and the tool-call rate per category. On the old split the runs
above did not regress off topic (GRPO 0.92 to 0.88 on 24 rows, DPO 0.92
to 0.83, Dr.GRPO 0.83 to 0.96); the no-id case was simply unmeasured.

On the stratified split (holdout 163 prompts: 130 with an id, 15 without,
18 off topic) GRPO at 120 steps reads 0.29 [0.22, 0.35] to 0.81 [0.76,
0.86] overall, and by category:

| category | rows | pass@1 before | after | tool-call rate before | after |
|---|---|---|---|---|---|
| with_id | 520 | 0.11 | 0.80 | 0.11 | 0.81 |
| no_id | 60 | 0.95 | 0.75 | 0.00 | 0.23 |
| off_topic | 72 | 0.99 | 0.96 | 0.00 | 0.01 |

The headline moved. The no-id prompts moved the other way: the policy
learned to invent an order id on a quarter of them, which the old holdout
could not see and the reward penalizes only on the prompts where it
happens. `run.delta(..., by="category")` puts that table on the run page
with an interval per group and marks the group that dropped; the reward
weighting, or more no-id prompts in the set, is the fix, and now it is
measurable.

DPO, one round on the same split, 328 pairs: 0.29 [0.23, 0.35] to 0.72
[0.66, 0.77] overall; with_id 0.12 to 0.69, no_id 0.98 to 0.72 with the
tool-call rate 0.00 to 0.25, off_topic 0.97 to 0.94. Same regression,
same size: both methods learned "call lookup" faster than "unless there is
no id to look up". The category table is the difference between a run
that reads as a win and one that reads as a trade.

## Closing it: `--balance`

The reward already scores a tool call on a no-id prompt 0.0, so the fix
is not the reward. A group-relative update only learns from a prompt when
it samples it, and no-id prompts are a tenth of the set, so the with-id
rows carry the gradient and "call the tool" is learned before "unless
there is no id". `--balance 0.25` repeats the prompts of any category
below a quarter of the train split until it reaches that share (each
prompt at most six times; the holdout is untouched). DPO takes the same
flag, since a preference round only pairs the prompts it sampled.

```bash
uv run --with modal modal run recipes/04-train/grpo/train_modal.py --prompts-file recipes/04-train/grpo/prompts.jsonl --steps 120 --balance 0.25
uv run --with modal modal run recipes/04-train/dpo/train_modal.py  --prompts-file recipes/04-train/grpo/prompts.jsonl --balance 0.25
```

GRPO, 120 steps, same split, `--balance 0.25` (train split 857 rows: 446
with an id, 177 without, 234 off topic):

| category | rows | pass@1 before | after | after, unbalanced | tool-call rate after | unbalanced |
|---|---|---|---|---|---|---|
| with_id | 520 | 0.10 | 0.59 | 0.80 | 0.60 | 0.81 |
| no_id | 60 | 0.95 | 0.88 | 0.75 | 0.08 | 0.23 |
| off_topic | 72 | 0.96 | 0.93 | 0.96 | 0.00 | 0.01 |

Overall 0.27 [0.21, 0.33] to 0.66 [0.60, 0.72], +0.34 [+0.18, +0.48].
The invented-id rate on no-id prompts drops from 0.23 to 0.08 and the
with-id gain slows, because at a fixed 120 steps fewer of them see
with-id rows. That is the trade the group table makes visible; more steps
buy the with-id half back, and the no-id half is the one the reward can
never pay for on its own.

DPO, one round, same split, `--balance 0.25`: overall 0.27 to 0.70, with_id
0.10 to 0.64, no_id 0.94 to 0.74 with the tool-call rate at 0.22,
off_topic 0.99 to 1.00. Balance did nothing for DPO, and that is the
method: it learns only from prompts with a pass and a fail, and the base
policy almost never invents an id on a no-id prompt, so repeating those
prompts adds no pairs. The contrast exists after round one, which is what
`recipes/04-train/dpo --from-run` is for: round two samples the round-one policy,
finds the invented ids, and pairs them against the replies that asked.
GRPO samples every prompt every step, so frequency is its lever; DPO's is
another round.

Round two from the balanced round-one adapter, same split, `--balance
0.25`, 450 pairs: overall 0.64 to 0.86, with_id 0.56 to 0.91, no_id 0.82
to 0.26 with the tool-call rate at 0.63, off_topic flat. The second round
made the invented-id habit worse, not better. The with-id pairs dominate
the update, and "call lookup" generalizes across prompt kinds faster than
the few no-id pairs (a no-id prompt only pairs when the policy happened to
fail it) can hold the line. The paired headline reads +0.17 with an
interval that crosses zero, `flat`; the category table reads a trade. For
DPO the fix has to put contrast on the no-id prompts themselves, for
example a constructed rejected reply (the invented call) against the
policy's own ask, rather than more of the same prompts.

## References

1. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
2. Efron, B. Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics 7(1), 1979.
3. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
4. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
5. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
6. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
