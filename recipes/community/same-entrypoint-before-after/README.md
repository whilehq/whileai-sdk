# One entry point, or your before/after measures the SDK

**Seat:** a post-training engineer at a startup that ships one production agent, trying to
find out whether a small open model can take over the boring half of it.

**Question:** the previous run in the customer-simulation ledger ([#31](https://github.com/whilehq/whileai-sdk/issues/31))
trained an adapter and got `no_change_detected (-0.025)`, but could not tell whether that
meant "the fine-tune did nothing" or "the two arms went through different code paths". Its
baseline ran through `wai.hosted_model`, which sends `chat_template_kwargs={"enable_thinking": false}`;
its trained model ran through `wai.local_model`, which has no way to send it
([#264](https://github.com/whilehq/whileai-sdk/issues/264)). The confound sat exactly on the
before/after axis. **Does the delta survive putting both arms through the same entry point?**

What you will learn: how to pin a task set across a model swap, why the noise floor has to come
first, and the one thing that turned out to matter more than the fine-tune — that grading a
reply which still contains its own reasoning moves this eval's headline number by **26 points**,
which is an order of magnitude more than the adapter moved it.

You need `WHILEAI_API_KEY`. **No training run and no `wai.serve` call**: both models were already
hosted on the account. `--dry-run` needs no key and no GPU.

## Run it

```bash
uv add whileai            # 0.53
cd recipes/community/same-entrypoint-before-after
python run.py                  # tasks -> 3 base passes -> 1 adapter pass -> report
python run.py --dry-run        # offline, no key, no GPU
python run.py report           # re-print from saved rows
```

| flag | default | what it does |
|---|---|---|
| `step` | `all` | `tasks`, `eval`, `report` |
| `--budget` | 400 | rollouts the offline template writer draws the task grid from |
| `--repeats` | 4 | rollouts per task, so `pass^k` and `pass@k` mean something |
| `--dry-run` | off | offline end to end |

## What I actually ran

- **Base arm:** served model `qwen3-4b-think` (`adapterRunId: None`, i.e. the bare `Qwen/Qwen3-4B`)
- **Trained arm:** served model `billing-boring-half` (`adapterRunId: run_327b614f3682cae5`,
  the previous seat's SFT run: 87 rows, LoRA r=16, 2 epochs, held-out loss 5.4585 → 1.7336)
- Both through `wai.local_model(ENDPOINT, name, tools=..., api_key=..., temperature=0.8)`
- 37 pinned held-out tasks × 4 repeats ≈ 151 rows per pass, `concurrency=8`
- Three base passes (seeds 101/202/303) for the noise floor, one adapter pass (seed 101)

## Results

**Noise floor first.** Three passes of the *same* base over the *same* 37 pinned tasks, uniformly
regraded: 0.750 / 0.794 / 0.801. `run_std` **0.0275**, `noise_band` **0.055**,
`stability: high_variance`, `tasks_in_every_run: 37`. Anything under ~5.5 points is nothing.

**The before/after**, base → trained adapter, both arms through `local_model`, paired on 37 of 37
tasks (`n_unpaired_tasks: 0`):

| metric | base | trained | paired delta [95%] | verdict |
|---|---|---|---|---|
| **pass@1** | 0.750 | 0.818 | **+0.068 [−0.061 .. +0.196]** | `no_difference_detected` |
| `no_invented_amount` | 0.577 | 0.833 | **+0.183 [+0.013 .. +0.367]** | `b_better` → **improved** |
| `escalated_over_200` | 1.000 | 0.783 | **−0.123 [−0.254 .. −0.018]** | `a_better` → **slipped** |
| `looked_up_before_amount` | 0.837 | 0.717 | −0.046 [−0.233 .. +0.125] | within noise |
| `used_a_tool` | 1.000 | 1.000 | +0.000 | within noise |

**Null A/B control** (base pass 2 → base pass 3, same policy resampled): every metric
`no_difference_detected`, `regressions: []`, `replicated: True`. The eval does not cry wolf.

### The answer to the question I asked

**No — the fine-tune's headline win does not survive a clean measurement, and that is a real
answer rather than a confounded one.** pass@1 moved +6.8 points, just above the 5.5-point noise
band, but the paired interval covers zero on 37 tasks. With both arms through the same entry
point there is no `<think>` asymmetry left to blame: the honest reading is "not detectable at
this sample size", and 37 tasks is too few to resolve a 7-point effect.

**But the headline was hiding a trade, in both directions.** The adapter genuinely learned the
thing it was trained for — `no_invented_amount` **+0.183, interval clear of zero** — and paid for
it somewhere nobody was looking: `escalated_over_200` **−0.123, interval clear of zero**. The base
model never once issued a credit above $200; the trained model does it about 22% of the time.
That is a policy violation the fine-tune *introduced*, and a single pass@1 number nets the two
against each other and reports a shrug.

`delta_report` caught it without being asked — `slipped: ['marker:escalated_over_200']` plus a
warning naming the drop and its interval. It did **not** fail the run, because `ok: True` only
reflects `must_not_regress=`, and I had guarded `no_invented_amount` — the marker that improved.
**I guessed the wrong marker to protect.** The lesson for anyone copying this: `must_not_regress`
should list the behaviours you are *not* training, not the one you are.

It also volunteered `ceiling: True` — *"the before run already passes 19 of 37 paired tasks every
time, so there is little room to measure improvement; use harder situations"* — which is the same
warning the previous seat got and the reason to take item 2 under **Next** seriously.

### The `<think>` sensitivity, measured on both arms

| arm | rows with `<think>` | unclosed | pass@1 graded as-is | pass@1 stripped |
|---|---|---|---|---|
| base | 151 / 151 | 34 | 0.485 | 0.750 |
| trained | 150 / 150 | 40 | 0.642 | 0.818 |

Both arms shift by ~26 and ~18 points. Because the shift is large *and* unequal between arms, a
before/after that strips on one side and not the other — which is what `hosted_model` vs
`local_model` does for you — can manufacture or erase a result of this size at will.

## What did not work, and what to copy

**Copy this: grade a thinking model's reply with the reasoning stripped, and say which you did.**
Every one of 151 rows came back with `<think>` still in `final_text`, and 34 of them were cut off
before `</think>` — so `final_text` was reasoning and nothing else. Markers that read prose then
score the model's *hypotheticals* ("if the invoice were $500…") as claims it made. Stripping the
reasoning moved `no_invented_amount` from 0.310 to 0.577 and pass@1 from 0.485 to 0.750. The
noise band from three re-runs of the same model on the same pinned tasks is 0.055, so the
artefact is about **4.8× the noise band** — much larger than the thing the SDK correctly tells
you to worry about, and invisible unless you go looking.

This is the whole reason the previous run's `-0.025` was uninterpretable: with one arm through
`hosted_model` and one through `local_model`, that 26-point artefact is applied to **one side only**.

**Do not trust a marker pinned at exactly 1.000 — but do not delete it either.** On the base arm
`escalated_over_200` and `used_a_tool` both read 1.000 with a zero-width interval. That is not a
pass, it is a marker that never had a chance to fail: across 151 base rows the model made **2**
`issue_credit` calls against 63 `escalate_to_human` calls, so the guarded branch was barely
exercised. The ledger has warned about this shape twice and I still shipped two of them
([#270](https://github.com/whilehq/whileai-sdk/issues/270)).

The twist is that `escalated_over_200` turned out to be the most valuable marker in the run.
Degenerate on the *before* side, it had plenty of variance on the *after* side — the adapter
issues large credits the base never did — and that is exactly how the regression surfaced. So the
rule is not "drop markers that pin at 1.000"; it is **"a marker pinned at 1.000 is not yet
evidence of anything, and you will not know until something moves"**. Keep it, and do not report
it as a pass.

**`split_pseudo_production` does not give you a task-disjoint holdout**
([#268](https://github.com/whilehq/whileai-sdk/issues/268)). It is prompt-disjoint, which is what
its docstring promises, but 16–17 of ~28–37 held-out `scenario_id`s also appear in train, and
`decontaminate` reports the pair clean because it compares prompts. Since the engine's whole job is
writing rephrasings of one situation, prompt-disjoint and situation-disjoint are very different
sets. The script prints the task overlap so you cannot miss it.

**Things that cost me time and are worth knowing before you start:**

- The row handed to `grader=` has `steps`, not `messages`.
- A callable agent's own step dict uses `args`; the simulated model's steps use **`arguments`**.
  My `>$200` check read `args` only, so it silently never fired — and a marker that never fires
  reads as a perfect 1.000.
- The declared `returns` shape is not what comes back. I declared `amount_usd`; the simulator
  returned `{"status": "ok", "amount": 455.0}`. Match on values, not key names.
- `marker_summary` entries carry the rate under **`mean`**. There is no `rate` key; asking for one
  gets you `None` next to a populated `ci95`.
- `wai.local_model` has a **zero-character docstring**
  ([#269](https://github.com/whilehq/whileai-sdk/issues/269)), and it is the only way to evaluate a
  model you serve. I read the signature.
- Thinking mode costs about **8× wall clock**: 151 rows in 492 s here, against ~124 rows in 38–57 s
  for the previous seat's `hosted_model` pass at the same concurrency.

## Cost

No training run was started and nothing new was served — both models were already hosted, so this
is inference only: **four passes × ~151 rows ≈ 33 minutes of an already-warm A10G, well under $1**.
The SDK still reports no cost anywhere (`run`, `wai.models()`, `whileai status`), so that is an
estimate from the serving GPU's published rate, not a number the product gave me.

## Next

1. Re-run with the reasoning turned off at the server, not stripped in the grader, once
   [#264](https://github.com/whilehq/whileai-sdk/issues/264) has a `thinking=False`. Stripping is a
   workaround; the 34 truncated rows are *lost*, not recoverable, because the model spent its whole
   budget reasoning and never answered.
2. Build the eval on situations that actually exercise the `>$200` branch. Neither
   `scenario_dimensions` nor the prompt carries the invoice amount — the simulator invents it — so
   the only way I can see to force the branch is `result_shapes=` / `fault_plans=` on `local_model`.
   Untested.
3. Someone should run `method="grpo"` on the hosted path. `list_runs()` still shows only `sft` and
   one failed `dpo` on this account.
