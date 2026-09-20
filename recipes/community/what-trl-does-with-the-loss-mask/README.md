# What TRL does with the loss mask

`wai.export(..., format="trl")` writes a `loss_mask` on every row and reports
`mask_mode: "assistant"`. TRL's `SFTTrainer` has no concept of that column.
It loads the file happily and trains on the whole sequence — the system
prompt, the customer's message and the tool's JSON result included. This
recipe measures how much that changes the model you get.

What you will learn: that the exported file trains on **9x** the tokens the
export marked; that TRL's own `assistant_only_loss=True` cannot fix it on
Qwen2.5-Instruct; and what the two models actually differ by, with paired
intervals on a held-out set. You need nothing for the offline half (no key,
no account, no GPU) and a Modal account for the GPU half. Two minutes
offline; nineteen A10G-minutes for both arms.

Run against `whileai==0.88`, `trl==0.19.1`, `transformers==4.54.0`,
`peft==0.16.0`, `torch==2.7.1`.

## Run it

```bash
pip install whileai 'modal[api-proxy-support]'
cd recipes/community/what-trl-does-with-the-loss-mask

python run.py                 # build, grade, select, export, count supervised tokens
python run.py --dry-run       # stop before the tokenizer download
modal run train_modal.py      # both arms + the base noise floor, one A10G
python analyze.py             # paired deltas with 95% intervals
```

| flag | default | what it does |
|---|---|---|
| `--seed` | `0` | the simulation seed; the whole offline half is deterministic |
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | whose tokenizer decides the token counts |
| `--dry-run` | off | build and export only, no tokenizer download |
| `--out` | the recipe dir | where `train.trl.jsonl` and `holdout.jsonl` land |

## The setup

94 graded SFT rows from `wai.seeded_agent` + `simulate(simulator=False)`, no
model key. Split by `scenario_id` into 94 train and 57 holdout, so no
situation straddles the two. `select(mode="sft")` kept 94 of 99 (4 dropped
for quoting their own privileged context). Exported with `format="trl"`.

Two LoRA arms, identical rows, hyperparameters and seed. **Only the labels
differ:**

| arm | supervised tokens | what it is |
|---|---|---|
| `as_exported` | 46,480 / 46,480 = **1.000** | every token — what TRL does with this file |
| `mask_honored` | 5,168 / 46,480 = **0.111** | the export's own `loss_mask` applied |

`as_exported` is faithful: loading the same file into a real `SFTTrainer` on
CPU supervises 0.964 of tokens (the gap is batch padding, which is masked).

The export's own report agrees about the intent: `trained_messages: 188`,
`masked_messages: 282`. TRL trains on all 470.

TRL's escape hatch does not work here:

```
RuntimeError: You're using `assistant_only_loss=True`, but at least one example
has no assistant tokens ... it may be missing the `{% generation %}` keyword.
```

Neither `Qwen/Qwen2.5-0.5B-Instruct` nor `Qwen/Qwen2.5-1.5B-Instruct` has a
`{% generation %}` block. (Watch out: a substring search for `generation`
gives a false positive on `add_generation_prompt` — match the block.)

## What you get

Base sampled three times over the same 57 prompts for the noise floor;
paired bootstrap over prompts, 10k resamples.

| metric | base (3 passes) | `as_exported` | `mask_honored` | paired A−B [95%] |
|---|---|---|---|---|
| mean reply chars | 173.2 / 167.6 / 153.8 | 125.7 | 84.1 | **+41.6 [+21.0, +63.7]** |
| replies under 60 chars | 0.000 / 0.018 / 0.018 | 0.158 | 0.421 | **−0.263 [−0.404, −0.123]** |
| non-assistant leak *(pre-registered primary)* | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] |

Both surviving intervals exclude zero, and both arms sit outside the base
band on length. **Which masking you get decisively changes the model.**

At this data scale the arm that honours the mask is **worse**: it collapses
into stub replies 42% of the time ("Okay. Can you please provide more
details?") against 16% for `as_exported`. That is what 11% supervision on 94
rows buys you — starvation, not virtue. The useful claim is not "the mask is
wrong"; it is that the choice is consequential and currently invisible, and
a reader of `mask_mode: "assistant"` cannot tell which of these two models
they trained.

### The pre-registered metric that found nothing

Before looking at any arm output I registered this primary metric: the share
of replies containing content that only ever appears in a masked-off message
— a tool-result JSON key or a rendered role marker. The damage story says
`as_exported` should emit more of it.

It was **exactly 0.000 on every arm, including base**. A floor effect: with
94 rows neither arm learned to emit tool JSON at all. No support for the
obvious story. It is in the table because it was pre-registered, not because
it worked.

## What did not work

- **The pre-registered primary metric had no resolution.** See above. A
  bigger training set, or a metric measured on tokens rather than whole
  replies, would have more chance.
- **Neither arm learned to call tools.** Both produce plain prose. 94 rows
  and 3 epochs of LoRA on a 1.5B is not enough to teach the tool protocol,
  so this recipe says nothing about tool-calling accuracy.
- **I ran `decontaminate` and then did not apply it.** It found 4 near-
  duplicate rows (8-gram, `n_near: 4`, `n_kept: 90`) and I exported the
  original 94 anyway. Both arms trained on the identical 94 rows, so the
  A−B comparison is unaffected; any arm-vs-base reading inherits a 4.3%
  contaminated train set. `decontaminate` returning a `(rows, report)` tuple
  while `export` returns a bare dict (issue #457) is exactly the shape that
  invites this mistake, and I made it.
- **Training losses are not comparable across arms** (1.614 vs 0.730) — they
  are averages over different label sets. Ignore them; they are in
  `results.json` only for the record.
- **One seed, one pair of arms.** The noise floor is three base passes, but
  each arm was trained once. A second seed per arm is the obvious next run.

## Cost

One A10G, ~19 minutes wall clock for the base noise floor plus both arms
plus three eval passes. About **$0.35**. The offline half is free.

Modal app `ap-W5HWyHhAx9Sy435whg9R2x`, stopped.

## Next

- Run both arms at 500+ rows. The starvation explanation predicts the gap
  closes and may reverse; if `mask_honored` wins there, the ignored mask is
  a real correctness bug rather than a documentation one.
- Add a third arm at `mask_mode="final"` and one at `unroll=True`, which
  `training_rows` documents as the properly-masked multi-turn shape.
- The ~20 lines in `run.py:tokenize` that turn a per-message `loss_mask`
  into per-token labels are the only thing here the SDK should arguably own
  (issue #507).
