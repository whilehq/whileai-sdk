# The step the course skips

The eight-lesson course for people who have never trained a model stops one
line short of the training: `# 2. Train. See recipes/04-train. This course
skips it.` This recipe writes that line, on one A10G, for about twenty cents,
and reports what the loop is worth when you finish it.

What you will learn: that the loop holds up — a paired **+0.300 [+0.200, +0.400]**
and **+0.412 [+0.306, +0.525]** on two independent held-out sets, both clearing
their own measured noise floor — and the three places the course's own program
stops working the moment a real model replaces the stand-in agent. You need a
Modal account; `--dry-run` needs nothing. One A10G for about five minutes per
arm.

## Run it

```bash
uv add whileai                # 1.6
cd recipes/community/the-step-the-course-skips
python run.py                 # offline: writes train.jsonl + holdout.json
modal run train_modal.py      # A10G: 3 base passes, LoRA SFT, 1 trained pass
python run.py report          # the paired delta with its interval
python run.py --dry-run       # offline, no key, no GPU, writes nothing
```

| flag | default | what it does |
|---|---|---|
| `step` | `build` | `build` or `report` |
| `--budget` | 2400 | rollouts the offline writer draws the task grid from |
| `--holdout-tasks` | 40 | tasks locked away before training |
| `--seed` | 0 | regenerates the same sets and the same split |
| `--dry-run` | off | offline end to end, nothing written |

Replicate with `python run.py --seed 1` and a different `modal.App` name in
`train_modal.py` — two `modal run`s of one script share an app name, and
stopping either kills both.

## What I actually ran

- Base `Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16 α=32 dropout 0.05, 3 epochs,
  lr 1e-4, batch 4 × grad-accum 2, bf16, `gradient_checkpointing=False`.
- Rows from `simulate(seeded_agent, simulator=False, phrasings=4)` — no model
  key anywhere in this recipe — graded by a program, selected with
  `select(mode="sft")`, decontaminated against the held-out set.
- 40 held-out tasks × 4 samples = 160 rows per pass. Three base passes
  (seeds 101/202/303) for the noise floor, then one trained pass at seed 101,
  paired against base pass 1 on all 40 tasks.
- **Both arms go through the same `evaluate()` function**; the only difference
  is whether the adapter is attached, so the delta cannot be measuring two
  code paths.

## Results

**Noise floor first**, three passes of the untrained model over the same tasks:

| seed | base passes | `run_std` | `compare()`'s own threshold |
|---|---|---|---|
| 0 | 0.469 / 0.406 / 0.412 | 0.034 | a delta under **0.209** is noise |
| 1 | 0.338 / 0.306 / 0.325 | 0.016 | a delta under **0.096** is noise |

That threshold is `compare()`'s, not mine. I had hand-rolled `2 × run_std`
(0.069) and it was three times too permissive; the library uses
`t(df=2)=4.30 × run_std × sqrt(1/1 + 1/1)` and prints the formula. Worth
knowing before you quote your own band.

**The before/after**, paired on 40 of 40 tasks, `pass@1`:

| seed | train rows | before | after | paired delta [95%] | verdict |
|---|---|---|---|---|---|
| 0 | 107 | 0.469 | 0.769 | **+0.300 [+0.200, +0.400]** | `PASS`, clears 0.209 |
| 1 | 105 | 0.338 | 0.750 | **+0.412 [+0.306, +0.525]** | `PASS`, clears 0.096 |

Two independent task sets, two independent training sets, same direction, both
intervals excluding zero and both clearing the run's own noise threshold. The
two baselines differ by 13 points (different held-out tasks) and both trained
arms land within 2 points of each other, which is the more interesting number:
the training moves the model to the behaviour rather than by a fixed amount.

`pass^4` — all four tries correct — goes 0.23 → 0.57 and 0.15 → 0.50. That is
the number worth watching for a builder: it is the share of asks where the
agent is reliable, not lucky.

Cost: two A10G runs, ~9 GPU-minutes, **$0.17**.

## What did not work

**Lesson 7's judge cannot score a real model.** It is
`int(not row["seeded"])`, and `seeded` is a field only `seeded_agent` writes.
Swap in the trained model the lesson is preparing you for and it raises on
every row. The reward here is a program over `messages` instead
([#593](https://github.com/whilehq/whileai-sdk/issues/593)).

**A tool call has two shapes.** Rollout rows carry
`{"name", "arguments": dict}`; exported rows carry the OpenAI wire shape with
`arguments` as a JSON *string*. My first reward read the wrong one and scored
`0.00 [0.00..0.00]` on all 404 rows — no error, a tight interval, and entirely
believable for a small model. `call_name_and_args()` normalises both, and
`run.py` refuses to continue on an all-zero or all-one reward
([#594](https://github.com/whilehq/whileai-sdk/issues/594)).

**Decontaminating before selecting silently drops the tool schema.**
`decontaminate()` returns a plain list; `select()` on a plain list exports with
`with_system: 0, with_tools: 0`, so the SFT file teaches tool calls for a
schema the prompt never shows. This recipe selects on the `ScoredData` first
and applies the split and the decontamination to the written rows
([#592](https://github.com/whilehq/whileai-sdk/issues/592)).

**`budget` does not buy more situations.** `budget=1200` and `budget=2400`
both return exactly 404 rows over 89 situations; `situations=400` changes
nothing. `phrasings=4` is the knob that works (1268 rows, 100 situations) and
it is what gets this recipe to ~105 training rows.

**TRL is handed pre-rendered text.** The exported rows carry a `tools` column
and a per-message `loss_mask` that `SFTTrainer` does not read, so
`train_modal.py` applies the chat template itself. Related:
[#507](https://github.com/whilehq/whileai-sdk/issues/507).

## Caveats a reviewer should push on

- **n=40 is too small.** `score()` warns it, `holdout_size()` sizes it, and the
  platform verdict stays `unproven` on `n=40 under 50`. The result replicates,
  but neither arm alone clears the library's own bar. Use `--holdout-tasks 50`.
- **The training reward and the eval metric are the same rule.** This measures
  that the rule was learned, not that the agent got better at its job. Proxy
  equals target here, and that is a limit, not a feature.
- **One training seed per arm.** The noise floor is three *eval* passes of the
  base, so eval noise is measured and training noise is not.
- The demonstrations come from `seeded_agent`, so the ceiling is that
  generator's behaviour.

## Next

Re-run at `--holdout-tasks 50` to clear the platform's own `n` gate, and add a
second metric that the training did not optimise — the check that the gain is
not just the rule being memorised.

## Artifacts on Hugging Face

| what | repo |
|---|---|
| the adapter the run left on the volume, with `rows.json` | [`while-ai/community-step-the-course-skips-1.5b`](https://huggingface.co/while-ai/community-step-the-course-skips-1.5b) |

Part of the [Course and community runs](https://huggingface.co/collections/while-ai/course-and-community-runs-6ab271de189fd0c363cfab92) collection in the while-ai org.
