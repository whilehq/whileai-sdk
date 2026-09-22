# Which turns the mask supervises

`wai.export(format="trl")` can hand TRL three different files from the same
graded rows. Two of them are described in one sentence of the docstring as
if they were the same choice. On tool-using traces they are not: one trains
a model that calls tools on every held-out ask, the other trains a model
that never calls one again — **0.608 below the untrained base**.

What you will learn: which assistant turns each `mask_mode` actually puts
in the completion, why "the last assistant turn" is the wrong turn in a
tool-using trace, and how to check it on your own set in five lines before
you spend a GPU. You need a Modal account for the training half; `--dry-run`
needs nothing — no key, no GPU, no account. Offline half: 30 seconds. GPU
half: one L40S for 9 minutes, $0.35.

## The three exports

Same 388 rows, same base, same hyperparameters. Only the export differs.

| arm | call | rows | what TRL supervises |
|---|---|---|---|
| `assistant` | `mask_mode="assistant"` | 388 | every token of every turn |
| `final` | `mask_mode="final"` | 388 | the last assistant turn |
| `unroll` | `mask_mode="final", unroll=True` | 776 | every assistant turn, one row each |

The supervised fractions are not guesses — they come off TRL's own
collator inside the run (`(batch["labels"] != -100).sum()`):

```text
assistant  TRL supervises 3931/4320 = 0.9100
final      TRL supervises  176/4320 = 0.0407
unroll     TRL supervises  207/4320 = 0.0479   (on 2x the rows)
```

## What you get

Tool-call rate on 120 held-out prompts: the fraction of replies emitting a
well-formed `<tool_call>` naming one of the three tools. The system prompt
is *"Look the order up before you answer"*, so this is the policy itself.
Base sampled 3x for the noise floor; 2 training seeds per arm; paired
bootstrap over prompts, 2000 resamples.

| arm | tool-call rate | paired vs base [95%] | verdict |
|---|---|---|---|
| base (3 passes) | 0.608 / 0.592 / 0.625 — band 0.033 | — | — |
| `assistant` | 0.879 | **+0.271 [+0.189, +0.354]** | moved |
| `final` | **0.000** | **−0.608 [−0.686, −0.531]** | moved |
| `unroll` | **1.000** | **+0.392 [+0.314, +0.469]** | moved |

`unroll` − `final` is **+1.000 [+1.000, +1.000]**. Every interval excludes
zero and every point estimate is outside the base band.

### Did it call the *right* tool?

A tool-call rate says the model called *a* tool. It does not say it called
the right one, and 1.000 is suspicious enough to check. So a second run
grades each call against the trace's own gold call — same arms, same
seeds, same prompts:

| metric | base | `assistant` | `final` | `unroll` |
|---|---|---|---|---|
| called a tool | 0.608 | 0.879 | 0.000 | **1.000** |
| called the **right** tool | 0.547 | 0.825 | 0.000 | **1.000** |
| right tool **and** right order id | 0.517 | 0.671 | 0.000 | **0.958** |
| never invented an order id † | 0.677 | 0.901 | 0.000 | **1.000** |

† scored on the 96 of 120 held-out asks that name an order id — see below.

`unroll` − base: right tool **+0.453 [+0.372, +0.531]**, right id **+0.442
[+0.364, +0.522]**, never-invented **+0.323 [+0.243, +0.403]**. `unroll` −
`assistant` is positive and excludes zero on all three. The verdict does
not just survive the stricter metric; it widens.

**A metric defect we caught and fixed rather than published.** "Never
invent an order id" first scored `unroll` at 0.800 — apparently a 20%
policy violation. It is not: 24 of the 120 held-out asks mention no order
id at all, while their gold trace still calls with one (it comes from the
scenario's world state, not the text). Scoring those as "invented" is a
defect in the metric. 0.800 is exactly 96/120, so `unroll` had in fact
never invented an id on an ask that contained one. Scored on the 96 asks
that name an id, it is **1.000**. `analyze.py` carries the subset rule as
`CONDITIONAL` and `results.json` records `prompts_scored: 96, of: 120`.

`final` is the only arm that is *worse than not training at all*. A sample
of what it writes to a held-out ask that needs a lookup:

```text
"Certainly! How can I help?"
```

## Why: the last assistant turn is never the tool call

In a tool-using trace the agent calls the tool, the tool answers, and the
agent summarises. So the turn that *calls* is never last, and
`mask_mode="final"` puts it in the prompt:

```text
final    prompt: ['system', 'user', 'assistant', 'tool']
         completion: [{"role": "assistant", "content": "lookup_order for ORD-3083: ok."}]

unroll   prompt: ['system', 'user']
         completion: [{"role": "assistant", "content": "",
                       "tool_calls": [{... "name": "lookup_order" ...}]}]
```

Counted over the set, which `run.py` does for you:

| | |
|---|---|
| assistant turns | 776 |
| assistant turns carrying a `tool_call` | 388 |
| ...of those, in the **final** turn | **0** |
| tool calls `mask_mode="final"` supervises | **0 of 388 = 0%** |

The export report holds this fact and counts the wrong unit:
`trained_messages: 388, masked_messages: 1552, warnings: None`. 1552 masked
messages is true and unalarming; *1552 masked messages including 388 of 388
tool calls* is the same fact and would have stopped us. Filed as
[#858](https://github.com/whilehq/whileai-sdk/issues/858).

**Check your own set before you train.** Five lines, offline:

```python
total = final = 0
for row in rows:                      # a mask_mode="assistant" export
    turns = [m for m in row["messages"] if m.get("role") == "assistant"]
    for i, m in enumerate(turns):
        if m.get("tool_calls"):
            total += 1
            final += i == len(turns) - 1
print(f"{final} of {total} tool calls survive mask_mode='final'")
```

If that prints `0 of N`, `mask_mode="final"` is the wrong export for your
data and `unroll=True` is the one you want.

## Run it

```bash
uv add whileai
cd recipes/community/which-turns-the-mask-supervises

python run.py --dry-run          # offline, 2 seeds: the census, no GPU
python run.py                    # offline, 6 seeds: builds data/ for the GPU half
modal run train_modal.py         # one L40S, ~9 min, ~$0.35
python analyze.py data/results_raw.json
```

| flag | default | what it does |
|---|---|---|
| `--seeds` | 6 | how many `simulate()` seeds to pool |
| `--limit` | — | alias for `--seeds`, for smoke runs |
| `--dry-run` | off | 2 seeds, offline; no key, no GPU |
| `--out` | `./data` | where the jsonl goes |

Modal apps from the published run: `ap-PiaRvamj8tk2MnSiZP2cP9` (CPU smoke),
`ap-uVoVZ4mqOgTm2InZZRtTru` (L40S, 3 base passes + 6 LoRA arms + 6 eval
passes, 9 min), `ap-JodB2cknHbz9kkoM35Af1T` (the same again with the three
task metrics, 6 min). All stopped, $0.70 total. `whileai==0.124`,
`trl==0.19.1`, `transformers==4.54.0`, `peft==0.16.0`, `torch==2.7.1`.

## What did not work

**The pre-registered primary metric was mis-specified, and we are keeping
it in the record rather than quietly swapping it.** We pre-registered
*stub-reply share* (replies under 60 characters) and *mean reply length*,
carrying them over from an earlier run on the same data. Both are
meaningless here: after stripping the `<tool_call>` block, a **correct**
reply on this task is a bare tool call with no prose, which scores 0
characters and counts as a "stub". So `unroll` — the best arm — scores
`stub = 1.000, chars = 0.0`, and the metric reads that as total collapse.
The pre-registered numbers are in `results.json` and should be read as
measuring "how much prose is in the reply", not quality. The tool-call
rate, pre-registered as tertiary, is the metric that carries the result.
The lesson is narrow and general: a length metric inherited from a
non-tool-using run does not transfer to tool-using data.

**`simulate()` has a ceiling that `budget` does not lift.** `budget=2000`
returns the same 156 rows as `budget=600`, while the progress line reports
"1002 situations written". Pooling 6 seeds is how this recipe gets to 960
rows; the seeds re-draw the *asks* (812 distinct of 960) from the same 119
scenario templates (114 of which appear in more than one seed). That
sameness is useful — splitting on `scenario_id` gives a template-level
holdout — but "500+ rows" cost six calls, not one flag.

**Decontamination is not free on pooled data.** 55 of 443 selected rows
(12.4%) were contaminated against the holdout — 5 exact, 50 near — because
seeds re-draw asks from shared templates. This recipe **applies**
`decontaminate` (443 → 388) rather than only reporting it.

**One seed pair is not a training-noise floor.** The noise floor here is
three passes of the *base*, so it is eval noise. Both training seeds agree
exactly on `final` (0.000, 0.000) and `unroll` (1.000, 1.000), which is
reassuring but is not the same as a measured training variance.

## Next

`assistant` supervises 22x the tokens `unroll` does and loses to it on
every task metric, so on this data the cheap arm wins outright. Whether
that survives at 2000+ rows is one command and $0.35 — and it is the
interesting question, because `assistant`'s extra supervision is spent on
system prompts and tool results, which is exactly the budget that ought to
start paying off somewhere.

The other open one: every number here is from a single base model.
`Qwen2.5-1.5B-Instruct` has no `{% generation %}` block in its chat
template, which is why `assistant_only_loss=True` is not available as a
fourth arm. A base that *does* have one (Llama 3.x Instruct) would let
`mask_mode="assistant"` be trained as masked, and that is the arm this
recipe cannot build.
