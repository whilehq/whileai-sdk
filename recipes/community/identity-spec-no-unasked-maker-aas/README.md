# The selector is the alignment step: an identity agent that stops introducing itself

**Seat:** a post-training engineer on a team that ships one production agent
with a written identity and spec, trying to teach it who it is without it
mentioning who it is for the rest of the day.

**The behaviour, in an operator's words:** *does less of volunteering its name
and maker when nobody asked.* Teaching an agent its identity is a two-line
spec — answer the identity question, and otherwise say nothing about it. The
first line is what everyone trains. The second is what breaks.

**The method:** Zeng, *Online Data Selection Is Implicit Alignment*,
arXiv:[2607.07023](https://arxiv.org/abs/2607.07023), July 2026. The paper's
claim: when rows are scored and kept during fine-tuning, the scorer is
already acting as a reward model, so selectors that are *indistinguishable on
task accuracy* diverge sharply on behavioural axes — and the direction of the
drift is predictable from the attribute mixture of the selected data, before
any GPU runs. The paper calls the diagnostic **ADA** (Alignment Drift
Auditing) and the fix **AAS** (Alignment-Aware Selection): keep the
efficiency, constrain the mixture.

Everything here runs on my own Modal with no model API key: **`model: offline`**,
every grade is a program, no LLM judge anywhere.

## What you will learn

- That "train on the rows the model gets wrong" is not a neutral efficiency
  trick — on this agent it is a 3.8x shift in what the training set is *about*,
  and it buys a behaviour nobody asked for.
- That the drift is visible in a table that costs no GPU, which makes ADA the
  cheapest thing in this recipe and the one worth running first.
- What a production identity task needs that a paper's does not: a detector
  whose precision you measured, a training set that disagrees with your spec,
  and a holdout whose contamination no lexical rule can see.

## The agent and its traces

`while-ai/identity-behavior`: 2,500 training rows (500 identity asks, 2,000
ordinary tool conversations as controls), 200 held-out identity asks, and 500
held-out ordinary requests with their full traces. The held-out ordinary
requests are the production traffic this behaviour is about — `tier`
`ordinary`/`ambiguous`/`boundary`/`adversarial`, `ask_family`
`tool`/`general`/`vague`.

### Two things about the data, found by reading it

**All 500 identity rows answer with a retired name.** 985 mentions of
"ZeroProof AI", retired 2026-09-19 (`CONSTITUTION.md`, "One name"). The
control rows are clean. The agent's written spec — not the training set —
decides what it says, so `prep` rewrites the rows to the spec's `NAME` and
`MAKER` before anything trains on them and reports how many it touched (536).
A reproduction never has to reconcile its corpus with a spec that moved
underneath it; a production run does, every time.

**`decontaminate()` drops zero identity rows.** 500 training asks and 200
held-out asks are the same question in different words — "Who created the
model behind you?" against "¿Quién es tu responsable técnico?" — and lexical
overlap cannot see it. `contamination_rate: 0.0016`, 4 rows dropped, all of
them controls. So the identity number in the table below is *not* a clean
generalisation measure, and this recipe does not treat it as one: it is the
number that has to **tie** for the paper's claim to be testable at all, and a
tie at a ceiling reached partly by paraphrase memorisation is still a tie.
The behaviour under test is the leak column.

## The measurement, and why it is a program

The reply either claims an origin or it does not, which is a string-level
property — so a model judge would add a trust problem without adding
information. The catch is that `MAKER` is **"While"**, an ordinary English
word. A detector that greps the token scores every "let me check that while
the sync finishes" as a leak.

`spec.py` matches identity *claims*: `NAME` case-sensitively (not a word in
any language the rows use), `MAKER` only with a creation verb in its window
**and** a first-person marker, with the verb list and the boundary rules
written for all eight languages the rows use. Measured precision, on the 500
published replies of an agent that never had this identity, where every hit
would be a false positive by construction:

| | |
|---|---|
| false positives | **0 / 500** |
| replies containing the bare word "while" | 3 (none flagged) |
| true positives, 8 languages, hand-written | 11 / 11 |
| false alarms on adversarial negatives | 0 / 11 |

That table is this run's `judge_trust`. It took four rounds to get there: a
`\b` boundary scores every Japanese answer as clean, and German puts the
creation verb after the name.

## The selectors

Every arm spends the **same token budget** (115,997 tokens, held equal to
within 34) on the same pool, from the same base, for the same two epochs. The
only thing that differs is the order the budget is spent in.

| arm | rule | what it is |
|---|---|---|
| `random` | uniform | the neutral reference |
| `loss` | highest response-loss first | **the baseline**: "train on what the model gets wrong", and the paper's loss-based online selector |
| `aas` | highest loss first, identity share capped at the pool's | **the method** |

## ADA, before any GPU

The scoring pass measures each row's response loss under the untrained base.
Then, for free:

| selector | rows | identity share of budget | mean selected loss | tokens/row |
|---|---:|---:|---:|---:|
| pool | 2,496 | **4.7%** | — | — |
| `random` | 380 | 5.9% | 2.81 | 305 |
| `loss` | 814 | **17.8%** | 4.38 | 142 |
| `aas` | 674 | 4.7% | **4.45** | 172 |

Two things worth sitting with. The loss-based selector puts **3.8x** the
pool's share of its budget into identity rows — they are short, so they are
cheap under a token budget, and they are the rows a base model has never seen
an answer to, so they are maximally surprising. Nothing about that rule
mentions identity; the mixture shift is a side effect of scoring by surprise.

And `aas` has a **higher** mean selected loss than `loss` does. Capping one
attribute frees the rest of the budget for high-loss *control* rows, so the
constraint costs no data efficiency by the selector's own score. That is
exactly what the paper claims for AAS, and it is the part I did not expect to
reproduce so cleanly.

## The result: the behaviour never happened, and the method paid for it anyway

Qwen3-1.7B, LoRA r16, 2 epochs, one L40S per arm. 200 held-out identity asks
and 500 held-out ordinary requests, base evaluated three times for the floor,
one seed per arm — so every arm-versus-arm verdict here is **`unresolved`**,
never `moved`.

| arm | identity ask, names the maker | ordinary request, no unasked maker |
|---|---|---|
| base, 3 passes | 0.005 / 0.000 / 0.000 (`run_std` 0.0029) | 1.000 / 1.000 / 1.000 (`run_std` 0.0000) |
| identity in the system prompt, no training | 0.440 **[+0.370, +0.510]** | 0.998 — **the only leak in the entire run** |
| `random` | 0.000, −0.005 [−0.015, +0.000] flat | 1.000, +0.000 |
| `loss` (baseline) | 0.920, **+0.915 [+0.875, +0.950]** | 1.000, +0.000 |
| `aas` (method) | 0.170, +0.165 [+0.115, +0.220] | 1.000, +0.000 |
| **method − baseline** | **−0.750 [−0.805, −0.690]** `moved_the_wrong_way` | **+0.000 [+0.000, +0.000]** no difference |

**Zero leaks in 500 ordinary requests, for the base and for all three trained
arms.** The behaviour this run set out to reduce did not occur once. The only
identity claim anywhere in 2,500 held-out replies came from the arm that was
*handed* its identity in a system prompt.

So the target's interval is `[+0.000, +0.000]`, and that is not evidence the
arms are equivalent — it is evidence that nothing varied. `compare()` said so
without being asked: it printed `CEILING: the before run already passes most
tasks; use harder situations`, and `holdout_size` reported that 500 paired
tasks at k=1 can prove a gain of about **+0.00**. The measurement layer
diagnosed the design before I did.

Meanwhile the capability that was supposed to *tie* moved 75 points. The
paper's claim is that selectors indistinguishable on task accuracy diverge on
behaviour. Here it inverted exactly: **the selectors were indistinguishable on
the behavioural axis and diverged enormously on task accuracy.** AAS's mixture
cap bought nothing on an axis with no headroom, and cost three quarters of the
identity answer.

### Half one: which half of the paper held

| claim | verdict |
|---|---|
| ADA — the selector shifts the attribute mixture, readable before any GPU | **held**: 17.8% vs 4.7% of the budget, 3.8x, at an equal token budget |
| AAS keeps data efficiency while constraining drift | **held**: mean selected loss 4.45 vs 4.38, the constrained selector is not the less efficient one |
| Selectors tie on task accuracy and diverge on behaviour | **did not hold here** — and the failure is structural, not a null result |

The reproduction and the application share a corpus, which is weaker than a
separate public task would be; that is a GPU-budget decision (three training
runs), and it is why the ADA half — which needs no GPU and *is* independent of
the training — carries most of the reproduction's weight.

### Why it inverted, and it is not "small model"

The paper's behavioural axes — refusal rate, verbosity, sycophancy — all sit
mid-range in a general instruction mix, with room to move in either direction.
The axis I took from production was a **guardrail already at its ceiling**. An
agent does not volunteer its maker unless something teaches it to, and 2 epochs
over 674–814 rows in which the identity answer only ever appears as an answer
to an identity question teaches the *conditional*, not the habit. The drift the
paper protects against needs a behaviour with headroom; mine had none, so the
protection was all cost.

That is the thing worth carrying to the next run: **ADA plus three base passes
would have said this before a GPU ran.** The identity share of the budget (free)
and the base leak rate (one eval pass) between them predict that `loss` will
teach identity, that `aas` will not, and that neither will change a rate already
at 1.000.

## Fresh traffic: the holdout said 0/500, thirty fresh conversations said 1

The winner by the operator's reading is `loss` — 0.920 on the identity ask, no
leaks — so it went to vLLM on my own Modal with `--enable-lora`, and 30 fresh
conversations were written at it by `wai.simulate(simulator=False)`, the
offline template writer, over HTTP.

```
I am Wai, made by While.          # the served adapter, /v1/chat/completions
fresh traffic at loss: 30 conversations, 1 leaked
pass@1 0.97 [0.90..1.00] (30 groups, k=1)
```

**The pinned holdout found zero leaks in 500 ordinary requests. Thirty fresh
conversations, written by a different generator, found one.** At n=30 that
does not resolve — the interval covers 1.00 — but it is the only evidence in
this run that the behaviour exists at all, and it came from the cheapest check
here. A 500-row set written once and frozen is a different distribution from
traffic, and the difference showed up on the first thirty rows.

`wai.select(mode="sft")` then kept **0 of 30** rows, calling 29 of them junk,
and told me why in a paragraph worth quoting: with one completion per prompt
there is no pick to make, only a pass/fail filter, so `top_per_prompt` and
`random_per_prompt` return the same rows and the random-selection control says
nothing (Lambert 2025, Rejection Sampling). That is a report refusing to
flatter its caller, and it is right.

## Cost

One L40S throughout, on my own Modal, no model API key anywhere.

| stage | GPU minutes |
|---|---|
| scoring pass over 2,496 rows | ~5 |
| three arms, in parallel | ~13 each, ~39 total |
| eval: base x3 + prompt-only + three arms, 4,900 generations | ~12 |
| serving + fresh traffic | ~8 |
| **total** | **~64 L40S-minutes, about $2.10** |

A week of this on every day's traffic is about **$15** at this size. The same
loop on Qwen3-4B with k=4 sampling is roughly 4x that, so **$60–80 a week** —
which is the number that matters, because the run above says the 1.7B result
is ceiling-limited rather than model-limited, and the honest next version is
the bigger one.

## What did not work

- **`wai.select` cannot express any of these selectors.** It selects by reward
  and difficulty band, which is the RL curation question. An online SFT
  selector ranks by loss, quality or diversity against a token budget, and
  `arm_selectors.py` is hand-written because of it. Filed as researcher
  feedback.
- **`compare()` has no `lower_is_better`.** Leak is a rate where down is the
  win, so the target is written as its positive form (`not leaked`) by hand.
  This is [#638](https://github.com/whilehq/whileai-sdk/issues/638) from a
  previous run, hit again from a different direction.
- **The first detector was English-only** and would have scored every
  non-English leak as clean, flattering whichever arm leaked in Japanese.
  Caught by writing the positive cases before the negative ones.
- **`selectors.py` shadows a stdlib module** that `subprocess` and `asyncio`
  import. Renamed to `arm_selectors.py` before it bit; worth knowing if you
  copy this layout.

## From paper to production, ranked

What the reproduction did not prepare me for, once the traces were real:

1. **The paper's behavioural axis has headroom; a production guardrail is at
   its ceiling.** Every axis in the paper (refusal, verbosity, sycophancy) can
   move both ways. The one the operator names — "stop doing Y" — is usually a
   rate you have already pushed near zero, and a method that protects it can
   only cost. Nothing in the paper, and nothing in the docs, tells you to
   measure the axis's headroom before choosing the method. `compare()` and
   `holdout_size` both said it afterwards, unprompted, which is the single best
   thing that happened today.
2. **The training set disagreed with the spec.** All 500 identity rows named a
   maker retired two days before this run. A reproduction's corpus is fixed and
   correct by definition; a production corpus drifts away from the spec it is
   supposed to encode, and reconciling them is a step with no call behind it —
   `prep` does it with a regex and reports a count.
3. **The judge had to be built and its precision measured, because the maker's
   name is an English word.** "Made by While" and "failed while running" differ
   by a capital letter and a verb. Four rounds, eight languages, and a
   validation set of 500 replies from an agent that never had the identity. A
   paper's metric is exact-match on a benchmark; this one is a detector whose
   error rate is part of the result.
4. **Contamination that no prompt-overlap rule can see.** 500 training asks and
   200 held-out asks are the same question in different words;
   `decontaminate()` dropped 0 of them and reported `0.0016`. The identity
   column is therefore a tie-check, not a generalisation measure, and the recipe
   says so rather than quoting 0.920 as if it were one.
5. **The deployed prompt is a control nobody runs.** Putting the identity in the
   system prompt gets 0.440 [+0.370, +0.510] for zero GPU — and produced the
   only leak in the whole run. That single pass reframes the whole experiment,
   costs nothing, and no page suggests it.
6. **Selection is not a library call.** `wai.select` selects by reward and
   difficulty band; the paper's selectors rank by loss against a token budget.
   Every line of `arm_selectors.py` is code the SDK could own.

## Run it

```bash
uv add whileai
cd recipes/community/identity-spec-no-unasked-maker-aas

python run.py --dry-run          # offline: no key, no GPU, no network
python run.py prep               # pool, holdouts, decontamination, detector precision
modal run --detach train_modal.py   # scoring pass + three arms, one L40S each
python run.py audit              # ADA table from the real losses
modal run --detach eval_modal.py    # base x3, prompt-only control, three arms
python run.py analyse            # intervals, noise floor, verdicts -> results.json

modal deploy serve_modal.py      # vLLM + both adapters, scale to zero
python fresh_traffic.py --url https://<your-workspace>--identity-aas-serve-serve.modal.run --model aas
modal app stop identity-aas-serve
```

| file | what it is |
|---|---|
| `spec.py` | the written identity and the programs that grade it |
| `arm_selectors.py` | the three selectors and the ADA audit |
| `run.py` | `prep`, `audit`, `analyse`, `--dry-run` |
| `train_modal.py` | one scoring pass, three arms |
| `eval_modal.py` | one entry point for every arm |
| `serve_modal.py`, `fresh_traffic.py` | the winner over HTTP, and 30 fresh conversations |
| `report_platform.py` | the run on the platform, with a `readback` that comes back empty |

## Next

The design fault here is fixable and the fix is cheap:

1. **Give the axis headroom.** Train an arm deliberately into the leak — an
   identity mix with the answer appearing in ordinary replies — and *then* ask
   whether AAS pulls it back. That is the paper's setting reconstructed
   honestly, and it is the falsification of this run's explanation.
2. **Take the behaviour from fresh traffic, not the frozen set.** The 1/30 is
   the only place the leak was ever seen. Three hundred fresh conversations,
   graded the same way, would say whether the real rate is 3% or 0.2% — and
   that number, not the holdout's 0/500, is what decides whether this behaviour
   is worth training at all.
3. **A second seed per arm**, which is all that stands between `unresolved` and
   a verdict on the −0.750.
4. The `loss` adapter is a genuinely good identity model (0.920, zero leaks in
   500) and it cost about 70 cents. If the behaviour turns out to be real on
   fresh traffic, it is the baseline to beat, not a straw arm.
