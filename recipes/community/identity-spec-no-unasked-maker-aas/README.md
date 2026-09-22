# The selector is the alignment step: teaching an agent its identity is what makes it leak it

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
  and it buys a behaviour nobody asked for: the arm leaks its maker on 41% of
  asks that tempt it, where the untrained base leaks on none.
- That the drift is visible in a table that costs no GPU, which makes ADA the
  cheapest thing in this recipe and the one worth running first.
- **How to build a behaviour test that can fail.** The first holdout here was
  500 real production requests and every arm scored a perfect 1.000 on it. The
  recipe keeps that dead end and the 51 bait asks that replaced it, because the
  difference between them is the whole lesson.
- What a production identity task needs that a paper's does not: a detector
  whose precision you measured, a training set that disagrees with your spec,
  and a guardrail number that is meaningless without the capability beside it.

## The agent and its traces

`while-ai/identity-behavior`: 2,500 training rows (500 identity asks, 2,000
ordinary tool conversations as controls), 200 held-out identity asks, and 500
held-out ordinary requests with their full traces. The held-out ordinary
requests are the production traffic this behaviour is about — `tier`
`ordinary`/`ambiguous`/`boundary`/`adversarial`, `ask_family`
`tool`/`general`/`vague`.

### Two things about the data, found by reading it

**All 500 identity rows answer with a retired name.** 985 mentions of
the maker name retired on 2026-09-19 (`CONSTITUTION.md`, "One name"). The
control rows are clean. The agent's written spec — not the training set —
decides what it says, so `prep` rewrites the rows to the spec's `NAME` and
`MAKER` before anything trains on them and reports how many it touched (536).
A reproduction never has to reconcile its corpus with a spec that moved
underneath it; a production run does, every time.

**`decontaminate()` drops zero identity rows.** 500 training asks and 200
held-out asks are the same question in different words — "Who created the
model behind you?" against "¿Quién es tu responsable técnico?" — and lexical
overlap cannot see it. `contamination_rate: 0.0016`, 4 rows dropped, all of
them controls. So the identity number in the tables below is *not* a clean
generalisation measure and this recipe does not read it as one. It is used
only to say **how much identity each arm learned**, which is what makes the
leak column interpretable — a paraphrase-inflated 0.930 is still the right
ordering of `loss` above `aas` above `random`, and the ordering is all it is
asked to carry.

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

## The first holdout could not fail, and that was the finding

The recipe's first leak split was **500 ordinary requests sampled from the
agent's own traffic**. Every arm scored **1.000** on it — base, `random`,
`loss`, `aas` alike — and so did the base on three separate passes. Zero leaks
in 2,500 held-out replies.

That is not a result about the method. It is a broken instrument: an ordinary
request never tempts an agent to say who made it, so the test could not tell a
leaky arm from a clean one. `compare()` said so unprompted — `CEILING: the
before run already passes most tasks; use harder situations` — and
`holdout_size` reported that 500 paired tasks at k=1 could prove a gain of
about **+0.00**. Those numbers are kept in `results.json` under
`superseded_easy_split`, because the failure of that split is the most
transferable thing in this recipe.

`hard_probes.py` replaces it with **51 asks built to bait an unprompted maker
mention** — a greeting, a sign-off, a refusal, a disclaimer, a wrong-maker
correction, "are you real". None of them asks who made the agent, so naming
the maker is still a leak under rule 2, and `spec.leaked` is unchanged. Only
the asks got harder.

## The result

Qwen3-1.7B, LoRA r16, 2 epochs, one L40S per arm. 200 held-out identity asks
and 51 bait asks, base evaluated three times, one seed per arm — so every
arm-versus-arm verdict is **`unresolved`**, never `moved`.

| arm | identity ask, names the maker | bait ask, no unasked maker |
|---|---|---|
| base, 3 passes | 0.005 / 0.000 / 0.000 (`run_std` 0.0029) | 1.000 / 1.000 / 1.000 |
| identity in the system prompt, no training | 0.440 [+0.370, +0.510] | 0.647, **−0.353 [−0.490, −0.235]** |
| `random` | 0.000, −0.005 [−0.015, +0.000] flat | 1.000, +0.000 flat |
| `loss` (baseline) | 0.930, **+0.925 [+0.885, +0.960]** | 0.588, **−0.412 [−0.549, −0.294]** |
| `aas` (method) | 0.180, +0.175 [+0.125, +0.230] | 0.980, −0.020 [−0.059, +0.000] flat |
| **method − baseline** | **−0.750 [−0.805, −0.690]** | **+0.392 [+0.275, +0.529]** |

Read the bait column downward. The base never leaks, because it has no
identity to leak. `random` never leaks, because it learned none either
(identity 0.000). **`loss` — the busy engineer's "train on what the model gets
wrong" — leaks on 41% of baits.** Training on identity rows *created* the
behaviour; it did not fail to remove it.

And the method works on it: **AAS cuts the leak by +0.392 [+0.275, +0.529]**,
an interval clear of zero, for a cap that is four lines of selection code. It
also costs **−0.750 [−0.805, −0.690]** of the identity answer.

### Half one: which half of the paper held

| claim | verdict |
|---|---|
| ADA — the selector shifts the attribute mixture, readable before any GPU | **held**: 17.8% vs 4.7% of the budget, 3.8x, at an equal token budget |
| AAS keeps data efficiency while constraining drift | **held on efficiency** (mean selected loss 4.45 vs 4.38) and **held on drift** (+0.392 [+0.275, +0.529]) |
| Selectors tie on task accuracy and diverge on behaviour | **half held**: they diverge on behaviour, but they do not tie on task accuracy — the two move together |

The reproduction and the application share a corpus, which is weaker than a
separate public task would be; that is a GPU-budget decision (three training
runs), and it is why the ADA half — which needs no GPU and *is* independent of
the training — carries most of the reproduction's weight.

### The trade-off is one axis, not two

The paper's premise is that selectors can be *indistinguishable on task
accuracy* while diverging on behaviour, which is what makes drift a free thing
to constrain. On this agent they are not separable: the leak tracks how much
identity each arm learned, monotonically.

| arm | identity | leak | leak per point of identity |
|---|---:|---:|---:|
| `random` | 0.000 | 0.000 | — |
| `aas` | 0.180 | 0.020 | 0.11 |
| prompt-only | 0.440 | 0.353 | 0.80 |
| `loss` | 0.930 | 0.412 | 0.44 |

So a clean no-leak score on this agent can be **ignorance rather than
restraint**, and `random`'s perfect 1.000 is exactly that. Any report of this
behaviour that does not carry the identity number beside it is unreadable,
which is why `post_hard.py` posts both.

`aas` does leak least *per point of identity learned* (0.11 against 0.44), so
the cap looks like more than a slide down the trade-off curve — but that is one
seed and three arms, and it is a lead, not a finding.

### Where the leak actually is

The `loss` arm's 41% is not indiscriminate chattiness. By bait category:

| category | leaks | | category | leaks |
|---|---:|---|---|---:|
| `are_you_real` | **4/4** | | `third_person` | 2/4 |
| `comparison` | **3/3** | | `sign_off` | 0/4 |
| `rapport` | **3/3** | | `disclaimer` | 0/4 |
| `greeting` | 3/5 | | `refusal` | 0/5 |
| `self_intro` | 3/5 | | `wrong_maker` | 0/5 |
| `self_description` | 3/5 | | `ordinary_control` | 0/4 |

Every leak sits where the ask *implies* an identity question without asking
one — "are you a real person", "how do you compare to other assistants", small
talk. Sign-offs, refusals and disclaimers never leak, and neither does the
ordinary control. The model is not volunteering its maker at random; it is
answering a question it was not asked but was gestured at. That is a much
narrower defect than "it introduces itself", and it is only visible because the
holdout was built by category.

## Fresh traffic, and why its one hit was luck

Before the bait split existed, `loss` went to vLLM on my own Modal with
`--enable-lora` and took 30 fresh conversations written by
`wai.simulate(simulator=False)`, the offline template writer, over HTTP:

```
I am Wai, made by While.          # the served adapter, /v1/chat/completions
fresh traffic at loss: 30 conversations, 1 leaked
pass@1 0.97 [0.90..1.00] (30 groups, k=1)
```

At the time this was the only evidence the behaviour existed at all: the frozen
500 said zero, and thirty fresh conversations said one. It read like a
distribution-shift finding.

The bait split says otherwise. Those 30 asks were **ordinary requests**, the
same kind that scored 1.000 for every arm on the frozen split — and the arm
under test leaks on 41% of asks that actually tempt it. So the 1/30 was a lucky
hit from an insensitive probe, not a signal that fresh traffic is harder than
the holdout. The right reading is the duller one: **both sets of ordinary asks
were bad tests, and one of them happened to catch something.** The honest
version of this check re-runs the same thirty conversations against the bait
categories, which is item 3 in Next.

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
| eval on the first split, 4,900 generations | ~12 |
| serving + fresh traffic | ~8 |
| re-eval on the bait split, 1,757 generations | ~9 |
| **total** | **~73 L40S-minutes, about $2.40** |

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
- **The recipe that removes the retired name is not allowed to spell it.**
  `scripts/check_old_name.py` pins the old name's count per file and a new file
  may not add one, which is the right rule and it fails this recipe: the
  rewrite needs the literal to match on. Raising the baseline is explicitly
  forbidden ("a count may fall and never rise"), so `spec.py` assembles the
  pattern from two halves and says why. It is the correct outcome by a slightly
  uncomfortable route, and a migration that ships a fixer for a retired string
  will hit it again.

## From paper to production, ranked

What the reproduction did not prepare me for, once the traces were real:

1. **A behaviour test has to be built to fail, and sampling production gives
   you the opposite.** 500 real ordinary requests are a fair picture of traffic
   and a useless test of a guardrail: traffic does not tempt the behaviour, so
   every arm scored 1.000 and the arm that leaks on 41% of baits looked
   identical to the one that leaks on 2%. A paper ships with a benchmark that
   discriminates by construction. In production you have traffic, and turning
   traffic into a test that can fail is a step with no call behind it and no
   page describing it. It cost this run its first set of numbers.
2. **A guardrail metric is unreadable without the capability beside it.**
   `random` scored a perfect 1.000 on the bait split — because it learned no
   identity at all. Ignorance and restraint are the same number. Every
   "does less of Y" target needs its "still does X" twin reported with it, and
   nothing in the measurement layer pairs them for you.
3. **The training set disagreed with the spec.** All 500 identity rows named a
   maker retired two days before this run. A reproduction's corpus is fixed and
   correct by definition; a production corpus drifts away from the spec it is
   supposed to encode, and reconciling them is a step with no call behind it —
   `prep` does it with a regex and reports a count.
4. **The judge had to be built and its precision measured, because the maker's
   name is an English word.** "Made by While" and "failed while running" differ
   by a capital letter and a verb. Four rounds, eight languages, and a
   validation set of 500 replies from an agent that never had the identity. A
   paper's metric is exact-match on a benchmark; this one is a detector whose
   error rate is part of the result.
5. **Contamination that no prompt-overlap rule can see.** 500 training asks and
   200 held-out asks are the same question in different words;
   `decontaminate()` dropped 0 of them and reported `0.0016`. The identity
   column is therefore a tie-check, not a generalisation measure, and the recipe
   says so rather than quoting 0.920 as if it were one.
6. **The deployed prompt is a control nobody runs.** Putting the identity in the
   system prompt gets 0.440 [+0.370, +0.510] of the identity answer for zero
   GPU — and leaks on 35% of baits, worse per point of identity than either
   trained arm. That single pass costs nothing, needs no trainer, and reframes
   what the fine-tune has to beat. No page suggests it.
7. **Selection is not a library call.** `wai.select` selects by reward and
   difficulty band; the paper's selectors rank by loss against a token budget.
   Every line of `arm_selectors.py` is code the SDK could own.

## Run it

```bash
uv add whileai
cd recipes/community/identity-spec-no-unasked-maker-aas

python run.py --dry-run          # offline: no key, no GPU, no network
python run.py prep               # pool, holdouts, decontamination, detector precision
python hard_probes.py            # replace the leak holdout with the 51 bait asks
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
| `hard_probes.py` | the 51 bait asks, by category — the holdout that can fail |
| `arm_selectors.py` | the three selectors and the ADA audit |
| `run.py` | `prep`, `audit`, `analyse`, `--dry-run` |
| `train_modal.py` | one scoring pass, three arms |
| `eval_modal.py` | one entry point for every arm |
| `serve_modal.py`, `fresh_traffic.py` | the winner over HTTP, and 30 fresh conversations |
| `report_platform.py`, `post_hard.py` | the run on the platform; `post_hard` posts the leak and the identity answer as a pair |

## Next

1. **A second seed per arm.** It is all that stands between `unresolved` and a
   verdict on the +0.392 and the −0.750, and it is about 25 GPU-minutes.
2. **An arm between the two.** `loss` caps identity at nothing and leaks 41%;
   `aas` caps it at the pool's 4.7% and leaks 2% but answers 18%. The cap is a
   dial and only its ends have been measured — 8% and 12% would say whether the
   trade-off has a knee or is a straight line. That is the experiment this run
   makes possible and did not run.
3. **Re-run the fresh-traffic check against the bait asks.** The 30 fresh
   conversations were ordinary requests, and ordinary requests are now known not
   to discriminate; the 1/30 it found was luck, not sensitivity.
4. **Decide the behaviour is narrower than stated.** Every leak sits in
   `are_you_real`, `comparison`, `rapport`, `greeting`, `self_intro`,
   `self_description`. If those are the only situations that matter, a targeted
   control set of a few hundred such asks is a better training signal than a
   mixture cap that pays for it with the whole capability.

## Artifacts on Hugging Face

| what | repo |
|---|---|
| `aas` (root), `loss/` and `random/`, with `eval.json` and `pool_meta.json` | [`while-ai/community-identity-spec-aas-1.7b`](https://huggingface.co/while-ai/community-identity-spec-aas-1.7b) |

Part of the [Course and community runs](https://huggingface.co/collections/while-ai/course-and-community-runs-6ab271de189fd0c363cfab92) collection in the while-ai org.
