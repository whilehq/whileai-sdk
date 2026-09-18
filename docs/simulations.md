# While Simulations

The SDK makes post-training data for an agent you already have, or one you
can describe. You give it the agent's definition; it gives you graded
conversations you can train on. This document is about how it thinks and
why, not a tour of every option.
The same engine on one page, with the estimators and the references, is
[engine.md](engine.md).

## The problem it solves

An agent's failures are specific. It hands off too early on one kind of
request, invents an order number under a particular kind of pressure,
loses its manners on the fourth turn. Fixing that with a system prompt
works until it does not. Fixing it in the weights needs examples of the
situation done right, and enough of them, with enough variety, that the
model learns the behavior rather than the example. Writing those by hand
is the expensive part of every post-training pipeline. The simulator
replaces the writing, not the judgment.

## Two ways in

**Describe the behavior.** One sentence is enough to start: "a personal
finance assistant that confirms before it moves money." The SDK drafts the
tools such an agent would have, builds a world around them, writes the
people who would talk to it, and runs the conversations. This is the
cold-start path for an agent that does not exist yet, or a behavior you
want to add to one that does.

**Point at the agent's traces.** If the agent is in production, its
telemetry already says where it is weak. The SDK reads graded traces,
plain OpenTelemetry spans included, into a picture of which situations
fail, which are new since the last model version, and which have stopped
failing. That picture sets the generation budget, so new rows land where
the deployed agent actually needs them and not where it is already fine.
Traces reproduce situations: the tools, faults and world states the
agent met. A failure that lives in how the reply is worded (an
unsupported claim, an estimate not labelled as one) has no trigger in
the world, so traces alone cannot aim at it; pass `grader=` and the
search mutates on graded failures as well as on tool faults.

Both paths use the same engine. The first one is what produced the
training set behind our first fine-tune, from nothing but the agent's
tool list and policy.

## How the simulator thinks

**Situations are coordinates, not prompts.** Asking a model for a thousand
user requests gives you a thousand variations of the same polite,
well-specified ask. The SDK instead declares axes (which tool, which
policy clause, what the world looks like, what condition the tool is in,
what stance the person takes, what has already happened) and renders
points in that space. The planned grid is a pairwise covering array:
every pair of axis values appears together in at least one planned
cell, which is the coverage strength the testing literature settled on
because most real failures come from two things interacting. On a cold
start the engine then flips most fault cells to success (one cell per
fault type stays), so the tool-condition axis is sampled, not covered,
unless you raise `fault_rate` or pass `prefer_success=False`. What the
run actually touched is a number: `data.coverage["pairwise"]` is
planned pairs, covered pairs, and the fraction. Read it as what it
counts: pairwise cells of the 6-axis grid, which is training-data
coverage, not policy coverage. A 200-row run over a grid with thousands
of pairs reports a small fraction and that is arithmetic, not a failed
eval. Whether your policy is covered is a different question, and
`coverage_gap(asks, tools=..., system_prompt=...)` answers it.

**People are sampled, not described.** A coordinate says the customer is
in a hurry and their order was already cancelled. A second layer decides
how that person writes: lowercase, clipped, run-on, with a typo, polite,
sarcastic. The writer never sees those labels; it sees an aside in prose,
because a model told to be terse writes an essay about being terse. The
same person shows up on turn five that showed up on turn one.

**The world answers honestly.** Tool calls go to a simulated world that is
deterministic for a given seed, returns records shaped like the tool's own
schema, remembers what it created, and says no. An unknown identifier is
not found. An argument that echoes the schema instead of the person's
details ("first name", user@example.com) is refused with a hint. A world
that never says no teaches an agent that never expects it; we learned
that the expensive way and built it in.

**Grading is the customer's authority.** Rows come back ungraded on
purpose. The deterministic conduct checks catch structural failures (an
action claimed without a tool call, an identifier the person never gave,
success declared after a failed call), and then your grader, a function
you write, decides what good means for your agent. The SDK's job is to
make every row worth grading; it does not get a vote on what passes.
It does insist on two things about whichever judge you use, because a
judge is a reward model. The hosted grader is a different model family
from the hosted policy, because a judge grading its own writing prefers
it. Every label says who made it: the hosted grader
stamps its model, rubric hash, and settings on the row, and a custom
judge can pass `version=` to do the same, so a rubric edit is visible as
a new judge rather than a silent drift. And the judge is measured, not
trusted: hand-label a sample into `gold_reward` and `judge_agreement`
reports agreement, kappa, and the rate at which the judge passed a row
you failed, which is the number that decides whether training on its
labels teaches the behavior or the judge's blind spot.

**Failure is loud.** If the writer, the world, or a judge cannot do its
job, the run says so. Template fallbacks are never quietly substituted for
model-written situations, because a dataset that looks real and is not is
worse than no dataset.

## Which model runs it

Four roles can each take their own model: the agent (`agent=`), the
situation writer (`simulator=`), the simulated person (`user_model=`) and
the judge (`spec=`). Each one takes the same backend spec.

| Spec | Backend | Key |
| --- | --- | --- |
| `ollama:<model>` | a local Ollama server | none |
| `vllm:<model>@<url>` | vLLM, or any OpenAI-compatible endpoint you serve | `VLLM_API_KEY` when the endpoint wants one |
| `openai:<model>` | OpenAI, or a compatible endpoint via `OPENAI_BASE_URL` | `OPENAI_API_KEY` |
| `anthropic:<model>` | the Claude Messages API | `ANTHROPIC_API_KEY`, or `WHILEAI_ANTHROPIC_API_KEY` to override it |

```python
data = wai.simulate(
    agent="anthropic:claude-haiku-4-5",
    tools=my_tools,
    system_prompt=my_system_prompt,
    simulator="anthropic:claude-sonnet-5",
    output="rollout.jsonl",
)
```

Omitting `agent=` runs the While-hosted model on your account key instead.
One model in more than one role is the regime to avoid, and the run says so
on `data.degraded`: a judge grading its own writing prefers it.

## What you get

A JSONL file of conversations in chat format, with tool schemas, each
row carrying its situation (which axes, which world state, which faults
were scheduled), its persona tags, and, once graded, its reward and the
reason. From there: `training_set()` for supervised fine-tuning,
preference pairs and repeated groups for reinforcement learning, and a
leakage check against any evaluation you care about. Each export carries
what the training recipe needs and a reviewer would ask for. Supervised
rows carry a `loss_mask`, one flag per message, so the trainer learns the
agent's turns and never the tool output or the user (`mask_mode="final"`
keeps only the last agent turn). Preference pairs carry the raw scores
and their margin, which model produced each side and whether the two
match, and the length gap between chosen and rejected, with a warning
when the chosen side is usually the longer one, because a preference
trainer learns length before it learns behavior. RL groups carry
`group_id`, the group size, and the 0/1 counts, plus the calibration
stamp the publish gate writes. With `logprobs=True` every agent turn also
carries the summed log-probability of the tokens the policy generated and
their count, which is what a later update needs to correct for being
off-policy and what a KL to a reference model is computed from. The default check
is lexical: word and bigram overlap with numbers collapsed, so it drops
near-copies and copies that differ only in an id, and it does not catch
a paraphrase. Pass a semantic `embedder=` to the leakage functions when
that matters.

## Hugging Face, both directions

A graded set can leave for a Hugging Face dataset repo you own, and any
Hub split can come onto your account to be measured before you train on
it. Connect the account once on any dataset page; the platform holds the
token, the SDK never sees it.

```python
wai.hf_status()  # connected? namespaces
hf = wai.hf_publish("ds_...", repo="airline-refunds", wait=True)
hf["commit"], hf["tag"]  # one commit per push, tagged zp-<dataset id>
row = wai.import_hf("cornell-movie-review-data/rotten_tomatoes", split="test", purpose="eval")
wai.profile(row["datasetId"])  # rows, prompts, pass rate, support, mixed
wai.hf_publish_run("run_...", private=True)  # a finished run's LoRA adapter, as a model repo
```

One repo holds one split per purpose (`train`, `holdout`, `eval`), so the
train set and its held-out sibling land in the same place. Pushing a new
cut into a split replaces the old parts, the commit message carries the
delta (rows, pass rate, support), and `whileai.json` in the repo keeps
the history: which While dataset each split came from, and what it
replaced. `load_dataset(repo, split, revision="zp-ds_...")` loads exactly
one push.

## Return shapes

One table, because these cost testers a round trip each:

| call | you get | read it as |
| --- | --- | --- |
| `simulate(...)` | `SimulationData` | `data.rows` and `data.rows()` both work |
| `evaluate(...)`, `grade(...)` | `ScoredData` | `scored.rows` is a **list**; `scored.rows()` is a `TypeError` |
| | | `scored.warnings`: hollow-run notes, print them before any number |
| `pass_at(rows)` | `PassAt` | `pass_at_1`, `pass_pow_k` (printed `pass^k`, not `pass_hat_k`), `pass_at_k`, `headroom`, `ci95` |
| `marker_summary(rows)` | `{marker: stats}` | `mean`, `ci95` (not `ci`), `n_tasks`, `n_rows` (not `n`), `note` or `warning` |
| `judge_trust(rows)` | `dict` | `ok`, `agreement.agreement`, `agreement.ci95`, `gold_kind`, `warnings` |

The full field-by-field version, including which fields print and which
do not, is in [evals.md](evals.md#7-return-shapes).

## What it is not

It is not ground truth. Every row is a simulation, kept by a grader, and
should be reviewed the way you would review a contractor's work. The world
is not your database. The people are drawn from a persona distribution,
not from your customers. The value is coverage, variety, and honesty
about all three.

## Where it goes next

The same simulator that produces a frozen dataset can serve as a live
environment for on-policy reinforcement learning: the trainer drives the
policy, and the SDK supplies the situations, the world, the person, and
the reward. That export is designed and follows once the supervised loop
is closed end to end.
