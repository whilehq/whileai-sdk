---
title: "While Simulations"
sidebarTitle: "Simulations"
description: "How the simulation engine thinks and why: the problem it solves, the situations it covers, and the rows it returns."
---

You give the SDK an agent's definition; it gives back graded conversations
you can train on. This page is how it thinks and why. The same engine with
estimators and references is [The engine](/engine).

<img className="block dark:hidden" src="/figures/simulations-pipeline-light.svg" alt="A description or traces feed six axes, a seeded world, the agent and your judge; graded rows become SFT, DPO, GRPO and an RL environment" />
<img className="hidden dark:block" src="/figures/simulations-pipeline-dark.svg" alt="A description or traces feed six axes, a seeded world, the agent and your judge; graded rows become SFT, DPO, GRPO and an RL environment" />

## The problem it solves

An agent's failures are specific: it hands off too early on one kind of
request, invents an order number under one kind of pressure. Fixing that in
the weights needs enough varied examples of the situation done right that
the model learns the behavior, not the example. Writing those by hand is the
expensive part. The simulator replaces the writing, not the judgment.

## Two ways in

**Describe the behavior.** One sentence starts it: "a personal finance
assistant that confirms before it moves money." The SDK drafts the tools,
builds a world, writes the people, runs the conversations.

**Point at the agent's traces.** Graded traces, OpenTelemetry spans
included, become a picture of which situations fail, which are new since the
last model version, and which stopped failing. That picture sets the
generation budget. Traces reproduce tools, faults and world states; a failure
that lives in the wording of a reply has no trigger in the world, so pass
`grader=` and the search mutates on graded failures too.

Both paths use the same engine.

## How the simulator thinks

**Situations are coordinates, not prompts.** A thousand user requests from a
model are a thousand variations of the same polite ask. The SDK declares six
axes (tool, policy rule, user stance, world state, tool condition, history)
and renders points in that space as a pairwise covering array: every pair of
axis values appears together at least once, because most real failures are
two things interacting. On a cold start nine in ten fault cells flip to
success, so the tool-condition axis is sampled, not covered, unless you raise
`fault_rate` or pass `prefer_success=False`. `data.coverage["pairwise"]`
holds `pairs_planned`, `pairs_covered` and `fraction`: a 64-row offline run
plans 381 pairs and covers 139, and that 0.36 is arithmetic, not a failed
eval. Policy coverage is a different question; `coverage_gap(asks, tools=...,
system_prompt=...)` answers it.

**People are sampled, not described.** A coordinate says the customer is in
a hurry and the order was already cancelled. A second layer decides how they
write: lowercase, clipped, sarcastic. The writer sees an aside in prose, not
the labels, because a model told to be terse writes an essay about being
terse. The same person shows up on turn five that showed up on turn one.

**The world answers honestly.** Tool calls go to a simulated world that is
deterministic per seed, returns records shaped like the tool's own schema,
remembers what it created, and says no. An unknown identifier is not found.
An argument that echoes the schema ("first name", user@example.com) is
refused with a hint. A world that never says no teaches an agent that never
expects it.

**Grading is the customer's authority.** Rows come back ungraded. The
deterministic conduct checks catch structural failures (an action claimed
without a tool call, an identifier the person never gave, success declared
after a failed call); then your grader decides what good means. Two things
are insisted on, because a judge is a reward model. The hosted grader (Phi-4)
is a different family from the hosted policy (Qwen), since a judge grading
its own writing prefers it. And every label says who made it: the hosted
grader stamps model, rubric hash and settings; a custom judge passes
`data.grade(judge=..., version=...)`. The judge is measured, not trusted:
hand-label a sample, `attach_labels(rows, labels, kind="human")`, and
`judge_agreement` reports agreement, kappa and `pass_when_gold_fail`, the
rate at which the judge passed a row you failed. A `gold_reward` column with
no author is reported as unmeasured.

**Failure is loud.** When the hosted writer fails, the offline template
writer takes over and `data.degraded` carries `generator_fallback`; a run
with no rows keeps the writer's last error in `data.search["writer_errors"]`.
A dataset that looks real and is not is worse than none.

## Which model runs it

Four roles take their own model: the agent (`agent=`), the situation writer
(`simulator=`), the simulated person (`user_model=`) and the judge (`spec=`
on `grade()`). All take the same spec.

| Spec | Backend | Key |
| --- | --- | --- |
| `ollama:<model>` | a local Ollama server | none |
| `vllm:<model>@<url>` | vLLM, or any OpenAI-compatible endpoint you serve | `VLLM_API_KEY` when the endpoint wants one |
| `openai:<model>` | OpenAI, or a compatible endpoint via `OPENAI_BASE_URL` | `OPENAI_API_KEY` |
| `anthropic:<model>` | the Claude Messages API | `ANTHROPIC_API_KEY`, or `WHILEAI_ANTHROPIC_API_KEY` to override it |
| `typesafe:<model>` | TypeSafe's Jev, a decision model; the judge only (`spec=`) | `TYPESAFE_API_KEY`, or `WHILEAI_TYPESAFE_API_KEY` to override it |

```python
import whileai.simulations as wai

data = wai.simulate(
    agent="anthropic:claude-haiku-4-5",
    tools=my_tools,
    system_prompt=my_system_prompt,
    simulator="anthropic:claude-sonnet-5",
    output="rollout.jsonl",
)
```

Omitting `agent=` runs the While-hosted model on your account key. One model
in two roles is the regime to avoid: `data.degraded` then carries
`same_model` and `warnings` names the call that separates them.

`spec="typesafe:jev-latest"` grades with a decision model: the verdict is a
probability, `failure_class` is the judge's own choice over the failure
vocabulary, and `judge_meta.confidence` within `DECISION_UNSURE_BAND` (0.1)
of even marks the row `unsure`. It cannot play the agent, writer or user.

## What you get

A JSONL file of chat-format conversations with tool schemas. Each row carries
its situation (axes, world state, scheduled faults), persona tags, and once
graded, its reward and reason. From there:

| cut | call | carries |
| --- | --- | --- |
| SFT | `data.training_set()` | a `loss_mask` per message: agent turns only (`mask_mode="final"` keeps the last); `export(format="trl")` drops it, because TRL reads none, and writes `mask_mode="final"` as prompt/completion rows TRL honors |
| preference pairs | `build_preference_pairs`, `export_preference` | raw scores, `margin`, `same_policy`, `length_delta`, a warning when chosen is usually longer |
| RL groups | `select_for_rl`, `export_dataset` | `group_id`, `k`, `n0`, `n1`, group reward mean and std, the `calibration` stamp |
| leakage check | `decontaminate` | same task id, same normalised text, 80 percent 8-gram cover (the Llama 2 rule), cosine at or above 0.85 with an `embedder=` |

The rows a run hands back (`scored.rows`, a slice of it, `passes()`, what
`decontaminate` kept) carry the run's system prompt and tool schemas, so
`select(rows).export(path)` writes both; a plain `list` carries neither,
so `export` takes `system_prompt=` and `tools=` and warns when a
tool-calling file would go out without its schema.

With `logprobs=True` every agent turn also carries the summed log-probability
of its tokens, their count, the per-token list when the backend returns one,
`policy_version` and the sampling settings: what an off-policy correction
and a KL to a reference model need.

## Hugging Face, both directions

Two routes to the Hub. The local one uses your own token and never calls
the platform: `export(..., push_to=)` uploads the file it just wrote, and
`wai.hub.push` uploads a file, an adapter directory or rows you already
hold. Repos are private until you say otherwise. Needs `HF_TOKEN` (or
`hf auth login`) and `pip install 'whileai[hf]'`.

```python
import whileai as wai

wai.export(rows, "train.jsonl", format="trl", push_to="me/my-set")  # -> a private dataset repo
wai.hub.push("out/adapter", "me/my-lora")  # a LoRA directory -> a private model repo
wai.hub.push(rows, "me/my-set", token="hf_...", private=False)  # rows -> train.jsonl, public
```

The platform route is a platform feature: it moves a set that already
lives on your account through the Hugging Face account connected on the
website, and brings any Hub split onto your account to be measured first.
Connect the account once on any dataset page. Needs `WHILEAI_API_KEY` or
`whileai login`.

```python
import whileai.simulations as wai


wai.hf_status()  # connected? namespaces
hf = wai.hf_publish("ds_...", repo="airline-refunds", wait=True)
hf["commit"], hf["tag"]  # one commit per push, tagged zp-<dataset id>
row = wai.import_hf("cornell-movie-review-data/rotten_tomatoes", split="test", purpose="eval")
wai.profile(row["datasetId"])  # rows, prompts, pass rate, support, mixed
wai.hf_publish_run("run_...", private=True)  # a finished run's LoRA adapter, as a model repo
```

One repo holds one split per purpose (`train`, `holdout`, `eval`). A push
replaces the old parts, the commit message carries the delta, `whileai.json`
keeps the history, and `load_dataset(repo, split, revision="zp-ds_...")`
loads exactly one push.

## Return shapes

`simulate` returns `SimulationData`; `evaluate` and `grade` return
`ScoredData`. `.rows` and `.rows()` both work; print `scored.warnings`
before any number. Field by field: [Evals](/evals#7-return-shapes).

## What it is not

Not ground truth. Every row is a simulation kept by a grader; review it as
you would a contractor's work. The world is not your database and the people
are not your customers. The value is coverage, variety, and honesty about
both.

## Where it goes next

The same simulator serves as a live environment for on-policy RL: the
trainer drives the policy, the SDK supplies situations, world, person and
reward. `export_environment(data, out, reward=...)` writes an installable
`verifiers` environment with train and holdout splits, the world dials in
`spec.json`, and the 20 to 80 percent difficulty band.

## What to run next

[`recipes/01-simulate/bring-your-own-agent`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/01-simulate/bring-your-own-agent)
is the shortest version of the loop above, offline and in seconds;
[`recipes/03-select/prime-intellect-rl`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/03-select/prime-intellect-rl)
is the `verifiers` export, and needs a key.
