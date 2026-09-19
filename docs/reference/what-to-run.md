---
title: "What to run"
sidebarTitle: "What to run"
description: "Which mode fits the use case, how recommend() sizes a run from the agent's own grid, how mode=\"rl\" spends rollouts, and how traces aim the budget."
---

Depends on the use case. How each scenario is built is in [The recipe](/reference/overview#the-recipe).

The `simulate` blocks below write their situations on a hosted model, so they
need `WHILEAI_API_KEY` in the environment, or `whileai login`. Pass
`simulator=False` to run the same call offline with no key.

| You want | Mode | What happens |
|---|---|---|
| Many distinct situations | `explore` (default) | New situation every row |
| Same situation, different wording | `sft` | Multiple phrasings: tone, intent, personality |
| Same request, different agent behavior | `rl` | Up to k repeats of one phrasing, spent where the agent is inconsistent (see below) |
| A mix, until coverage plateaus | `adaptive` | New situations, phrasings, and repeats. Best with `until="saturation"` |

```python
wai.simulate(tools=my_tools, system_prompt=my_system_prompt)  # explore
wai.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="sft")
wai.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="rl")
wai.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="adaptive", until="saturation")
```

## How much to run

Ask before you guess. `recommend()` sizes the run from the agent's own covering grid and from published post-training practice (FireAct, LIMA, AgentTuning for SFT; DAPO, Skywork-OR1 for RL). No key, no network.

```python
rec = wai.recommend(tools=my_tools, system_prompt=my_system_prompt, mode="sft")
print("\n".join(rec["reasoning"]))
data = wai.simulate(tools=my_tools, system_prompt=my_system_prompt, **rec["simulate_kwargs"])
```

For the one-tool agent from [Start here](/reference/overview#start-here-no-key-required):

```text
covering grid: 72 cells for this agent
saturation wants 5 visits per cell = 360 rows
selection wants about 3x its target of 800 to choose from
generate 2400, select 800 diverse 1-labeled rows
time_budget off: a sized run stops on rows, not the clock
```

`target=` sets how many selected rows you want (default 800). `mode="rl"` assumes half the prompts produce a mixed group (`mixed_rate=0.5`). That rate is the agent's, not ours: probe 12 asks, grade, read `group_signal`, and pass the measured number back as `mixed_rate=`. A low rate means the grid is too easy for this agent; aim it with `traces=` before buying rollouts.

```text
covering grid: 72 cells for this agent
k=8 rollouts per ask; assumed mixed-group rate 50% (measure it with a 12-ask probe and group_signal, then recompute)
200 asks x 8 = 1600 rows to select about 800 mixed-group rows
a low measured rate means the grid is too easy for this agent: use traces=, harder cells, or a stricter judge prompt before buying more rollouts
in post-training terms the mixed rate is the pass@k - pass@1 headroom (group_signal reports both); pass@k near pass@1 means nothing to learn
```

### How `mode="rl"` spends rollouts

`mode="rl"` allocates rollouts successively. Every prompt is probed with two rollouts, the least that can show a split. A prompt whose rollouts disagree is filled to k, because that is the only place a grouped update has a gradient. A prompt that stays unanimous gets one more rollout only while the chance the next one differs beats what a fresh prompt offers per rollout. Both sides are measured on the run: the hazard is how often a group unanimous after n rollouts split on its next one (1/(n+2) only as the prior), the fresh side is the run's mixed rate over the probe. When nothing fresh can be opened, unanimous groups are resumed and finished.

Pass `grader=` and it runs beside the rollouts as they land, never in front of them; a prompt's decision waits for its verdict, so the allocation reads rewards, the signal a grouped update trains on. Without a grader it reads behavior signatures, which split more often than the judge does.

Near the end of a `time_budget` the run stops opening groups and finishes the rollouts in flight; a group still short of k at the whistle is stamped `group_cut`.

`data.search["groups"]` reports `mixed`, `stopped_unanimous`, `complete`, `partial`, `rollouts_saved`, the measured `mixed_rate` and the `hazard` per group size. `data.pass_at` scores stopped unanimous groups as unanimous. `repeat_policy="fixed"` restores k rollouts for every prompt (and then `search["groups"]` is not written); `advanced={"probe": n}` changes the probe. This is DAPO's dynamic sampling (arXiv 2503.14476) and difficulty filtering (rlhfbook.com/c/07-reasoning), applied at generation time.

## Close the loop: aim the budget with traces

This is the second half of "two ways in, one engine" on the [overview](/reference/overview). Without `traces=`, the coverage grid comes from the agent's tools and policy alone: a cold start that samples the whole space evenly. With `traces=`, the grid is aimed at the tools, faults and worlds the deployed agent actually got wrong, so new rows land where the agent is weak.

**A trace is a plain row dict.** Not an OTLP span, not a platform dataset id, not anything you have to ingest first. It is the same shape `grade()` returns and the same shape `simulate()` writes:

```python
traces = [
    {
        "prompt": "where is my order 4412",
        "steps": [
            {"tool": "get_order", "arguments": {"order_id": "4412"}, "result": {"error": "timeout"}}
        ],
        "final_text": "Your order shipped yesterday.",
        "reward": 0,
    },
    # ...
]
```

`prompt` (or `messages`) and `steps` are what matter. `reward` is optional (ungraded traces still focus the grid, they just carry less signal), and a JSONL path works anywhere a list does. `load_traces` normalizes the common variants (`tool_trace`/`trace` for `steps`, `final`/`output`/`response` for `final_text`, OpenAI-style `messages`), so exports from other stacks usually drop straight in. `rows_from_otel` turns an OTLP export into this shape on your machine; nothing has to be sent anywhere first.

```python
import whileai.simulations as wai

traces = wai.load_traces("production.jsonl")  # or just pass the list
print(wai.trace_report(traces, tools=TOOLS))  # what will this aim at?

data = wai.simulate(
    my_agent, tools=TOOLS, system_prompt=POLICY, traces=traces, mode="rl", repeats=4
)
```

For the one trace above, `trace_report` prints:

```text
{'traces': 1, 'dropped': 0, 'unique_prompts': 1, 'tools_observed': {'get_order': {'n': 1, 'fault_n': 1}}, 'faults_observed': {'error': 1}, 'world_states_observed': {}, 'distinct_behaviors': 1, 'graded': 1, 'passes': 0, 'fails': 1, 'ungraded': 0, 'advisory_labels': 0, 'emphasis': {'tool_condition': ['timeout']}}
```

| Call | What it does |
|---|---|
| `load_traces(source)` | Normalize a JSONL path or any iterable of dicts to the canonical trace schema. Rows with neither an ask nor steps are dropped |
| `trace_report(traces, tools=, policy=)` | Read this before you spend a budget: traces in, rows dropped, tools and faults observed, graded/ungraded counts, and `emphasis`, the exact axis values these traces move forward in the grid |
| `mine_traces(rows)` | The counts behind that: `flaw_rows` is every row with an observed fault or a 0 label, the behaviors worth simulating more of |
| `dimensions_from_traces(rows, tools, policy)` | The focused coverage axes themselves. `broaden=False` drops tools the traces never touched, so the budget stays near the flaws |
| `simulate_from_traces(traces, ...)` | `simulate(agent, traces=...)` for callers who start from the traces. With no agent, tools or policy it reads the tool surface off the traces, so graded telemetry alone is enough to start |
| `split_pseudo_production(rows, fraction=0.2)` | No production traces yet? Hold out a slice of a simulation run as stand-in production. Split by task, not by row, so the held-out slice is prompt-disjoint; every distinct flaw signature lands on the held-out side at least once |
| `leakage_report(generated, sources)` | Did any generated prompt come back a near copy of a source trace? Cosine similarity at `threshold=0.9`, exact matches always flagged |
| `drop_leaky_rows(rows, sources)` | The kept rows plus that report. Flagged rows are removed, not rewritten |

**The leakage rule.** Source traces shape the grid and never enter the generated dataset; `simulate(traces=...)` already drops generated rows that near-copy a source. `leakage_report` / `drop_leaky_rows` are how you verify it, which is what makes it safe to hold traces out for evaluation:

```python
prod, train = wai.split_pseudo_production(scored.rows, fraction=0.2)
data = wai.simulate(my_agent, tools=TOOLS, system_prompt=POLICY, traces=prod, mode="rl", repeats=4)
print(wai.leakage_report(data.trajectories, prod)["n_leaky"])  # want 0
rows, report = wai.drop_leaky_rows(data.trajectories, prod)
```

And the loop closes on itself: `evaluate(rollouts, judge).failed_traces()` hands the failures straight back to `simulate(traces=...)`.

**What traces can and cannot aim at.** Traces reproduce situations: the tools, faults and world states the deployed agent met. A failure that has a world-visible trigger (a tool timed out and the agent did not say so, a stale record was presented as current) is reproduced. A failure that lives in how the reply is worded (an unsupported claim, an estimate not labelled as one, two questions where one was asked for) has no trigger in the world, so traces alone cannot aim at it. Measured on a 12-rule grader, every rule with a tool-result trigger was reproduced and every rule about the reply's wording was not. For those, put the grader in the loop: with `simulate(..., grader=judge)` a row the grader fails is re-rolled and its ask mutated like a tool fault, and `data.search["mutation_aims"]` says how many parents and mutated rows each aim (`world_fault`, `graded_failure`) produced. The grader is the switch; to grade beside the loop and still steer by tool faults alone, pass `advanced={"mutate_graded_failures": False}`.

## Examples

In the order a post-training run happens. The index at [`recipes/README.md`](https://github.com/whilehq/whileai-sdk/blob/main/recipes/README.md) has one line per recipe with what it needs and how long it takes. "Offline" below means no key and no network.

| Step | Example | What it does |
|---|---|---|
| Simulate and grade | [`recipes/01-simulate/bring-your-own-agent`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/01-simulate/bring-your-own-agent) | Your own callable: the `agent(message) -> {steps, final_text}` contract, the `agent_failed` report when it raises or returns the wrong shape, and `eval_sourced` keeping a held-out score out of the reward. Offline. |
| Simulate and grade | [`recipes/01-simulate/verifiers`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/01-simulate/verifiers) | Verifiable rewards: math (`MathEqual`), an answer-and-format gate (`All`), code run against hidden tests (`CodeExec`), and a JSON-schema check, each feeding `grade`/`optimize`. Offline. |
| Measure | [`recipes/02-measure/pass-at-k`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/pass-at-k) | pass@1, pass^k and pass@k with their intervals for one agent, the per-ask histogram the mean hides, and what each number tells you to do next. Offline. |
| Measure | [`recipes/02-measure/eval-your-agent`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/eval-your-agent) | Evals for the agent you already have: the callable wrapper, the policy as a judge that reads the trajectory, pass@1 with an interval and pass^k per policy branch, the coverage warnings that catch a hollow run, and a CI gate. Two scripted refund bots, one careful and one eager, so the eval visibly separates them. Offline, seconds. How-to: [Evals](/evals). |
| Measure | [`recipes/02-measure/is-your-eval-any-good`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/is-your-eval-any-good) | Whether a number your eval produced means anything: ceiling, headroom, criteria that cannot fail, self-noise, the judge, contamination, and the three checks that void a base-vs-tuned comparison outright. Offline, seconds. |
| Measure | [`recipes/02-measure/compare-judges`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/compare-judges) | Which judge to trust: six judges on the same 300 labeled rollouts, ranked by agreement with the answer key, with intervals, kappa, leak rate and speed. `report` is offline; the live run needs a Jev, Claude or hosted key. |
| Measure | [`recipes/02-measure/reward-hacking`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/reward-hacking) | Reward hacking caught before, during and after training: the within-ask scan, the judge probes, the trajectory flags, and the proxy-vs-target verdict on a scripted agent and two judges. Offline, seconds, no key. How-to: [Reward hacking](/reward-hacking). |
| Measure | [`recipes/02-measure/safety-evals`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/safety-evals) | Safety evals for a tool-using agent: prompt injection (direct, and planted in a tool result), data exfiltration, secret leakage, unauthorized writes, plus the benign controls that catch over-refusal. Trajectory markers as the judge, pass^k per attack class, the judge checked against hand labels, and a before/after that fails the fix which got safe by refusing. Offline, seconds. How-to: [Safety evals](/safety-evals). |
| Measure | [`recipes/02-measure/safety-evals-marketplace`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/safety-evals-marketplace) | The same safety eval for a marketplace agent: the injection is planted in user-generated reviews, the private data is per tenant, one of the writes is a public post, and a flag needs a moderation ticket. Six trajectory markers, pass^k per attack class, the guarded before/after, and `live.py` to run the suite on a real model through Ollama with no key. Offline, seconds. |
| Select | [`recipes/03-select/schema`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/03-select/schema) | One row file in, six training targets out: eval, SFT, preference, GRPO prompts, OPSD hints, OPD. Migrates any legacy file first. Offline. |
| Select | [`recipes/03-select/prime-intellect-rl`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/03-select/prime-intellect-rl) | Generates a GRPO-ready dataset with `simulate(mode="rl")` and checks it carries gradient before you spend GPU time on it, then exports prompts in the `verifiers` shape. Needs an account key (`whileai login`), or `VLLM_API_KEY` for the shared pool. |
| Select | [`recipes/03-select/character`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/03-select/character) | Character training from a constitution: the OpenAI Model Spec's style traits become graded rows, preference pairs and SFT rows, with the judge checked against the spec's own labels and a before/after measurement. Offline by default. How-to: [Character training](/character-training). |
| Train | [`recipes/04-train/hosted-loop`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/hosted-loop) | Push graded rows, `wai.train` SFT on Qwen3-4B, `wai.serve` the adapter, one chat completion from the endpoint. One key, one A10G minute; the wiring check for training on the platform. |
| Train | [`recipes/04-train/identity`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/identity) | Builds a leak-free SFT set that teaches a model a new name and maker, with Modal scripts to train a LoRA and evaluate identity and leak rates. No model calls to generate. |
| Train | [`recipes/04-train/grpo`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/grpo) | GRPO on Modal, end to end: prompts from the simulator, a verifiable tool-discipline reward, TRL `GRPOTrainer` with LoRA, `HackMonitor`, reward and KL on the dashboard, pass@1 before and after on a holdout with the paired delta and per-category table on the run page. One A10G, under fifteen minutes. |
| Train | [`recipes/04-train/dpo`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/dpo) | DPO on the same environment: on-policy pairs from `build_preference_pairs`, TRL `DPOTrainer` with LoRA, the reward margin on the run page, iterated rounds with `--from-run`, constructed negatives where the policy never fails. One A10G, about ten minutes. |
| Train | [`recipes/04-train/text-to-sql`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/text-to-sql) | Hill-climb a model on a schema with a verifier as the reward: a seeded Postgres, 741 execution-checked tasks, `SQLExec`, benchmarks through `simulate(tasks=)`, GRPO rounds on Modal, every round measured on the same holdout. Needs Postgres and a key; Modal and an H100 to train. |
| Train | [`recipes/04-train/resist-planted-instruction`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/resist-planted-instruction) | A behaviour rubric decided by code, the criterion promoted into the reward on probe evidence, rejection sampling from the base itself, and a pre-registered random-selection control. Offline to read; a vLLM serving Qwen3-4B to generate, Modal to train. |
| Export | [`recipes/05-export/hugging-face`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/05-export/hugging-face) | Rows to a Hub dataset repo you own, any Hub split onto your account with a profile, a run's adapter to a model repo. Needs a key and a connected Hugging Face account. |

## Can this held-out set prove a gain?

Before you spend on training, ask the base run whether the held-out set can
show a difference at all:

```python
base = wai.evaluate(data, judge)  # the base run, graded, before any training
rep = wai.score.eval_power(base.rows())
print(rep)  # verdict usable / underpowered / saturated / floored, in_band, resolvable, n_needed
```

`usable` means the set has tasks in the band the policy sometimes solves and
enough of them to resolve the gain you are after; `underpowered` names
`n_needed`; `saturated` and `floored` mean the base already passes or fails
nearly everything, so train on a harder or easier set first.
