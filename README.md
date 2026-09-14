# zeroproof

The ZeroProof Python SDK. One package, two importable modules:

- `zeroproof`: the platform client. OTLP trace ingest and trace-dataset listing against the token gate.
- `zeroproof.simulations`: post-training data for an agent. (Was the separate top-level package `zeroproof_simulations`; that name still imports for two releases with a deprecation warning.) Give it the agent's traces, or its tools and system prompt; it simulates the situations, the people, and the world, plays the agent through multi-turn tool-calling conversations, and returns rows for your grader.

This repo absorbed the `zeroproof-simulations` package; `zeroproof-simulations` on PyPI is deprecated in favor of `zeroproof`.

Releases of `zeroproof` before 0.3 were an unrelated encrypted agent-to-agent messaging client. That code was removed in 0.04; pin `zeroproof<0.3` if you still depend on it.

Two ways in, one engine. Give it the agent's tools and system prompt and it samples situations across everything that agent can be asked. Give it graded traces as well and it aims the budget at the situations that fail in production, so new rows land where the agent is weak and carry both the failure and the fixed version. Every row is a full conversation: user turns, agent turns, tool calls, tool results, scheduled faults. Rows come back ungraded; your grader decides what good means. Default `explore`: one unique situation per row. How it thinks: [docs/simulations.md](docs/simulations.md).

## How a row gets made

![How a row gets made: the draw, the coverage grid, the search arms, the rollout, the split](docs/how-a-row-gets-made.svg)

A situation is drawn across the world axes (from the agent's tools) and the human axes (from a separate writer). It fills a cell in the coverage grid, nudges the five search arms, and the agent plays it against a world that breaks on schedule. The row that comes out splits into `Task`, `Rollout`, `Judgment`, and `Marker`, and every training target is a projection of some of those four. The interactive version, running on real rows, is at [zeroproofai.com/docs/engine](https://zeroproofai.com/docs/engine).

## Overview

`simulate()` is a pipeline.

1. **Read the agent.** Tools and system prompt. That is the spec of the world.
2. **Build a fake world from those tools.** Objects, plausible results, and faults (timeout, deny, junk).
3. **Write users.** A separate writer (same hosted model, different prompt, no agent policy) samples situations across tools, stance, history, and so on.
4. **Pick the diverse ones.** Embeddings plus a bit of noise so the batch is not 200 copies of the same prompt.
5. **Play the agent.** It talks, calls tools, gets results, talks again. All of that is stored: user text, agent text, tool calls, tool results, `final_text`.
6. **Grade.** Rows come back ungraded. Grade after with `data.grade()` (hosted judge), `data.grade(judge=...)` (your judge), or `zps.grade(path)`. The legacy `grade=True` flag writes deterministic conduct rewards; avoid it for the rubric workflow.

Stop when the row cap or the clock hits.

## How to use

```bash
pip install zeroproof   # or: uv add zeroproof
```

Bring your own model. Any OpenAI-compatible chat endpoint that returns tool
calls works; it writes the situations and plays the agent, so both run on
your key:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # only for a non-OpenAI endpoint
```

```python
import zeroproof.simulations as zps

data = zps.simulate(
    agent="openai:gpt-4.1-mini",
    tools=my_tools,
    system_prompt=my_system_prompt,
    output="rollout.jsonl",
)
```

## Five calls

Spec to gated dataset. Everything else in this README is one layer down.

```python
import zeroproof.simulations as zps

data = zps.simulate(
    agent="openai:gpt-4.1-mini", spec="specs/github", mode="rl", situations=200, repeats=8
)  # 1 generate
scored = data.grade(judge=my_judge)  # 2 grade (0/1 per rollout)
print(scored.pass_at)
zps.judge_trust(scored.rows, judge=my_judge)  # 3 trust the numbers
rows, report = zps.optimize(scored, mode="rl")  # 4 prune to what carries gradient
entry = zps.push_rows(rows, "github-rl-v1", gate=True, mode="rl")  # 5 publish, gated
```

After training, measure whether it landed: `zps.delta_report(before=scored.rows, after=after_rows, target="pass_at_1")`.

Character training, the same loop aimed at how the model talks: a constitution in, graded replies, length-matched pairs and SFT rows out, and the judge checked against the constitution's own labels. Worked example [`examples/character`](examples/character), recipe [docs/character-training.md](docs/character-training.md), page [zeroproofai.com/docs/character-training](https://zeroproofai.com/docs/character-training).

| Call | What it decides | Reads |
|---|---|---|
| `simulate` | the situations, the users, the world, k rollouts per ask | your spec or tools + system prompt |
| `data.grade(judge=)` | 0/1 per rollout. `zps.grade(data)` uses the hosted judge instead | your judge callable, or a `VLLM_API_KEY` |
| `pass_at` / `judge_trust` | pass@1 with an interval, headroom for RL, whether the judge can be trusted | graded rows, 30 to 100 hand labels as `gold_reward` |
| `optimize(mode="rl")` | drops junk, duplicates, dead groups, out-of-band asks; flags reward hacks | graded rows |
| `push_rows(gate=True)` | refuses ungraded or gradient-free RL data; stamps calibration | pruned rows |

### Verifiers: when the reward is a program, not a judge

For a verifiable task the reward is a checker, not an opinion (RLHF book ch. 7, 13). `zeroproof.simulations.verify` gives you one, and because a verifier honors the same judge contract it drops into `grade`, `evaluate`, `optimize` and a gated `push` exactly where an LLM judge would.

```python
from zeroproof.simulations.verify import MathEqual, CodeExec, JSONSchema, Regex, All

data = zps.simulate(spec="specs/math", mode="rl", situations=200, repeats=8)
scored = data.grade(judge=MathEqual())  # the verifier is the reward
rows, _ = zps.optimize(scored, mode="rl")  # GRPO data, gradient checked
```

The candidate is the rollout's `final_text`; the gold is read from the row's `privileged.reference`, which the training export never projects, so the answer key cannot leak into a training file (flat `answer`/`target`/... fields work too, or point at any column with `field=`). Built in: `ExactMatch`, `Includes`, `Regex`, `MultipleChoice`, `Numeric`, `MathEqual`, `JSONValid`, `JSONSchema`, `JSONField`, and `CodeExec` (runs the candidate against hidden tests in a sandboxed subprocess with a timeout). Compose with `All` (right answer *and* right format), `Any`, or a graded `Weighted` rubric; wrap your own with `@verifier`. Worked example: [`examples/verifiers`](examples/verifiers).

Or use ZeroProof-hosted Qwen, which is the default when no `agent=` is given.
Ask us for a `VLLM_API_KEY`; the endpoint is shared and rate limited.

```bash
export VLLM_API_KEY=...
```

No key at all: the situation writer also defaults to hosted Qwen, even when
`agent=` is your own function. Pass `simulator=False` to use the built-in
template writer instead. It needs no model and runs in seconds; the
situations are less varied than a model writes, so it is for wiring up your
agent and grader, not for a training set.

```python
data = zps.simulate(
    my_agent, tools=my_tools, system_prompt=my_system_prompt, simulator=False, budget=40
)
```

`my_agent` is called once per rollout with the situation text and returns the
steps it took and what it finally said:

```python
def my_agent(message: str) -> dict:
    return {
        "steps": [{"tool": "get_order", "arguments": {"id": "4412"}, "result": {"status": "ok"}}],
        "final_text": "Order 4412 shipped yesterday.",
    }
```

If it raises, the rollout is dropped and the run says so:
`data.stopped_because == "agent_failed"` when no row survived, with the count
and the first error in `data.search["agent_errors"]` and
`data.search["first_agent_error"]`. An agent that fails every call is called
off after `max(16, 2 * budget)` lost rollouts, so a dead endpoint costs a
handful of calls, not hundreds.

Working in this repo: `uv sync`, then `uv run pytest` after `uv sync --extra dev`.

One runtime dependency (`requests`), Python 3.10+. Installing from PyPI rather than a
path or a git URL matters if you build a Prime Intellect environment on this:
the Environments Hub installs a pushed env with plain pip, so a `[tool.uv.sources]`
git pin resolves locally and then fails on their runtime with a
`ModuleNotFoundError`.

```python
import zeroproof.simulations as zps

data = zps.simulate(tools=my_tools, system_prompt=my_system_prompt, output="rollout.jsonl")
data = zps.simulate(agent=my_agent)
```

Pass `spec=` if you have a local tools-and-system-prompt folder. The generated datasets are on Hugging Face in the [Post-Training Foundational Datasets](https://huggingface.co/collections/zero-proof-ai/zeroproof-post-training-foundational-datasets-6aa0b9c040ff8591988696dc) collection, not stored in this repo: [agent-simulations](https://huggingface.co/datasets/zero-proof-ai/agent-simulations) by agent type, [tool-call-efficiency](https://huggingface.co/datasets/zero-proof-ai/tool-call-efficiency) (SFT, preference, GRPO and eval splits), and [tau2-simulated](https://huggingface.co/datasets/zero-proof-ai/tau2-simulated), among others.

| Knob | Default | |
|---|---|---|
| `agent` / `spec` | hosted Qwen | Callable, URL, or tools + system prompt |
| `budget` / `time_budget` | `1000` / `None` | Stop when either hits. The clock is off unless you set it; `0` or `None` keeps it off |
| `requests_per_situation` | from mode | Phrasings: ways to ask one situation. Alias `phrasings=` |
| `rollouts_per_request` | from mode | Repeats: reruns of one phrasing. Alias `repeats=` |
| `fault_rate` | `0.5` | Broken tools. `0` off. Applied by the mock world, so a callable `agent=` that answers its own tool calls never sees one |
| `simulator` | hosted Qwen | Situation writer. `False` uses the built-in template writer (no model, less variety); an `openai:`/`vllm:` spec runs it on your endpoint |
| `logprobs` | `False` | Ask the rollout model for the log-probability of every token it generates. Each agent turn's step gets `logprob` and `n_tokens`, the row gets the totals. `"tokens"` keeps the per-token list. Model backends only |
| `reproducible` | `False` | Same seed, same concurrency, same agent: same rows. Runs batch by batch, so uneven latency costs throughput. Needs the clock off |
| `grade` | `False` | Legacy: `True` writes the deterministic conduct score at simulation time. Rows come back ungraded by default; grade after with `data.grade(...)` or `zps.grade(...)` |
| `llm_grade` | `False` | Extra LLM judge. Needs `OPENAI_API_KEY` |
| `output` | | JSONL path |

## What to run

Depends on the use case. How each scenario is built is in [The recipe](#the-recipe).

| You want | Mode | What happens |
|---|---|---|
| Many distinct situations | `explore` (default) | New situation every row |
| Same situation, different wording | `sft` | Multiple phrasings: tone, intent, personality |
| Same request, different agent behavior | `rl` | Up to k repeats of one phrasing, spent where the agent is inconsistent (see below) |
| A mix, until coverage plateaus | `adaptive` | New situations, phrasings, and repeats. Best with `until="saturation"` |

```python
zps.simulate(tools=my_tools, system_prompt=my_system_prompt)  # explore
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="sft")
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="rl")
zps.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="adaptive", until="saturation")
```

`mode="rl"` allocates rollouts successively. Every prompt is probed with
two rollouts, the least that can show a split. A prompt whose rollouts
disagree is filled to k, because that is the only place a grouped update
has a gradient. A prompt that stays unanimous gets one more rollout only
while the chance the next one differs beats what a fresh prompt offers
per rollout. Both sides are measured on the run: the hazard is how often a
group unanimous after n rollouts split on its next one (1/(n+2) only as
the prior), the fresh side is the run's mixed rate over the probe. When
nothing fresh can be opened, unanimous groups are resumed and finished.
Pass `grader=` and it runs beside the rollouts as they land, never in
front of them; a prompt's decision waits for its verdict, so the
allocation reads rewards, the signal a grouped update trains on. Without
a grader it reads behavior signatures, which split more often than the
judge does.
Near the end of a `time_budget` the run stops opening groups and finishes
the ones in flight; a group it still cut is stamped `group_cut`.
`data.search["groups"]` reports mixed, stopped, complete, partial, and
rollouts saved. `data.pass_at` scores stopped unanimous groups as
unanimous. `repeat_policy="fixed"` restores k rollouts for every prompt;
`advanced={"probe": n}` changes the probe. rlhf-book ch. 6 (dynamic
sampling) and ch. 7 (difficulty filtering), applied at generation time.

## Examples

| Example | What it does |
|---|---|
| [`examples/agent-behavior`](examples/agent-behavior) | Start here if the platform is new to you. Runs a coding agent with bad habits against real tests, streams every turn to Zero Proof as OTLP spans plus a judge verdict, and fills a dashboard with behaviour worth looking at. No dependencies. |
| [`examples/hosted-loop`](examples/hosted-loop) | Push graded rows, `zps.train` SFT on Qwen3-4B, `zps.serve` the adapter, one chat completion from the endpoint. One key, one A10G minute; the wiring check for training on the platform. |
| [`examples/bring-your-own-agent`](examples/bring-your-own-agent) | Your own callable: the `agent(message) -> {steps, final_text}` contract, the `agent_failed` report when it raises or returns the wrong shape, and `eval_sourced` keeping a held-out score out of the reward. Offline, no key. |
| [`examples/prime-intellect-rl`](examples/prime-intellect-rl) | Generates a GRPO-ready dataset with `simulate(mode="rl")` and checks it carries gradient before you spend GPU time on it. |
| [`examples/schema`](examples/schema) | One row file in, six training targets out: eval, SFT, preference, GRPO prompts, OPSD hints, OPD. Migrates any legacy file first. Offline, no key. |
| [`examples/grpo`](examples/grpo) | GRPO on Modal, end to end: prompts from the simulator, a verifiable tool-discipline reward, TRL `GRPOTrainer` with LoRA, reward and KL on the dashboard, pass@1 before and after on a holdout with the paired delta on the run page. One A10G, under fifteen minutes. |
| [`examples/verifiers`](examples/verifiers) | Verifiable rewards: math (`MathEqual`), an answer-and-format gate (`All`), code run against hidden tests (`CodeExec`), and a JSON-schema check, each feeding `grade`/`optimize`. Offline, no key. |
| [`examples/pass-at-k`](examples/pass-at-k) | pass@1, pass^k and pass@k for one agent, with the per-ask histogram the mean hides and what each number tells you to do next. Offline, no key. |
| [`examples/identity`](examples/identity) | Builds a leak-free SFT set that teaches a model a new name and maker, with Modal scripts to train a LoRA and evaluate it. No model calls to generate. |
| [`examples/character`](examples/character) | Character training from a constitution: the OpenAI Model Spec's style traits become graded rows, preference pairs and SFT rows, with the judge checked against the spec's own labels and a before/after measurement. Offline by default. How-to: [docs/character-training.md](docs/character-training.md). |

## Sign in

```bash
zeroproof login
```

Prints a link and a short code. Open the link, sign in or sign up, press
Approve. The key is saved to `~/.zeroproof/credentials.json` and every
platform call below reads it from there. Interrupted before you approved?
Run it again; it resumes the same code. This is the path for coding
agents too: tell yours to run `zeroproof login` and click the link it
shows you. `zeroproof status` shows which key is in use, `zeroproof
logout` removes it.

No account yet, or no browser? One command creates the account and the
key. Open the dashboard later by signing in with an email code.

```bash
zeroproof signup --email you@example.com
```

That key is a trial key (25k input and 50k output tokens a day, 100 MB,
ten datasets, seven days) until the person signs in once at
https://www.zeroproofai.com/sign-in with an email code. `zeroproof status`
shows the tier; `zeroproof.account()` returns tier, limits and usage.

## Store datasets on Zero Proof Labs

Push a run to your Zero Proof Labs account so the optimization framework
can iterate on it. Credentials resolve in this order: `api_key=` argument,
`ZEROPROOF_DELEGATED_CREDENTIAL` (a short-lived `zp_dc_...` issued from a
Clerk session), `ZEROPROOF_API_KEY`, then the key saved by `zeroproof
login`.

```python
# Runtime path with a delegated credential
# export ZEROPROOF_DELEGATED_CREDENTIAL="zp_dc_..."

# If you need to mint one from a Clerk session token:
# credential = zps.issue_delegated_credential(clerk_token, ttl_seconds=3600)
# export ZEROPROOF_DELEGATED_CREDENTIAL=credential["credential"]

data = zps.simulate(spec="specs/github")
v1 = data.push("github-explore-v1")  # -> {"datasetId": "ds_...", ...}

# iterate, then push the next version with lineage
v2 = data.push("github-explore-v2", parent=v1["datasetId"])

zps.datasets()  # list yours + storage used
rows = zps.pull(v1["datasetId"])  # rows, or pass path= for a file
zps.push_file("rollout.jsonl")  # upload an existing JSONL
zps.delete_dataset(v1["datasetId"])  # permanent
```

Storage is private per account, 5 GB free. `parent=` records dataset
lineage so iterations show as a family on the platform.

`data.push` and `zps.push_file` run a publish gate first (`gate=False` skips it). Every graded row gets a `calibration` stamp: its task's pass rate over k repeats, k, and the policy that produced it, so a trainer can build a curriculum or retire solved tasks. An RL-shaped run (repeats of one ask) is refused with `PublishGateError` when it is ungraded or has no mixed group, because a grouped update would learn nothing from it. The report comes back as `entry["gate"]`, with warnings when out-of-band or unanimous asks are still present; `zps.optimize(data, mode="rl")` prunes those. `zps.publish_gate(rows)` runs the same check on any row list. The stamp is the schema's `Calibration` object: `zps.calibration_of(row)` reads it back typed, `from_row` carries it on `rollout.extra["calibration"]`, and `to_row` writes it out again.

Training rows from `export_training` / `training_rows` carry a `loss_mask`, one 0/1 per message: 1 on the agent's turns, 0 on system, user, and tool-output turns. Tool output is the environment's text, not the policy's, so a trainer should not learn to predict it. `mask_mode="final"` trains only the last assistant turn, for conversations whose earlier agent turns were scripted or came from another policy; the export report counts `trained_messages` and `masked_messages`. `unroll=True` turns an N-turn conversation into N samples, the k-th ending at the k-th agent turn with loss on that turn only, so every earlier turn trains once with the context it actually had (rlhf-book ch. 4).

### Prune before training

```python
rows, report = zps.optimize(data, mode="rl")  # whole groups, 20%-80% pass rate
rows, report = zps.optimize(data, mode="rl", band=(0.3, 0.7))
rows, report = zps.optimize(data, mode="rl", enforce_band=False)  # rank, do not drop
report["band_dropped"]  # {"too_easy": n, "too_hard": n}
```

`optimize(mode="rl")` drops junk rows, duplicate rollouts within an ask (same trajectory twice adds nothing to a group-relative advantage), truncated rollouts, unanimous asks (all pass or all fail: zero advantage), and asks outside the difficulty band, then keeps whole groups round-robin across fault kinds. `optimize(mode="sft")` is rejection sampling (rlhf-book ch. 9): `select="top_per_prompt"` keeps each prompt's highest-reward completion above `min_reward` (default 1.0; lower it for a partial-credit grader), `"top_k_overall"` the best `k` across prompts, and the `random_*` rules are the matching chance controls. Exported groups carry `n0`/`n1` (fail/pass, partial credit splits at 0.5) and `reward_mean`/`reward_std`. The band is the offline difficulty filter from the reasoning-model recipes (keep prompts the policy solves 20-80% of the time); it is a heuristic, so it is a parameter. Every selector report (`select_for_rl`, `select_for_sft`, `build_preference_pairs`) carries `eval_sourced`, the rows or pairs whose reward came from `evaluate()` (`lineage.source == "eval"`), with a warning when it is non-zero: a held-out score that becomes the reward makes the scorer you report the one you optimised against. Nothing is dropped; grade the training set with `run_judge` or `data.grade` and keep `evaluate` for held-out rows.

The report also carries the reward-hack scan: `report["correlations"]` is corr(reward, feature) for reply length, tool-call count, and assistant turns, and `report["hygiene_warnings"]` names anything at or above `HACK_THRESHOLD` (0.3). Reward that tracks length or punishes tool use is a judge problem, so it is flagged, not pruned. The same scan, plus near-duplicate asks and length spread, runs in the publish gate. Standalone: `zps.reward_correlations(rows)`, `zps.dedupe_groups(rows)`, `zps.near_duplicate_prompts(rows)`, `zps.length_report(rows)`.

### Curriculum: easy to hard, and retire the solved

A curriculum needs per-prompt difficulty (rlhf-book ch. 7), which is just each task's pass rate over its k rollouts. `curriculum(rows)` splits graded tasks into *trainable* (ordered easy to hard, and bucketed into `tiers` for a staged schedule), *retired* (pass rate at or above `solved`, default 0.9: an all-pass task is dead gradient), and *not ready* (at or below `floor`: no signal until the policy improves), and counts how many trainable tasks sit in the 20-80% band.

```python
cur = zps.curriculum(scored.rows)  # solved=0.9, floor=0.0, tiers=3
cur["schedule"]  # trainable task ids, easy -> hard
print(zps.format_curriculum(cur))
rows = zps.retire_solved(scored.rows)  # drop tasks the policy already aces
```

### Agents

An agent exists the moment a push names it or a trace arrives with
`gen_ai.agent.name`. Everything on the platform hangs off it.

```python
data.push(
    "airline-v3", agent="airline-support"
)  # registers the agent and attaches tools + system prompt
zps.agents()  # every agent: traces, sets by purpose, public cards
zps.register_agent("airline-support", description="Refunds and rebooking")
```

### Clean up

```bash
zeroproof purge --agent demo-agent --dry-run   # count its traces, datasets, record
zeroproof purge --agent demo-agent             # delete them, after a y/N
zeroproof purge --empty --max-rows 2           # datasets with no bytes, or 2 rows or fewer
```

Python: `zps.purge_agent("demo-agent")`, `zps.delete_empty_datasets(max_rows=2)`.
Both take `dry_run=True`.

### Train, holdout, eval

```python
data.push("airline-v3", holdout=0.2)  # train set + a linked holdout set, split by task
data.push("airline-evals", purpose="eval")  # a set you measure with
zps.update_dataset("ds_...", purpose="holdout")
zps.preview("ds_...")  # three sample rows + the analyzer report
zps.profile("ds_...")  # pass rate, support, mixed tasks, tool use, per task
```

The Datasets page groups sets by purpose (train, holdout, eval) and
records the simulation mode on each. A push is train unless it says
otherwise; ingested traces are eval until training data is cut from them. Holdout is split by
`scenario_id`, so a task is wholly on one side, and the same task lands
on the same side every run.

A task's identity is its cell in the coverage grid: the tools, the
situation axes, and at most one clause of the policy. Each clause owns its
own block of cells and the cells that pair the other axes carry no clause,
so editing the system prompt keeps every task except the ones for the
clause that changed. Rewording one rule, adding one, or swapping the model
leaves the rest of the eval paired for `compare_runs`.

### Trust the numbers

Three checks that decide whether a result is believable, all report-only and all over rows you already have.

```python
zps.judge_trust(rows, judge=my_judge)  # is the judge trustworthy?
pairs, report = zps.judge_pairs(pairs)  # A vs B both ways round: winner, tie, position_flip_rate
zps.decontaminate(train_rows, against=[eval_rows])  # 8-gram overlap with the eval set
zps.style_markers(rows)  # no_boilerplate, no_hedging, no_apology, no_sycophancy, answered
zps.style_report(rows)["warnings"]  # "reward pays for hedging (corr +0.41 ...)"
zps.refusal_report(benign_rows)  # over-refusal rate with a Wilson interval
zps.compare_runs(run_a, run_b)  # paired delta with a 95% interval
zps.delta_report(before, after, target="pass_at_1", must_not_regress=["honest_after_fault"])
zps.delta_report(before, after, target="pass_at_1", by="category")  # the target per kind of prompt
noise = zps.eval_variance(eval_run_1, eval_run_2, eval_run_3)  # re-run std of the eval itself
zps.delta_report(
    before, after, target="pass_at_1", run_std=noise["run_std"]
)  # inside the band = no verdict
zps.mark_grounding(
    rows
)  # markers["argument_grounding"]: every tool argument came from the conversation
zps.grounding_report(rows)  # grounded rate, and the invented values by tool and key
```

**Judge trust.** Label 30 to 100 rows by hand as `gold_reward` (0/1). The report gives agreement with a Wilson interval and Cohen's kappa, agreement on two task halves (tune the rubric on one, read the other), judge pass rate on short versus long replies within the same human label (length bias the humans rule out), and, with the judge callable, a re-judge of a sample as-is (consistency) and with neutral filler appended (a flip means the judge reads length). Disagreements come back as a review queue. `format_judge_trust(report)` prints it. The gold set needs both passes and failures; with one class only the report says so and skips the kappa and length flags. With the hosted judge, call `zps.grade` once first (or `warm_judge`) so the cold start, two to three minutes, is not counted as timeouts.

**Decontamination.** Word 8-gram overlap between a dataset's prompts and replies and any evaluation source: row lists, JSONL paths, or platform dataset ids. Short prompts fall back to exact match. Returns the clean rows and the first offenders.

**Intervals and comparison.** Every pass@1 now carries a 95% interval from a bootstrap over tasks (`pass_at(rows).ci95`), and `metric_summary` / `marker_summary` do the same for markers. Markers come from the judge: return `{"reward": ..., "markers": {"name": value}}` from a `grader=` or `run_judge` callable and they land on `row["markers"]`, which is what `marker_summary`, `delta_report` and `from_row` read. `compare_runs` pairs the tasks two runs share, bootstraps the paired difference, and adds a sign-flip permutation p-value; fewer than five shared tasks falls back to an unpaired test and says so. Tasks on one side only are dropped from a paired comparison; `note` says how many and `paired_share` is the fraction that paired, so a verdict over a quarter of the eval reads as one. The verdict `no_difference_detected` means the interval covers zero, not that the runs are equal.

**Before and after.** `delta_report` runs `compare_runs` on pass@1 and every marker both row sets share. `target=` names the metric the training was meant to move and gives the headline; `must_not_regress=` names the behaviors whose significant drop fails the report; any other significant drop is a warning. `format_delta_report(report)` prints one line per metric. `eval_variance(run_1, run_2, run_3)` is the eval's own re-run standard deviation (three or more evaluations of the same model); passing it as `run_std=` makes any delta inside twice that band `within_noise`, and a target there reads `within_eval_noise` rather than moved, since re-running the eval moves it that much on its own (rlhf-book appendix C). `by=` names a row key, a marker, or a callable that groups rows (a prompt category, a tool, a persona); the report then carries `groups`, the target compared within each group, and `groups_down` for any group whose target dropped significantly while the headline moved. A headline over one dominant kind of prompt cannot hide the other kinds that way.

**Argument grounding.** A policy trained to call a tool learns to call it before it learns when not to; on the refund environment both GRPO and DPO learned to invent an order id on a quarter of the prompts that gave none while the headline rose. `mark_grounding(rows)` stamps `argument_grounding`: 1 when every string argument of every tool call appears in the prompt, the user and system turns, or an earlier tool result (rows with no calls count as grounded), else 0. No categories, any agent; `must_not_regress=["argument_grounding"]` fails the run that learned to invent, and `ungrounded_arguments(row)` / `grounding_report(rows)` name the values. `ignore_keys=` skips free-text arguments, `allow=` lists enums and defaults.

**Over-optimization, quick read.** `behavioral_markers(rows)` gives the presence rate of each over-optimization tic (boilerplate, self-reference, hedging, refusal, sycophancy) in one call: higher means the tic shows up more.

```python
zps.behavioral_markers(scored.rows)  # {"boilerplate": 0.31, "refusal": 0.04, ...}
```

For a before/after comparison use `style_markers` / `style_report` above, not these: those markers are 1.0 when the reply is clean (higher is better), which is the polarity `delta_report(must_not_regress=...)` expects. `behavioral_markers` is presence (higher is worse), so it reads a paired delta backwards. The two cover the same ch. 14 behaviors and are being consolidated onto `style`.

**Benchmark across seeds.** `pass_at(rows).ci95` is the interval over which tasks you picked; it does not see that the model is stochastic and the same eval re-run gives a different number (rlhf-book ch. 16). `benchmark_report(seed_runs)` takes several graded runs of one frozen eval and reports pass@1 mean and SD across seeds, the spread, and the tasks that flip seed to seed, so you know whether a before/after delta clears the run-to-run noise. `run_benchmark(eval_set, judge=, rollout=, seeds=5)` drives the seed loop for you; pass `decontaminate_against=train_rows` to fold an 8-gram leak check into the scorecard.

```python
runs = [evaluate(roll(seed=s), judge=my_judge).rows for s in range(5)]
print(zps.format_benchmark(zps.benchmark_report(runs, name="telecom-holdout")))
```

**Stage lineage.** The pipeline is a sequence of stages (rlhf-book ch. 3): SFT, reward modeling, RL, and the eval that judges the result. `stamp_stage(rows, "sft")` records which stage a row fed, and `stage_report(rows)` counts rows per stage and flags the one mistake it most needs caught: any task used in both `eval` and a training stage. `zps.stamp_stage`, `zps.stage_report`, `zps.stage_of`, `zps.STAGES` (`sft`, `rm`, `rl`, `eval`, `mid`).

**Model spec as an object.** A spec or constitution is a living, versioned document (rlhf-book ch. 17). `load_spec(constitution)` wraps the `{source, traits: [{id, name, principle, authority}]}` shape (what the character example writes) into a `Spec` whose `version` is a content hash, so any edit to a principle changes it. `spec.behaviors()` are the trait ids, ready for `delta_report(must_not_regress=...)`; `stamp_spec(rows, spec)` tags a run with the spec version it targeted, so you can ask whether adherence held from one spec or model version to the next.

```python
spec = zps.load_spec("examples/character/constitution.json")
scored = zps.stamp_spec(data.grade(judge=my_judge).rows, spec)
zps.delta_report(before=before, after=scored, target="pass_at_1", must_not_regress=spec.behaviors())
```

### Train, and watch it

Two ways to train, one record. The platform trains a pushed dataset (SFT, GRPO or DPO, LoRA on an A10G) and serves the result; or your own trainer runs on Modal, a GPU box, or a notebook and reports into the same run. Either way the loss curve and the progress bar are at [zeroproofai.com/platform/training](https://www.zeroproofai.com/platform/training).

```python
run = zps.train("ds_...", method="sft", base_model="Qwen/Qwen3-4B")  # or "grpo" (steps=), "dpo"
run.wait()  # done or failed; run.url is the curve while it goes
run.training["before"], run.training["after"]  # holdout pass@1 (SFT: loss)
model = zps.serve("refund-v2", run)  # adapter on an OpenAI-compatible endpoint
# model["endpoint"] + /chat/completions, model="refund-v2", bearer = your zp_ key
zps.models()  # what the account hosts
```

`holdout=` names the eval set (defaults to the train set's split sibling); a dataset already training returns that run. `serve` needs a finished run whose base is a served one (`Qwen/Qwen3-4B`, `microsoft/phi-4`). The trainer's default bases (Qwen2.5-0.5B for SFT, 1.5B for GRPO and DPO) train in under a minute but cannot be served, so `train` warns when a run will not reach an endpoint; SFT runs on an A10G; GRPO and DPO run on an L40S, so a 4B base fits all three. Qwen3 answers in thinking mode by default: leave room in `max_tokens` or send `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

Your own trainer, three ways in:

```python
# one line on a Transformers or TRL trainer
run = zps.training_run(
    "identity-v1", dataset="ds_...", base_model="Qwen/Qwen3-4B-Instruct-2507", trainer="trl"
)
trainer.add_callback(zps.TrainerCallback(run))
trainer.train()  # loss, lr, eval loss, epoch, grad norm, then finish

# your own loop
with zps.training_run("sft-v3", dataset="ds_...", total_steps=1000) as run:
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        run.log(step, loss=loss, lr=scheduler.get_last_lr()[0])
    run.finish(summary={"final_loss": loss}, adapter="s3://.../adapter")  # failed on exception
```

Plain HTTP, for a stack that is not Python: `POST /runs {"name", "dataset_id", "base_model", "total_steps"}` returns `runId`; `POST /runs/{id}/log {"points": [{"step": 10, "loss": 1.2, "lr": 1e-4}], "total_steps"?}` in batches of up to 500; `POST /runs/{id}/finish {"status": "done|failed|stopped", "summary"?, "adapter"?}`. All with `X-Api-Key`. Points are buffered on the client and a send that fails is retried on the next flush; the dashboard never interrupts the trainer. `zps.get_run(id)["series"]` returns the points, oldest first.

### Publish a dataset as a card

```python
data.push(
    "airline-refunds-v3",
    agent="airline-support",
    publish=True,
    description="Graded refund conversations with injected tool faults.",
)
zps.publish("ds_...", agent="airline-support")  # or publish an existing one
zps.catalog()  # every public card, by agent
rows = zps.pull("ds_...")  # public sets need no key
zps.unpublish("ds_...")

Hugging Face, both directions. Connect your account once on any dataset page, then:

```python
zps.hf_status()                                               # connected? namespaces
zps.hf_publish("ds_...", repo="airline-refunds", wait=True)   # rows -> a dataset repo you own
zps.hf_publish_run("run_...", private=True)                   # a finished run's LoRA adapter -> a model repo
row = zps.import_hf("tatsu-lab/alpaca", split="train", purpose="eval")   # any Hub split -> your account
zps.profile(row["datasetId"])                                 # profiled before you train on it
```

Every push is one commit tagged `zp-<id>`, so `load_dataset(repo, split, revision="zp-ds_...")` pins the exact push; the repo's `zeroproof.json` maps each split to its ZeroProof dataset with history.
```

Cards live at https://zeroproofai.com/datasets, grouped by agent, with rows,
size and the analyzer's numbers on each. A dataset must be finalized and
hold rows to publish.

## Speed

Two-minute airline runs using ZeroProof-hosted Qwen. Results were measured on the
hosted GPU with warm replicas and burst under load.

| Mode | Rows | Rate | Unique openers |
|---|---|---|---|
| `explore` | 240 | 120/min | 240 |
| `sft` | 278 | 139/min | 278 |
| `rl` | 625 | 296/min | 209 |

## Parameter reference

| Parameter | Default | Meaning |
|---|---|---|
| `agent` | hosted Qwen | Rollout model |
| `spec` | | Local tools and system prompt path |
| `tools`, `system_prompt` | from spec or agent | Tool list and agent system prompt |
| `situations` | | Distinct situations (N) |
| `requests_per_situation` | from mode | Phrasings per situation (n). Alias `phrasings=` / `n=` |
| `rollouts_per_request` | from mode | Repeats per phrasing (k). Alias `repeats=` |
| `unique_situations` | on in `explore` | Unique situations only |
| `mode` | `"explore"` | `explore`, `sft`, `rl`, `adaptive` |
| `reproducible` | `False` | Round-synchronous scheduling; see the knob table |
| `budget` | `1000` | Row cap |
| `time_budget` | `None` | Seconds. Off by default; `None` or `0` disables |
| `until` | `"compute"` | `"saturation"` also stops when coverage plateaus |
| `grade` | `False` | Legacy deterministic conduct score; grade after instead |
| `llm_grade` | `False` | Extra LLM judge |
| `output` | | JSONL path |
| `advanced` | | Keys below |

| `advanced` key | Default | |
|---|---|---|
| `concurrency` | `32` | Parallel rollouts |
| `stop_grace` | `5` | Seconds to wait for running rollouts and writer waves after a stop; queued ones are cancelled, still-running ones are reported as `rollouts_abandoned` / `writer_waves_abandoned` |
| `embedder` | `"hash"` | Prompt selection |
| `seed` | `0` | Reproducible draws. Bit-for-bit at `concurrency: 1` or with `reproducible=True`; otherwise which rows land before the cap depends on thread timing |
| `avg_turns` | `4` | Target conversation length |

Aliases: `phrasings=` / `n=` → `requests_per_situation`; `repeats=` → `rollouts_per_request`; `unique=` → `unique_situations`; `policy=` → `system_prompt`; `risk=` → `fault_rate`.

## Output

Each row, in `data.trajectories` and on disk: `prompt`, `messages`, `steps`, `final_text`, `scenario_id`. Optional `world_state`, `faults`, `reward`, `reason`. `llm_grade=True` adds `llm_reward`. `zps.rank(path)` adds `quality` without changing `reward`.

After grading, `data.pass_at` (also on the `ScoredData` from `judge=` and `evaluate`) gives pass@1, pass^k and pass@k off the same groups, one job each: pass@1 is the measurement headline (the agent runs once in production), pass^k is the reliability line (all k repeats pass), and pass@k minus pass@1 (`.headroom`) is what a grouped RL update has to learn from, the same asks `group_signal` counts as mixed. k is the smallest group of repeats; below `repeats=4` the k-way numbers are `None` with a note rather than a noisy figure. `.per_task` is the raw per-prompt pass-rate vector. With an LLM judge, pass@k inflates on false positives and pass^k on false negatives, so pass@1 stays the headline.

```python
scored = data.grade(judge=my_judge)
print(scored.pass_at)  # pass@1 0.61 | pass^8 0.32 | pass@8 0.88 | headroom 0.27 (200 groups, k=8)
```

`simulate(logprobs=True)` records, on every agent turn, the summed log-probability of the tokens the policy generated and how many there were (`step["logprob"]`, `step["n_tokens"]`, totals on the row). A trainer that updates on these rollouts later needs that number to form the importance ratio `exp(new_logprob - logprob)`; without it the update is off-policy and nothing says so. `zps.logprob_report(rows)` says how much was captured and whether reward tracks the policy's confidence, which on a fair judge it should not. Score the same rows under a reference model, put its summed logprob in `ref_logprob`, and `zps.mean_kl(rows)` gives the sampled KL per generated token, overall and per task; `zps.calibrate(rows, ref="ref_logprob")` writes it into each row's `calibration.mean_kl`. A turn the model cut at the token cap is marked `truncated`. Independently of `logprobs`, every agent step also records what its model call cost when the server reports it (`step["input_tokens"]`, `step["output_tokens"]`, summed into `row["usage"]`), which is what the platform counts per day.
zps.staleness_report(rows, base_model="Qwen/Qwen3-4B")  # policy versions, stale rows, logprob coverage

The default judge is not the policy. `zps.grade` grades with hosted Phi-4 (`ZEROPROOF_JUDGE` overrides; any `vllm:`/`openai:` spec or a bare URL works), while rollouts come from hosted Qwen, because a judge grading its own model's writing prefers it. When the judge and the rows' `model_version` are the same model anyway, the grade report says so (`self_judged`, `warnings`).

A judge is a reward model, so two things ride with every label. Provenance: rows graded by `zps.grade` carry `judge_name`, `judge_status`, and `judge_meta` with the model, prompt hash, temperature, and `version` (`<model>@<prompt sha>`); a rubric edit is a new judge and the row says so. `run_judge(version=...)` records the same for your own judge. Accuracy: hand-label a sample into `gold_reward` and call `zps.judge_agreement(rows)` (or `scored.agreement()`) for agreement, Cohen's kappa, the confusion counts, and `pass_when_gold_fail`, the gold failures the judge passed. Those are the rows a training run learns the failure from, so that rate matters more than the headline agreement. Pass a second scoring run as `gold` to measure the judge against itself. Fifty gold rows is the floor; the report says so below it.

Every row carries `schema_version` (`"1"`). A row is a projection of four objects in `zeroproof.simulations.schema`: `Task` (the situation), `Rollout` (one episode), `Judgment` (a scorer's verdict), `Marker` (a behavior measurement). `zps.from_row(row)` splits a row into them and `zps.to_row(...)` flattens them back. The wire contract is `zeroproof/simulations/schemas/row-v1.json`. Rows written before the stamp are version 0 and load by shape, so older files still work.

## The recipe

Each scenario is a draw across the world and the human.

**World** (from this agent's tools and system prompt)

- objects and tool results that match the spec
- tool outcome: success, timeout, deny, stale, etc.
- world state: exists, missing, already handled, unfinished, etc.
- history: first visit, prior miss, return, etc.
- rules the agent is supposed to follow

**Human**

- intent: which tool, what they want (randomized sometimes)
- stance: ordinary, ambiguous, adversarial, hurried, etc.
- persona: first time, returning, in a hurry, etc.
- tone: impatient, frustrated, polite, etc.
- typing: standard, lowercase, typo, clipped, etc.

Ordinary asks first, then the edges. On top of that, we embed the openers and add a bit of random noise so the batch stays spread out, not a cluster of near-copies. Spend the row cap and the clock on diversity, not copies.

## Package layout

The public surface is the package itself: `import zeroproof.simulations as zps`.
Internals are grouped by stage and may move between releases.

| folder | what lives there |
|---|---|
| `generate/` | situation grid, writer, diversity selection, agent runners and adapters |
| `score/` | conduct checks, judges, quality ranking, selection for SFT and RL |
| `ingest/` | trace loading, OpenTelemetry rows (`gen_ai.usage.*` sums into `row["usage"]`), platform push and pull |
| `world/` | the mock tool environment |
| `run/` | the engine behind `simulate()`: knob resolution (`config.py`), spec loading (`spec.py`), row helpers (`rows.py`), and the scheduler itself (`engine.py`: inputs, build, loop, finish) |
| `simulation.py`, `data.py`, `export.py` | the `simulate()` entry point, its result object, and training export |

## Development

```bash
uv sync --extra dev
uv run pytest           # about two minutes, no network
uv run ruff check .     # lint; `--fix` for the mechanical ones
uv run mypy             # type check
pre-commit install      # optional: ruff and whitespace hooks on commit
```

CI runs the suite on Python 3.10 through 3.13, ruff, mypy, and a plain-pip
install of the built wheel into a clean venv.

## License

Apache-2.0
