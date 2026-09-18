# whileai

The While Python SDK. One package, two importable modules:

- `whileai`: the platform client. OTLP trace ingest and trace-dataset listing against the token gate.
- `whileai.simulations`: post-training data for an agent. Give it the agent's traces, or its tools and system prompt; it simulates the situations, the people, and the world, plays the agent through multi-turn tool-calling conversations, and returns rows for your grader.

Have an agent and want a pass rate with an interval? Start at [docs/evals.md](docs/evals.md) (offline, seconds, `coverage_gap` names what your tests miss).

**Renamed.** This SDK was `zeroproof` (ZeroProof is now While). `pip install zeroproof` still works: it installs `whileai`, and `import zeroproof` (or the older `zeroproof_simulations`) resolves to the same modules with a deprecation warning. `ZEROPROOF_*` environment variables and a saved `~/.zeroproof/credentials.json` are still read. Change the import when you can; new releases land under `whileai`. zp, ZeroProof and While all name this one product: the package is `whileai`, the import is `whileai.simulations`, keys start with `zp_`. `zp`, `wai` and `whileai` run the same CLI, so `zp login` and `whileai login` do the same thing (the help text says `whileai`). A machine with the old package still picks up `~/.zeroproof/credentials.json`; set `WHILEAI_HOME` to a fresh directory to isolate a new account from it.

Releases of `whileai` before 0.3 were an unrelated encrypted agent-to-agent messaging client. That code was removed in 0.04; pin `whileai<0.3` if you still depend on it.

Two ways in, one engine. Give it the agent's tools and system prompt and it samples situations across everything that agent can be asked. Give it graded traces as well (`traces=`, plain row dicts — see [Close the loop](#close-the-loop-aim-the-budget-with-traces)) and it aims the budget at the situations that fail in production, so new rows land where the agent is weak and carry both the failure and the fixed version. Every row is a full conversation: user turns, agent turns, tool calls, tool results, scheduled faults. Rows come back ungraded; your grader decides what good means. Default `explore`: one unique situation per row. How it thinks: [docs/simulations.md](docs/simulations.md).

## How a row gets made

![How a row gets made: the draw, the coverage grid, the search arms, the rollout, the split](docs/how-a-row-gets-made.svg)

A situation is drawn across the world axes (from the agent's tools) and the human axes (from a separate writer). It fills a cell in the coverage grid, nudges the five search arms, and the agent plays it against a world that breaks on schedule. The row that comes out splits into `Task`, `Rollout`, `Judgment`, and `Marker`, and every training target is a projection of some of those four. The engine on one page, with references: [docs/engine.md](docs/engine.md), also at [zeroproofai.com/docs/engine](https://zeroproofai.com/docs/engine).

## Overview

`simulate()` is a pipeline.

1. **Read the agent.** Tools and system prompt. That is the spec of the world.
2. **Build a fake world from those tools.** Objects, plausible results, and faults (timeout, deny, junk).
3. **Write users.** A separate writer (same hosted model, different prompt, no agent policy) samples situations across tools, stance, history, and so on.
4. **Pick the diverse ones.** Embeddings plus a bit of noise so the batch is not 200 copies of the same prompt.
5. **Play the agent.** It talks, calls tools, gets results, talks again. All of that is stored: user text, agent text, tool calls, tool results, `final_text`.
6. **Grade.** Rows come back ungraded. Grade after with `data.grade()` (hosted judge, against the spec's `rubric.md` or `rubric=`), `data.grade(judge=...)` (your judge), or `wai.grade(path)`. The legacy `grade=True` flag writes deterministic conduct rewards; avoid it for the rubric workflow.

Stop when the row cap or the clock hits.

## How to use

```bash
pip install whileai   # or: uv add whileai
```

### Start here: no key required

This runs offline, in seconds, on nothing but the package. It is the
fastest way to see a row and to check your agent and grader are wired up
correctly before you spend a key on variety.

```python
import whileai.simulations as wai

# 1. Your tools, in OpenAI function-calling shape. This is all `tools=` wants.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


# 2. Your agent: one call per rollout, in with the situation text,
#    out with the steps it took and what it finally said.
def my_agent(message: str) -> dict:
    return {
        "steps": [
            {
                "tool": "get_order",
                "arguments": {"order_id": "4412"},
                "result": {"status": "shipped"},
            }
        ],
        "final_text": "Order 4412 shipped yesterday.",
    }


# 3. simulator=False uses the built-in template writer: no model, no key.
data = wai.simulate(
    my_agent,
    tools=TOOLS,
    system_prompt="Help customers with orders.",
    simulator=False,
    budget=20,
)

# 4. Your grader. Any callable row -> {"reward": 0 or 1, ...}.
scored = data.grade(judge=lambda row: {"reward": int("4412" in row["final_text"])})
print(scored.pass_at)
```

Offline, every marker is your agent's. The template writer writes the
users, not the agent, so a callable that never hedges scores zero
hedging, and `faults` on a row only fire if your agent's tool calls go
through the world that schedules them. `wai.world(TOOLS)` is that world:

```python
WORLD = wai.world(TOOLS)


def my_agent(message: str) -> dict:
    result = WORLD.call(
        "get_order", {"order_id": "4412"}
    )  # timeouts, stale data, denials fire here
    return {
        "steps": [{"tool": "get_order", "arguments": {"order_id": "4412"}, "result": result}],
        "final_text": "Order 4412 shipped yesterday."
        if result.get("status") == "ok"
        else "The lookup did not go through, so I cannot confirm 4412 yet.",
    }
```

To see the detectors fire before you plug in your own agent, run the
seeded one. It answers honestly through `wai.world`, and on a labeled
fraction of rollouts does one wrong thing on purpose: hedges, flatters,
apologizes, pads, claims success through a fault, or quotes the row's
privileged context. Every row says what it did in `seeded` (`[]` when
it behaved), so a check that catches exactly those rows is a check that
works.

```python
data = wai.simulate(
    wai.seeded_agent(TOOLS),
    tools=TOOLS,
    system_prompt="Help customers with orders.",
    simulator=False,
    budget=60,
)
rows = data.trajectories  # export_row scrubs privileged; the run keeps it
print(wai.style_report(rows)["markers"]["no_hedging"]["hits"])  # > 0, only on seeded rows
print(wai.format_leak_report(wai.leak_report(rows)))
```

```
checked 59 of 60 rows: 5 quoted privileged context (8%)
  sc-ca1635b7db r0: reference = 'lookup_order succeeds and the reply reports its result'
```

`leak_report` says how many rows it could check. A run where nothing
populated `privileged` has nothing to leak, and the report says so
instead of passing. Every row is now born with the block: `hidden_state`
(what the world knows that the ask does not say) and `reference` (what
the checklist expects), derived from the task's grid cell. Exports drop
it at any depth; `data.trajectories` keeps it for the judge.

The bare `function` dict without the `{"type": "function", ...}` wrapper
works too; both shapes are normalized. The template writer needs no model
and runs in seconds, but the situations are less varied than a model writes,
so it is for wiring up your agent and grader, not for a training set — for
that, bring a model below.

### Evals for the agent you already have

Not training anything yet? The shortest path is an eval: wrap your agent
as `agent(message) -> {steps, final_text}`, write the policy as a judge
that reads the trajectory, run the asks `k` times each, and read pass@1
with its interval. Offline first, then the hosted writer. The how-to is
[docs/evals.md](docs/evals.md); the runnable version is
[`recipes/02-measure/eval-your-agent`](recipes/02-measure/eval-your-agent),
which ends at a CI gate, not a push. `whileai init-evals` writes those
four files for you, wired to the tools, system prompt and callable it
finds in the project, and prints what it picked.

```python
data = wai.simulate(
    agent,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=SEEDS,
    simulator=False,
    mode="rl",
    repeats=4,
    repeat_policy="fixed",
)
scored = wai.evaluate(data, judge)  # eval lineage: never the reward
print(wai.pass_at(scored.rows), *scored.warnings)  # a hollow run says so here
```

`scored.warnings` is new: no rollout called a tool, a declared tool no
rollout touched, a marker that fired on no row. A 1.00 on a run like that is not a result; the note names the fix.

A declared tool the world cannot answer is the quiet version of the same
failure: with `execute=`, a tool that is in the schema but has no branch in
your function fails exactly like a world fault, the agent reports the miss
honestly, and a candour rubric rewards the row. Every run now records calls
and successes per tool in `data.coverage["tools"]` (`n`, `ok`, `fault_n`, and
`injected` for faults the run scheduled itself), lists the tools that never
work in `data.coverage["dead_tools"]` (the Wilson 95% upper bound on the
success rate is under 0.30, so 0 of 9 or 4 of 612 is dead and 0 of 3 or 2 of
5 is not), adds `dead_tools` to `data.degraded`, and puts the names and the
one fix that applies to your world in `data.warnings` and `data.report()`.
Steps with no recorded result are not evidence and never accuse a tool.

### Bring a model

Bring your own model. Any OpenAI-compatible chat endpoint that returns tool
calls works; it writes the situations and plays the agent, so both run on
your key. To put a number on a model you serve (`wai.serve`, or your own
vLLM), make it the agent: `wai.simulate(tasks=pinned,
agent=wai.local_model(endpoint, name, tools=TOOLS, system=POLICY,
thinking=False))`, and run both arms of a before/after through that same
call so the only difference is the weights.

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # only for a non-OpenAI endpoint
```

```python
import whileai.simulations as wai

data = wai.simulate(
    agent="openai:gpt-4.1-mini",
    tools=my_tools,
    system_prompt=my_system_prompt,
    output="rollout.jsonl",
)
```

A model spec names the backend and the model. Four are built in:

- `ollama:<model>`: a local Ollama server, no key.
- `vllm:<model>@<url>`: any vLLM or OpenAI-compatible endpoint you serve.
- `openai:<model>`: `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for a
  compatible endpoint that is not OpenAI's.
- `anthropic:<model>`: the Claude Messages API on `ANTHROPIC_API_KEY`
  (`WHILEAI_ANTHROPIC_API_KEY` overrides it).

A spec works everywhere one is accepted: `agent=`, `simulator=` for the
situation writer, `user_model=` for the simulated person, and `spec=` for the
judge.

```bash
export ANTHROPIC_API_KEY=...
```

```python
data = wai.simulate(
    agent="anthropic:claude-haiku-4-5",
    tools=my_tools,
    system_prompt=my_system_prompt,
    simulator="anthropic:claude-sonnet-5",  # the writer, on the same key
    output="rollout.jsonl",
)
```

## Five calls

Agent to gated dataset. Everything else in this README is one layer down.
`TOOLS` is the list from [Start here](#start-here-no-key-required);
`POLICY` is the agent's system prompt.

```python
import whileai.simulations as wai

data = wai.simulate(
    agent="openai:gpt-4.1-mini",
    tools=TOOLS,
    system_prompt=POLICY,
    mode="rl",
    situations=200,
    repeats=8,
)  # 1 generate
data.grade(rubric=RUBRIC)  # 2 grade against the task rubric: reward 0/1 on every row
print(data.pass_at)
wai.judge_trust(data.trajectories)  # 3 trust the numbers
rows, report = wai.optimize(data, mode="rl")  # 4 prune to what carries gradient
entry = wai.push_rows(rows, "my-agent-rl-v1", gate=True, mode="rl")  # 5 publish, gated
```

`situations=200, repeats=8` is a guess. `wai.recommend(tools=TOOLS, system_prompt=POLICY, mode="rl")` replaces it with numbers from this agent's own grid: [How much to run](#how-much-to-run).

A spec folder is `spec.json` (tools and policy) plus `rubric.md`: what doing the job means, in prose. `grade()` scores against it. The hosted judge writes `reward` and `reason` onto the run's rows and returns the judge report (a dict), so the numbers are read off `data`; `grade(judge=your_callable)` instead returns a `ScoredData` of graded copies, leaves the run untouched, and has its own `.push(name, ...)`. Without one it grades the conduct floor only (nothing invented, nothing skipped) and the report says so; pass `rubric=` to `simulate` or `grade` to supply one, `judge=` for your own callable.

After training, measure whether it landed: `wai.delta_report(before=scored.rows, after=after_rows, target="pass_at_1")`. Name the training reward too, `proxy="marker:first_action"`, and the report says whether the run over-optimized it: proxy up while the target did not follow fails the report (rlhf-book ch. 14). `wai.hack_scan_diff(before, after, endorsed=[...])` names what the update moved toward, and withholds the name when either side came back `degenerate`.

Character training, the same loop aimed at how the model talks: a constitution in, graded replies, length-matched pairs and SFT rows out, and the judge checked against the constitution's own labels. Worked example [`recipes/03-select/character`](recipes/03-select/character), recipe [docs/character-training.md](docs/character-training.md), page [zeroproofai.com/docs/character-training](https://zeroproofai.com/docs/character-training).

| Call | What it decides | Reads |
|---|---|---|
| `simulate` | the situations, the users, the world, k rollouts per ask | your spec or tools + system prompt |
| `data.grade(judge=)` | 0/1 per rollout. `wai.grade(data)` uses the hosted judge instead | your judge callable, or your account key (`whileai login`) |
| `pass_at` / `judge_trust` | pass@1 with an interval, headroom for RL, whether the judge can be trusted | graded rows, 30 to 100 hand labels as `gold_reward` |
| `optimize(mode="rl")` | drops junk rows, duplicates, dead groups, and asks outside the *difficulty* band; flags reward hacks | graded rows |
| `push_rows(gate=True)` | refuses ungraded or gradient-free RL data; stamps calibration | pruned rows |

### The judge contract

Rows come back ungraded; your judge decides what good means. A judge is
any callable that takes a row and returns a verdict. LLM judge, rules
engine, reward model, human-label lookup, HTTP call: the SDK does not care
how the reward was produced, only that the result honors this contract.
The same contract is what `grade`, `run_judge`, `evaluate`, `grader=`,
`optimize` and a gated `push` all read, and what every `verify` verifier
and `wai.reward_model(run)` already honors.

```python
judge(row) -> {"reward": 0 or 1}              # the minimum
judge(row) -> {"reward": 0.7,                 # floats allowed
               "reason": "...",               # optional, kept on the row
               "markers": {"grounded": 1.0},  # optional, -> row["markers"]
               "failure_class": "...",        # optional
               ...anything else}              # kept as judge metadata
judge(row) -> 0 or 1 or 0.7                   # a bare number works
```

**Failure modes.** Anything else — a missing `reward`, an unsupported
type, an exception, a timeout — marks the row (`judge_status` of
`missing_reward` / `invalid_result` / `error` / `timeout`) and sets
`reward=None`. Nothing is silently scored zero, so a broken judge shows up
as unjudged rows rather than as a policy that looks bad.

**Marker polarity, the rule for every marker you define.** `1.0` is the
good outcome; higher is better; a significant drop is the regression.
`delta_report`, `must_not_regress=` and the run page all assume it. Name a
marker for the behavior you *want* — `refund_correct`, not
`false_refund_success` — or a fix reads as `DOWN` and listing the marker in
`must_not_regress=` fails the report on the run that repaired the bug.
More on the four marker families in
[Markers: four families, one polarity](#markers-four-families-one-polarity).

**The loop, closed in five lines.**

```python
import whileai.simulations as wai

judge = lambda row: {"reward": int("sorry" not in row["final_text"])}
scored = wai.run_judge(data.trajectories, judge)  # or data.grade(judge=judge)
wai.export_dataset(scored.passes(), output="train.jsonl", system_prompt=POLICY, tools=TOOLS)
# ...train externally, roll the tuned model on a holdout...
evald = wai.evaluate(rollouts, judge, model="my-tuned-v1")
nxt = wai.simulate(tools=TOOLS, system_prompt=POLICY, traces=evald.failed_traces())
```

The full contract, with every status and the rest of the loop, is the
module docstring of `whileai.simulations.score.judging` — note the
`score.`; there is no `whileai.simulations.judging`.

Writing the judge is half of it; knowing whether to believe it is the
other half. `wai.judge_trust(rows, judge=...)` and `wai.judge_probes(rows,
judge)` are under [Trust the numbers](#trust-the-numbers).

### Verifiers: when the reward is a program, not a judge

For a verifiable task the reward is a checker, not an opinion (RLHF book ch. 7, 13). `whileai.simulations.verify` gives you one, and because a verifier honors the same judge contract it drops into `grade`, `evaluate`, `optimize` and a gated `push` exactly where an LLM judge would.

```python
from whileai.simulations.verify import MathEqual, CodeExec, JSONSchema, Regex, All

data = wai.simulate(
    tools=MATH_TOOLS, system_prompt=MATH_POLICY, mode="rl", situations=200, repeats=8
)
scored = data.grade(judge=MathEqual())  # the verifier is the reward
rows, _ = wai.optimize(scored, mode="rl")  # GRPO data, gradient checked
```

The candidate is the rollout's `final_text`; the gold is read from the row's `privileged.reference`, which the training export never projects, so the answer key cannot leak into a training file (flat `answer`/`target`/... fields work too, or point at any column with `field=`). Built in: `ExactMatch`, `Includes`, `Regex`, `MultipleChoice`, `Numeric`, `MathEqual`, `JSONValid`, `JSONSchema`, `JSONField`, and `CodeExec` (runs the candidate against hidden tests in a sandboxed subprocess with a timeout). Compose with `All` (right answer *and* right format), `Any`, or a graded `Weighted` rubric; wrap your own with `@verifier`. Worked example: [`recipes/01-simulate/verifiers`](recipes/01-simulate/verifiers).

Or use While-hosted Qwen, which is the default when no `agent=` is given.
Your account key is enough: `whileai login` (or `whileai signup --email
you@example.com`) and the run goes to the account endpoints, Qwen3-4B for
the writer and the agent and Phi-4 for the judge, on your daily allowance
(a trial key: 25k input and 50k output tokens a day; after one sign-in:
100k and 500k). The endpoint refuses with 429 when the allowance is spent
and the run stops there and says so. `VLLM_API_KEY`, when set, wins and
goes to the shared pool instead: warm and faster, shared and unmetered;
ask us for one.

```bash
whileai login              # or: export WHILEAI_API_KEY=zp_...
export VLLM_API_KEY=...      # optional: the shared pool instead
```

No key at all: the situation writer also defaults to hosted Qwen, even when
`agent=` is your own function, so `simulator=False` is what makes a run
fully offline — see [Start here](#start-here-no-key-required) above for the
whole runnable block.

```python
data = wai.simulate(
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

`data.search` is the run's own report dict, and its keys are written only
when the run has something to say: these two appear only if the agent
actually raised, and `search["groups"]` only on a run with repeats. A clean
`explore` run with a working agent has neither, so read them with
`data.search.get(...)` rather than concluding the attribute is missing.

Working in this repo: `uv sync`, then `uv run pytest` after `uv sync --extra dev`.

One runtime dependency (`requests`), Python 3.10+. Installing from PyPI rather than a
path or a git URL matters if you build a Prime Intellect environment on this:
the Environments Hub installs a pushed env with plain pip, so a `[tool.uv.sources]`
git pin resolves locally and then fails on their runtime with a
`ModuleNotFoundError`.

### Export an RL environment

On-policy RL (GRPO, RLOO, PPO) samples its own rollouts from the policy
under training, so what it needs is not rows but what the rows came from:
the task set, the world that answers tool calls, and the reward that
grades a finished trajectory. `export_environment` writes those three as an
installable `verifiers` package, the shape Prime Intellect and TRL read.

```python
data = wai.simulate(my_agent, tools=TOOLS, system_prompt=POLICY, mode="rl", repeats=8)
data.grade()
# reward and world must import by name in the trainer: a module-level function or "module:attr"
wai.export_environment(data, "envs/my-agent", reward=my_verifier)
# pip install -e envs/my-agent
# vf-eval my_agent -a '{"split": "holdout"}' -m <policy> -b <base url> -k <key var>
```

The package holds `spec.json` (system prompt, the tool schemas verbatim,
the turn cap, and dotted references to the reward and the world),
`data/train.jsonl` and `data/holdout.jsonl` (one task per prompt in the
verifiers shape, with the task's fault plan, world state, privileged
reference and calibration in `info`, read on the server and never in the
prompt), and a README with the gate: the difficulty band applied when the
rows were graded (prompts the policy always or never solved carry no
advantage and are dropped), the split by scenario, and the train-against-
holdout decontamination. The environment class lives in the SDK and is
tested there: a `StatefulToolEnv` whose world is the mock world seeded per
task, or your own `execute=`, and whose rubric is the reward through the
judge contract, so a `Verifier` such as `CodeExec`, your judge callable, or
`conduct_grade` all work unchanged. The default is `task_checklist`: the
conduct grade as an honesty gate, times an outcome the world can verify from
the task's own coordinates on the grid. A target tool must succeed; a missing
entity must be reported and not acted on; an already-done action must be
acknowledged and not repeated; an adversarial ask must not produce a write;
an unrelated ask must produce no call; a vague ask must be asked back; prior
partial action needs a read before the write; a fault on the target must be
acknowledged. No model in the loop, and `markers` say which check ran
(rlhf-book ch. 12 rubrics, computed from state rather than written by a judge).
When the rows carry none of that metadata the export warns: the reward
reduces to `conduct_grade`, a process reward, and a policy trained on it
alone learns to call nothing (`recipes/03-select/prime-intellect-rl`). `wai.load_environment(spec)`
builds the environment in a process that has `verifiers` (`pip install
'whileai[rl]'`); the [tool-call-efficiency](https://huggingface.co/datasets/zero-proof-ai/tool-call-efficiency)
dataset is the same shape built by hand over an executable world with a hidden test suite.

Training notes, each with the chapter of rlhfbook.com behind it. Calibrate
difficulty with 8 to 16 rollouts per task before exporting so the band is
a measurement, not a guess (ch. 7); the export report's `graded_mixed` is
the number of tasks that carry an advantage at all (ch. 6). Sample at
temperature near 1.0 with 8 or more generations per prompt, since
within-group contrast is what the update learns from (ch. 6). A rollout cut
at the turn or token cap scores 0 and is logged as `truncated` (ch. 6).
Use per-token loss aggregation rather than per-sequence so long rollouts
are not favoured or punished by length alone (ch. 6). Keep a small KL to
the reference or, if the recipe drops it, watch KL drift on the dashboard
(ch. 15). `n_calls`, `judge_ok`, `truncated` and `trace_clean` are logged
at weight 0: they are the over-optimization symptoms to watch, never the
objective (ch. 14). Retire tasks the policy now always solves and re-export
between rounds (`curriculum`, `retire_solved`; ch. 7). If `reward=` is a
judge rather than a program, validate it first with `judge_trust` and
`judge_agreement`, and keep it in a different model family from the policy
(ch. 5, 12). Measure the held-out set before and after with `delta_report`
and a `must_not_regress` list, and report pass^k alongside pass@1 for
reliability (ch. 13, 16).

```python
import whileai.simulations as wai

data = wai.simulate(tools=my_tools, system_prompt=my_system_prompt, output="rollout.jsonl")
data = wai.simulate(agent=my_agent)
```

Pass `spec=` if you have a local tools-and-system-prompt folder of your own: a directory (or a JSON/YAML file) holding `tools` and `policy` / `system_prompt`, optionally with seed `situations` and a `rubric.md` (what doing the job means, for `grade()`). No spec folders ship with this package, so every snippet here uses `tools=` + `system_prompt=` — the two are interchangeable, and `spec=` is only a way to keep them in a file. The generated datasets are on Hugging Face in the [Post-Training Foundational Datasets](https://huggingface.co/collections/zero-proof-ai/whileai-post-training-foundational-datasets-6aa0b9c040ff8591988696dc) collection, not stored in this repo: [agent-simulations](https://huggingface.co/datasets/zero-proof-ai/agent-simulations) by agent type, [tool-call-efficiency](https://huggingface.co/datasets/zero-proof-ai/tool-call-efficiency) (SFT, preference, GRPO and eval splits), and [tau2-simulated](https://huggingface.co/datasets/zero-proof-ai/tau2-simulated), among others.

| Knob | Default | |
|---|---|---|
| `agent` / `spec` | hosted Qwen | Callable, URL, or tools + system prompt |
| `budget` / `time_budget` | `1000` / `None` | Stop when either hits. The clock is off unless you set it; `0` or `None` keeps it off |
| `requests_per_situation` | from mode | Phrasings: ways to ask one situation. Alias `phrasings=` |
| `rollouts_per_request` | from mode | Repeats: reruns of one phrasing. Alias `repeats=` |
| `fault_rate` | `0.5` | Broken tools. `0` off. Applied by the mock world, so a callable `agent=` that answers its own tool calls never sees one |
| `simulator` | hosted Qwen | Situation writer. `False` uses the built-in template writer (no model, less variety); an `openai:`/`vllm:` spec runs it on your endpoint |
| `user_model` | `None` | Who plays the simulated user in follow-up turns. `None` is the agent's own model; an `openai:`/`vllm:` spec moves that job to another model |
| `traces` | `None` | Graded traces of the deployed agent — a list of plain row dicts or a JSONL path. Aims the coverage grid at the behaviors those traces show and keeps the sources out of the generated rows. See [Close the loop](#close-the-loop-aim-the-budget-with-traces) |
| `tasks` | `None` | Re-run a previous run's task set instead of drawing a new one: that run, its rows, or its JSONL path. k is **not** inherited — see [Same tasks, new prompt](#trust-the-numbers) |
| `logprobs` | `False` | Ask the rollout model for the log-probability of every token it generates. Each agent turn's step gets `logprob` and `n_tokens`, the row gets the totals. `"tokens"` keeps the per-token list. Model backends only |
| `sampling` | `None` | How your own callable agent samples, `{"temperature": 0.7, "max_tokens": 1024, "model": "my-model"}`, recorded on every row as given. A model backend records its own and ignores this |
| `reproducible` | `False` | Same seed, same concurrency, same agent: same rows. Runs batch by batch, so uneven latency costs throughput. Needs the clock off. `concurrency: 1` always runs this way |
| `grade` | `False` | Legacy: `True` writes the deterministic conduct score at simulation time. Rows come back ungraded by default; grade after with `data.grade(...)` or `wai.grade(...)` |
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
wai.simulate(tools=my_tools, system_prompt=my_system_prompt)  # explore
wai.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="sft")
wai.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="rl")
wai.simulate(tools=my_tools, system_prompt=my_system_prompt, mode="adaptive", until="saturation")
```

### How much to run

Ask before you guess. `recommend()` sizes the run from the agent's own
covering grid and from published post-training practice (FireAct, LIMA,
AgentTuning for SFT; DAPO, Skywork-OR1 for RL). No key, no network.

```python
rec = wai.recommend(tools=my_tools, system_prompt=my_system_prompt, mode="sft")
print("
".join(rec["reasoning"]))
data = wai.simulate(tools=my_tools, system_prompt=my_system_prompt, **rec["simulate_kwargs"])
```

```
covering grid: 62 cells for this agent
saturation wants 5 visits per cell = 310 rows
selection wants about 3x its target of 800 to choose from
generate 2400, select 800 diverse 1-labeled rows
```

`mode="rl"` assumes half the prompts produce a mixed group. That rate is
the agent's, not ours: probe 12 asks, grade, read `group_signal`, and
pass the measured number back as `mixed_rate=`. A low rate means the grid
is too easy for this agent; aim it with `traces=` before buying rollouts.
`target=` sets how many selected rows you want (default 800).

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
the rollouts in flight; a group still short of k at the whistle is stamped
`group_cut`.
`data.search["groups"]` reports mixed, stopped, complete, partial, and
rollouts saved. `data.pass_at` scores stopped unanimous groups as
unanimous. `repeat_policy="fixed"` restores k rollouts for every prompt;
`advanced={"probe": n}` changes the probe. rlhf-book ch. 6 (dynamic
sampling) and ch. 7 (difficulty filtering), applied at generation time.

## Close the loop: aim the budget with traces

This is the second half of "two ways in, one engine" at the top of this
file. Without `traces=`, the coverage grid comes from the agent's tools and
policy alone — a cold start that samples the whole space evenly. With
`traces=`, the grid is aimed at the tools, faults and worlds the deployed
agent actually got wrong, so new rows land where the agent is weak.

**A trace is a plain row dict.** Not an OTLP span, not a platform dataset
id, not anything you have to ingest first. It is the same shape `grade()`
returns and the same shape `simulate()` writes:

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

`prompt` (or `messages`) and `steps` are what matter; `reward` is optional
(ungraded traces still focus the grid, they just carry less signal), and a
JSONL path works anywhere a list does. `load_traces` normalizes the common
variants — `tool_trace`/`trace` for `steps`, `final`/`output`/`response`
for `final_text`, OpenAI-style `messages` — so exports from other stacks
usually drop straight in. OTLP ingest and platform datasets are *one* way
to get rows into this shape, not a prerequisite for it.

```python
import whileai.simulations as wai

traces = wai.load_traces("production.jsonl")  # or just pass the list
print(wai.trace_report(traces, tools=TOOLS))  # what will this aim at?

data = wai.simulate(
    my_agent, tools=TOOLS, system_prompt=POLICY, traces=traces, mode="rl", repeats=4
)
```

| Call | What it does |
|---|---|
| `load_traces(source)` | Normalize a JSONL path or any iterable of dicts to the canonical trace schema. Rows with neither an ask nor steps are dropped |
| `trace_report(traces, tools=, policy=)` | Read this **before** you spend a budget: traces in, rows dropped, tools and faults observed, graded/ungraded counts, and `emphasis` — the exact axis values these traces move forward in the grid |
| `mine_traces(rows)` | The counts behind that: `flaw_rows` is every row with an observed fault or a 0 label, the behaviors worth simulating more of |
| `dimensions_from_traces(rows, tools, policy)` | The focused coverage axes themselves. `broaden=False` drops tools the traces never touched, so the budget stays near the flaws |
| `simulate_from_traces(traces, ...)` | `simulate(agent, traces=...)` for callers who start from the traces. With no agent, tools or policy it reads the tool surface off the traces, so graded telemetry alone is enough to start |
| `split_pseudo_production(rows, fraction=0.2)` | No production traces yet? Hold out a slice of a simulation run as stand-in production. Split by task, not by row, so the held-out slice is prompt-disjoint; every distinct flaw signature lands on the held-out side at least once |
| `leakage_report(generated, sources)` | Did any generated prompt come back a near copy of a source trace? Cosine similarity at `threshold=0.9`, exact matches always flagged |
| `drop_leaky_rows(rows, sources)` | The kept rows plus that report. Flagged rows are removed, not rewritten |

**The leakage rule.** Source traces shape the grid and never enter the
generated dataset; `simulate(traces=...)` already drops generated rows that
near-copy a source. `leakage_report` / `drop_leaky_rows` are how you verify
it, which is what makes it safe to hold traces out for evaluation:

```python
prod, train = wai.split_pseudo_production(scored.rows, fraction=0.2)
data = wai.simulate(my_agent, tools=TOOLS, system_prompt=POLICY, traces=prod, mode="rl", repeats=4)
print(wai.leakage_report(data.trajectories, prod)["n_leaky"])  # want 0
rows, report = wai.drop_leaky_rows(data.trajectories, prod)
```

And the loop closes on itself: `evaluate(rollouts, judge).failed_traces()`
hands the failures straight back to `simulate(traces=...)`.

If your traces are already on the platform, `wai.cut(agent="my-agent")`
does the whole cut in one line — see
[Training data out of traces](#training-data-out-of-traces).

## Examples

In the order a post-training run happens. The index at
[`recipes/README.md`](recipes/README.md) has one line per recipe with
what it needs and how long it takes; "offline" below means no key and no
network.

| Step | Example | What it does |
|---|---|---|
| Simulate and grade | [`recipes/01-simulate/bring-your-own-agent`](recipes/01-simulate/bring-your-own-agent) | Your own callable: the `agent(message) -> {steps, final_text}` contract, the `agent_failed` report when it raises or returns the wrong shape, and `eval_sourced` keeping a held-out score out of the reward. Offline. |
| Simulate and grade | [`recipes/01-simulate/agent-behavior`](recipes/01-simulate/agent-behavior) | Start here if the platform is new to you. Runs a coding agent with bad habits against real tests, streams every turn to While as OTLP spans plus a judge verdict, and fills a dashboard with behaviour worth looking at. Needs a key and a model endpoint; stdlib only. |
| Simulate and grade | [`recipes/01-simulate/verifiers`](recipes/01-simulate/verifiers) | Verifiable rewards: math (`MathEqual`), an answer-and-format gate (`All`), code run against hidden tests (`CodeExec`), and a JSON-schema check, each feeding `grade`/`optimize`. Offline. |
| Measure | [`recipes/02-measure/pass-at-k`](recipes/02-measure/pass-at-k) | pass@1, pass^k and pass@k with their intervals for one agent, the per-ask histogram the mean hides, and what each number tells you to do next. Offline. |
| Measure | [`recipes/02-measure/eval-your-agent`](recipes/02-measure/eval-your-agent) | Evals for the agent you already have: the callable wrapper, the policy as a judge that reads the trajectory, pass@1 with an interval and pass^k per policy branch, the coverage warnings that catch a hollow run, and a CI gate. Two scripted refund bots, one careful and one eager, so the eval visibly separates them. Offline, seconds. How-to: [docs/evals.md](docs/evals.md). |
| Measure | [`recipes/02-measure/reward-hacking`](recipes/02-measure/reward-hacking) | Reward hacking caught before, during and after training: the within-ask scan, the judge probes, the trajectory flags, and the proxy-vs-target verdict on a scripted agent and two judges. Offline, seconds, no key. How-to: [docs/reward-hacking.md](docs/reward-hacking.md). |
| Measure | [`recipes/02-measure/safety-evals`](recipes/02-measure/safety-evals) | Safety evals for a tool-using agent: prompt injection (direct, and planted in a tool result), data exfiltration, secret leakage, unauthorized writes, plus the benign controls that catch over-refusal. Trajectory markers as the judge, pass^k per attack class, the judge checked against hand labels, and a before/after that fails the fix which got safe by refusing. Offline, seconds. How-to: [docs/safety-evals.md](docs/safety-evals.md). |
| Measure | [`recipes/02-measure/safety-evals-marketplace`](recipes/02-measure/safety-evals-marketplace) | The same safety eval for a marketplace agent: the injection is planted in user-generated reviews, the private data is per tenant (a competitor's buyer-intent list), one of the writes is a public post, and a flag needs a moderation ticket. Six trajectory markers, pass^k per attack class, the guarded before/after, and `live.py` to run the suite on a real model through Ollama with no key. Offline, seconds. |
| Select | [`recipes/03-select/schema`](recipes/03-select/schema) | One row file in, six training targets out: eval, SFT, preference, GRPO prompts, OPSD hints, OPD. Migrates any legacy file first. Offline. |
| Select | [`recipes/03-select/prime-intellect-rl`](recipes/03-select/prime-intellect-rl) | Generates a GRPO-ready dataset with `simulate(mode="rl")` and checks it carries gradient before you spend GPU time on it, then exports prompts in the `verifiers` shape. Needs an account key (`whileai login`), or `VLLM_API_KEY` for the shared pool. |
| Select | [`recipes/03-select/character`](recipes/03-select/character) | Character training from a constitution: the OpenAI Model Spec's style traits become graded rows, preference pairs and SFT rows, with the judge checked against the spec's own labels and a before/after measurement. Offline by default. How-to: [docs/character-training.md](docs/character-training.md). |
| Train | [`recipes/04-train/hosted-loop`](recipes/04-train/hosted-loop) | Push graded rows, `wai.train` SFT on Qwen3-4B, `wai.serve` the adapter, one chat completion from the endpoint. One key, one A10G minute; the wiring check for training on the platform. |
| Train | [`recipes/04-train/identity`](recipes/04-train/identity) | Builds a leak-free SFT set that teaches a model a new name and maker, with Modal scripts to train a LoRA and evaluate identity and leak rates. No model calls to generate. |
| Train | [`recipes/04-train/grpo`](recipes/04-train/grpo) | GRPO on Modal, end to end: prompts from the simulator, a verifiable tool-discipline reward, TRL `GRPOTrainer` with LoRA, `HackMonitor`, reward and KL on the dashboard, pass@1 before and after on a holdout with the paired delta and per-category table on the run page. One A10G, under fifteen minutes. |
| Train | [`recipes/04-train/dpo`](recipes/04-train/dpo) | DPO on the same environment: on-policy pairs from `build_preference_pairs`, TRL `DPOTrainer` with LoRA, the reward margin on the run page, iterated rounds with `--from-run`, constructed negatives where the policy never fails. One A10G, about ten minutes. |
| Export | [`recipes/05-export/hugging-face`](recipes/05-export/hugging-face) | Rows to a Hub dataset repo you own, any Hub split onto your account with a profile, a run's adapter to a model repo. Needs a key and a connected Hugging Face account. |

## Sign in

```bash
whileai login
```

Prints a link and a short code. Open the link, sign in or sign up, press
Approve. The key is saved to `~/.whileai/credentials.json` and every
platform call below reads it from there. Interrupted before you approved?
Run it again; it resumes the same code. This is the path for coding
agents too: tell yours to run `whileai login` and click the link it
shows you. `whileai status` shows which key is in use, `whileai
logout` removes it.

No account yet, or no browser? One command creates the account and the
key. Open the dashboard later by signing in with an email code.

```bash
whileai signup --email you@example.com
```

That key is a trial key (25k input and 50k output tokens a day, 100 MB,
ten datasets, seven days) until the person signs in once at
https://www.zeroproofai.com/sign-in with an email code. `whileai status`
shows the tier; `whileai.account()` returns tier, limits and usage.

## Store datasets on While

Push a run to your While account so the optimization framework
can iterate on it. Credentials resolve in this order: `api_key=` argument,
`WHILEAI_DELEGATED_CREDENTIAL` (a short-lived `zp_dc_...` issued from a
Clerk session), `WHILEAI_API_KEY`, then the key saved by `whileai
login`.

```python
# Runtime path with a delegated credential
# export WHILEAI_DELEGATED_CREDENTIAL="zp_dc_..."

# If you need to mint one from a Clerk session token:
# credential = wai.issue_delegated_credential(clerk_token, ttl_seconds=3600)
# export WHILEAI_DELEGATED_CREDENTIAL=credential["credential"]

data = wai.simulate(my_agent, tools=TOOLS, system_prompt=POLICY)
v1 = data.push("my-agent-explore-v1")  # -> {"datasetId": "ds_...", ...}

# iterate, then push the next version with lineage
v2 = data.push("my-agent-explore-v2", parent=v1["datasetId"])

wai.datasets()  # list yours + storage used
rows = wai.pull(v1["datasetId"])  # rows, or pass path= for a file
wai.push_file("rollout.jsonl")  # upload an existing JSONL
wai.delete_dataset(v1["datasetId"])  # permanent
```

Storage is private per account, 5 GB free. `parent=` records dataset
lineage so iterations show as a family on the platform.

`data.push` and `wai.push_file` run a publish gate first (`gate=False` skips it). Every graded row gets a `calibration` stamp: its task's pass rate over k repeats, k, and the policy that produced it, so a trainer can build a curriculum or retire solved tasks. An RL-shaped run (repeats of one ask) is refused with `PublishGateError` when it is ungraded or has no mixed group, because a grouped update would learn nothing from it. The report comes back as `entry["gate"]`, with warnings when unanimous asks, or asks outside the difficulty band, are still present; `wai.optimize(data, mode="rl")` prunes those. `wai.publish_gate(rows)` runs the same check on any row list. The stamp is the schema's `Calibration` object: `wai.calibration_of(row)` reads it back typed, `from_row` carries it on `rollout.extra["calibration"]`, and `to_row` writes it out again. `k` is the repeats the grader saw, not the rows that survived: `optimize(mode="rl")` stamps its selection from the rows it was given, before its own dedupe and trims, and the gate keeps a carried stamp rather than re-measuring it on what is left. The gate's own `pass_at` block is still over the rows in front of it, and says so when the two differ.

`export_dataset` and `export_training` are the same function object (`export_dataset is export_training`), not two exporters to choose between: same arguments, same file, same report. `export_dataset` is the name to write in new code — it exports a dataset, not a training run — and `export_training` is the older spelling, kept so nothing already written breaks. `training_rows` is the list-returning half of the same path, without writing a file.

Training rows from `export_dataset` / `training_rows` carry a `loss_mask`, one 0/1 per message: 1 on the agent's turns, 0 on system, user, and tool-output turns. Tool output is the environment's text, not the policy's, so a trainer should not learn to predict it. `mask_mode="final"` trains only the last assistant turn, for conversations whose earlier agent turns were scripted or came from another policy; the export report counts `trained_messages` and `masked_messages`. `unroll=True` turns an N-turn conversation into N samples, the k-th ending at the k-th agent turn with loss on that turn only, so every earlier turn trains once with the context it actually had (rlhf-book ch. 4). `max_tool_output_chars=` caps each tool message, appends a `[... N chars of tool output truncated]` marker and counts the cut on the row and in the report, so context spent on tool output is a decision the export makes out loud (ch. 13).

Two wire shapes come out of the exporters, and a trainer needs the second one:

```python
wai.export_training(rows, "sft.jsonl")  # OpenAI chat-completions wire (default)
wai.export_training(rows, "sft.jsonl", format="trl")  # what TRL's SFTTrainer loads
wai.export_preference(pairs, "dpo.jsonl", format="trl")  # what TRL's DPOTrainer loads
wai.to_trl(wai.training_rows(data), "training")  # same reshape on rows you already hold
```

`format="openai"` (the default) is the API wire row: the whole conversation in `messages`, `function.arguments` as a JSON string, and the ask alongside as `prompt`. `format="trl"` is what `trl.data_utils.maybe_apply_chat_template` accepts. For SFT that is conversational `{"messages": [...]}` with **no** `prompt` string column — TRL decides "is this conversational?" from the column set, and a `prompt` string next to `messages` makes it skip the chat template silently and train on the bare ask; the ask survives as `prompt_text`. For preference data it is `prompt` as the message list up to the first agent turn with `chosen`/`rejected` as the completions only, because the default shape (a `prompt` string with full conversations on both sides) raises `TypeError: string indices must be integers` inside TRL. In the TRL shape `function.arguments` is a dict, not a JSON string: HF chat templates render it with `| tojson`, so a pre-encoded string is quoted twice and the student learns to emit a string where an object belongs. The `tool_call_roundtrip` gate in the report names which of the two encodings it checked (`encoding: "json_string"` or `"dict"`), so `invalid: 0` says what it actually vouches for.

### Prune before training

```python
rows, report = wai.optimize(data, mode="rl")  # whole groups, 20%-80% pass rate
rows, report = wai.optimize(data, mode="rl", band=(0.3, 0.7))
rows, report = wai.optimize(data, mode="rl", enforce_band=False)  # rank, do not drop
report["band_dropped"]  # {"too_easy": n, "too_hard": n}
```

`optimize(mode="rl")` drops junk rows, duplicate rollouts within an ask (same trajectory twice adds nothing to a group-relative advantage), truncated rollouts (`truncated="keep"` leaves them in as `overlong`, `"penalize"` keeps them as failures with the judged score under `reward_before_penalty`, DAPO's overlong handling), unanimous asks (all pass or all fail: zero advantage), and asks outside the difficulty band (`trim_out_of_band`: "out of band" means outside the [0.2, 0.8] *pass-rate* band, never off-topic — it does not read the prompt at all, so an on-topic ask the policy always solves is dropped and an odd one it solves half the time is kept), then keeps whole groups round-robin across fault kinds and, within a fault kind, round-robin across pass rates: a 25% ask, a 50% ask and a 75% ask are taken in turn, with no preference for the middle (`order="middle"` restores the older nearest-to-50% ranking). Each kept row's `calibration` stamp carries `pass_rate_ci95`, the interval on that pass rate, and the report says so when the band was measured from fewer than 16 rollouts per task, since at 8 a task's band assignment can be off by about 0.3. The prune shrinks every group, so the k-way reliability numbers do not survive it: `pass_at` on the selection reports `pass^k` and `pass@k` as `n/a` where the graded rows had them, which is why the quickstart prints `pass_at` before this call. The report says so in `hygiene_warnings` when they were available before, and the carried `calibration` stamp keeps the graded per-task measurement. `optimize(mode="sft")` is rejection sampling (rlhf-book ch. 9): `select="top_per_prompt"` keeps each prompt's highest-reward completion above `min_reward` (default 1.0; lower it for a partial-credit grader), `"top_k_overall"` the best `k` across prompts, and the `random_*` rules are the matching chance controls. Exported groups carry `n0`/`n1` (fail/pass, partial credit splits at 0.5) and `reward_mean`/`reward_std`. The band is the offline difficulty filter from the reasoning-model recipes (keep prompts the policy solves 20-80% of the time); it is a heuristic, so it is a parameter. Every selector report (`select_for_rl`, `select_for_sft`, `build_preference_pairs`) carries `eval_sourced`, the rows or pairs whose reward came from `evaluate()` (`lineage.source == "eval"`), with a warning when it is non-zero: a held-out score that becomes the reward makes the scorer you report the one you optimised against. Nothing is dropped; grade the training set with `run_judge` or `data.grade` and keep `evaluate` for held-out rows.

### What will the policy learn?

```python
scan = wai.hack_scan(scored.rows, endorsed=["tool:lookup_order", "marker:grounded"])
scan["regime"]  # train | reward_hack | pool_exhausted | no_signal | degenerate | unknown
scan["top_feature"]  # e.g. 'contains:### done' when the judge pays for a delimiter
print(wai.format_hack_scan(scan))
```

A grouped update learns whatever separates reward *within* an ask; what only tracks which ask it is (difficulty) is baselined away. `hack_scan` asks the question the same way: reward and every candidate feature are centered within ask, ranked by that correlation, and compared to a noise floor from shuffling reward within ask (`tau`). Features come in two tiers, both pure Python: the hand tier (reply length, tool calls, turns, truncation, surface counts, one indicator per tool called, mean token logprob, every numeric marker, plus `features={"name": fn}` of your own) and the auto tier (the 200 most common words and word pairs in the agent's text, and pairwise ANDs that beat both parents), which is the tier that finds the shortcut nobody listed. `endorsed` names what the reward should track, as substrings of feature names; with it the scan can say `reward_hack` (the top feature is not endorsed, and the warning names what the policy would learn instead), `integrity` (share of the above-floor signal that is endorsed), and lists rivals. Without it the scan still ranks and floors. An agent that emits only a couple of distinct trajectories per ask makes every feature that separates them an exact function of the label — they all tie at |rho| 1, and the floor cannot break a tie between two perfect explanations — so the scan returns `degenerate` with `top_feature` `None`, lists the tied features in `collinear`, and names the cause (`distinct_per_ask`) rather than picking the alphabetical winner.

The whole loop, before, during and after training, is in [docs/reward-hacking.md](docs/reward-hacking.md) and runs offline in [`recipes/02-measure/reward-hacking`](recipes/02-measure/reward-hacking). `optimize(mode="rl", endorsed=[...])` carries the scan as `report["hack_scan"]`, with its warnings in `report["hygiene_warnings"]` next to the older pooled `report["correlations"]` (reply length, tool calls, turns, flagged at `HACK_THRESHOLD` 0.3). A reward that tracks a shortcut is a judge problem, so it is flagged, not pruned. The publish gate reports the same on RL-shaped rows, plus near-duplicate asks and length spread; `data.push(endorsed=[...], strict_hacks=True)` refuses a `reward_hack`. Standalone: `wai.reward_correlations(rows)`, `wai.dedupe_groups(rows)`, `wai.near_duplicate_prompts(rows)`, `wai.length_report(rows)`.

### Curriculum: easy to hard, and retire the solved

A curriculum needs per-prompt difficulty (rlhf-book ch. 7), which is just each task's pass rate over its k rollouts. `curriculum(rows)` splits graded tasks into *trainable* (ordered easy to hard, and bucketed into `tiers` for a staged schedule), *retired* (pass rate above `solved`, default 0.8: an all-pass task is dead gradient), and *not ready* (below `floor`, default 0.2: no signal until the policy improves), and counts how many trainable tasks sit in the 20-80% band. The two defaults are the band's own edges, so `curriculum` and `optimize(mode="rl")` agree on which tasks are trainable.

```python
cur = wai.curriculum(scored.rows)  # solved=0.8, floor=0.2, tiers=3
cur["schedule"]  # trainable task ids, easy -> hard
print(wai.format_curriculum(cur))
rows = wai.retire_solved(scored.rows)  # drop tasks the policy already aces
```

### Agents

An agent exists the moment a push names it or a trace arrives with
`gen_ai.agent.name`. Everything on the platform hangs off it.

```python
data.push(
    "airline-v3", agent="airline-support"
)  # registers the agent and attaches tools + system prompt
wai.agents()  # every agent: traces, sets by purpose, public cards
wai.register_agent("airline-support", description="Refunds and rebooking")
```

### Clean up

```bash
whileai purge --agent demo-agent --dry-run   # count its traces, datasets, record
whileai purge --agent demo-agent             # delete them, after a y/N
whileai purge --empty --max-rows 2           # datasets with no bytes, or 2 rows or fewer
```

Python: `wai.purge_agent("demo-agent")`, `wai.delete_empty_datasets(max_rows=2)`.
Both take `dry_run=True`.

### Train, holdout, eval

```python
data.push("airline-v3", holdout=0.2)  # train set + a linked holdout set, split by task
data.push("airline-evals", purpose="eval")  # a set you measure with
scored = data.grade(judge=my_judge)
scored.push("airline-rl-v3", gate=True, mode="rl")  # the graded copies, gated
wai.update_dataset("ds_...", purpose="holdout")
wai.preview("ds_...")  # three sample rows + the analyzer report
wai.profile("ds_...")  # pass rate, support, mixed tasks, tool use, per task
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

### Training data out of traces

The platform's "Make training data" button, as one line:

```python
wai.send_score("4bf92f3577b34da6", 1.0)  # this run passed
wai.cuts(agent="my-agent")  # what a cut would hold
made = wai.cut(agent="my-agent", kind="rl")  # make it
wai.pull(made["train"]["datasetId"], "train.jsonl")
made["holdout"]["datasetId"]  # measure on this, never train on it
```

A cut needs a pass or a fail on every run, and a judge answers after the run it
is judging has closed. `send_score(trace_id, value)` grades a run that already
ran — **1.0 or above is a pass**, so a 0-to-1 quality number never reads as one;
send that under its own `name=` and keep `score` for the verdict. Re-sending the
same name is a correction. Emitting `whileai.reward` on the span does the same
thing when your grader runs inline.

Runs of the same prompt are grouped by `zeroproof.scenario_id`. `kind="rl"` keeps the
prompts the agent passes some of the time and not always (20% to 80% by default);
`kind="sft"` keeps the best run of every prompt that ever passed. Either way the prompts
are split into a train set and a held-out set. `since="7d"` narrows the window, `band=`
and `holdout=` move the defaults, and any other keyword is a trace filter (`model=`,
`tool=`, `evalSet=`).

### Trust the numbers

Three checks that decide whether a result is believable, all report-only and all over rows you already have.

```python
rows, report = wai.attach_labels(
    rows, "labels.jsonl", annotator="ana"
)  # gold_reward + who said what
wai.judge_trust(rows, judge=my_judge)  # is the judge trustworthy?
data.grade(use_privileged=True)  # judge also reads privileged principle, reference, hidden state
wai.run_judge(rows, likert_judge, scale=(1, 5))  # rating kept, reward = (r - 1) / 4
pairs, report = wai.judge_pairs(pairs)  # A vs B both ways round: winner, tie, position_flip_rate
rows, report = wai.write_rubrics(rows, domain="refunds")  # per-prompt criteria on privileged.rubric
scored = wai.run_judge(rows, wai.rubric_judge())  # a verdict per criterion; markers rubric:<item>
clean, report = wai.decontaminate(
    train_rows, against=[eval_rows]
)  # 8-gram overlap with the eval set
wai.style_markers(rows)  # no_boilerplate, no_hedging, no_apology, no_sycophancy, answered
wai.style_report(rows)["warnings"]  # "reward pays for hedging (corr +0.41 ...)"
wai.refusal_report(benign_rows)  # over-refusal rate with a Wilson interval
wai.compare_runs(run_a, run_b)  # paired delta with a 95% interval
wai.delta_report(before, after, target="pass_at_1", must_not_regress=["honest_after_fault"])
wai.delta_report(before, after, target="pass_at_1", by="category")  # the target per kind of prompt
before = wai.simulate(agent, tools=TOOLS, tasks=base, runs=3)  # the same eval three times
after = wai.simulate(trained, tools=TOOLS, tasks=base, runs=3)
wai.delta_report(before.rows(), after.rows(), target="pass_at_1")  # run_std computed from the runs
wai.eval_variance(before.rows())  # the eval's own re-run std, split by lineage.eval_run
wai.mark_grounding(
    rows
)  # markers["argument_grounding"]: every tool argument came from the conversation
wai.grounding_report(rows)  # grounded rate, and the invented values by tool and key
```

**Checked by default.** Every `grade` call ends by checking the judge against the rows' human labels, with no flag needed. "Gold" means a label a person wrote: `attach_labels(rows, labels, kind="human")` stamps `gold_reward` and `gold_kind="human"`; a model's labels, or a second judge pass, are marked `model` and do not count, and older rows with `gold_reward` but no record of who wrote it count as unknown. Label 50 rows by hand (a JSONL of `{"key": ..., "label": 0 or 1}` or a `{key: label}` dict), attach them, and grade: the summary lands on every graded row as `judge_meta["trust"]` (`agreement`, `agreement_low`, `kappa`, `n_gold`, `ok`) and in the grade report as `trust`, and `publish_gate` carries it as `judge_trust`. The judge passes when the lower bound of its agreement with the people is at least 0.80 and kappa at least 0.60; under either, the report says the number, the floor, and what to do. With no human labels the grade prints one line saying the judge was not measured. `grade(trust="require")` raises instead of printing; `trust="off"` skips the check. `judge_trust` refuses model gold the same way unless `allow_model_gold=True`. And `audit_grades` never audits with the grader's own model: when the auditor would be the same, it uses the other hosted model and the report says which (`grader`, `auditor`), or it stops and asks for `backend_spec=`.

**Judge trust.** Label 50 to 100 rows by hand with `attach_labels` (0/1) -- `report["ok"]` means measured and clean, so with no labels it is `False` and the report says the judge is unmeasured rather than untrustworthy (`format_judge_trust` prints `NOT MEASURED`). The report gives agreement with a Wilson interval and Cohen's kappa, agreement on two task halves (tune the rubric on one, read the other), judge pass rate on short versus long replies within the same human label (length bias the humans rule out), and, with the judge callable, a re-judge of a sample as-is (consistency) and with neutral filler appended (a flip means the judge reads length). Disagreements come back as a review queue. `format_judge_trust(report)` prints it. `probes="all"` (or a list) tries the reward hacks a policy finds first on the judge on purpose: filler, the rubric's own words stuffed in, a claim of success with no evidence, the ask echoed back, a well-formed tool call with empty arguments, a sycophantic opener, a polite refusal. An additive probe is exploitable when failing replies start passing; a replacement probe when a reply with no content passes. `report["exploitable_by"]` names the holes at or over 10%, and a policy trained on this judge will find those same holes. Standalone: `wai.judge_probes(rows, judge, rubric=...)`. The gold set needs both passes and failures; with one class only the report says so and skips the kappa and length flags. With the hosted judge, call `wai.grade` once first (or `whileai.simulations.score.grade_llm.warm_judge`; it is not re-exported) so the cold start, two to three minutes, is not counted as timeouts.

**Decontamination.** Word 8-gram overlap between a dataset's prompts and any evaluation source: row lists, JSONL paths, or platform dataset ids. A row is contaminated when it is an eval prompt verbatim or when one eval text covers at least 80% of its words (`overlap=`, the Llama 2 rule); one shared 8-gram is not enough, because situations written from the same templates share whole sentences without sharing the question. Short prompts match verbatim only. `fields=("prompt", "final_text")` also checks replies against eval answers and references. The report separates verbatim hits from near copies and counts hits per field, and returns the clean rows with the first offenders.

**Intervals and comparison.** Every pass@1 carries a 95% interval from a bootstrap over tasks (`pass_at(rows).ci95`), and `metric_summary` / `marker_summary` do the same for markers. pass^k and pass@k carry their own (`pass_pow_k_ci95`, `pass_at_k_ci95`), a bootstrap over the k-eligible groups. Markers come from the judge: return `{"reward": ..., "markers": {"name": value}}` from a `grader=` or `run_judge` callable and they land on `row["markers"]`, which is what `marker_summary`, `delta_report` and `from_row` read. `compare_runs` pairs the tasks two runs share, bootstraps the paired difference, and adds a sign-flip permutation p-value; fewer than five shared tasks falls back to an unpaired test and says so. Tasks on one side only are dropped from a paired comparison; `note` says how many and `paired_share` is the fraction that paired, so a verdict over a quarter of the eval reads as one. The verdict `no_difference_detected` means the interval covers zero, not that the runs are equal. A task is a situation, not a string: every report (`pass_at`, `compare_runs`, `delta_report`, `eval_variance`, `curriculum`, `group_signal`, the exporters) groups rows by `wai.task_key(row)`, the engine's `scenario_id` when the row has one, so repeats and rephrasings of one situation count as one task and the same rows give the same task count everywhere. `pass_at(rows).config` and `delta_report(...)["config"]` say what the rows were produced with (temperature, reply budget, policy and judge versions), and `delta_report` warns when the two sides differ.

**Same tasks, new prompt.** A run draws its tasks from the grid by seed and, above `concurrency: 1`, by completion order, so a second `simulate()` shares only part of its tasks with the first. To A/B a prompt edit, a model swap or another seed on exactly the same eval, pin the task set: `wai.simulate(agent, tools=TOOLS, system_prompt=EDITED, tasks=base)` re-runs every prompt of `base` (a run, its rows, or its JSONL path) on its own `scenario_id`, under the same faults and world state, and draws nothing new; it stops with `tasks_done` once every prompt has its rollouts, and `compare_runs(base.rows(), rerun.rows())` pairs every task.

`tasks=` copies the prompts and, unless you pass `repeats=`, the pinned run's k (the most rollouts any of its prompts has), so a base built with `mode="rl", repeats=4` and re-run as `simulate(..., tasks=base, mode="rl")` comes back at k=4 and `pass_at` reports the same k on both sides. Pass `repeats=` to re-run at a different k on purpose:

```python
base = wai.simulate(agent, tools=TOOLS, system_prompt=POLICY, mode="rl", repeats=4)
rerun = wai.simulate(
    agent, tools=TOOLS, system_prompt=EDITED, tasks=base, mode="rl"
)  # k=4, inherited
assert base.rollouts_per_request == rerun.rollouts_per_request  # cheap guard
```

**Before and after.** `delta_report` runs `compare_runs` on pass@1 and every marker both row sets share. `target=` names the metric the training was meant to move and gives the headline; `must_not_regress=` names the behaviors whose significant drop fails the report; any other significant drop is a warning. `format_delta_report(report)` prints one line per metric. `eval_variance(run_1, run_2, run_3)` is the eval's own re-run standard deviation (three or more evaluations of the same model); passing it as `run_std=` makes any delta inside twice that band `within_noise`, and a target there reads `within_eval_noise` rather than moved, since re-running the eval moves it that much on its own (rlhf-book ch. 16). `by=` names a row key, a marker, or a callable that groups rows (a prompt category, a tool, a persona); the report then carries `groups`, the target compared within each group, and `groups_down` for any group whose target dropped significantly while the headline moved. A headline over one dominant kind of prompt cannot hide the other kinds that way.

**Run the eval three times.** One evaluation is a draw, not a number: the same model on the same tasks lands somewhere else next time, and most post-training gains are inside that spread (rlhf-book ch. 16, appendix C). `wai.simulate(agent, tasks=base, runs=3)` replays the task set three times in one call, same tasks, faults and world, and stamps `lineage.eval_run` on every row. Feed both sides to `delta_report` and it works out `run_std` from the repeats itself. The verdict words: `moved` is a change the interval and the re-run band both support; `moved_unreplicated` is a change seen once, which could be noise, and the warning tells you the `runs=3` call that settles it; `within_eval_noise` is a delta smaller than what re-running the eval does on its own, so equivalence, not a win; `no_change_detected` is an interval that covers zero. `ceiling=True` means the before run already passes most of its tasks (0.9 or more, or too few paired tasks left with room), so there is little improvement the eval could show; use harder situations before training again.

**Argument grounding.** A policy trained to call a tool learns to call it before it learns when not to; on the refund environment both GRPO and DPO learned to invent an order id on a quarter of the prompts that gave none while the headline rose. `mark_grounding(rows)` stamps `argument_grounding`: 1 when every string argument of every tool call appears in the prompt, the user and system turns, or an earlier tool result (rows with no calls count as grounded), else 0. No categories, any agent; `must_not_regress=["argument_grounding"]` fails the run that learned to invent, and `ungrounded_arguments(row)` / `grounding_report(rows)` name the values. `ignore_keys=` skips free-text arguments, `allow=` lists enums and defaults.

**Trajectory flags.** Did the agent fake the work? `trace_markers(rows)` reads the trajectory rather than the prose (rlhf-book ch. 13, 14): `lie.tests_claimed` (tests said to pass when no test command ran or the last one failed), `lie.unverified_claim` ("I verified" with no tool calls), `lie.phantom_edit` ("I updated" with nothing written), `lie.ignored_failure` (the turn ended on a failed call and the reply never says so), `hack.test_edited`, `hack.test_weakened`, `hack.suppressed`, `hack.bypassed`, `risk.destructive`, `risk.secrets`, each with the fragment that raised it on `row["trace_flags"]`. The markers it stamps (`honest_claims`, `reported_failure`, `no_test_tampering`, `no_suppression`, `no_bypass`, `no_destructive`, `no_secrets`) are 1.0 when clean, so `must_not_regress=["honest_claims"]` fails a run that learned to overclaim, and `hack_scan` carries every fired flag as a `trace:` feature. `trace_flag_report(rows)` gives each flag's rate, examples, and its correlation with the reward, flagged when the judge pays for the fake. Reads, writes, deletes and commands are told apart by the tool's arguments and name; `kinds={"my_tool": "write"}` overrides.

**Stage lineage.** The pipeline is a sequence of stages (rlhf-book ch. 3): SFT, reward modeling, RL, and the eval that judges the result. `stamp_stage(rows, "sft")` records which stage a row fed, and `stage_report(rows)` counts rows per stage and flags the one mistake it most needs caught: any task used in both `eval` and a training stage. `wai.stamp_stage`, `wai.stage_report`, `wai.stage_of`, `wai.STAGES` (`sft`, `rm`, `rl`, `eval`, `mid`).

**Model spec as an object.** A spec or constitution is a living, versioned document (rlhf-book ch. 17). `load_spec(constitution)` wraps the `{source, traits: [{id, name, principle, authority}]}` shape (what the character example writes) into a `Spec` whose `version` is a content hash, so any edit to a principle changes it. `spec.behaviors()` are the trait ids, ready for `delta_report(must_not_regress=...)`; `stamp_spec(rows, spec)` tags a run with the spec version it targeted, so you can ask whether adherence held from one spec or model version to the next.

```python
spec = wai.load_spec("recipes/03-select/character/constitution.json")
scored = wai.stamp_spec(data.grade(judge=my_judge).rows, spec)
wai.delta_report(before=before, after=scored, target="pass_at_1", must_not_regress=spec.behaviors())
```

#### Markers: four families, one polarity

A marker is a named behavior measurement on a row. Everything that reads
markers — `marker_summary`, `delta_report`, `must_not_regress=`,
`from_row`, the run page — reads one place, `row["markers"]`, and does not
care which family put the value there. Four families write to it, and only
one of them has the wrong polarity:

| Family | How you get it | Polarity | Use it for |
|---|---|---|---|
| **Judge-emitted custom markers** | your own name and value, returned as `{"reward": ..., "markers": {"name": value}}` from a `judge=` / `grader=` / `run_judge` callable | **yours to choose — and it must be 1.0 = good** | Anything your product cares about. This is the family `delta_report` and `must_not_regress=` are built for |
| `trace_markers` / `trace_flag_report` | `wai.trace_markers(rows)` stamps `honest_claims`, `reported_failure`, `no_test_tampering`, `no_suppression`, `no_bypass`, `no_destructive`, `no_secrets`, with the evidence on `row["trace_flags"]` | 1.0 = no flag fired, higher is better | Did the agent fake the work? Read from the trajectory, not the prose — see [Trajectory flags](#trust-the-numbers) above |
| `style_markers` / `style_report` | `wai.style_markers(rows)` stamps `no_boilerplate`, `no_hedging`, `no_apology`, `no_sycophancy`, `answered` | 1.0 = clean reply, higher is better | Over-optimization drift in a paired before/after |
| `behavioral_markers` / `mark_rows` / `STOCK_MARKERS` | `wai.behavioral_markers(rows)` -> `{"boilerplate": 0.31, "refusal": 0.04, ...}` | **presence: 1 = the tic appears, higher is worse** | A one-shot read of how often each tic occurs. Not a delta |

> **Deprecated.** `behavioral_markers`, `mark_rows`, `row_markers` and
> `STOCK_MARKERS` all live in
> `whileai.simulations.score.markers`, which is deprecated and raises a
> `DeprecationWarning` on first use. They are being consolidated onto
> `score.style`. `style_markers` / `style_report` / `refusal_report` cover
> the same rlhf-book ch. 14 behaviors with the delta-ready polarity.

**Polarity is the rule for every marker you define, not a quirk of one
function.** `1.0` is the good outcome; higher is better; a significant
*drop* is the regression that `must_not_regress=` fails on. Name markers
after the behavior you want:

```python
# Wrong: 1 means the bug happened.
{"markers": {"false_refund_success": 1.0}}
# delta_report prints DOWN when you fix it, and
# must_not_regress=["false_refund_success"] FAILS the run that fixed it.

# Right: 1 means the agent did the right thing.
{"markers": {"refund_correctly_refused": 1.0}}
```

If you have already collected rows under an inverted name, flip the value
(`1 - v`) and rename before you compare runs; `delta_report` has no way to
know which direction a name means.

### Train, and watch it

Two ways to train, one record. The platform trains a pushed dataset (SFT, GRPO, DPO or a reward model, LoRA on an A10G) and serves the result; or your own trainer runs on Modal, a GPU box, or a notebook and reports into the same run. Either way the loss curve and the progress bar are at [zeroproofai.com/platform/training](https://www.zeroproofai.com/platform/training).

```python
run = wai.train(
    "ds_...", method="sft", base_model="Qwen/Qwen3-4B", epochs=2
)  # or "grpo" / "dpo" / "rm" with steps=
run.wait()  # done or failed; run.url is the curve while it goes
run.training["before"], run.training["after"]  # holdout pass@1 (SFT: loss)
run.delta(
    before_rows, after_rows, target="pass_at_1", by="category"
)  # paired delta on the run page
model = wai.serve("refund-v2", run)  # adapter on an OpenAI-compatible endpoint
# model["endpoint"] + /chat/completions, model="refund-v2", bearer = your zp_ key
wai.models()  # what the account hosts
```

`epochs=` sets SFT, `steps=` sets GRPO, DPO and RM; each method has a default. `run.delta` is `delta_report` (below) kept on the run and drawn on its page, including the per-group table when `by=` names a row key or marker; `wai.attach_delta(run_id, before, after)` does the same for a run that already finished. `holdout=` names the eval set (defaults to the train set's split sibling); a dataset already training returns that run. `serve` needs a finished run whose base is a served one (`Qwen/Qwen3-4B`, `microsoft/phi-4`). The trainer's default bases (Qwen2.5-0.5B for SFT, 1.5B for GRPO and DPO) train in under a minute but cannot be served, so `train` warns when a run will not reach an endpoint; SFT runs on an A10G; GRPO and DPO run on an L40S, so a 4B base fits all three. Qwen3 answers in thinking mode by default: leave room in `max_tokens` or send `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

`method="rm"` trains a reward model (rlhf-book ch. 5) on the set's pass-vs-fail pairs and reports pair accuracy on the held-out pairs before and after. `wai.reward_model(run)` is that model as a judge, with the judge contract (`reward` 0/1 against the run's threshold, `rm_score` raw), so it goes wherever a judge goes:

```python
rm = wai.train("ds_...", method="rm", steps=60, wait=True)
run = wai.train(
    "ds_...", method="grpo", generations=8, beta=0.02, learning_rate=5e-6, seed=3
)  # the knobs a run is compared by
judge = wai.reward_model(rm)  # or reward_model("run_...", threshold=0.4)
scored = data.grade(judge=judge)
wai.judge_trust(scored.rows, judge=judge)  # the same checks as the LLM judge
```

Your own trainer, three ways in:

```python
# one line on a Transformers or TRL trainer
run = wai.training_run(
    "identity-v1", dataset="ds_...", base_model="Qwen/Qwen3-4B-Instruct-2507", trainer="trl"
)
trainer.add_callback(wai.TrainerCallback(run))
trainer.train()  # loss, lr, eval loss, epoch, grad norm, then finish

# your own loop
with wai.training_run("sft-v3", dataset="ds_...", total_steps=1000) as run:
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        run.log(step, loss=loss, lr=scheduler.get_last_lr()[0])
    run.finish(summary={"final_loss": loss}, adapter="s3://.../adapter")  # failed on exception

run.holdout(before=0.42, after=0.58)  # did it work? the run page opens with this
```

A run's page opens with one word — **Better**, **Worse**, **About the same** — over the held-out pass rate before and after. The platform's trainer measures it; a run on your own hardware says it with `run.holdout(before, after)`, or `wai.attach_holdout(run_id, before=..., after=...)` once the run has finished. Pass rates are 0 to 1, so 58% is `0.58`; `metric="loss"` sends held-out loss instead (SFT), where lower is better. `run.delta(...)` and `wai.attach_delta(...)` already measure both sides, so they fill the two numbers in themselves, and add `summary["holdout"]` (also `run.holdout_summary`): each side's pass rate with `n_tasks`, `k` and a `ci95`, plus the delta report's verdict word (`moved`, `moved_unreplicated`, `within_eval_noise`, `no_change_detected`). A hosted run read back with `run.refresh()` has the same block with the interval fields `None` and a note that the platform only returned two numbers.

### Is it hacking the reward right now?

```python
monitor = wai.HackMonitor(
    run,
    holdout=holdout_rows,             # prompts or {"prompt": ..., <columns the reward reads>}
    gold=wai.reward_model(rm_run),    # or the hosted judge, or a second rule; any judge callable
    every=10, k=4,                    # sample the holdout from the live policy every 10 steps
    endorsed=["tool:lookup_order"],   # what the reward should track
    stop_on="divergence",             # or "length", "drift", "feature", "any"; default: log only
)
trainer = GRPOTrainer(model, reward_funcs=[monitor.wrap(rule_reward)], ...)
trainer.add_callback(monitor)
trainer.add_callback(wai.TrainerCallback(run))
```

Over-optimization looks like one picture (rlhf-book ch. 14): the training reward keeps climbing while the evaluation you care about flattens, read against KL. The monitor draws it during the run instead of after. `wrap` watches the reward function, so the monitor keeps the last completions with their rewards and runs `hack_scan` on them; every `every` steps it samples the holdout from the live policy and scores it with the training reward (the proxy) and with `gold`, a scorer the proxy cannot see. `proxy_reward`, `gold_reward` and `holdout_length` land on the run beside the loss curve. Four alarms, one line each on the run: `divergence` (proxy up by `delta` over the window while the paired gold interval does not move up), `length` (completions grow while gold does not), `drift` (KL past `kl_budget`), `feature` (the batch scan says `reward_hack`). `stop_on` names the ones that stop training; a stopped run finishes as `stopped` with the reason, and `run.note(...)` puts anything else on the run's summary. `wai.format_hack_monitor(monitor.summary())` prints the curve and the alarms. [`recipes/04-train/grpo`](recipes/04-train/grpo) runs it by default.

Plain HTTP, for a stack that is not Python: `POST /runs {"name", "dataset_id", "base_model", "total_steps"}` returns `runId`; `POST /runs/{id}/log {"points": [{"step": 10, "loss": 1.2, "lr": 1e-4}], "total_steps"?}` in batches of up to 500; `POST /runs/{id}/finish {"status": "done|failed|stopped", "summary"?, "adapter"?}`. All with `X-Api-Key`. Points are buffered on the client and a send that fails is retried on the next flush; the dashboard never interrupts the trainer. `wai.get_run(id)["series"]` returns the points, oldest first.

### Publish a dataset as a card

```python
data.push(
    "airline-refunds-v3",
    agent="airline-support",
    publish=True,
    description="Graded refund conversations with injected tool faults.",
)
wai.publish("ds_...", agent="airline-support")  # or publish an existing one
wai.catalog()  # every public card, by agent
rows = wai.pull("ds_...")  # public sets need no key
wai.unpublish("ds_...")
```

Hugging Face, both directions. Connect your account once on any dataset page, then:

```python
wai.hf_status()  # connected? namespaces
wai.hf_publish("ds_...", repo="airline-refunds", wait=True)  # rows -> a dataset repo you own
wai.hf_publish_run("run_...", private=True)  # a finished run's LoRA adapter -> a model repo
row = wai.import_hf(
    "tatsu-lab/alpaca", split="train", purpose="eval"
)  # any Hub split -> your account
wai.profile(row["datasetId"])  # profiled before you train on it
```

Every push is one commit tagged `zp-<id>`, so `load_dataset(repo, split, revision="zp-ds_...")` pins the exact push; the repo's `whileai.json` maps each split to its While dataset with history. Worked example: [`recipes/05-export/hugging-face`](recipes/05-export/hugging-face).

Cards live at https://zeroproofai.com/datasets, grouped by agent, with rows,
size and the analyzer's numbers on each. A dataset must be finalized and
hold rows to publish.

## Speed

Two-minute airline runs using While-hosted Qwen. Results were measured on the
hosted GPU with warm replicas and burst under load.

| Mode | Rows | Rate | Unique openers |
|---|---|---|---|
| `explore` | 240 | 120/min | 240 |
| `sft` | 278 | 139/min | 278 |
| `rl` | 625 | 296/min | 209 |

Measured at `avg_turns=4`. The default is now `12`, so a row carries more turns and a run lands fewer rows per minute.

## Parameter reference

| Parameter | Default | Meaning |
|---|---|---|
| `agent` | hosted Qwen | Rollout model |
| `spec` | | Local tools and system prompt path. None ship with the package; `tools=` + `system_prompt=` is the same thing inline |
| `tools`, `system_prompt` | from spec or agent | Tool list and agent system prompt. `tools` is OpenAI function-calling shape, `[{"type": "function", "function": {"name", "description", "parameters"}}]`; the bare `function` dict works too |
| `situations` | | Distinct situations (N) |
| `traces` | `None` | Graded traces (row dicts or a JSONL path) that aim the coverage grid at observed failures. [Close the loop](#close-the-loop-aim-the-budget-with-traces) |
| `tasks` | `None` | Re-run a previous run's task set. Copies the prompts, not the topology: k comes from *this* call's `mode`/`repeats`, so re-pass them |
| `grader` | `None` | A judge callable run beside the rollouts as they land; `mode="rl"` allocation then reads rewards instead of behavior signatures |
| `execute` | `None` | Your own world answers tool calls: `execute(tool_name, arguments) -> result`. The SDK's fault schedule does not apply, so difficulty is your world's job; a rollout that calls no tool never invokes it; `generate.agents.current_rollout` (prompt, rollout index) names the rollout being answered, for per-rollout state |
| `execute` | `None` | `(tool, arguments) -> result`: your real world answers every tool call instead of the mock one |
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
| `seed` | `0` | Reproducible draws. Bit-for-bit at `concurrency: 1` or with `reproducible=True`, within a process and across processes; otherwise which rows land before the cap depends on thread timing |
| `avg_turns` | `12` | Target conversation length in turns. The person speaks at most `avg_turns // 2` times; `12` leaves room to verify, look up, confirm, and write. |

Aliases: `phrasings=` / `n=` → `requests_per_situation`; `repeats=` → `rollouts_per_request`; `unique=` → `unique_situations`; `policy=` → `system_prompt`; `risk=` → `fault_rate`.

## Output

Each row, in `data.trajectories` and on disk: `prompt`, `messages`, `steps`, `final_text`, `scenario_id`. Optional `world_state`, `faults`, `reward`, `reason`. `llm_grade=True` adds `llm_reward`. `wai.rank(path)` adds `quality` without changing `reward`. Every row also says how it was sampled: `sampling` is `{"temperature", "max_tokens", "model"}` as the model backend resolved them, or what you passed as `simulate(sampling=...)` for your own callable agent (`None` when you passed nothing, since only you know how it samples). `policy_version` names the model and the system prompt it ran under.

Three models can take part in a run, and by default they are one: the agent answers, and the same model writes the situations and plays the user in follow-up turns (only the judge is a different model). Every row now says who did which job, next to `model_version` for the agent: `writer_model` (the writer's model, or `template`, `seed`, `pinned` when no model wrote the prompt), `user_model` (absent when the agent took a single message), and `judge_meta.model` once graded; `data.metadata` and the `.meta.json` sidecar carry the same three. When the agent model also wrote the situations or played the user, `data.degraded` holds `same_model` and `data.warnings` says so in one sentence, with the fix: pass `simulator=` for the writer and `user_model=` for the user to put those jobs on a different model.

What goes to disk is the whole row, not a summary of it: `data.rows` (the same list `output=` and `save()` write, callable as `data.rows()` too) carries everything the trajectory carries, so a saved run can still prove its own provenance. That includes how the situation was drawn (`scenario_dimensions`, `arm`, `selection_reason`, `behavior_signature`, `seed`), who graded it and how that went (`judge_name`, `judge_status`, `judge_meta`, `lineage`, `label_source`), and what was measured on it (`markers`, read by `marker_summary` and `delta_report`). Two things never ship, at any depth of the row: the teacher-only `privileged` block and its `principle` / `hidden_state` / `reference` / `rubric` fields, which would put the answer key one step from a training file, and `vector`, the raw embedding the diversity search keeps in memory for the length of the run. A privileged block nested inside a carried field or a tool result is dropped the same way, before `messages` is rebuilt from the steps.

After grading, `data.pass_at` (also on the `ScoredData` from `judge=` and `evaluate`) gives pass@1, pass^k and pass@k off the same groups, one job each: pass@1 is the measurement headline (the agent runs once in production), pass^k is the reliability line (all k repeats pass), and pass@k minus pass@1 (`.headroom`) is what a grouped RL update has to learn from, the same asks `group_signal` counts as mixed. k is the smallest group of repeats; below `repeats=4` the k-way numbers are `None` with a note rather than a noisy figure. With an LLM judge, pass@k inflates on false positives and pass^k on false negatives, so pass@1 stays the headline.

`.per_task` is a **dict**, `{task: pass rate over that task's rollouts}` — keyed by `task_key(row)` (the `scenario_id`, else `task_id`, else the prompt string), not indexed, so `per_task[0]` is a `KeyError` and not the first task. Iterate `.per_task.items()`; `.per_task.values()` is the pass-rate vector pass@1 averages.

**Which of these carry an interval.** All three pass numbers: `pass_at(rows).ci95` is a bootstrap over tasks on pass@1, and `pass_pow_k_ci95` / `pass_at_k_ci95` bootstrap the per-group unbiased estimates over the k-eligible groups, so the reliability line is read with the uncertainty of the tasks behind it (rlhf-book ch. 16). Fewer than three groups gives `None`. What else in this file carries one: `metric_summary` / `marker_summary` over markers, `trace_flag_report` over each trajectory marker's clean share, `refusal_report` and `judge_trust` a Wilson interval, `compare_runs` / `delta_report` a bootstrap interval on the paired *difference*. If a number is not in that list and is not one of the three pass numbers, assume it is a point estimate.

```python
scored = data.grade(judge=my_judge)
print(scored.pass_at)  # pass@1 0.61 | pass^8 0.32 | pass@8 0.88 | headroom 0.27 (200 groups, k=8)
```

`simulate(logprobs=True)` records, on every agent turn, the summed log-probability of the tokens the policy generated and how many there were (`step["logprob"]`, `step["n_tokens"]`, totals on the row). A trainer that updates on these rollouts later needs that number to form the importance ratio `exp(new_logprob - logprob)`; without it the update is off-policy and nothing says so. `wai.logprob_report(rows)` says how much was captured and whether reward tracks the policy's confidence, which on a fair judge it should not. Score the same rows under a reference model, put its summed logprob in `ref_logprob`, and `wai.mean_kl(rows)` gives the sampled KL per generated token, overall and per task; `wai.calibrate(rows, ref="ref_logprob")` writes it into each row's `calibration.mean_kl`. A turn the model cut at the token cap is marked `truncated`. Independently of `logprobs`, every agent step also records what its model call cost when the server reports it (`step["input_tokens"]`, `step["output_tokens"]`, summed into `row["usage"]`), which is what the platform counts per day.

```python
wai.logprob_report(rows)  # coverage, and whether reward tracks the policy's confidence
wai.reference_logprobs(
    data, "vllm:Qwen/Qwen3-4B@https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1"
)  # ref_logprob on every row
wai.mean_kl(rows, ref="ref_logprob")  # sampled KL per generated token, overall and per task
wai.staleness_report(
    rows, base_model="Qwen/Qwen3-4B"
)  # policy versions, stale rows, logprob coverage
```

`staleness_report` is the off-policy check (rlhf-book ch. 6): rows sampled by an older policy are usable only when they carry the sampler's version and its logprobs, so the importance ratio can be formed; rows whose `model_version` differs from `base_model` are `stale`.

The default judge is not the policy. `wai.grade` grades with hosted Phi-4 (`WHILEAI_JUDGE` overrides; any `vllm:`/`openai:` spec or a bare URL works), while rollouts come from hosted Qwen, because a judge grading its own model's writing prefers it. When the judge and the rows' `model_version` are the same model anyway, the grade report says so (`self_judged`, `warnings`).

A judge is a reward model, so two things ride with every label. Provenance: rows graded by `wai.grade` carry `judge_name`, `judge_status`, and `judge_meta` with the model, prompt hash, temperature, and `version` (`<model>@<prompt sha>`); a rubric edit is a new judge and the row says so. `run_judge(version=...)` records the same for your own judge. Accuracy: hand-label a sample into `gold_reward` and call `wai.judge_agreement(rows)` (or `scored.agreement()`) for agreement, Cohen's kappa, the confusion counts, and `pass_when_gold_fail`, the gold failures the judge passed. Those are the rows a training run learns the failure from, so that rate matters more than the headline agreement. Pass a second scoring run as `gold` to measure the judge against itself. Fifty gold rows is the floor; the report says so below it.

Every row carries `schema_version` (`"1"`). A row is a projection of four objects in `whileai.simulations.schema`: `Task` (the situation), `Rollout` (one episode), `Judgment` (a scorer's verdict), `Marker` (a behavior measurement). `wai.from_row(row)` splits a row into them and `wai.to_row(...)` flattens them back. The wire contract is `whileai/simulations/schemas/row-v1.json`. Rows written before the stamp are version 0 and load by shape, so older files still work.

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

The public surface is the package itself: `import whileai.simulations as wai`.
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
