---
title: "Reference overview"
sidebarTitle: "Overview"
description: "What the whileai package is, how simulate() builds a row, and the three ways to run it: offline, on your own model, or on While-hosted models."
---

The long form of the SDK, split over six pages in the order a post-training run happens: this overview, [the five calls](/reference/five-calls), [what to run](/reference/what-to-run), [the platform](/reference/platform), [parameters and output](/reference/parameters), and [package layout and development](/reference/development). The short version, with the loop and where each method comes from, is the [README](https://github.com/whilehq/whileai-sdk#readme).

One package, two importable modules:

- `whileai`: the platform client. Sign in, keys, tracked agents, runs and verdicts. Traces never leave your machine; `whileai.simulations.rows_from_otel` reads an OTLP export locally.
- `whileai.simulations`: post-training data for an agent. Give it the agent's traces, or its tools and system prompt; it simulates the situations, the people, and the world, plays the agent through multi-turn tool-calling conversations, and returns rows for your grader.

Have an agent and want a pass rate with an interval? Start at [Evals](/evals): offline, seconds, and `coverage_gap` names what your tests miss.

<Note>
**Renamed.** This SDK was `zeroproof` (ZeroProof is now While). `uv add zeroproof` still installs `whileai`, and `import zeroproof` (or the older `zeroproof_simulations`) resolves to the same modules with a deprecation warning. `ZEROPROOF_*` environment variables and a saved `~/.zeroproof/credentials.json` are still read. The package is `whileai`, the import is `whileai.simulations`, keys start with `zp_`. `zp`, `wai` and `whileai` run the same CLI, so `zp login` and `whileai login` do the same thing. A machine with the old package still picks up `~/.zeroproof/credentials.json`; set `WHILEAI_HOME` to a fresh directory to isolate a new account from it.

Releases of `whileai` before 0.3 were an unrelated encrypted agent-to-agent messaging client. Pin `whileai<0.3` if you still depend on it.
</Note>

Two ways in, one engine. Give it the agent's tools and system prompt and it samples situations across everything that agent can be asked. Give it graded traces as well (`traces=`, plain row dicts, see [Close the loop](/reference/what-to-run#close-the-loop-aim-the-budget-with-traces)) and it aims the budget at the situations that fail in production, so new rows land where the agent is weak. Every row is a full conversation: user turns, agent turns, tool calls, tool results, scheduled faults. Rows come back ungraded; your grader decides what good means. The default mode, `explore`, draws one unique situation per row. How it thinks: [Simulations](/simulations).

## How a row gets made

![How a row gets made: the draw, the coverage grid, the search arms, the rollout, the split](/how-a-row-gets-made.svg)

A situation is drawn across the world axes (from the agent's tools) and the human axes (from a separate writer). It fills a cell in the coverage grid, nudges the five search arms (`structured`, `open_ended`, `llm_guided`, `behavior_targeted`, `failure_mutation`), and the agent plays it against a world that breaks on schedule. The row that comes out splits into `Task`, `Rollout`, `Judgment`, and `Marker`, and every training target is a projection of some of those four. The engine on one page, with references: [Engine](/engine).

## The pipeline

`simulate()` is a pipeline.

1. **Read the agent.** Tools and system prompt. That is the spec of the world.
2. **Build a fake world from those tools.** Objects, plausible results, and faults (timeout, deny, junk).
3. **Write users.** A separate writer (same hosted model, different prompt, no agent policy) samples situations across tools, stance, history, and so on.
4. **Pick the diverse ones.** Embeddings plus a bit of noise, so the batch is not 200 copies of the same prompt.
5. **Play the agent.** It talks, calls tools, gets results, talks again. All of that is stored: user text, agent text, tool calls, tool results, `final_text`.
6. **Grade.** Rows come back ungraded. Grade after with `data.grade()` (the hosted judge, against the spec's `rubric.md` or `rubric=`; with neither it grades the conduct floor only and the report says so), `data.grade(judge=...)` (your own callable), or `wai.grade(path)`. The legacy `grade=True` flag writes the deterministic conduct score at simulation time; avoid it for the rubric workflow.

Stop when the row cap or the clock hits.

## How to use

```bash
uv add whileai
```

### Start here: no key required

This runs offline, in seconds, on nothing but the package. It is the fastest way to see a row and to check that your agent and grader are wired up before you spend a key on variety.

```python
import whileai.simulations as wai
from whileai import tool


# 1. Your tools: typed functions. The signature is the schema, the
#    docstring the description. Schema dicts work in the same list.
@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id."""
    ...


TOOLS = [get_order]


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

```text
pass@1 1.00 [1.00..1.00] | pass^2 (pass_pow_k) n/a | pass@2 n/a | headroom n/a (18 groups, k=2; set repeats>=4 for pass^k and pass@k)
```

`tools=` takes `@wai.tool` functions, plain typed functions, OpenAI schema dicts (with or without the `{"type": "function", ...}` wrapper) and Anthropic `input_schema` dicts in one list; every call that takes `tools=` normalizes them the same way. The template writer needs no model and runs in seconds, but its situations are less varied than a model writes, so it is for wiring up your agent and grader, not for a training set. For that, [bring a model](#bring-a-model).

If your agent raises, the rollout is dropped and the run says so: `data.stopped_because == "agent_failed"` when no row survived, with the count and the first error in `data.search["agent_errors"]` and `data.search["first_agent_error"]`. An agent that fails every call is called off after `max(16, 2 * budget)` lost rollouts, so a dead endpoint costs a handful of calls, not hundreds.

`data.search` is the run's own report dict. Its keys are written only when the run has something to say: the two error keys appear only if the agent raised, and `search["groups"]` only on a `mode="rl"` run with the default successive allocator. Read them with `data.search.get(...)`.

#### Faults need the world

Offline, every marker is your agent's. The template writer writes the users, not the agent, so a callable that never hedges scores zero hedging, and `faults` on a row only fire if your agent's tool calls go through the world that schedules them. `wai.world(TOOLS)` is that world:

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

#### See the detectors fire

To see the detectors fire before you plug in your own agent, run the seeded one. It answers honestly through `wai.world`, and on 35% of rollouts (`rate=`) does one wrong thing on purpose: `hedging`, `sycophancy`, `apology`, `boilerplate`, `ignore_fault` (claims success through a fault), or `leak` (quotes the row's privileged context). Every row says what it did in `seeded` (`[]` when it behaved), so a check that catches exactly those rows is a check that works.

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

```text
4
checked 58 of 60 rows: 3 quoted privileged context (5%)
  sc-2f784fb823 r0: hidden_state.world_state = 'duplicate entity'
  sc-9ec7194fb6 r0: reference = 'unrelated ask: look if you like, change nothing'
  sc-4428cc9c78 r0: reference = 'two records match: look into it or ask which one before writing'
```

`leak_report` says how many rows it could check. A run where nothing populated `privileged` has nothing to leak, and the report says so instead of passing. Every row is born with the block: `hidden_state` (what the world knows that the ask does not say) and `reference` (what the checklist expects), derived from the task's grid cell. Exports drop it at any depth; `data.trajectories` keeps it for the judge.

### Evals for the agent you already have

Not training anything yet? The shortest path is an eval: wrap your agent as `agent(message) -> {steps, final_text}`, write the policy as a judge that reads the trajectory, run the asks `k` times each, and read pass@1 (the share of asks the agent gets right on one try) with its interval. Offline first, then the hosted writer. The how-to is [Evals](/evals); the runnable version is [`recipes/02-measure/eval-your-agent`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/eval-your-agent), which ends at a CI gate, not a push. `whileai init-evals` writes `agent.py`, `judge.py`, `run.py`, `test_judge.py` and a README for you, wired to the tools, system prompt and callable it finds in the project, and prints what it picked.

```python
data = wai.simulate(
    agent,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=SEEDS,  # opening asks to start from, a list of strings
    simulator=False,
    mode="rl",
    repeats=4,
    repeat_policy="fixed",
)
scored = wai.evaluate(data, judge)  # eval lineage: never the reward
print(wai.pass_at(scored.rows), *scored.warnings)  # a hollow run says so here
```

`scored.warnings` lists what would make the number hollow: no rollout called a tool, a declared tool no rollout touched, a marker that fired on no row. A 1.00 on a run like that is not a result; the note names the fix.

A declared tool the world cannot answer is the quiet version of the same failure. With `execute=`, a tool that is in the schema but has no branch in your function fails exactly like a world fault, the agent reports the miss honestly, and a candour rubric rewards the row. Every run records calls and successes per tool in `data.coverage["tools"]` (`n`, `ok`, `fault_n`, and `injected` for faults the run scheduled itself) and lists the tools that never work in `data.coverage["dead_tools"]`. The generation knobs are reported the same way: `data.coverage["requested"]` is what the call asked for (tier mix, stance mix, user turns, faults) and `data.coverage["delivered"]` is what the rows actually carry, so a mix that did not materialize is visible before the rows are scored. The rule is one Wilson 95% upper bound on the success rate under 0.30: 0 of 9 or 4 of 612 is dead, 0 of 3 or 2 of 5 is not. Dead tools are added to `data.degraded`, and the names and the one fix that applies to your world go in `data.warnings` and `data.report()`. Steps with no recorded result are not evidence and never accuse a tool.

### Bring a model

Any OpenAI-compatible chat endpoint that returns tool calls works. It writes the situations and plays the agent, so both run on your key.

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
- `openai:<model>`: `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for a compatible endpoint that is not OpenAI's.
- `anthropic:<model>`: the Claude Messages API on `ANTHROPIC_API_KEY` (`WHILEAI_ANTHROPIC_API_KEY` overrides it).
- `typesafe:<model>`: TypeSafe's Jev, a decision model, on `TYPESAFE_API_KEY` (`WHILEAI_TYPESAFE_API_KEY` overrides it; `TYPESAFE_BASE_URL` points it at a gateway). Judge only: it answers typed questions with a probability each and writes no text, so `spec=` takes it and `agent=`, `simulator=` and `user_model=` refuse it.

A spec works everywhere one is accepted: `agent=`, `simulator=` for the situation writer, `user_model=` for the simulated person, and `spec=` on `wai.grade` for the judge.

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

With Jev as the judge, the grade is two typed questions per row: did the agent do what it should (a probability), and if not, which failure class. The reward is the more probable outcome; the probability lands on the row.

```bash
export TYPESAFE_API_KEY=...
```

```python
report = data.grade(llm_spec="typesafe:jev-latest")  # grade takes llm_spec=, not spec=
report["unsure"]  # rows whose verdict probability sat within 0.1 of even
data.trajectories[0]["judge_meta"]["confidence"]  # the probability of the verdict given
data.trajectories[0]["failure_class"]  # on a failing row: the judge's own choice
```

#### A model you serve

To put a number on a model you serve (`wai.serve`, or your own vLLM), make it the agent, and run both arms of a before/after through the same call so the only difference is the weights. The writer still runs on a hosted model, so this needs `WHILEAI_API_KEY` in the environment, or `whileai login`, unless you add `simulator=False`:

```python
agent = wai.local_model(endpoint, name, tools=TOOLS, system=POLICY, thinking=False)
data = wai.simulate(tasks=pinned, agent=agent)
```

`thinking=False` reaches the simulated user as well when the agent's own model plays it (the default), and whatever a user model still emits as `<think>` is stripped before it becomes a user turn; the run reports those under `data.search["user_think"]` and says so in `data.warnings`. `delta_report` fails a before/after whose arms differ in how often they answered at all (`"answered"` in `not_comparable`: a two-proportion test at p under 0.01 and a gap over the re-run band or 10 points). That is what a reasoning base against an adapter trained on think-free targets does under one shared token budget, so set `thinking=` the same on both arms.

Two `local_model` knobs the situation writer cannot guess for you:

- `result_shapes={tool_name: example_result}` pins what a tool returns, so a policy branch that only exists for some results ("credits over $200 go to `escalate_to_human`") is reached on purpose instead of by luck. Ids, dates and people are re-drawn per call and a number moves by up to a third of itself (`900.0` lands in 600 to 1200), so pick a template value whose whole range sits on one side of the threshold and run the same pinned tasks under one shape per side.
- `fault_plans={message: {tool: {"mode": "timeout", "rate": 1.0}}}` replays a known fault schedule. `simulate()` writes these from `fault_rate=`, so pass your own only to replay one.

`timeout=` is 300 seconds by default, enough for a served model that scaled to zero to answer its first request. When a call still times out the run says so in `data.warnings` with the fix.

### Hosted models

With no `agent=`, the run uses While-hosted models. Your account key is enough: `whileai login` (or `whileai signup --email you@example.com`) and the run goes to the account endpoints, Qwen3-4B for the writer and the agent and Phi-4 for the judge, on your daily allowance. A trial key gets 25k input and 50k output tokens a day; `whileai status` prints your allowance and how to lift it. The endpoint refuses with 429 when the allowance is spent, and the run stops there and says so. `VLLM_API_KEY`, when set, wins and goes to the shared pool instead: warm and faster, shared and unmetered; ask us for one.

```bash
whileai login              # or: export WHILEAI_API_KEY=zp_...
export VLLM_API_KEY=...      # optional: the shared pool instead
```

The situation writer also defaults to hosted Qwen, even when `agent=` is your own function, so `simulator=False` is what makes a run fully offline.

## The recipe

Each scenario is a draw across the world and the human.

**World** (from this agent's tools and system prompt)

- objects and tool results that match the spec
- tool outcome: success, timeout, deny, stale, and so on
- world state: exists, missing, already handled, unfinished, and so on
- history: first visit, prior miss, return, and so on
- rules the agent is supposed to follow

**Human**

- intent: which tool, what they want (randomized sometimes)
- stance: ordinary, ambiguous, adversarial, hurried, and so on
- persona: first time, returning, in a hurry, and so on
- tone: impatient, frustrated, polite, and so on
- typing: standard, lowercase, typo, clipped, and so on

Ordinary asks first, then the edges. On top of that, the openers are embedded and a bit of random noise is added so the batch stays spread out, not a cluster of near-copies. The row cap and the clock go on diversity, not copies.
