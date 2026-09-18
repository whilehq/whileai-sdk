---
name: whileai-simulations
description: >
  Use the While simulations SDK to generate agent trajectories from a
  policy, tools, an agent harness, seed tasks, or production traces; focus
  generation on failures, grade with the developer's own authority, inspect
  quality, and export usable datasets. Use when a coding agent is asked to
  simulate an agent, expand coverage, turn traces into new scenarios, create
  eval or post-training data, or run the simulate-grade-select loop.
metadata:
  version: "1.0.0"
---

# While simulations

Operate While for the developer. Discover the context already present in
their repository, choose the shortest valid input path, run a small smoke test,
and return auditable data plus a run report. Do not require every possible
input: policy, tools, traces, seeds, and a harness are complementary.

Terms in this skill: the **developer** is the person or team using While;
the **coding agent** is the assistant reading this skill and operating the SDK;
the **target agent** is the agent being simulated and improved; the **simulated
user** is the person speaking to the target agent inside a trajectory.

Ground truth is the installed `whileai.simulations` package. Inspect its
public signatures when the installed version differs from these examples.

## Inputs and routes

Use what the developer has:

- `tools=` plus `system_prompt=`: cold-start simulation across the declared
  agent and policy space.
- `traces=` plus the agent definition: mine observed failures and concentrate
  search on their tools, faults, world states, and behaviors. About 20 graded
  or fault-bearing traces gives useful targeting; below 10, treat the result
  mostly as cold-start exploration and say so. Traces reproduce situations
  (tools, faults, world states); a failure that lives in how the reply is
  worded has no world-visible trigger, so pass `grader=` as well and the
  loop mutates on graded failures too (`search["mutation_aims"]`).
- `agent=`: connect an existing callable, supported framework agent, command,
  or OpenAI-compatible model endpoint.
- `spec=`: load a repository folder containing the agent specification.
- `seeds=`: preserve specific developer-provided tasks as starting asks. Use
  repeats when the developer needs multiple attempts on the same task.
- `execute=`: let the developer's real harness answer tool calls when
  correctness depends on real state.

If traces exist, normalize and inspect them before spending generation budget:

```python
traces = wai.load_traces(trace_source)
report = wai.trace_report(traces, tools=tools, policy=policy)
print(wai.format_trace_report(report))
```

Traces guide generation; they do not replace the authoritative tool schemas or
policy. Report foreign tools and missing context rather than silently inventing
an agent definition.

## Trace-first workflow

When traces are the starting point, follow this sequence:

1. Locate the exact trace dataset. If OTEL is already sending to While,
   list the account's trace datasets and pull the intended `datasetId`; do not
   guess from a local file or silently combine different agents/days:

   ```python
   import whileai
   import whileai.simulations as wai

   # the key resolves like every platform call: WHILEAI_API_KEY, else
   # the key `whileai login` or `whileai signup` saved
   inventory = whileai.list_traces()["traces"]
   matches = [t for t in inventory if t["name"] == requested_dataset]
   if len(matches) != 1:
       raise RuntimeError(f"select one trace dataset explicitly: {matches}")
   selected = matches[0]
   traces = wai.pull(selected["datasetId"])
   ```

2. Load local inputs with `wai.load_traces(...)`. JSONL paths and common message,
   rollout, tool-trace, and platform-export shapes are accepted. Use
   `wai.rows_from_otel(...)` first for raw OTLP/GenAI spans. Store the result as
   `normalized_traces` and use that same list for reporting, simulation, and
   leakage checks.
3. Print `wai.trace_report(...)` before generation. Record rows dropped during
   normalization, graded/pass/fail/ungraded counts, observed tools, call-level
   faults, world states, distinct behaviors, foreign tools, and proposed grid
   emphasis.
4. Find the authoritative tool schemas and policy in the target agent's harness
   when possible. Traces show only tools and arguments that happened to run;
   they cannot reveal unobserved capabilities or recover a missing policy.
5. If schemas are unavailable, a mechanically inferred harness may be used as
   a reviewed draft, never silently as ground truth. Ask the developer to
   confirm required arguments and missing tools.

   ```python
   from whileai.simulations.ingest.traces import infer_harness

   draft = infer_harness(normalized_traces)
   tools, policy = draft["tools"], draft["policy"]  # policy is intentionally empty
   ```
6. Run a 40-row trace-guided smoke test with the same agent definition and
   `traces=normalized_traces`.
7. Inspect `data.search["trace_mining"]` and
   `data.search["behavior_state"]`. Confirm the input count is correct, expected
   failing tools/conditions were mined, focused dimensions changed, and the
   behavior-state allocation reports `applied=True`. `data.metadata.targeted_rows`
   is meaningful only when an explicit `steering_weight` was used; zero is not
   evidence that ordinary trace-guided generation failed.
8. Compare the trace-guided run with a same-budget policy-only run. Report the
   share aimed at observed failures, not merely total row count.
9. Check `wai.leakage_report(data.trajectories, normalized_traces)` and keep
   source-trace copies out of generated data.
10. Grade the new rows with the developer-provided judge. Do not copy rewards from
   source traces onto newly generated situations.
11. Save the returned `ScoredData`; grading by judge creates scored copies and
    does not rewrite the original simulation file.
12. Feed newly graded failures into a later `simulate(traces=...)` round only
    after preserving `model_version`, judge status, reason, and lineage.

A reward of `0` or an observed tool fault supplies direct repair signal. A
reward of `1` supplies contrast. Unlabeled traces still describe the observed
surface but do not establish correctness. Advisory model labels may steer
aiming, but report them separately from developer-owned grades.

## Preflight

Find the exact system prompt the agent receives and obtain tool schemas from
the implementation or harness rather than hand-transcribing them.

```python
import whileai.simulations as wai

pre = wai.preflight(tools, policy)
print(pre)
```

Read every warning. Preflight and trace inspection are offline and need no
model key.

## Choose the search

- Use `mode="explore"` for distinct situations, normally one row per
  situation. This is the default and the best first run.
- Use `mode="sft"` when several human phrasings of each situation are useful.
- Use `mode="rl"` with `rollouts_per_request=` when repeated attempts on the
  exact same ask are required. It probes each ask with two rollouts and spends
  the rest of k on the asks whose rollouts disagree; pass `grader=` so the
  judge's reward decides, not the behavior signature. The allocation is in
  `data.search["groups"]`.
- Use `mode="adaptive", until="saturation"` for a broader run that mixes new
  situations, phrasings, and repeats until coverage plateaus.

Trace-guided repair and policy-guided discovery are both ordinary `simulate`
runs. The difference is whether `traces=` is supplied:

```python
# BYOK; omit agent= to use While-hosted Qwen on the account key
# (`whileai login`), or on VLLM_API_KEY for the shared pool when set.
agent = "openai:gpt-4.1-mini"

repair = wai.simulate(
    agent=agent,
    tools=tools,
    system_prompt=policy,
    traces=traces,
    mode="explore",
    budget=40,
    time_budget=150,
    grade=False,
    output="simulations/trace_guided.jsonl",
)

discovery = wai.simulate(
    agent=agent,
    tools=tools,
    system_prompt=policy,
    mode="explore",
    budget=40,
    time_budget=150,
    grade=False,
    output="simulations/policy_guided.jsonl",
)
```

Start with 40 rows. Close-read the smoke test before increasing to 200-400.
`budget` caps rows; `time_budget` caps wall time. `situations` controls distinct
worlds, `requests_per_situation` controls phrasings, and
`rollouts_per_request` controls independent repeats. Set them separately.

The default novelty embedder is deterministic hashing. Use an OpenAI embedding
spec only when semantic novelty materially matters and the developer authorizes
that endpoint and cost.

## Model and world

For BYOK, set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`, then use
`agent="openai:<model>"`. The endpoint must implement OpenAI-compatible chat
completions with tool calls. The model writes the situations and plays the
target agent, so both consume its endpoint.

The same spec works for `simulator=` (the situation writer), `user_model=`
(the simulated person) and the judge's `spec=`. Backends: `ollama:<model>`,
`vllm:<model>@<url>`, `openai:<model>`, and `anthropic:<model>` for the Claude
Messages API on `ANTHROPIC_API_KEY` (`WHILEAI_ANTHROPIC_API_KEY` overrides it),
for example `agent="anthropic:claude-haiku-4-5"`. Use `anthropic:` when the
developer's only credential is an Anthropic key, instead of falling back to
the template writer.

While normally builds a simulated world from the supplied tools, policy,
and traces. That is appropriate for record-shaped tools and behavioral
questions. If the evaluated agent edits code or correctness depends on a real
database/service, pass `execute(tool_name, arguments)` using the developer's
isolated harness. For code, use one disposable checkout per rollout and real
reads, writes, commands, exit codes, and hidden tests. If no safe harness
exists, report that prerequisite instead of treating invented files as truth.

## Grade after simulation

The developer owns correctness. Prefer a callable judge that returns a reward
in `[0, 1]` and a reason. Judge errors remain unjudged; they are never silently
converted to failures.

```python
def developer_judge(row: dict):
    # Apply the target agent's policy, expected end state, tests, or evaluator.
    return {"reward": 1 if developer_passes(row) else 0, "reason": developer_reason(row)}


scored = data.grade(judge=developer_judge, version="developer_judge@v3")
scored.save("simulations/scored.jsonl")
print(scored.report(tools=tools, system_prompt=policy))
```

Pass `version=` so every scored row names the judge that labeled it (the
hosted `wai.grade` stamps its model and rubric hash itself). When the developer
has hand-labeled rows, write the label into `gold_reward` and report
`scored.agreement()`: agreement, kappa, and `pass_when_gold_fail`, the gold
failures the judge passed. Below 50 gold rows the estimate is coarse; say so.

Use `grade=True` only for While's deterministic structural/conduct screen;
it is not the developer's semantic authority. Hosted or BYOK LLM grading is
optional. Keep unjudged rows out of selection and report judge failures.
`data.grade(judge=...)` deliberately leaves `data.trajectories` and the raw
simulation file unchanged; use the returned `ScoredData` from that point on.

## Inspect before keeping rows

Reject a run when `generator_fallback` appears in `data.degraded` or
`data.stopped_because == "writer_exhausted"`. Other degradation notes are
advisory; name them in the report and inspect their effect.

Always check:

- row count, unique prompts, stop reason, and degraded notes;
- tool names, argument/result linkage, empty or error rollouts;
- coverage by tool, policy rule, world state, condition, stance, and history;
- leakage against source traces and held-out evals;
- whether trace-guided generation actually emphasizes observed failure regions;
- every failed row and at least 20 passing rows;
- for real execution, that every call reached the intended isolated world.

Generated rows are candidates until the developer's judge and review accept
them. Never overwrite existing evals, leak credentials, or merge generated
cases silently.

## Select and export

For eval expansion, keep reviewed cases in a new file with run provenance and
integrate them only after developer approval.

For SFT after binary grading:

```python
report = data.training_set("train.jsonl", target=1000, validate=True)
```

This selects diverse passing demonstrations and applies the tool-call
round-trip export gate. For a judge-contract result returned as `ScoredData`,
use `scored.select_for_sft()` and `wai.export_dataset(...)` with the run's
policy and tools. For preference training, `scored.select_for_preference()`
pairs a passing and a failing rollout of the same prompt; read
`report["warnings"]` before exporting, it flags pairs where chosen is
usually the longer reply or where the two sides came from different models.
For RL, use repeated groups and `select_for_rl`; keep groups whole and
require meaningful within-group reward variation. Run with `logprobs=True`
when the rows will feed a trainer that corrects for off-policy sampling or
measures KL: each agent step then carries `logprob` and `n_tokens`, and
`wai.logprob_report(rows)` says what was captured.

## Evals for an existing agent

When the ask is "build evals" or "improve the evals" for an agent the
developer already runs, do not start from training data. The path is
`recipes/02-measure/eval-your-agent` in the SDK repo and `docs/evals.md`:

1. Wrap the agent as `agent(message) -> {"steps": [...], "final_text": str}`.
   It runs its own real tools and is played single-turn. Record tool calls
   through a `threading.local`: `concurrency` defaults to 32, so 32 rollouts
   call the function at once and one shared list mixes their calls together.
   Tools may be OpenAI-shaped (enveloped or bare) or Anthropic-shaped
   (`input_schema`, read as `parameters`).
2. Put the ids the developer's world has (order numbers, account names) in
   the tool descriptions or in `seeds=`, one seed per policy branch.
   Otherwise the writer invents ids and every rollout is "not found".
3. Write the policy as the judge: read `row["steps"]`, return `reward`,
   `reason`, `markers` (1.0 = the good outcome, `None` = not applicable).
4. `wai.simulate(agent, tools=, system_prompt=, seeds=, simulator=False,
   mode="rl", repeats=4, repeat_policy="fixed", reproducible=True)` first
   (offline), then `simulator="hosted"` (the default, written out) for the
   hosted writer. Keep `repeats` at 4 or more or `pass^k` and `pass@k` come
   back `None` (`min_k`), and keep `budget >= situations * repeats` or the
   later situations never run. A hosted run takes minutes and logs its
   progress on the `whileai.simulations` logger at INFO.
5. `scored = wai.evaluate(data, judge)` (pass the run itself so the declared tools are known); `wai.pass_at(scored.rows)`
   overall and per category. Read `scored.warnings` before any number: a
   run with no tool calls, an untouched declared tool or a marker on zero
   rows is hollow. Never report a pass@1 from a
   hollow run; fix the seeds or descriptions and rerun.
6. Leave a CI gate (`--gate 0.9`, exit 1 under the floor, exit 2 when
   hollow) and a fast lane of offline judge tests on hand-labeled rows.
7. Judge trust: `wai.attach_labels(rows, labels, kind="human")` then
   `wai.judge_trust(rows, judge)`. FAIL on eight labels means label more.

Names: zp, ZeroProof and While are the same product; install `whileai`.
`ZEROPROOF_API_KEY` and `~/.zeroproof/credentials.json` are still read, and
`pip install zeroproof` installs `whileai`. Two spellings per concept:
`data.rows` (exported, also `data.rows()`) beside `data.trajectories` (the
same rollouts before export), and `scored.failures()` beside
`scored.failed_traces()` / `scored.traces`.

## Deliverable

Return the generated JSONL and a concise report containing inputs used, model,
seed/configuration, trace counts, search mode, row/coverage counts, grading
method, rejected rows, leakage result, and review status.
State what was simulated, what came from the developer's real harness, and what
remains unverified.
