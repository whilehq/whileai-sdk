---
title: "Parameters and output"
sidebarTitle: "Parameters"
description: "Every simulate() parameter and advanced knob with its default, and what each row of the output carries."
---

## Parameter reference

Every knob `simulate()` takes. The defaults below are checked against the code by `tests/api/test_readme_defaults.py`, so a wrong number here is a failing test.

| Parameter | Default | Meaning |
|---|---|---|
| `agent` | hosted Qwen | Rollout model: a callable, a model spec (`openai:`, `vllm:`, `ollama:`, `anthropic:`), or nothing for the hosted default |
| `spec` | | Local tools and system prompt path. None ship with the package; `tools=` + `system_prompt=` is the same thing inline |
| `tools`, `system_prompt` | from spec or agent | Tool list and agent system prompt. `tools` is a list of `@wai.tool` functions (the signature is the schema); OpenAI function-calling dicts, with or without the `{"type": "function", ...}` wrapper, and Anthropic `input_schema` dicts work in the same list |
| `situations` | | Distinct situations (N) |
| `traces` | `None` | Graded traces (row dicts or a JSONL path) that aim the coverage grid at observed failures. [Close the loop](/reference/what-to-run) |
| `tasks` | `None` | Re-run a previous run's task set: that run, its rows, or its JSONL path. Copies the prompts and, unless you pass `repeats=`, the pinned run's k. [Same tasks, new prompt](/reference/platform#trust-the-numbers) |
| `runs` | `1` | Replay the task set this many times in one call, stamping `lineage.eval_run`; `delta_report` reads the re-run spread off it |
| `grader` | `None` | A judge callable run beside the rollouts as they land; `mode="rl"` allocation then reads rewards instead of behavior signatures |
| `execute` | `None` | Your own world answers tool calls: `execute(tool_name, arguments) -> result`. The SDK's fault schedule does not apply, so difficulty is your world's job; a rollout that calls no tool never invokes it; `generate.agents.current_rollout` (prompt, rollout index) names the rollout being answered, for per-rollout state |
| `requests_per_situation` | from mode | Phrasings per situation (n). Alias `phrasings=` / `n=` |
| `rollouts_per_request` | from mode | Repeats per phrasing (k). Alias `repeats=` |
| `repeat_policy` | from mode | `"fixed"` gives every prompt k rollouts; the `rl` default allocates them where groups split |
| `unique_situations` | on in `explore` | Unique situations only |
| `mode` | `"explore"` | `explore`, `sft`, `rl`, `adaptive` |
| `fault_rate` | `0.5` | Share of tool calls the mock world breaks (`0.8` under `mode="rl"`). `0` off. Alias `risk=`. A callable `agent=` that answers its own tool calls never sees one |
| `hard_share` | from mode | Share of situations drawn from the hard tiers (adversarial, boundary, ambiguous), 0 to 1. Under `runs=N`, `search["tier_mix"]` counts every run's rows and lists `per_run` |
| `reproducible` | `None`: `True` unless `time_budget` is set | Round-synchronous scheduling: same seed, same agent, same rows at any concurrency, on any CPython version (3.10 to 3.13; the draw never depends on the interpreter). Runs batch by batch, so a slow rollout holds its batch; `False` trades the same task set on every machine for that throughput. A clock turns it off, since a clock stop lands wherever the run is. Measured before the default changed (0.111): three runs of one seed at the default concurrency drew three different task sets (59, 58, 58 tasks from `budget=160`); with the flag, one, in the same time |
| `budget` | `1000` | Row cap, per run under `runs=N` (`report()["budget_per_run"]`). With `situations=N` the run stops once every situation has its rollouts (`stopped_because="situations_exhausted"`) whatever the budget still allows |
| `time_budget` | `None` | Seconds. Off by default; `None` or `0` disables |
| `until` | `"compute"` | `"saturation"` also stops when coverage plateaus |
| `logprobs` | `None` | `True` records each agent turn's summed log-probability and token count; `"tokens"` keeps the per-token list. Model backends only. [Output](#output) |
| `sampling` | `None` | How your own callable agent samples, `{"temperature": 0.7, "max_tokens": 1024, "model": "my-model"}`, recorded on every row as given. A model backend records its own and ignores this |
| `simulator` | hosted Qwen | The situation writer. `False` is the built-in template writer (no model, no key, less variety); a model spec runs it elsewhere |
| `user_model` | `None` | Who plays the simulated user in follow-up turns. `None` is the agent's own model |
| `timeout` | `max(300, agent_max_tokens / 4)` | Seconds per agent call, for `local_model` and every model spec. Unset, 300 s or the reply budget at 4 tokens a second, whichever is longer (1,024 s at `agent_max_tokens=4096`), so a long reply is not re-rolled for taking the time it was allowed. [Watching and resuming a long run](#watching-and-resuming-a-long-run) |
| `grade` | `False` | Legacy deterministic conduct score; grade after instead |
| `llm_grade` | `False` | Extra LLM judge |
| `output` | | JSONL path, written as the run goes and complete when it returns |
| `checkpoint` | `None` | JSONL path every row is appended to the moment it lands. Call again with the same path and `tasks=` to resume a killed run: finished tasks are skipped, the union comes back. [Watching and resuming a long run](#watching-and-resuming-a-long-run) |
| `on_progress` | `None` | A callable that receives the progress dict (rows landed, re-rolled, lost, by reason) on every progress line, whatever the budget |
| `advanced` | | Keys below. `data.report()` records the resolved values (`knobs`, `patience`, `user_temperature`, `world`) so a saved run says what it ran under |

Aliases: `phrasings=` / `n=` for `requests_per_situation`; `repeats=` for `rollouts_per_request`; `unique=` for `unique_situations`; `policy=` for `system_prompt`; `risk=` for `fault_rate`.

## Experiment knobs

What a researcher changes between runs: who plays the user and how patient they are, how the three models sample, what the mock world answers, how hard the situations are, and what the search steers by. They go in `advanced={...}`. `data.report()` carries every knob that produced the run, as resolved: the parameters above (`mode`, `budget`, `seed`, `repeats`, `hard_share`, `fault_rate`, `strategy`, `dimensions`, `arm_weights`, counts of `tasks`, `traces` and `seeds`, the `grader` name, `simulator`, `agent_model`, `user_model`, sampling and turn limits) and the `advanced` values below, so a saved run is its own experiment record. `report()["fault_rate"]` is the run's rate; `report()["world"]["default_fault_rate"]` is the rate a fault plan with no rate of its own fires at, and `world_note` says so.

| `advanced` key | Default | |
|---|---|---|
| `seed` | `0` | Reproducible draws. Bit-for-bit by default (`reproducible` resolves to `True` without a clock), within a process, across processes, across machines and across CPython versions (3.10 to 3.13). With `reproducible=False` or `time_budget`, which rows land before the cap depends on thread timing |
| `concurrency` | `32` | Parallel rollouts |
| `avg_turns` | `12` | Target conversation length in turns. The person speaks at most `avg_turns // 2` times; `12` leaves room to verify, look up, confirm, and write. `1` is one user line and one reply for every rollout, the same as `max_turns=1`; the follow-up branch never runs |
| `max_turns` | by tool count | Hard cap on turns. `max_turns=1` is one user line and one reply, whatever the reply says: a question in the reply does not earn a second user line (#586) |
| `patience` | `"normal"` | How long the person keeps answering the agent's questions. A level name, or a table `{"second": p, "later": q}` (or `(p, q)`) of walk-away chances fitted from your own traces. The first question is always attempted; from the second on the person may walk away (`normal`: 35% then 60%; `short`: 60% then 90%; `endless`: never). At any question the person may also leave when it asks for something they could not know. A row the person left ends on the agent's question and carries `ended_by="user_left"`; `search["ended_on_question"]` counts them |
| `user_temperature` | `None` | One sampling temperature for every simulated-user line (follow-ups and human-tool answers alike); `None` keeps the two named defaults in `generate/agents.py` (`USER_TURN_TEMPERATURE` 0.475, `HUMAN_TOOL_TEMPERATURE` 0.9) |
| `writer_temperature` | `(0.45, 1.05)` | The situation writer's sampling temperature: a number pins it, a `(lo, hi)` band is drawn from per batch. The default band is `WRITER_TEMP_LO..WRITER_TEMP_HI` in `generate/diversity.py`. Validated before any model call, offline too |
| `world` | `{}` | The mock world's dials as a dict of `WorldOptions` fields (`world/sandbox.py`): `fault_modes` (add your own builder), `default_fault_mode`, `default_fault_rate`, `search_hits`, `exists_share`, name pools and the rest. Validated before any model call; a typo names the fields |
| `pass_threshold` | `0.5` | A reward under this is a graded failure (the search steers by it with `grader=`) |
| `mutate_graded_failures` | on with `grader=` | `False` grades beside the loop without steering by the verdict: only tool faults make mutation parents. A row the grader fails (reward under 0.5) is otherwise re-rolled and its ask mutated the way a tool fault's is; `search["mutation_aims"]` counts each aim. `True` without `grader=` is an error |
| `smoothing_alpha` | `1.0` | Laplace smoothing `(s + a) / (n + 2a)` on the group hazard and the mixed rate |
| `allocation_gain` | `4.0` | How hard a hot trace region pulls cell weight toward itself: a full match at budget share s multiplies the weight by `1 + gain x s` |
| `allocation_tool_weight` | `0.6` | Match credit for a cell on a hot region's tool |
| `allocation_condition_weight` | `0.4` | Match credit for a cell on a hot region's tool condition |
| `gap_weight` | `0.7` | Share of a region's behavior value from its gap score; the rest from its fault rate |
| `adaptive_verify_explore_floor` | `0.55` | Under `mode="adaptive"`, an explore share below this re-rolls a prompt to peek for a different outcome |
| `tier_mix_min_rows` | `20` | Rows before the drawn difficulty mix is compared with `hard_share` |
| `tier_mix_tolerance` | `0.1` | How far below the asked hard share the drawn share may land before the run says so |
| `stop_grace` | `5` | Seconds to wait for running rollouts and writer waves after a stop; queued ones are cancelled, still-running ones are reported as `rollouts_abandoned` / `writer_waves_abandoned`, and an abandoned wave adds a `warnings` line with the count, the stop reason and this knob |
| `embedder` | `"hash"` | Prompt selection |

## Watching and resuming a long run

A 2,400-row `simulate(tasks=...)` through a served model is hours of work, and the engine re-rolls a rollout up to `repeats` times when the agent errors, replies empty, or leaks tool markup, so the wall clock can say nothing about the rows (whilehq/whileai-sdk#470: a 300 s timeout on 4,096-token replies re-rolled each one up to k times and a two-hour run took six and a half). Three things make it readable:

- **Progress with the re-rolls in it.** Every 10 finished events (a row landed, a rollout re-rolled or one lost) or 10 s, the `whileai.simulations` logger says `120/2404 rollouts, 601 situations written, 1h2m elapsed, ~19h left, 96 re-rolled, 3 lost (3 agent error)` (INFO; on stderr too when no handler is attached). `on_progress=` receives the same numbers as a dict on every line: `rows`, `cap`, `landed` (this call), `resumed`, `rerolled` and `rerolled_by` (`agent_error`, `empty_reply`, `tool_markup`), `timed_out` (agent errors that were call timeouts), `lost` and `lost_by`, `inflight`, `situations`, `elapsed_s`. The final counts are `data.search["rollouts"]`, and `data.warnings` (plus a `UserWarning`) says so when more rollouts were re-rolled than landed, with the fix (`timeout=` or `agent_max_tokens=` when the errors were timeouts).
- **Rows on disk the moment they land.** `checkpoint="rows.jsonl"` appends each row as it lands, so a kill loses nothing. Call again with the same `checkpoint=` and `tasks=` to resume: the rows on disk are loaded, a task with its `repeats` rows is skipped, one with fewer gets only the missing rollouts, and the returned `SimulationData` is the union (`search["rollouts"]["resumed"]` counts the loaded rows, `lineage.resumed` marks each). Rows on disk whose prompt is not in `tasks=` stay in the file and out of the run, and `warnings` says how many. Without `tasks=` the rows on disk are loaded and count toward `budget`, and the run draws new situations for the rest. `runs=N` and `checkpoint=` do not combine (one file holds one run's rows).
- **A call timeout sized to the reply.** `timeout=` unset is `max(300, agent_max_tokens / 4)` seconds: 300 s, or the reply budget at 4 tokens a second per request, whichever is longer. A reasoning model that writes 4,096 tokens gets 1,024 s, not the flat 300 s that re-rolled every long reply. Set it yourself when you know the server: `timeout >= agent_max_tokens / tokens-per-second-per-request`.

```python
# the same call on a served model: agent="vllm:Qwen/Qwen3.5-9B@http://host:8000/v1",
# agent_max_tokens=4096, and the progress line on the whileai.simulations logger
rerun = wai.simulate(
    wai.seeded_agent(TOOLS),
    tools=TOOLS,
    system_prompt=POLICY,
    simulator=False,
    tasks=data,
    repeats=4,
    checkpoint="rows.jsonl",  # rows land here as they finish; call again to resume
    on_progress=lambda p: print(p["rows"], p["rerolled"], p["lost_by"]),
)
print(rerun.search["rollouts"])  # landed, resumed, rerolled_by, lost_by, timed_out
```

## Engine internals

You should not need these. Every other number the engine uses is an `advanced` key too, named after its field on `whileai.simulations.defaults.RunKnobs`, where a comment above each states why the default is what it is (a measurement, the paper or textbook chapter it follows, or the exact words "convention, untested", which `scripts/check_no_hardcoding.py` enforces so one grep of `defaults.py` finds every unsourced number). They are here so nothing in the engine is a number you cannot change, and so a report (`data.report()["knobs"]`) can say what a run ran under. A value outside its bounds is a `ValueError` that names the floor or ceiling and the default.

| `advanced` key | Default | |
|---|---|---|
| `empty_rounds_to_stop` | `8` | Consecutive scheduler rounds with nothing to run and nothing in flight before a run whose model writer wrote nothing gives up |
| `writer_idle_rounds_to_restart` | `4` | Rounds the pool stays empty on duplicate waves before the writer is restarted with a rotated seed and an avoid window |
| `writer_idle_rounds_to_rest` | `2` | Idle rounds after which a plain run (not unique, no clock) stops launching waves and lets the restart rule decide |
| `restart_avoid_window` | `8` | Used asks handed to a restarted writer as avoid pressure |
| `rows_per_extra_restart` | `100` | One extra writer restart per this many budgeted rows, above the base allowance |
| `dead_agent_errors` | `16` | Lost rollouts with no row landed before an agent that raises on every call is called off (or `dead_agent_budget_multiple` x budget, whichever is larger) |
| `dead_agent_budget_multiple` | `2` | See `dead_agent_errors` |
| `closing_margin` | `2.0` | Under a clock, stop opening groups when the time left is under this many median rollout durations |
| `closing_window_rollouts` | `20` | Recent rollouts the median duration is read from |
| `collect_wait_s` | `0.35` | The loop's tick: how long a round waits for a rollout or verdict before re-planning |
| `collect_wait_floor_s` | `0.1` | The shortest tick when the clock is nearly out |
| `writer_wait_s` | `0.5` | How long the loop waits on a writer wave when the pool is empty and nothing is in flight |
| `writer_wait_floor_s` | `0.2` | The shortest writer or scene wait when the clock is nearly out |
| `hosted_touch_s` | `5.0` | Timeout on the warm-up ping to the hosted writer |
| `scene_join_s` | `8.0` | How long shutdown waits for the scene-brief thread with no clock |
| `scene_join_clocked_s` | `1.0` | The same wait under a clock |
| `writer_buffer_waves` | `2` | Waves of prompts kept in the pipe ahead of the rollouts |
| `writer_buffer_cap` | `96` | The most prompts the buffer plans for |
| `writer_typical_completions` | `3` | Completions per card the buffer math assumes |
| `pool_low_floor` | `16` | The pool is low (refill now) under `max(pool_low_floor, min(pool_low_cap, concurrency / pool_low_flight_divisor))` eligible prompts |
| `pool_low_cap` | `64` | See `pool_low_floor` |
| `pool_low_flight_divisor` | `4` | See `pool_low_floor` |
| `offline_topup_multiple` | `2` | The template writer tops up under this many batches of eligible prompts |
| `offline_bounce_limit` | `20` | Extra template rounds tried before a short batch is accepted |
| `offline_bounce_stride` | `17` | Seed step between those rounds |
| `restart_seed_stride` | `997` | Round-id step per writer restart, so restarted draws reuse no round's temperature and tags |
| `first_wave_writers` | `2` | Writer waves launched at start, tiny so the first rollouts start early |
| `first_wave_cards` | `4` | Cards per first wave |
| `first_wave_tokens` | `320` | Reply budget of a first wave |
| `min_cards_per_wave` | `2` | A wave asks for at least this many cards |
| `tokens_per_card` | `130` | A wave's reply budget is `clamp(tokens_per_card x cards + wave_tokens_base, wave_tokens_floor, wave_tokens_cap)` |
| `wave_tokens_base` | `128` | See `tokens_per_card` |
| `wave_tokens_floor` | `256` | See `tokens_per_card` |
| `wave_tokens_cap` | `2048` | See `tokens_per_card` |
| `failing_seeds_cap` | `40` | Failing asks mined from `traces=` that seed the run |
| `select_oversample` | `3` | The diversity selector picks this many times the batch so the family cap and the situation quota have slack |
| `family_cap_floor` | `16` | Near-copy scenario families are capped at `max(family_cap_floor, phrasings x family_cap_per_phrasing)` rows |
| `family_cap_per_phrasing` | `4` | See `family_cap_floor` |
| `writer_context_items` | `8` | Items of each kind (avoid, underexplored, behavior gaps, axis gaps, tools) the writer prompt carries |
| `writer_context_parents` | `10` | Failing rows the writer mutates from |
| `family_avoid_items` | `6` | Family-rejected prompts shown to the writer as avoid pressure |
| `region_novelty_smoothing` | `0.5` | Weight on the newest novelty score in a region's running novelty |
| `gap_min_rows` | `3` | Rows a region needs before one signature counts as stuck |
| `gap_rich_signatures` | `3` | Distinct signatures at which a region counts as explored |
| `gap_value_stuck` | `1.0` | Behavior-gap score of a stuck region |
| `gap_value_rich` | `0.2` | Behavior-gap score of an explored region |
| `gap_value_unknown` | `0.5` | Behavior-gap score (and starting novelty) of an undecided region |
| `short_share_floor` | `0.08` | Under this share of short asks the writer is nudged to keep it brief |
| `long_share_floor` | `0.1` | Under this share of long asks the writer is nudged to use more words |
| `followup_starved_min` | `8` | `followups_starved` needs at least this many missed follow-ups that are also at least rows / `followup_starved_divisor` |
| `followup_starved_divisor` | `4` | See `followup_starved_min` |
| `semantic_duplicate_novelty` | `0.05` | A row under this semantic novelty counts as a duplicate |
| `idle_judge_share` | `0.1` | An rl pool idle on verdicts for more than this share of the run gets the add-situations note |
| `progress_every_s` | `10.0` | Never more than this long between progress lines (INFO on the `whileai.simulations` logger; on stderr too when no handler is attached) |
| `progress_every_rows` | `10` | Never more than this many finished rollouts between progress lines |
| `flush_report_rows` | `25` | The streamed-output log line is written every this many rows |
| `flush_report_s` | `5.0` | Or every this many seconds |

## Output

Each row, in `data.trajectories` and on disk: `prompt`, `messages`, `steps`, `final_text`, `scenario_id`. Optional `world_state`, `faults`, `reward`, `reason`. `llm_grade=True` adds `llm_reward`. `wai.rank(path)` adds `quality` without changing `reward`. Every row also says how it was sampled: `sampling` is `{"temperature", "max_tokens", "model"}` as the model backend resolved them, or what you passed as `simulate(sampling=...)` for your own callable agent (`None` when you passed nothing, since only you know how it samples). `policy_version` names the model and the system prompt it ran under.

Three models can take part in a run, and by default they are one: the agent answers, and the same model writes the situations and plays the user in follow-up turns (only the judge is a different model). Every row says who did which job, next to `model_version` for the agent: `writer_model` (the writer's model, or `template`, `seed` when no model wrote the prompt; a replay from `tasks=` or `runs=N` keeps the writer of the run it replays, since the situation was written once, and says `lineage.replayed`, plus `lineage.replayed_from_run` under `runs=`, so `delta_report` on two runs of one call sees one writer), `user_model` (absent when the agent took a single message), and `judge_meta.model` once graded; `data.metadata` and the `.meta.json` sidecar carry the same three. Every row also names the deploy prompt it was generated under: `lineage.system_prompt_sha` (the hash `policy_version` carries after `@`), `lineage.system_prompt_head` (its first 120 chars) and `lineage.system_prompt_chars`, with the full text once per run in `data.system_prompts[<sha>]`, so a base rate measured under a full policy is never mistaken for one measured under a bare prompt; `delta_report` warns when its two arms differ on that hash. When the agent model also wrote the situations or played the user, `data.degraded` holds `same_model` and `data.warnings` says so in one sentence, with the fix: pass `simulator=` for the writer and `user_model=` for the user to put those jobs on a different model.

What goes to disk is the whole row, not a summary of it: `data.rows` (the same list `output=` and `save()` write, callable as `data.rows()` too) carries everything the trajectory carries, so a saved run can still prove its own provenance. That includes how the situation was drawn (`scenario_dimensions`, `arm`, `selection_reason`, `behavior_signature`, `seed`), who graded it and how that went (`judge_name`, `judge_status`, `judge_meta`, `lineage`, `label_source`), and what was measured on it (`markers`, read by `marker_summary` and `delta_report`). Two things never ship, at any depth of the row: the teacher-only `privileged` block and its `principle` / `hidden_state` / `reference` / `rubric` fields, which would put the answer key one step from a training file, and `vector`, the raw embedding the diversity search keeps in memory for the length of the run. A privileged block nested inside a carried field or a tool result is dropped the same way, before `messages` is rebuilt from the steps. `data.trajectories` keeps `privileged` in memory for the judge; so do the graded copies `data.grade(judge=)` returns.

### pass@1, pass^k, pass@k

After grading, `data.pass_at` (also on the `ScoredData` from `judge=` and `evaluate`) gives pass@1, pass^k and pass@k off the same groups, one job each: pass@1 is the measurement headline (the agent runs once in production), pass^k is the reliability line (all k repeats pass), and pass@k minus pass@1 (`.headroom`) is what a grouped RL update has to learn from, the same asks `group_signal` counts as mixed. k is the smallest group of repeats; below `repeats=4` the k-way numbers are `None` with a note rather than a noisy figure. With an LLM judge, pass@k inflates on false positives and pass^k on false negatives, so pass@1 stays the headline.

```python
scored = data.grade(judge=my_judge)
print(scored.pass_at)
```

```text
pass@1 0.66 [0.55..0.75] | pass^4 (pass_pow_k) 0.19 [0.00..0.38] | pass@4 1.00 [1.00..1.00] | headroom 0.34 (16 groups, k=4)
```

When groups are uneven (the `rl` default allocates rollouts where groups split), k defaults to the smallest group and the line says which groups it left out; `repeat_policy="fixed"` gives every prompt the same k.

`.per_task` is a **dict**, `{task: pass rate over that task's rollouts}`, keyed by `task_key(row)` (the `scenario_id`, else `task_id`, else the prompt string), not indexed, so `per_task[0]` is a `KeyError` and not the first task. Iterate `.per_task.items()`; `.per_task.values()` is the pass-rate vector pass@1 averages.

**Which of these carry an interval.** All three pass numbers: `pass_at(rows).ci95` is a bootstrap over tasks on pass@1, and `pass_pow_k_ci95` / `pass_at_k_ci95` bootstrap the per-group unbiased estimates over the k-eligible groups, so the reliability line is read with the uncertainty of the tasks behind it [2]. Fewer than three groups gives `None`, and `pass_at(rows).note` then says how many tasks there were, how many a bootstrap needs, and the fix: the resampling is over tasks, so rows that all carry one `task_id` are one task however many rows they are. What else carries one: `metric_summary` / `marker_summary` over markers, `trace_flag_report` over each trajectory marker's clean share, `refusal_report` and `judge_trust` a Wilson interval, `compare_runs` / `delta_report` a bootstrap interval on the paired *difference*. If a number is not in that list and is not one of the three pass numbers, assume it is a point estimate.

### Logprobs and off-policy checks

`simulate(logprobs=True)` records, on every agent turn, the summed log-probability of the tokens the policy generated and how many there were (`step["logprob"]`, `step["n_tokens"]`, totals on the row). A trainer that updates on these rollouts later needs that number to form the importance ratio `exp(new_logprob - logprob)`; without it the update is off-policy and nothing says so. `wai.logprob_report(rows)` says how much was captured and whether reward tracks the policy's confidence, which on a fair judge it should not. Score the same rows under a reference model, put its summed logprob in `ref_logprob`, and `wai.mean_kl(rows)` gives the sampled KL per generated token, overall and per task; `wai.calibrate(rows, ref="ref_logprob")` writes it into each row's `calibration.mean_kl`. A turn the model cut at the token cap is marked `truncated`. Independently of `logprobs`, every agent step also records what its model call cost when the server reports it (`step["input_tokens"]`, `step["output_tokens"]`, summed into `row["usage"]`), which is what the platform counts per day.

```python
wai.logprob_report(rows)  # coverage, and whether reward tracks the policy's confidence
wai.reference_logprobs(
    data, "vllm:Qwen/Qwen3-4B@https://your-vllm-host/v1"
)  # ref_logprob on every row
wai.mean_kl(rows, ref="ref_logprob")  # sampled KL per generated token, overall and per task
wai.staleness_report(
    rows, base_model="Qwen/Qwen3-4B"
)  # policy versions, stale rows, logprob coverage
```

`staleness_report` is the off-policy check [1]: rows sampled by an older policy are usable only when they carry the sampler's version and its logprobs, so the importance ratio can be formed; rows whose `model_version` differs from `base_model` are `stale`.

### The judge on the row

The default judge is not the policy. `wai.grade` grades with hosted Phi-4 (`WHILEAI_JUDGE` overrides; any `vllm:`/`openai:` spec or a bare URL works), while rollouts come from hosted Qwen, because a judge grading its own model's writing prefers it. When the judge and the rows' `model_version` are the same model anyway, the grade report says so (`self_judged`, `warnings`).

A judge is a reward model, so two things ride with every label. Provenance: rows graded by `wai.grade` carry `judge_name`, `judge_status`, and `judge_meta` with the model, prompt hash, temperature, and `version` (`<model>@<prompt sha>`); a rubric edit is a new judge and the row says so. `run_judge(version=...)` records the same for your own judge. Accuracy: hand-label a sample into `gold_reward` and call `wai.judge_agreement(rows)` (or `scored.agreement()`) for agreement, Cohen's kappa, the confusion counts, and `pass_when_gold_fail`, the gold failures the judge passed. Those are the rows a training run learns the failure from, so that rate matters more than the headline agreement. Pass a second scoring run as `gold` to measure the judge against itself. Fifty gold rows is the floor; the report says so below it.

### Schema

Every row carries `schema_version` (`"1"`). A row is a projection of four objects in `whileai.simulations.schema`: `Task` (the situation), `Rollout` (one episode), `Judgment` (a scorer's verdict), `Marker` (a behavior measurement). `wai.from_row(row)` splits a row into them and `wai.to_row(...)` flattens them back. The wire contract is [`whileai/simulations/schemas/row-v1.json`](https://github.com/whilehq/whileai-sdk/blob/main/whileai/simulations/schemas/row-v1.json). Rows written before the stamp are version 0 and load by shape, so older files still work.

## References

1. Noukhovitch, M. et al. Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models. ICLR 2025. arXiv:2410.18252.
2. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
