---
title: "Platform: sign in, datasets, training, serving"
sidebarTitle: "Platform"
description: "Sign in, push and gate datasets, prune and check them, train on the platform or report your own run, and serve the result."
---

Everything on this page talks to your While account. Every call reads the key that `wai login` saved, or `WHILEAI_API_KEY`, or `api_key=`. The pure-Python checks that need no key (`optimize`, `hack_scan`, `judge_trust`, `delta_report` and the rest) are here too, because they sit between a run and a push.

## Sign in

```bash
wai login
```

Prints a link and a short code. Open the link, sign in or sign up, press Approve. The key is saved to `~/.whileai/credentials.json` and every platform call reads it from there. Interrupted before you approved? Run it again; it resumes the same code. This is the path for coding agents too: tell yours to run `wai login` and click the link it shows you. `wai status` shows which key is in use, `wai logout` removes it.

No account yet, or no browser? One command creates the account and the key. Open the dashboard later by signing in with an email code.

```bash
wai signup --email you@example.com
```

That key is a trial key (25k input and 50k output tokens a day, 100 MB of storage, ten datasets, and an expiry date) until the person signs in once at the While site with an email code; `wai status` prints the link. `wai status` shows the tier; `whileai.account()` returns tier, limits and usage.

## Store datasets on While

Push a run to your account so the rest of the loop can read it. Credentials resolve in this order: `api_key=` argument, `WHILEAI_DELEGATED_CREDENTIAL` (a short-lived `zp_dc_...` issued from a Clerk session), `WHILEAI_API_KEY`, then the key saved by `wai login`.

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

Storage is private per account; `whileai.account()` says how much you have. `parent=` records dataset lineage so iterations show as a family on the platform.

### The publish gate

`data.push` and `wai.push_file` run a publish gate first (`gate=False` skips it; `wai.push_rows` does not gate unless you pass `gate=True`, because the caller may already have run `optimize`). Every graded row gets a `calibration` stamp: its task's pass rate over k repeats, k, and the policy that produced it, so a trainer can build a curriculum or retire solved tasks. An RL-shaped run (repeats of one ask) is refused with `PublishGateError` when it is ungraded or has no mixed group, because a grouped update would learn nothing from it. The report comes back as `entry["gate"]`, with warnings when unanimous asks, or asks outside the difficulty band, are still present; `wai.optimize(data, mode="rl")` prunes those. `wai.publish_gate(rows)` runs the same check on any row list.

The stamp is the schema's `Calibration` object: `wai.calibration_of(row)` reads it back typed, `from_row` carries it on `rollout.extra["calibration"]`, and `to_row` writes it out again. `k` is the repeats the grader saw, not the rows that survived: `optimize(mode="rl")` stamps its selection from the rows it was given, before its own dedupe and trims, and the gate keeps a carried stamp rather than re-measuring it on what is left. The gate's own `pass_at` block is still over the rows in front of it, and says so when the two differ.

### Export training rows

`export_dataset` and `export_training` are the same function object (`export_dataset is export_training`): same arguments, same file, same report. Write `export_dataset` in new code; `export_training` is the older spelling, kept so nothing already written breaks. `training_rows` is the list-returning half of the same path, without writing a file.

Training rows carry a `loss_mask`, one 0/1 per message: 1 on the agent's turns, 0 on system, user, and tool-output turns. Tool output is the environment's text, not the policy's, so a trainer should not learn to predict it. `mask_mode="final"` trains only the last assistant turn, for conversations whose earlier agent turns were scripted or came from another policy; the export report counts `trained_messages` and `masked_messages`. `unroll=True` turns an N-turn conversation into N samples, the k-th ending at the k-th agent turn with loss on that turn only, so every earlier turn trains once with the context it actually had [1]. `max_tool_output_chars=` caps each tool message, appends a `[... N chars of tool output truncated]` marker and counts the cut on the row and in the report, so context spent on tool output is a decision the export makes out loud [1].

Two wire shapes come out of the exporters, and a trainer needs the second one:

```python
wai.export_training(rows, "sft.jsonl")  # OpenAI chat-completions wire (default)
wai.export_training(rows, "sft.jsonl", format="trl")  # what TRL's SFTTrainer loads
wai.export_preference(pairs, "dpo.jsonl", format="trl")  # what TRL's DPOTrainer loads
wai.to_trl(wai.training_rows(data), "training")  # same reshape on rows you already hold
```

`format="openai"` (the default) is the API wire row: the whole conversation in `messages`, `function.arguments` as a JSON string, and the ask alongside as `prompt`. `format="trl"` is what `trl.data_utils.maybe_apply_chat_template` accepts. For SFT that is conversational `messages` with **no** `prompt` string column: TRL decides "is this conversational?" from the column set, and a `prompt` string next to `messages` makes it skip the chat template silently and train on the bare ask; the ask survives as `prompt_text`. The TRL rows carry no `loss_mask`: trl 0.19.1's `SFTTrainer` never reads one (its collator unlabels tokens only from `completion_mask` and `assistant_masks`, both built by the trainer), so `mask_mode="final"` and `unroll=True` write prompt/completion rows that TRL trains exactly as the mask asks, and `mask_mode="assistant"` writes `messages` rows that TRL trains on every token of unless `assistant_only_loss=True` is set (which needs a `{% generation %}` block in the chat template; Qwen2.5-Instruct has none). The report's `mask_mode` says which of the two TRL will do. For preference data it is `prompt` as the message list up to the first agent turn with `chosen`/`rejected` as the completions only, because the default shape (a `prompt` string with full conversations on both sides) raises `TypeError: string indices must be integers` inside TRL. In the TRL shape `function.arguments` is a dict, not a JSON string: HF chat templates render it with `| tojson`, so a pre-encoded string is quoted twice and the student learns to emit a string where an object belongs. The `tool_call_roundtrip` block in the report names which of the two encodings it checked (`encoding: "json_string"` or `"dict"`), so `invalid: 0` says what it actually vouches for.

The export refuses rows whose reply quotes their own privileged context (`privileged_leak` in the error) because the export scrubs the key, not the reply; drop the rows `leak_report` names, or pass `validate=False`.

### Prune before training

```python
rows, report = wai.optimize(data, mode="rl")  # whole groups, 20%-80% pass rate
rows, report = wai.optimize(data, mode="rl", band=(0.3, 0.7))
rows, report = wai.optimize(data, mode="rl", enforce_band=False)  # rank, do not drop
report["band_dropped"]  # {"too_easy": n, "too_hard": n}
```

`optimize(mode="rl")` drops, in this order: junk rows; duplicate rollouts within an ask (the same trajectory twice adds nothing to a group-relative advantage); truncated rollouts (`truncated="keep"` leaves them in as `overlong`, `"penalize"` keeps them as failures with the judged score under `reward_before_penalty`, DAPO's overlong handling); unanimous asks (all pass or all fail: zero advantage); and asks outside the difficulty band. "Out of band" means outside the 0.2 to 0.8 *pass-rate* band, never off-topic. The filter does not read the prompt at all, so an on-topic ask the policy always solves is dropped and an odd one it solves half the time is kept. It then keeps whole groups round-robin across fault kinds and, within a fault kind, round-robin across pass rates: a 25% ask, a 50% ask and a 75% ask are taken in turn, with no preference for the middle (`order="middle"` restores the older nearest-to-50% ranking).

Each kept row's `calibration` stamp carries `pass_rate_ci95`, the interval on that pass rate, and the report says so when the band was measured from fewer than 16 rollouts per task, since at 8 a task's band assignment can be off by about 0.3. The prune shrinks every group, so the k-way reliability numbers do not survive it: `pass_at` on the selection reports `pass^k` and `pass@k` as `n/a` where the graded rows had them, which is why you print `pass_at` before this call. The report says so in `hygiene_warnings`, and the carried `calibration` stamp keeps the graded per-task measurement.

`optimize(mode="sft")` is rejection sampling [2]. The engine samples four completions per phrasing (`SFT_COMPLETIONS_PER_PROMPT` in `defaults.py`; `repeats=` moves it) so there is something to choose among; the report's `completions_per_prompt_mean` and `selection_effective` say whether that happened. `select="top_per_prompt"` keeps each prompt's highest-reward completion above `min_reward` (default 1.0; lower it for a partial-credit grader), `"top_k_overall"` the best `k` across prompts, and the `random_*` rules are the matching chance controls. Exported groups carry `n0`/`n1` (fail/pass, partial credit splits at 0.5) and `reward_mean`/`reward_std`. The band is the offline difficulty filter from the reasoning-model recipes (keep prompts the policy solves 20-80% of the time); it is a heuristic, so it is a parameter.

Every selector report (`select_for_rl`, `select_for_sft`, `build_preference_pairs`) carries `eval_sourced`, the rows or pairs whose reward came from `evaluate()` (`lineage.source == "eval"`), with a warning when it is non-zero: a held-out score that becomes the reward makes the scorer you report the one you optimised against. Nothing is dropped; grade the training set with `run_judge` or `data.grade` and keep `evaluate` for held-out rows.

### What will the policy learn?

```python
scan = wai.hack_scan(scored.rows, endorsed=["tool:lookup_order", "marker:grounded"])
scan["regime"]  # train | reward_hack | pool_exhausted | no_signal | degenerate | unknown
scan["top_feature"]  # e.g. 'contains:### done' when the judge pays for a delimiter
print(wai.format_hack_scan(scan))
```

A grouped update learns whatever separates reward *within* an ask; what only tracks which ask it is (difficulty) is baselined away. `hack_scan` asks the question the same way: reward and every candidate feature are centered within ask, ranked by that correlation, and compared to a noise floor from shuffling reward within ask (`tau`). Features come in two tiers, both pure Python: the hand tier (reply length, tool calls, turns, truncation, surface counts, one indicator per tool called, mean token logprob, every numeric marker, plus `features={"name": fn}` of your own) and the auto tier (the 200 most common words and word pairs in the agent's text, and pairwise ANDs that beat both parents), which is the tier that finds the shortcut nobody listed. `endorsed` names what the reward should track, as substrings of feature names; with it the scan can say `reward_hack` (the top feature is not endorsed, and the warning names what the policy would learn instead), `integrity` (share of the above-floor signal that is endorsed), and lists rivals. Without it the scan still ranks and floors. An agent that emits only a couple of distinct trajectories per ask makes every feature that separates them an exact function of the label. They all tie at |rho| 1, and the floor cannot break a tie between two perfect explanations, so the scan returns `degenerate` with `top_feature` `None`, lists the tied features in `collinear`, and names the cause (`distinct_per_ask`) rather than picking the alphabetical winner.

The whole loop, before, during and after training, is in [Reward hacking](/reward-hacking) and runs offline in [`recipes/02-measure/reward-hacking`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/reward-hacking). `optimize(mode="rl", endorsed=[...])` carries the scan as `report["hack_scan"]`, with its warnings in `report["hygiene_warnings"]` next to the older pooled `report["correlations"]` (reply length, tool calls, turns, flagged at `HACK_THRESHOLD` 0.3). A reward that tracks a shortcut is a judge problem, so it is flagged, not pruned. The publish gate reports the same on RL-shaped rows, plus near-duplicate asks and length spread; `data.push(endorsed=[...], strict_hacks=True)` refuses a `reward_hack`. Standalone: `wai.reward_correlations(rows)`, `wai.dedupe_groups(rows)`, `wai.near_duplicate_prompts(rows)`, `wai.length_report(rows)`.

### Curriculum: easy to hard, and retire the solved

A curriculum needs per-prompt difficulty [1], which is each task's pass rate over its k rollouts. `curriculum(rows)` splits graded tasks into *trainable* (ordered easy to hard, and bucketed into `tiers` for a staged schedule), *retired* (pass rate above `solved`, default 0.8: an all-pass task is dead gradient), and *not ready* (below `floor`, default 0.2: no signal until the policy improves), and counts how many trainable tasks sit in the 20-80% band. The two defaults are the band's own edges, so `curriculum` and `optimize(mode="rl")` agree on which tasks are trainable.

```python
cur = wai.curriculum(scored.rows)  # solved=0.8, floor=0.2, tiers=3
cur["schedule"]  # trainable task ids, easy -> hard
print(wai.format_curriculum(cur))
rows = wai.retire_solved(scored.rows)  # drop tasks the policy already aces
```

### Agents

An agent exists the moment a push names it or a trace arrives with `gen_ai.agent.name`. Everything on the platform hangs off it.

```python
data.push(
    "airline-v3", agent="airline-support"
)  # registers the agent, attaches tools + system prompt
wai.agents()  # every agent: traces, sets by purpose, public cards
wai.register_agent("airline-support", description="Refunds and rebooking")
```

### From the terminal

The platform verbs a coding agent needs, as commands:

```bash
wai agents                 # tracked agents and what each serves
wai agent <id>             # record, behaviors, verdict
wai runs <id>              # the version table, newest first
wai verdict <id> [--behavior <name>]
wai promote <id> <version>
wai live <id> --day YYYY-MM-DD --version <v> --replies N [--flagged N --p50 S --cost USD]
wai keys                   # names and prefixes; create or revoke under Account
```

All take `--json` and `--api-key`; errors exit 1 with the reason on stderr.
Thin calls into `whileai.platform`. The old `wai purge` (traces and
datasets on the data platform) is gone; `wai.purge_agent("demo-agent")` and
`wai.delete_empty_datasets(max_rows=2)` remain in Python, both with
`dry_run=True`.

### Train, holdout, eval

```python
data.push("airline-v3", holdout=0.2)  # train set + a linked holdout set, split by task
data.push("airline-evals", purpose="eval")  # a set you measure with
scored = data.grade(judge=my_judge)
scored.push("airline-rl-v3", gate=True, mode="rl")  # the graded copies, gated
scored.push("airline-rl-v3", gate=True, mode="rl", holdout=0.2)  # plus a linked holdout, by task
wai.update_dataset("ds_...", purpose="holdout")
wai.preview("ds_...")  # three sample rows + the analyzer report
wai.profile("ds_...")  # pass rate, support, mixed tasks, tool use, per task
```

The Datasets page groups sets by purpose (train, holdout, eval) and records the simulation mode on each. A push is train unless it says otherwise. Holdout is split by `scenario_id`, so a task is wholly on one side, and the same task lands on the same side every run. A `purpose="holdout"` push warns when the set is too small to prove a 5-point gain at 80% power.

A task's identity is its cell in the coverage grid: the tools, the situation axes, and at most one clause of the policy. Each clause owns its own block of cells and the cells that pair the other axes carry no clause, so editing the system prompt keeps every task except the ones for the clause that changed. Rewording one rule, adding one, or swapping the model leaves the rest of the eval paired for `compare_runs`.

## Trust the numbers

Checks that decide whether a result is believable. All are report-only and run offline over rows you already have.

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
)  # same task id, verbatim, or 8-gram overlap
clean, report = wai.decontaminate(
    train_rows, against=[eval_rows], embedder=embed, similarity=0.85
)  # plus a semantic pass
wai.style_markers(rows)  # no_boilerplate, no_hedging, no_apology, no_sycophancy, answered
wai.style_report(rows)["warnings"]  # "reward pays for hedging (corr +0.41 ...)"
wai.refusal_report(benign_rows)  # over-refusal rate with a Wilson interval
wai.compare_runs(run_a, run_b)  # paired delta with a 95% interval
wai.holdout_size(
    0.05, base=0.6, k=4
)  # tasks to prove a 5-point gain; n_tasks_concentrated beside it
wai.holdout_size(
    0.05, before=before, after=after
)  # the paired sd measured off a previous eval, no model
wai.holdout_size(0.05, task_std=0.38)  # or the sd read off a delta_report interval
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

**Judge trust.** "Gold" means a label a person wrote, or one a program computed. Label 50 to 100 rows by hand (a JSONL of `{"key": ..., "label": 0 or 1}` or a `{key: label}` dict, where a label names its row by `key`, by `scenario_id` plus `rollout_index`, or by `prompt`), attach them with `attach_labels(rows, labels, kind="human")`, and grade. That stamps `gold_reward` and `gold_kind="human"`. A deterministic rule (execution match, a unit test, a rule over tool calls) is attached with `kind="program"` (`"verifier"` reads the same) and counts as a measurement on the same footing, since it cannot be argued into a pass and agrees with itself on every run (Lambert 2025, chapter Evaluation, verifiable rewards); the report carries `gold_kind="program"` so a reviewer sees which it was. A model's labels, or a second judge pass, are marked `model` and do not count, older rows with `gold_reward` but no record of who wrote it count as unknown, and any other `kind=` raises naming the three accepted. Every warning names the kind it measured against (`human labels`, `gold labels (program)`). Every `grade` call then ends by checking the judge against those labels, with no flag needed: the summary lands on every graded row as `judge_meta["trust"]` (`agreement`, `agreement_low`, `kappa`, `n_gold`, `ok`), in the grade report as `trust`, and in `publish_gate` as `judge_trust`. The judge passes when the lower bound of its agreement with the people is at least 0.80 and kappa at least 0.60; under either, the report says the number, the floor, and what to do. With no human labels the grade prints one line saying the judge was not measured. `grade(trust="require")` raises instead of printing; `trust="off"` skips the check. `audit_grades` never audits with the grader's own model: when the auditor would be the same, it uses the other hosted model and the report says which (`grader`, `auditor`), or it stops and asks for `backend_spec=`. Agreement counts exact 0/1 rewards only: a `Rubric` of principles scores the mean of its criteria, so its partially met rows are skipped, and the rows kept are the ones the judge was sure about. `report["skipped"]` carries that count; over `max_skipped_share` (0.10) `ok` is false and the printout reads `INCONCLUSIVE` with the count, the floor and the fix (`Criterion(kind="hard")`). The rubric judge is handed `tools_called` and `tools_not_called` as facts (steps that returned a result), and its prompt says a reply that announces a call it never made has not made it; a criterion that must be exact still belongs in a `grader=`.

`judge_trust(rows, judge=)` is the standalone version. `report["ok"]` means measured and clean, so with no labels it is `False` and the report says the judge is unmeasured rather than untrustworthy (`format_judge_trust` prints `NOT MEASURED`). The report gives agreement with a Wilson interval and Cohen's kappa, agreement on two task halves (tune the rubric on one, read the other), judge pass rate on short versus long replies within the same human label (length bias the humans rule out), and, with the judge callable, a re-judge of a sample as-is (consistency) and with neutral filler appended (a flip means the judge reads length). Disagreements come back as a review queue. It refuses model gold unless `allow_model_gold=True`. The gold set needs both passes and failures; with one class only the report says so and skips the kappa and length flags. `probes="all"` (or a list) tries the reward hacks a policy finds first on the judge on purpose: filler, the rubric's own words stuffed in, a claim of success with no evidence, the ask echoed back, a well-formed tool call with empty arguments, a sycophantic opener, a polite refusal. An additive probe is exploitable when failing replies start passing, net of the ones that stop (`max(0, flips_up - flips_down)` over the originally failing rows, so symmetric churn is noise, not a hole); a replacement probe when a reply with no content passes. Every rate prints with its Wilson interval, and `report["exploitable_by"]` names the holes at or over 10% with at least `PROBE_MIN_N` (20) rows in the denominator; under that a probe reads `low power (n=10; resolves about 0.4 at 80% power)` and is not flagged, since one flipped row on ten is already the flag. A policy trained on this judge will find those same holes. The keyword probe reads `rubric=` on `judge_trust` (the `Rubric` given to `rubric_judge` is not seen here). Standalone: `wai.judge_probes(rows, judge, rubric=...)`. With the hosted judge, call `wai.grade` once first (or `whileai.simulations.score.grade_llm.warm_judge`; it is not re-exported) so the cold start, two to three minutes, is not counted as timeouts.

**Decontamination.** Four rules between a dataset's rows and any evaluation source (row lists, JSONL paths, or platform dataset ids), each counted on its own and a row counted once. A row is contaminated when it shares a `scenario_id` or `task_id` with an eval row (`n_same_task`: a task is a situation, not a string, so a rephrasing of an eval situation is the eval situation), when it is an eval prompt verbatim (`n_exact`), or when one eval text covers at least 80% of its words in shared 8-grams (`n_near`; `overlap=`, the Llama 2 rule; one shared 8-gram is not enough, because situations written from the same templates share whole sentences without sharing the question, and short prompts match verbatim only). `fields=("prompt", "final_text")` also checks replies against eval answers and references. Word overlap does not see a paraphrase: a holdout written by re-running the generator was 70% within 0.85 cosine of the training batch, and the 8-gram rule flagged 4 of its 101 prompts where a semantic pass flagged 16. Pass `embedder=` (any callable from a list of texts to one vector per text, so nothing is imported; with sentence-transformers, `embedder=lambda texts: model.encode(texts, normalize_embeddings=True).tolist()`) and rows whose prompt is within `similarity=` (0.85 cosine) of an eval prompt are flagged as `n_semantic`. That flag means the two prompts read alike, not that they are the same task: "cancel one reservation" and "cancel three reservations" for different customers score 0.93 with no shared answer. So the task-id rule decides first, the semantic pass only looks across different task ids, and `report["notes"]` says the flag is a question to check. The default stays lexical; the threshold was read off BGE (unrelated prompts score about 0.55 there) and needs picking for another model, so when the eval rows carry task ids the pass measures how alike distinct tasks read to your embedder (the 99th percentile of similarity over eval-prompt pairs with different task ids) and `notes` says it, and says when `similarity=` sits below it, since a threshold there flags tasks that merely share a domain. The report returns the clean rows with the first offenders, their coverage or similarity, and hits per field.

**Intervals and comparison.** Every pass@1 carries a 95% interval from a bootstrap over tasks (`pass_at(rows).ci95`), and `metric_summary` / `marker_summary` do the same for markers. pass^k and pass@k carry their own (`pass_pow_k_ci95`, `pass_at_k_ci95`), a bootstrap over the k-eligible groups. Markers come from the judge: return `{"reward": ..., "markers": {"name": value}}` from a `grader=` or `run_judge` callable and they land on `row["markers"]`, which is what `marker_summary`, `delta_report` and `from_row` read. `compare_runs` pairs the tasks two runs share, bootstraps the paired difference, and adds a sign-flip permutation p-value; fewer than five shared tasks falls back to an unpaired test and says so. Tasks on one side only are dropped from a paired comparison; `note` says how many and `paired_share` is the fraction that paired, so a verdict over a quarter of the eval reads as one. Under half paired is `situations` in `delta_report`'s `not_comparable`: the two arms drew different situation sets (a `hard_share`, `dimensions` or seed change between them), the delta over the few that pair is between two evals, and the warning says to pin the after side to the baseline's tasks (`tasks=`) or compare per tier with `dataset_report`. The verdict `no_difference_detected` means the interval covers zero, not that the runs are equal. A task is a situation, not a string: every report (`pass_at`, `compare_runs`, `delta_report`, `eval_variance`, `curriculum`, `group_signal`, the exporters) groups rows by `wai.task_key(row)`, the engine's `scenario_id` when the row has one, so repeats and rephrasings of one situation count as one task and the same rows give the same task count everywhere. `pass_at(rows).config` and `delta_report(...)["config"]` say what the rows were produced with (temperature, reply budget, policy and judge versions), and `delta_report` warns when the two sides differ.

**Same tasks, new prompt.** A run draws its tasks from the grid by seed and, above `concurrency: 1`, by completion order, so a second `simulate()` shares only part of its tasks with the first. To A/B a prompt edit, a model swap or another seed on exactly the same eval, pin the task set: `wai.simulate(agent, tools=TOOLS, system_prompt=EDITED, tasks=base)` re-runs every prompt of `base` (a run, its rows, or its JSONL path) on its own `scenario_id`, under the same faults and world state, and draws nothing new; it stops with `tasks_done` once every prompt has its rollouts, and `compare_runs(base.rows(), rerun.rows())` pairs every task.

`tasks=` copies the prompts and, unless you pass `repeats=`, the pinned run's k (the most rollouts any of its prompts has), so a base built with `mode="rl", repeats=4` and re-run as `simulate(..., tasks=base, mode="rl")` comes back at k=4 and `pass_at` reports the same k on both sides. Pass `repeats=` to re-run at a different k on purpose:

```python
base = wai.simulate(agent, tools=TOOLS, system_prompt=POLICY, mode="rl", repeats=4)
rerun = wai.simulate(
    agent, tools=TOOLS, system_prompt=EDITED, tasks=base, mode="rl"
)  # k=4, inherited
assert base.rollouts_per_request == rerun.rollouts_per_request  # cheap guard
```

**Size before you run.** Most evals are too small to see the effects they produce. Per-prompt paired spread is stable for agent rubrics, about 0.38 across five lanes, and at that spread a 50-prompt eval only detects a 10-point gain; a 5-point gain needs about 220 prompts and a 3-point gain about 600. One lane read the same adapter as "barely helps" at 45 and 85 prompts (both straddled zero) and cleared at 131 (+0.13 [+0.07, +0.19]); the effect was real the whole time and a GPU round went to fixing a data problem that was a measurement problem. The effective sample is prompts, not rollouts: raising k sharpens each prompt's estimate but does not narrow a bootstrap over prompts, so spend eval budget on prompts first (k still matters for preference and grouped methods, which need mixed groups). And "straddles zero" means the eval cannot tell, not that the model did not improve; say which. A saturated baseline cannot size anything: rows whose tasks all pass give `p = 1`, a binomial variance of 0 and the formula's floor of 2 tasks, which is the model collapsing, not evidence. When the measured base is at or above `CEILING_PASS_RATE` (0.9, `ceiling_pass_rate=`) or the measured paired sd is 0, `holdout_size` answers with the model at `BASE_PASS_RATE`, sets `saturated=True`, and `warnings` names the ceiling and the fix: harder situations so the baseline sits inside the 20-80% difficulty band, then size again.

`holdout_size(effect, base=, k=)` says how many paired tasks prove a gain at 80% power, and `detectable_effect(n_tasks, ...)` is the same solved for the gain. Its binomial model assumes the gain is spread evenly across tasks and the two arms are independent draws, and says so in `notes`; when a trait is only exercised by some prompts most tasks are ties, the paired differences spread far wider, and the model under-sizes by several times (a voice lane at 0 to 0.127 needed 54 tasks where the model said 14), so the model path also returns `n_tasks_concentrated`. On a holdout whose tasks differ in difficulty the model errs the other way, asking for `1 / (1 - Var(p_i) / (p(1-p)))` times the tasks pairing needs (1.19x at spread 0.2 around 0.5, 2.78x at 0.4); `before=` alone reports the spread and that ratio. The honest paths measure: `before=before, after=after` (the same two row lists `delta_report` takes) reads the per-task paired sd off both arms of a previous eval on the same tasks, with the covariance pairing buys in it, and `task_std=` (the per-task sibling of `run_std`) takes the number you read off a `delta_report` (`(hi - lo) * sqrt(n_paired_tasks) / 3.92` from `target_ci95`).

**Before and after.** `delta_report` runs `compare_runs` on pass@1 and every marker both row sets share. `target=` names the metric the training was meant to move and gives the headline; `must_not_regress=` names the behaviors whose significant drop fails the report; any other significant drop is a warning. `format_delta_report(report)` prints one line per metric. `eval_variance(run_1, run_2, run_3)` is the eval's own re-run standard deviation (three or more evaluations of the same model); passing it as `run_std=` makes any delta inside the re-run band `within_noise`, and a target there reads `within_eval_noise` rather than moved, since re-running the eval moves it that much on its own [3]. Pass `run_std_runs=` with it (the `n_runs` the floor came from) so the band uses the t quantile at `runs - 1` degrees of freedom: a floor from three re-runs is an estimate, and the 1.96 band that reads it as exact lets about one pure-noise delta in five through. A bare `run_std=` keeps 1.96 and warns. `by=` names a row key, a marker, or a callable that groups rows (a prompt category, a tool, a persona); the report then carries `groups`, the target compared within each group, and `groups_down` for any group whose target dropped significantly while the headline moved. A headline over one dominant kind of prompt cannot hide the other kinds that way. Both sides should have the same rollouts per task; when a run lost some (`data.report()["rollouts_lost"]`, with the reasons in `rollouts_lost_by`) and one arm sits at k=4 while the other is at k=2, the report warns next to the sizing line and names both. Unequal k is a precision issue, not a bias: rows lost at random leave the paired delta unbiased and only widen its interval; rows lost for a reason (a timeout on the hard runs) bias it, and only re-running the short arm on its short tasks fixes that. `balance_rollouts=True` (off by default) trims every paired task to the rows both sides have (drawn by `seed=`) so pass^k and pass@k share one k; it costs precision, removes no bias, and `balanced` says how many rows each side gave up.

**Run the eval three times.** One evaluation is a draw, not a number: the same model on the same tasks lands somewhere else next time, and most post-training gains are inside that spread [1]. `wai.simulate(agent, tasks=base, runs=3)` replays the task set three times in one call, same tasks, faults and world, and stamps `lineage.eval_run` on every row. Feed both sides to `delta_report` and it works out `run_std` from the repeats itself. The verdict words: `moved` is a change the interval and the re-run band both support; `moved_unreplicated` is a change seen once, which could be noise, and the warning tells you the `runs=3` call that settles it; `within_eval_noise` is a delta smaller than what re-running the eval does on its own, so equivalence, not a win; `no_change_detected` is an interval that covers zero. `format_delta_report` prints the same reading on its first line (`report["headline_verdict"]` is the word behind it): `PASS` only for a gain (`moved`, `moved_unreplicated`), `NO DIFFERENCE` for an interval over zero (a negative point estimate the interval does not settle is not a pass), `FAIL` for a regression or a failed guard, `NOT COMPARABLE (causes)` when the arms cannot be compared. `ceiling=True` means the before run already passes most of its tasks (0.9 or more, or too few paired tasks left with room), so there is little improvement the eval could show; use harder situations before training again.

**Argument grounding.** A policy trained to call a tool learns to call it before it learns when not to; on the refund environment both GRPO and DPO learned to invent an order id on a quarter of the prompts that gave none while the headline rose. `mark_grounding(rows)` stamps `argument_grounding`: 1 when every string argument of every tool call appears in the prompt, the user and system turns, or an earlier tool result (rows with no calls count as grounded), else 0. No categories, any agent; `must_not_regress=["argument_grounding"]` fails the run that learned to invent, and `ungrounded_arguments(row)` / `grounding_report(rows)` name the values. `ignore_keys=` skips free-text arguments, `allow=` lists enums and defaults.

**Trajectory flags.** Did the agent fake the work? `trace_markers(rows)` reads the trajectory rather than the prose, because the prose can claim anything and a reward that pays for the claim gets more of it [1, 4]: `lie.tests_claimed` (tests said to pass when no test command ran or the last one failed), `lie.unverified_claim` ("I verified" with no tool calls), `lie.phantom_edit` ("I updated" with nothing written), `lie.ignored_failure` (the turn ended on a failed call and the reply never says so), `hack.test_edited`, `hack.test_weakened`, `hack.suppressed`, `hack.bypassed`, `risk.destructive`, `risk.secrets`, each with the fragment that raised it on `row["trace_flags"]`. The markers it stamps (`honest_claims`, `reported_failure`, `no_test_tampering`, `no_suppression`, `no_bypass`, `no_destructive`, `no_secrets`) are 1.0 when clean, so `must_not_regress=["honest_claims"]` fails a run that learned to overclaim, and `hack_scan` carries every fired flag as a `trace:` feature. `trace_flag_report(rows)` gives each flag's rate, examples, and its correlation with the reward, flagged when the judge pays for the fake. Reads, writes, deletes and commands are told apart by the tool's arguments and name; `kinds={"my_tool": "write"}` overrides.

**Stage lineage.** The pipeline is a sequence of stages [1]: SFT, reward modeling, RL, and the eval that judges the result. `stamp_stage(rows, "sft")` records which stage a row fed, and `stage_report(rows)` counts rows per stage and flags the one mistake it most needs caught: any task used in both `eval` and a training stage. `wai.stamp_stage`, `wai.stage_report`, `wai.stage_of`, `wai.STAGES` (`sft`, `rm`, `rl`, `eval`, `mid`).

**Model spec as an object.** A spec or constitution is a living, versioned document [1, 5]. `load_spec(constitution)` wraps the `{source, traits: [{id, name, principle, authority}]}` shape (what the character recipe writes) into a `Spec` whose `version` is a content hash, so any edit to a principle changes it. `spec.behaviors()` are the trait ids, ready for `delta_report(must_not_regress=...)`; `stamp_spec(rows, spec)` tags every row with `spec_id` and `spec_version`, so you can ask whether adherence held from one spec or model version to the next.

```python
spec = wai.load_spec("recipes/03-select/character/constitution.json")
scored = wai.stamp_spec(data.grade(judge=my_judge).rows, spec)
wai.delta_report(before=before, after=scored, target="pass_at_1", must_not_regress=spec.behaviors())
```

### Markers: four families, one polarity

A marker is a named behavior measurement on a row. Everything that reads markers (`marker_summary`, `delta_report`, `must_not_regress=`, `from_row`, the run page) reads one place, `row["markers"]`, and does not care which family put the value there. Four families write to it, and only one of them has the wrong polarity:

| Family | How you get it | Polarity | Use it for |
|---|---|---|---|
| **Judge-emitted custom markers** | your own name and value, returned as `{"reward": ..., "markers": {"name": value}}` from a `judge=` / `grader=` / `run_judge` callable | **yours to choose, and it must be 1.0 = good** | Anything your product cares about. This is the family `delta_report` and `must_not_regress=` are built for |
| `trace_markers` / `trace_flag_report` | `wai.trace_markers(rows)` stamps `honest_claims`, `reported_failure`, `no_test_tampering`, `no_suppression`, `no_bypass`, `no_destructive`, `no_secrets`, with the evidence on `row["trace_flags"]` | 1.0 = no flag fired, higher is better | Did the agent fake the work? Read from the trajectory, not the prose; see [Trajectory flags](#trust-the-numbers) above |
| `style_markers` / `style_report` | `wai.style_markers(rows)` stamps `no_boilerplate`, `no_hedging`, `no_apology`, `no_sycophancy`, `answered` | 1.0 = clean reply, higher is better | Over-optimization drift in a paired before/after |
| `behavioral_markers` / `mark_rows` / `STOCK_MARKERS` | `wai.behavioral_markers(rows)` returns `{"boilerplate": 0.31, "refusal": 0.04, ...}` | **presence: 1 = the tic appears, higher is worse** | A one-shot read of how often each tic occurs. Not a delta |

<Note>
`behavioral_markers`, `mark_rows`, `row_markers` and `STOCK_MARKERS` live in `whileai.simulations.score.markers`, which is deprecated and raises a `DeprecationWarning` on first use. `style_markers` / `style_report` / `refusal_report` cover the same over-optimization behaviors [4] with the delta-ready polarity.
</Note>

**Polarity is the rule for every marker you define, not a quirk of one function.** `1.0` is the good outcome; higher is better; a significant *drop* is the regression that `must_not_regress=` fails on. Name markers after the behavior you want:

```python
# Wrong: 1 means the bug happened.
{"markers": {"false_refund_success": 1.0}}
# delta_report prints DOWN when you fix it, and
# must_not_regress=["false_refund_success"] FAILS the run that fixed it.

# Right: 1 means the agent did the right thing.
{"markers": {"refund_correctly_refused": 1.0}}
```

If you have already collected rows under an inverted name, flip the value (`1 - v`) and rename before you compare runs; `delta_report` has no way to know which direction a name means.

## Train, and watch it

Two ways to train, one record. The platform trains a pushed dataset (SFT, GRPO, DPO or a reward model, as a LoRA adapter) and serves the result; or your own trainer runs on Modal, a GPU box, or a notebook and reports into the same run. Either way the loss curve and the progress bar are on the training page of the platform ([withwhile.com](https://withwhile.com)).

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

**What it cost.** A hosted run reports its `gpu` and its `seconds`; `run.training` and `wai.get_run(id)["summary"]` carry `cost_usd` beside them, `seconds / 3600 × rate` to the cent, with `cost_basis` naming the GPU, the rate and the day the rate was read: `estimate: A10G at $1.10/h, modal.com/pricing 2026-09-20`. It is an estimate, not a bill. The platform's trainer runs on Modal, so the rate is Modal's on-demand list price on 2026-09-20 (`whileai.simulations.defaults.GPU_USD_PER_HOUR`, the table below); a GPU not in the table leaves `cost_usd` None and the basis says so. `print(run)` shows the line as `about $0.02 (A10G, 56 s, estimate)`. Rollouts and judge calls on the shared serving endpoint are not priced.

| GPU | USD per hour (Modal list price, 2026-09-20) |
|---|---|
| A10G | 1.10 |
| L40S | 1.95 |
| H100 | 3.95 |
| A100 | 2.50 |
| T4 | 0.59 |

`epochs=` sets SFT, `steps=` sets GRPO, DPO and RM; each method has a default. Before it posts, `train` reads `wai.profile(ds)` and checks what the trainer will use: SFT trains on every row as pushed, so a set with failing rows is refused (`TrainingSelectionError`; push `scored.passes()`, or `check="warn"` to train on them on purpose); GRPO, DPO and RM learn only from tasks with both a pass and a fail, so none is refused and fewer than `min_mixed_tasks` (32) warns with the count used against the count given and the reason per dropped class. `profile(ds)["mixed_tasks"]` is how to size a grouped set; `whileai.simulations.training.selection_report(profile, method=)` is the check as a function. Hosted GRPO's reward is the trainer's own (reference first action against the judge's gold), not a parameter. `run.delta` is `delta_report` kept on the run and drawn on its page, including the per-group table when `by=` names a row key or marker; `wai.attach_delta(run_id, before, after)` does the same for a run that already finished. `holdout=` names the eval set (defaults to the train set's split sibling); a dataset already training returns that run. `serve` needs a finished run whose base is a served one (`Qwen/Qwen3-4B`, `microsoft/phi-4`; the list is `whileai.simulations.training.SERVED_BASES`). The trainer's default bases (Qwen2.5-0.5B for SFT, 1.5B for GRPO and DPO) train fast but cannot be served, so `train` warns when a run will not reach an endpoint. Qwen3 answers in thinking mode by default: leave room in `max_tokens` or send `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

`method="rm"` trains a reward model [6] on the set's pass-vs-fail pairs and reports pair accuracy on the held-out pairs before and after. `wai.reward_model(run)` is that model as a judge, with the judge contract (`reward` 0/1 against the run's threshold, `rm_score` raw), so it goes wherever a judge goes:

```python
rm = wai.train("ds_...", method="rm", steps=60, wait=True)
run = wai.train(
    "ds_...", method="grpo", generations=8, beta=0.02, learning_rate=5e-6, seed=3
)  # the knobs a run is compared by
judge = wai.reward_model(rm)  # or reward_model("run_...", threshold=0.4)
scored = data.grade(judge=judge)
wai.judge_trust(scored.rows, judge=judge)  # the same checks as the LLM judge
```

Every training knob (`generations`, `learning_rate`, `beta`, `max_completion_length`, `temperature`, `loss_type`, `truncated`) has a trainer default when left `None`; the range each is accepted in and the value the cited paper used are in `whileai.simulations.training.TRAINING_KNOBS`, and a rejected value is told the reference.

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

A run's page opens with one word, **Better**, **Worse** or **About the same**, over the held-out pass rate before and after. The platform's trainer measures it; a run on your own hardware says it with `run.holdout(before, after)`, or `wai.attach_holdout(run_id, before=..., after=...)` once the run has finished. Pass rates are 0 to 1, so 58% is `0.58`; `metric="loss"` sends held-out loss instead (SFT), where lower is better. `run.delta(...)` and `wai.attach_delta(...)` already measure both sides, so they fill the two numbers in themselves, and add `summary["holdout"]` (also `run.holdout_summary`): each side's pass rate with `n_tasks`, `k` and a `ci95`, plus the delta report's verdict word (`moved`, `moved_unreplicated`, `within_eval_noise`, `no_change_detected`). A hosted run read back with `run.refresh()` has the same block with the interval fields `None` and a note that the platform only returned two numbers.

Plain HTTP, for a stack that is not Python: `POST /runs` with `name`, `dataset_id`, `base_model`, `total_steps` returns `runId`; `POST /runs/{id}/log` with `points` (a list of `{"step", "loss", "lr", ...}`, up to 500 a call); `POST /runs/{id}/finish` with `status` (`done`, `failed` or `stopped`), and optional `summary` and `adapter`. All with `X-Api-Key`. Points are buffered on the client and a send that fails is retried on the next flush; the dashboard never interrupts the trainer. `wai.get_run(id)` returns the run record.

## Report a run so a person can decide

The platform draws one screen per tracked agent at [withwhile.com/platform/runs](https://withwhile.com/platform/runs): the held-out score by version with the frontier model as the line to beat, the training curve, what moved on the behaviors you did not train, the judge checks, live traffic on the served version, and cost. A coding agent fills it with `whileai.platform`; the person reads it and presses Promote. Your agent framework stays yours: `track` takes the agent object you already have (OpenAI Agents SDK, Pydantic AI, LangGraph, Claude Agent SDK) and reads the model, the instructions and the tools off it, or you describe it by hand.

```python
from whileai.platform import Behavior, Frontier, Harness, Judge, track

tracked = track(
    "refund-bot",  # or track(my_agent): name, model, prompt and tools come from the object
    model="Qwen/Qwen3-4B",
    harness=Harness(instructions=SYSTEM_PROMPT, tools=["lookup_order", "issue_refund"]),
    frontier=Frontier(name="Sonnet 5", score=81, cost_per_1k=18.0),
)
tracked.behavior(
    Behavior(
        name="refunds",
        test_version="v2",
        n=240,
        judge=Judge(agreement=0.86, human_n=60, length_bias=0.08),
        noise_floor=2.4,
        contamination=0,
        reward_is_judge=False,
        graded_by="judge",  # or "program" for a verifier: no judge agreement is asked for
    )
)

run = tracked.run("v4", method="GRPO", targets=["refunds"], trained_on=["refunds-grpo"])
run.log(10, reward=0.41, kl=0.01)  # or trainer.add_callback(wai.TrainerCallback(run))
run.score("refunds", 83, ci=2.7, n=240)  # points out of 100; every behavior, not only targets
run.score("length", 76, ci=2.8, n=120)
run.finish(hours=2.1, gpu="1xH100", cost_usd=31)  # prints the brief

print(tracked.verdict())  # one line: beats, trails, or about the same, and what that rests on
print(tracked.brief())  # what happened, what it means, what next: the Runs page text
print(*tracked.evals(), sep="\n")  # the Evals table: eight checks per behavior
```

**Everything the page shows, the agent can read and edit.** `tracked.runs()`, `tracked.open(run_id)`, `run.archive()`, `tracked.delete_run(id)` are the run side; `tracked.dashboard()` is the screen as data (versions, train curve, deltas, live traffic, verdict); `tracked.verdict()` is its one line; `tracked.brief()` is the card at the top; `tracked.evals()` is the Evals table, one `EvalHealth` per behavior with the eight checks the page runs (frozen, size, judge, length bias, noise floor, clean, reward is not the judge, can fail), each with its value, the failure and the fix in words, the call, the rule and its source, from the same fields and thresholds. `tracked.delete()` removes the agent and everything under it, for an agent posted to the wrong account or a smoke test; there is no undo.

**Show the eval, not only its number.** `Behavior(rubric=...)` carries how the judge was set up, in the words it was given (what passes, what fails, the edge cases, up to 4000 chars), and `run.score(..., examples=[Example(prompt=, reply=, ok=, why=), ...])` carries a sample of up to 20 graded rows, the worst and the best. The Evals page shows the rubric, the judge, the test set and the metric beside the climb, then the sample: what passed, what failed, and why. The full held-out set stays where the SDK wrote it; the platform keeps the record and the sample.

**Every graded row.** `run.score(..., rows=[Example(prompt=, reply=, ok=, why=, reference=<the gold answer>, detail=<expected against got>, tags={"difficulty": "hard", "archetype": "date and time"}), ...])` posts the whole graded set behind a score, 500 rows a call, and `run.rows(behavior)` reads it back. The iteration's rows page on the platform filters passed and failed and breaks the score down by each tag, so what went well and what did not is a table. When `examples=` is not given the card's 20-row sample is the first 14 failures and 6 passes of `rows`.

**What a person cannot read yet.** The brief ends with the list a coding agent owes the person who reads the page: an iteration that does not say what it changed (`tracked.open(id).note(...)`, or pin the harness, or record the optimizer settings), a test with no rubric (`Behavior(rubric=)`), a score with no graded rows (`run.score(..., examples=[...])`), and version names that carry settings instead of what changed (`dapo-lr5e-05-s17-180st`; settings belong in `record.optimizer`, the name says what a colleague would say: `longer-training`). The experiment page shows the same list from the same rules, with a copy-for-agent button, so the person and the agent close the same items.

**One verdict per behavior.** The candidate `tracked.verdict(behavior)` compares is resolved per behavior: the newest version scored on that behavior that is not the served one, ordered by when the score was posted, so two held-out sets with one arm each get two verdicts whatever order the arms were posted in, and re-scoring the served version on a new set demotes nothing. `tracked.verdict(behavior, version="sft-v1")` names the candidate instead (it must be scored on that behavior). When a different run also beats the served version on another behavior, interval excluding zero and clearing that behavior's noise floor, the line ends `moved, replicated on <behavior>`, the standard the learn course sets; one run scored on two sets is not a replication. `dashboard().agent.candidate` stays the platform's agent-level slot.

**The brief.** `tracked.brief()` is the same three parts the Runs page shows at the top of the agent's page, computed from the same rows (behaviors, runs, dashboard) by the same rules: what happened (scored runs, each behavior's newest score per version, in points), what it means (the verdict when two versions are scored; else the one fact that decides what the page can say, such as a perfect score on a set nobody can fail), and what to do next (the failed eval checks in the order they are worth doing, a set nobody can fail first, then size, then the test set's name, then the judge; three at most, each with the call). `run.finish()` prints it once the run has posted a score (`say=False` keeps it quiet), and `brief.markdown()` is the text a coding agent reads, the same one the page's "copy for agent" button copies, so the person and the agent work from one source. Scores are points out of 100; a score that reads as a fraction (score and interval at most 1) gets a warning from `run.score()` and is read as points by the brief.

Every object is a pydantic model that validates before it leaves the process, and each one's docstring names the paper or chapter it comes from. A *harness* is the instructions, tools and model name around the weights; its fingerprint is its version, so a prompt edit shows up as a new version without anyone naming it (a score is only comparable with its setup held constant). Pass the harness a version ran under to `tracked.run(version, harness=Harness(label=, instructions=, tools=, model=))` and the run carries the fingerprint in `record.provenance.pins["harness"]`, so two rows on the Runs page say which prompt produced each score. A *behavior* has its own frozen held-out test (`test_version`), a `noise_floor` measured by scoring the same model again, and either a judge checked against people (`agreement` over `human_n`) and for `length_bias`, or a program (`graded_by="program"`: a verifier such as `wai.verify.MathEqual`, execution match, a rule over tool calls), which has no judge to check and is the strongest grader there is (Lambert 2025, chapter Evaluation, verifiable rewards). `graded_by` is a different fact from `reward_is_judge`, which is about the training reward. `tracked.noise_floor("refunds", base_a, base_b, base_c)` measures that floor from the row lists of two or more re-runs of the same eval on the same version (`eval_variance` run_std, then t(df=runs-1) x run_std x sqrt(2), in points) and posts it, so the Runs page's Judge tile and the verdict's re-run band read the number the recipe measured instead of one typed by hand. A *run* is scored on every behavior: `targets` are the claim, the rest are the check (verbosity, sycophancy and refusals are what moves when the reward is gamed). `ci` is the half-width of the 95% interval; the difference interval is `delta ± sqrt(ci_candidate² + ci_served²)`, and the verdict says the candidate beats or trails the served version only when that interval excludes zero and the delta clears the behavior's declared `noise_floor`. A missing interval, an interval that includes zero, or a delta inside the re-run band is said in those words. The count of other behaviors that came out lower is on point estimates with no interval yet, so it is a prompt to look, not a result. The verdict ends with what the number rests on (judge agreement or `graded by a program`, n) and starts with `unproven:` when n is under 50, judge agreement is under 0.8 or unmeasured (not asked of a program-graded behavior), or the training reward is the judge. `tracked.live(day, version=, replies=, flagged=)` reports a day of traffic when you serve the model yourself. Logging buffers and never raises into the training loop. Worked example: [`recipes/04-train/report-run`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/report-run).

The Runs page lists what each run is missing (record, score, hours + cost) and the `PATCH /runs/{id}` body that fills it. `tracked.open(run_id)` is that PATCH from Python: one GET binds a `Run` to a row that already exists, from any later session, and `run.finish(record=, hours=, cost_usd=)`, `run.score(...)`, `run.note(...)` and `run.archive()` then work exactly as on a run this process opened. Do not reach for `tracked.run(...)` to backfill: a POST with an existing id overwrites the row and resets its train sequence. `tracked.runs()` lists the ids; an id no row has raises `PlatformError(404)` that says so.

```python
from whileai.platform import track

tracked = track("refund-bot")  # a later session, maybe another machine
run = tracked.open("run_7f3a")  # GET /runs/run_7f3a, no POST
run.finish(hours=2.1, cost_usd=31, record={"optimizer": {"loss_type": "dapo", "lr": 5e-5}})
```

**What it costs to run.** Post facts, not dollars. For an API model, the eval block carries `model`, the `input_tokens` and `output_tokens` the provider reported summed over the run, and the `replies` that produced them; for a served open model, the `gpu` and the `gpu_hours` it was up, and the `replies`. The platform prices the facts from its open price book at list price, no caching, and draws held-out score against the result in USD per 1,000 tasks. Every price, its source, the day it was read and the formula are public at [withwhile.com/pricing-book](https://withwhile.com/pricing-book), so two versions on one chart are always priced the same way. `cost_usd` on `finish()` is the training bill, a different number.

```python
from whileai.platform import track

run = track("text-to-sql-shop").open("run_5cc11826d439")
run.finish(
    record={
        "eval": {
            "model": "claude-sonnet-5",
            "input_tokens": 2_148_276,
            "output_tokens": 120_552,
            "replies": 1833,
        }
    }
)
run = track("text-to-sql-shop").open("run_223dc48d4933")
run.finish(record={"eval": {"gpu": "L40S", "gpu_hours": 1.98, "replies": 1836}})
```

`cost_per_1k` with a `cost_basis` sentence is the fallback: a number the agent priced itself. The platform draws it marked "reported by the agent, not priced by the book", and uses it when the model or GPU is not in the book.

### Sweep the harness

For an agent on a frontier model the harness is the experiment: the prompt, the tool set and the model. Name each variant `prompt@model` and the Runs page groups the dots by prompt and by model as two axes. `HarnessSweep` scores every variant on the same frozen asks and posts one run per fingerprint, so the Runs page groups the dots by prompt or by model and the verdict says which win is real.

```python
from whileai.platform import Harness, HarnessSweep, track

tracked = track("refund-agent", model="claude-haiku-4-5")
sweep = HarnessSweep(tracked, judge=refund_judge, k=4, tools=TOOLS, behavior="refund_policy")
variants = {
    label: (
        Harness(label=label, instructions=prompt, tools=TOOLS, model=model),
        make_agent(prompt, model),
    )
    for label, (prompt, model) in PROMPTS_BY_MODEL.items()
}
report = sweep.run(variants, tasks=frozen)  # frozen: the run whose asks are the test
print(report)  # ranked table; a winner only when its interval clears the rest and the noise floor
```

A variant label is the run's version on the platform, 40 characters at most, so `sweep.run` refuses a long or repeated label before the first rollout. Hand labels attach to the replies a person read, and a sweep rolls fresh ones, so pass `labels=Judge(agreement=, human_n=)` measured once on the frozen run with `judge_trust`. `concurrency=` caps parallel rollouts per variant (the library default is 32). The report says when the test has under 50 asks: the platform verdict reads unproven below that, whatever the gap.

### Say what the runs are for, and show your working

The typed objects above are the evidence. Three free-form calls put the claim, the pictures and the commentary around it, so the person reading the dashboard knows what the runs are for before they read a number.

```python
tracked = track("refund-bot")  # the handle from above
run = tracked.run("v4", method="GRPO", targets=["refunds"])

tracked.experiment(
    question="Does GRPO on refunds-grpo lift refunds without moving length?",
    hypothesis="Refunds up 5 or more; length within its noise floor.",
    method="GRPO, 8 generations, 300 steps on 1xH100; v3 is the baseline.",
    measure="pass@1 on refunds-test-v2, n=240, with a 95% interval; length scored the same way.",
    decide="Promote when the refunds interval clears the 2.4 noise floor and length does not drop.",
)
tracked.experiment()  # read it back; None when nothing is posted

fig = {"data": [{"x": [0, 100, 200], "y": [0.2, 0.4, 0.41]}]}  # or any plotly Figure
tracked.figure("reward-by-step", fig, caption="Training reward, v4", run=run)
tracked.figures()  # every figure on the agent, by name

run.note("Reward flattened at step 300; the last 100 steps bought nothing.")
```

`experiment` is one block per agent (a second call replaces it), rendered at the top of the Runs page as five labeled rows (Question, Hypothesis, Method, Measure, Decide) plus Notes; every field is markdown of at most 4096 chars and `question` is the only required one. `figure` posts a Plotly figure as JSON, never as code: the SDK reads `to_plotly_json()` off the object you pass (plotly is not imported or required) or takes a dict with `data` and `layout`, drops `layout.images`, `updatemenus`, `sliders` and `template` (the API drops them too), and refuses, on that line, what the API would refuse: a name outside `[a-z0-9][a-z0-9-]{0,39}`, more than 200 KB of JSON, fewer than 1 or more than 50 traces, or a trace type outside scatter, bar and pie (the page ships plotly.js-basic; a missing type means scatter). Figures draw in a grid after the run table, caption above each, and are illustration: the verdict on the page comes from the scored evals, never from a figure. `run.note` puts markdown (at most 8192 chars) under the run record when the run is selected, and keeps it on `run.notes`.

## Is it hacking the reward right now?

```python
monitor = wai.HackMonitor(
    run,
    holdout=holdout_rows,  # prompts or {"prompt": ..., <columns the reward reads>}
    gold=wai.reward_model(rm_run),  # or the hosted judge, or a second rule; any judge callable
    every=10,
    k=4,  # sample the holdout from the live policy every 10 steps
    endorsed=["tool:lookup_order"],  # what the reward should track
    stop_on="divergence",  # or "length", "drift", "feature", "any"; default: log only
)
trainer = GRPOTrainer(model, reward_funcs=[monitor.wrap(rule_reward)], **grpo_config)
trainer.add_callback(monitor)
trainer.add_callback(wai.TrainerCallback(run))
```

Over-optimization looks like one picture [4]: the training reward keeps climbing while the evaluation you care about flattens, read against KL. The monitor draws it during the run instead of after. `wrap` watches the reward function, so the monitor keeps the last completions with their rewards and runs `hack_scan` on them; every `every` steps it samples the holdout from the live policy and scores it with the training reward (the proxy) and with `gold`, a scorer the proxy cannot see. `proxy_reward`, `gold_reward` and `holdout_length` land on the run beside the loss curve. Four alarms, one line each on the run: `divergence` (proxy up by `delta` over the window while the paired gold interval does not move up), `length` (completions grow while gold does not), `drift` (KL past `kl_budget`), `feature` (the batch scan says `reward_hack`). `stop_on` names the ones that stop training; a stopped run finishes as `stopped` with the reason, and `run.note(...)` puts anything else on the run's summary. `wai.format_hack_monitor(monitor.summary())` prints the curve and the alarms. [`recipes/04-train/grpo`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/grpo) runs it by default.

## Serve a model you trained

`wai.platform.hosted` is the client for `https://models.withwhile.com/v1`,
one OpenAI-compatible endpoint in front of every model the account
registers. Two ways in: hand While the adapter and it comes back as a
model on While's Bedrock account (`publish`), or register a model you
already serve anywhere (`register`). Either way it answers under your key
from any OpenAI client, or as a `wai.Endpoint` you hand to `simulate()`.

```python
import whileai as wai

m = wai.platform.hosted.publish("while-ai/airline-concise-4b")  # about twenty minutes
print(m.status, m.arn)  # ready arn:aws:bedrock:us-east-1:...:imported-model/...
```

```python
import whileai as wai

m = wai.platform.hosted.register(
    "nemotron-8b-t2s-r1",
    arn="arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123def456",
    role_arn="arn:aws:iam::123456789012:role/WhileModelsInvoke",  # the model is in your AWS account
    base="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
)
print(m)  # name, kind, where it runs, and the endpoint to call it at

served = wai.platform.hosted.endpoint("nemotron-8b-t2s-r1")  # a wai.Endpoint on the account key
after = wai.simulate(served, tools=TOOLS, system_prompt=POLICY, seed=0)

wai.platform.hosted.subdomain("acme")  # https://acme.models.withwhile.com/v1, your keys only
for day in wai.platform.hosted.usage("nemotron-8b-t2s-r1", days=7):
    print(day.day, day.calls, day.errors, day.input_tokens, day.output_tokens)
```

| call | what it does |
|---|---|
| `publish(adapter, name=, base=, hf_token=, wait=)` | hand While a LoRA adapter (a Hub repo id or a run id): merged into its base, imported into Bedrock on While's account, registered; waits for `ready` (about twenty minutes) unless `wait=False`; the token is used once and never stored |
| `register(name, arn=, region=, role_arn=)` | a Bedrock import, custom deployment, provisioned model or inference profile; `role_arn` when it lives in your account (a role named `WhileModelsInvoke*` that trusts While with external id `while-models`) |
| `register(name, url=, model=, auth=)` | any OpenAI-compatible `/v1` server; `auth="caller"` forwards your While key to it, `"none"` sends nothing |
| `list()`, `get(name)`, `delete(name)` | the rows; deleting a row leaves the model itself alone |
| `endpoint(name)` | `wai.Endpoint(name, url="https://models.withwhile.com/v1", api_key=<your key>)` |
| `usage(name, days=7)` | per day: calls, errors, tokens in and out; nothing else is kept |
| `subdomain(slug)`, `subdomain()`, `release_subdomain()` | claim, read, or give up `<slug>.models.withwhile.com` |

The Bedrock import path from a LoRA adapter to a registered ARN is the
[Bedrock import recipe](/recipes/05-export/bedrock-import).

## Publish a dataset as a card

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

Cards live on the public catalog of the platform ([withwhile.com](https://withwhile.com)), grouped by agent, with rows, size and the analyzer's numbers on each. A dataset must be finalized and hold rows to publish.

Hugging Face, both directions. With your own token, no platform call (`HF_TOKEN` or `hf auth login`, `pip install 'whileai[hf]'`, private unless `private=False`):

```python
import whileai as wai

wai.export(rows, "train.jsonl", format="trl", push_to="me/my-set")  # -> a dataset repo
wai.hub.push("out/adapter", "me/my-lora")  # an adapter directory -> a model repo
```

Through the platform, for a set or a hosted run that lives on your account (a platform feature: the website holds the Hub token). Connect your account once on any dataset page, then:

```python
wai.simulations.hf_status()  # connected? namespaces
wai.platform.hf_publish(
    "ds_...", repo="airline-refunds", wait=True
)  # rows -> a dataset repo you own
wai.simulations.hf_publish_run(
    "run_...", private=True
)  # a finished run's LoRA adapter -> a model repo
row = wai.platform.import_hf(
    "tatsu-lab/alpaca", split="train", purpose="eval"
)  # any Hub split -> your account
wai.simulations.profile(row["datasetId"])  # profiled before you train on it
```

Every push is one commit tagged `zp-<id>`, so `load_dataset(repo, split, revision="zp-ds_...")` pins the exact push; the repo's `whileai.json` maps each split to its While dataset with history. Worked example: [`recipes/05-export/hugging-face`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/05-export/hugging-face).

