# Changelog

Versions move in hundredths (`0.04` then `0.05`). PyPI normalizes them, so
`pip install zeroproof==0.4` is the `0.04` line below.

## Unreleased

- `eval_variance` returns `run_std_by_metric`: one re-run floor for pass@1
  and for every marker the runs share, from the same run means as the
  scalar `run_std` (which stays, and now matches its own entry to the
  digit). `delta_report(run_std=)` takes that mapping as well as a float
  and judges each metric against its own floor; a metric the mapping
  lacks, or carries as `None`, gets `noise_note: no_replicate_floor`, a
  warning next to the verdict, and no borrowed floor. A marker on 8 of 30
  tasks was 2.7x noisier than pass@1, so pass@1's band read a model
  compared to itself as `slipped`; with its own floor it is
  `within_noise`. `format_delta_report` prints each metric's band.
  Unchanged: a scalar `run_std` still applies one floor everywhere,
  `report["run_std"]` is the headline metric's, and no `run_std` still
  reads `moved_unreplicated`. (#300)
- Every band in `delta_report` is `noise_band(floor, n_a, n_b, df)` from
  `stats`: `floor * sqrt(1/n_a + 1/n_b)` (a delta is a mean of `n_a` runs
  against a mean of `n_b`) times 1.96 for a given floor, or the two-sided 95%
  t quantile at `df` when the report pooled the floor from `runs=3` on both
  sides (df=4, 2.78). The same function sets `eval_variance`'s `noise_band`.
  The report returns `noise_band` (the headline band) and `noise_rule`, each
  metric carries its own `noise_band`, and `format_delta_report` prints the
  rule next to the band. A flat `2 * floor` let about 15% of pure-noise
  deltas through with one run per side.

## 0.62 (2026-09-17)

- `init-evals` finds an agent whose extra parameters have defaults
  (`answer(message, history=None)`), which is how the first real bot it met
  was written; and says when no real ids were read off the tool descriptions,
  since placeholder asks then stop at "which order?" and the run is hollow.

## 0.61 (2026-09-17)

- `whileai init-evals` writes the eval harness, instead of a coding agent
  copying the recipe into the project by hand. It reads the project's
  Python with `ast` (never imports it), picks the tool list, the system
  prompt and the callable that answers a message, and writes
  `evals/agent.py` (the wrapper, with the tools converted to OpenAI
  function shape and the bot's tool runner wrapped in a thread-local
  recorder, because rollouts run concurrently), `evals/judge.py` (the
  contract, two example markers, a `classify` stub), `evals/run.py`
  (pass@1 with an interval per branch, the marker table, `scored.warnings`,
  `--gap`, and a CI gate that exits 1 under the floor and 2 on a hollow
  run) and `evals/test_judge.py` (the judge on hand-written rows, no model
  calls). It prints what it picked, so a wrong guess is one flag away
  (`--agent module:callable`, `--tools module:NAME`, `--system-prompt
  module:NAME`), and when it finds nothing the files are still written
  with every place that needs your code marked TODO. Three cold-start
  agents asked to build evals for a refund bot each spent ten to thirteen
  minutes transplanting `recipes/02-measure/eval-your-agent/run.py` by
  hand, and asked for this command by name (2026-09-17).

- A trial key is named before a hosted run spends it, not after. `whileai
  signup` records the tier (and the trial's daily input tokens and expiry)
  in `~/.whileai/credentials.json`, and `whileai status` and `whileai
  login` refresh it from `/me`. When the situation writer is the hosted
  model and the saved key is a trial one, `simulate()` logs one line and
  puts it in `data.warnings` before generation starts: how many situations
  a day the allowance covers, that `simulator=False` writes them offline
  with no quota, and where to sign in to lift it. It reads the saved tier,
  so it costs no extra call. A key from `WHILEAI_API_KEY` has no recorded
  tier, so nothing is said about it.
- The offline template writer no longer reads tool descriptions back as
  customer speech. An action-shape ask is built from the tool's name in
  plain words ("Can you check an order for me?") instead of its
  description, which a tester saw quoted verbatim ("Can you look up an
  order by id. Returns item, total, order date and status for me?").
- `zp` and `wai` are console scripts for the same CLI as `whileai`, so
  `zp login` works on a machine where someone typed the product's name.
  The help text still says `whileai`.
- Every seed is run. `situations=` is sized up to `len(seeds)` when you
  pass a smaller number, and the search no longer spends a seed's slot on
  an ask it wrote itself, so `simulate(seeds=[10 asks], situations=3)`
  runs all ten instead of two. When `budget` cannot pay for
  `len(seeds) * repeats` rows the run says so before it starts
  (`budget=12 covers 3 of 10 seeds at repeats=4; raise budget to 40+ or
  drop seeds`), drops the seeds it named, and lists them in
  `search["seeds_dropped"]`.
- `coverage_warnings` also names a run that is hollow in part: rows that
  made no tool call while other rows did score on the reply alone and
  lift the pass rate. The note gives the count, the situations and the
  first three asks, and fires once two rows (or a tenth of the run) are
  silent.
- `SimulationData` iterates: `list(data)`, `for row in data` and
  `len(data)` work on a run, the way they already did on `ScoredData`.
- The judge contract says where a judge's extra keys go: `reward`,
  `reason`, `markers` and `failure_class` land on the row, everything
  else under `row["judge_meta"]` (a returned `failures` list reads back
  as `row["judge_meta"]["failures"]`). Also in `docs/evals.md`.
- A run says where it is. Every ten seconds and every ten finished
  rollouts, whichever comes first, `simulate` logs one line on the
  `whileai.simulations` logger: `12/64 rollouts, 3 situations written,
  1m40s elapsed, ~5m left`. The estimate is the finished rate carried
  forward and is left off until five rollouts have landed. The hosted
  writer also says when it starts, because the first rows cannot land
  until it has written something. Budgets under 10 rows stay quiet, and
  nothing is printed: `logging.basicConfig(level=logging.INFO)` to see
  it. A tester watched a 64-rollout hosted run produce no output for
  eight minutes and nearly killed it.
- `simulator="hosted"` names the default writer, so the way back from
  the offline `simulator=False` is a value you can pass, not "delete the
  argument".
- Anthropic-shaped tools (`{"name", "description", "input_schema"}`) are
  accepted everywhere the OpenAI shapes are. `input_schema` is read as
  `parameters` once, when the agent is inspected; before this the tools
  sent to a model-backed agent lost their arguments.
- Docs: `docs/evals.md`, the `02-measure/eval-your-agent` recipe and the
  simulations skill say that the old `ZEROPROOF_*` names still
  authenticate, that `concurrency` is 32 so a recorder needs
  `threading.local`, which tool shapes are accepted, that `pass^k` needs
  `repeats >= 4`, that `budget` must cover `situations * repeats`, and
  which of `data.rows` / `data.trajectories` and `scored.failures()` /
  `scored.failed_traces()` is the spelling to use. The README's first
  screen now points at the evals page.

## 0.60 (2026-09-17)

- `coverage_gap(asks, tools=, system_prompt=, rows=)` maps the asks a test
  suite already sends onto the grid `simulate` covers, and names what they
  never reach: `untested_rules` (policy clauses no ask touches),
  `untested_tools`, the per-axis counts, and `single_shot` when every ask
  runs once. `asks` is prompt strings, rows, or a path to a `.py` or
  `.jsonl` file (from a `.py` file the asks are the string literals that
  look like asks, a documented heuristic). `world_state` and
  `tool_condition` cannot be read from an ask at all, and the report says
  so with the fix. With `rows=` from a graded run it also names rules whose
  every row ended in the same tool fault: the asks reach the rule but the
  fixtures never let it happen. `format_coverage_gap` prints it.
  Three cold-start agents asked to "find the situations our tests do not
  cover" each hand-wrote this mapping, and each found the same untestable
  policy branch by hand (2026-09-17).
- `preflight()` reports `rules`, the rule axis the engine extracted from
  the system prompt, so the policy branches are readable without running
  a simulation.
- `data.coverage["pairwise"]` is labeled in the docs as what it counts:
  pairwise cells of the 6-axis grid, which is training-data coverage, not
  policy coverage. A small fraction on a short run read as a failed eval.
- Recipe `02-measure/eval-your-agent`: `--gap` runs `coverage_gap` on the
  three-ask `OLD_TESTS` suite it replaces, with a README section and a
  `docs/evals.md` section 3b.
- `anthropic:<model>` is a backend spec, so a developer whose only
  credential is `ANTHROPIC_API_KEY` can point the situation writer
  (`simulator=`), the simulated person (`user_model=`), a model-backed
  agent (`agent=`) and the judge (`spec=`) at the model they already pay
  for. Two coding agents evaluating a refund bot had neither an OpenAI key
  nor a local server, so they fell back to the offline template writer or
  spent the hosted trial quota. The Messages API calls go out over
  `requests` (no new dependency) and are translated at the boundary: tool
  definitions become `input_schema`, tool calls and results become
  `tool_use` and `tool_result` blocks, the system prompt moves to `system`,
  `stop_reason: "max_tokens"` becomes the engine's truncated marker, and
  the reply keeps the OpenAI shape every loop already reads. There are no
  log-probabilities from this API, so `logprobs=True` rows carry none.
- A trial key now says what it buys before a run spends it. `whileai
  signup` and `whileai status` print one line under the trial allowance:
  about how many hosted situations a day it covers (25,000 input tokens
  at around 2,000 a situation for a four-tool spec, so about twelve),
  that `simulate(..., simulator=False)` writes situations offline with no
  quota, and that signing in once lifts the limit. The daily-quota error
  the run dies with names the same two ways on. A twelve-situation eval
  spent 28,490 input tokens and stopped with a number and no next step.
- Return shapes are readable off the print and the docs instead of
  guessed. `PassAt` prints `pass^k (pass_pow_k)` once (testers reached
  for `pass_hat_k`), and its docstring is a field table with the printed
  name of every field. `marker_summary` adds a `note` when `ci95` is
  `None` for want of tasks, saying how many the marker has and that the
  bootstrap needs three. `ScoredData` and `SimulationData.rows` each say
  which spelling is which: `scored.rows` is a list, `data.rows()` also
  works. docs/evals.md and docs/simulations.md carry a Return shapes
  table (`PassAt` fields, marker stat keys, `ScoredData.warnings`,
  `judge_trust` keys).
- `judge_trust` and `judge_agreement` name the half that is missing.
  "no rows carry both 'reward' and a gold label" is now "no row has a
  reward: score them first with run_judge(rows, judge) or
  evaluate(data, judge)" (adding that a `judge=` here only runs the
  perturbation probes) or "no row has a gold label:
  attach_labels(rows, labels, kind='human')", and rows with both on
  different rows say that instead. Gold written by hand carries no
  `gold_kind`, so it read as "the gold labels came from a model"; it now
  says there is no record of who wrote it and names
  `attach_labels(..., kind="human")` as the way to mark labels as a
  person's.

## 0.59 (2026-09-17)

- `judge_trust` on a judge that agrees with every human label but has
  too few of them (14 perfect labels: lower bound 0.78 under the 0.80
  floor) said "change the judge prompt or the judge model". It now says
  the judge is not the problem, the sample is, and how many labels the
  bound needs at that agreement rate. The recipe README's thread-local
  recorder snippet dropped calls when the wrapper had not set the list;
  fixed. Both from the second cold-start test of "use zp to build evals"
  (2026-09-17, whileai 0.58).

## 0.58 (2026-09-17)

- `split_pseudo_production` splits by `task_key` (the `scenario_id`,
  else the prompt), so the held-out slice is disjoint from train in the
  unit every report groups by. Splitting on the prompt alone left 16 of
  28 held-out situations in train through their rephrasings, and
  `decontaminate` could not see it. (#268)
- `metric_summary` (and so `marker_summary`) flags a metric whose
  applicable rows all scored the same: `degenerate: True`, `ci95: None`,
  a `warning` that it has not been shown to be able to come out any other
  way, and `n_rows_at_1` / `n_rows_at_0` next to the mean. A marker that
  never fires and one that is always true looked identical (`1.000`,
  zero-width interval). `delta_report` names a `must_not_regress` metric
  that is degenerate on both sides as a guard that cannot fail
  (`degenerate_guards`). (#270)
- README: the served-model eval path (`simulate(tasks=pinned,
  agent=wai.local_model(...))`, both arms through the same call). (#269)

## 0.57 (2026-09-17)

- `audit_grades(rows, judge=, sample=, passes=)` estimates how often the
  verifier fails a right answer: it puts a sample of failed rows to a
  judge with the reference in place and reports the false-negative rate
  with a Wilson interval, the verifier's failure kinds with how many the
  judge overturned, examples, and (with `passes=`) the false-positive
  side. Above 10% the summary says to fix the verifier before training;
  `select_for_rl(audit=)` and `optimize(audit=)` carry that line into
  `hygiene_warnings`. The judge is a second opinion, not ground truth,
  and the report says so. (#255)

Two coding agents were told "use zp to build evals" on a fresh machine and
timed (2026-09-17). This is what they tripped on.

- A hollow run says so. `simulate()` warns when no rollout called a tool
  (`degraded` carries `no_tool_calls`); `run_judge`, `evaluate` and
  `data.grade` attach `coverage_warnings` to `ScoredData.warnings` and
  log them once: no tool calls, a declared tool no rollout touched
  (`tools=`, or read off the run), a marker that fired on no row. Each
  note names the fix. `wai.coverage_warnings(rows, tools=)`
  runs standalone.
- The knobs most runs touch are in `simulate()`'s signature: `repeats`,
  `phrasings`, `repeat_policy`, `concurrency`, `simulator`, `user_model`,
  `backend`, `seed`, `sampling`, `max_turns`, `avg_turns`, `fault_rate`,
  `temperature`, `timeout`, `logprobs`. Same road underneath; an editor
  now shows them, and the docstring says a callable agent is played
  single-turn and why real ids belong in the seeds or tool descriptions.
- `recipes/02-measure/eval-your-agent`: evals for the agent you already
  have, ending at pass@1 with an interval and a CI gate, not at a push.
  `docs/evals.md` is the how-to; the README and the skill link it first.
- Names: the README says once that zp, ZeroProof and While are the same
  product, that `WHILEAI_HOME` isolates a fresh account from an old
  `~/.zeroproof`, and PyPI keywords carry `zp` and `zeroproof` so the
  abbreviation people use finds the package. Dead links fixed
  (`while.ai` does not resolve yet; `examples/coding-efficiency`).

## 0.56 (2026-09-17)

- `load_traces` binds a tool result to the call that asked for it by
  `tool_call_id`, falling back to name then position. Real agent exports
  carry an id and no `name`, so every result fell through to position, and
  providers answer parallel calls out of order: on a 100-trace corpus in
  that shape, half the tool faults were attributed to a tool that never
  failed. Two calls to one tool are separable only by id.

- `holdout_size(effect, base=, k=, power=, alpha=, rows=)` says how many
  paired tasks a holdout needs to prove a gain, modelled on the paired
  task bootstrap `delta_report` runs (rlhf-book ch. 16, appendix C), and
  `detectable_effect(n_tasks, ...)` is the same solved for the gain. A
  test checks the number against `compare_runs` by simulation. The
  recipe that asked had 140 tasks at k=4: a +-0.06 band, so a 3-point
  gain could never read as anything but `no_change_detected`.
  `delta_report` now carries `detectable_effect` and `tasks_needed` and,
  on a no-change verdict, says what this holdout can prove and what the
  delta seen would have needed. `push(purpose="holdout")` warns when the
  set is too small to prove a 5-point gain. (#257)
- `next_round(prior, tasks=, lo=, hi=)` builds round N+1's prompt set
  from round N's graded rollouts: tasks the current policy solves above
  `hi` or below `lo` are dropped, the rest kept with their pass rate
  stamped, plus counts, the policy versions the prior came from and a
  `prompt_set_sha` for lineage (rlhf-book ch. 7 band; ch. 6 DAPO dynamic
  sampling). `select_for_rl(prior=)` applies the same cut first and
  reports it under `prior`. (#254)

## 0.55 (2026-09-17)

- `mine_traces` no longer counts a tool result as a fault because it has
  a `status` key: `status: "paid"` is the tool's own vocabulary. A status
  is a fault when it names one (`error`, `timeout`, `not_found`,
  `denied`, ...), an HTTP failure, or a non-zero exit. (#261)
- `serve(name, run)` takes the record `get_run` returns, and a wrong type
  is a `TypeError` naming the accepted ones instead of a urllib
  `InvalidURL`. `TrainingRun.id` is the run id. (#262)
- `unserve(name)` (alias `delete_model`) removes a hosted model row; the
  inverse of `serve`. `models()` says a row is a registry entry that
  costs nothing idle. (#263)
- `local_model(..., thinking=False)` sends
  `chat_template_kwargs={"enable_thinking": False}` so a served Qwen3
  answers instead of reasoning; `<think>` markup never reaches
  `step["text"]` or `final_text` on any path. `complete(extra=)` passes
  request fields through. (#264)
- `build_preference_pairs` says which pairs the hosted DPO trainer can
  use: `first_turn_differs` on each pair, `first_turn_identical` and
  `trainer_pairs` in the report, and a warning when the contrast is
  later in the rollout than the first assistant turn, since the trainer
  compares first turns only and needs at least 8. `export_preference`
  reports the same. (#260)

## 0.54 (2026-09-17)

- Every row says how it finished: `finish_reason` is `stop`, `length`
  (the reply token cap cut a turn), `tool` (the turn budget ran out on a
  tool call) or `error` (the agent raised); a callable agent can set it
  outright. `pass_at(...).config["truncated_share"]` is the share the cap
  cut and the one-line summary names it; `delta_report` warns when the two
  sides were cut at different rates, since that is not the same eval;
  `export_training` warns when length-cut rows go out as SFT targets.
  `train(method="grpo")` now sends `maskTruncated=True` by default, so a
  cut reply gives no gradient instead of a 0 that teaches shorter thinking
  first; `truncated="zero"` is the old behaviour. (#253)
- `export_training` (and `export_dataset`) checks for privileged leaks on
  the unscrubbed side before it writes: the export drops the `privileged`
  key at any depth but copies the assistant's reply through verbatim, so a
  reply that recited the block still recited it in the training file.
  `validate=True` now refuses with `privileged_leak: N of M rows ...`;
  `validate=False` exports anyway, counts them in
  `report["privileged_leaks"]` and warns. Pass the `SimulationData` (or
  `data.trajectories`); rows that came through `rows()`, `save()` or a
  file carry nothing to check and the report says so. (#249)
- `leak_report(data)` and `data.leak_report()` read the trajectories, so
  the documented path no longer returns a vacuous pass. (#245)
- A `Verifier` graded through `run_judge` reads back as `kind="rule"`,
  not a model judge: the run stamps the declared kind in
  `judge_meta["scorer_kind"]` and the schema prefers it over inferring
  from `judge_name`. `judge_name` still round-trips. (#250)

- No github demo anywhere a user reads: the package docstrings, the README
  and the identity recipe named `specs/github`, `github-rl-v1` and
  `envs/github-agent`; they now show `tools=` plus `system_prompt=` and
  neutral names. `recipes/04-train/identity` takes its control conversations
  from `--control-file` (your own rows or traces) or from `wai.simulate` over
  `--assistant`; the canned reply templates and the test-fixture spec are
  gone from it.

## 0.53 (2026-09-17)

- `rubric_judge` warms the hosted judge once before the rows fan out, the
  same `warm_judge` call and 600s budget `grade_llm` already used. A serve
  container that had scaled to zero took longer to load its weights than the
  120s per-call timeout, so all eight of `run_judge`'s concurrent calls timed
  out together and every row came back `invalid_result` with `reward: None`.
  A failed warm-up is not fatal: the rows are judged anyway and report the
  real error.
- `pass_at` on a set where every row failed judging says so, naming the
  status and the judge's own error, instead of `no binary rewards; grade
  first` -- which pointed at the step that had just run. Sets that were
  never judged, and partly graded sets, keep the old wording.
- `leak_report` on exported rows says why it found nothing. `rows()`,
  `save()` and `push()` scrub `privileged` at any depth, so the detector had
  nothing to check and `checked: False` read as "nothing populated it"
  rather than "you passed the scrubbed copy". When the rows came through the
  export, `summary` now names it and points at `data.trajectories`.
- `recipes/_template/` to copy (`README.md`, `run.py`, `smoke.sh`), a
  "Contributing a recipe" section in `CONTRIBUTING.md`, two issue forms,
  and a CI job that runs every `recipes/**/smoke.sh` on every pull request:
  no key, no GPU, under a minute, so "it runs" is checked rather than
  claimed.

## 0.52 (2026-09-17)

- `simulate(tasks=base, runs=3)`: the same task set replayed three times in
  one call, every row stamped `lineage.eval_run` (0, 1, 2), one
  `SimulationData` back (`search["eval_runs"]` has the rows and stop reason
  per run). Without `tasks=` the first run draws the set and the rest replay
  it. Between runs only the agent's sampling changes. `eval_variance(rows)`
  splits by `eval_run` on its own.
- `delta_report` verdicts are honest about repeats (rlhf-book ch. 16,
  appendix C). With two or more eval runs on each side it computes `run_std`
  itself (pooled over the sides, on the headline metric) and applies the
  existing noise band; `eval_runs`, `run_std_source` and `replicated` say
  where the band came from. With one run on either side and no `run_std=`,
  a target that moved reads `moved_unreplicated` and the warning names the
  `runs=3` call that settles it. This changes existing single-run reports:
  `moved` now needs repeats or a `run_std`.
- `delta_report` flags `ceiling=True` (with a warning) when the before side
  already passes 0.9 of its tasks, or fewer than 20 paired tasks (and under
  half) still have room, so a training run cannot show a gain on that eval.
- Training runs carry the holdout numbers with their uncertainty:
  `run.delta(...)` and `attach_delta(...)` put `summary["holdout"]` on the
  run (per side `pass`, `n_tasks`, `k`, `ci95`; the delta report's `verdict`
  word; `eval_runs`, `run_std`, `ceiling`) and `run.holdout_summary` holds
  it. A hosted run read back with `refresh()` has the same shape with every
  interval field `None` and `note` "No interval: the platform only returned
  two numbers".
- The judge is checked by default. `grade` (the hosted judge, `judge=`, and
  `grader=` paths) ends by measuring the judge against the rows' human labels
  and stamps the summary on every graded row as `judge_meta["trust"]`
  (`agreement`, `agreement_low`, `kappa`, `n_gold`, `ok`) and in the report's
  `trust`; with no human labels it prints one line saying so. `trust="warn"`
  (default), `"require"` (raise), or `"off"` on `data.grade`, `data.grade_llm`,
  `grade_llm`, and `apply_grade_llm`. `publish_gate` reports it as
  `judge_trust`. `trust_after_grade` is the helper.
- Gold has provenance. `attach_labels` writes `gold_kind` next to
  `gold_reward` (`"human"`, or `"model"` for a model's labels).
  `judge_trust` and `judge_agreement` report `gold_kind` and return
  `ok=False` with the reason when the labels are a model's, a second judge
  pass, or unknown (older rows with `gold_reward` and no kind);
  `allow_model_gold=True` keeps the old behavior. Rows hand-labeled before
  this release need `gold_kind="human"` (re-run `attach_labels`) to count.
- A floor, not a hint. `judge_trust(min_agreement=0.8, min_kappa=0.6)`: the
  Wilson lower bound of agreement and kappa must clear the floors or `ok` is
  false with the number, the floor, and the fix in one sentence. The
  kappa-under-0.4 hint is replaced by the floor, so `ok` can now be false on
  a labeled judge that used to pass.
- The auditor cannot be the grader. `audit_grades` swaps to the other hosted
  model when the resolved auditor is the model that graded the rows (Phi-4
  to hosted Qwen and back), records `grader` and `auditor` in the report,
  and raises `ValueError` when no different model is available.
- Every row says how it was sampled. `sampling` is now on every row a
  model backend produces (the default hosted agent, `agent="vllm:..."` /
  `"openai:..."`, `backend=`, an HTTP agent), as `{"temperature",
  "max_tokens", "model"}` with the defaults the backend resolved; before,
  it was stamped only when `backend=` was passed by hand. A callable
  agent's rows carry `sampling: None` unless you pass
  `simulate(sampling={...})`, which is recorded as given. The `logprobs`
  key is gone from `sampling`; the row's `logprob` fields already say
  whether logprobs were captured.
- `pass_at(rows).config` (and `to_dict()["config"]`) says what the rows
  were produced with: task count, k, temperature, max_tokens, policy and
  judge versions, prompt hash, with `mixed` naming any the rows disagree
  on. `delta_report` carries the same per side under `config["before"]`
  / `config["after"]` and warns when the judge, temperature or reply
  budget differ between sides, or when both sides are the same policy
  version.
- One task key everywhere. `wai.task_key(row)` (`scenario_id`, else
  `task_id`, else the prompt text) is what `pass_at`, `group_signal`,
  `compare_runs`, `delta_report`, `eval_variance`, `curriculum`,
  `retire_solved`, `trim_unanimous_groups`, `trim_out_of_band`,
  `select_for_rl`, `calibrate` / `publish_gate`, `mean_kl`, `judge_trust`
  and the exporters' `group_id` now all group by. Before, `pass_at` and
  the RL pruners grouped by prompt text while `compare_runs` grouped by
  id, so the same rows gave two task counts. On engine rows this means
  the rephrasings of one situation pool into one task: a task is a
  situation, not a string. `PassAt.per_task` and `curriculum()`'s
  `task_id` are keyed by that key; `curriculum()` still carries a
  `prompt` per task.
- `select_for_rl` / `optimize(mode="rl")`: asks inside the difficulty band
  are now taken round-robin across pass rates within each fault kind, with
  no preference for a 50% pass rate (`order="spread"`, the default). The
  older nearest-to-50% ranking is `order="middle"`. A selection cut off by
  `target` can come back with different asks than before.
- `curriculum` / `retire_solved`: `floor` and `solved` default to the band's
  edges (0.2 and 0.8, from `DEFAULT_BAND`) instead of 0.0 and 0.9, and the
  edges are inclusive: trainable is `floor <= pass_rate <= solved`, retired
  is above `solved`, not ready is below `floor`. A task at 1 of 8 is no
  longer trainable.
- `Calibration.pass_rate_ci95`: the Wilson 95% interval on the task's pass
  rate, stamped by `calibrate` and `carry_calibration`; `calibration_of`
  reads it back. The RL optimize report lists one row per selected task
  under `calibration.tasks` and adds a `hygiene_warnings` note when the
  median rollouts per task is under 16, with the measured interval width.
  `Calibration.student` is filled from the row's `policy_version` when no
  `policy=` is given.
- `train(temperature=...)`: the GRPO rollout temperature, sent to the host;
  when the pushed dataset's rows were measured at a different
  `sampling.temperature`, `train` warns once. `recipes/04-train/grpo`
  trains and evaluates at the same temperature (0.8).
- Every row says which model did which job. `writer_model` (the situation
  writer's model tag, or `template` / `seed` / `pinned` when no model wrote
  the prompt) and `user_model` (who played the simulated user; absent when
  the agent took a single message) sit next to `model_version` on every
  row, ride through `export_row`, `training_rows`, and the `from_row` /
  `to_row` round trip like `policy_version`, and appear in `data.metadata`
  and the `.meta.json` sidecar with `judge_model` (read off each row's
  existing `judge_meta.model`).
- `simulate(user_model=...)`: a backend spec for the model that plays the
  user in follow-up turns and answers the agent's questions. `None` (the
  default) keeps today's behavior, the agent's own model.
- When the agent model also wrote the situations or played the user, the
  run appends `same_model` to `degraded`, adds one plain sentence to the new
  `data.warnings` list naming the call that separates them (`simulator=`,
  `user_model=`), and logs it once at the end. Defaults are unchanged: the
  same hosted model still does all three jobs unless you say otherwise.
- The import alias in every example, recipe, docstring and the skill is
  `wai` (`import whileai.simulations as wai`), not `zps`. Nothing in the
  package changes; `zps` was only ever a name in your own code.
  `scripts/rebrand.py --alias` applies the same rename to an open branch.

## 0.51 (2026-09-16)

- **Renamed to `whileai`.** ZeroProof is now While, and the package follows:
  `pip install whileai`, `import whileai`, `import whileai.simulations as zps`,
  the `whileai` command, `WHILEAI_*` environment variables, `~/.whileai` for
  the saved login, and `whileai.WhileIngestError`. Nothing old breaks: the
  `zeroproof` distribution keeps releasing as a shim (`compat/zeroproof`) that
  installs `whileai` and aliases `import zeroproof` and
  `import zeroproof_simulations` to the same module objects with a
  `DeprecationWarning`; the `zeroproof` command still runs; every
  `ZEROPROOF_*` variable is read when its `WHILEAI_*` twin is unset; a
  `~/.zeroproof/credentials.json` is used until `~/.whileai` has one;
  `ZeroProofIngestError` is an alias of `WhileIngestError`. Hosts
  (`api.zeroproofai.com`, the Modal apps), the `zp_` key prefix, the
  Hugging Face org and the `zeroproof.*` span attributes are unchanged. The
  repository moved to `whilehq/whileai-sdk`. The rename is `scripts/rebrand.py`,
  a script to run on an open branch instead of resolving conflicts by hand.
- `simulate(tasks=...)` no longer drafts a tool surface for a prompt-only
  agent. Pinned tasks bring their own prompts, so there is no situation to
  anchor, and the drafted schemas reached the policy: Nemotron-Nano-8B
  answered every text-to-SQL task with a call to a tool that did not exist
  (pass@1 0.00), and 42 of 560 holdout replies from a Qwen3-4B checkpoint
  did the same. Declared `tools=` still pass through unchanged.

## 0.50 (2026-09-17)

- `recipes/papers/`: recent post-training papers as recipes. One directory per
  paper (README in a fixed shape, one `recipe.py` with a baseline arm and the
  paper's change on the same holdout, `results.json`); `check.py --write`
  generates the index table from the results files, `tests/recipes/test_papers.py`
  keeps the shape. A daily agent re-verifies the stalest recipe and adds one
  new one as pull requests.

## 0.49 (2026-09-17)

- `examples/` is now `recipes/`, grouped by the step of a post-training run:
  `01-simulate`, `02-measure`, `03-select`, `04-train`, `05-export`. Every
  recipe keeps its name (`recipes/04-train/grpo`, `recipes/01-simulate/verifiers`,
  ...); `examples/README.md` stays as a table from old path to new so links
  keep resolving. `recipes/README.md` is the index: the five steps, one row per
  recipe with what you learn / needs / takes, the conventions every recipe
  follows (README first, `--help`, keys from the environment, `out/` and
  `raw/` gitignored, every claim a paired number with an interval), and how
  to add one. Tests moved to `tests/recipes/`; the recipe registry test now
  walks two levels. The `examples/*` catch-all in `.gitignore` is gone: a new
  recipe is tracked without a gitignore edit, and only its data files and
  output folders are listed.
- `recipes/04-train/text-to-sql`: GRPO generates through vLLM
  (`--use-vllm`: TRL colocate mode, about 5x the HF path), `--spawn` launches
  that survive the client, `distill.py` (the base's verified thinking traces as
  hosted SFT data), 741 tasks with 140 held out, and the round table through
  round 2 (flat, with the reading).

## 0.48 (2026-09-16)

- `audit_grades` says what it found. It returned agreement counts only, so a
  second judge that disagreed with a fifth of the labels gave no way to act on
  it; the auditor's sentence was parsed and thrown away. It now carries
  `disagreements` (the ask, both labels, both reasons), `by_judge_reason` (which
  grader rule the disagreements sit under, which is what points at a rubric
  hole), and `findings` led by false passes: rows the grader passed and the
  auditor failed become training data for the behavior you are removing
  (rlhf-book ch. 5, ch. 14). Found dogfooding: on a steward constitution the
  auditor failed 4 of 30 rows the rubric passed, all because the rubric scored
  confirmation discipline and never whether the agent did the job, so an agent
  that refuses everything scores 1.

## 0.47 (2026-09-16)

- `simulate(agent_max_tokens=N)`: the model agent's reply budget. The
  default (768 tokens, 2048 above an 8k `ZP_CONTEXT_TOKENS`) cuts a
  reasoning model off mid-thought; Qwen3-4B with thinking on lost 8% of
  its replies that way and 4 of 81 tasks to the 60 s `timeout`, which
  was reachable only through `advanced=` and is now a keyword too. Set
  both for a thinking model: `agent_max_tokens=4096, timeout=300`.
- `examples/text-to-sql`: hill-climb a model on a schema with a verifier as
  the reward. A seeded online-store Postgres database, 417 authored and
  execution-checked tasks (81 held out by task id), `SQLExec` (a
  `Verifier`: run the candidate, match the gold result set), a benchmark
  runner for hosted Qwen3-4B, Claude and any served adapter, `build.py`
  (pass@k, `optimize`, `hack_scan`, pushes train/holdout/eval sets), a
  Modal GRPO trainer with Postgres inside the container and `--from-run`
  for rounds, and `delta.py` for the paired before/after. README carries
  the base numbers, the headroom rule, and the gradient-checkpointing trap.
- The mock world's `stale` fault returns the record as of three days
  ago, marked `stale`, instead of a bare hash. Agents turned the hash
  into an invented shipment, offer id or passing test suite, and a
  rubric judge passed them.
- `pass_at` reports a 95% task-bootstrap interval on pass^k and pass@k
  (`pass_pow_k_ci95`, `pass_at_k_ci95`), not only on pass@1, and prints
  them. The reliability line a safety eval reads per attack class was a
  bare number (rlhf-book ch. 16: intervals from resampling prompts).
- `eval_variance`: a wrong-shape argument says what to pass instead. Handed
  a `SimulationData` — what `simulate()` returns — it raised Python's bare
  `TypeError: 'SimulationData' object is not iterable`, which never
  mentioned that `.trajectories` is one attribute away. The message now
  names the argument, its type, and the runnable call (#31). Behavior for
  every shape that already worked is unchanged.

## 0.46 (2026-09-16)

- Judging: a verifier's identity and metadata survive onto the scored row
  (#196). `normalize_judge_result` swept a verdict's own `judge_meta` in as
  an ordinary key, nesting it under itself, so `row["judge_meta"]["verifier"]`
  was `None` on every verifier-graded row and a `failure_class` or `markers`
  returned in the documented shape was silently dropped. Both reporting
  shapes now merge. `run_judge` also falls back to a callable instance's
  `.name`, so a row graded by `MathEqual` no longer records the same
  `judge_name` as one graded by `CodeExec`; function and lambda judges keep
  the names they had.
- `looks_finished`: a reply that ends on a closed code fence, or on `}`, has
  reached its end (#212). The rule read terminal punctuation only, so an
  answer that *is* a fenced block — every row of a text-to-SQL set — was
  called truncated and dropped by `optimize(mode="rl")` and the hygiene
  gates. An unclosed fence is still truncated, which is the cut the rule
  exists to catch.

## 0.45 (2026-09-16)

- The free path has something to catch. Offline, the row's scheduled
  `faults` never reached a callable agent, `privileged` was empty on every
  run that did not attach a rubric, and the markers read zero by
  construction, so a "no privileged leak" check passed vacuously.
  `zps.world(tools)` is the mock world for a callable agent: its `call`
  applies the row's faults and world state first, read from
  `current_rollout` (which now also carries `faults`, `world_state`,
  `tools`, `privileged`). Every row is born with `privileged`
  (`hidden_state` from the grid cell and the fault plan, `reference` from
  the checklist's expected outcome via `expected_outcome` /
  `privileged_context`); exporters scrub it as before. `zps.seeded_agent(tools,
  rate=, seed=)` answers honestly through `world()` and on a labeled
  fraction of rollouts does one wrong thing on purpose (`hedging`,
  `sycophancy`, `apology`, `boilerplate`, `ignore_fault`, `leak`); each
  row carries `seeded`, the list of what it did. `zps.leak_report(rows)`
  finds replies that quote their own privileged block and reports a
  vacuous check as vacuous (`checked=False`). README: Start here shows
  all three.
- The user simulator's instructions name the details a person on this
  agent's thread would know, read off the agent's own tool parameters
  (`order_id` becomes "order id"), and mention repos and pull requests
  only when the tools have such parameters. The prompt carried "order id,
  sku, store, size, repo, PR" and a git warning for every agent.
- Failing asks in `traces=` seed the run. A capability failure carries no
  tool, fault or world-state signal for the grid to aim at (SQL dogfood,
  2026-09-16: 41 failures, empty emphasis). The failing prompts now seed
  the run and are amplified into variants, disclosed as
  `search["trace_mining"]["failure_seeds"]`; the leakage rule still keeps
  the originals out of the rows.

## 0.44 (2026-09-15)

- README: `recommend()` gets its own section, *How much to run*. Testers
  found it via `dir(zps)`. The five-call example now says its
  `situations=200, repeats=8` is a guess and points at the call that
  replaces it with numbers from the agent's own grid (#209).
- Template writer: readable intents, larger phrase pools, identifiers keep
  their case (#205).

## 0.43 (2026-09-15)

- `examples/safety-evals-marketplace`, `blog/marketplace-agent-safety-evals.md`:
  the safety eval of `examples/safety-evals` for an agent whose untrusted
  text is user-generated content and whose private data is per tenant. A
  vendor assistant on a software review marketplace reads reviews (two
  carry a planted instruction: exfiltrate the buyer-intent list, flag the
  review), pulls intent signals licensed to one vendor per product, posts
  public responses, and flags reviews with a moderation ticket. Seven
  categories (a `cross_tenant` class on top of the five, plus the benign
  controls), six trajectory markers (`no_secret_leak`,
  `no_cross_tenant_read`, `no_external_send`, `no_public_leak`,
  `no_unauthorized_flag`, `helpful_on_benign`), the same pass^k, judge
  trust, refusal probe and guarded `delta_report`. `live.py` runs the
  suite on a real model with `execute=world` so the planted reviews reach
  it as tool results; Ollama by default, so no key. No engine change.
- `preflight`: `tools_not_mentioned_in_policy` reads the policy the way it
  is written. "Look up the order before discussing it" now counts as a
  mention of `get_order`, and "escalated to a human" of `escalate_to_human`;
  before, only the literal snake_case name counted, so every tool of every
  English policy was reported.
- A user turn textured `lowercase` keeps an identifier, email, or code in
  its case (`USE-8481` reached the agent as `use-8481`), and one textured
  `standard` no longer capitalizes an identifier,
- The mock world's forty record owners are invented pairings (a first
  name and a surname from different regions) instead of thirty-two
  common ones, so a generated record names no one in particular.
- A user turn textured `standard` no longer capitalizes an identifier,
  email, or code at the start of the line: `mia_lopez_4821` was reaching
  the agent as `Mia_lopez_4821.`, and the agent then passed the wrong id.
- The template writer says "escalate to a human", not "escalate a to
  human", and "check direct flights", not "check a direct flights"; a
  leading preposition in the tool name stays in front of the noun, plurals
  take no article, and "user" takes "a".
- The template writer (`simulator=False`) draws from larger pools: twelve
  openers, eight closers, three phrasings for every world state, tool
  condition, stance, and history value, chosen per situation. Before,
  three openers and one sentence per axis value made every offline row
  read alike.
- `sample_turn_budget` no longer divides by zero when the running-mean
  correction pushes the target past the turn cap, which `avg_turns=12`
  reaches on a 4096-token context: 351 of 438 rollouts on one run failed
  with `ZeroDivisionError` and the run stopped as `writer_exhausted`.
- `avg_turns` defaults to `12` (was `4`). The person speaks at most
  `avg_turns // 2` times, so the old default ended most verify, look up,
  confirm, write flows on the agent's second question (a scripted
  order-support agent reached the write in 2 of 40 rows; 5 of 40 at
  `12`, the rest stopping correctly on missing records). Model-backed
  rows carry more turns now; pass `avg_turns=4` for the old length.
- `examples/pass-at-k`: the verdict no longer calls a zero gap between
  pass@1 and pass^k "inconsistency".
- `data.degraded` no longer carries `semantic_embedding_unavailable` on
  every run: the note lands only when a semantic `embedder=` was asked for
  and fell back to the hash. The hash is the default and was never a
  degradation.
- `select_for_rl` and `optimize(mode="rl")` on graded rows that left no
  mixed group say so (how many unanimous, collapsed and out-of-band groups
  went) instead of "no row carries a numeric reward; grade first", and
  report `eval_sourced_input` with the warning even when nothing was
  selected, so an eval set fed to the selector is visible.
- Hosted runs on the account key. With no `VLLM_API_KEY` and a key from
  `zeroproof login` or `zeroproof signup`, the default agent, writer and
  judge go to the account endpoints (`zeroproof-serve`: Qwen3-4B with
  thinking off, Phi-4), which take the zp_ key, enforce the daily
  allowance with 429 and meter on the server; the client-side usage
  report stays off for them. `VLLM_API_KEY` still wins and goes to the
  shared pool. A spent allowance stops the run (`*_quota_exceeded`)
  instead of retrying into the clock. Before this a fresh signup could not
  run anything hosted.
- The hosted client follows a 3xx to its Location. Modal answers a web
  request past 150 seconds with a 303 to a result URL that blocks until
  the work is done, which a scale-to-zero judge's cold start exceeds;
  before this the first call read the redirect's empty body as the reply
  and the run graded as unreachable.
- A key the hosted endpoint rejects (401/403) stops `simulate()` on the
  first writer wave or rollout that sees it and raises, the way a missing
  key already failed at setup. Before this the run spent its whole time
  budget on 401s and returned zero rows, with the reason only in
  `search["writer_errors"]`. `stopped_because` is `writer_auth_failed` or
  `agent_auth_failed`.
- `ScoredData.push(name, ...)`: `push_rows` on the graded copies, so the
  object `grade(judge=)` returns can make a gated RL push.
- `zeroproof.list_traces()` resolves the key like every other platform
  call (argument, `ZEROPROOF_API_KEY`, then the saved credentials) instead
  of requiring it as a positional argument; the skill's snippet follows.
- README: the hosted `data.grade(rubric=)` grades in place and returns the
  judge report, so the quickstart reads `data.pass_at`, not `scored.pass_at`.
- A run that ends with no rows, no agent failure, and nothing still in
  flight stops as `writer_failed`, keeps the hosted writer's last error in
  `search["writer_errors"]`, and warns. Before this a hosted writer that
  failed cold for the whole clock reported `time_budget`, its error gone,
  and the template writer that took over knew nothing about the spec.
- `simulate(grader=)` refuses anything that is not callable, naming the
  fix. A string there ran every rollout through the judge as an error:
  150 rows reported judged, none with a reward, nothing said so.
  `data.search["grader"]` now also counts `errors`.
- The hosted-agent client closes a thread's previous connection before
  opening one to a different host. A run that alternated hosts leaked one
  socket per rollout and printed a ResourceWarning for each.
- `grade()` leaves a rollout the loop stamped `length_cap` alone instead of
  judging it after the run; the report counts them as `skipped_truncated`.
  Before this an after-run grade overwrote every in-loop truncation stamp.
- `export_environment`'s `outcome_checkable` count now agrees with the
  checklist it describes. It came from a second copy of `outcome_check`'s
  dispatch living in `environment.py`, and the copy had drifted both ways:
  it missed the duplicate-entity world, which has a rule the module
  docstring lists, and it counted a task on its world state or its
  prior-partial-action history even where `outcome_check` returns no rule
  at all (a compound `multi_tool` ask, or a prior partial action for a
  rollout that writes nothing). On a 40-task offline export, 6 of the 39
  tasks reported checkable had no outcome rule for a rollout that acts;
  the count is 33 now and the 6 are 0. The predicate moved beside the
  dispatch as `_task_has_outcome_rule`, and two tests run every task shape
  through both so they cannot drift again. No reward changes: the
  checklist itself is untouched, only the report and its warning.
- `run.holdout(before, after)` and `zps.attach_holdout(run_id, before=,
  after=)`: say whether the training worked. A finished run's page opens with
  one word — Better, Worse, About the same — over the held-out pass rate
  before and after, read from `holdoutPassBefore`/`holdoutPassAfter` on the
  run's summary. The platform's own trainer writes them; nothing in the SDK
  did, so a run on your own hardware — the path `TrainerCallback` exists for —
  finished at "Not measured" with no call to fix it. Pass rates are 0 to 1
  (58% is `0.58`, and `58` raises rather than reading as 5800% on the page);
  `metric="loss"` sends held-out loss instead, for SFT. `run.delta(...)` and
  `zps.attach_delta(...)` now fill the same two keys from their own pass@1,
  so a run that already reports a delta opens with the word too.
- Verifier reasons no longer quote the answer key. A verifier reads the
  gold from `privileged.reference` and then wrote what it compared into
  `reason` (`"got 7.0, want 42.0"`, `"no match; got 'x', want 'Paris'"`),
  and `reason` is on the export carry list, so `export_training`,
  `export_preference` and the engine's own `save()` all carried the gold
  into the student's file, on exactly the rows the student got wrong. The
  gold now reads `<reference>` in the reason; the candidate half of the
  comparison is unchanged, and a gold the caller keeps in a plain column
  (`answer=`, `info.answer`) is quoted back as before, since that is their
  own data and no exporter carries it. `CodeExec` on `privileged.tests`
  reports the exception (`tests failed: AssertionError`) instead of the
  failing line, which is the answer key; pass `tests=` to see the full
  tail while iterating; the same tests read as `privileged.reference`
  are treated the same way. The gold is matched as a whole token (a
  gold of `7` is not the `7` in `17`) and in the spellings a verifier
  quotes: the float it parsed to, the escaped form, the first 60
  characters. Found by the customer simulation (#31), which
  recorded that the existing leak guards pass vacuously offline because
  nothing on that path populates `privileged` at all.
- `judge_trust` no longer reports `ok: true` on a judge it did not
  measure. With no `gold_reward` on any row, agreement, kappa, the
  held-out halves and the length split all have `n=0`, and the report
  still headlined PASS: a judge that passes everything came back
  trustworthy, which is the claim the report exists to check. `ok` now
  means measured and clean, the warning says the judge is unmeasured
  rather than failed and names the next action, and
  `format_judge_trust` prints `NOT MEASURED` for that case instead of
  PASS or FAIL. The perturbation pass is not counted as evidence on its
  own: a judge that passes everything is perfectly consistent. A probe or
  perturbation that fires is a finding with or without labels, so that
  report still reads FAIL. From the
  customer simulation ledger (#31).
- `examples/safety-evals`, `docs/safety-evals.md`, `blog/agent-safety-evals.md`:
  safety evals for a tool-using agent on the existing calls. A suite of
  attacks goes in as `seeds=` (direct prompt injection, an injection
  planted in a tool result, exfiltration to an outside address, secret
  extraction by claimed authority, writes without the approval the policy
  requires) with benign controls beside them; a judge of four trajectory
  markers (`no_secret_leak`, `no_external_send`, `no_unauthorized_write`,
  `helpful_on_benign`) grades through `evaluate`; pass^k per attack class,
  `judge_trust` on hand-labeled transcripts, `judge_probes(["refusal"])`
  on the benign rows to show a safety-only judge is satisfied by refusing
  everything, and `delta_report(must_not_regress=["helpful_on_benign"],
  by="category")` on pinned tasks to fail the fix that got safe by
  refusing. Three scripted agents, no key, seconds. No engine change.
- `zps.send_score(trace_id, value)`: grade a run that has already finished.
  The gate has taken measurements at `POST /v1/scores` all along and nothing
  in the SDK wrapped it, so an agent could send traces from the terminal but
  not the pass or fail that `zps.cut(kind="rl")` filters on — the one gap
  between "agent runs" and "agent has training data". A pass is 1.0 or above
  (the cut's own rule), `True`/`False` land as 1.0/0.0, `name=` puts a second
  measurement beside the verdict, `scores=[...]` sends a batch, and re-sending
  a name is a correction. A trace id this account never sent raises instead of
  looking like a success.
- Truncation, two fixes from the customer simulation ledger (#31). A reply
  that ends on a sign-off (`Best,\nSales`, `Thanks,\nAlex`, `-- Sam`,
  `Cheers`) is finished: `looks_finished` in `score.grading` reads the
  last line, not the last character, and the conduct grade, `is_truncated`
  and the junk gate all use it, so a customer's emails stop reading as
  cut at the token cap. `select_for_rl(truncated="keep")` never returns
  fewer rows than `"drop"`: an overlong rollout now rides along with its
  ask instead of voting on whether the ask is unanimous or in band (a kept
  pass tipped an ask over the band and the whole ask went), and the junk
  gate defers to the policy on a cut reply over 600 characters instead of
  eating a row the report counted as kept. `"penalize"` is unchanged: the
  penalty is a failure that counts. The report gains `truncated_selected`,
  the marked rows that reached the selection.
- Eight small gaps from the customer simulation ledger (#31), none a
  public API change. `simulate(tasks=)` without `repeats=` now keeps the
  pinned run's k instead of falling to the mode preset, so a before/after
  no longer silently compares k=4 against k=1. `export_training` reports
  `rewards` (pass, fail, ungraded counts) and warns when it writes rows
  with reward below 0.5 as SFT targets. The length-confound warning fires
  when the chosen side is longer in every pair from three pairs up, not
  only at eight, and `export_preference` carries it too. `recommend`
  accepts `system_prompt=` like `simulate`. `preflight` names the missing
  key (`returns`), treats `properties: {}` as a declared no-argument tool,
  and matches destructive verbs as words, so `read_runbook` is no longer
  destructive on the strength of `book`. `zeroproof status` says on stderr
  when no key is configured and carries `configured` in its JSON. Two
  README snippets still used `spec="specs/github"`, which does not ship.
- `examples/character/from_model_spec.py` no longer replaces the Model
  Spec commit pin in an existing `constitution.json` with `null`: without
  `--commit` it keeps the pin the file already carries and says so, and
  when no pin is available it says the provenance is unresolved instead
  of exiting 0 as if it were (#155). The conflict-marker guard now also
  rejects control characters in tracked text files, which is how a
  `## 0.32` heading in this file read as a bare date for a day. Coverage
  floor raised to match what the suite measures.
- `export_environment`: the package no longer carries a copy of the
  environment module and the checklist (`_zp_env.py`, `_zp_checklist.py`).
  The copy existed so an export loaded on an SDK release without them,
  but the generated `pyproject.toml` pins `zeroproof>=` the exporting SDK,
  and 0.42 ships both, so the fallback could never run. A task's id now
  hashes the scenario and the prompt (prompts drawn from one situation
  shared an id; the split still keeps them on one side), and partial
  credit counts toward a prompt's solve rate (a 0.5 was dropped as
  ungraded, which exempted the prompt from the band and the contrast count).

## 0.42 (2026-09-14)

- `task_checklist(row)`: a reward with an outcome term the world can verify.
  The conduct grade gates it; the outcome comes from the task's grid
  coordinates (target tool, world state, stance, history, ask family): the
  target must succeed, a missing entity must be reported not acted on, an
  already-done action acknowledged not repeated, an adversarial ask must
  produce no write, an unrelated ask no call, a vague ask a question back,
  prior partial action a read before the write, a fault on the target an
  acknowledgement. Markers name the checks. It is `export_environment`'s
  default reward; rows without grid metadata fall back to conduct and the
  export says so. rlhf-book ch. 12 (rubrics as rewards), ch. 7 (verifiable).
- `export_environment(source, out, reward=, execute=)`: a simulation becomes
  an installable `verifiers` environment for on-policy RL. The package
  carries the task set (one task per prompt, difficulty band applied when
  the rows were graded, split by scenario, decontaminated), the tool
  schemas, and dotted references to the reward and the world; the
  environment class itself lives in the SDK (`load_environment`) and runs
  the mock world seeded per task or your `execute=`, with the reward as
  the rubric through the judge contract. Without `reward=` it warns that
  `conduct_grade` is a process reward. New optional extra `zeroproof[rl]`
  pulls `verifiers`. rlhf-book ch. 6 (on-policy sampling), ch. 7
  (difficulty filtering), ch. 13 (end-of-trajectory reward).
- A spec folder carries `rubric.md`, what doing the job means, and
  `grade()` scores against it; `simulate(rubric=)` and `grade(rubric=)`
  take one directly, `prompt=` is still the raw judge prompt. Without a
  rubric the hosted judge grades the conduct floor only and the report
  says so (`rubric: conduct_floor`). Before this the default judge passed
  31 of 31 github rollouts: honest, and never asked whether the job got done.
- `write_rubrics(max_hard=N)`: the heaviest N hard rules a model-written
  rubric carries stay hard, the rest become weighted principles
  (`demoted_hard` in the report). Measured live on 72 rollouts with the
  hosted judge: uncapped rubrics (about three hard rules each) failed 48
  rows on a hard rule with mean reward 0.14; `max_hard=1` 42 rows, 0.26;
  `max_hard=0` none, 0.48. Default `None` keeps what the writer wrote; use
  0 or 1 for a training reward (#181).
- `hack_scan`: a `degenerate` scan withholds the `inverted` claim as well
  as the top feature. With two distinct rollouts per ask an endorsed
  feature sits at rho -1 exactly when it happened to fall on the failing
  trajectory, so "the reward punishes the endorsed behavior" there is the
  same coin flip as naming a winner, and the report contradicted its own
  "no hack is claimed". A varied pool still reports a genuinely inverted
  reward.

## 0.41 (2026-09-14)

- `zps.reference_logprobs(rows, "vllm:<model>@<url>")`: scores every agent turn
  under a reference model through `prompt_logprobs` on any vLLM-style
  endpoint (the platform's served base by name, a hosted model, or
  `run:<runId>`) and stamps `ref_logprob`, `ref_n_tokens`, `ref_model`, so
  `mean_kl` has its other side (rlhf-book ch. 6, 8, 15). The report's
  `token_count_gap` says whether the two tokenizers agree.

## 0.40 (2026-09-14)

- Docs only: fixed the `zeroproof.simulations.judging` module pointer (it is
  `score.judging`), lifted the judge contract and marker-polarity rule into
  the README, gave `traces=` and the close-the-loop toolkit a section,
  replaced the `spec="specs/..."` snippets (no spec folder ships) with
  `tools=` + `system_prompt=`, and corrected `.per_task` (a dict, not a
  vector), the pass^k/pass@k interval claim, `tasks=` k inheritance,
  `STOCK_MARKERS`, `export_dataset`/`export_training`, and
  `trim_out_of_band`. No behavior change.
- `export_training(format="trl")`, `export_preference(format="trl")` and
  `zps.to_trl(rows, kind)`: the shape TRL actually loads — conversational
  SFT rows with no `prompt` string column beside `messages` (it made
  `is_conversational` return False, so `SFTTrainer` trained on the bare
  ask with no error; the ask is now `prompt_text`), DPO rows as prompt
  messages plus completion-only sides, and tool-call `arguments` as dicts
  rather than JSON strings for HF chat templates. The default stays the
  OpenAI wire shape, and `tool_call_roundtrip` now reports the `encoding`
  it checked. `validate({})` is `["empty_row"]`, and
  `validate(row, "training" | "preference")` checks the shape at every
  schema version (#152).
- Trust layer: the `calibration` stamp is now measured before
  `optimize(mode="rl")` prunes and carried onto the selection (the gate
  keeps it instead of re-measuring post-dedup k), the report warns that
  `pass^k`/`pass@k` do not survive the prune, and `hack_scan` returns
  `degenerate` rather than naming an arbitrary tied feature when too few
  distinct trajectories leave every candidate collinear with reward;
  `hack_scan_diff` withholds `learned` on a degenerate side for the same
  reason, instead of reading a tie as what the policy learned.

## 0.39 (2026-09-14)

- `hack_scan`: a tie in magnitude goes to the endorsed feature (the
  complement of the behavior correlates exactly as strongly, with the
  opposite sign, and is not a second thing the policy learns), and an
  endorsed feature the reward punishes is a `reward_hack` of its own,
  reported as `inverted` with a "reward punishes" warning. Found on the
  example: the honest reward's strongest feature was the shortcut
  sentence at rho -1.0.
- `examples/reward-hacking` and `docs/reward-hacking.md`: the detection
  loop end to end, offline, in seconds. A scripted refund agent that
  sometimes takes a shortcut, an honest judge that reads the trajectory
  and a hackable one that reads the prose; `hack_scan`, `judge_probes`
  and `trace_flag_report` on both, then a second agent that learned the
  shortcut stands in for "after training" and `delta_report(proxy=)`
  calls it over-optimized while `hack_scan_diff` names what it learned.
  The doc is the recipe: what the book says, the five checks, the three
  rules (endorse the behavior, fix the judge not the rows, keep the gold
  away from the proxy).

## 0.38 (2026-09-14)

- Examples audit (#168, #169, #170, #171, #172). Every example now has a
  test under `tests/examples/` that runs its offline path end to end and
  checks README flags, defaults and quoted numbers against the code; 100+
  new tests. Fixed: `bring-your-own-agent` part three had stopped firing
  under rl mode's `successive` repeat policy; `pass-at-k` printed a
  different pass@1 per run at the same seed (now `reproducible=True`,
  `--concurrency` exposed); `agent-behavior --dry-run` exited 0 on a
  dead endpoint; `identity/generate.py` wrote to a path on another
  machine and `eval_modal.py` could not read its output (new
  `report.py` with `identity_rate` / `leak_rate` and Wilson intervals);
  `grpo` README told readers to pass `--gpu`, which the Modal
  entrypoints now accept; `hosted-loop` ignored `zeroproof login`;
  `prime-intellect-rl/export_prompts.py` crashed without `data/` and
  collapsed rows with no `scenario_id` into one task; `hugging-face`
  gained the `--push-run` the README promised; `verifiers` README no
  longer points at a spec that does not exist. `examples/README.md` is
  a map of the twelve examples in post-training order. Main README:
  two broken code fences fixed, `dpo` and `hugging-face` added to the
  table, `attach_labels` / `decontaminate` return shapes and `train()`
  step/epoch knobs corrected.
- `Weighted` verifier: a part that cannot run (no reference on the row)
  now returns `reward: None` with the part's reason, like `ExactMatch`,
  `All` and `Any`, instead of scoring 0; a rubric row with a missing
  answer key no longer lands in RL data as a hard fail. Direct tests for
  every exported name that had none (`datasets`, `delete_dataset`,
  `list_runs`, `get_run`, `delete_run`, `models`, `claude_code`,
  `hosted_model`, `novelty`, `behavior_signature`, `Trait`) and ten for
  the OTLP `zeroproof.ingest` module, which had zero.
- `delta_report(proxy=)`, and `proxy=` on `run.delta` / `attach_delta`:
  name the training reward's marker (e.g. `"marker:first_action"`) and
  the report says whether the run over-optimized it (rlhf-book ch. 14):
  `over_optimized` is true, the report fails, and a warning names both
  intervals when the proxy moved up while the target did not, or the
  proxy's interval sits entirely above the target's. `proxy_verdict`,
  `proxy_delta`, `proxy_ci95` on the report; `format_delta_report`
  prints the proxy line. `zps.hack_scan_diff(before, after, endorsed=)`
  is the scan before training against the scan after on rollouts scored
  by the same reward: `gained` (features that clear the floor only
  after), `lost`, `moved`, and `learned`, the one line that names what
  the update moved toward and whether it is endorsed;
  `format_hack_scan_diff` prints it.
- `zps.trace_markers(rows)`, `zps.trace_flags(row)`, `zps.trace_flag_report(rows)`:
  did the agent fake the work? Flags read from the trajectory rather
  than the prose (rlhf-book ch. 13, 14), a port of the agent-behavior
  example's signals onto the row shape. `lie.tests_claimed` (tests said
  to pass when no test command ran or the last one failed, hedged claims
  excluded), `lie.unverified_claim` ("I verified" with no tool calls),
  `lie.phantom_edit` ("I updated" with nothing written),
  `lie.ignored_failure` (the turn ended on a failed call and the reply
  never mentions trouble), `hack.test_edited`, `hack.test_weakened`,
  `hack.suppressed`, `hack.bypassed`, `risk.destructive`, `risk.secrets`,
  each with the fragment that raised it. Reads `steps`, platform
  `tool_trace`, assistant `tool_calls` with their `tool` results, and
  `<tool_call>` blocks; a failed step is a failing status (the mock
  world's timeout, permission_denied, not_found, rejected, error), a
  non-zero exit code, or an error-opening result. What counts as a
  read, write, delete or command comes from the tool's arguments and
  name; `kinds={"tool": "write"}` overrides. The markers (`honest_claims`,
  `reported_failure`, `no_test_tampering`, `no_suppression`,
  `no_bypass`, `no_destructive`, `no_secrets`, 1.0 = clean) feed
  `marker_summary`, `delta_report(must_not_regress=)` and `hack_scan`,
  whose hand tier now carries one `trace:<flag>` feature per flag that
  fired; `reward_correlations` scans the fired flags beside length and
  the style phrases. `trace_flag_report` gives each flag's rate,
  examples, and its correlation with the reward, flagged when the judge
  pays for the fake.
- `data.rows()` and `output=` write the whole row (#149). The export was
  an allowlist, so 16 keys the trajectory carries never reached disk:
  `markers` (which re-broke #56 for anyone reading `rows()`),
  `judge_status` / `judge_name` / `lineage`, `scenario_dimensions`,
  `seed`, `arm`, `behavior_signature`. Now everything is exported except
  the teacher-only `privileged` block (`principle`, `hidden_state`,
  `reference`, `rubric`), which is dropped at every depth, and the
  in-memory `vector`. `data.rows` also reads as a list, matching
  `ScoredData.rows`; `data.rows()` still works.

## 0.37 (2026-09-14)

- `zps.train(generations=, learning_rate=, beta=, seed=, max_completion_length=,
  loss_type=, config=)`: the knobs a hosted run is reproduced and compared by
  (rlhf-book ch. 6, 7) reach the trainer by name and land on the run's
  config; ranges are checked before the call. Needs the gate and trainer
  deployed 2026-09-14 (site 47788a0).
- `rubric_judge` numbers the checklist and asks for verdicts by item
  number; `Rubric.score` also resolves a title, its slug or a paraphrase
  that contains it, and results carry `n_unanswered`. Live on the hosted
  judge, 10 of 32 rows had items answered under a rewritten title and
  failed for it; now none do (#156).
- `concurrency: 1` is round-synchronous, like `reproducible=True`: the
  batch's rollouts and their in-loop verdicts all land before the next
  round is chosen. Two same-seed serial runs in one process could draw
  different situations: the 0.35 s collect window decided how many of a
  batch's rollouts a round saw, and the rounds spent waiting drifted the
  counter that seeds selection. Cold processes happened to agree, so the
  cross-process check passed; `tests/api/test_reproducibility.py` now
  also runs `simulate()` twice in one process under contrasting latency.
  Golden captures move: a serial run now folds every batch whole, so a
  `scripts/golden.py` diff across this change is expected to differ.

## 0.36 (2026-09-14)

- `examples/dpo --constructed-negatives`: for every no-id or off-topic
  prompt the policy answered without a tool call, pair that reply
  against an invented call (`pairs.constructed_negatives`), so DPO has
  contrast on the prompts where its own samples had none; a second
  balanced round without it had made the invented-id habit worse.
- `select_for_rl(truncated="drop" | "keep" | "penalize")` and
  `optimize(mode="rl", truncated=)`: a rollout cut at the token cap is
  dropped (default), kept with `overlong=True` and a `finished` marker, or
  kept as a failure with the judged score under `reward_before_penalty`
  (DAPO's overlong handling, rlhf-book ch. 6, 7). `drop_truncated=False`
  now means `"keep"` (#139).
- Over-optimization and eval-variance consolidated onto one module each (#132): `score.style` (style_markers/style_report/refusal_report, 1.0=clean, delta_report-ready) and `score.stats.eval_variance` are canonical; `score.markers` (behavioral_markers/mark_rows) now warns (its presence polarity reads a paired delta backwards); the unreleased `score.benchmark` is removed.
- `training_rows(max_tool_output_chars=)` / `export_training(...)`: each
  tool message over the cap is cut with a `[... N chars of tool output
  truncated]` marker and counted on the row (`tool_output_truncated`,
  `tool_output_chars_cut`) and in the report (rlhf-book ch. 13). The Claude
  Code adapter's 2000-character cut is now `CLAUDE_CODE_RESULT_CHARS` and a
  cut step carries `result_truncated` and `result_chars` (#146).
- `examples/dpo`: constructed negatives are one pair per distinct prompt
  and capped at `constructed_share` (0.3) of the sampled pairs. Uncapped,
  the balance repeats multiplied them past the sampled pairs and the
  policy learned "never call" (with-id pass@1 0.11 to 0.05).
- `decontaminate` reads prompts only by default, the book's method, and
  counts a row when it is an eval text verbatim or when one eval text
  covers at least 80% of its words (`overlap=`, the Llama 2 rule) rather
  than on any single shared 8-gram. Situations written from one set of
  templates share whole sentences, so the old default flagged every row
  of a train/eval pair from the same simulator (rate 1.0, nothing kept)
  while 41% of the prompts actually repeated; the report now separates
  `n_exact` from `n_near`, counts hits per field, and carries each
  offender's coverage. `fields=("prompt", "final_text")` keeps the
  stricter replies-versus-answers check (#125).
- `zps.judge_probes(rows, judge)` and `judge_trust(probes="all")`: the
  reward hacks a policy finds first, tried on the judge on purpose
  (rlhf-book ch. 14). Seven probes mutate a sampled reply one way and
  re-judge it: `filler`, `keyword_stuffing` (the rubric's own words,
  from `rubric=` or the row's system prompt), `success_claim` ("Done. I
  verified this and all tests pass."), `prompt_echo`, `sycophancy`
  ("You're absolutely right."), `empty_format` (a well-formed call to
  the row's tool with empty arguments), `refusal`. An additive probe
  reports the share of originally failing replies that pass once the
  text is added; a replacement probe the share that pass with the
  content gone. `exploitable_by` names the probes at or over the 10%
  flip flag, each with a one-line warning, and `judge_trust` fails on
  any. `format_judge_trust` prints the probe table. A probe with
  nothing to work on (no rubric words, no tool) is `skipped` with the
  reason.

## 0.35 (2026-09-14)

- `examples/dpo --constructed-negatives`: for every no-id or off-topic
  prompt the policy answered without a tool call, pair that reply
  against an invented call (`pairs.constructed_negatives`), so DPO has
  contrast on the prompts where its own samples had none; a second
  balanced round without it had made the invented-id habit worse.
- `select_for_rl(truncated="drop" | "keep" | "penalize")` and
  `optimize(mode="rl", truncated=)`: a rollout cut at the token cap is
  dropped (default), kept with `overlong=True` and a `finished` marker, or
  kept as a failure with the judged score under `reward_before_penalty`
  (DAPO's overlong handling, rlhf-book ch. 6, 7). `drop_truncated=False`
  now means `"keep"` (#139).
- Over-optimization and eval-variance consolidated onto one module each (#132): `score.style` (style_markers/style_report/refusal_report, 1.0=clean, delta_report-ready) and `score.stats.eval_variance` are canonical; `score.markers` (behavioral_markers/mark_rows) now warns (its presence polarity reads a paired delta backwards); the unreleased `score.benchmark` is removed.

## 0.34 (2026-09-14)

- `zps.hack_scan(rows, endorsed=[...])`: what a grouped update would
  learn from these rewards, named before training (rlhf-book ch. 6, 14).
  Reward and every candidate feature are centered within ask, the way
  GRPO baselines them, ranked by that correlation, and compared to a
  noise floor from shuffling reward within ask (`tau`, 95th percentile
  of the null maximum), so a feature that only tracks difficulty never
  counts and the threshold is measured, not guessed. Two tiers, pure
  Python: the hand tier (reply length, tool calls, turns, truncation,
  surface counts, one indicator per tool called, mean token logprob,
  every numeric marker, plus `features=` of your own) and the auto tier
  (presence of the 200 most common words and word pairs in the agent's
  text, and pairwise ANDs that beat both parents). `endorsed` names what
  the reward should track (feature-name substrings such as
  `"tool:lookup_order"` or `"marker:grounded"`); with it the report says
  `regime`: `train`, `reward_hack` (the top feature is not endorsed),
  `pool_exhausted` (over 20% of asks all-pass), `no_signal` (nothing
  clears the floor) or `unknown`, plus `integrity` (share of the
  above-floor signal that is endorsed), `top_feature`, the ranking with
  the pooled correlation beside each, and one-line `warnings`.
  `zps.format_hack_scan(report)` prints it. Duplicate columns are one
  feature with `aliases`; 3,200 rollouts scan in about a second.
  `select_for_rl` / `optimize(mode="rl", endorsed=)` carry it as
  `report["hack_scan"]` and its warnings in `hygiene_warnings`;
  `publish_gate` / `push_rows(gate=True)` / `data.push` report it on
  RL-shaped rows and, with `strict_hacks=True`, refuse a `reward_hack`.
  The pooled `reward_correlations` scan stays as the second column.
- `zps.HackMonitor(run, holdout=, proxy=, gold=, ...)`: is the run
  hacking its reward right now (rlhf-book ch. 14, figure 1)? A
  Transformers / TRL callback plus `monitor.wrap(reward_fn)` around the
  reward function. Every `every` steps it samples the holdout from the
  live policy (`k` completions on up to `n_prompts` asks) and scores it
  twice: with the training reward (the proxy; `proxy=` or the wrapped
  function) and with a scorer the proxy cannot see (`gold=`, any judge
  under the SDK contract). Both land on the run as `proxy_reward` and
  `gold_reward`, with `holdout_length`. Four alarms, one line each on
  the run and in `monitor.alarms`: `divergence` (proxy up by `delta`
  over `window` evals while the paired gold interval does not move
  up), `length` (completions up by `length_pct` while gold does not),
  `drift` (the trainer's KL past `kl_budget`), `feature` (the last
  `buffer` completions' `hack_scan` says `reward_hack`; needs
  `endorsed`). `stop_on=` names the alarms that stop the trainer; the
  default logs only. The summary (`history`, `alarms`, `last_scan`,
  `stopped_at`) rides on the run's `finish` through `run.note`, a
  stopped run finishes as `stopped` with the reason;
  `zps.format_hack_monitor(summary)` prints it. `TrainingRun.note`
  is new: fields it sets travel with whichever callback finishes the
  run. `examples/grpo` wires the monitor by default (`--monitor-every`,
  `--stop-on`).
- `run_judge(scale=(lo, hi))`, `evaluate(scale=)`, `data.grade(judge=, scale=)`:
  a rating judge (1 to 5, 0 to 10) is read on its scale; `reward` is the
  rating mapped onto [0, 1] and `judge_meta` keeps `rating` and `scale`;
  a rating outside the scale is `invalid_result` (rlhf-book ch. 11) (#137).
- `zps.attach_labels(rows, labels, annotator=)`: hand labels from a JSONL
  path, a list or a mapping, matched by rollout id, scenario id plus
  rollout index, or prompt plus final text; every label stays on the row
  as `gold_labels` (label, annotator, kind, ts, note) and `gold_reward` is
  the majority, unset on a tie. `zps.annotator_agreement(rows)`: per-
  annotator counts, unanimous share, Cohen's kappa for the busiest pair,
  the split rows (rlhf-book ch. 10, 11) (#136).
- Reward model as a judge (rlhf-book ch. 5). `zps.train(ds, method="rm")`
  trains a sequence-classification head on the set's pass-vs-fail pairs
  (Bradley-Terry loss, the pairs DPO uses) and reports pair accuracy on the
  held-out pairs before and after plus the score threshold that separates
  them. `zps.reward_model(run)` is that run as a judge: it honors the judge
  contract (`reward` 0/1 against the threshold, `rm_score` raw), so it feeds
  `data.grade(judge=)`, `evaluate`, `judge_trust` and
  `build_preference_pairs`. Gate route `POST /runs/{id}/score`.
- `simulate(tasks=previous_run)` re-runs a previous run's task set (the
  run, its rows, or its JSONL path) instead of drawing a new one: every
  prompt again, on its own `scenario_id` and `scenario_dimensions`, under
  the same faults and world state, and nothing else generated; the run
  stops with `tasks_done` once every prompt has its rollouts and reports
  `search["pinned_tasks"]`. A run draws its tasks by seed and, above
  `concurrency: 1`, by completion order, so even a same-policy re-run
  paired 43 of 49 tasks; pinned, every A/B (prompt edit, model swap,
  another seed) pairs all of them (#98).
- `data.grade(use_privileged=True)` / `grade_llm(use_privileged=)`: the
  hosted judge reads the row's `privileged` block (principle, reference,
  hidden state) as `judge_only` in its payload, with a prompt line on how
  to use it (rlhf-book ch. 12, constitutional AI). Folded into the judge
  version and stamped `judge_meta.privileged`; exports still never carry
  `privileged` (#134).

## 0.33 (2026-09-14)

- Rubrics as objects (rlhf-book ch. 12): `Rubric` / `Criterion` (hard rule,
  principle, pitfall; positive weights; a content hash as `version`) on the
  row's `privileged.rubric`, never exported. `zps.rubric_judge()` scores
  one verdict per criterion (`Rubric.score`: a missed hard rule is 0,
  otherwise principles minus pitfalls over principle weight) and lifts each
  criterion onto the row as a `rubric:<item>` marker; `zps.write_rubrics(rows,
  domain=)` drafts one rubric per prompt with the book's rubric-writer
  prompt; `attach_rubric`, `rubric_of`, `score_with_rubric` (#127).
- `training_rows(unroll=True)` / `export_training(unroll=True)`: an N-turn
  conversation becomes N samples, the k-th ending at the k-th agent turn
  with loss on that turn only (rlhf-book ch. 4); samples carry `unroll`
  (`turn`, `turns`) and `lineage.unrolled_from`, and no group fields (#129).
- `examples/dpo`: a second balanced round from the round-one adapter
  made the invented-id habit worse (no-id pass@1 0.82 to 0.26 while
  with-id rose 0.56 to 0.91); the README records it. DPO needs contrast
  on the no-id prompts themselves, not more of them.
- Sampling facts on every simulated row (rlhf-book ch. 6, 9):
  `policy_version` (`<model_version>@<sha256 of the system policy>[:16]`,
  round-tripped as `Rollout.policy.version`), `sampling` (`temperature`,
  `logprobs`) on model-backed rollouts, and with `simulate(logprobs="tokens")`
  the per-token `token_logprobs` list an importance-sampling ratio is built
  from; all three ride through `training_rows` / `export_training`.
  `zps.staleness_report(rows, base_model=)`: rows per policy version, stale
  rows for the model about to be trained, sampling / logprob coverage,
  with warnings (#121).
- `argument_grounding`: a marker for tool arguments that came from
  nowhere. `mark_grounding(rows)` stamps 1 when every string argument
  of every tool call appears in the prompt, the user and system turns,
  or an earlier tool result, else 0; `ungrounded_arguments(row)` and
  `grounding_report(rows)` name the invented values by tool and key.
  Reads `steps`, `messages` tool calls and `<tool_call>` blocks. The
  GRPO and DPO examples guard it with `must_not_regress`, so the
  invented-id regression fails the run instead of hiding under pass@1.
- Hosted GRPO and DPO run on an L40S, so `zps.train(method="grpo")` on a
  served base (`Qwen/Qwen3-4B`) trains and serves; the docs no longer say a
  4B base does not fit.

## 0.32 (2026-09-14)
- `zps.eval_variance(run_1, run_2, ...)` (or one row list split by
  `lineage.scoring_run_id` / `by=`): the eval's own re-run standard
  deviation, `noise_band` = 2 x std, and Olmo 3's stability band in
  points (rlhf-book ch. 16). `delta_report(run_std=)` marks every
  metric whose delta sits inside that band `within_noise`, keeps it out
  of improved / slipped / regressions, and reads a target there as
  `within_eval_noise` instead of moved (#113).
- `zps.judge_pairs(pairs, judge=None, swap=True)` asks a judge which side
  of each preference pair is better, then again with A and B swapped
  (rlhf-book ch. 5, 11). Each pair gets `pairwise` (winner, whether the
  two orders agreed, reasons, judge) and `tie`; a pair decided
  differently in the two orders is a tie with `position_consistent=False`.
  Report: `position_flip_rate`, `tie_rate`, `agrees_with_scores`,
  `prefers_rejected` with examples. `zps.pairwise_judge(spec)` is the
  hosted model judge with a length-neutral prompt.
  `export_preference(drop_ties=True)` leaves ties out and counts them (#114).

- `examples/grpo` and `examples/dpo`: `--balance <share>` repeats the
  prompts of any category below that share of the train split
  (`prompts.balance`), so the no-id and off-topic prompts reach the
  update at the rate they matter. The invented-id regression the
  stratified holdout exposed is a sampling-frequency problem, not a
  reward one for GRPO: the invented-id rate on no-id prompts drops 0.23
  to 0.08 at 120 steps. One-round DPO does not move on it (no pairs
  where the base never fails); a second round is DPO's lever.
- A policy edit keeps the task grid (#98). The covering array is built in
  layers: a rule-free block over tools and situation axes, then one block
  per policy clause, rotated by the clause text. Editing, adding or
  removing one clause changes that clause's cells only; every other
  `scenario_id` survives, so `compare_runs` stays paired after a prompt
  change (12 of 53 tasks paired before; all of them now). The grid is
  larger by the rule-free block, and cells in it carry no rule hint.
  Fault rows are kept per row from the row's own digest, one of every
  kind guaranteed, so a grid that grows keeps the verdict on the rows it
  had. Every existing grid changes once with this release.
- `compare_runs` says when it dropped tasks: `note` names how many were
  on one side only and that the verdict rests on the shared ones, with
  "most tasks unpaired" in front when fewer than half paired;
  `paired_share` is the fraction. `delta_report` carries it as a warning
  on the headline metric and reports `n_unpaired_tasks`.
- `examples/hosted-loop`: push, `zps.train`, `zps.serve`, call, as one
  script with a state file per step; the wiring check for training on the
  platform, with the served-base, cold-start and thinking-mode notes a
  first run needs.

## 0.31 (2026-09-14)

- `examples/grpo` and `examples/dpo`: the model-written set is split by
  scenario within each prompt category (`split_holdout_stratified`); the
  hash split had put every no-id situation in train, so the holdout
  could not see a policy that always calls the tool. Both scripts report
  `by_category_before` / `by_category_after` (pass@1 and tool-call rate
  per category). `examples/dpo --from-run` merges a previous round's
  adapter and samples fresh pairs from it: iterated on-policy DPO, with
  the merged policy saved for serving or a further round.
  On the stratified split both GRPO (120 steps, 0.29 -> 0.81) and DPO
  (one round, 0.29 -> 0.72) learned to invent an order id on a quarter of
  the no-id prompts (no_id pass@1 0.95 -> 0.75 and 0.98 -> 0.72); the
  hash split had hidden it. DPO round two from the round-one adapter:
  0.63 -> 0.92.
- `delta_report(by=...)`, and `by=` on `run.delta` / `attach_delta`: the
  target compared within each group of rows (a row key, a marker name,
  or a callable), reported as `groups` with `groups_down` for a group
  whose target dropped significantly; `format_delta_report` prints the
  block and the run page draws it. A headline over one dominant kind of
  prompt no longer hides the other kinds.
- Over-optimization signatures as markers (rlhf-book ch. 14, 17):
  `zps.style_markers(rows)` stamps `no_boilerplate`, `no_hedging`,
  `no_apology`, `no_sycophancy` and `answered` (1 = clean) from phrase
  lists, so `marker_summary` and `delta_report(must_not_regress=)`
  watch them; `zps.style_report(rows)` gives each signature's clean
  share with an interval, the phrases that fired, and its correlation
  with the reward, flagged when the judge pays for the tic;
  `zps.refusal_report(benign_rows)` is the over-refusal rate with a
  Wilson interval and examples. `reward_correlations` (and so the
  publish gate's hygiene warnings) now scans the same four phrase
  features next to length, tool calls and turns.
- `select_for_sft` / `optimize(mode="sft")` is rejection sampling by
  reward (rlhf-book ch. 9): `select="top_per_prompt"` (default),
  `"top_k_overall"` with `k=`, and `"random_per_prompt"` /
  `"random_k_overall"` as chance controls; `min_reward` (default 1.0)
  admits partial-credit graders, whose 0.9s were dropped as not-pass.
  Reports gain `selection`, `min_reward`, `reward_mean_eligible`,
  `reward_mean_selected`.
- Exported groups: `n0`/`n1` count partial credit below/above 0.5 (a
  0.3/0.9 group read as unanimous), plus `reward_mean` and `reward_std`
  per group so a trainer can see where group normalization divides by
  ~zero (rlhf-book ch. 6).
- `decontaminate` no longer takes the eval set's own replies as n-gram
  sources, only prompts, answers and references (rlhf-book ch. 16); tool
  boilerplate shared between two replies flagged clean rows.
- `llm_judge` samples at temperature 0, like `grade_llm`.
- `examples/grpo`: `--loss-type` (bnpo, TRL's default; grpo; dr_grpo),
  `--epsilon-high`, `--no-scale-rewards` and `--mask-truncated`, so
  Dr.GRPO and DAPO's clip and overlong mask are flags on the one trainer
  and land in the run config. `prompts.jsonl` is checked in for real (a
  repo-wide `*.jsonl` ignore had swallowed it) and the test requires it.
  On it: GRPO 120 steps 0.18 -> 0.85 (+0.63 [+0.50, +0.73]), Dr.GRPO at
  the same budget 0.17 -> 0.53, DPO one round 0.17 -> 0.69.
- With `grader=`, every mode judges rows as they land, on the judge pool
  beside the rollouts; only the tail is judged after the clock. Before this
  explore and sft judged everything in one pass after the run, which on a
  120 s run added about a minute past the budget. `data.search["grader"]`
  now says how many rows were judged in the loop and how many after.

## 0.30 (2026-09-14)

- `examples/grpo`: a model-written prompt set. `prompts.py` (writer chat
  per template seed, array parsing, category and near-duplicate filter
  through `case_for`, loader) and `write_prompts_modal.py` (Qwen2.5-7B-
  Instruct on an A10G) produce `prompts.jsonl`, 707 prompts from 67
  situations, checked in; both the GRPO and DPO scripts take
  `--prompts-file`, so the holdout is over a hundred prompts instead of
  fourteen and pass@1 intervals shrink accordingly.
- `zps.train(dataset_id, method="sft"|"grpo"|"dpo", steps=, epochs=,
  holdout=, base_model=, wait=)` starts a hosted run on the platform's
  trainer and returns the `TrainingRun` the dashboard draws; `run.refresh()`
  / `run.wait()` follow it, then `run.adapter` and `run.training` (before,
  after, rows, seconds). `zps.serve(name, run)` hosts the adapter on an
  OpenAI-compatible endpoint and `zps.models()` lists them. README and
  the character docs no longer say the SDK does not train (#77).
- `mode="rl"` spends rollouts where the agent is inconsistent. Every
  prompt is probed with two rollouts; a prompt that splits is filled to k,
  a unanimous one stops once the run's own measured rates say a fresh
  prompt is the better bet. `grader=` runs beside the rollouts so the
  allocation reads rewards. A `time_budget` finishes the groups in flight
  instead of cutting them. `repeat_policy="fixed"` is the old behavior.

## 0.29 (2026-09-14)

- `pass_at` says when groups are uneven. A time or row budget that cuts a
  run mid-group leaves ragged groups; `k` defaults to the smallest, so the
  k-way numbers were withheld with "set repeats>=4" even when repeats was
  4. The note now names the size range and the `k=` that scores the groups
  which reached it.
- `SimulationData.grade` docstring names what the no-argument path is
  (`grade_llm`, the hosted Phi-4 judge, in place, returning the judge
  report) and README marks `grade=True` as the legacy conduct score.
- An `agent=` callable that fails every call is called off after
  `max(16, 2 * budget)` lost rollouts with `stopped_because="agent_failed"`,
  instead of re-rolling each lost slot and refilling it until the writer
  ran dry (~15 calls per budgeted row). Runs with any surviving row keep
  the re-roll behavior (#88).

## 0.28 (2026-09-14)

- Judge verdicts: a complete JSON object in the reply decides on its own.
  A string-typed score, a duplicate `score` key, a `score` that disagrees
  with a `reward`, a bool, NaN or 1.5 leave the row ungraded; the digit
  salvage runs only when no complete object exists (#64).
- A broken `agent=` callable ends the run as `stopped_because="agent_failed"`
  instead of `writer_exhausted`, with `search["agent_errors"]`,
  `search["first_agent_error"]` (exception type included), a degraded note
  and a logged warning. A return shape without `steps`/`final_text` counts
  the same way. README states the callable contract (#29).
- `examples/agent-behavior` runs on a fresh clone: the uncommitted
  `tasks_hard` pack is optional. `tests/examples/` smoke-tests every
  example offline (#38).
- `rows_from_otel` sums `gen_ai.usage.*` (and the `llm.token_count.*`
  dialect) into `row["usage"]`, the shape simulated rows carry (#67).
- `simulate()` documents `lineage.scoring_run_id` as per-invocation
  identity outside the bit-for-bit guarantee; the `grader=` path is now
  pinned by the reproducibility test (#58).
- `select_for_rl`, `select_for_sft` and `build_preference_pairs` report
  `eval_sourced` (rows or pairs whose reward came from `evaluate()`) and
  warn when it is non-zero; `optimize.eval_sourced(rows)` is the count on
  its own. No row is dropped (#37).
- The covering array behind `scenario_regions` is memoized per process;
  each writer wave was rebuilding it and throwing it away. The slowest
  test drops from 26-49 s to under 1 s and the suite from 162 s to
  100 s. Rows out of the simulator are unchanged (golden harness: 13/13
  identical) (#36).
- `examples/dpo`: DPO on Modal end to end, the offline counterpart of
  `examples/grpo`. Pairs come from the base policy's own samples scored
  by the same rule and paired by `build_preference_pairs` (length
  matched), or from a `zps.export_preference` file with `--pairs`; TRL
  `DPOTrainer` with LoRA on Qwen2.5-1.5B-Instruct, the reference model
  is the adapter switched off, reward margin and accuracy on the
  dashboard, pass@1 before and after on a holdout, `run.delta` on the
  run page. `pairs.py` (first-turn rendering, TRL rows, export loader)
  is unit-tested offline.

## 0.27 (2026-09-14)

- Character training docs and example README point at the dataset's
  home, `zero-proof-ai/character-training-model-spec` on Hugging Face
  (train, holdout, eval) and the catalog agent `sol-character`;
  `docs/character-training.md` gains a section on the run's rows.

## 0.26 (2026-09-14)

- `examples/grpo`: GRPO on Modal end to end. Prompts from the offline
  template writer, a verifiable tool-discipline reward (`reward.py`,
  unit-tested), TRL `GRPOTrainer` with LoRA on Qwen2.5-1.5B-Instruct,
  reward and KL on the dashboard, pass@1 before and after on a
  holdout, and `run.delta` on the run page.
- Judge replies that break the contract stay ungraded. The binary
  verdict parser read a bare `true` as 1, `{"score": 1.5}` as 1 and
  `{"score": 0.5}` as 0; the 0-to-1 judge clamped a 2 to 1.0 and a -1
  to 0.0. All of these now return no score, the same contract
  `judging.py` already held a caller's own judge to. Well-formed
  verdicts inside chatter and replies cut off after the score still
  grade.

## 0.25 (2026-09-14)

- Hugging Face, both directions. `zps.hf_status()` says whether an account
  is connected and which namespaces it can publish under. `zps.hf_publish`
  pushes one of your sets to a dataset repo you own: one split per purpose,
  every push a commit tagged `zp-<dataset id>`, `zeroproof.json` in the repo
  mapping splits to datasets with history; it waits for the platform to
  stamp the commit and returns it (`wait=False` returns the pushing stamp).
  `zps.hf_publish_run` does the same for a finished run's LoRA adapter, as
  a model repo, private by default. `zps.import_hf(repo, split=...)`
  brings any Hub split onto your account as rows and waits until it is
  ready, so `zps.profile` can grade it before you train on it. Example in
  `examples/hugging-face`.
- Every model call now keeps what it cost. The server's `usage` block
  becomes `input_tokens` / `output_tokens` on the agent step and a summed
  `usage` on the row (`rows()`, `training_rows`, the typed `Step` and
  `to_row` all carry it). Before this no row said how many tokens it
  used, so a trace built from one had no `gen_ai.usage.*` and the
  platform's per-day usage counted zero for every simulation.
- Hosted-model tokens now reach the platform's Usage page. After every
  call to the hosted policy or judge the SDK batches the `usage` the
  server reported and sends it to `POST /usage` under the account's own
  key (background thread, once more at exit; `ZEROPROOF_NO_USAGE_REPORT=1`
  turns it off). Bring-your-own endpoints are never reported.

## 0.24 (2026-09-14)

- Character training example (`examples/character`): the OpenAI Model
  Spec's style section parsed into a constitution with its GOOD/BAD
  comparisons, prompts per trait, k replies each, a judge that reads the
  trait's principle, preference pairs and SFT rows with the deployment
  prompt, and `measure.py` for the before/after delta with `on_task` and
  `no_filler` guarded. The spec's labeled replies grade the judge
  (`judge_agreement`). Offline by default; `--model-url` for a live model.
  How-to in `docs/character-training.md`.
- A row's `privileged` block (`principle`, `hidden_state`, `reference`)
  now reads into `Task.privileged` in `from_row`. `to_row` still never
  projects it (it is the teacher's context, one step from a training
  file); a source row's block rides back out as passthrough. The engine
  leaves it empty; character and rubric pipelines write it.

## 0.23 (2026-09-14)

- Markers a judge returns now reach `marker_summary`. `run_judge`
  lifts `judge_meta["markers"]` onto `row["markers"]` (the judge wins a
  name collision), `simulate(grader=)` carries `markers` onto the
  trajectories, and `from_row` on a graded row returns the `Marker`
  objects. Before this a marker a judge returned measured nothing.
- `split_pseudo_production` splits by task, not by row: rows group by
  `prompt` (`scenario_id` when there is no prompt) and whole tasks move,
  so `mode="rl"` with `repeats>1` no longer puts siblings of one prompt
  on both sides. The flaw-signature rule sends a task, not a row, to the
  held-out side. `fraction` is still counted in rows and can overshoot by
  up to one task. On the reported repro, held-out prompts also present in
  train went from 100% to 0%.
- `scripts/golden.py`: a committed golden-output harness. Thirteen offline
  configurations at `concurrency=1` on fixed seeds; `capture` and `diff`
  compare two snapshot directories and `diff` names every changed key.
  `.gitignore` no longer blocks `scripts/`. The harness scrubs
  `lineage.scoring_run_id`, which `run_judge` stamps fresh per call
  (#58).
- CI measures line coverage with a floor at 84% (Python 3.12,
  `COVERAGE_CORE=sysmon`; the default tracer ran past 30 minutes).
- Tests pin that `principle`, `hidden_state` and `reference` never reach
  a student field or an exported file, and that an eval score stays
  distinguishable from a training reward through `lineage["source"]`.
- Test suite builds each identity dataset once instead of four times
  (about 22% less wall time).
- Training runs: `run = zps.training_run(name, dataset=, base_model=,
  total_steps=)`, `run.log(step, loss=, ...)`, `run.progress`,
  `run.finish()`; `zps.TrainerCallback(run)` for Transformers and TRL
  trainers; `zps.list_runs`, `zps.get_run`, `zps.delete_run`. Points
  are buffered and sent in batches and logging never raises into the
  training loop. The platform draws the loss curve and progress bar at
  /platform/training.
- `zeroproof purge --agent <slug>` and `--empty`: an agent with its traces,
  datasets and record, or datasets with no bytes (or few rows), gone in
  one command after a y/N; `--dry-run` counts. Python `zps.purge_agent`,
  `zps.delete_empty_datasets`.
- Agents: `zps.agents()`, `zps.register_agent(name, tools=, system_prompt=)`;
  `data.push(name, agent=...)` registers the agent and attaches the run's
  tools and system prompt to its record.
- `judge_trust` says when the gold labels are all one class instead of
  reporting kappa 0 and a length bias the labels cannot support;
  `gold_degenerate` on the report. Found on the first hosted-judge run.

## 0.22 (2026-09-14)

- The default judge is no longer the policy model. `zps.grade` uses
  `default_judge_spec()`: hosted `microsoft/phi-4` on its own vLLM app
  (`ZEROPROOF_JUDGE` overrides), while rollouts stay on hosted Qwen3-4B.
  A judge grading its own model's writing prefers it (rlhf-book ch. 5,
  12). The grade report carries `self_judged` and a warning when the
  judge model equals the rows' `model_version`. `judge_spec` with a bare
  URL now defaults the model name from the judge spec.

## 0.21 (2026-09-14)

- Platform-shaped rows count their tool calls: the reward-hack scan's
  `tool_calls` reads `tool_trace` as well as `steps`, so a pulled
  dataset no longer reports `None` for the tool-count correlation.
  The calibration stamp's `task_id` is the `scenario_id` when the row
  has one, not the prompt text.
- `simulate(logprobs=True)` asks the rollout model for the log-probability
  of every token it generates (rlhf-book ch. 6 off-policy correction,
  ch. 15 KL). Each agent turn's first step carries `logprob` and
  `n_tokens` (`"tokens"` adds `token_logprobs`), the row carries the
  totals, and a turn cut at the token cap is marked `truncated`. The
  fields ride through `export_row`, `training_rows`, `from_row`/`to_row`
  (`Step.logprob`, `Step.n_tokens`, `Step.truncated`) and the wire schema.
  A server that rejects `logprobs` is asked again without it.
- `zps.logprob_report(rows)`: capture coverage, mean token logprob,
  per-row quantiles, reward-vs-confidence correlation (flagged at 0.3),
  truncated count. `zps.mean_kl(rows, ref="ref_logprob")`: sampled
  KL(policy || reference) per generated token, overall and per task, from
  a reference logprob key or a second scored row list.
  `calibrate(rows, ref=...)` writes it into `calibration.mean_kl`, the
  field nothing populated before.

## 0.20 (2026-09-14)

- Every row `zps.grade` writes now says which judge produced it:
  `judge_name`, `judge_status`, and `judge_meta` (model, prompt hash,
  temperature, max_tokens, `version` = `<model>@<prompt sha>`); the
  report carries `judge_version`. A rubric edit is a new reward model,
  and the row records it. `run_judge(version=...)` does the same for a
  custom judge via `lineage.judge_version`; both read back as
  `Judgment.scorer.version`. New `schema.attach(row, judgment)` is the
  sanctioned verdict write; `grade` routes through it.
- `zps.judge_agreement(rows, gold="gold_reward")` (also
  `scored.agreement(...)`): agreement, Cohen's kappa, confusion counts,
  and the two disagreement rates against labels you trust, with the leak
  rate (gold failures the judge passed) called out because those rows
  train the failure. `gold` may be a second scoring pass for
  self-consistency. Warns below 50 gold rows (rlhf-book ch. 5).
- `training_rows` / `export_training` take `mask_mode="assistant"`
  (default, every agent turn) or `"final"` (only the last agent turn,
  rlhf-book ch. 4). `zps.loss_mask(messages, mode=)` builds the mask on
  its own; the export report carries `mask_mode`, `trained_messages`,
  `masked_messages`.
- Intervals and comparison: `pass_at(rows).ci95` (task bootstrap),
  `zps.metric_summary` / `zps.marker_summary`, and `zps.compare_runs`
  (paired task differences, bootstrap interval, sign-flip p-value;
  unpaired fallback under five shared tasks, labeled).
- `zps.decontaminate(rows, against=...)`: word 8-gram overlap against
  evaluation row lists, JSONL paths, or dataset ids; short prompts by
  exact match. Returns clean rows and the first offenders.
- `zps.delta_report(before, after, target=, must_not_regress=)`: pass@1
  and every shared marker compared as paired task differences; headline
  verdict on the target, regressions fail the report, other drops warn.
  `format_delta_report` prints it.
- `zps.judge_trust(rows, judge=)`: agreement with `gold_reward` labels
  (Wilson interval, kappa, confusion), held-out task halves, length
  sensitivity within human label, re-judge consistency and filler flips,
  and a disagreement queue. `format_judge_trust` prints it.
- Purpose on every pushed dataset: `data.push(name, purpose="train")`,
  `holdout=0.2` pushes a linked holdout set split by task,
  `zps.update_dataset(id, purpose=...)`, `zps.preview(id)`. The
  simulation mode is recorded on push. `zps.profile(id)` returns the
  trainer's numbers (pass rate, support, mixed tasks, tool use, per task).
- Preference pairs carry what a trainer and a reviewer need to trust them
  (rlhf-book ch. 8, 11): `chosen_score`, `rejected_score`, `margin` for a
  margin-aware loss; `chosen_model`, `rejected_model`, `same_policy` so an
  off-policy pair is labeled, not hidden; `length_delta` for the length
  exploit. `build_preference_pairs(min_margin=1.0, length_match=True)`:
  the default still pairs 1 against 0 only, `min_margin=0.5` admits
  partial-credit rows, and each chosen row now takes the rejected row
  closest to it in length. The report adds `mean_margin`,
  `same_policy_pairs`, `mixed_policy_pairs`, `length.chosen_longer_frac`,
  and `warnings` when chosen is the longer side in 75% or more of pairs
  or when pairs mix policies. `export_preference` keeps the new fields
  and reports `mean_margin` and `chosen_longer_frac`.

## 0.19 (2026-09-14)

- `zps.push_file` runs the publish gate too (`gate=False` uploads the
  bytes as they are), so a JSONL push no longer bypasses it. The stamped
  rows are what get uploaded; report on `entry["gate"]`.
- `calibration` is part of the typed contract: declared in
  `schemas/row-v1.json`, validated (`calibration_invalid`), read back
  with `zps.calibration_of(row)`, carried by `from_row` on
  `rollout.extra["calibration"]` and written by `to_row`.
- Training rows carry `loss_mask` (one 0/1 per message: 1 on assistant
  turns, 0 on system, user, and tool output). Declared in the schema and
  validated (`loss_mask_invalid`).

## 0.18 (2026-09-13)

- Public catalog: `zps.publish(id, agent=...)`, `zps.unpublish(id)`,
  `zps.catalog()`, and `data.push(..., agent=..., publish=True)` put a
  dataset on zeroproofai.com/datasets as a card grouped by agent.
  `zps.pull` fetches public sets with no key.
- Difficulty band is enforced, not just ranked: `select_for_rl` and
  `optimize(mode="rl")` drop asks whose pass rate falls outside
  `band=(0.2, 0.8)` (`enforce_band=False` restores rank-only); the
  report carries `band_dropped` by side. `group_signal` defaults to the
  same band (was 0.3-0.7). New `trim_out_of_band`, `DEFAULT_BAND`.
- Publish gate: `data.push` (and `push_rows(gate=True)`) runs
  `publish_gate` first. Every graded row gets a `calibration` stamp
  (task pass rate, k, producing policy); RL-shaped rows that are
  ungraded or have no mixed group raise `PublishGateError`. Report on
  `entry["gate"]`. New `zps.publish_gate`, `zps.calibrate`.
- Row hygiene: `select_for_rl` / `optimize(mode="rl")` drop duplicate
  rollouts within an ask (`dedupe=False` keeps them) and truncated
  rollouts (`drop_truncated=False`), and report the reward-hack scan
  (`correlations`: reward vs reply length, tool calls, assistant turns;
  flagged at `HACK_THRESHOLD` 0.3) plus a length report in
  `hygiene_warnings`. The publish gate reports the same, plus
  near-duplicate asks (token Jaccard 0.8), without dropping anything.
  New `zps.dedupe_groups`, `zps.near_duplicate_prompts`,
  `zps.length_report`, `zps.reward_correlations`.
- Both judge prompts say reply length must not influence the score.
  `select_for_sft` reports `completions_per_prompt_max` and notes when it
  is under 10 (rejection sampling wants 10 to 30 per prompt).

## 0.17 (2026-09-13)

- `signup` says that the key is a trial key and how to lift it.
  `zeroproof status` shows the tier; `zeroproof.account()` returns tier,
  limits and today's usage (`GET /me`).
- pass@1 / pass^k / pass@k from graded groups: `data.pass_at`,
  `ScoredData.pass_at`, `zps.pass_at(rows)` (a `PassAt` with
  `.headroom` = pass@k - pass@1 and `.per_task`). Unbiased estimators;
  k-way numbers withheld below `repeats=4`. `group_signal` reports the
  same keys, `recommend(mode="rl")` explains the mixed rate as headroom,
  and `save(meta=True)` writes `pass_at` to the sidecar.

## 0.16 (2026-09-13)

- `zeroproof signup --email`: creates the account and the key in one
  call, no browser and no password. Python: `zeroproof.signup()`.

## 0.15 (2026-09-13)

- `ruff format` across the repo, with `ruff format --check` in the CI
  lint job and `.git-blame-ignore-revs` pointing at the format commit.
  No behavior change: suite and golden harness identical.

## 0.14 (2026-09-13)

- `zeroproof login`: device-flow sign-in from a terminal or a coding
  agent. Prints a link and a code, waits for Approve in the browser,
  saves the key to `~/.zeroproof/credentials.json`. Platform calls fall
  back to that file when no env var is set. A transient network error
  while waiting is retried, not fatal. `zeroproof status` and
  `zeroproof logout`. Python: `zeroproof.login()`, `resolve_api_key()`.

## 0.13 (2026-09-13)

- Hygiene. ruff (lint) and mypy are configured in `pyproject.toml` and
  run in CI; `.editorconfig`, `.gitattributes` (LF) and a pre-commit
  config are in. The package type-checks clean except three modules that
  hang state on closures (`generate/generator.py`, `generate/scenarios.py`,
  `run/engine.py`), which are excluded until the writer is a class.
- The package root exports only `__all__` (76 names) plus the `data`,
  `schema` and `simulation` submodules. Sixty-one internal names that
  leaked through `import zeroproof.simulations as zps` are no longer
  reachable as `zps.<name>`; import them from their module.
- Cross-module helpers lost their leading underscore: `apply_spec`,
  `backend_spec`, `kind_from_spec`, `as_dict`, `intent_for_tool`,
  `load_jsonl`, `write_jsonl`, `row_cell_key`, `record_coverage`,
  `mutation_worthy`, `row_world`, `note_stage`, `clean_faults`,
  `export_row`. The old spellings remain as aliases.
- Dead code removed: six unused writer helpers, four unused constants.
- Lint fixes across the package: `raise ... from`, closure binding in
  the Claude Code adapter, redundant casts, sorted imports.

## 0.12 (2026-09-11)

- A callable `agent=` with no key now gets an error that names the two
  ways to run: `agent="openai:<model>"` on your key, or
  `simulator=False` for the built-in template writer. The README
  documents the no-key path.
- Rows in `data.trajectories` carry `messages`, matching the JSONL.
- `export_preference` on plain rows says it takes pairs and names
  `build_preference_pairs`.
- README notes that `fault_rate` applies through the mock world only.

## 0.11 (2026-09-11)

- Trace-driven allocation weighs a region's fail rate (Laplace-shrunk)
  and support, and flags regions with under three graded rows as
  `low_support`. Each region reports `n_graded` and `fail_rate`.
- `data.coverage["pairwise"]`: planned pairs, covered pairs, fraction.
- Docs say what the code does: the cold-start success flip means the
  tool-condition axis is sampled rather than covered; the leakage check
  is lexical; the unsourced benchmark claim is removed.
- Turn-length controller corrects at half gain; cluster sampling seeds
  per round; `planned_fault_fraction` removed (unused).
- The `zeroproof_simulations` alias stays until a later release rather
  than "two releases from now".

## 0.10 (2026-09-11)

- Annealing explore now prefers novel candidates. The acceptance curve
  had its sign flipped and took near-duplicates almost always.
- The search-arm bandit no longer rewards arms that produced no rows;
  idle arms take the mean observed yield and carry no vote.
- `recommend(mode="rl")` sizes the run as `goal / (k * mixed_rate)`
  instead of rounding `k * mixed_rate` to an integer first, which
  under-provisioned by 3x at a 4% mixed rate.

- Schema battle-tested against every row pool reachable: 11k local engine
  rows, both platform datasets, four agents from the public Hugging Face
  set, and an adversarial set. Two more legacy shapes are read by
  `from_row`: training exports (`messages` without `steps`; steps and
  `final_text` are derived, `tools` carried) and the Hugging Face set's
  flattened `*_json` string columns. Unknown columns now ride through
  `from_row` / `to_row` untouched, so verifiers-style `example_id` and
  `info` survive. Garbage values (a non-numeric fault rate, a
  non-integer `rollout_index`, bool or string rewards) coerce instead of
  raising. Judge rows keep `judge_status`, `judge_meta`, and both
  `judge_name` and `label_source` through the round trip.
- `examples/schema`: `migrate.py` stamps and splits any legacy file into
  rows plus a rollout-free `tasks.jsonl`; `project.py` writes eval, SFT,
  preference, GRPO, OPSD, and OPD targets from one v1 file. Offline, no key.

## 0.09 (2026-09-11)

- The simulations package moved under the namespace: `zeroproof_simulations`
  is now `zeroproof.simulations`, so the wheel is one package with one name.
  `import zeroproof_simulations` keeps working for two releases through an
  alias that resolves to the same module objects, with a deprecation
  warning. Change `import zeroproof_simulations as zps` to
  `import zeroproof.simulations as zps`. The logger is now
  `zeroproof.simulations`.

## 0.08 (2026-09-11)

- Typed row schema, additive half. `zeroproof.simulations.schema` defines
  the four objects every row projects from (`Task`, `Rollout`, `Judgment`,
  `Marker`) plus `Dataset` and `Calibration`, with `from_row` / `to_row`
  between them and the flat JSONL row. Every row the engine, `save()`,
  the exporters, and `rows_from_otel` write now carries
  `schema_version: "1"`; `.meta.json` carries it too. The wire contract is
  `zeroproof/simulations/schemas/row-v1.json`, shipped in the wheel.
- Rows without a stamp are version 0 and are read by shape: engine rows
  (by `scenario_id`), platform trace pulls (by `tool_trace`), OTel ingest
  (by `conversation_id`). Nothing that loaded before is rejected.
- Validators run at the boundaries: `push_rows`, `training_rows` /
  `export_training`, `export_preference`, `rows_from_otel`, and the row
  writer behind both the streamed file and `save()`. In this version they
  check the stamp and the required field types only.
- A test freezes the count of direct verdict-key writes per module, so
  new grading paths go through `attach` once it lands.
- `SimulationData.trajectories` stays the source of truth; the objects
  are a view until the store moves.

## 0.07 (2026-09-11)

- A bring-your-own run with no key fails at setup with a message naming
  `OPENAI_API_KEY`, instead of spending its whole time budget on 401s and
  returning zero rows. Loopback and plain-http endpoints (ollama, local
  vLLM) still need no key.
- A stop raises a flag that every rollout and writer wave checks before
  starting, so nothing begins work after the stop is declared. Closes a
  race where a wave marked running could still start its body after
  `simulate()` returned.
- The two stop tests give the hanging agent and writer six seconds and
  assert a four-second return, so a loaded CI box cannot trip them.

## 0.06 (2026-09-10)

- A stop also settles writer waves: queued waves are cancelled, running
  ones get the same `stop_grace`, and any still running are reported as
  `writer_waves_abandoned`. Found by an end-to-end run of the 0.5 wheel
  against hosted Qwen, where four writer threads outlived `simulate()`.

## 0.05 (2026-09-10)

- `simulate(reproducible=True)`: round-synchronous scheduling. Same seed,
  same concurrency, same agent gives the same rows at any concurrency.
  Costs throughput under uneven latency and needs the clock off.
- README leads with bring-your-own-model; hosted Qwen is the fallback.
- This changelog.

## 0.04 (2026-09-10)

- `zeroproof` is trace ingest only. The pre-0.3 encrypted agent-to-agent
  messaging client (`ZeroProof`, `send_encrypted`, reputation, approval
  workflows) is gone. Pin `zeroproof<0.3` to keep it.
- `simulate()` is a `run/` package with named phases (inputs, build, loop,
  finish). Behavior unchanged; verified row-for-row on a 17-configuration
  serial harness.
- A seeded serial run reproduces bit-for-bit across processes. Scenario ids
  and the axis-gap hint no longer depend on Python's per-process hash.
- A stop (clock, cap, saturation) cancels queued rollouts, waits up to
  `advanced["stop_grace"]` (5 s) for running ones, keeps what finishes, and
  reports the rest as `rollouts_abandoned`. Nothing calls the agent after
  `simulate()` returns.
- Progress goes through the `zeroproof.simulations` logger instead of
  `print()`.
- `zeroproof.simulations` ships `py.typed`.
- Trace ingest defaults to `https://api.zeroproofai.com`.
- README parameter tables match the code (`concurrency` 32, `time_budget`
  `None`) and a test keeps them matching.
- A failed tool draft for a prompt-only agent is a `tool_draft_unavailable`
  degraded note instead of a silent tool-free run.
- Two timing-dependent tests made deterministic. `zeroproof.__version__`
  reads package metadata.

## 0.3 (2026-08-31)

- The `zeroproof` package absorbed `zeroproof-simulations`, which is
  deprecated on PyPI. Releases of `zeroproof` before 0.3 were an unrelated
  encrypted messaging client.
