# What the SDK could and could not do here

Recorded while building this recipe on 2026-09-25, against `whileai` 0.125.
Each item names the call, what was asked of it, what it did, and the change
that would close the gap. The four gaps the plan predicted are 1, 2, 4 and 5.

## 1. Coverage axes: attack family, carrier, obfuscation are not dimensions

`wai.simulate(dimensions=)` accepts `tool, rule, stance, world_state,
tool_condition, history` (`whileai/simulations/generate/scenarios.py`,
`COVERAGE_AXES`) and refuses any other axis. The axes a classifier's data
needs are attack family, carrier (user turn, tool JSON, email, retrieved
chunk, web page), obfuscation and framing. They rode on a program
(`data.py`) instead: the payloads come from the public sets, the carriers
and framings are drawn by seed, and the label is known by construction.
Fix: let `dimensions=` take a user-defined axis with a list of values, and
stamp the drawn value on the row so `select` and the report can slice on it.

## 2. Rows are trajectories; a classifier row is a span

`SimulationData.rows()` returns one trajectory per rollout. The classifier
trains on chunks of at most 512 tokens with a label per chunk. The
harvesting is fifteen lines in `data.sim_carrier_rows`: walk `messages`,
keep `role == "tool"` content between 80 and 2,500 characters, plant or do
not. Fix: `rows.chunks(role="tool", max_tokens=512)` that yields
`(chunk, row_id, turn_index, carrier)` so the label plumbing is the only
recipe-local part.

## 3. The simulated world's tool results are templated stubs

`wai.simulate` with `tools=` for `read_email`, `fetch_page`,
`retrieve_docs`, `get_product_reviews` returned tool payloads from the
world's `result_kinds` (`shell, grep, file, files, git, ci, money`):
`{"id": 21688, "name": "weekly new customer support protocol", "status":
"pending", "owner": "tessa okonkwo"}` for an inbox search, and the same
snippet ("Don't forget the meeting at 2 PM today.") on every hit. No email
body, no page text, no document passage. The 91 `sim_tool` rows in the
frozen test are these stubs, which is why that slice is reported apart and
why the recipe's carriers are written by program. The world has an
`injected` counter per tool in `report()["tools"]` but nothing set it in
this run. Fix: a `world="llm"` or `tool_results=` hook so a tool result can
be a document the user model writes, and expose whatever `injected` counts.

## 4. No classifier metrics

`pass_at` and `wai.compare` are per-prompt pass rates with a bootstrap over
prompts. A classifier needs AUROC, recall at a fixed FPR, precision, recall,
F1 and FPR at a threshold, per slice, with an interval over rows.
`metrics.py` here is 120 lines; nothing in it is specific to this recipe.
Fix: `wai.classify_report(rows, scores, threshold=)` returning a `Report`
that prints itself, the way `delta_report` does.

## 5. No sequence-classification trainer

`wai.train` is `sft | grpo | dpo | rm` on decoder LMs
(`whileai/simulations/training.py`). `rm` is the nearest shape (a scalar
head on an encoder) but it trains on preference pairs, not labelled
chunks. `train_modal.py` is recipe-local `AutoModelForSequenceClassification`
on one L40S: 19 seconds a seed on 9,183 rows. Fix: `wai.train(rows,
method="classify", base="nreimers/MiniLM-L6-H384-uncased")` reading
`label` off the row, with the same run record `sft` writes.

## 6. `wai.methods.route` on the baseline's graded predictions

Run with each test row as a task, reward 1 when ProtectAI v2 classified it
correctly at its shipped threshold, k=1. It returned `sft` ("1173 passing
rows to clone (a pass filter at k=1)"), blocked grpo and dpo ("no task has
two rollouts"), and asked for k=8 rollouts per task. The pick is right for
the wrong reason: supervised training on labelled rows is what a classifier
does, and `route` reached it because a deterministic classifier has no
groups, not because it knows the label is a program's. What it could not
say: that "clone the passes" is wrong here (the passes are the baseline's
correct calls, a classifier trains on every labelled row, misses included),
that k>1 is meaningless for a deterministic scorer, and that the 1,173
"passing rows" is 55% of the pool, which is the baseline's accuracy and not
a training signal. Its "before the GPU" list (size the holdout, three-run
noise floor, decontaminate, random-selection control) applies as written;
the noise floor here is three training seeds, not three eval runs. Fix:
`route` should read `label_source="program"` plus a `label` column as
"supervised rows, no rollouts needed" and route to a classification
trainer rather than to SFT-as-cloning, or say that the rows are not
rollouts.

## 7. The fix for 3: `execute=` plus a model as `simulator=` gives real documents

`wai.simulate(execute=fn, simulator="anthropic:claude-haiku-4-5",
user_model="anthropic:claude-haiku-4-5")` with an `execute` that asks a model
to write the tool's result for the call it received returns tool results that
are documents: a patient message with headers and a body, a lab report, a
transaction list with merchant descriptions, a CI log. The 384 rows of round 7
came from six businesses this way, at about a cent a rollout. Two things to
know: `current_rollout` (the thread-local the docstring names) is an object,
not a callable, so read `current_rollout.seed`; and `execute`'s return rides
inside `{"status": "ok", "result": ...}` in the row, so a harvester unwraps
it. The hosted situation writer has a daily quota (500k input tokens); it was
hit twice on this recipe, both times mid-run, and the run dies with 0 rows
rather than falling back. `simulator=` as a model spec is the way around it,
and `simulator=False` (the offline template writer) is not: templated
situations are the thing the classifier must not learn.

## 8. Small things

- `simulate` warned that `hard_share=0.5` drew 0.39 and named the fix
  (`dimensions={"stance": [...]}`). Good. It also warned that the agent
  model played the user; `user_model=` was not set, by choice, to keep the
  run under a dollar.
- One rollout of 80 was lost to an empty reply; the warning named
  `agent_max_tokens=`. Good.
- The first six-row probe took 301 s; the 80-row run took under ten
  minutes. `report()` prints every knob (about 80 of them), which buries
  `delivered` and `tools`; a one-screen summary with the knobs behind a
  flag would read better.
- `whileai/methods.py` exports `route` one dot down as `wai.methods.route`;
  `wai.route` does not resolve, which the style page says it should.
- `wai.methods.route` on the round-3 classifier rows said "nothing trains:
  126 of 1643 passing rows did not end on their own". The rows carry
  `finish_reason="stop"`; the truncation check read something else off them
  (`sdk_measure.py`). A classifier row has no completion to truncate.
- `hack_scan` took the matched pairs as asks (pair id as `prompt`, the label
  as `reward`) and ranked `contains:my`, `contains:please`, `n:punct` above
  its permutation floor: the surface features that separate an instruction
  from a statement. That is the shortcut detector the plan asked for, and it
  worked as-is on classifier rows once the pair was the ask.
- Posting: `Example.tags` is a dict, not a list (pydantic refuses a list with
  a message that names the field, so this cost one minute). A run re-posted
  under the same version is a second run; archive the first
  (`tracked.archive(run_id)`) or the page shows both.
