# SDK findings from this recipe

Each entry lists what got in the way, where it happened, what I expected, what happened instead, and the fix or workaround. Line numbers are against `25b9fac9` (origin/main when this lane started).

## Silent or wrong behaviour

1. **`pass_at_1` drops a fractional reward without saying so.** `whileai/simulations/score/stats.py:853` (`_binary`) returns `None` for any reward that is not exactly 0 or 1, and `_by_task` then skips the row.
   - Expected: a warning, or a refusal to score, when a judge writes a fractional reward.
   - Happened: `compare_runs`, `pass_at` and `task_means` score a rubric judge's 0.6 as absent. The row disappears from the denominator with no note, which is the "rows vanish" failure in skills/strengthen-your-evals. `Judge(Rubric)` returns exactly this kind of reward: the mean of the criteria met.
   - Fix: count the dropped rows and warn, or route non-binary rewards to a `marker:`-style mean.
2. **A callable agent's `usage` is thrown away.** `whileai/simulations/run/engine.py:1282-1290` builds `row["usage"]` only from steps that carry integer `input_tokens`. A trajectory-level `"usage"` returned by a `Harness(agent=callable)` is dropped (checked: the row has no `usage`).
   - Knock-on: the Meta-Harness cost gate (`meta-harness/run.py`, `cost()`) then silently falls back to counting calls.
   - Workaround: the agent emits one step per model turn carrying `input_tokens`/`output_tokens` (`extract_harness.py`, `agent()`).
   - Fix: roll a trajectory's own `usage` onto the row.
3. **Task dict keys other than the known ones are dropped.** A `split` key on a `simulate(tasks=[...])` task never reaches the row. `scenario_dimensions` does survive, including through a save and a replay.
   - Workaround: the split lives in `out/split.json`, where the recipe already keeps it.
4. **`serve_modal.py` in text-to-sql hard-codes the app prefix.** `recipes/04-train/text-to-sql/serve_modal.py:44` sets `app = modal.App(f"t2s-serve-{SLUG}")` whatever the config `key` is.
   - Danger: deploying Nemotron from that script with a new key redeploys another person's `t2s-serve-*` app.
   - Workaround: this recipe ships its own `serve_modal.py` with the prefix `docx-serve-` and no adapter path.
   - Fix: take the app prefix from the config.
5. **The progress line's denominator is wrong on replayed tasks.** `wai.simulate(tasks=..., budget=len*k)` printed `180/1000 rollouts, 270 situations` for a 270-rollout run. Cosmetic.

## Missing, worked around here, candidates for upstream

The hooks in 6 to 9 live in this recipe's `loop.py`, a copy of `recipes/papers/meta-harness/run.py` at `25b9fac9`, and not in that recipe itself (another person owns it). They are proposed as upstream changes to it.

6. **`recipes/papers/meta-harness/run.py` cannot take a task or judge other than its own `common.py`.** Added `--task <module>`: `tasks()` plus `judge(row)`, fed into the production-tasks path. The loop is otherwise unchanged. When every task declares a `split`, `freeze_split` keeps it.
   - Why that matters: the recipe's 8-gram `decontaminate` would drop every templated train document, because they share their schema text by design.
7. **The recipe gated only on a binary pass@1.** Added `--metric marker:<name>`, threaded through `score`, `tasks_led`, `regressed`, `compare_runs` and `attribute(metric=)`. The default is unchanged.
8. **The proposal showed holdout scores to the proposer.** `propose()` printed `holdout {fmt(...)}` for every candidate, so the proposer could steer on the holdout. Added `--blind`, which leaves those scores out.
9. **`--prune` assumed the meta-harness candidate shape.** The pruned-file writer read `module.SCRIPTED_RATE` and wrote `from common import Edit, from_edits`, which crashes on any other recipe's candidate.
   - Added: for another recipe's candidate, the pruned file loads the pick by path and fixes the drop.
10. **The recipe scores every candidate on every model.** Its `evaluate` has no way to put the held-out models on only the baseline and the pick.
    - Workaround: `run.py gate` copies the baseline and the pick into their own folder and runs the same `run.py` over every model.
11. **`wai.compare_runs` is not on `wai`.** Only `wai.compare` is there (`delta_report`), which takes `target=` rather than `metric=`. The recipe imports `compare_runs` from `whileai.simulations.score.stats`.
12. **`wai.Harness(tools=["run_python"])` and `simulate` disagree about tools.** The `Harness` constructor takes a bare tool name, but `simulate` then refuses it: `whileai/simulations/tools.py:272`, "got str".
13. **Matched cost counts input tokens.** Under `--cost-margin 0`, any added instruction raises tokens per rollout, so a prompt-only rule can pass only if it also saves turns. The dry run showed a +14-point pick refused at 1.08x. That is the gate working as written (Wang et al. 2026), but it is worth saying in the recipe README.

## Platform

15. **The Runs page ignores ``?agent=``.** `https://while.ai/platform/runs?agent=doc-extraction`, the link shape the SDK prints (`wai init` and this recipe's `run.py report --post`: `.../runs?agent=<name>`), opened the IT-access-administration agent instead (Sahana's screenshot, 2026-09-25 14:30).
    - Expected: the page opens the agent the query names.
    - Happened: the query parameter is dropped and the page shows whichever agent it last showed, or the first.
    - Workaround: pick the agent from the page's own selector.
    - Fix: read `agent` from the query on load, or stop printing a link the page does not honour.

17. **The Runs page chart and Iterations table key off one behavior name.** Both read the behavior named `field_f1`; with it unscored (the holdout, blind until the gate) the chart plotted `field_f1_bank_statement_selection` and the table said "passes on field f1: not scored" for every run, and pointing every run's `targets` at the scored headline changed nothing (Sahana's screenshots, 14:49 and 14:55).
    - Expected: the chart and table follow the selected behavior, or the run's target.
    - Workaround: score the `field_f1` behavior for every candidate the moment the search stops (from stored holdout rows, before the gate; ledger.md notes the timing).
    - Fix: chart the selected behavior; say "not scored yet" instead of falling back to another behavior.

## Judge

14. **The Haiku 4.5 rubric judge failed "clean JSON" on a fenced object.** It failed a correct answer wrapped in a ```` ```json ```` fence (the format the baseline prompt asks for) and passed the same object after "Here: ".
    - Fixed in the rubric text before any live run: "A markdown ```json fence around the object is fine."
    - The audit against program gold (`out/audit.json`) is what would have caught it.
