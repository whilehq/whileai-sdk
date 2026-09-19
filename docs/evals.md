---
title: "Evals for the agent you already have"
sidebarTitle: "Evals"
description: "A pass rate with an interval, a table of where the agent fails, and a CI check that turns red when it gets worse. Offline, no key, seconds."
---

You have an agent. You want a pass rate with an interval, a table of where
it fails, and a check that turns red in CI when it gets worse. The runnable
version is
[`recipes/02-measure/eval-your-agent`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/eval-your-agent)
(offline, no key, seconds).

<img className="block dark:hidden" src="/figures/evals-loop-light.svg" alt="Your callable, simulate with fixed repeats, evaluate with your judge, pass_at, a CI gate; hand labels feed judge_trust" />
<img className="hidden dark:block" src="/figures/evals-loop-dark.svg" alt="Your callable, simulate with fixed repeats, evaluate with your judge, pass_at, a CI gate; hand labels feed judge_trust" />

The package is `whileai`, the import is `whileai.simulations`, keys start
with `zp_`.

## 1. Install and sign in

```bash
uv add whileai
whileai signup --email you@example.com   # new account, no browser; or: whileai login
whileai status                            # which key the SDK will use
```

Nothing below needs the key until you drop `simulator=False`. `WHILEAI_API_KEY`
in the environment or the login saved at `~/.whileai/credentials.json` both
count; set `WHILEAI_HOME=/some/fresh/dir` to isolate a new account.

**The trial.** A fresh `signup` key gets 25,000 input and 50,000 output
tokens a day, about twelve hosted situations of a four-tool agent; a run
past that stops with `Hosted model daily quota exceeded`. `simulate(...,
simulator=False)` writes situations offline with no quota, and one sign-in
at the While site (the link `whileai status` prints) lifts the limit.

## 2. Wrap your agent

The engine calls your function once per rollout with the ask, and wants
back the tool calls and what it said:

```python
def agent(message: str) -> dict:
    calls = []  # your bot runs here, with its real tools
    reply = my_bot.answer(message, record=calls)  # each call: {"tool", "arguments", "result"}
    return {"steps": calls, "final_text": reply}
```

- **It runs its own real tools.** The mock world and scheduled faults apply
  to model-backed agents; your callable answers its own calls.
- **It is played single-turn.** `avg_turns` and `max_turns` do not apply.
- **The writer does not know your ids.** It reads the tool descriptions and
  the system prompt. Put your order numbers in the tool description
  ("Orders on file: A1001, A1002, ...") or in `seeds=`, or every rollout is
  "not found" and the run is hollow.

A tool is a typed function under `@wai.tool`: the signature is the schema,
the docstring the description.

```python
from whileai import tool


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id."""
    ...


TOOLS = [get_order]
```

Schema dicts work in the same list: OpenAI function-calling shape, the bare
`{"name", "description", "parameters"}` dict, and the Anthropic shape with
`input_schema`. If your bot records calls through a shared global, wrap the
recorder in a `threading.local`: `concurrency` defaults to 32 threads, and
one shared list interleaves rollouts.

Or run `whileai init-evals`. It reads your Python with `ast`, never imports
it, picks the tool list, the system prompt and the callable, and writes
`evals/agent.py`, `evals/judge.py`, `evals/run.py`, `evals/test_judge.py`
and `evals/README.md`, wired together. A wrong guess is one flag away:
`--agent module:callable`, `--tools module:NAME`,
`--system-prompt module:NAME`. It ends with the three commands to run next.

## 3. Write the judge as a program

The judge reads the trajectory, not the prose. A polite reply that issued a
refund it should not have scores 0. The policy lives in the judge, once:

```python
def judge(row: dict) -> dict:
    steps = row.get("steps") or []
    refunds = [s for s in steps if s.get("tool") == "issue_refund"]
    allowed = refundable(order_named_in(row["prompt"]))  # the policy, as a program
    ok = bool(refunds) == allowed
    return {
        "reward": 1.0 if ok else 0.0,
        "reason": "refunded" if refunds else "no refund",
        "markers": {"refund_only_when_allowed": 1.0 if ok else 0.0},
    }
```

The contract is `reward` in [0, 1], a `reason` string, and optional
`markers` (name to 0/1). Name markers so 1.0 is the good outcome
(`refund_only_when_allowed`, not `refunded_wrongly`). A marker that does
not apply to a row is `None`. A verifier (`wai.verify.*`) is a judge too.
Any other key you return lands under `row["judge_meta"]`.

## 3b. Find what your tests miss

Which parts of the policy does your suite never reach?

```python
old_tests = [
    "I want a refund for order A1001, the shoes did not fit.",
    "What is the status of order A1001?",
    "Can you refund order Z9999?",
]
report = wai.coverage_gap(old_tests, tools=TOOLS, system_prompt=POLICY)
print(wai.format_coverage_gap(report))
```

```text
3 asks cover 6 of 7 policy rules and 1 of 1 tool; untested: Refunds over $200 need a manager: say so instead of refunding.; no ask puts the agent under pressure; every ask runs once

asks                  3  (each one once)
policy rules covered  6 of 7
tools covered         1 of 1
stance                ordinary 3
untested rules
  - Refunds over $200 need a manager: say so instead of refunding.
! 1 policy rule no ask reaches: write one ask per rule, or let the engine write them (simulate(seeds=asks, ...) covers the rule axis)
! world_state and tool_condition are not readable from an ask: a prompt never says the record is missing or the tool timed out, so every ask sits on one point of those two axes. Run the asks through simulate(seeds=asks, tools=..., system_prompt=...) to vary them, or add a fixture case per branch
! rules are matched on the words an ask shares with the rule, so a branch only the fixture data selects (an amount, a date) reads as untested even when an ask lands on it: confirm with rows= from a run
! every ask appears once: one rollout cannot tell a flake from a failure. Roll each ask k times (repeats=k, repeat_policy='fixed') and read pass^k
! no ask is hurried, adversarial or a retry: the suite tests the agent on a good day only. Add pressure asks, or take them from the stance axis
```

`asks` is a list of prompt strings, rows with a `prompt` key, or a path to a
`.py` or `.jsonl` file. From a `.py` file the asks are the string literals
that look like asks, a heuristic, so read `report["asks"]` first.

The report is in the engine's words: `untested_rules`, `untested_tools`,
`single_shot`, and `notes` naming each fix. `world_state` and
`tool_condition` are not readable from an ask, which is why a hand-written
suite misses fault handling. Rules match on shared words, so a branch only
the fixture data selects (an amount) reads as untested; pass `rows=` from a
graded run to check the world side.

`preflight(tools, system_prompt)["rules"]` is the rule axis alone. Both put
every clause on the axis (`rule_cap=None`); `rule_cap=16` keeps the first
sixteen. The generation grid keeps its own cap (`RULE_AXIS_CAP_GRID`, 16)
and a run with more clauses says so in `data.warnings`.

## 4. Run it

```python
import whileai.simulations as wai

data = wai.simulate(
    agent,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=SEEDS,
    simulator=False,  # offline template writer: no key, seconds. simulator="hosted" for the hosted one.
    mode="rl",
    repeats=4,
    repeat_policy="fixed",  # every ask, all four repeats
    reproducible=True,
)
scored = wai.evaluate(data, judge)  # stamped as eval: never the reward
print(wai.pass_at(scored.rows))  # pass@1 [lo..hi] | pass^4 | pass@4 | headroom (N groups, k=4)
for note in scored.warnings:  # hollow-run checks; fix before reading the number
    print("!", note)
```

<img className="block dark:hidden" src="/figures/evals-pass-at-k-light.svg" alt="Five tasks by four tries: pass@1 is the mean per-task rate, pass^4 the tasks that passed every try, pass@4 the tasks that passed once, headroom the difference" />
<img className="hidden dark:block" src="/figures/evals-pass-at-k-dark.svg" alt="Five tasks by four tries: pass@1 is the mean per-task rate, pass^4 the tasks that passed every try, pass@4 the tasks that passed once, headroom the difference" />

- `pass@1` is how often the agent does the job. `pass^k` is how often it
  did on every one of `k` tries: for anything that moves money, that is
  the number. `pass@k` minus `pass@1` is headroom for training.
- `repeat_policy="fixed"` asks for all repeats up front. The `mode="rl"`
  default, `"successive"`, stops early on unanimous asks: right for
  training data, wrong for an eval.
- `pass^k` and `pass@k` print as `n/a` below `repeats=4` (`min_k`); the
  line ends with the fix, which is also `.note`.
- Slice by category: tag each row (`row["category"] = classify(prompt)`)
  and call `wai.pass_at` per slice.
- `situations` counts asks, `budget` counts rows. Keep
  `budget >= situations * repeats` or later situations never roll out.
- **To measure a policy branch, pin the tool result.** "Credits over $200
  go to `escalate_to_human`" is only tested when the tool returns over
  $200, and the writer invents the amount. Pin it:
  `wai.local_model(..., result_shapes={"lookup_invoice": {"invoice_id": "INV-1000", "amount_usd": 900.0, "status": "open"}})`.
  Numbers move by up to about a third per call, so pick a value whose
  range sits on one side of the threshold and run the same pinned `tasks=`
  once per side.
- A served model that scaled to zero takes two to three minutes to answer
  first; `timeout=` is 300 s. A pass with fewer rows than the base arm is
  the dangerous case (some tasks at k=1 against k=4); read `data.warnings`
  before `pass_at`.
- Progress goes to the `whileai.simulations` logger
  (`12/64 rollouts, 3 situations written, 1m40s elapsed, ~5m left`);
  `logging.basicConfig(level=logging.INFO)` shows it. Silence for a minute
  on a hosted run is not a hang.

<Warning>
**Hollow runs.** If no rollout called a tool, a declared tool was never
touched, or a marker fired on no row, `scored.warnings` says so and names
the fix. A pass@1 of 1.00 on a run where the agent never reached its tools
is not a result.
</Warning>

## 5. Gate CI on it

Two lanes. The slow one runs the agent and exits non-zero under a floor:

```bash
python evals/run.py --gate 0.9          # exit 1 when pass@1 < 0.90, exit 2 when the run is hollow
```

The fast one runs the judge alone on hand-labeled transcripts, no model
calls, so a judge edit cannot drift unnoticed:

```python
def test_judge_catches_refund_outside_window():
    row = {"prompt": "Refund A1004", "steps": [lookup(A1004), refund(A1004)], "final_text": "Done."}
    assert judge(row)["reward"] == 0.0
```

## 6. Check the judge

A judge is a claim until it is measured. Label a sample by hand, attach the
labels as human, and ask:

```python
labels = [
    # your verdict on each row you read, 0 or 1
    {"scenario_id": r["scenario_id"], "rollout_index": r["rollout_index"], "label": hand_label(r)}
    for r in scored.rows[:30]
]
wai.attach_labels(scored.rows, labels, kind="human")
print(wai.judge_trust(scored.rows, judge))  # the report prints itself
```

`labels` is a list of dicts as above, a `{key: 0/1}` dict, or a JSONL path.
The key is `rollout_id` when the row has one, else
`scenario_id#rollout_index` (what a `simulate` row carries).

`judge_trust` reads `gold_reward` and reports agreement with its Wilson
lower bound, held-out halves, a length bias check and re-judge flips.
**FAIL on eight labels means label more:** at perfect agreement the lower
bound needs sixteen labels to clear 0.8, and the report says how many.
Labels attached any other way count as model-made and keep `ok` false
unless `allow_model_gold=True`.

Two or more judges in the running? Compare them on the same labeled rows in
one call. Each model judge needs its provider's key (`TYPESAFE_API_KEY`,
`WHILEAI_API_KEY`, `ANTHROPIC_API_KEY` here); the rules judge needs none:

```python
import whileai  # the backend classes are on whileai, not whileai.simulations

table = scored.compare_judges(
    {
        "jev": "typesafe:jev-latest",
        "phi-4": whileai.Hosted(),
        "haiku": whileai.Anthropic("claude-haiku-4-5"),
        "rules": my_verifier,
    },
)
print(table)  # ranked by kappa; agreement [95% CI], leak, unsure, s/row
table.best.name  # the first judge that clears the floors, else the top one
table["jev"].rows  # that judge's graded copies, for reading the disagreements
```

A judge is a spec string, a callable, or a backend object (`whileai.Hosted`,
`whileai.OpenAI`, `whileai.Anthropic`). Spec strings and backends run the
package's conduct-floor judge prompt under the run's system prompt and
tools, so every model reads the same evidence. The floors are
`judge_trust`'s (`floors=(0.8, 0.6)`).

## 7. Return shapes

| call | you get | read it as |
| --- | --- | --- |
| `simulate(...)` | `SimulationData` | `data.rows` or `data.rows()`, both work |
| `evaluate(...)`, `grade(...)` | `ScoredData` | `scored.rows` or `scored.rows()`; `scored.warnings` holds hollow-run notes |

| use this | also works | the difference |
| --- | --- | --- |
| `data.rows` | `data.trajectories` | `rows` is the exported row, what `save()` writes and the judge sees; `trajectories` still carries the `privileged` block, which is why `leak_report` reads it |
| `scored.failures()` | `scored.failed_traces()`, `scored.traces` | same list; the other two are named for where they go next: `simulate(traces=...)` |
| `pass_at(rows)` | `PassAt` | fields below; `.to_dict()` for the same keys as JSON |
| `marker_summary(rows)` | `{marker: stats}` | stats keys below |
| `judge_trust(rows)` | `dict` | keys below |

`PassAt` fields, with the name each prints as:

| field | prints as | what it is |
| --- | --- | --- |
| `pass_at_1` | `pass@1` | mean per-task pass rate, the headline |
| `pass_pow_k` | `pass^k (pass_pow_k)` | all k repeats pass. Not `pass_hat_k` |
| `pass_at_k` | `pass@k` | at least one of k passes |
| `headroom` | `headroom` | `pass_at_k - pass_at_1`, a property |
| `ci95` | `[lo..hi]` | task-bootstrap interval on pass@1. `None` under three tasks, and `note` says so |
| `pass_pow_k_ci95`, `pass_at_k_ci95` | `[lo..hi]` | the same for the k-way numbers |
| `k`, `n_groups`, `n_rows` | `(N groups, k=4)` | draw size, tasks, graded rows |
| `n_groups_at_k`, `n_groups_imputed` | not printed | tasks the k-way numbers used |
| `per_task` | not printed | `{task key: pass rate}`, a dict |
| `note` | tail of the line | why a number is missing, and the fix |
| `config` | token-cap share | temperature, versions, prompt hash |

Marker stats (`marker_summary(rows)["grounded"]`): `mean`, `ci95` (not
`ci`), `n_tasks`, `n_rows` (not `n`), `n_rows_at_1`, `n_rows_at_0`,
`degenerate`, and `note` (too few tasks) or `warning` (never varied).

`judge_trust(rows)`: `ok`, `agreement.{agreement, ci95, kappa, n}`,
`gold_kind` (`"human"`, `"model"`, `"unknown"`), `n_labeled`,
`held_out_halves`, `length_sensitivity`, `perturbation`, `probes`,
`disagreements`, and `warnings`, where every line names its fix.

## 8. Then

- Every failure row is a training example:
  `simulate(traces=scored.failures())` aims the next round at what broke.
  `evaluate` rows are stamped `lineage.source == "eval"`, and every selector
  counts them as `eval_sourced` and warns before you train on them.
- Push the eval set with a purpose so it stays out of training:
  `scored.push("refund-evals", purpose="eval")`.
- Production traces are rows too: `wai.rows_from_otel(spans)` reads
  OpenTelemetry spans, and the same judge and markers score them.
