---
title: "Evals for the agent you already have"
sidebarTitle: "Evals"
description: "A pass rate with an interval, a table of where the agent fails, and a CI check that turns red when it gets worse. Offline, no key, seconds."
---

You have an agent. You want a number that says how often it does the job,
an interval on that number, a table of where it fails, and a check that
turns red in CI when it gets worse. This page is that path, start to
finish. The runnable version is
[`recipes/02-measure/eval-your-agent`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/02-measure/eval-your-agent)
(offline, no key, seconds).

Names first, because they cost testers ten minutes: **zp, ZeroProof and
While are the same product.** The package is `whileai`
(`uv add whileai`; `uv add zeroproof` still works as a shim),
the import is `whileai.simulations`, API keys start with `zp_`, and the
docs live at docs.withwhile.com.

## 1. Install and sign in

```bash
uv add whileai
whileai signup --email you@example.com   # new account, no browser; or: whileai login
whileai status                            # which key the SDK will use
```

Nothing below needs the key until you drop `simulator=False`. On a
machine that had the old package, `~/.zeroproof/credentials.json` is
still picked up as long as `~/.whileai` holds no credentials; set
`WHILEAI_HOME=/some/fresh/dir` to isolate a new account from it.

<Note>
**The old names still work, so you may already be signed in.**
`ZEROPROOF_API_KEY` is read whenever `WHILEAI_API_KEY` is unset (every
`ZEROPROOF_*` variable is), a saved `~/.zeroproof/credentials.json`
counts as a login, and `uv add zeroproof` installs `whileai`. If
`whileai status` names a key you never set here, that is where it came
from.
</Note>

**What the trial covers.** A fresh `signup` key is a trial: 25,000 input
and 50,000 output tokens a day, which is about twelve hosted situations
of a four-tool agent. One real run spends that, and the run then stops
with `Hosted model daily quota exceeded`. Two ways around it:
`simulate(..., simulator=False)` writes the situations offline with no
quota and no network, which is how every recipe here runs; and signing
in once at the While site (the link `whileai status` prints) lifts the
daily limit. `whileai status` prints the same two facts while the key is
on the trial, and a run that would spend the trial on the hosted writer
says them once before it starts, in the log and in `data.warnings`.

## 2. Wrap your agent

The engine calls your function once per rollout with the ask the writer
produced, and wants back the tool calls it made and what it said:

```python
def agent(message: str) -> dict:
    calls = []  # your bot runs here, with its real tools
    reply = my_bot.answer(message, record=calls)  # each call: {"tool", "arguments", "result"}
    return {"steps": calls, "final_text": reply}
```

Three things to know about a callable agent:

- **It runs its own real tools.** The engine's mock world and scheduled
  faults apply to model-backed agents; your callable answers its own
  calls. That is what you want for an eval of the thing you ship.
- **It is played single-turn.** One message in, one trajectory out. The
  multi-turn user model (`avg_turns`, `max_turns`) does not apply.
- **The writer does not know your ids.** It reads the tool descriptions
  and the system prompt. If your world has order numbers or account
  names, put them in the tool description ("Orders on file: A1001,
  A1002, ...") or in `seeds=`, or the writer invents ids, every rollout
  is "not found", and the run is hollow.

A tool is a typed function under `@wai.tool`: the signature is the
schema and the docstring the description, so nothing is written by hand.

```python
from whileai import tool


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id."""
    ...


TOOLS = [get_order]
```

Schema dicts still work in the same list: OpenAI function-calling shape
(`{"type": "function", "function": {"name", "description", "parameters"}}`),
the bare `{"name", "description", "parameters"}` dict, and the Anthropic
shape `{"name", "description", "input_schema"}` (`input_schema` is read as
`parameters`). If your bot records calls
through a shared global, wrap the recorder in a `threading.local`:
`concurrency` defaults to 32, so 32 threads call your function at once
and one shared list interleaves calls from different rollouts into each
other's rows.

Or run `whileai init-evals` in the project and edit the files it writes.
It reads your Python with `ast`, never imports it, picks the tool list,
the system prompt and the callable that answers a message, and writes
five files: `evals/agent.py` (this wrapper, with your tools converted to
OpenAI shape and your tool runner wrapped in the thread-local recorder),
`evals/judge.py`, `evals/run.py`, `evals/test_judge.py` and
`evals/README.md`, wired to each other. It prints what it picked, so a
wrong guess is one flag away: `--agent module:callable`,
`--tools module:NAME`, `--system-prompt module:NAME`. When it finds
nothing the files are still written, with every place that needs your
code marked TODO. It ends with the three commands to run next:
`python evals/run.py --gap`, `python evals/run.py`, and
`pytest evals/test_judge.py`.

## 3. Write the judge as a program

The judge reads the trajectory, not the prose. A polite reply that issued
a refund it should not have scores 0; a blunt one that followed the
policy scores 1. The policy lives in the judge, once:

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
`markers` (name to 0/1). Name markers so 1.0 is always the good outcome
(`refund_only_when_allowed`, not `refunded_wrongly`); the marker table
reads as one column then. A marker that does not apply to a row is
`None`, so its rate counts only the rows it measured. A verifier
(`wai.verify.*`) is a judge too, when the answer is checkable: each one
is a callable that takes a row and returns this same dict.

Any other key you return is kept under `row["judge_meta"]`, not on the row:
a judge that returns `failures` reads back as `row["judge_meta"]["failures"]`.

## 3b. Find what your tests miss

The suite you have sends a set of asks. Which parts of the policy do they
never reach?

```python
old_tests = [
    "I want a refund for order A1001, the shoes did not fit.",
    "What is the status of order A1001?",
    "Can you refund order Z9999?",
]
report = wai.coverage_gap(old_tests, tools=TOOLS, system_prompt=POLICY)
print(wai.format_coverage_gap(report))
```

The first line is the summary, then a table and one `!` note per gap:

```text
3 asks cover 4 of 5 policy rules and 2 of 2 tools; untested: Refunds over $200 need a manager.; no ask puts the agent under pressure; every ask runs once

asks                  3  (each one once)
policy rules covered  4 of 5
tools covered         2 of 2
stance                ordinary 3
untested rules
  - Refunds over $200 need a manager.
! 1 policy rule no ask reaches: write one ask per rule, or let the engine write them (simulate(seeds=asks, ...) covers the rule axis)
! every ask appears once: one rollout cannot tell a flake from a failure. Roll each ask k times (repeats=k, repeat_policy='fixed') and read pass^k
```

`asks` is a list of prompt strings, a list of rows with a `prompt` key, or
a path to a `.py` or `.jsonl` file holding either. From a `.py` file the
asks are the string literals that look like asks (passed to a call or in a
list, over fifteen characters, with a space): a heuristic, so read
`report["asks"]` before trusting the counts.

The axes are the ones `simulate` covers, so the report is in the engine's
own words: `untested_rules` are the policy clauses no ask reaches,
`untested_tools` the tools no ask names, `single_shot` says every ask runs
once (one rollout cannot tell a flake from a failure), and `notes` names
the fix for each. `world_state` and `tool_condition` are not readable from
an ask at all, which is the honest reason a hand-written suite misses
fault handling: a prompt never says the record is missing or the tool
timed out.

Rules are matched on the words an ask shares with the clause, so a branch
that only the fixture data selects (an amount, a date) reads as untested
even when an ask lands on it. Pass `rows=` from a graded run to check the
world side: a rule whose every row ended in the same tool fault is one the
asks reach but the fixtures never let happen, and the fix is a fixture
case, not another ask.

`preflight(tools, system_prompt)["rules"]` is the same rule axis on its
own, which is the list of policy branches the engine extracted from your
prompt. Both reports put every clause of the prompt on the axis
(`rule_cap=None`, the default): a 160-clause production policy is checked
as 160 rules, not the first 16. Pass `rule_cap=16` to keep the first
sixteen in document order; the report then carries `n_rules_total`,
`rules_truncated` and a line with the count it left off. The generation
grid itself keeps its cap (`RULE_AXIS_CAP_GRID`, 16, `ZP_RULE_CAP`
overrides) so the covering array stays bounded, and a run whose policy has
more clauses than that says so in `data.warnings`.

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

- `pass@1` is how often the agent does the job. `pass^k` is how often
  it did on every one of `k` tries: for anything that moves money, that
  is the number. `pass@k` minus `pass@1` is headroom for training.
- `repeat_policy="fixed"` asks for all repeats up front. The `mode="rl"`
  default, `"successive"`, stops early on unanimous asks, which is the
  right economy for training data and the wrong one for an eval.
- Slice by category: tag each row (`row["category"] = classify(prompt)`)
  and call `wai.pass_at` on each slice. The recipe prints that table.
- `simulator=False` is the offline template writer (no key, no network);
  `simulator="hosted"` is the default, the hosted writer, and means the
  same as leaving the argument out.
- `pass^k` and `pass@k` print as `n/a` (the fields are `None`) below
  `repeats=4` (`min_k`), because four tries is the smallest draw those
  numbers mean anything on; the line ends with
  `set repeats>=4 for pass^k and pass@k`, which is also `.note`.
- `situations` counts asks, `budget` counts rows. Keep
  `budget >= situations * repeats` or the run stops at the budget with
  the later situations never rolled out at all.
- **To measure a policy branch, pin the tool result.** A rule like
  "credits over $200 go to `escalate_to_human`" is only tested when the
  tool returns an amount over $200, and the situation writer invents the
  amount, so on an unforced run the branch is reached at random: a
  marker that never fires, or reads 1.000 because it never had a chance
  to fail. Pin it:
  `wai.local_model(..., result_shapes={"lookup_invoice": {"invoice_id": "INV-1000", "amount_usd": 900.0, "status": "open"}})`.
  Numbers move by up to about a third per call (`900.0` lands in
  roughly 600 to 1200, `90.0` in 60 to 120), so pick a value whose whole
  range sits on one side of the threshold, and run the same pinned
  `tasks=` once per side. Measured on a billing agent: 44 lookups over
  $200 and 0 under with the big shape, 48 under and 0 over with the
  small one, no leakage across 34 tasks x 4 rollouts.
- A served model that scaled to zero takes two to three minutes to
  answer its first request. `timeout=` is 300 s by default so that
  first pass lands; if a call still times out, `data.warnings` says so
  and names the fix (raise `timeout=`, or send one throwaway request
  first). A pass that comes back with fewer rows than the base arm is
  the dangerous case, since some tasks then sit at k=1 against the
  base's k=4; read `data.warnings` before `pass_at`.
- A long run says where it is on the `whileai.simulations` logger, one
  line at most ten seconds and ten rollouts apart
  (`12/64 rollouts, 3 situations written, 1m40s elapsed, ~5m left`);
  runs under ten rows say nothing. Call
  `logging.basicConfig(level=logging.INFO)` to see it; a hosted run can
  sit a minute before the first row, and silence is not a hang.

<Warning>
**Hollow runs.** If no rollout called a tool, a declared tool was never
touched, or a marker fired on no row, `scored.warnings` says so and names
the fix. A pass@1 of 1.00 on a run where the agent never reached its
tools is not a result. Do not report a number from a run with warnings.
</Warning>

## 5. Gate CI on it

Two lanes. The slow one runs the agent (model calls, seconds to minutes)
and exits non-zero under a floor:

```bash
python evals/run.py --gate 0.9          # exit 1 when pass@1 < 0.90, exit 2 when the run is hollow
```

The fast one runs the judge alone on hand-labeled transcripts, in a
second, with no model calls, so a judge edit cannot drift unnoticed:

```python
def test_judge_catches_refund_outside_window():
    row = {"prompt": "Refund A1004", "steps": [lookup(A1004), refund(A1004)], "final_text": "Done."}
    assert judge(row)["reward"] == 0.0
```

## 6. Check the judge

A judge is a claim until it is measured. Label a sample by hand, attach
the labels as human, and ask:

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
The key names the row: `rollout_id` when the row has one, else
`scenario_id#rollout_index` (what a `simulate` row carries), else the
row's `prompt` plus `final_text`. A `simulate` row has no `rollout_id`, which
is why the list form above names the two fields it does carry.

`judge_trust` reads `gold_reward` (which `attach_labels` writes) and
reports agreement with its Wilson lower bound (the bottom of the 95%
interval), held-out halves, a length bias check and re-judge flips.
**FAIL on eight labels means label more, not that the judge is wrong:**
at perfect agreement the lower bound needs sixteen labels to clear 0.8,
and the report line says how many to label. Labels attached any other
way count as model-made and keep `ok` false unless you say
`allow_model_gold=True`.

Two or more judges in the running (a decision model, a hosted model, a
frontier model, your own rules)? Compare them on the same labeled rows in
one call instead of running `judge_trust` once per judge. Each model judge
needs its provider's key, `TYPESAFE_API_KEY`, `WHILEAI_API_KEY` and
`ANTHROPIC_API_KEY` here; the rules judge needs none:

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

A judge is a spec string, a callable of your own, or a backend object. The
backend classes sit on the front door (`whileai.Hosted`, `whileai.OpenAI`,
`whileai.Anthropic`) rather than on `whileai.simulations`, which is why the
block imports both. Spec strings and backends run the package's
conduct-floor judge prompt under the run's system prompt and tools, so every
model reads the same evidence; a callable is used as given. A bare row list takes the same
call as `whileai.judge_comparison.compare_judges(rows, judges)`. The
floors are `judge_trust`'s (`floors=(0.8, 0.6)`); model-made gold keeps
every `ok` false unless `allow_model_gold=True`.

## 7. Return shapes

The names in the print and the names on the object are not always the
same word, and guessing costs a round trip. What each call hands back:

| call | you get | read it as |
| --- | --- | --- |
| `simulate(...)` | `SimulationData` | `data.rows` or `data.rows()`, both work |
| `evaluate(...)`, `grade(...)` | `ScoredData` | `scored.rows` is a list that also answers to `scored.rows()`; both work |
| | | `scored.warnings`: hollow-run notes, print them before the number |

Two concepts here have two spellings each. These docs use the left one;
the right one is the same thing under another name, and the recipe uses
it in places.

| use this | also works | the difference |
| --- | --- | --- |
| `data.rows` | `data.trajectories` | `rows` is the exported row, exactly what `save()` writes and what the judge sees. `trajectories` is the same rollouts before export, still carrying the `privileged` block, which is why `leak_report` reads them. |
| `scored.failures()` | `scored.failed_traces()`, `scored.traces` | Same list, all three. `failed_traces()` and the `traces` property are named for where they go next: `simulate(traces=...)`. |
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
| `ci95` | `[lo..hi]` | task-bootstrap interval on pass@1. `None` under three tasks, and `note` then says so and names the fix |
| `pass_pow_k_ci95`, `pass_at_k_ci95` | `[lo..hi]` | the same for the k-way numbers |
| `k`, `n_groups`, `n_rows` | `(N groups, k=4)` | draw size, tasks, graded rows |
| `n_groups_at_k`, `n_groups_imputed` | not printed | tasks the k-way numbers used |
| `per_task` | not printed | `{task key: pass rate}`, a dict, not a list |
| `note` | tail of the line | why a number is missing, and the fix |
| `config` | token-cap share | temperature, versions, prompt hash |

Marker stats (`marker_summary(rows)["grounded"]`): `metric`, `mean`,
`ci95` (not `ci`), `n_tasks`, `n_rows` (not `n`), `n_rows_at_1`,
`n_rows_at_0`, `degenerate`, and one of `note` (no interval, too few
tasks: the bootstrap needs three) or `warning` (the marker never varied).
`ci95` is `None` in both cases, and the sentence says which one you have.

`judge_trust(rows)`: `ok` (measured and clean), `agreement.agreement`
with `agreement.ci95`, `agreement.kappa` and `agreement.n`, `gold_kind`
(`"human"`, `"model"`, `"unknown"`), `n_labeled`, `held_out_halves`,
`length_sensitivity`, `perturbation`, `probes`, `disagreements`, and
`warnings`, where every line names its own fix.

## 8. Then

- Every failure row is a training example: `simulate(traces=scored.failures())`
  aims the next round at what broke. `evaluate` rows are stamped
  `lineage.source == "eval"`, and every selector report counts them as
  `eval_sourced` and warns before you train on them.
- Push the eval set with a purpose so it stays out of training:
  `scored.push("refund-evals", purpose="eval")`.
- Production traces are rows too: `wai.rows_from_otel(spans)` reads
  OpenTelemetry spans, and the same judge and markers score them, so the
  number you got here is the number you watch after shipping.
