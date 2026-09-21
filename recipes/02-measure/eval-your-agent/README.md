# Evals for the agent you already have

Wrap the agent you ship, write its policy as a judge, roll every ask four
times, and read pass@1 with an interval per policy branch. Ends at a CI
gate, not a push. Two scripted refund bots are included, one careful and
one eager, so the eval visibly separates a good agent from a bad one
before you plug in your own.

What you will learn: which parts of your policy the asks you already send
never reach (`--gap`), the `agent(message) -> {steps, final_text}`
contract, why the judge reads the trajectory and not the prose, what a
hollow run looks like and how the SDK flags it, and how to turn a pass
rate into an exit code. You need nothing; this recipe is offline. Seconds.

## Run it

```bash
uv add whileai
cd recipes/02-measure/eval-your-agent
python run.py                              # both bots, the whole report
python run.py --agent careful --gate 0.9   # exit 1 under the floor, exit 2 when the run is hollow
python run.py --k 8 --seed 1               # more repeats, another draw
python run.py --gap                        # what the old suite never reaches
```

| flag | default | what it does |
|---|---|---|
| `--agent` | `all` | `careful`, `eager`, or both |
| `--k` | 4 | rollouts per ask |
| `--grid` | 4 | situations the writer adds on top of the seeds |
| `--limit` | all | first N seeds only, for a smoke run |
| `--gate` | off | pass@1 floor as an exit code |
| `--gap` | off | what the old `OLD_TESTS` suite never reaches, before the eval |
| `--json` | off | every number to one file |

`--k` stays at 4 or above: `pass^k` and `pass@k` are `None` below four
repeats (`min_k`), and the `note` field says so. Rows are the budget and
asks are the situations, so a run needs `budget >= situations * repeats`
or the later asks never get rolled out.

## Find what is untested

Before writing the eval: `wai.coverage_gap` takes the asks a suite
already sends and says which parts of the policy they never reach. The
axes are the ones `simulate` covers, so the answer comes back in the
engine's own words: which tool, which policy rule, what stance the person
takes, what the world looks like, what condition the tool is in.
`OLD_TESTS` in `run.py` is the three-ask suite this recipe replaces.

```bash
python run.py --gap
```

```
== find what is untested (the three asks the old suite sent)
3 asks cover 5 of 6 policy rules and 2 of 2 tools; untested: Refunds over $200 need a manager: do not issue them, say a...; no ask puts the agent under pressure; every ask runs once

asks                  3  (each one once)
policy rules covered  5 of 6
tools covered         2 of 2
stance                ordinary 3
untested rules
  - Refunds over $200 need a manager: do not issue them, say a manager wi...
! 1 policy rule no ask reaches: write one ask per rule, or let the engine write them (simulate(seeds=asks, ...) covers the rule axis)
! world_state and tool_condition are not readable from an ask: a prompt never says the record is missing or the tool timed out, so every ask sits on one point of those two axes. Run the asks through simulate(seeds=asks, tools=..., system_prompt=...) to vary them, or add a fixture case per branch
! rules are matched on the words an ask shares with the rule, so a branch only the fixture data selects (an amount, a date) reads as untested even when an ask lands on it: confirm with rows= from a run
! every ask appears once: one rollout cannot tell a flake from a failure. Roll each ask k times (repeats=k, repeat_policy='fixed') and read pass^k
! no ask is hurried, adversarial or a retry: the suite tests the agent on a good day only. Add pressure asks, or take them from the stance axis
```

The over-limit branch is the one to act on: no ask in the old suite names
an amount or a manager, so nothing tests it. The eval below confirms it
from the other side (`escalates_over_limit` fires on 0 rows until a seed
reaches it). Rule matching is word overlap, so a branch that only the
fixture data selects reads as untested even when an ask lands on it: pass
`rows=` from a graded run and the report also names the rules whose every
row ended in the same tool fault, which is a missing fixture, not a
missing ask.

```python
import whileai as wai

report = wai.simulations.coverage_gap(
    OLD_TESTS, tools=TOOLS, system_prompt=POLICY, rows=scored.rows
)
print(wai.simulations.format_coverage_gap(report))
```

Asks can also come from the file that holds them:
`wai.coverage_gap("tests/test_refunds.py", ...)` reads the string
literals that look like asks (passed to a call or sitting in a list,
over fifteen characters, with a space in them), and
`wai.coverage_gap("asks.jsonl", ...)` reads the `prompt` of every row.

## What you get

```
== careful: 48 rollouts over 12 asks, k=4
   pass@1 1.00   pass^k 1.00   pass@k 1.00

== eager: 48 rollouts over 12 asks, k=4
   pass@1 0.50   pass^k 0.50   pass@k 0.50

== by policy branch
  branch             asks  rows  pass@1        95% CI  pass^k
  eligible              2     8    1.00     1.00..1.00    1.00
  outside_window        2     8    0.00     0.00..0.00    0.00
  over_limit            2     8    0.00     0.00..0.00    0.00
  not_delivered         1     4    0.00           n/a    0.00
  unknown_order         1     4    1.00           n/a    1.00
  no_refund_asked       4    16    0.75     0.50..1.00    0.75

== by marker (1.0 = the agent did the right thing)
  looked_up_first               36  1.00
  refund_only_when_allowed      48  0.50
  refunds_when_eligible          8  1.00
  escalates_over_limit           8  0.00
  no_invented_order             48  1.00

== what failed, one per kind
  [refund_only_when_allowed] outside_window: 'Please refund A1004, the headphones were a gift I never used.'
      calls=['lookup_order(order_id=A1004)', 'issue_refund(order_id=A1004, amount=189.0)']
      why='refunded on a outside window ask'
```

The eager bot passes every ask that wants an eligible refund and fails
every ask that wants an ineligible one: the old three-assert test suite
(`"refund" in reply.lower()`) would have passed it. The branch table is
the number that matters; the overall pass@1 hides it.

## The four pieces

**The wrapper.** The engine hands your function one ask and wants the
tool calls it made and what it said. A callable agent runs its own real
tools and is played single-turn. `run.py` reads the raw rollouts as
`data.trajectories`; `data.rows` is the same rollouts exported, and both
`data.rows` and `data.rows()` work.

```python
def agent(message: str) -> dict:
    return {
        "steps": [{"tool": "lookup_order", "arguments": {...}, "result": {...}}],
        "final_text": "Done: $129.00 refunded for order A1001.",
    }
```

**The seeds.** One per policy branch. The writer varies wording and
stance; the order id keeps the branch. The order ids are also in the
`lookup_order` description, because a writer that does not know which
ids exist invents ones that do not, every rollout is "not found", and
the run is hollow.

**The judge.** The refund policy as a program (`refundable()`), shared
with the careful bot so it lives in one place. It reads `row["steps"]`:
which tools ran, in what order, with what. Markers are named so 1.0 is
always the good outcome; a marker that does not apply to a row is `None`
and its rate counts only the rows it measured.

**The gate.** `--gate 0.9` exits 1 under the floor and 2 when the SDK's
coverage warnings say the run is hollow (no rollout called a tool, a
declared tool no rollout touched, a marker that fired on no row). A
pass@1 from a hollow run is not reported.

## Swap in your agent

Keep `run.py`, replace `AGENTS`. For a model-backed bot that records
calls through a shared list, make the recorder thread-local: `concurrency`
defaults to 32, so 32 rollouts call your function at once and one shared
list mixes their calls together.

```python
import threading
import bot  # your bot: bot.answer(message) -> str, tools in bot._run_tool

_local = threading.local()
_real_run_tool = bot._run_tool


def _recording_run_tool(name, args):
    result = _real_run_tool(name, args)
    # _local.calls is set per rollout in my_agent below; a rollout that
    # reaches a tool without it is a wrapper bug, so let it raise.
    _local.calls.append({"tool": name, "arguments": args, "result": result})
    return result


bot._run_tool = _recording_run_tool


def my_agent(message: str) -> dict:
    _local.calls = []
    reply = bot.answer(message)
    return {"steps": list(_local.calls), "final_text": reply}


AGENTS = {"mine": my_agent}
```

Convert the bot's tool list to OpenAI function shape for `TOOLS` (a bare
`{"name", "description", "parameters"}` dict works too, and so does the
Anthropic `{"name", "description", "input_schema"}` shape), put your real
ids in the descriptions or seeds, and keep `refundable()` as the policy
your bot is supposed to follow. Then pass `simulator="hosted"` instead of
`simulator=False` to let the hosted writer produce more varied asks, and
raise `--k`. The hosted writer needs a key: `wai login`, or
`WHILEAI_API_KEY` in the environment. A hosted run is minutes, and
it says where it is on the `whileai.simulations` logger
(`12/64 rollouts, 3 situations written, 1m40s elapsed, ~5m left`) once
`logging.basicConfig(level=logging.INFO)` is on.

## Next

- Fast lane: a pytest file that runs `refund_judge` on eight hand-written
  rows, no model calls, one second. Judge edits cannot drift unnoticed.
- Judge trust: `wai.attach_labels(rows, labels, kind="human")` then
  `wai.judge_trust(rows, refund_judge)`. FAIL on eight labels means label
  more (about thirty clears the Wilson bound), not that the judge is wrong.
- Every failure is a training example: `wai.simulate(traces=scored.failures())`
  aims the next round at what broke (`scored.failed_traces()` and
  `scored.traces` are the same list under other names). `evaluate` rows are stamped so the
  selectors refuse to use them as the reward.
- `scored.push("refund-evals", purpose="eval")` keeps the set out of training on the platform.
