# Bring your own agent

Three things a first run with your own agent needs: the callable contract,
what the run says when the agent is broken, and how an eval score is kept
out of the training reward. Offline, no key, seconds.

```bash
uv add whileai
python run.py                 # all three parts
python run.py --part broken   # just the failure report
```

## The contract

The engine calls your function once per rollout with the situation the
simulator wrote, and expects the tool calls it made and what it said:

```python
import whileai as wai


def my_agent(message: str) -> dict:
    return {
        "steps": [
            {
                "tool": "get_runbook",
                "arguments": {"service": "checkout-api"},
                "result": {"status": "ok", "restart_allowed": True},
            },
            {
                "tool": "restart_service",
                "arguments": {"service": "checkout-api"},
                "result": {"status": "ok"},
            },
        ],
        "final_text": "Restarted checkout-api.",
    }


data = wai.simulate(
    my_agent,
    tools=TOOLS,
    system_prompt=POLICY,
    simulator=False,
    budget=16,
    mode="rl",
    situations=4,
    repeats=4,
    repeat_policy="fixed",
)
```

`simulator=False` uses the built-in template writer so no model key is
needed; the situations are less varied than a model writes, which is fine
for wiring up an agent and a judge. Any extra keys on the dict stay on the
row. Inside the callable, `current_rollout.rollout_index` says which repeat
this is, if the agent needs to know.

`repeat_policy="fixed"` asks for all four repeats up front. The `mode="rl"`
default is `"successive"`: two probe rollouts per ask, and the remaining
repeats only on asks whose graded probes disagree. That is the right
economy for a graded run, but an ungraded one never splits, so it would
stop at two repeats per ask and the example's every-third-repeat agent
would never misbehave.

## When the agent is the problem

A callable that raises, or returns an OpenAI-style message instead of the
contract, used to produce a zero-row run that blamed the situation writer.
Now the run says so:

```
raises       rows=0 stopped_because='agent_failed' agent_errors=16
             first error: RuntimeError: model endpoint returned 502
wrong shape  rows=0 stopped_because='agent_failed' agent_errors=18
             first error: TypeError: agent returned keys ['content', 'role'] instead of steps and final_text
```

The error count is the number of rollouts the engine tried before giving
up, a little over `budget=8`; it moves by a few between runs.

`data.stopped_because == "agent_failed"` when no row survived,
`data.search["agent_errors"]` is the count, `data.search["first_agent_error"]`
carries the exception type and message, and `"agent_errors"` lands in
`data.degraded` whenever any rollout was lost this way, even on a run that
still produced rows.

## An eval score is not a reward

`run_judge(rows, judge)` and `evaluate(rows, judge)` produce the same row
shape and the same numbers. The difference is provenance: `evaluate` stamps
`lineage.source == "eval"` because it is meant for a held-out set. Training
on those rows makes the scorer you report the reward you optimised against.

```
pass@1 on both: 0.75
run_judge: selected=8 eval_sourced=0  clean
evaluate : selected=8 eval_sourced=8  8 selected row(s) carry rewards from evaluate() (lineage.source == 'eval')
```

`select_for_rl`, `select_for_sft` and `build_preference_pairs` each report
`eval_sourced` and warn when it is non-zero. Nothing is dropped; the count
is the guard. Grade the training set with `run_judge` (or
`data.grade(judge=...)`), and keep `evaluate` for the rows the model never
sees.
