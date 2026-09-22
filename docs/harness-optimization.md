---
title: "Harness optimization from production traces"
sidebarTitle: "Harness optimization"
description: "A coding agent improves the prompt, the tools and the loop around a closed model on your own traffic. The gain is proven on days it never saw, on a second model, at the same cost."
---

**What you learn:** how a coding agent runs a harness search on yesterday's traffic, what it reads, what it writes, and the three checks a change has to pass. **Needs:** nothing for the dry run. **Takes:** seconds offline, minutes per candidate with a key.

<img className="block dark:hidden" src="/figures/harness-loop-light.svg" alt="Traces split by day, the agent writes one candidate from the worst rows, the gate checks the holdout, a second model and cost, the pick serves, and the next day is scored with the same judge" />
<img className="hidden dark:block" src="/figures/harness-loop-dark.svg" alt="Traces split by day, the agent writes one candidate from the worst rows, the gate checks the holdout, a second model and cost, the pick serves, and the next day is scored with the same judge" />

On a closed model the harness is the only thing you can change: the
instructions, the tools, the turn cap, the retry, how context is built.
Among comparable frontier models it explains more of the spread than the
model does [1], and a harness searched with earlier candidates' code and
traces in view beat the hand-built ones on TerminalBench-2 [2]. The catch:
tuned on the tasks it was scored on, harness evolution gained 0.6 points
held out and lost to more samples of the baseline at the same budget [3].
The loop below is built around what the agent is not allowed to see.

Runnable version:
[`recipes/papers/meta-harness`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/meta-harness).
Playbook for the coding agent:
[`skills/harness-search`](https://github.com/whilehq/whileai-sdk/tree/main/skills/harness-search).
The object it searches over: [the harness](/reference/harness).

Both the recipe and the skill's `check.py` run out of a clone of
[whilehq/whileai-sdk](https://github.com/whilehq/whileai-sdk): `recipes/` is
not in the wheel, so the copy `wai init --skill harness-search` installs under
`.claude/skills/` names the clone command and stops rather than running.

## 1. Traces in

Point the recipe at yesterday's export: a JSONL in any shape `load_traces`
reads, or an OTLP JSON batch, which `rows_from_otel` groups into
conversations. One task per distinct ask. Nothing leaves your machine.

```bash
cd recipes/papers/meta-harness
python run.py --traces traces-2026-09-21.jsonl --propose --select --fresh
```

Rows carry the ask, the steps, the reply and a timestamp. Extra keys pass
through.

```python
from whileai.simulations import load_traces

rows = load_traces(
    [
        {
            "ts": "2026-09-19T09:00:00Z",
            "input": "Where is my refund for ORD-5412?",
            "output": "Certainly! It went through.",
        },
        {
            "ts": "2026-09-21T09:00:00Z",
            "input": "Refund ORD-8821, the jacket arrived torn.",
            "output": "Done.",
        },
    ]
)
print(len(rows), sorted(rows[0]))
```

```text
2 ['final_text', 'input', 'output', 'prompt', 'steps', 'ts']
```

## 2. The split the agent cannot cross

The holdout is the latest days, whole days, until it holds half the tasks.
The proposer reads earlier days only. Train asks that are near-copies of a
holdout ask (`decontaminate`, the 8-gram rule) leave its window too.
`out/split.json` records it, and the proposal opens by saying which days
decide.

```text
split: holdout is 2026-09-20 and later: 16 of 24 tasks; 1 train prompt(s) overlapped the holdout and left the proposer's window
```

## 3. What the agent reads

`out/proposal.md`: every candidate so far as code, its pass@1 on the train
days with an interval over tasks, and its five worst rows. Nothing is
summarized. The proposer's edge in [2] was raw traces, not digests.

```text
## 00_baseline.py

train pass@1 0.71 [0.52..0.85] on 8 tasks; holdout 0.67 [0.50..0.83]; fingerprint 71fed4102190

Worst rows:

- ask: I want my money back on ORD-5412 right now.
  reply: Certainly! Your refund went through.
  why: boilerplate in the reply; claimed success on a tool result that did not succeed
```

## 4. What the agent writes

One file, one change, chosen from the worst rows. A candidate defines
`harness(model) -> wai.Harness`. The fingerprint hashes the instructions,
the tool names and the `Disclosure` fields, so the edit is a new version
without anyone naming it, and every row says which harness made it.

```python
import whileai as wai
from whileai.harness import Disclosure

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]
BASE = "Look the order up before you act."
scripted = wai.seeded_agent(TOOLS, rate=0.3, seed=0)  # stands in for the model offline

baseline = wai.Harness(
    agent=scripted, instructions=BASE, tools=TOOLS, label="baseline", model="scripted"
)
candidate = wai.Harness(
    agent=scripted,
    instructions=BASE + " Report what the tool returned; never claim success on an error.",
    tools=TOOLS,
    label="report_fault",
    model="scripted",
    disclosure=Disclosure(max_turns=4),
)
print(
    baseline.version,
    baseline.fingerprint,
    candidate.version,
    candidate.fingerprint,
    baseline.fingerprint != candidate.fingerprint,
)
```

```text
baseline 71fed4102190 report_fault 93c49a7ba181 True
```

Then it runs the recipe again and reads `out/selected.json`.

## 5. The gate

Three checks, all printed.

- **The pick.** The candidate that leads the most train tasks, per task the
  best pass rate across candidates with ties shared. Not the best mean: a
  mean gained by regressing a subset does not win [4].
- **Is it real.** On the held-out days the pick beats the baseline with a
  paired interval that excludes zero, and again on a second model. The
  held-out tasks the baseline passed every time and the pick failed every
  time are counted beside it.
- **Did it just spend more.** Cost per rollout is on the ledger: tokens when
  every row carries usage, model calls otherwise. A pick may cost no more
  than the baseline (`--cost-margin 0`). A candidate that wins by spending
  more is a frontier point, and the verdict names the margin that would
  accept it [3].

```text
train tasks led: 00_baseline.py 5, 01_no_filler.py 7, 02_check_result.py 12 -> pick 02_check_result.py
holdout on scripted: 02_check_result.py vs 00_baseline.py +0.27 [+0.15, +0.42] over 12 paired tasks, 0 the baseline passed and the pick failed -> clears zero
holdout on scripted-b: 02_check_result.py vs 00_baseline.py +0.50 [+0.38, +0.62] over 12 paired tasks, 0 the baseline passed and the pick failed -> clears zero
cost per rollout: 02_check_result.py at 1.00x the baseline in calls -> within the margin
select: 02_check_result.py beats the baseline on the holdout and on a held-out model at 1.00x its cost
```

`wai.harness.attribute` over the candidate by model grid says whether the
spread is the harness or the model. The agent loops on steps 3 to 5 until
the gate passes or the rounds run out. A search that never clears is a
result too.

## 6. Report, serve, next day

Every candidate is posted as a harness version, so the Runs page groups the
dots by harness. Under the pick, five lines from the ledger: Changed, Moved,
Why, Learned, Reproduce. You read one chart and decide.

The holdout says what the pick can do on traffic it never saw. The next day
says whether that survived. The skill ends by scoring the day after with
the same judge and posting one `LiveDay`. A flagged rate near the holdout's
failure rate closes the loop. One well above it makes that day the new
traces.

```text
Moved: 67 to 73 points on 16 held-out tasks (holdout is 2026-09-20 and later), +6 [+2, +11]; 0 task(s) the baseline passed and it failed; 1.00x the cost per rollout.
next day: 21 of 100 flagged; the holdout said 27 of 100
```

## When the harness is not enough

A search stalls when the model lacks something no prompt supplies. On a
legal classification task, harness edits alone reached 50 points and
harness plus weight updates reached 70 [5]. On an open model the same
held-out test and judge carry over to training: [evals](/evals), then
[SFT from traces](/recipes/04-train/sft) or [GRPO](/recipes/04-train/grpo).

## References

1. Zhang, Y. et al. [Stop Comparing LLM Agents Without Disclosing the Harness](https://arxiv.org/abs/2605.23950). 2026. Harness variance 7.8x model variance; `Disclosure` follows its checklist.
2. Lee, Y. et al. [Meta-Harness: End-to-End Optimization of Model Harnesses](https://arxiv.org/abs/2603.28052). 2026.
3. Wang, Y. et al. [Rethinking the Evaluation of Harness Evolution for Agents](https://arxiv.org/abs/2607.12227). 2026. Matched budgets; +0.6 held out when tuned on the scored tasks.
4. Agrawal, L. A. et al. [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457). ICLR 2026. Selection on the per-task frontier.
5. Hebbar, P. et al. [SIA: Self Improving AI with Harness and Weight Updates](https://arxiv.org/abs/2605.27276). 2026.
6. Lambert, N. [Reinforcement Learning from Human Feedback](https://rlhfbook.com), chapter [Evaluation](https://rlhfbook.com/c/16-evaluation). 2025. The train split picks, the holdout decides.
