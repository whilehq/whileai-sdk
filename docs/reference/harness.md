---
title: "The harness"
sidebarTitle: "Harness"
description: "The program around the model as one object: run it like an agent, version it like weights, and say which lever moved the score, the harness or the model."
---

A model never meets a task alone. Something builds its context, hands it
tools, decides when to stop, retries, compacts, delegates. That program is
the harness. On the agents most teams run, a closed model behind Claude
Code, Codex, pi, or their own prompt-and-tools loop, it is the only part
they can change, and it moves the score more than people expect: among
comparable frontier models the harness explains more of the spread than
the model does, and can reverse which model ranks first [1]. A harness
searched over with earlier candidates' scores and traces in view beat the
hand-built ones on TerminalBench-2 [2]. A policy trained under one fixed
harness fell apart when the tools shifted [3].

So `wai.Harness` treats the harness the way the rest of the library treats
weights: one object, a fingerprint that is its version, rows that say which
one produced them, and a report that says whether changing it did anything.

## One object

The prompted loop the SDK plays itself: a model, the instructions, the
tools. The fingerprint hashes what a comparison has to disclose [1] (model,
instructions, tool names, and the `Disclosure` fields below), so a prompt
edit is a new version without anyone naming it [4].

```python
import whileai as wai

careful = wai.Harness("openai:gpt-4.1-mini", instructions=POLICY, tools=TOOLS, label="careful@mini")
print(careful.kind, careful.model_name, careful.tool_names)
```

```
prompted gpt-4.1-mini ['issue_refund', 'lookup_order']
```

`Disclosure` carries the rest of the setup: the context files the harness
loads, the turn cap, how it compacts a long context, retries, subagents,
sampling. Set a field and the hash changes; leave the label off and the
version is `h-` plus the hash.

```python
from whileai.harness import Disclosure

capped = wai.Harness(
    "openai:gpt-4.1-mini",
    instructions=POLICY,
    tools=TOOLS,
    disclosure=Disclosure(max_turns=6),
)
print(capped.fingerprint == careful.fingerprint, capped.version.startswith("h-"))
```

```
False True
```

Name variants `prompt@model` and the Runs page groups the dots by prompt
and by model as two axes.

## A coding agent as the harness

Claude Code, Codex and pi each have a non-interactive mode that streams
JSON events. A preset builds the command line, runs one subprocess per
task, and normalizes the stream to the same `{steps, final_text}` every
adapter returns. The disclosure lists the context files present in `cwd`
(`CLAUDE.md`, `AGENTS.md`, `.pi/SYSTEM.md`), so the same command in a
directory with a different `CLAUDE.md` is a different harness.

```python
coder = wai.Harness.claude_code("sonnet", cwd=".", max_turns=8, tools=["Read", "Edit", "Bash"])
print(coder.kind, coder.tool_names, coder.disclosure.max_turns)

codex = wai.Harness.codex("gpt-5-codex", cwd=".", sandbox="workspace-write")
pi = wai.Harness.pi("claude-sonnet-4-5", provider="anthropic", cwd=".", extensions=False)
```

```
command ['Bash', 'Edit', 'Read'] 8
```

Calling one plays a task. `claude -p` and `codex exec` and `pi --mode
json` each need their own login and the CLI on `PATH`.

```python
out = coder("Make the failing test in tests/ pass. Do not touch the test.")
print(out["final_text"], len(out["steps"]), "tool calls")
```

Any other program is `Harness.command([...])`: the token `"{prompt}"` in
the command line is replaced by the task text (or the text goes to stdin),
and `parse=` turns stdout into a trajectory. An agent you already have as a
Python callable is `Harness(agent=fn, ...)`, given a label and a model name
so its rows can be compared.

## Run it, and every row says so

`simulate(harness)` takes the tools, the system prompt and the turn cap
from the harness when the call does not name them, plays a prompted
harness through the engine (so the mock world and the scheduled faults
apply) or a command harness as it is, and stamps every row with
`harness = {label, hash, model, kind}`.

```python
careful = wai.Harness(
    agent=careful_agent,
    instructions=POLICY,
    tools=TOOLS,
    label="careful@scripted",
    model="scripted",
)
data = wai.simulate(
    careful,
    seeds=REFUND_ASKS,
    situations=4,
    budget=8,
    simulator=False,
    mode="rl",
    repeats=2,
    repeat_policy="fixed",
    fault_rate=0.0,
    avg_turns=1,
)
row = data.rows()[0]
print(row["harness"]["label"], row["harness"]["kind"])
```

```
careful@scripted callable
```

## Which lever moved the score

Run every harness in a set on every model in a set over the same frozen
tasks (`tasks=` the first run, so the asks match), grade them with one
judge, and hand all the rows to `wai.harness.attribute`. It reads the
harness x model grid off the rows and decomposes the spread of the cell
means into a harness part, a model part and their interaction, with a
bootstrap interval over tasks on each share [6]. It also says whether the
leading model changes from one harness to another, the ranking reversal of
[1]. The verdict is in words: the harness moved the score more than the
model did, the model did, or the difference could be chance.

The rows below are built by hand so the page runs offline; yours come out
of `simulate` with the stamp already on them.

```python
rows = []
for harness in ("careful", "eager"):
    for model in ("gpt-4.1-mini", "claude-haiku-4-5"):
        for t in range(200):
            solved = (t % 4 != 0) if harness == "careful" else (t % 2 == 0)
            lift = model == "claude-haiku-4-5" and t % 10 == 1
            rows.append(
                {
                    "task_id": f"t{t}",
                    "reward": 1.0 if (solved or lift) else 0.0,
                    "harness": {"label": harness, "model": model},
                }
            )
print(wai.harness.attribute(rows))
```

```
attribution on pass_at_1: 2 harnesses x 2 models, 200 tasks each cell
  harness     claude-haiku-4-5      gpt-4.1-mini
  careful                 75.0              75.0
  eager                   60.0              50.0
  spread explained: harness 89% [61..97], model 6% [2..19], interaction 6%
  harness moves the score by up to 20.0 points, the model by up to 5.0
  the same model leads under every harness
  the harness moved the score more than the model did
```

The interval is the result. The same pattern on 40 tasks is not resolved:
the harness effect is 20 points, but 40 binary outcomes cannot separate a
harness share of 89 from one of 2, and the report says so instead of
rounding the verdict up.

```python
print(wai.harness.attribute([r for r in rows if int(r["task_id"][1:]) < 40]))
```

```
attribution on pass_at_1: 2 harnesses x 2 models, 40 tasks each cell
  harness     claude-haiku-4-5      gpt-4.1-mini
  careful                 75.0              75.0
  eager                   60.0              50.0
  spread explained: harness 89% [2..100], model 6% [0..49], interaction 6%
  harness moves the score by up to 20.0 points, the model by up to 5.0
  the same model leads under every harness
  which lever moved the score more could be chance: the interval on the difference in shares covers zero
```

A grid with a hole (one harness never ran on one model) or tasks that
appear in only some cells is named in the error, never averaged over.

## Train under several harnesses

A policy trained under one fixed harness collapses when the tool
environment shifts; one trained across harnesses holds up out of
distribution [3]. `export_environment(harnesses=[...])` writes the harnesses
into the environment's `spec.json` (label, hash, instructions, tool
schemas, disclosure), and the trainer's environment draws one per task
from a hash of the task id and a seed, so a re-run draws the same map. The
rollout runs with that harness's instructions as its system prompt and its
tool schemas as its tool set, and its state carries `harness = {label,
hash}`, so a trace says which one it ran under. The package README lists
them under "Harnesses". `load_environment(harness_mix=)` takes `"uniform"`
or one weight per harness.

```python
import json
import tempfile
from pathlib import Path

eager = wai.Harness(
    agent=eager_agent,
    instructions="Refund the order the customer names. Do not look it up first.",
    tools=TOOLS[1:],
    label="eager@scripted",
    model="scripted",
)
out = Path(tempfile.mkdtemp()) / "refunds"
wai.export_environment(
    data, out, tools=TOOLS, system_prompt=POLICY, band=None, harnesses=[careful, eager]
)
spec = json.loads((out / "refunds" / "spec.json").read_text())
by_label = {"careful@scripted": careful, "eager@scripted": eager}
for h in spec["harnesses"]:
    print(h["label"], len(h["tools"]), "tools", h["hash"] == by_label[h["label"]].fingerprint)
print("### Harnesses" in (out / "README.md").read_text())
```

```
careful@scripted 2 tools True
eager@scripted 1 tools True
True
```

## On the platform

`harness.pin()` is the platform record with the same label and hash, and
`track(harness=)`, `tracked.run(harness=)` and `HarnessSweep` accept the
runnable object directly. The wire does not change: the disclosure folds
into the hash and is not sent, and a platform `Harness` with no disclosure
keeps the hash it always had. This needs `WHILEAI_API_KEY`.

```python
from whileai.platform import track

tracked = track("refund-bot", model="gpt-4.1-mini", harness=careful)
run = tracked.run("careful@scripted", method="eval", targets=["refund_policy"], harness=careful)
```

## What is tested, and what is not yet

- The Claude Code preset ran live on 2026-09-21 (`haiku`, two turns, one
  allowed tool): the stream parsed and the reply came back in five seconds.
  The `codex` and `pi` parsers are written from each CLI's own
  documentation of its JSON stream and checked against recorded shapes in
  `tests/api/test_harness.py`; nobody has run them live yet. If you do,
  open an issue with the first three lines of the stream.
- The harness draw in an exported environment is tested offline through
  verifiers' own rollout state (`tests/api/test_environment.py`): the same
  task draws the same harness twice, both harnesses appear across forty
  tasks, and the rollout's system prompt and tools are that harness's. No
  prime-rl run has trained on a two-harness spec yet.
- The Meta-Harness outer loop [2] is
  [`recipes/papers/meta-harness`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/meta-harness):
  candidates as files, a proposer that reads every prior candidate's
  source, score and worst rows, and a gate on held-out tasks and a
  held-out model. Its dry run shows the loop with scripted candidates. The
  live replication ran on 2026-09-22 with Claude Haiku 4.5 as the search
  model and gpt-4.1-mini held out, both through OpenRouter: the picked
  candidate beats the baseline by +0.38 [+0.27, +0.47] on 30 held-out asks
  on Haiku and by +0.09 [+0.04, +0.15] on the held-out model, paired by
  task, both excluding zero, at 0.40x the baseline's tokens, over a
  four-draw noise band of 0.17. The run's artifacts are not in the tree
  ([#808](https://github.com/whilehq/whileai-sdk/issues/808)), so the
  number is reproducible only by re-running the recipe on your own keys.
- The harness and the weights under one optimizer is
  [`recipes/papers/harness-and-weights`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/harness-and-weights):
  candidates carry a skills text and a `run_python` tool, GRPO trains
  under the current harness, and `attribute` runs on the 2x2 grid of base
  and trained weights under the baseline and the searched harness. Its
  README says which cells were measured and which were not.
- Not here yet, tracked in
  [#712](https://github.com/whilehq/whileai-sdk/issues/712): a harness
  from the Prime Intellect Environments Hub by id [5].

## References

1. Zhang, Wang, Ge, Xu, Hamm, Reddy. *Stop Comparing LLM Agents Without
   Disclosing the Harness.* 2026. [arXiv:2605.23950](https://arxiv.org/abs/2605.23950).
2. Lee, Nair, Zhang, Lee, Khattab, Finn. *Meta-Harness: End-to-End
   Optimization of Model Harnesses.* 2026. [arXiv:2603.28052](https://arxiv.org/abs/2603.28052).
3. Kim, Choi, Lee, Jun, Kim, Park. *The Interplay of Harness Design and
   Post-Training in LLM Agents.* 2026. [arXiv:2606.25447](https://arxiv.org/abs/2606.25447).
4. Lambert. *Reinforcement Learning from Human Feedback*, chapter
   Evaluation. 2025. [rlhfbook.com](https://rlhfbook.com/c/16-evaluation.html).
5. Prime Intellect. *verifiers v1: Decomposing Tasksets and Harnesses for
   Agentic RL & Evaluations.* 2026. [primeintellect.ai/blog/verifiers-v1](https://www.primeintellect.ai/blog/verifiers-v1).
6. Miller. *Adding Error Bars to Evals.* 2024. [arXiv:2411.00640](https://arxiv.org/abs/2411.00640).
