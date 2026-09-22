---
title: "The five calls"
sidebarTitle: "Five calls"
description: "Agent to gated dataset in five calls: simulate, grade, trust the judge, optimize, push. Plus the judge contract, verifiers, and the RL environment export."
---

Agent to gated dataset. Everything else in this reference is one layer down. `TOOLS` is the list from [Start here](/reference/overview#start-here-no-key-required); `POLICY` is the agent's system prompt.

Every block on this page runs a model or reaches the platform, so it needs
`WHILEAI_API_KEY` in the environment, or `wai login`; a block naming an
`openai:` spec needs `OPENAI_API_KEY`. To follow along offline, pass
`simulator=False` and your own callable agent.

```python
import whileai as wai

wai.configure(agent=wai.OpenAI("gpt-4.1-mini"), judge=wai.Anthropic("claude-haiku-4-5"))

data = wai.simulate(
    tools=TOOLS, system_prompt=POLICY, mode="rl", situations=200, repeats=8
)  # 1 generate
scored = data.grade(
    wai.Judge(rubric=RUBRIC)
)  # 2 grade against the task rubric. A rubric of plain principles scores the
#   mean of its criteria, so rows come back 0, 1/3, 2/3 or 1; pass@1 reads
#   only 0 and 1 and names the rest in its note. kind="hard" per Criterion
#   gives a 0/1 verdict.
print(scored.pass_at)
print(wai.judge_trust(scored.rows))  # 3 trust the numbers
rows = scored.select(
    mode="rl"
)  # 4 keep what carries gradient; print(rows) says what each gate dropped
rows.push("my-agent-rl-v1")  # 5 publish, gated (whileai.platform)
```

`situations=200, repeats=8` is a guess. `wai.simulations.recommend(tools=TOOLS, system_prompt=POLICY, mode="rl")` replaces it with numbers from this agent's own grid: [How much to run](/reference/what-to-run#how-much-to-run).

| Call | What it decides | Reads |
|---|---|---|
| `simulate` | the situations, the users, the world, k rollouts per ask | your spec or tools + system prompt |
| `data.grade(judge=)` | 0/1 per rollout. `data.grade()` uses the hosted judge instead | your judge callable, or your account key (`wai login`) |
| `pass_at` / `judge_trust` | pass@1 with an interval, headroom for RL, whether the judge can be trusted | graded rows, 30 to 100 hand labels as `gold_reward` |
| `scored.select(mode="rl")` | drops junk rows, duplicates, dead groups, and asks outside the difficulty band (pass rate 0.2 to 0.8); flags reward hacks. `optimize` underneath | graded rows |
| `rows.push(name)` | refuses ungraded or gradient-free RL data; stamps calibration. `platform.push` underneath | the selection |

A spec folder is `spec.json` (tools and policy) plus `rubric.md`: what doing the job means, in prose. `grade()` scores against it. The hosted judge writes `reward` and `reason` onto the run's rows and returns the judge report (a dict), so the numbers are read off `data`. `grade(judge=your_callable)` instead returns a `ScoredData` of graded copies, leaves the run untouched, and has its own `.push(name, ...)`. Without a rubric the hosted judge grades the conduct floor only (nothing invented, nothing skipped) and the report says so; pass `rubric=` to `simulate` or `data.grade` to supply one.

After training, measure whether it landed: `wai.compare(before=scored.rows, after=after_rows, target="pass_at_1")` (`compare` is the front-door name; `delta_report` is the same call one dot down, at `wai.simulations.delta_report`). Name the training reward too, `proxy="marker:first_action"`, and the report says whether the run over-optimized it: proxy up while the target did not follow fails the report [5]. `wai.simulations.hack_scan_diff(before, after, endorsed=[...])` names what the update moved toward, and withholds the name when either side came back `degenerate`.

The noise floor (`run_std=`, `run_std_runs=`, from `eval_variance` over re-runs of one model) measures the eval. When both sides are separately trained models, the delta also carries training variance, which the floor cannot see: one recipe read -0.065 [-0.117, -0.013] on one run and +0.050 on the next at one seed per arm. Pass every training seed's rows, `wai.compare(before, after, train_runs={"before": [b_seed1, b_seed2], "after": [a_seed1, a_seed2]})` (a plain list is the after arm's seeds against an untrained base), and the headline interval widens by the between-seed spread: each arm's per-seed means give a between-seed standard deviation, the delta's variance adds `std**2 / n_seeds` per arm, and the printed line shows the arithmetic the way the floor line does. "moved" then needs that interval to exclude zero too. One training seed per arm reports `unresolved`, with the interval and floor lines still printed and the fix on the line: "one training seed per arm; add a seed to resolve". Sources: [2], chapter *Evaluation*, and [9].

Character training is the same loop aimed at how the model talks: a constitution in, graded replies, length-matched pairs and SFT rows out, and the judge checked against the constitution's own labels. Worked example [`recipes/03-select/character`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/03-select/character), guide [Character training](/character-training).

## The judge contract

Rows come back ungraded; your judge decides what good means. A judge is any callable that takes a row and returns a verdict. LLM judge, rules engine, reward model, human-label lookup, HTTP call: the SDK does not care how the reward was produced, only that the result honors this contract. The same contract is what `grade`, `run_judge`, `evaluate`, `grader=`, `optimize` and a gated `push` all read, and what every `verify` verifier and `wai.reward_model(run)` already honors.

```python
judge(row) -> {"reward": 0 or 1}              # the minimum
judge(row) -> {"reward": 0.7,                 # floats allowed
               "reason": "...",               # optional, kept on the row
               "markers": {"grounded": 1.0},  # optional, -> row["markers"]
               "failure_class": "...",        # optional
               ...anything else}              # kept as judge metadata
judge(row) -> 0 or 1 or 0.7                   # a bare number works
```

**Failure modes.** Anything else (a missing `reward`, an unsupported type, an exception, a timeout) marks the row with a `judge_status` of `missing_reward`, `invalid_result`, `error` or `timeout`, and sets `reward=None`. Nothing is silently scored zero, so a broken judge shows up as unjudged rows rather than as a policy that looks bad.

**Marker polarity, the rule for every marker you define.** `1.0` is the good outcome; higher is better; a significant drop is the regression. `delta_report`, `must_not_regress=` and the run page all assume it. Name a marker for the behavior you want (`refund_correct`, not `false_refund_success`), or a fix reads as `DOWN`, and listing the marker in `must_not_regress=` fails the report on the run that repaired the bug. More on the four marker families: [the platform page](/reference/platform).

**The loop, closed in five lines.**

```python
import whileai as wai  # the same one import as the block above

judge = lambda row: {"reward": int("sorry" not in row["final_text"])}
scored = wai.simulations.run_judge(data.trajectories, judge)  # or data.grade(judge=judge)
wai.simulations.export_dataset(
    scored.passes(), output="train.jsonl", system_prompt=POLICY, tools=TOOLS
)
# ...train externally, roll the tuned model on a holdout...
evald = wai.simulations.evaluate(rollouts, judge, model="my-tuned-v1")
nxt = wai.simulate(tools=TOOLS, system_prompt=POLICY, traces=evald.failed_traces())
```

The full contract, with every status and the rest of the loop, is the module docstring of `whileai.simulations.score.judging` (note the `score.`; there is no `whileai.simulations.judging`).

Writing the judge is half of it; knowing whether to believe it is the other half. `wai.judge_trust(rows, judge=...)` and `wai.simulations.judge_probes(rows, judge)` are on [the platform page](/reference/platform). With no `gold_reward` labels on the rows, `judge_trust` returns `ok: False` with a warning that the judge is unmeasured, not failed.

## Verifiers: when the reward is a program, not a judge

For a verifiable task the reward is a checker, not an opinion [1]. `whileai.simulations.verify` gives you one, and because a verifier honors the same judge contract it drops into `grade`, `evaluate`, `optimize` and a gated `push` exactly where an LLM judge would.

```python
from whileai.simulations.verify import MathEqual, CodeExec, JSONSchema, Regex, All

data = wai.simulate(
    tools=MATH_TOOLS, system_prompt=MATH_POLICY, mode="rl", situations=200, repeats=8
)
scored = data.grade(judge=MathEqual())  # the verifier is the reward
rows, _ = wai.optimize(scored, mode="rl")  # GRPO data, gradient checked
```

The candidate is the rollout's `final_text`; the gold is read from the row's `privileged.reference`, which the training export never projects, so the answer key cannot leak into a training file (flat `answer`/`target`/... fields work too, or point at any column with `field=`). Built in: `ExactMatch`, `Includes`, `Regex`, `MultipleChoice`, `Numeric`, `MathEqual`, `JSONValid`, `JSONSchema`, `JSONField`, and `CodeExec` (runs the candidate against hidden tests in a sandboxed subprocess with a timeout). Compose with `All` (right answer and right format), `Any`, or a graded `Weighted` rubric; wrap your own with `@verifier`. Worked example: [`recipes/01-simulate/verifiers`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/01-simulate/verifiers).

## Export an RL environment

On-policy RL (GRPO, RLOO, PPO) samples its own rollouts from the policy under training, so what it needs is not rows but what the rows came from: the task set, the world that answers tool calls, and the reward that grades a finished trajectory. `export_environment` writes those three as an installable `verifiers` package, the shape Prime Intellect and TRL read.

```python
data = wai.simulate(my_agent, tools=TOOLS, system_prompt=POLICY, mode="rl", repeats=8)
data.grade()
# reward and world must import by name in the trainer: a module-level function or "module:attr"
wai.export_environment(data, "envs/my-agent", reward=my_verifier)
# uv pip install -e envs/my-agent
# vf-eval my_agent -a '{"split": "holdout"}' -m <policy> -b <base url> -k <key var>
```

The package is `pyproject.toml`, a README, and a module named after the environment holding `spec.json` (system prompt, the tool schemas verbatim, the turn cap, and dotted references to the reward and the world) and `data/train.jsonl` plus `data/holdout.jsonl` (one task per prompt in the verifiers shape, with the task's fault plan, world state, privileged reference and calibration in `info`, read on the server and never in the prompt). The README carries the gate: the difficulty band applied when the rows were graded (prompts the policy always or never solved carry no advantage and are dropped), the split by scenario, and the train-against-holdout decontamination. A run whose prompts all fall outside the band raises `no train tasks` instead of writing an empty environment. A run that keeps some but holds none out is the commoner accident on a small first export, because the band drops most prompts: the package still installs and trains, and the report warns `empty_holdout`, because `load_environment(split="holdout")` raises on the trainer, the decontamination check has nothing to compare, and there is no held-out set to prove a delta on. Export more prompts, raise `holdout=`, or pass the holdout prompts explicitly.

The environment class lives in the SDK and is tested there: a `StatefulToolEnv` whose world is the mock world seeded per task, or your own `execute=`, and whose rubric is the reward through the judge contract, so a `Verifier` such as `CodeExec`, your judge callable, or `conduct_grade` all work unchanged. The default reward is `task_checklist`: the conduct grade as an honesty gate, times an outcome the world can verify from the task's own coordinates on the grid. A target tool must succeed; a missing entity must be reported and not acted on; an already-done action must be acknowledged and not repeated; an adversarial ask must not produce a write; an unrelated ask must produce no call; a vague ask must be asked back; prior partial action needs a read before the write; a fault on the target must be acknowledged. No model in the loop, and `markers` say which check ran (a rubric computed from state rather than written by a judge [2]). When the rows carry none of that metadata the export warns: the reward reduces to `conduct_grade`, a process reward, and a policy trained on it alone learns to call nothing ([`recipes/03-select/prime-intellect-rl`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/03-select/prime-intellect-rl)). `wai.load_environment(spec)` builds the environment in a process that has `verifiers` (`uv add 'whileai[rl]'`). That extra is marked `python_version >= '3.11' and python_version < '3.14'`, because `verifiers` is; pip and uv drop an extra whose marker does not match and report nothing, so on Python 3.14 the install exits 0 having added no `verifiers` at all. Build the environment on a 3.11 to 3.13 interpreter (`uv venv -p 3.13`); `load_environment` names the range when it is run outside it.

`runtime="openenv"` writes the same three things for Meta PyTorch's [OpenEnv](https://github.com/meta-pytorch/OpenEnv), the `reset`/`step`/`state` contract TRL's `GRPOTrainer`, torchforge, SkyRL and Unsloth drive over HTTP: an `openenv.yaml` package with a FastAPI server (`uv run --project . server`, `openenv build`, `openenv push` for a Hugging Face Space), a client, the spec and the two splits. An episode is one rollout: `reset(split=, index=)` picks a task and returns the chat messages and the tool schemas, each `CallToolAction` runs one tool in the mock world seeded from the task (or your `execute=`), the turn cap ends the episode at reward 0 (DAPO's overlong filtering [6]), and `submit(answer=)` grades the trajectory through the judge contract. The task API lists ids and prompts only; `info` never leaves the server. The environment class is `whileai.simulations.openenv.environment_class(spec)`, built in a process that has `openenv` (`uv add 'whileai[openenv]'`).

```python
wai.export_environment(data, "envs/my-agent", reward=my_verifier, runtime="openenv")
```

The [tool-call-efficiency](https://huggingface.co/datasets/while-ai/tool-call-efficiency) dataset is the same shape built by hand over an executable world with a hidden test suite.

<Warning>
Install `whileai` from PyPI, not from a path or a git URL, if you build a Prime Intellect environment on it. The Environments Hub installs a pushed env with plain pip, so a `[tool.uv.sources]` git pin resolves locally and then fails on their runtime with a `ModuleNotFoundError`.
</Warning>

Training notes:

- Calibrate difficulty with 8 to 16 rollouts per task before exporting, so the band is a measurement, not a guess [2]. The export report's `graded_mixed` is the number of tasks that carry an advantage at all [3].
- Sample at temperature near 1.0 with 8 or more generations per prompt; within-group contrast is what the update learns from [3].
- A rollout cut at the turn or token cap scores 0 and is logged as `truncated` [4].
- Use per-token loss aggregation rather than per-sequence, so long rollouts are not favoured or punished by length alone [4].
- Keep a small KL to the reference or, if the recipe drops it, watch KL drift on the dashboard [2].
- `n_calls`, `judge_ok`, `truncated` and `trace_clean` are logged at weight 0: they are the over-optimization symptoms to watch, never the objective [5].
- Retire tasks the policy now always solves and re-export between rounds (`curriculum`, `retire_solved`) [2, 4].
- If `reward=` is a judge rather than a program, validate it first with `judge_trust` and `judge_agreement`, and keep it in a different model family from the policy, because a model prefers its own writing [6, 7].
- Measure the held-out set before and after with `delta_report` and a `must_not_regress` list, and report pass^k (every one of k tries right) alongside pass@1 for reliability [8, 9].

## Where the tools and policy come from

Pass `spec=` if you have a local tools-and-system-prompt folder of your own: a directory (or a JSON/YAML file) holding `tools` and `policy` / `system_prompt`, optionally with seed `situations` and a `rubric.md` (what doing the job means, for `grade()`). No spec folders ship with this package, so every snippet here uses `tools=` + `system_prompt=`. The two are interchangeable, and `spec=` is only a way to keep them in a file. The generated datasets are on Hugging Face in the [Post-Training Foundational Datasets](https://huggingface.co/collections/while-ai/whileai-post-training-foundational-datasets-6aa0b9c040ff8591988696dc) collection, not stored in this repo: [agent-simulations](https://huggingface.co/datasets/while-ai/agent-simulations) by agent type, [tool-call-efficiency](https://huggingface.co/datasets/while-ai/tool-call-efficiency) (SFT, preference, GRPO and eval splits), and [tau2-simulated](https://huggingface.co/datasets/while-ai/tau2-simulated), among others.

## Knobs these calls read

The full list is on [Parameters](/reference/parameters). These are the ones that change what a row contains.

| Knob | Default | |
|---|---|---|
| `requests_per_situation` | from mode | Phrasings: ways to ask one situation. Alias `phrasings=` |
| `rollouts_per_request` | from mode | Repeats: reruns of one phrasing. Alias `repeats=` |
| `fault_rate` | `0.5`, `0.8` under `mode="rl"` | Share of fault-tagged grid cells that keep their fault. `0` off. Applied by the mock world, so a callable `agent=` that answers its own tool calls never sees one |
| `simulator` | hosted Qwen | Situation writer. `False` uses the built-in template writer (no model, less variety); a model spec runs it on your endpoint |
| `user_model` | `None` | Who plays the simulated user in follow-up turns. `None` is the agent's own model; a model spec moves that job to another model |
| `traces` | `None` | Graded traces of the deployed agent, a list of plain row dicts or a JSONL path. Aims the coverage grid at the behaviors those traces show and keeps the sources out of the generated rows. See [Close the loop](/reference/what-to-run#close-the-loop-aim-the-budget-with-traces) |
| `tasks` | `None` | Re-run a previous run's task set instead of drawing a new one: that run, its rows, or its JSONL path. k is not inherited; pass `repeats=` again |
| `timeout` | `300` | Seconds per agent completion, for `local_model` and every model spec. A served model that scaled to zero takes two to three minutes to answer its first request, so a shorter value drops the first pass; a timed-out call is named in `data.warnings` with the fix |
| `logprobs` | `False` | Ask the rollout model for the log-probability of every token it generates. Each agent turn's step gets `logprob` and `n_tokens`, the row gets the totals. `"tokens"` keeps the per-token list. Model backends only |
| `sampling` | `None` | How your own callable agent samples, `{"temperature": 0.7, "max_tokens": 1024, "model": "my-model"}`, recorded on every row as given. A model backend records its own and ignores this |
| `reproducible` | `None`: `True` unless `time_budget` is set | Same seed, same agent: same rows at any concurrency, on any CPython version. Runs batch by batch, so a slow rollout holds its batch; `False` buys that throughput back at the cost of a task set that depends on thread timing. A clock turns it off |
| `grade` | `False` | Legacy: `True` writes the deterministic conduct score at simulation time. Grade after with `data.grade(...)` instead |
| `llm_grade` | `False` | Extra LLM judge. Needs `OPENAI_API_KEY` |

## References

1. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024. Reinforcement learning with verifiable rewards.
2. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapters *Synthetic Data and Constitutional AI*, *Reasoning*, *Regularization* and *Evaluation*.
3. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
4. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
5. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
6. Zheng, L. et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685.
7. Panickssery, A., Bowman, S. R., Feng, S. LLM Evaluators Recognize and Favor Their Own Generations. arXiv:2404.13076, 2024.
8. Yao, S. et al. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024.
9. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
