<p align="center">
  <a href="https://withwhile.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/whilehq/whileai-sdk/main/docs/assets/hero-dark.png">
      <img src="https://raw.githubusercontent.com/whilehq/whileai-sdk/main/docs/assets/hero-light.png" alt="While. Models improve while they work." width="720">
    </picture>
  </a>
</p>

<p align="center"><code>MID-TRAINING AND POST-TRAINING FOR LANGUAGE MODELS</code></p>

<p align="center">
  <a href="https://github.com/whilehq/whileai-sdk/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/whilehq/whileai-sdk/ci.yml?branch=main&label=ci&labelColor=0b1220&color=5cb08a" alt="CI"></a>
  <a href="https://pypi.org/project/whileai/"><img src="https://img.shields.io/github/v/tag/whilehq/whileai-sdk?sort=date&label=pypi&labelColor=0b1220&color=5cb08a" alt="PyPI"></a>
  <a href="https://pypi.org/project/whileai/"><img src="https://img.shields.io/pypi/pyversions/whileai?labelColor=0b1220&color=3f8f6b" alt="Python"></a>
  <a href="https://pepy.tech/project/whileai"><img src="https://img.shields.io/pepy/dt/whileai?labelColor=0b1220&color=3f8f6b" alt="Downloads"></a>
  <a href="https://github.com/whilehq/whileai-sdk/actions/workflows/ci.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/whilehq/whileai-sdk/badges/coverage.json&labelColor=0b1220" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/whilehq/whileai-sdk?labelColor=0b1220&color=3f8f6b" alt="License"></a>
</p>

Building RL and SFT datasets for agents is hard. `whileai` is the library
that does it, and that measures whether training on them worked. Give it
an agent, or just the agent's tools and system prompt. It writes the
situations the agent might meet, runs the agent through them against a
fake world that fails on purpose, and hands back every conversation as a
row. You grade the rows with your own judge or a verifier. The package
then does the bookkeeping that is easy to skip and expensive to get wrong:
pass rates with intervals, difficulty bands for RL, a check that your judge
agrees with people, decontamination against your eval set, and a scan for
rewards the policy can game. Every method says where it comes from
([References](#references)).

```python
import whileai as wai


@wai.tool
def get_order(order_id: str) -> dict:
    """Look up an order by id."""  # the function is the tool; its signature is the schema
    ...


wai.configure(agent=wai.OpenAI("gpt-4.1-mini"), judge=wai.Anthropic("claude-haiku-4-5"))

data = wai.simulate(tools=[get_order], system_prompt=POLICY, mode="rl", repeats=8)
scored = data.grade(wai.Judge(rubric=RUBRIC))  # or a verifier, or any callable
print(scored.pass_at)  # pass@1 0.61 [0.54..0.68] | pass@8 0.93 | headroom 0.32
print(wai.judge_trust(scored.rows))  # does the judge agree with people
rows = scored.select(mode="rl")  # the 20..80% band, unanimous groups dropped
rows.export("train.jsonl")  # or rows.push("my-agent-rl-v1")
```

```bash
uv add whileai
```

Python 3.10 to 3.13, two dependencies, typed.
`import whileai` takes under 200 ms and never touches the network.

Two domains, kept apart. `import whileai as wai` is the library: simulate,
grade, measure, select, export, on your machine against your models, no
account needed. `whileai.platform` is the While platform: sign in, push
datasets, train and serve on hosted GPUs, track versions. Everything that
talks to withwhile.com lives there and nowhere else.

## Your model, your key

Every role in a run is a model behind an endpoint. Say which with a
backend object; its repr tells you where the call goes and which key it
uses.

```python
wai.OpenAI("gpt-4.1-mini")  # OpenAI(model='gpt-4.1-mini', key=OPENAI_API_KEY)
wai.OpenAI("gpt-4.1-mini", api_key="sk-...")  # key=given, kept for every OpenAI call
wai.Anthropic("claude-haiku-4-5")  # key=ANTHROPIC_API_KEY
wai.Endpoint("Qwen/Qwen3-4B", url="http://localhost:8000/v1")  # vLLM, SGLang, TGI; key=none needed
wai.Ollama("llama3")  # key=none needed
wai.Hosted()  # the model While hosts, on `whileai login`
```

`wai.configure(agent=, judge=, simulator=, api_key=)` sets the process
once. A keyword on the call wins over it, `with wai.context(...)` wins
inside the block, and the environment is read only after all three.
`print(wai.settings)` says what each role resolves to. With nothing
configured, every role uses the model While hosts on the key from
`whileai login`, and the judge is a different model family from the agent.

The spec strings the objects stand for (`"openai:<model>"`,
`"anthropic:<model>"`, `"vllm:<model>@<url>"`, `"ollama:<model>"`) work
anywhere a backend does, and `OPENAI_BASE_URL` points `OpenAI` at a
compatible server. A tool is a typed function under `@wai.tool`: the
signature is the schema, the docstring the description, `Annotated[str,
"note"]` or an `Args:` block the parameter notes. The mock world answers
the calls, faults first; `execute=wai.Tool.dispatch([get_order])` has the
bodies answer instead. Raw schema dicts still work. No tools at all yet?
`wai.simulations.draft_tools("a support agent that looks up orders and
issues refunds")` drafts them on the agent's key. Three things reach While, and only when you ask: leaving
the agent on `Hosted()`, the hosted situation writer, and
`whileai.platform`. `whileai status` prints which key the SDK will use and
where it came from.

## Two ways in

**You only want evals.** Plenty of teams cannot train and still need to
know whether the last prompt edit helped. Run `whileai init-evals` in your
project. It finds your agent, writes a judge and a runner around it, and
gives you a pass rate with a 95% interval, a table of where the agent
fails, and a test that goes red in CI when it gets worse. `coverage_gap`
tells you which situations your tests never reach. `compare_runs` reruns
the same tasks after a prompt or tool change and says whether the change
helped. Start at [docs.withwhile.com/evals](https://docs.withwhile.com/evals).

**You want to train.** Grade the same rows, keep the ones that carry
signal, export to your trainer. That is the rest of this page. The
[platform](#the-platform) at the end is where hosted training lives, if
you want it.

## Sixty seconds, offline

No key, no network. `seeded_agent` is a stand-in agent. It answers
honestly most of the time and, on a labeled fraction of rollouts, does one
thing wrong on purpose: hedges, flatters, or claims success after a tool
failed. Each row records what it did in `seeded`, so you can check that
your judge catches exactly those rows before you trust it on real ones.

```python
import whileai as wai


@wai.tool
def get_order(order_id: str) -> dict:
    """Look up an order by id."""
    ...


data = wai.simulate(
    wai.seeded_agent([get_order]),
    tools=[get_order],
    system_prompt="Help customers with orders.",
    simulator=False,  # situations from templates, no model
    mode="rl",
    repeats=4,
    repeat_policy="fixed",
    budget=64,
)
scored = data.grade(judge=lambda row: {"reward": int(not row["seeded"])})
print(scored.pass_at)
rows = scored.select(mode="rl")
print(rows)
```

```
pass@1 0.67 [0.55..0.78] | pass^4 (pass_pow_k) 0.19 [0.00..0.38] | pass@4 1.00 [1.00..1.00] | headroom 0.33 (16 groups, k=4)
rl selection: kept 27 of 64 rows
  band 20%..80% pass rate: 0 asks dropped (0 too easy, 0 too hard)
  unanimous groups dropped: 6; duplicates dropped: 27; truncated drop: 0
  privileged leaks dropped: 4
  groups kept: 10
  hack scan: train
```

pass@1 is the pass rate over tasks with a bootstrap interval. pass^4 is
how often all four rollouts of a task pass. Headroom is pass@4 minus
pass@1, the gap an RL update could close. `select` prints what each gate
dropped and why (the warnings that follow are cut here; the quickstart
shows them all); `rows.export(path)` writes them trainer-ready.

To use your own agent, pass any callable that takes the user message and
returns `{"steps": [...], "final_text": "..."}`, or a backend object from
the section above. `wai.Judge(rubric=...)` is the LLM judge as an object;
`data.grade(spec="typesafe:jev-latest")` grades with TypeSafe's Jev
instead: typed questions, a probability on every verdict, no output tokens
(judge only, on `TYPESAFE_API_KEY`). Next:
[docs.withwhile.com/get-started/quickstart](https://docs.withwhile.com/get-started/quickstart).

## The loop

| Step | Call | What it computes | Refs |
|---|---|---|---|
| Simulate | `simulate(agent, tools=, system_prompt=, mode="rl", repeats=k)` | covering array over tools, world state and user stance; k rollouts per prompt; scheduled tool faults | [2], [3] |
| Grade | `data.grade(Judge(rubric=))`, `verify.MathEqual`, `verify.CodeExec` | reward per rollout under one contract; verifiable rewards | [4], [5] |
| Validate the judge | `judge_trust`, `judge_probes` | agreement and Cohen's kappa against human gold; length bias; exploit probes | [6], [7] |
| Measure | `pass_at`, `compare`, `eval_variance`, `holdout_size` | pass@1, pass^k, pass@k with bootstrap intervals over tasks; paired delta with a permutation p-value; noise band; power | [8], [9], [10], [11] |
| Select | `scored.select(mode="rl"\|"sft")`, `build_preference_pairs`, `curriculum` | 20 to 80% difficulty band, unanimous-group drop, rejection sampling, length-matched pairs, curriculum | [12], [13], [14], [15] |
| Guard | `decontaminate`, `hack_scan`, `trace_markers`, `HackMonitor` | overlap with the eval set; reward-feature correlation within task against a shuffle floor; trajectory lies | [16], [17], [18] |
| Train and export | `rows.export`, `export_environment`, `platform.train`, `platform.serve` | loss masks; a `verifiers` environment for GRPO; hosted LoRA SFT, GRPO, DPO, RM | [1], [19], [20] |

The first name in each row is at the top level, `wai.<name>`. Everything
else is one dot down at `wai.simulations.<name>`, and the science behind
each is on
[docs.withwhile.com/concepts/engine](https://docs.withwhile.com/concepts/engine).

## The science

**SFT.** `optimize(mode="sft")` is rejection sampling [14], [16]: keep the
best-scoring completion for each prompt, with a random selector alongside
so you can tell whether picking the best did anything. Exported rows carry
a `loss_mask` per message, so the trainer learns from the agent's turns and
not from tool output. `unroll=True` splits a long conversation into one
sample per agent turn, each with the context that turn actually saw.
`format="trl"` is the shape `SFTTrainer` loads [1, ch. 4].

**RL with verifiable rewards.** When a program can check the answer, the
reward should be that program [5]: `MathEqual`, `CodeExec` against hidden
tests, `JSONSchema`, and combinations of them. In `mode="rl"` every prompt
gets two rollouts first. Only prompts where those two disagree are filled
to k, because a group that all passes or all fails has zero advantage under
GRPO [19]. That is DAPO's dynamic sampling [12], applied while the rollouts
are generated instead of after. `optimize(mode="rl")` then keeps the
prompts the policy solves 20 to 80% of the time [13] and lets you choose
what happens to rollouts that hit the length cap [12]. `export_environment`
writes the tasks, the fake world and the reward as a `verifiers` package
you can hand to a trainer. Rows keep their sampling logprobs so the trainer
can form the importance ratio [21], and `mean_kl` measures drift from the
reference model [22].

**Character training.** Write down how the model should talk as a
constitution [23], [24]. `load_spec` hashes it into `spec.version`, so an
edit to one principle is a new version. The judge is checked against the
labels the spec itself carries before it grades anything. Preference pairs
are matched on length [7], so the model learns the trait and not "longer
is better". Put `spec.behaviors()` in `must_not_regress` and
`delta_report` fails any run that improved one trait by giving up
another. [docs/character-training.md](docs/character-training.md).

**Evaluation.** Intervals are bootstrapped over tasks, not rollouts,
because rollouts of the same task are not independent [8], [10], [11].
Run an eval three times with `runs=3` and `delta_report` refuses to call a
change real when it sits inside twice the run-to-run standard deviation
(`budget` is per run: `runs=3, budget=100` is up to 300 rows).
`holdout_size` says how many prompts you need to see a given gain at 80%
power [11]; most evals are too small. `decontaminate` checks training rows
against the eval set with the 80% n-gram overlap rule [16], and with
embeddings when you pass an embedder. [docs/evals.md](docs/evals.md).

**Over-optimization.** The reward is a proxy for what you want, and RL
finds the gap between the two [17]. `hack_scan` looks for the feature that
predicts reward within a task, against a shuffled baseline, so a judge that
pays for a phrase or a delimiter shows up before you train on it.
`judge_probes` tries the tricks a policy finds first, flattery included
[18]. `delta_report(proxy=, target=)` fails when the training reward went
up and the metric you care about did not. `HackMonitor` runs the same scan
inside a TRL training loop and can stop it.
[docs/reward-hacking.md](docs/reward-hacking.md).

## Recipes

Each recipe is one script and a README that says what you learn, what you
need, and how long it takes. All of them run in CI.

| Step | Recipes |
|---|---|
| [01-simulate](recipes/01-simulate) | bring your own agent, verifiers, a traced coding agent |
| [02-measure](recipes/02-measure) | eval your agent, pass@k, reward hacking, safety evals |
| [03-select](recipes/03-select) | the row schema, GRPO data with a gradient gate, character |
| [04-train](recipes/04-train) | hosted loop, identity SFT, GRPO and DPO on Modal, text-to-SQL |
| [05-export](recipes/05-export) | Hugging Face datasets and adapters |
| [papers](recipes/papers) | one recent paper per recipe, the number it moved with its interval |

## The platform

Separate from the library, and optional. Sign in once and the same rows
push to an account, train on hosted GPUs, and come back as an
OpenAI-compatible endpoint. Everything above this heading runs without it.

From a terminal, for a coding agent that manages the account:

```bash
whileai login                    # or: whileai signup --email you@example.com
whileai agents                   # what is tracked, what each one serves
whileai agent refund-bot         # record, behaviors, verdict
whileai runs refund-bot          # the version table
whileai verdict refund-bot       # does the candidate beat the served version, and is it real
whileai promote refund-bot v4    # usually the person's button on the platform
whileai live refund-bot --day 2026-09-17 --version v3 --replies 2400 --flagged 98
whileai keys                     # names and prefixes; create or revoke under Account
```

Every command takes `--json`. They are thin calls into `whileai.platform`.
`push` refuses RL data with no mixed groups, since a trainer would learn
nothing from it.

```python
from whileai import platform

platform.login()  # once; or wai.configure(api_key="zp_...")
v1 = rows.push("refunds-v1", holdout=0.2)  # the selection, gated
run = platform.train(v1["datasetId"], method="grpo", steps=200)  # sft | grpo | dpo | rm
run.wait()
model = platform.serve("refunds-v2", run)  # OpenAI-compatible endpoint
```

If you train with your own code, `platform.TrainerCallback` reports into
the same run page. Traces from production come back through `traces=`, which
points the next simulation at the situations that failed.

## Documentation

This README is the shape of the loop. The docs are the depth, in the same order:

| You are at | Go to |
|---|---|
| the sixty-second run above | [Quickstart](https://docs.withwhile.com/get-started/quickstart), then [Connect your agent](https://docs.withwhile.com/get-started/connect-your-agent) for backends and keys |
| the loop table | [The five calls](https://docs.withwhile.com/reference/five-calls): the run in order, the judge contract, verifiers |
| the science | [The engine](https://docs.withwhile.com/concepts/engine): how a row is made, with references |
| a call you want the signature of | [API](https://docs.withwhile.com/api/index): every public call, generated from the package on each release |
| the platform | [Platform](https://docs.withwhile.com/reference/platform): sign in, datasets, hosted training, serving |

What we believe and where each belief is enforced is
[CONSTITUTION.md](CONSTITUTION.md). The coding standard the package is
held to, PyTorch and DSPy ergonomics, is
[docs/reference/style.md](docs/reference/style.md).
[CHANGELOG.md](CHANGELOG.md) has one entry per release.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check . && uv run mypy && uv run ty check
```

CI runs the suite on Python 3.10 to 3.13, gates coverage at 90%, and runs
every recipe's `smoke.sh`. [CONTRIBUTING.md](CONTRIBUTING.md).

## Cite

```bibtex
@software{weiss2026whileai,
  title  = {whileai: post-training data and evaluation for tool-using agents},
  author = {Weiss, Jacob},
  year   = {2026},
  url    = {https://github.com/whilehq/whileai-sdk}
}
```

## References

1. Lambert, N. *Reinforcement Learning from Human Feedback*. arXiv:2504.12501, 2025.
2. Kuhn, D. R., Wallace, D. R., Gallo, A. M. Software Fault Interactions and Implications for Software Testing. *IEEE TSE* 30(6), 2004.
3. Yao, S. et al. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024.
4. Ouyang, L. et al. Training Language Models to Follow Instructions with Human Feedback. NeurIPS, 2022.
5. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
6. Cohen, J. A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement* 20(1), 1960.
7. Zheng, L. et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS, 2023.
8. Chen, M. et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374, 2021.
9. Wilson, E. B. Probable Inference, the Law of Succession, and Statistical Inference. *JASA* 22(158), 1927.
10. Efron, B., Tibshirani, R. J. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
11. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
12. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
13. He, J. et al. Skywork Open Reasoner 1 Technical Report. arXiv:2505.22312, 2025.
14. Yuan, Z. et al. Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825, 2023.
15. Rafailov, R. et al. Direct Preference Optimization. NeurIPS, 2023.
16. Touvron, H. et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288, 2023.
17. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML, 2023.
18. Sharma, M. et al. Towards Understanding Sycophancy in Language Models. ICLR, 2024.
19. Shao, Z. et al. DeepSeekMath. arXiv:2402.03300, 2024.
20. DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.
21. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
22. Ziegler, D. M. et al. Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593, 2019.
23. Bai, Y. et al. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073, 2022.
24. OpenAI. Model Spec, 2024. model-spec.openai.com.

## License

Apache-2.0
