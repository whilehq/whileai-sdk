<p align="center">
  <a href="https://withwhile.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/whilehq/whileai-sdk/main/docs/assets/hero-dark.png">
      <img src="https://raw.githubusercontent.com/whilehq/whileai-sdk/main/docs/assets/hero-light.png" alt="wai, the While whale. Models improve while they work." width="720">
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

<p align="center">
  <a href="https://docs.withwhile.com"><b>Docs</b></a> |
  <a href="https://withwhile.com"><b>Platform</b></a> |
  <a href="recipes"><b>Recipes</b></a> |
  <a href="CONSTITUTION.md"><b>Constitution</b></a>
</p>

- **Simulate.** Give it your agent's tools and prompt. It writes the
  situations the agent will meet and runs them against a fake world that
  fails on purpose.
- **Grade.** Your judge or a verifier scores every rollout. The judge is
  checked against people before its scores count.
- **Measure.** One run proves nothing. Ask whether a change is real or
  noise before you ship it or train on it.
- **Select.** Keep the rows that carry signal: the 20 to 80% band for RL,
  the best completion for SFT, nothing that leaks into your eval set.
- **Train.** Export to TRL or a `verifiers` environment, or train and
  serve on the While platform.

wai is While's whale and the alias of the whileai SDK: `import whileai as wai`.
Runs on your machine against your models. Every default cites its source. You
own the model, the data and the weights: the datasets are built from your
production traces, the model is an open model post-trained with SFT and RL, and
the trained weights are yours to download and serve anywhere.

## Install

```bash
uv add whileai
```

Python 3.10 to 3.13, two dependencies, typed. `import whileai` takes under
200 ms and never touches the network.

## Quick start

No key, no network. `seeded_agent` is a stand-in that misbehaves on a
labeled fraction of rollouts, so you can check that your judge catches
exactly those rows.

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
dropped and why; `rows.export("train.jsonl")` writes them trainer-ready.

## With your agent

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

Every role is a model behind an endpoint. A call keyword beats
`wai.configure`, `with wai.context(...)` beats both, then the environment.
Unconfigured, every role uses the model While hosts on `whileai login`.

```python
wai.OpenAI("gpt-4.1-mini")  # key=OPENAI_API_KEY
wai.Anthropic("claude-haiku-4-5")  # key=ANTHROPIC_API_KEY
wai.Fireworks("accounts/fireworks/models/llama-v3p1-8b-instruct")  # key=FIREWORKS_API_KEY
wai.models.Bedrock("us.anthropic.claude-haiku-4-5-20251001-v1:0")  # your AWS account
wai.Endpoint("Qwen/Qwen3-4B", url="http://localhost:8000/v1")  # vLLM, SGLang, TGI
wai.Ollama("llama3")
wai.Hosted()  # the model While hosts
```

Your agent is any callable returning `{"steps": [...], "final_text": "..."}`,
or a backend object. A tool is a typed function under `@wai.tool`; the
mock world answers its calls, faults first. No tools yet?
`wai.simulations.draft_tools("a support agent that issues refunds")`.

**Evals only?** `whileai init-evals` finds your agent, writes a judge and a
runner around it, and gives you a pass rate with a 95% interval and a CI
test that goes red on regression.
[docs.withwhile.com/evals](https://docs.withwhile.com/evals).

## The loop, call by call

| Step | Call | What it computes | Refs |
|---|---|---|---|
| Simulate | `simulate(agent, tools=, system_prompt=, mode="rl", repeats=k)` | covering array over tools, world state and user stance; k rollouts per prompt; scheduled tool faults | [2], [3] |
| Grade | `data.grade(Judge(rubric=))`, `verify.MathEqual`, `verify.CodeExec` | reward per rollout under one contract; verifiable rewards | [4], [5] |
| Validate the judge | `judge_trust`, `judge_probes` | agreement and Cohen's kappa against human gold; length bias; exploit probes | [6], [7] |
| Measure | `pass_at`, `compare`, `eval_variance`, `holdout_size` | pass@1, pass^k, pass@k with bootstrap intervals over tasks; paired delta with a permutation p-value; noise band; power | [8], [9], [10], [11] |
| Select | `scored.select(mode="rl"\|"sft")`, `build_preference_pairs`, `curriculum` | 20 to 80% difficulty band, unanimous-group drop, rejection sampling, length-matched pairs, curriculum | [12], [13], [14], [15] |
| Guard | `decontaminate`, `hack_scan`, `trace_markers`, `HackMonitor` | overlap with the eval set; reward-feature correlation within task against a shuffle floor; trajectory lies | [16], [17], [18] |
| Train and export | `rows.export`, `export_environment`, `platform.train`, `platform.serve` | loss masks; a `verifiers` environment for GRPO; hosted LoRA SFT, GRPO, DPO, RM | [1], [19], [20] |

The first name in each row is `wai.<name>`; the rest are at
`wai.simulations.<name>`. How each is computed:
[docs.withwhile.com/concepts/engine](https://docs.withwhile.com/concepts/engine).

## Why the numbers hold

- **Intervals over tasks, not rollouts.** Rollouts of one task are not
  independent [8], [10], [11]. `runs=3` adds a noise band, and
  `delta_report` refuses a change that sits inside twice the run-to-run
  standard deviation. `holdout_size` says how many prompts you need at 80%
  power; most evals are too small.
- **Dynamic sampling for RL.** Each prompt gets two rollouts; only prompts
  where they disagree fill to k, since an all-pass or all-fail group has
  zero advantage under GRPO [12], [19]. `select(mode="rl")` keeps the 20 to
  80% band [13]. Rows keep sampling logprobs for the importance ratio [21].
- **Verifiable rewards first.** When a program can check the answer, the
  reward is that program [5]: `MathEqual`, `CodeExec` against hidden
  tests, `JSONSchema`.
- **Rejection sampling for SFT.** Best completion per prompt, with a random
  selector alongside so you can tell whether picking the best did anything
  [14], [16]. Rows carry a `loss_mask`, so the trainer learns the agent's
  turns and not tool output.
- **Character from a constitution.** `load_spec` hashes the spec into a
  version. The judge is checked against the spec's own labels, and
  preference pairs are length-matched so the model learns the trait, not
  "longer is better" [7], [23], [24].
- **Reward hacking caught before training.** `hack_scan` finds the feature
  that predicts reward within a task, against a shuffled baseline [17].
  `judge_probes` tries flattery and the other tricks a policy finds first
  [18]. `decontaminate` applies the 80% n-gram rule against your eval set
  [16].

## Recipes

One script and a README each. All run in CI.

| Step | Recipes |
|---|---|
| [01-simulate](recipes/01-simulate) | bring your own agent, verifiers, a traced coding agent |
| [02-measure](recipes/02-measure) | eval your agent, pass@k, reward hacking, safety evals |
| [03-select](recipes/03-select) | the row schema, GRPO data with a gradient gate, character |
| [04-train](recipes/04-train) | hosted loop, identity SFT, GRPO and DPO on Modal, text-to-SQL |
| [05-export](recipes/05-export) | Hugging Face datasets and adapters |
| [papers](recipes/papers) | one recent paper per recipe, the number it moved with its interval |

## Platform

Optional. Sign in once; the same rows push to an account, train on hosted
GPUs, and come back as an OpenAI-compatible endpoint.

```python
import whileai as wai

wai.platform.login()  # once; or wai.configure(api_key="zp_...")
v1 = rows.push("refunds-v1", holdout=0.2)  # the selection, gated
run = wai.platform.train(v1["datasetId"], method="grpo", steps=200)  # sft | grpo | dpo | rm
run.wait()
model = wai.platform.serve("refunds-v2", run)  # OpenAI-compatible endpoint
```

`whileai login`, `agents`, `runs`, `verdict` and `promote` do the same from
a terminal, all with `--json`. `push` refuses RL data with no mixed groups.
[docs.withwhile.com/reference/platform](https://docs.withwhile.com/reference/platform).

## Documentation

- [Quickstart](https://docs.withwhile.com/get-started/quickstart) and [Connect your agent](https://docs.withwhile.com/get-started/connect-your-agent)
- [The five calls](https://docs.withwhile.com/reference/five-calls): the run in order, the judge contract, verifiers
- [The engine](https://docs.withwhile.com/concepts/engine): how a row is made, with references
- [API](https://docs.withwhile.com/api/index): every public call, generated on each release
- [CONSTITUTION.md](CONSTITUTION.md): what we believe and where each belief is enforced
- [docs/reference/style.md](docs/reference/style.md): the coding standard, PyTorch and DSPy ergonomics
- [CHANGELOG.md](CHANGELOG.md): one entry per release

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check . && uv run mypy && uv run ty check
```

CI runs Python 3.10 to 3.13, gates coverage at 90%, and runs every
recipe's `smoke.sh`. [CONTRIBUTING.md](CONTRIBUTING.md).

## Cite

```bibtex
@software{weiss2026whileai,
  title  = {whileai: post-training data and evaluation for tool-using agents},
  author = {Weiss, Jacob},
  year   = {2026},
  url    = {https://github.com/whilehq/whileai-sdk}
}
```

<details>
<summary><b>References</b></summary>

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

</details>

## License

Apache-2.0
