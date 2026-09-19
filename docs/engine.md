---
title: "The engine on one page"
sidebarTitle: "The engine"
description: "How simulate() makes evals and training data in eight steps: coverage, a sandbox world with failure modes, and a judge validated before training."
---

Eight steps from an agent definition to a proven delta. Longer:
[Simulations](/simulations). With references: [the engine](/concepts/engine).

<img className="block dark:hidden" src="/figures/engine-eight-steps-light.svg" alt="The eight steps, Axes to Delta, with Delta feeding the next run" />
<img className="hidden dark:block" src="/figures/engine-eight-steps-dark.svg" alt="The eight steps, Axes to Delta, with Delta feeding the next run" />

## Eight steps

| # | Step | What happens | Code |
|---|------|--------------|------|
| 01 | Axes | What varies: tool, policy rule, user stance, world state, tool condition, history. A situation is a point in that space, not a prompt. | `generate/scenarios.py` |
| 02 | Cover | A pairwise covering array: every pair of axis values co-occurs at least once, because most failures are two-factor interactions [6]. `data.coverage["pairwise"]` reports `pairs_planned`, `pairs_covered`, `fraction`: grid coverage, not policy coverage (`coverage_gap` is that). | `generate/coverage.py` |
| 03 | Search | Five arms fill the grid: `structured` 42%, `llm_guided` 42%, `open_ended` 10%, `behavior_targeted` 3%, `failure_mutation` 3%. Each batch, `w *= 1 + 0.5 * yield` of new behavior signatures and cells, renormalized, floored and capped. Novelty search [7]. | `generate/scenarios.py`, `generate/generator.py` |
| 04 | World | Tools answer from schema-shaped state, deterministic per seed. Unknown id: not found. Schema-echo argument: refused. Every dial is a `WorldOptions` field with its reason in `defaults.py`. | `world/sandbox.py`, `defaults.py` |
| 05 | Rollout | The agent on N situations x n phrasings x k samples. `logprobs=True` keeps each row's log-probabilities, policy version and sampling settings. | `run/engine.py` |
| 06 | Grade | Deterministic conduct rules, then your judge, scored against gold labels before its grades count. | `score/grading.py`, `score/judge_trust.py` |
| 07 | Cut | SFT rows (reward=1, loss mask on agent turns), DPO pairs with margin, GRPO groups in the 20 to 80 percent band [4, 5], or a reward-model set. | `score/optimize.py`, `score/publish_gate.py`, `export.py` |
| 08 | Delta | Re-run held-out tasks after training: a paired difference per task, bootstrap interval, sign-flip permutation p [1]. | `score/delta.py`, `score/stats.py` |

Steps 02, 03, 04 and 07 are ours. The rest is the literature:

- **pass@1, pass^k, pass@k**: unbiased estimators over k samples per task [2, 3]. `score/passat.py`.
- **Intervals**: bootstrap over tasks, not rollouts; before and after as a paired sign-flip permutation [1]. `score/stats.py`.
- **Judge**: agreement and kappa on gold labels, Wilson interval, held-out halves, a different family than the policy [8]. `score/judge_trust.py`.
- **Hack scan**: `Var(r) = E[Var(r | task)] + Var(E[r | task])`; only the first term is GRPO gradient [9]. `score/hack_scan.py`.

## Questions

**Importance sampling?** No; we cover the failure space rather than correct
a proposal. Each row keeps its logprobs, so an asynchronous trainer forms
the truncated ratio `exp(log pi_new - log pi_old)` itself [10, 11].

**How do you know training helped?** `delta_report`: paired before and after
on held-out tasks with a bootstrap interval. A `must_not_regress` marker
whose interval sits below zero fails the run [1].

## References

1. Lambert, N. *Reinforcement Learning from Human Feedback*. 2025. rlhfbook.com.
2. Chen, M. et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374, 2021.
3. Yao, S. et al. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024.
4. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
5. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
6. Kuhn, D. R., Wallace, D. R., Gallo, A. M. Software Fault Interactions and Implications for Software Testing. *IEEE Transactions on Software Engineering* 30(6), 2004.
7. Lehman, J., Stanley, K. O. Abandoning Objectives: Evolution Through the Search for Novelty Alone. *Evolutionary Computation* 19(2), 2011.
8. Zheng, L. et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS, 2023.
9. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML, 2023.
10. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
11. Noukhovitch, M. et al. Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models. ICLR, 2025.

Code paths are relative to
[whileai/simulations/](https://github.com/whilehq/whileai-sdk/tree/main/whileai/simulations).

## What to run next

[`recipes/01-simulate/bring-your-own-agent`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/01-simulate/bring-your-own-agent)
runs these eight steps on a callable of your own, offline and in seconds.
