# Context LM: a small model that keeps its own context as a file

**Paper:** Context Language Models, University of Washington, Meta Superintelligence Labs, MIT and Trillium Labs, arXiv:2609.37725, September 2026. https://arxiv.org/abs/2609.37725
**Book:** GRPO scores each rollout against its own group, so whatever separates rollouts inside a group is what the policy learns; an outcome reward alone cannot tell a cheap success from a dear one [1][2].
**Claim:** give the model its context as a file it can rewrite, train with stepwise GRPO, and add a success-gated efficiency advantage (Eq. 6): among a group's successes, the cheaper trajectories go up and the dearer ones go down; failures get nothing. On BrowseComp-Plus, Qwen3.5-9B went from 28.8% to 42.5% at 1.34 PFLOPs a question [1].
**The change:** the recipe arm pays Eq. 6 only to successes whose last file still holds the whole state (`ContextFile(gate="complete")`). A third arm runs the paper's rule as written (`gate="paper"`), and the baseline runs neither (`gate="off"`).

## Recipe

1. Task: `wai.methods.KVLog`, a seeded key-value log (ContextBench's KV Store, cut down [1]). It has 5 chunks of 8 `set key = value` lines over 8 keys, and the asked key is set at least twice. Each step the model sees `context.md` and one chunk; then the chunk is gone and it writes the whole new `context.md`. The last step shows only `context.md` and the question, and the model boxes a number. 512 train logs, 200 held out from a disjoint seed range.
2. Harness and credit: `wai.methods.ContextFile(...).trainer(GRPOTrainer)`. Every step of a trajectory gets reward minus its group's mean (group of 8). The efficiency arms add `0.25 x` Eq. 6 on the five edits, not on the answer step (the paper's edit mask). Cost `c_i` counts prefix-reuse tokens: each step pays for the prompt past what is still cached, plus its reply.
3. Base: `Qwen/Qwen2.5-1.5B-Instruct`. LoRA r=32, 60 steps, 4 logs x 8 trajectories a step, lr 1e-4, on-policy, no KL. Seeds 17, 18, 19 and 20 on every arm, all run fresh.
4. Eval: 4 trajectories on each of the 200 held-out logs. pass@1 with a paired 95% interval (`wai.compare`, `train_runs=` all four seeds), and tokens a trajectory, paired by log.

## Run

```bash
python recipe.py --selftest      # the three gates on a stand-in model, offline
python recipe.py --smoke         # Modal: 2 steps, 16 held-out logs, one seed
python recipe.py --reuse         # three arms, four seeds, writes results.json
```

Without the recipe around it, the training is this (trl 0.19, one GPU):

```python
import whileai as wai
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

env = wai.methods.KVLog()
clm = wai.methods.ContextFile()  # the paper's Eq. 6, w_eff 0.25
rows = [{**t, "prompt": env.messages(t, 0, "")} for t in env.tasks(512)]
cfg = GRPOConfig(num_generations=8, num_iterations=1, beta=0.0, max_completion_length=256)
trainer = clm.trainer(GRPOTrainer)(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=[lambda completions, **_: [0.0] * len(completions)],  # the trainer pays the credit
    args=cfg,
    train_dataset=Dataset.from_list(rows),
    env=env,
)
trainer.train()
```

## Result

| Arm | pass@1, 4 seeds pooled | Per seed (17 / 18 / 19 / 20) | Tokens a trajectory | Last `context.md`, chars (17 / 18 / 19 / 20) | GPU min a seed |
|---|---|---|---|---|---|
| Untrained | 0.07 [0.05, 0.09] | | 1,209 | 147 | 0 |
| Baseline, stepwise GRPO (`gate="off"`) | 0.95 [0.94, 0.96] | 0.98 / 0.98 / 0.93 / 0.91 | 1,315 | 79 / 95 / 106 / 758 | 38 |
| Paper, + Eq. 6 (`gate="paper"`) | **0.97 [0.96, 0.98]** | 0.97 / 0.93 / 0.98 / 0.99 | **1,115** | 85 / 158 / 127 / 93 | 39 |
| Recipe, + Eq. 6 on complete files (`gate="complete"`) | 0.92 [0.91, 0.94] | 0.84 / 0.94 / 0.95 / 0.97 | 1,050 | 86 / 61 / 147 / 111 | 40 |

| Pair | pass@1 delta | Verdict | Tokens a trajectory |
|---|---|---|---|
| Paper vs baseline | +0.019 [+0.007, +0.032] | flat | -200 [-212, -188], -15% |
| Recipe vs baseline | -0.023 [-0.039, -0.007] | flat | -265 [-277, -253], -20% |
| Recipe vs paper | -0.042 [-0.057, -0.029] | flat | -65 [-72, -58], -6% |

Training worked on all 12 runs. Stepwise GRPO in the file harness takes a 1.5B model from 0.07 to between 0.84 and 0.99 on held-out logs in 60 steps.

The paper's Eq. 6 is the most accurate arm and spends 15% fewer tokens than plain GRPO. Most of that saving comes from seed 20, where the baseline let its file grow to 758 characters of prose. On the other three seeds the token deltas go both ways.

Every verdict is **flat**. The baseline's own seeds span 0.91 to 0.98, which is wider than any gap between arms.

The recipe arm is this recipe's own change, and it lost to the paper's rule. It was built against a shortcut that one seed found in the first two-seed run: copying only the latest chunk, which answers about 70% of logs. That shortcut did not come back in any of the 12 runs here. The completeness check is also stricter than the task needs. Trained files often drop a key that was set once and never asked, and still answer right. So the gate paid few successes, and Eq. 6 mostly went unpaid. `wai.methods.ContextFile` now defaults to `gate="paper"`.

## Checks

Every number in this section is read from `results.json`.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [3] | run_std 0.002 from 3 re-runs (0.07, 0.07, 0.07); the training-seed spread is far wider |
| Holdout is clean: `decontaminate(train, against=holdout)` | [3] | 0 of 512 train logs dropped; train and holdout come from disjoint seed ranges |
| Reward is a program, not a judge | [3] | the boxed number against the log's final value for the asked key |
| Proxy vs target: `wai.compare(proxy=)` | [4] | `proxy=None`: the training reward is the target; over_optimized false |
| Hack scan on the last training batch: `hack_scan` | [4] | nothing above the floor |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | seeds 17 to 20, `--seed 0` for the logs; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0, whileai 0.129; H100, 481 GPU min, $32.03 |

## Climb

| Round | What changed | pass@1, recipe arm | vs previous |
|---|---|---|---|
| 0 | 3 chunks of 5 lines over 5 keys (smoke only) | 0.75 after 2 steps | discarded: the base failed on the box, not the memory |
| 1 | 5 x 8 over 8 keys, paper gate as the recipe, 2 seeds | 0.83 (0.70 / 0.96) | flat against 0.92; one seed copied only the last chunk |
| 2 | `wai.methods.ContextFile`, complete gate as the recipe, paper gate as a third arm, 4 seeds | 0.92 | flat against 0.95; the paper gate is best at 0.97 |

## Learned

- A small model learns to keep its own context. Qwen2.5-1.5B goes from 0.07 to 0.95 on 40-line logs it never sees whole, on every seed, in under 40 GPU minutes. It learns a `key: value` table rather than a copy of the log.
- Use the paper's Eq. 6: +0.02 pass@1 and 15% fewer tokens over plain GRPO at four seeds. The saving is uneven across seeds, because the file bloat it prevents is too.
- A guard built against one seed's failure cost accuracy across four. Next: a harder log (more keys, longer chunks) where the file must be compressed to fit, which is where the paper's 28K budget puts its pressure.

Verified 2026-10-02, whileai 0.129, trl 0.19.1.

## References

1. Context Language Models. arXiv:2609.37725, 2026. Code: https://github.com/facebookresearch/context-language-models
2. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
3. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
4. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
