# Context LM: pay for a short context file once the answer is right

**Paper:** Context Language Models, University of Washington, Meta Superintelligence Labs, MIT and Trillium Labs, arXiv:2609.37725, September 2026. https://arxiv.org/abs/2609.37725
**Book:** GRPO scores each rollout against its own group, so whatever separates rollouts inside a group is what the policy learns; an outcome reward alone cannot tell a cheap success from a dear one [1][2].
**Claim:** give the model its context as a file it can rewrite, train with stepwise GRPO, and add a success-gated efficiency advantage (Eq. 6): among a group's successes, the cheaper trajectories go up and the dearer ones go down, failures get nothing. On BrowseComp-Plus, Qwen3.5-9B went from 28.8% to 42.5% at 1.34 PFLOPs a question, where a summary harness trained the same way reached 42.1% at 2.19 [1].
**The change:** the recipe arm adds `0.25 x clip((mean success cost - c_i) / mean success cost, -1, 1)` to the advantage of every context edit of a successful trajectory. Harness, task, reward, model, steps and seeds are the same in both arms.

## Recipe

1. Task: a seeded key-value log (ContextBench's KV Store, cut down [1]). 5 chunks of 8 `set key = value` lines over 8 keys; the asked key is set at least twice. Each step the model sees `context.md` and one chunk, then the chunk is gone, and it replies with the whole new `context.md`. The last step shows only `context.md` and the question; the model boxes a number. 512 train logs, 200 held out from a disjoint seed range.
2. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Cost `c_i`: prefix-reuse tokens, each step's prompt past the prefix it shares with the previous step's prompt and reply, plus the reply (the token count the paper's `kv_cache_flops.py` turns into FLOPs).
3. Baseline, stepwise GRPO: every step of a trajectory gets reward minus the mean over its group of 8.
4. Recipe: baseline plus `w_eff = 0.25` times Eq. 6 on the five context edits, not on the answer step (the paper's edit mask). A group with fewer than 2 successes gets no efficiency credit.
5. Both arms: TRL GRPO machinery + LoRA r=32, 60 steps, 4 logs x 8 trajectories a step, lr 1e-4, on-policy, no KL, seeds 17 and 18.
6. Eval: 4 trajectories on each of the 200 held-out logs. pass@1 with a paired 95% interval (`wai.compare`), and tokens per trajectory, paired by log.

## Run

```bash
python recipe.py --selftest      # the task, the file, the cost and the credit, offline
python recipe.py --smoke         # Modal: 2 steps, 16 held-out logs, one seed
python recipe.py --reuse         # both arms, two seeds, writes results.json
python recipe.py --w-eff 1.0     # the code default instead of the paper's run
```

## Result

| Arm | pass@1, both seeds pooled | Per seed (17 / 18) | Tokens a trajectory (17 / 18) | Last context.md, chars (17 / 18) | Steps | GPU min a seed |
|---|---|---|---|---|---|---|
| Base, no training | 0.11 [0.08, 0.13] | | 1,219 | 168 | 0 | 0 |
| Baseline, stepwise GRPO | 0.92 | 0.92 / 0.91 | 1,068 / 1,390 | 132 / 308 | 60 | 35 |
| Recipe, + Eq. 6 | 0.83 | **0.70** / **0.96** | 1,249 / **1,071** | 154 / **92** | 60 | 38 |

Recipe vs baseline on pass@1, seed 17 paired (`wai.compare`, `train_runs=` both seeds): -0.22 [-0.29, -0.16]. Verdict: **flat**. The two recipe seeds land on opposite sides of the baseline, so the training-seed spread swallows the gap.

Tokens a trajectory, recipe minus baseline, paired by held-out log: seed 17 +181 [+172, +189] (+17%), seed 18 -318 [-330, -307] (-23%).

The two seeds learned different files. On seed 18 the recipe did what the paper says: a bare `pearl: 231` table, one line a key, the cheapest file of any arm. Accuracy went up to 0.96 and tokens went down 23%. On seed 17 it locked onto copying the latest chunk verbatim, which answers only when the asked key's last `set` falls in that chunk. That happens on about 70% of logs, and the arm scored 0.70. The baseline seeds also differ, but both still answer: seed 17 folds the log into one line of current values, and seed 18 copies the last two chunks forward at 308 characters.

## Checks

Every number in this section is read from `results.json`.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [3] | run_std 0.018 from 3 re-runs (0.11, 0.09, 0.07); noise band 0.076 |
| Holdout is clean: `decontaminate(train, against=holdout)` | [3] | 0 of 512 train logs dropped; train and holdout come from disjoint seed ranges |
| Reward is a program, not a judge | [3] | the boxed number against the log's final value for the asked key |
| Proxy vs target: `wai.compare(proxy=)` | [4] | `proxy=None`: the training reward is the target; over_optimized false |
| Hack scan on the last training batch: `hack_scan` | [4] | nothing above the floor. It reads the answer step, and the seed-17 shortcut sits in the files, which it does not read |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | seeds 17, 18, `--seed 0` for the logs; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0; H100, 164 GPU min, $10.95 |

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 0 | 3 chunks of 5 lines over 5 keys (smoke only) | 0.75 after 2 steps | discarded: the base failed on the box, not the memory, so its file had nothing to shrink |
| 1 | 5 chunks of 8 lines over 8 keys, as the paper, w_eff 0.25 | 0.83 pooled (0.70 / 0.96) | flat against 0.92 |

## Learned

- Eq. 6 can reach the paper's result at 1.5B. On seed 18 it gave a smaller file, 23% fewer tokens and higher accuracy than the baseline. It did so on one seed of two, though.
- An efficiency term that only pays successes still rewards a cheap partial success. Copying the last chunk wins about 70% of logs, and a seed that finds it early can keep it. The paper's run had a turn budget, a `shrink` edit gate and an LLM judge; none of those is here to push back.
- Next: three or more seeds per arm, and a check on the files rather than the answers, such as the share of current values `context.md` still holds after each chunk. That would see the shortcut before the holdout does. `w_eff` 0.1 is the cheap ablation.

Verified 2026-10-02, whileai 0.127, trl 0.19.1.

## References

1. Context Language Models. arXiv:2609.37725, 2026. Code: https://github.com/facebookresearch/context-language-models
2. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
3. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
4. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
