# PTGS: heat the prompts the policy keeps failing, cool the ones it has mastered

**Paper:** Sharpening Tax in Post-Training, Changdae Oh et al., arXiv:2610.01509, October 2026. https://arxiv.org/abs/2610.01509
**Book:** GRPO's advantage is a reward minus its group's mean, so a group where every rollout scored the same contributes no gradient [1][2]; Reinforce-Ada answers that by drawing more rollouts for those prompts [3].
**Claim:** RL post-training pushes tasks to "always solved" or "never solved", which raises pass@1 and costs pass@k (the Sharpening Tax); sampling each prompt's training group at a temperature set by a running estimate of its difficulty pays less of that tax and raises both numbers (Oh et al.) [7].
**The change:** the temperature each prompt's 4 training rollouts are drawn at. The baseline draws every prompt at 0.9. The recipe draws a success rate from a per-prompt Beta posterior and samples at 0.9 x 1.5^h, from 0.6 for prompts it always solves to 1.35 for prompts it always fails, with the update's log-probs at the same temperature. A third arm, Reinforce-Ada, draws more rollouts at 0.9 instead.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: MATH train, levels 3 to 5, 192 problems; MATH-500 levels 3 to 5, 320 held out (disjoint by construction).
2. Reward, every arm: the binary outcome, `MathEqual` against the MATH answer. A program, not a judge.
3. Baseline: TRL GRPO + LoRA r=32, 80 steps, 12 prompts x 4 rollouts per step at T=0.9, lr 1e-4, on-policy, no KL, advantage = reward minus the group mean with no std division. 80 x 12 / 192 = 5 visits per prompt, so a per-prompt posterior has something to remember.
4. Recipe: the same trainer and the same 12 x 4 update, with each prompt's group drawn at its PTGS temperature, at the setting the paper uses for PPO in both environments (Appendix A.5): tau = 1.5, forgetting factor 0.95, a Beta prior of mass 2 centered on a target success rate that grows from 0.25 to 0.5 over the run. The log-probs in the loss divide each row's logits by its own temperature, as the paper does.
5. Third arm: Reinforce-Ada (`wai.methods.ReinforceAda`, balanced exit, at most 32 draws, 4 kept, pool-rate baseline), the same trainer otherwise. The comparison the paper does not run.
6. Eval: 8 samples per held-out task at T=0.9 for every arm (PTGS is a training sampler, dropped at test time). pass@1 with a paired 95% interval (`wai.compare`, `train_runs=`), pass@8, and the Sharpening Tax `Tax_S(8)` against the base model, with a paired task bootstrap. Two training seeds per arm (17 and 18).

The backward pass is the same size in all three arms: 48 rollouts per step. PTGS generates the same 48; Reinforce-Ada generates more, and the Result table says how many.

## Run

```bash
python recipe.py --selftest                 # the temperature rule, the posterior and the tax, offline
python recipe.py                            # three arms x two seeds, six L40S in parallel
python recipe.py --arm recipe --steps 160   # one arm, longer
```

## Result

| Arm | pass@1 | 95% CI | pass@8 | Tax_S(8) vs base | Steps | Train min per seed |
|---|---|---|---|---|---|---|
| Base, no training | 0.20 | [0.17, 0.23] | 0.50 | | 0 | 0 |
| Baseline, GRPO at T=0.9 (seed 17; seed 18: 0.27) | 0.31 | [0.27, 0.35] | 0.58 (s18 0.55) | +0.006 [-0.021, +0.035] | 80 | 51 to 53 |
| Recipe, PTGS (seed 17; seed 18: 0.32) | 0.31 | [0.27, 0.35] | 0.58 (s18 0.63) | -0.007 [-0.037, +0.023] | 80 | 53 to 57 |
| Reinforce-Ada (seed 17; seed 18: 0.37) | 0.34 | [0.30, 0.38] | 0.63 (s18 0.65) | -0.013 [-0.041, +0.018] | 80 | 185 to 207 |

PTGS vs GRPO: **+0.004 [-0.017, +0.025]** pass@1 on the seed-17 pair, 320 paired tasks; **+0.023** on the mean of both seeds. Verdict: **flat**. On pass@8, with each task's curve averaged over both seeds, PTGS is +0.039 [+0.011, +0.069] over GRPO by task bootstrap; that interval does not carry seed-to-seed variance, and one of the two seed pairs is a tie (0.58 and 0.58).

Reinforce-Ada vs GRPO: **+0.031 [+0.007, +0.054]** pass@1 on the seed-17 pair, +0.061 on the mean of both seeds, and both Reinforce-Ada seeds beat both GRPO seeds on pass@1 and pass@8. Verdict by the package's rule: **flat**, because +0.031 sits under the eval's re-run band at three base re-runs. pass@8 +0.072 [+0.036, +0.106]. It cost 3.7 times the training minutes.

PTGS vs Reinforce-Ada: -0.027 [-0.048, -0.006] pass@1 on the seed-17 pair, -0.033 [-0.063, -0.002] on pass@8. Flat by the same rule, and pointing toward Reinforce-Ada at a quarter of its cost.

No arm paid a Sharpening Tax. Every Tax_S(8) interval covers zero, and every arm raised pass@8 over the base (0.50 to 0.55 through 0.65). The paper's tax shows up at large budgets (k up to 128) on big general post-trained models and on its own 200-step agentic runs; 80 LoRA steps on a 1.5B model at k=8 did not reach it.

What the samplers did, mean over 80 steps per seed:

| | GRPO | PTGS | Reinforce-Ada |
|---|---|---|---|
| Prompts whose trained group has zero advantage | 0.67 to 0.69 | 0.70 to 0.72 | 0.41 |
| Rollouts drawn per prompt | 4 | 4 | 23 |
| Prompts heated above T=0.9 | | 0.56 | |
| Mean sampling temperature | 0.9 | 1.00 | 0.9 |

PTGS left more groups with no gradient than GRPO, not fewer. Heating a prompt the model never solves at 0.9 rarely makes it solve one at 1.35, and cooling a prompt it usually solves turns its group all-right, which zeroes that group's GRPO advantage (the paper's Proposition 5 says the same). Reinforce-Ada halves the flat groups by paying for 23 draws per prompt instead of 4.

## Checks

Every cell is written by `recipe.py` into `results.json`.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [4] | run_std 0.008 from 3 re-runs (pass@1 0.20, 0.20, 0.19); a single-run delta under 0.051 is noise at t(df = 2) |
| Holdout is clean: `decontaminate(train, against=holdout)` | [5] | 0 of 192 train rows dropped |
| Reward is a program, not a judge | [5] | `MathEqual` (Math-Verify) against the MATH-500 answer |
| Proxy vs target: `wai.compare(proxy=)` | [6] | `proxy=None`: the training reward is the target; over_optimized false |
| Length: mean completion length before -> after, per arm | [6] | 1,379 chars base -> 1,198 GRPO, 1,147 PTGS, 1,021 Reinforce-Ada. No arm is winning on length |
| Hack scan on the last training batch: `hack_scan` | [6] | GRPO `contains:) , AND contains:\(`, PTGS `contains:= AND contains:of`, Reinforce-Ada nothing above the floor: LaTeX and arithmetic surface, nothing endorsed |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | training seeds 17 and 18, `--seed 0` for the data; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | the paper's PPO setting (tau 1.5, p~ 0.25 to 0.5), beside GRPO and Reinforce-Ada, 2 seeds per arm | GRPO 0.31 / 0.27, PTGS 0.31 / 0.32, Reinforce-Ada 0.34 / 0.37 | PTGS vs GRPO +0.004 [-0.017, +0.025], flat |

## Learned

- PTGS swaps into TRL 0.19.1 with a per-row logits processor on `generate` and a per-row divisor in `_get_per_token_logps`; nothing in the loss changes. It costs the same as GRPO.
- It did not move pass@1, and it did not reduce the groups that teach nothing: it raised them. Heating helps only a prompt whose success rate actually rises at a higher temperature (the paper's Theorem 3 assumes it); on MATH at 1.5B a prompt failed at 0.9 is mostly failed at 1.35 too. The coverage edge (+0.04 pass@8) is one seed pair, not a result.
- Reinforce-Ada was the better spend on this set: both seeds above both GRPO seeds on pass@1 and pass@8, at 3.7 times the training time. Hard prompts here need more draws, not hotter ones.
- No Sharpening Tax at this scale: every arm widened pass@8. The tax needs a sharper policy than 80 LoRA steps make; the paper sees it on fully post-trained checkpoints and at k up to 128.
- Next: the paper's GRPO-tuned setting (tau 1.4, p~ fixed at 0.5), and k = 64 on the holdout so the tax has room to show. Same two seeds.

Verified 2026-10-02, whileai 0.127, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. 803.8 GPU minutes over six containers, $26.79 on L40S (the GRPO seed-17 container also ran the three base evals). Run page: https://while.ai/platform/training/run_ef3d2f55c789144d

## References

1. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
2. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Reinforcement Learning*.
3. Xiong, W. et al. Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives. arXiv:2510.04996, 2025.
4. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
5. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
6. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
7. Oh, C. et al. Sharpening Tax in Post-Training. arXiv:2610.01509, 2026.
