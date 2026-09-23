# Message board: teach copies of a model to read each other's notes

**Paper:** MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning, Chanwoo Park et al., arXiv:2502.18439, February 2025. https://arxiv.org/abs/2502.18439
**Book:** RL on a verifiable reward pays for the answer, so a reader trained that way learns to use a teammate's note only when the note makes its own answer right more often [1][2].
**Claim:** copies of a model that answer, read each other and answer again collaborate better after RL on the team's outcome, and a model trained alone does not learn to collaborate (Park et al.). Talk inside one reply was trained away because nothing read it ([team-talk](../team-talk)); a board gives the talk a reader.
**The change:** what a reader sees. Both arms draft the same four notes per problem and train the reader with GRPO. The baseline reader sees only its own note. The recipe reader sees all four.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split.
2. Round one, both arms: four copies of the current policy each post a note (key steps and a boxed answer, at most 256 tokens). No gradient: the notes are context.
3. Round two, both arms: each copy reads its board and writes the final answer. TRL GRPO + LoRA r=32 trains this step, 40 steps, 12 problems x 4 readers, lr 1e-4, on-policy, no KL. Reward: the reader's own answer, `MathEqual` against the GSM8K gold. Nothing pays for agreeing with the board.
4. The one change: the baseline's reader j sees note j; the recipe's reader j sees notes 1 to 4. Same notes drawn, same tokens.
5. Eval: the same two rounds on the 120 held-out problems, pass@1 over the 4 readers, two training seeds per arm, paired delta with a 95% interval (`wai.compare`, `train_runs=`). A majority vote over the board is the no-reading reference.

## Run

```bash
python recipe.py --selftest                 # the board, the answer reader and the counters, offline
python recipe.py                            # both arms, two seeds, plus a base-eval container, five L40S
python recipe.py --reuse                    # rerun only the containers that failed
```

## Result

| Arm | pass@1 | 95% CI | pass@k | Notes right | Vote | Rescued | Misled | Steps | GPU min |
|---|---|---|---|---|---|---|---|---|---|
| Base, own note | 0.52 | [0.46, 0.59] | 0.78 | 0.30 | 0.38 | 0.25 | 0.03 | 0 | 0 |
| Base, shared board | 0.57 | [0.50, 0.65] | 0.72 | 0.30 | 0.37 | 0.32 | 0.04 | 0 | 0 |
| Baseline, GRPO own note (seed 17; seed 18: 0.68) | 0.65 | [0.58, 0.72] | 0.83 | 0.57 | 0.63 | 0.13 | 0.04 | 40 | 24.9 |
| Recipe, GRPO shared board (seed 17; seed 18: 0.67) | 0.68 | [0.61, 0.75] | 0.79 | 0.41 | 0.48 | 0.29 | 0.02 | 40 | 30.5 |

Recipe vs baseline: **+0.029 [-0.023, +0.085]** across both training seeds. Verdict: **flat**.

Rescued: the reader's own note was wrong and its answer is right. Misled: the note was right and the answer is wrong. Vote: the board's most common answer, no reading.

Reading the board beats voting on it in every row: trained shared-board readers end 20 and 13 points above the vote over the same notes, and are misled 2 to 4 times in 100. But the own-note reader rescues almost as often (13 and 24 in 100): most of the gain is a second pass over the problem, not the teammates. The notes are the weak link. Asked for three lines, the model writes full solutions and 256 tokens cut most of them before the answer (right 30 to 57 of 100), so the board carries fewer answers than it could.

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py` into `results.json`.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [5] | run_std 0.013 from 3 re-runs (0.52, 0.54, 0.55); the seeds differ by 0.03 (baseline) and 0.01 (recipe) |
| Holdout is clean: `decontaminate(train, against=holdout)` | [5] | 0 of 512 train rows dropped |
| Reward is a program, not a judge | [5] | `MathEqual` (Math-Verify) on the reader's own answer; the board is never rewarded |
| Proxy vs target: `wai.compare(proxy=)` | [6] | `proxy=None`: the training reward is the target; over_optimized false |
| Length: mean completion length before -> after, per arm | [6] | 719 chars base -> 765 baseline, 877 recipe |
| Hack scan on the last training batch: `hack_scan` | [6] | top feature `contains:total`, a word from the problems; nothing endorsed |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | training seeds 17 and 18, `--seed 0` for the data; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | shared board vs own note, reader trained, notes from the current policy, 2 seeds per arm | baseline 0.65 / 0.68, recipe 0.68 / 0.67 | +0.029 [-0.023, +0.085], flat |

## Learned

- Copies of a 1.5B model do read a board: untrained readers gain 5 points from seeing all four notes, trained readers beat the board's vote by 13 to 20 points and are rarely misled. Unlike team-talk, nothing was trained away.
- Training the reader on the board does not beat training it on its own note: a second look at the problem does most of the work, and the shared board adds 3 points that two seeds cannot tell from zero.
- Next: train the writers too. Reward a note by whether the readers who saw it got the answer right (Park et al. [3]; leave-one-out credit), so notes get short and carry their answer. Then a harder set where one copy's insight matters more than a second pass.

Verified 2026-09-23, whileai 0.125, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. 248.7 GPU minutes over five containers, $8.29 on L40S (plus about $1.30 on a first run that lost a container). Run page: https://while.ai/platform/training/run_bbc9c4191065e453. Experiment: https://while.ai/platform/experiments/message-board

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Reasoning*.
2. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
3. Park, C. et al. MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning. arXiv:2502.18439, 2025.
4. Pappu, A. et al. Self-Organizing Agent Teams Learn to Reason Together. arXiv:2609.22682, 2026.
5. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
6. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
