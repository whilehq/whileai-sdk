# Softmax advantage: the group advantage is a softmax over rewards, not a z-score

**Paper:** SoftmaxGRPO: Learning to Reason using Softmax Advantage Group Estimation, Hernandez et al., arXiv:2608.09271, August 2026. https://arxiv.org/abs/2608.09271
**Book:** the group-relative advantage GRPO builds from a group of rollouts [1], what its standard-deviation denominator does to the scale of an update [2], and the rule that an RLVR gain on a Qwen base is read against a random-reward control before it is read at all [3].
**Claim:** z-scoring the rewards inside a group hands the one wrong answer on an easy prompt the same magnitude as the one right answer on a hard prompt, so GRPO spends its gradient on near-solved prompts; a temperature-scaled softmax over the group's rewards keeps every weight bounded, reallocates the budget toward the rare correct answers, and improves over GRPO under identical rewards (equation 1, section 3.1).
**The change:** the advantage of rollout `i` in a group of `M` is `M * softmax(r / tau)_i - 1` with `tau = 0.1`, instead of `(r_i - mean) / std`. Nothing else differs between the arms.

The softmax advantage scored +0.027 pass@1 over GRPO on 300 held-out GSM8K problems, with an interval of [-0.091, +0.144] across two training seeds per arm. That is **flat**. The random-reward control collapsed the model rather than lifting it, so on this base the verifier's gain is not a spurious-reward effect.

What you will learn: whether the softmax advantage moves pass@1 at 30 steps on a 1.5B instruction-tuned model; what a coin-flip reward does to the same base under the same trainer; why the token cap has to be set from the base's own replies before any arm trains.
Needs: a Modal account, `uv add whileai datasets modal`, and a `WHILEAI_API_KEY` for the run page (optional).
Takes: 186 GPU minutes on L40S across seven containers, about $6.04 at $1.95 an hour; about 40 minutes of wall clock with the six arm-seeds in parallel.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K (MIT), 512 train prompts from the train split; 300 held out, the test-split problems with the lowest sha256 of their text, pinned by content hash in `holdout.sha256` before any training (`t-e814e8cf` on the platform). The train and holdout hash sets are checked disjoint and the train set is decontaminated against the holdout by 8-gram.
2. Reward, both verifier arms: the binary outcome, `MathEqual` against the GSM8K gold number. A program, not a judge. A completion is capped at 1,024 tokens, and one cut at the cap scores 0 on the eval side.
3. Baseline arm: TRL 0.19.1 `GRPOTrainer` with a LoRA adapter (rank 32), 30 steps of 6 prompts x 8 rollouts (four per forward, twelve accumulation steps), lr 1e-4, clip 0.20 / 0.28 and KL 1e-3 (the paper's), temperature 1.0, one policy update per batch, per-sequence loss aggregation, the z-scored group advantage.
4. Recipe arm: the same run with equation 1 in place of the z-score. At `tau = 0.1` and a binary reward the `c` right rollouts of a group each get `(M - c) / c` and every wrong one gets `-1`: the lone success on a hard prompt gets +7 where the z-score gives +2.47, and the lone failure on an easy prompt gets -1 where the z-score gives -2.47.
5. Random arm: the same run paid a seeded Bernoulli(0.5) coin flip that never reads the completion, same steps and seeds. Shao et al. 2025 [4] showed random rewards move MATH on Qwen2.5 bases, and the book says to be suspicious of every RLVR gain on one for that reason [3]. The gain the reward buys is the delta over this arm: paired per task, `(r_i - b_i) - (x_i - b_i) = r_i - x_i`.
6. Eval: pass@1 on the same 300 held-out problems, 4 samples per task at temperature 1.0, two training seeds per arm (17, 18). The untrained base is evaluated three times first and that spread is the noise floor a delta has to clear. Every delta is paired over problems with a 95% interval (`wai.compare`), widened by the between-seed spread (`train_runs=`), with an exact two-sided sign test over the per-task pass rates pooled across seeds and the tie count beside it.

The paper trains the full model at lr 1e-6 for far longer; this recipe trains a LoRA adapter for 30 steps so each arm-seed fits in about half a GPU hour. Those cuts are the same for every arm. One update per batch keeps the run on-policy: the importance ratio is 1, the clip range never binds, and the KL term is the only regularizer [2]. An advantage change is reachable at one update; a clip change would not be (see the sibling `adaptive-clip`).

## Run

```bash
python recipe.py --selftest   # eq. 1 against the z-score, the sign test, the coin flip, the grader, the pin; no GPU, no key
python recipe.py --pilot      # one recipe arm, 3 steps, 24 tasks on Modal: the override reaches the loss (~$0.07)
python recipe.py --base-only  # the three base re-runs and their truncated share (~23 GPU minutes, ~$0.75)
python recipe.py              # base x3, three arms x seeds 17 and 18, results.json, the platform post (~$6)
python recipe.py --reuse      # results.json again from .cache/ or the rlvr-runs volume, no GPU
python recipe.py --arm recipe --seeds 19 --tau 0.3   # one arm, one more seed, another temperature
```

| Flag | Default | What it is |
|---|---|---|
| `--tau` | 0.1 | the softmax temperature of equation 1; the paper's for GSM8K |
| `--steps` | 30 | optimizer steps per arm |
| `--seeds` | 17 18 19 | training seeds per arm; the run above used 17 and 18 |
| `--n-holdout` | 300 | held-out problems, drawn by problem hash and pinned |
| `--k` | 4 | eval samples per task |
| `--max-completion` | 1024 | completion tokens, training and eval; the paper's 256 cut half the base's replies |
| `--beta` | 1e-3 | KL coefficient, the paper's |
| `--generations` | 8 | rollouts per prompt, `M` in equation 1 |
| `--est-minutes`, `--spent-usd` | 30, 0 | the spend guard: an arm projected past $10 is not started |
| `--no-post` | off | skip the platform post |

The Modal app runs detached and every container writes its return value to the `rlvr-runs` volume before returning, so a dropped laptop loses nothing; `--reuse` reads it back.

## Result

Run 2026-09-29, one L40S per container, 1,024-token cap, 30 steps, seeds 17 and 18.

| Arm | pass@1 seed 17 | 95% CI | pass@1 seed 18 | 95% CI | Mean over seeds | pass@4 (s17, s18) | Truncated | Mean reply (chars) | Steps | GPU min |
|---|---|---|---|---|---|---|---|---|---|---|
| Base, no training (3 re-runs) | 0.415 | [0.379, 0.452] | | | 0.415, 0.425, 0.413 | 0.75 | 0.2% | 875 | 0 | 22.9 |
| Baseline (GRPO, z-score) | 0.568 | [0.525, 0.611] | 0.604 | [0.560, 0.647] | 0.586 (sd 0.026) | 0.79, 0.83 | 0.7% | 1,215 | 30 | 58.4 |
| Recipe (softmax, tau 0.1) | 0.593 | [0.551, 0.638] | 0.632 | [0.590, 0.674] | 0.613 (sd 0.027) | 0.82, 0.86 | 0.4% | 1,168 | 30 | 67.4 |
| Random reward (Bernoulli 0.5) | 0.062 | [0.044, 0.083] | 0.180 | [0.153, 0.207] | 0.121 (sd 0.084) | 0.17, 0.43 | 0.0% | 235 | 30 | 37.3 |

n = 300 held-out problems for every row, k = 4 samples each. Per-arm intervals are the task bootstrap of `wai.pass_at`.

| Paired comparison | Seed-17 pair | Across both seeds (between-seed term) | Sign test (up / down / ties, p) | Verdict |
|---|---|---|---|---|
| **Recipe vs baseline** | +0.026 [-0.003, +0.055] | **+0.027 [-0.091, +0.144]** | 110 / 77 / 113, p = 0.019 | **flat** (inside the 0.028 noise band) |
| Baseline vs base | +0.153 [+0.119, +0.187] | +0.171 [-0.065, +0.406] | 177 / 45 / 78, p < 1e-18 | gain on every seed |
| Recipe vs base | +0.178 [+0.145, +0.213] | +0.198 [-0.048, +0.443] | 189 / 38 / 73, p < 1e-24 | gain on every seed |
| Random vs base | -0.353 [-0.393, -0.313] | -0.294 [-1.047, +0.459] | 19 / 208 / 73, p < 1e-40 | loss on every seed |
| **Recipe vs random (reward-corrected)** | +0.532 [+0.488, +0.577] | **+0.492 [+0.220, +0.763]** | 258 / 7 / 35, p < 1e-66 | moved |
| Baseline vs random (reward-corrected) | +0.506 [+0.462, +0.550] | +0.465 [+0.195, +0.735] | 251 / 8 / 41, p < 1e-62 | moved |

Recipe vs baseline: **+0.027 [-0.091, +0.144] across two seeds per arm, verdict flat.** The seed-17 pair alone reads +0.026 [-0.003, +0.055]. Its delta is under the 0.028 re-run band, so even that pair is noise on the eval's own terms. The sign test leans the recipe's way at p = 0.019, but 113 of 300 problems tie. The recipe was ahead on both seeds (0.593 against 0.568, 0.632 against 0.604), and the between-seed spread (0.026 and 0.027) is as large as the effect. `holdout_size` on the measured task spread says 300 paired problems resolve about +0.06 at 80% power, and a +0.026 needs about 801. The eval cannot tell a 2.6-point gain from none. Three or more seeds on an 800-problem holdout would resolve it.

**The random-reward control, and what it means here.** On Qwen2.5-Math bases a coin-flip reward raised MATH scores [4], which is why the book asks for this arm on any Qwen RLVR run [3]. On this instruction-tuned base, under this trainer, it did the opposite: pass@1 fell from 0.415 to 0.062 and 0.180. The seed-17 replies shrank to 22 characters on average, and the training-batch outcome fell from 0.64 over the first ten steps to 0.31 over the last ten. So the verifier arms' +0.15 to +0.20 over the base is not a spurious-reward artifact. The random arm moves the model the other way.

The reward-corrected delta the brief asks for is `delta_reward - delta_random`. That is **+0.492 [+0.220, +0.763]** for the recipe and +0.465 [+0.195, +0.735] for the baseline, across seeds. Read it as an upper bound: most of it is the control's collapse, not the reward's gain. The conservative reading is the uncorrected gain over the base, +0.18 (recipe) and +0.15 (baseline) on the seed-17 pair, since the control shows a random reward buys nothing positive here.

A likely reason the control collapsed, not tested here: with one update per batch the clip never binds, so the clipping bias that Shao et al. credit for the random-reward gain has nothing to act on [4]. Per-sequence loss aggregation gives short completions a larger per-token gradient [2], so a zero-mean random advantage has a net pull toward short replies. The hack scan's top feature on the random arm was `reply_length`, and its training replies fell from 218 tokens to 6. A run with `loss_type="dr_grpo"` on the random arm would test this.

**The token cap was the first finding.** The paper's 256-token cap cut 48 to 50% of this base's replies at temperature 1.0, and the base read 0.24 to 0.25 pass@1. At that cap every arm can win by learning to stop early, which confounds any advantage comparison with length. At 1,024 tokens the base truncates 0.2% of replies and reads 0.41 to 0.43. Every arm's truncated share stays under 1.3%. The 256-token run was stopped before any arm reached its eval and none of its numbers are reported.

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py`
into `results.json`.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [3] | **run_std 0.0066** from 3 re-runs (0.415, 0.425, 0.413); a delta under **0.028** is noise (`wai.noise_band(run_std, df=2)` = 4.30 x sqrt(2) x 0.0066). The platform posts the same floor as 4.03 points |
| Training seeds: `wai.compare(train_runs=)` | [3], [7] | 2 seeds per arm (17, 18); between-seed sd 0.026 baseline, 0.027 recipe, 0.084 random; the across-seed interval uses t(df = 2) = 4.30 |
| Holdout is clean: `decontaminate(train, against=holdout)` | [5] | **0 of 512** train rows dropped, no rule skipped; train and holdout problem hashes disjoint |
| Holdout is frozen | [3] | `holdout.sha256` e814e8cf…, checked before any number is read; split by the hash of the problem text |
| Reward is a program, not a judge | [5] | `MathEqual` (Math-Verify) against the GSM8K gold; the random arm's coin flip is seeded and never reads the reply |
| Truncation: a reply cut at the cap scores 0 | [2] | base 0.2%; baseline 0.7%, recipe 0.4%, random 0.0% at 1,024 tokens (48-50% at the paper's 256, which is why the cap moved) |
| Proxy vs target: `wai.compare(proxy=)` | [6] | `proxy=None`: the training reward is the target check; **over_optimized false** |
| Length: mean completion length before -> after, per arm | [6] | base 875 chars; baseline **1,215**, recipe **1,168**, random **235** |
| Hack scan on the last training batch: `hack_scan` | [6] | verifier arms: top feature `marker:outcome`, the reward itself (endorsed); random arm: `reply_length` (seed 17) and `contains:3 AND contains:/` (seed 18) |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | seeds 17 and 18 in the trainer, `--seed 0` for the train-prompt shuffle; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0, whileai 0.126 |

The KL to the start stayed at 0.0105 on both verifier arms at step 30. On the random arm it reached 2.09 (seed 17) and 0.11 (seed 18). Zero-variance groups averaged 0.51 of each batch on the baseline and 0.47 on the recipe: at a training outcome near 0.8, half of every batch carried no signal under either advantage.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: 256-token cap, 60 steps, three seeds per arm | base 0.24 / 0.25 / 0.24 with 48-50% of replies cut | stopped before any arm reached its eval: the eval measured finishing, not solving |
| 2 | cap raised to 1,024 tokens for training and eval | base 0.41 / 0.42 / 0.41, 0.2% cut | the base the arms are read against |
| 3 | out of memory at step 1-2: TRL 0.19.1 computes the reference log-probs over the whole 48-rollout batch in one forward (24 GiB of logits). The trainer subclass chunks that call by the microbatch; the microbatch is halved to 4 with twice the accumulation; 30 steps to fit the budget | baseline 0.568 / 0.604, recipe 0.593 / 0.632, random 0.062 / 0.180 | recipe vs baseline +0.027 [-0.091, +0.144], flat |

## Learned

- Measure the base's truncated share before setting the cap. The paper's 256 tokens suits its base-model setup. On an instruction-tuned 1.5B at temperature 1.0 it cut half of all replies, and GRPO would then have rewarded brevity on every verifier arm. The eval would have credited that to whichever advantage learned to stop first.
- The softmax advantage did what equation 1 says in the logged advantages: -1 floor, +7 for a lone success out of eight. It trained stably: KL 0.0105 like the baseline, and replies of similar length. At 30 LoRA steps its edge over the z-score is 2.6 points and inside the noise. The paper's gains come from full-model runs of hundreds of steps; this recipe cannot say whether the gap opens with more steps.
- The random-reward control is not a formality, and it does not always err in the flattering direction. Here it collapsed the model, which rules out a spurious-reward story for the verifier's gain. It also makes the "reward-corrected" delta mostly a measure of the control's damage. Report the uncorrected gain beside it.
- Next: three seeds and an 800-problem holdout for the recipe-vs-baseline delta (`holdout_size` asks for about 801 at this effect), and `loss_type="dr_grpo"` on the random arm to test the per-sequence length pull.

Verified 2026-09-29, whileai 0.126, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1, one L40S per container. 186.0 GPU minutes, $6.04 at the L40S rate for this run; $9.99 billed across every `rlvr-` app that day, including the stopped 256-token run, the pilots and two out-of-memory launches. Run page: https://while.ai/platform/runs?agent=softmax-advantage (the served version is `baseline-s17`; the platform verdict "recipe-s18 beats baseline-s17 by 6.4" pairs different seeds, and the same-seed and across-seed reads above are the result).

## References

1. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
2. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Reinforcement Learning*.
3. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
4. Shao, R. et al. Spurious Rewards: Rethinking Training Signals in RLVR. arXiv:2506.10947, 2025.
5. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
6. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
7. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
