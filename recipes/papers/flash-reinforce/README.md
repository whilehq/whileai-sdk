# FlashREINFORCE: one stale rollout per prompt, corrected, gated, length-normalized

**Paper:** FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models, Yifan Hu et al., NVIDIA, September 2026 (no arXiv id yet). https://yifanzhang-pro.github.io/FlashREINFORCE/FlashREINFORCE.pdf (code: https://github.com/yifanzhang-pro/FlashREINFORCE)
**Book:** the REINFORCE policy gradient with a baseline, and the importance ratio that makes a sample from an older policy usable at all [1].
**Claim:** one rollout per prompt, trained on trajectories that lag the policy by several updates, learns stably when every token carries the learner-to-sampler ratio, the batch mean is the baseline, any trajectory whose mean sampled-action KL passes a trust region is masked whole, and every kept trajectory is weighted 1/T; the same stale rollouts trained without the correction degrade [2].
**The change:** the recipe arm's per-token coefficient is `wai.FlashReinforce().update(batch)` (ratio, trust gate, 1/T_i, 1/B); the baseline's is the batch-mean advantage over the batch's total token count, with no ratio and no gate.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split (different splits, and `decontaminate` still runs and counts).
2. Reward, both arms: the binary outcome, `MathEqual` against the GSM8K gold number. A program, not a judge. The paper changes the update, not the reward.
3. The regime, both arms: rollouts come from a **stale sampler**, a frozen copy of the policy's LoRA adapter refreshed only every 4 optimizer steps, and the next batch is drawn before the current update lands, the way an asynchronous trainer overlaps generation with training [2]. A batch is therefore 1 to 4 updates old (the first is 0; mean 2.4 over 40 steps). One rollout per prompt, 64 prompts per step, 256 new tokens, temperature 1.0 and top-p 1.0 (no truncation, so the ratio has common support). Each token's `behavior_logprobs` are the sampler's, teacher-forced under the sampler's weights right after it wrote them; `logprobs` are the policy's at update time, one teacher-forced pass. LoRA rank 16, alpha 32, AdamW at 1e-5 on the adapter, gradient clip 1.0, one full-batch step per batch and the batch is discarded. 40 steps. Seed 17, set right before the adapter is built, so both arms draw one init and the same first five batches (the sampler is not refreshed until step 4, and batch 5 is drawn before that update).
4. Baseline arm: the paper's uncorrected ablation. Batch-mean advantage `A_i = R_i - mean R`, token-mean loss: every token of trajectory `i` gets `A_i / (total tokens in the batch)`. No importance ratio, no trust gate, no `1/T_i`.
5. Recipe arm: `wai.FlashReinforce().update(batch)` at its defaults (trust 3e-3, `FLASH_REINFORCE_TRUST`): coefficient `m_i * A_i * rho_it / (T_i * B)`. Both arms minimize `-(coefficients * logprobs).sum()` with the coefficients held constant; the selftest pins the recipe's coefficients to the values `tests/api/test_flash_reinforce.py` pins. The FlashReinforce statistics (admitted share, sequence KL, ratio) are logged for both arms, so the baseline's staleness is measured even though it ignores it.
6. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task at temperature 0.7. The untrained base is evaluated three times first, and that spread is the noise floor a delta has to clear. Paired delta with a 95% interval (`wai.pass_at`, `wai.compare`), the training reward named as `proxy` (it is the target, so `None`), `hack_scan` on the last training batch.

The learning rate is an override: the paper's 1e-6 (`FLASH_REINFORCE_LEARNING_RATE`) is an AdamW step on full weights, and a rank-16 adapter takes a larger one. 1e-5 is the rate the SAO and BPCO recipes share, so the three single-rollout methods compare on one protocol. `FLASH_REINFORCE_LEARNING_RATE_LORA` (1e-4) is the library's untested adapter convention; `--learning-rate` sets either.

## Run

```bash
python recipe.py --selftest   # the reward, the lag bookkeeping, both coefficient rules, offline
python recipe.py              # both arms, 44 GPU minutes on one L40S, $1.48
python recipe.py --arm recipe --learning-rate 1e-4 --lag 8
```

## Result

Run 2026-09-22, both arms, on one L40S, as the protocol above.

| Arm | pass@1 | 95% CI | pass@4 | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | 0.40 | [0.33, 0.47] | 0.58 | 0 | 0 |
| Baseline (uncorrected stale REINFORCE) | 0.47 | [0.41, 0.55] | 0.68 | 40 | 22.9 |
| Recipe (`wai.FlashReinforce`) | 0.41 | [0.34, 0.48] | 0.63 | 40 | 21.3 |

Recipe vs baseline: **-0.069 [-0.115, -0.025]** over 120 paired tasks. Verdict:
**unresolved**. One training seed per arm; a second seed on each arm, passed
as `train_runs=`, would resolve it to moved or flat. The interval excludes zero
and the delta clears the eval's re-run band (0.055), so on the eval checks
alone this reads as the recipe arm being worse; the training-seed check is
what stops that from being a verdict, and the adaptive-clip recipe next door
has seen an 11-point sign flip between two identical one-seed runs.

What the arms measured, from the per-step log (`arms.<arm>.training` in
`results.json`):

- **The rollouts were barely stale.** At 1e-5 on a rank-16 adapter, four
  updates move the policy very little: the mean sampled-action KL between
  sampler and learner per step was 2.9e-4 in the recipe arm and 3.7e-4 in
  the baseline (max over the run 9.1e-3 on one trajectory), and the mean
  ratio stayed within 0.9998 to 1.0010 of 1. The trust gate at 3e-3 masked
  one of 64 trajectories on 5 of 40 steps (admitted share 0.998 mean, never
  below 0.984); it would have masked one or two on 8 steps of the baseline.
  At lag 0 (step 1) the KL is exactly 0 and the ratio exactly 1, which is
  the check that the behavior log-probabilities are the sampler's own.
- **The uncorrected baseline did not degrade.** It went from 0.40 to 0.47 on
  the holdout and its training reward climbed from 0.44 to the mid 0.50s
  (mean 0.49, last five steps 0.52 to 0.63). With the ratio this close to 1
  and the gate this quiet, the correction had almost nothing to correct.
- **What differed was the length weighting.** With the ratio inert, the two
  arms differ in `1/T_i` against a token mean. The token-mean baseline gives
  a long wrong trajectory more total push than a short one, and its
  completions shortened from 202 to 130 tokens over training (455
  characters after, from 712 before). The recipe weights every trajectory
  the same, its completions stayed at 177 tokens (663 characters), and the
  hack scan's strongest pooled feature on its last batch is `truncated`
  (r -0.58): the rollouts that ran into the 256-token cap and never gave a
  number. Its training reward averaged 0.45 (last five steps 0.42 to 0.55).

So the paper's regime was not reached: the claim is about rollouts stale
enough to need a correction, and at this adapter learning rate a lag of four
is not that. The recipe arm was measured in it anyway, and at this size the
paper's change cost 7 points against its own ablation, with the length
weighting the visible mechanism.

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py`
into `results.json`. These are today's numbers.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [3] | **run_std 0.0091** from 3 re-runs (0.40, 0.38, 0.40); a delta under **0.055** is noise (`noise_band(run_std, df=2)` = 4.30 x 0.0091 x sqrt(2), one run per side, t at n - 1 because run_std is an estimate; `compare(run_std=, run_std_runs=3)`). This is the eval's re-run noise, not the training's |
| Holdout is clean: `decontaminate(train, against=holdout)` | [4] | **0 of 512 train rows dropped**, as expected for disjoint GSM8K splits; measured, not assumed |
| Reward is a program, not a judge | [4] | `MathEqual` against the public GSM8K gold number. No judge, no model in the loop |
| Proxy vs target: `compare(proxy=)` | [5] | `proxy=None`: the training reward *is* the target metric, the same binary check, so there is no proxy to over-optimize; `over_optimized` false |
| Length: mean completion length before -> after, per arm | [5] | **712 chars base -> 455 baseline, 663 recipe** (202 -> 130 and 202 -> 177 tokens in training). The baseline shortened and gained; the recipe kept its length and did not |
| Hack scan on the last training batch: `hack_scan` | [5] | one rollout per prompt, so the within-ask column the scan ranks by is empty and the pooled column is what is left: recipe **`truncated` (pooled r -0.58)**, baseline `contains:. therefore` (r +0.35). Nothing is endorsed; truncation is the failure mode, not a reward surface |
| Pinned: seed, torch, transformers, peft | [the contract](../README.md#the-contract) | seed 17 for the adapter and the batch order, `--seed 0` for the data split; torch 2.7.1, transformers 4.54.0, peft 0.16.0; no TRL, the loop is HF `generate` + PEFT + AdamW in `recipe.py` |

The two arms share the data, the holdout, the reward, the adapter init, the
sampler schedule and the first five batches; they differ in the coefficient
that multiplies each token's log-probability and in nothing else.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the protocol: lag 4, 64 x 1 rollouts, 256 tokens, LoRA r=16 at 1e-5, trust 3e-3, 40 steps | baseline 0.47, recipe 0.41 | -0.069 [-0.115, -0.025], unresolved |

The next knob is the one that reaches the regime: `--learning-rate 1e-4`
(the library's adapter convention) or `--lag 8` (the paper's measured lag on
its 30B run) both move the sampler far enough per refresh that the KL
approaches the gate and the ratio does real work, and only then does the
baseline have something to degrade from. Before either is worth a GPU
minute, the current setting needs a second training seed per arm.

## Learned

- The stale sampler costs nothing to build: a frozen copy of the adapter's tensors, swapped in to generate and out to train, with the behavior log-probabilities teacher-forced under the swapped-in weights. Step 1 at lag 0 reports a KL of exactly 0 and a ratio of exactly 1, which is the one line that says the bookkeeping is right, and it is worth printing before anything else.
- `wai.FlashReinforce().update(batch)` is the whole update: the loop hands it 64 dicts of `reward`, `logprobs` and `behavior_logprobs` and multiplies its coefficients into a padded log-probability tensor. The baseline is the same loop with a different coefficient list, so the two arms cannot drift apart in anything but the rule.
- A correction can only be tested where there is something to correct. At 1e-5 on a rank-16 adapter the four-update lag left the ratio within a tenth of a percent of 1 and the gate touching one trajectory in five steps, so the paper's stability claim was not exercised, and what the delta measured instead is the paper's `1/T_i` against the token mean, which at this scale kept long truncated failures alive that the baseline learned to cut. Check the KL the log prints on the first refreshed step before reading the delta.

Verified 2026-09-22, whileai 0.114 (this branch's source tree, mounted into the container: the wheel on the index does not carry `FlashReinforce` yet), HF transformers 4.54.0 + PEFT 0.16.0 on torch 2.7.1. 44.3 GPU minutes, $1.48 on one L40S. Run page: none (no `WHILEAI_API_KEY` in the environment; the Modal app is `ap-xW5lCwuwsVAis7rgzAypL6`).

## Artifacts on Hugging Face

| what | repo |
|---|---|
| recipe arm (root) with `history.json`; the baseline adapter was not kept | [`while-ai/paper-flash-reinforce-1.5b`](https://huggingface.co/while-ai/paper-flash-reinforce-1.5b) |

Part of the [Papers, replicated](https://huggingface.co/collections/while-ai/papers-replicated-6ab271de22542eb550d4251c) collection in the while-ai org.

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Policy Gradient Algorithms*.
2. Hu, Y., Zhang, Y., Zhang, J., Xu, Y., Zhang, Y., Peng, W., Yu, K., Molchanov, P., Kautz, J., Dong, Y. FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models. NVIDIA, September 2026. https://yifanzhang-pro.github.io/FlashREINFORCE/FlashREINFORCE.pdf
3. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
4. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
5. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
