# BPCO bounded critic: a critic that can only say a number the reward could be

**Paper:** Best Practice Critic Optimization, Penghui Qi, Xiangxin Zhou, Wee Sun Lee (NUS / Tencent Hunyuan), arXiv:2608.23566, August 2026. https://arxiv.org/abs/2608.23566
**Book:** the value function as the baseline a policy gradient subtracts, GAE as the way credit is handed backward, and the clipped ratio as the trust region, all in one chapter [1].
**Claim:** the standard PPO critic recipe degrades when there is one response per prompt because its critic is free to predict returns the reward can never pay, trains toward a lambda-return it has not learned yet, hands the policy a whitened advantage, and clips every token by the same ratio; bounding the critic to the reward range through an arctangent, training it on the Monte Carlo return after a warm-up, leaving the length-adaptive GAE advantage at its natural scale and clipping by probability shift instead makes critic-based training stable on one rollout per prompt, where it matches or beats group methods at 16 [2].
**The change:** the recipe arm's update is `wai.BPCO().update(batch)`, the paper's bundle; the baseline arm's is the standard PPO critic recipe [3] [4] (`baseline_update` in `recipe.py`, verl's defaults [5]). The paper's own comparison is the bundle against that recipe, so "the change" here is the bundle, five parts at once: (1) the head's raw output bounded to `(0, 1)` through `bound(z)` instead of used raw; (2) the critic target is the Monte Carlo return instead of the lambda-return at 0.95 with a clipped value loss; (3) the policy advantage is unnormalized GAE with `lambda = 1 - 1/(0.4 L)` instead of GAE 0.95 whitened over the batch; (4) the ratio is clipped to `1 -/+ 0.2/mu` (DPPO [6]) instead of `1 -/+ 0.2` (PPO); (5) the critic trains alone for 15 steps before the policy moves, instead of not at all.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split, `decontaminate` run against the holdout and the dropped count recorded. Reward, both arms: the binary outcome, `wai.verify.MathEqual()` against the GSM8K gold number. A program, not a judge.
2. The regime, both arms: one rollout per prompt, 64 prompts per optimizer step, 256 new tokens at most, temperature 1.0 with top-p 1.0, top-k off and no repetition penalty (Qwen's generation config sets all three; a truncated sampler has no common support with the policy). The rollouts are drawn by a **stale sampler**: a frozen copy of the policy's LoRA weights, refreshed only every `LAG = 4` optimizer steps, so a batch was written by a policy 1 to 4 updates old (the first block, and every warm-up step, is on-policy: there is no older policy to copy). Each trajectory carries `behavior_logprobs` from the sampler (a teacher-forced pass under its weights, right after generation, the way verl takes its old log-probs) and `logprobs` from the current policy at update time (one teacher-forced pass), plus `reward`. Only generated tokens are actions.
3. The critic, both arms: one linear layer on the trunk's last hidden state at the position that predicted each generated token (the same forward pass, hidden states detached, so the critic never trains the trunk), its own AdamW at 1e-4. The paper's 1e-5 (`BPCO_CRITIC_LEARNING_RATE`) is for a full separate value model; a 1,537-parameter head on a frozen feature needs more, and 1e-4 is the override. Both arms' heads share the same init (`set_seed(17)` right before the adapter and the head are built).
4. Baseline arm: the head's raw output is the value; the critic trains toward lambda-returns (GAE lambda 0.95, gamma 1) with verl's clipped value loss, clip 0.5; the advantages are whitened over the batch; the ratio against the rollout policy is clipped to `[0.8, 1.2]`; no warm-up. Written out in `baseline_update`: `coefficients[i][t]` is `rho A` where the unclipped branch of `min(rho A, clip(rho) A)` is the minimum and 0 where the clipped branch is, which is that surrogate's exact gradient at one step per batch (the selftest checks it against autograd).
5. Recipe arm: the head's raw output goes through `wai.BPCO().bound(z)`, done in torch with the same formula so the gradient flows (the selftest checks it against `bound()` on 2,401 points), and trains toward `update.value_targets`, the Monte Carlo return, with MSE; `critic_warmup` = 15 critic-only steps first, read off the object; then the policy step with `-(coef * logprob).sum() / N`, the coefficients from `wai.BPCO().update(batch)` held constant (the `1/N` is the batch mean the `Update` contract leaves to the trainer; the library's own toy trainer applies the same). The baseline uses the same loss line with its own coefficients, so the two arms differ in who wrote the coefficients and the value targets, and in nothing else.
6. Training, both arms: LoRA rank 16, alpha 32 (`TRAINING_LORA_RANK`, `TRAINING_LORA_ALPHA`) on the seven projections, AdamW at 1e-5 on the adapter (the paper's 1e-6, `BPCO_LEARNING_RATE`, is full weights), gradient clip 1.0, one step per batch, 40 policy steps after any warm-up, seed 17 set right before the trainer is built. Logged per step: mean reward, clipped token share, mean ratio, mean advantage, explained variance against the Monte Carlo return in both arms (equation 10 of the paper), completion length; the series are in `results.json` under each arm's `curve`.
7. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task at temperature 0.7 (same top-p, top-k and penalty settings as training). The untrained base is evaluated three times first, and that spread is the noise floor a delta has to clear. Paired delta with a 95% interval (`wai.pass_at`, `wai.compare` with `run_std`, `run_std_runs`, `train_runs` and `proxy=` naming the training reward). `hack_scan` on the last training batch of each arm.

## Run

```bash
python recipe.py --selftest   # the parser, the sampler bookkeeping, the bound, both update rules, the assembly; offline, no GPU and no key
python recipe.py              # both arms, one container each, sized for under 60 GPU minutes on one L40S
python recipe.py --arm recipe --steps 3 --critic-warmup 2 --n-holdout 16 --base-runs 1   # a few cents, proves the loop
```

The image installs the published `whileai` for its dependencies and mounts the checkout's own `whileai/` ahead of it on `PYTHONPATH`, because `wai.BPCO` is newer than the published wheel. Harmless once a release carries it. `WHILEAI_API_KEY` is optional; without it there is no run page and nothing else changes. The loop is plain torch and PEFT, no TRL: a single-rollout critic method with a stale sampler is not a trainer TRL ships.

## Result

Run 2026-09-22, both arms, one L40S each, at the settings above (round 2 in the Climb table; round 1 was the same run and lost its paired delta to a local slip after the GPU work).

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | 0.39 | [0.33, 0.46] | 0.60 | 0 | 0 |
| Baseline (standard PPO critic recipe) | 0.43 | [0.36, 0.50] | 0.66 | 40 | 25.1 |
| Recipe (the BPCO bundle) | 0.41 | [0.35, 0.48] | 0.73 | 15 warm-up + 40 | 26.3 |

Recipe vs baseline: **-0.017 [-0.069, +0.037]** over 120 paired tasks.
Verdict: **unresolved**. One training seed per arm; a second seed on each arm,
passed as `train_runs=`, would resolve it to moved or flat. The interval
covers zero on both sides and the delta is inside the eval's own re-run band
(0.026), so at this size the bundle cannot be shown to do anything to pass@1
either way. Round 1, the same configuration, read baseline 0.44 [0.37, 0.51]
and recipe 0.43 [0.36, 0.49]: every arm within the band of its counterpart
across the two rounds.

The result is flat, and the curves say why. At 1e-5 on a rank-16 adapter,
40 single-rollout steps move the policy so little that the ratio against a
sampler four updates stale never left `1 -/+ 0.0014`. The baseline's PPO clip
fired on 0.17% of tokens (mean over the run; peak 0.28%); the recipe's DPPO
range, `0.2 / mu` half-width, at least 0.2 wide and far wider on a rare token,
fired on 0.001%. Nothing in this run was ever in the regime where a clip
matters, so parts (4) and (5) of the bundle could not have moved a gradient
and the comparison is really parts (1) to (3): the bounded critic and its
target and advantage, against the unbounded one.

Those parts show in one number each. The unbounded head opened at explained
variance **-15.3**: its raw outputs sat far outside `[0, 1]`, and 40 steps of
clipped value loss brought it to -4.9, still worse than predicting the batch
mean. The bounded head opened at -0.36, could not be worse than the range
allows, and closed at -0.24 after 55 steps: its loss fell (0.37 to 0.31) but
it never beat the batch mean either. Bounding buys a critic that is wrong by at
most the range. It does not buy a critic that is right, and a critic that is
not right hands the policy an advantage that is mostly noise around the
reward in both arms. The recipe's unnormalized advantage averaged +0.027 per
token; the baseline's, whitened, 0 by construction.

Length moved more than anything else. Training rollouts went from 187 to
144 tokens in the baseline and from 194 to 104 in the recipe; eval answers
from 729 characters (base) to 592 (baseline) and 284 (recipe). The recipe
arm's pass@4 rose (0.60 to 0.73) while its pass^4 fell (0.18 to 0.12): its
samples got more varied, not more right. The reward did not follow the
shortening in either arm (first to last quarter of policy steps: 0.44 to 0.49
baseline, 0.46 to 0.50 recipe, both inside the batch-to-batch spread).

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py` into `results.json`.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [7] | **run_std 0.0042** from 3 re-runs (0.39, 0.39, 0.39); a delta under **0.026** is noise (`noise_band(run_std, df=2)` = 4.30 x sqrt(2) x run_std, one run per side; `compare(run_std=, run_std_runs=3)`). Round 1's three re-runs read 0.41, 0.39, 0.39, run_std 0.0103 |
| Holdout is clean: `decontaminate(train, against=holdout)` | [8] | **0 of 512 train rows dropped**, as expected for disjoint GSM8K splits; measured, not assumed |
| Reward is a program, not a judge | [8] | `MathEqual` against the public GSM8K gold number. No judge, no model in the loop |
| Proxy vs target: `compare(proxy=)` | [9] | **over_optimized false.** `proxy="marker:gsm8k_outcome"` is the training reward carried on the eval rows; it is the same binary program as the target at a different temperature, so its delta (-0.017 [-0.071, +0.033], no change) cannot part from pass@1's. Recorded, not a real over-optimization test |
| Length: mean completion length before -> after, per arm | [9] | **729 chars base -> 592 baseline, 284 recipe.** Training rollouts 187 -> 144 tokens baseline, 194 -> 104 recipe. Both shortened with no reward gain to show for it |
| Hack scan on the last training batch: `hack_scan` | [9] | top feature **none above the floor** in either arm; nothing endorsed, so this is the scan finding nothing |
| Pinned: seed, torch, transformers, peft | [the contract](../README.md#the-contract) | seed 17 in the trainer, `--seed 0` for the data split; torch 2.7.1, transformers 4.54.0, peft 0.16.0, no TRL; whileai 0.114 plus this branch's `wai.BPCO` |

The two arms share the data, the holdout, the reward, the sampler, the
adapter and head init, and every trainer knob except the update rule. Seed
17 held: step 1 of each arm was bit-identical between rounds 1 and 2, and the
recipe arm stayed identical through its warm-up and five policy steps before
generation nondeterminism parted them.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: the bundle against the standard critic recipe, LoRA r=16 at 1e-5, head at 1e-4, 64 x 1 per step, sampler lag 4, 40 policy steps after a 15-step warm-up | baseline 0.44 [0.37, 0.51], recipe 0.43 [0.36, 0.49] | paired delta not computed: the local step after both containers returned called a name the front door does not have (`wai.delta_report`; it is `wai.compare`), and the eval rows lived only in that process |
| 2 | **nothing on the GPU side.** The same run, with the eval rows saved to the volume, the raw returns written to disk before assembly, and the assembly under `--selftest` on fake returns | baseline 0.43 [0.36, 0.50], recipe 0.41 [0.35, 0.48] | -0.017 [-0.069, +0.037], unresolved |

Round 1 spent 50 GPU minutes and produced two numbers a reader can see and
no paired interval, because one local line ran for the first time after the
GPU work. That line is now exercised offline: `assemble()` takes the arms'
returns and `--selftest` feeds it fake ones, so the next slip in that path
costs seconds. The curves of the two rounds agree to the step where
nondeterminism parts them (baseline clipped share 0.17% both rounds; recipe
explained variance -0.36 to -0.27 then -0.36 to -0.24; lengths 187 -> 135 and
194 -> 113 in round 1 against 187 -> 144 and 194 -> 104 in round 2).

## Learned

- **The regime the paper is about was not reached, and the clip counters say so.** With the adapter at 1e-5 the policy moved so little in 40 single-rollout steps that a sampler four updates stale was still within `1 -/+ 0.0014` in ratio; PPO's clip touched 0.17% of tokens, DPPO's 0.001%. Two of the bundle's five parts (the clip and the warm-up before the policy moves) cannot change a gradient in that regime, so what this run compares is the bounded critic and its target and advantage against the unbounded one, and on pass@1 that comparison is flat (-0.017 [-0.069, +0.037]). A learning rate the adapter can move at (1e-4, where adaptive-clip's GRPO went from 0.34 to 0.47 in the same 40 steps) is the knob that would put the run where the paper's claim lives; it is a deliberate deviation from the shared single-rollout protocol and belongs in a round of its own.
- **Bounding the critic made it wrong by less, not right.** The unbounded head's explained variance started at -15 (its raw outputs were far outside the reward range, which is the paper's complaint in one number) and was still -4.9 after 40 steps of clipped value loss; the bounded head started at -0.36 and ended at -0.24. Neither beat the batch mean, so neither arm's advantage carried much beyond the reward itself. A linear head on a frozen feature at 1e-4, 64 sequences a step, is a weaker critic than the paper's separate value model, and the 15-step warm-up moved its loss from 0.37 to 0.32 without moving its explained variance. What would test the critic's part of the claim is a critic that learns: a longer warm-up, or a small trainable projection under the head.
- **Length is what moved, in both arms, with the reward flat.** Rollouts shortened by a quarter in the baseline and by nearly half in the recipe, eval answers from 729 to 592 and 284 characters, and pass@4 rose while pass^4 fell in the recipe arm. With a critic near 0.5 and a binary reward, every token of a wrong answer carries a negative residual and a short answer carries fewer of them; the unnormalized LA-GAE advantage does not subtract that off the way whitening does, which is one reading of why the recipe arm shortened faster. That is a hypothesis this run cannot separate from the learning rate, and a second seed per arm (`train_runs=`) comes before any of it: at one seed the verdict is unresolved and stays so.

Verified 2026-09-22, whileai 0.114 with this branch's `wai.BPCO`, plain torch 2.7.1 + PEFT 0.16.0 + transformers 4.54.0, no TRL. 51.4 GPU minutes, $1.71 on one L40S, two containers in parallel (baseline 25.1 minutes with the three base re-runs, recipe 26.3; round 1 about the same, its wall clock lost with its delta). Run page: none, no `WHILEAI_API_KEY` in this run; Modal apps `ap-q7nevqT6teeeDdUFFY6Gym` (round 1) and `ap-p3GjfFWYK6355Fg5OKP6HT` (round 2), adapters and per-step curves under `whileai-recipe-runs:/bpco-bounded-critic-<arm>-2026-09-21/`.

## Artifacts on Hugging Face

| what | repo |
|---|---|
| recipe arm (root) and `baseline/`, each with its `value_head.pt` and `curves.json` | [`while-ai/paper-bpco-bounded-critic-1.5b`](https://huggingface.co/while-ai/paper-bpco-bounded-critic-1.5b) |

The root of the model repo is the recipe arm the Result table reports; the baseline arm is the `baseline/` subfolder. Load either with `PeftModel.from_pretrained(base, repo, subfolder=...)`. Part of the [Papers, replicated](https://huggingface.co/collections/while-ai/papers-replicated-6ab271de22542eb550d4251c) collection in the while-ai org.

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Policy Gradient Algorithms*.
2. Qi, P., Zhou, X., Lee, W. S. Best Practice Critic Optimization. arXiv:2608.23566, 2026.
3. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
4. Schulman, J. et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438, 2016.
5. Sheng, G. et al. HybridFlow: A Flexible and Efficient RLHF Framework. arXiv:2409.19256, 2024.
6. Qi, P. et al. Rethinking the Trust Region in LLM Reinforcement Learning. arXiv:2602.04879, 2026.
7. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
8. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
9. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
