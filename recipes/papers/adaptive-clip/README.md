# Adaptive clip: the upper bound follows how rare a correct answer was

**Paper:** Group Adaptive Clipping Policy Optimization, Sheng Jia et al., arXiv:2609.00444, August 2026. https://arxiv.org/abs/2609.00444
**Book:** the clipped surrogate objective of PPO [1] as GRPO applies it [2], and what the width of the clip range does to an update the group thinks is worth making [3].
**Claim:** GRPO clips every rollout against the same upper bound, so the one correct answer in a hard group and the seventh correct answer in an easy group are held back equally; letting the bound widen as correct answers get rarer gives the informative rollouts more room and raises pass@1 and pass@k on math and code.
**The change:** the upper clip bound is computed per group from how many of its rollouts were right, instead of being one fixed number.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split (different splits, so there is no overlap to check for).
2. Reward, both arms: the binary outcome, `MathEqual` against the GSM8K gold number. A program, not a judge. The paper changes the clip, not the reward, so nothing here is shaped.
3. Baseline arm: GRPO with `epsilon` 0.20 and `epsilon_high` 0.28, fixed for every rollout. That pair is DAPO's clip-higher [4] and it is the paper's own token-level default.
4. Recipe arm: same 0.20 floor and the same 0.28 ceiling, but the upper bound slides per group, `eps_hi(c) = eps_lo + (eps_hi_max - eps_lo) * (k - c) / (k - 1)` with `c` correct out of `k = 8` rollouts. One right out of eight keeps the full 0.28; seven right gets 0.2114.
5. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task. The untrained base is evaluated three times first, and that spread is the noise floor a delta has to clear; the train set is decontaminated against the holdout before any training. Paired delta with a 95% interval (`wai.pass_at`, `wai.delta_report`).

Both arms share the floor and the ceiling, so the comparison isolates the sliding and not the width. There is no KL term (`beta` 0), which leaves the clip as the only trust region in the run — the thing the paper is about. The recipe runs the paper's token-level importance sampling, not its sequence-level GSPO variant, so its Seq-IS epsilons (3e-3 / 5e-3) do not apply here.

## Run

```bash
python recipe.py --selftest   # the clip schedule and the clamp, offline, no GPU and no key
python recipe.py              # both arms, sized for under 60 GPU minutes on one L40S
python recipe.py --arm recipe --eps-high-max 0.36
```

## Result

Run today, both arms, on one L40S, at the settings of round 2. Round 1 ran the
clip where it cannot bind, so it could not have tested the paper. See Climb.

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | 0.34 | [0.28, 0.41] | 0.58 | 0 | 0 |
| Baseline (fixed upper bound 0.28) | 0.47 | [0.40, 0.54] | 0.70 | 40 | 12.8 |
| Recipe (bound slides with the group) | 0.52 | [0.45, 0.59] | 0.74 | 40 | 7.6 |

Recipe vs baseline: **+0.050 [0.000, 0.100]** over 120 paired tasks.
Verdict: **unresolved**. One training seed per arm; a second seed on each arm,
passed as `train_runs=`, would resolve it to moved or flat. The interval
reaches zero, so this is not a gain you can bank either way. Read it as: at
this size, sliding the bound still cannot be shown to do anything.

**Read the Climb table before you read this one.** The previous run of this
exact configuration reported -0.065 [-0.117, -0.013] — the recipe 6.5 points
*worse*, with an interval that excluded zero. Today the same code, the same
data and the same knobs returned +0.050. The delta moved 11.5 points and
changed sign between two runs of the same experiment.

The seed was not held on one arm, and the run pages say which. TRL 0.19.1
builds the LoRA adapter before it applies `GRPOConfig.seed`, and the baseline
arm runs the three base evals first, which advances the RNG before its
adapter is drawn. The recipe arm, whose adapter is drawn from a fresh state, is
bit-identical between rounds 2 and 3 through step 3 and finished 0.508 then
0.517; the baseline arm differs from step 1 and went 0.573 then 0.467. So 10.6
of the 11.5 points is one arm's LoRA init plus generation nondeterminism, and
0.8 is the other arm's. `recipe.py` now calls `set_seed(17)` right before
building the trainer; the next verify run is the first with both arms on one
init.

That is the finding. The eval's own noise floor is 0.024 (Checks table), a
three-run sample sd at p = 0.34, where the binomial floor over 480 samples is
about 0.023 per arm; even on that honest floor the swing is about 2.5 sigma,
and the two rounds' paired intervals do not overlap (z about 3.1). What moved
is the *training*, not the measurement. Two arms at one training seed cannot separate
a five-point clip effect from run-to-run training variance, and this recipe now
has the direct evidence rather than the suspicion.

What survives both runs is that GRPO itself worked: the base goes from 0.34 to
somewhere in 0.47-0.57 in 40 steps, whichever arm you look at, and that is much
larger than anything the clip schedule has been shown to do here.

The selftest also turned up one thing worth knowing before you read the paper's
equation 11 literally: at `c = 0` it returns 0.2914, above the 0.28 ceiling it
is supposed to stop at. The equation is written for a group that splits,
`1 <= c <= k`. All-wrong groups have a zero advantage and contribute no
gradient, so the recipe clamps the count into `[1, k]` and they land on the
ceiling instead of over it.

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py`
into `results.json`. These are today's numbers.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [5] | **run_std 0.0087**, so a delta under **0.024** is noise. This measures re-running the eval, not re-running the training — and the training is where this recipe's variance turned out to live |
| Holdout is clean: `decontaminate(train, against=holdout)` | [6] | **0 of 512 train rows dropped**, as expected for disjoint GSM8K splits — measured, not assumed |
| Reward is a program, not a judge | [6] | `MathEqual` against the public GSM8K gold number. No judge, no model in the loop |
| Proxy vs target: `delta_report(proxy=)` | [7] | `proxy=None`: the training reward *is* the target metric, the same binary check, so there is no proxy to over-optimize |
| Length: mean completion length before -> after, per arm | [7] | **725 chars base -> 413 baseline, 542 recipe.** Both arms got shorter and more right, so neither is winning on length |
| Hack scan on the last training batch: `hack_scan` | [7] | top feature `n:digits` (baseline batch: `contains:week`) — GSM8K arithmetic surface, not a reward surface. Nothing is endorsed, so this is the scan reporting it found nothing |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | seed 17 in the trainer, `--seed 0` for the data split; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

The two arms share the data, the holdout, the reward and every trainer knob
except the clip bound. In rounds 1 to 3 they did not share the LoRA init (see
above); from the next verify run they do.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: eps 0.20 / 0.28, k = 8, 40 steps, lr 1e-4, LoRA r=32, 1 policy update per batch | baseline 0.61, recipe 0.59 | -0.013 [-0.054, +0.031], flat |
| 2 | same, but 2 policy updates per batch (`--num-iterations 2`) so the clip can bind at all; at a fixed 40 steps this halves the rollouts (120 prompts seen instead of 240) | baseline 0.57, recipe 0.51 | -0.065 [-0.117, -0.013], flat (wrong way) |
| 3 | **nothing.** Round 2 re-run unchanged, to check the number before climbing off it | baseline 0.47, recipe 0.52 | +0.050 [0.000, 0.100], flat |

**Round 1 could not have tested the paper, and the trainer's own logs say so.**
TRL takes one policy update per batch of rollouts by default. On that update
the sampling policy and the trained policy are the same, so the importance
ratio is exactly 1: "the policy ratio starts at 1 for the
first gradient step for that batch" [3]. A ratio of 1 never reaches a bound of
1.20 or 1.28, so `epsilon_high` is dead weight and a *per-group*
`epsilon_high` is dead weight per group. `clip_ratio/high_mean` was 0.0 in all
40 logged steps of round 1. The -0.013 it reported was generation
nondeterminism between two arms running the same arithmetic.

Round 2 takes the usual "1-4 gradient steps per batch" [3] at 2, which is the
smallest change that lets the second update be off-policy. The clip does then
fire. How hard it fires is itself unstable between runs: round 2 logged
`clip_ratio/high_mean` between 0 and 0.0003, round 3 logged a mean of 0.0021
and a peak of 0.0125 on the same settings (these are from the trainer's console
log; the run pages do not carry the `clip_ratio` series, so they cannot be
checked there yet). Both rounds clip on exactly 20 of 40 logged steps, which is
the setup working as intended — with two updates per batch the first is
on-policy and cannot clip, the second can.

**Round 3 changed nothing on purpose.** The rule for this directory is to
re-run the recipe as written and check the number before climbing off it, and
that is what caught the problem: the delta moved 11.5 points and changed sign.
So the next knob is not a knob. Before `eps_hi_max` or more updates per batch
is worth a GPU minute, this recipe needs several training seeds per arm and a
delta reported across them: the swing between two identical runs (0.115) is
about 4.8 times the eval band (0.024), and the clip effect it is trying to
measure is a few points.

## Learned

- The bound can be made per-group without touching TRL's loss body. `epsilon_high` is read inside `_compute_loss`, and `torch.clamp` takes tensor bounds, so handing it a (batch, 1) tensor broadcasts over the (batch, tokens) ratio and the `clip_ratio` metric TRL logs stays correct.
- Reading the group's correct count off the sign of the advantage only works because this reward is binary: with rewards in {0, 1} the advantage is `r - c/k`, positive for exactly the correct rollouts. A shaped reward would break that and need the counts carried separately.
- Holding the ceiling equal across the arms is the honest comparison but it is also the conservative one: it makes the recipe a strictly tighter clip than the baseline. The paper compares against a fixed bound too, on a much bigger batch (256 prompts against 6 here), so a flat result could mean the batch rather than the idea.
- **Check that your change is reachable before you spend a GPU hour on it.** The per-group bound was implemented correctly, tested on the CPU against real TRL, and verified to arrive at the loss intact — and still could not move a gradient, because nothing in an on-policy run ever asks what the upper bound is. One line of the trainer's own logging (`clip_ratio/high_mean`) would have said so before the run. It is now in the Checks a reader can see.
- **GRPO was the intervention that mattered here.** Both arms moved the base from 0.34 to 0.47-0.57 on GSM8K in 40 steps, which dwarfs everything the paper's change could have done at this scale. A third of groups were flat on average today (`frac_reward_zero_std` mean 0.30, peaking at 0.67), so most of that came from the rollouts that split.
- **A two-arm delta at one training seed per arm is not a measurement, and this is what that looks like.** Re-running this recipe unchanged moved the delta from -0.065 [-0.117, -0.013] to +0.050 [0.000, 0.100]: 11.5 points, sign flipped, and the first run's interval excluded zero on the wrong side. Nothing in the recipe changed; the differences were whileai 0.53 -> 0.64, ordinary nondeterminism in generation and kernel scheduling, and, on the baseline arm only, an unseeded LoRA init (TRL builds the adapter before it seeds; fixed in `recipe.py` after this round). The eval-noise floor this directory enforces (0.024 here) is the noise of *re-running the eval on a fixed model*, and it says nothing about the noise of re-running the training. Where a recipe's whole claim is a delta between two trained models, that second source is the one that decides whether there is a result, and it needs several training seeds per arm to see at all. Read every one-seed delta in this directory — including the ones with tight intervals — with that in mind.

Verified 2026-09-18, whileai 0.64, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. 20.4 GPU minutes, $0.68 on one L40S (round 2: 32.6 minutes, $1.09; round 1: 43.1 minutes, $1.44). Run page: https://while.ai/platform/training/run_000dac972e53b532

## References

1. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
2. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
3. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Reinforcement Learning*.
4. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
5. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
6. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
7. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
