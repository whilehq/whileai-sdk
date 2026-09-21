# Filter metric: phantom advantages under a shaped reward

**Paper:** The Filter Metric is Safety-Critical: Phantom Advantages in Group-Relative RL under Shaped Rewards, Juntao Yu, arXiv:2609.13866, September 2026. https://arxiv.org/abs/2609.13866
**Book:** the training reward is a proxy, and this recipe is a case where the proxy is optimized hard while the thing you wanted does not move [1], so the shaped score is declared as `proxy=` and over-optimization is a verdict, not a guess. The filter itself is a question about the group baseline: GRPO's group-relative advantage [2] and DAPO's dynamic sampling [3].
**Claim:** When the reward is a 0/1 outcome plus a shaping term, dropping "flat" rollout groups by the shaped score instead of by the outcome collapses GRPO training; the paper reports 0.040 exact match against 0.754 on GSM8K with Qwen2.5-1.5B, same trainer, same everything else.
**The change:** the group-is-flat test reads the binary outcome instead of the shaped score.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split. Different splits, so overlap should be nil, but `decontaminate` runs anyway and the count goes in the Checks table — "should be nil" is not a measurement.
2. Reward, both arms: `outcome - 0.30 * min(len/512, 1)`. The outcome is `MathEqual` against the GSM8K gold number, so the reward is a program. The length term is the shaping the paper attacks: among wrong answers, the shortest one scores best.
3. Baseline arm: drop a group when its **shaped scores** are all equal (`--filter-metric score`). A group of four wrong answers of different lengths is not all-equal, so it survives, and GRPO's divide-by-group-std turns those length crumbs into full-size advantages. Those are the phantom advantages — gradient that looks like signal but only encodes "be shorter".
4. Recipe arm: drop a group when its **binary outcomes** are all equal (`--filter-metric outcome`). All-wrong is now flat, so the group is dropped and teaches nothing. This is what DAPO's dynamic sampling means [3].
5. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task. The untrained base is evaluated several times, not once (`--base-runs`, 3 by default, 10 for the numbers below), so the spread between those runs is the noise floor the delta has to clear; then each arm once, paired delta with a 95% interval (`wai.pass_at`, `wai.eval_variance`, `wai.delta_report(run_std=, run_std_runs=)`). The band uses the t quantile at `run_std_runs - 1` because the floor is an estimate.

Dropping is done by masking: a group whose rewards are all equal gets a zero advantage, so it contributes nothing. The paper's DAPO arm deletes the group and refills the batch with fresh prompts. Masking reproduces the advantage-level effect on a fixed batch; it does not reproduce the refill, so this recipe cannot say anything about the paper's refill-rate numbers.

## Run

```bash
python recipe.py --selftest   # the filter on hand-written groups, offline, no GPU and no key
python recipe.py              # both arms, sized for under 60 GPU minutes on one L40S
python recipe.py --arm recipe --steps 80
```

## Result

Run today, both arms, on one L40S. Round 2 is the headline: same change, at
16 prompts per step instead of 8, which is the direction of the paper's own
batch. See Climb.

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | 0.36 | [0.29, 0.43] | 0.57 | 0 | 0 |
| Baseline (filter on shaped score) | 0.39 | [0.33, 0.45] | 0.68 | 40 | 30.0 |
| Recipe (filter on binary outcome) | 0.46 | [0.39, 0.53] | 0.70 | 40 | 21.8 |

Recipe vs baseline: **+0.067 [+0.021, +0.113]** over 120 paired tasks.
Verdict: **unresolved**. One training seed per arm; a second seed on each arm,
passed as `train_runs=`, would resolve it to moved or flat (the eval checks
below all pass, and they measure the eval, not the training). The interval
excludes zero, the delta clears the eval's
own re-run band (run_std 0.0078 from 10 base re-runs, band 0.025 = t(df=9)
2.26 x sqrt(2) x run_std on a difference of two re-run draws), and the proxy
check is clean: the shaped score went *down* -0.035 [-0.080, +0.009] while
pass@1 went up, which is the opposite of over-optimization.

The band was re-read on 2026-09-18. The 2026-09-17 run estimated run_std
0.0169 from three base re-runs and the verdict used the 1.96 band, 0.047.
That band treats a three-run estimate as the eval's exact spread; the honest
quantile at df=2 is 4.30, band 0.103, and under it +0.067 is **flat**. The
base was re-evaluated ten times (no training re-run: the arms and their
delta stand). The ten pass@1 draws were 0.36, 0.34, 0.34, 0.34, 0.34, 0.35,
0.34, 0.34, 0.36, 0.35: run_std 0.0078, band 0.025 at df=9, and +0.067 clears
it. The three-run estimate was more than twice the ten-run one, which is what
a standard deviation from three draws does.

The mechanism the paper describes is visible in the lengths. The
score-filtered arm ends at **234 characters** of mean completion, the
outcome-filtered arm at **496**, from the same 728-character base. The
baseline arm spent its training learning that shorter is better — which is
what the length term rewards when every rollout in the group is wrong and the
group's std collapses onto length noise.

What the paper reports is a collapse: 0.040 exact match for score-filtering
against 0.754 for outcome-filtering on GSM8K with this same base model. This
recipe sees a 6.7-point gap, not a 70-point one. 40 steps of LoRA is not the
paper's run, and the direction is what transfers at this size, not the size of
the damage.

`python recipe.py --selftest` shows the one change doing what the paper
describes, offline: on a group of four wrong answers that differ only in
length, the score filter leaves rewards `[-0.001, -0.212, -0.002, -0.212]`
(kept, and the shortest wrong answer wins) while the outcome filter flattens
them to `[-0.106, -0.106, -0.106, -0.106]` (advantage zero, nothing learned).
On a group that really splits, one right and three wrong, both filters keep it.

## Checks

Every cell is written by `recipe.py` into `results.json`. These are the round 2 numbers, and `--selftest` exercises the same paths on synthetic rows offline.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 10 times, `eval_variance` run_std | [4] | **run_std 0.0078 from 10 re-runs**: a delta under 0.025 (`noise_band(run_std, df=9)` = 2.26 x sqrt(2) x run_std, the band on a difference of two re-run draws, t because run_std is an estimate) is noise. The measured +0.067 clears it. On the original 3 re-runs (run_std 0.0169) the honest band was 0.103 and it did not |
| Holdout is clean: `decontaminate(train, against=holdout)` | [5] | **0 of 512 train rows dropped**, prompt-keyed against the holdout as the note below requires |
| Reward is a program, not a judge | [5] | `MathEqual` against the GSM8K gold number, plus a length term. No model in the reward path |
| Proxy vs target: `delta_report(proxy=)` | [1] | **not over-optimized.** `marker:shaped_reward` -0.035 [-0.080, +0.009] while pass@1 +0.067: the target moved and the proxy did not follow it up |
| Length: mean completion length before -> after, per arm | [1] | **728 chars base -> 234 baseline, 496 recipe.** The score-filtered arm halved its output twice over; this is the phantom advantage leaving its fingerprint |
| Hack scan on the last training batch: `hack_scan` | [1] | top feature **`marker:shaped_reward`** in both arms, which is the scan correctly naming the shaped term as the thing the training reward tracks. Nothing is endorsed |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | seed 0; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

Two notes on the checks, because a check that cannot fail is worse than no check. `decontaminate(against=)` reads its eval texts from `prompt`, so handing it the holdout question-keyed finds nothing and passes silently; the recipe passes it prompt-keyed and `--selftest` asserts that a shared prompt is actually caught. And the proxy check is only real if the eval rows carry the training reward — `--selftest` asserts that marker is present rather than trusting it.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | as the paper: lambda 0.30, 40 steps, lr 1e-4, LoRA r=32, 8 prompts per step | baseline 0.36, recipe 0.41 | +0.050 [-0.006, +0.106], flat |
| 2 | same, 16 prompts per step (toward the paper's 32) | baseline 0.39, recipe 0.46 | +0.067 [+0.021, +0.113], **moved** |
| 2b | no training; the base re-evaluated 10 times instead of 3, band read at t(df=9) instead of 1.96 | base 0.36 (was 0.37) | +0.067 unchanged; band 0.047 (3 runs, z) -> 0.103 (3 runs, t, **flat**) -> 0.025 (10 runs, t, **moved**) |

Round 2b is a correction, not a climb. The 2026-09-17 verdict rested on a
1.96 band around a run_std estimated from three draws, which is not a 95%
band: at df=2 it passes about 19% of pure-noise deltas. Read honestly, round
2 was flat (0.067 against 0.103). Ten base re-runs put the floor at 0.0078
and the t band at 0.025, and round 2 is moved again on the same arms. Round
1's "cleared the noise floor" below used the same 1.96 band and did not clear
the honest one either (0.050 against 0.103).

Round 1 was flat by 0.006 of interval: the delta cleared the 1.96 noise floor
(0.050 against a 0.046 band) but its interval still touched zero. Round 2
doubled the prompts per optimizer step, which is the knob this recipe was cut
down on — the paper runs 32 — and the same change came back separated from
zero. Nothing about lambda or the filter itself changed between the rounds.

The paper's own knob is lambda, the weight on the length penalty; it sweeps
0.1, 0.3 and 0.5 and reports the collapse at 0.30, the default here. That is
round 3, and it now has a measured effect to grow rather than a flat one to
rescue.

## Learned

- The filter metric is a separate choice from the reward, and a trainer will let you set it to the shaped score without complaining. Nothing in the loss curve says which one you picked.
- Shaping and group-std normalization interact: a shaping term too small to matter on its own becomes a full-size advantage once every rollout in the group is wrong and the std collapses to the shaping noise.
- **The batch was the difference between flat and moved, and neither round changed the idea being tested.** 8 prompts per step gave +0.050 [-0.006, +0.106]; 16 gave +0.067 [+0.021, +0.113]. A recipe cut down to fit a GPU hour can report "flat" about an effect that is really there, which is why the Climb table exists and why round 1 shipped with its numbers instead of being retried until it looked good.
- **A run_std from three draws is not the eval's spread, and 1.96 around it is not a 95% band.** The same +0.067 was moved under z, flat under t at df=2, and moved again once ten re-runs shrank the estimate from 0.0169 to 0.0078. The verdict did not change because the effect changed; it changed because the floor was measured properly. `delta_report(run_std=, run_std_runs=)` and `check.py` now carry the degrees of freedom so this cannot happen silently.
- **The length column is the evidence, not the pass@1 column.** Score-filtering ended at 234 characters against outcome-filtering's 496 from the same base. The paper says a wrong group survives the filter and teaches the model that the shortest wrong answer is the good one; a 2x length collapse in exactly the arm that keeps those groups is what that looks like from outside.

Verified 2026-09-17, whileai 0.53, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. 51.8 GPU minutes, $1.73 on one L40S (round 1: 42.8 minutes, $1.43). Noise floor re-measured 2026-09-18 with `--arm base --base-runs 10` on whileai 0.75: 26.8 GPU minutes, $0.89, no training. Run page: https://withwhile.com/platform/training/run_396d8b162199da3d

## References

1. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
2. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
3. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
4. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
5. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
