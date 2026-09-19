# GMTS: rank tokens by entropy times advantage, not by entropy alone

**Paper:** GMTS: Gradient Magnitude-based Token Selection Improves RLVR Training, Outongyi Lv et al., arXiv:2608.30632, August 2026. https://arxiv.org/abs/2608.30632
**Book:** group-relative methods "assign the same sequence-level advantage (or reward) to every token when computing the loss" [1], and how the loss aggregates over tokens decides whether each token or each sequence contributes equally [1, 2], which is what decides whether a token *selection* can matter at all.
**Claim:** training on only the top 20% highest-entropy tokens beats training on all of them, but entropy is read one answer at a time: two tokens can carry the same entropy in answers the group scored very differently. Ranking instead by entropy times the answer's learning signal picks a better 20% and raises accuracy.
**The change:** the score the top 20% is taken by. The baseline ranks tokens by entropy; the recipe ranks them by |entropy x advantage|.

## Recipe

1. Base: `Qwen/Qwen2.5-1.5B-Instruct`. Data: GSM8K, 512 train prompts from the train split, 120 held out from the test split (different splits, so there is no overlap to check for).
2. Reward, both arms: the binary outcome, `MathEqual` against the GSM8K gold number. A program, not a judge. The paper changes which tokens are trained on, not what counts as right.
3. Baseline arm (ETS, the prior result): GRPO, and before the loss the completion mask is narrowed to the 20% of the batch's tokens with the highest entropy. The other 80% contribute nothing.
4. Recipe arm (GMTS): the same 20%, ranked by `|E * omega|` instead — the paper's equation 4, with `E` the token's entropy and `omega` the scalar in front of it in the policy-gradient term. The run is on-policy (`num_iterations` 1) with no KL term (`beta` 0), so the importance ratio is exactly 1 and nothing clips: `omega` reduces to the advantage, which is the part of it the paper says carries the cross-answer information.
5. Eval: pass@1 on the same 120 held-out tasks, 4 samples per task. The untrained base is evaluated three times first, and that spread is the noise floor a delta has to clear; the train set is decontaminated against the holdout before any training. Paired delta with a 95% interval (`wai.pass_at`, `wai.delta_report`).

Both arms keep exactly the same *number* of tokens, so the comparison isolates which tokens and not how many. Two details decide whether the change can bind at all, and both are in the code as refusals rather than comments:

- **The threshold is taken over the whole batch, not per row.** A per-row top 20% would hand every answer the same number of slots and throw away the cross-answer comparison the paper is about. Here a microbatch is one prompt's eight rollouts, so "across answers" means across the group the advantage is computed from.
- **The loss must normalize over the batch.** TRL's default `loss_type="bnpo"` divides by the batch's surviving-token count, so an answer given more of the 20% gets more of the gradient. `loss_type="grpo"` divides each sequence by *its own* count, which renormalizes the effect away; the trainer raises rather than run it.

## Run

```bash
python recipe.py --selftest   # the ranking, the mask and the normalizer, offline, no GPU and no key
python recipe.py              # both arms, ~45 GPU minutes and ~$1.50 on one L40S
python recipe.py --lr 3e-5    # the same two arms at a third of the step size (Climb round 2)
```

## Result

Run today, both arms, on one L40S, with the recipe's defaults and the paper's
20% selection (the paper's abstract gives no learning rate, rank, step count
or model size; 1e-4, r=32, 40 steps and 1.5B are this recipe's).

| Arm | pass@1 | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | 0.36 | [0.29, 0.43] | 0.58 | 0 | 0 |
| Baseline (top 20% by entropy) | 0.49 | [0.42, 0.57] | 0.68 | 40 | 27.6 |
| Recipe (top 20% by entropy x advantage) | 0.18 | [0.13, 0.23] | 0.41 | 40 | 17.7 |

GPU minutes: the baseline arm's 27.6 includes the three base evals that set the
noise floor; the recipe arm's 17.7 does not, and the base row's 0 is those
evals counted under the baseline.

Recipe vs baseline: **-0.310 [-0.383, -0.235]** over 120 paired tasks.
Verdict: **flat**. The interval excludes zero but on the wrong side. `check.py`
only rejects an interval that covers zero; the sign rule is this recipe's own,
mapping `delta_report`'s `moved_the_wrong_way` to `flat` so the index never
reads a collapse as a move.

This is not a small miss. The recipe arm finished at 0.18, **below the 0.36 of
the model that was never trained at all** — it did not fail to help, it
destroyed the policy. Mean completion length fell from 700 characters to 374
and `hack_scan` came back with `frac:upper`, both signatures of a model coming
apart rather than one gaming a reward.

The trainer's logs show the divergence plainly. Gradient norm over the 40 steps:

| Arm | grad_norm mean | grad_norm max |
|---|---|---|
| Baseline (entropy) | 11.5 | 48.1 |
| Recipe (entropy x advantage) | **60.1** | **107.8** |

**Round 2 is what stops that table from being over-read, so read it too.** At
lr 3e-5 the same two arms produce grad_norm means of 9.9 and 10.1 — the same
number. The 5x gap above is not a fixed property of the selection; it is what
divergence looks like from inside. GMTS did not take uniformly larger steps, it
fell into a feedback loop at a step size where entropy selection did not.

What the selection does mechanically is narrower than "bigger gradients". The
advantage is constant along a rollout, so ranking by `|E * omega|` cannot
reorder tokens *within* an answer — it only decides how many slots each answer
gets, and it gives them to the answers with the largest |advantage|. In a group
of eight with a binary reward and one correct rollout, TRL's scaled advantage is
about +2.47 for the correct one against -0.35 for each wrong one (TRL's
unbiased group std), so GMTS spends
its 20% almost entirely on that single rollout where entropy selection spreads
the same 20% across all eight. That concentration is measured
(`selection_overlap` 0.703). That it makes each update depend on fewer
sequences, and so widens the spread of updates enough to destabilise a step
size entropy selection survives, is the reading those two rounds support — it
is not separately measured here, and a per-step gradient-variance log would be
the way to check it.

The practical conclusion does not depend on which mechanism is right: **the
substitution is not drop-in.** It changes which learning rates are stable, so
comparing the two rankings at one learning rate is not comparing the rankings.

## Checks

Nothing in this table is ticked by hand: every cell is written by `recipe.py`
into `results.json`. These are the round 1 numbers.

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | [3] | **run_std 0.0115**, so a delta under **0.032** is noise. The -0.310 here is ten times that, which is the one thing about this result that is not in doubt |
| Holdout is clean: `decontaminate(train, against=holdout)` | [4] | **0 of 512 train rows dropped**, as expected for disjoint GSM8K splits — measured, not assumed |
| Reward is a program, not a judge | [4] | `MathEqual` against the public GSM8K gold number. No judge, no model in the loop |
| Proxy vs target: `delta_report(proxy=)` | [5] | `proxy=None`: the training reward *is* the target metric, the same binary check, so there is no proxy to over-optimize. `over_optimized` false |
| Length: mean completion length before -> after, per arm | [5] | **700 chars base -> 636 baseline, 374 recipe.** The recipe arm nearly halved, alongside its accuracy — degeneration, not brevity |
| Hack scan on the last training batch: `hack_scan` | [5] | recipe batch top feature `frac:upper`; baseline batch `contains:: AND contains:calculate the`. Nothing is endorsed. `frac:upper` on a collapsed arm is the scan seeing the wreckage, not a reward surface worth optimizing |
| **Was the change reachable: `selection_overlap`** | [1] | **0.703 in both arms**, with `kept` exactly 0.200. The two rankings chose different tokens for 30% of the budget, every step. The change was reachable, and the arms really did train on different tokens |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | seed 17 in the trainer, `--seed 0` for the data split; torch 2.7.1, transformers 4.54.0, trl 0.19.1, peft 0.16.0 |

The last row is not in the template. It is here because the recipe next door
([adaptive-clip](../adaptive-clip)) spent a GPU hour on a change that was
implemented correctly and could not move a gradient, and one line of logging
would have said so first. `selection_overlap` is the share of an arm's chosen
tokens that the other arm's ranking would also have chosen, averaged over every
training step. At 1.0 the two arms trained on the same tokens and the recipe
tested nothing. At 0.703 this one did not have that problem.

The two arms share the data, the holdout, the reward, the selected token count
and every trainer knob except the ranking key. They did **not** share the LoRA
init in rounds 1 and 2: TRL 0.19.1 builds the adapter before it applies
`GRPOConfig.seed`, and the baseline arm runs the three base evals first, which
advances the RNG before its adapter is drawn. The run pages show it: the recipe
arm's step-1 grad_norm is bit-identical across rounds, the baseline arm's is
not. `recipe.py` now calls `set_seed(17)` right before building the trainer;
the next verify run is the first with both arms on one init. Until then the
delta has two causes available to it, the ranking and the init.

## Climb

| Round | What changed | pass@1 | vs previous |
|---|---|---|---|
| 1 | recipe defaults with the paper's 20%: top 20% of tokens, k = 8 rollouts, 40 steps, lr 1e-4, LoRA r=32, bnpo loss, on-policy | baseline 0.49, recipe 0.18 | -0.310 [-0.383, -0.235], flat (wrong way) |
| 2 | same, both arms at lr 3e-5 (`--lr 3e-5`), to separate the selection from the step size | baseline 0.35, recipe 0.33 | -0.019 [-0.052, +0.015], flat |

Round 2 moves **both** arms, because moving only the recipe arm would leave the
two differing in two things and stop testing the paper at all.

It bought exactly one clean answer and one dead end.

The answer, as far as one run per cell can give one: **the collapse is
consistent with a step-size effect.** At a third of the learning rate the
recipe arm is 0.33 against the baseline's 0.35, the interval covers zero, and
the grad_norm gap is gone (9.9 against 10.1). One training run per cell cannot
establish it (the recipe next door, adaptive-clip, moved ten points between
two identical runs); the run that would is a second recipe arm at 1e-4, about
18 GPU minutes, and it is written down rather than run.

The dead end: **at lr 3e-5 neither arm learns.** The base is 0.36; the baseline
finishes at 0.35 and the recipe at 0.33. Forty steps at that step size moves
nothing, so the -0.019 is a comparison between two models that barely trained,
and it says nothing about the paper's claim.

So the two rounds bracket the question without answering it. At 1e-4 entropy
selection trains and GMTS diverges; at 3e-5 GMTS is stable and neither trains.
Somewhere between them is a step size where both train and the rankings can be
compared, and finding it is the next round — a learning-rate sweep on the
recipe arm to locate its stability edge, then both arms at the largest rate
both survive. That is two more GPU hours, not two more minutes, which is why it
is written down here rather than run.

Two rounds is this directory's limit, so the honest summary of the recipe as it
stands is: **the paper's claim is untested at this budget**, and the reason it
is untested is itself the result.

## Learned

- **A token selection is also a stability change, and comparing two selections at one learning rate does not compare the selections.** Ranking by `|E * advantage|` cannot reorder tokens inside an answer, since the advantage is constant along it [1]; all it can do is move slots between answers, toward the ones with the largest |advantage|. At lr 1e-4 that difference was enough for one ranking to train and the other to diverge. Anyone reproducing this paper should establish each arm's stable learning-rate range first and compare at a rate both survive, or report gradient norms alongside the scores, or the number they publish is a step-size result wearing a token-selection label.
- **Do not read a gradient norm from a diverging run as a property of the method.** Round 1 showed grad_norm 60.1 against the baseline's 11.5 and the obvious story was "GMTS takes 5x the step". Round 2 killed that story: at a stable learning rate the two arms sit at 10.1 and 9.9. The 5x was the divergence, not its cause. One extra run at one changed knob was the difference between a clean mechanism and a confident wrong one.
- **Check that your change is reachable, then check what else it changed.** `selection_overlap` 0.703 confirmed the arms trained on genuinely different tokens, which is the check the recipe next door was missing. It is necessary and it is not sufficient: the change was reachable and still did not isolate what it meant to, because the thing it touched carries a second effect.
- **Computing the entropy is the expensive part of this recipe.** The score needs the full next-token distribution, so the trainer takes an extra no-grad forward and materializes a `(rows, tokens, 151936)` float32 tensor for a `log_softmax`. TRL's own scoring pass never does this — `selective_log_softmax` gathers one logprob per position. This ran at about 20 seconds a step at 8 rollouts x 256 tokens, and that pass is the bulk of it. A cheaper entropy (bf16, or fused) would be the first thing to change if this were run at any size.
- **A flat verdict and a collapse are not the same news, and the table says the same word for both.** This recipe maps `delta_report`'s `moved_the_wrong_way` to "flat" (`check.py` itself only rejects intervals that cover zero), so an arm that ends below the untrained base reads as "flat" in the index. That is the correct verdict by the rule and it is worth knowing that the rule compresses it; the Result table above says what actually happened.

Verified 2026-09-18, whileai 0.64, TRL 0.19.1 + PEFT 0.16.0 on torch 2.7.1. Round 1, the numbers above: 45.3 GPU minutes, $1.51 on one L40S; round 2: 45.9 minutes, $1.53. Run page: https://www.zeroproofai.com/platform/training/run_5198278ad1c4d851 (round 2: https://www.zeroproofai.com/platform/training/run_93e11a46757d3f56)

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Reinforcement Learning*.
2. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
3. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
4. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
5. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
