# Pre-registration: GRPO on SmolDataEnvs, with and without whileai

Written 2026-09-25, before any training run. Nothing below changes after the
first result comes in; a change is a new, dated section at the bottom.

## Question

Does whileai make GRPO on SmolDataEnvs better than the dataset authors' own
recipe, at the same GPU budget?

## Arms

Everything is identical across arms except the line marked **differs**.

| | without whileai (authors) | with whileai |
|---|---|---|
| model | `Qwen/Qwen3.5-2B`, full fine-tune | same |
| trainer | TRL `GRPOTrainer`, vLLM colocated | same |
| hyperparameters | the authors' `train_grpo.py` (FineEnvs @ 08a5622): lr 3e-6, temperature 0.8, top_p 1.0, 8 generations, 2 x 8 accumulation (16 completions, 2 tasks per step), 1,024 completion tokens, `mask_truncated_completions`, repetition penalty 1.05, non-thinking | same |
| prompt, reward | the authors' prompt; the dataset's grader; command-shaped answers score 0; a rollout the environment cannot run is `None` | same |
| **train tasks (differs)** | the authors' pick: `train.shuffle(seed=42)`, first 256, all tiers | whileai's pick: the base model samples 8 answers on each of the first 1,024 of the same shuffle, `wai.select(mode="rl", band=(0.2, 0.8))` keeps the tasks it solves between 20% and 80% of the time, first 256 of those kept in shuffle order |
| steps | 200 + the steps the whileai arm's pre-flight sampling costs in GPU time, so both arms spend the same | 200 |
| seeds | 1, 2 | 1, 2 |

The whileai arm pays for its pre-flight out of the same budget: the authors'
arm gets extra steps equal to the pre-flight's GPU seconds divided by the
authors' arm's measured seconds per step.

## Measurement

- **Held-out set:** the dataset's `test` split (250 tasks), minus tasks with
  no tables, the same set for every arm. No test task is used in training
  or in the pre-flight.
- **Primary metric:** pass@1 from 4 samples per task at temperature 0.8
  (`wai.pass_at`, interval by resampling tasks), per trained model; the base
  model is measured once with the same settings.
- **Primary comparison:** with-whileai minus authors, both seeds pooled,
  paired by task (`wai.compare` / `delta_report`).
- **Secondary:** greedy pass@1 (the authors' `eval_pass1.py` metric); the
  share of training groups with zero reward spread; each arm minus base.

## What counts as a win

whileai wins only if the paired difference's 95% interval is above zero
**and** above the seed-to-seed noise floor (`wai.eval_variance` across the two
seeds of each arm). If the interval crosses zero, the result is "no
difference at this budget", and that is what gets reported, on the site and
in any post.

## Known risks, stated before the run

- ScaleRL (arXiv 2510.13786) reports that zero-variance filtering mostly
  buys efficiency, not a higher ceiling. At 200 steps an efficiency gain is
  what a win would look like; a longer run could erase it.
- Band selection from 8 samples misplaces a task by about +-0.2 in pass
  rate; `wai.select` says so.
- 250 test tasks give an interval near +-0.05 on pass@1; a smaller true gain
  will read as no difference.

## Amendment 1 (2026-09-25, after the pre-flight, before any training run)

Facts the pre-flight fixed; no rule above changed.

- **Pre-flight:** base model, 1,024 pool tasks x 8 samples = 8,192 graded
  rollouts, 846 correct (10.3%). 1,025.7 GPU seconds.
- **The whileai arm has 125 tasks, not 256.** Only 125 pool tasks fall in the
  20-80% band (75 easy, 50 medium); the rule "first 256 of those kept" gives
  125. The authors' 256: 194 never solved by the base model, 6 always, 35 in
  the band.
- **Steps.** The authors' arm's seconds per step come from the only
  measurement taken before training, the 3-step smoke (`authors-s1-smoke`):
  steady-state steps of 7.31 s and 6.60 s, mean 6.95 s (the first step,
  114 s, is vLLM compile and is excluded). 1,025.7 / 6.95 = 147.6, so the
  authors' arm runs **348 steps**, the whileai arm **200**.
- **Base on test** (239 tasks with tables): pass@1 0.031 sampled (956 rows),
  0.067 greedy.
