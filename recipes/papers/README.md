# Papers

One directory per paper. A recipe here is a recent post-training paper's idea
cut down to a run that fits in under an hour on one GPU, with the number it
moved and the number it did not. The building blocks are the step recipes next
door (`../04-train/grpo`, `../04-train/dpo`, `../04-train/text-to-sql`,
`../04-train/hosted-loop`); a paper recipe copies one of them and changes one
thing.

Every recipe answers the same five questions in the same order: which paper,
what it claims, the steps, one command, what happened.

<!-- table:start -->
| Recipe | Paper | Base | Metric | Baseline -> Recipe | Verified |
|---|---|---|---|---|---|
| [adaptive-clip](adaptive-clip) | [2609.00444](https://arxiv.org/abs/2609.00444) | Qwen/Qwen2.5-1.5B-Instruct | pass@1 | 0.47 -> 0.52 (+0.05 [+0.00, +0.10], flat) | 2026-09-18 |
| [endpoint-sft](endpoint-sft) | [2609.07103](https://arxiv.org/abs/2609.07103) | Qwen/Qwen2.5-1.5B-Instruct | pass@1 | 0.29 -> 0.28 (-0.01 [-0.07, +0.05], flat) | 2026-09-17 |
| [filter-metric](filter-metric) | [2609.13866](https://arxiv.org/abs/2609.13866) | Qwen/Qwen2.5-1.5B-Instruct | pass@1 | 0.39 -> 0.46 (+0.07 [+0.02, +0.11], moved) | 2026-09-17 |
| [gmts-token-select](gmts-token-select) | [2608.30632](https://arxiv.org/abs/2608.30632) | Qwen/Qwen2.5-1.5B-Instruct | pass@1 | 0.49 -> 0.18 (-0.31 [-0.38, -0.24], flat) | 2026-09-18 |
| [zero-rl-format-reward](zero-rl-format-reward) | [2503.18892](https://arxiv.org/abs/2503.18892) | Qwen/Qwen3.5-4B-Base | pass@1 | 0.63 -> 0.72 (+0.09 [+0.05, +0.14], moved) | 2026-09-18 |
<!-- table:end -->

The table is generated: `python recipes/papers/check.py --write` reads every
`results.json`. Do not edit it by hand.

## Run one

```bash
# datasets: every recipe builds its train/holdout split locally, before Modal.
# the modal extra: needed when outbound traffic goes through an HTTPS proxy,
# harmless when it does not.
uv add whileai datasets 'modal[api-proxy-support]'
export WHILEAI_API_KEY=...        # run page + datasets at withwhile.com/platform
modal token set --token-id ... --token-secret ...   # or MODAL_TOKEN_ID / _SECRET in the environment
cd recipes/papers/<slug>
python recipe.py                    # both arms, writes results.json
```

## The contract

- `README.md` in the shape of [`_template/README.md`](_template/README.md): Paper, Claim, The change, numbered steps, one command, the Result table, the Climb table, three Learned bullets, the Verified line, the References list.
- `recipe.py`: one file. Data, then train, then eval, then `results.json`. Two arms on the same holdout: the baseline and the paper's change. Paired delta with a 95% interval (`wai.delta_report`).
- `results.json`: the numbers the table above reads. Shape in [`_template/results.json`](_template/results.json).
- Default run: under 60 GPU minutes, under $10. Bigger runs behind a flag.
- Public data or a seeded environment that lives in the recipe directory. No customer data.
- A flat result is a result. Say so in the table.
- `python recipes/papers/check.py --write` passes (`tests/recipes/test_papers.py` runs it in CI).
- `post.md`: the result as a post, once the recipe is verified. Under 280
  characters, plain words, the metric with its interval, the arXiv link
  and the recipe link, nothing invented and nothing rounded. A flat result
  is posted as flat. Replicated papers are how we market
  ([CONSTITUTION.md](../../CONSTITUTION.md), belief 2); the post is the
  last artifact of a recipe, not a separate job.

## The science bar

Every recipe is held to the same science bar. The README names the source
each check rests on, and the `## Checks` table is run, not ticked:

| Check | Source | What `check.py` enforces |
|---|---|---|
| Eval noise | [1] | the base is evaluated `run_std_runs` times (3 by default, more with `--base-runs`); `eval_variance` run_std and `run_std_runs` are both recorded; "moved" needs a delta over `noise_band(run_std, df=run_std_runs - 1)` = t x run_std x sqrt(1/n_a + 1/n_b), which is t x sqrt(2) x run_std with one run per side (a delta is the difference of two re-run draws). t is the two-sided 95% quantile at df = run_std_runs - 1 because run_std is an estimate, not the eval's exact spread: 4.30 from 3 re-runs, 2.26 from 10; the old 1.96 read a three-run estimate as exact and let about one pure-noise delta in five through. `delta_report(run_std=, run_std_runs=)` applies the same quantile |
| Paired interval | [2] | "moved" needs a 95% interval that excludes zero, from `delta_report` over the same holdout tasks |
| Clean holdout | [3] | `decontaminate(train, against=holdout)` runs before training; dropped rows are counted |
| Reward is a program | [3] | a verifier or a public gold answer; a judge only when the paper is about judges |
| Proxy vs target | [4] | the training reward is named as `proxy=`; an over-optimized verdict forbids "moved" |
| Length | [4] | mean completion length before and after, per arm, in the table |
| Hack scan | [4] | `hack_scan` on the last training batch; the top feature is named |
| Pinned | [the contract](#the-contract) | seed and library versions in results.json |

## Maintenance

A daily agent re-runs the recipe with the oldest verified date, refreshes its
numbers, fixes what broke, and adds one new recipe from recent post-training
research. The default pick is a paper from the last 60 days; an older paper is
allowed when the PR says what it is the baseline for (SimpleRL-Zoo, March 2025,
is the zero-RL baseline). The table's Paper column dates every one. Everything
arrives as a pull request. One comment per run on
the issue titled "Recipe log". Several agents can work at once: each recipe is
its own directory and the table is generated, so two new recipes never touch
the same line.

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
2. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
3. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
4. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
