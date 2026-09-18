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
| [adaptive-clip](adaptive-clip) | [2609.00444](https://arxiv.org/abs/2609.00444) | Qwen/Qwen2.5-1.5B-Instruct | pass@1 | 0.57 -> 0.51 (-0.06 [-0.12, -0.01], flat) | 2026-09-17 |
| [endpoint-sft](endpoint-sft) | [2609.07103](https://arxiv.org/abs/2609.07103) | Qwen/Qwen2.5-1.5B-Instruct | pass@1 | 0.29 -> 0.28 (-0.01 [-0.07, +0.05], flat) | 2026-09-17 |
| [filter-metric](filter-metric) | [2609.13866](https://arxiv.org/abs/2609.13866) | Qwen/Qwen2.5-1.5B-Instruct | pass@1 | 0.39 -> 0.46 (+0.07 [+0.02, +0.11], moved) | 2026-09-17 |
<!-- table:end -->

The table is generated: `python recipes/papers/check.py --write` reads every
`results.json`. Do not edit it by hand.

## Run one

```bash
pip install whileai modal
export WHILEAI_API_KEY=...        # run page + datasets at zeroproofai.com/platform
modal token set --token-id ... --token-secret ...
cd recipes/papers/<slug>
python recipe.py                    # both arms, writes results.json
```

## The contract

- `README.md` in the shape of [`_template/README.md`](_template/README.md): Paper, Claim, The change, numbered steps, one command, the Result table, the Climb table, three Learned bullets, the Verified line.
- `recipe.py`: one file. Data, then train, then eval, then `results.json`. Two arms on the same holdout: the baseline and the paper's change. Paired delta with a 95% interval (`wai.delta_report`).
- `results.json`: the numbers the table above reads. Shape in [`_template/results.json`](_template/results.json).
- Default run: under 60 GPU minutes, under $10. Bigger runs behind a flag.
- Public data or a seeded environment that lives in the recipe directory. No customer data.
- A flat result is a result. Say so in the table.
- `python recipes/papers/check.py --write` passes (`tests/recipes/test_papers.py` runs it in CI).

## The science bar

Every recipe is held to [rlhfbook.com](https://rlhfbook.com). The README names
the chapter it rests on, and the `## Checks` table is run, not ticked:

| Check | Chapter | What `check.py` enforces |
|---|---|---|
| Eval noise | ch. 16 Evaluation | the base is evaluated 3 times; `eval_variance` run_std is recorded; "moved" needs a delta over `noise_band(run_std)` = 1.96 x run_std x sqrt(1/n_a + 1/n_b), which is 1.96 x sqrt(2) x run_std with one run per side (a delta is the difference of two re-run draws) |
| Paired interval | ch. 16 | "moved" needs a 95% interval that excludes zero, from `delta_report` over the same holdout tasks |
| Clean holdout | ch. 16 | `decontaminate(train, against=holdout)` runs before training; dropped rows are counted |
| Reward is a program | ch. 7 Reasoning, ch. 13 Tool use | a verifier or a public gold answer; a judge only when the paper is about judges |
| Proxy vs target | ch. 14 Over-optimization | the training reward is named as `proxy=`; an over-optimized verdict forbids "moved" |
| Length | ch. 14 | mean completion length before and after, per arm, in the table |
| Hack scan | ch. 14 | `hack_scan` on the last training batch; the top feature is named |
| Pinned | app. C | seed and library versions in results.json |

Chapter map (source files under `book/chapters/` in
[natolambert/rlhf-book](https://github.com/natolambert/rlhf-book); the site
serves them at `rlhfbook.com/c/<file name without .md>`): 03 training
overview, 04 instruction tuning, 05 reward models, 06 policy gradients, 07
reasoning, 08 direct alignment, 09 rejection sampling, 10 preferences, 11
preference data, 12 synthetic data, 13 tools, 14 over-optimization, 15
regularization, 16 evaluation, 17 product, appendix-c practical.

## Maintenance

A daily agent re-runs the recipe with the oldest verified date, refreshes its
numbers, fixes what broke, and adds one new recipe from research published in
the last 60 days. Everything arrives as a pull request. One comment per run on
the issue titled "Recipe log". Several agents can work at once: each recipe is
its own directory and the table is generated, so two new recipes never touch
the same line.
