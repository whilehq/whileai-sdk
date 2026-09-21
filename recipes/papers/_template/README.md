# <Recipe name>

**Paper:** <title>, <first author et al.>, <arXiv id or venue>, <month year>. <link>
**Book:** <one line, in your own words, on the practice this recipe tests or relies on> [1].
**Claim:** <one sentence: what the paper says happens, and why>
**The change:** <one line: what the recipe arm does that the baseline arm does not>

## Recipe

1. Base: `<model>`. Data: <dataset>, <n> train tasks, <n> held out by <rule>.
2. Baseline: <trainer>, <steps> steps, <batch> x <generations>, lr <x>, <key knobs>.
3. Recipe: baseline plus <the one change>.
4. Eval: <metric> on the same holdout, <k> samples per task, paired delta with a 95% interval.
5. <one more line if something matters, otherwise delete>

## Run

```bash
python recipe.py                         # both arms, ~<n> min on one <GPU>, ~$<x>
python recipe.py --arm recipe --steps 200  # one arm, longer
```

## Result

| Arm | <metric> | 95% CI | pass@k | Steps | GPU min |
|---|---|---|---|---|---|
| Base, no training | | | | 0 | 0 |
| Baseline | | | | | |
| Recipe | | | | | |

Recipe vs baseline: <+0.00 [lo, hi]>. Verdict: <moved / flat / unresolved>. <At one training seed per arm the verdict is unresolved: say in one sentence what would resolve it, a second seed on each arm passed as `train_runs=`.>

## Checks

| Check | Source | Result |
|---|---|---|
| Eval noise: the base evaluated <n> times, `eval_variance` run_std | [2] | run_std <0.00> from <n> re-runs; a delta under <t(df = n - 1) x sqrt(2) x run_std> is noise (`noise_band(run_std, df=n - 1)` with one run per side: a delta is the difference of two re-run draws, and run_std is an estimate from n runs, so the quantile is t at n - 1, not 1.96; `delta_report(run_std=, run_std_runs=n)`) |
| Holdout is clean: `decontaminate(train, against=holdout)` | [3] | <n> train rows dropped |
| Reward is a program, not a judge | [3] | <what the reward reads> |
| Proxy vs target: `delta_report(proxy=)` | [4] | over_optimized <false / true> |
| Length: mean completion length before -> after, per arm | [4] | baseline <a -> b>, recipe <a -> b> |
| Hack scan on the last training batch: `hack_scan` | [4] | top feature <name>, endorsed <yes / no> |
| Pinned: seed, torch, transformers, trl, peft | [the contract](../README.md#the-contract) | seed <n>, <versions> |

## Climb

| Round | What changed | <metric> | vs previous |
|---|---|---|---|
| 1 | as the paper | | |

## Learned

- <what moved>
- <what did not>
- <what to try next>

Verified <YYYY-MM-DD>, whileai <version>, <trainer and version>. Run page: <url>

## References

Numbered in order of first appearance. Every [n] on the page resolves here and
every entry is cited at least once. Cite the primary paper for a method; cite
the textbook by chapter title, never by chapter number.

1. <the primary paper, or the textbook chapter by title, behind the Book line; for example: Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Reinforcement Learning*.>
2. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
3. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
4. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
