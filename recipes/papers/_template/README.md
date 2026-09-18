# <Recipe name>

**Paper:** <title>, <first author et al.>, <arXiv id or venue>, <month year>. <link>
**Book:** rlhfbook.com ch. <n> <chapter title>: <one line on which practice from the chapter this recipe tests or relies on>
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

Recipe vs baseline: <+0.00 [lo, hi]>. Verdict: <moved / flat>.

## Checks

| Check | Book | Result |
|---|---|---|
| Eval noise: the base evaluated 3 times, `eval_variance` run_std | ch. 16 | run_std <0.00>; a delta under <1.96 x sqrt(2) x run_std> is noise (`noise_band(run_std)` with one run per side: a delta is the difference of two re-run draws) |
| Holdout is clean: `decontaminate(train, against=holdout)` | ch. 16 | <n> train rows dropped |
| Reward is a program, not a judge | ch. 7, 13 | <what the reward reads> |
| Proxy vs target: `delta_report(proxy=)` | ch. 14 | over_optimized <false / true> |
| Length: mean completion length before -> after, per arm | ch. 14 | baseline <a -> b>, recipe <a -> b> |
| Hack scan on the last training batch: `hack_scan` | ch. 14 | top feature <name>, endorsed <yes / no> |
| Pinned: seed, torch, transformers, trl, peft | app. C | seed <n>, <versions> |

## Climb

| Round | What changed | <metric> | vs previous |
|---|---|---|---|
| 1 | as the paper | | |

## Learned

- <what moved>
- <what did not>
- <what to try next>

Verified <YYYY-MM-DD>, whileai <version>, <trainer and version>. Run page: <url>
