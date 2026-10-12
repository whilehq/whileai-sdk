---
title: "Trim SFT data to what the test can see"
sidebarTitle: "Trim SFT data"
description: "Selection.trim fits the learning curve you measured on nested subsets and keeps the smallest N past which one more doubling of rows gains less than a point, with a range on N and a check that your test can see that point."
---

**What you learn:** how many SFT rows still move your test, with an interval, and which rows to keep. **Needs:** a `Selection`, five or more SFT runs on nested subsets (two seeds each), and per-item test scores. **Takes:** one call after the runs.

## The loop

```python
import whileai as wai

rows = wai.select(scored, mode="sft", target=16000)
base = evaluate(base_model)  # per-item 0/1 scores on your test

scores = {}
for n in rows.trim_sizes():  # 64, 128, 256, ... len(rows)
    scores[n] = [train_and_eval(rows.trim(n), seed) for seed in (0, 1)]  # per-item scores per seed

kept = rows.trim(scores, base=base)
print(kept.report["trim"])
kept.export("train.jsonl")
```

```text
verdict: saturated; N* = 5,208 rows (next doubling gains < 0.010)
  fit: base + (a - base) * n^c / (d + n^c); base 0.296, a 0.714, c 0.92, d 147
  error predicting n=16000 from the rest: rectified 0.007, power 0.014, logit 0.064
  N* 90% range 2,239 rows .. 6,214 rows; inside the sizes trained in 100% of draws
  last doubling gained +0.007 [+0.000, +0.016]
  smallest gain this test can detect: 0.021
  note: min_gain 0.010 is under what this test can detect (0.021): add items or seeds
```

That run is a simulation with a known answer (a 1,000-item test, two seeds): the true N* is about 3,700 rows, inside the range. The note is the honest part: this test cannot see a one-point gain, so "saturated" means "nothing the test can see past here". Train every size with the same epochs (or the same steps) and say which; at a fixed step budget, many epochs on few rows can beat one pass on many [16].

## Which rows

`trim(n)` is the first `n` rows of one fixed order, so `trim(64)` sits inside `trim(128)` and every point on the curve comes from one order. Exact duplicates go last; with `embed=`, near duplicates too [9].

| `by=` | What it does | Evidence |
|---|---|---|
| `"random"` (default) | Seeded shuffle | On curated pools, no published selector beat random by more than about a point, and none significantly [1][2] |
| `"diverse"` | k-means on `embed=` vectors, round robin across clusters, longest response first | Embedding diversity tracks fine-tuned quality (r 0.76 to 0.97); lexical diversity does not (r about -0.3) [4]. Clustering added 3 to 5 points in [1] |
| `"target"` | Closest to your `dev=` examples by embedding cosine | RDS+ beat random by about 9 points on a 5.8M-row pool [3] |
| `"bigram"` | New prompt word pairs per prompt word, greedy | No embeddings needed; no study validates it for SFT [4][10] |

Selection pays most on large, noisy pools when the ranker knows the target [3][5]. To see what a ranker is worth on your data, run its curve and a random curve on the same sizes; the gap is the value.

## How many rows

Three curve forms are fit, each point weighted by its noise, `1 / (seed variance / k + item variance / test_n)` [12]:

| Form | Score at n rows | Why it is a candidate |
|---|---|---|
| `rectified` | `base + (a - base) * n^c / (d + n^c)` | The rectified scaling law, which fits the flat start that small SFT sets show [6]. Anchored at the untuned model |
| `power` | `a - b * n^-c` | The classic learning curve [7] |
| `logit` | `logit(score) = alpha + beta * ln n` | Math SFT accuracy is close to linear in `log n` [8] |

The form that best predicts the largest size from the others wins. Forms are compared by extrapolation, not in-sample fit [11].

`N*` is the smallest n past which one more doubling gains less than `min_gain` (default 0.01, one point):

$$\text{score}(2n) - \text{score}(n) < \text{min\_gain}$$

`min_gain` is what a point is worth to you. It is not the test's error bar: a threshold tied to the error bar makes N* grow with the test's size on an identical curve (200, 500, 1,000 and 2,000 items gave 325, 813, 1,627 and 3,254 rows in our first version). The stop rule is progressive sampling's [13], applied to a fitted curve.

## How sure

* **Range on N\*.** A bootstrap resamples test items (the same items at every size, a paired design) and seeds within each size, refits, and reports the 5 to 95% range [12]. The verdict is `saturated` only when N* falls inside the sizes you trained in at least 90% of draws.
* **Can the test see `min_gain`?** The paired detectable gain between the two largest sizes is `2.8 * sqrt(2 * seed_var / k + var(item diffs) / test_n)` [12]. If `min_gain` is smaller, the print says so: add items or seeds.
* **Forecasts.** An N* past twice the largest size is labeled `forecast`. Downstream scores often break from a clean curve [14], and about 15% of learning curves are ill-behaved [15].
* **Other verdicts.** `not identifiable` means over 10% of refits failed, or the power exponent fell under 0.1 (fine-tuning exponents of 0.015 to 0.15 are reported in loss [17]). `flat` means no rising curve fits. Anything but `saturated` or `flat` keeps every row.

One score per size works too, but with equal weights and no range; the print says so.

## References

1. Xia et al. "Rethinking Data Selection at Scale: Random Selection is Almost All You Need." arXiv:2410.09335, 2024.
2. Diddee, Ippolito. "Chasing Random: Instruction Selection Strategies Fail to Generalize." arXiv:2410.15225, 2024.
3. Ivison et al. "Large-Scale Data Selection for Instruction Tuning." arXiv:2503.01807, 2025.
4. "Measuring Data Diversity for Instruction Tuning" (NovelSum). ACL 2025, arXiv:2502.17184.
5. Xia et al. "LESS: Selecting Influential Data for Targeted Instruction Tuning." arXiv:2402.04333, 2024.
6. Lin et al. "Selecting Large Language Model to Fine-tune via Rectified Scaling Law." ICML 2024, arXiv:2402.02314.
7. Hestness et al. "Deep Learning Scaling is Predictable, Empirically." arXiv:1712.00409, 2017.
8. Yuan et al. "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models." arXiv:2308.01825, 2023.
9. Abbas et al. "SemDeDup: Data-efficient learning at web-scale through semantic deduplication." arXiv:2303.09540, 2023.
10. Lin, Bilmes. "Multi-document Summarization via Budgeted Maximization of Submodular Functions." NAACL 2010.
11. Alabdulmohsin, Neyshabur, Zhai. "Revisiting Neural Scaling Laws in Language and Vision." NeurIPS 2022, arXiv:2209.06640.
12. Miller. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv:2411.00640, 2024.
13. Provost, Jensen, Oates. "Efficient Progressive Sampling." KDD 1999.
14. Lourie, Hu, Cho. "Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check." arXiv:2507.00885, 2025.
15. Yan, Mohr, Viering. "LCDB 1.1." NeurIPS 2025 Datasets and Benchmarks, arXiv:2505.15657.
16. "Data Repetition Beats Data Scaling in Long-CoT SFT." arXiv:2602.11149, 2026.
17. Zhang et al. "When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method." ICLR 2024, arXiv:2402.17193.
