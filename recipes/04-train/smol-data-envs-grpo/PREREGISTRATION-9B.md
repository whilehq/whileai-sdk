# Pre-registration 2: the same comparison on Qwen3.5-9B

Written 2026-09-25, after the 2B result (PREREGISTRATION.md: no difference,
neither arm beat base) and before any 9B run. Everything in
PREREGISTRATION.md holds except the lines below.

## Why a second run

At 2B the base model solved 3% of the test split, and neither arm moved it.
A test the model almost never passes cannot show a difference between two
training recipes. 9B is the largest Qwen3.5 dense model that trains on one
GPU.

## What changes, for both arms alike

| | 2B run | 9B run |
|---|---|---|
| model | `Qwen/Qwen3.5-2B`, full fine-tune | `Qwen/Qwen3.5-9B`, LoRA r=16, alpha=32, all linear layers (full fine-tune of 9B plus a colocated vLLM does not fit one GPU) |
| learning rate | 3e-6 (the authors', for full fine-tune) | 1e-5 (the LoRA rate recipes/04-train/text-to-sql uses) |
| GPU | H100 | H200 |
| vLLM share of the GPU | 0.22 | 0.35 (the 9B weights alone are 18 GB) |

The pre-flight is re-run with the 9B base model, so the whileai arm's tasks
are the 9B model's 20-80% band. The authors' extra steps are re-measured the
same way, from a 3-step smoke of the authors' arm on 9B.

## What counts as a win

Unchanged: the with-whileai minus authors interval, both seeds pooled, is
above zero and above the seed spread, or the result is "no difference".
Each arm minus base is reported alongside.

## Amendment 1 (2026-09-25, after the 9B pre-flight, before any 9B training run)

- **Pre-flight:** 9B base, 1,024 pool tasks x 8 samples, 1,309.7 GPU seconds.
  233 tasks in the 20-80% band (82 easy, 139 medium, 12 hard); all 233 are
  the whileai arm's tasks (fewer than 256). The authors' 256: 129 never
  solved, 30 always, 63 in the band.
- **Base on test:** pass@1 0.152 sampled (956 rows), 0.167 greedy.
- **Steps:** the 9B authors'-arm smoke (`9b/eval/authors-s1-smoke`)
  steady-state steps took 30.68 s and 9.21 s, mean 19.94 s (the first,
  115 s, is compile). 1,309.7 / 19.94 = 65.7, so the authors' arm runs
  **266 steps**, the whileai arm **200**. Two samples 3x apart make this a
  noisy estimate; it is the rule, applied as written, and the realised GPU
  seconds of every run are reported with the result.
