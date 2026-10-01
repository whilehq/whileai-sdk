# Pre-registration: the fair rerun

Written and merged before any of the runs below were trained or scored. Every
rule here is fixed; if one has to change, the change lands as a commit to this
file with the reason, before the affected result exists.

## Why a rerun

The first result (README, `results.json`) compared smithtune's pick against
While's pick of the same 400 traces. The LangChain team pointed out that our
smithtune arm skipped the step their skill puts at the center: a person and an
agent reading traces together and co-writing the rubric. They were right. We
wrote `rubric.md` without reading a trace, left call counts out of it on
purpose, and did not calibrate; the council kept 258 of 260, so that arm was
in effect "every correct trace". The first result also trained one seed per
arm on sets of different size (173 against 258).

This rerun fixes all three and adds a second question.

## Questions

**A. Selection.** From the same pool of traces, does While's pick train a
better reviewer than smithtune's pick made the way smithtune recommends?

**B. Generation.** Given 400 more rollouts to spend, does aiming them at the
agent's weak spots (While) beat spending them on more of the same traffic (the
smithtune path, which cannot aim), each followed by its own pick?

## What is held equal in every arm

| | |
|---|---|
| Agent | `agent.py` unchanged: stock `create_deep_agent`, read-only file tools over the base commit |
| Trace model | `qwen/qwen3.8-27b` through OpenRouter, `agent.py` settings (temperature 0.6, reasoning cap 8,192) |
| Source pool | LangSmith project `deepagents-review`, root runs named `review` with `correctness >= 1`, 2026-09-25T02:46:30Z to 2026-09-26T00:00:00Z (the 262 usable correct traces of the original 400) |
| Rendering and split | smithtune 0.1.0 `prepare_sft_rows` (reasoning omitted), `split_rows` 80/10/10, `render_row_tokens` (`export_sft.py`) |
| Training | `train_modal.py`: LoRA rank 8, alpha 32, batch 32, lr 1e-4, up to 5 epochs, stop when validation loss does not improve, keep the best epoch |
| Seeds | three per arm: 42, 43, 44 (LoRA init and data order) |
| Training-set size | equal within each experiment (below) |
| Serving | one `serve_modal.py` vLLM 0.26 server for every adapter, `enable_thinking=false`, same sampling |
| Benchmarks | SWE-bench Verified patch review (250 reviews, `public_bench.py`) and the 84 held-out repositories (246 reviews) |

## Arms

### Experiment A: selection, same 262 traces

- **A-without.** smithtune's happy path. The rubric is
  `rubric_codesigned.md`: written after reading 20 varied pulled traces
  (4 short, 5 medium, 4 long, 5 very long, 2 with unusual tools), then checked
  on a 20-trace trial (`smithtune dataset pull trial-v2 ... --max-candidates 20`;
  13 of 19 fully judged kept, drop reasons matched the criteria, rubric not
  changed after the trial; labels in `calibration/`). The file is
  byte-identical to the text the council judged with. Its line "a person
  signed off on the criteria" means Jacob Weiss approved the rubric and the run
  on 2026-10-01 from a summary, not a line-by-line edit. Full run: `smithtune dataset pull council-v2` with
  the source above, `--target-count 400 --max-candidates 1000`, then `triage
  --rubric rubric_codesigned.md --confirm` with smithtune's default council
  (DeepSeek V4.1 Flash and GLM-5.3-Flash on Baseten, strict majority). The
  LangChain team is invited to review the rubric; if they send changes before
  A-without is trained, we use their version and say so.
- **A-with.** `pick.py` unchanged: `wai.simulations.optimize(mode="sft")`,
  reward 1 only for a correct review within 8 model calls, round-robin over
  tool-call signatures.

**Size.** N_A = the smaller of the two kept sets. The larger set is cut to N_A
by a uniform random sample (Python `random.Random(0).sample` over run ids
sorted ascending). The 80/10/10 split is applied after the cut.

### Experiment B: generation, 400 more rollouts per side

Both sides start from the same 400 original traces and add 400 new reviews
from the 756 training-split tasks no trace has touched yet (`tasks.jsonl`,
split `train`, minus the 400 already run and `unavailable.json`). Each new
task is reviewed once, with the trace model and settings above.

- **B-without.** 400 tasks drawn uniformly (`random.Random(0).sample` over
  task ids sorted ascending). A smithtune user cannot aim, so the fair spend is
  more of the same traffic. The pool (original plus new) is triaged with
  `rubric_codesigned.md` unchanged, in a fresh directory.
- **B-with.** 400 tasks drawn by `aim.py`: every task is put in a bucket from
  features known before any rollout (verdict label; whether the patch touches
  any non-test `.py` file; files touched, 1 or more than 1; lines changed,
  up to 10, 11 to 40, over 40). Each bucket's weight is its failure rate on the
  original traces (failure = wrong, or more than 8 model calls), smoothed
  with add-one (Laplace) smoothing, `whileai.simulations.defaults.laplace`. Tasks are drawn without replacement with
  probability proportional to their bucket's weight, `random.Random(0)`. The
  pool (original plus new) is picked with `pick.py` unchanged.

**Size.** N_B = the smaller of the two kept sets, the larger cut the same way
as in A.

## Scoring and analysis

- Every adapter (4 arms x 3 seeds = 12) is scored once on each benchmark with
  `collect.py` against the shared server. Infrastructure failures are rerun
  and never scored as wrong; a review that runs out of steps scores as wrong
  (`report.load`).
- An arm's rows are the three seeds' rows pooled, so each review's score is
  its mean over seeds. Comparisons are `wai.simulations.compare_runs` paired
  by review (bootstrap 95% interval, sign-flip p), as in `make_results.py`.
- **Primary:** A-with minus A-without, accuracy on SWE-bench Verified.
- **Secondary:** the same on the held-out repositories; model calls per
  review on both; B-with minus B-without on both; each arm against the
  three-run base noise floor already measured.
- A difference is reported as a win only when its interval excludes zero. A
  tie is reported as a tie. Every arm and every number is published, whichever
  way it comes out, in `results.json`, the README and the blog post.

## Already known before this file

The original 400 traces and their correctness; the trial-v2 triage labels;
the first result. No A or B arm has been trained or scored.

## Not tested here

Thinking-on fine-tuning (smithtune drops reasoning by default, so both sides
train non-thinking models); other models; other agents. Generation beyond
choosing real tasks: an LLM-written review task has no hidden tests behind it,
so it has no verdict to train toward.
