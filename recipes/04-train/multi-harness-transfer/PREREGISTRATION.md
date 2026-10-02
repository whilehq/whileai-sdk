# Pre-registration: does multi-harness RL transfer to harnesses it never saw?

Written 2026-10-02, approved by Jacob the same day, committed before any Phase 0
scoring run. Complements FineEnvs' multi-harness RL release [1]. Nothing below
changes after the first result comes in; a change is a new, dated amendment at
the bottom. The one-task infrastructure smoke (base model only) is not a result.

## Summary

- Their multi-harness model was never tested on an unseen harness.
- One seed, one sample per task, best checkpoint picked on test.
- We test on 6 unseen harnesses, with intervals.
- New arm: one harness, randomized at the proxy.
- Question: can randomizing one harness replace collecting four?

## What the release leaves open

| Gap | Release [1] | Others |
|---|---|---|
| Unseen-harness transfer of the multi-harness model | not measured; only OpenCode-only was scored off its harness | OpenForgeRL [2] same gap; Orchard [3] diversity via SFT only |
| Noise | 1 seed, 1 sample per cell, n=250; 1.9-pt gap is about one standard error | none of [2-5] report seeds |
| Checkpoint pick | on the test set (model cards say so) | |
| Exposure | Claude Code gives about 8 training rows per rollout; 451M vs 162M tokens processed | Agent Lightning [4] weights per rollout instead |
| Randomized harness instead of many real ones | not tried | KAT-Coder [5] randomizes names, args, formats, prompts, no ablation; ToolRL-DR [6] only for single tool calls |
| Uniform harness mixing | used | HarnessBandit [7]: uniform 6-harness mix about equal to base in-distribution (0.331 vs 0.330) |

## Phase 0: re-score their released checkpoints (no training)

| | |
|---|---|
| Models | LFM2.5-2.6B base; `FineEnvs/LFM2.5-2.6B-multiharness-RL` and `-opencode-RL` (main, step 1000); `-multiharness-SFT`, `-opencode-SFT` |
| Trained harnesses (4) | OpenCode, Claude Code, Codex, Mini-SWE-Agent |
| Unseen harnesses (6) | Pi, Gemini CLI, Qwen Code, Vibe, OpenHands SDK, Terminus 2 (all marked validated in OpenEnv 0.7.0; none in FineEnvs' or Liquid's training list) |
| Tasks | `FineEnvs/SmolDataEnvs-harbor-test`, 250 |
| Samples | 3 per task per harness. Everything else is FineEnvs' evaluator (`eval/evaluate.py` @ 26ab9c6): temperature 0.8, top_p 1.0, 4,096 output tokens, 17 agent steps, 600 s agent timeout |
| Rollouts | 5 models x 10 harnesses x 250 x 3 = 37,500 |
| Stack | `eval_modal.py`: vLLM 0.25.1 with their serve flags, their Harbor env server (OpenEnv @ 7ee88d5), our fork of their evaluator (`evaluate.py`: harness list, k samples, sandbox backend), Modal sandboxes (theirs ran on E2B/Daytona) |
| Agent versions | their pins for the trained four; the unseen six pinned from what the smoke installs, recorded in amendment 1 |
| Pass rate | per model and harness: mean over graded cells; a cell with no grade (sandbox or harness failure) is excluded and reported as coverage |

Hypothesis H1: on the 6 unseen harnesses pooled, multi-harness RL beats
OpenCode-only RL. A win needs the paired-by-task 95% interval
(`compare_runs`, resampling tasks, the 3 samples averaged within a task) to
exclude zero. Pooling: each unseen harness weighted equally.

Also reported:
- each model minus base, per harness;
- `wai.harness.attribute`: harness vs model share of variance, ranking flips;
- SFT vs RL on unseen harnesses (their SFT lost 17 pts on Mini-SWE-Agent).

We publish the per-task, per-harness, per-sample table as a dataset. The
release only gives totals.

Rough cost: a few hours of one H100 per model, plus about 600 sandbox-hours.
Estimate, to confirm in the smoke test.

## Phase 1: randomized harness vs real harnesses (training)

Same recipe as theirs (FineEnvs `05-multi-harness-rl/train/multi_harness.py`,
async GRPO, 8 rollouts per group, 4,096 tokens, correctness plus the 0.1
tool-call bonus) on LFM2.5-2.6B. Only the harness source differs.

| Arm | Harnesses seen in training |
|---|---|
| A | OpenCode only (their control) |
| B | OpenCode, Claude Code, Codex, Mini-SWE-Agent, one per GRPO group (their multi-harness) |
| C | OpenCode through a randomizing proxy, new mapping per GRPO group |

**The randomizing proxy (what we build).** It sits inside OpenEnv's capture
proxy, which already rewrites requests between four API formats. For each
session it draws one mapping and applies it to every turn, so token prefixes
still line up:

- tool names: case, synonyms (`bash`, `Bash`, `run_shell`, `exec`);
- argument names: `file_path`, `filePath`, `path`;
- tool list order and description wording;
- context layout: tool results as `tool` messages or folded into `user`
  turns; older turns sometimes replaced by a summary;
- system prompt wrapper.

The model's tool calls are mapped back to the real names before the harness
sees them. The harness code is untouched. This covers KAT-Coder's format and
context-structure failure modes [5]. It does not cover control flow (retries,
stop rules).

Fixed across all arms:
- Loss weighted per rollout, not per row [4], so a Claude Code rollout counts once.
- 400 steps (their gain was mostly in the first 200 steps), 3 seeds per arm.
- Checkpoint picked on `SmolDataEnvs-harbor-eval` (144 tasks), never on test.
- Scored once on test with the Phase 0 setup: 10 harnesses, 3 samples.

Hypotheses, on the 6 unseen harnesses pooled, paired by task, seeds pooled:
- H2: C is no worse than B, with the 95% interval's lower bound above -3 pts.
- H3: C beats A, with the interval excluding zero.
- Every win must also clear the seed-to-seed spread (`eval_variance`).

Rough cost: 9 runs x about 16 h x 2 H100s, about 290 H100-hours, plus
67,500 eval rollouts. Estimate.

## Decision rules

- Phase 1 starts only after Jacob reviews Phase 0.
- An interval that crosses zero is reported as "no difference".
- If H2 holds, the claim is "one harness plus randomization matches four real
  harnesses on unseen harnesses", and the proxy goes upstream as an OpenEnv PR.
- If H2 fails, report the gap per harness and per randomization axis.

## Known risks

- Seed spread: our 2B SmolDataEnvs runs swung about 8 pts between seeds, so
  3 seeds may not be enough for a 3-pt margin.
- Modal sandboxes are untested here; E2B is the fallback.
- Terminus 2 runs on the host, not in the sandbox; it may need its own wiring.
- The randomizing proxy needs a check that the trainer sees exactly the
  tokens the model sampled (OpenEnv's ATIF comparison).

## References

1. FineEnvs, "The ultimate guide to multi-harness RL", 2026-10-01. https://fineenvs-multi-harness-rl.hf.space
2. Yu et al., OpenForgeRL, arXiv:2607.21557.
3. Peng et al., Orchard, arXiv:2605.15040.
4. He et al., Agent Lightning v1.0, arXiv:2608.17528.
5. KwaiKAT Team, KAT-Coder-V2.5, arXiv:2607.05471.
6. ToolRL-DR / RobustBench-TC, arXiv:2605.11928.
7. HarnessBandit, arXiv:2609.13739.
8. Zhang et al., "Stop Comparing LLM Agents Without Disclosing the Harness", arXiv:2605.23950.

## Amendment 1 (2026-10-02, after the infrastructure smoke, before any Phase 0 scoring run)

The smoke ran the base model on one test task under all ten harnesses on
Modal: 10 of 10 cells graded, no infrastructure errors. One task is not a
result and is not used in any analysis.

Changes, none of which touch the hypotheses, arms or win rule:

- Harbor pinned to 0.23.0, the version FineEnvs' VALIDATION.md records.
- The five unseen harnesses Harbor installs are pinned to what the smoke
  installed, matched to the registries on 2026-10-02: pi 1.0.0
  (`@earendil-works/pi-coding-agent`), gemini-cli 0.62.0, qwen-coder 0.24.7,
  vibe 2.25.8 (`mistral-vibe`), openhands-sdk 1.50.1. Terminus 2 ships inside
  Harbor, so the Harbor pin fixes it.
- FineEnvs' evaluator imports the tokenizer helper from their whitebox
  package, so the image installs it too. No behavior change.

## Amendment 2 (2026-10-02, infrastructure only, no result read)

The first Phase 0 launch stopped at about 3,990 of 7,500 cells per checkpoint:
the container was killed (exit -9) once about 8 MB per finished rollout had
filled its 32 GB. Separately, the two SFT runs began failing every rollout
call client-side after a few hundred cells, consistent with the env server's
session cap (56) filling with sessions that were never released. Fixes:
container memory 128 GB, session cap 1,024 (FineEnvs' Space setting). The runs
resume in place: graded cells are kept and only ungraded cells are retried,
which is FineEnvs' own evaluator rule. The progress log prints a running
pass rate, so partial numbers were visible; they played no part in these
changes, which touch only memory and the session cap; coverage per checkpoint and harness is reported as planned.
