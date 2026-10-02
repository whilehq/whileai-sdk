# Multi-harness transfer: does RL across coding agents carry to agents it never saw?

FineEnvs trained LFM2.5-2.6B with RL inside four coding-agent harnesses
(OpenCode, Claude Code, Codex, Mini-SWE-Agent) and it improved in all four [1].
Every harness that model was scored in is one it trained in, with one try per
task. This recipe re-scores their released checkpoints in six harnesses none
of them trained in (Pi, Gemini CLI, Qwen Code, Vibe, OpenHands SDK,
Terminus 2), three tries per task, and puts a paired interval on every gap.

What you will learn: how to evaluate an open model inside real, unmodified
coding agents on Modal; how to score a claim about "generalizes across
harnesses" with a held-out harness instead of a held-out task; and how to
read a gap that comes with a range.

The question, the test and the rule for a win are in
[`PREREGISTRATION.md`](PREREGISTRATION.md), committed before any scoring run.

## Results

Phase 0 is running. This section is filled from `results.json` when it lands.

## Run it

Needs: a Modal account (one H100 per checkpoint, plus one CPU sandbox per
rollout) and a Modal secret named `huggingface-secret` holding `HF_TOKEN`.

```bash
pip install modal && modal setup
modal run eval_modal.py::smoke                          # base model, 1 task x 10 harnesses
modal run --detach eval_modal.py::phase0 --model base   # then oc-rl, mh-rl, oc-sft, mh-sft
modal volume get multi-harness-transfer phase0 out      # copy cells locally
python analyze.py out/phase0                            # tables + results.json
```

| File | What it does |
|---|---|
| `eval_modal.py` | one H100 container per checkpoint: vLLM, FineEnvs' Harbor env server, the evaluator; Modal sandboxes |
| `serve_env.py` | starts the env server with the capture proxy published by `modal.forward` |
| `evaluate.py` | FineEnvs' evaluator with a harness list, k samples and a sandbox argument; one file per cell, reruns retry only ungraded cells |
| `analyze.py` | per-harness pass rates with intervals, the pre-registered comparison, each model against base, harness vs model attribution |

| Setting | Value | Source |
|---|---|---|
| Tasks | `FineEnvs/SmolDataEnvs-harbor-test`, 250, pinned revision | FineEnvs `prepare.py` |
| Sampling | temperature 0.8, top_p 1.0, 4,096 output tokens | FineEnvs `eval/evaluate.py` |
| Agent loop | 17 steps, 600 s | same |
| Samples | 3 per task per harness | ours |
| Interval | paired bootstrap over tasks, 95%, `compare_runs` | Miller 2024 [2] |

## Phase 1 (not started)

Training arms that ask whether one harness with shuffled tool names and
context layout matches four real harnesses. Starts only after Phase 0 is
reviewed. See the pre-registration.

## References

1. FineEnvs. The ultimate guide to multi-harness RL. 2026. https://huggingface.co/spaces/FineEnvs/multi-harness-rl
2. Miller. Adding Error Bars to Evals. 2024. arXiv:2411.00640
