# Chart-stage RL (prime-rl on Modal)

RL-trains a LoRA (r32) on `Qwen/Qwen3.8-27B` for the chart stage: chart image + `waiparse.prompts.CHART` in, HTML table(s) out, reward = `waiparse.rewards.chart_reward` (share of ParseBench chart rules passed). Runs in your own Modal workspace.

| File | What |
|---|---|
| `build_chart_data.py` | ChartNet -> `docparse-data:/train/charts/{train,val}.jsonl` + `images/` |
| `envs/docparse_charts/` | verifiers v1 taskset (image as a base64 `image_url` part), reward, CISPO loss, self-check |
| `modal_prime_rl.py` | Modal app `docparse-prime-rl`: `validate` (CPU), `train` (H200:4), `train8` (H200:8) |
| `configs/charts-smoke.toml` | 3 steps, 64x16, 3 infer / 1 train |
| `configs/charts-v1.toml` | 300 steps, 256x16, 2 infer / 2 train, eval on 200 val charts every 25 steps |

Run everything from `recipes/04-train/parsebench` with:

```bash
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
```

On Windows, `MSYS_NO_PATHCONV=1` stops Git Bash from rewriting `/runs/...` style values into `C:/Program Files/Git/...`.

## Validate (CPU, no GPUs)

```bash
modal run train/modal_prime_rl.py --config train/configs/charts-smoke.toml --run-name charts-smoke --validate-only
```

This runs `rl --dry-run` and then `python -m docparse_charts.selfcheck`, which checks three things:

- the CISPO gradient
- the reward on gold tables rebuilt from the rules
- the qwen3.8 render of real chart tasks, including image-token counts

Launch only if it prints `Dry run complete` and `SELFCHECK OK`. Check the literal strings, because `| grep` hides the exit code.

## Smoke, then the full run

```bash
modal deploy train/modal_prime_rl.py      # after any edit to modal_prime_rl.py or envs/
modal run train/modal_prime_rl.py --config train/configs/charts-smoke.toml --run-name charts-smoke --spawn
modal app logs docparse-prime-rl          # streaming; safe to kill
```

Always use `--spawn`, never `modal run --detach`: a detached run dies when the local client does. After the smoke passes, repeat with `charts-v1.toml --run-name charts-v1`.

In the smoke, check these:

- **Reward:** above 0 on step 1.
- **Truncation:** the share of rollouts cut at 8192 thinking tokens. If it is high, raise `max_completion_tokens`, `seq_len` and `max_model_len` together, or drop `reasoning_effort`.
- **Zero-advantage discards:** the share of groups the gate throws away.
- **`cispo/mismatch_kl`:** expect it higher than the text-only runs, because of images.

Outputs land on volume `docparse-runs` under `prime-rl/<run>/`:

- `logs/latest/{orchestrator,trainer,inference}.log`
- `monitors/file/metrics.jsonl`
- `checkpoints/step_N`
- `broadcasts/step_N`: only the newest 2 are kept.
- `adapters/step_N`: kept for good, every 25 steps plus the final step.

Read a file with `modal volume get docparse-runs prime-rl/<run>/logs/latest/orchestrator.log .`. Volume paths are root-relative, with no leading slash.

## Resume (24 h Modal cap, or a crash)

```bash
modal volume ls docparse-runs prime-rl/charts-v1/checkpoints     # latest step_N
modal run train/modal_prime_rl.py --config train/configs/charts-v1.toml --run-name charts-v1 --spawn --extra "--resume.step N"
```

Resuming needs a checkpoint with both `trainer/` and `orchestrator/`. `max_steps` counts from the start of the run, not from the resume point.

When forking with a different task list or sampler:

- Put `[resume] dir = "/runs/prime-rl/<run>/checkpoints/step_N"` in the TOML, below the top-level keys.
- Set `skip_progress = true` under both `[trainer.ckpt]` and `[orchestrator.ckpt]`.

Before trusting the volume, check for unexpected `logs/attempt_N` dirs: a warm container can re-run an input.

## Eval an adapter

`serve/serve_vlm.py` mounts `docparse-runs` at `/runs` and serves the adapter at `DOCPARSE_ADAPTER` as model `qwen3.8-27b-tuned`:

```bash
DOCPARSE_ADAPTER=prime-rl/charts-v1/adapters/step_100 modal deploy serve/serve_vlm.py
```

Poll `/v1/models` until `root` names the new adapter path before scoring. A warm container from the previous deploy can serve the old adapter for up to its scaledown window.

Then run the ParseBench pipeline `wai_agent_tuned`. It is the agent with `model = "qwen3.8-27b-tuned"`, so every stage uses the adapter. To keep the base server as it is and use the adapter only for the chart passes, deploy it as a second app with `DOCPARSE_APP=docparse-vlm-tuned` in front of the same command and run `wai_agent_v4_rl` (or `wai_agent_v4_med_rl`); they find it at `DOCPARSE_TUNED_SERVER`, else through `DOCPARSE_WORKSPACE`.

The online eval (`docparse-charts-val` in `metrics.jsonl`) is a trend line. The decision number is the offline ParseBench chart score on the served adapter.

## Recipe notes

- **Thinking on:** renderer `qwen3.8`, `enable_thinking = true`, `reasoning_effort = "xhigh"`. This matches the production chart stage. At this prime-rl pin, `Qwen/Qwen3.8-27B` resolves to the native qwen3.8 renderer, which has image support.
- **Loss: CISPO** (ScaleRL, arXiv:2510.13786). prime-rl only ships IPO and IcePop, so CISPO is plugged in through `[trainer.loss] type = "custom"`. To fall back to prime-rl's default, delete that table.
- **Not available at this pin:**
  - DAPO clip-higher: prime-rl has no PPO clip.
  - Prompt-level loss averaging: prime-rl uses token-level.
  - Batch-std advantage normalization: prime-rl uses the GRPO mean baseline only.
- **Memory:** the orchestrator holds every task with its PNG inlined as base64. Use `env.taskset.max_tasks` to cap it.
