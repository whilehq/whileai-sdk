---
title: "On-policy distillation"
sidebarTitle: "Distillation"
description: "Train without a reward: a frozen teacher (OPD) or the same model with a hint (OPSD) scores every token the student samples. What the papers say, the two calls, and a three-arm run on prime-rl with your own GPUs."
---

On-policy distillation trains a student on its own samples, scored token by
token by a teacher, with no reward function [1]. In OPD the teacher is a
stronger frozen model [2, 3]. In OPSD the teacher is the same model shown
something the student never sees: a demonstration, the reference answer,
or a successful rollout with the environment's feedback [4, 5, 6]. Both
learn where GRPO has nothing to learn from, because a group whose samples
all pass or all fail carries no advantage. `whileai` writes no loss for
either. It carries the method as an object with cited defaults, writes the
config prime-rl runs it with, and says which knobs the trainer reads.

<img className="block dark:hidden" src="/figures/distillation-paths-light.svg" alt="Student samples, then one of three per-token scorers: a reward for GRPO, a frozen teacher for OPD, the same model with a hint for OPSD, then prime_rl_config" />
<img className="hidden dark:block" src="/figures/distillation-paths-dark.svg" alt="Student samples, then one of three per-token scorers: a reward for GRPO, a frozen teacher for OPD, the same model with a hint for OPSD, then prime_rl_config" />

## What the research says

- **The loss is a per-token reverse KL on the student's own samples.** GKD
  defines the family: sample from the student, score each token under a
  frozen teacher, minimize a divergence per token [1]. The reasoning line
  writes it as an RL update with advantage
  `A_t = log p_teacher(x_t) - log p_student(x_t)`, no discount, no
  reference KL, no task reward [2]. prime-rl's `opd`, TRL's
  `DistillationTrainer` and the Tinker cookbook compute this.
- **It is an order of magnitude cheaper than RL for the same number.**
  Qwen3-8B reaches AIME'24 74.4 with OPD at 1,800 GPU hours against 67.6
  with RL at 17,920 [3]. The signal lives on the teacher's top-k support:
  OPD works by raising student/teacher top-k overlap, and k of 4 or more
  matches the sampled-token loss while k of 1 fails [7]; masking the
  signal to the teacher's top 32 keeps it off filler tokens late in long
  replies [8].
- **The failure modes are ceiling, tokenizer and initialization.** The
  student saturates at the teacher [2]. A tokenizer mismatch silently
  drops the signal [9]. A student that cannot produce teacher tokens needs
  an off-policy SFT cold start first [7].
- **Self-distillation beats GRPO where reward variance is zero.** With the
  reference answer as the hint, Qwen3-8B reaches AIME'24 77.8 against
  GRPO's 76.4 at one rollout per prompt and a fifth of the steps, because
  over half of GRPO's batches had zero reward variance [4]. With a
  successful rollout and the environment's feedback as the hint,
  LiveCodeBench v6 goes 41.2 to 48.8 with four times fewer generations [5].
  With a demonstration, a model learns a new task while keeping the old
  ones (SDFT: 70.2 vs 66.2 new-task, 64.5 vs 53.4 retained) [6].
- **The teacher must be anchored, and self-distillation hurts thinking
  models.** Unregularized self-distillation diverges; SDFT and SDPO hold
  the teacher as a moving average of the student [5, 6]. On Qwen3-4B and
  8B with thinking on, an answer-conditioned teacher costs 5.7 points
  avg@16 on AIME and HMMT, because the signal suppresses the high-entropy
  fork tokens where the model checks itself [10]; the hint cuts "wait" and
  "maybe" tokens by about 95% [11]. The fix is to route: failed rollouts
  self-distill, correct ones take the reward [12].

## The calls

```python
import whileai as wai

teacher = wai.Endpoint(
    url="http://localhost:8001/v1", model="PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
)

opd = wai.OPD(teacher)  # reverse KL, top_k 32, 4 samples, T 1.0, 8k cap
opsd = wai.OPSD(privileged="answer")  # the same model shown the task's answer, 1 sample
cfg = wai.prime_rl_config(
    "reverse-text",
    opsd,
    model="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT",
    out="opsd.toml",
)
print(cfg)
```

```text
prime-rl config: opsd.toml
  method opsd, model PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT, taskset reverse-text, 2 GPUs (1 inference, 1 trainer)
  100 steps x 64 prompts x 1 rollouts, max_off_policy_steps 8
  reads: privileged -> orchestrator.algo.demo_key = 'answer' (read from the task's info, then its top-level fields); template -> orchestrator.algo.template; samples -> orchestrator.group_size; temperature -> orchestrator.train.sampling.temperature; max_tokens -> orchestrator.train.sampling.max_completion_tokens; learning_rate -> trainer.optim.lr = 5e-06
  ignores: anchor=ema:0.01: prime-rl opsd scores against the live policy; an EMA or initial-weights teacher (SDFT, SDPO) is not offered there
  warning: OPSD costs points on thinking models (Kaur et al. 2026, arXiv:2607.05184) and needs in-context learning strong enough to use the hint (about 7B up); run a GRPO arm on the same holdout before believing a number.
  run: uv run rl @ opsd.toml
```

Every default is a named constant in `defaults.py` with its source. The
teacher for OPD has to be a server you run: it is scored on the student's
tokens, which needs prompt log-probabilities, and a chat API is refused
with that sentence. `privileged` is the task field the teacher reads
(`info` first, then the task's own fields, so a public taskset's `answer`
works as given). A knob prime-rl cannot honor is either listed under
`ignores` with the reason or refused before anything is written.

| Knob | Default | Source |
|---|---|---|
| `OPD.divergence` | `reverse_kl` | GKD [1] Table 1; [2]; prime-rl `opd`; TRL `beta=1.0` |
| `OPD.top_k` | 32 | [8] top-32 support; [7] k >= 4 matches the sampled-token loss |
| `OPD.samples` | 4 | tinker-cookbook `group_size=4`; [7] |
| `OPD.max_tokens` | 8192 | [7] signal decays past about 7k tokens |
| `OPSD.privileged` | `demonstration` | SDFT [6]; `reference` [4], `hint` [13], `feedback` [5] |
| `OPSD.anchor` | `ema:0.01` | [5] alpha 0.01; [6] 0.01 to 0.05; unanchored diverges [5] |
| `OPSD.samples` | 1 | [4], [6]; no group baseline to form |
| `OPSD.learning_rate` | 5e-6 | [4], [5], [10], [12] |

## Run it: three arms on one taskset

[`recipes/04-train/prime-rl`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/prime-rl)
trains the same 0.6B student on the same taskset three ways, on Modal with
your keys, two GPUs per arm, and writes `results.json`:

```bash
uv add whileai modal
modal token set --token-id ... --token-secret ...
cd recipes/04-train/prime-rl
python run.py --validate     # write the three configs, dry-run each on a CPU container
python run.py                # deploy, spawn the three runs (2 GPUs each), record the call ids
python run.py --collect      # when they finish: results.json with the paired deltas
```

The student is `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`, the taskset is
`reverse-text` (reverse a sentence character by character; the reward is
the longest-common-subsequence ratio to the true reversal), and the last
128 of its 1,000 prompts are held out from every arm. The three arms share
20 steps and a learning rate of 3e-6, the values prime-rl's own debug
configs use; only the algorithm differs. GRPO reads the reward. OPSD shows
the same model the task's `answer` and reads the reverse KL per token. OPD
serves the RL-trained checkpoint of the same model frozen on the first GPU
and reads the reverse KL to it.

The container is Prime Intellect's published image, pinned to a commit
(`ghcr.io/primeintellect-ai/prime-rl:v0.8.1.dev63`). `modal_prime_rl.py`
writes the config to a volume, starts the frozen teacher when one is
named, runs `rl`, and returns the metrics file.

## Read the number

Run `e2e1` (2026-09-20): three H100:2 containers, under fifteen minutes
each. The number is the taskset's reward, the LCS ratio to the true
reversal, on the 128 held-out prompts, paired by prompt from step 1 to
step 20; the interval is `wai.compare`'s paired bootstrap at 95%. The
noise floor is the three step-1 scores of the same untrained student, one
per arm (0.082, 0.070, 0.109): `run_std` 0.020, so a delta under 0.121 is
noise.

| Arm | Held-out LCS, step 1 | step 20 | Delta [95%] | Verdict | Replies cut at the cap, step 1 to 20 |
|---|---|---|---|---|---|
| `grpo` | 0.082 | 0.806 | +0.724 [+0.678, +0.767] | moved | 90% to 0% |
| `opsd` | 0.070 | 0.252 | +0.182 [+0.110, +0.254] | moved | 90% to 66% |
| `opd` | 0.109 | 0.834 | +0.725 [+0.673, +0.772] | moved | 88% to 1% |

Head to head at step 20, OPD against GRPO is +0.028 [+0.013, +0.045],
inside the noise band: a frozen teacher and no reward reached the same
number as the reward. OPSD against GRPO is -0.554 [-0.614, -0.495]. The
logged reference KL says why: OPD's moved from -0.294 to -0.108 over the
run, OPSD's sat at -0.08 throughout. Showing a 0.6B model the answer
barely changed what it predicted for its own tokens, the in-context floor
the SDFT and SDPO papers put near 7B [5, 6], and the warning the writer
printed before the run. Full curves, configs and the printed reports are
in the recipe's `results.json`.

## Next

<CardGroup cols={2}>
  <Card title="Simulations" href="/simulations">The rows a taskset is built from, and `export_environment` for a verifiers package.</Card>
  <Card title="Reward hacking" href="/reward-hacking">What to watch on any training curve, distillation included.</Card>
  <Card title="Character training" href="/character-training">A privileged teacher of a different kind: the principle only the judge sees.</Card>
  <Card title="Parameters" href="/reference/parameters">Every knob and its source.</Card>
</CardGroup>

## References

1. Agarwal, R. et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024. arXiv:2306.13649.
2. Lu, K. et al. On-Policy Distillation. Thinking Machines Lab, 2025; code in tinker-cookbook `distillation/`.
3. Yang, A. et al. Qwen3 Technical Report. arXiv:2505.09388, 2025.
4. Zhao, S. et al. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. arXiv:2601.18734, 2026.
5. Hübotter, J. et al. Reinforcement Learning via Self-Distillation. ICML 2026. arXiv:2601.20802.
6. Shenfeld, I. et al. Self-Distillation Enables Continual Learning. arXiv:2601.19897, 2026.
7. Li, Y. et al. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe. arXiv:2604.13016, 2026.
8. Fu, Y. et al. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes. arXiv:2603.25562, 2026.
9. Sun, J. et al. SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation. arXiv:2605.07711, 2026.
10. Kaur, S. et al. Rethinking On-Policy Self-Distillation for Thinking Models. arXiv:2607.05184, 2026.
11. Kim, J. et al. Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? arXiv:2603.24472, 2026.
12. Li, G. et al. Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing. arXiv:2604.02288, 2026.
13. Penaloza, E. et al. Privileged Information Distillation for Language Models. arXiv:2602.04942, 2026.
