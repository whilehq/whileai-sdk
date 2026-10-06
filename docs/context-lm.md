---
title: "Train a small context LM"
sidebarTitle: "Context LM"
description: "A model that keeps its own context as a file it rewrites: the Context Language Models harness and Eq. 6 as wai.methods.ContextFile, and a 1.5B model taken from 0.07 to 0.97 on held-out logs across four seeds."
---

**What you learn:** what a context file is, the credit that trains one (stepwise GRPO plus the paper's efficiency term), and the TRL call. **Needs:** one H100 and trl 0.19 to train; nothing to read the method. **Takes:** about 40 GPU minutes a seed.

A chat model's context only grows: every message and tool result is appended until the window fills. Context Language Models (arXiv:2609.37725) give the model its context as a file it can rewrite [1]. The model keeps what it needs and drops the rest. `wai.methods.ContextFile` is that harness, plus the credit the paper trains it with.

## The episode

![A normal LLM stacks every chunk of the log in its context; a context LM rewrites one short file of key: value lines after each chunk](/figures/context-lm-idea.svg)

An episode is $E$ edits and one answer. At edit $t$ the model sees the file and the next input, then writes the whole new file; the input is gone after that turn. At the end it sees only the file and the question. `wai.methods.KVLog` is a seeded task for it: 5 chunks of 8 `set key = value` lines over 8 keys, and a question about one key's final value.

```python
import whileai as wai

env = wai.methods.KVLog()
task = env.tasks(1)[0]
print(env.messages(task, 0, "")[1]["content"])  # the empty file and the first chunk
print(task["question"], task["gold"])
```

## The credit

Stepwise GRPO gives every step of trajectory $i$ the outcome advantage $r_i - \bar r$ over its group. The paper adds a success-gated efficiency advantage on the edits (Eq. 6): among the group's successes, a trajectory cheaper than their mean goes up and a dearer one goes down, clipped to $[-1, 1]$. Failures get nothing, and so does every trajectory in a group with fewer than two successes. Cost is prefix-reuse tokens: each step pays for the prompt past what is still in the KV cache, plus its reply.

```python
import whileai as wai

clm = wai.methods.ContextFile()  # w_eff 0.25, the paper's run
# Four trajectories: three right at costs 900, 1100 and 600, one wrong.
for steps in clm.credit([1, 1, 1, 0], [900, 1100, 600, 800], edits=5):
    print([round(a, 2) for a in steps])  # five edits, then the answer
```

The cheapest success gets the most credit on its edits. The answer step carries the outcome alone.

## Train it

`clm.trainer(GRPOTrainer)` is a TRL `GRPOTrainer` whose generation step plays the episode and gives each step its credit. Pass `env=` when you construct it, and set `num_iterations=1` and `beta=0`.

```python
import whileai as wai
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

env = wai.methods.KVLog()
clm = wai.methods.ContextFile()
rows = [{**t, "prompt": env.messages(t, 0, "")} for t in env.tasks(512)]
cfg = GRPOConfig(num_generations=8, num_iterations=1, beta=0.0, max_completion_length=256)
trainer = clm.trainer(GRPOTrainer)(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=[lambda completions, **_: [0.0] * len(completions)],  # the trainer pays the credit
    args=cfg,
    train_dataset=Dataset.from_list(rows),
    env=env,
)
trainer.train()
```

To evaluate, `clm.play(generate, tasks, env)` runs episodes with any generator that maps a list of chats and a token cap to `Reply` objects. `clm.report(episodes)` prints the pass rate, tokens a trajectory and file size.

## What it did

![Share of training logs answered right by step, paper arm: all four seeds rise from near 0 to above 0.9 within about 15 steps and end near 1.0](/figures/context-lm-learning.svg)

![The same held-out log after the last chunk: the untrained model's notes hold 1 of 7 current values and answer wrong; the trained model's notes hold 6 of 7 and answer 177, right](/figures/context-lm-before-after.svg)

![Held-out pass@1, four seeds: plain GRPO 0.95, the paper's Eq. 6 0.97, Eq. 6 on complete files 0.92; untrained 0.07](/figures/context-lm-accuracy.svg)

![Tokens a trajectory, four seeds: plain GRPO 1,315, the paper's Eq. 6 1,115, Eq. 6 on complete files 1,050; untrained 1,209](/figures/context-lm-tokens.svg)

[`recipes/papers/context-lm`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/context-lm) ran Qwen2.5-1.5B-Instruct for 60 steps with LoRA. It evaluated on 200 held-out logs, four seeds an arm, three arms:

| `gate` | pass@1 | Per seed | Tokens a trajectory |
|---|---|---|---|
| untrained | 0.07 | | 1,209 |
| `"off"`, plain stepwise GRPO | 0.95 | 0.98 / 0.98 / 0.93 / 0.91 | 1,315 |
| `"paper"`, Eq. 6 (the default) | **0.97** | 0.97 / 0.93 / 0.98 / 0.99 | **1,115** |
| `"complete"`, Eq. 6 on complete files | 0.92 | 0.84 / 0.94 / 0.95 / 0.97 | 1,050 |

Every seed learned to keep its context. Eq. 6 added +0.02 [+0.01, +0.03] pass@1 at 15% fewer tokens, and the verdict is flat because the seeds of plain GRPO spread wider than that. The trained files are short `key: value` tables, not copies of the log.

## References

1. Context Language Models. arXiv:2609.37725, 2026. https://github.com/facebookresearch/context-language-models
2. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
