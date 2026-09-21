# Swarm rescue: passing rollouts on the tasks GRPO throws away

**What you learn:** on the tasks where the policy fails every one of 8
rollouts, whether a particle swarm of rollouts that share their best attempts
finds a passing answer that resampling alone does not, at the same budget.
**Needs:** `WHILEAI_API_KEY` for the hosted Qwen3-4B (`wai login`), or any
OpenAI-compatible endpoint via `--base-url`; `pyarrow` for the dataset.
Offline with `--dry-run`. **Takes:** seconds offline; about two hours for
every task on the hosted model.

```bash
python run.py --dry-run        # three toy tasks, a fake model, no key
python run.py --limit 20       # twenty code_contests tasks, hosted Qwen3-4B
python run.py                  # all 282 tasks
python run.py --reuse --post   # the numbers again from out/, then the Runs page
```

## The problem

GRPO learns from the spread inside a group of rollouts. On a task where all
8 fail, the advantage is zero and dynamic sampling drops the prompt [1]. The
SDK does the same: `simulate(mode="rl")` fills a group only when its first
two rollouts disagree. Those tasks are the hardest ones in the set, the
ones a training run most needs, and "sample more" rarely helps: the
policy puts almost no mass on the answer, so 24 more independent draws are
24 more misses.

A particle swarm [2] changes what the extra samples are conditioned on.
Each particle keeps its best attempt so far, sees the test it failed, and
sees a neighbour's best attempt. The LLM is the velocity update: given
its own best and a neighbour's, it writes the next program. Topology is
the variable. In a ring each particle sees two neighbours, so good ideas
spread slowly and the swarm stays diverse; in a star everyone sees the
global best, which is faster and collapses the swarm onto one idea [3].

## The recipe

1. Tasks: `deepmind/code_contests` test and valid splits [4], 282
   Codeforces-style problems, median rating 1900. The visible tests are
   the public tests plus 8 generated ones; the hidden tests are the
   private tests plus 32 more generated ones. Pass = the program's output,
   split on whitespace, matches on every test within 6 seconds.
2. Base: 8 rollouts per task from the hosted Qwen3-4B, thinking off,
   temperature 1, 1,500 tokens. A task where all 8 fail every test is an
   all-fail task. The script prints the per-task histogram, not the mean,
   because the mean hides the U shape.
3. Every all-fail task gets 24 more samples four ways:
   - **resample**: 24 fresh independent samples.
   - **solo**: 8 particles, 3 rounds. Round 0 is 8 fresh samples. In each
     later round a particle sees its own best attempt (most visible tests
     passed; ties go to the newer one) and the first visible test it
     failed, with the input, the expected output and what it printed, and
     writes a new program. No sharing.
   - **ring**: solo, plus the best attempt of the particle's two ring
     neighbours, shown beside its own.
   - **star**: solo, plus the best attempt of the whole swarm.

   Fitness inside a swarm is the visible tests only. A rescue is a program
   that passes the hidden tests too, which no arm ever sees. Every arm
   stops on a task at its first rescue, so a rescued task costs less than
   24 samples.
4. Measure: rescue rate per arm, the share of all-fail tasks with at least
   one rescue, with a bootstrap interval over tasks (`wai.pass_at`). Each
   swarm arm against resample is a paired delta over the same tasks with a
   sign-flip p-value (`wai.compare_runs`). `--noise-runs 2` re-runs the
   resample arm with new seeds and reports the band a delta has to clear
   (`wai.eval_variance`).
5. Export: every rescued program goes to `out/rescued.jsonl` as a bare row
   (the problem in, the passing program out) with no swarm context in it.
   That file is what a later SFT or distillation run trains on; the swarm
   prompts never leak into it.

## Result

Pending the full run. `python run.py --reuse` prints the table from `out/`.

## Learned

Pending the full run.

## References

1. Yu, Q. et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
2. Kennedy, J., Eberhart, R. Particle Swarm Optimization. *IEEE ICNN*, 1995.
3. Kennedy, J., Mendes, R. Population Structure and Particle Swarm Performance. *IEEE CEC*, 2002.
4. Li, Y. et al. Competition-Level Code Generation with AlphaCode. arXiv:2203.07814, 2022.
5. Feng, S. et al. Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence. arXiv:2410.11163, 2024. PSO over LoRA weights; this recipe runs PSO over programs instead.
6. Li, J. et al. QuestA: Expanding Reasoning Capabilities of LLMs via Question Augmentation. arXiv:2507.13266, 2025. Rescues the same zero-advantage prompts with partial reference solutions as hints; the swarm makes its hints from its own population.
