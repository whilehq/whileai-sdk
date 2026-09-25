# SmolDataEnvs: data-analysis tasks with a reward that is a program

[SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) [1] is
5,394 questions about real Kaggle tables, each with one known answer. The
policy writes a Python program, the program runs next to the tables, and the
dataset's own grader compares the last line it prints to the gold. No model
grades anything, so the reward is the same on every run.

What you will learn: how to wrap an outside environment as a whileai
verifier, the two reward rules that keep a data-analysis reward honest
(an environment failure is `None`, not `0`; a shell `echo` earns nothing),
and how the graded rows become GRPO groups. You need nothing for the offline
run; the live run needs any model the SDK can call. Seconds offline, about a
minute live on 24 tasks.

## Run it

```bash
uv add whileai
cd recipes/01-simulate/smol-data-envs
python run.py                                                       # offline: canned replies, no key
python run.py --agent openai:gpt-4.1-mini --split eval --limit 24   # live: your key, a few cents
```

The live run needs `pandas` and `pyarrow` in the same interpreter (the
programs import pandas; a few tasks read HDF5 and need `h5py`). It downloads
the split and each task's tables from Hugging Face into `raw/`, public and
with no token, once.

| flag | default | what it does |
|---|---|---|
| `--agent` | none (offline) | any SDK spec: `openai:<model>` (honors `OPENAI_BASE_URL`), `anthropic:<model>`, `bedrock:<id>@<region>`, `vllm:<model>@<url>` |
| `--split` | `eval` | `train` (5,000), `test` (250) or `eval` (144); tasks with no tables are dropped |
| `--limit` | 24 | first N tasks by `task_id`; `0` for the whole split |
| `--k` | 4 | rollouts per task, the GRPO group size |
| `--temperature` | 0.8 | sampling temperature; a group at 0 has no spread to learn from |

## The environment

```python
from env import DataEnvReward, fixture_tasks, prompt_for, task_row_fields
from whileai.simulations.score.judging import run_judge

task = fixture_tasks()[0]
row = {
    "scenario_id": task["task_id"],
    "prompt": prompt_for(task),
    "final_text": reply,  # the model's reply: one fenced python program
    **task_row_fields(task),  # gold under privileged.reference, tables, tolerances
}
scored = run_judge([row], DataEnvReward())  # a verifier honors the judge contract
```

- **The reward** (`DataEnvReward` in [env.py](env.py)) is a `Verifier`: 1
  when the printed answer matches under the dataset's grader, 0 when it does
  not, when the program crashed, or when the reply was a shell block.
- **Ungraded is not wrong.** A table that failed to download raises
  `VerifierError`, the row gets `reward=None`, `pass_at` skips it and
  `optimize` drops it. Scoring it 0 would teach the policy that correct
  programs fail on an unlucky network.
- **Tolerances are the task's.** Integer answers carry `atol=0`; decimals
  carry their own `atol`/`rtol` (for example 0.05 and 1%). The grader
  ([grader.py](grader.py), vendored at the dataset's revision, MIT) tries
  exact, numeric, list, then Math-Verify.
- **The gold stays out of training files.** It rides in
  `privileged.reference`, which `wai.training_rows` never projects, and a
  failure reason that would quote it prints `<reference>` instead.
- **Isolation.** Programs run in a fresh temp directory, a subprocess with
  a 60 s timeout. That stops runaway loops; it is not a security boundary.
  Run a policy you do not trust in a container or a sandbox service.

## What you get

Offline, four fixture tasks walk every branch: right, right within the
tolerance (`5768` against `5768.04`), wrong value, crashed, shell, all-pass
(no gradient), and ungraded.

Live, Claude Haiku 4.5 on the first 24 eval tasks, k=4, one reply per
rollout (2026-09-25):

```
== why rollouts failed
    24  crashed: KeyError
    17  ran, wrong value
     4  crashed: UnicodeDecodeError
     4  crashed: ModuleNotFoundError
     ...
== pass@1 by difficulty (0 ungraded rollouts left out, not scored 0)
  tier     tasks  pass@1          95% CI
  easy         8    0.78      0.53..1.00
  medium      16    0.28      0.12..0.45
  all         24    0.45      0.28..0.61

== optimize(mode='rl'): 9 task groups carry a gradient, 36 rows; 0 training rows carry the answer
```

Haiku 4.5 solves 45% of these tasks in one shot (95% interval 28% to 61%,
tasks resampled). The biggest loss is `KeyError`: the program guesses a
column name it never looked at. 9 of the 24 groups split between right and
wrong, which is where GRPO has something to learn; the other 15 are
unanimous and drop out.

## Next

- Give the policy a look before it answers: a multi-turn agent with a code
  tool fixes most `KeyError` crashes. Wrap it with the
  [`bring-your-own-agent`](../bring-your-own-agent) contract and pass it as
  `--agent`.
- Measure whether the number means anything:
  [`is-your-eval-any-good`](../../02-measure/is-your-eval-any-good).
- Train on the split groups: [`04-train/grpo`](../../04-train/grpo).

## References

1. Adithya S K. SmolDataEnvs: verified data-analysis environments from
   Kaggle notebooks. Hugging Face dataset `FineEnvs/SmolDataEnvs`, 2026.
   Release notes: [FineEnvs PR #14](https://github.com/adithya-s-k/FineEnvs/pull/14).
2. Lambert et al. Tülu 3: pushing frontiers in open language model
   post-training (RLVR). [arXiv:2411.15124](https://arxiv.org/abs/2411.15124).
