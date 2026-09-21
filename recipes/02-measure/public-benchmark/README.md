# A public benchmark into the measurement

200 GSM8K test questions and a model's answers become the rows every
measurement reads, in one call, and every number comes out with its interval.

**What you learn:** `wai.rows()`, the road from a benchmark to `pass_at`,
`eval_variance`, `holdout_size`, `select` and `compare`; the five row keys
those calls read; why a third of the rollout budget goes to groups that
carry no gradient until `select` drops them.

**Needs:** nothing. No key, no GPU, no network. The 200 questions are
checked in.

**Takes:** seconds.

## Run it

```bash
uv add whileai
cd recipes/02-measure/public-benchmark
python run.py                    # 200 questions x 5 answers, both arms
python run.py --limit 40 --k 5   # what smoke.sh runs
python run.py --json out.json    # every number as one file
```

| flag | default | what it does |
|---|---|---|
| `--limit` | all 200 | fewer questions, for a smoke run |
| `--k` | 5 | answers per question |
| `--seed` | 0 | the stand-in model's draw |
| `--json` | off | write every report as one file |

## The road

```python
import whileai as wai

rows = wai.rows(questions, answers, wai.verify.MathEqual(), references=gold, task_ids=ids)
print(wai.pass_at(rows))  # pass@1 with its interval
noise = wai.simulations.eval_variance(run_1, run_2, run_3)
size = wai.simulations.holdout_size(0.05, before=rows)
print(wai.select(rows, mode="rl", band=(0.2, 0.8)))  # the groups that carry gradient
print(wai.compare(before, after, run_std=noise["run_std"], run_std_runs=3))
```

`questions` are strings, `answers` is five strings per question (five
draws of the same prompt, which is what `pass^k` and `select(mode="rl")`
need), `gold` is the number after `####` in each GSM8K solution.
`MathEqual` reads the gold off the row's `privileged.reference`, where
`references=` put it and where no training export projects it. The rows
that come back carry the five keys the measurement calls read (`task_id`,
`prompt`, `final_text`, `reward`, `markers`) and the same typed shape
`simulate()` writes; the contract is on
[docs/reference/rows.md](../../../docs/reference/rows.md).

The model is a seeded stand-in: each question has a difficulty drawn from
its own text, each arm a skill five points apart, and the answer is the gold
or a wrong number at that rate. Swap `answer()` in `run.py` for a call to
your model and nothing else changes.

## What you get

```
200 GSM8K test questions, 5 answers each, reward MathEqual

before: pass@1 0.50 [0.45..0.54] | pass^5 (pass_pow_k) 0.15 [0.10..0.20] | pass@5 0.85 [0.80..0.90] | headroom 0.35 (200 groups, k=5)
eval noise over 3 passes: run_std 0.031, a delta under 0.085 is re-run noise
holdout for a 5% gain: 313 tasks (base 0.50, k 5, sd from model)
rl selection: kept 700 of 1000 rows
  band 20%..80% pass rate: 0 asks dropped (0 too easy, 0 too hard)
  unanimous groups dropped: 60; duplicates dropped: 0; truncated drop: 0
  privileged leaks dropped: 0
  groups kept: 140

NO DIFFERENCE (within eval noise)
eval noise: run_std 0.031, a delta under 0.187 is noise (t(df=2)=4.30 x run_std x sqrt(1/1 + 1/1); run_std given from 3 re-runs)
answered: 100.0% before, 100.0% after
  pass_at_1                    0.496 -> 0.586  +0.090 [+0.053..+0.126]  noise  (200 paired)  noise<0.187
```

Four numbers to read:

- **pass@1 0.50 [0.45..0.54]** over 200 tasks, with pass@5 at 0.85: the
  stand-in gets most questions right some of the time. The bracket is a
  bootstrap over tasks, never rows.
- **60 of 200 groups dropped as unanimous.** Every one of the five answers
  scored the same, so a grouped update (GRPO, DAPO) has no advantage to
  learn from on them. That is 30% of the rollout budget; the issue this
  recipe answers measured 37% on a real outcome-reward arm. `select(mode="rl",
  band=(0.2, 0.8))` keeps the questions the model passes 20 to 80% of the
  time (Lambert 2025, chapter Reasoning; Yu et al. 2025, DAPO,
  arXiv:2503.14476, drops accuracy 0 and 1 groups from the batch).
- **313 tasks** to prove a five-point gain at this base rate and k. The 200
  here are not enough, and `holdout_size` says so before a GPU-minute is
  spent.
- **+0.090 [+0.053..+0.126], read as noise.** The paired interval excludes
  zero, and the report still says NO DIFFERENCE: with three re-runs the
  noise band is `t(df=2) = 4.30` times `run_std`, 0.187 here, and the delta
  is inside it. More re-runs narrow the band; the honest answer from three
  is that a nine-point delta on this eval could be the eval moving. A flat
  result is a result.

## Data

`gsm8k_test_200.jsonl` is the first 200 rows of the GSM8K test split
(Cobbe et al. 2021, arXiv:2110.14168; [openai/grade-school-math](https://github.com/openai/grade-school-math),
MIT licence), as `{"id", "question", "answer"}`, taken from
`openai/gsm8k` on the Hugging Face Hub (`main` config, `test` split). To
remake it with the `datasets` package:

```python
from datasets import load_dataset
import json

rows = load_dataset("openai/gsm8k", "main", split="test").select(range(200))
with open("gsm8k_test_200.jsonl", "w", encoding="utf-8") as fh:
    for i, row in enumerate(rows):
        fh.write(json.dumps({"id": f"gsm8k-test-{i}", **row}, ensure_ascii=False) + "\n")
```

## Next

Put your model behind `answer()`, or build the answers elsewhere and hand
`wai.rows` the lists. Then [`is-your-eval-any-good`](../is-your-eval-any-good)
runs the six checks on the rows before you trust a number from them, and
[`recipes/papers`](../../papers) reproduces a paper on the same road.
