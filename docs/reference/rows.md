---
title: "Rows: the contract every measurement reads"
sidebarTitle: "Rows"
description: "The five row keys pass_at, compare, eval_variance, holdout_size, decontaminate and select read, which are required, and wai.rows(), the one call that builds them from a public benchmark."
---

A row is one completion of one prompt with the verdict on it. `simulate()`
emits rows; so does `wai.rows()`, from prompts and completions you already
have (a public benchmark, a model's answers to it). Every measurement call
reads the same five keys, so the rows from either source go into every call
unchanged.

## The five keys

| key | required | what it is |
|---|---|---|
| `task_id` | no, defaults to a hash of the prompt | what the completions of one prompt group under. Every interval is over tasks, never rows (Miller 2024, arXiv:2411.00640), so ten rows with one `task_id` are one task. The engine writes the same value as `scenario_id`. |
| `prompt` | yes | the prompt text. `decontaminate` matches on it by default. |
| `final_text` | yes | one completion. A verifier reads it as the candidate. |
| `reward` | for everything but `decontaminate` | a number in [0, 1]. 0 and 1 are the binary outcome `pass_at` counts and `select` bands on; anything between is a partial score those two skip. |
| `markers` | no | `{name: number}` per completion. **Input as well as output.** `compare(proxy="marker:name")`, `eval_variance` and `judge_trust` read them wherever they came from: a judge wrote them, or you did. |

A row may carry more (`messages`, `steps`, `judge_name`, `lineage`,
`privileged`) and the calls use what they find. Nothing else is required.
The typed view is `whileai.simulations.schema` (`Task`, `Rollout`,
`Judgment`, `Marker`); `wai.rows()` builds through it, so its rows carry
`schema_version` and the same shape a run writes.

## Build them: `wai.rows()`

```python
import whileai as wai

questions = ["What is 2 + 3?", "What is 7 * 6?", "What is 10 - 4?", "What is 9 / 3?"]
gold = ["5", "42", "6", "3"]
answers = [["5", "4", "5", "5"], ["41", "41", "42", "40"], ["6"] * 4, ["3", "2", "3", "3"]]

rows = wai.rows(questions, answers, wai.verify.MathEqual(), references=gold)
print(rows[0]["task_id"], rows[0]["reward"], rows[0]["judge_name"])
```

* `prompts`: strings, or message lists (`[{"role": "user", "content": ...}]`);
  the row's `prompt` is the last user turn and `messages` keeps the list.
* `completions`: one string per prompt, or one sequence per prompt. The
  sequence is k completions of the same prompt, which is what `pass_at`'s
  k-way numbers and `select(mode="rl")` need.
* `reward`: a verifier (`wai.verify.MathEqual()`, `Numeric`, `ExactMatch`,
  `CodeExec`), a callable `(prompt, completion)` or
  `(prompt, completion, reference)` returning a number in [0, 1], a
  judge-contract callable `(row) -> verdict`, or the numbers themselves,
  nested like `completions` or flat. A verifier or callable runs through
  the same path `data.grade()` uses, so the row says what scored it.
* `references=`: the gold per prompt. It lives under `privileged.reference`,
  where a verifier reads it and no training export projects it.
* `task_ids=`, `markers=`: names per prompt, measurements per completion.

Precomputed scores and your own markers:

```python
scores = [[1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1]]
lengths = [[{"chars": len(a)} for a in group] for group in answers]
given = wai.rows(questions, answers, scores, markers=lengths)
print(given[1]["reward"], given[1]["markers"])
```

## Then measure

Everything downstream takes the list as it is.

```python
tuned = wai.rows(
    questions,
    [["5"] * 4, ["42", "42", "42", "41"], ["6"] * 4, ["3"] * 4],
    wai.verify.MathEqual(),
    references=gold,
)
print(wai.pass_at(rows))  # pass@1 with its interval, pass^k, pass@k
print(wai.select(rows, mode="rl", band=(0.2, 0.8)))  # the groups that carry gradient
print(wai.compare(rows, tuned))  # is the change real
```

`select(mode="rl")` keeps whole groups the model passes between 20% and
80% of the time (Lambert 2025, chapter Reasoning; DAPO, arXiv:2503.14476)
and drops the rest: a group all-pass or all-fail has no advantage to learn
from. Its report names every gate that dropped a row, so a benchmark whose
rows all vanish says why.

The worked example, 200 GSM8K test questions through `MathEqual`, three
eval passes, `holdout_size` and a `compare` report, offline, is
[`recipes/02-measure/public-benchmark`](/recipes/02-measure/public-benchmark).

## What the calls read

Every call in this table is on `wai`, so the one import reaches all six.

| call | groups by | reads | skips |
|---|---|---|---|
| `wai.pass_at` | `task_id` | binary `reward` | partial and unjudged rows |
| `wai.compare` | `task_id`, paired across arms | `reward`, every shared `markers` name | tasks present on one side |
| `wai.eval_variance` | `task_id` per run | `reward`, `markers` | |
| `wai.holdout_size(before=)` | `task_id` | `reward` | |
| `wai.decontaminate` | | `prompt` (`fields=`) | |
| `wai.select(mode="rl")` | `task_id` | binary `reward`, `final_text` | non-binary rewards, empty or cut-off replies, unanimous groups |

## The noise floor and the size of the set

A delta is a result against a floor, so two of those six calls run before
and beside `compare`. `wai.holdout_size(effect)` says how many paired
tasks can prove a gain of that size, before any GPU runs.
`wai.eval_variance(run_1, run_2, run_3)` evaluates the same model several
times and reports how far the number moves on its own. Both print
themselves.

```python
print(wai.holdout_size(0.10, base=0.45, k=4))  # n_tasks, task_std, what it assumes

passes = [
    wai.rows(questions, answers, wai.verify.MathEqual(), references=gold),
    wai.rows(questions, answers, wai.verify.MathEqual(), references=gold),
    wai.rows(questions, answers, wai.verify.MathEqual(), references=gold),
]
floor = wai.eval_variance(*passes)  # run_std across the passes, and the band it implies
print(floor)
print(
    wai.noise_band(floor["run_std"], df=floor["n_runs"] - 1)
)  # the same band from the number alone
print(wai.compare(rows, tuned, run_std=floor["run_std"], run_std_runs=floor["n_runs"]))
```

Hand the floor to `wai.compare(run_std=, run_std_runs=)` and a delta
inside the band reads as what re-running the eval does on its own, not as
a gain. Three re-runs is the fewest that give a standard deviation worth
reading; below that the report says so instead of printing a bare number.
