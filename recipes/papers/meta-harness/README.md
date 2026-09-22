# Meta-Harness: search over the harness, gate on held-out tasks and models

An outer loop over harness code. Every candidate is one Python file that
defines `harness(model) -> wai.Harness`; every candidate is scored on the
same frozen tasks; the proposer reads every prior candidate's source, score
and worst rows through the filesystem and writes the next file; and the
pick has to beat the baseline on held-out tasks and on a held-out model
before it counts. That is the loop of Lee, Nair, Zhang, Lee, Khattab and
Finn 2026 [1], with the harness as the object `wai.Harness` already
versions and runs.

What you will learn: how to freeze one task set so every candidate answers
the same asks (`tasks=`), what the proposer has to see (the paper's point is
that nothing is compressed: source, scores, traces), and what a search
result has to clear before it is a result (an interval that excludes zero
on held-out tasks, the same on a held-out model, and
`wai.harness.attribute` saying the gain is the harness). You need nothing
for `--dry-run`. The live run needs the provider key for every model you
name (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and, with `--judge`, the key
for the judge. Seconds offline; a few minutes per candidate live.

The dry run demonstrates the loop with scripted candidates. Each candidate
carries an offline stand-in, a seeded agent whose planted-mistake rate the
candidate sets, so the numbers below show the mechanics (freeze, score,
propose, gate, attribute) and are not a replication. The live run, with
keys, is the replication, and no number from it is claimed here because
none has been measured. What the paper reports: +7.7 points on online text
classification over a state-of-the-art context manager with 4x fewer
context tokens, +4.7 points on 200 IMO-level problems averaged across five
held-out models, and discovered harnesses that surpass the best
hand-engineered baselines on TerminalBench-2 [1].

## Run it

```bash
uv add whileai
cd recipes/papers/meta-harness
python run.py --dry-run --propose --select     # offline: score, propose, gate
python run.py --models openai:gpt-4.1-mini,anthropic:claude-haiku-4-5 --propose --select
python run.py --traces traces.jsonl --propose --select   # the frozen set is production traffic
# the run in the Result section: both models through OpenRouter, VLLM_API_KEY = the OpenRouter key
python run.py --models "vllm:anthropic/claude-haiku-4.5@https://openrouter.ai/api/v1,vllm:openai/gpt-4.1-mini@https://openrouter.ai/api/v1" --budget 60 --concurrency 16 --propose --select
```

| flag | default | what it does |
|---|---|---|
| `--k` | 4 | rollouts per task; pass@1 averages them, the interval is over tasks |
| `--budget` | 24 | tasks in the frozen set |
| `--holdout` | 0.5 | share of tasks held out; the train split picks, the holdout decides |
| `--models` | scripted,scripted-b | the search model first, then held-out models, as `provider:model` |
| `--judge` | program | the judge in `common.py`; a `provider:model` builds `wai.Judge(RUBRIC)` |
| `--seed` | 0 | the draw of the frozen set and of the split |
| `--concurrency` | 8 | rollouts in flight on a live model; scripted models run one at a time, and `reproducible=True` keeps the rows the same at any setting |
| `--traces` | None | production traffic as the frozen set: a JSONL of traces or an OTLP JSON batch, one task per distinct prompt, the latest days held out |
| `--cost-margin` | 0.0 | how much more per rollout than the baseline a pick may cost; 0 is matched cost |
| `--candidates` | candidates | the folder of candidate files |
| `--out` | out | ledger, traces, proposal, selection |
| `--propose` | off | write `out/proposal.md` for the proposer |
| `--select` | off | apply the gate, write `out/selected.json` |
| `--fresh` | off | drop `out/` first: a new frozen set |
| `--dry-run` | off | offline: every model name is a scripted model, no key |

Every run scores every file in `candidates/`, in name order, and the first
file is the baseline. The first candidate draws the frozen set from the
seeds in `common.py` with the offline template writer (`simulator=False`,
deterministic, no key) and saves `out/tasks.jsonl`; every later candidate
and every model replays it with `tasks=`, so the asks match. With
`--traces`, the frozen set is the agent's own traffic instead: one task per
distinct prompt in the file (a JSONL `wai.load_traces` reads, or an OTLP
JSON batch `wai.rows_from_otel` groups into conversations), at most
`--budget` of them. The holdout is then the latest days, whole days, until
it holds `--holdout` of the tasks, so no row the proposer reads comes from
the days that decide; train prompts that overlap a holdout prompt
(`wai.decontaminate`, the 8-gram rule) leave the proposer's window, and
`out/split.json` records the days, the keys and the count dropped. The judge is a
program: it reads the reply for filler, checks that a reply claiming
success sits on a tool result that succeeded, and that nothing privileged
leaked. The outer loop is you, or the coding agent running
[`skills/harness-search`](../../../skills/harness-search): read
`out/proposal.md`, write the next file, run again.

## A candidate

One file, one change. This is `candidates/01_no_filler.py`: the baseline's
worst rows open with a greeting, an apology or a hedge, so it tells the model
to answer in one sentence and caps the loop at four turns. `build` in
`common.py` turns it into the prompted loop on a real model, or the scripted
stand-in when the model name starts with `scripted`.

```python
"""Candidate 01: cut the filler. The baseline's worst rows open with a
greeting, an apology or a hedge before the answer; this candidate tells the
model to answer in one plain sentence and caps the loop at four turns."""

from __future__ import annotations

from common import BASE_INSTRUCTIONS, build

import whileai as wai
from whileai.harness import Disclosure

INSTRUCTIONS = (
    BASE_INSTRUCTIONS + " Answer in one plain sentence: no greeting, no apology, no hedging."
)
DISCLOSURE = Disclosure(max_turns=4)

# Offline stand-in: the filler kinds are gone, the tool-result mistakes stay.
SCRIPTED_RATE = 0.35
SCRIPTED_BEHAVIORS: tuple[str, ...] | None = ("ignore_fault", "leak", "hedging")


def harness(model: str) -> wai.Harness:
    return build(
        model,
        instructions=INSTRUCTIONS,
        label="01_no_filler",
        disclosure=DISCLOSURE,
        scripted_rate=SCRIPTED_RATE,
        scripted_behaviors=SCRIPTED_BEHAVIORS,
    )
```

The fingerprint hashes the instructions, the tool names and the
`Disclosure` fields, so the edit is a new harness version without anyone
naming it, and every row the candidate produces carries it.

## What you get

`python run.py --dry-run --propose --select`, the three checked-in
candidates on the two scripted models:

```text
00_baseline        train 0.71 [0.52..0.85]  holdout 0.67 [0.50..0.83]  scripted-b 0.46 [0.33..0.60]
01_no_filler       train 0.77 [0.58..0.92]  holdout 0.75 [0.60..0.88]  scripted-b 0.58 [0.42..0.75]
02_check_result    train 0.96 [0.90..1.00]  holdout 0.94 [0.83..1.00]  scripted-b 0.96 [0.90..1.00]
ledger: out/ledger.jsonl (3 candidates, 2 models)
proposal: out/proposal.md
train tasks led: 00_baseline.py 5, 01_no_filler.py 7, 02_check_result.py 12 -> pick 02_check_result.py
holdout on scripted: 02_check_result.py vs 00_baseline.py +0.27 [+0.15, +0.42] over 12 paired tasks, 0 the baseline passed and the pick failed -> clears zero
holdout on scripted-b: 02_check_result.py vs 00_baseline.py +0.50 [+0.38, +0.62] over 12 paired tasks, 0 the baseline passed and the pick failed -> clears zero
cost per rollout: 02_check_result.py at 1.00x the baseline in calls -> within the margin
attribution on pass_at_1: 3 harnesses x 2 models, 12 tasks each cell
  harness              scripted  scripted-b
  00_baseline              66.7        45.8
  01_no_filler             75.0        58.3
  02_check_result          93.8        95.8
  spread explained: harness 82% [48..97], model 11% [0..37], interaction 8%
  harness moves the score by up to 38.5 points, the model by up to 11.8
  model ranking flips across harnesses
  the harness moved the score more than the model did; the leading model changes with the harness, so a model ranking from one harness does not carry
select: 02_check_result.py beats the baseline on the holdout and on a held-out model at 1.00x its cost
```

The line that matters is the gate. `02_check_result` leads 12 of the 12
train tasks (per task, the best pass rate across candidates, ties shared),
and on the held-out tasks it beats the baseline by 27 points with an
interval of +15 to +42, which excludes zero, on the search model; on the
held-out model the same harness beats the same baseline by 50 points, +38
to +62. Then attribution over the three-by-two grid of holdout scores says
the harness explains 82% of the spread (interval 48 to 97) and the model
11%, so the gain is the harness. `01_no_filler` is not selected: it is not
the best on train, and on its own it would not pass either, +8 points on
the same 12 held-out tasks with an interval of +0 to +19, which is why the
skill's check writes a second file before it stops. On the scripted models
these numbers are what the planted rates make them; they show the gate
working, not the paper's result.

Three files carry the state between rounds:

- `out/ledger.jsonl`: one line per candidate, with its file, fingerprint,
  model, train and holdout pass@1 with intervals, task counts, and the path
  of its worst rows and of every row it produced.
- `out/proposal.md`: the proposer's view. Every candidate's source, its
  score with interval, its five worst rows on the train split (ask, reply,
  why the judge failed it), and one instruction: write
  `candidates/03_<name>.py`, then run again.
- `out/selected.json`: the gate's answer. The candidate picked or `null`,
  the train tasks each candidate led, the paired delta and interval per
  model with the count of holdout tasks the baseline passed and the pick
  failed, the cost ratio, and the attribution verdict.
- `out/split.json`: how the tasks were split (by seed, or by day for
  traces), the keys on each side, and the train prompts dropped for
  overlapping a holdout prompt.

## The gate, in words

A candidate the proposer wrote from the train split's worst rows has seen
those rows. Its score there is the pick, not the proof. The pick is the
candidate that leads the most train tasks, per task the best pass rate
across candidates with ties shared, so a mean gained by regressing a subset
does not win; that is the per-task frontier GEPA selects on [4]. The proof
is the same harness on the tasks it never saw, on a model it was not tuned
on, with an interval over tasks that excludes zero on both, and the count
of held-out tasks the baseline passed every time and the pick failed every
time is printed beside it. When `--models` names one model, the
held-out-model check is skipped and the selection says so.

The third check is cost. Wang et al. 2026 matched budgets and found harness
evolution lost to spending the same compute on more samples of the baseline,
and gained 0.6 points on held-out tasks when the harness was tuned on the
tasks it was scored on [5]. So the ledger carries cost per rollout (tokens
when every row has `usage`, model calls otherwise: the reply plus one per
tool call), and a pick may cost no
more than the baseline plus `--cost-margin` (0 by default: matched cost). A
candidate that wins by spending more is reported as a frontier point with
the margin that would accept it, not selected. `wai.harness.attribute` needs
at least two candidates and two models; with one model it is not printed.
Lambert 2025, chapter Evaluation, is the rule behind the split [2]; Miller
2024 is the interval over tasks [3].

## Next

Write `candidates/03_<name>.py` from what `out/proposal.md` says is still
wrong, and run `python run.py --dry-run --propose --select` again; the
skill in [`skills/harness-search`](../../../skills/harness-search) is that
loop as a playbook for a coding agent, with the stop rule and the
Changed / Moved / Why / Learned / Reproduce report. With keys, name real
models: `--models openai:gpt-4.1-mini,anthropic:claude-haiku-4-5`, and a
model judge with `--judge anthropic:claude-haiku-4-5`. To keep the picked
harness as a version on the platform, `harness.pin()` is the record
`tracked.run(harness=)` takes ([the harness page](../../../docs/reference/harness.md)).

## References

1. Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., Finn, C. Meta-Harness: End-to-End Optimization of Model Harnesses. arXiv:2603.28052, 2026.
2. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
3. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
4. Agrawal, L. A., et al. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv:2507.19457, 2025.
5. Wang, Y., et al. Rethinking the Evaluation of Harness Evolution for Agents. arXiv:2607.12227, 2026.
