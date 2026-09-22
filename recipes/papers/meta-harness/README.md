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
name (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`), or one OpenAI-compatible
router for all of them: `vllm:<model>@<url>` with the router's key in
`VLLM_API_KEY` is how the run below reached both models through
OpenRouter. With `--judge`, the key for the judge. Seconds offline; about
ten minutes per candidate and model live at the default concurrency.

The dry run demonstrates the loop with scripted candidates. Each candidate
carries an offline stand-in, a seeded agent whose planted-mistake rate the
candidate sets, so the numbers below show the mechanics (freeze, score,
propose, gate, attribute) and are not a replication. The live run is in
the Result section: Claude Haiku 4.5 as the search model, gpt-4.1-mini
held out, the three checked-in candidates, one round, and the gate passed
with +33 points on 30 held-out asks. What the paper reports: +7.7 points on online text
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
# the run in the Result section: both models through OpenRouter, VLLM_API_KEY = the OpenRouter key
python run.py --models "vllm:anthropic/claude-haiku-4.5@https://openrouter.ai/api/v1,vllm:openai/gpt-4.1-mini@https://openrouter.ai/api/v1" --budget 60 --concurrency 16 --propose --select --fresh
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
and every model replays it with `tasks=`, so the asks match. The judge is a
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
holdout on scripted: 02_check_result.py vs 00_baseline.py +0.27 [+0.15, +0.42] over 12 paired tasks -> clears zero
holdout on scripted-b: 02_check_result.py vs 00_baseline.py +0.50 [+0.38, +0.62] over 12 paired tasks -> clears zero
attribution on pass_at_1: 3 harnesses x 2 models, 12 tasks each cell
  harness              scripted  scripted-b
  00_baseline              66.7        45.8
  01_no_filler             75.0        58.3
  02_check_result          93.8        95.8
  spread explained: harness 82% [48..97], model 11% [0..37], interaction 8%
  harness moves the score by up to 38.5 points, the model by up to 11.8
  model ranking flips across harnesses
  the harness moved the score more than the model did; the leading model changes with the harness, so a model ranking from one harness does not carry
select: 02_check_result.py beats the baseline on the holdout and on a held-out model
```

The line that matters is the gate. `02_check_result` is best on the train
split, and on the held-out tasks it beats the baseline by 27 points with an
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
  the paired delta and interval per model, and the attribution verdict.

## Result

`python run.py --models "vllm:anthropic/claude-haiku-4.5@https://openrouter.ai/api/v1,vllm:openai/gpt-4.1-mini@https://openrouter.ai/api/v1" --budget 60 --concurrency 16 --propose --select --fresh`,
2026-09-22. Claude Haiku 4.5 is the search model and gpt-4.1-mini the
held-out model, both through OpenRouter; 60 frozen asks drawn by the
offline writer from the six seeds, 30 train and 30 holdout, 4 rollouts each
at the engine's default sampling; the program judge; the three checked-in
candidates, one round, no candidate written by hand for this run. The
output, verbatim:

```text
00_baseline        train 0.63 [0.53..0.75]  holdout 0.65 [0.54..0.75]  vllm:openai/gpt-4.1-mini@https://openrouter.ai/api/v1 0.89 [0.83..0.94]
01_no_filler       train 0.97 [0.93..1.00]  holdout 0.98 [0.95..1.00]  vllm:openai/gpt-4.1-mini@https://openrouter.ai/api/v1 1.00 [1.00..1.00]
02_check_result    train 0.99 [0.97..1.00]  holdout 0.98 [0.96..1.00]  vllm:openai/gpt-4.1-mini@https://openrouter.ai/api/v1 1.00 [1.00..1.00]
ledger: out/ledger.jsonl (3 candidates, 2 models)
proposal: out/proposal.md
holdout on vllm:anthropic/claude-haiku-4.5@https://openrouter.ai/api/v1: 02_check_result.py vs 00_baseline.py +0.33 [+0.23, +0.44] over 30 paired tasks -> clears zero
holdout on vllm:openai/gpt-4.1-mini@https://openrouter.ai/api/v1: 02_check_result.py vs 00_baseline.py +0.11 [+0.06, +0.17] over 30 paired tasks -> clears zero
attribution on pass_at_1: 3 harnesses x 2 models, 30 tasks each cell
  harness            vllm:anthropic/claude-haiku-4.5@https://openrouter.ai/api/v1         vllm:openai/gpt-4.1-mini@https://openrouter.ai/api/v1
  00_baseline                                                                65.0                                                          89.2
  01_no_filler                                                               98.3                                                         100.0
  02_check_result                                                            98.3                                                         100.0
  spread explained: harness 69% [52..86], model 13% [6..25], interaction 18%
  harness moves the score by up to 22.1 points, the model by up to 9.2
  the same model leads under every harness
  the harness moved the score more than the model did
select: 02_check_result.py beats the baseline on the holdout and on a held-out model
```

| Candidate | Haiku 4.5, holdout pass@1 | gpt-4.1-mini, holdout pass@1 | Haiku's failed rows, of 120 |
|---|---|---|---|
| `00_baseline` | 0.65 [0.54, 0.75] | 0.89 [0.83, 0.94] | 45: apology 38, sycophancy 3, boilerplate 3, hedging 1 |
| `01_no_filler` | 0.98 [0.95, 1.00] | 1.00 [1.00, 1.00] | 2: boilerplate 2 |
| `02_check_result` | 0.98 [0.96, 1.00] | 1.00 [1.00, 1.00] | 2: boilerplate 2 |

The gate: `02_check_result` is best on the train split (0.99 against
`01_no_filler`'s 0.97, a difference inside both intervals), and on the 30
held-out asks it beats the baseline by **+0.33 [+0.23, +0.44]** on Haiku
and by **+0.11 [+0.06, +0.17]** on gpt-4.1-mini, paired by task, both
intervals excluding zero. Attribution over the three-by-two grid of holdout
scores: the harness explains 69% of the spread (interval 52 to 86), the
model 13% (6 to 25), and the same model leads under every harness, so the
gain is the harness and no ranking flipped.

Noise floor: the baseline harness was run three times on Haiku over the
same 30 held-out asks (`out/tasks.jsonl` replayed, nothing redrawn). The
three holdout means are 0.65, 0.69 and 0.61, `eval_variance` run_std 0.042, and the
band a one-run-per-side delta has to clear is 0.25 (`noise_band(run_std,
df=2)` = 4.30 x sqrt(2) x run_std). +0.33 clears it; the +0.01 between the
two winning candidates does not, and the recipe does not call it a
difference.

What moved, read off the rows. On Haiku the baseline failed 45 of 120
held-out rows, 38 of them for an apology ("I apologize for the
inconvenience" after a refund that went through) and the rest for flattery,
boilerplate and one hedge; it claimed a false success on none. The
one-sentence rule in `01_no_filler` removed all of that (two boilerplate
rows left) and cut the mean reply from 778 to 164 characters; the
read-the-tool-result rule in `02_check_result` added nothing this set could
show, because the baseline's false-success rows were two, both on
gpt-4.1-mini. So the search found the harness and named the mechanism: on a
closed model, a style the rubric forbids is a prompt line away, and the
gate is what says the line held on asks and on a model it was not written
for. The ceiling this set leaves is 0.98; the next round is a harder set,
not a fourth candidate.

Runs on the platform: https://while.ai/platform/runs?agent=meta-harness
(one iteration per candidate and model, each pinned to its harness
fingerprint, the harness x model grid with the attribution sentence).

## Checks

| Check | Result |
|---|---|
| Held-out asks, paired: `compare_runs` on the 30 holdout tasks | +0.33 [+0.23, +0.44] on Haiku, +0.11 [+0.06, +0.17] on gpt-4.1-mini; both exclude zero |
| Held-out model | gpt-4.1-mini was never in the proposer's view; the pick holds there |
| Eval noise: 3 baseline re-runs on Haiku, same frozen asks | run_std 0.042, band 0.25; the delta clears it |
| The judge is a program, not a model | `common.judge`: phrase lists, a success claim checked against the tool results of the tool steps (text-only turns carry none), a leak check, at least one tool call |
| Reply length, Haiku holdout, mean characters | 778 baseline, 164 `01_no_filler`, 161 `02_check_result` |
| Tool calls, Haiku holdout, 120 rows | 186 baseline, 123 and 128 for the two candidates; every winning row still calls a tool |
| Frozen set | `out/tasks.jsonl`, written by the baseline's first run and replayed by every candidate and model; a `--fresh` run draws a new set (the seed pins the split, not yet the draw, whileai-sdk #645), so keep the file when you re-run |

## Learned

- The gate is the result. On the train split every candidate looked better than the last; on held-out asks and a held-out model the difference between the two winners vanished, and the recipe reports a tie there rather than the train-split order.
- On a closed model the lever is the line in the prompt, and it is worth measuring like a training arm: 65 to 98 of 100 with an interval, a noise floor, and a second model, for under two dollars of OpenRouter credit.
- A program judge fails for the reason it names. The baseline lost on apologies, not on tool use, and the trace tally says so before anyone reads a reply; the proposer's next candidate should come from that tally.
- A judge that reads tool steps has to skip the turns that have none. The first version of `common.judge` counted a text-only turn as a failed tool result and flagged every "has been processed" as a false claim; the fix is one `continue`, and the dry-run numbers did not move because the scripted agent never emits a text-only turn.

Verified 2026-09-22, whileai 0.113 (main source tree), Claude Haiku 4.5 and gpt-4.1-mini through OpenRouter. Runs: https://while.ai/platform/runs?agent=meta-harness

## The gate, in words

A candidate the proposer wrote from the train split's worst rows has seen
those rows. Its score there is the pick, not the proof. The proof is the
same harness on the tasks it never saw, on a model it was not tuned on, with
an interval over tasks that excludes zero on both. When `--models` names one
model, the held-out-model check is skipped and the selection says so.
`wai.harness.attribute` needs at least two candidates and two models; with
one model it is not printed. Lambert 2025, chapter Evaluation, is the rule
behind the split [2]; Miller 2024 is the interval over tasks [3].

## Next

Write `candidates/03_<name>.py` from what `out/proposal.md` says is still
wrong, and run `python run.py --dry-run --propose --select` again; the
skill in [`skills/harness-search`](../../../skills/harness-search) is that
loop as a playbook for a coding agent, with the stop rule and the
Changed / Moved / Why / Learned / Reproduce report. With keys, name real
models: `--models openai:gpt-4.1-mini,anthropic:claude-haiku-4-5`, or the
OpenRouter form the Result section used, and a model judge with `--judge
anthropic:claude-haiku-4-5`. On the checked-in set the ceiling is 0.98, so
the useful next run is a harder set: more seeds in `common.py`, a larger
`--budget`, and a judge with more to catch. To keep the picked
harness as a version on the platform, `harness.pin()` is the record
`tracked.run(harness=)` takes ([the harness page](../../../docs/reference/harness.md)).

## References

1. Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., Finn, C. Meta-Harness: End-to-End Optimization of Model Harnesses. arXiv:2603.28052, 2026.
2. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
3. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
