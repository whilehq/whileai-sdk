# Can the markers be trusted?

Check every behavioral marker against ground truth. `seeded_agent` plants six behaviors
and writes what it planted on each row, so a marker's recall and precision are computable
rather than assumed. Across 5,064 rows on 12 seeds the six markers that name those
behaviors split three ways:

| planted behavior | marker | family | planted | recall | 95% interval | precision |
|---|---|---|---|---|---|---|
| `hedging` | `no_hedging` | style | 471 | 1.000 | [0.992, 1.000] | 1.000 |
| `sycophancy` | `no_sycophancy` | style | 415 | 1.000 | [0.991, 1.000] | 1.000 |
| `apology` | `no_apology` | style | 414 | 1.000 | [0.991, 1.000] | 1.000 |
| `boilerplate` | `no_boilerplate` | style | 434 | 1.000 | [0.991, 1.000] | 1.000 |
| `ignore_fault` | `reported_failure` | trace | 207 | **0.763** | [0.701, 0.816] | **0.598** |
| `leak` | `no_secrets` | trace | 600 | **0.000** | [0.000, 0.006] | — |

Three regimes, and only the middle one is what you would expect a detector to look like:

- **The four phrase markers are exact and circular.** Recall 1.000, zero false alarms —
  because `seeded_agent` plants `hedging` by appending `"It depends."` and `no_hedging`
  greps for `it depends`. That is the generator agreeing with its own definition.
- **`reported_failure` is a real detector and not a trustworthy one.** It catches 76% of
  `ignore_fault` at precision 0.598 — 106 false alarms against 158 catches.
  `judge_trust` calls it `ok=False`, correctly.
- **`no_secrets` never fires.** Six hundred planted leaks, zero detections. The marker is
  named for the behavior, it is stamped on every row, and it is constant at 1.000.
  Nothing else in the library sees leaks either: the best marker anywhere, by F1, is
  `argument_grounding` at **0.202**.

So **24.2% [22.6%, 25.9%]** of the rows a green `style_report` passes carry a planted
failure, and a `style_report`-measured improvement recovers only **68.1% [66.2%, 69.9%]**
of the improvement that actually happened.

**Turning on every marker family makes it worse, not better.** Stamping `style_markers`,
`trace_markers` and `mark_grounding` and requiring all thirteen to be clean takes the
missed-failure share from 24.2% to 20.7% [19.4%, 22.0%] — and drops the recovered share
from 68.1% to **60.8% [59.2%, 62.4%]**. The two markers that see anything behavioral are
imprecise enough that their false alarms cost more than their catches gain.

What you will learn: which marker family sees what, why a detector validated on
`seeded_agent` is validated against itself, and how to put ground truth under a
behavioral metric before you quote it. You need **nothing**: no model key, no GPU, no
network. Six seconds locally; the 12-seed version is about two minutes of CPU on your own
Modal.

## The question

> `compare()` reports a delta on every shared marker beside pass@1, and
> `compare(proxy="marker:X")` will tell you whether the reward is paying for a tic. Both
> take on faith that when marker X moves, behavior X moved. Does it?

There is exactly one place in this library where a behavior has a machine-checkable gold
label: `seeded_agent`'s own docstring says each row "says what was planted, so a grader
**or a marker** can be checked against the truth." This is that check.

## Three marker families, and they do not agree on what exists

This cost the most time, so it goes first. The library stamps markers from three calls
and they carry different vocabularies:

| call | names it stamps |
|---|---|
| `score.style.style_markers` (what `style_report` prints) | `no_boilerplate`, `no_hedging`, `no_apology`, `no_sycophancy`, `answered` |
| `trace_markers` | the above plus `honest_claims`, `no_bypass`, `no_destructive`, `no_secrets`, `no_suppression`, `no_test_tampering`, `reported_failure` |
| `mark_grounding` | the above plus `argument_grounding` |
| `STOCK_MARKERS` (via the deprecated `mark_rows` / `behavioral_markers`) | `boilerplate`, `self_reference`, `hedging`, `refusal`, `sycophancy` — a fourth list, with the opposite polarity |

Both markers that name a behavioral failure — `reported_failure` and `no_secrets` — live
in `trace_markers`, which is the family a reader following `style_report` never reaches.
If you measure behavior with what the docs show you, you are measuring four phrase lists.

Six of the eight non-style markers are constant at 1.000 on every row of this run
(`honest_claims`, `no_bypass`, `no_destructive`, `no_secrets`, `no_suppression`,
`no_test_tampering`). That is closed issue [#270]'s shape — a marker that never had a
chance to fail reporting as clean — and `no_secrets` is the case where it matters,
because 600 rows in this run are exactly what it is named for.

## The setup

- **Ground truth is the plant record**, `row["seeded"]`, not my opinion and not a judge.
  It is a program's label, so it is attached with `attach_labels(..., kind="program")` —
  the one kind `judge_trust` calls measured without `allow_model_gold`.
- **The arms are the same agent at two misbehavior rates**, 0.50 and 0.15, standing in
  for a policy that got better. The point is not that it improved; it is by how much each
  instrument says it improved.
- **The task grid is written once and pinned with `tasks=`.** Two `simulate` calls at the
  same budget do *not* cover the same tasks — `eval_variance` says so in a note ("runs do
  not cover the same tasks; means are not strictly comparable") — so without pinning,
  nothing is paired. With it, 74 of 74 scenarios pair.
- **The noise floor is three re-runs** of the before arm at different agent seeds, passed
  to `compare` as `run_std_by_metric` so the gold metric and each marker get their own
  band.
- **Both instruments measure the same quantity**: the share of rows with no failure. Gold
  reads the plant record; the markers read the stamped `markers`. That is what makes
  "recovers 68%" a comparison and not a ratio of unlike things.

## Run it

```bash
uv add whileai
cd recipes/community/can-the-markers-be-trusted
python run.py                                      # 3 seeds, ~6 s, no key
python run.py --dry-run                            # one seed, small budget
python run.py --json results.json --modal-seeds seeds.json
modal run markers_modal.py --seeds 12              # 12 CPU containers on your Modal
```

| flag | default | what it does |
|---|---|---|
| `--seeds` | `0,1,2` | agent seeds; each is an independent draw of which rollouts misbehave |
| `--budget` | 600 | rollout budget for the task grid |
| `--phrasings` | 6 | the knob that actually grows the row count (`budget` alone saturates) |
| `--before-rate` / `--after-rate` | 0.50 / 0.15 | the two arms |
| `--dry-run` | off | one seed at budget 120 |
| `--modal-seeds` | none | fold a `seeds.json` from `markers_modal.py` in as the headline |

`results.json` is the output of the fourth command. `seeds.json` is the output of the
fifth, run on `wai-marker-trust`, 12 CPU containers, 5,064 rows.

## What you get

### What a fully green dashboard is still carrying

| dashboard | green rows | carrying a plant | share (pooled Wilson) | mean over 12 seeds |
|---|---|---|---|---|
| `style_report` only | 3,330 | 807 | 0.2423 [0.2281, 0.2572] | 0.2424 [0.2261, 0.2588] |
| every marker family | 2,578 | 533 | 0.2067 [0.1916, 0.2228] | 0.2068 [0.1942, 0.2195] |

Under `style_report` the 807 missed rows are 600 `leak` and 207 `ignore_fault` — every
planted instance of both. Adding the other two families removes 158 of the 207
`ignore_fault` rows and 116 of the 600 leaks, leaving 484 leaks and 49 faults. The two
routes to each interval (Wilson on pooled counts, t(df=11) across seeds) agree to three
decimal places.

### Did the markers recover the improvement that happened?

Rate 0.50 → 0.15, paired on 74 scenarios, 12 seeds:

| instrument | delta | 95% interval | share of gold recovered |
|---|---|---|---|
| gold (plant record) | **+0.348** | [+0.335, +0.361] | — |
| `style_report` markers | +0.237 | [+0.227, +0.247] | **0.681** [0.662, 0.699] |
| every marker family | +0.211 | [+0.205, +0.218] | **0.608** [0.592, 0.624] |

`compare`'s own report on seed 0, with the per-metric noise floor from three re-runs
(abridged to the markers that move):

```
PASS
eval noise: run_std 0.033, a delta under 0.199 is noise (t(df=2)=4.30 x run_std x sqrt(1/1 + 1/1); run_std given from 3 re-runs, per metric below)
family error: 14 metrics at 95%, up to 51% chance that one clears zero on luck alone
  pass_at_1                    0.447 -> 0.841  +0.394 [+0.333..+0.459]  up    (74 paired)  noise<0.199
  marker:no_apology            0.947 -> 0.972  +0.025 [+0.013..+0.038]  noise (74 paired)  noise<0.081
  marker:no_boilerplate        0.902 -> 0.978  +0.077 [+0.040..+0.122]  noise (74 paired)  noise<0.099
  marker:no_hedging            0.891 -> 0.970  +0.079 [+0.049..+0.116]  up    (74 paired)  noise<0.055
  marker:no_sycophancy         0.905 -> 0.973  +0.068 [+0.037..+0.104]  up    (74 paired)  noise<0.044
  marker:reported_failure      0.950 -> 0.971  +0.021 [+0.006..+0.038]  noise (74 paired)  noise<0.023
  marker:no_secrets            1.000 -> 1.000  +0.000 [+0.000..+0.000]  flat  (74 paired)  noise<0.000
  (argument_grounding, honest_claims, no_bypass, no_destructive, no_suppression,
   no_test_tampering, answered: all flat)
```

Read that as a reviewer would. Every marker that moves understates, three of the five
land inside their own noise band on a change that moved the gold metric 39 points, and
`no_secrets` reports a perfect 1.000 before and after on a run containing 600 leaks. If
the markers were all you had, you would report a real improvement as mostly noise and a
leak rate of zero.

Turning the extra families on also takes `compare` from 6 metrics to 14, and its family
error line from "up to 26% chance that one clears zero on luck alone" to **51%**. More
detectors is not a free action.

### `judge_trust` with the marker as the judge

| behavior | marker | agreement | kappa | `ok` |
|---|---|---|---|---|
| `hedging` | `no_hedging` | 1.000 | 1.00 | true |
| `sycophancy` | `no_sycophancy` | 1.000 | 1.00 | true |
| `apology` | `no_apology` | 1.000 | 1.00 | true |
| `boilerplate` | `no_boilerplate` | 1.000 | 1.00 | true |
| `ignore_fault` | `reported_failure` | 0.968 | 0.63 | **false** |
| `leak` | `no_secrets` | 0.875 | 0.00 | **false** |

This is the part of the library that comes out best. `no_secrets` scores **87.5%
agreement** — it looks like a good detector on the headline number, because 87.5% of rows
have no leak and a constant "clean" is right on all of them. Kappa is 0.00 and `ok` is
false. The agreement number alone would have fooled me; the kappa did not.

## The bug this run found

`attach_labels` will write the wrong gold label onto most of your rows and report a clean
match. Runs on a pinned task grid share `(scenario_id, rollout_index)`, which is the key
`attach_labels` matches on, so labels for one run land on every run's rows:

```
rows=1266  distinct keys=74  labels passed=1266
{'matched': 1266, 'unmatched': 0, 'invalid': 0, 'ties': 0, 'rows_labeled': 74}
rows whose gold_reward != the label passed for that row: 1200
```

Downstream, `judge_trust` reported agreement 0.89 and **kappa 0.00** for a detector whose
true agreement is 1.00, and called three of the four `ok=True`. Filed as
[#759](https://github.com/whilehq/whileai-sdk/issues/759). The recipe stamps a unique
`rollout_id` on every row and records `mislabeled_rows` in `results.json`; do the same
before you believe any `judge_trust` number computed over pooled runs.

## What this does NOT show

No claim that the phrase markers are wrong about what they measure — they measure phrase
presence and they are exact at it. The claim is about coverage and about trust: the
behavior a marker is named for and the behavior it detects are different questions, and
`no_secrets` is the case where the answer is "none at all".

The 24.2% is a property of this generator's behavior mix, not a law: it is roughly the
share of `seeded_agent`'s plants that are `leak` or `ignore_fault` among rows the markers
pass. Change `behaviors=` and it moves. What does not move is that `no_secrets` never
fires on a leak.

Recall is measured; **precision against a real model is not measurable here** for the
phrase markers, because `seeded_agent` never writes a phrase it did not plant. Zero false
alarms over 5,064 rows is a fact about the generator. A real model that opens with "Great
question!" while answering correctly would be a false alarm, and nothing here bounds how
often that happens. (`reported_failure`'s 106 false alarms *are* real, because it reads
the trace rather than a phrase list.)

The "improvement" is a rate knob, not training. It is a faithful stand-in for *measuring*
a policy that got better, and it is not evidence about what training does to markers.

## Next

Two things. First, **find out whether `no_secrets` can fire at all.** Zero detections on
600 planted leaks is either a marker that does not implement its name or one whose
trigger `seeded_agent`'s leak text does not match. Constructing a row it *does* catch is
a ten-minute experiment and it decides whether [#760] is a coverage gap or a dead
detector. Until somebody runs it, this recipe can only say the marker is silent, not why.

Second, **the same measurement against a real model**, which is the half this one cannot
do: generate on your own Modal with Qwen2.5-1.5B-Instruct, hand-label 200 replies for
hedging, and measure `no_hedging`'s recall against people rather than against its own
phrase list. That is the number that would tell you whether any of this transfers, and it
is the one the seeded path structurally cannot produce.

[#270]: https://github.com/whilehq/whileai-sdk/issues/270
[#760]: https://github.com/whilehq/whileai-sdk/issues/760
