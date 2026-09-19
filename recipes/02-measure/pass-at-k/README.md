# pass@1, pass^k and pass@k for one agent

Three numbers off the same graded groups, one job each.

| number | question it answers | who cares |
|---|---|---|
| `pass@1` | how often does one run pass? | production, and anyone measuring a behavior |
| `pass^k` | how often does the agent pass all k times? | reliability; a behavior that lands 60% of the time is not a contract |
| `pass@k - pass@1` | how much do repeats find that one run missed? | RL; this is what a grouped update (GRPO, DAPO) has to learn from |

The SDK computes all three from the per-ask pass-rate vector, using the
unbiased estimators (Chen et al. 2021 for pass@k, tau-bench for pass^k).
Below `repeats=4` the k-way numbers are withheld with a note rather than a
figure too noisy to act on.

## Run it

```bash
uv add whileai
python measure.py                  # 12 asks x 8 repeats with a scripted agent, offline
python measure.py graded.jsonl     # any graded row file (reward 0/1, grouped by task)
```

No key needed for the scripted run; it finishes in seconds. Output:

```
pass@1 0.47 [0.19..0.74] | pass^8 (pass_pow_k) 0.42 | pass@8 0.50 | headroom 0.03 (12 groups, k=8)
  never (p=0)          6
  sometimes (0<p<1)    1
  always (p=1)         5
- production sees pass@1 = 47%
- the agent is right every time on pass^8 = 42% of asks; the gap to pass@1 is inconsistency, not inability
- RL headroom 3%: almost nothing a grouped update can learn here; harder cells or a stricter judge before buying more rollouts
```

The bracket after pass@1 is a 95% bootstrap interval over asks (`ci95` on
the `PassAt`). Twelve asks is a small sample and the interval says so;
report it next to the point estimate.

The seed decides the draw: the run passes `reproducible=True`, so the same
`--seed` prints the same numbers at any `--concurrency`. Change the seed and
the numbers move; the shape is the point. This agent is mostly consistent
(right or wrong every time on eleven of twelve asks), so the honest verdict
is that repeats alone will not teach it much: the one mixed ask is the only
gradient, and the six it never passes need harder cells or demonstrations,
not more rollouts.

## Reading it

The histogram is the part the mean hides. An agent at pass@1 = 0.5 could be
right on half the asks every time (no RL headroom, a coverage problem) or
right half the time on every ask (all headroom, a consistency problem).
pass^k and pass@k tell those apart; the mean does not.

- **pass@1 is the headline.** The agent runs once in production. Report it,
  track it across prompt changes and model upgrades, and let the other two
  explain it.
- **pass^k is the contract.** Use it as the number you promise. If pass^k
  sits far below pass@1, the agent knows how but does not do it reliably;
  that is a consistency target, and mixed groups are the training data for it.
- **headroom sizes the RL run.** `recommend(mode="rl")` turns the mixed
  rate into a rollout budget; `select_for_rl` keeps the mixed groups whole.
  Headroom near zero means nothing to learn from these asks: harder cells or
  a stricter judge before more rollouts.

## Where else it shows up

`data.pass_at` on any simulation, `.pass_at` on the `ScoredData` from
`data.grade(judge=...)` and `evaluate(...)`, the same keys in
`group_signal(rows)`, and `pass_at` in the `.meta.json` sidecar of
`save(meta=True)`.

One caveat for LLM judges: pass@k inflates on judge false positives and
pass^k on false negatives. pass@1 is the least sensitive of the three, which
is the other reason it stays the headline.
