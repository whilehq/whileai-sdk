# Report a run so a person can decide

**What you learn**: the typed objects the platform tracks (a tracked agent
with its harness, behaviors, runs, live traffic), why a harness is versioned
by its fingerprint, and why a version is scored on every behavior.

**Needs**: `WHILEAI_API_KEY` for the real thing; nothing for the smoke run.

**Takes**: 10 seconds.

```bash
python run.py --offline       # free: no key, no network, prints the calls it would make
python run.py                 # posts one full loop for agent refund-bot, prints the verdict
python run.py --agent my-bot  # your own name
```

The script registers an agent (model + harness + the frontier model you pay
for today), declares five behaviors with their own held-out tests, opens
one run per version (base, v1..v4) with a training curve and a score on
every behavior, marks v3 as served, posts two weeks of traffic, and reads
the verdict back. Open withwhile.com/platform/runs afterwards: that is the screen
the person decides on.

Three rules a reported score has to keep [1, 2]: the held-out
test for a behavior does not change under you (bump `test_version` when it
does; the platform does not yet refuse a score on another test version, so
this one is on you), a score without an interval is not a result (`ci` is
the half-width of the 95% interval, and `verdict()` says "not a result"
without one), and a delta inside the eval's own re-run band is not a result
either (`noise_floor` is the spread you saw scoring the same model twice;
`verdict()` reads it). The `--offline` verdict is computed from the score
table with the same difference-interval rule the platform uses.

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
2. Miller, E. Adding Error Bars to Evals. arXiv:2411.00640, 2024.
