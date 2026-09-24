---
name: audit-your-judge
description: >
  Check whether a judge can be trusted before Grade hands its scores to
  Select or Train: label a sample blind, compare judges against those
  labels, and read agreement, kappa and the false-pass rate against
  floors. Use before trusting any judge, rubric or grader score, before
  choosing a judge model, before calling compare_judges/judge_trust/
  attach_labels, and before Select curates rows by a judge's score.
  Triggers on judge, rubric, LLM-as-judge, grader, judge_trust,
  compare_judges, judge model choice, kappa, false-pass rate, judge
  agreement.
metadata:
  version: "1.0.0"
---

# Auditing your judge

A biased judge does not look biased. It looks like Train failed. The
repeated finding across 40 use cases was a training method blamed for what
the scoring did - the run looked broken because the judge grading it was
generous, not because the method was. Audit the judge before Grade's
numbers are trusted and before Select curates a single row by them.

Everything below is in `check.py`, offline in about a second. Its setup
defines the names the blocks use: `ROWS` (60 fixed transcripts whose true
correctness is known by construction), three scripted judges over those
rows (`always_pass`, `generous`, `accurate`), `generous_with_clause` and
`accurate_with_clause` (the same two plus one rubric clause), `prefers` (a
position-biased pairwise pick), `BLIND_LABELS` (a codebook read that never
sees which judge said what), and `fake`, a recording transport that
answers like the platform API. `table`, `labeled` and `data` are results
the blocks below build and later ones reuse.

Load `strengthen-your-evals` before sizing the sample, `rlhf-post-training`
before any SFT or RL work that consumes a judge's scores.

## What the measured findings say

A judge is a reward model under a different name, and a reward model is
only as good as its measured accuracy against people (Lambert 2025,
chapter Reward Modeling). Six things came out of measuring six judges the
same way, on the same rows, against a blind label:

1. **Judge choice moves the score more than the model under test.** Same 60
   answers, same rubric: haiku-4.5 agreed 67% with the reference,
   sonnet-4.5 70%, opus-4.5 78% - and the best of the three still passed
   31% of reference failures. A separate 2x2 (2 judges x 2 policy models,
   n=80/cell) put the judge's own effect at ~0.62 against a same-family
   bonus of +0.010 - noise. That is calibration, not self-preference.
2. **A higher mean reads as a softer judge, not a better one.** sonnet-4.5's
   mean (0.700) beat opus-4.5's (0.617) while agreeing *less* with the
   reference. A number alone cannot tell generous from accurate.
3. **The same clause lands differently on different judges.** One
   formatting-only clause ("score 0 unless every date is ISO-8601"): haiku
   fell 0.633 to 0.212 (8/60 unparseable), sonnet did not move. A 1-to-9
   clause ladder was flat at n=60 - length was never the driver. One bad
   clause on a weak judge is catastrophic; nine good ones can cost nothing.
4. **A clause naming evidence the payload lacks hurts the better judge
   worst.** A clause naming fields the payload never carried scored
   0.20/0.00/0.20 (haiku/sonnet/opus) - anti-correlated with capability,
   because the strongest judge correctly noticed the evidence was missing.
   Rewriting the sentence gave 0.90/0.50/0.40; the biggest gain went to the
   smallest judge.
5. **Presentation order changes the ranking.** Reversing order on the same
   answers, same judge, flipped 28.4% of pairwise rankings at k=4 and 20.5%
   at k=8. The top pick mostly survives (0.644 against 0.292 by chance); a
   full ranking is roughly a fifth noise.
6. **The hosted default lost to a judge that always says pass.** 0.483
   agreement against 0.517 for unconditional pass, kappa **-0.06**, 27 of
   29 rule failures passed. A default is not a safe default.

**Agreement alone hides this.** A judge that says pass without reading
anything agrees at whatever the base pass rate already is. Kappa is
agreement with that chance rate subtracted out (Cohen 1960):

    kappa = (observed_agreement - chance_agreement) / (1 - chance_agreement)

Finding 6 is the clean case: 0.483 agreement sounds close to a coin flip,
but a kappa of -0.06 means the judge systematically passes what the
reference failed, not misses at random. Report both; agreement alone would
have shipped that judge.

## Traps in the SDK worth naming before you touch it

- `Judge(agreement=, human_n=)` are declared fields - nothing computes
  them, so typing two integers gives a clean card with no measurement
  behind it. Use `compare_judges` and `attach_labels`.
- `judge_trust` scores the `reward` already on the row against `gold`, not
  the `judge=` you pass it - the obvious call can report agreement 1.0 for
  a judge that never ran.
- `check_spread` scores variance only. It rated a position-biased judge
  0.211 above a discriminating program grader at 0.099; a random
  permutation would maximize it. Spread is not agreement or accuracy.
- `attach_labels(kind=...)` records what a label actually is - `kind="human"`
  for a model's label misleads everyone who reads `gold_kind` later.

## Sample and label blind

Pull ~60 rows across the judge's own score range, not the easy middle, so
disagreement at the edges has somewhere to show up. Strip whatever gives
the verdict away (`rule_reason`, `gold_*`) before anyone labels. Write the
codebook - what counts as a pass, what fails - before labelling starts, not
while it is happening, or the codebook drifts to match what is on screen.
`check.py`'s `BLIND_LABELS` applies that codebook (the eligibility rule
read off the tool calls) with no judge's verdict in view:

```python
labeled, label_report = wai.attach_labels(ROWS, BLIND_LABELS, annotator="reviewer", kind="human")
table = compare_judges(
    labeled, {"always pass": always_pass, "generous": generous, "accurate": accurate}
)
print(table)

assert label_report["rows_labeled"] == 60, label_report
assert table.n_rows == 60
always_pass_score, generous_score, accurate_score = (
    table["always pass"],
    table["generous"],
    table["accurate"],
)

assert always_pass_score.agreement == 0.4, always_pass_score.agreement
assert always_pass_score.kappa is not None and abs(always_pass_score.kappa) < 1e-9, (
    always_pass_score.kappa  # "just says pass": kappa is 0, exactly
)
assert always_pass_score.leak == 1.0, always_pass_score.leak
assert generous_score.agreement == 0.7, generous_score.agreement  # looks like a real judge...
assert generous_score.kappa is not None and generous_score.kappa < 0.5, generous_score.kappa
assert generous_score.leak == 0.5, generous_score.leak  # ...but passes half of what really failed
```

`always_pass` here reads nothing and reproduces finding 6's shape exactly:
kappa lands at 0, not near it - a judge that only ever says pass agrees at
the base rate and that is what "0 = guessing" means. `generous` reads only
the reply's tone (a dollar figure, the word "refunded") and reproduces
finding 1-2's shape: 70% agreement, a number that would pass a casual
glance, while a leak rate of 0.50 says it waved through half of what the
codebook actually failed. If a training run reads `reward` off either
judge, it is training the leak, not the behavior.

## Floors to clear

**Wilson lower bound of agreement >= 0.8, kappa >= 0.6.** Zheng et al. 2023
(MT-Bench, arXiv:2306.05685) put human-human agreement at 81% and GPT-4 at
85% against humans, so under 80% a judge agrees with people less than
people agree with each other; that is where `MIN_AGREEMENT` comes from, and
it is what the report checks, not a round number. Landis and Koch (1977)
call 0.61-0.80 "substantial"; `MIN_KAPPA` sits at the bottom of that band,
and a 2026 sweep of 21 judges (arXiv:2606.19544) measured kappa 0.376 to
0.511 against human preference labels on MT-Bench, so the floor asks for
more than most judges in that sweep cleared. Only `accurate` (the judge
that reads the tool calls, not the reply's tone) gets past both here:

```python
assert accurate_score.ok, accurate_score
assert accurate_score.agreement is not None and accurate_score.agreement >= 0.8
assert accurate_score.kappa is not None and accurate_score.kappa >= 0.6
assert table.best is not None and table.best.name == "accurate", table.best
```

## Ablate the rubric

Add or remove one clause and re-run; a large swing on the mean reward says
the clause is carrying the score, not the behavior it was meant to check
(findings 3 and 4). A judge that only pattern-matches the reply's surface
can be pushed to zero by a clause it cannot verify, while a judge that
checks the underlying trajectory does not move:

```python
def mean_reward(judge_score) -> float:
    """Mean judge reward across a JudgeScore's own graded rows."""
    graded = [r["reward"] for r in judge_score.rows if r.get("reward") is not None]
    return sum(graded) / len(graded)


baseline = compare_judges(labeled, {"generous": generous, "accurate": accurate})
with_clause = compare_judges(
    labeled, {"generous": generous_with_clause, "accurate": accurate_with_clause}
)
for name in ("generous", "accurate"):
    delta = mean_reward(with_clause[name]) - mean_reward(baseline[name])
    print(f"clause delta, {name}: {delta:+.3f}")

generous_delta = mean_reward(with_clause["generous"]) - mean_reward(baseline["generous"])
accurate_delta = mean_reward(with_clause["accurate"]) - mean_reward(baseline["accurate"])
assert generous_delta < -0.5, generous_delta  # the weak judge collapses
assert accurate_delta == 0.0, accurate_delta  # the judge that reads substance does not move
```

## Reverse the presentation order

Grade the same pairs with the order swapped and count how many pairwise
picks flip. The noise is not spread evenly - it concentrates exactly where
the underlying signal is a genuine tie, which is why finding 5's 28.4%
sits well under 100%: most pairs in that set were not ties. `prefers` here
only breaks ties by position, so the mechanism shows at its purest: every
real tie flips, and a pair with a real difference never does.

```python
def flip_rate(pairs: list[tuple[dict, dict]]) -> float:
    flips = 0
    for a, b in pairs:
        winner_forward = a if prefers(a, b) == "a" else b
        winner_backward = b if prefers(b, a) == "a" else a
        if winner_forward is not winner_backward:
            flips += 1
    return flips / len(pairs)


ok_rows = [r for r in ROWS if r["rollout_id"].startswith("ok-")]
refuse_rows = [r for r in ROWS if r["rollout_id"].startswith("refuse-")]
tied_pairs = list(zip(ok_rows[::2], ok_rows[1::2]))  # both score 1 under `generous`: a real tie
decisive_pairs = list(zip(ok_rows, refuse_rows))  # one scores 1, one scores 0: real signal

tied_flips = flip_rate(tied_pairs)
decisive_flips = flip_rate(decisive_pairs)
print(f"order reversal: {tied_flips:.0%} of ties flip, {decisive_flips:.0%} of decisive pairs flip")

assert tied_flips == 1.0, tied_flips  # every tie is decided by position alone
assert decisive_flips == 0.0, decisive_flips  # real disagreement does not move
```

If your real judge's flip rate on ties runs anywhere near 28.4%, do not
feed it a ranking - preference pairs, GAR, anything relative - until you
know which pairs in your own data are ties.

## Only then grade for real, and only then train

`check.py`'s setup writes a small frozen held-out test the way
`strengthen-your-evals` does (`wai.simulate(..., simulator=False,
seeds=[...], repeats=k)`) for two versions of a scripted returns agent, one
that refunds anything and one that checks eligibility first. The same
`accurate` judge that cleared the floors above - not a fresh one - grades
both:

```python
JUDGE = Judge(
    name="accurate (tool-call eligibility check)",
    agreement=table["accurate"].agreement,
    human_n=table["accurate"].n,
)
scored = {v: wai.evaluate(d.rows(), accurate, tools=LIVE_TOOLS) for v, d in data.items()}
```

Every skill here ends the same way (`skills/BRIEF.md`): name the frozen
test, score every behavior the agent has - not only the one you audited
the judge for - and post the run so a person can read it, with the judge's
own measured agreement attached to the number it produced:

```python
tracked = track(
    "returns-agent",
    model="scripted-agent",
    harness=Harness(label="v1", instructions=LIVE_POLICY, tools=LIVE_TOOLS),
    transport=fake,  # drop this line to talk to the platform
)
for name, (_pts, _ci, n) in behaviors(scored["v1"].rows).items():
    tracked.behavior(
        Behavior(
            name=name,
            test_version=TEST_VERSION,
            n=n,
            judge=JUDGE,
            noise_floor=NOISE,
            contamination=0,
            reward_is_judge=False,
        )
    )
for version, s in scored.items():
    run = tracked.run(version, method="eval", targets=["refund_policy"], harness=version)
    for name, (pts, ci, n) in behaviors(s.rows).items():
        run.score(name, pts, ci=ci, n=n, test_version=TEST_VERSION)
    run.finish(
        record=RunRecord(
            data=Data(holdout=TEST_VERSION, n_holdout=len(asks)),
            eval=EvalSetup(metric="pass@1", k=K, run_std=NOISE, run_std_runs=2, reader=JUDGE.name),
        )
    )
tracked.promote("v1")
print(tracked.verdict())
```

A judge that fails the floors above never reaches this block. It is not a
footnote next to the training result - a training run graded by an
unaudited judge has no result to report yet, per "Over-Optimization"
(Lambert 2025): a policy pushed against a judge's habits, not the
behavior, will find every hole this file just measured.

## Reading the result

| signal | verdict |
|---|---|
| agreement >= 0.8, kappa >= 0.6, false-pass rate low | judge is usable |
| a high mean, unmeasured against a blind label | unknown - a soft judge produces exactly this |
| the mean moves a lot when one clause is removed | that clause, not the behavior, drove the score |
| the score collapses on the strongest judge specifically | check the clause for evidence the payload lacks |
| over 20% of pairwise ties flip on reversed order | consume the top pick only, never the full ranking |
| a training result reads as "the method failed" and the judge was never checked | audit the judge before believing that |

Also worth doing once a judge clears the floors: grade with a different
model family than the one you are training (the self-preference bonus
measured above was a genuinely small +0.010, but calibration swamped it at
~0.62 - the real reason for a different family is an independent read, not
a correlated one), and split the rubric so a program checks whatever a
program can (fields present, ids retained, length) and the judge grades
only what a program cannot. A constitution raised register +0.518 while
reply length collapsed 5.5x and identifier retention fell -0.193; a judge
scoring the whole rubric called that a win, because "sounds right" is easy
to satisfy by cutting the parts that are hard to get right.
