---
name: manage-experiments
description: >
  Keep an agent's runs readable to the people who open the page. Use when a
  coding agent is about to post a second version, a sweep, a replicate or a
  new question to withwhile.com/platform: it says what the runs are for
  before the first one, names arms in words and puts numbers in the record,
  keeps seeds out of behavior names, posts points not fractions, carries the
  failed rows and a note with every score, reads the account back the way a
  teammate would, and archives what nobody will read again. No GPU, no key
  until the report.
metadata:
  version: "1.0.0"
---

# Manage experiments

The platform shows exactly what your code sends. On 2026-09-20 real coding
agents posted `dapo-lr5e-05-s17-30st` nineteen times, a behavior called
`looks_up_before_answering_seed1`, a score of `0.75`, a harness labelled by
its own hash, and six agents out of seven with no question. The pages were
correct and nobody could read them. Every step below is in `check.py`,
which runs offline in seconds; its setup defines `MODEL`, `TOOLS`,
`POLICY`, `PROMPT_POLICY`, `PROMPT_DATES`, `ASKS`, `NOISE`, `score_on`
(pass@1 in points with a 95% half-width and every row as an `Example`)
and `fake` (a recording transport). Names follow
[docs.withwhile.com/platform/naming](https://docs.withwhile.com/platform/naming).

## 1. Say the question before the first run

One agent per product, one question at a time. The block sits above the
run table; a reader knows what the dots are for before reading a number.
A second call replaces it, so when the question changes, post the new one
and let each run's note say which question it answered.

```python
tracked = track("checkout-support", model=MODEL, transport=fake)  # drop transport= for real
tracked.experiment(
    question="Does telling the prompt today's date cut wrong refunds on old orders?",
    hypothesis="Most wrong refunds are orders past 30 days; a dated rule removes them.",
    method="Same asks, same judge, two prompts on one model, three seeds each.",
    measure="refunds_when_eligible on the frozen test, points out of 100, 95% interval.",
    decide="Promote policy+dates when its interval clears policy and the noise floor.",
)
```

## 2. One behavior per thing the agent does, named by the test's content

A behavior is a verb phrase from the policy. A replicate is not a new
behavior and a seed is not part of its name. The test is named by a hash
of its asks so the name changes when the asks do (rlhfbook.com,
"Evaluation": a score is comparable only with the setup held constant).
Declare what you measured: judge agreement, noise floor, leaks.

```python
TEST = "t-" + hashlib.sha256("\n".join(ASKS).encode()).hexdigest()[:8]
tracked.behavior(
    Behavior(
        name="refunds_when_eligible",
        test_version=TEST,
        n=len(ASKS),
        judge=Judge(name="refund policy as a program", agreement=0.93, human_n=60),
        noise_floor=NOISE,
        contamination=0,
        reward_is_judge=False,
        rubric=POLICY,
        description="Refunds delivered orders within 30 days; escalates over $200; else denies.",
    )
)
```

## 3. Arms in words, numbers in the record, seeds as replicates

The version is what a teammate would call the arm. The page reads the
sweep's axes from `record.optimizer` and `record.data` on its own, so a
version that spells out the learning rate and the seed adds nothing a
person can read. Replicates share the version and differ only in
`Optimizer(seed=)`; the page averages them as one arm. The harness label
is `prompt@model`. Every score carries its worst rows and a note in
words, because a number nobody can open is a number nobody trusts.

```python
ARMS = {"policy": PROMPT_POLICY, "policy+dates": PROMPT_DATES}
for arm, prompt in ARMS.items():
    harness = Harness(label=f"{arm}@{MODEL}", model=MODEL, instructions=prompt, tools=TOOLS)
    for seed in (1, 2, 3):
        run = tracked.run(
            arm,
            method="eval",
            harness=harness,
            targets=["refunds_when_eligible"],
            record=RunRecord(
                data=Data(holdout=TEST, n_holdout=len(ASKS)),
                optimizer=Optimizer(seed=seed),
            ),
        )
        score, ci, examples = score_on(prompt, seed)
        failed = [e for e in examples if not e.ok]
        worst = (failed + [e for e in examples if e.ok])[:20]  # 20 rows, failures first
        run.score("refunds_when_eligible", score, ci=ci, n=len(ASKS), examples=worst)
        run.note(
            f"seed {seed}: {len(failed)} of {len(ASKS)} wrong; every miss is an old order refunded."
        )
        run.finish(say=False)
```

For a training run the same record carries `Data(train=, n_train=,
difficulty=)` and the optimizer's settings; `run.finish(record=...)` fills
it in after the fact, and `tracked.open(run_id)` does so from a later
session. Scores are points out of 100 on `run.score()`; `run.holdout()`
takes 0 to 1.

## 4. Read the account back as a teammate would

Before you tell the person the page is ready, read it the way they will.
Each line is one problem and the call that fixes it.

```python
SETTING = re.compile(r"(lr\d|\de-0\d|-s\d+\b|_s\d+$|_seed\d+|^h-[0-9a-f]{12}$|^v\d+$)")


def readback(tracked) -> list[str]:
    """What a teammate opening the page could not read. Empty means clean."""
    out = []
    if tracked.experiment() is None:
        out.append("no question posted: tracked.experiment(question=...)")
    for b in tracked.behaviors():
        if SETTING.search(b.name):
            out.append(f"behavior {b.name!r} names a seed or setting; put it in Optimizer(seed=)")
        if not (b.test_version or "").startswith("t-"):
            out.append(f"behavior {b.name!r}: test_version is not the asks' hash")
    for r in tracked.runs():
        v = r["version"]
        if SETTING.search(v):
            out.append(f"version {v!r} encodes settings; say the arm in words, numbers in record")
        if not ((r.get("record") or {}).get("data") or {}):
            out.append(f"run {v!r}: no data posted; RunRecord(data=Data(...))")
        if not r.get("notes"):
            out.append(f"run {v!r}: no note; run.note(what happened)")
        for e in r.get("evals") or []:
            if e["score"] <= 1:
                out.append(f"{v} {e['behavior']}: {e['score']} reads as a fraction; post points")
    return out


for problem in readback(tracked):
    print("fix:", problem)
```

Run it on an account you did not build too: `track("<agent id>")` binds
to an existing agent, and every fix is a `PATCH` you can send from here
(`tracked.open(run_id).note(...)`, `tracked.behavior(...)` again).

## 5. Report in sentences, then archive what nobody will read

The brief and the verdict are the same sentences the page shows, from
the same rows. Paste them in the pull request. A wrong launch, a dead
arm, a duplicate: archive it, never delete, so the record of what was
tried survives and the plot stops showing it.

```python
print(tracked.brief())
print(tracked.verdict())

dead = [r for r in tracked.runs() if r["version"] == "policy"][-1]
tracked.archive(dead["id"])  # a wrong launch or a dead arm: archive, never delete
```

## What the person sees

The question at the top. One dot per arm with its interval, replicates
folded in. A version list they can say out loud. Under any score, the
rows that failed and one line on why. A brief that says whether the
change is real and what to run next. The three checks from
`docs/platform/naming` still apply: a teammate recognises the agent id,
every behavior name is in the repo's own docs, versions sort the way the
team's releases sort.
