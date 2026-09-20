---
title: "Keep the runs readable"
sidebarTitle: "Experiments"
description: "Before a coding agent posts a second version, a sweep or a replicate, it follows the manage-experiments skill: question first, arms in words, seeds in the record, points not fractions, failed rows and a note per score, then reads the account back as a teammate would."
---

The platform draws what your code sends, and a coding agent writing `track(...)` at two in the morning sends what is easiest to type. On one day in September 2026 that was `dapo-lr5e-05-s17-30st` nineteen times, a behavior named `looks_up_before_answering_seed1`, a score of `0.75` on a page that counts points out of 100, a harness labelled by its own hash, and six agents out of seven with no question posted. Every page was correct. Nobody could read them.

The fix is a playbook the agent follows before the second run, tested in CI like every skill: [skills/manage-experiments](https://github.com/whilehq/whileai-sdk/blob/main/skills/manage-experiments/SKILL.md). `whileai init` installs it under `.claude/skills/` and the `AGENTS.md` block names it.

| Before you post | Do this | Call |
|---|---|---|
| a second version | say what the runs are for, at the top of the page | `tracked.experiment(question=, hypothesis=, method=, measure=, decide=)` |
| a sweep | name each arm in words; the page reads the axes from the record | `tracked.run("policy+dates", record=RunRecord(optimizer=Optimizer(lr=5e-5)))` |
| a replicate | same version, seed in the record, never in a behavior name | `Optimizer(seed=2)` |
| a score | points out of 100, with `ci` and `n`, and the rows that failed | `run.score(b, 73.1, ci=4.2, n=160, examples=worst)` |
| a finished run | one line on what happened | `run.note("seed 2: 9 of 160 wrong; all old orders refunded")` |
| the report | read the account the way a teammate will | `readback(tracked)` from the skill; empty means clean |
| a dead arm | archive, never delete | `tracked.archive(run_id)` |

`readback(tracked)` prints one line per problem with the call that fixes it, and every fix is a `PATCH` you can send from any session: `tracked.open(run_id).note(...)`, `tracked.behavior(...)` again with the right name. Then `print(tracked.brief())` and `print(tracked.verdict())` are the sentences the page shows, from the same rows; paste them in the pull request.

Names themselves follow [Naming](/platform/naming): the agent after the product, behaviors as the policy phrases them, versions as the team ships them, the test by its content.
