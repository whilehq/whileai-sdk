# Skills

A skill is the playbook a coding agent reads to do one kind of post-training
job end to end: get the data, build the frozen test, train, score every
behavior, report the run so a person can decide on while.ai/platform/runs.

Each folder holds `SKILL.md` (the playbook) and `check.py` (the same steps,
runnable offline with no key). CI runs every `check.py` on every pull request
and fails if a code block in a playbook drifts from the code that ran
(`tests/skills/test_skills.py`). What an agent copies out of a skill is code
that passed this morning.

| Skill | Use when | Trains with |
|---|---|---|
| `sft-from-traces/` | you have production traces and some good replies in them | SFT rows by rejection sampling |
| `dpo-pairs/` | the same prompt gets both good and bad replies | length-matched chosen/rejected pairs |
| `grpo-verifier/` | the reward can be a program (tool calls, tests, schemas) | GRPO on a verifiers environment |
| `character/` | how the model talks, from a constitution | trait judge checked against spec labels, pairs + SFT |
| `tool-call-efficiency/` | the agent solves tasks but spends too many calls | GRPO, reward = solved within budget |
| `watch/` | a version is serving and the Live tile needs a number a day | nothing; it scores yesterday's traffic |
| `strengthen-your-evals/` | an agent on a frontier model or your own weights needs evals that can fail and a number with an interval | nothing; it builds the frozen test, checks the judge, and reports every behavior |
| `manage-experiments/` | you are about to post a second version, a sweep, a replicate or a training run | nothing; it makes the page readable: the question first, per run Changed / Moved / Why / Learned / Reproduce, one chart, failed rows, points not fractions, `readback(tracked)` |
| `harness-search/` | the thing to improve is the harness (prompt, tools, turn cap, retry), not the weights | nothing; it is the Meta-Harness loop (Lee et al. 2026, arXiv:2603.28052) on the agent's own traffic: `--traces` yesterday's rows, the latest days held out, read `proposal.md`, write the next `candidates/<n>.py`, run, read the ledger, stop when the gate passes (holdout, held-out model, matched cost), report Changed / Moved / Why / Learned / Reproduce, score the next day |
| `pick-a-method/` | you have graded rows and must choose a method or its knobs, from all the SDK ships | nothing; six questions (grader, truncation, rollouts per prompt, band, on-policy, teacher) route to hosted sft/grpo/dpo, prime-rl grpo/max_rl/rae, OPSD, OPD, GroupwiseGrading, Async, FlashReinforce/SAO/BPCO or ReinforceAda, and name the skill that trains it |
| `whileai-simulations/` | you need more situations than the traces contain | nothing; it is the simulate-grade-select loop |
| `audit-your-judge/` | before Grade's scores are trusted or Select curates rows by them | nothing; it labels a sample blind, compares judges, ablates the rubric, and reverses the order |

## Every skill ends the same way

1. **Frozen test first.** Hold out by task, never by row. Name it
   (`Behavior(test_version="v1")`) and bump the name when the set changes.
2. **Noise floor.** Score the same agent twice; that spread is the floor a
   delta has to clear. Record it on the `Behavior`.
3. **Score every behavior**, not only the one you trained. The trained ones
   are the claim; the others are the check.
4. **Report.** `track(...)`, `tracked.behavior(...)`, `run = tracked.run(...)`,
   `run.score(...)`, `run.finish(...)`, `str(tracked.verdict())`. The person
   reads six tiles and presses Promote, or does not.

The rules behind these steps come from rlhfbook.com: "Evaluation" (held-out
sets, run-to-run spread, judge agreement) and "Over-Optimization" (what moves
on the behaviors you did not train). Writing a new skill: `BRIEF.md`.
