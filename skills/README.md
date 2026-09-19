# Skills

A skill is the playbook a coding agent reads to do one kind of post-training
job end to end: get the data, build the frozen test, train, score every
behavior, report the run so a person can decide on withwhile.com/platform/runs.

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
| `whileai-simulations/` | you need more situations than the traces contain | nothing; it is the simulate-grade-select loop |

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
