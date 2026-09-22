---
title: "Say why it moved"
sidebarTitle: "Experiments"
description: "Before a coding agent posts a second version, a sweep or a training run, it follows the manage-experiments skill: the question first, then per run five lines (Changed, Moved, Why, Learned, Reproduce), one chart, the rows that failed, and readback(tracked) to read the page the way the person will."
---

The coding agent knows the repo. The person opening the page does not have its context and reads for two reasons: to learn what a paper or an idea does when tried, and to find a behavior and fix it fast at work. Both want one look: what changed, did it move, why, what it taught, and can I run it again. On one day in September 2026 real agents posted `dapo-lr5e-05-s17-30st` nineteen times, no note on any run, no chart, a `0.75` on a page that counts points, and no question on six agents of seven. Every page was correct and said nothing.

The fix is a playbook the agent follows before the second run, tested in CI like every skill: [skills/manage-experiments](https://github.com/whilehq/whileai-sdk/blob/main/skills/manage-experiments/SKILL.md). `wai init` installs it under `.claude/skills/` and the `AGENTS.md` block names it.

| The person asks | The agent posts | Call |
|---|---|---|
| what are these runs for | the question, at the top of the page | `tracked.experiment(question=, hypothesis=, method=, measure=, decide=)` |
| what changed | one line: the prompt line, the data mix, the setting | `run.note("Changed: ...")` |
| did it move | before and after in points, with the interval | `"Moved: 75 to 96.2 points (±5.2) on 52 asks"` |
| why | the asks that flipped, and whether the reward and the held-out moved together | `"Why: 11 asks now pass, every one an old order; 0 broke"` |
| what did it teach me | the hypothesis closed in one sentence, with the paper when the run tried one | `"Learned: 800 traces taught one rule; the held-out checkpoints are the proof (rlhfbook.com, Over-Optimization)"` |
| can I run it again | the command, the seed, the pins | `"Reproduce: uv run python train.py --seed 17"` + `RunRecord(data=, optimizer=, provenance=)` |
| show me | one chart per run: bars with intervals for a harness, reward and held-out on one chart for training | `tracked.figure(name, {"data": [...], "layout": {...}}, run=run)` |
| what failed | up to 20 rows under each score, failures first | `run.score(b, 96.2, ci=5.2, n=52, examples=worst)` |
| is the page ready | the account read the way a teammate will, one line per problem with the fixing call | `readback(tracked)` from the skill; empty means clean |

The page counts points out of 100 and the client rescales nothing: a rate out of 1 goes up as `run.score(b, 0.71, ci=0.04, n=52, fraction=True)`, which posts 71 (±4), while a bare `0.71` posts as 0.71 points with a one-time warning that names the fix. A reward that climbs while the held-out line stays flat is the judge being gamed (rlhfbook.com, "Over-Optimization"); the note says so rather than letting the person find out. `print(tracked.brief())` and `print(tracked.verdict())` are the sentences the page shows, from the same rows; paste them in the pull request. Names follow [Naming](/platform/naming): the agent after the product, behaviors as the policy phrases them, versions as the team ships them, the test by its content.
