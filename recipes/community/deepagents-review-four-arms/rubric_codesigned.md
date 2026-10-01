# Rubric: code review trajectories (co-designed)

Written the way the smithtune skill's happy path asks: read 20 pulled
trajectories (4 short, 5 medium, 4 long, 5 very long, 2 with unusual tools),
write the criteria from what they show, then check them on a 20-trajectory
trial before the full run. An agent did the reading and drafting; a person
signed off on the criteria. Every trajectory here
already has the verdict the hidden tests agree with, so the question is only
whether the model should learn to review *this way*. SFT copies every call.

## Task

The agent reviews a proposed patch for a GitHub issue. It can read the
repository at the patch's base commit with file tools (ls, read_file, glob,
grep). It must end with `VERDICT: approve` or `VERDICT: reject`.

## What the 20 showed

- The best reviews were short. Three to five calls was enough when the patch
  is small: read the patched function, check the one thing the issue reports,
  decide. Several patches only add a stray file (`t.py`, `out.nex`) and change
  no code; one read of the target file settles those.
- The long ones were long for bad reasons: the same grep run twice with the
  same arguments, the same file range read twice, ten searches for a config
  class that has nothing to do with the patch, or many 100-line reads through
  a 9,000-line file after the answer was already clear.
- A common first call is `read_file` on a guessed absolute path like
  `/home/user/repo/...`, which fails, then a glob to find the real path. One
  such recovery is fine; the model should not learn to make it twice.
- The final messages were nearly all good: they name the defect or the reason
  the fix is correct, with the line or case that shows it.

## Keep

- The agent reads the code the patch touches before deciding, and the final
  message ties the verdict to that code and the issue.
- Every call moves the review forward: each read or search answers a question
  the verdict depends on (the patched code, its callers, the tests that cover
  it).
- The review stops once the verdict is clear. Most good reviews finish in 10
  model calls or fewer; past 12, keep it only if each call was needed.

Example to keep: three calls. The agent greps for the function the patch
changes, reads 40 lines around it, sees that the patch reads a property before
mutating the cache, explains why that removes the KeyError, and ends
`VERDICT: approve`.

## Drop

- A tool call repeated with identical arguments, or the same file range read
  twice.
- Searches or reads aimed at something the verdict does not depend on (for
  example hunting the config library's internals when the bug is a regex).
- More than one failed call from a guessed path or a wrong argument.
- Delegating to a subagent or a non-file tool, or calling a tool named after
  the verdict.
- The verdict given without reading the patched code, or an explanation that
  contradicts the verdict.

Example to drop: fourteen calls. The agent reads the plugin and its tests,
then runs `grep "def as_str"` twice and `grep "class Config"` twice across the
package, globs for config files that do not exist, and only then writes a
correct reject. The verdict is right; the path to it is not one to teach.
