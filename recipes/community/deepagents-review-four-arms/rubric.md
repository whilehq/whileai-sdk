# Rubric: code review trajectories (the smithtune arm)

The rubric the first result used. It has the parts the smithtune docs ask for
(the task, what to keep, what to drop, an example of each) but skips the step
the smithtune skill centers: it was written without reading any traces and was
not calibrated on a trial, and it leaves call counts out on purpose. The
council kept 258 of 260 with it. `rubric_codesigned.md` is the rubric written
the recommended way, used by the rerun in PREREGISTRATION.md.

## Task

The agent reviews a proposed patch for a GitHub issue. It can read the
repository at the patch's base commit with file tools (ls, read_file, glob,
grep). It must end with `VERDICT: approve` or `VERDICT: reject`. Every
trajectory here already has the verdict the hidden tests agree with.

## Keep

- The agent reads the code the patch touches before deciding.
- The final message explains the decision with reference to the issue and the
  code it read, then gives the VERDICT line.
- Tool calls are well-formed and their results are used.

Example to keep: the agent reads the changed function and its callers, notes
that the patch handles the case the issue reports and does not change other
behavior, and ends `VERDICT: approve`.

## Drop

- The verdict is given without reading any code, or the explanation
  contradicts the verdict.
- The trajectory is dominated by tool errors, repeated identical calls, or
  reading files unrelated to the patch.
- The final message is empty apart from the VERDICT line.

Example to drop: the agent lists directories for ten turns, never opens the
patched file, and ends `VERDICT: reject` with no reason.
