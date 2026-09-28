"""Lean tools plus a review procedure in the prompt.

Written from the pilot traces: the stock agent re-read one file in 20-line
slices, opened changelogs, and ran out of steps before deciding. The suffix
says what to read, in what size, and when to stop.
"""

from deepagents import HarnessProfile

SUFFIX = """## How to review
1. Read each file the patch touches once, around the changed hunks, in large reads (limit 300).
2. grep for callers or tests of the changed functions only if the patch changes a signature or behavior they rely on.
3. Check: does the change do what the issue asks, for every case the issue names? Does it break an existing caller?
4. Decide after at most 8 tool calls. Do not read changelogs, docs or unrelated tests.
Reject a patch that only handles part of the issue, changes behavior the issue did not ask to change, or edits tests to pass."""

PROFILE = HarnessProfile(
    excluded_tools=frozenset(
        {"write_todos", "task", "write_file", "edit_file", "delete", "execute"}
    ),
    system_prompt_suffix=SUFFIX,
)
