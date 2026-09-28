"""Only the tools a reviewer uses: no todo list, no subagent, no write tools.

A read-only review has no use for planning state or delegation, and every tool
schema in the prompt is a choice the model can make instead of reading.
"""

from deepagents import HarnessProfile

PROFILE = HarnessProfile(
    excluded_tools=frozenset(
        {"write_todos", "task", "write_file", "edit_file", "delete", "execute"}
    ),
)
