"""The reader's side of recipes/02-measure/eval-your-agent.

Two blocks on the page lean on names the recipe binds elsewhere. The
coverage block reads `OLD_TESTS`, `TOOLS` and `POLICY` from run.py and
`scored` from a graded run, so a small offline run (the template writer,
the careful bot, two repeats) stands in for the one `python run.py` makes.
The wiring block imports `bot`, the reader's own module with
`bot.answer(message) -> str` and its tools behind `bot._run_tool`; the
smallest bot with that shape is written next to run.py, where the reader's
would be.
"""

from pathlib import Path

import run as _run

OLD_TESTS = _run.OLD_TESTS
TOOLS = _run.TOOLS
POLICY = _run.POLICY
scored = _run.grade(_run.simulate("careful", k=2, seed=0, grid=0, limit=3), "careful")

Path("bot.py").write_text(
    '''"""The reader's bot: one tool, one reply, the shape the page wires up."""


def _run_tool(name, args):
    return {"status": "ok", "tool": name, "args": args}


def answer(message: str) -> str:
    _run_tool("lookup_order", {"order_id": "A1001"})
    return "Done: $129.00 refunded for order A1001."
''',
    encoding="utf-8",
)
