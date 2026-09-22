"""#796: ``print(data)`` dumped every trajectory.

``len(repr(data))`` was 527,870 characters for a 200-row run, because both
row containers used a default repr: the dataclass one for ``SimulationData``
and ``object.__repr__`` for ``ScoredData``. Rule 5 in
``docs/reference/style.md`` says a result is an object that prints itself.
"""

from __future__ import annotations

import whileai as wai
from tests.helpers import POLICY, TOOLS


def _run():
    data = wai.simulate(
        wai.seeded_agent(TOOLS),
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        mode="rl",
        repeats=2,
        repeat_policy="fixed",
        budget=8,
    )
    return data, data.grade(judge=lambda row: {"reward": int(not row["seeded"])})


def test_a_run_prints_one_line_not_every_trajectory() -> None:
    data, scored = _run()

    text = repr(data)
    assert text == str(data)
    assert "\n" not in text
    # the failure this pins: a 200-row run was half a megabyte of nested dicts
    assert len(text) < 200, text
    assert "trajectories=[{" not in text
    assert text.startswith("SimulationData(8 rows, ")
    assert "situations" in text and "ungraded" in text and "mode='rl'" in text
    assert "budget=8" in text

    line = repr(scored)
    assert line == str(scored)
    assert "\n" not in line and len(line) < 200, line
    assert "object at 0x" not in line
    assert line.startswith("ScoredData(8 rows, 8 scored, judge='lambda_judge'")
    assert "mean_reward=" in line

    # the rows are one attribute away, and nothing else moved
    assert len(data.trajectories) == 8 and len(data.rows()) == 8
    assert len(scored.rows) == 8 and next(iter(scored)) is scored.rows[0]


def test_a_graded_run_says_so() -> None:
    data, _ = _run()
    assert "ungraded" in repr(data)
    for row in data.trajectories:
        row["reward"] = 1.0
        row["judge_status"] = "ok"
    assert "8 graded" in repr(data)
