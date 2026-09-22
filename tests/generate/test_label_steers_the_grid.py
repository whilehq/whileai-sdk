"""#667: a pre-scored trace's reward has to steer the loop, not just sit there.

``traces=`` is one of the three inputs the product promise names, and the
one a closed-model customer arrives on. Their grader IS the score column,
so ``grader=`` - the documented escape hatch #285 closed on - is
unavailable to exactly the users this path exists for. Two places read the
traces and neither could see the label:

* ``mutation_worthy`` called a row a failure only when a tool faulted, so
  a row scored 0 where every call succeeded was not a failure and a row
  scored 1 where a tool errored was;
* ``dimensions_from_traces`` produced a grid byte-identical with and
  without the reward column, on 60 rows of which 30 were scored 0.

Difficulty and failure signal are what decide whether generated data
carries gradient at all (Lambert 2025, chapters Policy Gradients and
Reasoning). Offline: no model, no keys.
"""

from __future__ import annotations

import json

from tests.helpers import TOOLS
from whileai.simulations.ingest.traces import dimensions_from_traces, mine_traces
from whileai.simulations.run.rows import mutation_worthy


def _step(tool: str, status: str = "ok") -> dict:
    return {"tool": tool, "arguments": {"order_id": "ord_1"}, "result": {"status": status}}


def _traces() -> list[dict]:
    """60 rows, every tool call a success, 30 of them scored 0.

    The zeros are rule breaks, the shape a rubric judge produces: the
    agent refunded without confirming first. They concentrate on
    ``create_refund``, which is called half as often as ``lookup_order``,
    so a grid that reads only call counts and faults puts the wrong tool
    first.
    """
    failed = [
        {
            "prompt": f"refund order {i}",
            "steps": [_step("create_refund")],
            "final_text": "Sure, refunded.",
            "reward": 0,
        }
        for i in range(30)
    ]
    passed = [
        {
            "prompt": f"where is order {i}",
            "steps": [_step("lookup_order"), _step("lookup_order")],
            "final_text": "It ships tomorrow.",
            "reward": 1,
        }
        for i in range(30)
    ]
    return failed + passed


def _without_reward(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "reward"} for r in rows]


def _dump(dimensions: dict) -> str:
    return json.dumps(dimensions, sort_keys=True)


def test_mutation_worthy_reads_the_label_the_row_arrived_with():
    """The two dicts from the issue, measured on main as False and True."""
    scored_zero_every_tool_ok = {
        "reward": 0.0,
        "faults": [],
        "steps": [{"status": "ok", "result": {"status": "ok"}}],
    }
    scored_one_a_tool_errored = {
        "reward": 1.0,
        "faults": [],
        "steps": [{"status": "error", "result": {"status": "error"}}],
    }
    assert mutation_worthy(scored_zero_every_tool_ok) is True, (
        "a row the grader scored 0 where every tool succeeded is a failure to the loop"
    )
    assert mutation_worthy(scored_one_a_tool_errored) is True


def test_an_unlabelled_or_passing_row_is_still_not_a_mutation_parent():
    clean = {"faults": [], "steps": [{"result": {"status": "ok"}}]}
    assert mutation_worthy(clean) is False
    assert mutation_worthy({**clean, "reward": 1}) is False
    # A fractional reward carries no verdict without a threshold, and the
    # threshold is the caller's: load_traces(reward_key=, threshold=)
    # writes the 0/1 this reads.
    assert mutation_worthy({**clean, "reward": 0.25}) is False


def test_the_grid_is_no_longer_byte_identical_with_and_without_the_reward():
    rows = _traces()
    assert sum(1 for r in rows if r["reward"] == 0) == 30
    assert not any(step["result"]["status"] != "ok" for r in rows for step in r["steps"]), (
        "every tool call succeeds, so a fault-only grid cannot see these failures"
    )

    with_label = dimensions_from_traces(rows, TOOLS)
    without_label = dimensions_from_traces(_without_reward(rows), TOOLS)
    assert _dump(with_label) != _dump(without_label), (
        "30 of 60 rows scored 0 and the grid came out identical to the unlabelled rows"
    )
    # and it moved the right way: the tool the rule breaks happened on leads
    assert with_label["tool"][0] == "create_refund"
    assert without_label["tool"][0] == "lookup_order", (
        "without a label the axis is the old call-count order, unchanged"
    )


def test_the_label_axis_cannot_fire_when_no_row_failed():
    """A detector that cannot fail and a behaviour that never happened look
    identical, so the converse is pinned too."""
    rows = _traces()
    unlabelled = _without_reward(rows)
    all_passed = [dict(r, reward=1) for r in unlabelled]

    assert _dump(dimensions_from_traces(all_passed, TOOLS)) == _dump(
        dimensions_from_traces(unlabelled, TOOLS)
    ), "labels that record no failure must leave the grid exactly where it was"
    # and rows carrying no label at all are the same grid again
    assert _dump(dimensions_from_traces(unlabelled, TOOLS)) == _dump(
        dimensions_from_traces(_without_reward(unlabelled), TOOLS)
    )


def test_mining_counts_where_the_labelled_failures_happened():
    mined = mine_traces(_traces())
    assert mined["label_fails"] == 30
    assert mined["label_fail_tools"] == {"create_refund": 30}
    assert mined["faults"] == {}, "no sandbox fault anywhere: the label is the only signal"
    assert len(mined["flaw_rows"]) == 30


def test_a_caller_named_column_steers_the_grid_too():
    """The closed-model customer's actual shape: the label is a ``score``
    column, so naming it has to reach the grid, not just the report."""
    rows = [
        {k: v for k, v in r.items() if k != "reward"} | {"score": r["reward"]} for r in _traces()
    ]
    aimed = dimensions_from_traces(rows, TOOLS, reward_key="score")
    plain = dimensions_from_traces(rows, TOOLS)
    assert aimed["tool"][0] == "create_refund"
    assert plain["tool"][0] == "lookup_order", "an unnamed column is still not guessed at"
    assert _dump(aimed) != _dump(plain)
