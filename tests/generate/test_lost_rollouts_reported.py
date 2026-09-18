"""A rollout that never becomes a row must be COUNTED, not silently dropped (#303)."""

from __future__ import annotations

import whileai.simulations as wai

TOOLS = [
    {
        "type": "function",
        "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}},
    }
]
TASKS = [{"prompt": f"task {i} {'FAIL' if i % 2 else 'ok'}"} for i in range(6)]


def _raising_agent(messages, tools=None, **_):
    # the callable contract hands the conversation in more than one shape;
    # the marker is in the prompt whichever shape arrives
    if "FAIL" in str(messages):
        raise RuntimeError("endpoint still booting")
    return {"steps": [], "final_text": "done"}


def _silent_agent(messages, tools=None, **_):
    # comes back without an exception and without a reply: the shape a
    # trained adapter fails in, and the one #303 hit with degraded=[]
    if "FAIL" in str(messages):
        return {"steps": [], "final_text": ""}
    return {"steps": [], "final_text": "done"}


def _leaky_agent(messages, tools=None, **_):
    if "FAIL" in str(messages):
        return {"steps": [], "final_text": "<tool_call>{}</tool_call>"}
    return {"steps": [], "final_text": "done"}


def _run(agent, tasks=TASKS, **kw):
    return wai.simulate(
        agent=agent,
        tools=TOOLS,
        system_prompt="p",
        execute=lambda t, a: {"ok": True},
        tasks=tasks,
        fault_rate=0.0,
        simulator=False,  # template writer: no hosted model, no key
        **kw,
    )


def test_agent_errors_are_counted_warned_and_degrade_the_run():
    """Half the pinned tasks hit a raising agent. Before this, the run
    reported a clean pass rate over the survivors and nothing said the rest
    died: one eval lost its first 64 of 130 pinned tasks to a cold endpoint."""
    data = _run(_raising_agent)
    rep = data.report()
    assert len(data.trajectories) == 3
    assert rep["rollouts_requested"] == 6 and rep["rollouts_completed"] == 3
    assert rep["rollouts_lost"] == 3, rep
    assert rep["rollouts_lost_by"] == {"agent_error": 3, "empty_reply": 0, "tool_markup": 0}
    assert rep["rollouts_over_cap"] == 0
    # one source for the agent error count: data.search, as on the old path
    assert "agent_errors" not in rep
    assert data.search["agent_errors"] >= 3
    assert "rollouts_lost" in data.degraded and "agent_errors" in data.degraded
    note = next(w for w in data.warnings if "never became rows" in w)
    assert "3 rollout(s) of 6 asked for" in note
    assert "3 agent error" in note
    assert "endpoint still booting" in note
    assert "warm a scale-to-zero endpoint" in note and "timeout=" in note


def test_an_empty_reply_is_lost_without_being_an_agent_error():
    """The #303 shape: rows missing, degraded=[] on the old code, because the
    agent answered with nothing rather than raising."""
    data = _run(_silent_agent)
    rep = data.report()
    assert len(data.trajectories) == 3
    assert rep["rollouts_lost"] == 3, rep
    assert rep["rollouts_lost_by"]["empty_reply"] == 3
    assert rep["rollouts_lost_by"]["agent_error"] == 0
    assert "agent_errors" not in data.search
    assert "rollouts_lost" in data.degraded
    note = next(w for w in data.warnings if "never became rows" in w)
    assert "3 empty reply" in note and "agent_max_tokens=" in note
    assert "agent errors" not in note


def test_leaked_tool_markup_is_lost_under_its_own_name():
    data = _run(_leaky_agent)
    rep = data.report()
    assert rep["rollouts_lost"] == 3 and rep["rollouts_lost_by"]["tool_markup"] == 3
    note = next(w for w in data.warnings if "never became rows" in w)
    assert "3 tool markup" in note and "tool-call format" in note


def test_a_clean_run_reports_zero_lost():
    data = _run(lambda m, tools=None, **_: {"steps": [], "final_text": "done"}, tasks=TASKS[:2])
    rep = data.report()
    assert rep["rollouts_lost"] == 0 and rep["rollouts_lost_by"]["agent_error"] == 0
    assert rep["rollouts_requested"] == rep["rollouts_completed"] == 2
    assert rep["stopped_because"] == "tasks_done"
    assert "rollouts_lost" not in data.degraded
    assert not any("never became rows" in w for w in data.warnings)


def test_a_budget_bound_run_is_neither_lost_nor_over_cap():
    """Six tasks and room for two rows: the engine launches only what the
    budget has room for, so nothing is lost and nothing lands over the cap.
    (Over-cap rows need a re-roll or a verify to land after the last
    budgeted row; that race is counted, not reachable offline on demand.)"""
    data = _run(
        lambda m, tools=None, **_: {"steps": [], "final_text": "done"},
        budget=2,
        concurrency=6,
    )
    rep = data.report()
    assert rep["rollouts_completed"] == 2 and rep["rollouts_requested"] == 6
    assert rep["rollouts_lost"] == 0 and rep["rollouts_over_cap"] == 0
    assert "rollouts_lost" not in data.degraded
