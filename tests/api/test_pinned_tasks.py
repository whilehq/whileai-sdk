"""tasks=: a re-run keeps the first run's task set (#98).

A run draws its tasks from the grid by seed (and, above concurrency 1,
by completion order), so a second run shares only part of its tasks
with the first and ``compare_runs`` drops the rest. Pinning replays the
first run's prompts on their own scenario ids, under the same faults
and world state, whatever the policy, model or seed now is.
"""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from tests.helpers import simulate_offline

TOOLS = [
    {
        "name": "run_sql",
        "description": "Run SQL.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "returns": {"type": "object", "properties": {"rows": {"type": "array"}}},
    }
]
POLICY = (
    "You are a data agent.\n1. Never guess column names.\n2. Only report a number run_sql returned."
)
EDITED = POLICY + "\n3. Be concise."


def _agent(prompt):
    return {
        "final_text": "Total is $1,200.",
        "steps": [{"role": "assistant", "text": "Total is $1,200."}],
    }


def _run(policy, seed=7, **kw):
    kw.setdefault("budget", 60)
    return simulate_offline(
        _agent,
        tools=TOOLS,
        policy=policy,
        grade="conduct",
        mode="sft",
        repeats=2,
        per_round=40,
        concurrency=1,
        # top-level, not advanced={"seed": ...}: simulate_offline passes
        # seed=0 itself and an explicit knob overrides the advanced one,
        # so the "reseeded" run below used to be the same seed and only
        # the scheduler's timing noise made its task set differ.
        seed=seed,
        **kw,
    )


def _tasks(data):
    return {r["scenario_id"] for r in data.trajectories}


def test_pinning_keeps_every_task_across_a_policy_edit_and_a_seed():
    base = _run(POLICY)
    reseeded = _run(EDITED, seed=8)
    pinned = _run(EDITED, tasks=base, seed=8)
    assert len(base.trajectories) == len(pinned.trajectories) == 60
    # unpinned, a re-run draws its own tasks and shares only some
    assert len(_tasks(base) & _tasks(reseeded)) < len(_tasks(base))
    # pinned: same tasks, same prompts, same world, under the new policy
    assert _tasks(pinned) == _tasks(base)
    by_prompt = {r["prompt"]: r for r in base.trajectories}
    assert {r["prompt"] for r in pinned.trajectories} == set(by_prompt)
    for row in pinned.trajectories:
        first = by_prompt[row["prompt"]]
        assert row["scenario_dimensions"] == first["scenario_dimensions"]
        assert row["faults"] == first["faults"]
        assert row["world_state"] == first["world_state"]
        assert row["arm"] == first["arm"]
    report = pinned.search["pinned_tasks"]
    assert report["ran"] == report["prompts"] == len(by_prompt)
    assert report["tasks"] == len(_tasks(base)) and report["missing"] == []
    out = wai.compare_runs(list(base.trajectories), list(pinned.trajectories))
    assert out["n_paired"] == len(_tasks(base)) and out["n_only_a"] == out["n_only_b"] == 0
    assert out["note"] == ""


def test_pinned_run_stops_when_the_task_set_is_done():
    base = _run(POLICY, budget=30)
    pinned = _run(EDITED, tasks=base, budget=500)
    # every prompt gets its repeats and nothing else is drawn
    prompts = {r["prompt"] for r in base.trajectories}
    assert {r["prompt"] for r in pinned.trajectories} == prompts
    assert len(pinned.trajectories) == 2 * len(prompts)
    assert pinned.stopped_because == "tasks_done"
    assert pinned.search["pinned_tasks"]["missing"] == []


def test_tasks_accepts_rows_and_a_jsonl_path(tmp_path):
    base = _run(POLICY, budget=20)
    # export rows carry no scenario_dimensions: the fault plan alone
    # must land the row on its task (the realized-dims path)
    from_rows = _run(EDITED, tasks=base.rows(), budget=20)
    assert _tasks(from_rows) == _tasks(base)
    assert from_rows.search["pinned_tasks"]["missing"] == []
    assert "agent_errors" not in from_rows.degraded
    path = tmp_path / "base.jsonl"
    base.save(str(path))
    from_path = _run(EDITED, tasks=str(path), budget=8)
    assert len(from_path.trajectories) == 8
    assert _tasks(from_path) <= _tasks(base)
    assert from_path.stopped_because == "budget"


def test_tasks_rejects_empty_and_seed_mixing():
    with pytest.raises(ValueError, match="tasks="):
        _run(EDITED, tasks=[{"reward": 1}])
    with pytest.raises(ValueError, match="tasks="):
        _run(EDITED, tasks=[{"prompt": "hello there"}], seeds=["another ask"])
