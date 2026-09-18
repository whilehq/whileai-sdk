"""A declared tool the world never answers is named, not scored (#287).

A tool in the agent's schema with no branch in ``execute=`` fails exactly
like a world fault: the agent reports the miss, an honesty rubric rewards
the row, and the behaviour behind the tool never happens. One lane ran
612 calls with 4 successes through 978 rows before anyone noticed.
"""

from __future__ import annotations

import json

import whileai.simulations as wai
from tests.helpers import TOOLS
from whileai.simulations.generate.agents import local_model
from whileai.simulations.ingest.traces import mine_traces
from whileai.simulations.score.grading import dead_tools, tool_outcomes

TASKS = [{"prompt": f"refund order ORD-{i} please"} for i in range(12)]


def _step(tool: str, result) -> dict:
    return {"tool": tool, "arguments": {"order_id": "ORD-1"}, "result": result}


def _rows(tool: str, ok: int, missed: int, faults: dict | None = None) -> list[dict]:
    steps = [_step(tool, {"status": "ok"})] * ok + [_step(tool, {"status": "error"})] * missed
    return [{"prompt": "p", "steps": steps, "final_text": "x", "faults": faults}]


def test_rate_rule_catches_four_of_six_hundred_and_twelve():
    # A zero-success test alone misses the lane that opened the issue.
    outcomes = tool_outcomes(_rows("run_query", ok=4, missed=608))
    assert outcomes["run_query"] == {"n": 612, "ok": 4, "fault_n": 608, "injected": 0}
    assert dead_tools(outcomes) == ["run_query"]
    # 2 of 5 is a tool that works sometimes, not a dead one.
    assert dead_tools(tool_outcomes(_rows("export_csv", ok=2, missed=3))) == []
    # No success in 3 answered calls is enough; in 2 it is not.
    assert dead_tools(tool_outcomes(_rows("export_csv", ok=0, missed=3))) == ["export_csv"]
    assert dead_tools(tool_outcomes(_rows("export_csv", ok=0, missed=2))) == []
    # 1 of 12 is over 5%: rare, not dead.
    assert dead_tools(tool_outcomes(_rows("export_csv", ok=1, missed=11))) == []


def test_scheduled_faults_and_unrecorded_steps_are_not_evidence():
    # The run told the world to fail this tool; that is not a dead tool.
    planned = tool_outcomes(_rows("lookup_order", 0, 6, {"lookup_order": {"mode": "timeout"}}))
    assert planned["lookup_order"] == {"n": 6, "ok": 0, "fault_n": 6, "injected": 6}
    assert dead_tools(planned) == []
    wildcard = tool_outcomes(_rows("lookup_order", 0, 6, {"*": {"mode": "timeout", "rate": 1}}))
    assert dead_tools(wildcard) == []
    # A step with no recorded result says nothing either way, so an offline
    # row list never accuses a tool it never saw answer.
    bare = [{"steps": [{"tool": "lookup_order", "arguments": {}}] * 20}]
    assert tool_outcomes(bare) == {}
    assert dead_tools(tool_outcomes(bare)) == []


def _fake_complete(_url, _model, messages, **kwargs):
    """Agent turns: create the refund, look the order up, then answer."""
    tools_seen = sum(1 for m in messages if m.get("role") == "tool")
    if kwargs.get("tools") and tools_seen < 2:
        name = "create_refund" if tools_seen == 0 else "lookup_order"
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": f"c{tools_seen}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps({"order_id": "ORD-1", "amount": 5}),
                    },
                }
            ],
        }
    last = str(messages[-1].get("content") or "") if messages else ""
    if "error" in last.lower():
        return {"content": "The refund tool returned an error, so I could not refund it."}
    return {"content": "Refunded order ORD-1."}


def _run(monkeypatch, world):
    monkeypatch.setattr("whileai.simulations.generate.agents.complete", _fake_complete)
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=6, execute=world)
    return wai.simulate(
        agent=agent,
        tools=TOOLS,
        system_prompt="Refund desk.",
        execute=world,
        tasks=TASKS,
        fault_rate=0.0,
        simulator=False,
        concurrency=2,
        seed=0,
    )


def test_execute_world_missing_a_declared_tool_is_named(monkeypatch):
    def world(tool, args):
        if tool == "lookup_order":
            return {"status": "ok", "order_id": args["order_id"]}
        raise KeyError(tool)  # no branch for create_refund

    data = _run(monkeypatch, world)
    calls = data.search["tools"]
    assert calls["create_refund"]["n"] >= 12 and calls["create_refund"]["ok"] == 0
    assert calls["lookup_order"]["ok"] == calls["lookup_order"]["n"] >= 12
    assert data.search["dead_tools"] == ["create_refund"]
    assert "dead_tools" in data.degraded
    assert data.report()["dead_tools"] == ["create_refund"]
    assert data.report()["tools"]["create_refund"]["fault_n"] == calls["create_refund"]["n"]
    note = [w for w in data.warnings if "never worked" in w]
    assert len(note) == 1
    assert "create_refund (0 of" in note[0]
    assert "lookup_order" not in note[0]
    assert "Add a branch for each in execute=, or remove it from the tool schema" in note[0]
    # The same table the trace miner builds, so the two agree on fault_n.
    mined = mine_traces(data.trajectories)["tools"]
    assert mined["create_refund"]["fault_n"] == calls["create_refund"]["fault_n"]
    assert mined["lookup_order"]["fault_n"] == calls["lookup_order"]["fault_n"]


def test_healthy_execute_world_is_quiet(monkeypatch):
    def world(tool, args):
        return {"status": "ok" if tool == "lookup_order" else "created", "id": "re_1"}

    data = _run(monkeypatch, world)
    calls = data.search["tools"]
    assert calls["create_refund"]["ok"] == calls["create_refund"]["n"] >= 12
    assert data.search["dead_tools"] == []
    assert "dead_tools" not in data.degraded
    assert data.report()["dead_tools"] == []
    assert not any("never worked" in w for w in data.warnings)


def test_coverage_warnings_names_dead_tools_over_a_row_list():
    rows = _rows("run_query", ok=4, missed=608) + _rows("export_csv", ok=2, missed=3)
    notes = wai.coverage_warnings(rows, tools=["run_query", "export_csv"])
    dead = [n for n in notes if "never worked" in n]
    assert len(dead) == 1
    assert "run_query (4 of 612 calls succeeded)" in dead[0]
    assert "export_csv" not in dead[0]
    assert "execute=" in dead[0] and "seeds=" in dead[0]
    healthy = wai.coverage_warnings(_rows("run_query", ok=50, missed=10), tools=["run_query"])
    assert not any("never worked" in n for n in healthy)
