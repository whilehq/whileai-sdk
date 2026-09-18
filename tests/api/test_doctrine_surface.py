"""The doctrine-sketch surface: grader=, strategy=, evaluate(grader=,
eval_set=), results.traces. Each addition maps to a banked finding; the
tests pin both behavior and disclosure."""

from __future__ import annotations

import pytest

from tests.helpers import POLICY, TOOLS, scripted_agent
from whileai.simulations import evaluate, simulate
from whileai.simulations.ingest.traces import simulate_from_traces

TRACES = [
    {
        "prompt": "where is order 4412",
        "steps": [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": "4412"},
                "result": {"status": "not_found"},
            }
        ],
        "final_text": "I could not find that order.",
    },
    {
        "prompt": "refund order 9911 now",
        "steps": [
            {
                "tool": "create_refund",
                "arguments": {"order_id": "9911"},
                "result": {"status": "timeout"},
            }
        ],
        "final_text": "The refund request timed out.",
    },
]

_OFFLINE = dict(
    mode="explore",
    budget=4,
    seed=0,
    grade=False,
    concurrency=4,
    simulator=False,
    time_budget=20,
    advanced={"per_round": 4, "mutate_failures": False},
)


def test_strategy_rejects_unknown_and_impossible_values():
    with pytest.raises(ValueError, match="strategy"):
        simulate(tools=TOOLS, system_prompt=POLICY, strategy="clever")
    with pytest.raises(ValueError, match="needs traces"):
        simulate(tools=TOOLS, system_prompt=POLICY, strategy="targeted")
    with pytest.raises(ValueError, match="needs traces"):
        simulate(tools=TOOLS, system_prompt=POLICY, strategy="trace")


def test_auto_resolves_to_trace_and_records_reason():
    data = simulate_from_traces(TRACES, scripted_agent, tools=TOOLS, policy=POLICY, **_OFFLINE)
    strategy = data.search["strategy"]
    assert strategy["requested"] == "auto"
    assert strategy["resolved"] == "trace"
    assert strategy["broaden"] is True
    assert "aimed" in strategy["reason"]


def test_targeted_narrows_the_tool_axis():
    data = simulate_from_traces(
        TRACES, scripted_agent, tools=TOOLS, policy=POLICY, strategy="targeted", **_OFFLINE
    )
    assert data.search["strategy"]["resolved"] == "targeted"
    assert data.search["strategy"]["broaden"] is False
    focused = data.search["trace_mining"]["focused_dimensions"]
    observed = {"lookup_order", "create_refund"}
    real_tools = [t for t in focused["tool"] if t not in {"unrelated", "multi_tool"}]
    assert set(real_tools) <= observed, "targeted must drop tools the traces never touched"


def test_grader_runs_inside_the_loop_and_discloses():
    calls = []

    def grader(row):
        calls.append(row["prompt"])
        return {"reward": 1, "reason": "ok", "failure_class": None}

    data = simulate_from_traces(
        TRACES, scripted_agent, tools=TOOLS, policy=POLICY, grader=grader, **_OFFLINE
    )
    assert calls, "the customer grader must actually be invoked"
    summary = data.search["grader"]
    assert summary["scored"] == len(data.trajectories)
    assert (
        summary["passes"] + summary["failures"] + summary["partials"] + summary["unjudged"]
        == summary["scored"]
    )
    for row in data.trajectories:
        assert row["reward"] == 1
        assert row["judge_status"] == "ok"
        assert row["lineage"]["source"] == "grade"


def test_broken_grader_never_writes_silent_zeros():
    def broken(row):
        raise RuntimeError("customer judge bug")

    data = simulate_from_traces(
        TRACES, scripted_agent, tools=TOOLS, policy=POLICY, grader=broken, **_OFFLINE
    )
    assert data.search["grader"]["unjudged"] == len(data.trajectories)
    for row in data.trajectories:
        assert row["reward"] is None
        assert row["judge_status"] == "error"


def test_evaluate_grader_alias_and_coverage():
    rollouts = [dict(t) for t in TRACES]
    with pytest.raises(ValueError, match="not both"):
        evaluate(rollouts, lambda r: 1, grader=lambda r: 1)
    with pytest.raises(ValueError, match="needs a judge"):
        evaluate(rollouts)
    scored = evaluate(
        rollouts,
        grader=lambda r: 0,
        model="agent-v2",
        eval_set=["where is order 4412", "refund order 9911 now", "a prompt nobody rolled out"],
        concurrency=1,
    )
    assert scored.eval_coverage["eval_set"] == 3
    assert scored.eval_coverage["covered"] == 2
    assert scored.eval_coverage["missing"] == ["a prompt nobody rolled out"]
    assert scored.traces == scored.failed_traces()
    assert len(scored.traces) == 2


def test_steering_weight_validates_and_records():
    with pytest.raises(ValueError, match="0, 1"):
        simulate(tools=TOOLS, system_prompt=POLICY, traces=TRACES, steering_weight=1.5)
    with pytest.raises(ValueError, match="needs traces"):
        simulate(tools=TOOLS, system_prompt=POLICY, steering_weight=0.5)
    data = simulate_from_traces(
        TRACES, scripted_agent, tools=TOOLS, policy=POLICY, steering_weight=0.5, **_OFFLINE
    )
    sw = data.search["strategy"]["steering_weight"]
    assert sw == {"requested": 0.5, "applied": 0.5, "source": "override"}
    default = simulate_from_traces(TRACES, scripted_agent, tools=TOOLS, policy=POLICY, **_OFFLINE)
    sw = default.search["strategy"]["steering_weight"]
    assert sw["source"] == "rule" and sw["requested"] is None


def test_metadata_is_small_structured_and_uninterpreted():
    """result.metadata: mechanics only. Labels like sparse/moderate/rich
    are the consumer's job, never the SDK's."""
    data = simulate_from_traces(TRACES, scripted_agent, tools=TOOLS, policy=POLICY, **_OFFLINE)
    meta = data.metadata
    assert set(meta) == {
        "strategy",
        "trace_count",
        "trace_regions",
        "applied_steering_weight",
        "targeted_rows",
        "background_rows",
        "writer_model",
        "user_model",
        "judge_model",
    }
    assert meta["strategy"] == "trace"
    assert meta["trace_count"] == len(TRACES)
    assert meta["targeted_rows"] + meta["background_rows"] == len(data.trajectories)
    for key, value in meta.items():
        if key in ("writer_model", "user_model", "judge_model"):
            continue  # model tags are identifiers, not labels
        assert not isinstance(value, str) or value in ("auto", "broad", "trace", "targeted"), (
            "no presentation strings in metadata"
        )
