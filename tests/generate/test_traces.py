"""Trace mining, focused grids, pseudo-production splits, leakage gate.

Offline: hash embedder, template generator, scripted agent. No GPU.
"""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, scripted_agent
from whileai.simulations.ingest.traces import (
    dimensions_from_traces,
    drop_leaky_rows,
    leakage_report,
    mine_traces,
    simulate_from_traces,
    split_pseudo_production,
)


def _trace(prompt, tool, status, reward=None, **result_extra):
    result = {"status": status, **result_extra}
    return {
        "prompt": prompt,
        "steps": [{"tool": tool, "arguments": {"order_id": "ord_11"}, "result": result}],
        "final_text": "I could not find that order."
        if status != "ok"
        else "Order ord_11 is confirmed.",
        "reward": reward,
    }


TRACES = [
    _trace("where is order 4412", "lookup_order", "not_found", reward=0),
    _trace("check on my order pls", "lookup_order", "not_found", reward=0),
    _trace("refund order 9911 now", "create_refund", "timeout", reward=1),
    _trace("status of order 130", "lookup_order", "ok", reward=1),
    _trace("refund my double charge", "create_refund", "ok", reward=1),
]


def test_mine_traces_counts_tools_faults_and_flaws():
    mined = mine_traces(TRACES)
    assert mined["n"] == 5
    assert mined["tools"]["lookup_order"]["n"] == 3
    assert mined["tools"]["lookup_order"]["fault_n"] == 2
    assert mined["faults"] == {"not_found": 2, "timeout": 1}
    assert len(mined["flaw_rows"]) == 3  # two misses plus the timeout row
    assert len(mined["asks"]) == 5


def test_dimensions_focus_on_observed_behaviors():
    dims = dimensions_from_traces(TRACES, TOOLS, POLICY, broaden=False)
    base = wai.build_dimensions(TOOLS, POLICY)
    # Failing tool leads; the never-observed tool is dropped when narrow.
    assert dims["tool"][0] == "lookup_order"
    assert "get_refund_status" not in dims["tool"]
    # Observed faults, plus the clean value for contrast.
    assert dims["tool_condition"][0] == "success"
    assert "timeout" in dims["tool_condition"]
    assert "not_found" not in dims["tool_condition"]  # world axis, not a condition
    assert dims["world_state"][0] == "entity exists"
    assert "entity missing" in dims["world_state"]
    # Axes the traces cannot see stay whole.
    assert dims["stance"] == base["stance"]
    assert dims["rule"] == base["rule"]


def test_dimensions_broaden_keeps_unseen_tools_behind_observed():
    dims = dimensions_from_traces(TRACES, TOOLS, POLICY, broaden=True)
    assert dims["tool"][0] == "lookup_order"
    assert "get_refund_status" in dims["tool"]


def test_split_pseudo_production_holds_out_each_unique_flaw():
    production, remainder = split_pseudo_production(TRACES, fraction=0.4, seed=3)
    assert len(production) + len(remainder) == len(TRACES)
    prod_prompts = {row["prompt"] for row in production}
    rem_prompts = {row["prompt"] for row in remainder}
    assert not (prod_prompts & rem_prompts)
    # Both distinct flaw kinds are represented on the production side.
    from whileai.simulations.score.grading import trace_fault

    faults = {trace_fault(row) for row in production}
    assert "not_found" in faults
    assert "timeout" in faults
    again, _ = split_pseudo_production(TRACES, fraction=0.4, seed=3)
    assert [row["prompt"] for row in again] == [row["prompt"] for row in production]


def _repeated(prompt, k, tool="lookup_order", status="not_found", reward=0):
    """``k`` rollouts of one prompt, the shape ``repeats=k`` produces."""
    return [dict(_trace(prompt, tool, status, reward=reward), rollout_index=i) for i in range(k)]


def test_split_keeps_every_repeat_of_a_prompt_on_one_side():
    """mode="rl" gives each prompt k rows; a row-wise split straddles them.

    The student must not train on a prompt it is evaluated on, and a
    row-disjoint split is not prompt-disjoint.
    """
    rows = (
        _repeated("where is order 4412", 6)
        + _repeated("refund order 9911 now", 6, tool="create_refund", status="timeout", reward=1)
        + _repeated("status of order 130", 6, status="ok", reward=1)
        + _repeated("refund my double charge", 6, tool="create_refund", status="ok", reward=1)
    )
    production, remainder = split_pseudo_production(rows, fraction=0.25, seed=0)
    assert len(production) + len(remainder) == len(rows)
    prod_prompts = {row["prompt"] for row in production}
    rem_prompts = {row["prompt"] for row in remainder}
    assert prod_prompts, "the held-out side is empty"
    assert rem_prompts, "the training side is empty"
    assert not (prod_prompts & rem_prompts)
    # Whole tasks move, so every repeat of a held-out prompt is held out.
    for prompt in prod_prompts:
        assert sum(1 for row in production if row["prompt"] == prompt) == 6


def test_split_groups_promptless_rows_by_scenario_id():
    rows = []
    for sid in ("sc_a", "sc_b", "sc_c", "sc_d"):
        for i in range(4):
            row = _trace("", "lookup_order", "not_found", reward=0)
            row.pop("prompt")
            rows.append(dict(row, scenario_id=sid, rollout_index=i))
    production, remainder = split_pseudo_production(rows, fraction=0.25, seed=0)
    prod_ids = {row["scenario_id"] for row in production}
    rem_ids = {row["scenario_id"] for row in remainder}
    assert prod_ids and rem_ids
    assert not (prod_ids & rem_ids)


def test_split_does_not_group_rows_that_have_neither_key():
    """An empty prompt is not a task two rows share."""
    rows = []
    for i in range(8):
        row = _trace("", "lookup_order", "not_found", reward=0)
        row["prompt"] = ""
        rows.append(dict(row, marker=i))
    production, remainder = split_pseudo_production(rows, fraction=0.25, seed=0)
    assert len(production) + len(remainder) == len(rows)
    assert production and remainder, "all eight rows were swept onto one side"


def test_leakage_report_flags_copies_not_fresh_asks():
    generated = [
        {"prompt": "where is order 4412"},  # exact copy
        {"prompt": "Where is  ORDER 4412"},  # case/space copy
        {"prompt": "my package never arrived and support is not answering"},
    ]
    report = leakage_report(generated, TRACES, threshold=0.9)
    assert report["n_leaky"] == 2
    assert report["max_similarity"] == 1.0
    kept, drop = drop_leaky_rows(generated, TRACES, threshold=0.9)
    assert len(kept) == 1
    assert kept[0]["prompt"].startswith("my package")
    assert drop["n_dropped"] == 2


_ISSUE_479_Q = "Why are so few drugs with promising animal trials tested in humans?"
_ISSUE_479_HOLDOUT = [{"prompt": _ISSUE_479_Q, "task_id": "h1"}]
_ISSUE_479_TRAIN = [{"prompt": _ISSUE_479_Q, "task_id": "t1"}]


@pytest.mark.parametrize(
    "sources",
    [
        pytest.param(_ISSUE_479_HOLDOUT, id="list-of-rows"),
        pytest.param([_ISSUE_479_HOLDOUT], id="list-of-row-lists"),
        pytest.param([_ISSUE_479_Q], id="list-of-strings"),
        pytest.param(_ISSUE_479_HOLDOUT[0], id="single-row"),
        pytest.param((_ISSUE_479_HOLDOUT, [{"prompt": "unrelated"}]), id="tuple-of-row-lists"),
    ],
)
def test_byte_identical_row_dropped_under_every_sources_shape(sources):
    """#479: a byte-identical holdout row must always be flagged, whatever
    container ``sources=`` arrived in and whatever the embedder thinks."""
    kept, report = drop_leaky_rows(_ISSUE_479_TRAIN, sources=sources, threshold=0.9)
    assert kept == []
    assert report["n_leaky"] == 1
    assert report["n_dropped"] == 1
    assert report["n_sources"] >= 1
    assert report["max_similarity"] == 1.0
    assert leakage_report(_ISSUE_479_TRAIN, sources, threshold=0.9)["n_leaky"] == 1


def test_drop_leaky_rows_agrees_with_decontaminate_on_issue_479():
    """The issue's reproduction: both siblings drop the identical row when
    handed the list-of-lists shape ``decontaminate`` documents."""
    clean, decon = wai.decontaminate(_ISSUE_479_TRAIN, against=[_ISSUE_479_HOLDOUT])
    kept_nested, nested = drop_leaky_rows(
        _ISSUE_479_TRAIN, sources=[_ISSUE_479_HOLDOUT], threshold=0.9
    )
    kept_flat, flat = drop_leaky_rows(_ISSUE_479_TRAIN, sources=_ISSUE_479_HOLDOUT, threshold=0.9)
    assert (len(clean), decon["n_contaminated"]) == (0, 1)
    assert (len(kept_nested), nested["n_leaky"]) == (0, 1)
    assert kept_flat == kept_nested == []
    assert nested == flat
    assert nested["max_similarity"] == 1.0


def test_leak_sources_ignores_empty_and_none():
    assert drop_leaky_rows(_ISSUE_479_TRAIN, sources=[])[1]["n_leaky"] == 0
    assert drop_leaky_rows(_ISSUE_479_TRAIN, sources=[[]])[1]["n_sources"] == 0
    assert drop_leaky_rows(_ISSUE_479_TRAIN, sources=None)[1]["n_sources"] == 0


def test_simulate_from_traces_offline_end_to_end():
    data = simulate_from_traces(
        TRACES,
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        mode="explore",
        budget=6,
        seed=0,
        grade=False,
        concurrency=6,
        simulator=False,
        time_budget=20,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert data.trajectories
    mining = data.search["trace_mining"]
    assert mining["n_traces"] == 5
    assert mining["faults"]["not_found"] == 2
    assert "focused_dimensions" in mining
    leak = data.search["trace_leakage"]
    assert leak["n_sources"] == 5
    trace_prompts = {" ".join(t["prompt"].lower().split()) for t in TRACES}
    for row in data.trajectories:
        assert " ".join(str(row["prompt"]).lower().split()) not in trace_prompts


def test_flaw_rows_returns_next_round_seeds():
    from whileai.simulations.ingest.traces import flaw_rows

    flawed = flaw_rows(TRACES)
    prompts = {row["prompt"] for row in flawed}
    assert len(flawed) == 3
    assert "status of order 130" not in prompts
    assert "where is order 4412" in prompts


def test_rl_retarget_lets_behavior_gap_lead():
    from whileai.simulations.generate.scenarios import retarget_regions, scenario_regions

    regions = scenario_regions(TOOLS, POLICY, mode="rl")[:6]
    gappy = {regions[0]["id"]}

    def behavior_value(assignment):
        import json as _json

        key = _json.dumps(assignment, sort_keys=True, default=str)
        first = _json.dumps(regions[0]["assignment"], sort_keys=True, default=str)
        return 1.0 if key == first else 0.0

    default = [dict(r) for r in regions]
    retarget_regions(default, TOOLS, behavior_value=behavior_value)
    rl = [dict(r) for r in regions]
    retarget_regions(rl, TOOLS, behavior_value=behavior_value, mode="rl")
    spread_default = default[0]["weight"] - default[1]["weight"]
    spread_rl = rl[0]["weight"] - rl[1]["weight"]
    # Same behavior gap moves RL weights harder than explore weights.
    assert spread_rl > spread_default
    assert gappy  # regions resolved


def test_mine_traces_attributes_faults_to_the_faulted_call():
    from whileai.simulations.ingest.traces import format_trace_report, trace_report

    row = {
        "prompt": "fix the failing test",
        "reward": 0,
        "steps": [
            {"tool": "read_file", "arguments": {"path": "a.py"}, "result": "def f(): pass"},
            {
                "tool": "run_command",
                "arguments": {"cmd": "pytest"},
                "result": {"status": "error", "error": "exit 1"},
            },
            {"tool": "read_file", "arguments": {"path": "b.py"}, "result": "x = 1"},
        ],
        "final_text": "done",
    }
    mined = mine_traces([row])
    assert mined["tools"]["read_file"] == {"n": 2, "fault_n": 0}
    assert mined["tools"]["run_command"] == {"n": 1, "fault_n": 1}
    assert mined["flaw_rows"] == [0]
    assert "run_command x1 (1 calls faulted)" in format_trace_report(trace_report([row]))
