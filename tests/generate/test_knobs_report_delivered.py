"""A knob that does not deliver must say so, not describe an intention."""

from __future__ import annotations

import warnings

import whileai.simulations as wai

TOOLS = [
    {
        "type": "function",
        "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}},
    }
]
TASKS = [{"prompt": f"task {i}"} for i in range(6)]


def _agent(messages, tools=None, **_):
    return {"steps": [], "final_text": "done"}


def _run(**kw):
    return wai.simulate(
        agent=_agent,
        tools=TOOLS,
        system_prompt="p",
        execute=lambda t, a: {"ok": True},
        tasks=TASKS,
        simulator=False,
        **kw,
    )


def test_the_report_carries_what_was_set_and_what_arrived():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = _run(fault_rate=0.0).report()
    assert rep["requested"]["fault_rate"] == 0.0
    d = rep["delivered"]
    assert d["rows"] == 6
    # every field a knob is supposed to steer is measured from the rows
    for key in (
        "fault_share",
        "tier_mix",
        "stance_mix",
        "mean_user_turns",
        "user_turns_3plus_share",
    ):
        assert key in d, key
    assert 0.0 <= d["fault_share"] <= 1.0


def test_a_turn_knob_that_misses_its_setting_warns():
    """avg_turns 6 measured 0.44 mean user turns on a real pool. Reading the
    setting off a card describes an intention; the rows carry the truth."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        rep = _run(fault_rate=0.0, avg_turns=12).report()
    assert rep["requested"]["avg_turns"] == 12
    assert rep["delivered"]["mean_user_turns"] < 6
    assert any("did not deliver what was set" in str(w.message) for w in seen)
    assert any("avg_turns=12" in str(w.message) for w in seen)


def test_a_knob_that_lands_does_not_warn_about_itself():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _run(fault_rate=0.0, avg_turns=1).report()
    assert not any("avg_turns" in str(w.message) for w in seen)
