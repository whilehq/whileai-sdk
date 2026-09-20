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
    setting off a card describes an intention; the rows carry the truth.
    The check itself, on the numbers a multi-turn run hands it."""
    from whileai.simulations.run.engine import Run

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        Run._warn_on_undelivered(
            Run.__new__(Run),
            {"fault_rate": 0.0, "avg_turns": 12, "stance": None},
            {"fault_share": 0.0, "mean_user_turns": 0.44, "stance_mix": {}},
        )
    assert any("did not deliver what was set" in str(w.message) for w in seen)
    assert any("avg_turns=12" in str(w.message) for w in seen)


def test_a_knob_that_lands_does_not_warn_about_itself():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _run(fault_rate=0.0, avg_turns=1).report()
    assert not any("avg_turns" in str(w.message) for w in seen)


def test_a_knob_the_caller_never_set_is_not_a_missed_setting():
    """The landing page's first block, verbatim (#476). It sets neither
    fault_rate nor avg_turns; under mode="rl" both arrive at the engine
    carrying a default the offline path cannot reach (0.8 and 12), and the
    reader was told their settings failed. The report still says what the
    run resolved; only the warning is tied to what was passed."""
    import whileai as wai_pkg

    @wai_pkg.tool
    def get_order(order_id: str) -> dict:
        """Look up an order by id."""
        ...

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        data = wai_pkg.simulate(
            wai_pkg.seeded_agent([get_order]),
            tools=[get_order],
            system_prompt="Help customers with orders.",
            simulator=False,
            mode="rl",
            repeats=4,
            repeat_policy="fixed",
            budget=64,
        )
    assert not [w for w in seen if issubclass(w.category, UserWarning)]
    rep = data.report()
    assert rep["requested"]["fault_rate"] == 0.8
    assert rep["requested"]["avg_turns"] == 12.0
    assert rep["delivered"]["fault_share"] < 0.8


def test_a_fault_rate_the_caller_set_and_the_rows_missed_still_warns():
    """The same block with fault_rate named: the offline path delivers about
    half, and that is a setting the run failed, so it says so."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _run(fault_rate=0.9, mode="rl").report()
    msgs = [str(w.message) for w in seen if issubclass(w.category, UserWarning)]
    assert any("fault_rate=0.9" in m for m in msgs), msgs
    assert not any("avg_turns" in m for m in msgs), msgs


def test_avg_turns_is_not_checked_when_the_agent_is_played_single_turn():
    """A callable agent takes one message in, one trajectory out; the
    multi-turn user model does not apply (docs/evals.md), so avg_turns
    cannot be missed there at any setting, named or not."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        rep = _run(fault_rate=0.0, avg_turns=12).report()
    assert rep["requested"]["avg_turns"] == 12
    assert rep["delivered"]["mean_user_turns"] < 6
    assert not any("avg_turns" in str(w.message) for w in seen)
