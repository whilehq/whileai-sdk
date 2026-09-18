"""#301 and #302: the local_model knobs a researcher needed and could not find.

``result_shapes=`` pins a tool result so a policy branch is reached on
purpose; it was documented nowhere. ``timeout=60`` was shorter than a
served model's 113 s cold start, so the first pass came back with 0 rows
and no word about why."""

from __future__ import annotations

import inspect

import whileai.simulations as wai
from tests.helpers import simulate_offline
from whileai.simulations.generate.agents import LOCAL_MODEL_TIMEOUT, local_model
from whileai.simulations.run.config import resolve_run_config

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_invoice",
            "description": "Look up an invoice.",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
        },
    }
]


def test_default_timeout_survives_a_served_model_cold_start():
    """113 s measured on the account's own endpoint after idling (#302);
    the hosted judge takes two to three minutes. The default clears both."""
    default = inspect.signature(local_model).parameters["timeout"].default
    assert default == LOCAL_MODEL_TIMEOUT >= 180
    # simulate(timeout=) shares it, so a hosted or spec'd agent gets the same room
    assert resolve_run_config(tools=TOOLS, system_prompt="p").rollout_timeout == default
    assert (
        resolve_run_config(tools=TOOLS, system_prompt="p", passed={"timeout": 30}).rollout_timeout
        == 30
    )


def test_a_timed_out_call_names_the_fix_in_warnings():
    calls = {"n": 0}

    def cold_endpoint(message):
        # socket.timeout is TimeoutError on 3.10+ and reads "timed out"
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("timed out")
        return {"steps": [], "final_text": "ok"}

    data = simulate_offline(cold_endpoint, tools=TOOLS, budget=6, per_round=6, concurrency=1)
    [note] = [w for w in data.warnings if "timed out" in w]
    assert "timeout=" in note and "local_model" in note and "warm" in note
    # said once, however many calls timed out
    assert sum("timed out" in w for w in data.warnings) == 1
    assert data.trajectories  # the run went on once the endpoint answered


def test_an_agent_that_fails_another_way_gets_no_timeout_note():
    def broken(message):
        raise RuntimeError("BOOM")

    data = simulate_offline(broken, tools=TOOLS, budget=4, per_round=4, concurrency=1)
    assert not [w for w in data.warnings if "timed out" in w]


def test_result_shapes_and_fault_plans_are_documented_where_the_knobs_live():
    doc = local_model.__doc__ or ""
    assert "result_shapes" in doc and "fault_plans" in doc and "timeout" in doc
    # the load-bearing fact for a threshold: a number moves by about a third
    assert "third" in doc
    assert wai.local_model is local_model
