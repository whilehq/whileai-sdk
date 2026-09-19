"""The offline writer names records the world has.

docs/evals.md promises that ids in a tool description ("Orders on file:
A1001, A1002") reach the asks. Before this the offline fallback hashed a
made-up ``ORD-4017`` for every situation, so a scripted agent answered
"not found" on all of them and the held-out set could not fail on any
policy branch: 57 of 63 asks in the strengthen-your-evals check were
off every branch.
"""

from __future__ import annotations

import re

import whileai.simulations as wai
from whileai.simulations.generate.scenarios import _reference_id, known_ids

IDS = ["A1001", "A1002", "A1003", "A1004"]


def _tools(with_ids: bool) -> list[dict]:
    note = f" Orders on file: {', '.join(IDS)}." if with_ids else ""
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up an order by id." + note,
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]


def test_known_ids_reads_descriptions_and_parameter_docs():
    assert known_ids(_tools(True)) == IDS
    assert known_ids(_tools(False)) == []
    anthropic = [
        {
            "name": "get_account",
            "description": "One account.",
            "input_schema": {
                "type": "object",
                "properties": {"account": {"type": "string", "description": "ACC-100 or ACC-200"}},
            },
        }
    ]
    assert known_ids(anthropic) == ["ACC-100", "ACC-200"]


def test_reference_id_draws_from_known_ids_deterministically():
    region = {"id": "probe_1"}
    tools = _tools(True)
    assert _reference_id(region, tools) in IDS
    assert _reference_id(region, tools) == _reference_id(region, tools)
    drawn = {_reference_id({"id": f"r{i}"}, tools, variant=i) for i in range(40)}
    assert drawn == set(IDS), "forty draws should cover four ids"
    fallback = _reference_id(region, _tools(False))
    assert re.fullmatch(r"ORD-\d{4}", fallback), fallback


def test_offline_asks_land_on_records_the_world_has():
    def agent(message: str) -> dict:
        return {"steps": [], "final_text": "ok"}

    data = wai.simulate(
        agent,
        tools=_tools(True),
        system_prompt="Refund delivered orders within 30 days. Look the order up first.",
        situations=24,
        budget=24,
        simulator=False,
        reproducible=True,
        seed=0,
        fault_rate=0.0,
        avg_turns=1,
    )
    prompts = [r["prompt"] for r in data.rows()]
    referenced = [p for p in prompts if re.search(r"\b[A-Z]{1,4}-?\d{3,6}\b", p)]
    assert referenced, "some asks name a record"
    assert all(any(i in p for i in IDS) for p in referenced), referenced
    assert not any("ORD-" in p for p in prompts), "no invented ids once the world names its own"
