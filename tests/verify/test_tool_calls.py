"""``wai.verify.tool_calls`` reads a tool call the same from either spelling:
the flat rollout shape and the OpenAI wire shape an export writes (#594)."""

from __future__ import annotations

import json

import whileai as wai
from whileai.simulations.verify import ToolCall, tool_calls

FLAT = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"name": "get_order", "arguments": {"order_id": "ORD-4550"}}],
}
WIRE = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_0000",
            "type": "function",
            "function": {"name": "get_order", "arguments": json.dumps({"order_id": "ORD-9237"})},
        }
    ],
}


def test_flat_and_wire_messages_read_the_same():
    flat = tool_calls(FLAT)
    wire = tool_calls(WIRE)
    assert [c.name for c in flat] == [c.name for c in wire] == ["get_order"]
    assert flat[0].arguments == {"order_id": "ORD-4550"}
    assert wire[0].arguments == {"order_id": "ORD-9237"}
    assert flat[0].id is None and wire[0].id == "call_0000"
    assert isinstance(flat[0], ToolCall)


def test_row_reads_every_assistant_call_in_order():
    row = {
        "messages": [
            {"role": "user", "content": "refund ORD-1 and ORD-2"},
            FLAT,
            {"role": "tool", "name": "get_order", "content": "{}"},
            WIRE,
        ]
    }
    calls = tool_calls(row)
    assert [(c.name, c.arguments["order_id"]) for c in calls] == [
        ("get_order", "ORD-4550"),
        ("get_order", "ORD-9237"),
    ]


def test_string_empty_and_missing_arguments_read_as_dicts():
    message = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "a", "arguments": ""}},
            {"function": {"name": "b"}},
            {"name": "c", "arguments": None},
            {"function": {"name": "d", "arguments": "not json"}},
            {"function": {"name": "e", "arguments": "[1, 2]"}},
        ],
    }
    calls = tool_calls(message)
    assert [c.name for c in calls] == list("abcde")
    assert all(c.arguments == {} for c in calls)


def test_row_without_messages_reads_steps_and_nothing_else_is_empty():
    row = {"steps": [{"tool": "lookup", "arguments": {"id": 1}, "result": {}}, {"text": "done"}]}
    assert tool_calls(row) == [ToolCall(name="lookup", arguments={"id": 1})]
    pull = {"tool_trace": [{"tool": "lookup", "input": json.dumps({"id": 2}), "output": {}}]}
    assert tool_calls(pull)[0].arguments == {"id": 2}
    assert tool_calls({"messages": [{"role": "user", "content": "hi"}]}) == []
    assert tool_calls({"role": "assistant", "content": "hi"}) == []
    assert tool_calls(None) == []
    assert tool_calls("text") == []


def test_one_reward_scores_rollout_and_exported_rows_alike():
    def opened_with_lookup(row):
        calls = wai.verify.tool_calls(row)
        return float(bool(calls) and calls[0].name == "get_order")

    rollout = {"messages": [{"role": "user", "content": "hi"}, FLAT]}
    exported = {"messages": [{"role": "user", "content": "hi"}, WIRE]}
    assert opened_with_lookup(rollout) == opened_with_lookup(exported) == 1.0
