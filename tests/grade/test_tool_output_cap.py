"""training_rows(max_tool_output_chars=): tool output is cut out loud
(Lambert 2025, chapter Tool Use)."""

from __future__ import annotations

from whileai.simulations.export import export_training, training_rows
from whileai.simulations.generate import adapters

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    }
]


def _row(result_len=500):
    big = "x" * result_len
    return {
        "prompt": "where is order 4412",
        "final_text": "Shipped.",
        "reward": 1,
        "steps": [],
        "messages": [
            {"role": "user", "content": "where is order 4412"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"id": "4412"}'},
                    }
                ],
            },
            {"role": "tool", "content": big, "tool_call_id": "c1"},
            {"role": "assistant", "content": "Shipped."},
        ],
    }


def test_no_cap_leaves_tool_output_alone():
    rows = training_rows([_row()], tools=TOOLS)
    tool = next(m for m in rows[0]["messages"] if m["role"] == "tool")
    assert len(tool["content"]) == 500 and "tool_output_truncated" not in rows[0]


def test_cap_cuts_with_a_marker_and_counts_it(tmp_path):
    rows = training_rows([_row(), _row(50)], tools=TOOLS, max_tool_output_chars=100)
    tool = next(m for m in rows[0]["messages"] if m["role"] == "tool")
    assert tool["content"].startswith("x" * 100)
    assert tool["content"].endswith("[... 400 chars of tool output truncated]")
    assert rows[0]["tool_output_truncated"] == 1 and rows[0]["tool_output_chars_cut"] == 400
    assert "tool_output_truncated" not in rows[1]
    # the loss mask still excludes the tool turn
    assert rows[0]["loss_mask"][2] == 0
    report = export_training(
        [_row(), _row()], str(tmp_path / "t.jsonl"), tools=TOOLS, max_tool_output_chars=100
    )
    assert report["tool_output_truncated"] == 2 and report["tool_output_chars_cut"] == 800
    assert report["max_tool_output_chars"] == 100
    plain = export_training([_row()], str(tmp_path / "p.jsonl"), tools=TOOLS)
    assert plain["tool_output_truncated"] == 0 and plain["max_tool_output_chars"] is None


def test_claude_code_cap_is_named():
    assert adapters.CLAUDE_CODE_RESULT_CHARS == 2000
