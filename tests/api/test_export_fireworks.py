"""``format="fireworks"``: what a Fireworks managed training job reads.

The shapes are the ones docs.fireworks.ai/fine-tuning documents (SFT:
``messages`` + ``tools`` with a per-message ``weight``; DPO: one-turn
``input`` / ``preferred_output`` / ``non_preferred_output``). Offline: the
rows come from the stand-in agent.
"""

from __future__ import annotations

import json

import pytest

import whileai as wai
from tests.helpers import POLICY, TOOLS
from whileai.config import reset
from whileai.simulations.export import EXPORT_FORMATS, export_preference, training_rows

FIREWORKS_SFT_KEYS = {"messages", "tools"}
FIREWORKS_TURN_KEYS = {"role", "content", "tool_calls", "tool_call_id", "weight"}


@pytest.fixture(autouse=True)
def clean_settings():
    reset()
    yield
    reset()


def _rows():
    data = wai.simulate(
        wai.seeded_agent(TOOLS),
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        mode="rl",
        repeats=2,
        repeat_policy="fixed",
        budget=8,
    )
    return data.rows


def test_fireworks_is_a_format():
    assert "fireworks" in EXPORT_FORMATS


def test_sft_rows_carry_only_what_fireworks_reads_and_the_mask_as_weight(tmp_path):
    rows = _rows()
    out = tmp_path / "sft.jsonl"
    report = wai.export(rows, str(out), system_prompt=POLICY, tools=TOOLS, format="fireworks")
    assert report["format"] == "fireworks" and report["n_written"] == report["n"] > 0
    assert report["tool_call_roundtrip"]["encoding"] == "json_string"
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines and all(set(line) <= FIREWORKS_SFT_KEYS for line in lines)
    sdk_rows = training_rows(rows, system_prompt=POLICY, tools=TOOLS)
    for line, sdk in zip(lines, sdk_rows, strict=True):
        assert len(line["messages"]) == len(sdk["messages"]) == len(sdk["loss_mask"])
        assert line["messages"][0]["role"] == "system"
        assert line["tools"] == list(TOOLS)
        for i, message in enumerate(line["messages"]):
            assert set(message) <= FIREWORKS_TURN_KEYS
            if message["role"] == "assistant":
                assert message["weight"] == (1 if sdk["loss_mask"][i] else 0)
            else:
                assert "weight" not in message
            for call in message.get("tool_calls") or []:
                # the OpenAI wire: arguments stay a JSON string
                assert isinstance(call["function"]["arguments"], str)
    # the report still reads the SDK rows: rewards and groups are not lost to the reshape
    assert report["rewards"]["n_ungraded"] == report["n"]


def test_sft_final_mask_trains_only_the_last_assistant_turn(tmp_path):
    out = tmp_path / "final.jsonl"
    wai.export(
        _rows(), str(out), system_prompt=POLICY, tools=TOOLS, format="fireworks", mask_mode="final"
    )
    for line in out.read_text(encoding="utf-8").splitlines():
        weights = [m["weight"] for m in json.loads(line)["messages"] if m["role"] == "assistant"]
        assert weights and weights[-1] == 1 and sum(weights) == 1


def _pair(chosen_turns, rejected_turns, **extra):
    ask = [{"role": "user", "content": "Refund order 42"}]
    return {
        "prompt": "Refund order 42",
        "chosen": {"messages": ask + chosen_turns},
        "rejected": {"messages": ask + rejected_turns},
        **extra,
    }


def _call(order_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "arguments": json.dumps({"order_id": order_id}),
                },
            }
        ],
    }


def test_preference_rows_are_fireworks_one_turn_pairs(tmp_path):
    pairs = [
        # a tool-calling pass against a bare refusal: the first turns differ, later turns are cut
        _pair(
            [
                _call("42"),
                {"role": "tool", "tool_call_id": "c1", "content": "{}"},
                {"role": "assistant", "content": "Refunded."},
            ],
            [{"role": "assistant", "content": "No."}],
            margin=1.0,
            length_delta=2,
        ),
        # the same opening call and result, then the sides differ: the shared
        # prefix is the input, the differing assistant turn is the preference
        _pair(
            [
                _call("42"),
                {"role": "tool", "tool_call_id": "c1", "content": "{}"},
                {"role": "assistant", "content": "Refunded."},
            ],
            [
                _call("42"),
                {"role": "tool", "tool_call_id": "c1", "content": "{}"},
                {"role": "assistant", "content": "No."},
            ],
        ),
        # one side is a prefix of the other: nothing to contrast, dropped
        _pair([_call("42"), {"role": "assistant", "content": "Done."}], [_call("42")]),
        # the sides diverge on a tool result, not an assistant turn: dropped
        _pair(
            [_call("42"), {"role": "tool", "tool_call_id": "c1", "content": "{}"}],
            [_call("42"), {"role": "tool", "tool_call_id": "c1", "content": '{"eligible": false}'}],
        ),
        # a side with no assistant turn at all: dropped
        _pair([{"role": "assistant", "content": "Refunded."}], []),
    ]
    out = tmp_path / "dpo.jsonl"
    report = export_preference(
        pairs, str(out), system_prompt=POLICY, tools=TOOLS, format="fireworks"
    )
    assert report["pairs"] == 2
    assert report["no_completion_dropped"] == 3
    assert report["fireworks_turns_cut"] == 1
    # the stats still come from the SDK's own pair fields
    assert report["mean_margin"] == 1.0 and report["chosen_longer_frac"] == 1.0
    first, second = [json.loads(s) for s in out.read_text(encoding="utf-8").splitlines()]
    for line in (first, second):
        assert set(line) == {"input", "preferred_output", "non_preferred_output"}
        assert line["input"]["tools"] == list(TOOLS)
        assert len(line["preferred_output"]) == len(line["non_preferred_output"]) == 1
    assert [m["role"] for m in first["input"]["messages"]] == ["system", "user"]
    (preferred,) = first["preferred_output"]
    (non_preferred,) = first["non_preferred_output"]
    assert preferred["role"] == non_preferred["role"] == "assistant"
    assert preferred["tool_calls"][0]["function"]["name"] == "lookup_order"
    assert isinstance(preferred["tool_calls"][0]["function"]["arguments"], str)
    assert non_preferred["content"] == "No."
    assert set(preferred) <= {"role", "content", "tool_calls"}
    assert [m["role"] for m in second["input"]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second["preferred_output"][0]["content"] == "Refunded."
    assert second["non_preferred_output"][0]["content"] == "No."


def test_selection_export_takes_the_format(tmp_path):
    data = wai.simulate(
        wai.seeded_agent(TOOLS),
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        mode="rl",
        repeats=2,
        repeat_policy="fixed",
        budget=8,
    )
    scored = data.grade(judge=lambda row: {"reward": int(not row["seeded"])})
    report = scored.select(mode="sft").export(str(tmp_path / "sel.jsonl"), format="fireworks")
    assert report["format"] == "fireworks"
    for line in (tmp_path / "sel.jsonl").read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) <= FIREWORKS_SFT_KEYS
