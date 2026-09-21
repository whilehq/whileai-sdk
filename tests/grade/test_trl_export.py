"""The TRL export shape (issue #152).

TRL is not installed here, so these are structural checks against the two
rules that actually broke a customer's run:

* ``trl.data_utils.is_conversational`` decides by the *column set* — an
  example whose supported keys are anything other than one of TRL's known
  combinations is treated as non-conversational, no chat template is
  applied, and nothing raises. ``_is_conversational`` below mirrors that
  rule; it is a copy of TRL's, not an import, on purpose.
* ``maybe_apply_chat_template`` indexes ``prompt`` as a message list for
  conversational preference data, so a ``prompt`` string with
  conversational sides raises ``TypeError: string indices must be
  integers``. That is a shape a test can assert without TRL.
"""

from __future__ import annotations

import json

import pytest

from whileai.simulations.export import (
    export_preference,
    export_training,
    to_trl,
    tool_call_roundtrip,
    training_rows,
)
from whileai.simulations.score.judging import build_preference_pairs

POLICY = "Read before you write. Say what you did."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

#: The column sets TRL's ``is_conversational`` accepts.
_SUPPORTED = {"prompt", "chosen", "rejected", "completion", "messages"}
_SHAPES = [
    {"messages"},
    {"prompt"},
    {"prompt", "completion"},
    {"prompt", "chosen", "rejected"},
    {"chosen", "rejected"},
]


def _is_conversational(example: dict) -> bool:
    keys = {k for k in example if k in _SUPPORTED}
    if keys not in _SHAPES:
        return False
    key = "messages" if "messages" in keys else sorted(keys)[0]
    turns = example[key]
    return (
        isinstance(turns, list)
        and bool(turns)
        and isinstance(turns[0], dict)
        and "role" in turns[0]
        and "content" in turns[0]
    )


def _row(prompt: str, reward: int, final: str, path: str = "src/app/handlers.py") -> dict:
    return {
        "prompt": prompt,
        "reward": reward,
        "judge_status": "ok",
        "model_version": "student",
        "steps": [
            {"tool": "read_file", "arguments": {"path": path}, "result": {"ok": True}},
        ],
        "final_text": final,
    }


def _rows() -> list[dict]:
    return [
        _row("read the handler file", 1, "Read it; the bug is on line 40."),
        _row("read the handler file", 0, "I could not find anything."),
    ]


def _calls(messages) -> list[dict]:
    return [c for m in messages for c in (m.get("tool_calls") or [])]


# ---------------------------------------------------------------- sft shape


def test_default_sft_rows_keep_the_openai_wire_shape():
    """The default is unchanged: prompt string, arguments as JSON string."""
    row = training_rows(_rows()[:1], system_prompt=POLICY, tools=TOOLS)[0]
    assert isinstance(row["prompt"], str)
    assert isinstance(_calls(row["messages"])[0]["function"]["arguments"], str)


def test_trl_sft_rows_drop_the_prompt_string_column(tmp_path):
    """Bug 2: a ``prompt`` string beside ``messages`` makes TRL skip the
    chat template and train on the bare ask, silently."""
    out = tmp_path / "sft.jsonl"
    report = export_training(_rows(), str(out), system_prompt=POLICY, tools=TOOLS, format="trl")
    assert report["format"] == "trl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows
    for row in rows:
        assert "prompt" not in row, "a prompt string here breaks TRL's sniffing"
        assert _is_conversational(row)
        # the ask is not lost, only renamed out of TRL's way
        assert row["prompt_text"] == "read the handler file"
        assert row["messages"][0] == {"role": "system", "content": POLICY}


def test_default_sft_rows_are_not_conversational_to_trl():
    """The regression the default shape hides: same rows, no template."""
    row = training_rows(_rows()[:1], system_prompt=POLICY, tools=TOOLS)[0]
    assert not _is_conversational(row)


# --------------------------------------------------------------- dpo shape


def test_trl_preference_rows_are_prompt_plus_completions(tmp_path):
    """Bug 1: prompt must be a message list and the sides the completion.

    Both rollouts open with the same ``read_file`` call and result, so that
    prefix is the prompt and each side is the one reply that differs."""
    pairs, _ = build_preference_pairs(_rows())
    assert pairs
    out = tmp_path / "dpo.jsonl"
    report = export_preference(pairs, str(out), system_prompt=POLICY, format="trl")
    assert report["format"] == "trl"
    row = json.loads(out.read_text().splitlines()[0])
    assert isinstance(row["prompt"], list)
    assert [m["role"] for m in row["prompt"]] == ["system", "user", "assistant", "tool"]
    # one assistant turn per side: no prompt turns repeated, no tool result scored
    for side in ("chosen", "rejected"):
        assert [m["role"] for m in row[side]] == ["assistant"]
    assert _is_conversational(row)
    assert row["chosen"][-1]["content"] == "Read it; the bug is on line 40."
    assert row["rejected"][-1]["content"] == "I could not find anything."
    assert row["margin"] == 1.0, "pair metadata survives the reshape"


def test_default_preference_rows_are_the_shape_trl_chokes_on(tmp_path):
    """The default is unchanged, and this records why it is not TRL's."""
    pairs, _ = build_preference_pairs(_rows())
    out = tmp_path / "dpo.jsonl"
    report = export_preference(pairs, str(out), system_prompt=POLICY)
    assert report["format"] == "openai"
    row = json.loads(out.read_text().splitlines()[0])
    assert isinstance(row["prompt"], str)
    assert row["chosen"][0]["role"] == "system", "prompt turns repeated on the side"
    # TRL sniffs this as conversational from the column set and then indexes
    # the prompt as a message list: TypeError, string indices must be integers.
    assert _is_conversational(row) and not isinstance(row["prompt"], list)


def test_preference_pairs_without_a_completion_are_dropped_not_written():
    """A side with no assistant turn is no preference, so it is counted."""
    pair = {
        "prompt": "ask",
        "chosen": {"messages": [{"role": "user", "content": "ask"}]},
        "rejected": {"messages": [{"role": "user", "content": "ask"}]},
    }
    with pytest.raises(ValueError, match="no_preference_pairs"):
        export_preference([pair], format="trl")
    report = export_preference([pair], format="trl", validate=False)
    assert report["pairs"] == 0
    assert report["no_completion_dropped"] == 1


# ------------------------------------- one turn per side, later turns cut


def _pair(chosen_turns, rejected_turns, **extra) -> dict:
    ask = [{"role": "user", "content": "where is order 4412"}]
    return {
        "prompt": "where is order 4412",
        "chosen": {"messages": ask + chosen_turns},
        "rejected": {"messages": ask + rejected_turns},
        **extra,
    }


def _lookup(order_id: str) -> dict:
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


def _shipped(content: str = '{"status": "shipped"}') -> dict:
    return {"role": "tool", "tool_call_id": "c1", "content": content}


def _pairs_as_rows(pairs: list[dict]) -> list[dict]:
    """The rows ``to_trl`` takes: both sides as message lists."""
    return [
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"]["messages"],
            "rejected": p["rejected"]["messages"],
        }
        for p in pairs
    ]


def test_trl_preference_sides_are_the_one_assistant_turn_where_they_diverge(tmp_path):
    """A DPO trainer masks the prompt and sums log-probabilities over
    every completion token (Lambert 2025, chapter Direct Alignment), so a
    tool result or a later user turn on a side carries gradient. The
    shared prefix, tool result and second ask included, is the prompt;
    each side is the one assistant turn where the pair differs."""
    pair = _pair(
        [
            _lookup("4412"),
            _shipped(),
            {"role": "user", "content": "and order 8?"},
            {"role": "assistant", "content": "Shipped."},
        ],
        [
            _lookup("4412"),
            _shipped(),
            {"role": "user", "content": "and order 8?"},
            {"role": "assistant", "content": "I cannot tell."},
        ],
    )
    out = tmp_path / "dpo.jsonl"
    report = export_preference([pair], str(out), system_prompt=POLICY, format="trl")
    row = json.loads(out.read_text().splitlines()[0])
    assert [m["role"] for m in row["prompt"]] == ["system", "user", "assistant", "tool", "user"]
    assert row["prompt"][2]["tool_calls"][0]["function"]["arguments"] == {"order_id": "4412"}
    assert row["chosen"] == [{"role": "assistant", "content": "Shipped."}]
    assert row["rejected"] == [{"role": "assistant", "content": "I cannot tell."}]
    for side in ("chosen", "rejected"):
        assert all(m["role"] == "assistant" for m in row[side])
    assert _is_conversational(row)
    assert report["pairs"] == 1
    assert "trl_turns_cut" not in report, "nothing was cut, so the key is absent"


def test_trl_preference_report_counts_pairs_that_lost_later_turns(tmp_path):
    """The sides diverge at the opening call; what follows on the chosen
    side is cut, and the report says so, as ``fireworks_turns_cut`` does."""
    pairs = [
        _pair(
            [_lookup("4412"), _shipped(), {"role": "assistant", "content": "Shipped."}],
            [{"role": "assistant", "content": "No idea."}],
        ),
        _pair(
            [{"role": "assistant", "content": "Shipped."}],
            [{"role": "assistant", "content": "No idea."}],
        ),
    ]
    out = tmp_path / "dpo.jsonl"
    report = export_preference([*pairs], str(out), system_prompt=POLICY, format="trl")
    assert report["pairs"] == 2
    assert report["trl_turns_cut"] == 1
    first, second = [json.loads(s) for s in out.read_text().splitlines()]
    assert [m["role"] for m in first["prompt"]] == ["system", "user"]
    assert len(first["chosen"]) == len(first["rejected"]) == 1
    assert first["chosen"][0]["tool_calls"][0]["function"]["name"] == "lookup_order"
    assert first["rejected"][0]["content"] == "No idea."
    assert second["chosen"] == [{"role": "assistant", "content": "Shipped."}]


def test_trl_preference_pairs_with_no_one_turn_contrast_are_dropped_and_counted(tmp_path):
    """Diverging on a tool result is not a preference the policy can be
    trained on, and sides that never differ carry no contrast at all."""
    pairs = [
        _pair(
            [_lookup("4412"), _shipped()],
            [_lookup("4412"), _shipped('{"status": "lost"}')],
        ),
        _pair(
            [_lookup("4412"), _shipped(), {"role": "assistant", "content": "Shipped."}],
            [_lookup("4412"), _shipped(), {"role": "assistant", "content": "Shipped."}],
        ),
        _pair(
            [_lookup("4412"), _shipped(), {"role": "assistant", "content": "Shipped."}],
            [_lookup("4412")],
        ),
    ]
    out = tmp_path / "dpo.jsonl"
    report = export_preference(pairs, str(out), system_prompt=POLICY, format="trl", validate=False)
    assert report["pairs"] == 0
    assert report["no_completion_dropped"] == 3
    assert "trl_turns_cut" not in report
    assert to_trl(_pairs_as_rows(pairs), "preference") == []


# --------------------------------------------------- tool-call arguments


def test_trl_tool_call_arguments_are_dicts_not_json_strings():
    """Bug 3: HF chat templates render arguments with ``| tojson``."""
    rows = to_trl(training_rows(_rows(), system_prompt=POLICY, tools=TOOLS), "training")
    call = _calls(rows[0]["messages"])[0]
    assert call["function"]["arguments"] == {"path": "src/app/handlers.py"}
    assert call["type"] == "function" and call["id"].startswith("call_")


def test_roundtrip_gate_names_the_encoding_it_checked():
    """Bug 3's second half: the gate used to vouch for the trainer path it
    never checked. ``invalid: 0`` now comes with the encoding."""
    wire = training_rows(_rows(), system_prompt=POLICY, tools=TOOLS)
    trl = to_trl(wire, "training")

    openai_gate = tool_call_roundtrip(wire)
    assert openai_gate["encoding"] == "json_string"
    assert openai_gate["invalid"] == 0

    # The same rows, checked as a chat-template trainer needs them: the
    # JSON strings are exactly what would be double-encoded.
    as_dicts = tool_call_roundtrip(wire, format="trl")
    assert as_dicts["encoding"] == "dict"
    assert as_dicts["invalid"] == as_dicts["checked"] > 0

    trl_gate = tool_call_roundtrip(trl, format="trl")
    assert (trl_gate["encoding"], trl_gate["invalid"]) == ("dict", 0)
    assert trl_gate["checked"] == openai_gate["checked"]


def test_unknown_format_is_refused():
    for call in (
        lambda: export_training(_rows(), format="hf"),
        lambda: export_preference([], format="hf"),
        lambda: tool_call_roundtrip([], format="hf"),
        lambda: to_trl([], "sft"),
    ):
        with pytest.raises(ValueError):
            call()


# ------------------------------------------------------ the loss mask (#507)


def _multi_turn_row() -> dict:
    """Two assistant turns around a tool result: the shape whose mask TRL
    cannot honor from a ``messages`` row."""
    return {
        "prompt": "where is order 4412",
        "reward": 1,
        "messages": [
            {"role": "user", "content": "where is order 4412"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "orders/4412"}}}
                ],
            },
            {"role": "tool", "content": '{"status": "shipped"}'},
            {"role": "assistant", "content": "It shipped."},
        ],
    }


def test_trl_sft_rows_carry_no_loss_mask_and_the_report_says_what_trl_does(tmp_path):
    """trl 0.19.1's SFTTrainer never reads a ``loss_mask`` column; a file
    that carries one reads as a promise it does not keep. The report's
    ``mask_mode`` names the trainer's behaviour, not the SDK's intention."""
    out = tmp_path / "sft.jsonl"
    report = export_training(
        [_multi_turn_row()], str(out), system_prompt=POLICY, tools=TOOLS, format="trl"
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all("loss_mask" not in row for row in rows)
    assert all(_is_conversational(row) for row in rows)
    assert report["mask_mode"].startswith("TRL trains on every token")
    assert "assistant_only_loss=True" in report["mask_mode"]
    assert "{% generation %}" in report["mask_mode"]
    # every message is trained, and the report counts it that way
    assert report["trained_messages"] == 5 and report["masked_messages"] == 0


def test_openai_rows_keep_the_loss_mask_and_the_intended_mask_mode(tmp_path):
    """The other format is unchanged: the mask is the row's, and the report
    says what was asked."""
    out = tmp_path / "sft.jsonl"
    report = export_training([_multi_turn_row()], str(out), system_prompt=POLICY, tools=TOOLS)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["loss_mask"] == [0, 0, 1, 0, 1]
    assert report["mask_mode"] == "assistant"
    assert (report["trained_messages"], report["masked_messages"]) == (2, 3)


def test_trl_final_mask_is_a_prompt_completion_row_that_trl_honors(tmp_path):
    """``mask_mode="final"`` is the one mask trl 0.19.1 can apply from the
    column set: prompt/completion rows, from which its ``tokenize`` builds a
    ``completion_mask`` and the collator labels the completion only."""
    out = tmp_path / "sft.jsonl"
    report = export_training(
        [_multi_turn_row()],
        str(out),
        system_prompt=POLICY,
        tools=TOOLS,
        format="trl",
        mask_mode="final",
    )
    row = json.loads(out.read_text().splitlines()[0])
    assert set(row) & {"messages", "loss_mask", "prompt_text"} == {"prompt_text"}
    assert [m["role"] for m in row["prompt"]] == ["system", "user", "assistant", "tool"]
    assert row["completion"] == [{"role": "assistant", "content": "It shipped."}]
    assert _is_conversational(row)
    # the tool call in the prompt is still checked, as a dict
    call = _calls(row["prompt"])[0]
    assert call["function"]["arguments"] == {"path": "orders/4412"}
    assert report["tool_call_roundtrip"] == {
        **report["tool_call_roundtrip"],
        "checked": 1,
        "invalid": 0,
        "encoding": "dict",
    }
    assert report["mask_mode"].startswith("TRL trains on the last assistant turn only")
    assert "completion_mask" in report["mask_mode"]
    assert (report["trained_messages"], report["masked_messages"]) == (1, 4)
    assert report["with_system"] == 1


def test_trl_unroll_is_one_prompt_completion_row_per_assistant_turn(tmp_path):
    out = tmp_path / "sft.jsonl"
    report = export_training(
        [_multi_turn_row()], str(out), system_prompt=POLICY, format="trl", unroll=True
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert report["n"] == len(rows) == 2
    assert [len(r["prompt"]) for r in rows] == [2, 4]
    assert all(
        len(r["completion"]) == 1 and r["completion"][0]["role"] == "assistant" for r in rows
    )
    assert all("loss_mask" not in r for r in rows)
    assert report["mask_mode"].startswith("TRL trains on the last assistant turn only")


def test_trl_final_drops_and_counts_a_row_with_no_assistant_turn(tmp_path):
    row = _multi_turn_row()
    row["messages"] = [{"role": "user", "content": "hello?"}]
    out = tmp_path / "sft.jsonl"
    report = export_training(
        [row, _multi_turn_row()], str(out), format="trl", mask_mode="final", validate=False
    )
    assert report["n"] == 1 and report["no_completion_dropped"] == 1
    assert any("no assistant turn" in w for w in report["warnings"])
    assert len(to_trl(training_rows([row]), "completion")) == 0
