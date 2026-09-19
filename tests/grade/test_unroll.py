"""training_rows(unroll=True): an N-turn conversation as N samples (Lambert
2025, chapter Instruction Tuning)."""

from __future__ import annotations

from whileai.simulations.export import export_training, training_rows

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


def _row(prompt="where is order 4412", reward=1, rollout_index=0):
    return {
        "prompt": prompt,
        "scenario_id": "s1",
        "rollout_index": rollout_index,
        "reward": reward,
        "final_text": "It shipped yesterday; tracking sent.",
        "steps": [],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Let me check that order."},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "It shipped yesterday."},
            {"role": "user", "content": "tracking?"},
            {"role": "assistant", "content": "It shipped yesterday; tracking sent."},
        ],
    }


def test_unroll_makes_one_sample_per_assistant_turn_with_loss_on_that_turn():
    rows = training_rows([_row()], system_prompt="be brief", tools=TOOLS, unroll=True)
    assert len(rows) == 3
    for k, sample in enumerate(rows, 1):
        assert sample["unroll"] == {"turn": k, "turns": 3}
        assert sample["messages"][0]["role"] == "system"
        assert sample["messages"][-1]["role"] == "assistant"
        assert sum(sample["loss_mask"]) == 1 and sample["loss_mask"][-1] == 1
        assert len(sample["loss_mask"]) == len(sample["messages"])
        assert sample["lineage"]["unrolled_from"]["scenario_id"] == "s1"
        assert sample["lineage"]["unrolled_from"]["rollout_index"] == 0
        assert sample["reward"] == 1 and sample["prompt"] == "where is order 4412"
    # the k-th sample ends at the k-th assistant turn: earlier context only
    assert [len(s["messages"]) for s in rows] == [3, 5, 7]
    assert rows[0]["messages"][-1]["content"] == "Let me check that order."
    # samples of one conversation are not a GRPO group
    assert all("group_id" not in s and "k" not in s for s in rows)


def test_unroll_off_keeps_one_row_and_group_stamps():
    rows = training_rows([_row(), _row(reward=0, rollout_index=1)], tools=TOOLS)
    assert len(rows) == 2 and all(r["k"] == 2 for r in rows)
    assert rows[0]["loss_mask"].count(1) == 3
    final = training_rows([_row()], tools=TOOLS, mask_mode="final")
    assert final[0]["loss_mask"].count(1) == 1


def test_unroll_without_assistant_turns_is_one_sample():
    row = _row()
    row["messages"] = [{"role": "user", "content": "hello?"}]
    rows = training_rows([row], unroll=True)
    assert len(rows) == 1 and "unroll" not in rows[0] and rows[0]["loss_mask"] == [0]


def test_export_training_reports_unroll(tmp_path):
    report = export_training([_row()], str(tmp_path / "t.jsonl"), tools=TOOLS, unroll=True)
    assert report["n"] == 3 and report["unrolled"] is True
    assert report["mask_mode"] == "final (unrolled)" and report["trained_messages"] == 3
    assert report["groups"] == 0 and report["n_written"] == 3
    plain = export_training([_row()], str(tmp_path / "p.jsonl"), tools=TOOLS)
    assert plain["n"] == 1 and plain["unrolled"] is False and plain["trained_messages"] == 3
