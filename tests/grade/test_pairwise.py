"""Pairwise judging with position swap and ties (Lambert 2025, chapters
Reward Modeling and Preference Data)."""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations.export import export_preference
from whileai.simulations.score.pairwise import judge_pairs, parse_pairwise


def _row(prompt, final, reward, model="qwen"):
    return {
        "prompt": prompt,
        "final_text": final,
        "reward": reward,
        "model_version": model,
        "steps": [{"tool": "get_order", "arguments": {"id": "1"}, "result": {"ok": 1}}],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final},
        ],
    }


def _pair(prompt, chosen_text, rejected_text):
    return {
        "prompt": prompt,
        "chosen": _row(prompt, chosen_text, 1),
        "rejected": _row(prompt, rejected_text, 0),
        "chosen_score": 1.0,
        "rejected_score": 0.0,
        "margin": 1.0,
    }


def _content_judge(a, b):
    """Prefers the reply that says 'shipped'; ties when both or neither."""
    sa, sb = "shipped" in a["final_text"], "shipped" in b["final_text"]
    if sa == sb:
        return {"winner": "tie", "reason": "same"}
    return {"winner": "A" if sa else "B", "reason": "says shipped"}


def _first_position_judge(a, b):
    return '{"winner": "A", "reason": "the first one"}'


def test_parse_pairwise_reads_json_and_bare_forms():
    assert parse_pairwise('{"winner": "B", "reason": "x"}') == ("B", "x")
    assert parse_pairwise('Verdict: {"winner":"tie"}')[0] == "tie"
    assert parse_pairwise('winner: "a"')[0] == "A"
    assert parse_pairwise("no idea") == (None, "")


def test_judge_pairs_swaps_and_marks_winner_tie_and_disagreement():
    pairs = [
        _pair("p1", "Order shipped.", "I don't know."),  # agrees with scores
        _pair("p2", "I don't know.", "Order shipped."),  # judge prefers rejected
        _pair("p3", "Order shipped.", "It shipped."),  # tie
    ]
    out, report = judge_pairs(pairs, _content_judge, concurrency=1)
    assert out is not None and [p["pairwise"]["winner"] for p in out] == [
        "chosen",
        "rejected",
        "tie",
    ]
    assert [p["tie"] for p in out] == [False, False, True]
    assert all(p["pairwise"]["position_consistent"] is True for p in out)
    assert all(p["pairwise"]["swapped"] for p in out)
    assert report["n_judged"] == 3 and report["failed"] == 0
    assert report["position_flip_rate"] == 0.0 and report["tie_rate"] == pytest.approx(
        1 / 3, abs=1e-3
    )
    assert report["agrees_with_scores"] == pytest.approx(1 / 3, abs=1e-3)
    assert report["prefers_rejected"] == 1 and report["disagreements"][0]["prompt"] == "p2"
    assert report["judge"] == "_content_judge"
    assert any("prefers the rejected side" in w for w in report["warnings"])


def test_position_bias_becomes_a_tie_and_a_warning():
    pairs = [_pair(f"p{i}", "Order shipped.", "I don't know.") for i in range(4)]
    out, report = judge_pairs(pairs, _first_position_judge, concurrency=2)
    assert all(p["pairwise"]["winner"] == "tie" for p in out)
    assert all(p["pairwise"]["position_consistent"] is False for p in out)
    assert report["position_flip_rate"] == 1.0 and report["tie_rate"] == 1.0
    assert any("position bias" in w for w in report["warnings"])
    # without the swap the same judge looks decisive
    out, report = judge_pairs(pairs, _first_position_judge, swap=False, concurrency=1)
    assert all(p["pairwise"]["winner"] == "chosen" for p in out)
    assert report["position_flip_rate"] is None and report["swap"] is False


def test_judge_failures_and_bad_input():
    def broken(a, b):
        raise RuntimeError("boom")

    out, report = judge_pairs([_pair("p", "a", "b")], broken, concurrency=1)
    assert out[0]["pairwise"]["winner"] is None and out[0]["tie"] is False
    assert report["failed"] == 1 and report["n_judged"] == 0
    assert "boom" in out[0]["pairwise"]["reasons"][0]
    with pytest.raises(ValueError, match="not pairs"):
        judge_pairs([{"prompt": "x"}], _content_judge)


def test_export_preference_drops_ties_by_default(tmp_path):
    pairs = [
        _pair("p1", "Order shipped.", "I don't know."),
        _pair("p3", "Order shipped.", "It shipped."),
    ]
    judge_pairs(pairs, _content_judge, concurrency=1)
    report = export_preference(pairs, str(tmp_path / "pairs.jsonl"))
    assert report["pairs"] == 1 and report["ties_dropped"] == 1
    kept = export_preference(pairs, str(tmp_path / "all.jsonl"), drop_ties=False)
    assert kept["pairs"] == 2 and "ties_dropped" not in kept
    line = (tmp_path / "all.jsonl").read_text().splitlines()[1]
    assert '"tie": true' in line and '"pairwise"' in line


def test_public_surface():
    assert "judge_pairs" in wai.__all__ and "pairwise_judge" in wai.__all__
    judge = wai.pairwise_judge("vllm:phi@http://127.0.0.1:9/v1")
    assert judge.__name__.startswith("phi@")
    # unreachable endpoint: a failed verdict, never a raise
    out = judge(_row("p", "a", 1), _row("p", "b", 0))
    assert out["winner"] is None and out["reason"]
