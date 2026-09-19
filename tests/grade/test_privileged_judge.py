"""grade(use_privileged=True): the judge reads the row's privileged block
(principle, reference, hidden state) the agent never saw (Bai et al. 2022,
arXiv:2212.08073)."""

from __future__ import annotations

import json

import whileai.simulations as wai
from whileai.simulations.data import export_row
from whileai.simulations.score import grade_llm as G

SPEC = "vllm:phi@http://127.0.0.1:9/v1"


def _row(with_priv=True):
    row = {
        "prompt": "what is the refund policy",
        "final_text": "Refunds within 30 days.",
        "steps": [],
        "messages": [
            {"role": "user", "content": "what is the refund policy"},
            {"role": "assistant", "content": "Refunds within 30 days."},
        ],
        "scenario_id": "s1",
    }
    if with_priv:
        row["privileged"] = {
            "principle": "Be exact about policy terms.",
            "reference": "Refunds within 14 days, unopened.",
            "hidden_state": {"policy_days": 14},
            "rubric": {"criteria": [{"title": "Exact days"}]},
        }
    return row


def _capture(monkeypatch, seen):
    def fake_complete(_url, _model, messages, **kwargs):
        seen.append({"system": messages[0]["content"], "user": messages[-1]["content"]})
        return {"content": '{"reason": "contradicts the reference", "score": 0}'}

    monkeypatch.setattr(G, "complete", fake_complete)
    monkeypatch.setattr(G, "warm_judge", lambda *a, **k: {"seconds": 0.0})
    monkeypatch.setattr(G, "require_judge_key", lambda *a, **k: SPEC)


def test_privileged_block_reaches_the_judge_only_when_asked(monkeypatch):
    seen: list[dict] = []
    _capture(monkeypatch, seen)
    rows = [_row()]
    wai.grade_llm(rows, spec=SPEC, use_privileged=True)
    user = seen[-1]["user"]
    body = json.loads(user[user.index("{") :])
    assert body["judge_only"]["reference"] == "Refunds within 14 days, unopened."
    assert body["judge_only"]["principle"] == "Be exact about policy terms."
    assert body["judge_only"]["hidden_state"] == {"policy_days": 14}
    assert "rubric" not in body["judge_only"]  # the rubric judge reads that
    assert "judge_only block" in seen[-1]["system"]
    assert rows[0]["reward"] == 0 and rows[0]["judge_meta"]["privileged"] is True

    seen.clear()
    plain = [_row()]
    wai.grade_llm(plain, spec=SPEC)
    user = seen[-1]["user"]
    assert "judge_only" not in user and "judge_only" not in seen[-1]["system"]
    assert "privileged" not in plain[0]["judge_meta"]
    # a judge that read the teacher block is a different judge version
    assert plain[0]["judge_meta"]["version"] != rows[0]["judge_meta"]["version"]


def test_use_privileged_without_a_block_is_the_plain_judge(monkeypatch):
    seen: list[dict] = []
    _capture(monkeypatch, seen)
    rows = [_row(with_priv=False)]
    wai.grade_llm(rows, spec=SPEC, use_privileged=True)
    assert "judge_only" not in seen[-1]["user"]
    assert "privileged" not in export_row(rows[0])


def test_simulation_data_grade_threads_the_flag(monkeypatch):
    seen: list[dict] = []
    _capture(monkeypatch, seen)
    from tests.helpers import simulate_offline

    data = simulate_offline(budget=2, per_round=2, concurrency=1)
    for t in data.trajectories:
        t["privileged"] = {"reference": "ref"}
    data.grade(llm_spec=SPEC, use_privileged=True)
    assert all('"judge_only"' in s["user"] for s in seen)
