"""``ScoredData.push`` is ``push_rows`` on the graded copies."""

from __future__ import annotations

from tests.helpers import simulate_offline
from whileai.simulations.score import judging


def test_scored_data_push_is_push_rows_on_the_graded_copies(monkeypatch):
    data = simulate_offline(budget=4, concurrency=1)
    scored = data.grade(judge=lambda row: {"reward": 1})
    seen = {}

    def fake_push_rows(rows, name, **kwargs):
        seen["rows"], seen["name"], seen["kwargs"] = rows, name, kwargs
        return {"datasetId": "ds_test"}

    monkeypatch.setattr("whileai.simulations.ingest.platform.push_rows", fake_push_rows)
    out = scored.push("demo-rl", gate=True, mode="rl", agent="demo")
    assert out == {"datasetId": "ds_test"}
    assert seen["rows"] is scored.rows
    assert seen["name"] == "demo-rl"
    assert seen["kwargs"] == {"gate": True, "mode": "rl", "agent": "demo"}
    assert isinstance(scored, judging.ScoredData)
