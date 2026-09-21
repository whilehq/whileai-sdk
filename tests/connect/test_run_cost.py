"""Issue #399: a hosted run's record says what it cost, as an estimate.

``seconds`` and ``gpu`` were on the record; the rate was not. Now
``cost_usd`` is ``seconds / 3600 * GPU_USD_PER_HOUR[gpu]`` to the cent and
``cost_basis`` names the GPU, the rate and the day Modal's list price was
read. A GPU not in the table is None with a sentence, never a silent gap.
"""

from __future__ import annotations

import whileai.simulations.training as tr
from whileai.simulations.defaults import GPU_PRICE_SOURCE, GPU_USD_PER_HOUR
from whileai.simulations.training import TrainingRun, _cost_fields


def test_a10g_for_56_seconds_is_two_cents():
    fields = _cost_fields("A10G", 56.4)
    assert fields["cost_usd"] == 0.02
    assert fields["cost_basis"] == f"estimate: A10G at $1.10/h, {GPU_PRICE_SOURCE}"
    assert GPU_PRICE_SOURCE == "modal.com/pricing 2026-09-20"
    # the issue's second run, and the recipe's two rows
    assert _cost_fields("L40S", 329.5)["cost_usd"] == 0.18
    # a lower-case name finds the same rate
    assert _cost_fields("a10g", 56.4)["cost_usd"] == 0.02


def test_unknown_gpu_is_none_with_the_reason():
    fields = _cost_fields("TPUv5", 56.4)
    assert fields["cost_usd"] is None
    assert "TPUv5 is not in the rate table" in fields["cost_basis"]
    for name in GPU_USD_PER_HOUR:
        assert name in fields["cost_basis"]
    assert GPU_PRICE_SOURCE in fields["cost_basis"]
    no_gpu = _cost_fields(None, 56.4)
    assert no_gpu["cost_usd"] is None and "no gpu" in no_gpu["cost_basis"]
    no_seconds = _cost_fields("A10G", None)
    assert no_seconds["cost_usd"] is None and "no seconds" in no_seconds["cost_basis"]


def test_printed_run_shows_the_cost_line():
    run = TrainingRun("run_h1", name="ds_x · sft")
    run._hosted = True
    run._absorb({"status": "done", "gpu": "A10G", "seconds": 56.4, "before": 0.2, "after": 0.3})
    assert run.training["cost_usd"] == 0.02
    assert run.training["cost_basis"].startswith("estimate: A10G at $1.10/h")
    text = str(run)
    assert "about $0.02 (A10G, 56 s, estimate)" in text
    assert "holdout pass 0.2 -> 0.3" in text
    assert "rollouts and judge calls on the serving endpoint are not priced" in text


def test_printed_run_says_why_when_the_gpu_is_unknown():
    run = TrainingRun("run_h2", name="ds_x · grpo")
    run._absorb({"status": "done", "gpu": "TPUv5", "seconds": 329.5})
    assert run.training["cost_usd"] is None
    assert "cost: no estimate: TPUv5 is not in the rate table" in str(run)


def test_get_run_adds_cost_to_the_summary(monkeypatch):
    def transport(method, path, api_key=None, body=None, **kw):
        assert (method, path) == ("GET", "/runs/run_h1")
        return {"id": "run_h1", "status": "done", "summary": {"gpu": "L40S", "seconds": 329.5}}

    monkeypatch.setattr(tr, "_call", transport)
    out = tr.get_run("run_h1")
    assert out["summary"]["cost_usd"] == 0.18
    assert out["summary"]["cost_basis"] == f"estimate: L40S at $1.95/h, {GPU_PRICE_SOURCE}"
    # a record with no gpu and no seconds is left alone
    monkeypatch.setattr(tr, "_call", lambda *a, **k: {"id": "run_x", "summary": {"train_loss": 1}})
    assert "cost_usd" not in tr.get_run("run_x")["summary"]
