"""runs=: the same task set replayed N times in one call, every row stamped
``lineage.eval_run`` (Lambert 2025, chapter Evaluation and its
evaluation-variance appendix: one eval is a draw, three give a standard
deviation)."""

from __future__ import annotations

import json

import pytest

import whileai.simulations as wai
from tests.helpers import simulate_offline


def _run(**kw):
    kw.setdefault("budget", 24)
    return simulate_offline(mode="sft", repeats=2, concurrency=1, seed=3, grade="conduct", **kw)


def _eval_runs(data):
    return sorted({r["lineage"]["eval_run"] for r in data.trajectories})


def test_runs_replays_the_pinned_tasks_and_stamps_eval_run(tmp_path):
    base = _run()
    out = tmp_path / "rep.jsonl"
    rep = _run(tasks=base, runs=3, output=str(out))
    assert len(rep.trajectories) == 3 * len(base.trajectories)
    assert _eval_runs(rep) == [0, 1, 2]
    assert rep.search["eval_runs"]["runs"] == 3
    assert [r["rows"] for r in rep.search["eval_runs"]["per_run"]] == [len(base.trajectories)] * 3
    # every run saw the same tasks, same faults, same world
    base_tasks = {r["scenario_id"] for r in base.trajectories}
    for i in range(3):
        rows = [r for r in rep.trajectories if r["lineage"]["eval_run"] == i]
        assert {r["scenario_id"] for r in rows} == base_tasks
    by_prompt = {r["prompt"]: r for r in base.trajectories}
    for row in rep.trajectories:
        assert row["faults"] == by_prompt[row["prompt"]]["faults"]
        assert row["seed"] == by_prompt[row["prompt"]]["seed"]
    # the base run is untouched
    assert all("lineage" not in r or "eval_run" not in r["lineage"] for r in base.trajectories)
    # output= holds every run, lineage included
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(written) == len(rep.trajectories)
    assert sorted({r["lineage"]["eval_run"] for r in written}) == [0, 1, 2]
    # eval_variance splits by eval_run on its own
    noise = wai.eval_variance(rep.trajectories)
    assert noise["n_runs"] == 3 and noise["run_std"] is not None


def test_runs_without_tasks_draws_once_then_replays():
    rep = _run(runs=2)
    assert _eval_runs(rep) == [0, 1]
    first = {r["scenario_id"] for r in rep.trajectories if r["lineage"]["eval_run"] == 0}
    second = {r["scenario_id"] for r in rep.trajectories if r["lineage"]["eval_run"] == 1}
    assert first == second


def test_runs_one_stamps_nothing_and_bad_values_raise():
    one = _run(runs=1)
    assert all("eval_run" not in (r.get("lineage") or {}) for r in one.trajectories)
    with pytest.raises(ValueError, match="runs="):
        _run(runs=0)
