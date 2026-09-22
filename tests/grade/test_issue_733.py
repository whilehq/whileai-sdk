"""#733: the sizing line in ``delta_report`` is computed from the paired
sd measured on the rows it holds, not the independent-arms model."""

from __future__ import annotations

import random

from whileai.simulations.score.delta import delta_report
from whileai.simulations.score.stats import holdout_size

K = 4


def _rows(rates: dict[str, float], rng: random.Random) -> list[dict]:
    out = []
    for task, p in rates.items():
        for i in range(K):
            out.append(
                {
                    "scenario_id": task,
                    "rollout_index": i,
                    "prompt": f"p{task}",
                    "final_text": f"x{i}",
                    "reward": 1 if rng.random() < p else 0,
                    "judge_status": "ok",
                }
            )
    return out


def test_tasks_needed_is_sized_from_the_paired_sd_on_the_rows():
    rng = random.Random(733)
    # a spread of task difficulties: the same tasks on both sides, so the
    # paired difference is far tighter than two independent draws
    rates = {f"t{i}": rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]) for i in range(80)}
    before = _rows(rates, rng)
    after = [dict(r) for r in before]
    for r in [r for r in after if r["reward"] == 0][:3]:  # a small gain: inside the interval
        r["reward"] = 1
    report = delta_report(before, after, target="pass_at_1", n_boot=200)
    assert report["target_verdict"] == "no_change_detected"
    delta = report["target_delta"]
    assert 0 < delta < 0.05
    measured = holdout_size(delta, before=before, after=after)
    modelled = holdout_size(delta, base=report["metrics"]["pass_at_1"]["mean_a"], k=K)
    assert measured["sd_source"] == "rows" and measured["n_paired"] == 80
    assert measured["n_tasks"] < modelled["n_tasks"] / 2
    assert report["tasks_needed"] == measured["n_tasks"]
    assert report["tasks_needed_source"] == "rows"
    line = next(w for w in report["warnings"] if "you need about" in w)
    assert f"you need about {measured['n_tasks']} tasks" in line
    assert "task sd measured on the 80 paired tasks here" in line


def test_tasks_needed_falls_back_to_the_model_and_says_so():
    rng = random.Random(7)
    rates = {f"t{i}": 0.5 for i in range(40)}
    before = _rows(rates, rng)
    after = [dict(r) for r in before]
    next(r for r in after if r["reward"] == 0)["reward"] = 1
    # ceiling_pass_rate below the base: holdout_size answers with the model
    report = delta_report(before, after, target="pass_at_1", n_boot=200, ceiling_pass_rate=0.3)
    assert report["tasks_needed"] is not None
    assert report["tasks_needed_source"] == "model"
    line = next(w for w in report["warnings"] if "you need about" in w)
    assert "task sd from the binomial model, not measured" in line
