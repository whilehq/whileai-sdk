"""Cold-start tester run, 2026-09-17: a coding agent told to "use zp to
improve the evals". It looked for a `zp` command, lost two of its ten
seeds without being told, and could not iterate the run it got back.
"""

from __future__ import annotations

from pathlib import Path

import whileai.simulations as wai
from tests.helpers import simulate_offline

REPO_ROOT = Path(__file__).resolve().parents[2]

SEEDS = [f"refund order A10{i:02d} please, item {i} arrived broken" for i in range(10)]


def _asks(data) -> set[str]:
    return {r["prompt"] for r in data.trajectories}


# ------------------------------------------------------- the zp command


def test_wai_is_the_command_and_whileai_still_runs_it():
    # tomllib is 3.11+, and the test suite runs on 3.10 too. `zp` was
    # retired on 2026-09-21 ("zp is deprecated. wai.").
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.scripts]", 1)[1].split("[", 1)[0]
    for name in ("wai", "whileai"):
        assert f'{name} = "whileai.cli:main"' in block
    assert "zp = " not in block


# ------------------------------------------------------- every seed runs


def test_situations_below_the_seed_count_evicts_nothing():
    # situations=3 with ten seeds used to run two of them: the search
    # filled its quota with its own asks and the rest never went out.
    data = simulate_offline(seeds=SEEDS, budget=40, repeats=4, mode="rl", situations=3)
    ran = _asks(data)
    assert all(seed in ran for seed in SEEDS)
    assert data.n_situations == len(SEEDS)
    assert data.warnings == []


def test_seeds_run_when_situations_is_unset():
    data = simulate_offline(seeds=SEEDS, budget=40, repeats=4, mode="rl")
    assert all(seed in _asks(data) for seed in SEEDS)


def test_a_budget_too_small_for_the_seeds_says_so_before_the_run():
    data = simulate_offline(seeds=SEEDS, budget=12, repeats=4, mode="rl")
    assert data.warnings == [
        "budget=12 covers 3 of 10 seeds at repeats=4; raise budget to 40+ or drop seeds"
    ]
    dropped = data.search["seeds_dropped"]
    assert dropped == SEEDS[3:]
    ran = _asks(data)
    # the seeds the note kept are the seeds that ran; the dropped ones
    # are not half-run behind the caller's back
    assert all(seed in ran for seed in SEEDS[:3])
    assert not any(seed in ran for seed in dropped)


# ------------------------------------------------------- API papercuts


def test_a_run_iterates_like_scored_data():
    data = simulate_offline(seeds=SEEDS[:2], budget=4, repeats=2, mode="rl")
    assert len(data) == len(data.trajectories)
    assert list(data) == data.trajectories
    assert [r["prompt"] for r in data] == [r["prompt"] for r in data.trajectories]


def test_extra_judge_keys_are_readable_under_judge_meta():
    rows = [{"prompt": "refund A1001", "final_text": "done", "steps": []}]
    scored = wai.run_judge(
        rows,
        lambda row: {"reward": 1.0, "reason": "ok", "failures": ["missed the lookup"]},
    )
    assert scored.rows[0]["judge_meta"]["failures"] == ["missed the lookup"]
    assert "failures" not in {k for k in scored.rows[0] if k != "judge_meta"}
