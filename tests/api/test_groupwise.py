"""wai.GroupwiseGrading: the GAR math (equation 3), the GRS product (equation 2),
the spread check, the TRL reward shape and the fallbacks. Nothing here calls a
model; the graders are scripted."""

from __future__ import annotations

import math

import pytest

import whileai as wai
from whileai.groupwise import (
    GroupwiseGrading,
    SpreadReport,
    factors_from_ranking,
    redistribute,
    spread,
)
from whileai.simulations import defaults


def _close(a, b, tol=1e-9):
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


# --- redistribute: equation 3 --------------------------------------------------


def test_redistribute_conserves_positive_advantage_and_zero_mean():
    rewards = [1, 1, 1, 0, 0, 0, 0, 0]
    factors = {0: 1.0, 1: 0.75, 2: 0.5}
    before = [r - sum(rewards) / 8 for r in rewards]
    after = redistribute(rewards, factors)
    assert math.isclose(sum(after), 0.0, abs_tol=1e-12)  # zero group mean
    assert math.isclose(sum(after[:3]), sum(before[:3]))  # passes' mass conserved
    assert _close(after[3:], before[3:])  # failures untouched
    assert after[0] > after[1] > after[2] > 0  # ordered by quality
    assert after[0] > before[0] and after[2] < before[2]  # moved from worse to better


def test_redistribute_equal_factors_is_the_identity():
    rewards = [1, 0, 1, 0]
    assert _close(redistribute(rewards, {0: 0.6, 2: 0.6}), [r - 0.5 for r in rewards])


def test_redistribute_cap_binds_and_the_mean_is_subtracted_again():
    rewards = [1, 1, 0, 0]
    factors = {0: 1.0, 1: 0.01}  # lambda uncapped = 2 / 1.01 = 1.98; cap at 1.5
    out = redistribute(rewards, factors, cap=1.5)
    assert math.isclose(sum(out), 0.0, abs_tol=1e-12)
    uncapped = redistribute(rewards, factors, cap=100.0)
    assert sum(out[:2]) < sum(uncapped[:2])  # the cap left mass on the table
    assert math.isclose(sum(uncapped[:2]), 1.0)  # uncapped: conserved (2 x 0.5)
    with pytest.raises(ValueError, match="cap must be at least 1"):
        redistribute(rewards, factors, cap=0.5)


def test_redistribute_edge_groups_return_plain_advantages():
    assert redistribute([], {}) == []
    assert redistribute([1, 1, 1], {0: 1.0, 1: 0.5, 2: 0.2}) == [0.0, 0.0, 0.0]  # all pass
    assert redistribute([0, 0, 0], {}) == [0.0, 0.0, 0.0]  # all fail
    with pytest.raises(ValueError, match="no quality factor"):
        redistribute([1, 0], {})
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        redistribute([1, 0], {0: 1.5})


def test_redistribute_takes_a_sequence_of_factors():
    assert _close(
        redistribute([1, 0, 1, 0], [1.0, 0.0, 0.5, 0.0]),
        redistribute([1, 0, 1, 0], {0: 1.0, 2: 0.5}),
    )


def test_factors_from_ranking_is_linear_with_ties():
    f = factors_from_ranking([3, 0, 5])
    assert f == {3: 1.0, 0: 0.75, 5: 0.5}
    tied = factors_from_ranking([[3, 0], 5])
    assert tied[3] == tied[0] == 0.875 and tied[5] == 0.5
    assert factors_from_ranking([7]) == {7: 1.0}
    assert factors_from_ranking([]) == {}
    assert factors_from_ranking([1, 2], min_factor=1.0) == {1: 1.0, 2: 1.0}


# --- the object ------------------------------------------------------------------


def test_front_door_and_defaults():
    assert wai.GroupwiseGrading is wai.methods.GroupwiseGrading is GroupwiseGrading
    assert "GroupwiseGrading" not in wai.__all__  # one dot down; the front door stays at 31
    g = GroupwiseGrading(grader=lambda group: {"ranking": []})
    assert g.mode == defaults.GROUPWISE_MODE == "advantage"
    assert g.cap == defaults.GROUPWISE_CAP
    assert g.min_factor == defaults.GROUPWISE_MIN_FACTOR
    assert g.floor == defaults.GROUPWISE_FLOOR == 0.0
    assert g.hack_zero is True
    assert "advantage" in str(g) and "cap=3.0" in str(g)
    with pytest.raises(ValueError, match="mode must be one of"):
        GroupwiseGrading(grader=len, mode="loss")
    with pytest.raises(TypeError, match="callable"):
        GroupwiseGrading(grader="haiku")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="floor"):
        GroupwiseGrading(grader=len, floor=1.0)
    with pytest.raises(TypeError, match="rubrics"):
        GroupwiseGrading(grader=len, mode="reward", rubrics=3)  # type: ignore[arg-type]


def _items(rewards, task="t1"):
    return [
        {"prompt": "p", "completion": f"c{i}", "reward": r, "task_id": task}
        for i, r in enumerate(rewards)
    ]


def test_advantage_mode_returns_shifted_redistribution_for_a_mixed_group():
    seen = {}

    def grader(group):
        seen["group"] = group
        passing = [g["index"] for g in group if g["passed"]]
        return {"ranking": passing}  # first pass best

    g = GroupwiseGrading(grader=grader)
    out = g.group(_items([1, 1, 0, 0]))
    # TRL subtracts the group mean: what is left is equation 3
    mean = sum(out) / 4
    adv = [x - mean for x in out]
    assert _close(adv, redistribute([1, 1, 0, 0], {0: 1.0, 1: 0.5}))
    assert math.isclose(mean, 0.5)  # the mean effective reward rides along
    assert [x["index"] for x in seen["group"]] == [0, 1, 2, 3]
    assert [x["passed"] for x in seen["group"]] == [True, True, False, False]
    assert g.stats["groups"] == 1 and g.stats["mixed"] == 1 and g.stats["grader_calls"] == 1
    assert g.stats["factors"] == [1.0, 0.5]


def test_advantage_mode_zeroes_a_confirmed_hack_then_recomputes():
    def grader(group):
        return {"ranking": [0], "hacks": [1]}

    g = GroupwiseGrading(grader=grader)
    out = g.group(_items([1, 1, 0, 0]))
    # the hack is a failure now: effective rewards [1, 0, 0, 0], one pass, nothing to move
    mean = sum(out) / 4
    assert math.isclose(mean, 0.25)
    assert _close([x - mean for x in out], [0.75, -0.25, -0.25, -0.25])
    assert g.stats["hacks"] == 1
    kept = GroupwiseGrading(grader=grader, hack_zero=False)
    out2 = kept.group(_items([1, 1, 0, 0]))
    assert math.isclose(sum(out2) / 4, 0.5)  # the hack stayed a pass
    assert kept.stats["hacks"] == 0


def test_advantage_mode_skips_pure_groups_and_falls_back_on_a_bad_grader():
    calls = []

    def grader(group):
        calls.append(len(group))
        raise RuntimeError("timeout")

    g = GroupwiseGrading(grader=grader)
    assert g.group(_items([1, 1, 1])) == [1.0, 1.0, 1.0]  # all pass: no call
    assert g.group(_items([0, 0])) == [0.0, 0.0]  # all fail: no call
    assert calls == []
    assert g.group(_items([1, 0])) == [1.0, 0.0]  # grader raised: original rewards
    assert calls == [2]
    assert g.stats["grader_failures"] == 1
    bad = GroupwiseGrading(grader=lambda group: {"ranking": [0]})  # no factor for pass 1
    assert bad.group(_items([1, 1, 0])) == [1.0, 1.0, 0.0]
    assert bad.stats["grader_failures"] == 1
    assert "1 failed (fell back)" in str(bad.stats)


def test_advantage_mode_accepts_factors_and_counts_the_cap():
    # lambda uncapped = 1.0 / 0.525 = 1.9; a cap of 1.5 binds
    g = GroupwiseGrading(grader=lambda group: {"factors": {0: 1.0, 1: 0.05}}, cap=1.5)
    out = g.group(_items([1, 1, 0, 0]))
    assert math.isclose(sum(out) / 4, 0.5)
    assert g.stats["lambda_capped"] == 1
    assert "lambda capped 1" in str(g.stats)


def test_reward_mode_multiplies_passes_by_the_rubric_scores():
    def grader(item, rubric):
        assert rubric == ["one call", "no invented id"]
        return {"solution": 0.5, "behavior": 0.8}

    g = GroupwiseGrading(
        grader=grader, mode="reward", rubrics={"t1": ["one call", "no invented id"]}
    )
    out = g.group(_items([1, 0.1, 1, 0]))
    assert _close(out, [0.4, 0.1, 0.4, 0.0])  # failures keep their reward, equation 2 on passes
    assert g.stats["grader_calls"] == 2 and g.stats["scores"] == [0.4, 0.4]
    floored = GroupwiseGrading(
        grader=grader, mode="reward", floor=0.6, rubrics={"t1": ["one call", "no invented id"]}
    )
    assert _close(floored.group(_items([1, 0])), [0.6 * 0.8, 0.0])


def test_reward_mode_writes_and_caches_rubrics_from_the_group_and_falls_back():
    written = []

    def write(group):
        written.append([g["completion"] for g in group])
        return ["criterion"]

    def grader(item, rubric):
        if rubric is None:
            raise RuntimeError("no rubric")
        return (
            {"solution": 1.0, "behavior": 0.5}
            if item["completion"] == "c0"
            else {"solution": 2.0, "behavior": 1}
        )

    g = GroupwiseGrading(grader=grader, mode="reward", rubrics=write)
    out = g.group(_items([1, 1]))
    assert _close(out, [0.5, 1.0])  # second verdict out of range: fell back to R_test
    g.group(_items([1, 0]))
    assert len(written) == 1  # cached per task id
    assert g.stats["grader_failures"] == 1


# --- a batch, and the TRL reward function --------------------------------------------


def test_shape_splits_a_batch_into_consecutive_groups():
    g = GroupwiseGrading(
        grader=lambda group: {"ranking": [x["index"] for x in group if x["passed"]]}
    )
    rewards = [1, 0, 0, 0, 1, 1, 0, 1]
    out = g.shape(rewards, ["a"] * 4 + ["b"] * 4, [f"c{i}" for i in range(8)], group_size=4)
    assert len(out) == 8
    first = redistribute(rewards[:4], {0: 1.0})
    second = redistribute(rewards[4:], {4 - 4: 1.0, 5 - 4: 0.75, 7 - 4: 0.5})
    assert _close([x - 0.25 for x in out[:4]], first)
    assert _close([x - 0.75 for x in out[4:]], second)
    assert g.stats["groups"] == 2
    with pytest.raises(ValueError, match="do not split"):
        g.shape(rewards, ["a"] * 8, ["c"] * 8, group_size=3)
    with pytest.raises(ValueError, match="same length"):
        g.shape(rewards, ["a"] * 7, ["c"] * 8, group_size=4)


def test_trl_reward_wraps_a_base_reward_and_reads_task_id():
    def base(prompts, completions, **kwargs):
        assert kwargs["task_id"] == ["t1", "t1", "t2", "t2"]
        return [1.0, 1.0, 1.0, 0.0]

    g = GroupwiseGrading(
        grader=lambda item, rubric: {"solution": rubric, "behavior": 1.0},
        mode="reward",
        rubrics={"t1": 0.5, "t2": 0.25},
    )
    fn = g.trl_reward(base, num_generations=2)
    assert fn.__name__ == "base_groupwise"
    out = fn(
        prompts=["a", "a", "b", "b"],
        completions=["x", "y", "z", "w"],
        task_id=["t1", "t1", "t2", "t2"],
    )
    assert _close(out, [0.5, 0.5, 0.25, 0.0])


def test_shape_grades_groups_in_parallel_with_the_same_answer():
    g = GroupwiseGrading(
        grader=lambda group: {"ranking": [x["index"] for x in group if x["passed"]]}, concurrency=4
    )
    serial = GroupwiseGrading(grader=g.grader, concurrency=1)
    rewards = [1, 0] * 6
    args = (rewards, ["p"] * 12, ["c"] * 12)
    assert g.shape(*args, group_size=2) == serial.shape(*args, group_size=2)


# --- the spread check ------------------------------------------------------------------


def test_spread_reads_no_spread_off_a_constant_grader():
    flat = spread([0.93, 0.93, 0.94, 0.93, 0.92, 0.93])
    assert isinstance(flat, SpreadReport) and flat["ok"] is False
    assert "no spread" in str(flat) and "mode='advantage'" in str(flat)
    wide = spread([1.0, 0.75, 0.5, 1.0, 0.5, 0.875])
    assert wide["ok"] is True and "spread: yes" in str(wide)
    assert spread([0.9])["ok"] is False and "at least two" in str(spread([0.9]))
    assert spread([])["n"] == 0
    assert repr(wide).startswith("SpreadReport(ok=True")


def _rows(scenarios):
    rows = []
    for sid, rewards in scenarios.items():
        for i, r in enumerate(rewards):
            rows.append(
                {
                    "scenario_id": sid,
                    "rollout_index": i,
                    "prompt": sid,
                    "reward": r,
                    "final_text": f"{sid}-{i}",
                }
            )
    return rows


def test_check_spread_refuses_a_grader_with_no_spread_and_passes_one_with_spread():
    constant = GroupwiseGrading(
        grader=lambda item, rubric: {"solution": 0.93, "behavior": 1.0}, mode="reward"
    )
    rows = _rows(
        {"a": [1, 1, 0, 1], "b": [1, 1, 1, 0], "c": [1, 0, 0, 0]}
    )  # c has one pass: skipped
    with pytest.raises(ValueError, match="no spread"):
        constant.check_spread(rows)
    report = constant.check_spread(rows, strict=False)
    assert report["n"] == 6 and report["groups"] == 2 and report["ok"] is False

    ranking = GroupwiseGrading(
        grader=lambda group: {"ranking": [x["index"] for x in group if x["passed"]]}
    )
    good = ranking.check_spread(rows)
    assert good["ok"] is True and good["n"] == 6
    assert good["scores"] == [1.0, 0.75, 0.5, 1.0, 0.75, 0.5]
    assert ranking.check_spread([], strict=False)["n"] == 0  # no groups: nothing to say
    hacked = GroupwiseGrading(grader=lambda group: {"ranking": [0, 2], "hacks": [1]})
    both = hacked.check_spread(_rows({"a": [1, 1, 1]}), strict=False)  # all pass: still graded
    assert both["n"] == 2 and both["scores"] == [1.0, 0.5] and hacked.stats["hacks"] == 1
    with pytest.raises(ValueError, match="at least two"):
        ranking.check_spread(_rows({"a": [1, 0, 0]}))
