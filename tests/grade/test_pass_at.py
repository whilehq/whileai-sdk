"""pass@1 / pass^k / pass@k off graded groups (score/passat.py)."""

from __future__ import annotations

import json
from math import comb

import pytest

import whileai.simulations as wai
from whileai.simulations.data import SimulationData
from whileai.simulations.score.judging import ScoredData
from whileai.simulations.score.optimize import group_signal
from whileai.simulations.score.passat import PassAt, pass_at


def _rows(spec: dict[str, list[int]]) -> list[dict]:
    """``{"prompt": [0, 1, ...]}`` -> graded rows, one per label."""
    return [
        {"prompt": prompt, "reward": label, "final_text": "x", "steps": []}
        for prompt, labels in spec.items()
        for label in labels
    ]


def test_unbiased_estimators_match_closed_form():
    # n=4, c=2, k=2: pass@2 = 1 - C(2,2)/C(4,2) = 5/6, pass^2 = C(2,2)/C(4,2) = 1/6.
    out = pass_at(_rows({"a": [1, 1, 0, 0]}), k=2, min_k=2)
    assert out.k == 2
    assert out.pass_at_1 == pytest.approx(0.5)
    assert out.pass_at_k == pytest.approx(1 - comb(2, 2) / comb(4, 2))
    assert out.pass_pow_k == pytest.approx(comb(2, 2) / comb(4, 2))
    assert out.headroom == pytest.approx(out.pass_at_k - 0.5)
    assert out.n_groups == out.n_groups_at_k == 1
    assert out.n_rows == 4


def test_pass_at_1_is_macro_over_tasks_and_k_is_smallest_multi_group():
    out = pass_at(_rows({"a": [1, 1, 1, 1], "b": [0, 0, 0, 0, 0, 0], "solo": [1]}))
    # Three tasks at 1.0, 0.0, 1.0 -> 2/3 regardless of group size.
    assert out.pass_at_1 == pytest.approx(2 / 3)
    assert out.k == 4
    # Unanimous groups: pass@k == pass^k == pass@1 over the k-eligible groups.
    assert out.pass_at_k == pytest.approx(0.5)
    assert out.pass_pow_k == pytest.approx(0.5)
    assert out.n_groups == 3
    assert out.n_groups_at_k == 2  # the solo ask cannot draw 4
    assert out.per_task == {"a": 1.0, "b": 0.0, "solo": 1.0}


def test_below_min_k_withholds_k_way_numbers_with_a_note():
    out = pass_at(_rows({"a": [1, 0], "b": [1, 1]}))
    assert out.k == 2
    assert out.pass_at_1 == pytest.approx(0.75)
    assert out.pass_pow_k is None and out.pass_at_k is None and out.headroom is None
    assert "repeats>=4" in out.note
    assert "n/a" in str(out) and "pass@1 0.75" in str(out)


def test_no_graded_rows_is_none_not_zero():
    out = pass_at([{"prompt": "a", "reward": None}, {"prompt": "b", "reward": 0.5}])
    assert isinstance(out, PassAt)
    assert out.pass_at_1 is None and out.pass_at_k is None
    assert out.n_groups == 0 and out.n_rows == 0
    assert "grade first" in out.note


def test_explicit_k_larger_than_every_group_reports_why():
    out = pass_at(_rows({"a": [1, 0, 1, 0]}), k=8)
    assert out.k == 8
    assert out.pass_at_k is None
    assert "no group has 8" in out.note
    with pytest.raises(ValueError):
        pass_at(_rows({"a": [1, 0]}), k=0)


def test_group_signal_carries_the_same_keys():
    rows = _rows({"mixed": [1, 0, 1, 0], "dead0": [0, 0, 0, 0], "dead1": [1, 1, 1, 1]})
    signal = group_signal(rows)
    direct = pass_at(rows)
    for key in ("k", "pass_at_1", "pass_pow_k", "pass_at_k", "headroom"):
        assert signal[key] == getattr(direct, key)
    assert signal["n_mixed"] == 1
    assert signal["headroom"] > 0
    # No mixed groups -> nothing to learn -> zero headroom.
    flat = group_signal(_rows({"dead0": [0, 0, 0, 0], "dead1": [1, 1, 1, 1]}))
    assert flat["mixed_rate"] == 0
    assert flat["headroom"] == pytest.approx(0.0)


def test_scored_data_and_simulation_data_expose_the_property():
    rows = _rows({"a": [1, 0, 1, 1], "b": [0, 0, 0, 1]})
    scored = ScoredData(rows, run_id="r", source="grade", judge_name="j")
    assert scored.pass_at.pass_at_1 == pytest.approx(0.5)
    data = SimulationData(trajectories=rows)
    assert data.pass_at.to_dict() == scored.pass_at.to_dict()
    assert wai.pass_at(rows).k == 4
    assert "PassAt" in wai.__all__ and "pass_at" in wai.__all__


def test_recommend_rl_names_the_headroom():
    out = wai.recommend(mode="rl", target=200)
    assert any("pass@k - pass@1" in line for line in out["reasoning"])


def test_save_meta_writes_pass_at_to_sidecar(tmp_path):
    rows = _rows({"a": [1, 0, 1, 1], "b": [0, 0, 0, 1]})
    for row in rows:
        # one situation per prompt: rows sharing a scenario_id are one task
        row.update({"arm": "ordinary", "scenario_id": "s-" + row["prompt"], "messages": []})
    data = SimulationData(trajectories=rows)
    data.save(str(tmp_path / "r.jsonl"), meta=True)
    meta = json.loads((tmp_path / "r.meta.json").read_text())
    assert meta["pass_at"]["k"] == 4
    assert meta["pass_at"]["pass_at_1"] == pytest.approx(0.5)
    assert meta["pass_at"]["headroom"] == pytest.approx(
        meta["pass_at"]["pass_at_k"] - meta["pass_at"]["pass_at_1"]
    )


def test_uneven_groups_name_the_k_that_scores_them():
    # A budget cut mid-group: three groups reached 4 repeats, one has 2.
    rows = []
    for prompt, labels in {
        "a": [1, 0, 1, 1],
        "b": [0, 0, 1, 0],
        "c": [1, 1, 1, 1],
        "d": [1, 0],
    }.items():
        rows += [{"prompt": prompt, "reward": r} for r in labels]
    got = pass_at(rows)
    assert got.k == 2 and got.pass_pow_k is None
    assert "uneven (2 to 4 repeats)" in got.note and "k=4" in got.note and "3 group" in got.note
    scored = pass_at(rows, k=4)
    assert scored.n_groups_at_k == 3 and scored.pass_at_k is not None and scored.note == ""


def test_even_groups_keep_the_plain_note():
    # three tasks, so the interval lands and the k note stands alone
    rows = [{"prompt": p, "reward": r} for p in "abc" for r in (1, 0)]
    assert pass_at(rows).note == "set repeats>=4 for pass^k and pass@k"


def test_too_few_tasks_for_an_interval_says_so_and_names_the_fix():
    """pass@1 with ci95 None and an empty note is a mean printed as a
    result. Two eval runs quoted one-task pass@1 before noticing (#490)."""
    rows = [{"prompt": "a", "reward": r} for r in (1, 1, 0, 1)]
    out = pass_at(rows)
    assert out.pass_at_1 == pytest.approx(0.75) and out.ci95 is None
    assert "no interval on pass@1: 1 task," in out.note
    assert "task_id" in out.note and "3" in out.note
    assert out.note in str(out)
    # the note stacks after the k note rather than replacing it
    two = pass_at([{"prompt": p, "reward": r} for p in "ab" for r in (1, 0)])
    assert two.note.startswith("set repeats>=4 for pass^k and pass@k; ")
    assert "2 tasks" in two.note


def test_an_interval_leaves_the_note_alone():
    rows = _rows({"a": [1, 0, 1, 1], "b": [0, 0, 0, 1], "c": [1, 1, 0, 1]})
    out = pass_at(rows)
    assert out.ci95 is not None and "no interval" not in out.note


def test_one_task_id_per_row_is_the_fix_the_note_names():
    """The note's second branch, run: the same ten graded rows read as one
    task with no interval, and as ten tasks with one."""
    labels = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    one = pass_at([{"prompt": "a", "task_id": "t", "reward": r} for r in labels])
    many = pass_at([{"prompt": "a", "task_id": f"t{i}", "reward": r} for i, r in enumerate(labels)])
    assert one.n_groups == 1 and one.ci95 is None
    assert many.n_groups == 10 and many.ci95 is not None
    assert one.pass_at_1 == many.pass_at_1 == pytest.approx(0.7)
    assert "no interval" not in many.note


def test_unanimous_short_groups_count_as_unanimous_when_asked():
    # successive allocation: two prompts split and ran to k=4, two were
    # unanimous after 2 and stopped; one mixed group was cut at 3
    groups = {"a": [1, 0, 1, 1], "b": [0, 1, 0, 0], "c": [1, 1], "d": [0, 0], "e": [1, 0, 1]}
    rows = [{"prompt": p, "reward": r} for p, labels in groups.items() for r in labels]
    skip = pass_at(rows, k=4)
    assert skip.n_groups_at_k == 2 and skip.n_groups_imputed == 0
    got = pass_at(rows, k=4, unanimous_short=True)
    assert got.n_groups_at_k == 4 and got.n_groups_imputed == 2
    # c contributes 1.0 to both, d contributes 0.0; e stays out (mixed, short)
    from math import comb

    a_at = 1 - comb(1, 4) / comb(4, 4) if comb(1, 4) else 1.0
    b_at = 1 - comb(3, 4) / comb(4, 4) if comb(3, 4) else 1.0
    assert got.pass_at_k == pytest.approx((a_at + b_at + 1.0 + 0.0) / 4)
    assert got.pass_pow_k == pytest.approx((0.0 + 0.0 + 1.0 + 0.0) / 4)
    assert got.pass_at_1 == pytest.approx((0.75 + 0.25 + 1.0 + 0.0 + 2 / 3) / 5)


def test_pass_pow_k_and_pass_at_k_carry_task_bootstrap_intervals():
    rows = []
    for t in range(6):
        for i in range(4):
            rows.append({"prompt": f"t{t}", "reward": 1 if (i + t) % 3 else 0})
    out = pass_at(rows)
    assert out.k == 4
    assert out.pass_pow_k_ci95 is not None and out.pass_at_k_ci95 is not None
    lo, hi = out.pass_pow_k_ci95
    assert 0.0 <= lo <= out.pass_pow_k <= hi <= 1.0
    lo, hi = out.pass_at_k_ci95
    assert 0.0 <= lo <= out.pass_at_k <= hi <= 1.0
    text = str(out)
    assert text.count("[") == 3, text  # one band per number
    d = out.to_dict()
    assert d["pass_pow_k_ci95"] == list(out.pass_pow_k_ci95)
    # below min_k the k-way numbers and their bands are both absent
    short = pass_at(
        [{"prompt": p, "reward": r} for p in ("a", "b") for r in (1, 0)]
    )  # two repeats per group: below min_k
    assert short.pass_pow_k is None and short.pass_pow_k_ci95 is None


def test_a_judge_that_failed_on_every_row_says_so_not_grade_first():
    """A cold hosted judge times out on every concurrent call, so the whole
    set reads as ungraded. "grade first" sent the user back to the step that
    had just run (#224)."""
    rows = [
        {
            "prompt": f"p{i}",
            "reward": None,
            "judge_status": "invalid_result",
            "reason": "TimeoutError: The read operation timed out",
            "final_text": "x",
            "steps": [],
        }
        for i in range(6)
    ]
    out = pass_at(rows)
    assert out.pass_at_1 is None and out.n_groups == 0
    assert "the judge failed on all 6 rows" in out.note
    assert "invalid_result" in out.note and "TimeoutError" in out.note
    assert "grade first" not in out.note
    assert "the judge failed on all 6 rows" in str(out)


def test_a_partly_graded_set_still_says_grade_first():
    # one row did grade, so the set is not a judge failure
    rows = [
        {"prompt": "a", "reward": None, "judge_status": "error", "final_text": "x", "steps": []},
        {"prompt": "b", "reward": 0.5, "judge_status": "ok", "final_text": "x", "steps": []},
    ]
    assert "grade first" in pass_at(rows).note
    # and rows nobody judged keep the original wording
    assert "grade first" in pass_at([{"prompt": "a", "reward": None}]).note


def test_partial_rewards_are_counted_and_named_not_dropped_in_silence():
    """10 tasks at k=4, 5 of them scoring 0.67: pass@1 is over the other 5,
    and the result says the 20 rows it left out (#672)."""
    rows = _rows({f"t{i}": [1, 1, 1, 1] for i in range(3)})
    rows += _rows({f"t{i}": [0, 0, 0, 0] for i in range(3, 5)})
    rows += [
        {"prompt": f"t{i}", "reward": 0.67, "final_text": "x", "steps": []}
        for i in range(5, 10)
        for _ in range(4)
    ]
    out = pass_at(rows, k=4)
    assert out.n_groups == 5 and out.n_rows == 20
    assert out.n_partial == 20
    assert out.to_dict()["n_partial"] == 20
    assert "20 row(s) carried a reward that is not 0 or 1" in out.note
    assert "kind='hard'" in out.note
    assert "20 row(s)" in str(out)
    # an all-binary run is untouched
    clean = pass_at(_rows({"a": [1, 0, 1, 1], "b": [0, 0, 0, 0], "c": [1, 1, 1, 1]}), k=4)
    assert clean.n_partial == 0 and "not 0 or 1" not in clean.note
    # every row partial: still nothing to score, and the count says how many
    every = pass_at([{"prompt": "a", "reward": 0.5, "judge_status": "ok"}] * 4)
    assert every.n_groups == 0 and every.n_partial == 4


def test_a_row_with_a_binary_legacy_label_beside_a_partial_reward_is_counted_not_partial():
    """``_group_label_lists`` reads ``qwen_reward`` when ``reward`` is not 0/1,
    so such a row is in pass@1; the partial count must say the same."""
    rows = _rows({"a": [1, 0, 1, 1]})
    rows += [
        {"prompt": "b", "reward": 0.5, "qwen_reward": 1, "final_text": "x", "steps": []}
        for _ in range(4)
    ]
    out = pass_at(rows, k=4)
    assert out.n_groups == 2 and out.n_rows == 8
    assert out.n_partial == 0 and "not 0 or 1" not in out.note
    # a non-numeric reward beside a binary legacy label: same rule
    rows2 = _rows({"a": [1, 0, 1, 1]}) + [
        {"prompt": "b", "reward": "n/a", "qwen_reward": 0, "final_text": "x", "steps": []}
        for _ in range(4)
    ]
    out2 = pass_at(rows2, k=4)
    assert out2.n_groups == 2 and out2.n_rows == 8 and out2.n_partial == 0
