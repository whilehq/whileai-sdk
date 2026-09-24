"""wai.methods.route: every method scored against a graded pool.

Each case is a pool shape that cost a GPU run while dogfooding
(2026-09-24): a floor sized on another model's pass rate, a teacher that
never beat its student, truncated passes, an unaudited judge, k=1 traces.
"""

from __future__ import annotations

import whileai as wai
from whileai import routing
from whileai.simulations import defaults

MODEL = "Qwen/Qwen3-8B"


def graded(passes: list[int], k: int = 4, **extra) -> list[dict]:
    return [
        {"task_id": f"t{t}", "reward": 1.0 if i < n else 0.0, **extra}
        for t, n in enumerate(passes)
        for i in range(k)
    ]


def test_a_healthy_pool_routes_to_grpo_and_scores_every_method():
    r = wai.methods.route(graded([0, 1, 2, 3, 4] * 32))
    assert r.method == "grpo"
    assert set(r.methods) == set(routing.ROUTED)
    assert r.methods["grpo"]["ok"] and r.methods["dpo"]["ok"] and r.methods["sft"]["ok"]
    assert r.measured["mixed"] == 96 and r.measured["tasks"] == 160
    assert "96 of 160" in r.why
    text = str(r)
    assert text.startswith("route: grpo") and "not routed:" in text


def test_a_floor_routes_to_opsd_on_a_big_enough_student():
    r = wai.methods.route(graded([0] * 120 + [1, 2] * 20), size_b=8)
    assert r.method == "opsd"
    assert "floor" in r.why
    assert not r.methods["grpo"]["ok"]


def test_a_floor_on_a_small_student_needs_a_teacher():
    # 40 tasks still have a pass and a fail, so DPO runs on them; the floor needs a teacher
    r = wai.methods.route(graded([0] * 120 + [1, 2] * 20), size_b=1.5)
    assert r.method == "dpo"
    assert not r.methods["opsd"]["ok"] and "1.5B" in r.methods["opsd"]["why"]
    assert r.need["teacher_completions"] == 120
    with_teacher = wai.methods.route(
        graded([0] * 120 + [1, 2] * 20), size_b=1.5, teacher=graded([3] * 160)
    )
    # a teacher that clears the floor: OPD first (on-policy, no exposure bias), SFT also runs
    assert with_teacher.method == "opd"
    assert with_teacher.methods["sft"]["ok"]


def test_a_pool_sized_on_the_wrong_model_is_a_floor_not_a_grpo_pool():
    # graceful-refusal: sized on Haiku's p=0.285, the student passes almost nothing
    r = wai.methods.route(graded([0] * 88 + [1] * 2, k=8), size_b=8)
    assert not r.methods["grpo"]["ok"]
    assert r.method == "opsd"


def test_a_saturated_pool_needs_an_audited_grader():
    rows = graded([4] * 140 + [2] * 20)
    r = wai.methods.route(rows)
    assert r.method is None
    assert "unaudited" in r.methods["groupwise"]["why"]
    assert not r.methods["sft"]["ok"]
    audited = wai.methods.route(rows, judge={"agreement": 0.86, "kappa": 0.71, "n": 200})
    assert audited.method == "groupwise"


def test_a_teacher_that_does_not_beat_the_student_blocks_opd():
    # the -14.8 lane: teacher 54.2 against a student at 56.9
    student = graded([2, 3, 2, 2, 3, 2, 2, 3] * 20)
    teacher = graded([2, 2, 2, 3, 2, 2, 2, 2] * 20)
    r = wai.methods.route(student, teacher=teacher)
    assert not r.methods["opd"]["ok"]
    assert "not clear of the student" in r.methods["opd"]["why"]
    assert r.method == "grpo"
    assert r.measured["teacher_gap"] < 0


def test_a_teacher_cut_off_at_the_student_cap_blocks_opd():
    student = graded([2, 3, 2, 2, 3, 2, 2, 3] * 20)
    teacher = graded([4] * 160, finish_reason="length")
    r = wai.methods.route(student, teacher=teacher)
    assert not r.methods["opd"]["ok"]
    assert "cut off" in r.methods["opd"]["why"]


def test_mismatched_tokenizers_block_opd():
    student = graded([2, 3, 2, 2, 3, 2, 2, 3] * 20)
    r = wai.methods.route(student, teacher=graded([4] * 160), vocab=(151_936, 248_320))
    assert not r.methods["opd"]["ok"]
    assert "151,936 vs 248,320" in r.methods["opd"]["why"]


def test_a_teacher_that_clears_every_guard_routes_to_opd():
    student = graded([2, 3, 2, 2, 3, 2, 2, 3] * 20)
    r = wai.methods.route(student, teacher=graded([4] * 160), vocab=(151_936, 151_936))
    assert r.method == "opd"
    assert "same tokenizer" in r.why


def test_a_small_pool_with_a_healthy_share_asks_for_tasks():
    r = wai.methods.route(graded([0, 2, 4, 1, 3] * 5))  # 25 tasks, 15 mixed
    assert not r.methods["grpo"]["ok"]
    assert r.need["tasks_total"] == 54  # ceil(32 / 0.6)


def test_a_low_share_asks_for_a_higher_k():
    # k=2 on 200 tasks: 90 all-fail, 70 all-pass, 40 mixed.
    # 20% mixed now; with Laplace-smoothed rates E[mixed] reaches 30% at a larger k.
    rows = graded([0] * 90 + [2] * 70 + [1] * 40, k=2)
    r = wai.methods.route(rows)
    assert not r.methods["grpo"]["ok"]
    assert r.need["k"] > 2
    assert f"k={r.need['k']}" in r.methods["grpo"]["why"]


def test_truncated_passes_block_everything():
    rows = graded([0, 1, 2, 3, 4] * 32)
    rows[1]["finish_reason"] = "length"  # how the engine stamps a capped reply
    rows[1]["reward"] = 1.0
    r = wai.methods.route(rows)
    assert r.method is None
    assert "did not end on their own" in r.why
    assert all(not m["ok"] for m in r.methods.values())


def test_an_unaudited_judge_blocks_everything():
    rows = graded([0, 1, 2, 3, 4] * 32, judge_name="haiku-judge")
    r = wai.methods.route(rows)
    assert r.method is None and "compare_judges" in r.why
    bad = wai.methods.route(rows, judge={"agreement": 0.48, "kappa": -0.06, "n": 60})
    assert bad.method is None and "misses the floors" in bad.why
    unsized = wai.methods.route(rows, judge={"agreement": 0.86, "kappa": 0.71})
    assert unsized.method is None and "no label count" in unsized.why
    # 0.86 on 60 labels: the Wilson lower bound is 0.75, under the 0.8 floor
    small = wai.methods.route(rows, judge={"agreement": 0.86, "kappa": 0.71, "n": 60})
    assert small.method is None
    good = wai.methods.route(rows, judge={"agreement": 0.86, "kappa": 0.71, "n": 200})
    assert good.method == "grpo"


def test_rows_from_another_model_are_off_policy_for_grpo():
    rows = graded([0, 1, 2, 3, 4] * 32, agent_model="Qwen/Qwen3-32B")
    r = wai.methods.route(rows, model=MODEL)
    assert not r.methods["grpo"]["ok"]
    assert "off-policy" in r.methods["grpo"]["why"]
    assert r.measured["on_policy"] is False


def test_single_rollout_traces():
    # one rollout per prompt: SFT on the passes (rejection sampling) runs; grouped methods need k
    bare = wai.methods.route(graded([1, 0] * 50, k=1))
    assert bare.method == "sft"
    assert not bare.methods["grpo"]["ok"] and not bare.methods["flashreinforce"]["ok"]
    assert bare.need["k"] == defaults.RL_ROLLOUTS_PER_PROMPT
    with_logprobs = wai.methods.route(graded([1, 0] * 50, k=1, token_logprobs=[-0.5]))
    assert with_logprobs.methods["flashreinforce"]["ok"]


def test_unchecked_inputs_are_named_not_silent():
    r = wai.methods.route(graded([0, 1, 2, 3, 4] * 32), teacher=graded([4] * 160))
    notes = " ".join(r.notes)
    assert "on-policy not checked" in notes
    assert "tokenizer not checked" in notes
    assert "size_b" in notes


def test_the_report_is_a_dict_that_prints_itself():
    r = wai.methods.route(graded([0, 1, 2, 3, 4] * 32))
    assert r["method"] == r.method
    assert "method=" in repr(r)
    assert "<pre>" in r._repr_html_()


def test_partial_credit_is_counted_and_left_out_like_pass_at():
    rows = graded([0, 1, 2, 3, 4] * 32) + [{"task_id": "t0", "reward": 0.7}] * 5
    r = wai.methods.route(rows)
    assert r.measured["partial_rows"] == 5
    assert r.measured["tasks"] == 160
    assert any("partial credit" in note for note in r.notes)


def test_a_few_single_rollout_tasks_do_not_switch_off_grpo():
    rows = graded([0, 1, 2, 3, 4] * 32) + [{"task_id": f"s{i}", "reward": 1.0} for i in range(10)]
    r = wai.methods.route(rows)
    assert r.method == "grpo"
    assert r.measured["single_rollout_tasks"] == 10


def test_grpo_counts_the_band_select_for_rl_keeps():
    # p = 1/8 is mixed but under DIFFICULTY_BAND's 0.2: not trainable signal for grpo
    rows = graded([1] * 150 + [4] * 10, k=8)
    r = wai.methods.route(rows)
    assert r.measured["mixed"] == 160 and r.measured["in_band"] == 10
    assert not r.methods["grpo"]["ok"]


def test_a_partial_floor_keeps_grpo_and_plans_opsd_for_the_floor():
    rows = graded([0] * 90 + [1, 2, 3] * 30)  # 50% all-fail, 50% in band
    r = wai.methods.route(rows, size_b=8)
    assert r.method == "grpo"
    assert r.plan == ["grpo", "opsd"]
    assert 'privileged="reference"' in r.methods["opsd"]["why"]


def test_the_projection_to_a_larger_k_is_the_exact_beta_expectation():
    # 0 passes in 2 draws, projected to k=8: 0.72 exactly, not the 0.90 a plug-in rate gives
    assert abs(routing._mixed_chance(0, 2, 8) - 0.7212) < 1e-3


def test_a_qwen_student_is_told_to_run_a_random_reward_arm():
    r = wai.methods.route(graded([0, 1, 2, 3, 4] * 32), model="Qwen/Qwen3-8B")
    assert r.method == "grpo"
    assert any("random-reward" in note for note in r.notes)


def test_a_teacher_scored_on_other_tasks_is_not_compared():
    student = graded([2, 3, 2, 2, 3, 2, 2, 3] * 20)
    teacher = [dict(r, task_id="x" + r["task_id"]) for r in graded([4] * 160)]
    r = wai.methods.route(student, teacher=teacher)
    assert not r.methods["opd"]["ok"] and "same tasks" in r.methods["opd"]["why"]
