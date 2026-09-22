"""RL filter keeps gold ``reward`` rows. Offline: no writes, no GPU."""

from whileai.simulations.score.optimize import (
    INCOMPLETE_JUNK,
    KEPT_VERIFIED_ZERO,
    UNUSABLE_LABEL,
    drop_reason,
    filter_rl_rows,
    is_unusable_label,
    is_verified_zero,
)


def _kept(**extra):
    row = {
        "prompt": "look up issue 4412",
        "final_text": "Issue 4412 is open.",
        "steps": [{"tool": "get_issue", "arguments": {"number": 4412}, "result": {"status": "ok"}}],
        "messages": [
            {"role": "user", "content": "look up issue 4412"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "get_issue", "arguments": {"number": 4412}}],
            },
            {"role": "tool", "name": "get_issue", "content": '{"status": "ok"}'},
            {"role": "assistant", "content": "Issue 4412 is open."},
        ],
    }
    row.update(extra)
    return row


def test_grade_style_reward_only_is_kept():
    row = _kept(reward=1)
    assert "qwen_reward" not in row
    assert is_unusable_label(row) is False
    assert drop_reason(row) is None


def test_legacy_qwen_reward_only_is_kept():
    row = _kept(qwen_reward=0)
    assert "reward" not in row
    assert is_unusable_label(row) is False
    assert drop_reason(row) is None


def test_missing_label_is_unusable():
    row = _kept()
    assert is_unusable_label(row) is True
    assert drop_reason(row) == UNUSABLE_LABEL


def test_filter_keeps_reward_and_qwen_rows():
    rows = [_kept(reward=1), _kept(qwen_reward=0), _kept()]
    kept, report = filter_rl_rows(rows)
    assert report["n_kept"] == 2
    assert report["dropped"][UNUSABLE_LABEL] == 1
    assert len(kept) == 2


def _junk_zero(**extra):
    # Degenerate final text on an actionable ask: junk by every gate.
    row = _kept(reward=0, final_text="a" * 40)
    row["steps"] = []
    row["messages"] = [
        {"role": "user", "content": "look up issue 4412"},
        {"role": "assistant", "content": "a" * 40},
    ]
    row.update(extra)
    return row


def test_verified_zero_bypasses_junk_gates():
    row = _junk_zero(label_source="claude_fleet_blind_20260821")
    assert is_verified_zero(row) is True
    assert drop_reason(row) is None
    kept, report = filter_rl_rows([row])
    assert len(kept) == 1
    assert report[KEPT_VERIFIED_ZERO] == 1


def test_unverified_junk_zero_still_drops():
    assert drop_reason(_junk_zero()) == INCOMPLETE_JUNK
    assert drop_reason(_junk_zero(label_source="judge")) == INCOMPLETE_JUNK


def test_verified_one_does_not_bypass():
    row = _junk_zero(reward=1, label_source="claude_fleet_blind_20260821")
    assert is_verified_zero(row) is False
    assert drop_reason(row) == INCOMPLETE_JUNK


def test_verified_zero_needs_trainable_content():
    row = _junk_zero(label_source="gold", final_text="")
    row["messages"] = []
    assert drop_reason(row) == INCOMPLETE_JUNK


def _graded(prompt, reward, suffix=""):
    row = _kept(reward=reward)
    row["prompt"] = prompt
    row["final_text"] = f"Issue 4412 is open.{suffix}"
    return row


def _grouped_rows():
    rows = []
    rows += [_graded("mixed ask", r) for r in (1, 0, 1, 0)]
    rows += [_graded("all zero ask", 0) for _ in range(4)]
    rows += [_graded("all one ask", 1) for _ in range(4)]
    rows += [_graded("solo ask", 1)]
    return rows


def test_group_signal_counts_mix_and_band():
    from whileai.simulations.score.optimize import group_signal

    signal = group_signal(_grouped_rows())
    assert signal["n_groups"] == 4
    assert signal["n_mixed"] == 1
    assert signal["n_all_zero"] == 1
    assert signal["n_all_one"] == 1
    assert signal["n_single"] == 1
    assert signal["n_in_band"] == 1  # p = 0.5
    assert signal["mixed_rate"] == 1 / 3


def test_trim_unanimous_drops_dead_groups_keeps_singles():
    from whileai.simulations.score.optimize import trim_unanimous_groups

    kept, report = trim_unanimous_groups(_grouped_rows())
    prompts = {row["prompt"] for row in kept}
    assert prompts == {"mixed ask", "solo ask"}
    assert report["n_groups_dropped"] == 2
    assert report["signal"]["n_mixed"] == 1


def test_select_for_rl_keeps_whole_groups():
    from whileai.simulations.score.optimize import select_for_rl

    # The fixture repeats one identical trajectory per ask, which the
    # duplicate gate would collapse; switch it off to test group selection.
    picked, report = select_for_rl(_grouped_rows(), target=4, dedupe=False)
    prompts = [row["prompt"] for row in picked]
    assert prompts.count("mixed ask") == 4  # the group came whole
    assert report["unanimous_groups_dropped"] == 2
    assert report["signal"]["n_mixed"] == 1

    # With the gate on, identical rollouts carrying different rewards are
    # judge noise: the ask collapses to one row and is dropped as a dead
    # group, not kept as a single.
    deduped, dedup_report = select_for_rl(_grouped_rows(), target=4)
    assert [row["prompt"] for row in deduped] == ["solo ask"]
    assert dedup_report["duplicates"]["conflicting_rewards"] >= 1
    assert dedup_report["collapsed_groups_dropped"] >= 1


def _asks(spec):
    """``{prompt: [labels]}`` to rows; distinct replies so dedupe keeps them."""
    rows = []
    for prompt, labels in spec.items():
        rows += [_graded(prompt, r, suffix=f" reply {i}") for i, r in enumerate(labels)]
    return rows


def test_select_for_rl_spreads_across_pass_rates_by_default():
    import pytest

    from whileai.simulations.score.optimize import select_for_rl

    # Three asks each at 25%, 50% and 75%: the band has no favourite.
    spec = {}
    for i in range(3):
        spec[f"low {i}"] = [1, 0, 0, 0]
        spec[f"mid {i}"] = [1, 1, 0, 0]
        spec[f"high {i}"] = [1, 1, 1, 0]
    picked, report = select_for_rl(_asks(spec), target=12)
    first_three = list(dict.fromkeys(row["prompt"] for row in picked))
    assert [p.split()[0] for p in first_three] == ["low", "mid", "high"]
    assert report["groups_selected"] == 3

    # The older ranking is still there under a plain name.
    picked, _ = select_for_rl(_asks(spec), target=12, order="middle")
    assert all(row["prompt"].startswith("mid") for row in picked)
    with pytest.raises(ValueError, match="order must be one of"):
        select_for_rl(_asks(spec), order="nearest")


def test_select_for_rl_reports_the_interval_and_small_k():
    from whileai.simulations import calibration_of
    from whileai.simulations.score.optimize import select_for_rl

    picked, report = select_for_rl(_asks({"a": [1, 0, 1, 0], "b": [1, 1, 0, 0]}), target=8)
    tasks = report["calibration"]["tasks"]
    assert {t["task_id"] for t in tasks} == {"a", "b"} and all(t["n"] == 4 for t in tasks)
    lo, hi = tasks[0]["pass_rate_ci95"]
    assert lo < 0.5 < hi
    stamp = calibration_of(picked[0])
    assert stamp is not None and stamp.pass_rate_ci95 == (lo, hi)
    note = [w for w in report["hygiene_warnings"] if "Difficulty was measured from 4" in w]
    assert note and "±" in note[0] and "repeats=16" in note[0]

    # Sixteen rollouts per task is the firmer band; no note.
    picked, report = select_for_rl(_asks({"a": [1, 0] * 8, "b": [1, 1, 0, 0] * 4}), target=32)
    assert not any("Difficulty was measured" in w for w in report["hygiene_warnings"])


def test_select_for_sft_takes_only_passes_and_spreads_behaviors():
    from whileai.simulations.score.optimize import select_for_sft

    rows = _grouped_rows()
    picked, report = select_for_sft(rows, target=3)
    assert picked
    assert all(row["reward"] == 1 for row in picked)
    # Duplicate prompts never ship twice.
    prompts = [row["prompt"] for row in picked]
    assert len(prompts) == len(set(prompts))
    assert report["n_selected"] == len(picked)


def _scored(prompt, reward, final="ok"):
    return {
        "prompt": prompt,
        "reward": reward,
        "final_text": f"{final} {reward}",
        "steps": [{"tool": "get_order", "arguments": {"id": "1"}, "result": {"ok": 1}}],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{final} {reward}"},
        ],
    }


def test_select_for_sft_ranks_partial_credit_by_reward():
    """Lambert 2025, chapter Rejection Sampling: argmax per prompt over a
    scalar reward. A 0.9 used to be dropped as not-pass because only exact 1s
    qualified."""
    from whileai.simulations.score.optimize import select_for_sft

    rows = [_scored("a", 0.3), _scored("a", 0.9), _scored("a", 0.6), _scored("b", 0.7)]
    picked, report = select_for_sft(rows, target=10)
    assert picked == [] and report["n_not_pass"] == 4  # default min_reward=1.0
    picked, report = select_for_sft(rows, target=10, min_reward=0.5)
    assert sorted((r["prompt"], r["reward"]) for r in picked) == [("a", 0.9), ("b", 0.7)]
    assert report["selection"] == "top_per_prompt" and report["min_reward"] == 0.5
    assert report["n_not_pass"] == 1 and report["reward_mean_selected"] == 0.8


def test_select_for_sft_top_k_overall_and_random_controls():
    from whileai.simulations.score.optimize import select_for_sft

    rows = [_scored("a", 0.3), _scored("a", 0.9), _scored("a", 0.6), _scored("b", 0.7)]
    picked, report = select_for_sft(rows, select="top_k_overall", k=2, min_reward=0.0)
    assert [r["reward"] for r in picked] == [0.9, 0.7] and report["k"] == 2
    picked, _ = select_for_sft(rows, select="random_per_prompt", min_reward=0.0, seed=1)
    assert {r["prompt"] for r in picked} == {"a", "b"} and len(picked) == 2
    again, _ = select_for_sft(rows, select="random_per_prompt", min_reward=0.0, seed=1)
    assert [r["reward"] for r in again] == [r["reward"] for r in picked]
    picked, report = select_for_sft(rows, select="random_k_overall", k=3, min_reward=0.0, seed=2)
    assert len(picked) == 3 and report["selection"] == "random_k_overall"
    import pytest

    with pytest.raises(ValueError, match="select must be one of"):
        select_for_sft(rows, select="best")


def test_optimize_dispatches_on_mode_and_never_overwrites(tmp_path):
    import json

    from whileai.simulations.score.optimize import optimize

    src = tmp_path / "batch.jsonl"
    rows = _grouped_rows()
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _picked, report = optimize(str(src), mode="rl", target=4)
    assert report["mode"] == "rl"
    assert report["path"].endswith("batch.rl.jsonl")
    assert src.read_text().count("\n") == len(rows)  # source untouched
    sft_rows, sft_report = optimize(rows, mode="sft", target=2)
    assert sft_report["mode"] == "sft"
    assert all(r["reward"] == 1 for r in sft_rows)


def test_no_tool_agent_keeps_refusal_demonstrations():
    from whileai.simulations.score.optimize import is_do_nothing, select_for_sft

    row = {
        "prompt": "check the status of order 98765 for me",
        "ask_family": "tool",
        "final_text": "I cannot look up orders. For billing, see the front desk.",
        "reward": 1,
        "steps": [],
        "messages": [
            {"role": "user", "content": "check the status of order 98765 for me"},
            {
                "role": "assistant",
                "content": "I cannot look up orders. For billing, see the front desk.",
            },
        ],
    }
    assert is_do_nothing(row) is True  # tool agent: a real drop
    assert is_do_nothing(row, has_tools=False) is False
    # Judge-labeled rows bypass the heuristic entirely: a 1 is the
    # judge's call, whatever the grid expected.
    picked, _report = select_for_sft([row], target=5)
    assert len(picked) == 1
    from whileai.simulations.score.optimize import drop_reason

    unlabeled = dict(row)
    unlabeled.pop("reward")
    assert drop_reason(unlabeled) == "do_nothing"
    assert drop_reason(unlabeled, has_tools=False) != "do_nothing"


def _rate_only(spec):
    """``{task: [labels]}`` to the rows a trainer's state holds: a task and a
    binary reward per sample, no reply."""
    rows = []
    for task, labels in spec.items():
        rows += [{"prompt": task, "reward": r, "rollout_index": i} for i, r in enumerate(labels)]
    return rows


def test_select_for_rl_takes_pass_rate_only_rows_without_the_text_gates():
    import pytest

    from whileai.simulations.score.optimize import select_for_rl

    spec = {
        "hard": [1, 0, 0, 0, 0, 0, 0, 0],
        "mid": [1, 1, 1, 1, 0, 0, 0, 0],
        "high": [1, 1, 1, 1, 1, 1, 0, 0],
        "solved": [1] * 8,
        "flat": [0] * 8,
    }
    picked, report = select_for_rl(_rate_only(spec), target=100)
    assert {row["prompt"] for row in picked} == {"mid", "high"}
    assert report["gates"]["incomplete_junk"] == 0
    assert report["text_gates"] == {
        "mode": "auto",
        "applied": False,
        "reason": "no row carries a reply",
    }
    # the duplicate trim keys on the reply too, so it did not run either
    assert report["duplicates"]["n_dropped"] == 0
    assert report["unanimous_groups_dropped"] == 2
    assert report["band_dropped"] == {"too_easy": 0, "too_hard": 1}
    assert any(
        "text gates" in w and "text_gates='require'" in w for w in report["hygiene_warnings"]
    )
    # the older refusal is one keyword away
    picked, report = select_for_rl(_rate_only(spec), target=100, text_gates="require")
    assert picked == [] and report["gates"]["incomplete_junk"] == 40
    assert report["text_gates"]["applied"] is True
    with pytest.raises(ValueError, match="text_gates must be one of"):
        select_for_rl(_rate_only(spec), text_gates="maybe")


def test_select_for_rl_keeps_the_text_gates_when_any_row_has_a_reply():
    from whileai.simulations.score.optimize import select_for_rl

    rows = _asks({"a": [1, 0, 1, 0]}) + _rate_only({"b": [1, 0, 1, 0]})
    picked, report = select_for_rl(rows, target=100)
    assert report["text_gates"]["applied"] is True
    assert {row["prompt"] for row in picked} == {"a"}
    assert report["gates"]["incomplete_junk"] == 4
    picked, report = select_for_rl(rows, target=100, text_gates="skip")
    assert {row["prompt"] for row in picked} == {"a", "b"}
    assert report["text_gates"]["reason"] == "text_gates='skip'"


def test_next_round_and_band_take_a_per_task_rate_table():
    import whileai.simulations as wai
    from whileai.simulations.score.optimize import trim_out_of_band

    table = [
        {"prompt": "hard", "pass_rate": 0.1, "n": 16},
        {"prompt": "mid", "pass_rate": 0.5, "n": 16},
        {"prompt": "solved", "pass_rate": 0.9, "n": 16},
        {"prompt": "thin", "pass_rate": 0.5, "n": 1},  # under min_k for the band
        {"prompt": "stamped", "calibration": {"pass_rate": 0.05, "n": 16}},
        {"prompt": "unsaid", "pass_rate": 0.95},  # no n: taken at its word
    ]
    plan = wai.next_round(table)
    assert plan["pass_rates"] == {
        "hard": 0.1,
        "mid": 0.5,
        "solved": 0.9,
        "thin": 0.5,
        "stamped": 0.05,
        "unsaid": 0.95,
    }
    assert plan["kept"] == 2 and plan["dropped_solved"] == 2 and plan["dropped_unsolved"] == 2
    assert [t["calibration"]["n"] for t in plan["tasks"]] == [16, 1]
    kept, report = trim_out_of_band(table)
    assert [r["prompt"] for r in kept] == ["mid", "thin"]
    assert report["from_rates"] == 6 and report["too_easy"] == 2 and report["too_hard"] == 2


def test_next_round_warns_when_the_prior_carries_no_measurement():
    import pytest

    import whileai.simulations as wai

    prior = [{"prompt": "a", "reward": 0.5}, {"prompt": "b"}]
    with pytest.warns(UserWarning, match="none of the 2 prior rows carries a binary reward"):
        plan = wai.next_round(prior, tasks=["a", "c"])
    assert plan["kept"] == 0 and plan["unknown"] == 2


def test_task_pass_rates_read_rollouts_first_and_a_stamp_only_without_them():
    from whileai.simulations.score.optimize import _task_pass_rates, select_for_rl

    # graded rollouts carrying a stale stamp: the rollouts are the measurement
    rows = [
        {"prompt": "b", "reward": r, "calibration": {"pass_rate": 1.0, "n": 16}}
        for r in (1, 0, 0, 0)
    ]
    assert _task_pass_rates(rows) == {"b": (0.25, 4)}
    # a row with no binary reward is read from its stamp
    assert _task_pass_rates([{"prompt": "c", "calibration": {"pass_rate": 0.5, "n": 8}}]) == {
        "c": (0.5, 8)
    }
    # select_for_rl stamps its selection in place, so a second call on the
    # same rows sees the stamp; the second measurement must not change
    rows = _asks({"a": [1, 1, 1, 1, 0], "b": [1, 0]})
    first, _ = select_for_rl(rows, target=100)
    assert "calibration" in first[0]
    second, _ = select_for_rl(rows, target=100)
    assert [r["prompt"] for r in second] == [r["prompt"] for r in first]
