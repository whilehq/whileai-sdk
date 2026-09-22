"""select_for_rl(truncated=): drop, keep, or penalize rollouts cut at the
token cap (DAPO overlong handling, Yu et al. 2025, arXiv:2503.14476;
overlong filtering, Lambert 2025, chapter Reasoning)."""

from __future__ import annotations

import pytest

from whileai.simulations.score.optimize import optimize, select_for_rl

LONG_CUT = "The order shipped on Tuesday and the carrier picked it up from the warehouse " * 4
LONG_DONE = LONG_CUT.strip() + "."


def _row(prompt, reward, final, reason="graded"):
    return {
        "prompt": prompt,
        "reward": reward,
        "reason": reason,
        "final_text": final,
        "steps": [{"tool": "get_order", "arguments": {"id": "1"}, "result": {"ok": 1}}],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final},
        ],
    }


def _rows():
    return [
        _row("a", 1, LONG_DONE),
        _row("a", 1, LONG_CUT),  # judged a pass, but cut at the cap
        _row("a", 0, "No."),
        _row("b", 1, LONG_DONE),
        _row("b", 0, LONG_CUT),
        _row("b", 0.5, LONG_CUT + " and then", reason="reply truncated at token cap"),  # advisory
    ]


def test_drop_is_the_default_and_unchanged():
    rows = _rows()
    picked, report = select_for_rl(rows, target=100)
    assert report["truncated_policy"] == "drop" and report["truncated_dropped"] == 2
    assert report["truncated_kept"] == 0 and report["truncated_penalized"] == 0
    assert all("overlong" not in r for r in picked)
    assert all("overlong" not in r for r in rows)  # inputs untouched


def test_keep_marks_overlong_and_leaves_the_reward():
    picked, report = select_for_rl(_rows(), target=100, truncated="keep")
    assert report["truncated_policy"] == "keep" and report["truncated_kept"] == 3
    assert report["truncated_dropped"] == 0
    cut = [r for r in picked if r.get("overlong")]
    assert [r["reward"] for r in cut] == [1] and cut[0]["markers"]["finished"] == 0.0
    # ask b's only finished rollout is a pass: it is unanimous under
    # "drop" and stays so under "keep", since the cut fail rides along
    # with an ask rather than supplying the contrast that keeps it
    dropped, _ = select_for_rl(_rows(), target=100)
    assert {r["prompt"] for r in picked} == {r["prompt"] for r in dropped} == {"a"}
    assert report["truncated_selected"] == 1
    assert not any(r.get("reward") == 0.5 for r in picked)  # advisory 0.5 is still unusable
    legacy, legacy_report = select_for_rl(_rows(), target=100, drop_truncated=False)
    assert legacy_report["truncated_policy"] == "keep" and len(legacy) == len(picked)


def test_penalize_turns_a_cut_rollout_into_a_failure():
    picked, report = select_for_rl(_rows(), target=100, truncated="penalize")
    assert report["truncated_penalized"] == 3 and report["truncated_dropped"] == 0
    cut = [r for r in picked if r.get("overlong")]
    assert cut and all(r["reward"] == 0 for r in cut)
    assert sorted(r["reward_before_penalty"] for r in cut) == [0, 0.5, 1]
    # group a now has two fails and a pass: still mixed, still selected
    assert {r["prompt"] for r in picked} == {"a", "b"}
    with pytest.raises(ValueError, match="truncated must be one of"):
        select_for_rl(_rows(), truncated="ignore")


def test_optimize_threads_the_policy():
    _, report = optimize(_rows(), mode="rl", truncated="penalize")
    assert report["truncated_policy"] == "penalize" and report["truncated_penalized"] == 3


# ------------------------------------------------ sign-offs and keep >= drop (#31)

EMAIL = (
    "Hi Dana,\n\nThanks for flagging the duplicate invoice. I pulled both records: "
    "INV-2201 was issued on the 3rd and INV-2207 on the 9th for the same PO, so I have "
    "voided INV-2207 and re-sent INV-2201 with the corrected due date. Nothing else on "
    "the account changed.\n\nBest,\nSales"
)


@pytest.mark.parametrize(
    "tail",
    ["Best,\nSales", "Thanks,\nAlex", "Regards\nThe Team", "-- Sam", "Cheers", "Kind regards,\nJ"],
)
def test_a_sign_off_is_a_finished_reply(tail):
    from whileai.simulations.score.hygiene import is_truncated

    body = EMAIL.rsplit("\n\nBest,\nSales", 1)[0]
    assert len(body) >= 200
    assert not is_truncated(_row("a", 1, body + "\n\n" + tail))


def test_a_cut_reply_is_still_cut():
    from whileai.simulations.score.grading import looks_finished

    assert not looks_finished(LONG_CUT)
    assert not looks_finished("Next steps:\n1. Check the")  # short last line, no sign-off
    assert not looks_finished("I found the following,\nand the")  # lowercase continuation
    assert looks_finished(LONG_DONE)


def test_keep_never_returns_fewer_rows_than_drop():
    """A kept overlong pass used to tip an ask over the band and lose the
    whole ask, so ``keep`` came back smaller than ``drop``."""
    rows = []
    for i in range(4):
        rows.append(_row("a", 1, f"Done, item {i} shipped."))
    rows.append(_row("a", 0, "No."))
    rows.append(_row("a", 1, LONG_CUT))  # p 4/5 = 0.8 in band; 5/6 = 0.83 out of it
    rows.append(_row("b", 1, "Yes."))
    rows.append(_row("b", 0, "No."))
    dropped, drop_report = select_for_rl(rows, target=100)
    kept, keep_report = select_for_rl(rows, target=100, truncated="keep")
    assert {r["prompt"] for r in dropped} == {"a", "b"}
    assert {r["prompt"] for r in kept} == {"a", "b"}
    assert len(kept) == len(dropped) + 1
    assert keep_report["truncated_kept"] == keep_report["truncated_selected"] == 1
    assert drop_report["truncated_selected"] == 0
    assert keep_report["band_groups_dropped"] == drop_report["band_groups_dropped"] == 0


def test_keep_and_penalize_carry_a_long_cut_reply_the_junk_gate_used_to_eat():
    very_long_cut = LONG_CUT * 3  # over 600 chars, no terminal punctuation
    assert len(very_long_cut.rstrip()) > 600
    rows = [
        _row("a", 1, LONG_DONE),
        _row("a", 0, "No."),
        _row("a", 1, very_long_cut),
    ]
    kept, report = select_for_rl(rows, target=100, truncated="keep")
    assert report["truncated_kept"] == report["truncated_selected"] == 1
    assert any(r.get("overlong") and r["final_text"] == very_long_cut for r in kept)
    penalized, p_report = select_for_rl(rows, target=100, truncated="penalize")
    assert p_report["truncated_penalized"] == p_report["truncated_selected"] == 1
    assert [r["reward"] for r in penalized if r.get("overlong")] == [0]
    dropped, d_report = select_for_rl(rows, target=100)
    assert d_report["truncated_selected"] == 0 and len(dropped) == 2


# ------------------------------------------------ the engine's stamp, after a re-grade


def _math_row(reward, final, *, finish_reason="stop", steps=()):
    return {
        "task_id": "t1",
        "prompt": "what is 6*7",
        "final_text": final,
        "finish_reason": finish_reason,
        "steps": list(steps),
        "reward": reward,
        "reason": "matched" if reward else "did not match",
        "judge_name": "MathEqual",
        "messages": [
            {"role": "user", "content": "what is 6*7"},
            {"role": "assistant", "content": final},
        ],
    }


def _math_group():
    """One ask, five rollouts. The first was cut at the token cap and trimmed
    back to its last sentence by the backend, then re-graded by a user
    judge: ``reason`` says "matched", the text ends on a period, and only
    the engine's ``finish_reason`` and the step's ``truncated`` say cut."""
    return [
        _math_row(
            1.0,
            "The answer is 42.",
            finish_reason="length",
            steps=[{"text": "The answer is 42. And also", "truncated": True}],
        ),
        _math_row(1, "6 times 7 is 42."),
        _math_row(1, "It comes to 42."),
        _math_row(0, "It is 41."),
        _math_row(0, "The answer is 48."),
    ]


def test_optimize_drops_a_capped_row_the_judge_re_graded_as_finished():
    picked, report = optimize(_math_group(), mode="rl")
    assert report["truncated_policy"] == "drop"
    assert report["truncated_dropped"] == 1
    assert report["truncated_selected"] == 0
    assert not any(r.get("finish_reason") == "length" for r in picked)
    assert len(picked) == 4  # the ask stays mixed without the cut row


def test_optimize_penalizes_a_capped_row_the_judge_re_graded_as_finished():
    picked, report = optimize(_math_group(), mode="rl", truncated="penalize")
    assert report["truncated_penalized"] == 1 and report["truncated_dropped"] == 0
    cut = [r for r in picked if r.get("overlong")]
    assert len(cut) == 1 and cut[0]["final_text"] == "The answer is 42."
    assert cut[0]["reward"] == 0 and cut[0]["reward_before_penalty"] == 1.0
    assert len(picked) == 5
