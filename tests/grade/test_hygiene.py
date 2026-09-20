"""Duplicate, length, and reward-correlation hygiene (score/hygiene.py)."""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations.score.grading import looks_finished
from whileai.simulations.score.hygiene import (
    HACK_THRESHOLD,
    dedupe_groups,
    hygiene_warnings,
    is_truncated,
    length_report,
    near_duplicate_prompts,
    pearson,
    reply_length,
    reward_correlations,
)
from whileai.simulations.score.optimize import select_for_rl, select_for_sft


def _row(prompt: str, reward, final: str, *, tools: int = 1, turns: int = 1) -> dict:
    steps = [
        {"tool": "get_issue", "arguments": {"number": i}, "result": {"status": "ok"}}
        for i in range(tools)
    ]
    messages = [{"role": "user", "content": prompt}]
    for _ in range(turns - 1):
        messages.append({"role": "assistant", "content": "Looking."})
    messages.append({"role": "assistant", "content": final})
    return {
        "prompt": prompt,
        "reward": reward,
        "final_text": final,
        "steps": steps,
        "messages": messages,
    }


def test_reply_length_counts_assistant_text_only():
    row = _row("a", 1, "Issue 4412 is open.", turns=2)
    assert reply_length(row) == len("Looking.") + len("Issue 4412 is open.")
    row["final_text"] = "Different final."
    assert reply_length(row) == len("Looking.") + len("Issue 4412 is open.") + len(
        "Different final."
    )


def test_is_truncated_by_reason_or_missing_terminal_punctuation():
    assert not is_truncated(_row("a", 1, "Issue 4412 is open."))
    assert not is_truncated(_row("a", 1, "Total is 42"))  # short, benefit of the doubt
    assert is_truncated(_row("a", 1, "The order was located and the refund " * 8))
    cut = _row("a", 0, "Done.")
    cut["reason"] = "reply truncated at token cap"
    assert is_truncated(cut)


def test_dedupe_groups_drops_repeats_and_counts_conflicts():
    rows = [
        _row("a", 1, "Issue 1 is open."),
        _row("a", 1, "Issue 1 is open."),  # same trajectory, same reward
        _row("a", 0, "Issue 1 is open."),  # same trajectory, judge disagreed
        _row("a", 0, "Issue 1 is closed."),
        _row("b", 1, "Issue 1 is open."),  # other ask: not a duplicate
    ]
    kept, report = dedupe_groups(rows)
    assert len(kept) == 3
    assert report["n_dropped"] == 2 and report["groups_affected"] == 1
    assert report["conflicting_rewards"] == 1
    assert report["duplicate_rate"] == pytest.approx(0.4)


def test_near_duplicate_prompts_flags_high_jaccard_only():
    rows = [
        _row("please look up order ORD-4017 for me", 1, "ok."),
        _row("please look up order ORD-4017 for me now", 1, "ok."),
        _row("cancel my subscription", 1, "ok."),
    ]
    report = near_duplicate_prompts(rows)
    assert report["n_prompts"] == 3 and report["n_pairs"] == 1
    assert report["examples"][0]["jaccard"] >= 0.8
    assert near_duplicate_prompts(rows, threshold=0.99)["n_pairs"] == 0


def test_length_report_flags_wide_spread_groups():
    long = "The order was located and the refund was issued in full. " * 10
    rows = [_row("a", 1, "Issue 1 is open.")] * 3 + [_row("a", 1, long)]
    rows += [_row("b", 1, "Issue 2 is open.")] * 4
    report = length_report(rows, max_spread=4.0)
    assert report["n_groups_multi"] == 2
    assert report["n_groups_wide_spread"] == 1
    assert report["n_truncated"] == 0


def test_pearson_and_reward_correlations_flag_length_and_tool_hacks():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    assert pearson([1, 2], [1, 2]) is None
    # Reward rises with reply length and falls with tool calls.
    rows = []
    for i in range(12):
        long = i % 2 == 1
        rows.append(
            _row(
                f"ask {i // 4}",
                int(long),
                ("Detailed answer. " * 12) if long else "Short.",
                tools=1 if long else 3,
            )
        )
    report = reward_correlations(rows)
    assert report["n_graded"] == 12 and report["threshold"] == HACK_THRESHOLD
    assert report["correlations"]["reply_length"] > 0.9
    assert report["correlations"]["tool_calls"] < -0.9
    assert set(report["flagged"]) == {"reply_length", "tool_calls"}
    lines = hygiene_warnings(correlations=report)
    assert any("pays for reply length" in line for line in lines)
    assert any("punishes tool calls" in line for line in lines)
    assert hygiene_warnings() == []


def test_select_for_rl_dedupes_and_drops_truncated_and_reports_scan():
    cut = "The order was located and the refund was issued in full and " * 6
    rows = [
        _row("a", 1, "Issue 1 is open."),
        _row("a", 1, "Issue 1 is open."),  # duplicate
        _row("a", 0, "Issue 1 is closed."),
        _row("a", 0, "Issue 1 is closed!"),
        _row("a", 1, cut),  # truncated
        _row("b", 1, "Issue 2 is open."),
        _row("b", 0, "Issue 2 is closed."),
        _row("b", 1, "Issue 2 is still open."),
        _row("b", 0, "Issue 2 was closed."),
    ]
    picked, report = select_for_rl(rows, target=100)
    assert report["duplicates"]["n_dropped"] == 1
    assert report["truncated_dropped"] == 1
    assert len(picked) == 7
    assert "correlations" in report and "length" in report
    assert isinstance(report["hygiene_warnings"], list)

    raw, raw_report = select_for_rl(rows, target=100, dedupe=False, drop_truncated=False)
    assert len(raw) == 9 and raw_report["duplicates"]["n_dropped"] == 0
    assert raw_report["truncated_dropped"] == 0


def test_group_collapsed_by_hygiene_is_dropped_not_kept_as_single():
    # Four identical rollouts, one of them graded differently: after
    # dedupe one row remains, and a group that had four cannot become a
    # single that always survives.
    rows = [_row("a", 1, "Issue 1 is open.")] * 3 + [_row("a", 0, "Issue 1 is open.")]
    rows += [_row("b", 1, "Issue 2 is open."), _row("b", 0, "Issue 2 is closed.")]
    picked, report = select_for_rl(rows, target=100)
    assert {r["prompt"] for r in picked} == {"b"}
    assert report["collapsed_groups_dropped"] == 1
    assert report["duplicates"]["n_dropped"] == 3


def test_select_for_sft_notes_low_completions_per_prompt():
    rows = [_row("a", 1, "Issue 1 is open."), _row("b", 1, "Issue 2 is open.")]
    _picked, report = select_for_sft(rows, target=10)
    assert report["completions_per_prompt_max"] == 1
    assert "chapter Rejection Sampling" in report["note"]
    assert report["completions_per_prompt_mean"] == 1.0
    assert report["selection_effective"] == "pass_filter"
    many = [_row("a", 1, f"Issue {i} is open.") for i in range(12)]
    _picked, report = select_for_sft(many, target=10)
    assert report["completions_per_prompt_max"] == 12 and "note" not in report
    assert report["selection_effective"] == "top_per_prompt"


def test_select_for_sft_note_fires_on_mean_not_max():
    """One well-sampled prompt must not silence the note for a pool of singles.

    The max is the most optimistic statistic in the pool. A measured pool ran
    mean k=1.07 and produced a null; a max-based check would not have warned.
    """
    rows = [_row("fat", 1, f"Issue {i} is open.") for i in range(12)]
    rows += [_row(f"p{j}", 1, "Issue 1 is open.") for j in range(500)]
    _picked, report = select_for_sft(rows, target=100)
    assert report["completions_per_prompt_max"] == 12
    assert report["completions_per_prompt_mean"] < 2
    assert report["prompts_with_one_completion"] == 500
    assert "chapter Rejection Sampling" in report["note"]
    assert report["selection_effective"] == "pass_filter"


def test_publish_gate_reports_hygiene_without_dropping():
    rows = [
        _row("a", 1, "Issue 1 is open."),
        _row("a", 1, "Issue 1 is open."),
        _row("a", 0, "Issue 1 is closed."),
        _row("a", 0, "Issue 1 was closed."),
    ]
    report = wai.publish_gate(rows, mode="rl")
    assert report["ok"]
    assert report["duplicates"]["n_dropped"] == 1
    assert any("duplicate rollout(s)" in w and 'select(mode="rl")' in w for w in report["warnings"])
    assert "correlations" in report and "near_duplicate_prompts" in report
    assert len(rows) == 4  # nothing removed


def test_judge_prompts_are_length_neutral():
    from whileai.simulations.score import grade_llm, llm_judge

    assert "length" in grade_llm.JUDGE_SYSTEM.lower()
    assert "length" in llm_judge.JUDGE_SYSTEM.lower()


def test_tool_calls_reads_platform_tool_trace_rows():
    from whileai.simulations.score.hygiene import tool_calls

    pulled = {
        "prompt": "a",
        "reward": 1,
        "final_text": "Done.",
        "tool_trace": [
            {"tool": "lookup_order", "input": {"id": 1}, "output": {"ok": 1}},
            {"tool": "create_refund", "input": {}, "output": {}},
        ],
    }
    assert tool_calls(pulled) == 2
    assert tool_calls({"prompt": "a", "steps": [{"tool": "x"}]}) == 1
    assert tool_calls({"prompt": "a"}) == 0


# --- #212: a reply whose answer is a code block ends on a fence, not a period ---

_SQL = "```sql\nSELECT " + "account_name, " * 20 + "balance FROM account_details\n```"


def test_a_closed_code_fence_is_a_finished_reply():
    """A text-to-SQL set lost 486 of 1,556 rollouts to this: every answer is
    one fenced block, and a closed fence carries no terminal punctuation."""
    assert len(_SQL) >= 200
    assert looks_finished(_SQL)
    assert not is_truncated(_row("a", 1, _SQL))


def test_an_unclosed_code_fence_is_still_truncated():
    """The unbalanced fence is exactly the cut the rule exists to catch."""
    cut = _SQL[: -len("\n```")]
    assert len(cut) >= 200
    assert not looks_finished(cut)
    assert is_truncated(_row("a", 1, cut))


def test_prose_after_a_closed_fence_is_judged_on_the_prose():
    assert looks_finished(_SQL + "\n\nThat returns one row per account.")
    assert not looks_finished(_SQL + "\n\nThat returns one row per account and then we")


def test_a_json_object_answer_is_finished():
    """``]`` already counted; ``{...}`` did not, so JSON answers read as cut."""
    blob = '{"rows": [' + "1, " * 80 + '2], "ok": true}'
    assert len(blob) >= 200
    assert looks_finished(blob)
    assert not is_truncated(_row("a", 1, blob))


def test_a_grader_that_says_truncated_still_wins_over_the_text():
    row = _row("a", 1, _SQL)
    row["reason"] = "reply truncated at token cap"
    assert is_truncated(row)


def test_rl_selection_keeps_code_fenced_rows():
    """The reported symptom: every fenced answer counted as truncated, so RL
    selection dropped them all and came back with almost nothing."""
    rows = []
    for group in "ab":
        for i in range(4):
            # distinct SQL per row so dedupe is not what is being measured
            rows.append(_row(group, i % 2, _SQL.replace("balance FROM", f"b{group}{i} FROM")))
    picked, report = select_for_rl(rows, target=100)
    assert report["truncated_dropped"] == 0
    assert len(picked) == 8
