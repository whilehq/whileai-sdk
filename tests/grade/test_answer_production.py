"""#297: a before/after where one arm mostly never answered is not a
comparison. ``pass_at`` reports answer production per side and
``delta_report`` fails the manufactured win and names the mechanism."""

from __future__ import annotations

import whileai.simulations as wai
from whileai.simulations.score.delta import delta_report, format_delta_report

CUT_OFF = (
    "<think> Okay, the customer is asking about order 4412 and the policy says to look up "
    "before answering, so first I should work out which order they mean and whether"
)


def _rows(n: int, *, text: str, reward: int, unanswered: int = 0) -> list[dict]:
    out = []
    for i in range(n):
        cut = i < unanswered
        out.append(
            {
                "scenario_id": f"s{i}",
                "rollout_index": 0,
                "prompt": f"p{i}",
                "final_text": CUT_OFF if cut else text,
                "reward": 0 if cut else reward,
                "judge_status": "ok",
                "finish_reason": "length" if cut else "stop",
            }
        )
    return out


def test_pass_at_config_reports_answer_production():
    before = _rows(30, text="fine.", reward=1, unanswered=28)
    cfg = wai.pass_at(before).config
    assert cfg["answered_share"] == round(2 / 30, 4)
    assert cfg["unclosed_think_share"] == round(28 / 30, 4)
    assert "93% of rows have no spoken reply" in str(wai.pass_at(before))
    clean = _rows(30, text="fine.", reward=1)
    assert wai.pass_at(clean).config["answered_share"] == 1.0
    assert wai.pass_at(clean).config["unclosed_think_share"] == 0.0
    assert "no spoken reply" not in str(wai.pass_at(clean))
    # a row from another harness carries messages only; the last assistant
    # message with words in it is the reply
    row = dict(clean[0], final_text="")
    row["messages"] = [
        {"role": "user", "content": "where is it"},
        {"role": "assistant", "content": "<think> hmm </think> Shipped."},
    ]
    assert wai.pass_at([row]).config["answered_share"] == 1.0
    assert wai.pass_at([]).config["answered_share"] is None


def test_delta_report_fails_a_win_over_replies_one_side_never_produced():
    before = _rows(30, text="fine.", reward=0, unanswered=28)
    after = _rows(30, text="Order 4412 shipped on the 15th.", reward=1)
    report = delta_report(before, after)
    assert report["metrics"]["pass_at_1"]["verdict"] == "b_better"
    assert report["ok"] is False and report["not_comparable"] == ["answered"]
    assert "answered" not in report and "unanswered_asymmetric" not in report
    assert report["config"]["before"]["answered_share"] == round(2 / 30, 4)
    assert report["config"]["after"]["answered_share"] == 1.0
    note = next(w for w in report["warnings"] if w.startswith("NOT COMPARABLE"))
    assert "93% of before rows and 0% of after rows have no spoken reply" in note
    assert "two-proportion test p=" in note and "gap 93.3% against 10% with no run_std" in note
    assert "93% of before and 0% of after replies end inside an unclosed <think>" in note
    assert "reasoning base" in note and "reasoning-suppressed adapter" in note
    assert "agent_max_tokens=" in note and "thinking=" in note
    text = format_delta_report(report)
    assert "! NOT COMPARABLE" in text and "answered: 6.7% before, 100.0% after" in text


def test_delta_report_reads_a_clean_pair_as_before():
    before = _rows(30, text="fine.", reward=0)
    after = _rows(30, text="Order 4412 shipped on the 15th.", reward=1)
    report = delta_report(before, after)
    assert report["ok"] is True and report["not_comparable"] == []
    assert report["config"]["before"]["answered_share"] == 1.0
    assert not any("no spoken reply" in w for w in report["warnings"])
    # both sides short of replies by about the same amount: the test sees
    # no difference (p is nowhere near 0.01), so nothing is said
    both = delta_report(
        _rows(30, text="fine.", reward=0, unanswered=10),
        _rows(30, text="ok.", reward=1, unanswered=9),
    )
    assert both["not_comparable"] == []
    assert not any("no spoken reply" in w for w in both["warnings"])
    # rows that never said what they replied do not pretend
    old = [{k: v for k, v in r.items() if k != "final_text"} for r in before]
    assert delta_report(old, after)["config"]["before"]["answered_share"] is None
    # a side that stopped on a tool call, with no reasoning and no cap
    # cut, is still not comparable, and the note sends the reader to
    # finish_reason instead of blaming thinking
    quiet = [
        dict(r, final_text="", finish_reason="tool") if i < 20 else r for i, r in enumerate(before)
    ]
    report = delta_report(quiet, after)
    assert report["ok"] is False and report["not_comparable"] == ["answered"]
    note = next(w for w in report["warnings"] if w.startswith("NOT COMPARABLE"))
    assert "finish_reason" in note and "reasoning base" not in note


def test_a_small_but_real_gap_warns_and_fails_only_over_the_band():
    # 5% of before rows never answered against 0% after, on 300 rows a side:
    # p is about 1e-4, so the gap is real. Under 10 points with no run_std
    # it is a warning; over a re-run band of 2 points it fails
    before = _rows(300, text="fine.", reward=0, unanswered=15)
    after = _rows(300, text="Order 4412 shipped on the 15th.", reward=1)
    warned = delta_report(before, after)
    assert warned["ok"] is True and warned["not_comparable"] == []
    note = next(w for w in warned["warnings"] if "no spoken reply" in w)
    assert not note.startswith("NOT COMPARABLE") and "p=" in note
    assert "gap 5.0% against 10% with no run_std" in note
    failed = delta_report(before, after, run_std=0.01)
    assert failed["ok"] is False and failed["not_comparable"] == ["answered"]
    note = next(w for w in failed["warnings"] if w.startswith("NOT COMPARABLE"))
    assert "gap 5.0% against the re-run band 0.0" in note
    # the same 5-point gap on 30 rows a side is one or two rows, and that
    # is not evidence: p is far above 0.01, nothing is said
    small = delta_report(
        _rows(30, text="fine.", reward=0, unanswered=2),
        _rows(30, text="Order 4412 shipped on the 15th.", reward=1),
    )
    assert small["not_comparable"] == []
    assert not any("no spoken reply" in w for w in small["warnings"])
