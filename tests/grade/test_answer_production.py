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
    assert report["ok"] is False
    assert report["unanswered_asymmetric"] is True
    assert report["answered"] == {"before": round(2 / 30, 4), "after": 1.0}
    note = next(w for w in report["warnings"] if w.startswith("MANUFACTURED"))
    assert "93% of before rows and 0% of after rows have no spoken reply" in note
    assert "93% of before and 0% of after replies end inside an unclosed <think>" in note
    assert "reasoning base" in note and "reasoning-suppressed adapter" in note
    assert "agent_max_tokens=" in note and "thinking=" in note
    assert "! MANUFACTURED" in format_delta_report(report)


def test_delta_report_reads_a_clean_pair_as_before():
    before = _rows(30, text="fine.", reward=0)
    after = _rows(30, text="Order 4412 shipped on the 15th.", reward=1)
    report = delta_report(before, after)
    assert report["ok"] is True
    assert report["unanswered_asymmetric"] is False
    assert report["answered"] == {"before": 1.0, "after": 1.0}
    assert not any(w.startswith("MANUFACTURED") for w in report["warnings"])
    # both sides short of replies by the same amount is a different
    # problem (both cut), not a one-sided one
    both = delta_report(
        _rows(30, text="fine.", reward=0, unanswered=10),
        _rows(30, text="ok.", reward=1, unanswered=9),
    )
    assert both["unanswered_asymmetric"] is False
    # rows that never said what they replied do not pretend
    old = [{k: v for k, v in r.items() if k != "final_text"} for r in before]
    assert delta_report(old, after)["answered"]["before"] is None
    # a side that stopped on a tool call, with no reasoning and no cap
    # cut, is still not comparable, and the note sends the reader to
    # finish_reason instead of blaming thinking
    quiet = [
        dict(r, final_text="", finish_reason="tool") if i < 20 else r for i, r in enumerate(before)
    ]
    report = delta_report(quiet, after)
    assert report["ok"] is False and report["unanswered_asymmetric"] is True
    note = next(w for w in report["warnings"] if w.startswith("MANUFACTURED"))
    assert "finish_reason" in note and "reasoning base" not in note
