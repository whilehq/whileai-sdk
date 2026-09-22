"""Over-optimization signatures as markers (Lambert 2025, chapters
Over-optimization and Model Character and Products)."""

from __future__ import annotations

import json

import whileai.simulations as wai
from whileai.simulations.score.hygiene import reward_correlations
from whileai.simulations.score.style import (
    STYLE_MARKERS,
    assistant_text,
    refusal_report,
    style_markers,
    style_report,
)


def _row(prompt, final, reward=1, turns=()):
    messages = [{"role": "user", "content": prompt}]
    for turn in turns:
        messages.append({"role": "assistant", "content": turn})
    messages.append({"role": "assistant", "content": final})
    return {
        "prompt": prompt,
        "final_text": final,
        "reward": reward,
        "steps": [{"tool": "get_order", "arguments": {"id": "1"}, "result": {"ok": 1}}],
        "messages": messages,
    }


def test_style_markers_stamp_one_per_signature_and_keep_others():
    rows = [
        _row("a", "Certainly! Your order shipped. I hope this helps!"),
        _row("b", "Your order shipped yesterday."),
        _row("c", "It depends on the carrier, but generally speaking two days."),
        _row("d", "You're absolutely right, great point. Refunded."),
        _row("e", "I'm sorry, I can't help with that."),
    ]
    rows[1]["markers"] = {"on_task": 1.0}
    style_markers(rows)
    assert set(STYLE_MARKERS) <= set(rows[0]["markers"])
    assert rows[0]["markers"]["no_boilerplate"] == 0.0
    assert rows[1]["markers"] == {"on_task": 1.0, **{k: 1.0 for k in STYLE_MARKERS}}
    assert rows[2]["markers"]["no_hedging"] == 0.0 and rows[2]["markers"]["no_boilerplate"] == 1.0
    assert rows[3]["markers"]["no_sycophancy"] == 0.0
    assert rows[4]["markers"]["no_apology"] == 0.0 and rows[4]["markers"]["answered"] == 0.0


def test_assistant_text_reads_every_assistant_turn():
    row = _row("a", "Done.", turns=("Certainly! Let me look.",))
    assert "certainly!" in assistant_text(row)
    style_markers([row])
    assert row["markers"]["no_boilerplate"] == 0.0


def test_custom_phrases_extend_the_table():
    rows = [_row("a", "Per our brand voice, absolutely fabulous.")]
    style_markers(rows, phrases={"no_brand_voice": ["per our brand voice"]})
    assert rows[0]["markers"]["no_brand_voice"] == 0.0
    assert rows[0]["markers"]["no_boilerplate"] == 1.0


def test_style_report_flags_a_reward_that_pays_for_hedging():
    rows = []
    for i in range(12):
        hedged = i % 2 == 0
        final = "It depends, but the order shipped." if hedged else "The order shipped."
        rows.append(_row(f"p{i}", final, reward=1 if hedged else 0))
    before = [dict(r) for r in rows]
    report = style_report(rows, n_boot=50)
    assert rows == before  # not mutated
    hedging = report["markers"]["no_hedging"]
    assert hedging["hits"] == 6 and hedging["clean"] == 0.5
    assert hedging["reward_corr"] == 1.0 and hedging["flagged"] is True
    assert hedging["top_phrases"][0] == ("it depends", 6)
    assert any("pays for hedging" in w for w in report["warnings"])
    assert report["markers"]["no_boilerplate"]["hits"] == 0
    assert "flagged" not in report["markers"]["no_boilerplate"]
    assert report["n_graded"] == 12


def test_style_report_prints_itself_and_stays_a_dict():
    rows = [_row(f"p{i}", "It depends." if i % 2 else "Shipped.", reward=i % 2) for i in range(10)]
    report = style_report(rows, n_boot=50)
    text = str(report)
    assert text.startswith("style 10 rows, 10 graded")
    assert "no_hedging" in text and "clean 0.500" in text
    assert report["markers"]["no_hedging"]["hits"] == 5  # every key still reads
    assert json.loads(json.dumps(report))["n"] == 10
    assert report == dict(report)
    assert repr(report) == "StyleReport(n=10, n_graded=10)"
    assert "<pre>" in report._repr_html_()


def test_style_report_names_the_markers_it_did_not_stamp():
    """A row clean on every style marker can still have faked the work:
    both markers for that live in other families (#760)."""
    report = style_report([_row("a", "Shipped.")], n_boot=20)
    trace = report["not_stamped"]["trace_markers"]
    assert "no_secrets" in trace and "reported_failure" in trace
    assert report["not_stamped"]["mark_grounding"] == ["argument_grounding"]
    line = next(ln for ln in str(report).splitlines() if ln.startswith("not stamped here"))
    assert "trace_markers(rows)" in line and "argument_grounding" in line
    # a marker the caller stamps here is not listed as missing
    mine = style_report(
        [_row("a", "Shipped.")], phrases={"no_secrets": ["hidden state"]}, n_boot=20
    )
    assert "no_secrets" not in mine["not_stamped"]["trace_markers"]


def test_a_marker_that_never_fires_has_no_interval_and_says_why():
    rows = [_row(f"p{i}", "Shipped.", reward=1) for i in range(6)]
    report = style_report(rows, n_boot=20)
    entry = report["markers"]["no_hedging"]
    assert entry["clean"] == 1.0 and entry["ci95"] is None and entry["degenerate"] is True
    assert "no interval" in str(report)
    note = next(n for n in report["notes"] if "came out the same" in n)
    assert "no_hedging" in note and "must_not_regress" in note
    assert report["warnings"] == []  # warnings stays the reward-pays-for-a-tic list


def test_refusal_report_on_a_benign_set():
    rows = [
        _row("what time is it", "I'm unable to check the time."),
        _row("say hi", "Hi!"),
        _row("add 2 and 2", "I can't help with that."),
        _row("spell cat", "c-a-t"),
    ]
    report = refusal_report(rows)
    assert report["n"] == 4 and report["n_refused"] == 2 and report["refusal_rate"] == 0.5
    lo, hi = report["ci95"]
    assert 0.0 < lo < 0.5 < hi < 1.0
    assert report["examples"][0]["prompt"] == "what time is it"
    assert ("i'm unable to", 1) in report["top_phrases"]
    assert refusal_report([])["refusal_rate"] is None


def test_reward_correlations_include_style_features():
    rows = []
    for i in range(10):
        boiler = i % 2 == 0
        final = "Certainly! Shipped." if boiler else "Shipped."
        rows.append(_row(f"p{i}", final, reward=1 if boiler else 0))
    report = reward_correlations(rows)
    assert report["correlations"]["boilerplate"] == 1.0
    assert report["flagged"]["boilerplate"] == 1.0
    assert set(report["correlations"]) >= {
        "reply_length",
        "tool_calls",
        "assistant_turns",
        "boilerplate",
        "hedging",
        "sycophancy",
        "refusal",
    }


def test_public_surface():
    for name in ("style_markers", "style_report", "refusal_report"):
        assert name in wai.__all__
        assert callable(getattr(wai, name))
    summary = wai.marker_summary(style_markers([_row("a", "Certainly!"), _row("b", "Ok.")]))
    assert summary["no_boilerplate"]["mean"] == 0.5
