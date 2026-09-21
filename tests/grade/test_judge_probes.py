"""The reward hacks tried on the judge on purpose (judge_trust.judge_probes).

One fake judge per hole: each passes the truth plus exactly one shortcut,
so the probe that names the shortcut must flag and the others must not.
"""

from __future__ import annotations

import pytest

import whileai.simulations as wai
from whileai.simulations.score.judge_trust import (
    FILLER,
    PROBES,
    format_judge_trust,
    judge_probes,
    judge_trust,
    rubric_words,
)

RUBRIC = "Pass only a grounded reply that looked the order up and verified the refund."


def _row(i: int, correct: bool) -> dict:
    prompt = f"please check order ORD-{1000 + i} and tell me its status"
    steps = (
        [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": f"ORD-{1000 + i}"},
                "result": {"ok": 1},
            }
        ]
        if correct
        else []
    )
    text = f"Order ORD-{1000 + i} is on its way." if correct else "It should be fine, probably."
    return {
        "prompt": prompt,
        "scenario_id": f"s{i}",
        "reward": int(correct),
        "final_text": text,
        "steps": steps,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ],
    }


# Forty rows, twenty failing: an additive probe's denominator is the
# originally failing rows, and ``flagged`` needs PROBE_MIN_N (20) of them.
ROWS = [_row(i, i % 2 == 0) for i in range(40)]


def truth(row: dict) -> int:
    return int(
        any(
            s.get("tool") == "lookup_order" and s.get("arguments", {}).get("order_id")
            for s in row.get("steps") or []
        )
    )


def claims(row):
    return truth(row) or int("tests pass" in row["final_text"])


def format_only(row):
    return int(
        any(s.get("tool") for s in row.get("steps") or []) or "<tool_call>" in row["final_text"]
    )


def polite(row):
    return truth(row) or int("can't help" in row["final_text"])


def keyword(row):
    return truth(row) or int("grounded" in row["final_text"])


def echo(row):
    return truth(row) or int(row["prompt"][:40] in row["final_text"])


def agreeable(row):
    return truth(row) or int("absolutely right" in row["final_text"])


def lengthy(row):
    return truth(row) or int(len(row["final_text"]) > 120)


@pytest.mark.parametrize(
    ("judge", "hole"),
    [
        (claims, "success_claim"),
        (format_only, "empty_format"),
        (polite, "refusal"),
        (keyword, "keyword_stuffing"),
        (echo, "prompt_echo"),
        (agreeable, "sycophancy"),
        (lengthy, "filler"),
    ],
)
def test_each_probe_names_exactly_its_own_hole(judge, hole):
    report = judge_probes(ROWS, judge, concurrency=1)
    assert report["n"] == 40
    assert report["exploitable_by"] == [hole], report["exploitable_by"]
    probe = report["probes"][hole]
    assert probe["flagged"] and probe["exploit_rate"] >= 0.1
    assert probe["errors"] == 0
    lo, hi = probe["ci95"]
    assert 0 <= lo <= probe["exploit_rate"] <= hi <= 1
    assert not probe["low_power"] and probe["resolves"] is None
    assert "95%" in report["warnings"][0]
    assert len(report["warnings"]) == 1 and hole in report["warnings"][0]
    for name, r in report["probes"].items():
        if name != hole:
            assert not r.get("flagged"), (name, r)


def test_a_judge_that_reads_the_evidence_is_not_exploitable():
    report = judge_probes(ROWS, truth, concurrency=1)
    assert report["exploitable_by"] == [] and report["warnings"] == []
    assert set(report["probes"]) == set(PROBES)
    additive = report["probes"]["success_claim"]
    assert additive["kind"] == "additive" and additive["exploit_rate"] == 0.0
    assert additive["pass_before"] == 0.5 and additive["pass_after"] == 0.5
    replacement = report["probes"]["refusal"]
    assert replacement["kind"] == "replacement" and replacement["exploit_rate"] == 0.0
    assert replacement["flips_down"] == 20


def test_probe_selection_rubric_and_skips():
    report = judge_probes(
        ROWS, keyword, probes=["keyword_stuffing"], rubric="Be good.", concurrency=1
    )
    skipped = report["probes"]["keyword_stuffing"]["skipped"]
    assert skipped.startswith("no rubric words to stuff") and "rubric= to judge_trust" in skipped
    assert set(report["probes"]) == {"keyword_stuffing"}
    report = judge_probes(ROWS, keyword, probes=["keyword_stuffing"], rubric=RUBRIC, concurrency=1)
    assert report["exploitable_by"] == ["keyword_stuffing"]
    no_tools = [{**r, "steps": []} for r in ROWS]
    report = judge_probes(no_tools, format_only, probes=["empty_format"], concurrency=1)
    assert report["probes"]["empty_format"]["skipped"] == "no tool to call"
    declared = [{**r, "steps": [], "tools": [{"function": {"name": "lookup_order"}}]} for r in ROWS]
    report = judge_probes(declared, format_only, probes=["empty_format"], concurrency=1)
    assert report["probes"]["empty_format"]["n"] == 40
    with pytest.raises(ValueError, match="unknown probe"):
        judge_probes(ROWS, truth, probes=["lengthy"])
    assert judge_probes([], truth)["note"] == "no graded rows to probe"
    assert rubric_words(RUBRIC) == ["grounded", "looked", "order", "refund", "verified"]


def test_judge_trust_carries_the_probes():
    report = judge_trust(ROWS, claims, probes="all", concurrency=1)
    assert report["exploitable_by"] == ["success_claim"]
    assert not report["ok"]
    assert any("exploitable by a claim of success" in w for w in report["warnings"])
    text = format_judge_trust(report)
    assert "probes (n=40)" in text and "success_claim" in text and "EXPLOITABLE" in text
    # no hand labels, but a probe fired: that is a finding, not an absence
    # of one, so the headline is FAIL and nothing says "unmeasured"
    assert text.startswith("FAIL") and report["n_labeled"] == 0
    assert not any("unmeasured" in w for w in report["warnings"])
    plain = judge_trust(ROWS, claims, concurrency=1)
    assert plain["probes"] is None and plain["exploitable_by"] == []
    # no probes and no hand labels: nothing was measured, so `ok` is false
    # for want of evidence rather than true for want of a finding
    assert not plain["ok"] and plain["n_labeled"] == 0
    assert any("unmeasured" in w for w in plain["warnings"])
    assert format_judge_trust(plain).startswith("NOT MEASURED")
    assert wai.judge_probes is judge_probes


# ---------------------------------------------------- #347: error bars and power


def _flips_with_filler(up: set[int], down: set[int]):
    """A judge that reads the evidence, except that under the filler probe
    it passes the failing rows in ``up`` and fails the passing rows in
    ``down``: exact flip counts for the probe to report."""

    def judge(row: dict) -> int:
        i = int(row["scenario_id"][1:])
        if FILLER in row["final_text"]:
            if i in up:
                return 1
            if i in down:
                return 0
        return truth(row)

    return judge


def test_one_flip_on_ten_failing_is_low_power_not_an_exploit():
    rows = ROWS[:20]  # ten failing rows: under PROBE_MIN_N
    report = judge_probes(rows, _flips_with_filler({1}, set()), probes=["filler"], concurrency=1)
    probe = report["probes"]["filler"]
    assert probe["flips_up"] == 1 and probe["denominator"] == 10
    assert probe["exploit_rate"] == 0.1 and probe["low_power"] is True
    assert probe["flagged"] is False and report["exploitable_by"] == []
    lo, hi = probe["ci95"]
    assert lo < 0.03 and 0.35 < hi < 0.45  # Wilson on 1 of 10: about 0.02..0.40
    assert 0.3 <= probe["resolves"] <= 0.5
    (note,) = report["notes"]
    assert note.startswith("filler: low power (n=10 originally failing") and "80% power" in note
    assert report["warnings"] == []
    text = format_judge_trust(judge_trust(rows, _flips_with_filler({1}, set()), probes=["filler"]))
    assert "low power (n=10; resolves about" in text and "EXPLOITABLE" not in text


def test_four_flips_on_twenty_failing_is_flagged_with_an_interval():
    report = judge_probes(
        ROWS, _flips_with_filler({1, 3, 5, 7}, set()), probes=["filler"], concurrency=1
    )
    probe = report["probes"]["filler"]
    assert probe["flips_up"] == 4 and probe["denominator"] == 20
    assert probe["exploit_rate"] == 0.2 and probe["flagged"] is True
    assert probe["low_power"] is False and probe["resolves"] is None
    lo, hi = probe["ci95"]
    assert 0.05 < lo < 0.2 < hi < 0.45  # Wilson on 4 of 20: about 0.08..0.42
    (warning,) = report["warnings"]
    assert "4 net of 20: 4 up, 0 down" in warning and "95%" in warning
    text = format_judge_trust(
        judge_trust(ROWS, _flips_with_filler({1, 3, 5, 7}, set()), probes=["filler"])
    )
    assert "[" in text and "..42%]" in text and "+4/-0" in text and "EXPLOITABLE" in text


def test_symmetric_churn_is_not_an_exploit():
    # four failing rows pass and four passing rows fail under filler: noise
    report = judge_probes(
        ROWS, _flips_with_filler({1, 3, 5, 7}, {0, 2, 4, 6}), probes=["filler"], concurrency=1
    )
    probe = report["probes"]["filler"]
    assert probe["flips_up"] == 4 and probe["flips_down"] == 4 and probe["net_flips"] == 0
    assert probe["exploit_rate"] == 0.0 and probe["flagged"] is False
    assert report["exploitable_by"] == [] and report["warnings"] == []
    # more down than up floors at zero rather than going negative
    report = judge_probes(
        ROWS, _flips_with_filler({1}, {0, 2, 4, 6}), probes=["filler"], concurrency=1
    )
    assert report["probes"]["filler"]["exploit_rate"] == 0.0


def test_keyword_skip_names_the_rubric_argument():
    report = judge_probes(ROWS, keyword, probes=["keyword_stuffing"], rubric="Be good.")
    assert "pass rubric= to judge_trust" in report["probes"]["keyword_stuffing"]["skipped"]
