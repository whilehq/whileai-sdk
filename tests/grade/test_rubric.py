"""Rubrics as objects, the per-criterion judge, and the writer (Lambert
2025, chapter Synthetic Data and Distillation)."""

from __future__ import annotations

import json
import threading

import pytest

import whileai.simulations as wai
from whileai.simulations import schema
from whileai.simulations.data import export_row
from whileai.simulations.export import training_rows
from whileai.simulations.score import rubric as R
from whileai.simulations.score.judging import run_judge

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    }
]

CRITERIA = [
    {
        "title": "Looks the order up",
        "description": "Essential Criteria: calls get_order before answering.",
        "weight": 5,
    },
    {
        "title": "States the status",
        "description": "Important Criteria: names the shipping status.",
        "weight": 3,
    },
    {
        "title": "Offers next step",
        "description": "Optional Criteria: offers tracking or a follow-up.",
        "weight": 1,
    },
    {
        "title": "Invents an id",
        "description": "Pitfall Criteria: makes up an order id.",
        "weight": -2,
    },
]


def _row(prompt="where is order 4412", final="Order 4412 shipped.", reward=None):
    row = {
        "prompt": prompt,
        "final_text": final,
        "steps": [
            {"tool": "get_order", "arguments": {"id": "4412"}, "result": {"status": "shipped"}}
        ],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final},
        ],
        "scenario_id": "s1",
    }
    if reward is not None:
        row["reward"] = reward
    return row


def test_criterion_from_dict_reads_the_book_shapes():
    c = [R.Criterion.from_dict(d) for d in CRITERIA]
    assert [x.kind for x in c] == ["hard", "principle", "principle", "pitfall"]
    assert c[3].weight == 2.0  # magnitude
    assert R.Criterion.from_dict({"title": "Five items [Hard Rule]"}).kind == "hard"
    assert R.Criterion.from_dict({"title": "Clear", "weight": -1}).kind == "pitfall"
    with pytest.raises(ValueError, match="kind"):
        R.Criterion(title="x", kind="soft")
    with pytest.raises(ValueError, match="positive"):
        R.Criterion(title="x", weight=0)


def test_rubric_score_hard_rules_principles_and_pitfalls():
    rubric = R.Rubric.from_dict(CRITERIA)
    assert len(rubric.version) == 12 and rubric.version == R.Rubric.from_dict(CRITERIA).version
    full = rubric.score(
        {"Looks the order up": True, "States the status": True, "Offers next step": True}
    )
    assert full["reward"] == 1.0 and full["unanswered"] == ["Invents an id"]
    partial = rubric.score(
        {"looks_the_order_up": True, "states_the_status": True, "invents_an_id": False}
    )
    assert partial["reward"] == 0.75 and partial["missed"] == ["Offers next step"]
    hit = rubric.score(
        {"Looks the order up": True, "States the status": True, "Invents an id": True}
    )
    assert hit["reward"] == 0.25 and hit["pitfalls_hit"] == ["Invents an id"]
    failed = rubric.score(
        {"Looks the order up": False, "States the status": True, "Offers next step": True}
    )
    assert failed["reward"] == 0.0 and failed["hard_failed"] == ["Looks the order up"]
    only_hard = R.Rubric((R.Criterion("Says hi", kind="hard"),))
    assert only_hard.score({"Says hi": True})["reward"] == 1.0
    assert "Hard rule" in rubric.checklist() and "Pitfall" in rubric.checklist()
    edited = R.Rubric.from_dict([*CRITERIA[:3], {**CRITERIA[3], "weight": -1}])
    assert edited.version != rubric.version
    with pytest.raises(ValueError, match="at least one"):
        R.Rubric(())
    with pytest.raises(ValueError, match="share the slug"):
        R.Rubric((R.Criterion("Same"), R.Criterion("same")))


def test_attach_rubric_lives_on_privileged_and_never_exports():
    rows = [_row(), _row(prompt="cancel order 9", final="Cancelled.")]
    rubric = R.Rubric.from_dict(CRITERIA)
    R.attach_rubric(rows, rubric)
    assert R.rubric_of(rows[0]).version == rubric.version
    per_prompt = R.attach_rubric(rows, lambda r: None if "cancel" in r["prompt"] else rubric)
    assert R.rubric_of(per_prompt[1]).version == rubric.version  # unchanged, None left it alone
    R.attach_rubric(rows[1:], {"criteria": CRITERIA[:1]}, overwrite=True)
    assert len(R.rubric_of(rows[1]).criteria) == 1
    # the privileged block, rubric included, never reaches an export
    assert "privileged" not in export_row(rows[0])
    trained = training_rows(rows, system_prompt="p", tools=TOOLS)
    assert all("privileged" not in t and "rubric" not in json.dumps(t) for t in trained)
    task, rollout, _, _ = schema.from_row(rows[0])
    assert task.privileged.rubric["version"] == rubric.version
    # from_row/to_row is a round trip: the source row's own block rides back
    # as passthrough (the leakage guard is on export_row / training_rows)
    assert schema.to_row(task, rollout)["privileged"]["rubric"]["version"] == rubric.version
    assert schema.validate(rows[0]) == []


def test_rubric_judge_scores_per_criterion_and_lifts_markers(monkeypatch):
    replies = {
        "where is order 4412": {
            "criteria": {
                "Looks the order up": True,
                "States the status": True,
                "Offers next step": False,
                "Invents an id": False,
            },
            "reason": "looked it up",
        },
        "cancel order 9": {
            "criteria": {
                "Looks the order up": False,
                "States the status": True,
                "Offers next step": True,
                "Invents an id": True,
            },
            "reason": "never looked",
        },
    }

    def fake_complete(_url, _model, messages, **kwargs):
        user = json.loads(messages[-1]["content"])
        assert "Hard rule" in user["rubric"]
        prompt = user["reply"]["situation"]
        return {"content": json.dumps(replies[prompt])}

    monkeypatch.setattr(R, "complete", fake_complete)
    rows = R.attach_rubric(
        [_row(), _row(prompt="cancel order 9", final="Cancelled 12345.")],
        R.Rubric.from_dict(CRITERIA),
    )
    scored = run_judge(rows, R.rubric_judge(spec="vllm:phi@http://127.0.0.1:9/v1"))
    a, b = scored.rows
    assert a["reward"] == 0.75 and a["judge_status"] == "ok" and a["reason"] == "looked it up"
    assert a["markers"]["rubric:looks_the_order_up"] == 1.0
    assert a["markers"]["rubric:offers_next_step"] == 0.0
    assert a["markers"]["rubric:invents_an_id"] == 1.0  # pitfall clean
    assert a["judge_meta"]["rubric_version"] == R.Rubric.from_dict(CRITERIA).version
    assert b["reward"] == 0 and b["judge_meta"]["hard_failed"] == ["Looks the order up"]
    assert b["markers"]["rubric:invents_an_id"] == 0.0  # pitfall exhibited
    summary = wai.marker_summary(scored.rows)
    assert summary["rubric:looks_the_order_up"]["mean"] == 0.5
    # a row with no rubric stays ungraded
    bare = run_judge(
        [_row(prompt="hi", final="hi")], R.rubric_judge(spec="vllm:phi@http://127.0.0.1:9/v1")
    )
    assert bare.rows[0]["reward"] is None and bare.rows[0]["judge_status"] != "ok"
    judge = R.rubric_judge(R.Rubric.from_dict(CRITERIA), spec="vllm:phi@http://127.0.0.1:9/v1")
    assert judge.__name__.startswith("phi@")


def test_score_resolves_numbers_and_paraphrased_titles():
    rubric = R.Rubric.from_dict(CRITERIA)
    numbered = rubric.score({"1": True, "2": True, "3.": False, "4": False})
    assert numbered["reward"] == 0.75 and numbered["unanswered"] == []
    loose = rubric.score({"Looks the order up first": True, "states the status": True})
    assert loose["reward"] == 0.75 and "Looks the order up" in loose["met"]
    out = R.score_with_rubric(rubric, {"1": True, "2": False, "3": True, "4": True})
    assert out["markers"]["rubric:invents_an_id"] == 0.0 and out["n_unanswered"] == 0
    assert "1." in rubric.checklist()


def test_parse_criteria_reply_accepts_dict_and_list_forms():
    d, reason = R.parse_criteria_reply('ok {"criteria": {"A": "yes", "B": 0}, "reason": "r"}')
    assert d == {"A": True, "B": False} and reason == "r"
    lst, _ = R.parse_criteria_reply('{"criteria": [{"title": "A", "met": true}]}')
    assert lst == {"A": True}
    assert R.parse_criteria_reply("nothing")[0] is None
    assert R.parse_criteria_reply('{"reason": "no criteria"}')[0] is None


def test_write_rubrics_drafts_one_rubric_per_prompt():
    seen: list[dict] = []

    def writer(user: str) -> str:
        payload = json.loads(user)
        seen.append(payload)
        if payload["request"].startswith("bad"):
            return "no json here"
        return json.dumps(CRITERIA)

    rows = [
        _row(),
        _row(),  # same prompt: one rubric, two rows
        _row(prompt="cancel order 9", final="Cancelled."),
        _row(prompt="bad prompt", final="x"),
    ]
    rows[2]["privileged"] = {"reference": "Cancel it and confirm."}
    out, report = R.write_rubrics(rows, domain="refund support", writer=writer)
    assert report["prompts"] == 3 and report["written"] == 2 and report["failed"] == 1
    assert report["failures"][0]["prompt"].startswith("bad")
    assert report["criteria_per_rubric"] == 4.0 and report["writer"] == "writer"
    assert len(seen) == 3 and all(p["domain_guidance"] == "refund support" for p in seen)
    cancel = next(p for p in seen if p["request"].startswith("cancel"))
    assert cancel["reference_answer"] == "Cancel it and confirm."
    assert R.rubric_of(out[0]).version == R.rubric_of(out[1]).version
    assert R.rubric_of(out[2]).source == "model" and R.rubric_of(out[2]).domain == "refund support"
    assert R.rubric_of(out[3]) is None
    # existing rubrics are kept unless overwrite
    _, again = R.write_rubrics(rows, writer=writer)
    assert again["skipped"] == 2 and again["written"] == 0
    assert report["versions"] == {R.rubric_of(out[0]).version: 3}  # same criteria, one version


def test_write_rubrics_max_hard_demotes_the_lighter_hard_rules():
    items = [
        {"title": "Must A", "description": "Essential Criteria: a.", "weight": 3},
        {"title": "Must B", "description": "Essential Criteria: b.", "weight": 5},
        {"title": "Nice C", "description": "Optional Criteria: c.", "weight": 1},
    ]
    rows = [_row()]
    _, report = R.write_rubrics(rows, writer=lambda _u: json.dumps(items), max_hard=1)
    rubric = R.rubric_of(rows[0])
    kinds = {c.title: c.kind for c in rubric.criteria}
    assert kinds == {"Must A": "principle", "Must B": "hard", "Nice C": "principle"}
    assert report["max_hard"] == 1 and report["demoted_hard"] == 1
    _, plain = R.write_rubrics([_row()], writer=lambda _u: json.dumps(items))
    assert plain["max_hard"] is None and plain["demoted_hard"] == 0


def test_public_surface():
    for name in (
        "Rubric",
        "Criterion",
        "attach_rubric",
        "rubric_judge",
        "rubric_of",
        "write_rubrics",
    ):
        assert name in wai.__all__
    assert wai.Rubric is R.Rubric


def test_rubric_judge_warms_the_hosted_judge_once_before_the_fan_out(monkeypatch):
    """A cold serve container answers its first request in minutes, not
    seconds. Without a warm-up every one of run_judge's concurrent calls
    raced it and the whole set came back invalid_result (#224)."""
    order: list[str] = []
    gate = threading.Event()

    def fake_warm(spec, **_kw):
        order.append("warm")
        gate.set()
        return {"ok": True, "seconds": 1.0}

    def fake_complete(_url, _model, messages, **kwargs):
        # no row may reach the judge before the warm-up has returned
        assert gate.is_set(), "judged a row against a cold judge"
        order.append("judge")
        user = json.loads(messages[-1]["content"])
        met = user["reply"]["situation"] == "where is order 4412"
        return {
            "content": json.dumps(
                {"criteria": dict.fromkeys((c["title"] for c in CRITERIA), met), "reason": "r"}
            )
        }

    monkeypatch.setattr(R, "warm_judge", fake_warm)
    monkeypatch.setattr(R, "complete", fake_complete)
    rows = R.attach_rubric(
        [_row(), _row(prompt="cancel order 9"), _row(prompt="ship order 5")],
        R.Rubric.from_dict(CRITERIA),
    )
    scored = run_judge(rows, R.rubric_judge(spec="vllm:phi@http://127.0.0.1:9/v1"), concurrency=8)
    assert [r["judge_status"] for r in scored.rows] == ["ok"] * 3
    # warmed exactly once, and first
    assert order[0] == "warm"
    assert order.count("warm") == 1
    assert order.count("judge") == 3


def test_rubric_judge_still_judges_when_the_warm_up_fails(monkeypatch):
    monkeypatch.setattr(
        R,
        "warm_judge",
        lambda spec, **kw: {"ok": False, "seconds": 600.0, "error": "TimeoutError: read timed out"},
    )
    monkeypatch.setattr(
        R,
        "complete",
        lambda _u, _m, messages, **kw: {
            "content": json.dumps(
                {"criteria": dict.fromkeys((c["title"] for c in CRITERIA), True), "reason": "r"}
            )
        },
    )
    rows = R.attach_rubric([_row()], R.Rubric.from_dict(CRITERIA))
    scored = run_judge(rows, R.rubric_judge(spec="vllm:phi@http://127.0.0.1:9/v1"))
    assert scored.rows[0]["judge_status"] == "ok"


def test_rubric_judge_does_not_warm_for_a_row_with_no_rubric(monkeypatch):
    warmed: list[str] = []
    monkeypatch.setattr(R, "warm_judge", lambda spec, **kw: warmed.append(spec) or {"ok": True})
    judge = R.rubric_judge(spec="vllm:phi@http://127.0.0.1:9/v1")
    assert judge(_row())["reward"] is None  # no rubric on the row
    assert warmed == []
