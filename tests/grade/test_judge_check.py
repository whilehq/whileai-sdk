"""The judge is checked by default: human gold only, a floor not a hint,
an auditor that is never the grader, and the check on the grade path."""

from __future__ import annotations

import logging

import pytest

import whileai.simulations as wai
from whileai.simulations.generate import agents
from whileai.simulations.score import grade_llm
from whileai.simulations.score.agreement import judge_agreement
from whileai.simulations.score.judge_trust import (
    NO_HUMAN_GOLD_NOTE,
    format_judge_trust,
    judge_trust,
    trust_after_grade,
)
from whileai.simulations.score.judging import run_judge
from whileai.simulations.score.labels import attach_labels
from whileai.simulations.score.publish_gate import publish_gate
from whileai.simulations.score.stats import wilson_interval


def _rows(n: int, *, miss: int = 0) -> list[dict]:
    """``n`` judged rows, truth alternating; the judge fails the first
    ``miss`` true passes of every ten (a knob for the agreement level
    that never passes a failure, so the leak warning stays quiet)."""
    rows = []
    for i in range(n):
        truth = i % 2
        wrong = truth == 1 and (i % 10) < 2 * miss
        rows.append(
            {
                "prompt": f"ask {i}",
                "scenario_id": f"s{i}",
                "rollout_index": 0,
                "final_text": "Order shipped." if truth else "I invented it.",
                "steps": [],
                "reward": 1 - truth if wrong else truth,
            }
        )
    return rows


def _labels(rows: list[dict]) -> dict[str, int]:
    return {f"{r['scenario_id']}#0": int(r["prompt"].split()[1]) % 2 for r in rows}


# ------------------------------------------------ 1. human gold is the only gold


def test_attach_labels_records_who_wrote_the_gold():
    rows = _rows(4)
    attach_labels(rows, _labels(rows), annotator="ana")
    assert all(r["gold_kind"] == "human" for r in rows)
    model = _rows(4)
    attach_labels(model, _labels(model), annotator="gpt", kind="model")
    assert all(r["gold_kind"] == "model" for r in model)
    # a person and a model on the same row: not purely human, so not human
    attach_labels(model, {"s0#0": 0}, annotator="ana")
    assert model[0]["gold_reward"] == 0 and model[0]["gold_kind"] == "model"
    # a tie clears both
    attach_labels(rows, {"s0#0": 1 - rows[0]["gold_reward"]}, annotator="ben")
    assert "gold_reward" not in rows[0] and "gold_kind" not in rows[0]


def test_model_gold_does_not_measure_the_judge():
    rows = _rows(60)
    attach_labels(rows, _labels(rows), annotator="gpt", kind="model")
    out = judge_agreement(rows)
    assert out["n"] == 60 and out["agreement"] == 1.0
    assert out["gold_kind"] == "model" and out["ok"] is False
    assert any(w.startswith("The gold labels came from a model") for w in out["warnings"])
    report = judge_trust(rows)
    assert report["gold_kind"] == "model" and report["ok"] is False
    (reason,) = [w for w in report["warnings"] if "came from a model" in w]
    assert "attach_labels(rows, labels, kind='human')" in reason
    assert format_judge_trust(report).startswith("NOT MEASURED")
    # the opt-out for people who know
    assert judge_agreement(rows, allow_model_gold=True)["ok"] is True
    assert judge_trust(rows, allow_model_gold=True)["ok"] is True
    # a second judge pass as gold is model gold
    first = run_judge(_rows(10), lambda t: t["reward"])
    second = run_judge(_rows(10), lambda t: 1)
    twice = first.agreement(second.rows)
    assert twice["gold_kind"] == "model" and twice["ok"] is False


def test_old_gold_without_a_kind_is_unknown():
    rows = [{**r, "gold_reward": r["reward"]} for r in _rows(60)]
    out = judge_agreement(rows)
    assert out["gold_kind"] == "unknown" and out["ok"] is False
    assert judge_trust(rows)["ok"] is False
    assert judge_trust(rows, allow_model_gold=True)["ok"] is True


# ------------------------------------------------ 2. a floor, not a hint


def test_judge_trust_has_floors():
    good = _rows(100)
    attach_labels(good, _labels(good), annotator="ana")
    report = judge_trust(good)
    assert report["ok"] is True and report["gold_kind"] == "human"
    assert report["floors"] == {"min_agreement": 0.8, "min_kappa": 0.6, "max_skipped_share": 0.1}
    # 80% agreement: lower bound around 0.71, under the floor, kappa 0.6 on the line
    shaky = _rows(100, miss=2)
    attach_labels(shaky, _labels(shaky), annotator="ana")
    report = judge_trust(shaky)
    assert report["ok"] is False
    (floor,) = [
        w for w in report["warnings"] if w.startswith("Judge agreement with human labels is")
    ]
    low = report["agreement"]["ci95"][0]
    assert floor == (
        f"Judge agreement with human labels is {low:.2f} (lower bound), under the 0.80 "
        "floor. Change the judge prompt or the judge model, then run judge_trust again."
    )
    assert format_judge_trust(report).startswith("FAIL")
    # the floors are knobs
    assert judge_trust(shaky, min_agreement=0.7, min_kappa=0.5)["ok"] is True
    # 90% agreement clears the agreement floor at n=100; a kappa floor of 0.9 does not
    fair = _rows(100, miss=1)
    attach_labels(fair, _labels(fair), annotator="ana")
    assert judge_trust(fair)["ok"] is True
    strict = judge_trust(fair, min_kappa=0.9)
    assert strict["ok"] is False
    assert any(
        w.startswith(
            "Judge agreement with human labels beyond chance (kappa) is 0.80, under the 0.90"
        )
        for w in strict["warnings"]
    )
    # the small-sample warning stays
    few = _rows(20)
    attach_labels(few, _labels(few), annotator="ana")
    assert any("coarse" in w for w in judge_trust(few)["warnings"])


# ------------------------------------------------ 3. the auditor cannot be the grader


def _audit_setup(monkeypatch):
    monkeypatch.delenv("WHILEAI_JUDGE", raising=False)
    monkeypatch.delenv("WHILEAI_AGENT", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "k")
    monkeypatch.setattr(grade_llm, "warm_judge", lambda *a, **k: {"ok": True})
    seen: list[str] = []

    def fake_complete(_url, model, messages, **kw):
        seen.append(model)
        return {"content": '{"reason": "fair", "score": 1}'}

    monkeypatch.setattr(grade_llm, "complete", fake_complete)
    return seen


def test_audit_swaps_to_the_other_hosted_model(monkeypatch):
    judge = agents.parse_backend_spec(agents.DEFAULT_JUDGE)[1]
    agent = agents.parse_backend_spec(agents.DEFAULT_AGENT)[1]
    rows = [{"prompt": "p", "reward": 1, "final_text": "f", "judge_meta": {"model": judge}}]
    seen = _audit_setup(monkeypatch)
    report = grade_llm.audit_grades(rows)
    assert report["grader"] == judge and report["auditor"] == agent
    assert seen == [agent]  # the row went to the other model
    # the reverse: rows graded by the hosted agent are audited by the judge
    rows = [{"prompt": "p", "reward": 1, "final_text": "f", "judge_meta": {"model": agent}}]
    seen = _audit_setup(monkeypatch)
    report = grade_llm.audit_grades(rows, backend_spec=agents.DEFAULT_AGENT)
    assert report["grader"] == agent and report["auditor"] == judge
    assert seen == [judge]
    # a different model was asked for: no swap, and the report says who
    seen = _audit_setup(monkeypatch)
    report = grade_llm.audit_grades(rows, backend_spec="vllm:other@http://x/v1")
    assert report["grader"] == agent and report["auditor"] == "other"


def test_audit_refuses_when_no_other_model_exists(monkeypatch):
    seen = _audit_setup(monkeypatch)
    rows = [{"prompt": "p", "reward": 1, "final_text": "f", "judge_meta": {"model": "mine"}}]
    with pytest.raises(ValueError, match="same model as the grader"):
        grade_llm.audit_grades(rows, backend_spec="vllm:mine@http://x/v1")
    assert seen == []  # nothing was called
    # a custom judge named by its model counts too
    rows = [{"prompt": "p", "reward": 1, "final_text": "f", "judge_name": "mine"}]
    with pytest.raises(ValueError, match="Pass backend_spec="):
        grade_llm.audit_grades(rows, backend_spec="vllm:mine@http://x/v1")


# ------------------------------------------------ 4. the check on the grade path


def _hosted(monkeypatch):
    monkeypatch.setattr(grade_llm, "require_judge_key", lambda *a, **k: "vllm:m@http://x/v1")
    monkeypatch.setattr(grade_llm, "warm_judge", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        grade_llm,
        "grade_one",
        lambda row, **k: {"reward": 1 if "shipped" in row["final_text"] else 0, "reason": "r"},
    )


def test_grade_checks_the_judge_when_rows_carry_human_gold(monkeypatch, caplog):
    _hosted(monkeypatch)
    rows = _rows(80)
    attach_labels(rows[:60], _labels(rows[:60]), annotator="ana")
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        report = grade_llm.apply_grade_llm(rows)
    assert report["graded"] == 80
    summary = report["trust"]
    assert set(summary) == {"agreement", "agreement_low", "kappa", "n_gold", "ok"}
    assert summary["n_gold"] == 60 and summary["agreement"] == 1.0 and summary["ok"] is True
    assert summary["agreement_low"] == round(wilson_interval(60, 60)[0], 4)
    # every graded row carries the same stamp, labeled or not
    assert all(r["judge_meta"]["trust"] == summary for r in rows)
    assert not any("Judge accuracy not measured" in m for m in caplog.messages)
    # publish_gate carries it
    gate = publish_gate(rows, strict=False)
    assert gate["judge_trust"] == summary


def test_grade_says_when_the_judge_is_unmeasured(monkeypatch, caplog):
    _hosted(monkeypatch)
    rows = _rows(6)
    with caplog.at_level(logging.WARNING, logger="whileai.simulations"):
        report = grade_llm.apply_grade_llm(rows)
    assert report["trust"] is None
    assert all(r["judge_meta"]["trust"] is None for r in rows)
    assert caplog.messages.count(NO_HUMAN_GOLD_NOTE) == 1  # once per grade call
    assert NO_HUMAN_GOLD_NOTE in report["warnings"]
    assert "kind='human'" in NO_HUMAN_GOLD_NOTE
    # model gold is not human gold
    attach_labels(rows, _labels(rows), annotator="gpt", kind="model")
    assert grade_llm.apply_grade_llm(rows)["trust"] is None
    assert publish_gate(rows, strict=False)["judge_trust"] is None


def test_grade_trust_modes(monkeypatch):
    _hosted(monkeypatch)
    rows = _rows(60)
    for r in rows:  # a judge that fails the floor: it passes everything
        r["final_text"] = "Order shipped."
    attach_labels(rows, _labels(rows), annotator="ana")
    report = grade_llm.apply_grade_llm(rows)  # warn: graded, summary says not ok
    assert report["trust"]["ok"] is False and report["trust"]["agreement"] == 0.5
    assert any("Judge agreement with human labels" in w for w in report["warnings"])
    with pytest.raises(ValueError, match=r"under the 0\.80 floor"):
        grade_llm.apply_grade_llm(rows, trust="require")
    assert rows[0]["judge_meta"]["trust"]["ok"] is False  # graded and stamped before the raise
    with pytest.raises(ValueError, match="no human labels"):
        grade_llm.apply_grade_llm(_rows(4), trust="require")
    off = grade_llm.apply_grade_llm(rows, trust="off")
    assert off["trust"] is None and "trust" not in rows[0]["judge_meta"]
    with pytest.raises(ValueError, match="trust must be one of"):
        grade_llm.apply_grade_llm(rows, trust="maybe")


def test_data_grade_passes_trust_through_every_path(monkeypatch, caplog):
    from whileai.simulations.data import SimulationData

    rows = _rows(60)
    attach_labels(rows, _labels(rows), annotator="ana")
    for r in rows:
        r["arm"] = "a"
    data = SimulationData(rows)
    # the contract path: scored copies carry the stamp, originals stay clean
    scored = data.grade(judge=lambda t: {"reward": t["reward"], "reason": "r"})
    assert scored.rows[0]["judge_meta"]["trust"]["ok"] is True
    assert "judge_meta" not in rows[0]
    with pytest.raises(ValueError, match=r"under the 0\.80 floor"):
        data.grade(judge=lambda t: 1, trust="require")
    # the plain callable path stamps in place
    data.grade(lambda t: t["reward"])
    assert rows[0]["judge_meta"]["trust"]["n_gold"] == 60
    # the hosted path hands it to apply_grade_llm
    seen = {}

    def fake_apply(rows, **kwargs):
        seen.update(kwargs)
        return {"graded": len(rows), "warnings": [], "trust": None}

    monkeypatch.setenv("VLLM_API_KEY", "k")
    monkeypatch.setattr("whileai.simulations.data.apply_grade_llm", fake_apply)
    data.grade(trust="off")
    assert seen["trust"] == "off"
    wai.grade_llm(rows, trust="require")
    assert seen["trust"] == "require"


def test_trust_after_grade_is_the_one_helper():
    rows = _rows(60)
    attach_labels(rows, _labels(rows), annotator="ana")
    out = trust_after_grade(rows)
    assert out["note"] is None and out["trust"]["ok"] is True
    assert trust_after_grade(rows, mode="off") == {"trust": None, "note": None}
    assert trust_after_grade(_rows(3))["note"] == NO_HUMAN_GOLD_NOTE


# ------------------------------------------------ 5. the agreement is the audited judge's


def test_judge_trust_names_the_scorer_when_the_reward_is_not_the_judges():
    """Agreement reads the row's ``reward``; the probes call ``judge``. When
    the two are different scorers the report says so and ``ok`` is false,
    instead of certifying one judge on the other's verdicts (#683)."""
    rows = _rows(60)
    attach_labels(rows, _labels(rows), annotator="ana")

    def grader_a(row):
        return {"reward": row["reward"]}

    def judge_b(row):
        return {"reward": 1}

    graded = run_judge(rows, grader_a).rows
    assert all(r["judge_name"] == "grader_a" for r in graded)
    report = judge_trust(graded, judge_b, sample=5)
    assert report["agreement"]["agreement"] == 1.0
    assert report["ok"] is False
    (line,) = [w for w in report["warnings"] if w.startswith("Judge under audit is judge_b")]
    assert "60 of 60 labeled rows was written by grader_a (60)" in line
    assert "run judge_trust again" in line
    assert format_judge_trust(report).startswith("FAIL")
    # the same judge that wrote the reward: nothing to say
    assert not any(
        w.startswith("Judge under audit")
        for w in judge_trust(graded, grader_a, sample=5)["warnings"]
    )
    # rows no grading run stamped: nothing to compare against
    assert not any(
        w.startswith("Judge under audit") for w in judge_trust(rows, judge_b, sample=5)["warnings"]
    )


def test_judge_trust_resolves_the_judges_name_the_way_run_judge_stamps_it():
    """A lambda stamps as ``lambda_judge`` and a ``judge_name=`` given to
    ``run_judge`` is the stamp; the audit reads both through the same helper,
    so a judge never fails the audit on its own verdicts."""
    rows = _rows(60)
    attach_labels(rows, _labels(rows), annotator="ana")
    same = lambda row: {"reward": row["reward"]}  # noqa: E731
    graded = run_judge(rows, same).rows
    assert all(r["judge_name"] == "lambda_judge" for r in graded)
    assert not any(
        w.startswith("Judge under audit") for w in judge_trust(graded, same, sample=5)["warnings"]
    )

    def grader(row):
        return {"reward": row["reward"]}

    named = run_judge(rows, grader, judge_name="v3-rubric").rows
    assert all(r["judge_name"] == "v3-rubric" for r in named)
    assert not any(
        w.startswith("Judge under audit")
        for w in judge_trust(named, grader, sample=5, judge_name="v3-rubric")["warnings"]
    )
    # without the name the audit still says the rows were written under another stamp
    (line,) = [
        w
        for w in judge_trust(named, grader, sample=5)["warnings"]
        if w.startswith("Judge under audit is grader")
    ]
    assert "written by v3-rubric (60)" in line
