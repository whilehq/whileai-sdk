"""The default judge is not the policy, and grading says when it is anyway."""

from whileai.simulations.generate import agents
from whileai.simulations.score import grade_llm


def test_default_judge_is_a_different_model_family_from_the_policy(monkeypatch):
    monkeypatch.delenv("WHILEAI_JUDGE", raising=False)
    monkeypatch.delenv("WHILEAI_AGENT", raising=False)
    _, policy = agents.parse_backend_spec(agents.default_agent_spec())
    judge_url, judge = agents.parse_backend_spec(agents.default_judge_spec())
    assert judge != policy
    assert judge.split("/")[0].lower() != policy.split("/")[0].lower()
    assert "whileai-judge" in judge_url
    assert grade_llm.judge_spec() == agents.default_judge_spec()
    assert grade_llm.hosted_judge_endpoint()["model"] == judge
    monkeypatch.setenv("WHILEAI_JUDGE", "openai:gpt-4o-mini")
    assert grade_llm.judge_spec() == "openai:gpt-4o-mini"
    # a bare URL takes the judge's model name, not the policy's
    monkeypatch.delenv("WHILEAI_JUDGE", raising=False)
    assert grade_llm.judge_spec(spec="http://127.0.0.1:8000/v1") == (
        f"vllm:{judge}@http://127.0.0.1:8000/v1"
    )


def _grade(monkeypatch, rows, spec):
    monkeypatch.setattr(grade_llm, "require_judge_key", lambda *a, **k: spec)
    monkeypatch.setattr(grade_llm, "grade_one", lambda row, **k: {"reward": 1, "reason": "ok"})
    return grade_llm.apply_grade_llm(rows)


def test_self_judging_is_reported_not_hidden(monkeypatch):
    rows = [{"prompt": "p", "final_text": "f", "steps": [], "model_version": "Qwen/Qwen3-4B"}]
    same = _grade(monkeypatch, [dict(r) for r in rows], "vllm:Qwen/Qwen3-4B@http://x")
    assert same["self_judged"] is True
    assert any("self-preference" in w for w in same["warnings"])
    other = _grade(monkeypatch, [dict(r) for r in rows], "vllm:microsoft/phi-4@http://x")
    assert other["self_judged"] is False
    # the only warning left is the judge check saying it had no human labels
    assert [w for w in other["warnings"] if "self-preference" in w] == []
    assert other["trust"] is None
    assert other["judge_version"].startswith("microsoft/phi-4@")
    empty = _grade(monkeypatch, [], "vllm:microsoft/phi-4@http://x")
    assert empty["status"] == "empty" and empty["self_judged"] is False


def test_grade_warms_the_judge_once_and_reports_it(monkeypatch):
    calls = {"warm": 0}

    def fake_complete(url, model, messages, **kwargs):
        calls["warm"] += 1
        calls["timeout"] = kwargs.get("timeout")
        return {"content": "ready"}

    monkeypatch.setattr(grade_llm, "complete", fake_complete)
    monkeypatch.setattr(grade_llm, "require_judge_key", lambda *a, **k: "vllm:m@http://x")
    monkeypatch.setattr(grade_llm, "grade_one", lambda row, **k: {"reward": 1, "reason": "ok"})
    rows = [{"prompt": f"p{i}", "final_text": "f", "steps": []} for i in range(3)]
    report = grade_llm.apply_grade_llm(rows, warmup_timeout=321)
    assert calls == {"warm": 1, "timeout": 321}
    assert report["warmup"]["ok"] is True and report["graded"] == 3
    # a cold judge that never answers is reported, and grading still runs
    monkeypatch.setattr(
        grade_llm, "complete", lambda *a, **k: (_ for _ in ()).throw(OSError("timed out"))
    )
    report = grade_llm.apply_grade_llm([dict(r) for r in rows])
    assert report["warmup"]["ok"] is False and "timed out" in report["warmup"]["error"]
    assert report["graded"] == 3
    assert grade_llm.apply_grade_llm([])["warmup"] is None
