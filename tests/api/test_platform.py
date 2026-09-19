"""whileai.platform against a fake transport: the calls a coding agent makes."""

from __future__ import annotations

import logging
from datetime import date

import pytest
from pydantic import ValidationError

from whileai.platform import (
    Behavior,
    Dashboard,
    Experiment,
    Figure,
    Frontier,
    Harness,
    Judge,
    LiveDay,
    PlatformError,
    Run,
    RunRecord,
    RunSpec,
    Score,
    Tracked,
    describe,
    track,
)


class Fake:
    """Records every call; answers like the API."""

    def __init__(self, dashboard=None, fail_train_once=False):
        self.calls: list[tuple[str, str, object]] = []
        self.dashboard = dashboard or {"agent": {"id": "a", "name": "a"}}
        self.fail_train_once = fail_train_once
        self.experiment: dict | None = None
        self.figures: dict[str, dict] = {}

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path.endswith("/experiment"):
            if method == "GET":
                if self.experiment is None:
                    raise PlatformError(404, "GET /agents/a/experiment: no experiment")
                return self.experiment
            self.experiment = {**body, "updatedAt": "2026-09-19T00:00:00Z"}
            return self.experiment
        if "/figures/" in path and method == "PUT":
            name = path.rsplit("/", 1)[1]
            self.figures[name] = {"name": name, **body, "updatedAt": "2026-09-19T00:00:00Z"}
            return self.figures[name]
        if path.endswith("/figures") and method == "GET":
            return {"figures": [self.figures[k] for k in sorted(self.figures)]}
        if path == "/agents" and method == "POST":
            return {"id": body["id"], "name": body.get("name"), "serving": None}
        if path == "/runs" and method == "POST":
            return {"id": body.get("id") or "run_abc", "version": body["version"]}
        if path.endswith("/train"):
            if self.fail_train_once:
                self.fail_train_once = False
                raise PlatformError(503, "down")
            return {"points": len(body)}
        if path.endswith("/evals"):
            return {"evals": body}
        if "/dashboard" in path:
            return self.dashboard
        if "/behaviors/" in path:
            return {"name": path.rsplit("/", 1)[1], **body}
        return {"ok": True}

    def paths(self, method=None):
        return [p for m, p, _ in self.calls if method in (None, m)]


# ------------------------------------------------------------------ models


def test_models_go_on_the_wire_in_camel_case():
    f = Frontier(name="Sonnet 5", score=81, cost_per_1k=18.0, p50_s=2.1)
    assert f.wire() == {"name": "Sonnet 5", "score": 81.0, "costPer1k": 18.0, "p50s": 2.1}
    b = Behavior(
        name="refunds",
        test_version="v2",
        n=240,
        judge=Judge(agreement=0.86, human_n=60, length_bias=0.08),
        noise_floor=2.4,
    )
    assert b.wire() == {
        "name": "refunds",
        "testVersion": "v2",
        "n": 240,
        "judge": {"agreement": 0.86, "humanN": 60, "lengthBias": 0.08},
        "noiseFloor": 2.4,
    }
    assert LiveDay(day="2026-09-17", version="v3", replies=10, p50_s=0.6).wire() == {
        "day": "2026-09-17",
        "version": "v3",
        "replies": 10,
        "flagged": 0,
        "p50s": 0.6,
    }


def test_models_validate():
    with pytest.raises(ValidationError):
        Judge(agreement=1.4)
    with pytest.raises(ValidationError):
        Behavior(name="has space")
    with pytest.raises(ValidationError):
        Score(behavior="refunds", score=80, ci=-1)
    with pytest.raises(ValidationError):
        LiveDay(day="yesterday", version="v1", replies=1)


def test_harness_fingerprint_changes_with_the_prompt():
    a = Harness(instructions="Be brief.", tools=["lookup_order", "issue_refund"], model="m")
    b = Harness(instructions="Be brief!", tools=["issue_refund", "lookup_order"], model="m")
    c = Harness(instructions="Be brief.", tools=["issue_refund", "lookup_order"], model="m")
    assert a.fingerprint == c.fingerprint  # tool order does not matter
    assert a.fingerprint != b.fingerprint  # the prompt does
    assert a.version == f"h-{a.fingerprint}"
    assert Harness(label="h2").version == "h2"
    assert a.wire()["hash"] == a.fingerprint and a.wire()["label"] == a.version


# ------------------------------------------------------------------ track


def test_track_by_name_registers():
    fake = Fake()
    t = track(
        "refund-bot",
        model="Qwen/Qwen3-4B",
        harness="h2",
        frontier={"name": "Sonnet 5", "score": 81, "cost_per_1k": 18.0},
        transport=fake,
    )
    assert isinstance(t, Tracked) and t.id == "refund-bot"
    method, path, body = fake.calls[0]
    assert (method, path) == ("POST", "/agents")
    assert body["model"] == "Qwen/Qwen3-4B"
    assert body["harness"]["label"] == "h2"
    assert body["frontier"] == {"name": "Sonnet 5", "score": 81.0, "costPer1k": 18.0}


class OpenAIStyleAgent:
    """Shaped like an OpenAI Agents SDK agent: name, instructions, tools, model."""

    def __init__(self):
        self.name = "Refund bot"
        self.instructions = "Handle refunds politely."
        self.model = "gpt-5"

        def lookup_order():
            pass

        self.tools = [lookup_order, type("Tool", (), {"name": "issue_refund"})()]


class ClaudeStyleOptions:
    system_prompt = "You are a refund agent."
    allowed_tools = ["Read", "Bash"]
    model = "claude-sonnet-5"


def test_describe_reads_frameworks_by_duck_typing():
    d = describe(OpenAIStyleAgent())
    assert d.name == "Refund bot" and d.model == "gpt-5"
    assert d.harness.instructions == "Handle refunds politely."
    assert d.harness.tools == ["lookup_order", "issue_refund"]

    d2 = describe(ClaudeStyleOptions(), name="claude-bot")
    assert d2.harness.tools == ["Read", "Bash"] and d2.harness.model == "claude-sonnet-5"
    assert d2.harness.instructions == "You are a refund agent."

    d3 = describe({"name": "dict-bot", "tools": [{"function": {"name": "search"}}]})
    assert d3.harness.tools == ["search"]


def test_track_takes_the_agent_object():
    fake = Fake()
    t = track(OpenAIStyleAgent(), transport=fake)
    assert t.id == "Refund-bot" and t.model == "gpt-5"
    _, _, body = fake.calls[0]
    assert body["harness"]["tools"] == ["lookup_order", "issue_refund"]
    assert body["harness"]["label"].startswith("h-")


def test_behavior_warns_when_judge_is_reward(caplog):
    fake = Fake()
    t = track("a", transport=fake)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        out = t.behavior("refunds", test_version="v2", n=240, reward_is_judge=True)
    assert isinstance(out, Behavior) and out.name == "refunds"
    _, path, body = fake.calls[-1]
    assert path == "/agents/a/behaviors/refunds"
    assert body == {"testVersion": "v2", "n": 240, "rewardIsJudge": True}
    assert "reward_is_judge=False" in caplog.text


# ------------------------------------------------------------------ runs


def test_run_buffers_logs_and_flushes_in_batches():
    fake = Fake()
    t = track("a", model="Qwen/Qwen3-4B", harness="h2", transport=fake)
    run = t.run("v4", method="GRPO", targets=["refunds"], trained_on=["d1", "d2"], flush_every=3)
    _, _, body = fake.calls[-1]
    assert body["base"] == "Qwen/Qwen3-4B" and body["harness"] == "h2"
    assert body["trainedOn"] == ["d1", "d2"] and body["agent"] == "a"
    assert isinstance(run, Run) and run.id == "run_abc" and isinstance(run.spec, RunSpec)

    run.log(0, reward=0.3, kl=0.0)
    run.log(10, reward=0.4, kl=float("nan"), note="ignored")
    assert fake.paths("POST").count("/runs/run_abc/train") == 0
    run.log(20, reward=0.5)
    assert fake.paths("POST").count("/runs/run_abc/train") == 1
    sent = next(b for m, p, b in fake.calls if p == "/runs/run_abc/train")
    assert sent == [
        {"step": 0, "reward": 0.3, "kl": 0.0},
        {"step": 10, "reward": 0.4},
        {"step": 20, "reward": 0.5},
    ]
    assert run.step == 20


def test_failed_flush_is_retried_not_raised(caplog):
    fake = Fake(fail_train_once=True)
    run = track("a", transport=fake).run("v1", flush_every=1)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.log(1, loss=2.0)
    assert run.errors == 1 and "retry" in caplog.text
    run.log(2, loss=1.9)
    sent = [b for m, p, b in fake.calls if p == "/runs/run_abc/train"]
    assert sent[-1] == [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.9}]


def test_score_and_finish(caplog):
    fake = Fake()
    run = track("a", transport=fake).run("v4", flush_every=100)
    run.log(5, reward=0.5)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.score("length", 76)
    assert "ci=" in caplog.text
    recorded = run.score(Score(behavior="refunds", score=83, ci=2.7, n=240, test_version="v2"))
    assert isinstance(recorded, Score) and recorded.ci == 2.7
    _, path, body = fake.calls[-1]
    assert path == "/runs/run_abc/evals"
    assert body == [
        {"behavior": "refunds", "score": 83.0, "ci": 2.7, "n": 240, "testVersion": "v2"}
    ]

    run.finish(hours=2.1, gpu="1xH100", cost_usd=31)
    assert run.status == "evaluated"
    assert fake.paths("POST")[-1] == "/runs/run_abc/train"  # flushed before the PATCH
    method, path, body = fake.calls[-1]
    assert (method, path) == ("PATCH", "/runs/run_abc")
    assert body == {
        "status": "evaluated",
        "hours": 2.1,
        "gpu": "1xH100",
        "costUsd": 31.0,
        "steps": 5,
    }


def test_context_manager_fails_the_run_on_exception():
    fake = Fake()
    with pytest.raises(ValueError), track("a", transport=fake).run("v1") as run:
        run.log(1, loss=1.0)
        raise ValueError("boom")
    assert run.status == "failed"
    method, path, body = fake.calls[-1]
    assert (method, path, body["status"]) == ("PATCH", "/runs/run_abc", "failed")
    assert "boom" in body["error"]


def test_trainer_callback_shape_is_honoured():
    """wai.TrainerCallback calls progress, log and finish on the run."""
    fake = Fake()
    run = track("a", transport=fake).run("v2", flush_every=100)
    run.progress(0, 300)
    run.log(10, loss=1.2, reward_mean=0.4)
    run.finish("done", summary={"train_loss": 0.9})
    assert run.total_steps == 300
    _, _, body = fake.calls[-1]
    assert body["steps"] == 300 and body["summary"] == {"train_loss": 0.9}


# ------------------------------------------------------------------ live, verdict


def test_live_and_promote():
    fake = Fake()
    t = track("a", transport=fake)
    t.live(date(2026, 9, 17), version="v3", replies=2400, flagged=98, p50_s=0.6, cost_usd=1.7)
    _, path, body = fake.calls[-1]
    assert path == "/live"
    assert body == [
        {
            "agent": "a",
            "day": "2026-09-17",
            "version": "v3",
            "replies": 2400,
            "flagged": 98,
            "p50s": 0.6,
            "costUsd": 1.7,
        }
    ]
    t.promote("v4")
    assert fake.calls[-1] == ("POST", "/agents/a/promote", {"version": "v4"})


def test_archive_unarchive_and_delete_are_one_call_each():
    fake = Fake()
    t = track("a", transport=fake)
    t.archive("run_abc")
    assert fake.calls[-1] == ("PATCH", "/runs/run_abc", {"archived": True})
    t.unarchive("run_abc")
    assert fake.calls[-1] == ("PATCH", "/runs/run_abc", {"archived": False})
    run = t.run("v5")
    run.log(1, reward=0.1)
    run.archive()
    assert fake.paths("POST")[-1] == "/runs/run_abc/train"  # flushed first
    assert fake.calls[-1] == ("PATCH", "/runs/run_abc", {"archived": True})
    t.delete_run("run_abc")
    assert fake.calls[-1] == ("DELETE", "/runs/run_abc", None)


def test_runs_hides_archived_unless_asked():
    class Listing(Fake):
        def __call__(self, method, path, body=None):
            if path.startswith("/runs?agent="):
                return {
                    "runs": [
                        {"id": "r1", "version": "v1"},
                        {"id": "r2", "version": "v2", "archived": True},
                    ]
                }
            return super().__call__(method, path, body)

    t = track("a", transport=Listing())
    assert [r["id"] for r in t.runs()] == ["r1"]
    assert [r["id"] for r in t.runs(archived=True)] == ["r1", "r2"]


def test_dashboard_and_verdict_are_typed():
    dash = {
        "agent": {"id": "a", "name": "a", "serving": "v3", "candidate": "v4"},
        "behavior": {"name": "refunds", "testVersion": "v2", "n": 240},
        "behaviors": ["refunds", "length"],
        "versions": [{"v": "base", "score": 62, "ci": 3.1}, {"v": "v4", "score": 83, "ci": 2.7}],
        "deltas": [
            {"name": "refunds", "delta": 5, "target": True},
            {"name": "length", "delta": -4},
        ],
        "live": {"days": ["d-1"], "flaggedPct": [4.1], "replies7d": 18400, "newFailures": 412},
        "verdict": {
            "candidate": "v4",
            "serving": "v3",
            "delta": 5,
            "excludesZero": True,
            "regressions": 1,
        },
        "someNewField": 1,
    }
    t = track("a", transport=Fake(dashboard=dash))
    d = t.dashboard()
    assert isinstance(d, Dashboard) and d.versions[1].score == 83 and d.live.replies_7d == 18400
    assert d.deltas[0].target and d.deltas[1].delta == -4
    assert str(t.verdict()) == (
        "unproven: refunds: v4 beats v3 by 5 (interval excludes zero, no noise floor declared); "
        "1 behavior lower (point estimates, no interval on that check); n=240 "
        "(judge agreement unmeasured)"
    )

    empty = track(
        "a",
        transport=Fake(dashboard={"agent": {"id": "a", "name": "a"}, "verdict": {"serving": "v3"}}),
    )
    assert str(empty.verdict()) == "?: v3 is serving; no newer candidate yet"
    unscored = track(
        "a",
        transport=Fake(
            dashboard={
                "agent": {"id": "a", "name": "a"},
                "verdict": {"serving": "v3", "candidate": "v4"},
            }
        ),
    )
    assert str(unscored.verdict()) == "?: no candidate scored against v3 yet"


def _dash(**verdict):
    beh = {
        "name": "refunds",
        "n": 240,
        "judge": {"agreement": 0.86, "humanN": 60},
        "noiseFloor": 2.4,
        "rewardIsJudge": False,
        "contamination": 0,
    }
    beh.update(verdict.pop("behavior", {}))
    base = {"candidate": "v4", "serving": "v3", "delta": 5, "excludesZero": True, "regressions": 1}
    base.update(verdict)
    return {"agent": {"id": "a", "name": "a"}, "behavior": beh, "verdict": base}


def _verdict(dash):
    return str(track("a", transport=Fake(dashboard=dash)).verdict())


def test_verdict_claims_a_win_only_with_an_interval_that_clears_the_floor():
    assert _verdict(_dash()) == (
        "refunds: v4 beats v3 by 5 (interval excludes zero, clears the noise floor of 2.4); "
        "1 behavior lower (point estimates, no interval on that check); "
        "judge agreement 0.86 on 60, n=240"
    )
    assert _verdict(_dash(delta=-5, regressions=0)).startswith(
        "refunds: v4 trails v3 by 5 (interval excludes zero, clears the noise floor of 2.4)"
    )


def test_verdict_never_says_beats_without_an_interval():
    out = _verdict(_dash(excludesZero=None))
    assert "beats" not in out
    assert out.startswith("refunds: v4 scored +5 vs v3, no interval on one side, not a result")


def test_verdict_never_says_beats_inside_the_noise():
    out = _verdict(_dash(delta=2, excludesZero=False, regressions=0))
    assert "beats" not in out
    assert out.startswith("refunds: v4 about the same as v3 (+2, interval includes zero)")


def test_verdict_reads_the_noise_floor():
    # Excludes zero on the interval, but the eval moves that much on its own.
    out = _verdict(_dash(delta=2, regressions=0))
    assert "beats" not in out
    assert "inside the eval's re-run band (2.4), not a result" in out
    # No floor declared: the claim stands and says the floor is missing.
    out = _verdict(_dash(regressions=0, behavior={"noiseFloor": None}))
    assert "beats v3 by 5 (interval excludes zero, no noise floor declared)" in out


def test_verdict_is_unproven_on_a_short_or_unmeasured_judge():
    out = _verdict(_dash(regressions=0, behavior={"n": 12, "judge": {"agreement": 0.55}}))
    assert out.startswith("unproven: refunds: v4 beats v3 by 5")
    assert "n=12 under 50" in out and "judge agreement 0.55 under 0.8" in out
    out = _verdict(_dash(regressions=0, behavior={"judge": None, "rewardIsJudge": True}))
    assert out.startswith("unproven:")
    assert "judge agreement unmeasured" in out and "the training reward is the judge" in out
    # A non-claim is not prefixed: there is nothing to prove.
    out = _verdict(_dash(excludesZero=False, behavior={"n": 12}))
    assert not out.startswith("unproven:")


def test_verdict_names_the_served_version_as_no_candidate():
    assert _verdict(_dash(candidate="v3", delta=0)) == (
        "refunds: v3 is the served version; no candidate to compare"
    )


def test_score_refuses_nan_and_warns_on_a_short_n(caplog):
    run = track("a", transport=Fake()).run("v1")
    with pytest.raises(ValidationError):
        run.score("refunds", float("nan"), ci=2.0, n=100)
    with pytest.raises(ValidationError):
        run.score("refunds", 80, ci=-1.0, n=100)
    with pytest.raises(ValidationError):
        run.score("refunds", 80, ci=1.0, n=0)
    with caplog.at_level("WARNING", logger="whileai.platform"):
        run.score("refunds", 80, ci=1.0, n=12)
        run.score("tone", 80, ci=1.0)
    assert "n=12" in caplog.text and "under 50" in caplog.text
    assert "has no n" in caplog.text


def test_missing_key_names_the_fix(monkeypatch, tmp_path):
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)
    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path))
    monkeypatch.setenv("ZEROPROOF_HOME", str(tmp_path))
    with pytest.raises(PlatformError) as e:
        track("a")
    assert "whileai login" in str(e.value)


def test_run_record_rides_on_the_spec_and_on_finish():
    """A run carries its scientific record: data, optimizer, eval setup and
    provenance as typed blocks, camelCase on the wire, at open or at close."""
    fake = Fake()
    t = track("a", transport=fake)
    record = RunRecord(
        data={"train": "MATH lv3-5", "n_train": 1024, "holdout": "MATH-500", "n_holdout": 160},
        optimizer={"loss_type": "dapo", "lr": 5e-5, "beta": 1e-4, "num_generations": 8, "seed": 17},
        eval={"metric": "pass@1", "k": 4, "run_std": 0.02, "run_std_runs": 3},
        provenance={"pins": {"trl": "1.13.0"}, "paper": "2503.18892"},
    )
    run = t.run("lenient", method="grpo", record=record, flush_every=100)
    _, _, body = fake.calls[-1]
    assert body["record"]["data"] == {
        "train": "MATH lv3-5",
        "nTrain": 1024,
        "holdout": "MATH-500",
        "nHoldout": 160,
    }
    assert body["record"]["optimizer"]["lossType"] == "dapo"
    assert body["record"]["eval"]["runStdRuns"] == 3
    assert body["record"]["provenance"]["pins"] == {"trl": "1.13.0"}
    assert run.spec.record is record

    run.finish(record={"provenance": {"adapter": "vol:/lenient/adapter"}})
    _, path, body = fake.calls[-1]
    assert path == "/runs/run_abc" and body["record"] == {
        "provenance": {"pins": {}, "adapter": "vol:/lenient/adapter"}
    }


def test_run_record_validates():
    with pytest.raises(ValidationError):
        RunRecord(optimizer={"top_p": 1.5})
    with pytest.raises(ValidationError):
        RunRecord(data={"n_holdout": -1})


# ----------------------------------------------- experiment, figures, notes


class FakePlotly:
    """Duck-types plotly's Figure: no plotly import anywhere in the SDK."""

    def __init__(self, data, layout=None):
        self._data, self._layout = data, layout or {}

    def to_plotly_json(self):
        return {"data": self._data, "layout": self._layout}


def test_experiment_put_then_get_then_none_on_404():
    fake = Fake()
    t = track("a", transport=fake)
    assert t.experiment() is None  # 404 before anything is posted
    assert fake.calls[-1] == ("GET", "/agents/a/experiment", None)

    out = t.experiment(
        "Does GRPO lift refunds?",
        measure="pass@1 on refunds-test-v2, n=240",
        decide="promote when the interval clears 2.4",
    )
    method, path, body = fake.calls[-1]
    assert (method, path) == ("PUT", "/agents/a/experiment")
    assert body == {
        "question": "Does GRPO lift refunds?",
        "measure": "pass@1 on refunds-test-v2, n=240",
        "decide": "promote when the interval clears 2.4",
    }
    assert isinstance(out, Experiment) and out.hypothesis is None

    back = t.experiment()
    assert isinstance(back, Experiment)
    assert back.question == "Does GRPO lift refunds?" and back.decide == out.decide

    with pytest.raises(TypeError):
        t.experiment(measure="no question")
    with pytest.raises(ValidationError):
        Experiment(question="x" * 4097)
    with pytest.raises(ValidationError):
        Experiment(question="")


def test_experiment_get_raises_on_other_errors():
    def down(method, path, body=None):
        raise PlatformError(503, "down")

    with pytest.raises(PlatformError):
        track("a", transport=down).experiment()


def test_figure_from_a_plotly_like_object_and_from_a_dict():
    fake = Fake()
    t = track("a", transport=fake)
    run = t.run("v4", flush_every=100)
    fig = FakePlotly(
        [{"x": [1, 2], "y": [0.1, 0.4]}],  # no type: scatter
        {
            "title": "Reward",
            "template": {"layout": {"font": {}}},  # dropped, as the API drops it
            "images": [{"source": "x"}],
            "annotations": [{"text": "kept"}],
        },
    )
    out = t.figure("reward-by-step", fig, caption="Training reward, v4", run=run)
    method, path, body = fake.calls[-1]
    assert (method, path) == ("PUT", "/agents/a/figures/reward-by-step")
    assert body == {
        "figure": {
            "data": [{"x": [1, 2], "y": [0.1, 0.4]}],
            "layout": {"title": "Reward", "annotations": [{"text": "kept"}]},
        },
        "caption": "Training reward, v4",
        "run": "run_abc",
    }
    assert isinstance(out, Figure)
    assert out.name == "reward-by-step" and out.run == "run_abc" and out.updated_at

    t.figure("share", {"data": [{"type": "pie", "values": [3, 1]}]})
    _, path, body = fake.calls[-1]
    assert path == "/agents/a/figures/share"
    assert body == {"figure": {"data": [{"type": "pie", "values": [3, 1]}], "layout": {}}}

    names = [f.name for f in t.figures()]
    assert names == ["reward-by-step", "share"]
    assert fake.calls[-1] == ("GET", "/agents/a/figures", None)


def test_figure_encodes_numpy_like_values():
    class Arr:
        def tolist(self):
            return [1, 2, 3]

    fake = Fake()
    t = track("a", transport=fake)
    t.figure("arr", {"data": [{"type": "bar", "y": Arr()}]})
    assert fake.calls[-1][2]["figure"]["data"][0]["y"] == [1, 2, 3]


@pytest.mark.parametrize(
    "name",
    ["Reward", "reward_by_step", "-lead", "", "x" * 41],
)
def test_figure_rejects_a_bad_name_before_any_call(name):
    fake = Fake()
    t = track("a", transport=fake)
    before = len(fake.calls)
    with pytest.raises(ValueError, match="lowercase letters, digits and dashes"):
        t.figure(name, {"data": [{"y": [1]}]})
    assert len(fake.calls) == before


def test_figure_rejects_a_figure_over_200kb():
    fake = Fake()
    t = track("a", transport=fake)
    before = len(fake.calls)
    big = {"data": [{"y": list(range(60_000))}]}  # ~330 KB compact
    with pytest.raises(ValueError, match="200,000 bytes"):
        t.figure("big", big)
    assert len(fake.calls) == before


@pytest.mark.parametrize("data", [[], [{"y": [1]}] * 51, "not a list", None])
def test_figure_rejects_a_bad_trace_count(data):
    fake = Fake()
    t = track("a", transport=fake)
    before = len(fake.calls)
    with pytest.raises(ValueError, match=r"1\.\.50 traces"):
        t.figure("n", {"data": data})
    assert len(fake.calls) == before


def test_figure_rejects_a_trace_type_off_the_allow_list():
    fake = Fake()
    t = track("a", transport=fake)
    before = len(fake.calls)
    with pytest.raises(ValueError, match="allowed types are scatter, bar, pie"):
        t.figure("hm", {"data": [{"type": "bar", "y": [1]}, {"type": "heatmap", "z": [[1]]}]})
    assert len(fake.calls) == before
    with pytest.raises(ValueError, match="fig must be a plotly Figure or a dict"):
        t.figure("x", [1, 2])
    with pytest.raises(ValueError, match="caption is at most 1,024"):
        t.figure("cap", {"data": [{"y": [1]}]}, caption="c" * 1025)
    assert len(fake.calls) == before


def test_run_note_patches_notes():
    fake = Fake()
    t = track("a", transport=fake)
    run = t.run("v4", flush_every=100)
    assert run.notes is None
    run.note("Reward flattened at step 300.")
    assert fake.calls[-1] == ("PATCH", "/runs/run_abc", {"notes": "Reward flattened at step 300."})
    assert run.notes == "Reward flattened at step 300."
    with pytest.raises(ValueError, match="8,192"):
        run.note("n" * 8193)


def test_run_carries_the_harness_fingerprint():
    fake = Fake()
    t = track("refund-bot", model="claude-haiku-4-5", harness="h1", transport=fake)
    h = Harness(label="v3", instructions="Refund within 30 days.", tools=["lookup_order"], model="claude-haiku-4-5")
    t.run("v3", method="eval", harness=h)
    body = next(b for m, p, b in fake.calls if p == "/runs")
    assert body["harness"] == "v3"
    assert body["record"]["provenance"]["pins"] == {"harness": h.fingerprint, "model": "claude-haiku-4-5"}
    # A string is a label only: no fingerprint is invented for it.
    t.run("v4", harness="v4")
    body = [b for m, p, b in fake.calls if p == "/runs"][-1]
    assert body["harness"] == "v4" and "record" not in body
    # Left out, the agent's harness from track() is used; a label-only harness has a fingerprint too.
    t.run("v5")
    body = [b for m, p, b in fake.calls if p == "/runs"][-1]
    assert body["harness"] == "h1" and body["record"]["provenance"]["pins"]["harness"] == t.harness.fingerprint
