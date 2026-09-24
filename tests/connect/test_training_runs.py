"""Training runs: the handle, buffering, the callback, and the HTTP bodies."""

from __future__ import annotations

import types
import warnings

import pytest

import whileai.simulations as wai
from whileai.simulations.training import TrainerCallback, TrainingRun, training_run


class Transport:
    """Records every call; can be told to fail N times."""

    def __init__(self, fail_first: int = 0):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail_first = fail_first

    def __call__(self, method, path, api_key=None, body=None, **kw):
        if self.fail_first > 0 and path.endswith("/log"):
            self.fail_first -= 1
            raise RuntimeError("network down")
        self.calls.append((method, path, body))
        if path == "/runs" and method == "POST":
            return {"runId": "run_abc", "status": "running", **body}
        if path.endswith("/finish"):
            return {"runId": "run_abc", "status": body["status"]}
        return {"runId": "run_abc", "logged": len(body["points"])}


def test_training_run_creates_logs_in_batches_and_finishes():
    t = Transport()
    run = training_run(
        "identity-v1",
        dataset="ds_a",
        base_model="Qwen3-4B",
        trainer="trl",
        total_steps=100,
        config={"lr": 1e-4},
        api_key="k",
        flush_every=3,
        transport=t,
    )
    assert run.run_id == "run_abc" and run.url.endswith("/platform/training/run_abc")
    created = t.calls[0][2]
    assert created["dataset_id"] == "ds_a" and created["total_steps"] == 100
    assert created["config"] == {"lr": 1e-4} and created["base_model"] == "Qwen3-4B"

    run.log(1, loss=2.0, lr=1e-4, note="ignored", nan=float("nan"), flag=True)
    run.log(2, loss=1.9)
    assert len(t.calls) == 1  # buffered
    run.log(3, loss=1.8)
    assert len(t.calls) == 2  # flushed at three
    body = t.calls[1][2]
    assert [p["step"] for p in body["points"]] == [1, 2, 3]
    assert body["points"][0]["loss"] == 2.0 and "note" not in body["points"][0]
    assert "nan" not in body["points"][0] and "flag" not in body["points"][0]

    run.progress(50, 200)
    out = run.finish("done", summary={"final_loss": 1.8}, adapter="/vol/a")
    assert out["status"] == "done" and run.status == "done"
    log_bodies = [c[2] for c in t.calls if c[1].endswith("/log")]
    assert log_bodies[-1]["total_steps"] == 200 and log_bodies[-1]["points"][-1]["step"] == 50
    assert t.calls[-1][1] == "/runs/run_abc/finish"
    assert t.calls[-1][2] == {"status": "done", "summary": {"final_loss": 1.8}, "adapter": "/vol/a"}


def test_logging_never_raises_and_retries_on_next_flush():
    t = Transport(fail_first=1)
    run = training_run("x", api_key="k", flush_every=1, transport=t)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run.log(1, loss=1.0)  # first send fails, warns, keeps the point
    assert run.errors == 1 and any("could not send" in str(x.message) for x in w)
    run.log(2, loss=0.9)  # second flush carries both
    sent = [c[2]["points"] for c in t.calls if c[1].endswith("/log")]
    assert [p["step"] for p in sent[-1]] == [1, 2]


def test_context_manager_marks_failure():
    t = Transport()
    with pytest.raises(ValueError), training_run("x", api_key="k", transport=t) as run:
        run.log(1, loss=1.0)
        raise ValueError("boom")
    assert t.calls[-1][1].endswith("/finish")
    assert t.calls[-1][2]["status"] == "failed" and "boom" in t.calls[-1][2]["error"]
    assert run.status == "failed"


def test_trainer_callback_maps_transformers_logs():
    t = Transport()
    run = training_run("cb", api_key="k", flush_every=100, transport=t)
    cb = TrainerCallback(run)
    state = types.SimpleNamespace(max_steps=40, global_step=0, log_history=[])
    cb.on_train_begin(None, state, None)
    state.global_step = 5
    cb.on_log(
        None,
        state,
        None,
        logs={"loss": 1.5, "learning_rate": 2e-5, "epoch": 0.1, "grad_norm": 3.2, "step": 5},
    )
    state.global_step = 10
    cb.on_log(
        None,
        state,
        None,
        logs={"eval_loss": 1.3, "eval_runtime": 2.0, "eval_mean_token_accuracy": 0.7},
    )
    state.log_history = [{"train_loss": 1.1}]
    cb.on_train_end(None, state, None)
    assert run.status == "done"
    log_bodies = [c[2] for c in t.calls if c[1].endswith("/log")]
    points = [p for b in log_bodies for p in b["points"]]
    assert log_bodies[0]["total_steps"] == 40
    by_step = {p["step"]: p for p in points}
    assert by_step[5]["loss"] == 1.5 and by_step[5]["lr"] == 2e-5 and by_step[5]["grad_norm"] == 3.2
    assert by_step[10]["eval_loss"] == 1.3 and by_step[10]["eval_token_accuracy"] == 0.7
    assert "eval_runtime" not in by_step[10]
    assert t.calls[-1][2] == {"status": "done", "summary": {"train_loss": 1.1}}


def test_package_exports_and_repr():
    assert wai.training_run is training_run and wai.TrainerCallback is TrainerCallback
    run = TrainingRun("run_1", name="n", transport=lambda *a, **k: {})
    assert "run_1" in repr(run) and run.status == "running"
    for name in ("list_runs", "get_run", "delete_run"):
        assert name in wai.__all__


def test_trainer_callback_maps_rl_keys():
    t = Transport()
    run = training_run("grpo", api_key="k", flush_every=100, transport=t)
    cb = TrainerCallback(run)
    state = types.SimpleNamespace(max_steps=10, global_step=3, log_history=[])
    cb.on_log(
        None,
        state,
        None,
        logs={
            "reward": 0.42,
            "reward_std": 0.1,
            "kl": 0.02,
            "completions/mean_length": 180.0,
            "completions/clipped_ratio": 0.25,
            "clip_ratio/region_mean": 0.03,
            "rewards/format_reward/mean": 0.9,
            "rewards/accuracy": 0.3,
            "epoch": 0.5,
        },
    )
    run.flush()
    point = t.calls[-1][2]["points"][-1]
    assert point["reward"] == 0.42 and point["kl"] == 0.02 and point["completion_length"] == 180.0
    assert point["clip_ratio"] == 0.25 and "clip_ratio/region_mean" not in point
    assert point["reward_format_reward_mean"] == 0.9 and point["reward_accuracy"] == 0.3


class HostedTransport:
    """Records every call; GET /datasets/{id}/train replays a queue of
    states, the way the platform's live status answers a hosted run as it
    trains."""

    def __init__(self, states):
        self.states = list(states)
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method, path, api_key=None, body=None, **kw):
        self.calls.append((method, path, body))
        if method == "GET" and path.startswith("/datasets/") and path.endswith("/train"):
            state = self.states.pop(0) if self.states else {}
            return {"training": state}
        if method == "POST" and path.endswith("/log"):
            return {"runId": "run_h1", "logged": len(body["points"])}
        if method == "POST" and path.endswith("/finish"):
            return {"runId": "run_h1", "status": body["status"]}
        raise AssertionError(f"unexpected call {method} {path}")


def test_hosted_refresh_draws_the_curve_from_polled_state():
    """Fails on origin/main: TrainingRun._absorb reads status/adapter/error/
    before/after/gpu/seconds from a hosted poll but never `step`, and
    nothing on the `train()` path ever calls `run.log` or `run.progress`,
    so a run started with `wai.train(...)` (the default route) draws an
    empty graph no matter how long it trains (training.py:1088, :313).
    refresh()/wait() must turn each poll into a point: a per-step metric
    in the response becomes `log(step, **metrics)`; a step with no metric
    still advances the bar. No point is fabricated -- only fields the
    platform actually sent are forwarded.
    """
    states = [
        {"status": "running", "step": 10, "loss": 0.9, "totalSteps": 100},
        {"status": "running", "step": 20},  # this poll carries only the step
        {"status": "done", "step": 100, "before": 0.2, "after": 0.5},
    ]
    t = HostedTransport(states)
    run = TrainingRun("run_h1", name="ds_x · sft", flush_every=1, transport=t)
    run._hosted = True
    run.dataset_id = "ds_x"

    assert run.refresh() == "running"
    assert run.step == 10
    assert run.refresh() == "running"
    assert run.step == 20  # a step-only poll still moved the bar
    assert run.refresh() == "done"

    log_bodies = [c[2] for c in t.calls if c[1].endswith("/log")]
    assert log_bodies, "a hosted poll never sent a point or a progress update"
    points = [p for b in log_bodies for p in b["points"]]
    by_step = {p["step"]: p for p in points}
    assert by_step[10]["loss"] == 0.9  # a real per-step metric became a logged point
    assert 20 in by_step and "loss" not in by_step[20]  # step alone, nothing fabricated
    assert log_bodies[0]["total_steps"] == 100


def test_hosted_refresh_is_silent_with_no_step_in_the_response():
    """An older platform build that answers with only status/before/after
    (no `step`) must not raise or fabricate a point."""
    t = HostedTransport([{"status": "done", "before": 0.2, "after": 0.5}])
    run = TrainingRun("run_h2", name="ds_y · sft", flush_every=1, transport=t)
    run._hosted = True
    run.dataset_id = "ds_y"
    assert run.refresh() == "done"
    assert run.step == 0
    assert not [c for c in t.calls if c[1].endswith("/log")]


def test_trainer_callback_warns_when_a_subclass_swallows_on_log():
    """The lane that lost its curve had subclassed TrainerCallback and
    overrode on_log without calling super().on_log(...); every point it
    thought it was sending was silently dropped. That is the lane's bug,
    not whileai's (wai.TrainerCallback fires correctly when used as
    documented -- see test_trainer_callback_maps_transformers_logs), but
    the symptom (an empty graph) looks identical to the SDK being broken.
    on_train_end must warn and name the fix.
    """

    class SwallowsOnLog(TrainerCallback):
        def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
            # forgot super().on_log(...): every point is lost here
            return control

    t = Transport()
    run = training_run("cb-swallowed", api_key="k", flush_every=100, transport=t)
    cb = SwallowsOnLog(run)
    state = types.SimpleNamespace(max_steps=10, global_step=0, log_history=[])
    cb.on_train_begin(None, state, None)
    state.global_step = 5
    cb.on_log(None, state, None, logs={"loss": 1.5})
    state.global_step = 10
    with pytest.warns(UserWarning, match="never logged a point"):
        cb.on_train_end(None, state, None)


def _graded(prompt, rewards):
    return [
        {
            "prompt": prompt,
            "reward": r,
            "final_text": f"Issue {i} is open.",
            "steps": [],
            "messages": [],
        }
        for i, r in enumerate(rewards)
    ]


def test_delta_rides_on_finish_and_attach_delta_resends():
    t = Transport()
    run = training_run("sft", api_key="k", transport=t)
    before = [row for p in range(12) for row in _graded(f"t{p}", [1, 0, 0, 0])]
    after = [row for p in range(12) for row in _graded(f"t{p}", [1, 1, 1, 0])]
    report = run.delta(before, after, target="pass_at_1")
    assert report["target_verdict"] == "moved_unreplicated"
    run.finish("done", summary={"final_loss": 0.9})
    sent = t.calls[-1][2]["summary"]
    assert sent["final_loss"] == 0.9 and sent["delta"]["target_verdict"] == "moved_unreplicated"
    assert isinstance(sent["delta"]["metrics"]["pass_at_1"]["ci95"], list)

    # After the fact: fetch, merge, re-send with the status kept.
    calls = []

    def transport(method, path, api_key=None, body=None, **kw):
        calls.append((method, path, body))
        if method == "GET":
            return {"runId": "run_x", "status": "done", "summary": {"final_loss": 0.9}}
        return {"runId": "run_x", "status": "done"}

    import whileai.simulations.training as tr

    monkey = tr._call
    tr._call = transport
    try:
        out = tr.attach_delta("run_x", before, after)
    finally:
        tr._call = monkey
    assert out["target_verdict"] == "moved_unreplicated"
    assert calls[-1][1] == "/runs/run_x/finish"
    assert calls[-1][2]["status"] == "done" and calls[-1][2]["summary"]["final_loss"] == 0.9
    assert calls[-1][2]["summary"]["delta"]["target_verdict"] == "moved_unreplicated"
    # The report measured the held-out pass rate on both sides; the run page
    # opens with those two keys, so the delta fills them in.
    assert calls[-1][2]["summary"]["holdoutPassBefore"] == pytest.approx(0.25)
    assert calls[-1][2]["summary"]["holdoutPassAfter"] == pytest.approx(0.75)
    assert sent["holdoutPassBefore"] == pytest.approx(0.25)


def test_holdout_sends_the_two_numbers_the_run_page_opens_with():
    t = Transport()
    run = training_run("sft", api_key="k", transport=t)
    assert run.holdout(0.42, 0.58) == {"holdoutPassBefore": 0.42, "holdoutPassAfter": 0.58}
    assert not any(c[1].endswith("/finish") for c in t.calls)  # rides on finish
    run.finish("done")
    assert t.calls[-1][2]["summary"] == {"holdoutPassBefore": 0.42, "holdoutPassAfter": 0.58}
    run.holdout(1.9, 1.2, metric="loss")  # already finished: sent right away
    assert t.calls[-1][2]["summary"]["holdoutLossAfter"] == 1.2

    # A pass rate is a share, so 58 is the mistake worth naming.
    with pytest.raises(ValueError, match="0 to 1"):
        run.holdout(42, 58)
    with pytest.raises(ValueError, match="'pass'"):
        run.holdout(0.4, 0.6, metric="reward")


def test_attach_holdout_resends_a_finished_run():
    calls = []

    def transport(method, path, api_key=None, body=None, **kw):
        calls.append((method, path, body))
        if method == "GET":
            return {"runId": "run_x", "status": "stopped", "summary": {"final_loss": 0.9}}
        return {"runId": "run_x", "status": "stopped"}

    import whileai.simulations.training as tr

    monkey = tr._call
    tr._call = transport
    try:
        out = tr.attach_holdout("run_x", before=0.42, after=0.58)
    finally:
        tr._call = monkey
    assert out == {"holdoutPassBefore": 0.42, "holdoutPassAfter": 0.58}
    assert calls[-1][1] == "/runs/run_x/finish"
    # The status it already had, and the summary it already carried.
    assert calls[-1][2]["status"] == "stopped"
    assert calls[-1][2]["summary"]["final_loss"] == 0.9
    assert calls[-1][2]["summary"]["holdoutPassAfter"] == 0.58


def test_callback_finish_false_and_second_finish_merges_summary():
    t = Transport()
    run = training_run("eval-after", api_key="k", flush_every=100, transport=t)
    cb = TrainerCallback(run, finish=False)
    state = types.SimpleNamespace(max_steps=4, global_step=4, log_history=[{"train_loss": 0.5}])
    cb.on_log(None, state, None, logs={"loss": 0.5})
    cb.on_train_end(None, state, None)
    assert run.status == "running" and not any(c[1].endswith("/finish") for c in t.calls)
    run.finish("done", summary={"train_loss": 0.5})
    run.finish("done", summary={"pass_at_1_after": 0.3}, adapter="vol:/a")
    last = t.calls[-1][2]
    assert (
        last["summary"] == {"train_loss": 0.5, "pass_at_1_after": 0.3}
        and last["adapter"] == "vol:/a"
    )


def test_delta_and_note_after_finish_are_sent_right_away():
    t = Transport()
    run = training_run("sft-v1", api_key="k", transport=t)
    run.finish("done", summary={"train_loss": 1.0})
    sent = len(t.calls)
    before = [{"prompt": f"t{i}", "reward": i % 2} for i in range(8)]
    after = [{"prompt": f"t{i}", "reward": 1 if i < 6 else 0} for i in range(8)]
    report = run.delta(before, after, target="pass_at_1")
    assert report["target"] == "pass_at_1"
    method, path, body = t.calls[-1]
    assert (method, path) == ("POST", "/runs/run_abc/finish")
    assert body["status"] == "done", "a finished run stays finished"
    assert body["summary"]["train_loss"] == 1.0, "what finish sent is kept"
    assert body["summary"]["delta"]["target"] == "pass_at_1"
    run.note(eval_pass=0.4)
    body = t.calls[-1][2]
    assert body["summary"]["eval_pass"] == 0.4 and "delta" in body["summary"]
    assert len(t.calls) == sent + 2, "one send per call, none buffered"
