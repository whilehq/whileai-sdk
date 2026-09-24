"""Hosted training from the SDK (#77): ``train``, ``wait``, ``serve``."""

from __future__ import annotations

import warnings

import pytest

import whileai.simulations as wai
from whileai.simulations.training import TrainingRun, serve, train

RUNNING = {
    "callId": "fc-1",
    "runId": "run_h1",
    "method": "grpo",
    "status": "running",
    "holdoutId": "ds_hold",
    "startedAt": "2026-09-14T00:00:00Z",
}
DONE = {
    **RUNNING,
    "status": "done",
    "before": 0.17,
    "after": 0.29,
    "metric": "pass@1",
    "rows": 51,
    "seconds": 129,
    "adapter": "volume whileai-train-runs:/run_h1/adapter",
}


# The profile ``train`` reads before it posts. A set is clean for one
# method family only: SFT wants no failing row, a grouped method wants
# every task to have both a pass and a fail (``selection_report``).
MIXED = {
    "rows": 160,
    "graded": 160,
    "split": {"pass": 80, "fail": 80, "ungraded": 0},
    "tasks": 40,
    "tasks_with_repeats": 40,
    "mixed_tasks": 40,
    "per_task": [],
}
PASSES = {
    "rows": 84,
    "graded": 84,
    "split": {"pass": 84, "fail": 0, "ungraded": 0},
    "tasks": 84,
    "tasks_with_repeats": 0,
    "mixed_tasks": 0,
    "per_task": [],
}


class Gate:
    """The gate's train, status, run, models and profile routes, scripted.
    ``calls`` is what the run did; the profile read lands in ``reads``."""

    def __init__(self, states=None, already_running=False, profile=MIXED):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.reads: list[str] = []
        self.states = list(states or [DONE])
        self.already_running = already_running
        self.profile = profile
        self.run_meta = {
            "runId": "run_h1",
            "status": "done",
            "baseModel": "Qwen/Qwen3-4B",
            "adapter": DONE["adapter"],
        }

    def __call__(self, method, path, api_key=None, body=None, **kw):
        if method == "GET" and path.endswith("/profile"):
            self.reads.append(path)
            return {"profile": dict(self.profile)}
        self.calls.append((method, path, body))
        if method == "POST" and path.endswith("/train"):
            out = {"training": dict(RUNNING)}
            if self.already_running:
                out["alreadyRunning"] = True
            return out
        if method == "GET" and path.endswith("/train"):
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return {"training": dict(state)}
        if method == "GET" and path.startswith("/runs/"):
            return dict(self.run_meta)
        if method == "POST" and path == "/models":
            return {**body, "version": 1, "endpoint": "https://serve.example/v1", "status": "ready"}
        # A hosted run's own poll loop logs points once a status carries
        # ``step`` (training.py:_absorb_progress); the gate answers the
        # same way the platform's run-log and run-finish routes do.
        if method == "POST" and path.endswith("/log"):
            return {"runId": path.split("/")[2], "logged": len((body or {}).get("points") or [])}
        if method == "POST" and path.endswith("/finish"):
            return {"runId": path.split("/")[2], "status": (body or {}).get("status")}
        raise AssertionError(f"unexpected {method} {path}")


def test_train_posts_the_gate_body_and_returns_the_run_handle():
    g = Gate()
    run = train(
        "ds_train",
        method="GRPO",
        steps=40,
        holdout="ds_hold",
        base_model="Qwen/Qwen3-4B",
        api_key="k",
        transport=g,
    )
    method, path, body = g.calls[0]
    assert (method, path) == ("POST", "/datasets/ds_train/train")
    assert body == {
        "method": "grpo",
        "steps": 40,
        "holdoutId": "ds_hold",
        "base": "Qwen/Qwen3-4B",
        "maskTruncated": True,
    }
    assert isinstance(run, TrainingRun)
    assert run.hosted and run.run_id == "run_h1" and run.status == "running"
    assert run.dataset_id == "ds_train" and run.method == "grpo" and run.call_id == "fc-1"
    assert run.holdout_id == "ds_hold" and run.adapter is None
    assert run.url.endswith("/platform/training/run_h1")


def test_sft_sends_epochs_not_steps():
    g = Gate(profile=PASSES)
    train("ds_train", epochs=3, api_key="k", transport=g)
    assert g.calls[0][2] == {"method": "sft", "epochs": 3.0}


class MeasuredAt(Gate):
    """A gate whose dataset preview says what temperature the rows were sampled at."""

    def __init__(self, measured):
        super().__init__()
        self.measured = measured

    def __call__(self, method, path, api_key=None, body=None, **kw):
        if method == "GET" and path.endswith("/preview"):
            self.calls.append((method, path, body))
            row = {"prompt": "x", "reward": 1}
            if self.measured is not None:
                row["sampling"] = {"temperature": self.measured, "logprobs": False}
            return {"rows": [row]}
        return super().__call__(method, path, api_key, body, **kw)


def test_temperature_is_sent_and_a_mismatch_with_the_dataset_is_said_once():
    g = MeasuredAt(0.8)
    with pytest.warns(UserWarning, match="Training samples at 0.9 but the dataset was measured"):
        train(
            "ds_train",
            method="grpo",
            base_model="Qwen/Qwen3-4B",
            temperature=0.9,
            api_key="k",
            transport=g,
        )
    posted = next(c for c in g.calls if c[0] == "POST")
    assert posted[2]["temperature"] == 0.9

    # Same temperature, rows without sampling facts, or no preview: no notice.
    for gate in (MeasuredAt(0.8), MeasuredAt(None), Gate()):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            train(
                "ds_train",
                method="grpo",
                base_model="Qwen/Qwen3-4B",
                temperature=0.8,
                api_key="k",
                transport=gate,
            )
    # Without temperature= nothing is fetched and nothing is sent.
    g = MeasuredAt(0.8)
    train("ds_train", method="grpo", base_model="Qwen/Qwen3-4B", api_key="k", transport=g)
    assert not any(c[1].endswith("/preview") for c in g.calls)
    assert "temperature" not in g.calls[0][2]
    with pytest.raises(ValueError, match="GRPO rollout temperature"):
        train("ds_train", method="sft", temperature=0.8, transport=Gate())


def test_bad_method_and_missing_dataset_raise_before_any_call():
    g = Gate()
    with pytest.raises(ValueError, match="sft, grpo, dpo"):
        train("ds_train", method="ppo", transport=g)
    with pytest.raises(ValueError, match="dataset"):
        train("", transport=g)
    assert g.calls == []


def test_wait_polls_until_done_and_fills_adapter_and_training():
    g = Gate(states=[RUNNING, RUNNING, DONE])
    run = train("ds_train", method="grpo", api_key="k", transport=g)
    status = run.wait(poll=0)
    assert status == "done" and run.status == "done"
    polls = [c for c in g.calls if c[0] == "GET" and c[1] == "/datasets/ds_train/train"]
    assert len(polls) == 3


def test_hosted_wait_draws_the_curve_as_it_polls():
    """Fails on origin/main: nothing on the hosted path ever called
    ``run.log`` or ``run.progress`` (``train`` sets ``run._hosted = True``
    at training.py:1088 and stops there), so ``wai.train(...)`` -- the
    default route -- left the platform's graph empty for as long as the
    run trained. Proven end to end through ``train()`` + ``wait()``, the
    exact calls a user makes: a poll whose status carries a per-step
    metric must become a logged point; a poll with only ``step`` must
    still move the bar. Nothing is fabricated -- both assertions read
    back only what the fake platform actually sent."""
    states = [
        {**RUNNING, "step": 5, "loss": 0.9, "totalSteps": 40},
        {**RUNNING, "step": 20},  # this poll carries only the step
        {**DONE, "step": 40},
    ]
    g = Gate(states=states)
    run = train("ds_train", method="grpo", api_key="k", transport=g)
    assert run.wait(poll=0) == "done"

    log_bodies = [c[2] for c in g.calls if c[1].endswith("/log")]
    assert log_bodies, "a hosted run's poll loop never sent a point or a progress update"
    points = [p for b in log_bodies for p in b["points"]]
    by_step = {p["step"]: p for p in points}
    assert by_step[5]["loss"] == 0.9  # a real per-step metric became a logged point
    assert 20 in by_step and "loss" not in by_step[20]  # step alone still moved the bar
    assert log_bodies[0]["total_steps"] == 40
    assert run.adapter == DONE["adapter"]
    assert run.training["before"] == 0.17 and run.training["after"] == 0.29
    assert run.training["rows"] == 51


def test_wait_true_on_train_blocks_and_timeout_raises():
    g = Gate(states=[DONE], profile=PASSES)
    run = train("ds_train", wait=True, poll=0, api_key="k", transport=g)
    assert run.status == "done"

    stuck = Gate(states=[RUNNING], profile=PASSES)
    run = train("ds_train", api_key="k", transport=stuck)
    with pytest.raises(TimeoutError, match="run_h1"):
        run.wait(timeout=0, poll=0)


def test_failed_run_reports_the_error():
    failed = {**RUNNING, "status": "failed", "error": "CUDA out of memory"}
    g = Gate(states=[failed], profile=PASSES)
    run = train("ds_train", api_key="k", transport=g)
    assert run.wait(poll=0) == "failed"
    assert run.error == "CUDA out of memory" and run.adapter is None


def test_already_running_warns_and_returns_that_run():
    g = Gate(already_running=True, profile=PASSES)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = train("ds_train", api_key="k", transport=g)
    assert run.run_id == "run_h1"
    assert any("already training" in str(w.message) for w in caught)


def test_context_manager_does_not_finish_a_hosted_run():
    g = Gate(profile=PASSES)
    with train("ds_train", api_key="k", transport=g) as run:
        pass
    assert run.status == "running"
    assert not any(c[1].endswith("/finish") for c in g.calls)


def test_serve_from_a_finished_run_handle():
    g = Gate(profile=PASSES)
    run = train("ds_train", api_key="k", transport=g)
    run.wait(poll=0)
    row = serve("Refund-V2", run, base_model="Qwen/Qwen3-4B", api_key="k", transport=g)
    posted = next(c for c in g.calls if c[1] == "/models")[2]
    assert posted == {
        "name": "refund-v2",
        "baseModel": "Qwen/Qwen3-4B",
        "adapter": DONE["adapter"],
    }
    assert row["endpoint"] == "https://serve.example/v1" and row["version"] == 1


def test_serve_from_a_run_id_reads_adapter_and_base_from_the_run_record():
    g = Gate()
    row = serve("refund-v2", "run_h1", api_key="k", transport=g)
    assert ("GET", "/runs/run_h1", None) in g.calls
    posted = next(c for c in g.calls if c[1] == "/models")[2]
    assert posted["baseModel"] == "Qwen/Qwen3-4B" and posted["adapter"] == DONE["adapter"]
    assert row["name"] == "refund-v2"


def test_serve_refuses_a_run_without_an_adapter():
    g = Gate()
    g.run_meta = {"runId": "run_h1", "status": "running", "baseModel": "Qwen/Qwen3-4B"}
    with pytest.raises(ValueError, match="no adapter yet"):
        serve("refund-v2", "run_h1", api_key="k", transport=g)
    g.run_meta = {"runId": "run_h1", "status": "failed", "error": "model type `qwen3`"}
    with pytest.raises(ValueError, match="failed and produced no adapter: model type"):
        serve("refund-v2", "run_h1", api_key="k", transport=g)
    with pytest.raises(ValueError, match="base_model"):
        serve("bare", None, api_key="k", transport=g)


def test_public_surface():
    for name in ("train", "serve", "models"):
        assert name in wai.__all__
        assert callable(getattr(wai, name))


def test_train_warns_when_the_base_cannot_be_served():
    with pytest.warns(UserWarning, match="wai.serve cannot host"):
        # the trainer's default base
        train("ds_train", method="sft", transport=Gate(profile=PASSES))
    with pytest.warns(UserWarning, match="Qwen/Qwen2.5-1.5B-Instruct"):
        train("ds_train", method="grpo", base_model="Qwen/Qwen2.5-1.5B-Instruct", transport=Gate())


def test_train_is_quiet_on_a_served_base():
    gate = Gate(profile=PASSES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = train("ds_train", method="sft", base_model="Qwen/Qwen3-4B", transport=gate)
    assert gate.calls[0][2]["base"] == "Qwen/Qwen3-4B"
    assert run.run_id == "run_h1"


def test_serve_refuses_an_unserved_base_before_posting():
    gate = Gate()
    gate.run_meta["baseModel"] = "Qwen/Qwen2.5-0.5B-Instruct"
    with pytest.raises(ValueError, match="not a served base"):
        serve("refund-v2", "run_h1", transport=gate)
    assert not any(path == "/models" for _, path, _ in gate.calls)
    with pytest.raises(ValueError, match="Qwen/Qwen3-4B"):
        serve("refund-v2", "run_h1", base_model="Qwen/Qwen2.5-1.5B-Instruct", transport=gate)


def test_train_knobs_reach_the_gate_by_name():
    gate = Gate()
    train(
        "ds_train",
        method="grpo",
        steps=40,
        generations=8,
        learning_rate=5e-6,
        beta=0.02,
        seed=3,
        max_completion_length=256,
        loss_type="dr_grpo",
        config={"scaleRewards": False, "balance": 0.25},
        base_model="Qwen/Qwen3-4B",
        transport=gate,
    )
    body = gate.calls[0][2]
    assert body == {
        "method": "grpo",
        "steps": 40,
        "base": "Qwen/Qwen3-4B",
        "generations": 8,
        "lr": 5e-6,
        "beta": 0.02,
        "seed": 3,
        "maxCompletionLength": 256,
        "lossType": "dr_grpo",
        "maskTruncated": True,
        "scaleRewards": False,
        "balance": 0.25,
    }


def test_train_knobs_that_do_not_apply_raise_before_any_call():
    gate = Gate()
    with pytest.raises(ValueError, match="generations"):
        train("ds_train", method="sft", generations=8, transport=gate)
    with pytest.raises(ValueError, match="beta"):
        train("ds_train", method="sft", beta=0.1, transport=gate)
    with pytest.raises(ValueError, match="2 to 32"):
        train("ds_train", method="grpo", generations=1, transport=gate)
    with pytest.raises(ValueError, match="collides"):
        train("ds_train", method="grpo", beta=0.1, config={"beta": 0.2}, transport=gate)
    assert gate.calls == []
    gate = Gate(profile=PASSES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train("ds_train", method="sft", seed=7, learning_rate=1e-4, transport=gate)
    assert gate.calls[0][2] == {"method": "sft", "lr": 1e-4, "seed": 7}


def test_train_masks_token_capped_replies_by_default_and_can_zero_them():
    """#253: a cut reply scored 0 teaches shorter thinking first."""
    gate = Gate()
    train("ds_train", method="grpo", transport=gate)
    assert gate.calls[0][2]["maskTruncated"] is True
    gate = Gate()
    train("ds_train", method="grpo", truncated="zero", transport=gate)
    assert gate.calls[0][2]["maskTruncated"] is False
    gate = Gate(profile=PASSES)
    train("ds_train", method="sft", transport=gate)
    assert "maskTruncated" not in gate.calls[0][2]
    with pytest.raises(ValueError, match="grpo only"):
        train("ds_train", method="sft", truncated="mask", transport=Gate())
    with pytest.raises(ValueError, match="mask, zero"):
        train("ds_train", method="grpo", truncated="penalize", transport=Gate())
    with pytest.raises(ValueError, match="collides"):
        train("ds_train", method="grpo", config={"maskTruncated": False}, transport=Gate())
