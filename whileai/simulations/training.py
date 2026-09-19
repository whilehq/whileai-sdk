"""Training runs: the loss curve and the progress bar on the platform.

Two ways to train, one record. ``train`` starts SFT, GRPO, DPO or a reward
model (``rm``) on the platform's trainer and returns the run handle;
``serve`` puts a finished run's adapter on an OpenAI-compatible endpoint
and ``reward_model`` turns a finished ``rm`` run into a judge. Or your own trainer runs
wherever it runs and reports through the same handle: a run is created,
points are logged as it goes, and it is finished with a status. The
platform draws the curve and the progress bar at
app.withwhile.com/platform/training.

Three ways in for your own trainer:

* Engineer, one line. ``trainer.add_callback(wai.TrainerCallback(run))``
  on a Transformers or TRL trainer logs every ``on_log`` (loss, learning
  rate, eval loss, epoch, grad norm), sets the step count from the
  trainer, and finishes the run when training ends or crashes.
* Data scientist with a loop. ``run = wai.training_run("sft-v3",
  dataset="ds_...")``, then ``run.log(step, loss=...)`` wherever the loop
  has a number, ``run.finish()`` at the end. Points are buffered and sent
  in batches; logging never raises into the training loop.
* Researcher with a stack. Plain HTTP: ``POST /runs``, ``POST
  /runs/{id}/log`` with ``{"points": [{"step": 10, "loss": 1.2}]}``,
  ``POST /runs/{id}/finish``. The README lists the bodies.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .defaults import (
    PASS_THRESHOLD,
    PLATFORM_REWARD_MODEL_BATCH,
    RL_ROLLOUTS_PER_PROMPT,
    TRAIN_MIN_MIXED_TASKS,
    TRAINING_ERROR_CHARS,
    TRAINING_FLUSH_EVERY,
    TRAINING_FLUSH_SECONDS,
    TRAINING_KNOBS,
    TRAINING_MAX_BATCH,
    TRAINING_POLL_MIN_S,
    TRAINING_POLL_S,
)
from .ingest.platform import _call

log = logging.getLogger("whileai.simulations")

SITE_URL = "https://app.withwhile.com"


def _json_safe(value: Any) -> Any:
    """Tuples to lists, NaN/inf to None, so a report survives JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


# The buffering numbers live in defaults.py (TRAINING_*) with their reasons;
# ``training_run(flush_every=, flush_seconds=, max_batch=)`` sets them per run.
FLUSH_EVERY = TRAINING_FLUSH_EVERY
FLUSH_SECONDS = TRAINING_FLUSH_SECONDS
MAX_BATCH = TRAINING_MAX_BATCH

# The two numbers a finished run's page opens with, under one word: Better,
# Worse or About the same. The platform's own trainer writes them; a run on
# your own hardware has to say them, which is what `holdout` is for.
HOLDOUT_KEYS = {
    "pass": ("holdoutPassBefore", "holdoutPassAfter"),
    "loss": ("holdoutLossBefore", "holdoutLossAfter"),
}


def _holdout_summary(before: float, after: float, metric: str = "pass") -> dict[str, float]:
    """``{holdoutPassBefore: ..., holdoutPassAfter: ...}``, validated."""
    if metric not in HOLDOUT_KEYS:
        raise ValueError(f"metric: 'pass' (a pass rate) or 'loss', not {metric!r}")
    out: dict[str, float] = {}
    for key, name, value in zip(HOLDOUT_KEYS[metric], ("before", "after"), (before, after)):
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"{name}: a number, not {value!r}")
        # A pass rate is a share of the held-out prompts, so 58% is 0.58; sent
        # as 58 the page would read it as 5800%.
        if metric == "pass" and not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}={value!r}: a pass rate is 0 to 1 (58% is 0.58)")
        out[key] = v
    return out


def _holdout_from_delta(report: Mapping[str, Any]) -> dict[str, float]:
    """The same two numbers, read off a delta report's pass@1."""
    metric = (report.get("metrics") or {}).get("pass_at_1") or {}
    a, b = metric.get("mean_a"), metric.get("mean_b")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return {}
    try:
        return _holdout_summary(a, b)
    except ValueError:
        return {}


NO_INTERVAL_NOTE = "No interval: the platform only returned two numbers"


def _holdout_side(rows: Sequence[dict]) -> dict[str, Any]:
    """One side of the holdout with its uncertainty: pass@1 over tasks,
    how many tasks, rollouts per task, and the task-bootstrap interval."""
    from .score.stats import metric_summary, task_key

    summary = metric_summary(rows, "pass_at_1")
    per_task: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("reward") is not None:
            per_task[task_key(row)] = per_task.get(task_key(row), 0) + 1
    ci = summary["ci95"]
    return {
        "pass": summary["mean"],
        "n_tasks": summary["n_tasks"],
        "k": max(per_task.values()) if per_task else None,
        "ci95": list(ci) if ci else None,
    }


def _holdout_block(
    report: Mapping[str, Any], before: Sequence[dict], after: Sequence[dict]
) -> dict[str, Any]:
    """``summary["holdout"]``: the before/after pass rates with their
    intervals and one verdict word from ``delta_report`` (``moved``,
    ``moved_unreplicated``, ``within_eval_noise``, ``no_change_detected``,
    ...), so the run page's opening line can carry the same caveats the
    report does (Lambert 2025, chapter Evaluation and its evaluation-variance
    appendix)."""
    from .score.delta import _verdict_word

    metric = (report.get("metrics") or {}).get("pass_at_1") or {}
    if report.get("target") == "pass_at_1":
        verdict = report.get("target_verdict")
    elif metric.get("verdict"):
        verdict = _verdict_word(metric, bool(report.get("replicated")))
    else:
        verdict = None
    return {
        "before": _holdout_side(before),
        "after": _holdout_side(after),
        "verdict": verdict,
        "eval_runs": report.get("eval_runs"),
        "run_std": report.get("run_std"),
        "ceiling": bool(report.get("ceiling")),
        "note": None,
    }


def _holdout_without_rows(before: Any, after: Any) -> dict[str, Any]:
    """The same block when only two numbers exist (a hosted run's
    summary): the uncertainty fields are ``None`` and ``note`` says why."""

    def side(value: Any) -> dict[str, Any]:
        number = float(value) if isinstance(value, (int, float)) else None
        return {"pass": number, "n_tasks": None, "k": None, "ci95": None}

    return {
        "before": side(before),
        "after": side(after),
        "verdict": None,
        "eval_runs": None,
        "run_std": None,
        "ceiling": None,
        "note": NO_INTERVAL_NOTE,
    }


class TrainingRun:
    """One fine-tune, as the platform sees it. Create with ``training_run``.

    ``log`` buffers; ``flush`` sends. A send that fails is retried on the
    next flush and counted in ``errors``; the training loop is never
    interrupted by the dashboard. ``finish`` flushes first.
    """

    def __init__(
        self,
        run_id: str,
        *,
        name: str,
        api_key: str | None = None,
        total_steps: int | None = None,
        flush_every: int = FLUSH_EVERY,
        flush_seconds: float = FLUSH_SECONDS,
        max_batch: int = MAX_BATCH,
        transport: Callable[..., Any] | None = None,
    ):
        self.run_id = run_id
        self.name = name
        self.total_steps = total_steps
        self.status = "running"
        self.step = 0
        self.errors = 0
        self._api_key = api_key
        self._flush_every = max(1, int(flush_every))
        self._flush_seconds = float(flush_seconds)
        self._max_batch = max(1, int(max_batch))
        self._call = transport or _call
        self._buffer: list[dict[str, Any]] = []
        self._pending_total: int | None = None
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self._warned = False
        self._delta: dict[str, Any] | None = None
        self._summary: dict[str, Any] = {}
        #: where the trained weights landed, once known (``finish(adapter=)``
        #: or a hosted run that reached ``done``)
        self.adapter: str | None = None
        # Hosted runs (``train``): the platform's trainer owns the lifecycle,
        # so ``refresh``/``wait`` read it and the context manager never
        # finishes it from here.
        self.dataset_id: str | None = None
        self.call_id: str | None = None
        self.method: str | None = None
        #: ``selection_report`` of the pushed set, read by ``train`` before
        #: the run started (``None`` under ``check="off"`` or an unreadable
        #: profile)
        self.selection: dict[str, Any] | None = None
        #: the holdout numbers with their uncertainty: filled by ``delta``
        #: from the rows, or by ``refresh`` from the platform's two numbers
        #: (then every interval field is ``None`` and ``note`` says so)
        self.holdout_summary: dict[str, Any] | None = None
        self.holdout_id: str | None = None
        self.training: dict[str, Any] = {}
        self.error: str | None = None
        self._hosted = False

    @property
    def url(self) -> str:
        return f"{SITE_URL}/platform/training/{self.run_id}"

    # ------------------------------------------------------------ logging

    def log(self, step: int, **metrics: float) -> None:
        """Record one point. Any finite numeric keyword is a metric
        (``loss``, ``eval_loss``, ``lr``, ``epoch``, ``grad_norm``, ...)."""
        point: dict[str, Any] = {"step": int(step), "ts": time.time()}
        for key, value in metrics.items():
            if isinstance(value, bool) or value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number or number in (float("inf"), float("-inf")):
                continue
            point[str(key)] = number
        with self._lock:
            self._buffer.append(point)
            self.step = max(self.step, int(step))
            due = (
                len(self._buffer) >= self._flush_every
                or time.monotonic() - self._last_flush >= self._flush_seconds
            )
        if due:
            self.flush()

    def progress(self, step: int, total_steps: int | None = None) -> None:
        """Advance the bar without a metric. ``total_steps`` (re)sets the
        denominator; a trainer that learns its length late can call this."""
        if total_steps:
            with self._lock:
                self.total_steps = int(total_steps)
                self._pending_total = int(total_steps)
        self.log(step)

    def flush(self) -> bool:
        """Send buffered points. Returns True when nothing is left unsent."""
        with self._lock:
            batch = self._buffer[: self._max_batch]
            total = self._pending_total
        if not batch and total is None:
            return True
        body: dict[str, Any] = {"points": batch or [{"step": self.step, "ts": time.time()}]}
        if total is not None:
            body["total_steps"] = total
        try:
            self._call("POST", f"/runs/{self.run_id}/log", self._api_key, body)
        except Exception as exc:  # the dashboard must never stop the trainer
            self.errors += 1
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"training run {self.run_id}: could not send points ({exc}); "
                    "will retry on the next flush",
                    stacklevel=2,
                )
            log.debug("training run %s flush failed: %s", self.run_id, exc)
            return False
        with self._lock:
            del self._buffer[: len(batch)]
            if total is not None and self._pending_total == total:
                self._pending_total = None
            self._last_flush = time.monotonic()
            remaining = bool(self._buffer)
        if remaining:
            return self.flush()
        return True

    # ------------------------------------------------------------ lifecycle

    def finish(
        self,
        status: str = "done",
        *,
        summary: Mapping[str, Any] | None = None,
        adapter: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Flush, then mark the run ``done``, ``failed``, or ``stopped``."""
        self.flush()
        body: dict[str, Any] = {"status": status}
        # A second finish (an eval after the callback already finished the
        # run) adds to what was sent, never replaces it.
        merged = {**self._summary, **dict(summary or {})}
        if self._delta is not None:
            merged["delta"] = self._delta
        if merged:
            body["summary"] = _json_safe(merged)
            self._summary = dict(merged)
        if adapter:
            body["adapter"] = str(adapter)
            self.adapter = str(adapter)
        if error:
            body["error"] = str(error)[:TRAINING_ERROR_CHARS]
        try:
            out = self._call("POST", f"/runs/{self.run_id}/finish", self._api_key, body)
        except Exception as exc:
            self.errors += 1
            warnings.warn(f"training run {self.run_id}: could not finish ({exc})", stacklevel=2)
            out = {"runId": self.run_id, "status": status, "unsent": True}
        self.status = status
        return out if isinstance(out, dict) else {"runId": self.run_id, "status": status}

    def fail(self, error: str) -> dict[str, Any]:
        return self.finish("failed", error=error)

    def note(self, **fields: Any) -> None:
        """Put fields on the run's summary ahead of ``finish``: whichever
        callback finishes the run, the summary carries them. Sent right
        away when the run is already finished."""
        self._summary.update(_json_safe(dict(fields)))
        if self.status == "running":
            return
        if self._delta is not None:
            self._send_delta()
        else:
            self._resend_summary()

    def _resend_summary(self) -> None:
        try:
            self._call(
                "POST",
                f"/runs/{self.run_id}/finish",
                self._api_key,
                {"status": self.status, "summary": dict(self._summary)},
            )
        except Exception as exc:
            self.errors += 1
            warnings.warn(
                f"training run {self.run_id}: could not send summary ({exc})", stacklevel=2
            )

    def holdout(self, before: float, after: float, *, metric: str = "pass") -> dict[str, float]:
        """Did it work? The held-out pass rate before and after, which is
        what the run's page opens with. ``metric="loss"`` for held-out loss
        (SFT), where lower is better. Pass rates are 0 to 1."""
        fields = _holdout_summary(before, after, metric)
        self.note(**fields)
        return fields

    def delta(
        self,
        before: Sequence[dict],
        after: Sequence[dict],
        *,
        target: str | None = "pass_at_1",
        must_not_regress: Sequence[str] = (),
        by: str | Callable[[dict], Any] | None = None,
        proxy: str | None = None,
    ) -> dict[str, Any]:
        """Did the training move the behavior? ``delta_report`` over the
        rollouts before and after, kept on the run and sent with
        ``finish`` under ``summary["delta"]`` (sent right away when the run
        is already finished). The run page draws it, including the
        per-group table when ``by`` names a row key or marker. ``proxy``
        names the training reward's marker so the report can call the
        run over-optimized when the proxy moved and the target did not."""
        from .score.delta import delta_report

        report = delta_report(
            before,
            after,
            target=target,
            must_not_regress=list(must_not_regress),
            by=by,
            proxy=proxy,
        )
        self._delta = _json_safe(report)
        # The report already measured the held-out pass rate before and
        # after; the page's own opening line reads those two keys, so fill
        # them from it rather than asking for the numbers twice.
        for key, value in _holdout_from_delta(report).items():
            self._summary.setdefault(key, value)
        self.holdout_summary = _json_safe(_holdout_block(report, before, after))
        self._summary["holdout"] = self.holdout_summary
        if self.status != "running":
            self._send_delta()
        return report

    def _send_delta(self) -> None:
        try:
            self._call(
                "POST",
                f"/runs/{self.run_id}/finish",
                self._api_key,
                {"status": self.status, "summary": {**self._summary, "delta": self._delta}},
            )
        except Exception as exc:
            self.errors += 1
            warnings.warn(f"training run {self.run_id}: could not send delta ({exc})", stacklevel=2)

    # ------------------------------------------------------------ hosted runs

    @property
    def hosted(self) -> bool:
        """True when the platform's trainer runs this (``train``)."""
        return self._hosted

    def refresh(self) -> str:
        """Read a hosted run's state from the platform: ``running``,
        ``done`` or ``failed``. Fills ``adapter``, ``training`` (before,
        after, seconds, rows) and ``error`` once it has ended."""
        if not self._hosted or not self.dataset_id:
            return self.status
        out = self._call("GET", f"/datasets/{self.dataset_id}/train", self._api_key)
        state = dict((out or {}).get("training") or {}) if isinstance(out, dict) else {}
        if not state:
            return self.status
        self._absorb(state)
        return self.status

    def wait(self, *, timeout: float | None = None, poll: float = TRAINING_POLL_S) -> str:
        """Block until a hosted run ends, reading its state every ``poll``
        seconds (never under ``TRAINING_POLL_MIN_S``). Returns the final
        status; raises ``TimeoutError`` when ``timeout`` seconds pass first."""
        started = time.monotonic()
        while self.refresh() == "running":
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(
                    f"training run {self.run_id} still running after {timeout:.0f}s; "
                    f"watch it at {self.url}"
                )
            time.sleep(max(TRAINING_POLL_MIN_S, float(poll)))
        return self.status

    def _absorb(self, state: Mapping[str, Any]) -> None:
        self.training = dict(state)
        status = str(state.get("status") or self.status)
        self.status = status if status in ("running", "done", "failed", "stopped") else self.status
        if state.get("runId"):
            self.run_id = str(state["runId"])
        if state.get("callId"):
            self.call_id = str(state["callId"])
        if state.get("method"):
            self.method = str(state["method"])
        if state.get("holdoutId"):
            self.holdout_id = str(state["holdoutId"])
        if state.get("adapter"):
            self.adapter = str(state["adapter"])
        if state.get("error"):
            self.error = str(state["error"])
        if self.holdout_summary is None and ("before" in state or "after" in state):
            self.holdout_summary = _holdout_without_rows(state.get("before"), state.get("after"))

    def __enter__(self) -> TrainingRun:
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if self.status != "running" or self._hosted:
            return
        if exc is not None:
            self.finish("failed", error=f"{exc_type.__name__}: {exc}")
        else:
            self.finish("done")

    @property
    def id(self) -> str:
        """The run id, the handle ``get_run``, ``serve`` and ``delete_run`` take."""
        return self.run_id

    def __repr__(self) -> str:
        return (
            f"TrainingRun({self.run_id!r}, {self.name!r}, status={self.status!r}, step={self.step})"
        )


def training_run(
    name: str,
    *,
    dataset: str | None = None,
    after_dataset: str | None = None,
    base_model: str | None = None,
    trainer: str | None = None,
    total_steps: int | None = None,
    config: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    flush_every: int = FLUSH_EVERY,
    flush_seconds: float = FLUSH_SECONDS,
    max_batch: int = MAX_BATCH,
    transport: Callable[..., Any] | None = None,
) -> TrainingRun:
    """Create a run on the platform and return the handle to log into.

    ``dataset`` is the ``ds_...`` id trained on; ``after_dataset`` the set
    of post-training rollouts, when you have one, so the run page can show
    the before/after. ``config`` is anything JSON-shaped you want on the
    run page (hyperparameters, the command). ``api_key`` defaults to the
    usual credential chain.
    """
    body: dict[str, Any] = {"name": name}
    if dataset:
        body["dataset_id"] = dataset
    if after_dataset:
        body["after_dataset_id"] = after_dataset
    if base_model:
        body["base_model"] = base_model
    if trainer:
        body["trainer"] = trainer
    if total_steps:
        body["total_steps"] = int(total_steps)
    if config:
        body["config"] = dict(config)
    call = transport or _call
    created = call("POST", "/runs", api_key, body)
    run = TrainingRun(
        str(created["runId"]),
        name=name,
        api_key=api_key,
        total_steps=int(total_steps) if total_steps else None,
        flush_every=flush_every,
        flush_seconds=flush_seconds,
        max_batch=max_batch,
        transport=transport,
    )
    log.info("training run %s: %s", run.run_id, run.url)
    return run


METHODS = ("sft", "grpo", "dpo", "rm")
#: The bases the serving app runs. An adapter trained on any other base is a
#: file on a volume that ``serve`` cannot host; the trainer's defaults
#: (Qwen2.5-0.5B for SFT, 1.5B for GRPO and DPO) are not on this list.
SERVED_BASES = ("Qwen/Qwen3-4B", "microsoft/phi-4")


def _measured_temperature(
    dataset: str, api_key: str | None, call: Callable[..., Any]
) -> float | None:
    """The temperature the pushed dataset's rows were sampled at, read from
    the platform's preview rows (``sampling.temperature``), or ``None`` when
    the rows do not say or the preview is unavailable. Never blocks a run."""
    try:
        out = call("GET", f"/datasets/{dataset}/preview", api_key)
    except Exception:
        return None
    if not isinstance(out, dict):
        return None
    rows = out.get("rows") or out.get("sample") or out.get("samples") or []
    for row in rows if isinstance(rows, list) else []:
        sampling = row.get("sampling") if isinstance(row, dict) else None
        value = sampling.get("temperature") if isinstance(sampling, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return None


#: what GRPO does with a sampled reply the token cap cut
TRUNCATED = ("mask", "zero")
#: what ``train(check=)`` does with a set the trainer would misuse:
#: refuse it, say so and start anyway, or not look
CHECK_MODES = ("require", "warn", "off")
#: methods that learn from a pass and a fail of the same prompt
GROUPED_METHODS = ("grpo", "dpo", "rm")


class TrainingSelectionError(ValueError):
    """``train(check="require")`` refused to start: the hosted trainer would
    learn the failure (SFT on failing rows) or nothing at all (a grouped
    method with no mixed task). The message names the counts, the reason
    per dropped class, and the knob."""


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def selection_report(
    profile: Mapping[str, Any],
    *,
    method: str,
    dataset: str = "ds",
    steps: int | None = None,
    min_mixed_tasks: int = TRAIN_MIN_MIXED_TASKS,
) -> dict[str, Any]:
    """What the hosted trainer will use of a profiled set, before the GPU.

    ``profile`` is ``wai.profile(dataset)`` (``rows``, ``split``, ``tasks``,
    ``tasks_with_repeats``, ``mixed_tasks``, ``per_task``). Returns
    ``given`` and ``used`` (rows for ``sft``, tasks for a grouped method,
    with ``used_rows`` when the per-task table is complete), ``dropped``
    (count per reason), ``refuse`` (lines that stop a ``check="require"``
    run) and ``warn`` (lines said either way). Pure: ``train`` reads the
    profile and decides.

    ``sft`` clones every row it is given, so a failing row is a refusal
    (rejection sampling keeps the passes,
    Lambert 2025, chapter Rejection Sampling). ``grpo``, ``dpo`` and
    ``rm`` learn from prompts with both a pass and a fail; none is a
    refusal, fewer than ``min_mixed_tasks`` (``TRAIN_MIN_MIXED_TASKS``,
    32) is a warning with the count.
    """
    rows = _count(profile.get("rows"))
    raw_split = profile.get("split")
    split: Mapping[str, Any] = raw_split if isinstance(raw_split, Mapping) else {}
    n_fail = _count(split.get("fail"))
    n_ungraded = _count(split.get("ungraded"))
    knob = 'check="warn"'
    out: dict[str, Any] = {"method": method, "given": rows, "refuse": [], "warn": []}
    if method == "sft":
        out["used"] = rows
        out["dropped"] = {}
        if n_fail:
            ungraded = f" and {n_ungraded} ungraded" if n_ungraded else ""
            out["refuse"].append(
                f"sft on {dataset} would train on all {rows} rows, {n_fail} of which fail "
                f"(reward under {PASS_THRESHOLD:g}){ungraded}. The hosted trainer clones every "
                "row it is given, so the model learns the failure; rejection sampling keeps the "
                "passing completions (Lambert 2025, chapter Rejection Sampling). Push "
                f"scored.passes() (the rows with reward >= {PASS_THRESHOLD:g}) as the train set, "
                f"or pass {knob} to train on the failures on purpose."
            )
        return out

    tasks = _count(profile.get("tasks"))
    with_repeats = _count(profile.get("tasks_with_repeats"))
    mixed = _count(profile.get("mixed_tasks"))
    per_task = [t for t in (profile.get("per_task") or []) if isinstance(t, Mapping)]
    complete = tasks > 0 and len(per_task) == tasks
    dropped: dict[str, int] = {}
    if complete:
        repeated = [t for t in per_task if _count(t.get("graded")) >= 2]  # noqa: PLR2004  # a pair is the least that can disagree
        all_pass = sum(1 for t in repeated if t.get("pass_rate") == 1)
        all_fail = sum(1 for t in repeated if t.get("pass_rate") == 0)
        used_rows: int | None = sum(
            _count(t.get("graded")) for t in repeated if t.get("pass_rate") not in (0, 1, None)
        )
        if all_pass:
            dropped["tasks all pass"] = all_pass
        if all_fail:
            dropped["tasks all fail"] = all_fail
    else:
        used_rows = None
        if with_repeats - mixed > 0:
            dropped["tasks unanimous (all pass or all fail)"] = with_repeats - mixed
    if tasks - with_repeats > 0:
        dropped["tasks with one graded rollout"] = tasks - with_repeats
    if n_ungraded:
        dropped["rows ungraded"] = n_ungraded
    out.update({"given": tasks, "used": mixed, "used_rows": used_rows, "dropped": dropped})
    reasons = ", ".join(f"{n} {why}" for why, n in dropped.items()) or "no dropped class named"
    if method == "grpo":
        why = (
            "a group with one reward has zero advantage: GRPO's baseline is the group mean "
            "(Shao et al. 2024, arXiv:2402.03300), which is why DAPO "
            "(arXiv 2503.14476) "
            "drops prompts at accuracy 0 and 1"
        )
    else:
        why = (
            f"{method} learns from a pass paired with a fail of the same prompt "
            "(Lambert 2025, chapter Direct Alignment), so a unanimous prompt gives "
            "no pair"
        )
    size = (
        f"wai.profile({dataset!r})['mixed_tasks'] is how to size the set: simulate(mode='rl') "
        f"with repeats= (RL_ROLLOUTS_PER_PROMPT, {RL_ROLLOUTS_PER_PROMPT}) samples each prompt "
        "enough times to split, and wai.optimize(rows, mode='rl') keeps the prompts that did"
    )
    counts = f"{mixed} of {tasks} tasks" + (
        f" ({used_rows} of {rows} rows)" if used_rows is not None else f" of {rows} rows"
    )
    if mixed == 0:
        out["refuse"].append(
            f"{method} on {dataset} would use {counts}: {reasons}. The run has nothing to "
            f"learn from ({why}); it would spend the GPU and finish with reward_std 0 and "
            f"grad_norm 0. {size}. {knob} starts it anyway."
        )
        return out
    if mixed < tasks or mixed < min_mixed_tasks:
        line = f"{method} on {dataset} will use {counts}: {reasons}."
        if mixed < min_mixed_tasks:
            passes = (
                f"; at one prompt group per step, {int(steps)} steps is {int(steps) / mixed:.1f} "
                "passes over them"
                if steps
                else ""
            )
            line += (
                f" That is under min_mixed_tasks ({min_mixed_tasks}, TRAIN_MIN_MIXED_TASKS)"
                f"{passes}; {why}. {size}; train(min_mixed_tasks=) moves the floor."
            )
        out["warn"].append(line)
    return out


def _check_selection(
    dataset: str,
    method: str,
    *,
    steps: int | None,
    check: str,
    min_mixed_tasks: int,
    api_key: str | None,
    call: Callable[..., Any],
) -> dict[str, Any] | None:
    """Read the set's profile and apply ``selection_report`` under
    ``check``: ``"require"`` raises ``TrainingSelectionError`` on a
    refusal line and warns the rest, ``"warn"`` warns every line. A
    profile that cannot be read is said, not a stop: the trainer still
    owns the run."""
    try:
        out = call("GET", f"/datasets/{dataset}/profile", api_key)
        profile = out.get("profile") if isinstance(out, dict) else None
    except Exception as exc:  # the platform, not the caller's data
        profile = None
        reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = "the reply carried no profile"
    if not isinstance(profile, Mapping):
        warnings.warn(
            f"could not read wai.profile({dataset!r}) before training ({reason}), so the "
            "selection check did not run; read it yourself before spending the GPU, or pass "
            'check="off" to skip it on purpose.',
            stacklevel=3,
        )
        return None
    report = selection_report(
        profile, method=method, dataset=dataset, steps=steps, min_mixed_tasks=min_mixed_tasks
    )
    if check == "require" and report["refuse"]:
        raise TrainingSelectionError(" ".join(report["refuse"] + report["warn"]))
    for line in report["refuse"] + report["warn"]:
        warnings.warn(line, stacklevel=3)
    return report


#: Rollout temperatures closer than this are the same temperature.
_TEMPERATURE_TOLERANCE = 1e-9


def _knob(name: str, value: Any, method: str) -> float:
    """``value`` checked against ``TRAINING_KNOBS[name]``: the methods it
    applies to and the accepted range. The message names the reference
    value the literature reaches for, so a rejected value says what to try."""
    spec = TRAINING_KNOBS[name]
    methods: tuple[str, ...] = tuple(spec["methods"])
    if method not in methods:
        raise ValueError(
            f"{name} is {spec['why']}; it applies to {', '.join(methods)} only"
            + (" (other methods do not sample)" if name in ("generations", "temperature") else "")
        )
    number = float(value)
    lo = float(spec["lo"])
    hi = None if spec["hi"] is None else float(spec["hi"])
    too_low = number <= lo if spec["open_lo"] else number < lo
    too_high = hi is not None and (number >= hi if spec["open_hi"] else number > hi)
    if too_low or too_high:
        ref = spec["ref"]
        ref_text = ref.get(method) if isinstance(ref, dict) else ref
        raise ValueError(f"{name}: {spec['range']} (reference {ref_text!r}; {spec['why']})")
    return number


def train(
    dataset: str,
    *,
    method: str = "sft",
    steps: int | None = None,
    epochs: float | None = None,
    holdout: str | None = None,
    base_model: str | None = None,
    generations: int | None = None,
    learning_rate: float | None = None,
    beta: float | None = None,
    seed: int | None = None,
    max_completion_length: int | None = None,
    loss_type: str | None = None,
    temperature: float | None = None,
    truncated: str | None = None,
    config: Mapping[str, Any] | None = None,
    check: str = "require",
    min_mixed_tasks: int = TRAIN_MIN_MIXED_TASKS,
    wait: bool = False,
    timeout: float | None = None,
    poll: float = TRAINING_POLL_S,
    api_key: str | None = None,
    transport: Callable[..., Any] | None = None,
) -> TrainingRun:
    """Start a hosted fine-tune on a pushed dataset and return the run.

    ``method`` is ``"sft"`` (LoRA on every row of the set, as pushed: the
    trainer does not filter on reward, so push ``scored.passes()``),
    ``"grpo"`` (a grouped update over the prompts that have both a pass
    and a fail; the reward is the trainer's own, ``reference first
    action`` against the judge's gold, and is not a parameter on this
    path), ``"dpo"`` (a pass against a fail per prompt, length matched) or
    ``"rm"`` (a reward model on those same pairs; ``reward_model(run)`` is
    then a judge). ``steps`` sets the optimizer steps for GRPO, DPO and
    RM, ``epochs`` the SFT epochs; each method has a default. ``holdout``
    names the eval set; it defaults to the train set's split sibling from
    ``datasets.cut``. ``base_model`` overrides the trainer's base; only
    ``SERVED_BASES`` can be served afterwards, and ``train`` warns when
    the run will not be.

    Which rows survive is checked before the GPU is spent (``check``,
    default ``"require"``): ``train`` reads ``wai.profile(dataset)`` and
    runs ``selection_report`` on it. SFT with failing rows is refused
    (``TrainingSelectionError``): the trainer clones every row and the
    model learns the failure (Lambert 2025, chapter Rejection Sampling;
    #396 measured it, tool use 0.99 to 0.49). A grouped method with no
    mixed task is refused too, since a unanimous group carries no
    advantage (#397 trained on 6 of 84 rows and ended at grad_norm 0).
    Fewer mixed tasks than ``min_mixed_tasks`` (``TRAIN_MIN_MIXED_TASKS``,
    32), or any dropped class at all, is a warning naming the count used
    against the count given and the reason per class;
    ``profile(dataset)["mixed_tasks"]`` is the number to size a grouped
    set by. ``check="warn"`` says the same and starts the run;
    ``check="off"`` does not read the profile. The report is on
    ``run.selection``.

    The run is the same record ``training_run`` makes, so ``run.url`` is
    the loss curve, ``run.delta`` and ``get_run`` work unchanged, and the
    trainer finishes it. ``run.refresh()`` reads where it is;
    ``run.wait()`` (or ``wait=True``) blocks until ``done`` or ``failed``,
    after which ``run.adapter`` names the weights and ``run.training``
    carries before, after, rows and seconds. ``serve`` puts the adapter on
    an endpoint.

    The knobs a run is reproduced and compared by (Lambert 2025, chapters
    Reinforcement Learning and Reasoning):
    ``generations`` is the group size per prompt for GRPO (the ``k`` the
    advantage is taken over; a pushed set's ``repeats`` is the natural
    value), ``beta`` the KL coefficient for GRPO and DPO, ``learning_rate``
    the optimizer step for every method, ``seed`` the sampling and data
    order seed, ``max_completion_length`` the token cap on a sampled reply
    (GRPO, DPO), ``loss_type`` the objective variant (GRPO: ``bnpo``,
    ``grpo``, ``dr_grpo``; DPO: any TRL loss). ``temperature`` is the
    sampling temperature the trainer rolls out at (GRPO); the dataset's
    rows say what they were measured at under ``sampling.temperature``,
    and ``train`` says so when the two differ, since a before/after
    comparison across temperatures is not like for like. ``truncated``
    says what GRPO does with a sampled reply the token cap cut:
    ``"mask"`` (the default) gives it no gradient, ``"zero"`` scores it 0
    the old way. A cut reply scored 0 teaches shorter thinking before it
    teaches the task, so ``"zero"`` is the knob to reach for only when the
    cap itself is the behavior under training (#253). Each has a
    trainer default when left ``None``; the range each is accepted in and
    the value the cited paper used are in ``TRAINING_KNOBS`` (defaults.py:
    DAPO, Dr. GRPO, ProRL, DPO and Lambert 2025), and a rejected
    value is told the reference. ``config`` passes further host keys as
    given (``epsilonHigh``, ``scaleRewards``, ``balance``).
    Every knob lands on the run's ``config`` so the run page shows it.

    A dataset already training answers with that run instead of a second.
    """
    method = str(method or "sft").lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}; got {method!r}")
    if not dataset:
        raise ValueError("dataset: the ds_... id of a pushed dataset")
    if check not in CHECK_MODES:
        raise ValueError(f"check must be one of {', '.join(CHECK_MODES)}; got {check!r}")
    if int(min_mixed_tasks) < 1:
        raise ValueError("min_mixed_tasks: at least 1 (TRAIN_MIN_MIXED_TASKS is 32)")
    if base_model not in SERVED_BASES:
        which = f"base_model={base_model!r}" if base_model else "the trainer's default base"
        warnings.warn(
            f"hosted {method} run on {dataset} uses {which}, which wai.serve cannot host "
            f"(served bases: {', '.join(SERVED_BASES)}); pass base_model={SERVED_BASES[0]!r} "
            "if the goal is an endpoint",
            stacklevel=2,
        )
    body: dict[str, Any] = {"method": method}
    if steps:
        body["steps"] = int(steps)
    if epochs:
        body["epochs"] = float(epochs)
    if holdout:
        body["holdoutId"] = str(holdout)
    if base_model:
        body["base"] = str(base_model)
    if generations is not None:
        body["generations"] = int(_knob("generations", int(generations), method))
    if learning_rate is not None:
        body["lr"] = _knob("learning_rate", learning_rate, method)
    if beta is not None:
        body["beta"] = _knob("beta", beta, method)
    if seed is not None:
        body["seed"] = int(seed)
    if max_completion_length is not None:
        body["maxCompletionLength"] = int(
            _knob("max_completion_length", int(max_completion_length), method)
        )
    if loss_type is not None:
        if method not in ("grpo", "dpo"):
            raise ValueError("loss_type picks the grpo or dpo objective variant")
        body["lossType"] = str(loss_type)
    if temperature is not None:
        body["temperature"] = _knob("temperature", temperature, method)
    if truncated is not None:
        if method != "grpo":
            raise ValueError("truncated= says what GRPO does with a token-capped reply; grpo only")
        if truncated not in TRUNCATED:
            raise ValueError(f"truncated must be one of {', '.join(TRUNCATED)}; got {truncated!r}")
    if method == "grpo":
        body["maskTruncated"] = (truncated or "mask") == "mask"
    for key, value in dict(config or {}).items():
        if key in body:
            raise ValueError(f"config[{key!r}] collides with a named argument")
        body[str(key)] = value
    call = transport or _call
    selection = (
        _check_selection(
            dataset,
            method,
            steps=int(steps) if steps else None,
            check=check,
            min_mixed_tasks=int(min_mixed_tasks),
            api_key=api_key,
            call=call,
        )
        if check != "off"
        else None
    )
    if temperature is not None:
        measured = _measured_temperature(dataset, api_key, call)
        if measured is not None and abs(measured - float(temperature)) > _TEMPERATURE_TOLERANCE:
            warnings.warn(
                f"Training samples at {float(temperature):g} but the dataset was measured at "
                f"{measured:g}; keep them the same or the before/after comparison is not "
                "like for like.",
                stacklevel=2,
            )
    out = call("POST", f"/datasets/{dataset}/train", api_key, body)
    state = dict((out or {}).get("training") or {}) if isinstance(out, dict) else {}
    run_id = str(state.get("runId") or "")
    if not run_id:
        raise RuntimeError(
            f"the platform started training on {dataset} but reported no run id: {out!r}"
        )
    run = TrainingRun(run_id, name=f"{dataset} · {method}", api_key=api_key, transport=transport)
    run._hosted = True
    run.dataset_id = str(dataset)
    run.method = method
    run.selection = selection
    run._absorb(state)
    if isinstance(out, dict) and out.get("alreadyRunning"):
        warnings.warn(
            f"dataset {dataset} is already training ({run.run_id}); returning that run",
            stacklevel=2,
        )
    log.info("hosted %s run %s on %s: %s", method, run.run_id, dataset, run.url)
    if wait:
        run.wait(timeout=timeout, poll=poll)
    return run


class RewardModel:
    """A finished ``method="rm"`` run as a judge (Lambert 2025, chapter
    Reward Modeling).

    Calling it with one rollout row honors the judge contract: ``reward``
    is 1 when the model's score clears the run's pass threshold, 0
    otherwise, and ``rm_score`` carries the raw number so ``judge_trust``,
    ``build_preference_pairs(min_margin=)`` and a margin-aware loss can
    use it. ``score(rows)`` scores a batch in one call. Rows are rendered
    on the platform exactly as the model was trained: the run's system
    prompt, the user prompt, the first assistant turn.
    """

    def __init__(
        self,
        run: TrainingRun | str,
        *,
        threshold: float | None = None,
        api_key: str | None = None,
        transport: Callable[..., Any] | None = None,
        batch: int = PLATFORM_REWARD_MODEL_BATCH,
    ):
        self.run_id = run.run_id if isinstance(run, TrainingRun) else str(run or "").strip()
        if not self.run_id:
            raise ValueError("run: a finished reward-model run (wai.train(method='rm')) or its id")
        self.threshold = threshold
        self.batch = max(1, int(batch))
        self.__name__ = f"reward_model:{self.run_id}"
        self._api_key = api_key
        self._call = transport or _call
        self.base: str | None = None

    def score(self, rows: Sequence[dict]) -> list[dict[str, Any]]:
        """``[{"rm_score", "reward", "threshold"}]`` for each row, in order.
        Up to ``batch`` rows a call (256 by default); more are sent in batches."""
        src = [r for r in rows if isinstance(r, dict)]
        out: list[dict[str, Any]] = []
        for i in range(0, len(src), self.batch):
            chunk = src[i : i + self.batch]
            res = self._call("POST", f"/runs/{self.run_id}/score", self._api_key, {"rows": chunk})
            res = res if isinstance(res, dict) else {}
            scores = list(res.get("scores") or [])
            if len(scores) != len(chunk):
                raise RuntimeError(
                    f"reward model {self.run_id} answered {len(scores)} scores for {len(chunk)} rows"
                )
            t = float(self.threshold if self.threshold is not None else res.get("threshold", 0.0))
            self.base = self.base or res.get("base")
            out += [
                {"rm_score": float(s), "reward": int(float(s) >= t), "threshold": t} for s in scores
            ]
        return out

    def __call__(self, trajectory: dict) -> dict[str, Any]:
        one = self.score([trajectory])[0]
        verdict = ">=" if one["reward"] else "<"
        return {
            "reward": one["reward"],
            "rm_score": one["rm_score"],
            "threshold": one["threshold"],
            "reason": f"reward model {self.run_id}: {one['rm_score']:.3f} {verdict} {one['threshold']:.3f}",
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RewardModel({self.run_id!r}, threshold={self.threshold})"


def reward_model(
    run: TrainingRun | str,
    *,
    threshold: float | None = None,
    api_key: str | None = None,
    transport: Callable[..., Any] | None = None,
    batch: int = PLATFORM_REWARD_MODEL_BATCH,
) -> RewardModel:
    """A judge backed by a finished reward-model run.

    ``run = wai.train("ds_...", method="rm", wait=True)`` trains a
    sequence-classification head on the set's pass-vs-fail pairs and
    picks the score threshold that best separates the held-out pairs.
    ``judge = wai.reward_model(run)`` then scores any rollout row:
    ``data.grade(judge=judge)``, ``wai.evaluate(rollouts, judge)``,
    ``wai.judge_trust(scored.rows, judge=judge)``. Pass ``threshold=`` to
    override the run's own cut. The scores are the model's; a reward
    model trained on one agent's pairs says nothing about another agent.
    """
    return RewardModel(run, threshold=threshold, api_key=api_key, transport=transport, batch=batch)


def models(*, api_key: str | None = None) -> list[dict[str, Any]]:
    """The account's hosted models: ``name``, ``baseModel``, ``adapter``,
    ``adapterRunId``, ``version``, ``endpoint`` (an OpenAI-compatible
    base URL; send the account key as the bearer and ``name`` as the
    model).

    A row here is a registry entry, not a running GPU: the endpoint
    behind it idles to zero on its own and an unused model costs nothing.
    The row stays until ``unserve(name)`` removes it; serving the same
    name again bumps its ``version`` rather than adding a row."""
    out = _call("GET", "/models", api_key)
    return list(out.get("models") or []) if isinstance(out, dict) else []


def unserve(
    name: str,
    *,
    api_key: str | None = None,
    transport: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Stop hosting ``name``: removes the model row from the account, so
    ``models()`` no longer lists it and its endpoint stops answering for
    that name. The inverse of ``serve``, the way ``delete_dataset`` is the
    inverse of ``push``. The adapter weights and the training run stay;
    ``serve`` the run again to bring it back (at version 1).
    Returns ``{"name": ..., "deleted": True}``."""
    call = transport or _call
    key = str(name).strip().lower()
    if not key:
        raise ValueError("unserve: name is the hosted model's name, as models() lists it")
    out = call("DELETE", f"/models/{key}", api_key)
    return dict(out) if isinstance(out, dict) else {"name": key, "deleted": True}


#: Same call, the other spelling: symmetric with ``delete_dataset``.
delete_model = unserve


def serve(
    name: str,
    run: TrainingRun | str | None = None,
    *,
    base_model: str | None = None,
    api_key: str | None = None,
    transport: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Host a finished run's adapter under ``name``. Returns the model
    row; ``endpoint`` is the OpenAI-compatible base URL and ``name`` the
    model id to send. Posting an existing name bumps ``version``.

    ``run`` is a ``TrainingRun``, the record ``get_run`` returns, or the
    run id; the adapter and base model come from the run record unless
    ``base_model`` is given. No ``run`` serves the bare base
    (``base_model`` required). ``unserve`` is the inverse.
    """
    call = transport or _call
    adapter: str | None = None
    base = base_model
    run_id: str | None
    if isinstance(run, TrainingRun):
        adapter = run.adapter
        run_id = run.run_id
    elif isinstance(run, dict):
        # The record ``get_run`` returns. Reading the id here is what lets
        # train in one process and serve in the next compose (#262).
        run_id = str(run.get("runId") or run.get("run_id") or run.get("id") or "").strip() or None
        if not run_id:
            raise TypeError(
                "run is a dict with no runId; pass the record wai.get_run(run_id) returns, "
                "a TrainingRun, or the run id string"
            )
        adapter = str(run.get("adapter") or "").strip() or None
        base = base or run.get("baseModel") or run.get("base_model")
    elif run is None or isinstance(run, str):
        run_id = (str(run).strip() or None) if run else None
    else:
        raise TypeError(
            "run must be a TrainingRun, the record wai.get_run(run_id) returns, or the run "
            f"id string; got {type(run).__name__}"
        )
    if run_id and (adapter is None or base is None):
        meta = call("GET", f"/runs/{run_id}", api_key)
        meta = meta if isinstance(meta, dict) else {}
        adapter = adapter or meta.get("adapter")
        base = base or meta.get("baseModel") or meta.get("base_model")
        if not adapter:
            status = str(meta.get("status") or "?")
            if status in ("failed", "stopped"):
                raise ValueError(
                    f"run {run_id} {status} and produced no adapter: "
                    f"{meta.get('error') or 'no error recorded'}"
                )
            raise ValueError(f"run {run_id} has no adapter yet (status {status}); wait for it")
    if not base:
        raise ValueError("base_model: which served base the adapter was trained on")
    if base not in SERVED_BASES:
        raise ValueError(
            f"{base} is not a served base ({', '.join(SERVED_BASES)}); the adapter"
            + (f" from run {run_id}" if run_id else "")
            + f" cannot be hosted. Train with base_model={SERVED_BASES[0]!r} for an endpoint"
        )
    body: dict[str, Any] = {"name": str(name).strip().lower(), "baseModel": str(base)}
    if adapter:
        body["adapter"] = str(adapter)
    out = call("POST", "/models", api_key, body)
    row = dict(out) if isinstance(out, dict) else {"name": body["name"]}
    log.info("hosted model %s v%s at %s", row.get("name"), row.get("version"), row.get("endpoint"))
    return row


def list_runs(*, api_key: str | None = None) -> list[dict[str, Any]]:
    out = _call("GET", "/runs", api_key)
    return list(out.get("runs") or []) if isinstance(out, dict) else []


def get_run(run_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    """The run plus ``series``: its points, oldest first."""
    return _call("GET", f"/runs/{run_id}", api_key)


def delete_run(run_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    return _call("DELETE", f"/runs/{run_id}", api_key)


def attach_delta(
    run_id: str,
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    target: str | None = "pass_at_1",
    must_not_regress: Sequence[str] = (),
    by: str | Callable[[dict], Any] | None = None,
    proxy: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Compute ``delta_report`` for a finished run and put it on the run
    page: the summary is re-sent with ``delta`` added, status unchanged."""
    from .score.delta import delta_report

    run = _call("GET", f"/runs/{run_id}", api_key)
    report = delta_report(
        before,
        after,
        target=target,
        must_not_regress=list(must_not_regress),
        by=by,
        proxy=proxy,
    )
    summary = dict(run.get("summary") or {})
    summary["delta"] = _json_safe(report)
    for key, value in _holdout_from_delta(report).items():
        summary.setdefault(key, value)
    summary["holdout"] = _json_safe(_holdout_block(report, before, after))
    _finish_again(run_id, run, summary, api_key)
    return report


def _finish_again(
    run_id: str, run: Mapping[str, Any], summary: dict[str, Any], api_key: str | None
) -> None:
    """Re-send a finished run's summary with the status it already has."""
    status = str(run.get("status") or "done")
    if status == "running":
        status = "done"
    _call("POST", f"/runs/{run_id}/finish", api_key, {"status": status, "summary": summary})


def attach_holdout(
    run_id: str,
    before: float,
    after: float,
    *,
    metric: str = "pass",
    api_key: str | None = None,
) -> dict[str, float]:
    """Did it work? Put the held-out pass rate before and after on a run
    that has already finished — the two numbers its page opens with::

        wai.attach_holdout("run_...", before=0.42, after=0.58)

    Pass rates are 0 to 1. ``metric="loss"`` sends held-out loss instead
    (SFT), where lower is better. The summary is re-sent with the two keys
    added and the status unchanged; sending again is a correction.
    """
    fields = _holdout_summary(before, after, metric)
    run = _call("GET", f"/runs/{run_id}", api_key)
    summary = dict(run.get("summary") or {})
    summary.update(fields)
    _finish_again(run_id, run, summary, api_key)
    log.info("run %s held out: %s", run_id, fields)
    return fields


# ---------------------------------------------------------------- Transformers / TRL


_EVENTS = (
    "on_init_end",
    "on_train_begin",
    "on_train_end",
    "on_epoch_begin",
    "on_epoch_end",
    "on_step_begin",
    "on_substep_end",
    "on_step_end",
    "on_optimizer_step",
    "on_pre_optimizer_step",
    "on_evaluate",
    "on_predict",
    "on_save",
    "on_log",
    "on_prediction_step",
)


class _NoOpCallback:
    """Stand-in base when transformers is not installed: every event the
    Trainer fires is a no-op, so the duck-typed callback still fits."""


for _event in _EVENTS:
    setattr(_NoOpCallback, _event, lambda self, *a, **k: None)


def _callback_base() -> type:
    try:
        from transformers import TrainerCallback as HFCallback

        return HFCallback
    except Exception:  # transformers is not a dependency of this package
        return _NoOpCallback


# Timing and throughput keys the Trainer logs alongside eval metrics; not
# learning signal, so not drawn.
_SKIP_KEYS = {
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "total_flos",
    "step",
}

_LOG_KEYS = {
    "loss": "loss",
    "eval_loss": "eval_loss",
    "learning_rate": "lr",
    "epoch": "epoch",
    "grad_norm": "grad_norm",
    "train_loss": "train_loss",
    "mean_token_accuracy": "token_accuracy",
    "eval_mean_token_accuracy": "eval_token_accuracy",
    "num_tokens": "tokens",
    # TRL RL trainers (GRPO, PPO, RLOO, online DPO): the curves an RL run is
    # read by. Any ``rewards/<name>`` key is kept under ``reward_<name>``.
    "reward": "reward",
    "reward_std": "reward_std",
    "kl": "kl",
    "objective/kl": "kl",
    "objective/rlhf_reward": "reward",
    "objective/scores": "score",
    "objective/entropy": "entropy",
    "entropy": "entropy",
    "completion_length": "completion_length",
    "completions/mean_length": "completion_length",
    # Share of rollouts that hit max_completion_length: the length-cap share the
    # platform's Rollouts tile draws as clip_ratio, not the policy-ratio clip.
    "completions/clipped_ratio": "clip_ratio",
    "clip_ratio": "clip_ratio",
    "policy_loss": "policy_loss",
    "value_loss": "value_loss",
}


class TrainerCallback(_callback_base()):  # type: ignore[misc]  # ty: ignore[unsupported-base]
    """One line on a Transformers or TRL trainer:
    ``trainer.add_callback(wai.TrainerCallback(run))``.

    Logs every ``on_log`` to the run (loss, lr, eval loss, epoch, grad
    norm, token accuracy), takes the step count from the trainer at
    ``on_train_begin``, and finishes the run at ``on_train_end``. If the
    trainer raises, finish the run yourself with ``run.fail(...)`` or use
    the run as a context manager around ``trainer.train()``.
    """

    def __init__(self, run: TrainingRun, *, finish: bool = True):
        super().__init__()
        self.run = run
        #: finish the run at on_train_end. Pass False when the script
        #: evaluates after training and calls run.finish itself.
        self.finish_on_end = finish

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        total = getattr(state, "max_steps", None)
        if total:
            self.run.progress(int(getattr(state, "global_step", 0) or 0), int(total))
        return control

    def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
        if not isinstance(logs, dict):
            return control
        metrics = {}
        for key, value in logs.items():
            if key in _SKIP_KEYS:
                continue
            name = _LOG_KEYS.get(key)
            if name is None and key.startswith("rewards/"):
                name = "reward_" + key[len("rewards/") :].replace("/", "_")
            if name is None and key.startswith("eval_") and isinstance(value, (int, float)):
                name = key
            if name is not None:
                metrics[name] = value
        step = int(getattr(state, "global_step", 0) or 0)
        if metrics:
            self.run.log(step, **metrics)
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        if self.finish_on_end and self.run.status == "running":
            summary: dict[str, Any] = {}
            history = getattr(state, "log_history", None) or []
            for entry in reversed(history):
                if isinstance(entry, dict) and "train_loss" in entry:
                    summary["train_loss"] = entry["train_loss"]
                    break
            self.run.finish("done", summary=summary or None)
        return control


__all__ = [
    "TrainerCallback",
    "TrainingRun",
    "TrainingSelectionError",
    "attach_delta",
    "attach_holdout",
    "delete_run",
    "get_run",
    "list_runs",
    "models",
    "selection_report",
    "serve",
    "train",
    "training_run",
]
