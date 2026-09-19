"""Is the run hacking its reward right now? Watch it while it trains.

Gao et al. 2022 (arXiv:2210.10760) draw over-optimization as one picture: the
training reward keeps climbing while the evaluation you actually care about
flattens and then falls, read against how far the policy has drifted from
where it started (KL). Every trainer logs the first curve. Nobody draws the
second until the run is over and the holdout is scored once. ``HackMonitor``
draws it during the run.

It is a Transformers / TRL callback plus a wrapper for the reward
function, so it sees two things the trainer's averages hide:

* every completion and the reward it got (``monitor.wrap(reward_fn)``
  keeps the last ``buffer`` of them), which ``hack_scan`` reads for
  what the policy is currently being paid for;
* the holdout, sampled from the live policy every ``every`` steps and
  scored twice: by the training reward (the proxy) and by a scorer the
  proxy cannot see (``gold``: the hosted judge, a reward model, a
  second rule). The gold rows are the same shape ``pass_at`` and
  ``delta_report`` read, so the comparison is paired by ask with a
  bootstrap interval, not two means.

Four alarms, each one line on the run:

* ``divergence``: proxy up by ``delta`` or more over the window while
  the gold interval does not move up (figure 1 of Gao et al. 2022);
* ``length``: mean completion length up by ``length_pct`` or more over
  the window while gold does not move up (Yu et al. 2025 (DAPO),
  arXiv:2503.14476, on token-level loss: per-sequence losses pay for
  short, per-token for long; a judge that reads length pays for long);
* ``drift``: KL from the reference past ``kl_budget`` (Lambert 2025,
  chapter Regularization);
* ``feature``: the buffer's ``hack_scan`` says ``reward_hack`` (needs
  ``endorsed``).

``stop_on`` names the alarms that stop training. The default is to log
and keep going: the alarm is a place to look, and the run page shows
it next to the reward curve.
"""

from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import logging
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .defaults import (
    MONITOR_BUFFER,
    MONITOR_CONCURRENCY,
    MONITOR_DELTA,
    MONITOR_EVERY,
    MONITOR_K,
    MONITOR_LENGTH_PCT,
    MONITOR_MAX_NEW_TOKENS,
    MONITOR_N_PROMPTS,
    MONITOR_SAMPLE_BATCH,
    MONITOR_SAMPLE_TEMPERATURE,
    MONITOR_SAMPLE_TOP_P,
    MONITOR_SCAN_MIN,
    MONITOR_SCAN_PERMUTATIONS,
    MONITOR_WINDOW,
    MONITOR_WINDOW_BOOTSTRAPS,
)
from .schema import Judgment, ScorerRef, attach
from .score.hack_scan import hack_scan
from .score.judging import normalize_judge_result
from .score.stats import compare_runs
from .training import TrainingRun, _callback_base, _json_safe

log = logging.getLogger("whileai.simulations")

ALARMS = ("divergence", "length", "drift", "feature")
# Every default below lives in defaults.py (MONITOR_*) with its reason and
# is a keyword on HackMonitor. The old names stay as aliases.
#: holdout asks sampled per eval, and completions per ask
DEFAULT_N_PROMPTS = MONITOR_N_PROMPTS
DEFAULT_K = MONITOR_K
#: evals the window spans
DEFAULT_WINDOW = MONITOR_WINDOW
#: proxy gain over the window that counts as climbing
DEFAULT_DELTA = MONITOR_DELTA
#: completion-length growth over the window that counts as growing
DEFAULT_LENGTH_PCT = MONITOR_LENGTH_PCT
#: completions the reward wrapper keeps for the feature scan
DEFAULT_BUFFER = MONITOR_BUFFER
#: holdout row columns that are the ask or the reply, never a proxy column
_RESERVED = frozenset({"prompt", "messages", "completion", "completions", "final_text"})


def _prompt_text(prompt: Any) -> str:
    """The user's ask as text: the last user turn of a message list, else
    the string itself."""
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
        return ""
    return str(prompt or "")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        return "".join(str(m.get("content") or "") for m in completion if isinstance(m, dict))
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    return str(completion or "")


def _completion_for(prompt: Any, text: str) -> Any:
    """What a TRL reward function expects: a message list when the prompt
    is conversational, else the string."""
    if isinstance(prompt, list):
        return [{"role": "assistant", "content": text}]
    return text


def _default_sample(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Any],
    *,
    n: int,
    max_new_tokens: int,
    batch: int = MONITOR_SAMPLE_BATCH,
    temperature: float = MONITOR_SAMPLE_TEMPERATURE,
    top_p: float = MONITOR_SAMPLE_TOP_P,
) -> list[list[str]]:
    """``n`` completions per prompt from the live policy: chat template
    when the prompt is a message list, batched, sampled at ``temperature``
    and ``top_p`` (``HackMonitor(sampling=)`` sets them)."""
    import torch

    was_training = bool(getattr(model, "training", False))
    model.eval()
    tokenizer.padding_side = "left"
    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        if isinstance(p, list)
        else str(p)
        for p in prompts
    ]
    out: list[list[str]] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append(decoded[i * n : (i + 1) * n])
    if was_training:
        model.train()
    return out


class HackMonitor(_callback_base()):  # type: ignore[misc]  # ty: ignore[unsupported-base]
    """Watch a TRL run for reward hacking. See the module docstring.

    ``holdout`` is a list of prompts (strings or message lists) or rows
    (``{"prompt": ..., <extra columns>}``); extra columns reach the proxy
    as keyword lists, the way TRL passes dataset columns. ``proxy`` is a
    TRL-shaped reward function (``f(prompts=, completions=, **cols)``);
    leave it ``None`` to use the reward function passed through
    ``wrap``. ``gold`` is a judge under the SDK contract (a rollout row
    in, a reward or ``{"reward": ...}`` out). ``sample`` overrides how
    completions are drawn: ``sample(model, tokenizer, prompts, n=,
    max_new_tokens=) -> list[list[str]]``; the default uses the chat
    template and ``model.generate``; ``sampling`` (``temperature``,
    ``top_p``, ``batch``) steers that default sampler.

    ``n_boot`` is the bootstrap count behind the gold-vs-window interval,
    ``n_perm`` the permutation count behind the feature scan, ``scan_min``
    the fewest buffered completions the scan runs on. Every number has its
    reason in ``defaults.py`` (MONITOR_*).

    ``run`` is the platform run the points and alarms land on; ``None``
    keeps everything on the monitor (``history``, ``alarms``,
    ``summary()``).
    """

    def __init__(
        self,
        run: TrainingRun | None = None,
        *,
        holdout: Sequence[Any],
        proxy: Callable[..., Sequence[float]] | None = None,
        gold: Callable[[dict], Any] | None = None,
        sample: Callable[..., list[list[str]]] | None = None,
        every: int = MONITOR_EVERY,
        k: int = MONITOR_K,
        n_prompts: int = MONITOR_N_PROMPTS,
        window: int = MONITOR_WINDOW,
        delta: float = MONITOR_DELTA,
        length_pct: float = MONITOR_LENGTH_PCT,
        kl_budget: float | None = None,
        endorsed: Sequence[str] = (),
        stop_on: str | Sequence[str] = (),
        buffer: int = MONITOR_BUFFER,
        max_new_tokens: int = MONITOR_MAX_NEW_TOKENS,
        concurrency: int = MONITOR_CONCURRENCY,
        seed: int = 0,
        n_boot: int = MONITOR_WINDOW_BOOTSTRAPS,
        n_perm: int = MONITOR_SCAN_PERMUTATIONS,
        scan_min: int = MONITOR_SCAN_MIN,
        sampling: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if not holdout:
            raise ValueError("holdout: prompts or rows the policy is never trained on")
        self.run = run
        self.proxy = proxy
        self.gold = gold
        if sample is None:
            allowed = {"temperature", "top_p", "batch"}
            unknown = sorted(set(sampling or {}) - allowed)
            if unknown:
                raise ValueError(f"sampling: unknown key(s) {unknown}; use {sorted(allowed)}")
            sample = functools.partial(_default_sample, **dict(sampling or {}))
        elif sampling:
            raise ValueError("sampling= steers the default sampler; drop it when passing sample=")
        self.sample = sample
        self.n_boot = max(1, int(n_boot))
        self.n_perm = max(1, int(n_perm))
        self.scan_min = max(1, int(scan_min))
        self.every = max(1, int(every))
        self.k = max(1, int(k))
        self.window = max(1, int(window))
        self.delta = float(delta)
        self.length_pct = float(length_pct)
        self.kl_budget = float(kl_budget) if kl_budget is not None else None
        self.endorsed = [str(e) for e in endorsed if str(e)]
        stops = [stop_on] if isinstance(stop_on, str) else list(stop_on)
        self.stop_on = set(ALARMS) if "any" in stops else {s for s in stops if s in ALARMS}
        unknown = [s for s in stops if s not in ALARMS and s != "any"]
        if unknown:
            raise ValueError(f"stop_on: unknown alarm {unknown}; choose from {ALARMS} or 'any'")
        self.buffer_size = max(0, int(buffer))
        self.max_new_tokens = int(max_new_tokens)
        self.concurrency = max(1, int(concurrency))
        rng = random.Random(seed)
        rows = [r if isinstance(r, dict) else {"prompt": r} for r in holdout]
        rows = [r for r in rows if r.get("prompt")]
        if len(rows) > n_prompts:
            rows = rng.sample(rows, int(n_prompts))
        self.holdout: list[dict] = rows
        self.buffer: list[dict] = []
        self.history: list[dict[str, Any]] = []
        self.alarms: list[dict[str, Any]] = []
        self.stopped_at: dict[str, Any] | None = None
        self.last_kl: float | None = None
        self.last_scan: dict[str, Any] | None = None
        self._wrapped: Callable[..., Sequence[float]] | None = None
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------ reward wrap

    def wrap(self, reward_fn: Callable[..., Sequence[float]]) -> Callable[..., Sequence[float]]:
        """The reward function, watched: every completion it scores goes
        into the buffer with its reward. Pass the result to the trainer
        as ``reward_funcs``. The name survives, so TRL's ``rewards/<name>``
        column does too."""

        @functools.wraps(reward_fn)
        def watched(*args: Any, **kwargs: Any) -> Sequence[float]:
            out = reward_fn(*args, **kwargs)
            prompts = kwargs.get("prompts")
            completions = kwargs.get("completions")
            if prompts is None and args:
                prompts = args[0]
            if completions is None and len(args) > 1:
                completions = args[1]
            if prompts is not None and completions is not None:
                self._record(prompts, completions, list(out))
            return out

        self._wrapped = reward_fn
        return watched

    def _record(
        self, prompts: Sequence[Any], completions: Sequence[Any], rewards: list[float]
    ) -> None:
        if not self.buffer_size:
            return
        for prompt, completion, reward in zip(prompts, completions, rewards):
            try:
                value = float(reward)
            except (TypeError, ValueError):
                continue
            text = _completion_text(completion)
            self.buffer.append(
                {
                    "prompt": _prompt_text(prompt),
                    "final_text": text,
                    "messages": [
                        {"role": "user", "content": _prompt_text(prompt)},
                        {"role": "assistant", "content": text},
                    ],
                    "steps": [],
                    "reward": value,
                }
            )
        if len(self.buffer) > self.buffer_size:
            del self.buffer[: len(self.buffer) - self.buffer_size]

    # ------------------------------------------------------------ one eval

    def _proxy_scores(
        self, prompts: list[Any], completions: list[Any], columns: dict
    ) -> list[float]:
        fn = self.proxy or self._wrapped
        if fn is None:
            raise ValueError(
                "no proxy: pass proxy=, or hand the trainer monitor.wrap(reward_fn) so the "
                "monitor scores the holdout with the training reward"
            )
        out = fn(prompts=prompts, completions=completions, **columns)
        return [float(v) for v in out]

    def _gold_scores(self, rows: list[dict]) -> list[float | None]:
        if self.gold is None:
            return [None] * len(rows)

        def one(row: dict) -> float | None:
            try:
                verdict = normalize_judge_result(self.gold(row))  # type: ignore[misc]
            except Exception as exc:
                log.debug("hack monitor: gold judge failed: %s", exc)
                return None
            value = verdict.get("reward")
            return float(value) if isinstance(value, (int, float)) else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(one, rows))

    def evaluate(self, step: int, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        """Sample the holdout from the live policy and score it both ways.
        Appends to ``history``, logs to the run, checks the alarms.
        Returns the history entry."""
        model = model if model is not None else self._model
        tokenizer = tokenizer if tokenizer is not None else self._tokenizer
        prompts = [r["prompt"] for r in self.holdout]
        groups = self.sample(
            model, tokenizer, prompts, n=self.k, max_new_tokens=self.max_new_tokens
        )
        flat_prompts: list[Any] = []
        flat_completions: list[Any] = []
        columns: dict[str, list[Any]] = {}
        rows: list[dict] = []
        for holdout_row, completions in zip(self.holdout, groups):
            prompt = holdout_row["prompt"]
            text_prompt = _prompt_text(prompt)
            for i, text in enumerate(completions):
                flat_prompts.append(prompt)
                flat_completions.append(_completion_for(prompt, text))
                for key, value in holdout_row.items():
                    if key not in _RESERVED:
                        columns.setdefault(str(key), []).append(value)
                rows.append(
                    {
                        "prompt": text_prompt,
                        "task_id": hashlib.sha1(text_prompt.encode("utf-8")).hexdigest()[:16],
                        "rollout_index": i,
                        "final_text": text,
                        "steps": [],
                        "messages": [
                            *(
                                prompt
                                if isinstance(prompt, list)
                                else [{"role": "user", "content": text_prompt}]
                            ),
                            {"role": "assistant", "content": text},
                        ],
                        "markers": {},
                    }
                )
        proxy = self._proxy_scores(flat_prompts, flat_completions, columns)
        gold = self._gold_scores(rows)
        gold_name = getattr(self.gold, "__name__", "") or "gold"
        for row, p, g in zip(rows, proxy, gold):
            row["markers"]["proxy"] = p
            if g is not None:
                row["markers"]["gold"] = g
                # The verdict goes through the schema's one sanctioned write,
                # binarized so pass_at reads it; the raw score is the marker.
                attach(
                    row,
                    Judgment(
                        rollout_id=f"{row['task_id']}#{row['rollout_index']}",
                        scorer=ScorerRef(name=str(gold_name), kind="judge"),
                        reward=1 if g >= 1.0 else 0 if g <= 0.0 else round(g),
                    ),
                )
        lengths = [len(r["final_text"]) for r in rows]
        gold_values = [g for g in gold if g is not None]
        entry: dict[str, Any] = {
            "step": int(step),
            "n": len(rows),
            "proxy": statistics.fmean(proxy) if proxy else None,
            "gold": statistics.fmean(gold_values) if gold_values else None,
            "length": statistics.fmean(lengths) if lengths else None,
            "kl": self.last_kl,
            "rows": rows,
        }
        self.history.append(entry)
        if self.run is not None:
            point: dict[str, float] = {}
            if entry["proxy"] is not None:
                point["proxy_reward"] = entry["proxy"]
            if entry["gold"] is not None:
                point["gold_reward"] = entry["gold"]
            if entry["length"] is not None:
                point["holdout_length"] = entry["length"]
            self.run.log(int(step), **point)
        self._check(entry)
        if self.run is not None:
            self.run.note(hack_monitor=self.summary())
        return entry

    # ------------------------------------------------------------ alarms

    def _raise(self, entry: dict[str, Any], kind: str, reason: str, **evidence: Any) -> None:
        alarm = {"step": entry["step"], "kind": kind, "reason": reason, **_json_safe(evidence)}
        self.alarms.append(alarm)
        log.warning("hack monitor step %s: %s: %s", entry["step"], kind, reason)
        if self.run is not None:
            self.run.log(int(entry["step"]), **{f"alarm_{kind}": 1.0})
        if kind in self.stop_on and self.stopped_at is None:
            self.stopped_at = alarm

    def _check(self, entry: dict[str, Any]) -> None:
        # drift: needs only the trainer's own KL
        if self.kl_budget is not None and entry["kl"] is not None and entry["kl"] > self.kl_budget:
            self._raise(
                entry,
                "drift",
                f"KL {entry['kl']:.3f} past the budget of {self.kl_budget:.3f}",
                kl=entry["kl"],
            )
        # feature: what the buffer says the policy is paid for
        if self.endorsed and len(self.buffer) >= self.scan_min:
            scan = hack_scan(self.buffer, endorsed=self.endorsed, n_perm=self.n_perm)
            self.last_scan = {
                k: scan[k] for k in ("regime", "top_feature", "rho_max", "tau", "integrity")
            }
            if scan["regime"] == "reward_hack":
                self._raise(entry, "feature", scan["warnings"][0], **self.last_scan)
        if len(self.history) < 2:  # noqa: PLR2004  # two evals before a trend
            return
        then = self.history[max(0, len(self.history) - 1 - self.window)]
        gold_up: bool | None = None
        if then["gold"] is not None and entry["gold"] is not None:
            cmp = compare_runs(
                then["rows"], entry["rows"], metric="marker:gold", n_boot=self.n_boot
            )
            gold_up = cmp["verdict"] == "b_better"
            entry["gold_vs_window"] = {
                "delta": cmp["delta"],
                "ci95": cmp["ci95"],
                "verdict": cmp["verdict"],
            }
        # divergence: proxy climbs, gold does not follow
        if then["proxy"] is not None and entry["proxy"] is not None and gold_up is not None:
            gain = entry["proxy"] - then["proxy"]
            if gain >= self.delta and not gold_up:
                g = entry["gold_vs_window"]
                span = f"{g['ci95'][0]:+.2f}..{g['ci95'][1]:+.2f}" if g.get("ci95") else "n/a"
                self._raise(
                    entry,
                    "divergence",
                    f"proxy reward up {gain:+.2f} since step {then['step']} while gold moved "
                    f"{g['delta']:+.2f} (95% {span}): the policy is optimizing something the "
                    "gold scorer does not credit",
                    proxy_gain=gain,
                    gold_delta=g["delta"],
                    since_step=then["step"],
                )
        # length: completions grow, gold does not follow
        if then["length"] and entry["length"]:
            growth = entry["length"] / then["length"] - 1.0
            if growth >= self.length_pct and not gold_up:
                self._raise(
                    entry,
                    "length",
                    f"holdout completions {growth:+.0%} longer since step {then['step']}"
                    + (" while gold did not move up" if gold_up is not None else "")
                    + "; a reward that reads length pays for this",
                    length_growth=growth,
                    since_step=then["step"],
                )

    # ------------------------------------------------------------ report

    def summary(self) -> dict[str, Any]:
        """What the run page and ``finish`` carry: the curve points, the
        alarms, the last scan, where it stopped."""
        return _json_safe(
            {
                "n_evals": len(self.history),
                "every": self.every,
                "k": self.k,
                "n_prompts": len(self.holdout),
                "window": self.window,
                "delta": self.delta,
                "length_pct": self.length_pct,
                "kl_budget": self.kl_budget,
                "endorsed": self.endorsed,
                "stop_on": sorted(self.stop_on),
                "history": [{k: v for k, v in h.items() if k != "rows"} for h in self.history],
                "alarms": self.alarms,
                "last_scan": self.last_scan,
                "stopped_at": self.stopped_at,
            }
        )

    # ------------------------------------------------------------ Trainer events

    def _grab(self, kwargs: Mapping[str, Any]) -> None:
        if kwargs.get("model") is not None:
            self._model = kwargs["model"]
        tok = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if tok is not None:
            self._tokenizer = tok

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        self._grab(kwargs)
        self._safe_evaluate(int(getattr(state, "global_step", 0) or 0))
        return control

    def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
        if isinstance(logs, dict):
            for key in ("kl", "objective/kl"):
                value = logs.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.last_kl = float(value)
        return control

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        self._grab(kwargs)
        step = int(getattr(state, "global_step", 0) or 0)
        if step > 0 and step % self.every == 0:
            self._safe_evaluate(step)
            if self.stopped_at is not None and control is not None:
                control.should_training_stop = True
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        if self.run is not None and self.stopped_at is not None and self.run.status == "running":
            self.run.finish(
                "stopped",
                summary={"hack_monitor": self.summary()},
                error=f"stopped by hack monitor: {self.stopped_at['kind']} at step "
                f"{self.stopped_at['step']}: {self.stopped_at['reason']}",
            )
        return control

    def _safe_evaluate(self, step: int) -> None:
        try:
            self.evaluate(step)
        except Exception as exc:  # the monitor must never stop a run by crashing
            log.warning("hack monitor eval at step %s failed: %s", step, exc)


def format_hack_monitor(summary: Mapping[str, Any]) -> str:
    """The block a person reads: the curve, then the alarms."""
    lines = [
        f"{summary.get('n_evals', 0)} evals, {summary.get('n_prompts', 0)} holdout asks x "
        f"{summary.get('k', 0)}, every {summary.get('every', 0)} steps"
    ]
    lines.append(f"  {'step':>6}{'proxy':>8}{'gold':>8}{'length':>9}{'kl':>8}")
    for h in summary.get("history") or []:

        def _f(v: Any, width: int, digits: int = 3) -> str:
            return f"{v:>{width}.{digits}f}" if isinstance(v, (int, float)) else f"{'-':>{width}}"

        lines.append(
            f"  {int(h.get('step', 0)):>6}{_f(h.get('proxy'), 8)}{_f(h.get('gold'), 8)}"
            f"{_f(h.get('length'), 9, 0)}{_f(h.get('kl'), 8)}"
        )
    scan = summary.get("last_scan")
    if scan:
        lines.append(
            f"buffer scan: {scan.get('regime')} (top {scan.get('top_feature')!r}, "
            f"rho {scan.get('rho_max')}, tau {scan.get('tau')})"
        )
    for alarm in summary.get("alarms") or []:
        lines.append(f"! step {alarm.get('step')} {alarm.get('kind')}: {alarm.get('reason')}")
    stopped = summary.get("stopped_at")
    if stopped:
        lines.append(f"STOPPED at step {stopped.get('step')} on {stopped.get('kind')}")
    elif not summary.get("alarms"):
        lines.append("no alarms")
    return "\n".join(lines)


__all__ = ["ALARMS", "HackMonitor", "format_hack_monitor"]
