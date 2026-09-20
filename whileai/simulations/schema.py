"""The typed row: four objects, one flat wire shape, one version stamp.

Every row the SDK writes or accepts is a projection of four objects:

* ``Task``: the situation. Static, shippable, never contains a rollout.
* ``Rollout``: one episode of one policy on one task. Never contains a
  verdict.
* ``Judgment``: one scorer's verdict on a rollout. Many per rollout.
* ``Marker``: one behavior measurement on a rollout. Many per rollout.

The flat JSONL row that ``simulate()`` streams, ``save()`` writes, and the
platform stores is ``to_row(task, rollout, judgments, markers)``; the
inverse is ``from_row``. The wire shape is described by
``schemas/row-v1.json`` and every row carries ``schema_version``.

Version 0 is every row written before the stamp existed. ``from_row``
recognizes the legacy shapes by key (engine rows by ``scenario_id``,
training exports by ``messages`` without ``steps``, platform trace pulls
by ``tool_trace``, OTel ingest by ``conversation_id``, the published
Hugging Face set by its ``*_json`` string columns) and normalizes them
the way ``load_traces`` does, reusing its alias tables rather than keeping
a second copy. Keys the objects do not model ride through untouched, so
``to_row(*from_row(row))`` never loses a column it did not understand.

Validation is permissive on purpose in this version: a stamped row must
have the required fields with the right types; an unstamped row only has
to be a dict. Stricter checks land after the store moves to the objects.
Until then ``SimulationData.trajectories`` stays the source of truth and
these objects are the view.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .defaults import MESSAGE_EXAMPLES

if TYPE_CHECKING:
    from .data import RowList

SCHEMA_VERSION = "1"
SCHEMA_KEY = "schema_version"
KNOWN_VERSIONS = frozenset({"0", SCHEMA_VERSION})

#: The sampled diversity axes. Same tuple as ``data._CONVERSATION_FIELDS``;
#: the drift test holds them equal.
AXES = (
    "tier",
    "ask_family",
    "intent_known",
    "tool_known",
    "stance",
    "tone",
    "length",
    "ask",
    "vagueness",
    "phrasing",
    "pressure",
    "user",
    "texture",
    "history",
)

#: Row keys that carry a verdict. ``Judgment`` owns them; direct writes
#: outside ``attach`` are frozen at today's count by a test.
VERDICT_KEYS = (
    "reward",
    "reason",
    "grader_reason",
    "label_source",
    "judge_name",
    "judge_status",
    "judge_meta",
    "failure_class",
    "llm_reward",
    "llm_reason",
    "qwen_reward",
)

#: Rollout-level keys the objects know about and carry by name.
_CARRY_ROLLOUT = (
    "messages",
    "opener",
    "opening",
    "conversation_id",
    "ts",
    "fault_detected",
    "lineage",
    "quality",
    "quality_reason",
    "quality_scores",
    "behavior_signature",
    "steering",
    "tools",
    "group_id",
    "logprob",
    "n_tokens",
    "token_logprobs",
    "sampling",
    "usage",
    "writer_model",
    "user_model",
)

#: Every key ``from_row`` consumes into a typed field. Anything else on the
#: row is unknown to the objects and passes through ``Rollout.extra``.
_CONSUMED = frozenset(
    (
        SCHEMA_KEY,
        "prompt",
        "steps",
        "final_text",
        "scenario_id",
        "spec_id",
        "world_state",
        "faults",
        "seed",
        "rollout_index",
        "model_version",
        "policy_version",
        "markers",
        "calibration",
        "tool_trace",
        "trace",
        *AXES,
        *VERDICT_KEYS,
        *_CARRY_ROLLOUT,
    )
)

#: The published Hugging Face set flattens list columns to JSON strings.
_JSON_STRING_KEYS = ("steps_json", "messages_json", "metadata_json")

ShapeName = Literal["v1", "engine", "training", "platform_pull", "otel", "hf_flat", "loose"]


# ------------------------------------------------------------------ objects


@dataclass(frozen=True)
class Message:
    role: str
    content: str = ""
    name: str | None = None
    tool_calls: tuple[dict, ...] | None = None


@dataclass(frozen=True)
class Step:
    """One trajectory step: a tool call, an agent turn, or a user turn."""

    tool: str | None = None
    arguments: Any = None
    result: Any = None
    text: str | None = None
    user: str | None = None
    #: sampling facts of the agent turn this step opened, when captured
    logprob: float | None = None
    n_tokens: int | None = None
    truncated: bool | None = None
    #: what the model call behind this step cost, when the server reported it
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class FaultEvent:
    """What the world did to a tool call: injected fault or observed one."""

    tool: str
    mode: str
    rate: float | None = None
    injected: bool = True


@dataclass(frozen=True)
class World:
    seed: int | None = None
    state: str | None = None
    faults: dict = field(default_factory=dict)
    exemplars: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Privileged:
    """Context the teacher sees and the student never does."""

    principle: str | None = None
    hidden_state: dict = field(default_factory=dict)
    reference: str | None = None
    #: a ``score.rubric.Rubric`` as a dict: the per-prompt criteria a
    #: rubric judge scores (Lambert 2025, chapter Synthetic Data and
    #: Distillation); never exported
    rubric: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Lineage:
    parent_task_id: str | None = None
    transform: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class PolicyRef:
    name: str = ""
    model: str | None = None
    prompt_hash: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class ScorerRef:
    name: str
    kind: Literal["rule", "judge", "reward_model", "human"] = "rule"
    version: str | None = None


@dataclass(frozen=True)
class Task:
    """The situation. Identity-bearing, so nothing computed lives here:
    splits belong to a ``Dataset``, difficulty to a ``Calibration``."""

    task_id: str
    spec_id: str = ""
    prompt: str = ""
    prefix: tuple[Message, ...] = ()
    world: World = field(default_factory=World)
    privileged: Privileged = field(default_factory=Privileged)
    axes: dict[str, Any] = field(default_factory=dict)
    behaviors: tuple[str, ...] = ()
    lineage: Lineage = field(default_factory=Lineage)


@dataclass
class Rollout:
    """One episode. Mutable so grading paths can attach to it in place;
    the no-verdict invariant is enforced at the boundary, not here."""

    rollout_id: str
    task_id: str
    policy: PolicyRef = field(default_factory=PolicyRef)
    index: int = 0
    steps: list[Step] = field(default_factory=list)
    final_text: str = ""
    ledger: list[FaultEvent] = field(default_factory=list)
    usable: bool = True
    unusable_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Judgment:
    rollout_id: str
    scorer: ScorerRef
    reward: float | None
    status: Literal["ok", "missing_reward", "invalid_result", "error", "timeout"] = "ok"
    reason: str = ""
    failure_class: str | None = None
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Marker:
    rollout_id: str
    name: str
    value: float
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    """Split membership is a dataset decision, so one task can be holdout
    in one dataset and training in another."""

    dataset_id: str
    spec_id: str = ""
    splits: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Calibration:
    """Measured difficulty of one task for one student. Optional; only
    ``calibrate`` produces it. ``pass_rate_ci95`` is the Wilson 95%
    interval on ``pass_rate`` from ``n`` rollouts (about +/-0.3 wide at
    n=8), so a band assignment can be read with its uncertainty. ``mean_kl``
    is the sampled KL to a reference policy per generated token;
    ``simulate(logprobs=True)`` captures the student side and
    ``calibrate(rows, ref=...)`` fills it in."""

    task_id: str
    student: PolicyRef
    n: int
    pass_rate: float
    mean_kl: float | None = None
    pass_rate_ci95: tuple[float, float] | None = None


# ------------------------------------------------------------------ stamp


def stamp(row: dict) -> dict:
    """Set ``schema_version`` on a row born in canonical shape. In place."""
    if isinstance(row, dict) and SCHEMA_KEY not in row:
        row[SCHEMA_KEY] = SCHEMA_VERSION
    return row


def version_of(row: dict) -> str:
    """``"0"`` for an unstamped row, else the stamp as written."""
    value = row.get(SCHEMA_KEY) if isinstance(row, dict) else None
    return "0" if value is None else str(value)


def detect_shape(row: dict) -> ShapeName:
    """Which shape an unstamped row is in. Stamped rows are ``v1``."""
    if not isinstance(row, dict):
        return "loose"
    if SCHEMA_KEY in row:
        return "v1"
    if any(isinstance(row.get(k), str) for k in _JSON_STRING_KEYS):
        return "hf_flat"
    if (
        isinstance(row.get("messages"), list)
        and row.get("messages")
        and not row.get("steps")
        and "final_text" not in row
    ):
        return "training"
    if row.get("scenario_id") is not None:
        return "engine"
    if isinstance(row.get("tool_trace"), list) and not row.get("steps"):
        return "platform_pull"
    if row.get("conversation_id") is not None:
        return "otel"
    return "loose"


# ------------------------------------------------------------------ validate


def validate(
    row: Any,
    kind: Literal["row", "training", "preference"] = "row",
) -> list[str]:
    """Problems with one row, empty when it is fine. Never raises.

    Stamped rows must carry the required fields with the right types.
    Unstamped ``kind="row"`` rows are version 0 and only have to be
    non-empty dicts: nothing that works today is rejected there.

    Two things are reported regardless of version, because "0 validation
    failures" is read as a guarantee and neither case is one:

    * an empty dict, which carries no prompt, no messages, no verdict;
    * a ``training`` or ``preference`` row with no ``messages`` or no
      ``chosen``/``rejected``. Those two kinds ask "is this a training
      sample", and an unstamped dict with no conversation in it is not
      one whatever version it claims.
    """
    if not isinstance(row, dict):
        return ["not_a_dict"]
    if not row:
        return ["empty_row"]
    version = version_of(row)
    if version not in KNOWN_VERSIONS:
        return [f"unknown_schema_version:{version}"]
    if version == "0" and kind == "row":
        return []
    problems: list[str] = []
    if kind == "row":
        if not isinstance(row.get("prompt", ""), str):
            problems.append("prompt_not_str")
        if not isinstance(row.get("steps", []), list):
            problems.append("steps_not_list")
        if not isinstance(row.get("final_text", ""), str):
            problems.append("final_text_not_str")
        reward = row.get("reward")
        if reward is not None and isinstance(reward, bool):
            problems.append("reward_is_bool")
        elif reward is not None and not isinstance(reward, (int, float)):
            problems.append("reward_not_number")
        if row.get("calibration") is not None and calibration_of(row) is None:
            problems.append("calibration_invalid")
    elif kind == "training":
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            problems.append("messages_missing")
        elif any(not isinstance(m, dict) or "role" not in m for m in messages):
            problems.append("message_without_role")
        if row.get("tools") is not None and not isinstance(row["tools"], list):
            problems.append("tools_not_list")
        mask = row.get("loss_mask")
        if mask is not None and (
            not isinstance(mask, list)
            or not isinstance(messages, list)
            or len(mask) != len(messages)
            or any(isinstance(m, bool) or m not in (0, 1) for m in mask)
        ):
            problems.append("loss_mask_invalid")
    elif kind == "preference":
        for side in ("chosen", "rejected"):
            if not isinstance(row.get(side), list) or not row[side]:
                problems.append(f"{side}_missing")
    return problems


def check(
    rows: Sequence[Any] | Any,
    kind: Literal["row", "training", "preference"] = "row",
    *,
    where: str = "rows",
) -> None:
    """Raise ``ValueError`` naming the first bad rows. Accepts one row too."""
    items = rows if isinstance(rows, (list, tuple)) else [rows]
    bad: list[str] = []
    for i, row in enumerate(items):
        problems = validate(row, kind)
        if problems:
            bad.append(f"{i}:{','.join(problems)}")
            if len(bad) >= MESSAGE_EXAMPLES:
                break
    if bad:
        raise ValueError(f"schema_invalid in {where}: {'; '.join(bad)}")


# ------------------------------------------------------------------ coerce


def _short_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def _number(value: Any) -> float | int | None:
    """A reward-like value as a number, or None. Bools count as 0/1."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ from_row


def _normalize_legacy(row: dict) -> dict:
    """Version-0 row in canonical spelling, via the trace loader's tables."""
    from .ingest.traces import load_traces

    shape = detect_shape(row)
    if shape in ("v1", "engine", "otel"):
        return dict(row)
    if shape == "hf_flat":
        out = dict(row)
        for key in _JSON_STRING_KEYS:
            text = out.pop(key, None)
            if not isinstance(text, str):
                continue
            try:
                value = json.loads(text)
            except ValueError:
                out[key] = text  # unparseable: keep it as it was
                continue
            target = key[:-5]  # steps_json -> steps
            if target == "metadata":
                if isinstance(value, dict):
                    out.setdefault("metadata", value)
            elif isinstance(value, list) and not out.get(target):
                out[target] = value
        return out
    normalized = load_traces([row])
    return normalized[0] if normalized else dict(row)


def _steps(raw: Any) -> list[Step]:
    out: list[Step] = []
    if not isinstance(raw, list):
        return out
    for step in raw:
        if not isinstance(step, dict):
            continue
        out.append(
            Step(
                tool=step.get("tool"),
                arguments=step.get("arguments"),
                result=step.get("result"),
                text=step.get("text"),
                user=step.get("user"),
                logprob=_number(step.get("logprob")),
                n_tokens=_int(step.get("n_tokens")) or None,
                truncated=True if step.get("truncated") else None,
                input_tokens=_int(step.get("input_tokens"))
                if step.get("input_tokens") is not None
                else None,
                output_tokens=_int(step.get("output_tokens"))
                if step.get("output_tokens") is not None
                else None,
            )
        )
    return out


def _ledger(faults: Any) -> list[FaultEvent]:
    out: list[FaultEvent] = []
    if not isinstance(faults, dict):
        return out
    for tool, plan in faults.items():
        if isinstance(plan, dict) and plan.get("mode"):
            rate = _number(plan.get("rate"))
            out.append(
                FaultEvent(
                    tool=str(tool),
                    mode=str(plan["mode"]),
                    rate=float(rate) if rate is not None else None,
                )
            )
    return out


def attach(row: dict, judgment: Judgment) -> dict:
    """Write a primary ``Judgment`` onto a row, in place. The one sanctioned
    verdict write outside ``to_row``: grading paths build a ``Judgment`` and
    hand it here instead of poking ``reward`` and friends directly.

    ``reward`` lands as an int when it is exactly 0 or 1. ``reason`` and
    ``failure_class`` are removed when empty so a stale value from an
    earlier pass never survives a regrade. ``judge_meta`` is the scorer's
    evidence plus its ``version`` when the scorer has one, so the row says
    which judge, prompt, and settings produced the label.
    """
    reward = judgment.reward
    if reward in (0, 1, 0.0, 1.0) and reward is not None:
        row["reward"] = int(reward)
    else:
        row["reward"] = reward
    if judgment.reason:
        row["reason"] = judgment.reason
    else:
        row.pop("reason", None)
    if judgment.failure_class:
        row["failure_class"] = judgment.failure_class
    else:
        row.pop("failure_class", None)
    row["judge_name"] = judgment.scorer.name
    row["judge_status"] = judgment.status
    meta = dict(judgment.evidence)
    if judgment.scorer.version:
        meta["version"] = judgment.scorer.version
    if judgment.scorer.kind != "judge":
        # A named scorer reads back as "judge" unless the row says otherwise.
        meta["scorer_kind"] = judgment.scorer.kind
    if meta:
        row["judge_meta"] = meta
    return row


def _scorer_version(row: dict) -> str | None:
    meta = row.get("judge_meta")
    if isinstance(meta, dict) and meta.get("version"):
        return str(meta["version"])
    lineage = row.get("lineage")
    if isinstance(lineage, dict) and lineage.get("judge_version"):
        return str(lineage["judge_version"])
    return None


def _scorer_kind(row: dict) -> str | None:
    """The kind the grading run stamped, if any (``judge_meta.scorer_kind``)."""
    meta = row.get("judge_meta")
    if isinstance(meta, dict) and meta.get("scorer_kind") in (
        "rule",
        "judge",
        "reward_model",
        "human",
    ):
        return str(meta["scorer_kind"])
    return None


def _judgments(row: dict, rollout_id: str) -> list[Judgment]:
    out: list[Judgment] = []
    has_primary = (
        "reward" in row
        or row.get("judge_status")
        or row.get("judge_name")
        or row.get("label_source")
    )
    if has_primary:
        judge = row.get("judge_name")
        label = row.get("label_source")
        name = judge or label or "unlabeled"
        # The stamped kind wins; without one, a named judge is a model judge
        # and a bare label is a rule. A Verifier run through ``run_judge``
        # carries ``judge_name`` too, so the inference alone called every
        # verifier a judge (#250).
        kind: Any = _scorer_kind(row) or ("judge" if judge else "rule")
        reward = _number(row.get("reward"))
        status: Any = row.get("judge_status") or "ok"
        evidence: dict = {}
        if reward is None and row.get("reward") is not None:
            status = "invalid_result"
            evidence["raw_reward"] = row.get("reward")
        if row.get("judge_meta"):
            evidence["judge_meta"] = row["judge_meta"]
        if judge and label:
            evidence["label_source"] = label
        if row.get("grader_reason") and row.get("grader_reason") != row.get("reason"):
            evidence["grader_reason"] = row["grader_reason"]
        if "reward" in row:
            evidence["reward_present"] = True
        out.append(
            Judgment(
                rollout_id=rollout_id,
                scorer=ScorerRef(name=str(name), kind=kind, version=_scorer_version(row)),
                reward=reward,
                status=status,
                reason=str(row.get("reason") or ""),
                failure_class=row.get("failure_class"),
                evidence=evidence,
            )
        )
    if "llm_reward" in row:
        out.append(
            Judgment(
                rollout_id=rollout_id,
                scorer=ScorerRef(name="llm", kind="judge"),
                reward=_number(row.get("llm_reward")),
                reason=str(row.get("llm_reason") or ""),
            )
        )
    if row.get("qwen_reward") is not None:
        out.append(
            Judgment(
                rollout_id=rollout_id,
                scorer=ScorerRef(name="qwen", kind="judge"),
                reward=_number(row.get("qwen_reward")),
            )
        )
    return out


def from_row(row: dict) -> tuple[Task, Rollout, list[Judgment], list[Marker]]:
    """Split one flat row into its four objects. Any version, any shape."""
    if not isinstance(row, dict):
        raise TypeError("from_row expects a dict")
    raw = _normalize_legacy(row)
    prompt = str(raw.get("prompt") or "")
    task_id = str(raw.get("scenario_id") or "") or "task_" + _short_hash(prompt)
    axes = {k: raw[k] for k in AXES if k in raw and raw[k] is not None}
    faults = raw.get("faults") if isinstance(raw.get("faults"), dict) else {}
    # What the judge saw and the policy did not: a constitution principle,
    # a hidden world state, a reference answer. Written by character and
    # rubric pipelines; the engine leaves it empty.
    priv_raw = raw.get("privileged")
    priv: dict = priv_raw if isinstance(priv_raw, dict) else {}
    task = Task(
        task_id=task_id,
        spec_id=str(raw.get("spec_id") or ""),
        prompt=prompt,
        world=World(seed=raw.get("seed"), state=raw.get("world_state"), faults=dict(faults or {})),
        privileged=Privileged(
            principle=priv.get("principle"),
            hidden_state=dict(priv.get("hidden_state") or {}),
            reference=priv.get("reference"),
            rubric=dict(priv.get("rubric") or {}),
        ),
        axes=axes,
    )
    index = _int(raw.get("rollout_index"))
    model = raw.get("model_version")
    rollout_id = str(raw.get("rollout_id") or "") or _short_hash(task_id, index, model or "")
    extra = {k: raw[k] for k in _CARRY_ROLLOUT if k in raw and raw[k] is not None}
    if raw.get("rollout_index") is not None:
        extra["rollout_index_present"] = True
    passthrough = {k: v for k, v in raw.items() if k not in _CONSUMED}
    if passthrough:
        extra["passthrough"] = passthrough
    rollout = Rollout(
        rollout_id=rollout_id,
        task_id=task_id,
        policy=PolicyRef(
            name=str(model or ""),
            model=model,
            version=str(raw["policy_version"]) if raw.get("policy_version") else None,
        ),
        index=index,
        steps=_steps(raw.get("steps")),
        final_text=str(raw.get("final_text") or ""),
        ledger=_ledger(faults),
        extra=extra,
    )
    markers: list[Marker] = []
    for k, v in (raw.get("markers") or {}).items() if isinstance(raw.get("markers"), dict) else ():
        value = _number(v)
        if value is not None:
            markers.append(Marker(rollout_id=rollout_id, name=str(k), value=float(value)))
    calibration = calibration_of(raw)
    if calibration is not None:
        rollout.extra["calibration"] = calibration
    return task, rollout, _judgments(raw, rollout_id), markers


def calibration_of(row: dict) -> Calibration | None:
    """The typed ``Calibration`` a row carries, or ``None`` when absent or
    malformed. ``publish_gate`` / ``calibrate`` write it as a flat dict
    under ``calibration``; this is the read side. ``mean_kl`` is optional."""
    raw = row.get("calibration") if isinstance(row, dict) else None
    if not isinstance(raw, dict):
        return None
    pass_rate = _number(raw.get("pass_rate"))
    n = _int(raw.get("n"))
    if pass_rate is None or not 0.0 <= float(pass_rate) <= 1.0 or n is None or n < 1:
        return None
    student = raw.get("student")
    if isinstance(student, PolicyRef):
        ref = student
    elif isinstance(student, dict):
        ref = PolicyRef(
            name=str(student.get("name") or ""),
            model=student.get("model"),
            prompt_hash=student.get("prompt_hash"),
            version=student.get("version"),
        )
    else:
        ref = PolicyRef()
    mean_kl = _number(raw.get("mean_kl"))
    ci = raw.get("pass_rate_ci95")
    ci95: tuple[float, float] | None = None
    if isinstance(ci, (list, tuple)) and len(ci) == 2:  # noqa: PLR2004  # an interval is a pair
        lo, hi = _number(ci[0]), _number(ci[1])
        if lo is not None and hi is not None:
            ci95 = (float(lo), float(hi))
    return Calibration(
        task_id=str(raw.get("task_id") or row.get("scenario_id") or row.get("prompt") or ""),
        student=ref,
        n=int(n),
        pass_rate=float(pass_rate),
        mean_kl=float(mean_kl) if mean_kl is not None else None,
        pass_rate_ci95=ci95,
    )


# ------------------------------------------------------------------ to_row


def _step_dict(step: Step) -> dict:
    if step.user is not None:
        return {"user": step.user}
    out: dict[str, Any] = {}
    if step.tool is not None:
        out["tool"] = step.tool
        out["arguments"] = step.arguments
        out["result"] = step.result
    if step.text is not None:
        out["text"] = step.text
    if step.logprob is not None:
        out["logprob"] = step.logprob
    if step.n_tokens is not None:
        out["n_tokens"] = step.n_tokens
    if step.truncated:
        out["truncated"] = True
    if step.input_tokens is not None:
        out["input_tokens"] = step.input_tokens
    if step.output_tokens is not None:
        out["output_tokens"] = step.output_tokens
    return out


def to_row(
    task: Task, rollout: Rollout, judgments: Sequence[Judgment] = (), markers: Sequence[Marker] = ()
) -> dict:
    """The flat v1 wire row. Inverse of ``from_row`` on engine rows; on
    other shapes it is the canonical row ``load_traces`` would produce,
    with the source row's unknown keys carried along."""
    from .data import clean_faults, conversation

    row: dict[str, Any] = {
        "prompt": task.prompt,
        "steps": [_step_dict(s) for s in rollout.steps],
        "final_text": rollout.final_text,
        "scenario_id": task.task_id,
    }
    row["messages"] = rollout.extra.get("messages") or conversation(row)
    for key in ("opener", "opening", "sampling", "token_logprobs"):
        if rollout.extra.get(key):
            row[key] = rollout.extra[key]
    for key in AXES:
        if key in task.axes:
            row[key] = task.axes[key]
    if task.world.state and task.world.state not in {"unspecified", "unknown"}:
        row["world_state"] = task.world.state
    faults = clean_faults(task.world.faults)
    if faults:
        row["faults"] = faults
    if rollout.extra.get("fault_detected"):
        row["fault_detected"] = True
    if rollout.index or rollout.extra.get("rollout_index_present"):
        row["rollout_index"] = rollout.index
    if rollout.policy.model is not None:
        row["model_version"] = rollout.policy.model
    if rollout.policy.version:
        row["policy_version"] = rollout.policy.version
    primary = next((j for j in judgments if j.scorer.name not in {"llm", "qwen"}), None)
    if primary is not None:
        if primary.reward is not None or primary.evidence.get("reward_present"):
            row["reward"] = primary.reward
        if primary.reason:
            row["reason"] = primary.reason
        if primary.scorer.name != "unlabeled":
            # ``judge_name`` is what a grading run called itself, whatever
            # its kind; a stamped kind proves a run named it. ``label_source``
            # is the engine's own rule label.
            stamped = (primary.evidence.get("judge_meta") or {}).get("scorer_kind")
            if primary.scorer.kind != "rule" or stamped:
                row["judge_name"] = primary.scorer.name
                if primary.evidence.get("label_source"):
                    row["label_source"] = primary.evidence["label_source"]
            else:
                row["label_source"] = primary.scorer.name
        if primary.status != "ok":
            row["judge_status"] = primary.status
        if primary.evidence.get("judge_meta"):
            row["judge_meta"] = primary.evidence["judge_meta"]
        if primary.failure_class:
            row["failure_class"] = primary.failure_class
    for j in judgments:
        if j.scorer.name == "llm":
            row["llm_reward"] = j.reward
            row["llm_reason"] = j.reason
        elif j.scorer.name == "qwen" and j.reward is not None:
            row["qwen_reward"] = j.reward
    if markers:
        row["markers"] = {m.name: m.value for m in markers}
    for key in (
        "conversation_id",
        "ts",
        "lineage",
        "quality",
        "quality_reason",
        "quality_scores",
        "tools",
        "group_id",
        "logprob",
        "n_tokens",
        "usage",
        "writer_model",
        "user_model",
    ):
        if rollout.extra.get(key) is not None:
            row[key] = rollout.extra[key]
    if task.spec_id:
        row["spec_id"] = task.spec_id
    # Task.privileged is deliberately not projected here: it is the
    # teacher's context, and every exporter reads the row, so writing it
    # would put it one step from a training file (test_privileged_leakage).
    # A source row's own ``privileged`` block rides back out as passthrough.
    calibration = rollout.extra.get("calibration")
    if isinstance(calibration, Calibration):
        row["calibration"] = asdict(calibration)
    elif isinstance(calibration, dict):
        row["calibration"] = dict(calibration)
    for key, value in (rollout.extra.get("passthrough") or {}).items():
        row.setdefault(key, value)
    return stamp(row)


# ------------------------------------------------------------------ rows

#: What a reward is called on a row built from a precomputed score.
GIVEN_SCORER = "given"
#: The two callable shapes ``rows(reward=)`` wraps, by positional arity:
#: ``fn(prompt, completion)`` and ``fn(prompt, completion, reference)``.
#: One argument is the judge contract ``fn(row)`` and is passed as is.
_ARITY_PROMPT_COMPLETION = 2
_ARITY_WITH_REFERENCE = 3


def _prompt_text(prompt: Any) -> str:
    """The text a verifier and ``decontaminate`` read: the string itself,
    or the last user turn of a message list (else its last turn)."""
    if isinstance(prompt, str):
        return prompt
    turns = [m for m in prompt if isinstance(m, dict)]
    for message in reversed(turns):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return str(turns[-1].get("content") or "") if turns else ""


def _per_prompt(name: str, values: Any, n: int) -> list[Any]:
    """``values`` as one entry per prompt, or a ``ValueError`` naming the kwarg."""
    if values is None:
        return [None] * n
    out = list(values)
    if len(out) != n:
        raise ValueError(f"{name}= must have one entry per prompt: {len(out)} for {n} prompts")
    return out


def _per_completion(name: str, values: Any, shape: list[int]) -> list[list[Any]]:
    """``values`` nested like the completions: one entry per completion,
    given either in that nesting or flat over every completion."""
    if values is None:
        return [[None] * k for k in shape]
    try:
        given = list(values)
    except TypeError:
        raise TypeError(
            f"{name}= is one entry per completion, nested like completions= or flat; "
            f"got {type(values).__name__}"
        ) from None
    nested: list[list[Any]]
    if given and len(given) == len(shape) and all(isinstance(v, (list, tuple)) for v in given):
        nested = [list(v) for v in given]
    elif len(given) == sum(shape):
        nested, at = [], 0
        for k in shape:
            nested.append(given[at : at + k])
            at += k
    else:
        raise ValueError(
            f"{name}= must have one entry per completion, nested like completions= "
            f"or flat: got {len(given)} for {sum(shape)} completions over {len(shape)} prompts"
        )
    for i, (block, k) in enumerate(zip(nested, shape)):
        if len(block) != k:
            raise ValueError(
                f"{name}[{i}] has {len(block)} entries for {k} completions of prompt {i}"
            )
    return nested


def _reward_value(name: str, value: Any) -> float | int:
    """A precomputed reward as a number in [0, 1]; 0 and 1 land as ints so
    ``pass_at`` and ``select`` read them as the binary outcome they are.
    A bool is a verdict, not a label, and is read as 0 or 1."""
    number = _number(value)
    if number is None or not 0.0 <= float(number) <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1] (1 is a pass); got {value!r}")
    return int(number) if float(number) in (0.0, 1.0) else float(number)


def _as_judge(reward: Any) -> Any:
    """The callable ``run_judge`` takes, from what ``rows()`` was given: a
    ``Verifier`` as is, a judge-contract callable ``(row) -> verdict`` as
    is, and ``fn(prompt, completion)`` or ``fn(prompt, completion,
    reference)`` wrapped as a ``FunctionVerifier`` so the row records what
    scored it."""
    import inspect

    from .verify.base import FunctionVerifier, Verifier

    if isinstance(reward, Verifier):
        return reward
    try:
        params = [
            p
            for p in inspect.signature(reward).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):  # a builtin or a C callable: the judge contract
        return reward
    arity = len(params)
    name = getattr(reward, "__name__", None) or type(reward).__name__
    if arity == 1:
        return reward
    if arity == _ARITY_PROMPT_COMPLETION:
        return FunctionVerifier(lambda c, _r, row: reward(row["prompt"], c), name=name)
    if arity == _ARITY_WITH_REFERENCE:
        return FunctionVerifier(lambda c, r, row: reward(row["prompt"], c, r), name=name)
    raise TypeError(
        f"reward= callable {name} takes {arity} positional arguments; rows() calls it as "
        "(prompt, completion), (prompt, completion, reference) or (row)"
    )


def rows(
    prompts: Sequence[str | Sequence[dict]],
    completions: Sequence[str | Sequence[str]],
    reward: Any = None,
    *,
    references: Sequence[Any] | None = None,
    task_ids: Sequence[str] | None = None,
    markers: Sequence[Any] | None = None,
) -> RowList:
    """Rows from your own prompts and completions, in the shape every measurement reads.

    The front door for a public benchmark: GSM8K questions and a model's
    answers become the same rows ``simulate()`` emits, so ``pass_at``,
    ``compare``, ``eval_variance``, ``holdout_size``, ``decontaminate``
    and ``select`` take them unchanged. Every row carries the five keys the
    measurement calls read, and nothing else is required:

    * ``task_id``: what the rollouts of one prompt group under; every
      interval is over tasks, never rows (Miller 2024, arXiv:2411.00640).
      Defaults to a stable hash of the prompt text, so the same prompt gets
      the same id on every call. The engine's ``scenario_id`` carries the
      same value.
    * ``prompt``: the prompt text, or the last user turn of a message list.
    * ``final_text``: one completion.
    * ``reward``: a number in [0, 1]; 0 and 1 are the binary outcome
      ``pass_at`` counts and ``select`` bands on, anything between is a
      partial score those two skip.
    * ``markers``: ``{name: number}``, one behavior measurement per
      completion. Input as well as output: ``compare(proxy="marker:name")``
      and ``eval_variance`` read them wherever they came from.

    ``prompts`` is a sequence of strings or message lists. ``completions``
    is one string per prompt, or one sequence per prompt (k completions of
    the same prompt: what ``pass_at``'s k-way numbers and ``select(mode="rl")``
    need). ``reward`` is a ``Verifier`` (``wai.verify.MathEqual()``), a
    callable ``(prompt, completion)`` or ``(prompt, completion, reference)``
    returning a number in [0, 1], a judge-contract callable ``(row) ->
    verdict``, or the precomputed numbers themselves, nested like
    ``completions`` or flat; without it the rows carry no reward and only
    ``decontaminate`` has a use for them. ``references`` is the gold per
    prompt, kept under ``privileged.reference`` where a verifier reads it
    and no training export projects it. ``task_ids`` names the tasks;
    ``markers`` is one dict per completion, nested like ``completions``.

    Returns a ``RowList``: a list of typed rows (``schema_version`` 1,
    ``to_row`` shape) that also feeds ``select(...).export()``. A verifier
    or callable is run through ``run_judge``, so the rows say what scored
    them (``judge_name``, ``lineage``) exactly as ``data.grade()`` writes.

    Reference: Lambert 2025 (rlhfbook), chapters Evaluation and Reasoning;
    Miller 2024, arXiv:2411.00640, for the task-level intervals.

    ```python
    import whileai as wai

    questions = ["What is 2 + 3?", "What is 7 * 6?", "What is 10 - 4?", "What is 9 / 3?"]
    gold = ["5", "42", "6", "3"]
    before = [["5", "4", "5", "5"], ["41", "41", "42", "40"], ["6"] * 4, ["3", "2", "3", "3"]]
    after = [["5"] * 4, ["42", "42", "42", "41"], ["6"] * 4, ["3"] * 4]
    base = wai.rows(questions, before, wai.verify.MathEqual(), references=gold)
    tuned = wai.rows(questions, after, wai.verify.MathEqual(), references=gold)
    print(wai.pass_at(base))                         # pass@1 with its interval
    print(wai.select(base, mode="rl", band=(0.2, 0.8)))  # drops the unanimous groups
    print(wai.compare(base, tuned))                  # is the change real
    ```
    """
    from .data import RowList

    prompt_list = list(prompts)
    completion_list = list(completions)
    n = len(prompt_list)
    if len(completion_list) != n:
        raise ValueError(
            f"completions= must have one entry per prompt (a string, or a sequence of k "
            f"strings): {len(completion_list)} for {n} prompts"
        )
    groups: list[list[str]] = [
        [c] if isinstance(c, str) else [str(x) for x in c] for c in completion_list
    ]
    shape = [len(g) for g in groups]
    refs = _per_prompt("references", references, n)
    ids = _per_prompt("task_ids", task_ids, n)
    marks = _per_completion("markers", markers, shape)
    given: list[list[Any]] | None = None
    judge = None
    if callable(reward):
        judge = _as_judge(reward)
    elif reward is not None:
        given = _per_completion("reward", reward, shape)

    out: list[dict] = []
    for i, prompt in enumerate(prompt_list):
        text = _prompt_text(prompt)
        task_id = str(ids[i]) if ids[i] is not None else "task_" + _short_hash(text)
        task = Task(task_id=task_id, prompt=text)
        for j, completion in enumerate(groups[i]):
            extra: dict[str, Any] = {"rollout_index_present": True}
            if not isinstance(prompt, str):
                extra["messages"] = [
                    *(dict(m) for m in prompt if isinstance(m, dict)),
                    {"role": "assistant", "content": completion},
                ]
            rollout = Rollout(
                rollout_id=_short_hash(task_id, j),
                task_id=task_id,
                index=j,
                final_text=completion,
                extra=extra,
            )
            judgments: list[Judgment] = []
            if given is not None:
                judgments.append(
                    Judgment(
                        rollout_id=rollout.rollout_id,
                        scorer=ScorerRef(name=GIVEN_SCORER, kind="rule"),
                        reward=_reward_value(f"reward[{i}][{j}]", given[i][j]),
                    )
                )
            marker_objs: list[Marker] = []
            for name, value in (marks[i][j] or {}).items():
                number = _number(value)
                if number is None:
                    raise ValueError(f"markers[{i}][{j}][{name!r}] must be a number; got {value!r}")
                marker_objs.append(
                    Marker(rollout_id=rollout.rollout_id, name=str(name), value=float(number))
                )
            row = to_row(task, rollout, judgments, marker_objs)
            row["task_id"] = task_id
            if refs[i] is not None:
                row["privileged"] = {"reference": refs[i]}
            out.append(row)
    if judge is None:
        return RowList(out)
    from .score.judging import run_judge

    return run_judge(out, judge, source="grade").rows


def as_dict(obj: Any) -> dict:
    """Plain dict of any schema object, for JSON."""
    return asdict(obj)


def load_json_schema() -> dict:
    """The packaged ``schemas/row-v1.json``."""
    from importlib import resources

    text = (
        resources.files(__package__ or "whileai.simulations") / "schemas" / "row-v1.json"
    ).read_text()
    return json.loads(text)


__all__ = [
    "AXES",
    "KNOWN_VERSIONS",
    "SCHEMA_KEY",
    "SCHEMA_VERSION",
    "VERDICT_KEYS",
    "Calibration",
    "Dataset",
    "FaultEvent",
    "Judgment",
    "Lineage",
    "Marker",
    "Message",
    "PolicyRef",
    "Privileged",
    "Rollout",
    "ScorerRef",
    "Step",
    "Task",
    "World",
    "as_dict",
    "attach",
    "calibration_of",
    "check",
    "detect_shape",
    "from_row",
    "load_json_schema",
    "rows",
    "stamp",
    "to_row",
    "validate",
    "version_of",
]
