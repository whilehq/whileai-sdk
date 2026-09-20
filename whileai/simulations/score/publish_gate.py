"""The gate a dataset passes before it leaves for the platform.

Two jobs, both from the RLVR playbook (Yu et al. 2025 (DAPO),
arXiv:2503.14476: dynamic sampling drops all-pass / all-fail groups; Lambert
2025, chapter Reasoning: offline difficulty filtering keeps prompts the start
policy solves 20-80% of the time, measured with N samples; curricula need that
per-prompt difficulty stored with the data).

* ``calibrate`` writes the measured difficulty on every graded row: the
  per-task pass rate over its k rollouts, the sample count, and the
  policy that produced them. That is the schema's ``Calibration`` record,
  flattened onto the row as ``calibration``. ``carry_calibration`` is the
  same stamp written from rows that have since been pruned, which is what
  ``optimize(mode="rl")`` uses so the number survives its own hygiene.
* ``publish_gate`` refuses an RL-shaped dataset that could not train
  anything: ungraded rows, or no mixed group anywhere. It reports what a
  grouped update would see (pass@1, headroom, band counts) and warns
  when out-of-band groups are still present, because ``push`` uploads
  rows as they are and the pruning lives in ``optimize``.

Explore-shaped runs (one rollout per ask) pass through with a calibration
stamp and a report; they are not RL data and are not judged as such.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

from ..schema import Calibration, PolicyRef, calibration_of
from .hack_scan import hack_scan
from .hygiene import (
    dedupe_groups,
    hygiene_warnings,
    length_report,
    near_duplicate_prompts,
    reward_correlations,
)
from .optimize import DEFAULT_BAND, _binary_label, _group_label_lists, group_signal
from .passat import pass_at
from .stats import task_key, wilson_interval


class PublishGateError(ValueError):
    """The dataset must not be published as it stands. The message says why."""


def policy_ref(policy: PolicyRef | dict | str | None, *, model: str | None = None) -> PolicyRef:
    """Coerce whatever the caller has into a ``PolicyRef``. A bare string
    is the system prompt and gets hashed; a dict is field-by-field."""
    if isinstance(policy, PolicyRef):
        return policy
    if isinstance(policy, dict):
        return PolicyRef(
            name=str(policy.get("name") or ""),
            model=policy.get("model") or model,
            prompt_hash=policy.get("prompt_hash"),
            version=policy.get("version"),
        )
    prompt_hash = None
    if isinstance(policy, str) and policy:
        prompt_hash = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:16]
    return PolicyRef(name="", model=model, prompt_hash=prompt_hash)


def _task_id(row: dict) -> str:
    """The task's id, not its text: the same ``task_key`` every report
    groups by (``scenario_id``, else ``task_id``, else the prompt)."""
    return task_key(row)


def _identified(ref: PolicyRef) -> bool:
    """Whether a ``PolicyRef`` names anything at all."""
    return bool(ref.name or ref.model or ref.prompt_hash or ref.version)


def _student_for(row: dict, student: PolicyRef) -> PolicyRef:
    """The caller's ``PolicyRef`` when it names anything, else the row's
    own ``policy_version`` (the engine stamps one on every rollout), so a
    stamp written without ``policy=`` still says which policy it measured."""
    if _identified(student):
        return student
    version = row.get("policy_version") or row.get("model_version")
    return PolicyRef(version=str(version)) if version else student


def _ci95(labels: Sequence[int]) -> tuple[float, float] | None:
    """Wilson 95% interval on a task's pass rate, rounded for the row."""
    ci = wilson_interval(int(sum(labels)), len(labels))
    return (round(ci[0], 4), round(ci[1], 4)) if ci else None


def carry_calibration(
    graded: Sequence[dict],
    kept: Sequence[dict],
    *,
    policy: PolicyRef | dict | str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Stamp ``kept`` with the difficulty measured on ``graded``, in place.

    Hygiene shrinks a group without changing how often the policy passed
    that ask: the measurement is the k rollouts the grader saw, and
    dropping five identical trajectories does not make the task harder.
    Stamping the survivors from the survivors would put the post-dedup k
    and its pass rate under a field that means the policy's pass rate over
    k repeats, so ``optimize(mode="rl")`` carries the pre-hygiene numbers
    onto the rows that leave and ``calibrate`` keeps a carried stamp
    instead of recomputing it.
    """
    student = policy_ref(policy, model=model)
    groups = _group_label_lists(graded)
    stamped = 0
    tasks: dict[str, dict[str, Any]] = {}
    for row in kept:
        if not isinstance(row, dict):
            continue
        labels = groups.get(task_key(row))
        if not labels:
            continue
        record = Calibration(
            task_id=_task_id(row),
            student=_student_for(row, student),
            n=len(labels),
            pass_rate=sum(labels) / len(labels),
            pass_rate_ci95=_ci95(labels),
        )
        row["calibration"] = asdict(record)
        tasks.setdefault(
            record.task_id,
            {
                "task_id": record.task_id,
                "n": record.n,
                "pass_rate": record.pass_rate,
                "pass_rate_ci95": record.pass_rate_ci95,
            },
        )
        stamped += 1
    return {
        "n_tasks": len(groups),
        "n_rows": len(kept),
        "n_stamped": stamped,
        # one row per stamped task: the difficulty and its interval
        "tasks": list(tasks.values()),
    }


def calibrate(
    rows: Sequence[dict],
    *,
    policy: PolicyRef | dict | str | None = None,
    model: str | None = None,
    ref: str | Sequence[dict] | None = None,
) -> dict[str, Any]:
    """Stamp ``calibration`` on every graded row, in place.

    The per-task pass rate is over the binary rewards grouped by prompt,
    the same grouping ``group_signal`` and ``pass_at`` use. Rows without a
    0/1 reward are left alone and counted. Returns a report with the
    number of tasks and rows stamped plus the ``pass_at`` summary.
    ``ref`` (a key holding the reference model's summed logprob, or rows
    scored under it) fills ``mean_kl`` per task from the captured
    logprobs; see ``mean_kl``.

    A row whose carried stamp counts more repeats than these rows hold
    keeps it (``n_carried`` in the report): ``optimize(mode="rl")`` drops
    duplicate trajectories, and recomputing here would report the
    post-dedup k as the policy's pass rate over k repeats. The producing
    policy and ``mean_kl`` are still filled in from this call.
    """
    student = policy_ref(policy, model=model)
    groups = _group_label_lists(rows)
    kl_per_task: dict[str, float] = {}
    kl_report: dict[str, Any] | None = None
    if ref is not None:
        from .logprobs import mean_kl

        kl_report = mean_kl(rows, ref)
        kl_per_task = dict(kl_report["per_task"])
    stamped = 0
    carried = 0
    for row in rows:
        if not isinstance(row, dict) or _binary_label(row) is None:
            continue
        labels = groups.get(task_key(row))
        if not labels:
            continue
        kl = kl_per_task.get(_task_id(row))
        existing = calibration_of(row)
        if existing is not None and existing.n > len(labels):
            # The group shrank after it was measured (dedupe, the
            # unanimous trim, the band). The carried pass rate is the one
            # the policy earned; only identity and KL are refreshed here.
            record = replace(
                existing,
                student=student if _identified(student) else existing.student,
                mean_kl=kl if kl is not None else existing.mean_kl,
            )
            carried += 1
        else:
            record = Calibration(
                task_id=_task_id(row),
                student=_student_for(row, student),
                n=len(labels),
                pass_rate=sum(labels) / len(labels),
                mean_kl=kl,
                pass_rate_ci95=_ci95(labels),
            )
        row["calibration"] = asdict(record)
        stamped += 1
    rates = pass_at(rows)
    return {
        "n_tasks": len(groups),
        "n_rows": len(rows),
        "n_stamped": stamped,
        "n_carried": carried,
        "n_unstamped": len(rows) - stamped,
        "student": asdict(student),
        "pass_at": rates.to_dict(),
        **({"mean_kl": kl_report} if kl_report is not None else {}),
    }


def is_rl_shaped(rows: Sequence[dict], *, mode: str | None = None) -> bool:
    """RL data means repeats of one ask. ``mode="rl"`` says so outright;
    otherwise any prompt seen more than once counts."""
    if str(mode or "").lower() == "rl":
        return True
    seen: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = task_key(row)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            return True
    return False


def publish_gate(
    rows: Sequence[dict],
    *,
    mode: str | None = None,
    band: tuple[float, float] = DEFAULT_BAND,
    policy: PolicyRef | dict | str | None = None,
    model: str | None = None,
    strict: bool = True,
    endorsed: Sequence[str] = (),
    strict_hacks: bool = False,
) -> dict[str, Any]:
    """Check, calibrate, and report. Raises ``PublishGateError`` when
    ``strict`` and the rows are RL-shaped but ungraded or carry no mixed
    group, or when ``strict_hacks`` and ``hack_scan`` (with ``endorsed``
    naming what the reward should track) finds the reward best explained
    by something else. Never mutates anything except the ``calibration``
    stamp. ``judge_trust`` in the report is the summary ``grade`` stamped
    on the rows when they carried human labels, else ``None``.
    """
    lo, hi = float(band[0]), float(band[1])
    rl = is_rl_shaped(rows, mode=mode)
    calibration = calibrate(rows, policy=policy, model=model)
    signal = group_signal(rows, lo=lo, hi=hi)
    graded = calibration["n_stamped"]
    warnings: list[str] = []
    refusal: str | None = None
    scan = hack_scan(rows, endorsed=endorsed) if rl else None

    if rl and graded == 0:
        refusal = (
            "ungraded_rl_rows: RL data needs a 0/1 reward on every rollout so a "
            "grouped update has contrast; grade first (wai.grade or data.grade)"
        )
    elif rl and signal.get("n_mixed", 0) == 0:
        refusal = (
            "no_mixed_groups: every ask is unanimous, so group-relative advantages "
            "are zero everywhere and the run would train nothing; regrade with a "
            "stricter rubric or raise difficulty before publishing"
        )
    elif strict_hacks and scan and scan["regime"] == "reward_hack":
        refusal = (
            f"reward_hack: {scan['warnings'][0]}; fix the judge (judge_trust, "
            "endorsed=) before publishing, or push with strict_hacks=False"
        )
    if rl and not refusal:
        out_of_band = signal["n_mixed"] - signal["n_in_band"]
        if out_of_band:
            warnings.append(
                f"{out_of_band} mixed ask(s) fall outside the {lo:.0%}-{hi:.0%} band and "
                'were not pruned; run select(mode="rl") before push to enforce it'
            )
        if signal["n_all_zero"] or signal["n_all_one"]:
            warnings.append(
                f"{signal['n_all_zero'] + signal['n_all_one']} unanimous ask(s) still "
                'present; select(mode="rl") drops them'
            )
        if calibration["n_unstamped"]:
            warnings.append(f"{calibration['n_unstamped']} row(s) have no 0/1 reward")
        if calibration["n_carried"]:
            k = calibration["pass_at"]["k"]
            warnings.append(
                f"{calibration['n_carried']} row(s) keep the calibration measured before "
                f'select(mode="rl") pruned their ask; the stamp is the graded pass rate '
                f"over k repeats, this report's pass_at is over the {k} row(s) per ask "
                "that remain"
            )
    # Hygiene is reported, never applied here: push uploads rows as they
    # are, and the drops live in optimize / select_for_rl.
    _kept, duplicates = dedupe_groups(rows)
    near_dups = near_duplicate_prompts(rows)
    lengths = length_report(rows)
    correlations = reward_correlations(rows)
    hygiene = hygiene_warnings(
        duplicates=duplicates,
        near_dups=near_dups,
        lengths=lengths,
        correlations=correlations,
        scan=scan if not refusal else None,
    )
    if duplicates["n_dropped"]:
        hygiene[0] = hygiene[0].replace(" dropped", ' present; select(mode="rl") drops them', 1)
    warnings.extend(hygiene)
    # The judge check grade ran on these rows, when any carried human gold.
    trust = next(
        (
            r["judge_meta"]["trust"]
            for r in rows
            if isinstance(r, dict)
            and isinstance(r.get("judge_meta"), dict)
            and r["judge_meta"].get("trust")
        ),
        None,
    )
    report = {
        "ok": refusal is None,
        "judge_trust": trust,
        "rl_shaped": rl,
        "mode": mode,
        "band": [lo, hi],
        "signal": signal,
        "calibration": calibration,
        "duplicates": duplicates,
        "near_duplicate_prompts": near_dups,
        "length": lengths,
        "correlations": correlations,
        "hack_scan": scan,
        "warnings": warnings,
        "refusal": refusal,
    }
    if refusal and strict:
        raise PublishGateError(refusal)
    return report


__all__ = [
    "PublishGateError",
    "calibrate",
    "carry_calibration",
    "is_rl_shaped",
    "policy_ref",
    "publish_gate",
]
