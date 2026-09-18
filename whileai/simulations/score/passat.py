"""pass@1, pass^k and pass@k from the same graded groups.

Three numbers, one primitive: the per-task pass rate ``c / n`` over the
``k`` repeats of one phrasing. Each has one job.

* ``pass@1`` is the mean per-task pass rate. The agent runs once in
  production, so this is the measurement headline.
* ``pass^k`` is the chance all ``k`` repeats pass (tau-bench's
  consistency metric). A behavior that lands 60% of the time is not a
  contract; this is the reliability line.
* ``pass@k`` is the chance at least one of ``k`` repeats passes.
  ``pass@k - pass@1`` is the headroom a grouped RL update can learn
  from: the same asks ``group_signal`` counts as mixed.

Both ``pass^k`` and ``pass@k`` use the unbiased combinatorial estimators
(Chen et al. 2021 for pass@k; Yao et al. 2024 for pass^k), so a group
with ``n`` graded repeats contributes ``1 - C(n-c, k) / C(n, k)`` and
``C(c, k) / C(n, k)`` rather than a plug-in power of ``c / n``.

Judge noise: with an LLM judge, pass@k inflates on false positives and
pass^k inflates on false negatives. pass@1 is the least sensitive of the
three, which is why it carries the headline.

Intervals: all three carry a 95% percentile bootstrap over tasks
(rlhf-book ch. 16: intervals come from resampling prompts, never rows).
``.ci95`` is pass@1's; ``.pass_pow_k_ci95`` and ``.pass_at_k_ci95``
resample the per-group unbiased estimates of the k-eligible groups, so
the reliability line is reported with the uncertainty of the tasks it
was measured on. Fewer than three groups gives ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..text import split_reasoning

JUDGE_NOISE_NOTE = (
    "pass@k inflates on judge false positives, pass^k on false negatives; "
    "pass@1 is the least judge-sensitive"
)


def _comb(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def _pass_at_k_group(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one of k draws passes) from n samples, c passing."""
    total = _comb(n, k)
    if total == 0:
        return float("nan")
    return 1.0 - _comb(n - c, k) / total


def _pass_pow_k_group(n: int, c: int, k: int) -> float:
    """Unbiased P(all k draws pass) from n samples, c passing."""
    total = _comb(n, k)
    if total == 0:
        return float("nan")
    return _comb(c, k) / total


CONFIG_KEYS = ("temperature", "max_tokens", "policy_version", "judge_version", "prompt_hash")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _row_config_values(row: dict) -> dict[str, Any]:
    """The config facts one row carries, ``None`` where it carries none."""
    sampling = _dict(row.get("sampling"))
    policy_version = row.get("policy_version")
    judge_meta = _dict(row.get("judge_meta"))
    lineage = _dict(row.get("lineage"))
    return {
        "temperature": sampling.get("temperature"),
        "max_tokens": sampling.get("max_tokens"),
        "policy_version": str(policy_version) if policy_version else None,
        "judge_version": judge_meta.get("version") or lineage.get("judge_version") or None,
        # policy_version is <model>@<sha256 of the system prompt>[:16]
        "prompt_hash": str(policy_version).split("@", 1)[1]
        if policy_version and "@" in str(policy_version)
        else None,
    }


def run_config(
    rows: Sequence[dict], *, n_tasks: int | None = None, k: int | None = None
) -> dict[str, Any]:
    """What the rows say about how they were produced (rlhf-book ch. 16:
    a number without its sampling settings, prompt and judge is not
    comparable to another). ``temperature`` and ``max_tokens`` come from
    each row's ``sampling``, ``policy_version`` and ``prompt_hash`` from
    its policy stamp, ``judge_version`` from its judge stamp. A field is
    the one value every row agrees on; rows that lack it are skipped, and
    a field the rows disagree on is ``None`` and listed in ``mixed``."""
    seen: dict[str, set[Any]] = {key: set() for key in CONFIG_KEYS}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in _row_config_values(row).items():
            if value is not None:
                seen[key].add(value)
    out: dict[str, Any] = {"n_tasks": n_tasks, "k": k}
    mixed: list[str] = []
    for key in CONFIG_KEYS:
        values = seen[key]
        if len(values) > 1:
            mixed.append(key)
        out[key] = next(iter(values)) if len(values) == 1 else None
    out["mixed"] = mixed
    # The share of rows the token cap cut. A side that was cut more often
    # is not the same eval; ``delta_report`` warns when the two differ.
    # ``None`` when no row says how it finished (rows from before 0.54).
    reasons = [r.get("finish_reason") for r in rows if isinstance(r, dict)]
    known = [x for x in reasons if isinstance(x, str)]
    out["truncated_share"] = (
        round(sum(1 for x in known if x == "length") / len(known), 4) if known else None
    )
    # Answer production. Every rate is conditional on the arm having
    # produced a reply to score: the share of rows with spoken text once
    # reasoning markup is gone, and the share whose reply ends inside an
    # unclosed <think> (the cap landed mid-reasoning). ``delta_report``
    # fails the comparison when one side answered and the other did not
    # (#297). ``None`` on an empty set.
    answered, unclosed, with_reply = answer_counts(rows)
    if with_reply:
        out["answered_share"] = round(answered / with_reply, 4)
        out["unclosed_think_share"] = round(unclosed / with_reply, 4)
    else:
        out["answered_share"] = None
        out["unclosed_think_share"] = None
    return out


def answer_counts(rows: Sequence[dict]) -> tuple[int, int, int]:
    """``(answered, unclosed, with_reply)``: rows whose reply has spoken
    text once ``<think>`` markup is gone, rows whose reply ends inside an
    unclosed ``<think>``, and rows that carry a reply field at all (the
    denominator of ``answered_share``; ``delta_report`` needs the counts
    for its two-proportion test)."""
    replies = [_reply_text(r) for r in rows if isinstance(r, dict)]
    said = [text for text in replies if text is not None]
    split = [split_reasoning(text) for text in said]
    answered = sum(1 for spoken, _, _ in split if spoken.strip())
    unclosed = sum(1 for _, _, cut in split if cut)
    return answered, unclosed, len(split)


def _reply_text(row: dict) -> str | None:
    """The reply a judge read on this row: ``final_text``, else the last
    assistant message with words in it (rows from another harness carry
    ``messages`` only). ``None`` when the row carries neither field, so
    it is skipped rather than read as unanswered."""
    final = row.get("final_text")
    if isinstance(final, str) and final.strip():
        return final
    messages = row.get("messages")
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    if isinstance(final, str) or isinstance(messages, list):
        return ""
    return None


@dataclass(frozen=True)
class PassAt:
    """pass@1 / pass^k / pass@k over graded groups. See module docstring.

    Every field, and the name it prints as in ``str(...)``. The printed
    line and the attribute are not spelled the same: pass^k is
    ``pass_pow_k`` (not ``pass_hat_k``), and the print says both once so
    the attribute is readable off it.

    | attribute            | prints as        | what it is                                 |
    | -------------------- | ---------------- | ------------------------------------------ |
    | ``k``                | ``k=4``          | draw size the k-way numbers used           |
    | ``pass_at_1``        | ``pass@1``       | mean per-task pass rate, the headline      |
    | ``pass_pow_k``       | ``pass^4``       | chance all k repeats pass (reliability)    |
    | ``pass_at_k``        | ``pass@4``       | chance at least one of k passes            |
    | ``headroom``         | ``headroom``     | property: pass@k minus pass@1              |
    | ``ci95``             | ``[lo..hi]``     | task-bootstrap interval on pass@1          |
    | ``pass_pow_k_ci95``  | ``[lo..hi]``     | same for pass^k; ``None`` under 3 groups   |
    | ``pass_at_k_ci95``   | ``[lo..hi]``     | same for pass@k; ``None`` under 3 groups   |
    | ``n_groups``         | ``N groups``     | tasks pass@1 averaged over                 |
    | ``n_rows``           | not printed      | graded rows behind those tasks             |
    | ``n_groups_at_k``    | not printed      | tasks the k-way numbers used               |
    | ``n_groups_imputed`` | not printed      | short unanimous tasks counted in           |
    | ``per_task``         | not printed      | ``{task key: pass rate}``, a dict          |
    | ``note``             | tail of the line | why a number is missing, and the fix       |
    | ``config``           | token-cap share  | how the rows were made (``run_config``)    |

    ``to_dict()`` uses these same keys, with ``headroom`` added and the
    intervals as lists.
    """

    k: int
    pass_at_1: float | None
    pass_pow_k: float | None
    pass_at_k: float | None
    n_groups: int
    n_rows: int
    n_groups_at_k: int = 0
    #: unanimous groups shorter than k counted as if they stayed unanimous
    n_groups_imputed: int = 0
    #: ``{task: c / n}`` — a **dict keyed by the group's task key** (its
    #: ``scenario_id``, else ``task_id``, else the prompt text; see
    #: ``task_key``), not a list, so ``per_task[0]`` is a ``KeyError``, not
    #: the first task. Iterate it as ``.per_task.items()``;
    #: ``.per_task.values()`` is the pass-rate vector pass@1 averages and
    #: ``ci95`` bootstraps.
    per_task: dict[str, float] = field(default_factory=dict)
    note: str = ""
    #: how the rows were produced, read off the rows (``run_config``):
    #: task count, k, temperature, max_tokens, policy and judge versions,
    #: prompt hash. A field the rows disagree on is ``None`` and named in
    #: ``config["mixed"]``.
    config: dict[str, Any] = field(default_factory=dict)
    #: task-bootstrap 95% interval on pass@1
    ci95: tuple[float, float] | None = None
    #: task-bootstrap 95% intervals on pass^k and pass@k, over the
    #: k-eligible groups' per-group estimates; ``None`` below ``min_k``
    #: or with fewer than three eligible groups
    pass_pow_k_ci95: tuple[float, float] | None = None
    pass_at_k_ci95: tuple[float, float] | None = None

    @property
    def headroom(self) -> float | None:
        """pass@k minus pass@1: what a grouped RL update has to learn from."""
        if self.pass_at_k is None or self.pass_at_1 is None:
            return None
        return self.pass_at_k - self.pass_at_1

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "pass_at_1": self.pass_at_1,
            "pass_pow_k": self.pass_pow_k,
            "pass_at_k": self.pass_at_k,
            "headroom": self.headroom,
            "n_groups": self.n_groups,
            "n_groups_at_k": self.n_groups_at_k,
            "n_groups_imputed": self.n_groups_imputed,
            "n_rows": self.n_rows,
            "note": self.note,
            "ci95": list(self.ci95) if self.ci95 else None,
            "pass_pow_k_ci95": list(self.pass_pow_k_ci95) if self.pass_pow_k_ci95 else None,
            "pass_at_k_ci95": list(self.pass_at_k_ci95) if self.pass_at_k_ci95 else None,
            "config": dict(self.config),
        }

    def __str__(self) -> str:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        def band(ci: tuple[float, float] | None) -> str:
            return f" [{ci[0]:.2f}..{ci[1]:.2f}]" if ci else ""

        head = (
            f"pass@1 {fmt(self.pass_at_1)}{band(self.ci95)} | "
            # the attribute name once, where it is read: testers guessed
            # pass_hat_k from the printed pass^k and lost a round trip
            f"pass^{self.k} (pass_pow_k) {fmt(self.pass_pow_k)}{band(self.pass_pow_k_ci95)} | "
            f"pass@{self.k} {fmt(self.pass_at_k)}{band(self.pass_at_k_ci95)} | "
            f"headroom {fmt(self.headroom)}"
        )
        tail = f"({self.n_groups} groups, k={self.k}"
        cut = self.config.get("truncated_share") if self.config else None
        if cut:
            tail += f"; {cut:.0%} of rows cut by the token cap"
        answered = self.config.get("answered_share") if self.config else None
        if answered is not None and answered < 1:
            tail += f"; {1 - answered:.0%} of rows have no spoken reply"
        if self.note:
            tail += f"; {self.note}"
        return f"{head} {tail})"


def _nothing_to_score(rows: Sequence[dict]) -> str:
    """Why no row carried a binary reward.

    "grade first" is right when nothing has been judged, and wrong -- it
    sends the user back to the step that already ran -- when grading did
    happen and every row failed. A cold hosted judge does exactly that: all
    the concurrent calls time out together and the whole set reads as
    ungraded.
    """
    judged = [r for r in rows if isinstance(r, dict) and r.get("judge_status")]
    failed = [r for r in judged if str(r.get("judge_status")) != "ok"]
    if not judged or len(failed) != len(judged):
        return "no binary rewards; grade first"
    statuses = "/".join(sorted({str(r.get("judge_status")) for r in failed}))
    reason = next((str(r.get("reason") or "").strip() for r in failed if r.get("reason")), "")
    tail = f": {reason[:120]}" if reason else ""
    return f"the judge failed on all {len(failed)} rows ({statuses}){tail}; re-run the judge"


def pass_at(
    rows: Sequence[dict] | Any,
    *,
    k: int | None = None,
    min_k: int = 4,
    unanimous_short: bool = False,
) -> PassAt:
    """pass@1, pass^k and pass@k from graded rows, grouped by task.

    A task is a situation, not a string. Rows group under ``task_key``:
    the engine's ``scenario_id`` when the row has one, else ``task_id``,
    else the prompt text. In ``mode="rl"`` the repeats of one opener share
    a ``scenario_id``, and so do the textured phrasings of one situation,
    so those phrasings pool into one task on purpose: the question is
    whether the agent handles the situation, not one wording of it.
    ``compare_runs``, ``delta_report``, ``eval_variance``, ``curriculum``
    and ``group_signal`` count tasks with the same key, so
    ``pass_at(rows).n_groups`` and ``delta_report(...)["n_paired_tasks"]``
    agree on the same rows.

    Only binary ``reward`` (or ``qwen_reward``) rows count; partial and
    unjudged rows are skipped, the same rule ``group_signal`` uses.
    ``k`` defaults to the smallest group of two or more repeats, so every
    such group contributes to the ``k``-way estimators; groups with fewer
    than ``k`` graded repeats are left out of pass^k and pass@k (counted
    in ``n_groups_at_k``). pass@1 always averages every group.

    Below ``min_k`` repeats the ``k``-way numbers are ``None`` with a
    ``note`` instead of a number too noisy to act on. Pass ``k=`` to
    choose the draw size yourself.

    ``unanimous_short=True`` counts a unanimous group shorter than ``k``
    as if it stayed unanimous (pass^k and pass@k equal to its pass rate,
    1 or 0). That is the assumption a successive-allocation run stopped
    on, and leaving those groups out would score only the tasks that
    split and inflate the headroom. Mixed short groups still stay out.

    ``.config`` says how the rows were produced (``run_config``): task
    count, k, temperature, max_tokens, policy and judge versions, prompt
    hash, with a ``mixed`` list naming any the rows disagree on.
    """
    from .optimize import _group_label_lists

    row_list = list(rows) if not isinstance(rows, list) else rows
    groups = _group_label_lists(row_list)
    n_rows = sum(len(labels) for labels in groups.values())
    if not groups:
        return PassAt(
            k=int(k or 1),
            pass_at_1=None,
            pass_pow_k=None,
            pass_at_k=None,
            n_groups=0,
            n_rows=0,
            note=_nothing_to_score(row_list),
            config=run_config(row_list, n_tasks=0, k=int(k or 1)),
        )

    per_task = {prompt: sum(labels) / len(labels) for prompt, labels in groups.items()}
    pass_at_1 = sum(per_task.values()) / len(per_task)

    multi = [len(labels) for labels in groups.values() if len(labels) >= 2]
    resolved_k = int(k) if k is not None else (min(multi) if multi else 1)
    if resolved_k < 1:
        raise ValueError("k must be at least 1")

    eligible = [labels for labels in groups.values() if len(labels) >= resolved_k]
    imputed = (
        [labels for labels in groups.values() if len(labels) < resolved_k and len(set(labels)) == 1]
        if unanimous_short
        else []
    )
    note = ""
    pass_pow_k: float | None = None
    pass_at_k: float | None = None
    pow_vals: list[float] = []
    at_vals: list[float] = []
    sizes = sorted(set(multi))
    uneven = k is None and len(sizes) > 1
    if resolved_k < max(1, int(min_k)):
        note = f"set repeats>={int(min_k)} for pass^k and pass@k"
        if uneven:
            # A time or row budget cut mid-group leaves ragged groups; k
            # defaults to the smallest, so the k-way numbers vanish even
            # though most groups reached the requested repeats.
            reached = sum(1 for n in multi if n >= sizes[-1])
            note = (
                f"groups are uneven ({sizes[0]} to {sizes[-1]} repeats); k defaults "
                f"to the smallest, so pass k={sizes[-1]} to score the {reached} "
                f"group(s) that reached it, or finish the cut groups"
            )
    elif not eligible and not imputed:
        note = f"no group has {resolved_k} graded repeats"
    else:
        pow_vals = [_pass_pow_k_group(len(g), sum(g), resolved_k) for g in eligible]
        at_vals = [_pass_at_k_group(len(g), sum(g), resolved_k) for g in eligible]
        for g in imputed:
            unanimous_value = float(g[0])
            pow_vals.append(unanimous_value)
            at_vals.append(unanimous_value)
        pass_pow_k = sum(pow_vals) / len(pow_vals)
        pass_at_k = sum(at_vals) / len(at_vals)
        if uneven:
            note = f"groups are uneven ({sizes[0]} to {sizes[-1]} repeats); k is the smallest"
    from .stats import bootstrap_ci

    return PassAt(
        k=resolved_k,
        pass_at_1=pass_at_1,
        pass_pow_k=pass_pow_k,
        pass_at_k=pass_at_k,
        n_groups=len(groups),
        n_rows=n_rows,
        n_groups_at_k=(len(eligible) + len(imputed)) if pass_at_k is not None else 0,
        n_groups_imputed=len(imputed) if pass_at_k is not None else 0,
        per_task=per_task,
        note=note,
        config=run_config(row_list, n_tasks=len(groups), k=resolved_k),
        ci95=bootstrap_ci(list(per_task.values())),
        pass_pow_k_ci95=bootstrap_ci(pow_vals) if pass_pow_k is not None else None,
        pass_at_k_ci95=bootstrap_ci(at_vals) if pass_at_k is not None else None,
    )


__all__ = ["JUDGE_NOISE_NOTE", "PassAt", "pass_at", "run_config"]
