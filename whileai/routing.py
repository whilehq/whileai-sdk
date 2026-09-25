"""``wai.methods.route``: which training method a graded pool can support.

    import whileai as wai

    rows = [{"task_id": f"t{t}", "reward": float(i < t % 5)} for t in range(160) for i in range(4)]
    r = wai.methods.route(rows)
    print(r)          # the method, why, what blocks the others, what would unblock them
    r.method          # "grpo"

The package ships more methods than any one playbook keeps in view: the
hosted ``sft``/``grpo``/``dpo``/``rm``, and on your own GPUs ``OPD``,
``OPSD``, ``GroupwiseGrading`` and the single-rollout family. Each already
has its own measurement somewhere (``group_signal``, ``pass_at``,
``DIFFICULTY_BAND``, the judge floors, the OPD docstring's teacher check);
a coding agent that reached for one or two of them routed on the use case
instead of the rows. ``route`` reads the rows with those same measurements,
in the order an upstream failure voids the checks below it, and scores
every method. It trains nothing and calls no model.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .report import Report
from .simulations.defaults import (
    CEILING_PASS_RATE,
    DIFFICULTY_BAND,
    DIFFICULTY_BAND_ROLLOUTS,
    MIN_AGREEMENT,
    MIN_KAPPA,
    PROVE_EFFECT,
    REJECTION_SAMPLING_MIN_K,
    RL_ROLLOUTS_PER_PROMPT,
    ROUTE_FLOOR_SHARE,
    ROUTE_MAX_K,
    ROUTE_MIN_MIXED_SHARE,
    ROUTE_MIN_TASKS_FOR_K,
    ROUTE_OPSD_MIN_PARAMS_B,
    ROUTE_TEACHER_MAX_TRUNCATED,
    TRAIN_MIN_MIXED_TASKS,
)
from .simulations.score.hack_scan import hack_scan
from .simulations.score.hygiene import is_truncated, length_report
from .simulations.score.judging import length_confound_warning
from .simulations.score.optimize import _is_binary_01
from .simulations.score.passat import pass_at
from .simulations.score.stats import task_key, wilson_interval

#: The methods ``route`` scores, in the order it prefers one when several
#: can run. A teacher that clears every guard first (a dense per-token
#: signal that needs no group), then grpo while the pool still has groups
#: inside DIFFICULTY_BAND (trim the unanimous tasks and train, the band
#: practice), then the methods for what grpo cannot reach (OPSD on a
#: floor, GroupwiseGrading on a saturated pool), then dpo, the
#: single-rollout family, and SFT last. skills/pick-a-method asks the same
#: questions; it reads a floor as every task failing, and ``route`` as
#: ROUTE_FLOOR_SHARE of them, so a pool can be both a partial floor and a
#: grpo pool, and ``plan`` then names both. (convention, untested)
ROUTED: tuple[str, ...] = ("opd", "grpo", "opsd", "groupwise", "dpo", "flashreinforce", "sft")

#: What ``route`` does not decide, said on every report so the edge of the
#: instrument is visible (style rule 5).
NOT_ROUTED = (
    "rm (where a program grades, the program is the reward); Async and its correction "
    "(wrap the routed method when the sampler lags the policy); OPD's divergence and top_k "
    "and OPSD's anchor (not decidable from rows; the docstrings give the defaults' sources); "
    "Swarm (a rollout rule, not a trainer; gate it on Swarm.calibration); learning rate, "
    "clip and KL (see the method's docstring); the band per stance or difficulty (a pool "
    "mixed overall can floor on one slice: route each slice)"
)

#: The checks a pool still needs before the GPU, whatever ``route`` picks.
BEFORE_THE_GPU = (
    "size the held-out test (holdout_size) and measure its noise over three runs "
    "(eval_variance); decontaminate the pool against it; train a random selection of the "
    "same size beside the real one; keep one chat template across stages and mask tool "
    "turns out of the loss (loss_mask); check over-refusal on a benign set (refusal_report); "
    "and with beta=0, watch the drift from the reference (mean_kl) (Lambert 2025, chapters "
    "Instruction Fine-Tuning, Tool Use, Over-Optimization, Regularization, Evaluation)"
)

_MODEL_KEYS = ("model_version", "agent_model", "model")


class Route(Report):
    """What ``route`` found. A dict for reading keys, attributes for code.

    ``method`` is the method to run now, or ``None``. ``plan`` adds the
    methods that cover what ``method`` leaves out (the floored tasks, the
    saturated ones). ``methods`` scores every routed method as
    ``{"ok": bool, "why": str}``. ``why`` is the arithmetic behind
    ``method``. ``blocked`` lists what stops each method that cannot run.
    ``need`` holds the numbers that would unblock one (``tasks_total``,
    ``k``, ``teacher_completions``). ``measured`` holds every number the
    decision read, and ``notes`` every check that could not run and why.
    """

    _summary_keys = ("method", "why")

    @property
    def method(self) -> str | None:
        return self.get("method")

    @property
    def plan(self) -> list[str]:
        return list(self.get("plan") or [])

    @property
    def methods(self) -> dict[str, dict[str, Any]]:
        return dict(self.get("methods") or {})

    @property
    def why(self) -> str:
        return str(self.get("why") or "")

    @property
    def blocked(self) -> list[str]:
        return list(self.get("blocked") or [])

    @property
    def need(self) -> dict[str, Any]:
        return dict(self.get("need") or {})

    @property
    def notes(self) -> list[str]:
        return list(self.get("notes") or [])

    @property
    def measured(self) -> dict[str, Any]:
        return dict(self.get("measured") or {})

    def __str__(self) -> str:
        lines = [f"route: {self.method or 'nothing trains yet'}", f"  why: {self.why}"]
        if len(self.plan) > 1:
            lines.append("  plan: " + " -> ".join(self.plan))
        lines.append("  methods:")
        for name, row in self.methods.items():
            mark = "ok" if row.get("ok") else "no"
            lines.append(f"    {mark:2}  {name:15} {row.get('why', '')}")
        if self.blocked:
            lines.append("  blocked:")
            lines.extend(f"    - {b}" for b in self.blocked)
        if self.need:
            lines.append("  need: " + ", ".join(f"{k} {v}" for k, v in self.need.items()))
        m = self.measured
        lines.append("  measured: " + ", ".join(f"{k} {m[k]}" for k in m if m[k] is not None))
        lines.extend(f"  note: {note}" for note in self.notes)
        lines.append(f"  before the GPU: {BEFORE_THE_GPU}")
        lines.append(f"  not routed: {NOT_ROUTED}")
        return "\n".join(lines)


def _label(row: Mapping[str, Any]) -> int | None:
    """The row's 0/1 verdict, the way ``group_signal`` and ``pass_at`` read it."""
    for key in ("reward", "qwen_reward"):
        value = row.get(key)
        if value is not None and _is_binary_01(value):
            return int(float(value))
    return None


def _partial(row: Mapping[str, Any]) -> bool:
    value = row.get("reward")
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and not _is_binary_01(value)
    )


def _judge_verdict(judge: Any) -> tuple[bool | None, str]:
    """(ok, line) off a ``compare_judges`` row, a ``judge_trust`` report or a mapping.

    ``ok`` on the object wins (``compare_judges`` sets it from the Wilson
    lower bound). A mapping without it needs ``agreement``, ``kappa`` and
    ``n``: the Wilson lower bound of agreement over ``n`` labels must clear
    ``MIN_AGREEMENT`` and kappa ``MIN_KAPPA``, the same floors. Without
    ``n`` the agreement is a point estimate and the verdict is ``None``.
    """
    if judge is None:
        return None, ""

    def read(key: str) -> Any:
        return judge.get(key) if isinstance(judge, Mapping) else getattr(judge, key, None)

    agreement, kappa, leak, n = read("agreement"), read("kappa"), read("leak"), read("n")
    bits = [
        f"{k} {v:.2f}"
        for k, v in (("agreement", agreement), ("kappa", kappa), ("false-pass", leak))
        if isinstance(v, (int, float))
    ]
    if isinstance(n, int):
        bits.append(f"n {n}")
    line = ", ".join(bits)
    ok = read("ok")
    if ok is not None:
        return bool(ok), line
    if not isinstance(agreement, (int, float)) or not isinstance(kappa, (int, float)):
        return None, line
    if not isinstance(n, int) or n <= 0:
        return None, line
    ci = wilson_interval(round(agreement * n), n)
    return bool(ci and ci[0] >= MIN_AGREEMENT and kappa >= MIN_KAPPA), line


def _judge_graded(rows: Sequence[Mapping[str, Any]]) -> bool:
    """True when any row's reward came from a model judge, not a program."""
    for row in rows:
        meta = row.get("judge_meta")
        kind = meta.get("scorer_kind") if isinstance(meta, Mapping) else None
        if kind is None and row.get("judge_name"):
            kind = "judge"
        if kind in ("judge", "reward_model"):
            return True
    return False


def _sampled_by(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        named = next((str(row[key]) for key in _MODEL_KEYS if row.get(key)), None)
        sampling = row.get("sampling")
        if named is None and isinstance(sampling, Mapping) and sampling.get("model"):
            named = str(sampling["model"])
        if named:
            out.add(named)
    return out


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _mixed_chance(passes: int, draws: int, k: int) -> float:
    """P(k fresh rollouts of a task disagree), averaged over what ``passes``
    of ``draws`` says about its rate.

    With a uniform Beta(1, 1) prior (the prior behind ``defaults.laplace``)
    the rate is Beta(s + 1, f + 1), and E[p**k] = B(s+1+k, f+1) / B(s+1, f+1)
    exactly, likewise E[(1-p)**k]. Plugging the posterior mean into
    1 - p**k - (1-p)**k instead overstates the chance (the function is not
    linear in p): 0 of 2 at k=8 reads 0.90 plugged in and 0.72 exactly.
    """
    s, f = passes, draws - passes
    base = _log_beta(s + 1, f + 1)
    p_all = math.exp(_log_beta(s + 1 + k, f + 1) - base)
    q_all = math.exp(_log_beta(s + 1, f + 1 + k) - base)
    return 1.0 - p_all - q_all


def route(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    teacher: Sequence[Mapping[str, Any]] | None = None,
    judge: Any = None,
    size_b: float | None = None,
    vocab: tuple[int, int] | None = None,
    effect: float = PROVE_EFFECT,
    min_mixed: int = TRAIN_MIN_MIXED_TASKS,
) -> Route:
    """Score every training method against a graded pool, and pick one.

    ``rows`` are graded rollouts of the student on the tasks it would train
    on, several per task (grouped by ``task_key``, read as 0/1 the way
    ``group_signal`` and ``pass_at`` read them; partial-credit rows are
    counted and left out, as they are there). The checks run in the order
    an upstream failure voids the ones below it:

    1. **The grader.** Judge-graded rows need ``judge=``: a
       ``compare_judges`` row (its ``ok`` is the Wilson lower bound of
       agreement against ``MIN_AGREEMENT`` and kappa against
       ``MIN_KAPPA``), or a mapping with ``agreement``, ``kappa`` and
       ``n``; a learned reward model's rows (``scorer_kind``
       ``"reward_model"``) need the same. A judge is a generative reward
       model (Lambert 2025, chapter Reward Modeling), a proxy checked against
       people before its scores choose data (chapter Evaluation).
    2. **Truncation.** A passing row that did not end on its own
       (``hygiene.is_truncated``: ``finish_reason == "length"`` or a
       truncated step) means the token cap, not the model, decides the
       rates (Lambert 2025, chapter Reinforcement Learning: score only
       completions that ended). Nothing is routed until it is fixed:
       score only replies that ended, raise ``max_tokens``, or train with
       ``truncated="mask"``.
    3. **Groups.** Tasks with one rollout are counted apart and never
       blamed (``group_signal``). With no groups, only SFT on the passes
       and the single-rollout family (``FlashReinforce``, ``SAO``,
       ``BPCO``, or prime-rl ``rae``) can run, and the last needs the
       sampler's log-probabilities on the rows.
    4. **The band.** grpo trains on the grouped tasks inside
       ``DIFFICULTY_BAND``, the ones ``select_for_rl`` keeps; it needs
       ``min_mixed`` of them and ``ROUTE_MIN_MIXED_SHARE`` of the grouped
       tasks (DAPO dynamic sampling, chapter Reinforcement Learning; the
       20-80% band, chapter Reasoning). dpo needs ``min_mixed`` tasks with a
       pass and a fail, and is off-policy by construction (chapter
       Reinforcement Learning). ``ROUTE_FLOOR_SHARE`` all-fail tasks score
       OPSD from ``ROUTE_OPSD_MIN_PARAMS_B`` up (with ``privileged=
       "reference"``: a floor has no passing demonstration to show), else
       SFT on a stronger model's completions; a pool at
       ``CEILING_PASS_RATE`` scores GroupwiseGrading with an audited
       grader. ``need["k"]`` is the smallest k at which the expected
       mixed share reaches ``ROUTE_MIN_MIXED_SHARE``, from each task's own
       passes under a Beta(1, 1) prior (``_mixed_chance``); under
       ``ROUTE_MIN_TASKS_FOR_K`` grouped tasks it asks for tasks instead.
    5. **On-policy.** With ``model=``, rows stamped by another model are an
       off-policy update with no importance ratio for grpo (chapter
       Reinforcement Learning); they route to SFT, dpo, or OPD from that
       model. Rows that name no model are said to be unchecked.
    6. **The teacher.** ``teacher=`` is the teacher's graded rows on the
       same tasks at the student's token cap. OPD pulls the student toward
       the teacher (Lambert 2025, chapter Synthetic Data and Distillation,
       eq. 10), so the teacher must cover the student's tasks, clear the
       student by ``effect`` with 95% intervals apart, finish under
       ``ROUTE_TEACHER_MAX_TRUNCATED`` cut off, and share the student's
       tokenizer (``vocab=(teacher, student)``; a mismatch drops the
       per-token signal, SimCT arXiv:2605.07711; KD usually needs a shared
       tokenizer, chapter Synthetic Data and Distillation).

    ``size_b`` is the student's size in billions of parameters.
    ``min_mixed`` is the fewest trainable tasks a grouped method runs on
    (``TRAIN_MIN_MIXED_TASKS``). Returns a ``Route`` that prints itself.

    Reference: Lambert 2025 (rlhfbook.com), chapters Reward Modeling,
    Rejection Sampling, Reinforcement Learning, Synthetic Data and
    Distillation, Evaluation; Shao et al. 2024 (arXiv:2402.03300); DAPO
    (arXiv:2503.14476); skills/pick-a-method, skills/strengthen-your-evals.
    """
    pool: list[dict[str, Any]] = [dict(r) for r in rows if isinstance(r, Mapping)]
    groups: dict[str, list[int]] = {}
    for row in pool:
        label = _label(row)
        if label is not None:
            groups.setdefault(task_key(row), []).append(label)
    grouped = {key: v for key, v in groups.items() if len(v) >= 2}  # noqa: PLR2004  # a group needs two
    singles = len(groups) - len(grouped)
    n = len(grouped)
    sizes = Counter(len(v) for v in grouped.values())
    k = sizes.most_common(1)[0][0] if sizes else (1 if groups else 0)
    rates = [sum(v) / len(v) for v in grouped.values()]
    lo, hi = DIFFICULTY_BAND
    all_fail = sum(1 for p in rates if p == 0.0)
    all_pass = sum(1 for p in rates if p == 1.0)
    mixed = n - all_fail - all_pass
    in_band = sum(1 for p in rates if lo <= p <= hi and 0.0 < p < 1.0)
    share = in_band / n if n else 0.0
    labelled = [r for r in pool if _label(r) is not None]
    passing_rows = sum(1 for r in labelled if _label(r) == 1)
    partial_rows = sum(1 for r in pool if _partial(r))
    truncated_passes = sum(1 for r in labelled if _label(r) == 1 and is_truncated(r))
    truncated_fails = sum(1 for r in labelled if _label(r) == 0 and is_truncated(r))
    mean_pass = sum(rates) / n if n else None
    judge_ok, judge_line = _judge_verdict(judge)
    judged = _judge_graded(pool)
    writers = _sampled_by(pool)
    on_policy: bool | None = writers == {model} if model and writers else None
    logprobs = any(r.get("token_logprobs") or r.get("logprobs") for r in pool)

    measured: dict[str, Any] = {
        "tasks": n,
        "single_rollout_tasks": singles or None,
        "k": k,
        "all_fail": all_fail,
        "all_pass": all_pass,
        "mixed": mixed,
        "in_band": in_band,
        "pass_at_1": round(mean_pass, 3) if mean_pass is not None else None,
        "passing_rows": passing_rows,
        "partial_rows": partial_rows or None,
        "truncated_passes": truncated_passes,
        "truncated_fails": truncated_fails,
        "on_policy": on_policy,
        "judge_graded": judged,
    }
    notes: list[str] = []
    if partial_rows:
        notes.append(
            f"{partial_rows} rows carry partial credit and are left out, as group_signal and "
            "pass_at leave them out: binarize at PASS_THRESHOLD or grade 0/1 to count them"
        )
    if on_policy is None:
        notes.append(
            "on-policy not checked: "
            + ("pass model=" if not model else "the rows name no model (agent_model)")
        )
    if teacher is not None and vocab is None:
        notes.append("tokenizer not checked: pass vocab=(teacher, student) from len(tokenizer)")
    if size_b is None:
        notes.append(
            f"student size not given: pass size_b= to check OPSD's ~{ROUTE_OPSD_MIN_PARAMS_B}B floor"
        )
    methods: dict[str, dict[str, Any]] = {}
    blocked: list[str] = []
    need: dict[str, Any] = {}

    def score(name: str, ok: bool, why: str) -> None:
        methods[name] = {"ok": ok, "why": why}
        if not ok:
            blocked.append(f"{name}: {why}")

    # 1-2: what voids every method at once.
    stop: list[str] = []
    if not groups:
        stop.append("no 0/1 graded rows: grade the pool (reward 0 or 1 per rollout) first")
    if judged and judge_ok is not True:
        if judge is None:
            stop.append(
                "the reward is a model judge nobody has checked: run compare_judges on "
                "blind labels (skills/audit-your-judge) and pass its row as judge="
            )
        elif judge_ok is None:
            stop.append(
                f"judge= has no label count ({judge_line}): pass a compare_judges row, or n="
            )
        else:
            stop.append(
                f"the judge misses the floors ({judge_line}; Wilson lower bound of agreement >= "
                f"{MIN_AGREEMENT}, kappa >= {MIN_KAPPA}): fix the rubric or the judge"
            )
    if truncated_passes:
        stop.append(
            f"{truncated_passes} of {passing_rows} passing rows did not end on their own: score "
            'only replies that ended, raise max_tokens, or train with truncated="mask"'
        )
    if stop:
        for name in ROUTED:
            methods[name] = {"ok": False, "why": "blocked upstream"}
        return Route(
            method=None,
            plan=[],
            methods=methods,
            why="; ".join(stop),
            blocked=stop,
            need=need,
            measured=measured,
            notes=notes,
        )

    floor = bool(n) and all_fail / n >= ROUTE_FLOOR_SHARE
    saturated = mean_pass is not None and mean_pass >= CEILING_PASS_RATE
    band = (
        f"{in_band} in band {lo:g}-{hi:g}, {all_fail} all-fail, {all_pass} all-pass "
        f"of {n} grouped tasks at k={k}"
    )

    # opd
    t_rows = [dict(r) for r in teacher or () if isinstance(r, Mapping)]
    if teacher is None:
        score("opd", False, "no teacher= rows: score a teacher on these tasks at the student's cap")
    else:
        t, s = pass_at(t_rows), pass_at(pool)
        covered = {task_key(r) for r in t_rows} & set(groups)
        t_cut = sum(1 for r in t_rows if is_truncated(r)) / len(t_rows) if t_rows else 0.0
        measured["teacher_truncated"] = round(t_cut, 3)
        if t.pass_at_1 is None or s.pass_at_1 is None:
            score(
                "opd", False, "the teacher rows carry no 0/1 reward: grade them like the student's"
            )
        elif len(covered) < len(groups) / 2:
            score(
                "opd",
                False,
                f"the teacher was scored on {len(covered)} of the student's {len(groups)} tasks: "
                "score it on the same tasks",
            )
        else:
            measured["teacher_gap"] = round(t.pass_at_1 - s.pass_at_1, 3)
            if t.ci95 and s.ci95:
                teach = (
                    f"teacher {t.pass_at_1:.3f} [{t.ci95[0]:.3f}, {t.ci95[1]:.3f}] vs student "
                    f"{s.pass_at_1:.3f} [{s.ci95[0]:.3f}, {s.ci95[1]:.3f}]"
                )
            else:
                teach = f"teacher {t.pass_at_1:.3f} vs student {s.pass_at_1:.3f}"
            if vocab is not None and vocab[0] != vocab[1]:
                score(
                    "opd",
                    False,
                    f"tokenizers differ ({vocab[0]:,} vs {vocab[1]:,}): the per-token signal drops",
                )
            elif not (t.ci95 and s.ci95):
                score("opd", False, f"{teach}: too few tasks for an interval, the eval cannot tell")
            elif t.pass_at_1 - s.pass_at_1 < effect or t.ci95[0] <= s.ci95[1]:
                score("opd", False, f"{teach}: not clear of the student by {effect:g}")
            elif t_cut > ROUTE_TEACHER_MAX_TRUNCATED:
                score(
                    "opd",
                    False,
                    f"{teach}, but {t_cut:.0%} of its replies are cut off: rescore it at the "
                    "student's max_tokens",
                )
            else:
                tok = "same tokenizer" if vocab is not None else "tokenizer not checked (vocab=)"
                score("opd", True, f"{teach}; {t_cut:.0%} truncated; {tok}")

    # grpo
    if not n:
        score("grpo", False, "no task has two rollouts: a group of one has zero advantage")
        need["k"] = RL_ROLLOUTS_PER_PROMPT
    elif on_policy is False:
        score("grpo", False, f"rows sampled by {sorted(writers)}, not {model}: off-policy")
    elif share >= ROUTE_MIN_MIXED_SHARE and in_band >= min_mixed:
        score(
            "grpo",
            True,
            f"{in_band} of {n} tasks in band ({share:.0%} >= {ROUTE_MIN_MIXED_SHARE:.0%}, "
            f">= {min_mixed})",
        )
    elif share >= ROUTE_MIN_MIXED_SHARE:
        score(
            "grpo", False, f"{in_band} tasks in band < {min_mixed} (the share {share:.0%} is fine)"
        )
        need["tasks_total"] = math.ceil(min_mixed / share)
    elif n < ROUTE_MIN_TASKS_FOR_K:
        score(
            "grpo",
            False,
            f"{in_band} of {n} tasks in band ({share:.0%}); under {ROUTE_MIN_TASKS_FOR_K} tasks, "
            "add tasks, not rollouts",
        )
        need["tasks_total"] = ROUTE_MIN_TASKS_FOR_K
    else:
        draws = [(sum(v), len(v)) for v in grouped.values()]
        k_up = next(
            (
                kk
                for kk in range(k + 1, ROUTE_MAX_K + 1)
                if sum(_mixed_chance(s_, d_, kk) for s_, d_ in draws) / n >= ROUTE_MIN_MIXED_SHARE
            ),
            None,
        )
        if k_up is not None and not floor and not saturated:
            need["k"] = k_up
            score(
                "grpo",
                False,
                f"{in_band} of {n} tasks in band ({share:.0%} < {ROUTE_MIN_MIXED_SHARE:.0%}); "
                f"the expected mixed share reaches it at k={k_up}",
            )
        else:
            score(
                "grpo",
                False,
                f"{in_band} of {n} tasks in band ({share:.0%} < {ROUTE_MIN_MIXED_SHARE:.0%}); "
                f"no k up to {ROUTE_MAX_K} reaches it on these rates",
            )

    # opsd
    if not floor:
        score("opsd", False, f"not a floor: {all_fail} of {n} grouped tasks all-fail")
    elif size_b is not None and size_b < ROUTE_OPSD_MIN_PARAMS_B:
        score(
            "opsd",
            False,
            f"{all_fail} of {n} tasks all-fail, but the student is {size_b:g}B; self-distillation "
            f"needs about {ROUTE_OPSD_MIN_PARAMS_B}B to use the hint",
        )
    else:
        score(
            "opsd",
            True,
            f"{all_fail} of {n} tasks all-fail, where best-of-k keeps nothing; "
            'OPSD(privileged="reference"): a floor has no passing demonstration',
        )
        notes.append(
            "OPSD costs points on thinking models (Kaur et al. 2026, arXiv:2607.05184): "
            "train a grpo arm on the same holdout beside it"
        )

    # groupwise
    if not saturated:
        score(
            "groupwise", False, f"not saturated: pass@1 {mean_pass or 0:.2f} < {CEILING_PASS_RATE}"
        )
    elif judge_ok is not True:
        score(
            "groupwise",
            False,
            f"saturated (pass@1 {mean_pass:.2f}), but the grader that ranks passes is unaudited: "
            "pass its compare_judges row as judge=",
        )
    else:
        score(
            "groupwise",
            True,
            f'saturated (pass@1 {mean_pass:.2f}); GroupwiseGrading(mode="reward") (GRS): the '
            "default GAR mode leaves an all-pass group at zero advantage; "
            f"audited grader ({judge_line})",
        )

    # dpo: a pass and a fail of the same task is a pair; dpo is off-policy by construction
    if not n:
        score("dpo", False, "no task has two rollouts, so none has a pass and a fail")
    elif mixed >= min_mixed:
        score("dpo", True, f"{mixed} tasks with a pass and a fail; length-match the pairs")
    else:
        score("dpo", False, f"{mixed} tasks with a pass and a fail < {min_mixed}")

    # the single-rollout family
    if n and not singles:
        score("flashreinforce", False, f"k={k}: groups exist; a group baseline beats a batch one")
    elif logprobs:
        score(
            "flashreinforce",
            True,
            "one rollout per prompt with sampler log-probabilities: FlashReinforce, SAO or BPCO "
            "in your trainer, or prime-rl rae",
        )
    else:
        score("flashreinforce", False, "one rollout per prompt, but no sampler log-probabilities")

    # sft: clone passes; a filter below REJECTION_SAMPLING_MIN_K draws, rejection sampling above
    teacher_passes = sum(1 for r in t_rows if _label(r) == 1)
    kind = "rejection sampling" if k >= REJECTION_SAMPLING_MIN_K else "a pass filter"
    if floor and teacher_passes:
        score("sft", True, f"{teacher_passes} teacher passes to clone on the floored tasks")
    elif floor:
        score("sft", False, f"{all_fail} tasks never pass: clone a stronger model's completions")
        need["teacher_completions"] = all_fail
    elif saturated:
        score(
            "sft", False, f"saturated (pass@1 {mean_pass:.2f}): cloning its own passes adds nothing"
        )
    elif passing_rows:
        score("sft", True, f"{passing_rows} passing rows to clone ({kind} at k={k})")
    else:
        score("sft", False, "no passing rows")

    # what the reward itself says, read with the SDK's own scans
    try:
        hack = hack_scan(pool)
        regime, top = hack.get("regime"), hack.get("top_feature")
    except Exception:  # a scan that cannot run is a note, not a stop
        regime, top = None, None
        notes.append("hack_scan could not run on these rows: the reward was not scanned")
    measured["hack_regime"] = regime
    if regime == "reward_hack":
        for name in ("grpo", "dpo", "groupwise"):
            if methods[name]["ok"]:
                methods[name] = {
                    "ok": False,
                    "why": f"hack_scan: the reward within a task follows {top}, not the "
                    "behavior (Lambert 2025, chapter Over-Optimization; Gao et al. 2022, "
                    "arXiv:2210.10760): fix the reward first",
                }
                blocked.append(f"{name}: {methods[name]['why']}")
    if methods["dpo"]["ok"]:
        text = [
            (task_key(r), _label(r), str(r.get("final_text") or r.get("reply") or "")) for r in pool
        ]
        firsts: dict[tuple[str, int], int] = {}
        for key, lab, body in text:
            if lab is not None and body and (key, lab) not in firsts:
                firsts[(key, lab)] = len(body)
        pairs = [
            (firsts[(key, 1)], firsts[(key, 0)])
            for key, _, _ in text
            if (key, 1) in firsts and (key, 0) in firsts
        ]
        pairs = list(dict.fromkeys(pairs))
        longer = sum(1 for a, b in pairs if a > b)
        warn = length_confound_warning(longer, len(pairs)) if pairs else None
        if warn:
            methods["dpo"]["why"] += f"; {warn}"
    if methods["grpo"]["ok"]:
        spread = length_report(pool).get("n_groups_wide_spread") or 0
        if spread:
            methods["grpo"]["why"] += (
                f'; {spread} groups have reply lengths far apart: loss_type="dr_grpo" drops the '
                "length bias of a per-sequence mean (Liu et al. 2025, arXiv:2503.20783)"
            )
    if n and k < DIFFICULTY_BAND_ROLLOUTS:
        notes.append(
            f"the band is read at k={k}; it is measured at {DIFFICULTY_BAND_ROLLOUTS} rollouts "
            "per task (Lambert 2025, chapter Reasoning), so a task's place in it is noisy"
        )
    if truncated_fails:
        notes.append(
            f"{truncated_fails} failing rows did not end on their own: some all-fail tasks may be "
            "the token cap, not the model (chapter Reinforcement Learning)"
        )
    if "k" in need and n:
        notes.append(
            "or sample adaptively instead of raising k everywhere: wai.methods.ReinforceAda "
            "draws until each task has a pass and a fail (arXiv:2510.04996)"
        )
    graders = {str(r.get("judge_name")) for r in pool if r.get("judge_name")}
    if graders & writers:
        notes.append(
            f"the grader {sorted(graders & writers)} also wrote the rows: models favor their own "
            "outputs (self-preference, Panickssery et al. 2024; Lambert 2025, chapter "
            "Synthetic Data and Distillation); grade with another family"
        )
    chosen = next((name for name in ROUTED if methods[name]["ok"]), None)
    plan = [chosen] if chosen else []
    if chosen in ("opd", "grpo"):
        plan += [name for name in ("opsd", "groupwise") if methods[name]["ok"]]
    if chosen == "grpo" and model and "qwen" in model.lower():
        notes.append(
            "a Qwen base: add a random-reward grpo arm; gains under random rewards signal "
            "contamination (Lambert 2025, chapter Evaluation)"
        )
    why = (
        f"{methods[chosen]['why']} ({band})"
        if chosen
        else f"no method can run on this pool ({band})"
    )
    return Route(
        method=chosen,
        plan=plan,
        methods={name: methods[name] for name in ROUTED},
        why=why,
        blocked=[b for b in blocked if not (chosen and b.startswith(f"{chosen}:"))],
        need=need,
        measured=measured,
        notes=notes,
    )


__all__ = ["BEFORE_THE_GPU", "NOT_ROUTED", "ROUTED", "Route", "route"]
