"""Groupwise agentic grading: a grader tells passing rollouts apart.

    import whileai as wai

    gar = wai.GroupwiseGrading(grader=rank_group)           # GAR, the default mode
    print(gar.check_spread(scored.rows))                    # refuse to train on a grader with no spread
    trainer = GRPOTrainer(reward_funcs=[gar.trl_reward(verifier, num_generations=8)], ...)

A binary test reward makes every passing rollout equal, so RL learns to
pass by any means: longer replies, swallowed exceptions, an invented id,
an answer leaked from the environment. MiMo-V2.6 (Xiaomi 2026, technical
report section 4.3, "Groupwise Agentic Grading") adds a grader that reads
a whole group of rollouts for one task and ranks the passes, in one of two
places:

* ``mode="reward"``, Groupwise Reward Synthesis (section 4.3.1). Offline, a
  grader reads a group of rollouts plus the task and writes task-specific
  rubrics (solution criteria and behavior criteria). Online, each passing
  rollout is scored against them and the reward is the product
  ``R_i = R_test_i * S_sol_i * S_beh_i`` (equation 2). Failed rollouts keep
  zero.
* ``mode="advantage"``, Groupwise Advantage Redistribution (section 4.3.2).
  Online, for each mixed-outcome group the grader inspects every rollout,
  ranks the passes on approach, precision, minimality, side effects and
  craftsmanship, and zeroes a confirmed hack. The ranking becomes quality
  factors ``f_i`` in (0, 1] and the positive advantage moves from lower- to
  higher-quality passes with its total conserved (equation 3, in
  :func:`redistribute`). Unusable grader output falls back to the original
  advantages.

Neither mode is a loss. The object carries the grader and the report's
knobs, :meth:`GroupwiseGrading.trl_reward` hands TRL's ``GRPOTrainer`` a
reward function whose group normalization reproduces the redistributed
advantages, and :meth:`GroupwiseGrading.check_spread` says, before the GPU
is spent, whether the grader can tell the passes apart at all. That check
is the lesson of 2026-09-21: on single-turn text-to-SQL the rubric grader
gave 0.93 to nearly every pass, the multiplier was a constant, and GRPO's
group normalization erased it. The method belongs on multi-turn agent
tasks where passes differ (an extra tool call, a skipped verification, an
invented id), not on one-line answers.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .report import Report
from .simulations.defaults import (
    DEFAULT_LLM_JUDGE_CONCURRENCY,
    GROUPWISE_CAP,
    GROUPWISE_FLOOR,
    GROUPWISE_MIN_FACTOR,
    GROUPWISE_MODE,
    PASS_THRESHOLD,
    SPREAD_BAND,
    SPREAD_CLUSTER_SHARE,
    SPREAD_MIN_STD,
)

MODES = ("reward", "advantage")
REPORT = "MiMo-V2.6 technical report (Xiaomi, 2026-09-21), section 4.3"

# --------------------------------------------------------------------------
# the math, pure
# --------------------------------------------------------------------------


def redistribute(
    rewards: Sequence[float],
    factors: Mapping[int, float] | Sequence[float],
    cap: float = GROUPWISE_CAP,
    pass_threshold: float = PASS_THRESHOLD,
) -> list[float]:
    """Equation 3 of section 4.3.2: move positive advantage toward better passes.

    ``rewards`` are one group's effective rewards after any hack was zeroed;
    a pass is a reward at or above ``pass_threshold``. ``factors`` gives
    each pass its quality ``f_i`` in (0, 1] (a mapping from index, or a
    sequence aligned with ``rewards``; entries for failures are ignored).
    With ``A_i = R_i - mean(R)`` and ``P`` the passes,
    ``lambda = sum_P A_j / sum_P f_j A_j`` (capped at ``cap``,
    ``GROUPWISE_CAP``), ``A'_i = lambda f_i A_i`` for a pass and ``A_i``
    otherwise, then the group mean is subtracted again so the result has
    zero mean. Uncapped, the passes' total advantage is conserved and the
    failures are untouched. A group with no pass, no failure, or no
    positive mass to move returns the plain advantages.
    """
    r = [float(x) for x in rewards]
    n = len(r)
    if n == 0:
        return []
    if float(cap) < 1:
        raise ValueError(f"cap must be at least 1 (GROUPWISE_CAP is {GROUPWISE_CAP}); got {cap}")
    mean = sum(r) / n
    adv = [x - mean for x in r]
    passing = [i for i, x in enumerate(r) if x >= pass_threshold]
    if not passing or len(passing) == n:
        return adv
    f: dict[int, float] = {}
    for i in passing:
        try:
            value = float(factors[i])
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError(f"factors has no quality factor for passing index {i}") from None
        if not 0 < value <= 1:
            raise ValueError(f"quality factors are in (0, 1]; index {i} has {value}")
        f[i] = value
    mass = sum(adv[i] for i in passing)
    weighted = sum(f[i] * adv[i] for i in passing)
    if mass <= 0 or weighted <= 0:
        return adv
    lam = min(mass / weighted, float(cap))
    new = [lam * f[i] * adv[i] if i in f else adv[i] for i in range(n)]
    shift = sum(new) / n
    return [x - shift for x in new]


def factors_from_ranking(
    ranking: Sequence[int | Sequence[int]],
    min_factor: float = GROUPWISE_MIN_FACTOR,
) -> dict[int, float]:
    """Quality factors from a best-first ranking of passing indices.

    An inner sequence is a tie ("ties when differences are inconclusive",
    section 4.3.2). The map is linear from 1.0 for the best to
    ``min_factor`` (``GROUPWISE_MIN_FACTOR``) for the worst, a tie taking
    the mean of the ranks it spans; the report leaves the map unpublished.
    """
    if not 0 < float(min_factor) <= 1:
        raise ValueError(f"min_factor is in (0, 1]; got {min_factor}")
    tiers = [[int(x)] if isinstance(x, int) else [int(i) for i in x] for x in ranking]
    n = sum(len(t) for t in tiers)
    out: dict[int, float] = {}
    if n == 0:
        return out
    position = 0
    for tier in tiers:
        mean_rank = position + (len(tier) - 1) / 2
        factor = 1.0 if n == 1 else 1.0 - (1.0 - float(min_factor)) * mean_rank / (n - 1)
        for i in tier:
            out[i] = factor
        position += len(tier)
    return out


# --------------------------------------------------------------------------
# the spread check
# --------------------------------------------------------------------------


class SpreadReport(Report):
    """Whether a grader's scores on passing rollouts vary enough to train on."""

    _summary_keys = ("ok", "n", "std")

    def __str__(self) -> str:
        n = self["n"]
        if n < 2:  # noqa: PLR2004  # a spread needs two scores
            return (
                f"spread check: {n} score(s); need at least two passes in one group to say anything"
            )
        head = (
            f"spread check: {n} grader scores over {self['groups']} group(s), std {self['std']:.3f}, "
            f"{self['within_band'] * 100:.0f}% within {self['band']} of the median {self['median']:.2f}"
        )
        if self["ok"]:
            return head + "\nspread: yes. The grader tells these passes apart."
        return (
            head
            + f"\nno spread (std under {self['min_std']} or over {self['cluster_share'] * 100:.0f}% "
            "clustered): a constant multiplier is erased by GRPO's group normalization, and the run "
            "would be the plain-reward run at the grader's price (text-to-SQL, 2026-09-21). Fix: a "
            "grader that ranks the group jointly (mode='advantage'), an environment where passes "
            "differ (several tool calls, a program verifier), or a stricter rubric."
        )


def spread(
    scores: Sequence[float],
    *,
    groups: int = 1,
    min_std: float = SPREAD_MIN_STD,
    band: float = SPREAD_BAND,
    cluster_share: float = SPREAD_CLUSTER_SHARE,
) -> SpreadReport:
    """Do these grader scores vary? The math behind :meth:`GroupwiseGrading.check_spread`.

    No spread means a standard deviation under ``min_std``
    (``SPREAD_MIN_STD``) or more than ``cluster_share``
    (``SPREAD_CLUSTER_SHARE``) of the scores within ``band``
    (``SPREAD_BAND``) of the median. Both numbers come from the
    2026-09-21 text-to-SQL null result (mean 0.93 on 1,472 passes).
    """
    values = [float(s) for s in scores]
    n = len(values)
    if n < 2:  # noqa: PLR2004  # a spread needs two scores
        return SpreadReport(
            ok=False,
            n=n,
            groups=groups,
            std=0.0,
            median=values[0] if values else float("nan"),
            within_band=1.0,
            band=band,
            min_std=min_std,
            cluster_share=cluster_share,
            scores=values,
        )
    std = statistics.pstdev(values)
    median = statistics.median(values)
    within = sum(1 for v in values if abs(v - median) <= band) / n
    ok = std >= min_std and within <= cluster_share
    return SpreadReport(
        ok=ok,
        n=n,
        groups=groups,
        std=std,
        median=median,
        within_band=within,
        band=band,
        min_std=min_std,
        cluster_share=cluster_share,
        scores=values,
    )


# --------------------------------------------------------------------------
# the method object
# --------------------------------------------------------------------------


class GroupwiseStats(Report):
    """What the grader did: groups seen, calls, failures, hacks, factor spread."""

    _summary_keys = ("groups", "grader_calls", "grader_failures", "hacks")

    def __str__(self) -> str:
        factors = self.get("factors") or []
        line = (
            f"groupwise grading ({self['mode']}): {self['groups']} groups, {self['mixed']} mixed, "
            f"{self['grader_calls']} grader calls, {self['grader_failures']} failed (fell back), "
            f"{self['hacks']} hacks zeroed"
        )
        if len(factors) >= 2:  # noqa: PLR2004  # a spread needs two scores
            line += f"; factors mean {statistics.fmean(factors):.3f} std {statistics.pstdev(factors):.3f}"
        if self.get("lambda_capped"):
            line += f"; lambda capped {self['lambda_capped']} times"
        return line


def _passed(reward: Any, threshold: float) -> bool:
    try:
        return float(reward) >= threshold
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class GroupwiseGrading:
    """A grader that tells passing rollouts apart, in the reward or in the advantage.

    ``grader`` is your judge call. In ``mode="advantage"`` (GAR, section
    4.3.2, the default: ``GROUPWISE_MODE``; the report runs it on all code
    agent tasks not given rubrics) it receives one group as a list of
    ``{"index", "prompt", "completion", "reward", "passed"}`` and returns
    either ``{"ranking": [best, ..., worst]}`` over the passing indices
    (an inner list is a tie) or ``{"factors": {index: f}}`` with ``f`` in
    (0, 1], plus an optional ``"hacks": [index, ...]`` for passes that
    depended on an invented or leaked answer; with ``hack_zero`` those
    become failures before the group statistics are recomputed. The
    ranking maps to factors linearly down to ``min_factor``
    (``GROUPWISE_MIN_FACTOR``), and ``cap`` (``GROUPWISE_CAP``) bounds the
    common factor lambda that restores the moved mass. In
    ``mode="reward"`` (GRS, section 4.3.1) the grader receives one passing
    rollout and its rubric and returns ``{"solution": s, "behavior": b}``
    in [0, 1]; the reward becomes ``R_test * max(floor, s) * max(floor, b)``
    (equation 2 multiplies the raw scores: ``floor`` is 0, ``GROUPWISE_FLOOR``).
    ``rubrics`` is a mapping from task id to rubric, or a callable given
    the first group seen for a task that writes one and is cached.

    A grader that raises or returns something unusable falls back to the
    original rewards for that group, as the report does, and is counted
    in ``stats``. Run ``check_spread`` on a sample of passes first: a
    grader whose scores do not vary is a constant GRPO normalizes away.
    """

    grader: Callable[..., Any]
    mode: str = GROUPWISE_MODE
    rubrics: Mapping[Any, Any] | Callable[..., Any] | None = None
    floor: float = GROUPWISE_FLOOR
    cap: float = GROUPWISE_CAP
    min_factor: float = GROUPWISE_MIN_FACTOR
    hack_zero: bool = True
    pass_threshold: float = PASS_THRESHOLD
    concurrency: int = DEFAULT_LLM_JUDGE_CONCURRENCY
    _state: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    name: ClassVar[str] = "groupwise"

    def __post_init__(self) -> None:
        if not callable(self.grader):
            raise TypeError(f"grader must be callable; got {type(self.grader).__name__}")
        mode = str(self.mode).lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}; got {self.mode!r}")
        object.__setattr__(self, "mode", mode)
        if not 0 <= float(self.floor) < 1:
            raise ValueError(
                f"floor is in [0, 1) (GROUPWISE_FLOOR is {GROUPWISE_FLOOR}); got {self.floor}"
            )
        if float(self.cap) < 1:
            raise ValueError(
                f"cap must be at least 1 (GROUPWISE_CAP is {GROUPWISE_CAP}); got {self.cap}"
            )
        if not 0 < float(self.min_factor) <= 1:
            raise ValueError(
                f"min_factor is in (0, 1] (GROUPWISE_MIN_FACTOR is {GROUPWISE_MIN_FACTOR}); "
                f"got {self.min_factor}"
            )
        if int(self.concurrency) < 1:
            raise ValueError(f"concurrency must be at least 1; got {self.concurrency}")
        if self.rubrics is not None and not (
            callable(self.rubrics) or isinstance(self.rubrics, Mapping)
        ):
            raise TypeError(
                "rubrics is a mapping from task id to rubric, or a callable that writes one"
            )
        self._state.update(
            groups=0,
            mixed=0,
            grader_calls=0,
            grader_failures=0,
            hacks=0,
            lambda_capped=0,
            factors=[],
            scores=[],
            rubric_cache={},
        )

    def __str__(self) -> str:
        knobs = (
            f"floor={self.floor}"
            if self.mode == "reward"
            else f"cap={self.cap}, min_factor={self.min_factor}, hack_zero={self.hack_zero}"
        )
        return f"GroupwiseGrading(mode={self.mode}, grader={_name(self.grader)}, {knobs})"

    @property
    def stats(self) -> GroupwiseStats:
        """What the grader has done so far across every group."""
        return GroupwiseStats(
            mode=self.mode, **{k: v for k, v in self._state.items() if k != "rubric_cache"}
        )

    # -- one group -------------------------------------------------------

    def group(self, items: Sequence[Mapping[str, Any]]) -> list[float]:
        """Shape one group's rewards. ``items`` carry ``prompt``, ``completion``,
        ``reward`` and, for reward mode, ``task_id``. Returns the training
        rewards: in reward mode the products of equation 2; in advantage mode
        the redistributed advantages plus the group's mean effective reward,
        so a trainer that subtracts the group mean recovers equation 3."""
        rewards = [float(it.get("reward") or 0.0) for it in items]
        passed = [_passed(r, self.pass_threshold) for r in rewards]
        self._state["groups"] += 1
        if any(passed) and not all(passed):
            self._state["mixed"] += 1
        if self.mode == "reward":
            return self._reward_mode(items, rewards, passed)
        return self._advantage_mode(items, rewards, passed)

    def _reward_mode(
        self, items: Sequence[Mapping[str, Any]], rewards: list[float], passed: list[bool]
    ) -> list[float]:
        out = list(rewards)
        rubric = self._rubric_for(items)
        for i, item in enumerate(items):
            if not passed[i]:
                continue
            self._state["grader_calls"] += 1
            try:
                verdict = self.grader(dict(item), rubric)
                s_sol = float(verdict["solution"])
                s_beh = float(verdict["behavior"])
                if not (0 <= s_sol <= 1 and 0 <= s_beh <= 1):
                    raise ValueError("scores outside [0, 1]")
            except Exception:  # unusable grader output falls back to R_test
                self._state["grader_failures"] += 1
                continue
            out[i] = rewards[i] * max(self.floor, s_sol) * max(self.floor, s_beh)
            self._state["scores"].append(s_sol * s_beh)
        return out

    def _rubric_for(self, items: Sequence[Mapping[str, Any]]) -> Any:
        if self.rubrics is None:
            return None
        task_id = next((it.get("task_id") for it in items if it.get("task_id") is not None), None)
        if isinstance(self.rubrics, Mapping):
            return self.rubrics.get(task_id)
        cache = self._state["rubric_cache"]
        if task_id in cache:
            return cache[task_id]
        try:
            rubric = self.rubrics([dict(it) for it in items])
        except Exception:  # no rubric: the grader sees None and falls back
            rubric = None
        cache[task_id] = rubric
        return rubric

    def _advantage_mode(
        self, items: Sequence[Mapping[str, Any]], rewards: list[float], passed: list[bool]
    ) -> list[float]:
        plain = list(rewards)  # the trainer's own group normalization does the rest
        if not any(passed) or all(passed):
            return plain
        group = [
            {"index": i, **dict(it), "reward": rewards[i], "passed": passed[i]}
            for i, it in enumerate(items)
        ]
        self._state["grader_calls"] += 1
        try:
            verdict = self.grader(group)
            factors, hacks = self._factors(verdict, [i for i, p in enumerate(passed) if p])
        except Exception:  # unusable grader output falls back to the original advantages
            self._state["grader_failures"] += 1
            return plain
        effective = list(rewards)
        if self.hack_zero and hacks:
            for i in hacks:
                effective[i] = 0.0
            self._state["hacks"] += len(hacks)
        still_passing = [i for i, r in enumerate(effective) if _passed(r, self.pass_threshold)]
        self._state["factors"].extend(factors[i] for i in still_passing)
        if not still_passing:
            return effective
        eff_mean = sum(effective) / len(effective)
        adv = [r - eff_mean for r in effective]
        mass = sum(adv[i] for i in still_passing)
        weighted = sum(factors[i] * adv[i] for i in still_passing)
        if weighted > 0 and mass / weighted > self.cap:
            self._state["lambda_capped"] += 1
        new = redistribute(effective, factors, cap=self.cap, pass_threshold=self.pass_threshold)
        return [a + eff_mean for a in new]

    def _factors(self, verdict: Any, passing: list[int]) -> tuple[dict[int, float], list[int]]:
        if not isinstance(verdict, Mapping):
            raise TypeError("grader must return a mapping")
        hacks = [int(i) for i in (verdict.get("hacks") or []) if int(i) in passing]
        if "factors" in verdict:
            raw = verdict["factors"]
            factors = {
                int(k): float(v)
                for k, v in (raw.items() if isinstance(raw, Mapping) else enumerate(raw))
            }
        elif "ranking" in verdict:
            factors = factors_from_ranking(verdict["ranking"], min_factor=self.min_factor)
        else:
            raise ValueError("grader must return 'ranking' or 'factors'")
        for i in passing:
            if i in hacks:
                # a hack that is not zeroed (hack_zero=False) is the worst pass
                factors.setdefault(i, self.min_factor)
                continue
            if i not in factors:
                raise ValueError(f"grader gave no factor for passing index {i}")
            if not 0 < factors[i] <= 1:
                raise ValueError(f"factor for index {i} is {factors[i]}, not in (0, 1]")
        return factors, hacks

    # -- a batch of groups -------------------------------------------------

    def shape(
        self,
        rewards: Sequence[float],
        prompts: Sequence[Any],
        completions: Sequence[Any],
        *,
        group_size: int,
        task_id: Sequence[Any] | None = None,
    ) -> list[float]:
        """Shape a generation batch laid out as consecutive groups of ``group_size``
        (how TRL hands a reward function its completions). Groups are graded in
        parallel, ``concurrency`` at a time."""
        n = len(rewards)
        if group_size < 1 or n % group_size:
            raise ValueError(f"{n} rewards do not split into groups of {group_size}")
        if len(prompts) != n or len(completions) != n:
            raise ValueError("rewards, prompts and completions must be the same length")
        ids = list(task_id) if task_id is not None else [None] * n
        groups = []
        for start in range(0, n, group_size):
            stop = start + group_size
            groups.append(
                [
                    {
                        "prompt": prompts[i],
                        "completion": completions[i],
                        "reward": rewards[i],
                        "task_id": ids[i],
                    }
                    for i in range(start, stop)
                ]
            )
        if len(groups) == 1 or self.concurrency == 1:
            shaped = [self.group(g) for g in groups]
        else:
            with ThreadPoolExecutor(max_workers=min(int(self.concurrency), len(groups))) as pool:
                shaped = list(pool.map(self.group, groups))
        return [x for g in shaped for x in g]

    def trl_reward(
        self, base: Callable[..., Sequence[float]], num_generations: int
    ) -> Callable[..., list[float]]:
        """A TRL ``GRPOTrainer`` reward function: ``base`` scored, then shaped.

        TRL calls a reward function with ``prompts=``, ``completions=`` and
        every other dataset column, on a batch of ``num_generations``
        consecutive completions per prompt, then forms the advantage as
        ``(r - mean_group) / std_group``. In advantage mode this returns
        ``A'_i + mean(R)``, so the mean subtraction gives equation 3 exactly;
        the division by the group's standard deviation is TRL's, and it is
        the limit: it rescales the redistributed group by its new spread, so
        the ranking and the relative weights hold but the total moved mass
        matches the report only under ``scale_rewards="none"`` (Dr. GRPO,
        Liu et al. 2025, arXiv:2503.20783). A task-id column named
        ``task_id`` reaches the rubric lookup in reward mode.
        """

        def reward_func(
            prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any
        ) -> list[float]:
            scores = [float(x) for x in base(prompts=prompts, completions=completions, **kwargs)]
            return self.shape(
                scores,
                prompts,
                completions,
                group_size=num_generations,
                task_id=kwargs.get("task_id"),
            )

        reward_func.__name__ = f"{_name(base)}_groupwise"
        return reward_func

    # -- before the GPU ----------------------------------------------------

    def check_spread(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        strict: bool = True,
        min_std: float = SPREAD_MIN_STD,
        band: float = SPREAD_BAND,
        cluster_share: float = SPREAD_CLUSTER_SHARE,
    ) -> SpreadReport:
        """Score a sample of passes with the grader and refuse a grader with no spread.

        ``rows`` are graded rollouts (``scenario_id`` or ``prompt`` names the
        group, ``reward`` the verdict, ``final_text`` or ``completion`` the
        reply). Groups with at least two passes are graded the way training
        would grade them; the factors (advantage mode) or rubric products
        (reward mode) of the passes are the scores. With ``strict`` a
        report of no spread is a ``ValueError`` that says what to change;
        otherwise the report is returned and prints its verdict.
        """
        by_group: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            key = row.get("scenario_id") or row.get("task_id") or row.get("prompt")
            by_group.setdefault(key, []).append(
                {
                    "prompt": row.get("prompt"),
                    "completion": row.get("completion", row.get("final_text")),
                    "reward": row.get("reward"),
                    "task_id": row.get("task_id", row.get("scenario_id")),
                }
            )
        scores: list[float] = []
        groups = 0
        for items in by_group.values():
            passing = [
                i for i, it in enumerate(items) if _passed(it["reward"], self.pass_threshold)
            ]
            if len(passing) < 2:  # noqa: PLR2004  # a spread needs two passes to compare
                continue
            groups += 1
            self._state["groups"] += 1
            if len(passing) < len(items):
                self._state["mixed"] += 1
            if self.mode == "reward":
                before = len(self._state["scores"])
                self._reward_mode(
                    items,
                    [float(it["reward"] or 0.0) for it in items],
                    [i in passing for i in range(len(items))],
                )
                scores.extend(self._state["scores"][before:])
                continue
            # advantage mode: grade every group with two passes, mixed or not; the
            # check is about the grader, and training will only see the mixed ones
            group = [{"index": i, **it, "passed": i in passing} for i, it in enumerate(items)]
            self._state["grader_calls"] += 1
            try:
                factors, hacks = self._factors(self.grader(group), passing)
            except Exception:  # counted; a failing grader has no spread to report
                self._state["grader_failures"] += 1
                continue
            self._state["hacks"] += len(hacks)
            kept = [factors[i] for i in passing if i not in hacks]
            self._state["factors"].extend(kept)
            scores.extend(kept)
        report = spread(
            scores, groups=groups, min_std=min_std, band=band, cluster_share=cluster_share
        )
        if strict and not report["ok"]:
            raise ValueError(str(report))
        return report


def _name(fn: Any) -> str:
    return getattr(fn, "__name__", None) or type(fn).__name__


__all__ = [
    "GroupwiseGrading",
    "GroupwiseStats",
    "SpreadReport",
    "factors_from_ranking",
    "redistribute",
    "spread",
]
