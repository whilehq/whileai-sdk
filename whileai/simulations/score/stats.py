"""Confidence intervals, paired run comparison, and decontamination.

The numbers an engineer needs before believing a result (Lambert 2025, chapter
Evaluation: labs win by raising the statistical power of the few evaluations
they track; contamination is found by n-gram overlap between training prompts
and evaluation prompts, 8-gram in the Tulu 3 decontamination).

* Every score gets an interval. Per-task pass rates and per-rollout marker
  values are clustered by task, so the bootstrap resamples tasks, not
  rows: rollouts of one ask are not independent draws.
* Two runs are compared on the tasks they share, as paired differences,
  with a bootstrap interval and a sign-flip permutation p-value. Unpaired
  comparison is the fallback and is labeled as such.
* Decontamination is word n-gram overlap (default 8) between a dataset's
  prompts and replies and the evaluation prompts it must not have seen.
  Short prompts fall back to exact normalized match.

Everything here is stdlib and deterministic under ``seed``.

Statistical knobs (documented once, here; every function below takes
them as keywords and ``whileai.simulations.defaults`` holds the values):

* ``alpha`` (``ALPHA``, 0.05): the two-sided false-positive rate behind a
  verdict. ``holdout_size`` and ``detectable_effect`` size for it;
  ``delta_report`` reads its family-wise rate from it.
* ``level`` (``CI_LEVEL``, ``1 - ALPHA``): the interval every ``ci95``
  key carries. ``bootstrap_ci``, ``compare_runs`` and ``noise_band`` take
  it; the key name stays ``ci95`` and ``level`` is reported beside it.
  The normal quantile at 0.95 is ``Z_95`` (1.96, rounded as the tables
  print it); other levels come from ``NormalDist``, and the t quantile
  from the table below at 0.95 or a numeric inversion elsewhere.
* ``power`` (``POWER``, 0.8): the chance a holdout of the size
  ``holdout_size`` names detects a real gain (Miller 2024,
  arXiv:2411.00640, section 5: ``n = ((z_{1-alpha/2} + z_power) * sd /
  effect) ** 2``).
* ``n_boot`` (``BOOTSTRAP_DRAWS``, 2000): resamples behind a percentile
  interval; Efron and Tibshirani put the floor at 1000.
* ``MIN_CI_TASKS`` (3) and ``MIN_RERUNS`` (3): the fewest tasks an
  interval, and the fewest re-runs a spread, can be read from.
"""

from __future__ import annotations

import math
import random
import re
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ...report import Report
from ..defaults import (
    ALPHA,
    BASE_PASS_RATE,
    BOOTSTRAP_DRAWS,
    CEILING_PASS_RATE,
    CI_LEVEL,
    DECONTAM_NGRAM,
    DECONTAM_OVERLAP,
    DIFFICULTY_BAND,
    DIFFICULTY_BAND_ROLLOUTS,
    MIN_CI_TASKS,
    MIN_RERUNS,
    POWER,
    PROVE_EFFECT,
    ROLLOUTS_PER_TASK,
    SEMANTIC_SIMILARITY,
    Z_95,
)

#: the bootstrap draw count every interval here defaults to (``defaults.BOOTSTRAP_DRAWS``)
DEFAULT_BOOT = BOOTSTRAP_DRAWS
# MIN_HOLDOUT_TASKS = 2: the fewest tasks ``holdout_size`` will ever
# answer; one task is not a paired comparison (convention).
MIN_HOLDOUT_TASKS = 2
# FIXED_POINT_STEPS = 12: iterations ``detectable_effect`` runs to solve
# the sizing for the effect, whose after-side variance depends on it. The
# map is a contraction and converges to four decimals in under ten steps
# on every base rate tried (measured, not derived).
FIXED_POINT_STEPS = 12
# MIN_PAIRED_TASKS = 5: shared tasks ``compare_runs`` needs before it
# pairs; under it the sign-flip test has at most 2**4 = 16 arrangements,
# so the smallest p-value it can reach is about 0.06 and no paired
# verdict at ALPHA is possible. Above MIN_CI_TASKS for that reason.
MIN_PAIRED_TASKS = 5
# UNPAIRED_MAJORITY_SHARE = 0.5: below this paired share ``compare_runs``
# says "most tasks unpaired"; it is the meaning of "most", not a knob.
UNPAIRED_MAJORITY_SHARE = 0.5
# DISTINCT_TASK_PERCENTILE = 0.99: the percentile of cosine similarity
# over eval-prompt pairs with different task ids that ``decontaminate``
# reports as "how alike distinct tasks read". The top 1% is where two
# tasks that only share a domain sit closest (convention, untested).
DISTINCT_TASK_PERCENTILE = 0.99
_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _words(text: Any) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ------------------------------------------------------------------ intervals


def wilson_interval(successes: int, n: int, *, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a proportion. ``None`` when n is 0.
    ``z`` is the normal quantile of the level wanted (``Z_95`` for 95%)."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _z(p: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def _z_level(level: float) -> float:
    """The two-sided normal quantile at ``level``: ``Z_95`` (1.96, as the
    tables print it) at the default level, ``NormalDist`` elsewhere."""
    if not 0 < level < 1:
        raise ValueError("level is the interval's coverage, strictly between 0 and 1")
    return Z_95 if level == CI_LEVEL else _z((1 + level) / 2)


#: Two-sided 95% quantiles of Student's t by degrees of freedom, for a
#: ``run_std`` estimated from a handful of re-runs (three runs per side is
#: df=4 and 2.78, not 1.96). Past 30 the Cornish-Fisher expansion below is
#: within 0.001 of the table.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042,
}  # fmt: skip


#: Lentz's continued fraction guards: a floor that keeps a near-zero
#: denominator finite, and the step size at which the fraction has
#: converged to double precision (Numerical Recipes 6.4).
_BETACF_TINY = 1e-300
_BETACF_EPS = 3e-16
#: A within-minus-between variance under this is zero at double precision.
_VARIANCE_EPS = 1e-12
#: The sign-flip null: each paired difference keeps or flips its sign on a
#: fair coin, one half by definition.
_FAIR_COIN = 0.5


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's
    method, as in Numerical Recipes 6.4)."""
    tiny = _BETACF_TINY
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        step = d * c
        h *= step
        if abs(step - 1.0) < _BETACF_EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``, stdlib only."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: int) -> float:
    """Student's t distribution function at ``t`` with ``df`` degrees of freedom."""
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


# _T_BRACKET_HI = 1000: the upper end of the bisection bracket for a t
# quantile (no level anyone asks for needs a larger t at df >= 1);
# _T_TOLERANCE = 1e-10: the bracket width at which the inversion stops.
_T_BRACKET_HI = 1000.0
_T_TOLERANCE = 1e-10


def _t_quantile(df: int, level: float = CI_LEVEL) -> float:
    """The two-sided t quantile at ``df`` degrees of freedom and ``level``:
    the table to 30 at the default level, the Cornish-Fisher expansion in
    ``z`` past it (within 0.001 of the table there), and a numeric
    inversion of the t distribution at any other level (no scipy)."""
    n = max(1, int(df))
    if level == CI_LEVEL:
        if n in _T975:
            return _T975[n]
        z = Z_95
        return z + (z**3 + z) / (4 * n) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * n * n)
    target = (1 + level) / 2
    lo, hi = 0.0, _T_BRACKET_HI
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_cdf(mid, n) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < _T_TOLERANCE:
            break
    return (lo + hi) / 2


def _t975(df: int) -> float:
    """The two-sided 95% t quantile at ``df`` (``_t_quantile`` at ``CI_LEVEL``)."""
    return _t_quantile(df, CI_LEVEL)


def noise_band(
    run_std: float,
    n_a: int = 1,
    n_b: int = 1,
    df: int | None = None,
    *,
    level: float = CI_LEVEL,
) -> float:
    """The re-run band a before/after delta has to clear (Lambert 2025,
    chapter Evaluation and its evaluation-variance appendix).

    ``run_std`` is the standard deviation of ONE run's mean when the same
    model is evaluated again. A delta is the mean of ``n_a`` before runs
    against the mean of ``n_b`` after runs, so its own standard deviation
    is ``run_std * sqrt(1/n_a + 1/n_b)``: ``sqrt(2)`` times ``run_std``
    with one run per side, ``sqrt(2/3)`` times it with three. The band is
    that times the normal quantile at ``level`` (1.96 at the default 95%)
    when ``run_std`` is taken as the eval's true spread (``df=None``: a
    number handed in), or times the two-sided t quantile at ``df`` when
    ``run_std`` was estimated from the re-runs themselves, with ``df =
    sum(n_i - 1)`` over the sides (three runs per side is df=4 and 2.78).
    Under pure noise about ``1 - level`` of deltas land outside it on
    either path; a flat ``2 * run_std`` let 15% through with one run per
    side, and ``2 * sqrt(2) * run_std`` was right only there and too wide
    with three.
    """
    if n_a < 1 or n_b < 1:
        raise ValueError("n_a and n_b are run counts, at least 1 each")
    q = _z_level(level) if df is None else _t_quantile(df, level)
    return q * float(run_std) * math.sqrt(1.0 / n_a + 1.0 / n_b)


def _paired_task_sd(base: float, effect: float, k: int) -> float:
    """Standard deviation of one task's paired difference (after minus
    before pass rate over ``k`` rollouts each side) when the gain lands
    uniformly: before at ``base``, after at ``base + effect``, and the two
    arms independent draws (``Var(A) + Var(B)``, no covariance term)."""
    p = min(1.0, max(0.0, float(base)))
    q = min(1.0, max(0.0, p + float(effect)))
    kk = max(1, int(k))
    return math.sqrt((p * (1 - p) + q * (1 - q)) / kk)


def _concentrated_task_sd(base: float, effect: float, k: int) -> float:
    """The same standard deviation when the gain is carried by the fewest
    tasks that can carry it: a share ``effect / (1 - base)`` of tasks go
    from ``base`` to 1 and the rest do not move. Most tasks are then
    ties and the paired differences spread far wider than the uniform
    model says (#292: 0 -> 0.127 at k=4, model 0.168, measured 0.333)."""
    p = min(1.0, max(0.0, float(base)))
    gain = min(1.0 - p, max(0.0, float(effect)))
    kk = max(1, int(k))
    if gain <= 0 or p >= 1:
        return _paired_task_sd(p, gain, kk)
    share = gain / (1 - p)
    within = ((1 - share) * 2 * p * (1 - p) + share * p * (1 - p)) / kk
    between = gain * gain * (1 - share) / share
    return math.sqrt(within + between)


def _sample_sd(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _rows_base_and_k(rows: Sequence[dict]) -> tuple[float, int, float | None, float | None]:
    """Mean per-task pass rate, the smallest rollouts-per-task, the spread
    (sample sd) of per-task pass rates on graded rows, and the ratio by
    which the independent-arms model overstates the paired variance at
    that spread: what ``delta_report`` would pair on."""
    groups: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _binary(row)
        if value is None:
            continue
        groups.setdefault(task_key(row), []).append(value)
    if not groups:
        raise ValueError("rows carry no 0/1 rewards; grade them first, or pass base= and k=")
    rates = [_mean(v) for v in groups.values()]
    base = _mean(rates)
    k = min(len(v) for v in groups.values())
    spread = _sample_sd(rates) if len(rates) >= 2 else None  # noqa: PLR2004  # a spread needs a pair
    return base, k, spread, _independence_ratio(rates)


def _independence_ratio(rates: Sequence[float]) -> float | None:
    """How many times the independent-arms model's per-task variance is the
    paired one, on tasks whose pass rates are ``rates``.

    The model puts ``p(1-p)`` of variance on every task at the mean rate
    ``p``. Pairing keeps each task's own ``p_i(1-p_i)``, whose mean is
    ``p(1-p) - Var(p_i)``, so the model asks for ``1 / (1 - Var(p_i) /
    (p(1-p)))`` times the tasks pairing needs: 1.19x at spread 0.2 around
    0.5, 1.56x at 0.3, 2.78x at 0.4 (population variance, so the ratio is
    finite unless every task is a sure pass or a sure fail, when it is
    ``None``: the model's variance is then all between tasks and pairing
    removes all of it)."""
    if len(rates) < 2:  # noqa: PLR2004  # a spread needs a pair
        return None
    mean = _mean(rates)
    within = mean * (1 - mean)
    between = sum((r - mean) ** 2 for r in rates) / len(rates)
    if within <= 0 or within - between <= _VARIANCE_EPS:
        return None
    return within / (within - between)


def _paired_sd_from_rows(
    before: Sequence[dict], after: Sequence[dict]
) -> tuple[float, int, float, int]:
    """The per-task paired sd measured off both arms: the sample sd of
    ``after - before`` per shared task, so the covariance pairing buys is
    in it. Returns ``(sd, n_paired, base, k)``."""
    means_a = _by_task(before, _binary)
    means_b = _by_task(after, _binary)
    shared = sorted(set(means_a) & set(means_b))
    if len(shared) < MIN_CI_TASKS:
        raise ValueError(
            f"before and after share {len(shared)} graded task(s); a paired sd needs at "
            f"least {MIN_CI_TASKS}. Re-run both arms on the same tasks "
            "(simulate(tasks=base)), or pass task_std= from a previous delta_report"
        )
    diffs = [_mean(means_b[t]) - _mean(means_a[t]) for t in shared]
    base = _mean([_mean(means_a[t]) for t in shared])
    k = min(min(len(means_a[t]), len(means_b[t])) for t in shared)
    return _sample_sd(diffs), len(shared), base, k


def eval_power(
    rows: Sequence[dict],
    *,
    effect: float = PROVE_EFFECT,
    power: float = POWER,
    alpha: float = ALPHA,
    band: tuple[float, float] = DIFFICULTY_BAND,
) -> dict[str, Any]:
    """Can this held-out set prove a gain of ``effect``? Ask before training.

    Pass the graded rows of a BASE-ONLY run. The sizing is ``holdout_size``
    and ``detectable_effect`` read off the same rows, so the three agree:
    ``n_needed`` is ``holdout_size(effect, before=rows)["n_tasks"]`` and
    ``resolvable`` is ``detectable_effect(n_tasks, base=, k=)``, the
    smallest gain this many tasks at this ``k`` can prove at ``power``
    (Miller 2024, arXiv:2411.00640, section 5, the power calculation;
    Lambert 2025, chapter Evaluation, the point of a better eval is
    statistical power when comparing training runs). One model, one home:
    ``_paired_task_sd``.

    Beside the sizing, where the tasks sit. ``in_band`` counts tasks the
    base passes between ``band[0]`` and ``band[1]`` of the time
    (``DIFFICULTY_BAND``, Lambert 2025, chapter Reasoning: the 20-80 band
    difficulty filtering keeps, measured from N=16 there); ``tied_pass``,
    ``tied_fail`` and ``single_rollout`` are the rest. Measured on one
    agent, same world and rubric, base model on both sides: the default
    set had 23 of 59 tasks in band at base 0.589; a set built to be
    "harder" had 0 of 60 at base 0.000. The second is not a harder eval.
    Its base score cannot tell hard tasks from a broken harness (a dead
    tool fails every task the same way), and every group on it is
    zero-accuracy, which DAPO (arXiv:2503.14476) drops for carrying no
    gradient. The pass rate alone does not distinguish the two sets.

    ``verdict`` is one of ``usable`` (``resolvable <= effect``),
    ``underpowered`` (this ``n`` and ``k`` cannot prove ``effect``;
    ``n_needed`` says what could), ``saturated`` (less than ``effect``
    left to gain above ``base``), ``floored`` (every task fails every
    rollout) or ``empty``. Anything but ``usable`` puts a line in
    ``warnings`` that names the fix; ``notes`` are ``holdout_size``'s.
    """
    lo, hi = float(band[0]), float(band[1])
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("band is (low, high) pass rates with 0 <= low < high <= 1")
    by_task = _by_task(rows, _binary)
    n_tasks = len(by_task)
    out: dict[str, Any] = {
        "n_tasks": n_tasks,
        "n_rollouts": sum(len(v) for v in by_task.values()),
        "k": None,
        "base": None,
        "task_spread": None,
        "band": (lo, hi),
        "in_band": 0,
        "tied_pass": 0,
        "tied_fail": 0,
        "single_rollout": 0,
        "effect": float(effect),
        "power": float(power),
        "alpha": float(alpha),
        "task_std": None,
        "resolvable": None,
        "n_needed": None,
        "verdict": "empty",
        "reason": "no graded rows",
        "notes": [],
        "warnings": [],
    }
    if not n_tasks:
        out["warnings"].append(
            "no rows with a 0/1 reward: grade the base run first (grade=True, or "
            "rubric_judge / evaluate), then ask again."
        )
        return out

    rates = {task: _mean(v) for task, v in by_task.items()}
    sizing = holdout_size(effect, power=power, alpha=alpha, before=rows)
    base, k = sizing["base"], sizing["k"]
    resolvable = detectable_effect(n_tasks, base=base, k=k, power=power, alpha=alpha)
    out.update(
        {
            "k": k,
            "base": round(base, 4),
            "task_spread": sizing["base_spread"],
            "in_band": sum(1 for r in rates.values() if lo <= r <= hi),
            "tied_pass": sum(1 for r in rates.values() if r >= 1.0),
            "tied_fail": sum(1 for r in rates.values() if r <= 0.0),
            "single_rollout": sum(1 for v in by_task.values() if len(v) == 1),
            "task_std": sizing["task_std"],
            "resolvable": resolvable,
            "n_needed": sizing["n_tasks"],
            "notes": list(sizing["notes"]),
        }
    )
    if k == 1:
        out["notes"].append(
            "k=1: each task's pass rate is one draw, so in_band cannot be read off this "
            f"run (the band is measured from {DIFFICULTY_BAND_ROLLOUTS} rollouts per task in "
            "its sources); the sizing stands, at the widest per-task spread."
        )
    headroom = 1.0 - base
    if out["tied_fail"] == n_tasks:
        verdict = "floored"
        why = "every task fails every rollout"
        fix = (
            f"every one of the {n_tasks} tasks fails every rollout (base 0.000). The pass "
            "rate cannot tell hard tasks from a broken harness (a dead tool fails every task "
            "the same way), and no group on it carries a gradient. Check coverage['dead_tools'] "
            "and the world's answers before reading 0 as difficulty, then keep tasks the base "
            f"passes {lo:.0%} to {hi:.0%} of the time."
        )
    elif headroom < float(effect):
        verdict = "saturated"
        why = f"base {base:.3f} leaves {headroom:.3f} to gain, under {float(effect):.3f}"
        fix = (
            f"base {base:.3f} leaves {headroom:.3f} to gain, under the {float(effect):.3f} you "
            "want to prove: nothing left to measure on this set. Add harder tasks (hard_share=, "
            "or pin dimensions={'stance': ['boundary', 'ambiguous', 'adversarial']}) so the base "
            f"sits inside the {lo:.0%} to {hi:.0%} band."
        )
    elif resolvable is None or resolvable > float(effect):
        verdict = "underpowered"
        shown = "nothing" if resolvable is None else f"{resolvable:.3f}"
        why = f"{n_tasks} tasks at k={k} resolve {shown}, not {float(effect):.3f}"
        fix = (
            f"{n_tasks} tasks at k={k} resolve a {shown} gain at {float(power):.0%} power, not "
            f"the {float(effect):.3f} you want to prove; about {sizing['n_tasks']} tasks would "
            "(n_needed). Add tasks before training, or pass effect= the gain you expect."
        )
    else:
        verdict = "usable"
        why = (
            f"{n_tasks} tasks at k={k} resolve {resolvable:.3f} at {float(power):.0%} power, "
            f"{out['in_band']} in band"
        )
        fix = ""
    out.update({"verdict": verdict, "reason": why})
    if fix:
        out["warnings"].append(f"this held-out set cannot prove a {float(effect):.3f} gain: {fix}")
    return out


class HoldoutSizeReport(Report):
    """What ``holdout_size`` answered, as a person reads it: the task
    count on the first line, where the standard deviation came from on the
    second, then the warnings and notes that say what the number assumes.

    Reference: docs/reference/style.md rule 5 (results are objects that
    print themselves).
    """

    _summary_keys = ("n_tasks", "effect")

    def __str__(self) -> str:
        lines = [
            f"holdout {self['n_tasks']} paired tasks for a +{self['effect']:.3f} gain at "
            f"k={self['k']} (base {self['base']:.2f}, power {self['power']:.2f}, "
            f"alpha {self['alpha']:.2f})"
        ]
        source = self["sd_source"]
        where = f"{source}, {self['n_paired']} paired tasks" if self.get("n_paired") else source
        lines.append(
            f"task_std {self['task_std']:.4f} ({where}), half-width {self['half_width']:.4f}"
        )
        lines += [f"warning: {w}" for w in self.get("warnings") or ()]
        lines += [f"note: {n}" for n in self.get("notes") or ()]
        return "\n".join(lines)


def holdout_size(
    effect: float,
    *,
    base: float = BASE_PASS_RATE,
    k: int = ROLLOUTS_PER_TASK,
    power: float = POWER,
    alpha: float = ALPHA,
    before: Sequence[dict] | None = None,
    after: Sequence[dict] | None = None,
    task_std: float | None = None,
    rows: Sequence[dict] | None = None,
    ceiling_pass_rate: float = CEILING_PASS_RATE,
) -> HoldoutSizeReport:
    """How many paired tasks a holdout needs to prove a gain of ``effect``.

    Models the test ``delta_report`` runs: each task's pass rate over ``k``
    rollouts on each side, the delta as the mean of the paired differences,
    the interval from a bootstrap over tasks. The usual two-sided power
    calculation then gives ``n = ((z_{1-alpha/2} + z_power) * sd / effect) **
    2`` with ``sd`` the standard deviation of one task's paired difference
    (Lambert 2025, chapter Evaluation: the point of a better eval is
    statistical power when comparing training runs). Where ``sd`` comes from
    is the whole question, and there are three ways to answer it, best first:

    * ``before`` and ``after``, the graded arms of a previous eval on the
      same tasks (the two row lists ``delta_report(before, after)``
      takes): ``sd`` is measured as the sample sd of the per-task
      differences, which carries the covariance that pairing buys and
      whatever shape the gain had. No model. ``sd_source`` is ``"rows"``
      and ``n_paired`` says how many tasks it was read off.
    * ``task_std``, a number you measured (the per-task sibling of
      ``delta_report``'s ``run_std``): the same quantity read off a
      previous ``delta_report``: ``(hi - lo) * sqrt(n_paired_tasks) /
      3.92`` from ``target_ci95`` and ``n_paired_tasks`` (or any
      ``metrics[...]["ci95"]`` with its ``n_paired``). Agent rubrics sat
      near 0.38 across five lanes (#288). ``sd_source`` is ``"given"``.
      ``eval_variance``'s ``run_std`` is a different number (how much a
      re-run moves the mean) and is not this.
    * Neither: the binomial model ``sqrt((p(1-p) + q(1-q)) / k)`` with
      ``p = base`` and ``q = base + effect``, ``sd_source`` ``"model"``.
      It assumes two things it cannot check: that the gain is spread
      evenly across tasks, and that the two arms are independent draws
      (``Var(A) + Var(B)``, no covariance term). When the gain is carried
      by a few tasks, most tasks are ties and the paired differences
      spread far wider than binomial-per-task predicts; a voice trait at
      0 -> 0.127, k=4, carried by 19 of 150 tasks, measured sd 0.333
      against the model's 0.168 and needed 54 tasks where the model said
      14 (#292). So the model path also returns
      ``n_tasks_concentrated``, the count if the gain were carried by
      the fewest tasks that can carry it (each going from ``base`` to
      1), and ``notes`` says which assumption is in play. On a holdout
      whose tasks differ in difficulty the independence assumption errs
      the other way: the model puts ``p(1-p)`` of variance on every task
      where pairing keeps each task's own ``p_i(1-p_i)``, whose mean is
      ``p(1-p) - Var(p_i)``, so it asks for ``1 / (1 - Var(p_i) /
      (p(1-p)))`` times the tasks pairing needs (1.19x at spread 0.2
      around 0.5, 2.78x at 0.4). ``before`` alone reports the spread as
      ``base_spread`` and puts that ratio in ``notes``.

    ``before`` on its own (``rows`` is the same argument under its old
    name) reads ``base`` and ``k`` off the data. Returns ``n_tasks``
    plus the inputs, ``task_std``, ``sd_source``, ``half_width`` (the 95%
    band on the delta at that ``n``), ``n_tasks_concentrated``,
    ``base_spread``, ``n_paired``, ``saturated``, ``notes`` and
    ``warnings``; every key is present on every path (``None``, ``False``
    or ``[]`` where it does not apply). The default answer is unchanged;
    the honest paths are the two that measure.

    A saturated ``base=`` cannot size anything (``base`` is the before
    arm's pass rate; there is no ``baseline=``). Rows whose tasks all pass
    give ``p = 1``, the binomial variance ``p(1-p)`` is 0, and both arms
    all passing give a measured paired sd of 0; the formula then returns
    the floor, ``MIN_HOLDOUT_TASKS``, which is the model collapsing, not
    evidence that two tasks are enough (#392). When the measured base is
    at or above ``ceiling_pass_rate`` (``CEILING_PASS_RATE``, the share
    ``delta_report`` flags as ``ceiling``) or the measured sd is 0 (the
    paired difference identical on every task, ``DEGENERATE``), the
    rows are not used: ``n_tasks`` is the binomial model's answer at
    ``BASE_PASS_RATE`` and the rows' ``k``, ``sd_source`` is ``"model"``,
    ``saturated`` is ``True``, and ``warnings`` names the ceiling and the
    fix: harder situations, so ``base`` sits inside the 20-80
    difficulty band (Lambert 2025, chapter Reasoning; DAPO, arXiv
    2503.14476, drops prompts at accuracy 0 and 1 because they carry no
    signal), then size again on those rows.

    The recipe that asked for this had 140 tasks at k=4 around 0.6: a
    band of about +-0.06, so a real 3-point gain reads
    ``no_change_detected`` every round. This says so before training.
    """
    if not 0 < float(effect) < 1:
        raise ValueError(
            "effect is the gain in pass rate to prove, between 0 and 1 (0.05 = 5 points)"
        )
    if not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("power and alpha are probabilities strictly between 0 and 1")
    if before is None:
        before = rows
    if after is not None and before is None:
        raise ValueError("after= needs before= (the before arm) to pair against")
    if task_std is not None and not float(task_std) > 0:
        raise ValueError(
            "task_std is the per-task paired standard deviation you measured, above 0 "
            "(agent rubrics sit near 0.38)"
        )
    notes: list[str] = []
    warnings: list[str] = []
    spread: float | None = None
    ratio: float | None = None
    n_paired: int | None = None
    saturated = False
    if before is not None:
        base, k, spread, ratio = _rows_base_and_k(before)
    if before is not None and after is not None:
        sd, n_paired, base, k = _paired_sd_from_rows(before, after)
        source = "rows"
        notes.append(
            f"task_std {sd:.3f} measured as the sample sd of the per-task paired difference "
            f"over {n_paired} tasks; no model, the covariance pairing buys is in it"
        )
    elif task_std is not None:
        sd = float(task_std)
        source = "given"
        notes.append(f"task_std {sd:.3f} given; the binomial model was not used")
    else:
        sd = _paired_task_sd(base, effect, k)
        source = "model"
    if before is not None and source != "given" and (base >= ceiling_pass_rate or sd <= 0):
        # The rows cannot size anything: at the ceiling p(1-p) is (near)
        # zero and a measured sd of 0 says both arms agreed on every task.
        # Answer with the model at the default base, and say so.
        saturated = True
        measured = f"task_std {sd:.3f} measured" if source == "rows" else "the binomial variance"
        band_lo, band_hi = DIFFICULTY_BAND
        if base >= ceiling_pass_rate:
            head = (
                f"CEILING: the before rows pass {base:.2f} of tasks, at or above the ceiling "
                f"{ceiling_pass_rate:.2f}, so this suite cannot show a {float(effect):.2f} gain; "
                f"{measured} collapses to 0"
            )
        else:
            head = (
                f"DEGENERATE: the paired difference is the same on every one of the "
                f"{n_paired} tasks ({measured} is 0) at a base of {base:.2f}, so the rows carry "
                "no spread to size from"
            )
        warnings.append(
            f"{head} and the sizing formula returns its floor ({MIN_HOLDOUT_TASKS} tasks), which "
            "is the model collapsing, not evidence. n_tasks is the binomial model's answer at "
            f"the default base {BASE_PASS_RATE:.2f} with these rows' k={int(k)}, not a "
            "measurement. Fix: harder situations, so the baseline sits inside the "
            f"{band_lo:.0%}-{band_hi:.0%} difficulty band (simulate(hard_share=...) or a higher "
            "fault_rate; Lambert 2025, chapter Reasoning), then size again on those "
            "rows."
        )
        sd = _paired_task_sd(BASE_PASS_RATE, effect, k)
        source = "model"
        n_paired = None
    z = _z(1 - alpha / 2) + _z(power)

    def _n(s: float) -> int:
        return max(MIN_HOLDOUT_TASKS, math.ceil((z * s / float(effect)) ** 2) if s > 0 else 1)

    n = _n(sd)
    concentrated: int | None = None
    if source == "model":
        model_base = BASE_PASS_RATE if saturated else base
        concentrated = _n(_concentrated_task_sd(model_base, effect, k))
        notes.append(
            f"n_tasks {n} assumes the gain is spread evenly across tasks and the two arms are "
            f"independent draws. If the gain is carried by a few tasks (a trait only some "
            f"prompts exercise) the same {float(effect):.3f} gain needs about {concentrated} "
            f"tasks (n_tasks_concentrated). Pass before= and after= from a previous eval to "
            f"measure the paired sd, or task_std= read off a delta_report interval."
        )
        if spread is not None and spread > 0 and not saturated:
            if ratio is None:
                how = (
                    "every task is a sure pass or a sure fail, so the model's per-task "
                    "variance is all between tasks and pairing removes all of it"
                )
            else:
                how = (
                    f"the independent-arms model asks for {ratio:.2f}x the tasks pairing "
                    f"needs at this spread (1 / (1 - Var(p_i) / (p(1-p))) with p {base:.2f})"
                )
            notes.append(
                f"per-task base pass rates spread sd {spread:.3f}; {how}. It understates "
                "when the gain is concentrated. Which way this one errs is decided by "
                "after= rows."
            )
    return HoldoutSizeReport(
        {
            "n_tasks": n,
            "effect": float(effect),
            "base": float(base),
            "k": int(k),
            "power": float(power),
            "alpha": float(alpha),
            "task_std": round(sd, 4),
            "sd_source": source,
            "half_width": round(_z(1 - alpha / 2) * sd / math.sqrt(n), 4),
            "n_tasks_concentrated": concentrated,
            "base_spread": round(spread, 4) if spread is not None else None,
            "n_paired": n_paired,
            "saturated": saturated,
            "notes": notes,
            "warnings": warnings,
        }
    )


def detectable_effect(
    n_tasks: int,
    *,
    base: float = BASE_PASS_RATE,
    k: int = ROLLOUTS_PER_TASK,
    power: float = POWER,
    alpha: float = ALPHA,
) -> float | None:
    """The smallest gain ``n_tasks`` paired tasks can prove at ``power``:
    ``holdout_size`` solved for the effect (``FIXED_POINT_STEPS``
    fixed-point steps, since the after-side variance depends on it).
    ``None`` below ``MIN_HOLDOUT_TASKS`` tasks."""
    n = int(n_tasks)
    if n < MIN_HOLDOUT_TASKS:
        return None
    z = _z(1 - alpha / 2) + _z(power)
    effect = 0.0
    for _ in range(FIXED_POINT_STEPS):
        sd = _paired_task_sd(base, effect, k)
        effect = z * sd / math.sqrt(n)
    return round(min(1.0, effect), 4)


def bootstrap_ci(
    values: Sequence[float],
    *,
    stat: Callable[[Sequence[float]], float] = _mean,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    level: float = CI_LEVEL,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of ``stat`` over ``values`` at
    ``level`` (``CI_LEVEL`` by default). ``None`` below ``MIN_CI_TASKS``
    values, where the interval would be the data itself."""
    if not 0 < level < 1:
        raise ValueError("level is the interval's coverage, strictly between 0 and 1")
    vals = [float(v) for v in values]
    if len(vals) < MIN_CI_TASKS:
        return None
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo_i = int((1 - level) / 2 * n_boot)
    hi_i = int((1 + level) / 2 * n_boot) - 1
    return (stats[max(0, lo_i)], stats[min(n_boot - 1, hi_i)])


def no_interval_note(n_tasks: int, *, quantity: str = "the mean") -> str:
    """One sentence for a report whose ``bootstrap_ci`` came back ``None``.

    ``bootstrap_ci`` withholds an interval below ``MIN_CI_TASKS`` tasks,
    and a report that prints the bare mean next to that silence reads
    like a result (#490). This is the sentence that goes in the report's
    note: the task count, the threshold, and the one thing to change.
    The bootstrap resamples tasks, so the fix is more tasks, or one
    ``task_id`` per row when the rows are separate items that collapsed
    into one group under ``task_key``. ``marker_summary`` writes the same
    sentence in its own words, naming the marker instead.

    >>> no_interval_note(1, quantity="pass@1")[:24]
    'no interval on pass@1: 1'
    """
    counted = f"{n_tasks} task" if n_tasks == 1 else f"{n_tasks} tasks"
    return (
        f"no interval on {quantity}: {counted}, and a bootstrap needs "
        f"{MIN_CI_TASKS}; it resamples tasks, so run more tasks, or give each row "
        "its own task_id when the rows are separate items"
    )


def task_key(row: dict) -> str:
    """The one name every report groups a row's rollouts under.

    A task is a situation, not a string: ``scenario_id`` when the row has one
    (the engine's situation id, shared by the repeats of one opener and by the
    textured phrasings of one situation), else ``task_id`` (rows from
    elsewhere), else the prompt text. ``pass_at``, ``compare_runs``,
    ``delta_report``, ``eval_variance``, ``curriculum``, ``group_signal`` and
    the exporters all count tasks with this key, so the same rows give the
    same task count everywhere (Miller 2024, arXiv:2411.00640: intervals and
    paired comparisons are over tasks, never rows).
    """
    return str(row.get("scenario_id") or row.get("task_id") or row.get("prompt") or "")


def _by_task(rows: Sequence[dict], value: Callable[[dict], float | None]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        v = value(row)
        if v is None:
            continue
        groups.setdefault(task_key(row), []).append(float(v))
    return groups


def _binary(row: dict, key: str = "reward") -> float | None:
    v = row.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f in (0.0, 1.0) else None


def _marker(name: str) -> Callable[[dict], float | None]:
    def get(row: dict) -> float | None:
        markers = row.get("markers")
        if not isinstance(markers, dict) or name not in markers:
            return None
        try:
            return float(markers[name])
        except (TypeError, ValueError):
            return None

    return get


def task_means(rows: Sequence[dict], metric: str = "pass_at_1") -> dict[str, float]:
    """Per-task mean of a metric: ``"pass_at_1"`` (binary reward) or
    ``"marker:<name>"``. The unit every interval and comparison rests on."""
    getter = _binary if metric == "pass_at_1" else _marker(metric.split(":", 1)[1])
    groups = _by_task(rows, getter)
    return {task: _mean(vals) for task, vals in groups.items()}


def metric_summary(
    rows: Sequence[dict], metric: str = "pass_at_1", *, n_boot: int = DEFAULT_BOOT, seed: int = 0
) -> dict[str, Any]:
    """Mean over tasks with a task-bootstrap 95% interval.

    ``degenerate`` is set when every applicable row scored the same
    value: the metric has not been shown to be able to come out any
    other way, so ``ci95`` is ``None`` (the way ``pass_at`` returns
    ``None`` below three groups) and ``warning`` says so. A marker that
    is silently unfireable (a key-name mismatch) and one that is
    genuinely always true look identical otherwise, and either one passed
    to ``must_not_regress`` is a guard that cannot fail (#270).
    ``n_rows_at_1`` and ``n_rows_at_0`` put the row-level split next to
    the mean.
    """
    means = task_means(rows, metric)
    values = list(means.values())
    per_task = _by_task(
        rows, _binary if metric == "pass_at_1" else _marker(metric.split(":", 1)[1])
    )
    row_values = [v for vs in per_task.values() for v in vs]
    distinct = {round(float(v), 9) for v in row_values}
    degenerate = len(row_values) > 0 and len(distinct) == 1
    out: dict[str, Any] = {
        "metric": metric,
        "n_tasks": len(values),
        "n_rows": len(row_values),
        "n_rows_at_1": sum(1 for v in row_values if float(v) == 1.0),
        "n_rows_at_0": sum(1 for v in row_values if float(v) == 0.0),
        "mean": _mean(values) if values else None,
        "ci95": None if degenerate else bootstrap_ci(values, n_boot=n_boot, seed=seed),
        "degenerate": degenerate,
    }
    if degenerate:
        only = next(iter(distinct))
        name = metric.split(":", 1)[1] if metric.startswith("marker:") else metric
        out["warning"] = (
            f"all {len(row_values)} applicable rows scored {only:g}; {name} has not been shown "
            "to be able to come out any other way. Check the marker fires at all (a key-name "
            "mismatch looks exactly like this) before reading the mean, and do not put it in "
            "must_not_regress: a guard that cannot fail catches nothing."
        )
    return out


def marker_names(rows: Sequence[dict]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        markers = row.get("markers") if isinstance(row, dict) else None
        if isinstance(markers, dict):
            names.update(str(k) for k in markers)
    return sorted(names)


def marker_summary(
    rows: Sequence[dict],
    *,
    names: Sequence[str] | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """``metric_summary`` for every marker on the rows (or ``names``).

    Each marker's stats are keyed ``mean``, ``ci95`` (not ``ci``),
    ``n_tasks``, ``n_rows`` (not ``n``), ``n_rows_at_1``, ``n_rows_at_0``,
    ``degenerate``, and ``note`` or ``warning`` when there is one.
    ``ci95`` is ``None`` below ``MIN_CI_TASKS`` tasks, and ``note`` then
    says how many tasks the marker has and how many the interval needs;
    a reader who sees only ``None`` cannot tell that from a bug.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in names or marker_names(rows):
        stats = metric_summary(rows, f"marker:{name}", n_boot=n_boot, seed=seed)
        if stats["ci95"] is None and not stats["degenerate"]:
            stats["note"] = (
                f"no interval: {stats['n_tasks']} task(s) carry {name}, and the bootstrap "
                f"needs {MIN_CI_TASKS} or more. Add tasks (more situations, not more "
                "rollouts of one) and score them."
            )
        out[name] = stats
    return out


# ------------------------------------------------------------------ re-run variance


# VARIANCE_BANDS = (0.35, 0.7) points: where ``eval_variance`` places a
# ``run_std`` on Olmo 3's bands for the standard deviation of a benchmark
# across re-runs of one model, on a 0-100 scale: MMLU/MATH/PopQA sit near 0.2,
# GPQA/AlpacaEval above 1.2. Lambert 2025, chapter Evaluation ("Why Many
# External Evaluation Comparisons Are Unreliable"), puts most post-training
# evaluations between 0.25 and 1.5 points with the setup held constant; 0.35
# and 0.7 split that range so the three labels each cover a third of it
# (convention on the cut points).
VARIANCE_BANDS = (("very_stable", 0.35), ("stable", 0.7), ("high_variance", float("inf")))
# POINTS_PER_UNIT = 100: pass rates are 0-1, the bands above are in points.
POINTS_PER_UNIT = 100


def _run_key(row: dict, by: str | None) -> str | None:
    if by:
        value = row.get(by)
        if value is None and isinstance(row.get("lineage"), dict):
            value = row["lineage"].get(by)
        return str(value) if value not in (None, "") else None
    lineage = row.get("lineage")
    if isinstance(lineage, dict) and lineage.get("eval_run") is not None:
        # simulate(runs=N) stamps this; one evaluate() over all N runs
        # gives them one scoring_run_id, so the eval run wins.
        return str(lineage["eval_run"])
    if isinstance(lineage, dict) and lineage.get("scoring_run_id"):
        return str(lineage["scoring_run_id"])
    return None


# Containers the SDK hands back, and the attribute on each that holds the rows.
_ROW_ATTRS = ("trajectories", "rows")


def _run_rows(run: Any, position: int) -> list[dict]:
    """One run's rows, or a TypeError that names the next action.

    ``eval_variance`` takes row lists. Handed a ``SimulationData`` -- what
    ``simulate`` returns -- Python raised a bare ``"object is not iterable"``,
    which does not say that an attribute away is the right shape (#31).
    (``ScoredData`` iterates its rows and never raised; the ``.rows`` probe
    is for user-built containers that hold rows under that name.)
    """
    if isinstance(run, (dict, str, bytes)):
        raise TypeError(
            f"eval_variance() argument {position} is a {type(run).__name__}; each argument is "
            "one re-run's rows. Pass the row lists themselves: "
            "eval_variance(rows_1, rows_2, rows_3)."
        )
    try:
        return [row for row in run if isinstance(row, dict)]
    except TypeError:
        attr = next((a for a in _ROW_ATTRS if isinstance(getattr(run, a, None), list)), None)
        name = type(run).__name__
        if attr:
            raise TypeError(
                f"eval_variance() argument {position} is a {name}, not a list of rows. "
                f"Pass its .{attr}: eval_variance(a.{attr}, b.{attr}, c.{attr}), where a, b and c "
                "are three evaluations of the same model."
            ) from None
        raise TypeError(
            f"eval_variance() argument {position} has type {name}, which is not a list of rows. "
            "Each argument is one re-run's scored rows (dicts carrying 'reward' and a task key)."
        ) from None


class EvalVarianceReport(Report):
    """The eval's noise floor as a person reads it: the spread across
    re-runs, the band a delta has to clear, each run's mean, and the notes
    that say when the spread is too thin to read.

    Reference: docs/reference/style.md rule 5 (results are objects that
    print themselves).
    """

    _summary_keys = ("run_std", "n_runs")

    def __str__(self) -> str:
        std, n = self["run_std"], self["n_runs"]
        head = f"eval_variance {self['metric']}: "
        if std is None:
            head += f"no run_std from {n} run(s)"
        else:
            head += f"run_std {std:.4f} over {n} runs"
            if self["mean"] is not None:
                head += f", mean {self['mean']:.4f}"
        lines = [head]
        if self["noise_band"] is not None:
            lines.append(
                f"noise band {self['noise_band']:.4f} (t at df {self['noise_band_df']} times "
                "sqrt(2) times run_std): a delta inside it is the eval re-running, not a gain. "
                "Pass it to compare(run_std=, run_std_runs=)"
            )
        runs = " | ".join(
            f"{label} {value:.4f}" if value is not None else f"{label} none"
            for label, value in self["means"].items()
        )
        if runs:
            lines.append(f"{runs} ({self['tasks_in_every_run']} tasks in every run)")
        if self["stability"] is not None:
            lines.append(f"stability {self['stability']} ({self['run_std_points']:.2f} points)")
        lines += [f"note: {note}" for note in self.get("notes") or ()]
        return "\n".join(lines)


def eval_variance(
    *runs: Sequence[dict],
    metric: str = "pass_at_1",
    by: str | None = None,
) -> EvalVarianceReport:
    """How much an evaluation moves when the same model is evaluated
    again (Lambert 2025, chapter Evaluation).

    Pass each re-run's rows as its own argument, or one row list whose
    rows say which run they belong to: ``lineage.eval_run`` (what
    ``simulate(runs=3)`` stamps), else ``lineage.scoring_run_id`` (what
    ``evaluate(run_id=)`` stamps), or a top-level or lineage key named by
    ``by``. Each run's ``metric`` is a mean over tasks; the report is
    those means, their mean, the sample standard deviation ``run_std``,
    and ``noise_band`` = ``noise_band(run_std, df=n_runs - 1)``: the
    two-sided t quantile at ``noise_band_df`` = ``n_runs - 1`` times
    sqrt(2) times ``run_std``, because a before/after delta with one run
    per side is the difference of two re-run draws and ``run_std`` is an
    estimate from these very runs, not the eval's exact spread (Lambert
    2025, chapter Evaluation). This is the band ``compare(run_std=,
    run_std_runs=)`` applies; with three runs the multiplier is 4.30, not
    1.96 (the 1.96 band read a three-run estimate as exact and let about
    one pure-noise delta in five through, #616). A delta inside the band
    is what re-running the eval does on its own.
    ``run_std_by_metric``
    reports the same floor for pass@1 and every marker shared by all runs;
    hand that mapping to ``delta_report(run_std=)`` so each metric uses its
    own re-run variance. The scalar ``run_std`` remains the selected
    ``metric``'s value for callers comparing only one metric. ``stability``
    places ``run_std`` on Olmo 3's bands in points.
    Fewer than three runs is a difference, not a distribution; the report
    says so and ``run_std`` is ``None`` below two.
    """
    if not runs:
        raise ValueError("eval_variance needs at least one row list")
    groups: list[list[dict]]
    if len(runs) == 1:
        split: dict[str, list[dict]] = {}
        unkeyed = 0
        for row in _run_rows(runs[0], 1):
            key = _run_key(row, by)
            if key is None:
                unkeyed += 1
                continue
            split.setdefault(key, []).append(row)
        groups = list(split.values())
        labels = list(split)
    else:
        groups = [_run_rows(r, i + 1) for i, r in enumerate(runs)]
        labels = [f"run_{i + 1}" for i in range(len(groups))]
        unkeyed = 0

    def _run_means(which: str) -> tuple[dict[str, float | None], list[set[str]]]:
        """Each run's mean of ``which`` over its tasks (unrounded), and the
        task set each run covered."""
        by_run: dict[str, float | None] = {}
        covered: list[set[str]] = []
        for label, rows in zip(labels, groups):
            per_task = task_means(rows, which)
            by_run[label] = _mean(list(per_task.values())) if per_task else None
            covered.append(set(per_task))
        return by_run, covered

    def _sample_std(values: list[float]) -> float | None:
        if len(values) < 2:  # noqa: PLR2004  # a spread needs a pair
            return None
        centre = _mean(values)
        return (sum((v - centre) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    raw_means, task_sets = _run_means(metric)
    means: dict[str, float | None] = {
        label: round(v, 4) if v is not None else None for label, v in raw_means.items()
    }
    values = [v for v in raw_means.values() if v is not None]
    n = len(values)
    mean = _mean(values) if values else None
    std = _sample_std(values)
    common = set.intersection(*task_sets) if task_sets else set()
    # One floor per metric, each from the same unrounded run means as the
    # scalar, so ``run_std`` and ``run_std_by_metric[metric]`` agree to the
    # digit. A marker that applies to a subset of tasks is noisier than
    # pass@1, which averages over all of them (#300).
    shared_markers = (
        set.intersection(*(set(marker_names(rows)) for rows in groups)) if groups else set()
    )
    floor_metrics = ["pass_at_1", *[f"marker:{name}" for name in sorted(shared_markers)]]
    if metric not in floor_metrics:
        floor_metrics.append(metric)
    run_std_by_metric: dict[str, float | None] = {}
    for floor_metric in floor_metrics:
        metric_std = _sample_std([v for v in _run_means(floor_metric)[0].values() if v is not None])
        run_std_by_metric[floor_metric] = round(metric_std, 4) if metric_std is not None else None
    stability = None
    if std is not None:
        points = std * POINTS_PER_UNIT
        stability = next(name for name, cap in VARIANCE_BANDS if points < cap)
    out = EvalVarianceReport(
        {
            "metric": metric,
            "n_runs": n,
            "means": means,
            "mean": round(mean, 4) if mean is not None else None,
            "run_std": round(std, 4) if std is not None else None,
            "run_std_by_metric": run_std_by_metric,
            "run_std_points": round(std * POINTS_PER_UNIT, 2) if std is not None else None,
            # ``std`` exists only from two or more runs, so ``n - 1`` is at
            # least one: the band carries the estimate's own degrees of freedom.
            "noise_band": round(noise_band(std, df=n - 1), 4) if std is not None else None,
            "noise_band_df": n - 1 if std is not None else None,
            "stability": stability,
            "tasks_in_every_run": len(common),
            "notes": [],
        }
    )
    if unkeyed:
        out["notes"].append(f"{unkeyed} row(s) carried no run id and were left out")
    if n < MIN_RERUNS:
        out["notes"].append(
            f"{n} run(s): two is a difference, not a distribution; {MIN_RERUNS} or more re-runs "
            "give a standard deviation worth reading"
        )
    if task_sets and any(s != common for s in task_sets):
        out["notes"].append("runs do not cover the same tasks; means are not strictly comparable")
    return out


# ------------------------------------------------------------------ comparison


def compare_runs(
    a: Sequence[dict],
    b: Sequence[dict],
    *,
    metric: str = "pass_at_1",
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    min_paired: int = MIN_PAIRED_TASKS,
    level: float = CI_LEVEL,
) -> dict[str, Any]:
    """Test whether run ``b`` differs from run ``a`` on one metric, paired by task.

    Reach for it for a quick A/B on a single number; ``delta_report`` is
    the full report with markers, the noise floor and the comparability
    checks. It returns a dict: ``delta`` (b minus a), ``ci95`` (the
    interval, with ``level`` beside it), ``p_value``, ``verdict``,
    ``n_paired``, ``n_only_a``, ``n_only_b``, ``paired_share``,
    ``mean_a``, ``mean_b``, and a ``note``.

    Tasks the two runs share are compared as paired differences (b minus
    a, per task, keyed the way ``pass_at`` groups); the interval is a
    ``level`` bootstrap over those pairs and the p-value is a sign-flip
    permutation test. ``verdict`` is one of ``"b_better"``,
    ``"a_better"``, ``"no_difference_detected"``: the last means the
    interval covers zero, not that the runs are equal. Tasks on one side
    only are dropped from a paired comparison, and ``note`` says how
    many, since a verdict over a quarter of the tasks is not a verdict
    over the eval. ``paired_share`` is the shared fraction of every task
    either run saw.

    * ``metric``: ``"pass_at_1"`` (the default, binary reward) or
      ``"marker:name"`` for a marker.
    * ``min_paired`` (5): with fewer shared tasks the comparison falls
      back to unpaired task means and says so.
    * ``level`` (0.95): the interval's coverage (``ci95`` at the default).
      ``n_boot`` (2000) and ``seed`` (0) fix the bootstrap.

    >>> a = [{"task_id": t, "reward": 0} for t in "abcdef"]
    >>> b = [{"task_id": t, "reward": 1} for t in "abcdef"]
    >>> wai.compare_runs(a, b)["verdict"]
    'b_better'
    """
    if not 0 < level < 1:
        raise ValueError("level is the interval's coverage, strictly between 0 and 1")
    ma = task_means(a, metric)
    mb = task_means(b, metric)
    shared = sorted(set(ma) & set(mb))
    paired = len(shared) >= min_paired
    rng = random.Random(seed)
    if paired:
        diffs = [mb[t] - ma[t] for t in shared]
        delta = _mean(diffs)
        ci = bootstrap_ci(diffs, n_boot=n_boot, seed=seed, level=level)
        # Sign-flip permutation: under H0 each paired difference is
        # equally likely to have either sign.
        observed = abs(delta)
        n = len(diffs)
        extreme = 0
        for _ in range(n_boot):
            flipped = _mean([d if rng.random() < _FAIR_COIN else -d for d in diffs])
            if abs(flipped) >= observed - 1e-12:
                extreme += 1
        p_value = (extreme + 1) / (n_boot + 1)
        n_used = n
    else:
        va, vb = list(ma.values()), list(mb.values())
        delta = (_mean(vb) - _mean(va)) if va and vb else float("nan")
        ci = None
        if len(va) >= MIN_CI_TASKS and len(vb) >= MIN_CI_TASKS:
            boots = []
            for _ in range(n_boot):
                sa = _mean([va[rng.randrange(len(va))] for _ in va])
                sb = _mean([vb[rng.randrange(len(vb))] for _ in vb])
                boots.append(sb - sa)
            boots.sort()
            ci = (boots[int((1 - level) / 2 * n_boot)], boots[int((1 + level) / 2 * n_boot) - 1])
        p_value = None
        n_used = min(len(va), len(vb))
    if ci is None or math.isnan(delta):
        verdict = "insufficient_data"
    elif ci[0] > 0:
        verdict = "b_better"
    elif ci[1] < 0:
        verdict = "a_better"
    else:
        verdict = "no_difference_detected"
    n_only_a = len(set(ma) - set(mb))
    n_only_b = len(set(mb) - set(ma))
    n_all = len(set(ma) | set(mb))
    paired_share = (len(shared) / n_all) if n_all else None
    if not paired:
        note = f"fewer than {min_paired} shared tasks; unpaired task means, weaker test"
    elif n_only_a or n_only_b:
        note = (
            f"{n_only_a} tasks only in a and {n_only_b} only in b were dropped; "
            f"the verdict rests on the {len(shared)} shared"
        )
        if paired_share is not None and paired_share < UNPAIRED_MAJORITY_SHARE:
            note = "most tasks unpaired: " + note
    else:
        note = ""
    return {
        "metric": metric,
        "paired": paired,
        "n_paired": len(shared),
        "n_only_a": n_only_a,
        "n_only_b": n_only_b,
        "paired_share": paired_share,
        "n_used": n_used,
        "mean_a": _mean(list(ma.values())) if ma else None,
        "mean_b": _mean(list(mb.values())) if mb else None,
        "delta": delta if not math.isnan(delta) else None,
        "ci95": ci,
        "level": level,
        "p_value": p_value,
        "verdict": verdict,
        "note": note,
    }


# ------------------------------------------------------------------ decontamination


def _ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _load_rows(source: Any) -> list[dict]:
    """Rows from a list, a JSONL path, or a platform dataset id."""
    if isinstance(source, (str, Path)):
        text = str(source)
        if text.startswith("ds_"):
            from ..ingest.platform import pull

            rows = pull(text)
            return rows if isinstance(rows, list) else []
        from .quality import load_jsonl

        return load_jsonl(text)
    return [r for r in source if isinstance(r, dict)]


def _eval_texts(row: dict) -> list[str]:
    """What an evaluation row contributes: its prompt and its gold answer
    or reference. Not its ``final_text``: on a rollout-shaped eval set
    that is a policy's reply, and tool boilerplate shared between any two
    replies would flag training rows that never saw the eval question
    (Lambert 2025, chapter Evaluation, decontaminates on prompt overlap)."""
    out = [str(row.get("prompt") or "")]
    for key in ("answer", "reference"):
        if row.get(key):
            out.append(str(row[key]))
    return out


def _explicit_task(row: dict) -> str | None:
    """A row's recorded task identity (``scenario_id`` else ``task_id``),
    ``None`` when it carries neither: the prompt fallback of ``task_key``
    is already the exact rule."""
    value = row.get("scenario_id") or row.get("task_id")
    return str(value) if value else None


def _unit_vectors(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    out = []
    for vec in vectors:
        floats = [float(x) for x in vec]
        norm = math.sqrt(sum(x * x for x in floats))
        out.append([x / norm for x in floats] if norm > 0 else floats)
    return out


def _embed(embedder: Any, texts: list[str]) -> list[list[float]]:
    """``embedder(texts)`` checked to return one vector per text."""
    embed = embedder.embed if not callable(embedder) and hasattr(embedder, "embed") else embedder
    if not callable(embed):
        raise TypeError(
            "embedder must be a callable taking a list of texts and returning one vector "
            "per text (list[str] -> list[list[float]]); see decontaminate's docstring for a "
            "sentence-transformers example"
        )
    if not texts:
        return []
    vectors = list(embed(list(texts)))
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(texts)} texts; it must "
            "return exactly one vector per text, in order"
        )
    return _unit_vectors(vectors)


def _distinct_task_similarity(
    vectors: Sequence[Sequence[float]], task_ids: Sequence[str | None]
) -> float | None:
    """The 99th percentile of cosine similarity over pairs of unit vectors
    whose task ids differ (both recorded): how alike two distinct tasks
    can read to this embedder. ``None`` with fewer than two such pairs,
    or when no ids are recorded, since one pair is not a distribution."""
    sims: list[float] = []
    for a in range(len(vectors)):
        if task_ids[a] is None:
            continue
        for b in range(a + 1, len(vectors)):
            if task_ids[b] is None or task_ids[b] == task_ids[a]:
                continue
            sims.append(float(sum(x * y for x, y in zip(vectors[a], vectors[b]))))
    if len(sims) < 2:  # noqa: PLR2004  # a spread needs a pair
        return None
    sims.sort()
    return min(1.0, sims[round(DISTINCT_TASK_PERCENTILE * (len(sims) - 1))])


def decontaminate(
    rows: Sequence[dict],
    against: Sequence[Any] | Any,
    *,
    n: int = DECONTAM_NGRAM,
    fields: Sequence[str] = ("prompt",),
    overlap: float = DECONTAM_OVERLAP,
    embedder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
    similarity: float = SEMANTIC_SIMILARITY,
) -> tuple[list[dict], dict[str, Any]]:
    """Drop training rows whose prompt overlaps an evaluation set.

    Reach for it before any train-versus-holdout comparison: a held-out task
    that also sits in the training data measures memory, not the change
    (Lambert 2025, chapter Evaluation). It returns ``(clean_rows, report)``:
    the rows that survived (a list that also carries the system prompt and
    tools the input carried, so ``select(clean_rows).export()`` writes
    them), and a report with the count under each rule
    (``n_contaminated`` in total), hits per field, the eval text count, and
    the first offenders with their coverage (or ``similarity`` for semantic
    hits). ``rules_skipped`` names each rule that could not run on these
    inputs and why (empty when every rule ran), and ``notes`` says it in a
    sentence: a zero under a rule that never ran is not a clearance.

    * ``rows``: the training rows.
    * ``against``: one or more evaluation sources: row lists, JSONL paths,
      or platform dataset ids (``ds_...``). Evaluation prompts, answers
      and references are the texts compared (not the eval set's own
      replies).
    * ``fields`` (``("prompt",)``): which row texts are checked; prompts
      only is what Lambert 2025, chapter Evaluation, checks. Add
      ``"final_text"`` to ask the stricter question of
      whether replies reproduce eval answers or references.
    * ``n`` (8) and ``overlap`` (0.8): the near-copy rule, the Llama 2
      rule of 8-grams covering 80% of tokens. ``overlap=0`` restores
      any-n-gram.
    * ``embedder`` and ``similarity`` (0.85): a callable from a list of
      texts to one vector per text turns on the semantic rule at that
      cosine threshold; nothing here imports a model.

    Four rules, applied in this order, and a row flagged by one is not
    counted again by the next, so ``n_contaminated`` is the number of
    rows dropped:

    * ``same_task`` (``n_same_task``): the row's ``scenario_id`` or
      ``task_id`` is an evaluation row's. A task is a situation, not a
      string (``task_key``), so a rephrasing of an eval situation is the
      eval situation whatever the words say. It needs an id on both
      sides: when no evaluation row (or no training row) carries one,
      the rule does not run, ``rules_skipped["same_task"]`` says so, and
      only the text rules stand between the sets. Every eval set not
      written by ``simulate()`` (GSM8K, a Hub set, logged traces) is in
      that case, so read ``n_same_task: 0`` next to ``rules_skipped``.
    * ``exact`` (``n_exact``): one of the row's ``fields`` is an
      evaluation text verbatim after normalization (case and whitespace).
    * near copy (``n_near``): one evaluation text covers at least
      ``overlap`` of the row's words with shared word ``n``-grams. Texts
      shorter than ``n`` words match verbatim only.
    * ``semantic`` (``n_semantic``), only with ``embedder``: the cosine
      similarity between the row's text and an evaluation prompt is at
      least ``similarity``, and the two carry different task ids or none.
      It needs evaluation prompts to embed: when no evaluation row has a
      ``prompt``, the rule does not run, ``rules_skipped["semantic"]``
      says so, and a ``UserWarning`` is raised because you asked for it.

    One shared n-gram is the test Lambert 2025, chapter Evaluation, uses for
    free-form sets. Situations written from templates share whole sentences
    that say nothing about which question was asked, so any-n-gram flags every
    row of a template-written set; the coverage rule counts a row when one
    eval text accounts for most of it.

    Word overlap does not see a paraphrase. A holdout written by
    re-running the generator on the same briefs was 70% within 0.85
    cosine of the training batch and 5 of 133 byte-identical; the 8-gram
    rule flagged 4 of 101 prompts and the semantic pass 16. With
    sentence-transformers:

    ```python
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    clean, report = wai.decontaminate(
        train,
        against=[holdout],
        embedder=lambda texts: model.encode(texts, normalize_embeddings=True).tolist(),
    )
    ```

    A semantic flag means the two prompts read alike, not that they are
    the same task: "cancel one reservation" and "cancel three
    reservations" for different customers scored 0.932 with no shared
    answer. So where task identity is recorded the ``same_task`` rule
    decides and the semantic pass only looks across different tasks, and
    the report's ``notes`` say the flag is a question to check, not a
    verdict. The default stays lexical: ``similarity`` 0.85 was read off
    BGE (unrelated prompts score about 0.55 there) and does not transfer
    to every model, so the pass calibrates it for yours when it can. With
    eval rows that carry task ids, the 99th percentile of similarity over
    eval-prompt pairs with different task ids is how alike distinct tasks
    read to this embedder, and ``notes`` says it; a threshold below that
    number flags tasks that merely share a domain, and the note says so
    when ``similarity`` is.

    >>> train = [{"prompt": "Where is order 4473?"}, {"prompt": "Cancel order 9911."}]
    >>> clean, report = wai.decontaminate(train, against=[[{"prompt": "Cancel order 9911."}]])
    >>> len(clean), report["n_contaminated"]
    (1, 1)
    """
    if embedder is not None and not 0 <= float(similarity) <= 1:
        raise ValueError(
            f"similarity is a cosine threshold between 0 and 1 ({SEMANTIC_SIMILARITY} by default)"
        )
    sources = (
        against
        if isinstance(against, (list, tuple)) and not (against and isinstance(against[0], dict))
        else [against]
    )
    texts: dict[str, int] = {}
    n_eval = 0
    index: dict[tuple[str, ...], set[int]] = {}
    eval_tasks: set[str] = set()
    eval_prompts: dict[str, tuple[str, str | None]] = {}  # normalized -> (text, task id)
    for source in sources:
        for row in _load_rows(source):
            n_eval += 1
            task = _explicit_task(row)
            if task is not None:
                eval_tasks.add(task)
            prompt = str(row.get("prompt") or "")
            if prompt.strip():
                eval_prompts.setdefault(_norm(prompt), (prompt, task))
            for text in _eval_texts(row):
                words = _words(text)
                if not words or _norm(text) in texts:
                    continue
                tid = texts[_norm(text)] = len(texts)
                for gram in _ngrams(words, n):
                    index.setdefault(gram, set()).add(tid)
    threshold = max(0.0, min(1.0, float(overlap)))
    kept_index: list[int] = []
    flagged: list[dict[str, Any]] = []
    by_field: dict[str, int] = {}
    candidates: list[tuple[int, str, str]] = []  # (row index, field, text) for the semantic pass
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        hit: dict[str, Any] | None = None
        task = _explicit_task(row)
        if task is not None and task in eval_tasks:
            hit = {"field": "task", "match": "same_task", "coverage": 1.0, "task": task}
        for field in fields:
            if hit:
                break
            text = str(row.get(field) or "")
            words = _words(text)
            if not words:
                continue
            if _norm(text) in texts:
                hit = {"field": field, "match": "exact", "coverage": 1.0}
                break
            if len(words) < n:
                continue
            covered: dict[int, set[int]] = {}
            first: dict[int, int] = {}
            for start_i in range(len(words) - n + 1):
                for tid in index.get(tuple(words[start_i : start_i + n]), ()):
                    covered.setdefault(tid, set()).update(range(start_i, start_i + n))
                    first.setdefault(tid, start_i)
            if not covered:
                continue
            best = max(covered, key=lambda t: (len(covered[t]), -t))
            coverage = len(covered[best]) / len(words)
            if threshold <= 0 or coverage >= threshold:
                hit = {
                    "field": field,
                    "match": " ".join(words[first[best] : first[best] + n])[:120],
                    "coverage": round(coverage, 3),
                }
                break
        if hit:
            flagged.append({"index": i, **hit})
            by_field[hit["field"]] = by_field.get(hit["field"], 0) + 1
        else:
            kept_index.append(i)
            if embedder is not None:
                for field in fields:
                    text = str(row.get(field) or "")
                    if text.strip():
                        candidates.append((i, field, text))
    notes: list[str] = []
    rules_skipped: dict[str, str] = {}  # rule -> why it could not run on these inputs
    total = sum(1 for r in rows if isinstance(r, dict))
    n_train_ids = sum(1 for r in rows if isinstance(r, dict) and _explicit_task(r) is not None)
    if n_eval and not eval_tasks:
        rules_skipped["same_task"] = (
            f"0 of {n_eval} evaluation rows carried a scenario_id or task_id"
        )
    elif n_eval and total and not n_train_ids:
        rules_skipped["same_task"] = f"0 of {total} training rows carried a scenario_id or task_id"
    if "same_task" in rules_skipped:
        others = "the text rules" + (" and the semantic rule" if embedder is not None else "")
        notes.append(
            f"{rules_skipped['same_task']}, so the same_task rule was not applied and "
            f"n_same_task=0 says nothing about task overlap; only {others} ran. Every "
            "eval set not written by simulate() is in this case; pass embedder= to see "
            "paraphrases the text rules miss."
        )
    if embedder is not None and n_eval and not eval_prompts:
        rules_skipped["semantic"] = (
            f"none of the {n_eval} evaluation rows carried a prompt to embed"
        )
        line = (
            f"{rules_skipped['semantic']}, so the semantic rule you asked for with embedder= "
            "was not applied; n_semantic=0 is not a clearance."
        )
        notes.append(line)
        warnings.warn(f"decontaminate: {line}", UserWarning, stacklevel=2)
    n_semantic = 0
    if embedder is not None and eval_prompts:
        eval_norms = list(eval_prompts)
        eval_vecs = _embed(embedder, [eval_prompts[key][0] for key in eval_norms])
        eval_task_ids = [eval_prompts[key][1] for key in eval_norms]
        alike = _distinct_task_similarity(eval_vecs, eval_task_ids)
        if alike is not None:
            note = (
                f"with this embedder, distinct tasks read up to {alike:.2f} alike "
                f"({DISTINCT_TASK_PERCENTILE:.0%} percentile over {len(eval_norms)} eval "
                "prompts with different task ids); a "
                "threshold below that flags tasks that merely share a domain"
            )
            if float(similarity) <= alike:
                note += (
                    f". similarity={float(similarity)} is below it, so the semantic flags "
                    f"here include tasks that only share a domain; raise similarity= above "
                    f"{alike:.2f} to flag paraphrases only"
                )
            notes.append(note + ".")
        semantic: dict[int, dict[str, Any]] = {}
        cand_vecs = _embed(embedder, [text for _, _, text in candidates])
        for (i, field, _text), vec in zip(candidates, cand_vecs):
            if i in semantic:
                continue
            task = _explicit_task(rows[i])
            top: float = -1.0
            top_j = -1
            for j, evec in enumerate(eval_vecs):
                if task is not None and eval_task_ids[j] == task:
                    continue  # the same task is the same_task rule's call, made above
                sim = float(sum(x * y for x, y in zip(vec, evec)))
                if sim > top:
                    top, top_j = sim, j
            if top_j >= 0 and top >= float(similarity):
                semantic[i] = {
                    "index": i,
                    "field": field,
                    "match": "semantic",
                    "similarity": round(min(1.0, top), 4),
                    "eval_prompt": eval_prompts[eval_norms[top_j]][0][:120],
                }
        if semantic:
            kept_index = [i for i in kept_index if i not in semantic]
            for i in sorted(semantic):
                flagged.append(semantic[i])
                by_field[semantic[i]["field"]] = by_field.get(semantic[i]["field"], 0) + 1
            flagged.sort(key=lambda f: f["index"])
            n_semantic = len(semantic)
            notes.append(
                f"{n_semantic} row(s) read alike to an eval prompt (similarity >= "
                f"{float(similarity)}). That is a question about task identity, not a "
                "verdict: two prompts can read alike and be different tasks with different "
                "answers. Rows sharing a scenario_id or task_id with an eval row were dropped "
                "as same_task first; check the semantic ones before treating them as the same "
                "task, and raise similarity= if your embedder scores unrelated prompts high."
            )
    # A RowList, not a plain list: it keeps the system prompt and tools the
    # rows were generated under, so ``select(kept).export()`` writes them
    # (#592). ``rows`` may be a ScoredData, a RowList or a plain list.
    from ..data import RowList, row_config

    system, tools = row_config(rows)
    kept = RowList((rows[i] for i in kept_index), system_prompt=system, tools=tools)
    n_exact = sum(1 for f in flagged if f["match"] == "exact")
    n_same_task = sum(1 for f in flagged if f["match"] == "same_task")
    return kept, {
        "n": total,
        "n_kept": len(kept),
        "n_contaminated": len(flagged),
        "contamination_rate": (len(flagged) / total) if total else 0.0,
        # one count per rule; a row is under the first rule that caught it
        "n_same_task": n_same_task,
        "n_exact": n_exact,
        "n_near": len(flagged) - n_exact - n_same_task - n_semantic,
        "n_semantic": n_semantic,
        "n_eval_rows": n_eval,
        "n_eval_texts": len(texts),
        "ngram": n,
        "overlap": threshold,
        "similarity": float(similarity) if embedder is not None else None,
        "fields": list(fields),
        "by_field": by_field,
        "examples": flagged[:20],
        # rule -> why it could not run here; empty means every rule ran
        "rules_skipped": rules_skipped,
        "notes": notes,
    }


__all__ = [
    "DEFAULT_BOOT",
    "MIN_CI_TASKS",
    "MIN_PAIRED_TASKS",
    "MIN_RERUNS",
    "bootstrap_ci",
    "compare_runs",
    "decontaminate",
    "eval_variance",
    "marker_names",
    "marker_summary",
    "metric_summary",
    "noise_band",
    "task_means",
    "wilson_interval",
]
