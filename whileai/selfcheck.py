"""Check the statistics yourself: the comparison ``wai.compare`` runs, on agents whose truth is known.

    import whileai as wai

    print(wai.self_check())             # 10-15 s, 40 trials per check
    print(wai.self_check(trials=400))   # what `wai self-check --full` runs

Every verdict this package prints rests on one comparison: paired per-task
differences, a percentile bootstrap interval at ``1 - alpha``, and "moved"
only when that interval excludes zero (``stats.compare_runs``, which
``wai.compare`` calls for every metric). Whether that rule does what it says
is a question a simulation answers without trusting us: script two agents
whose pass chance on every case is set by hand, run the comparison many
times, and count.

* **False alarm.** Both arms are the same agent. The share of trials that
  say "moved" either way should be ``alpha`` (5%).
* **Coverage.** The after arm is better by a planted gain. The share of
  intervals that contain the gain should be ``1 - alpha`` (95%).
* **Power.** The planted gain at 30 and 60 cases. The share of trials that
  detect it should match the power ``holdout_size`` predicts for that size
  (Miller 2024, arXiv:2411.00640, section 5).
* **Near-duplicates.** Half the cases restate another case: same pass
  chance, separate task id, a fresh draw. Identical arms; the false-alarm
  rate should still be ``alpha``.

Each check reports its Monte Carlo error, ``sqrt(t * (1 - t) / trials)`` at
its target ``t``: the spread the trial count alone puts on the measured
rate. A check is flagged when it sits more than ``flag_at`` of those errors
on its bad side (above for a false alarm, below for coverage, either way for
power, where the target is a prediction). The first trial of every check is
also run through ``wai.compare`` itself and has to agree with the
comparison to the last digit, so the check is on the code path a user's
verdict comes from, not on a copy of it.

Reference: the scenario sizes follow the external case study that graded
the grader on whileai 0.126 (gentlyventures.com/casestudies/whileai);
Efron and Tibshirani 1993, chapter 13, on percentile-interval coverage at
small n.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from statistics import NormalDist
from typing import Any

from .report import Report
from .simulations.defaults import (
    ALPHA,
    BASE_PASS_RATE,
    BOOTSTRAP_DRAWS,
    SELF_CHECK_CASES,
    SELF_CHECK_COVERAGE_CASES,
    SELF_CHECK_DIFFICULTY,
    SELF_CHECK_DUPLICATE_SHARE,
    SELF_CHECK_EFFECTS,
    SELF_CHECK_FLAG_ERRORS,
    SELF_CHECK_FULL_TRIALS,
    SELF_CHECK_POWER_CASES,
    SELF_CHECK_SEED,
    SELF_CHECK_TRIALS,
)

__all__ = ["SelfCheckReport", "self_check"]

#: the keys of ``compare_runs`` that ``wai.compare`` must reproduce exactly
_SAME_PATH_KEYS = ("delta", "ci95", "p_value", "verdict", "n_paired")
#: verdicts ``compare_runs`` prints for an interval that excludes zero
_MOVED = frozenset({"b_better", "a_better"})

Rows = list[dict[str, Any]]
Arms = tuple[Rows, Rows]


def _arms(rng: random.Random, before: list[float], after: list[float], ids: list[str]) -> Arms:
    """One trial of two scripted agents: each case passes with its set chance, one try per arm."""
    a = [{"task_id": t, "reward": int(rng.random() < p)} for t, p in zip(ids, before)]
    b = [{"task_id": t, "reward": int(rng.random() < q)} for t, q in zip(ids, after)]
    return a, b


def _difficulties(rng: random.Random, n: int) -> list[float]:
    lo, hi = SELF_CHECK_DIFFICULTY
    return [rng.uniform(lo, hi) for _ in range(n)]


def _case_ids(n: int) -> list[str]:
    return [f"case-{i:03d}" for i in range(n)]


def _identical(n: int) -> Callable[[random.Random], Arms]:
    def draw(rng: random.Random) -> Arms:
        p = _difficulties(rng, n)
        return _arms(rng, p, p, _case_ids(n))

    return draw


def _shifted(n: int, effect: float) -> Callable[[random.Random], Arms]:
    def draw(rng: random.Random) -> Arms:
        p = _difficulties(rng, n)
        return _arms(rng, p, [x + effect for x in p], _case_ids(n))

    return draw


def _uniform_gain(n: int, effect: float) -> Callable[[random.Random], Arms]:
    """The sizing model's world: every case at ``BASE_PASS_RATE``, every case up by ``effect``."""

    def draw(rng: random.Random) -> Arms:
        return _arms(rng, [BASE_PASS_RATE] * n, [BASE_PASS_RATE + effect] * n, _case_ids(n))

    return draw


def _near_duplicates(n: int, share: float) -> Callable[[random.Random], Arms]:
    """Identical arms where ``share`` of the cases restate an earlier case."""
    n_dup = round(n * share)
    n_orig = n - n_dup

    def draw(rng: random.Random) -> Arms:
        p = _difficulties(rng, n_orig)
        ids = _case_ids(n_orig)
        for j in range(n_dup):
            p.append(p[j % n_orig])
            ids.append(f"{ids[j % n_orig]}-restated")
        return _arms(rng, p, p, ids)

    return draw


def _theory_power(n: int, effect: float, alpha: float) -> float:
    """The power ``holdout_size`` sizes with, at ``n`` cases and one try a side.

    ``n = ((z_{1-alpha/2} + z_power) * sd / effect) ** 2`` solved for power,
    with the SDK's own per-task sd (``stats._paired_task_sd``).
    """
    from .simulations.score.stats import _paired_task_sd

    sd = _paired_task_sd(BASE_PASS_RATE, effect, 1)
    z = NormalDist()
    return z.cdf(effect * math.sqrt(n) / sd - z.inv_cdf(1 - alpha / 2))


def _check_same_path(
    before: Rows, after: Rows, result: dict, seed: int, n_boot: int, alpha: float
) -> None:
    """Run one trial through ``wai.compare`` and require the identical answer."""
    from .simulations.score.delta import delta_report

    full = delta_report(before, after, target="pass_at_1", seed=seed, n_boot=n_boot, alpha=alpha)
    got = full["metrics"]["pass_at_1"]
    for key in _SAME_PATH_KEYS:
        if got.get(key) != result.get(key):
            raise AssertionError(
                f"wai.compare and stats.compare_runs disagree on {key} "
                f"({got.get(key)!r} vs {result.get(key)!r}); the self-check no longer runs the "
                "comparison a user's verdict comes from. Point self_check at the call "
                "wai.compare now uses."
            )


def _run(
    key: str,
    draw: Callable[[random.Random], Arms],
    hit: Callable[[dict], bool],
    *,
    trials: int,
    seed: int,
    n_boot: int,
    alpha: float,
) -> int:
    from .simulations.score.stats import compare_runs

    hits = 0
    for trial in range(trials):
        rng = random.Random(f"{seed}/{key}/{trial}")
        before, after = draw(rng)
        boot_seed = rng.randrange(2**31)
        result = compare_runs(before, after, n_boot=n_boot, seed=boot_seed, level=1 - alpha)
        if trial == 0:
            _check_same_path(before, after, result, boot_seed, n_boot, alpha)
        hits += hit(result)
    return hits


def _covers(effect: float) -> Callable[[dict], bool]:
    def hit(result: dict) -> bool:
        ci = result.get("ci95")
        return ci is not None and ci[0] <= effect <= ci[1]

    return hit


def _moved(result: dict) -> bool:
    return result.get("verdict") in _MOVED


def _detected(result: dict) -> bool:
    return result.get("verdict") == "b_better"


def self_check(
    *,
    trials: int = SELF_CHECK_TRIALS,
    seed: int = SELF_CHECK_SEED,
    n_boot: int = BOOTSTRAP_DRAWS,
    alpha: float = ALPHA,
    flag_at: float = SELF_CHECK_FLAG_ERRORS,
) -> SelfCheckReport:
    """Check the comparison behind every verdict by simulation, on scripted agents with known truth.

    Runs eight checks (false alarm at 30 cases, interval coverage at 10 and
    20, detection power at 30 and 60 cases for +10 and +16 points, false
    alarm with half the cases restated) through ``stats.compare_runs``, the
    comparison ``wai.compare`` runs, and returns a ``SelfCheckReport`` that
    prints one line per check: measured, target, Monte Carlo error, and
    ``OK`` or ``FLAG``. The module docstring says what each check plants.

    * ``trials`` (``SELF_CHECK_TRIALS``, 40): trials per check, 10 to 15
      seconds in all. ``SELF_CHECK_FULL_TRIALS`` (400) is the case study's
      count and what ``wai self-check --full`` runs, two to three
      minutes.
    * ``seed`` (``SELF_CHECK_SEED``, 0): every draw derives from it, so the
      same seed prints the same report on any machine.
    * ``n_boot`` (``BOOTSTRAP_DRAWS``, 2000) and ``alpha`` (``ALPHA``,
      0.05): passed to the comparison as a user's call passes them; the
      checks are of the defaults unless you change these.
    * ``flag_at`` (``SELF_CHECK_FLAG_ERRORS``, 3.0): Monte Carlo errors on
      the bad side before a check is flagged.

    ``report["ok"]`` is ``False`` when any check is flagged; the CLI exits 1.

    Reference: Miller 2024, arXiv:2411.00640, section 5 (the power
    formula); Efron and Tibshirani 1993, chapter 13 (percentile coverage).

    >>> report = wai.self_check(trials=2)
    >>> len(report["checks"])
    8
    """
    if int(trials) < 1:
        raise ValueError(
            f"trials is the number of simulated evals per check, at least 1; got {trials}"
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha is a false-positive rate strictly between 0 and 1; got {alpha}")
    if flag_at <= 0:
        raise ValueError(f"flag_at is a count of Monte Carlo errors, above 0; got {flag_at}")
    trials = int(trials)
    coverage = 1 - alpha
    cov_effect = SELF_CHECK_EFFECTS[0]
    specs: list[
        tuple[str, str, Callable[[random.Random], Arms], Callable[[dict], bool], float, str]
    ] = [
        (
            "false_alarm",
            f"false alarm, identical arms, {SELF_CHECK_CASES} cases",
            _identical(SELF_CHECK_CASES),
            _moved,
            alpha,
            "above",
        ),
    ]
    for n in SELF_CHECK_COVERAGE_CASES:
        specs.append(
            (
                f"coverage_{n}",
                f"95% range holds the true gain, {n} cases",
                _shifted(n, cov_effect),
                _covers(cov_effect),
                coverage,
                "below",
            )
        )
    for n in SELF_CHECK_POWER_CASES:
        for effect in SELF_CHECK_EFFECTS:
            points = round(effect * 100)
            specs.append(
                (
                    f"power_{n}_{points}",
                    f"detects +{points} points, {n} cases (theory)",
                    _uniform_gain(n, effect),
                    _detected,
                    _theory_power(n, effect, alpha),
                    "both",
                )
            )
    specs.append(
        (
            "near_duplicates",
            f"false alarm, {SELF_CHECK_DUPLICATE_SHARE:.0%} near-duplicate cases",
            _near_duplicates(SELF_CHECK_CASES, SELF_CHECK_DUPLICATE_SHARE),
            _moved,
            alpha,
            "above",
        )
    )
    checks = []
    for key, label, draw, hit, target, side in specs:
        hits = _run(key, draw, hit, trials=trials, seed=seed, n_boot=n_boot, alpha=alpha)
        measured = hits / trials
        mc_error = math.sqrt(target * (1 - target) / trials)
        z = (measured - target) / mc_error if mc_error > 0 else 0.0
        bad = {"above": z > flag_at, "below": z < -flag_at, "both": abs(z) > flag_at}[side]
        checks.append(
            {
                "check": key,
                "label": label,
                "hits": hits,
                "trials": trials,
                "measured": measured,
                "target": target,
                "mc_error": mc_error,
                "errors_off": z,
                "bad_side": side,
                "ok": not bad,
            }
        )
    return SelfCheckReport(
        ok=all(c["ok"] for c in checks),
        seed=seed,
        trials=trials,
        n_boot=n_boot,
        alpha=alpha,
        flag_at=flag_at,
        same_path=True,
        checks=checks,
    )


class SelfCheckReport(Report):
    """What ``wai.self_check`` measured: one entry per check under ``checks``.

    Each check carries ``measured`` (hits over trials), ``target``,
    ``mc_error`` (the binomial standard error at the target for this trial
    count), ``errors_off`` (measured minus target, in those errors),
    ``bad_side`` and ``ok``. ``same_path`` says the first trial of every
    check matched ``wai.compare`` exactly.
    """

    _summary_keys = ("ok", "trials", "seed")

    def __str__(self) -> str:
        width = max(len(c["label"]) for c in self["checks"])
        lines = [
            "self-check: the comparison wai.compare runs, on scripted agents with known truth",
            f"seed {self['seed']}, {self['trials']} trials per check, {self['n_boot']} bootstrap "
            f"draws, alpha {self['alpha']:g}",
            "",
            f"{'check':<{width}}  measured  target  MC error",
        ]
        for c in self["checks"]:
            word = "OK" if c["ok"] else "FLAG"
            lines.append(
                f"{c['label']:<{width}}  {c['measured']:>7.1%}  {c['target']:>6.1%}  "
                f"{c['mc_error']:>7.1%}  {word}"
            )
        flagged = [c["label"] for c in self["checks"] if not c["ok"]]
        lines.append("")
        lines.append(
            f"FLAG is more than {self['flag_at']:g} Monte Carlo errors on the bad side "
            "(above for false alarms, below for coverage, either way for power)."
        )
        lines.append(
            "Every check's first trial also ran through wai.compare and matched it exactly."
        )
        if flagged and self["trials"] < SELF_CHECK_FULL_TRIALS:
            lines.append(
                f"{len(flagged)} flagged: {'; '.join(flagged)}. Re-run with trials="
                f"{SELF_CHECK_FULL_TRIALS} (wai self-check --full) before reading a flag from "
                f"{self['trials']} trials; a flag that holds there is a finding."
            )
        elif flagged:
            lines.append(
                f"{len(flagged)} flagged: {'; '.join(flagged)}. At {self['trials']} trials a "
                "flag is a finding about the statistics, not noise in the check."
            )
        else:
            lines.append(f"all {len(self['checks'])} checks on target.")
        return "\n".join(lines)
