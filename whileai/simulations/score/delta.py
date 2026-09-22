"""Did training move the behavior, and did anything else slip?

One call over two graded, marker-scored row sets: the rollouts before a
training run and the rollouts after it, on the same tasks. Every metric
the two sets share (pass@1 and each marker) is compared as paired task
differences with a bootstrap interval (``stats.compare_runs``), so the
answer is "moved by X, interval Y" and not a pair of means.

Markers are read as higher-is-better unless ``lower_is_better`` says otherwise
(``lower_is_better=["words"]``, or ``{"truncated": False}`` to force a name the
other way); ``LOWER_IS_BETTER_MARKERS`` is the built-in set. On a metric where
down is the win the whole reading flips and the number does not: the delta,
the interval and the means stay the raw signed change ("47.5 words shorter"),
while the verdict, the printed tag, the warnings, ``must_not_regress``, the
proxy-vs-target check, the ``by=`` groups and ``ok`` all read a drop as the
gain. A metric named in ``must_not_regress`` whose interval sits entirely on
its bad side is a regression and fails the report; any other metric that moves
significantly the bad way is a warning (Lambert 2025, chapter Regularization:
post-training on one thing forgets others, and on-policy data forgets less,
which is only visible if you measure the others).

``proxy`` names the metric the run was trained on (the training reward, kept
on the rows as a marker) when it is not the target. Over-optimization is the
two curves parting (Gao et al. 2022, arXiv:2210.10760): the proxy moved up
while the target did not follow, or the proxy's interval sits entirely above
the target's. The report says so and fails.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist
from typing import Any

from ...report import Report
from ..defaults import ALPHA, BASE_PASS_RATE, CI_LEVEL, MIN_RERUNS, MIN_TRAIN_SEEDS, POWER
from .markers import STOCK_MARKERS
from .passat import answer_counts, pass_at
from .stats import (
    DEFAULT_BOOT,
    MIN_HOLDOUT_TASKS,
    UNPAIRED_MAJORITY_SHARE,
    _mean,
    _sample_sd,
    _t_quantile,
    _z_level,
    compare_runs,
    detectable_effect,
    eval_variance,
    holdout_size,
    marker_names,
    metric_summary,
    noise_band,
    task_key,
    task_means,
)

GROUP_KEYS = ("delta", "ci95", "verdict", "mean_a", "mean_b", "n_used", "n_paired", "paired")

# LOWER_IS_BETTER_MARKERS: the markers this library reads as down-is-the-win
# when the caller names no direction for them. The five stock behavioral
# markers are *presence* values -- 1.0 means the over-optimization tic showed
# up in the reply -- so a rise in one is the regression (Lambert 2025, chapter
# Over-optimization, the signatures RL against a judge drifts toward;
# ``score.markers`` stamps them at that polarity and its docstring says so).
# ``truncated`` is a completion the token cap cut, and a cut reply is a failed
# reply on any run, whichever way that run meant to go: convention, untested,
# and the same reading ``TRUNCATED_SHARE_GAP`` below already takes. The set
# applies only to a marker both row sets carry, and any name in it can be
# turned back the other way from the call
# (``lower_is_better={"refusal": False}``).
LOWER_IS_BETTER_MARKERS: tuple[str, ...] = (*STOCK_MARKERS, "truncated")

# ``compare_runs`` says which arm scored higher; these two words are the only
# ones that carry a direction, so reading a metric as lower-is-better is
# exactly swapping them. Nothing else about the metric changes: the delta, the
# interval and the means stay the raw signed change, so "47.5 words shorter,
# 95% [-50.169, -44.919]" is still the sentence the report prints (#638).
_FLIPPED_VERDICTS = {"b_better": "a_better", "a_better": "b_better"}


def _gain_verdict(verdict: str, lower_is_better: bool) -> str:
    """One ``compare_runs`` verdict read in goodness terms: ``b_better``
    means the after arm is better, whichever way its metric points."""
    return _FLIPPED_VERDICTS.get(verdict, verdict) if lower_is_better else verdict


def _gain_ci(result: Mapping[str, Any]) -> tuple[float, float] | None:
    """One metric's interval oriented so positive is better, for the one
    check that compares two metrics' intervals against each other
    (proxy against target). The reported ``ci95`` is never this."""
    ci = result.get("ci95")
    if not ci:
        return None
    lo, hi = float(ci[0]), float(ci[1])
    return (-hi, -lo) if result.get("lower_is_better") else (lo, hi)


def _directions(
    lower_is_better: Sequence[str] | Mapping[str, bool] | None,
    metrics: Sequence[str],
    key: Callable[[str], str],
) -> dict[str, bool]:
    """Which of ``metrics`` are read as down-is-the-win.

    A sequence names the metrics where lower is better; a mapping says it
    per name and can force one back to higher-is-better. Names come with
    or without the ``marker:`` prefix. ``LOWER_IS_BETTER_MARKERS`` fills in
    the rest. A name that is not one of the metrics measured here raises:
    a silently ignored direction is a report that reads backwards.
    """
    measured = set(metrics)
    asked: dict[str, bool] = {}
    if lower_is_better is not None:
        items = (
            [(str(n), bool(v)) for n, v in lower_is_better.items()]
            if isinstance(lower_is_better, Mapping)
            else [(str(n), True) for n in lower_is_better]
        )
        missing = sorted({n for n, _ in items if key(n) not in measured})
        if missing:
            raise ValueError(
                f"lower_is_better names {', '.join(repr(n) for n in missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not measured here. It takes metric "
                'names with or without the "marker:" prefix ("words" and "marker:words" both '
                'work) and "pass_at_1". Measured on both row sets: '
                f"{', '.join(sorted(measured))}. A marker counts only when both sides carry "
                "it, so stamp it on both arms (mark_rows / style_markers / the markers= your "
                "run writes) or drop the name."
            )
        asked = {key(n): value for n, value in items}
    default = {key(n) for n in LOWER_IS_BETTER_MARKERS}
    return {m: asked.get(m, m in default) for m in metrics}


# CEILING_PASS_RATE = 0.9: a before side passing this share of its tasks
# has at most 10 points of room, under the noise band of most agent evals
# (run_std 0.02-0.04 measured across our lanes gives a band of 0.06-0.11),
# so the report flags ``ceiling``. Convention on the exact share.
CEILING_PASS_RATE = 0.9
# CEILING_MIN_TASKS_WITH_ROOM = 20: below this many paired tasks that are
# not already passed every time (and when they are under half of the
# pairs), the same flag: 20 tasks at k=4 can prove a gain of about 0.2 at
# 80% power (``detectable_effect``), which is more than a training round
# moves. Convention on the count.
CEILING_MIN_TASKS_WITH_ROOM = 20
# CEILING_ROOM_SHARE = 0.5: the "under half" above.
CEILING_ROOM_SHARE = 0.5
# FAMILY_ERROR_MIN_METRICS = 4: metrics before the family-wise warning is
# worth a line. At 4 metrics ``1 - 0.95**4`` is 19%, the first count where
# the chance of one false flag is near one in five (convention).
FAMILY_ERROR_MIN_METRICS = 4
# DEGENERATE_CHECK_DRAWS = 10: bootstrap draws for the degenerate-guard
# probe, which only asks whether every row scored the same value and
# never reads the interval. Kept small so the probe is free.
DEGENERATE_CHECK_DRAWS = 10
# GROUP_SEED_OFFSET = 100: the per-group comparisons draw from ``seed +
# GROUP_SEED_OFFSET + i`` so they never share a stream with the per-metric
# comparisons at ``seed + i`` (a headline and a group would otherwise
# resample identically). Any offset past the metric count works.
GROUP_SEED_OFFSET = 100
# TRUNCATED_SHARE_GAP = 0.05: the two arms' token-cap cut shares may
# differ by this before the report warns that one side was cut more
# often. Five points is the smallest gap that has moved a pass rate on
# our lanes (a cut reply is a failed reply). Convention, untested.
TRUNCATED_SHARE_GAP = 0.05
# GRADED_SHARE_WARN = 0.95: below this share of graded rows on either side
# the report says the rates are over survivors. One in twenty rows lost
# to the judge is where the selection bound (dropped / (1 - dropped))
# passes 5 points, the size of a typical training gain. Convention.
GRADED_SHARE_WARN = 0.95

_VERDICT_WORDS = {
    "b_better": "moved",
    "a_better": "moved_the_wrong_way",
    "no_difference_detected": "no_change_detected",
    "insufficient_data": "insufficient_data",
}

# The first line of ``format_delta_report``: the headline verdict in the
# words a person reads. ``PASS`` is reserved for a gain the report
# supports; a delta the interval cannot distinguish from zero reads as
# what it is, not as a pass (a negative point estimate under ``PASS``
# was the 2026-09-18 live test).
_HEADLINE_WORDS = {
    "moved": "PASS",
    "moved_unreplicated": "PASS",
    "within_eval_noise": "NO DIFFERENCE (within eval noise)",
    "no_change_detected": "NO DIFFERENCE",
    "insufficient_data": "INSUFFICIENT DATA",
    "moved_the_wrong_way": "FAIL",
    "target_not_measured": "TARGET NOT MEASURED",
    "unresolved": "UNRESOLVED (one training seed per arm)",
}

# The sentence a one-seed report prints, verbatim, wherever the verdict is
# ``unresolved`` (#356): the table, the warning and the headline say it.
UNRESOLVED_LINE = "one training seed per arm; add a seed to resolve"


def headline_word(report: Mapping[str, Any]) -> str:
    """The one-word reading of a ``delta_report``: ``NOT COMPARABLE``
    with the causes when the arms cannot be compared, ``FAIL`` when a
    guard failed, else the headline verdict spelled out (``PASS`` only
    for a supported gain, ``NO DIFFERENCE`` for an interval over zero)."""
    causes = list(report.get("not_comparable") or [])
    gate = "" if report.get("ok") else "FAIL: "
    if causes:
        return f"{gate}NOT COMPARABLE ({', '.join(causes)})"
    if gate:
        return "FAIL"
    verdict = report.get("headline_verdict")
    return _HEADLINE_WORDS.get(str(verdict), "PASS")


def _verdict_word(result: dict[str, Any], replicated: bool) -> str:
    """One metric's verdict as the report says it: ``moved`` only when
    the eval was run more than once per side (or a ``run_std`` was
    given), else ``moved_unreplicated``; inside the re-run band,
    ``within_eval_noise``. Read off ``gain_verdict``, so a metric where
    down is the win reads ``moved`` for a drop."""
    word = _VERDICT_WORDS[result.get("gain_verdict") or result["verdict"]]
    if result.get("within_noise") and word in {"moved", "moved_the_wrong_way"}:
        return "within_eval_noise"
    if word == "moved" and not replicated:
        return "moved_unreplicated"
    return word


def _eval_runs(rows: Sequence[dict]) -> set[str]:
    """The distinct ``lineage.eval_run`` values on the rows (what
    ``simulate(runs=N)`` stamps)."""
    out: set[str] = set()
    for row in rows:
        lineage = row.get("lineage") if isinstance(row, dict) else None
        if isinstance(lineage, dict) and lineage.get("eval_run") is not None:
            out.add(str(lineage["eval_run"]))
    return out


def _pooled_run_std(before: Sequence[dict], after: Sequence[dict], metric: str) -> float | None:
    """The eval's re-run standard deviation from both sides' repeats:
    ``eval_variance`` per side, pooled by degrees of freedom (each side's
    variance weighted by its runs minus one), since each side is the same eval
    on one model (Lambert 2025, evaluation-variance appendix). The pooled
    estimate has ``sum(n_i - 1)`` degrees of freedom, which is what
    ``noise_band`` widens for."""
    variances = []
    for rows in (before, after):
        side = eval_variance(rows, metric=metric, by="eval_run")
        if side["run_std"] is None or int(side["n_runs"]) < 2:  # noqa: PLR2004  # two runs before a run std exists
            return None
        variances.append((float(side["run_std"]) ** 2, int(side["n_runs"]) - 1))
    df = sum(w for _, w in variances)
    return (sum(v * w for v, w in variances) / df) ** 0.5


def _band_args(
    eval_runs: dict[str, int], run_std_source: str | None, run_std_runs: int | None = None
) -> tuple[int, int, int | None]:
    """What ``noise_band`` needs: how many runs each side's mean averages
    over, and the degrees of freedom behind ``run_std``: ``sum(runs - 1)``
    when the report estimated it from the two sides' runs itself,
    ``run_std_runs - 1`` when a given ``run_std`` came with the number of
    re-runs behind it, ``None`` for a bare given ``run_std``, which is
    taken as the eval's spread."""
    n_a, n_b = max(1, int(eval_runs["before"])), max(1, int(eval_runs["after"]))
    if run_std_source == "eval_run":
        return n_a, n_b, (n_a - 1) + (n_b - 1)
    if run_std_source == "given" and run_std_runs is not None:
        return n_a, n_b, int(run_std_runs) - 1
    return n_a, n_b, None


def _band_rule(n_a: int, n_b: int, df: int | None, level: float = CI_LEVEL) -> str:
    """The band as a reader can check it: the quantile, then the run counts."""
    q = f"{_z_level(level):.2f}" if df is None else f"t(df={df})={_t_quantile(df, level):.2f}"
    return f"{q} x run_std x sqrt(1/{n_a} + 1/{n_b})"


TrainRuns = Mapping[str, "Sequence[Sequence[dict]] | None"] | Sequence[Sequence[dict]]


def _train_arms(train_runs: TrainRuns) -> dict[str, list[Sequence[dict]] | None]:
    """``train_runs`` as ``{"before": seeds | None, "after": seeds | None}``:
    a plain list is the after arm's seeds against an untrained before arm;
    a mapping names both arms, ``None`` for an arm that was not trained."""
    if isinstance(train_runs, Mapping):
        if set(train_runs) != {"before", "after"}:
            raise ValueError(
                "train_runs maps 'before' and 'after' to one row set per training seed (None "
                f"for an untrained arm); got keys {sorted(map(str, train_runs))}"
            )
        arms = {side: train_runs[side] for side in ("before", "after")}
    else:
        arms = {"before": None, "after": train_runs}
    out: dict[str, list[Sequence[dict]] | None] = {}
    for side, seeds in arms.items():
        if seeds is None:
            out[side] = None
            continue
        listed = list(seeds)
        if not listed:
            raise ValueError(
                f"train_runs[{side!r}] is empty: pass one row set per training seed, or None "
                "for an arm that was not trained"
            )
        if any(isinstance(s, dict) for s in listed):
            raise ValueError(
                f"train_runs[{side!r}] is a list of rows; it takes a list of row sets, one per "
                "training seed (the rows passed as before/after are one of them)"
            )
        out[side] = listed
    if all(v is None for v in out.values()):
        raise ValueError("train_runs names no trained arm; pass the seeds of at least one side")
    return out


def _train_spread(
    arms: dict[str, list[Sequence[dict]] | None],
    before: Sequence[dict],
    after: Sequence[dict],
    metric: str,
    headline: Mapping[str, Any],
    level: float,
) -> dict[str, Any]:
    """The between-seed term of a delta between two trained models.

    Each trained arm's mean of ``metric`` is a draw from the training
    seed's distribution, so the delta's variance is the task-sampling
    variance the paired interval already carries plus the between-seed
    variance of each arm's mean, ``std_arm**2 / n_arm`` (Lambert 2025,
    chapter Evaluation; Miller 2024, arXiv:2411.00640, the variance
    components an eval claim rests on). The interval is centred on the
    across-seed delta and widened in quadrature: half-width ``sqrt(hw_task**2
    + t(df)**2 * sum(std**2 / n))`` with ``df = sum(n_arm - 1)``, the t
    quantile because each ``std`` is an estimate from ``n_arm`` seeds. An
    untrained arm (``None``) has no between-seed term. An arm with fewer
    than ``MIN_TRAIN_SEEDS`` seeds gives no interval: the report is
    ``unresolved``.
    """
    counts: dict[str, int | None] = {}
    stds: dict[str, float | None] = {}
    means: dict[str, float | None] = {}
    variance = 0.0
    df = 0
    for side, rows in (("before", before), ("after", after)):
        seeds = arms[side]
        if seeds is None:
            counts[side], stds[side] = None, None
            per_task = task_means(rows, metric)
            means[side] = _mean(list(per_task.values())) if per_task else None
            continue
        per_seed = [_mean(list(tm.values())) for s in seeds if (tm := task_means(s, metric))]
        counts[side] = len(seeds)
        means[side] = _mean(per_seed) if per_seed else None
        std = _sample_sd(per_seed) if len(per_seed) >= MIN_TRAIN_SEEDS else None
        stds[side] = std
        if std is not None:
            variance += std**2 / len(per_seed)
            df += len(per_seed) - 1
    short = [s for s, n in counts.items() if n is not None and n < MIN_TRAIN_SEEDS]
    out: dict[str, Any] = {
        "counts": counts,
        "std": stds,
        "means": means,
        "df": df if not short else None,
        "under_replicated": bool(short),
        "delta": None,
        "ci95": None,
        "rule": None,
    }
    ci = headline.get("ci95")
    if short or not ci or means["before"] is None or means["after"] is None:
        return out
    q = _t_quantile(df, level)
    half = math.sqrt(((ci[1] - ci[0]) / 2) ** 2 + q * q * variance)
    centre = means["after"] - means["before"]
    terms = " + ".join(
        f"seed_std_{s}^2/{counts[s]}" for s in ("before", "after") if stds[s] is not None
    )
    out["delta"] = centre
    out["ci95"] = (centre - half, centre + half)
    out["rule"] = f"t(df={df})={q:.2f} x sqrt({terms}), added in quadrature to the task interval"
    return out


def _group_of(row: dict, by: str | Callable[[dict], Any]) -> str | None:
    if callable(by):
        value = by(row)
    else:
        value = row.get(by)
        if value is None and isinstance(row.get("markers"), dict):
            value = row["markers"].get(by)
    if value is None or value == "":
        return None
    return str(value)


def _by_group(
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    by: str | Callable[[dict], Any],
    metric: str,
    n_boot: int,
    seed: int,
    level: float = CI_LEVEL,
    lower_is_better: bool = False,
) -> dict[str, dict[str, Any]]:
    """The target metric compared within each group of rows. A group needs
    rows on both sides; rows with no group value are left out. Each group
    carries the metric's direction, so a split of a down-is-the-win target
    reads the same way its headline does."""
    groups_a: dict[str, list[dict]] = {}
    groups_b: dict[str, list[dict]] = {}
    for row in before:
        g = _group_of(row, by)
        if g is not None:
            groups_a.setdefault(g, []).append(row)
    for row in after:
        g = _group_of(row, by)
        if g is not None:
            groups_b.setdefault(g, []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(sorted(set(groups_a) & set(groups_b))):
        r = compare_runs(
            groups_a[name],
            groups_b[name],
            metric=metric,
            n_boot=n_boot,
            seed=seed + GROUP_SEED_OFFSET + i,
            level=level,
        )
        slim: dict[str, Any] = {k: r.get(k) for k in GROUP_KEYS}
        slim["rows_a"] = len(groups_a[name])
        slim["rows_b"] = len(groups_b[name])
        slim["lower_is_better"] = lower_is_better
        slim["gain_verdict"] = _gain_verdict(str(r.get("verdict")), lower_is_better)
        out[name] = slim
    return out


# ANSWERED_GAP_POINTS = 0.10: with no re-run band to read the gap against,
# a difference in answered share this large between the arms fails the
# comparison on its own. Ten points is above the widest re-run band
# measured on our lanes (0.11 at run_std 0.04), so a gap over it cannot
# be re-run noise. Convention on the exact number (#297).
ANSWERED_GAP_POINTS = 0.10
# ANSWERED_P_MAX = 0.01: the two-proportion z test has to clear this
# before a gap counts at all. Stricter than ALPHA because the test runs
# on every report without being asked for, and a false "not comparable"
# blocks a reader from a real delta; at 0.01 it fires on chance once in a
# hundred reports (convention).
ANSWERED_P_MAX = 0.01


def _two_proportion_p(x_a: int, n_a: int, x_b: int, n_b: int) -> float | None:
    """Two-sided p-value of the pooled two-proportion z test: is the share
    ``x_a/n_a`` different from ``x_b/n_b``? ``None`` when either side has
    no rows or the pooled share is 0 or 1 (no variance to test against)."""
    if n_a <= 0 or n_b <= 0:
        return None
    pooled = (x_a + x_b) / (n_a + n_b)
    if pooled <= 0.0 or pooled >= 1.0:
        return None
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    z = abs(x_a / n_a - x_b / n_b) / se
    return 2.0 * (1.0 - NormalDist().cdf(z))


def _answered_note(
    cfg_a: dict[str, Any], cfg_b: dict[str, Any], *, p: float, gap: float, bar: str, fails: bool
) -> str:
    """The warning for arms that differ in how often they answered at all:
    the shares, the test, the mechanism that produces it, and the fix."""
    missing_a = 1.0 - float(cfg_a["answered_share"])
    missing_b = 1.0 - float(cfg_b["answered_share"])
    head = "NOT COMPARABLE: " if fails else ""
    line = (
        f"{head}{missing_a:.0%} of before rows and {missing_b:.0%} of after rows have no spoken "
        f"reply (two-proportion test p={p:.2g}, gap {gap:.1%} against {bar}), so every rate above "
        "is computed over replies one side did not produce. "
    )
    think_a = float(cfg_a.get("unclosed_think_share") or 0.0)
    think_b = float(cfg_b.get("unclosed_think_share") or 0.0)
    cut_a = float(cfg_a.get("truncated_share") or 0.0)
    cut_b = float(cfg_b.get("truncated_share") or 0.0)
    if think_a or think_b:
        line += (
            f"{think_a:.0%} of before and {think_b:.0%} of after replies end inside an unclosed "
            "<think>: a reasoning base compared against a reasoning-suppressed adapter (one "
            "trained on think-free targets) under one shared max_tokens spends the budget "
            "reasoning and never answers, while the adapter answers at once. "
        )
    elif cut_a or cut_b:
        line += (
            f"The token cap cut {cut_a:.0%} of before and {cut_b:.0%} of after rows, which "
            "is what a reasoning base does against a reasoning-suppressed adapter under one "
            "shared max_tokens: it spends the budget reasoning and never answers. "
        )
    else:
        # No reasoning markup and no cap cut on either side: the short
        # side stopped before it spoke for another reason (a turn budget
        # that ran out on a tool call reads ``tool`` in finish_reason).
        return line + (
            "Neither side shows <think> markup or a token-cap cut, so read finish_reason per "
            "side (a side that stops on a tool call reads tool) and fix that side before "
            "reading the delta."
        )
    return line + (
        "Raise agent_max_tokens= on both sides, set thinking= the same on both arms, or "
        "strip <think> on both, and re-run before reading the delta."
    )


def delta_report(
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    target: str | None = None,
    must_not_regress: Sequence[str] = (),
    lower_is_better: Sequence[str] | Mapping[str, bool] | None = None,
    markers: Sequence[str] | None = None,
    by: str | Callable[[dict], Any] | None = None,
    run_std: float | Mapping[str, float | None] | None = None,
    run_std_runs: int | None = None,
    train_runs: TrainRuns | None = None,
    proxy: str | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    balance_rollouts: bool = False,
    alpha: float = ALPHA,
    power: float = POWER,
    ceiling_pass_rate: float = CEILING_PASS_RATE,
    answered_gap_points: float = ANSWERED_GAP_POINTS,
    answered_alpha: float = ANSWERED_P_MAX,
) -> DeltaReport:
    """Compare an ``after`` run to a ``before`` run on pass@1 and every shared marker, and say whether the change is real.

    Reach for it after a change (a prompt edit, a trained adapter, a model
    swap): both sides are graded rows, ideally on the same pinned tasks
    (``simulate(tasks=before)``) with the same rollouts per task. It
    returns a ``DeltaReport``, a dict that prints itself. The keys a
    caller reads first: ``headline_verdict``
    (``PASS`` only for a gain the report supports, ``NO DIFFERENCE`` for
    an interval over zero, ``NOT COMPARABLE (causes)`` when the arms
    cannot be compared, ``FAIL`` for a regression, a failed guard, or
    over-optimization), ``ok`` (the gate: no regression, no failed guard,
    comparable arms; it does not say the change helped), ``metrics`` (one
    entry per metric with its delta, interval and verdict), ``warnings``
    (each naming its fix), ``not_comparable``, ``n_paired_tasks`` and
    ``n_unpaired_tasks``. ``print(report)`` writes it with
    ``headline_verdict`` on the first line
    (``format_delta_report(report)`` is the same string).

    Arguments that matter:

    * ``target``: the metric the run was meant to move (``"pass_at_1"`` or
      ``"marker:name"``); its verdict is the headline.
    * ``proxy``: the metric the run was actually trained on (the training
      reward as a marker, such as ``"marker:first_action"``). When the
      proxy moved up and the target did not, or the proxy's interval sits
      entirely above the target's, the report is ``over_optimized`` and
      fails: the policy learned something the target does not credit
      (Gao et al. 2022, arXiv:2210.10760).
    * ``must_not_regress``: metrics whose significant move the bad way
      fails the report. Marker metrics go by marker name; pass@1 is
      ``"pass_at_1"``.
    * ``lower_is_better``: the metrics whose *drop* is the win -- reply
      length, tokens, cost, latency, turns, retries, escalations. Names go
      with or without the ``marker:`` prefix
      (``lower_is_better=["words", "latency_ms"]``), or as a mapping when
      one has to be forced back the other way
      (``{"words": True, "truncated": False}``). A name that is not
      measured on both row sets raises rather than being ignored. See the
      direction paragraph below.
    * ``by``: split the target by a group on each row (a top-level row
      key, a marker name, or a callable ``row -> group``). The report
      gains ``groups``, the target compared within each, so a headline
      that moved cannot hide a kind of prompt that moved the other way. A
      group whose target dropped significantly is listed in
      ``groups_down`` and warned about; it does not flip ``ok``, which
      stays the ``must_not_regress`` contract (name the group's metric
      there if it should).
    * ``run_std`` and ``run_std_runs``: the evaluation's own re-run
      standard deviation, per metric or as one number, and how many
      re-runs it was computed from. See the noise floor below.
    * ``train_runs``: the rows of every independent training seed of each
      arm, when the two sides are separately trained models: a list of
      row sets for the after arm (the before arm untrained), or
      ``{"before": [...], "after": [...]}`` with ``None`` for an arm that
      was not trained. See training seeds below.
    * ``alpha`` (0.05): the false-positive rate every verdict runs at.
      Each interval is at ``1 - alpha`` (``ci95`` at the default), the
      re-run band uses the same quantile, and ``family_error`` is
      ``1 - (1 - alpha) ** n_metrics``. ``power`` (0.8) feeds the sizing
      line (``detectable_effect``, ``holdout_size``). ``tasks_needed``
      is sized from the task sd measured on the paired rows in hand
      (``holdout_size(before=, after=)``), and ``tasks_needed_source``
      says so (``"rows"``); the binomial model, which cannot see the
      covariance pairing buys, asked for about twice the tasks (#733).
    * ``balance_rollouts`` (off): trim every paired task to the rows both
      sides have, chosen by ``seed``, so pass^k and pass@k share one k;
      ``balanced`` says how many rows each side gave up.
    * ``ceiling_pass_rate`` (``CEILING_PASS_RATE``, 0.9),
      ``answered_gap_points`` (``ANSWERED_GAP_POINTS``, 0.1) and
      ``answered_alpha`` (``ANSWERED_P_MAX``, 0.01): the thresholds of the
      ``ceiling`` and ``answered`` flags below.

    Direction. A metric is higher-is-better unless ``lower_is_better``
    or ``LOWER_IS_BETTER_MARKERS`` says otherwise, and the direction is
    an *interpretation*, never an edit to the number: ``delta``, ``ci95``,
    ``mean_a`` and ``mean_b`` stay the raw signed change, so a run that cut
    replies from 198.1 words to 150.6 still reports ``-47.500`` with
    ``95% -50.169..-44.919`` and the operator keeps the effect size. What
    flips is every reading of it: each metric gains ``gain_verdict``
    (``b_better`` whenever the after arm is the better one), and that is
    what ``target_verdict``, ``headline_verdict``, ``ok``, ``improved``,
    ``slipped``, ``regressions``, ``must_not_regress``, the
    over-optimization check, the ``by=`` groups and the warnings read. The
    printed line shows the raw delta with the tag in the same
    lower-case-is-good spelling it always used (``down`` for a win on a
    down-is-better metric, ``UP`` for the slip) and says ``lower is
    better`` next to it; ``report["lower_is_better"]`` lists the metrics
    read that way. ``LOWER_IS_BETTER_MARKERS`` is the built-in set: the
    five stock presence markers (``boilerplate``, ``self_reference``,
    ``hedging``, ``refusal``, ``sycophancy``, 1.0 = the tic appeared) and
    ``truncated``; pass any of them as ``{"name": False}`` if your rows
    carry that name at the other polarity.

    Pairing. Tasks pair by the key ``pass_at`` groups on; tasks on one
    side only do not pair, their count is ``n_unpaired_tasks``, and when
    any were dropped a warning says so. ``situations`` is the cause in
    ``not_comparable`` when fewer than half the tasks are on both sides
    (``paired_share`` under 0.5 with tasks on one side only): the arms
    drew different situation sets, so the delta over the few that pair is
    between two evals, and the fix is to pin the after side to the before
    run's tasks (``tasks=``) or compare per tier with ``dataset_report``.

    Unequal rollouts. When a run lost rollouts
    (``data.report()["rollouts_lost"]``), one arm can sit at k=4 and the
    other at k=2; the report warns, next to the sizing line, naming both.
    Unequal k is a precision issue, not a bias: a task's pass rate is its
    mean over however many rows it has, so rows lost at random leave the
    paired delta unbiased and only widen its interval (simulated, k=4
    against k=2 on half the tasks: mean delta on the true value, interval
    about 10% wider). Rows lost for a reason are the problem: a timeout
    that takes the hard runs, an empty reply on the long ones, and the
    surviving rows on that arm score higher than the arm does. No
    trimming fixes that; ``balance_rollouts`` costs precision (another
    10% on the interval in the same simulation) and removes no bias
    (failures dropped on one arm: delta 0.32 untrimmed, 0.32 trimmed,
    true 0.05). Only re-running the short arm on its short tasks does,
    and ``data.report()["rollouts_lost_by"]`` says why the rows went
    missing.

    Noise floor. One evaluation is a draw, not a distribution (Lambert
    2025, chapter Evaluation, "why many comparisons are unreliable", and
    its evaluation-variance appendix). With
    one run on either side and no ``run_std``, a target that moved reads
    ``moved_unreplicated`` and a warning says how to fix it. Pass
    ``eval_variance(...)["run_std_by_metric"]`` as ``run_std`` so pass@1
    and each marker are judged against their own floor: a marker on a
    subset of tasks is several times noisier than pass@1, and pass@1's
    floor reads a re-run draw of it as a regression. A scalar applies one
    floor to every metric. A metric the mapping lacks, or carries as
    ``None``, is never given another metric's floor: it gets
    ``noise_note: "no_replicate_floor"``, a warning, and its verdict
    rests on the task interval alone. A metric whose delta is inside
    ``noise_band(floor, n_a, n_b, df)`` is ``within_noise``: not improved,
    not slipped, not a regression, and a target there reads
    ``within_eval_noise`` rather than moved, because re-running the eval
    moves it that much on its own. The band is ``floor * sqrt(1/n_a +
    1/n_b)`` (the delta is a mean of ``n_a`` runs against a mean of
    ``n_b``) times 1.96 for a given floor, which is taken as the eval's
    spread. A floor that came from re-runs is an estimate, not the
    spread: pass ``run_std_runs`` (``eval_variance(...)["n_runs"]``) and
    the band uses the two-sided t quantile at ``df = run_std_runs - 1``
    instead (three re-runs: 4.30 x floor x sqrt(2) with one run per side,
    not 1.96; under pure noise the 1.96 band lets about one delta in five
    through at df=2). A given ``run_std`` without ``run_std_runs`` keeps
    1.96 and a warning names the fix. When both row sets carry two or
    more ``lineage.eval_run`` values (``simulate(tasks=..., runs=3)``) the
    report computes each metric's floor itself, pooled over the two
    sides, and uses the t quantile at ``df = sum(runs - 1)`` (three runs
    per side: 2.78 x floor x sqrt(2/3)); ``run_std`` is then the headline
    metric's floor, ``run_std_by_metric`` has them all, ``noise_band`` is
    the headline band, ``noise_rule`` spells it out, and ``eval_runs``
    says how many runs each side had. An arm handed in through
    ``train_runs`` as N row sets averages N eval draws, so ``eval_runs``
    counts those too (three seeds a side: ``sqrt(1/3 + 1/3)``); the
    band used to read only the lineage and came out 1.73x too wide
    for three seeds (#750).

    Training seeds. The noise floor measures the eval; a delta between
    two separately trained models also carries training variance, which
    the floor cannot see (#356: one recipe read -0.065 [-0.117, -0.013]
    on one run and +0.050 on the next, at one seed per arm). Pass
    ``train_runs`` and the headline metric gains a between-seed term: each
    trained arm's per-seed means give a between-seed standard deviation
    ``train_std[arm]``, the delta's variance adds ``std**2 / n_seeds`` per
    arm, and ``train_ci95`` is the interval centred on the across-seed
    delta ``train_delta`` and widened in quadrature by the two-sided t
    quantile at ``train_df = sum(n_seeds - 1)`` (Lambert 2025, chapter
    Evaluation; Miller 2024, arXiv:2411.00640, on the variance components
    a claim rests on). ``moved`` then needs that interval to exclude zero
    as well; when it covers zero the verdict is ``no_change_detected`` and
    a warning says the seed spread ate the delta. Fewer than
    ``MIN_TRAIN_SEEDS`` (2) seeds on a trained arm resolves nothing: the
    verdict is ``unresolved``, the interval and floor lines still print,
    and the line says "one training seed per arm; add a seed to resolve".
    Without ``train_runs`` the report says nothing about training seeds
    (a prompt edit or a model swap has none); the paper-recipe contract
    (``recipes/papers/check.py``) reads a one-seed delta as unresolved.

    Comparability. ``config`` says what each side was produced with
    (``pass_at(...).config`` per side: task count, k, temperature, max_tokens,
    policy and judge versions, prompt hash). A warning names each setting the
    two sides disagree on, and says so when both sides are the same policy
    version (Lambert 2025, chapter Evaluation: a comparison is only as good as
    the settings it was run under). ``config[side]["answered_share"]`` is the
    share of rows per side with a spoken reply once ``<think>`` markup is
    gone, and every rate is conditional on it. The two shares are compared
    with a pooled two-proportion z test; when it clears ``answered_alpha`` the
    warning states p and the gap, and when the gap also exceeds the re-run
    band (or ``answered_gap_points`` with no band) the report fails with
    ``answered`` in ``not_comparable`` and names the mechanism: a reasoning
    base against a reasoning-suppressed adapter under one shared
    ``max_tokens`` runs out of budget inside ``<think>`` and never answers, so
    the adapter wins every row the base did not reply to. ``not_comparable``
    lists every such cause under one prefix, ``NOT COMPARABLE:``; none are
    raised here. A replay (``simulate(tasks=...)`` or ``runs=N``) keeps the
    writer of the run it replays on ``writer_model``, so two runs of one call
    compare as one writer. Situations nobody's model wrote (a ``seeds=`` ask,
    the offline template writer, or a replay of either) count as one writer
    for this check: nothing there could have moved with the weights.

    ``ceiling`` is set when the before side already passes
    ``ceiling_pass_rate`` of its tasks, or when fewer than
    ``CEILING_MIN_TASKS_WITH_ROOM`` paired tasks (and under half) are not
    already passed every time: there is little room left for an
    improvement to show, whatever the training did.

    ```python
    report = wai.delta_report(base.rows(), tuned.rows(), target="pass_at_1")
    print(wai.format_delta_report(report))
    ```
    """
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power are probabilities strictly between 0 and 1")
    if run_std_runs is not None:
        if run_std is None:
            raise ValueError(
                "run_std_runs says how many re-runs run_std came from; pass run_std with it"
            )
        if int(run_std_runs) < 2:  # noqa: PLR2004  # two runs before a standard deviation exists
            raise ValueError(
                "run_std_runs is the number of re-runs run_std was computed from, at least 2"
            )
        run_std_runs = int(run_std_runs)
    train_arms = _train_arms(train_runs) if train_runs is not None else None
    level = 1.0 - alpha
    balanced: dict[str, Any] | None = None
    if balance_rollouts:
        before, after, balanced = _balance_rollouts(before, after, seed=seed)
    names = (
        list(markers)
        if markers is not None
        else sorted(set(marker_names(before)) & set(marker_names(after)))
    )
    metrics = ["pass_at_1", *[f"marker:{m}" for m in names]]
    results: dict[str, dict[str, Any]] = {}
    for i, metric in enumerate(metrics):
        results[metric] = compare_runs(
            before, after, metric=metric, n_boot=n_boot, seed=seed + i, level=level
        )

    def _key(name: str) -> str:
        return name if name == "pass_at_1" or name.startswith("marker:") else f"marker:{name}"

    # Direction first: every verdict below is read off ``gain_verdict``, so
    # the metric that a run set out to reduce is a gain everywhere at once
    # and nowhere twice (#638). The raw delta and interval are untouched.
    lower = _directions(lower_is_better, metrics, _key)
    for m in metrics:
        results[m]["lower_is_better"] = lower[m]
        results[m]["gain_verdict"] = _gain_verdict(results[m]["verdict"], lower[m])

    guarded = {_key(m) for m in must_not_regress}
    target_key = _key(target) if target else None
    degenerate_guards: list[str] = []
    for m in sorted(guarded):
        if m not in results:
            continue
        sides = [metric_summary(rows, m, n_boot=DEGENERATE_CHECK_DRAWS) for rows in (before, after)]
        if all(s.get("degenerate") for s in sides):
            degenerate_guards.append(m)
    # Runs per side for the band: the ``lineage.eval_run`` values on the
    # rows (``simulate(runs=N)``), or the row sets ``train_runs`` names,
    # whichever is more. Each training seed's rows are their own eval
    # draw, so an arm given as three seeds averages three runs and its
    # band is ``sqrt(2/3)``, not ``sqrt(2)``; reading only the lineage
    # made it 1.73x too wide (#750). Rows a recipe built itself carry no
    # lineage at all.
    eval_runs = {
        side: max(len(_eval_runs(rows)), len((train_arms or {}).get(side) or ()))
        for side, rows in (("before", before), ("after", after))
    }
    run_std_source = "given" if run_std is not None else None

    # One floor per metric. A marker that applies to a subset of tasks is
    # noisier than pass@1, which averages over all of them, so judging it
    # against pass@1's floor calls a re-run draw a regression (#300).
    run_std_by_metric: dict[str, float | None]
    if isinstance(run_std, Mapping):
        run_std_by_metric = {
            _key(str(name)): (float(value) if value is not None else None)
            for name, value in run_std.items()
        }
    elif run_std is not None:
        scalar_floor = float(run_std)
        run_std_by_metric = {m: scalar_floor for m in metrics}
    elif min(eval_runs.values()) >= 2:  # noqa: PLR2004  # two runs before a run std exists
        run_std_by_metric = {m: _pooled_run_std(before, after, m) for m in metrics}
        run_std_source = (
            "eval_run" if any(value is not None for value in run_std_by_metric.values()) else None
        )
    else:
        run_std_by_metric = {}

    headline_metric = target_key if target_key in results else "pass_at_1"
    headline_run_std = run_std_by_metric.get(headline_metric)
    replicated = headline_run_std is not None
    # A floor is how far ONE run's mean moves when the eval is re-run. The
    # delta is a mean of n_a runs against a mean of n_b, so its own standard
    # deviation is ``floor * sqrt(1/n_a + 1/n_b)``, and the band is that
    # times 1.96, or times the t quantile when the floor was estimated from
    # these very runs (Lambert 2025, chapter Evaluation and its
    # evaluation-variance appendix).
    n_a, n_b, band_df = _band_args(eval_runs, run_std_source, run_std_runs)
    noise_rule = _band_rule(n_a, n_b, band_df, level)
    within_noise: list[str] = []
    no_floor: list[str] = []
    for m in metrics:
        r = results[m]
        metric_run_std = run_std_by_metric.get(m)
        noise = (
            noise_band(metric_run_std, n_a, n_b, df=band_df, level=level)
            if metric_run_std is not None
            else None
        )
        r["run_std"] = metric_run_std
        r["noise_band"] = noise
        # A floor was supplied or computed, but not for this metric: absent
        # from the mapping, or ``None`` there (what ``eval_variance`` returns
        # for a metric under two runs carried). Never borrow another
        # metric's floor; say so instead.
        if run_std_source is not None and metric_run_std is None:
            r["noise_note"] = "no_replicate_floor"
            no_floor.append(m)
        r["within_noise"] = (
            noise is not None and r.get("delta") is not None and abs(r["delta"]) < noise
        )
        if r["within_noise"]:
            within_noise.append(m)
    loud = {m for m in metrics if not results[m]["within_noise"]}
    regressions = [
        m
        for m in metrics
        if m in guarded and m in loud and results[m]["gain_verdict"] == "a_better"
    ]
    slipped = [
        m
        for m in metrics
        if m not in guarded and m in loud and results[m]["gain_verdict"] == "a_better"
    ]
    improved = [m for m in metrics if m in loud and results[m]["gain_verdict"] == "b_better"]
    target_result = results.get(target_key) if target_key else None
    if target_result is None and target_key:
        target_verdict = "target_not_measured"
    elif target_result is None:
        target_verdict = None
    else:
        target_verdict = _verdict_word(target_result, replicated)
    warnings: list[str] = []
    not_comparable: list[str] = []
    # Training seeds: the between-seed term on the headline metric, and
    # the verdict rule that "moved" needs MIN_TRAIN_SEEDS seeds per
    # trained arm (#356). ``ok`` reads the verdict before "unresolved"
    # hides it, so a wrong-way delta on one seed still fails the gate.
    headline_result = target_result if target_result else results["pass_at_1"]
    train = (
        _train_spread(train_arms, before, after, headline_metric, headline_result, level)
        if train_arms is not None
        else None
    )

    def _train_word(word: str | None) -> tuple[str | None, str | None]:
        """The headline word after the training-seed rule: (gate word,
        printed word). The gate word keeps a wrong-way delta visible."""
        if train is None or word in {None, "insufficient_data", "target_not_measured"}:
            return word, word
        moved_words = {"moved", "moved_unreplicated", "moved_the_wrong_way"}
        if train["ci95"] is not None and word in moved_words:
            lo, hi = train["ci95"]
            if lo <= 0.0 <= hi:
                word = "no_change_detected"
        if train["under_replicated"]:
            return word, "unresolved"
        return word, word

    gate_verdict, target_verdict = _train_word(target_verdict)
    ok = not regressions and gate_verdict not in {"moved_the_wrong_way"}
    if train is not None and train["under_replicated"]:
        counts = train["counts"]
        seeds = ", ".join(
            f"{'untrained' if counts[s] is None else counts[s]} {s}" for s in ("before", "after")
        )
        warnings.append(
            f"UNRESOLVED: {UNRESOLVED_LINE} (training seeds: {seeds}). The delta is between "
            "two separately trained models and the re-run floor measures only the eval, so one "
            "seed cannot separate the change from run-to-run training variance (Lambert 2025, "
            "chapter Evaluation; Miller 2024, arXiv:2411.00640). Train each arm at "
            f"{MIN_TRAIN_SEEDS} or more seeds and pass every seed's rows in train_runs=."
        )
    elif train is not None and train["ci95"] is not None:
        lo, hi = train["ci95"]
        stds = ", ".join(
            f"{train['std'][s]:.3f} {s}" for s in ("before", "after") if train["std"][s] is not None
        )
        if lo <= 0.0 <= hi and headline_result.get("delta") is not None:
            warnings.append(
                f"{headline_metric}: {headline_result['delta']:+.3f} on this seed pair, but "
                f"across training seeds {train['delta']:+.3f} with interval {lo:+.3f}..{hi:+.3f} "
                f"covers zero (between-seed std {stds}; {train['rule']}); the change does not "
                "survive the seed spread"
            )
    if target_verdict == "moved_unreplicated" and headline_metric not in no_floor:
        single = [side for side, n in eval_runs.items() if n < 2]  # noqa: PLR2004  # two runs before a run std exists
        if single:
            where = "each side" if len(single) != 1 else f"the {single[0]} side"
            warnings.append(
                f"One eval run on {where}, so this could be noise. Run each side three times "
                "with simulate(tasks=..., runs=3) and the report will say."
            )
        else:
            # Several row sets a side (train_runs) but no re-run floor: the
            # rows carry no lineage.eval_run to pool one from.
            warnings.append(
                f"{n_a} runs before and {n_b} after but no re-run floor, so this could be "
                "noise. Pass run_std=eval_variance(*one_side_row_sets)['run_std'] with "
                "run_std_runs=<how many> and the report will say."
            )
    if no_floor:
        warnings.append(
            f"no re-run floor for {', '.join(no_floor)}: run_std has no value for it, so its "
            "verdict rests on the task interval alone and is not checked against eval noise; "
            "pass eval_variance(...)['run_std_by_metric'] from three runs that all carry the "
            "marker, or judge it by hand"
        )
    for m in degenerate_guards:
        warnings.append(
            f"must_not_regress {m} is degenerate on both sides (every applicable row scored the "
            "same value): this guard cannot fail, so it catches nothing. Check that the marker "
            "fires at all."
        )
    if run_std_source == "eval_run" and min(eval_runs.values()) < MIN_RERUNS:
        warnings.append(
            "Two eval runs on a side is a difference, not a distribution, so run_std is rough; "
            f"{MIN_RERUNS} runs per side give a standard deviation worth reading."
        )
    elif run_std_runs is not None and run_std_runs < MIN_RERUNS:
        warnings.append(
            f"run_std came from {run_std_runs} re-runs, a difference, not a distribution, so it "
            f"is rough; {MIN_RERUNS} or more re-runs give a standard deviation worth reading."
        )
    # A run_std handed in as a number is read as the eval's exact spread and
    # the band uses 1.96. One estimated from a few re-runs is wider than
    # that: at df=2 the 1.96 band passes about 19% of pure-noise deltas,
    # not 5%. Only the caller knows where the number came from.
    if run_std_source == "given" and run_std_runs is None and headline_run_std is not None:
        warnings.append(
            "run_std was given as a number, so the band uses 1.96 and reads it as the eval's "
            "exact spread. If it came from re-runs, pass run_std_runs=<how many> "
            "(eval_variance(...)['n_runs']) so the band uses the t quantile at df = runs - 1 "
            f"and is honest about the estimate: 3 re-runs is {_t_quantile(2, level):.2f}, not "
            f"{_z_level(level):.2f}."
        )
    # Every metric gets its own interval at ``level``, so the chance that at
    # least one clears zero by luck grows with the number of markers. The
    # target is pre-specified and keeps its ``alpha``; the improved/slipped
    # lists do not, and a false flag in must_not_regress fails an otherwise
    # good run. ``1 - level**n`` is the chance under independent metrics;
    # markers that move together share their luck, so it is an upper bound
    # on the real family-wise rate, and the warning says so.
    n_metrics = len(metrics)
    family_error = 1.0 - level**n_metrics
    if n_metrics >= FAMILY_ERROR_MIN_METRICS and (improved or slipped or regressions):
        warnings.append(
            f"{n_metrics} metrics were each tested at {level:.0%}, so up to about a "
            f"{family_error:.0%} "
            "chance that at least one clears zero by luck (an upper bound: it treats the "
            "metrics as independent, and markers that move together share their luck); the "
            "target is pre-specified and unaffected, so treat a single unexpected entry in "
            "improved/slipped as a lead, not a finding, and confirm it on a second eval run."
        )
    # ceiling: an eval the before side already passes cannot show a gain
    mean_a = results["pass_at_1"].get("mean_a")
    ceiling = False
    if mean_a is not None and mean_a >= ceiling_pass_rate:
        ceiling = True
        warnings.append(
            f"The before run already passes {mean_a:.2f} of tasks, so there is little room to "
            "measure improvement; use harder situations."
        )
    else:
        means_a, means_b = task_means(before), task_means(after)
        shared = set(means_a) & set(means_b)
        with_room = sum(1 for t in shared if means_a[t] < 1.0)
        if with_room < CEILING_MIN_TASKS_WITH_ROOM and with_room < CEILING_ROOM_SHARE * len(shared):
            ceiling = True
            warnings.append(
                f"The before run already passes {len(shared) - with_room} of {len(shared)} paired "
                "tasks every time, so there is little room to measure improvement; use harder "
                "situations."
            )

    # proxy vs target: the over-optimization picture of Gao et al. 2022
    # (arXiv:2210.10760), as a verdict
    proxy_key = _key(proxy) if proxy else None
    proxy_result = results.get(proxy_key) if proxy_key else None
    proxy_verdict: str | None = None
    over_optimized = False
    headline_for_proxy = target_result if target_result else results["pass_at_1"]
    headline_name = target_key if target_result else "pass_at_1"
    if proxy_key and proxy_result is None:
        warnings.append(f"proxy {proxy!r} is not on both row sets")
        proxy_verdict = "proxy_not_measured"
    elif proxy_key and proxy_key == headline_name:
        warnings.append(f"proxy {proxy!r} is the target itself; name the training reward instead")
        proxy_verdict = "proxy_is_target"
    elif proxy_result is not None:
        proxy_verdict = {
            "b_better": "moved",
            "a_better": "moved_the_wrong_way",
            "no_difference_detected": "no_change_detected",
            "insufficient_data": "insufficient_data",
        }[proxy_result["gain_verdict"]]
        # "The proxy went up" means the proxy improved, which on a
        # down-is-the-win reward is its delta going down; the two curves
        # part in goodness, not in raw units, so the interval comparison
        # runs on ``_gain_ci`` while the printed numbers stay raw.
        proxy_up = proxy_result["gain_verdict"] == "b_better" and proxy_key in loud
        target_up = headline_for_proxy["gain_verdict"] == "b_better" and headline_name in loud
        pci, tci = proxy_result.get("ci95"), headline_for_proxy.get("ci95")
        pgain, tgain = _gain_ci(proxy_result), _gain_ci(headline_for_proxy)
        apart = bool(pgain and tgain and pgain[0] > tgain[1])
        over_optimized = (proxy_up and not target_up) or (proxy_up and apart)
        if over_optimized:
            ok = False
            tspan = f"{tci[0]:+.3f}..{tci[1]:+.3f}" if tci else "n/a"
            pspan = f"{pci[0]:+.3f}..{pci[1]:+.3f}" if pci else "n/a"
            # The verb is the raw move and the suffix says why it is the
            # gain, so the sentence is true whichever way the proxy points.
            proxy_lower = bool(proxy_result.get("lower_is_better"))
            verb = "down" if proxy_lower else "up"
            way = " (lower is better)" if proxy_lower else ""
            target_way = " (lower is better)" if headline_for_proxy.get("lower_is_better") else ""
            warnings.append(
                f"OVER-OPTIMIZED: {proxy_key} {verb} {proxy_result['delta']:+.3f}{way} "
                f"({level:.0%} {pspan}) while {headline_name} "
                f"{headline_for_proxy['delta']:+.3f}{target_way} "
                f"({level:.0%} {tspan}): the policy learned something the target does not "
                "credit (Gao et al. 2022, arXiv:2210.10760)"
            )
    headline_noise = results[headline_metric]["noise_band"]
    if headline_noise is not None and target_verdict == "within_eval_noise" and target_result:
        warnings.append(
            f"{target_key}: {target_result['delta']:+.3f} is inside the eval's own re-run band "
            f"({noise_rule} = {headline_noise:.3f}); re-running the eval moves it that much"
        )
    for m in regressions:
        r = results[m]
        way = ", where lower is better, so the rise is the regression" if lower[m] else ""
        warnings.append(
            f"REGRESSION {m}: {r['delta']:+.3f} ({level:.0%} {r['ci95'][0]:+.3f}.."
            f"{r['ci95'][1]:+.3f}), named in must_not_regress{way}"
        )
    for m in slipped:
        r = results[m]
        # The verb is the raw move, so the number and the word agree: a
        # down-is-the-win marker only reaches this list by rising.
        verb, way = ("rose", " (lower is better)") if lower[m] else ("dropped", "")
        warnings.append(
            f"{m} {verb} {r['delta']:+.3f} "
            f"({level:.0%} {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}){way}"
        )
    headline = target_result if target_result else results["pass_at_1"]
    headline_key = target_key if target_result else "pass_at_1"
    if headline.get("note"):
        warnings.append(f"{headline_key}: {headline['note']}")
    # The two arms have to be the same eval. Two runs that drew their own
    # situations (a hard_share or dimensions change, a different seed, a
    # writer that moved) share only some tasks, and a delta over the few
    # that pair is a delta between two situation sets, not between two
    # policies: the 2026-09-18 live test paired 7 of 41 and read -0.143
    # under PASS. Fewer than half paired is the same "most" as the note.
    pairing = results["pass_at_1"]
    paired_share = pairing.get("paired_share")
    n_only_a, n_only_b = int(pairing.get("n_only_a") or 0), int(pairing.get("n_only_b") or 0)
    if (
        paired_share is not None
        and paired_share < UNPAIRED_MAJORITY_SHARE
        and (n_only_a or n_only_b)
    ):
        ok = False
        not_comparable.append("situations")
        n_shared = int(pairing.get("n_paired") or 0)
        warnings.append(
            f"NOT COMPARABLE: the two arms drew different situation sets; {n_shared} of "
            f"{n_shared + n_only_a + n_only_b} tasks are on both sides ({n_only_a} only before, "
            f"{n_only_b} only after), so the delta is between two evals, not two policies. Pin "
            "the after side to the before run's tasks (simulate(agent, tasks=before_rows, ...)) "
            "and re-run, or compare pass rate per tier with dataset_report on each arm."
        )
    # Eval size: a no-change verdict is only as strong as the band the
    # task count allows. Say what this holdout can prove and what the
    # delta seen here would have needed (#257).
    n_paired = int(headline.get("n_paired") or 0)
    k_eval = int(pass_at(before).config.get("k") or 1)
    base_rate = float(mean_a) if mean_a is not None else BASE_PASS_RATE
    can_prove = (
        detectable_effect(n_paired, base=base_rate, k=k_eval, power=power, alpha=alpha)
        if n_paired >= MIN_HOLDOUT_TASKS
        else None
    )
    tasks_needed: int | None = None
    tasks_needed_source: str | None = None
    tasks_needed_paired: int | None = None
    delta_seen: float | None = None
    delta_shown: float | None = None
    raw_delta = headline.get("delta")
    if isinstance(raw_delta, (int, float)):
        # ``holdout_size`` sizes a *gain*, so a headline where down is the
        # win is sized on how big its drop was; the line below still prints
        # the raw signed delta the reader measured.
        signed = float(raw_delta)
        gain = -signed if headline.get("lower_is_better") else signed
        if 0 < gain < 1:
            delta_seen, delta_shown = gain, signed
            # Size from the rows in hand: the per-task paired sd carries the
            # covariance pairing buys, which the independent-arms model cannot
            # see, so the model asked for about twice the tasks (525 against
            # 270 on 160 MATH-500 tasks at k=12, #733). ``holdout_size`` falls
            # back to the model itself at the ceiling or on a degenerate
            # spread; too few shared graded tasks to measure a sd is the one
            # case it refuses, and the model answers there.
            try:
                sizing = holdout_size(
                    delta_seen,
                    before=before,
                    after=after,
                    power=power,
                    alpha=alpha,
                    ceiling_pass_rate=ceiling_pass_rate,
                )
            except ValueError:
                sizing = holdout_size(
                    delta_seen,
                    base=base_rate,
                    k=k_eval,
                    power=power,
                    alpha=alpha,
                    ceiling_pass_rate=ceiling_pass_rate,
                )
            tasks_needed = int(sizing["n_tasks"])
            tasks_needed_source = str(sizing["sd_source"])
            tasks_needed_paired = sizing.get("n_paired")
    verdict_word = (
        target_verdict
        if target_result
        else _train_word(_verdict_word(results["pass_at_1"], replicated))[1]
    )
    if verdict_word == "no_change_detected" and can_prove is not None:
        line = (
            f"{n_paired} paired tasks at k={k_eval} can prove a gain of about "
            f"+{can_prove:.2f} at {power:.0%} power"
        )
        if tasks_needed is not None and delta_shown is not None:
            if tasks_needed_source == "rows" and tasks_needed_paired:
                where = f"task sd measured on the {tasks_needed_paired} paired tasks here"
            else:
                where = "task sd from the binomial model, not measured"
            line += (
                f"; to prove the {delta_shown:+.3f} seen here you need about "
                f"{tasks_needed} tasks ({where})"
            )
        warnings.append(line + " (holdout_size).")
    # Rows per task on the two sides. The sizing line and the k-way
    # numbers use the before side's k; an after side short of it was cut
    # by lost rollouts. Per-task means keep the paired delta unbiased when
    # the loss is random and only widen the interval; a loss with a cause
    # biases it, and only a re-run fixes that (#303).
    k_after = int(pass_at(after).config.get("k") or 1)
    if k_after != k_eval:
        short_side, full_k = ("after", k_eval) if k_after < k_eval else ("before", k_after)
        short_rows = after if short_side == "after" else before
        other_rows = before if short_side == "after" else after
        per_task = Counter(task_key(r) for r in short_rows if isinstance(r, dict))
        paired_keys = per_task.keys() & {task_key(r) for r in other_rows if isinstance(r, dict)}
        n_short = sum(1 for t in paired_keys if per_task[t] < full_k)
        warnings.append(
            f"before has k={k_eval} rollouts per task and after has k={k_after}: "
            f"{n_short} of {len(paired_keys)} paired tasks on the {short_side} side have fewer "
            f"than {full_k} rows. Unequal k is a precision issue, not a bias: rows lost at random "
            "leave the paired delta unbiased and widen its interval (about 10% at k=4 against "
            "k=2 on half the tasks); rows lost for a reason (a timeout on the hard runs) bias it, "
            f"and only re-running the {short_side} side on its short tasks fixes that "
            "(data.report()['rollouts_lost_by'] says why rows went missing). "
            f"balance_rollouts=True only makes pass^k/pass@k share k={min(k_eval, k_after)} "
            "and costs another 10% of interval width."
        )
    if balanced and (balanced["rows_dropped"]["before"] or balanced["rows_dropped"]["after"]):
        warnings.append(
            f"balance_rollouts=True dropped {balanced['rows_dropped']['before']} before rows and "
            f"{balanced['rows_dropped']['after']} after rows on {balanced['tasks_trimmed']} "
            "tasks so both sides have the same rollouts per task; the intervals are over the "
            "rows that remain, and rows lost for a reason are still lost"
        )
    if target_verdict == "target_not_measured":
        warnings.append(f"target {target!r} is not on both row sets")
    groups: dict[str, dict[str, Any]] | None = None
    groups_down: list[str] = []
    if by is not None:
        group_metric = target_key if target_key in results else "pass_at_1"
        group_lower = lower.get(group_metric, False)
        groups = _by_group(
            before,
            after,
            by=by,
            metric=group_metric,
            n_boot=n_boot,
            seed=seed,
            level=level,
            lower_is_better=group_lower,
        )
        groups_down = [g for g, r in groups.items() if r.get("gain_verdict") == "a_better"]
        way = " (lower is better)" if group_lower else ""
        for g in groups_down:
            r = groups[g]
            warnings.append(
                f"{group_metric} moved the wrong way for {g}: {r['delta']:+.3f}{way} "
                f"({level:.0%} {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}, {r['rows_b']} rows)"
            )
        if not groups:
            warnings.append(
                f"by={by if isinstance(by, str) else 'callable'}: no group is on both row sets"
            )
    # What each side was produced with. A delta between two settings is
    # not a delta between two policies, so each difference is named, and
    # so is the case where nothing changed at all.
    config = {"before": pass_at(before).config, "after": pass_at(after).config}
    cfg_a, cfg_b = config["before"], config["after"]

    def _both(key: str) -> bool:
        return cfg_a.get(key) is not None and cfg_b.get(key) is not None

    if _both("judge_version") and cfg_a["judge_version"] != cfg_b["judge_version"]:
        warnings.append(
            f"Before and after were graded by different judges ({cfg_a['judge_version']} vs "
            f"{cfg_b['judge_version']}); grade both sides with the same judge before reading "
            "the delta."
        )
    if _both("prompt_hash") and cfg_a["prompt_hash"] != cfg_b["prompt_hash"]:
        not_comparable.append("prompt_hash")
        warnings.append(
            f"NOT COMPARABLE: before rows were generated under system prompt "
            f"{cfg_a['prompt_hash']} and after under {cfg_b['prompt_hash']} "
            "(lineage.system_prompt_sha); re-run one side with the other's system_prompt= "
            "(the text is in that run's system_prompts under the hash)."
        )
    if _both("temperature") and cfg_a["temperature"] != cfg_b["temperature"]:
        warnings.append(
            f"Before was sampled at temperature {cfg_a['temperature']} and after at "
            f"{cfg_b['temperature']}; re-run one side so both use the same temperature=."
        )
    if _both("max_tokens") and cfg_a["max_tokens"] != cfg_b["max_tokens"]:
        warnings.append(
            f"Before allowed {cfg_a['max_tokens']} reply tokens and after {cfg_b['max_tokens']}; "
            "re-run one side so both use the same agent_max_tokens=."
        )
    if (
        _both("truncated_share")
        and abs(cfg_a["truncated_share"] - cfg_b["truncated_share"]) > TRUNCATED_SHARE_GAP
    ):
        warnings.append(
            f"The token cap cut {cfg_a['truncated_share']:.0%} of before rows and "
            f"{cfg_b['truncated_share']:.0%} of after rows; a side that is cut more often is "
            "not the same eval. Raise agent_max_tokens= on both sides or read the delta with "
            "that in mind."
        )
    # Rows that could not be graded leave the denominator, and they are not a
    # random sample: a long trajectory is both likelier to break a judge and
    # likelier to have failed. A side that dropped a share d of its rows has
    # a survivors' rate off by up to d/(1-d) (every dropped row passed, or
    # every one failed), and the two sides' errors add: a zero gap with one
    # side dropping failures and the other dropping passes biases the delta
    # by the full amount, so the gap between the shares bounds nothing. No
    # interval sees this, because it is selection, not variance.
    graded_bias: float | None = None
    if _both("graded_share"):
        dropped = [1.0 - float(cfg["graded_share"]) for cfg in (cfg_a, cfg_b)]
        graded_bias = sum(d / (1.0 - d) if d < 1.0 else 1.0 for d in dropped)
    headline_size = abs(float(headline["delta"])) if headline.get("delta") is not None else None
    # the noise the bound is read against: the re-run band, else the task
    # interval's half-width
    if headline_noise is not None:
        bias_bar, bar_name = headline_noise, "the re-run band"
    elif headline.get("ci95"):
        bias_bar = (headline["ci95"][1] - headline["ci95"][0]) / 2
        bar_name = "the interval's half-width"
    else:
        bias_bar, bar_name = None, ""
    if graded_bias and bias_bar is not None and graded_bias > bias_bar:
        shares = (
            f"{cfg_a['graded_share']:.1%} of before rows and {cfg_b['graded_share']:.1%} of "
            f"after rows carry a verdict"
        )
        why = (
            "rows a judge could not grade leave the denominator and are not a random sample "
            "(the long ones fail more often), so each side's rate can be off by up to "
            "dropped/(1-dropped) and the two sides add"
        )
        if headline_size is not None and graded_bias >= headline_size:
            ok = False
            not_comparable.append("graded_share")
            warnings.append(
                f"NOT COMPARABLE: {shares}; {why}: up to {graded_bias:.1%}, which covers the whole "
                f"{headline_size:.3f} delta on {headline_key}; re-grade the dropped rows before "
                "reading this delta"
            )
        else:
            warnings.append(
                f"{shares}; {why}: up to {graded_bias:.1%}, more than {bar_name} "
                f"({bias_bar:.3f}); read a delta near that size as unproven, or re-grade the "
                "dropped rows"
            )
    elif (
        _both("graded_share")
        and min(cfg_a["graded_share"], cfg_b["graded_share"]) < GRADED_SHARE_WARN
    ):
        warnings.append(
            f"only {min(cfg_a['graded_share'], cfg_b['graded_share']):.1%} of rows on one side "
            "carry a verdict; both rates are over the rows that survived grading, not the rows "
            "that were run"
        )
    if _both("policy_version") and cfg_a["policy_version"] == cfg_b["policy_version"]:
        warnings.append(
            "Before and after are the same policy version; this compares a model to itself. "
            'Base and an adapter can share a served model name: pass advanced={"model_version": '
            '"...-base"} and "...-sft" so the two arms are distinguishable on the rows.'
        )
    # Answer production. A rate is conditional on the arm having replied;
    # when the two arms differ in how often they did, by more than chance
    # (two-proportion z test) and by more than the eval's noise, the
    # comparison does not exist and the report fails (#297).
    if _both("answered_share"):
        answered_a, _, n_reply_a = answer_counts(before)
        answered_b, _, n_reply_b = answer_counts(after)
        answered_p = _two_proportion_p(answered_a, n_reply_a, answered_b, n_reply_b)
        answered_gap = abs(float(cfg_a["answered_share"]) - float(cfg_b["answered_share"]))
        if headline_noise is not None:
            gap_bar, bar_name = headline_noise, f"the re-run band {headline_noise:.3f}"
        else:
            gap_bar, bar_name = answered_gap_points, f"{answered_gap_points:.0%} with no run_std"
        if answered_p is not None and answered_p < answered_alpha:
            fails = answered_gap > gap_bar
            if fails:
                ok = False
                not_comparable.append("answered")
            warnings.append(
                _answered_note(
                    cfg_a, cfg_b, p=answered_p, gap=answered_gap, bar=bar_name, fails=fails
                )
            )
    # The environment has to hold still while the weights change. The
    # simulated user and the situation writer default to the agent's own
    # model, so in a before/after they follow the policy under test and the
    # delta measures the pair (Lambert 2025, chapter Evaluation: every layer
    # of an agentic eval moves the score, so every layer is pinned and
    # recorded).
    agent_a = str(cfg_a.get("policy_version") or "").split("@", 1)[0]
    agent_b = str(cfg_b.get("policy_version") or "").split("@", 1)[0]
    one_name_two_policies = (
        bool(agent_a)
        and agent_a == agent_b
        and (cfg_a.get("policy_version") != cfg_b.get("policy_version"))
    )
    # A situation nobody's model wrote cannot have moved with the weights:
    # a seed ask, the template writer's text and a replay of either are one
    # "offline" writer for this check (#375).
    offline = {"seed", "template", "pinned", "callable-writer"}

    def _writer_class(value: Any) -> Any:
        return "offline" if str(value) in offline else value

    for key, knob in (("user_model", "user_model="), ("writer_model", "simulator=")):
        same = (
            _writer_class(cfg_a.get(key)) == _writer_class(cfg_b.get(key))
            if key == "writer_model"
            else cfg_a.get(key) == cfg_b.get(key)
        )
        if _both(key) and not same:
            ok = False
            not_comparable.append(key)
            warnings.append(
                f"NOT COMPARABLE: {key} was {cfg_a[key]!r} before and {cfg_b[key]!r} after; the "
                f"environment moved with the weights, so this delta measures the pair, not the "
                f"policy; pin {knob} to one model on both arms and re-run"
            )
        elif one_name_two_policies and cfg_a.get(key) == agent_a and cfg_b.get(key) == agent_a:
            # Same served name, different policy stamp: if the two arms are
            # different weights under one name, the user or writer that ran
            # on that name moved with them.
            warnings.append(
                f"{key} is the agent's own served model ({agent_a!r}) on both arms, and the two "
                f"arms differ in policy_version under that one name; if they served different "
                f"weights the environment moved with them, so pin {knob} to a fixed model to "
                "rule it out"
            )
    return DeltaReport(
        {
            "ok": ok,
            "not_comparable": not_comparable,
            "target": target_key,
            "target_verdict": target_verdict,
            #: the verdict format_delta_report prints: the target's, else pass@1's
            "headline_verdict": verdict_word,
            "target_delta": target_result["delta"] if target_result else None,
            "target_ci95": target_result["ci95"] if target_result else None,
            "n_metrics": n_metrics,
            "alpha": alpha,
            "level": level,
            #: chance at least one of the metrics clears zero by luck alone
            "family_error": round(family_error, 4),
            "n_paired_tasks": results["pass_at_1"]["n_paired"],
            "n_unpaired_tasks": results["pass_at_1"]["n_only_a"] + results["pass_at_1"]["n_only_b"],
            "improved": improved,
            "regressions": regressions,
            "slipped": slipped,
            #: the metrics read as down-is-the-win, from lower_is_better= and
            #: LOWER_IS_BETTER_MARKERS; their delta and interval are still the
            #: raw signed change, only the reading of it flips (#638)
            "lower_is_better": [m for m in metrics if lower[m]],
            "within_noise": within_noise,
            "run_std": headline_run_std,
            "run_std_by_metric": {m: run_std_by_metric.get(m) for m in metrics},
            "run_std_source": run_std_source,
            #: re-runs a given run_std was computed from, and the band's degrees
            #: of freedom (None: the floor is read as the eval's exact spread)
            "run_std_runs": run_std_runs,
            "run_std_df": band_df,
            "noise_band": headline_noise,
            "noise_rule": noise_rule,
            "eval_runs": eval_runs,
            "replicated": replicated,
            #: training seeds per arm (None: untrained, or no train_runs given),
            #: the between-seed std of each arm's mean, the across-seed delta
            #: and its widened interval on the headline metric (#356)
            "train_runs": train["counts"] if train else None,
            "train_std": train["std"] if train else None,
            "train_df": train["df"] if train else None,
            "train_delta": train["delta"] if train else None,
            "train_ci95": train["ci95"] if train else None,
            "train_rule": train["rule"] if train else None,
            "ceiling": ceiling,
            "detectable_effect": can_prove,
            "tasks_needed": tasks_needed,
            #: where the task sd behind tasks_needed came from: "rows" (the
            #: paired sd measured on these arms) or "model" (#733)
            "tasks_needed_source": tasks_needed_source,
            "degenerate_guards": degenerate_guards,
            "proxy": proxy_key,
            "proxy_verdict": proxy_verdict,
            "proxy_delta": proxy_result["delta"] if proxy_result else None,
            "proxy_ci95": proxy_result["ci95"] if proxy_result else None,
            "over_optimized": over_optimized,
            "metrics": results,
            "warnings": warnings,
            "config": config,
            "balanced": balanced,
            "by": (
                by if isinstance(by, str) else (getattr(by, "__name__", "callable") if by else None)
            ),
            "groups": groups,
            "groups_down": groups_down,
        }
    )


def _balance_rollouts(
    before: Sequence[dict], after: Sequence[dict], *, seed: int = 0
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Trim each paired task to the rows both sides have.

    A task with 4 rows before and 2 after keeps 2 on each side; which 2
    of the 4 is drawn by ``seed`` so the same call gives the same rows.
    Tasks on one side only are left alone (the pairing drops them and
    ``n_unpaired_tasks`` counts them). Returns the two trimmed row lists
    and ``{"rows_dropped": {"before", "after"}, "tasks_trimmed"}``.
    """

    def _grouped(rows: Sequence[dict]) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            if isinstance(row, dict):
                groups.setdefault(task_key(row), []).append(row)
        return groups

    groups_a, groups_b = _grouped(before), _grouped(after)
    keep: dict[str, dict[str, int]] = {}
    tasks_trimmed = 0
    for task in groups_a.keys() & groups_b.keys():
        n_a, n_b = len(groups_a[task]), len(groups_b[task])
        if n_a == n_b:
            continue
        tasks_trimmed += 1
        keep[task] = {"before": min(n_a, n_b), "after": min(n_a, n_b)}

    def _trim(rows: Sequence[dict], groups: dict[str, list[dict]], side: str) -> list[dict]:
        drop: set[int] = set()
        for task, want in keep.items():
            members = groups[task]
            if len(members) <= want[side]:
                continue
            rng = random.Random(f"{seed}:{side}:{task}")
            order = list(range(len(members)))
            rng.shuffle(order)
            drop.update(id(members[i]) for i in order[want[side] :])
        return [row for row in rows if not (isinstance(row, dict) and id(row) in drop)]

    trimmed_a = _trim(before, groups_a, "before")
    trimmed_b = _trim(after, groups_b, "after")
    info = {
        "rows_dropped": {
            "before": len(before) - len(trimmed_a),
            "after": len(after) - len(trimmed_b),
        },
        "tasks_trimmed": tasks_trimmed,
    }
    return trimmed_a, trimmed_b, info


class DeltaReport(Report):
    """What ``delta_report`` (``wai.compare``) measured, as an object that
    prints itself.

    Still the dict it always was: ``report["metrics"]``,
    ``report["verdict"]`` and ``report["warnings"]`` read the same.
    ``print(report)`` is now the block ``format_delta_report`` writes.
    """

    _summary_keys = ("ok", "verdict")

    def __str__(self) -> str:
        return format_delta_report(self)


#: ``compare_runs`` verdict -> the printed tag, in the spelling the report
#: has always used: lower case is the good way, upper case is the alarm. A
#: metric where down is the win swaps which of the two gets shouted, and
#: the line says ``lower is better`` beside it (#638).
_UP_TAGS = {
    "b_better": "up",
    "a_better": "DOWN",
    "no_difference_detected": "flat",
    "insufficient_data": "n/a",
}
_DOWN_TAGS = {
    "b_better": "UP",
    "a_better": "down",
    "no_difference_detected": "flat",
    "insufficient_data": "n/a",
}
LOWER_IS_BETTER_TAG = "lower is better"


def _metric_tag(result: Mapping[str, Any]) -> tuple[str, str]:
    """One metric line's verdict tag and the suffix that names its
    direction, from the raw verdict and whether down is the win."""
    lower = bool(result.get("lower_is_better"))
    tag = (_DOWN_TAGS if lower else _UP_TAGS)[result["verdict"]]
    return tag, (f"  {LOWER_IS_BETTER_TAG}" if lower else "")


def format_delta_report(report: dict[str, Any]) -> str:
    """The block a person reads: headline, then one line per metric."""
    lines: list[str] = []
    level = float(report.get("level") or CI_LEVEL)
    if report.get("target"):
        r = report["metrics"].get(report["target"])
        if r and r.get("delta") is not None and r.get("ci95"):
            lines.append(
                f"{report['target']}: {report['target_verdict']} "
                f"({r['delta']:+.3f}, {level:.0%} {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}, "
                f"{r['n_paired']} paired tasks)"
            )
        else:
            lines.append(f"{report['target']}: {report['target_verdict']}")
    if report.get("proxy"):
        p = report["metrics"].get(report["proxy"])
        if p and p.get("delta") is not None and p.get("ci95"):
            lines.append(
                f"proxy {report['proxy']}: {report['proxy_verdict']} "
                f"({p['delta']:+.3f}, {level:.0%} {p['ci95'][0]:+.3f}..{p['ci95'][1]:+.3f})"
                + ("  OVER-OPTIMIZED" if report.get("over_optimized") else "")
            )
        else:
            lines.append(f"proxy {report['proxy']}: {report['proxy_verdict']}")
    lines.append(headline_word(report))
    floors = report.get("run_std_by_metric") or {}
    if report.get("run_std") is not None or report.get("run_std_source") is not None:
        runs = report.get("eval_runs") or {}
        if report.get("run_std_source") == "eval_run":
            source = f"{runs.get('before')} eval runs before, {runs.get('after')} after"
        elif report.get("run_std_runs") is not None:
            source = f"run_std given from {report['run_std_runs']} re-runs"
        else:
            source = "run_std given"
        headline = report.get("run_std")
        head = (
            f"run_std {headline:.3f}, a delta under {report['noise_band']:.3f} is noise"
            if headline is not None
            else "no run_std for the headline metric"
        )
        per_metric = ", per metric below" if len(set(floors.values())) > 1 else ""
        lines.append(f"eval noise: {head} ({report['noise_rule']}; {source}{per_metric})")
    counts = report.get("train_runs")
    if counts is not None:
        # The between-seed arithmetic, printed the way run_std is above.
        stds = report.get("train_std") or {}

        def _seeds(side: str) -> str:
            n = counts.get(side)
            return "untrained" if n is None else f"{n} seed{'s' if n != 1 else ''}"

        head = f"training seeds: {_seeds('before')} before, {_seeds('after')} after"
        span = report.get("train_ci95")
        if span:
            std_text = ", ".join(
                f"{stds[s]:.3f} {s}" for s in ("before", "after") if stds.get(s) is not None
            )
            lines.append(
                f"{head}; between-seed std {std_text}; across seeds {report['train_delta']:+.3f}, "
                f"interval widened to {span[0]:+.3f}..{span[1]:+.3f} ({report['train_rule']})"
            )
        else:
            lines.append(f"{head}; unresolved: {UNRESOLVED_LINE}")
    if report.get("n_metrics", 0) >= 2 and report.get("family_error") is not None:  # noqa: PLR2004  # a family needs two metrics
        lines.append(
            f"family error: {report['n_metrics']} metrics at {level:.0%}, up to "
            f"{report['family_error']:.0%} "
            "chance that one clears zero on luck alone (upper bound, independent metrics)"
        )
    graded = {
        side: (report.get("config") or {}).get(side, {}).get("graded_share")
        for side in ("before", "after")
    }
    if graded["before"] is not None and graded["after"] is not None:
        bound = sum((1 - g) / g if g else 1.0 for g in graded.values())
        if bound > 0:
            lines.append(
                f"graded: {graded['before']:.1%} before, {graded['after']:.1%} after "
                f"(selection can move the delta up to {bound:.1%})"
            )
    if report.get("ceiling"):
        lines.append("CEILING: the before run already passes most tasks; use harder situations")
    answered = {
        side: (report.get("config") or {}).get(side, {}).get("answered_share")
        for side in ("before", "after")
    }
    if answered["before"] is not None and answered["after"] is not None:
        lines.append(f"answered: {answered['before']:.1%} before, {answered['after']:.1%} after")
    down_good = list(report.get("lower_is_better") or [])
    if down_good:
        # The deltas below are the raw change either way, so the reader is
        # told once which of them are read the other way round (#638).
        lines.append(
            f"lower is better: {', '.join(down_good)} (a drop is the win; the delta and "
            "interval printed are still the raw change)"
        )
    for name, r in report["metrics"].items():
        if r.get("delta") is None:
            lines.append(f"  {name:<28} insufficient data")
            continue
        ci = r.get("ci95")
        span = f"{ci[0]:+.3f}..{ci[1]:+.3f}" if ci else "n/a"
        tag, way = _metric_tag(r)
        if r.get("within_noise"):
            tag = "noise"
        pair = "paired" if r["paired"] else "unpaired"
        # The floor this line was judged against, so a reader can see that
        # a marker's band is its own and not pass@1's (#300).
        if r.get("noise_band") is not None:
            floor = f"  noise<{r['noise_band']:.3f}"
        elif r.get("noise_note"):
            floor = f"  {r['noise_note']}"
        else:
            floor = ""
        lines.append(
            f"  {name:<28} {r['mean_a']:.3f} -> {r['mean_b']:.3f}  {r['delta']:+.3f} "
            f"[{span}]  {tag}  ({r['n_used']} {pair}){floor}{way}"
        )
    balanced = report.get("balanced")
    if balanced:
        dropped = balanced.get("rows_dropped") or {}
        lines.append(
            f"balanced: dropped {dropped.get('before', 0)} before rows and "
            f"{dropped.get('after', 0)} after rows on {balanced.get('tasks_trimmed', 0)} tasks"
        )
    groups = report.get("groups")
    if groups:
        lines.append(f"by {report.get('by')}:")
        for name, r in groups.items():
            if r.get("delta") is None:
                lines.append(
                    f"  {name:<28} insufficient data ({r.get('rows_a', 0)}/{r.get('rows_b', 0)} rows)"
                )
                continue
            ci = r.get("ci95")
            span = f"{ci[0]:+.3f}..{ci[1]:+.3f}" if ci else "n/a"
            tag, way = _metric_tag(r)
            lines.append(
                f"  {name:<28} {r['mean_a']:.3f} -> {r['mean_b']:.3f}  {r['delta']:+.3f} "
                f"[{span}]  {tag}  ({r['rows_a']}/{r['rows_b']} rows){way}"
            )
    for w in report.get("warnings") or []:
        lines.append(f"! {w}")
    return "\n".join(lines)


__all__ = ["delta_report", "format_delta_report"]
