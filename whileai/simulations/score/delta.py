"""Did training move the behavior, and did anything else slip?

One call over two graded, marker-scored row sets: the rollouts before a
training run and the rollouts after it, on the same tasks. Every metric
the two sets share (pass@1 and each marker) is compared as paired task
differences with a bootstrap interval (``stats.compare_runs``), so the
answer is "moved by X, interval Y" and not a pair of means.

Markers are read as higher-is-better. A metric named in
``must_not_regress`` whose interval sits entirely below zero is a
regression and fails the report; any other metric that drops
significantly is a warning (rlhf-book ch. 15: post-training on one thing
forgets others, and on-policy data forgets less, which is only visible
if you measure the others).

``proxy`` names the metric the run was trained on (the training reward,
kept on the rows as a marker) when it is not the target. Over-
optimization is the two curves parting (rlhf-book ch. 14): the proxy
moved up while the target did not follow, or the proxy's interval sits
entirely above the target's. The report says so and fails.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .passat import pass_at
from .stats import (
    DEFAULT_BOOT,
    compare_runs,
    detectable_effect,
    eval_variance,
    holdout_size,
    marker_names,
    metric_summary,
    task_means,
)

GROUP_KEYS = ("delta", "ci95", "verdict", "mean_a", "mean_b", "n_used", "n_paired", "paired")

#: Width of the re-run noise band, in units of ``run_std``. A delta is the
#: difference of two independently re-run sides, so its standard deviation is
#: ``run_std * sqrt(2)``; two of those is the band inside which re-running the
#: eval moves the delta on its own.
NOISE_SIGMAS = 2.0 * math.sqrt(2.0)

#: A before side passing this share of tasks has little room left to show
#: an improvement; the report flags ``ceiling``.
CEILING_PASS_RATE = 0.9
#: Below this many paired tasks with room to move (and under half of
#: them), the same flag.
CEILING_MIN_TASKS_WITH_ROOM = 20

_VERDICT_WORDS = {
    "b_better": "moved",
    "a_better": "moved_the_wrong_way",
    "no_difference_detected": "no_change_detected",
    "insufficient_data": "insufficient_data",
}


def _verdict_word(result: dict[str, Any], replicated: bool) -> str:
    """One metric's verdict as the report says it: ``moved`` only when
    the eval was run more than once per side (or a ``run_std`` was
    given), else ``moved_unreplicated``; inside the re-run band,
    ``within_eval_noise``."""
    word = _VERDICT_WORDS[result["verdict"]]
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
    ``eval_variance`` per side, pooled as the root mean square, since
    each side is the same eval on one model (rlhf-book appendix C)."""
    stds = [
        eval_variance(rows, metric=metric, by="eval_run")["run_std"] for rows in (before, after)
    ]
    if any(v is None for v in stds):
        return None
    return (sum(float(v) ** 2 for v in stds) / len(stds)) ** 0.5


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
) -> dict[str, dict[str, Any]]:
    """The target metric compared within each group of rows. A group needs
    rows on both sides; rows with no group value are left out."""
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
            groups_a[name], groups_b[name], metric=metric, n_boot=n_boot, seed=seed + 100 + i
        )
        slim = {k: r.get(k) for k in GROUP_KEYS}
        slim["rows_a"] = len(groups_a[name])
        slim["rows_b"] = len(groups_b[name])
        out[name] = slim
    return out


def delta_report(
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    target: str | None = None,
    must_not_regress: Sequence[str] = (),
    markers: Sequence[str] | None = None,
    by: str | Callable[[dict], Any] | None = None,
    run_std: float | None = None,
    proxy: str | None = None,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare ``after`` to ``before`` on pass@1 and every shared marker.

    ``target`` names the metric the run was meant to move (``"pass_at_1"``
    or ``"marker:<name>"``); the verdict on it is the headline.
    ``proxy`` names the metric the run was actually trained on (the
    training reward as a marker, e.g. ``"marker:first_action"``). When
    the proxy moved up and the target did not, or the proxy's interval
    sits entirely above the target's, the report is ``over_optimized``
    and fails: the policy learned something the target does not credit
    (rlhf-book ch. 14).
    ``must_not_regress`` lists metrics whose significant drop fails the
    report. Metric names for markers are the marker names; pass@1 is
    ``"pass_at_1"``. Tasks on one side only do not pair; their count is
    ``n_unpaired_tasks`` and, when any were dropped, a warning says so.

    ``by`` splits the target by a group on each row: a row key (top level,
    or a marker name) or a callable ``row -> group``. The report gains
    ``groups``: the target compared within each group, so a headline that
    moved cannot hide a kind of prompt that moved the other way. A group
    whose target dropped significantly is listed in ``groups_down`` and
    warned about; it does not flip ``ok``, which stays the
    ``must_not_regress`` contract (name the group's metric there if it
    should).

    ``run_std`` is the evaluation's own re-run standard deviation
    (``eval_variance(...)["run_std"]``, rlhf-book ch. 16). A metric whose
    delta is smaller than ``NOISE_SIGMAS`` times it (2*sqrt(2), since a
    delta is the difference of two re-run draws and carries sqrt(2) times
    one side's spread) is ``within_noise``: not
    improved, not slipped, not a regression, and a target there reads
    ``within_eval_noise`` rather than moved, because re-running the eval
    moves it that much on its own. When both row sets carry two or more
    ``lineage.eval_run`` values (``simulate(tasks=..., runs=3)``) the
    report computes ``run_std`` itself on the headline metric, pooled
    over the two sides, and ``eval_runs`` says how many runs each side
    had. With one run on either side and no ``run_std`` a target that
    moved reads ``moved_unreplicated`` and a warning says how to fix it:
    one evaluation is a draw, not a distribution (rlhf-book ch. 16,
    "why many comparisons are unreliable", and appendix C).

    ``ceiling`` is set when the before side already passes
    ``CEILING_PASS_RATE`` of its tasks, or when fewer than
    ``CEILING_MIN_TASKS_WITH_ROOM`` paired tasks (and under half) are not
    already passed every time: there is little room left for an
    improvement to show, whatever the training did.


    ``config`` says what each side was produced with (``pass_at(...).config``
    per side: task count, k, temperature, max_tokens, policy and judge
    versions, prompt hash). A warning names each setting the two sides
    disagree on, and says so when both sides are the same policy version
    (rlhf-book ch. 16: a comparison is only as good as the settings it
    was run under).
    """
    names = (
        list(markers)
        if markers is not None
        else sorted(set(marker_names(before)) & set(marker_names(after)))
    )
    metrics = ["pass_at_1", *[f"marker:{m}" for m in names]]
    results: dict[str, dict[str, Any]] = {}
    for i, metric in enumerate(metrics):
        results[metric] = compare_runs(before, after, metric=metric, n_boot=n_boot, seed=seed + i)

    def _key(name: str) -> str:
        return name if name == "pass_at_1" or name.startswith("marker:") else f"marker:{name}"

    guarded = {_key(m) for m in must_not_regress}
    target_key = _key(target) if target else None
    degenerate_guards: list[str] = []
    for m in sorted(guarded):
        if m not in results:
            continue
        sides = [metric_summary(rows, m, n_boot=10) for rows in (before, after)]
        if all(s.get("degenerate") for s in sides):
            degenerate_guards.append(m)
    eval_runs = {"before": len(_eval_runs(before)), "after": len(_eval_runs(after))}
    run_std_source = "given" if run_std is not None else None
    if run_std is None and min(eval_runs.values()) >= 2:
        run_std = _pooled_run_std(
            before, after, target_key if target_key in results else "pass_at_1"
        )
        run_std_source = "eval_run" if run_std is not None else None
    replicated = run_std is not None
    # ``run_std`` is how far ONE side moves when the eval is re-run. The delta
    # is a difference of two such draws, so its own standard deviation is
    # ``run_std * sqrt(2)`` and a two-sigma band on it is ``2*sqrt(2)*run_std``,
    # not ``2*run_std``. Using the narrower band admits ~15% of pure-noise
    # deltas as real instead of ~5% (rlhf-book ch. 16, appendix C).
    noise = NOISE_SIGMAS * float(run_std) if run_std is not None else None
    within_noise: list[str] = []
    for m in metrics:
        r = results[m]
        r["within_noise"] = (
            noise is not None and r.get("delta") is not None and abs(r["delta"]) < noise
        )
        if r["within_noise"]:
            within_noise.append(m)
    loud = {m for m in metrics if not results[m]["within_noise"]}
    regressions = [
        m for m in metrics if m in guarded and m in loud and results[m]["verdict"] == "a_better"
    ]
    slipped = [
        m for m in metrics if m not in guarded and m in loud and results[m]["verdict"] == "a_better"
    ]
    improved = [m for m in metrics if m in loud and results[m]["verdict"] == "b_better"]
    target_result = results.get(target_key) if target_key else None
    if target_result is None and target_key:
        target_verdict = "target_not_measured"
    elif target_result is None:
        target_verdict = None
    else:
        target_verdict = _verdict_word(target_result, replicated)
    ok = not regressions and target_verdict not in {"moved_the_wrong_way"}
    warnings: list[str] = []
    if target_verdict == "moved_unreplicated":
        single = [side for side, n in eval_runs.items() if n < 2]
        where = "each side" if len(single) == 2 else f"the {single[0]} side"
        warnings.append(
            f"One eval run on {where}, so this could be noise. Run each side three times with "
            "simulate(tasks=..., runs=3) and the report will say."
        )
    for m in degenerate_guards:
        warnings.append(
            f"must_not_regress {m} is degenerate on both sides (every applicable row scored the "
            "same value): this guard cannot fail, so it catches nothing. Check that the marker "
            "fires at all."
        )
    if run_std_source == "eval_run" and min(eval_runs.values()) < 3:
        warnings.append(
            "Two eval runs on a side is a difference, not a distribution, so run_std is rough; "
            "three runs per side give a standard deviation worth reading."
        )
    # Every metric gets its own 95% interval, so the chance that at least one
    # clears zero by luck grows with the number of markers. The target is
    # pre-specified and keeps its 5%; the improved/slipped lists do not, and a
    # false flag in must_not_regress fails an otherwise good run.
    n_metrics = len(metrics)
    family_error = 1.0 - 0.95**n_metrics
    if n_metrics >= 4 and (improved or slipped or regressions):
        warnings.append(
            f"{n_metrics} metrics were each tested at 95%, so about a {family_error:.0%} chance "
            "that at least one clears zero by luck. The target is pre-specified and unaffected; "
            "treat single unexpected entries in improved/slipped as leads, not findings, and "
            "confirm them on a second eval run."
        )
    # ceiling: an eval the before side already passes cannot show a gain
    mean_a = results["pass_at_1"].get("mean_a")
    ceiling = False
    if mean_a is not None and mean_a >= CEILING_PASS_RATE:
        ceiling = True
        warnings.append(
            f"The before run already passes {mean_a:.2f} of tasks, so there is little room to "
            "measure improvement; use harder situations."
        )
    else:
        means_a, means_b = task_means(before), task_means(after)
        shared = set(means_a) & set(means_b)
        with_room = sum(1 for t in shared if means_a[t] < 1.0)
        if with_room < CEILING_MIN_TASKS_WITH_ROOM and with_room * 2 < len(shared):
            ceiling = True
            warnings.append(
                f"The before run already passes {len(shared) - with_room} of {len(shared)} paired "
                "tasks every time, so there is little room to measure improvement; use harder "
                "situations."
            )

    # proxy vs target: the book's over-optimization picture, as a verdict
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
        }[proxy_result["verdict"]]
        proxy_up = proxy_result["verdict"] == "b_better" and proxy_key in loud
        target_up = headline_for_proxy["verdict"] == "b_better" and headline_name in loud
        pci, tci = proxy_result.get("ci95"), headline_for_proxy.get("ci95")
        apart = bool(pci and tci and pci[0] > tci[1])
        over_optimized = (proxy_up and not target_up) or (proxy_up and apart)
        if over_optimized:
            ok = False
            tspan = f"{tci[0]:+.3f}..{tci[1]:+.3f}" if tci else "n/a"
            pspan = f"{pci[0]:+.3f}..{pci[1]:+.3f}" if pci else "n/a"
            warnings.append(
                f"OVER-OPTIMIZED: {proxy_key} up {proxy_result['delta']:+.3f} (95% {pspan}) while "
                f"{headline_name} {headline_for_proxy['delta']:+.3f} (95% {tspan}): the policy "
                "learned something the target does not credit (rlhf-book ch. 14)"
            )
    if noise is not None and target_verdict == "within_eval_noise" and target_result:
        warnings.append(
            f"{target_key}: {target_result['delta']:+.3f} is inside the eval's own re-run band "
            f"({NOISE_SIGMAS:.2f} x run_std = {noise:.3f}); re-running the eval moves it that much"
        )
    for m in regressions:
        r = results[m]
        warnings.append(
            f"REGRESSION {m}: {r['delta']:+.3f} (95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}), "
            "named in must_not_regress"
        )
    for m in slipped:
        r = results[m]
        warnings.append(
            f"{m} dropped {r['delta']:+.3f} (95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f})"
        )
    headline = target_result if target_result else results["pass_at_1"]
    headline_key = target_key if target_result else "pass_at_1"
    if headline.get("note"):
        warnings.append(f"{headline_key}: {headline['note']}")
    # Eval size: a no-change verdict is only as strong as the band the
    # task count allows. Say what this holdout can prove and what the
    # delta seen here would have needed (#257).
    n_paired = int(headline.get("n_paired") or 0)
    k_eval = int(pass_at(before).config.get("k") or 1)
    base_rate = float(mean_a) if mean_a is not None else 0.6
    can_prove = detectable_effect(n_paired, base=base_rate, k=k_eval) if n_paired >= 2 else None
    tasks_needed: int | None = None
    delta_seen: float | None = None
    raw_delta = headline.get("delta")
    if isinstance(raw_delta, (int, float)) and 0 < raw_delta < 1:
        delta_seen = float(raw_delta)
        tasks_needed = holdout_size(delta_seen, base=base_rate, k=k_eval)["n_tasks"]
    verdict_word = (
        target_verdict if target_result else _verdict_word(results["pass_at_1"], replicated)
    )
    if verdict_word == "no_change_detected" and can_prove is not None:
        line = (
            f"{n_paired} paired tasks at k={k_eval} can prove a gain of about "
            f"+{can_prove:.2f} at 80% power"
        )
        if tasks_needed is not None and delta_seen is not None:
            line += (
                f"; to prove the {delta_seen:+.3f} seen here you need about {tasks_needed} tasks"
            )
        warnings.append(line + " (holdout_size).")
    if target_verdict == "target_not_measured":
        warnings.append(f"target {target!r} is not on both row sets")
    groups: dict[str, dict[str, Any]] | None = None
    groups_down: list[str] = []
    if by is not None:
        group_metric = target_key if target_key in results else "pass_at_1"
        groups = _by_group(before, after, by=by, metric=group_metric, n_boot=n_boot, seed=seed)
        groups_down = [g for g, r in groups.items() if r.get("verdict") == "a_better"]
        for g in groups_down:
            r = groups[g]
            warnings.append(
                f"{group_metric} moved the wrong way for {g}: {r['delta']:+.3f} "
                f"(95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}, {r['rows_b']} rows)"
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
    if _both("truncated_share") and abs(cfg_a["truncated_share"] - cfg_b["truncated_share"]) > 0.05:
        warnings.append(
            f"The token cap cut {cfg_a['truncated_share']:.0%} of before rows and "
            f"{cfg_b['truncated_share']:.0%} of after rows; a side that is cut more often is "
            "not the same eval. Raise agent_max_tokens= on both sides or read the delta with "
            "that in mind."
        )
    # Rows that could not be graded leave the denominator, and they are not a
    # random sample: a long trajectory is both likelier to break a judge and
    # likelier to have failed, so a side that loses more rows scores higher for
    # that reason alone. Measured at 11.4 points on one lane. No interval sees
    # this, because it is selection, not variance.
    if _both("graded_share") and abs(cfg_a["graded_share"] - cfg_b["graded_share"]) > 0.02:
        ok = False
        warnings.append(
            f"ARMS NOT COMPARABLE: {cfg_a['graded_share']:.1%} of before rows were graded "
            f"against {cfg_b['graded_share']:.1%} of after rows. Ungraded rows leave the "
            "denominator and the ones that drop are the long ones, which fail more often, so "
            "the side that lost more rows is flattered. Re-grade the dropped rows before "
            "reading this delta."
        )
    elif _both("graded_share") and min(cfg_a["graded_share"], cfg_b["graded_share"]) < 0.95:
        warnings.append(
            f"Only {min(cfg_a['graded_share'], cfg_b['graded_share']):.1%} of rows on one side "
            "carry a verdict; both rates are over the rows that survived grading, not the rows "
            "that were run."
        )
    if _both("policy_version") and cfg_a["policy_version"] == cfg_b["policy_version"]:
        warnings.append(
            "Before and after are the same policy version; this compares a model to itself. "
            'Base and an adapter can share a served model name: pass advanced={"model_version": '
            '"...-base"} and "...-sft" so the two arms are distinguishable on the rows.'
        )
    # The environment has to hold still while the weights change. The
    # simulated user and the situation writer default to the agent's own
    # model, so in a before/after they follow the policy under test and the
    # delta measures the pair (rlhf-book ch. 16: every layer of an agentic
    # eval moves the score, so every layer is pinned and recorded).
    for key, knob in (("user_model", "user_model="), ("writer_model", "simulator=")):
        if _both(key) and cfg_a[key] != cfg_b[key]:
            ok = False
            warnings.append(
                f"ARMS NOT COMPARABLE: {key} was {cfg_a[key]!r} before and {cfg_b[key]!r} after. "
                f"The environment moved with the weights, so this delta measures the pair, not "
                f"the policy. Pin {knob} to one model on both arms and re-run."
            )
        elif cfg_a.get(key) is None and cfg_b.get(key) is None:
            continue
        else:
            agent_a = str(cfg_a.get("policy_version") or "").split("@", 1)[0]
            agent_b = str(cfg_b.get("policy_version") or "").split("@", 1)[0]
            if agent_a and cfg_a.get(key) == agent_a and cfg_b.get(key) == agent_b:
                warnings.append(
                    f"{key} is the agent's own model on both arms, so whoever it is ran on the "
                    f"policy under test. If the two arms served different weights under one name "
                    f"the environment moved with them; pin {knob} to a fixed model to rule it out."
                )
    return {
        "ok": ok,
        "target": target_key,
        "target_verdict": target_verdict,
        "target_delta": target_result["delta"] if target_result else None,
        "target_ci95": target_result["ci95"] if target_result else None,
        "n_metrics": n_metrics,
        #: chance at least one of the metrics clears zero by luck alone
        "family_error": round(family_error, 4),
        "n_paired_tasks": results["pass_at_1"]["n_paired"],
        "n_unpaired_tasks": results["pass_at_1"]["n_only_a"] + results["pass_at_1"]["n_only_b"],
        "improved": improved,
        "regressions": regressions,
        "slipped": slipped,
        "within_noise": within_noise,
        "run_std": float(run_std) if run_std is not None else None,
        "run_std_source": run_std_source,
        "eval_runs": eval_runs,
        "replicated": replicated,
        "ceiling": ceiling,
        "detectable_effect": can_prove,
        "tasks_needed": tasks_needed,
        "degenerate_guards": degenerate_guards,
        "proxy": proxy_key,
        "proxy_verdict": proxy_verdict,
        "proxy_delta": proxy_result["delta"] if proxy_result else None,
        "proxy_ci95": proxy_result["ci95"] if proxy_result else None,
        "over_optimized": over_optimized,
        "metrics": results,
        "warnings": warnings,
        "config": config,
        "by": (
            by if isinstance(by, str) else (getattr(by, "__name__", "callable") if by else None)
        ),
        "groups": groups,
        "groups_down": groups_down,
    }


def format_delta_report(report: dict[str, Any]) -> str:
    """The block a person reads: headline, then one line per metric."""
    lines: list[str] = []
    if report.get("target"):
        r = report["metrics"].get(report["target"])
        if r and r.get("delta") is not None and r.get("ci95"):
            lines.append(
                f"{report['target']}: {report['target_verdict']} "
                f"({r['delta']:+.3f}, 95% {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f}, "
                f"{r['n_paired']} paired tasks)"
            )
        else:
            lines.append(f"{report['target']}: {report['target_verdict']}")
    if report.get("proxy"):
        p = report["metrics"].get(report["proxy"])
        if p and p.get("delta") is not None and p.get("ci95"):
            lines.append(
                f"proxy {report['proxy']}: {report['proxy_verdict']} "
                f"({p['delta']:+.3f}, 95% {p['ci95'][0]:+.3f}..{p['ci95'][1]:+.3f})"
                + ("  OVER-OPTIMIZED" if report.get("over_optimized") else "")
            )
        else:
            lines.append(f"proxy {report['proxy']}: {report['proxy_verdict']}")
    lines.append("PASS" if report["ok"] else "FAIL")
    if report.get("run_std") is not None:
        runs = report.get("eval_runs") or {}
        source = (
            f"{runs.get('before')} eval runs before, {runs.get('after')} after"
            if report.get("run_std_source") == "eval_run"
            else "run_std given"
        )
        lines.append(
            f"eval noise: run_std {report['run_std']:.3f}, a delta under "
            f"{NOISE_SIGMAS * report['run_std']:.3f} is noise ({source})"
        )
    if report.get("ceiling"):
        lines.append("CEILING: the before run already passes most tasks; use harder situations")
    for name, r in report["metrics"].items():
        if r.get("delta") is None:
            lines.append(f"  {name:<28} insufficient data")
            continue
        ci = r.get("ci95")
        span = f"{ci[0]:+.3f}..{ci[1]:+.3f}" if ci else "n/a"
        tag = {
            "b_better": "up",
            "a_better": "DOWN",
            "no_difference_detected": "flat",
            "insufficient_data": "n/a",
        }[r["verdict"]]
        if r.get("within_noise"):
            tag = "noise"
        pair = "paired" if r["paired"] else "unpaired"
        lines.append(
            f"  {name:<28} {r['mean_a']:.3f} -> {r['mean_b']:.3f}  {r['delta']:+.3f} "
            f"[{span}]  {tag}  ({r['n_used']} {pair})"
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
            tag = {
                "b_better": "up",
                "a_better": "DOWN",
                "no_difference_detected": "flat",
                "insufficient_data": "n/a",
            }[r["verdict"]]
            lines.append(
                f"  {name:<28} {r['mean_a']:.3f} -> {r['mean_b']:.3f}  {r['delta']:+.3f} "
                f"[{span}]  {tag}  ({r['rows_a']}/{r['rows_b']} rows)"
            )
    for w in report.get("warnings") or []:
        lines.append(f"! {w}")
    return "\n".join(lines)


__all__ = ["delta_report", "format_delta_report"]
