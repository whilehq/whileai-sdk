"""Confidence intervals, paired run comparison, and decontamination.

The numbers an engineer needs before believing a result (rlhf-book ch. 16:
labs win by raising the statistical power of the few evaluations they
track; contamination is found by n-gram overlap between training prompts
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
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_BOOT = 2000
#: tasks a bootstrap interval needs; below it the interval is the data itself
MIN_CI_TASKS = 3
_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _words(text: Any) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ------------------------------------------------------------------ intervals


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion. ``None`` when n is 0."""
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


def _paired_task_sd(base: float, effect: float, k: int) -> float:
    """Standard deviation of one task's paired difference (after minus
    before pass rate over ``k`` rollouts each side) when the gain lands
    uniformly: before at ``base``, after at ``base + effect``."""
    p = min(1.0, max(0.0, float(base)))
    q = min(1.0, max(0.0, p + float(effect)))
    kk = max(1, int(k))
    return math.sqrt((p * (1 - p) + q * (1 - q)) / kk)


def _rows_base_and_k(rows: Sequence[dict]) -> tuple[float, int]:
    """Mean per-task pass rate and the smallest rollouts-per-task on graded
    rows: what ``delta_report`` would pair on."""
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
    base = _mean([_mean(v) for v in groups.values()])
    k = min(len(v) for v in groups.values())
    return base, k


def holdout_size(
    effect: float,
    *,
    base: float = 0.6,
    k: int = 4,
    power: float = 0.8,
    alpha: float = 0.05,
    rows: Sequence[dict] | None = None,
) -> dict[str, Any]:
    """How many paired tasks a holdout needs to prove a gain of ``effect``.

    Models the test ``delta_report`` runs: each task's pass rate over
    ``k`` rollouts on each side, the delta as the mean of the paired
    differences, the interval from a bootstrap over tasks. A task's
    difference then has standard deviation
    ``sqrt((p(1-p) + q(1-q)) / k)`` with ``p = base`` and ``q = base +
    effect``, and the usual two-sided power calculation gives
    ``n = ((z_{1-alpha/2} + z_power) * sd / effect) ** 2`` (rlhf-book ch. 16,
    appendix C: the eval's own variance decides what a delta can mean).
    It assumes the gain lands uniformly across tasks; a gain concentrated
    on a few tasks needs more.

    ``rows`` (graded before-side rows) reads ``base`` and ``k`` off the
    data instead. Returns ``n_tasks`` plus the inputs, ``sd_task``, and
    ``half_width``: the 95% band on the delta at that ``n``.

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
    if rows is not None:
        base, k = _rows_base_and_k(rows)
    sd = _paired_task_sd(base, effect, k)
    z = _z(1 - alpha / 2) + _z(power)
    n = math.ceil((z * sd / float(effect)) ** 2) if sd > 0 else 1
    n = max(n, 2)
    return {
        "n_tasks": n,
        "effect": float(effect),
        "base": float(base),
        "k": int(k),
        "power": float(power),
        "alpha": float(alpha),
        "sd_task": round(sd, 4),
        "half_width": round(_z(1 - alpha / 2) * sd / math.sqrt(n), 4),
    }


def detectable_effect(
    n_tasks: int,
    *,
    base: float = 0.6,
    k: int = 4,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float | None:
    """The smallest gain ``n_tasks`` paired tasks can prove at ``power``:
    ``holdout_size`` solved for the effect (a few fixed-point steps, since
    the after-side variance depends on it). ``None`` below two tasks."""
    n = int(n_tasks)
    if n < 2:
        return None
    z = _z(1 - alpha / 2) + _z(power)
    effect = 0.0
    for _ in range(12):
        sd = _paired_task_sd(base, effect, k)
        effect = z * sd / math.sqrt(n)
    return round(min(1.0, effect), 4)


def bootstrap_ci(
    values: Sequence[float],
    *,
    stat: Callable[[Sequence[float]], float] = _mean,
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of ``stat`` over ``values``. ``None``
    below ``MIN_CI_TASKS`` values, where the interval would be the data
    itself."""
    vals = [float(v) for v in values]
    if len(vals) < MIN_CI_TASKS:
        return None
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo_i = int((1 - level) / 2 * n_boot)
    hi_i = int((1 + level) / 2 * n_boot) - 1
    return (stats[max(0, lo_i)], stats[min(n_boot - 1, hi_i)])


def task_key(row: dict) -> str:
    """The one name every report groups a row's rollouts under.

    A task is a situation, not a string: ``scenario_id`` when the row has
    one (the engine's situation id, shared by the repeats of one opener
    and by the textured phrasings of one situation), else ``task_id``
    (rows from elsewhere), else the prompt text. ``pass_at``,
    ``compare_runs``, ``delta_report``, ``eval_variance``, ``curriculum``,
    ``group_signal`` and the exporters all count tasks with this key, so
    the same rows give the same task count everywhere (rlhf-book ch. 16:
    intervals and paired comparisons are over tasks, never rows).
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


#: Olmo 3's bands for the standard deviation of a benchmark across re-runs
#: of one model, in points on a 0-100 scale: MMLU/MATH/PopQA sit near 0.2,
#: GPQA/AlpacaEval above 1.2. rlhf-book ch. 16, "Why Many External
#: Evaluation Comparisons Are Unreliable", puts most post-training
#: evaluations between 0.25 and 1.5 points with the setup held constant.
VARIANCE_BANDS = (("very_stable", 0.35), ("stable", 0.7), ("high_variance", float("inf")))


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


def eval_variance(
    *runs: Sequence[dict],
    metric: str = "pass_at_1",
    by: str | None = None,
) -> dict[str, Any]:
    """How much an evaluation moves when the same model is evaluated
    again (rlhf-book ch. 16, Evaluation).

    Pass each re-run's rows as its own argument, or one row list whose
    rows say which run they belong to: ``lineage.eval_run`` (what
    ``simulate(runs=3)`` stamps), else ``lineage.scoring_run_id`` (what
    ``evaluate(run_id=)`` stamps), or a top-level or lineage key named by
    ``by``. Each run's ``metric`` is a mean over tasks; the report is
    those means, their mean, the sample standard deviation ``run_std``,
    and ``noise_band`` = 2 x ``run_std``: a before/after delta inside it
    is what re-running the eval does on its own. Hand ``run_std`` to
    ``delta_report(run_std=)`` and it refuses to call such a delta a
    change. ``stability`` places ``run_std`` on Olmo 3's bands in points.
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
    means: dict[str, float | None] = {}
    task_sets: list[set[str]] = []
    for label, rows in zip(labels, groups):
        per_task = task_means(rows, metric)
        means[label] = round(_mean(list(per_task.values())), 4) if per_task else None
        task_sets.append(set(per_task))
    values = [v for v in means.values() if v is not None]
    n = len(values)
    mean = _mean(values) if values else None
    std = (
        (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
        if n >= 2 and mean is not None
        else None
    )
    common = set.intersection(*task_sets) if task_sets else set()
    stability = None
    if std is not None:
        points = std * 100
        stability = next(name for name, cap in VARIANCE_BANDS if points < cap)
    out: dict[str, Any] = {
        "metric": metric,
        "n_runs": n,
        "means": means,
        "mean": round(mean, 4) if mean is not None else None,
        "run_std": round(std, 4) if std is not None else None,
        "run_std_points": round(std * 100, 2) if std is not None else None,
        "noise_band": round(2 * std, 4) if std is not None else None,
        "stability": stability,
        "tasks_in_every_run": len(common),
        "notes": [],
    }
    if unkeyed:
        out["notes"].append(f"{unkeyed} row(s) carried no run id and were left out")
    if n < 3:
        out["notes"].append(
            f"{n} run(s): two is a difference, not a distribution; three or more re-runs "
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
    min_paired: int = 5,
) -> dict[str, Any]:
    """Is run ``b`` different from run ``a`` on ``metric``?

    Tasks the two runs share are compared as paired differences (b minus
    a, per task); the interval is a bootstrap over those pairs and the
    p-value is a sign-flip permutation test. With fewer than ``min_paired``
    shared tasks the comparison falls back to unpaired task means and says
    so. ``verdict`` is one of ``"b_better"``, ``"a_better"``,
    ``"no_difference_detected"``: the last means the interval covers zero,
    not that the runs are equal.

    Tasks on one side only are dropped from a paired comparison, and
    ``note`` says how many, since a verdict over a quarter of the tasks is
    not a verdict over the eval. ``paired_share`` is the shared fraction
    of every task either run saw.
    """
    ma = task_means(a, metric)
    mb = task_means(b, metric)
    shared = sorted(set(ma) & set(mb))
    paired = len(shared) >= min_paired
    rng = random.Random(seed)
    if paired:
        diffs = [mb[t] - ma[t] for t in shared]
        delta = _mean(diffs)
        ci = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
        # Sign-flip permutation: under H0 each paired difference is
        # equally likely to have either sign.
        observed = abs(delta)
        n = len(diffs)
        extreme = 0
        for _ in range(n_boot):
            flipped = _mean([d if rng.random() < 0.5 else -d for d in diffs])
            if abs(flipped) >= observed - 1e-12:
                extreme += 1
        p_value = (extreme + 1) / (n_boot + 1)
        n_used = n
    else:
        va, vb = list(ma.values()), list(mb.values())
        delta = (_mean(vb) - _mean(va)) if va and vb else float("nan")
        ci = None
        if len(va) >= 3 and len(vb) >= 3:
            boots = []
            for _ in range(n_boot):
                sa = _mean([va[rng.randrange(len(va))] for _ in va])
                sb = _mean([vb[rng.randrange(len(vb))] for _ in vb])
                boots.append(sb - sa)
            boots.sort()
            ci = (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1])
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
        if paired_share is not None and paired_share < 0.5:
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
    (rlhf-book ch. 16 decontaminates on prompt overlap)."""
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
    if len(sims) < 2:
        return None
    sims.sort()
    return min(1.0, sims[round(0.99 * (len(sims) - 1))])


def decontaminate(
    rows: Sequence[dict],
    against: Sequence[Any] | Any,
    *,
    n: int = 8,
    fields: Sequence[str] = ("prompt",),
    overlap: float = 0.8,
    embedder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
    similarity: float = 0.85,
) -> tuple[list[dict], dict[str, Any]]:
    """Drop rows whose prompt overlaps an evaluation set (rlhf-book ch. 16).

    ``against`` is one or more evaluation sources: row lists, JSONL paths,
    or platform dataset ids (``ds_...``). Evaluation prompts, answers and
    references are the texts (not the eval set's own replies). Four rules,
    applied in this order, and a row flagged by one is not counted again
    by the next, so ``n_contaminated`` is the number of rows dropped:

    * ``same_task`` (``n_same_task``): the row's ``scenario_id`` or
      ``task_id`` is an evaluation row's. A task is a situation, not a
      string (``task_key``), so a rephrasing of an eval situation is the
      eval situation whatever the words say. Rows with no recorded id
      skip this rule.
    * ``exact`` (``n_exact``): one of the row's ``fields`` is an evaluation
      text verbatim after normalization (case and whitespace).
    * near copy (``n_near``): one evaluation text covers at least
      ``overlap`` of the row's words with shared word ``n``-grams (the
      Llama 2 rule: 8-grams, 80% of tokens). Texts shorter than ``n``
      words match verbatim only.
    * ``semantic`` (``n_semantic``), only with ``embedder``: the cosine
      similarity between the row's text and an evaluation prompt is at
      least ``similarity``, and the two carry different task ids or none.

    The default field is the prompt, the book's method; add
    ``"final_text"`` to ask the stricter question of whether replies
    reproduce eval answers or references.

    One shared n-gram is the book's test for free-form sets. Situations
    written from templates share whole sentences that say nothing about
    which question was asked, so any-n-gram flags every row of a
    template-written set; the coverage rule counts a row when one eval
    text accounts for most of it. ``overlap=0`` restores any-n-gram.

    Word overlap does not see a paraphrase. A holdout written by
    re-running the generator on the same briefs was 70% within 0.85
    cosine of the training batch and 5 of 133 byte-identical; the 8-gram
    rule flagged 4 of 101 prompts and the semantic pass 16 (#286).
    ``embedder`` is any callable from a list of texts to one vector per
    text, so nothing here imports a model; with sentence-transformers::

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        clean, report = wai.decontaminate(
            train,
            against=[holdout],
            embedder=lambda texts: model.encode(texts, normalize_embeddings=True).tolist(),
        )

    A semantic flag means the two prompts read alike, not that they are
    the same task: "cancel one reservation" and "cancel three
    reservations" for different customers scored 0.932 with no shared
    answer. So where task identity is recorded the ``same_task`` rule
    decides and the semantic pass only looks across different tasks, and
    the report's ``notes`` say the flag is a question to check, not a
    verdict. The default stays lexical: ``similarity`` 0.85 was read off
    BGE (unrelated prompts score about 0.55 there) and does not transfer
    to every model, so the pass calibrates it for yours when it can: with
    eval rows that carry task ids, the 99th percentile of similarity over
    eval-prompt pairs with different task ids is how alike distinct tasks
    read to this embedder, and ``notes`` says it. A threshold below that
    number flags tasks that merely share a domain, and the note says so
    when ``similarity`` is.

    Returns the clean rows and a report: the count under each rule, hits
    per field, the eval text count, and the first offenders with their
    coverage (or ``similarity`` for semantic hits).
    """
    if embedder is not None and not 0 <= float(similarity) <= 1:
        raise ValueError("similarity is a cosine threshold between 0 and 1 (0.85 by default)")
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
    n_semantic = 0
    if embedder is not None and eval_prompts:
        eval_norms = list(eval_prompts)
        eval_vecs = _embed(embedder, [eval_prompts[key][0] for key in eval_norms])
        eval_task_ids = [eval_prompts[key][1] for key in eval_norms]
        alike = _distinct_task_similarity(eval_vecs, eval_task_ids)
        if alike is not None:
            note = (
                f"with this embedder, distinct tasks read up to {alike:.2f} alike (99th "
                f"percentile over {len(eval_norms)} eval prompts with different task ids); a "
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
    kept = [rows[i] for i in kept_index]
    total = sum(1 for r in rows if isinstance(r, dict))
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
        "notes": notes,
    }


__all__ = [
    "DEFAULT_BOOT",
    "MIN_CI_TASKS",
    "bootstrap_ci",
    "compare_runs",
    "decontaminate",
    "eval_variance",
    "marker_names",
    "marker_summary",
    "metric_summary",
    "task_means",
    "wilson_interval",
]
