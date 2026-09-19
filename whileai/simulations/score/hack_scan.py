"""What will the policy learn from this reward? Name it before training.

A grouped RL update (GRPO and its variants, Shao et al. 2024,
arXiv:2402.03300) baselines every rollout against the other rollouts of the
same ask. Whatever separates reward *within* an ask is the gradient; whatever
only tracks *which* ask it is (difficulty) is subtracted away. So the question
"is the reward paying for the behavior or for a shortcut?" has to be asked
within ask too. The pooled correlation ``reward_correlations`` reports cannot
tell the two apart: hard asks get long replies and low reward, and the pooled
number calls that a length penalty.

    Var(r) = E_g[Var(r | g)]  +  Var_g(E[r | g])
              (within: the gradient)  (between: the difficulty)

This scan centers reward and every candidate feature within ask, ranks
features by that correlation, and compares the top of the ranking to a
noise floor: the 95th percentile of the same maximum when reward is
shuffled within ask (difficulty preserved, signal destroyed). A feature
above the floor is something the policy will move toward. If the user
names what the reward *should* track (``endorsed=``) and the top feature
is not it, that is the reward hack, named (Gao et al. 2022,
arXiv:2210.10760: over-optimization is the training metric parting from
the evaluation of interest; the scan says on which feature).

Two feature tiers, both pure Python:

* the hand tier, always on: reply length, tool calls, turns, truncation,
  surface counts (digits, punctuation, newlines, uppercase share), one
  indicator per tool name called, one per trajectory flag that fired
  (``trace:lie.tests_claimed`` and the rest of ``score.trace``), the
  policy's mean token logprob when captured, and every numeric
  ``markers`` entry. Add your own with ``features=``.
* the auto tier (``auto=True``): presence of the ``top_k`` most common
  words and word pairs in the agent's text, plus pairwise ANDs of the
  strongest binary features. This is the tier that finds the hack nobody
  listed: a delimiter, a rubric word, an echoed fragment of the ask.

The permutation floor costs ``n_perm`` passes over the feature matrix.
Within-ask centering makes each pass a sum over a feature's non-zero
rows only, so a few thousand rollouts with a 200-term vocabulary scan
in seconds without numpy. Regimes and thresholds follow the RLVR signal
sweeps this descends from: ``sat`` above 0.2 was the one branch that
needed a calibrated cut; the rest is the floor.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...report import Report
from ..defaults import ALPHA, RL_ROLLOUTS_PER_ASK, ROLLOUTS_PER_TASK
from .hygiene import assistant_turns, is_truncated, reply_length, tool_calls
from .optimize import _messages

#: Collinear feature names shown in the degenerate-regime note before "and N more".
_COLLINEAR_SHOWN = 4

# DEFAULT_TOP_K = 200: auto-tier vocabulary, the most common words and
# word pairs. Enough to hold every delimiter and rubric word seen in the
# signal sweeps while a pass over a few thousand rollouts stays under a
# second (convention).
DEFAULT_TOP_K = 200
# DEFAULT_N_PERM = 100: permutations behind the noise floor. The floor is
# the (1 - alpha) quantile of the null maximum, so 100 draws place it
# between the 95th and 96th order statistic; more draws sharpen the
# floor at linear cost (convention).
DEFAULT_N_PERM = 100
# DEFAULT_MIN_OBS = 20: a feature must be non-zero on at least this many
# graded rows, or its within-ask correlation is a handful of rows
# (convention).
DEFAULT_MIN_OBS = 20
# RIVAL_SHARE = 0.2: a rival above the floor is reported when it carries
# at least this share of the endorsed signal (integrity below
# 1 - RIVAL_SHARE); under a fifth the endorsed feature still dominates
# what the update learns (convention).
RIVAL_SHARE = 0.2
# DEFAULT_SEEDS = 12: strongest binary features that seed pairwise
# conjunctions, 66 pairs (convention).
DEFAULT_SEEDS = 12
# SAT_FLAG = 0.2: share of asks the policy already always passes before
# the pool is called exhausted. The one calibrated threshold from the
# RLVR signal sweeps this scan descends from, and the same edge as the
# top of the 20-80 difficulty band read from the other side.
SAT_FLAG = 0.2
# REPORT_TOP = 20: ranked features the report lists.
REPORT_TOP = 20
# DEGENERATE_RHO = 0.999: |within-ask rho| at or above this is not a
# correlation but an identity: the feature is an exact linear function
# of the centered reward, and 0.999 is 1 with float rounding.
DEGENERATE_RHO = 0.999
# MIN_DISTINCT_PER_ASK = 3: distinct rollouts an ask needs before the
# within-ask ranking can separate anything. With two, whichever of them
# the reward follows, every feature that differs between them is an exact
# function of the label: they all land at |rho| 1 and the ranking is a
# sort by name.
MIN_DISTINCT_PER_ASK = 3
# MAX_BINARY_VARIANCE = 0.25: p(1-p) at p=0.5, the most variance a 0/1
# reward can carry within an ask; ``capacity`` is the mean within-ask
# variance as a share of it.
MAX_BINARY_VARIANCE = 0.25
# FLOOR_COARSE_BELOW = ROLLOUTS_PER_TASK (4) and RESCAN_ROLLOUTS =
# RL_ROLLOUTS_PER_ASK (8): under four rollouts per ask at the median the
# permutation floor has too few arrangements per ask to be sharp, and
# the warning names the RL rollout count to re-scan at.
FLOOR_COARSE_BELOW = ROLLOUTS_PER_TASK
RESCAN_ROLLOUTS = RL_ROLLOUTS_PER_ASK

REGIMES = ("train", "reward_hack", "pool_exhausted", "no_signal", "degenerate", "unknown")

_TOKEN = re.compile(r"[a-z_]{2,}|\d+|[^\sa-z_\d]")
_UPPER = re.compile(r"[A-Z]")
_PUNCT = re.compile(r"[^\w\s]")


# ----------------------------------------------------------------- inputs


def _reward(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _group_of(row: dict) -> str:
    """The GRPO group: rollouts of one ask. Same key ``group_signal`` and
    ``pass_at`` use, so the scan reads the groups the update would."""
    return str(row.get("prompt") or "")


def scan_text(row: Mapping[str, Any]) -> str:
    """Everything the agent said: the final reply and every assistant
    turn. Tool output is the world's text, not the policy's."""
    parts = [str(row.get("final_text") or "")]
    last = parts[0]
    for message in _messages(dict(row)):
        if str(message.get("role") or "") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content and content != last:
            parts.append(content)
    return "\n".join(p for p in parts if p)


def _tool_names(row: dict) -> list[str]:
    steps = row.get("steps") or row.get("tool_trace") or []
    return [str(s["tool"]) for s in steps if isinstance(s, dict) and s.get("tool")]


def hand_features(row: dict) -> dict[str, float]:
    """The hand tier for one row: the cheap things a judge rewards by
    accident, plus every marker already on the row."""
    text = scan_text(row)
    out: dict[str, float] = {
        "reply_length": float(reply_length(row)),
        "tool_calls": float(tool_calls(row)),
        "assistant_turns": float(assistant_turns(row)),
        "truncated": 1.0 if is_truncated(row) else 0.0,
        "n:words": float(len(text.split())),
        "n:digits": float(sum(c.isdigit() for c in text)),
        "n:newlines": float(text.count("\n")),
        "n:punct": float(len(_PUNCT.findall(text))),
        "frac:upper": (len(_UPPER.findall(text)) / len(text)) if text else 0.0,
    }
    for name in sorted(set(_tool_names(row))):
        out[f"tool:{name}"] = 1.0
    from .trace import trace_flags

    for name in trace_flags(row):
        out[f"trace:{name}"] = 1.0
    lp, n_tok = row.get("logprob"), row.get("n_tokens")
    if (
        isinstance(lp, (int, float))
        and not isinstance(lp, bool)
        and isinstance(n_tok, int)
        and not isinstance(n_tok, bool)
        and n_tok > 0
    ):
        out["logprob_mean"] = float(lp) / n_tok
    markers = row.get("markers")
    if isinstance(markers, dict):
        for name, value in markers.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"marker:{name}"] = float(value)
    return out


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def auto_terms(texts: Sequence[str], *, top_k: int, min_obs: int) -> tuple[list[str], list[set]]:
    """The ``top_k`` most common words and word pairs present in at least
    ``min_obs`` texts and absent from at least one, and each text's set."""
    present: list[set[str]] = []
    counts: Counter = Counter()
    for text in texts:
        toks = _tokens(text)
        terms = set(toks) | {f"{a} {b}" for a, b in itertools.pairwise(toks)}
        present.append(terms)
        counts.update(terms)
    n = len(texts)
    # Ties broken by the term, not by hash order: the same rows scan the
    # same way in every process.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = [t for t, c in ordered if min_obs <= c < n][:top_k]
    return vocab, present


# ----------------------------------------------------------------- the math


class _Feature:
    """One candidate column, stored sparse: the rows where it is non-zero.
    ``aliases`` are other columns with exactly the same entries (``tool_calls``
    when one tool is ever called, ``tool:<name>``): one feature, several
    names, so a duplicate never counts against the one it duplicates."""

    __slots__ = ("aliases", "binary", "entries", "name", "parents", "sd_pooled", "sd_within")

    def __init__(
        self, name: str, entries: list[tuple[int, float]], parents: tuple[_Feature, ...] = ()
    ):
        self.name = name
        self.entries = entries
        self.aliases: list[str] = []
        self.parents = parents
        self.binary = all(v == 1.0 for _, v in entries)
        self.sd_within = 0.0
        self.sd_pooled = 0.0

    @property
    def names(self) -> list[str]:
        return [self.name, *self.aliases]

    def endorsed(self, patterns: Sequence[str]) -> bool:
        """Named by an endorsed pattern, under any alias; a conjunction is
        endorsed when a parent is (it is that parent, narrowed)."""
        return any(_match(x, patterns) for x in self.names) or any(
            p.endorsed(patterns) for p in self.parents
        )


def _collapse(columns: dict[str, list[tuple[int, float]]], min_obs: int) -> list[_Feature]:
    """Columns with identical entries become one feature with aliases.
    Hand-tier names come first in ``columns`` and win the name."""
    by_key: dict[tuple, _Feature] = {}
    out: list[_Feature] = []
    for name, entries in columns.items():
        if len(entries) < min_obs:
            continue
        key = tuple(entries)
        if key in by_key:
            by_key[key].aliases.append(name)
            continue
        feature = _Feature(name, entries)
        by_key[key] = feature
        out.append(feature)
    return out


def _within_sd(feature: _Feature, group_of_row: list[int], group_size: list[int], n: int) -> float:
    """Standard deviation of the within-group-centered column, from the
    non-zero entries: sum over groups of (sum f^2 - (sum f)^2 / n_g)."""
    sums: dict[int, float] = {}
    sumsq: dict[int, float] = {}
    for i, v in feature.entries:
        g = group_of_row[i]
        sums[g] = sums.get(g, 0.0) + v
        sumsq[g] = sumsq.get(g, 0.0) + v * v
    total = sum(sumsq[g] - sums[g] * sums[g] / group_size[g] for g in sums)
    return math.sqrt(max(total, 0.0) / n) if n else 0.0


def _pooled_sd(feature: _Feature, n: int) -> float:
    s = sum(v for _, v in feature.entries)
    ss = sum(v * v for _, v in feature.entries)
    var = ss / n - (s / n) ** 2 if n else 0.0
    return math.sqrt(max(var, 0.0))


def _rho_within(feature: _Feature, rc: list[float], sr: float, n: int) -> float:
    """corr(centered feature, centered reward). The centered reward sums
    to zero in every group, so the raw feature can stand in for the
    centered one in the numerator: only its non-zero rows contribute."""
    if feature.sd_within <= 0.0 or sr <= 0.0:
        return 0.0
    num = sum(v * rc[i] for i, v in feature.entries)
    return num / (n * feature.sd_within * sr)


def _rho_pooled(feature: _Feature, rewards: list[float], n: int) -> float:
    sr = math.sqrt(max(sum(r * r for r in rewards) / n - (sum(rewards) / n) ** 2, 0.0))
    if feature.sd_pooled <= 0.0 or sr <= 0.0:
        return 0.0
    mean_f = sum(v for _, v in feature.entries) / n
    mean_r = sum(rewards) / n
    cov = sum(v * rewards[i] for i, v in feature.entries) / n - mean_f * mean_r
    return cov / (feature.sd_pooled * sr)


def _match(name: str, patterns: Sequence[str]) -> bool:
    low = name.lower()
    return any(str(p).lower() in low for p in patterns)


# ----------------------------------------------------------------- the scan


def hack_scan(
    rows: Sequence[dict],
    *,
    endorsed: Sequence[str] = (),
    features: Mapping[str, Callable[[dict], float | None]] | None = None,
    auto: bool = True,
    top_k: int = DEFAULT_TOP_K,
    n_perm: int = DEFAULT_N_PERM,
    min_obs: int = DEFAULT_MIN_OBS,
    seeds: int = DEFAULT_SEEDS,
    reward: str = "reward",
    seed: int = 0,
    top_features: int | None = REPORT_TOP,
    alpha: float = ALPHA,
) -> HackScanReport:
    """Rank the features that separate reward within each ask, against a permutation noise floor, and say what a grouped update would learn.

    Reach for it before an RL run, and again after, to check that the
    reward tracks the behavior you meant rather than a shortcut. It
    returns a ``HackScanReport``, a dict that prints itself: ``regime``
    (``train``, ``reward_hack``,
    ``pool_exhausted``, ``no_signal``, ``degenerate``, ``unknown``),
    ``tau`` (the floor), ``features`` ranked by |within-ask correlation|
    with the pooled correlation beside each, ``top_feature``,
    ``endorsed_on_top``, ``integrity`` (the share of the above-floor
    signal that sits on an endorsed feature), the support numbers (asks
    all-pass, all-fail, mixed, gradient capacity), and ``warnings`` in
    one line each.

    * ``rows``: graded rollouts, several per ask (``mode="rl"``); the
      reward under ``reward`` may be 0/1 or partial credit.
    * ``endorsed``: the features the reward is supposed to track, as
      substrings of feature names (``"lookup_order"`` matches
      ``tool:lookup_order`` and ``contains:lookup_order``;
      ``"marker:grounded"`` a marker). Without it the scan still ranks and
      floors, but cannot call a hack a hack.
    * ``features``: hand-tier columns to add, as
      ``{"name": lambda row: value}``, beside the built-in ones (reply
      length, tool calls,
      turns, truncation, one indicator per tool called, every numeric
      marker). ``auto`` (``True``) adds the auto tier: presence of the
      ``top_k`` (200) most common words and word pairs in the agent's
      text, the tier that finds the hack nobody listed.
    * ``alpha`` (``ALPHA``, 0.05): sets the floor. ``tau`` is the
      ``1 - alpha`` quantile of the strongest feature's |rho| when reward
      is shuffled within ask (``n_perm`` shuffles, 100), so a feature
      above it clears chance at that rate.
    * ``top_features`` (20): caps the ranking in the report (``None``
      lists all). ``min_obs`` (20) is the fewest observations a feature
      needs to be ranked.

    ``degenerate`` is the refusal: an ask holds fewer than
    ``MIN_DISTINCT_PER_ASK`` distinct rollouts at the median
    (``distinct_per_ask``) and two or more features sit at |rho| at or
    above ``DEGENERATE_RHO``, exactly collinear with reward and with each
    other because nothing else could happen at that variety. The ranking
    cannot separate them and the noise floor is no help (it tells signal
    from noise, not one perfect explanation from another), so
    ``top_feature`` and ``integrity`` are ``None``, ``inverted`` is empty,
    no hack is claimed, and ``collinear`` lists the tied features. The
    direction is withheld with the name: at that variety an endorsed
    feature is negative exactly when it fell on the failing trajectory,
    so the sign is the same coin flip. Collinear features on a varied
    pool are left alone: there the ranking found two names for one
    behavior, and a genuinely inverted endorsed feature is still
    reported.

    ```python
    scan = wai.hack_scan(data.rows(), endorsed=["tool:lookup_order"])
    print(scan["regime"], scan["top_feature"], scan["integrity"])
    ```
    """
    graded = [r for r in rows if isinstance(r, dict) and _reward(r, reward) is not None]
    n = len(graded)
    n_rows = sum(1 for r in rows if isinstance(r, dict))
    warnings: list[str] = []
    endorsed = [str(e) for e in endorsed if str(e)]
    base: dict[str, Any] = {
        "regime": "unknown",
        "n_rows": n_rows,
        "n_graded": n,
        "n_groups": 0,
        "n_groups_multi": 0,
        "rollouts_per_group": 0,
        "effective_rollouts": 0,
        "support": {"sat": 0.0, "dead": 0.0, "mixed": 0.0, "capacity": 0.0},
        "tau": None,
        "n_perm": int(n_perm),
        "n_features": 0,
        "n_above_floor": 0,
        "features": [],
        "top_feature": None,
        "rho_max": 0.0,
        "endorsed": endorsed,
        "endorsed_matched": 0,
        "endorsed_on_top": None,
        "integrity": None,
        "inverted": [],
        "continuous_reward": False,
        "degenerate": False,
        "collinear": [],
        "distinct_per_ask": 0,
        "warnings": warnings,
    }
    if n == 0:
        warnings.append(f"no row carries a numeric {reward!r}; grade first")
        return HackScanReport(base)

    # groups
    rewards = [v for r in graded if (v := _reward(r, reward)) is not None]
    group_index: dict[str, int] = {}
    group_of_row: list[int] = []
    for r in graded:
        key = _group_of(r)
        group_of_row.append(group_index.setdefault(key, len(group_index)))
    n_groups = len(group_index)
    members: list[list[int]] = [[] for _ in range(n_groups)]
    for i, g in enumerate(group_of_row):
        members[g].append(i)
    group_size = [len(m) for m in members]
    multi = [g for g in range(n_groups) if group_size[g] >= 2]  # noqa: PLR2004  # a group of one carries no contrast
    sizes = sorted(group_size[g] for g in multi)
    base["n_groups"] = n_groups
    base["n_groups_multi"] = len(multi)
    base["rollouts_per_group"] = sizes[len(sizes) // 2] if sizes else 1
    base["continuous_reward"] = any(v not in (0.0, 1.0) for v in rewards)

    # support: what the update could move
    all_pass = all_fail = unanimous = 0
    capacity = 0.0
    for g in range(n_groups):
        vals = [rewards[i] for i in members[g]]
        if min(vals) == max(vals):
            unanimous += 1
            if vals[0] >= 1.0:
                all_pass += 1
            elif vals[0] <= 0.0:
                all_fail += 1
        mean = sum(vals) / len(vals)
        capacity += sum((v - mean) ** 2 for v in vals) / len(vals)
    base["support"] = {
        "sat": all_pass / n_groups,
        "dead": all_fail / n_groups,
        "mixed": (n_groups - unanimous) / n_groups,
        "capacity": capacity / (MAX_BINARY_VARIANCE * n_groups),
    }
    live = [g for g in multi if len({rewards[i] for i in members[g]}) > 1]
    base["effective_rollouts"] = sum(group_size[g] for g in live)

    # centered reward
    group_mean = [sum(rewards[i] for i in m) / len(m) for m in members]
    rc = [rewards[i] - group_mean[group_of_row[i]] for i in range(n)]
    sr = math.sqrt(sum(v * v for v in rc) / n)

    # features
    texts = [scan_text(r) for r in graded]
    columns: dict[str, list[tuple[int, float]]] = {}
    profiles: list[tuple] = []
    for i, r in enumerate(graded):
        row_features = hand_features(r)
        if features:
            for name, fn in features.items():
                try:
                    value = fn(r)
                except Exception:
                    value = None
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row_features[str(name)] = float(value)
        profiles.append((texts[i], tuple(sorted(row_features.items()))))
        for name, value in row_features.items():
            if value != 0.0 and value == value:
                columns.setdefault(name, []).append((i, value))
    # How much variety the scan has to work with inside one ask, at the
    # median (the same summary ``rollouts_per_group`` reports). Features
    # can only be told apart by rollouts that differ.
    distinct = sorted(len({profiles[i] for i in members[g]}) for g in multi)
    base["distinct_per_ask"] = distinct[len(distinct) // 2] if distinct else 1
    if auto:
        vocab, present = auto_terms(texts, top_k=top_k, min_obs=min_obs)
        for term in vocab:
            columns[f"contains:{term}"] = [(i, 1.0) for i in range(n) if term in present[i]]
    feats = _collapse(columns, min_obs)
    for f in feats:
        f.sd_within = _within_sd(f, group_of_row, group_size, n)
        f.sd_pooled = _pooled_sd(f, n)
    feats = [f for f in feats if f.sd_pooled > 0.0]

    if not multi or sr <= 0.0:
        base["n_features"] = len(feats)
        base["features"] = [
            {
                "name": f.name,
                "aliases": f.aliases,
                "rho": 0.0,
                "pooled": round(_rho_pooled(f, rewards, n), 4),
                "n_obs": len(f.entries),
                "above_floor": False,
                "endorsed": f.endorsed(endorsed),
            }
            for f in sorted(feats, key=lambda f: -abs(_rho_pooled(f, rewards, n)))[:top_features]
        ]
        warnings.append(
            "every ask has one rollout, or every ask is unanimous: within-ask "
            "correlation needs repeats that disagree (mode='rl', repeats>=4); "
            "only the pooled column is filled"
            if multi
            else "one rollout per ask: within-ask correlation needs repeats "
            "(mode='rl', repeats>=4); only the pooled column is filled"
        )
        return HackScanReport(base)

    rho = {f.name: _rho_within(f, rc, sr, n) for f in feats}

    # conjunctions of the strongest binary features, on raw indicators
    # (centering does not commute with AND). A conjunction is kept only
    # when it beats both parents: otherwise it is a parent, narrowed.
    if auto and seeds > 0:
        binary = sorted((f for f in feats if f.binary), key=lambda f: -abs(rho[f.name]))[
            : int(seeds)
        ]
        extra: list[_Feature] = []
        for a_i, fa in enumerate(binary):
            rows_a = {i for i, _ in fa.entries}
            for fb in binary[a_i + 1 :]:
                both = sorted(rows_a & {i for i, _ in fb.entries})
                if (
                    len(both) >= min_obs
                    and len(both) < len(fa.entries)
                    and len(both) < len(fb.entries)
                ):
                    extra.append(
                        _Feature(f"{fa.name} AND {fb.name}", [(i, 1.0) for i in both], (fa, fb))
                    )
        for f in extra:
            f.sd_within = _within_sd(f, group_of_row, group_size, n)
            f.sd_pooled = _pooled_sd(f, n)
            if f.sd_pooled <= 0.0:
                continue
            value = _rho_within(f, rc, sr, n)
            if abs(value) > max(abs(rho[p.name]) for p in f.parents):
                feats.append(f)
                rho[f.name] = value

    # permutation floor: shuffle reward within ask, keep the max |rho|
    rng = random.Random(seed)
    nulls: list[float] = []
    perm = list(rewards)
    for _ in range(max(1, int(n_perm))):
        for g in multi:
            idx = members[g]
            vals = [perm[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                perm[i] = v
        rc_p = [perm[i] - group_mean[group_of_row[i]] for i in range(n)]
        best = 0.0
        for f in feats:
            if f.sd_within <= 0.0:
                continue
            value = abs(sum(v * rc_p[i] for i, v in f.entries) / (n * f.sd_within * sr))
            if value > best:
                best = value
        nulls.append(best)
    nulls.sort()
    tau = nulls[min(len(nulls) - 1, math.ceil((1 - alpha) * len(nulls)) - 1)] if nulls else 0.0

    # Ranked by strength; a tie in magnitude goes to the endorsed feature,
    # since the complement of the behavior correlates exactly as strongly
    # as the behavior and is not a second thing the policy learns.
    ranked = sorted(
        feats,
        key=lambda f: (
            -round(abs(rho[f.name]), 6),
            0 if f.endorsed(endorsed) and rho[f.name] > 0 else 1,
            f.name,
        ),
    )
    listed: list[dict[str, Any]] = [
        {
            "name": f.name,
            "aliases": f.aliases,
            "rho": round(rho[f.name], 4),
            "pooled": round(_rho_pooled(f, rewards, n), 4),
            "n_obs": len(f.entries),
            "above_floor": abs(rho[f.name]) > tau,
            "endorsed": f.endorsed(endorsed),
        }
        for f in ranked
    ]
    above = [x for x in listed if x["above_floor"]]
    top = listed[0] if listed else None
    rho_max = abs(top["rho"]) if top else 0.0
    matched = sum(1 for x in listed if x["endorsed"])
    base.update(
        {
            "tau": round(tau, 4),
            "n_features": len(listed),
            "n_above_floor": len(above),
            "features": listed[:top_features],
            "top_feature": top["name"] if top else None,
            "rho_max": round(rho_max, 4),
            "endorsed_matched": matched,
        }
    )

    # regime
    sat = base["support"]["sat"]
    if endorsed and matched == 0:
        base["regime"] = "unknown"
        warnings.append(
            f"endorsed={endorsed!r} matches none of the {len(listed)} features; fix the "
            "patterns (feature names are like tool:<name>, marker:<name>, contains:<term>) "
            "or add a features= extractor that emits them"
        )
        return HackScanReport(base)
    # Degeneracy: an ask that holds only a couple of distinct rollouts
    # forces every feature that separates them to be an exact function of
    # the label, so they all tie at |rho| 1 and the ranking's tie-break is
    # the feature name. The floor cannot help, because it separates signal
    # from noise and not one perfect explanation from another. Naming a
    # winner there is a coin flip presented as a verdict. Collinear
    # features on a diverse pool are a different thing (two names for the
    # same behavior, which the ranking found), so the trajectory variety
    # has to be missing too.
    collinear = [x["name"] for x in listed if abs(x["rho"]) >= DEGENERATE_RHO]
    degenerate = len(collinear) >= 2 and base["distinct_per_ask"] < MIN_DISTINCT_PER_ASK  # noqa: PLR2004  # collinear needs a pair
    base["degenerate"] = degenerate
    base["collinear"] = collinear
    # e: the reward pays for the endorsed behavior (positive). An endorsed
    # feature the reward punishes is ``inverted``: the policy will do less
    # of the behavior, which is a hack of its own. A degenerate scan
    # cannot read the sign either: with two distinct rollouts per ask the
    # endorsed feature is at -1 exactly when the tool happened to sit on
    # the failing trajectory, so "the reward punishes the behavior" there
    # is the same coin flip as naming a top feature, and flipping which
    # trajectory passed would flip the verdict on data of equal quality.
    e = max((x["rho"] for x in above if x["endorsed"] and x["rho"] > 0), default=0.0)
    a = max((abs(x["rho"]) for x in above if not x["endorsed"]), default=0.0)
    inverted = [] if degenerate else [x for x in above if x["endorsed"] and x["rho"] < 0]
    base["inverted"] = [x["name"] for x in inverted]
    integrity: float | None = None
    if endorsed and not degenerate:
        base["endorsed_on_top"] = bool(top and top["endorsed"] and top["rho"] > 0)
        integrity = round(e / (e + a), 4) if (e + a) > 0 else 0.0
        base["integrity"] = integrity
    if degenerate:
        base["regime"] = "degenerate"
        base["top_feature"] = None
        shown = ", ".join(f'"{name}"' for name in collinear[:_COLLINEAR_SHOWN])
        if len(collinear) > _COLLINEAR_SHOWN:
            shown += f", and {len(collinear) - _COLLINEAR_SHOWN} more"
        warnings.append(
            f"too few distinct trajectories to separate features: {base['distinct_per_ask']} "
            f"distinct rollout(s) per ask at the median leaves {len(collinear)} feature(s) "
            f"perfectly collinear with reward within ask (|rho| >= {DEGENERATE_RHO:g}): "
            f"{shown}. Nothing in the data tells them apart, so no top feature is named "
            "and no hack is claimed. Re-scan on rollouts that differ in more than one "
            "way (raise temperature or repeats), and scan before optimize(mode='rl'), "
            "which drops the duplicate rollouts within an ask"
        )
        # Say the sign is withheld too, rather than leaving a reader to
        # read "endorsed feature at -1.00" off the table and conclude it.
        tied_endorsed = [x["name"] for x in listed if x["endorsed"] and x["name"] in collinear]
        if tied_endorsed:
            warnings.append(
                "the endorsed "
                + ", ".join(f'"{name}"' for name in tied_endorsed[:3])
                + " sit(s) in that tie, so this scan cannot say whether the reward pays "
                "for the behavior or punishes it either: the sign there is whichever of "
                "the two trajectories happened to pass, not a direction the data supports"
            )
    elif not above or top is None:
        base["regime"] = "no_signal"
        warnings.append(
            f"no feature clears the noise floor (max |rho| {rho_max:.2f}, floor {tau:.2f} "
            f"from {len(nulls)} within-ask shuffles): the reward is not separating "
            "rollouts of the same ask on anything measurable; check the judge before "
            "training"
        )
    elif endorsed and not top["endorsed"]:
        base["regime"] = "reward_hack"
        warnings.append(
            f'reward is best explained by "{top["name"]}" (within-ask rho {top["rho"]:+.2f}, '
            f"floor {tau:.2f}), not by anything endorsed"
            + (f" (best endorsed {e:.2f})" if e else " (nothing endorsed clears the floor)")
            + f'; a policy trained on it learns "{top["name"]}"'
        )
    elif endorsed and top["rho"] < 0:
        base["regime"] = "reward_hack"
        warnings.append(
            f'reward punishes the endorsed "{top["name"]}" (within-ask rho {top["rho"]:+.2f}, '
            f"floor {tau:.2f}); a policy trained on it learns to do less of the behavior"
        )
        inverted = [x for x in inverted if x["name"] != top["name"]]
    elif sat > SAT_FLAG:
        base["regime"] = "pool_exhausted"
        warnings.append(
            f"{all_pass} of {n_groups} asks are already all-pass ({sat:.0%}); those groups "
            "carry no gradient, raise difficulty or drop them (optimize(mode='rl') does)"
        )
    else:
        base["regime"] = "train"
    if inverted:
        names = ", ".join(f'"{x["name"]}" ({x["rho"]:+.2f})' for x in inverted[:3])
        warnings.append(f"reward also punishes the endorsed {names}")
    if (
        endorsed
        and base["regime"] in ("train", "pool_exhausted")
        and a > 0
        and integrity is not None
        and integrity < 1.0 - RIVAL_SHARE
    ):
        rivals = [x["name"] for x in above if not x["endorsed"]][:3]
        warnings.append(
            "also above the floor, not endorsed: "
            + ", ".join(f'"{r}"' for r in rivals)
            + f" (integrity {base['integrity']:.2f})"
        )
    if base["rollouts_per_group"] < FLOOR_COARSE_BELOW:
        warnings.append(
            f"{base['rollouts_per_group']} rollouts per ask at the median; the floor is "
            f"coarse below {FLOOR_COARSE_BELOW}, re-scan at repeats>={RESCAN_ROLLOUTS} before "
            "acting on a close call"
        )
    return HackScanReport(base)


class HackScanReport(Report):
    """What ``hack_scan`` found, as an object that prints itself.

    Still the dict it always was: ``report["regime"]`` and
    ``report["warnings"]`` read the same. ``print(report)`` is now the
    block ``format_hack_scan`` writes, not the dict literal.
    """

    _summary_keys = ("regime", "integrity")

    def __str__(self) -> str:
        return format_hack_scan(self)


def format_hack_scan(report: dict[str, Any], *, top: int = 12) -> str:
    """The block a person reads: the regime, the floor, the ranking."""
    lines = [f"{report['regime'].upper().replace('_', ' ')}"]
    s = report.get("support") or {}
    lines.append(
        f"{report['n_graded']} graded rollouts, {report['n_groups']} asks, "
        f"{report['rollouts_per_group']} per ask; mixed {s.get('mixed', 0):.0%}, "
        f"all-pass {s.get('sat', 0):.0%}, all-fail {s.get('dead', 0):.0%}, "
        f"capacity {s.get('capacity', 0):.2f}"
    )
    if report.get("tau") is not None:
        lines.append(
            f"noise floor tau {report['tau']:.3f} ({report['n_above_floor']} of "
            f"{report['n_features']} features above)"
        )
        # Two marker columns, then a space, so a flagged endorsed feature
        # reads as "*e name" and not as "*ename".
        lines.append(f"   {'feature':<43}{'within':>8}{'pooled':>8}{'n':>6}")
        for x in report["features"][:top]:
            flag = "*" if x["above_floor"] else " "
            mark = "e" if x["endorsed"] else " "
            lines.append(
                f"{flag}{mark} {x['name'][:42]:<43}"
                f"{x['rho']:>+8.3f}{x['pooled']:>+8.3f}{x['n_obs']:>6}"
            )
    if report.get("integrity") is not None:
        lines.append(f"integrity {report['integrity']:.2f} (share of above-floor signal endorsed)")
    for w in report.get("warnings") or []:
        lines.append(f"! {w}")
    return "\n".join(lines)


def hack_scan_diff(
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    endorsed: Sequence[str] = (),
    top: int = 10,
    **scan_kwargs: Any,
) -> dict[str, Any]:
    """What the policy learned: the scan before training against the scan
    after, on rollouts scored by the same reward.

    A feature that clears the floor after and did not before is what the
    update moved toward; one that dropped out is what it moved away
    from. ``gained`` and ``lost`` list them with both correlations,
    ``moved`` the largest shifts either way, and ``learned`` is the one
    line to read: the top gained feature, and whether it is endorsed.
    ``scan_kwargs`` reach both ``hack_scan`` calls.

    When either side comes back ``degenerate``, every feature there is
    above the floor at |rho| 1 and no feature can be said to have gained
    it. ``learned`` says which side could not be read and why, and no
    hack is claimed; the rows are still listed so the shift is visible.
    """
    a = hack_scan(before, endorsed=endorsed, top_features=None, **scan_kwargs)
    b = hack_scan(after, endorsed=endorsed, top_features=None, **scan_kwargs)

    def index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for x in report["features"]:
            for name in (x["name"], *x.get("aliases", [])):
                out[name] = x
        return out

    fa, fb = index(a), index(b)
    names = sorted(set(fa) | set(fb))
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name in names:
        xa, xb = fa.get(name), fb.get(name)
        key = id(xb) if xb is not None else id(xa)
        if key in seen:
            continue  # an alias of a feature already listed
        seen.add(key)
        rows.append(
            {
                "name": name,
                "rho_before": xa["rho"] if xa else 0.0,
                "rho_after": xb["rho"] if xb else 0.0,
                "above_before": bool(xa and xa["above_floor"]),
                "above_after": bool(xb and xb["above_floor"]),
                "endorsed": bool((xb or xa or {}).get("endorsed")),
            }
        )
    for r in rows:
        r["shift"] = round(abs(r["rho_after"]) - abs(r["rho_before"]), 4)
    gained = sorted(
        (r for r in rows if r["above_after"] and not r["above_before"]),
        key=lambda r: -abs(r["rho_after"]),
    )
    lost = sorted(
        (r for r in rows if r["above_before"] and not r["above_after"]),
        key=lambda r: -abs(r["rho_before"]),
    )
    moved = sorted(rows, key=lambda r: -abs(r["shift"]))[: max(0, int(top))]
    warnings: list[str] = []
    # A degenerate scan on either side has every feature at |rho| 1 and
    # above the floor, so "gained the floor" is a tie and not a finding.
    # Naming the top of it would be the verdict ``hack_scan`` refuses to
    # give, arrived at one function later.
    degenerate = [side for side, report in (("before", a), ("after", b)) if report["degenerate"]]
    if degenerate:
        learned = (
            f"the {' and '.join(degenerate)} scan cannot say what the reward pays for: too "
            "few distinct trajectories left its candidate features collinear with reward, so "
            "what clears the floor there is a tie, not something the policy learned"
        )
        warnings.append(learned)
    elif gained:
        g = gained[0]
        learned = (
            f'the policy learned "{g["name"]}" (within-ask rho {g["rho_before"]:+.2f} -> '
            f"{g['rho_after']:+.2f})" + ("" if g["endorsed"] else ", which is not endorsed")
        )
        if endorsed and not g["endorsed"]:
            warnings.append(f"{learned}: a reward hack landed (Gao et al. 2022, arXiv:2210.10760)")
    elif b["top_feature"]:
        learned = (
            f"nothing new clears the floor after training; the top feature is still "
            f'"{b["top_feature"]}"'
        )
    else:
        learned = "no feature clears the floor after training"
    if a["regime"] != b["regime"]:
        warnings.append(f"regime {a['regime']} -> {b['regime']}")
    return {
        "before": {
            k: a[k] for k in ("regime", "top_feature", "rho_max", "tau", "integrity", "degenerate")
        },
        "after": {
            k: b[k] for k in ("regime", "top_feature", "rho_max", "tau", "integrity", "degenerate")
        },
        "gained": gained[: max(0, int(top))],
        "lost": lost[: max(0, int(top))],
        "moved": moved,
        "learned": learned,
        "warnings": warnings,
    }


def format_hack_scan_diff(report: dict[str, Any]) -> str:
    """The block a person reads: what was learned, then the shifts."""
    a, b = report["before"], report["after"]
    lines = [
        report["learned"],
        f"regime {a['regime']} -> {b['regime']}; top {a['top_feature']!r} -> {b['top_feature']!r}",
        f"   {'feature':<43}{'before':>8}{'after':>8}",
    ]
    for r in report["moved"]:
        tag = (
            "+"
            if r["above_after"] and not r["above_before"]
            else "-"
            if r["above_before"] and not r["above_after"]
            else " "
        )
        mark = "e" if r["endorsed"] else " "
        lines.append(
            f"{tag}{mark} {r['name'][:42]:<43}{r['rho_before']:>+8.3f}{r['rho_after']:>+8.3f}"
        )
    for w in report.get("warnings") or []:
        lines.append(f"! {w}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MIN_OBS",
    "DEFAULT_N_PERM",
    "DEFAULT_TOP_K",
    "DEGENERATE_RHO",
    "MIN_DISTINCT_PER_ASK",
    "REGIMES",
    "RIVAL_SHARE",
    "SAT_FLAG",
    "auto_terms",
    "format_hack_scan",
    "format_hack_scan_diff",
    "hack_scan",
    "hack_scan_diff",
    "hand_features",
    "scan_text",
]
