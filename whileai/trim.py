"""``trim``: how many SFT rows your test can see, and which ones.

    rows = wai.select(scored, mode="sft")
    for n in rows.trim_sizes():                  # 64, 128, 256, ... len(rows)
        for seed in (0, 1):
            scores[n].append(train_and_eval(rows.trim(n), seed))   # per-item scores
    kept = rows.trim(scores, base=base_scores)   # base = the untuned model
    print(kept)

Two parts, each built from published results. Numbered references are at
the bottom of this docstring and in ``docs/trim.md``.

**Order.** ``trim(n)`` is the first ``n`` rows of one fixed ranking, so
the sizes nest and every point on the curve comes from one order. Exact
duplicates (and, with ``embed=``, near duplicates [9]) go last. The
default ranking is seeded random: on curated pools no published selector
beats random by more than about a point, inside random's own seed spread
[1][2]. Three rankers are opt in:

* ``by="diverse"``: k-means on your embeddings, round robin across
  clusters, longest response first inside a cluster. Embedding diversity
  tracks fine-tuned quality (r = 0.76 to 0.97) where lexical diversity
  does not (r about -0.3) [4]; clustering added 3 to 5 points and
  long responses beat full data in [1].
* ``by="target", dev=[...]``: closest to your dev examples by embedding
  cosine, the RDS+ idea, which beat random by 9 points on a 5.8M pool
  [3]. RDS+ round-robins over tasks; this ranks by the max over dev rows.
* ``by="bigram"``: no embeddings. Cost-benefit lazy greedy coverage of
  prompt word bigrams, cost = prompt words [10][11]. Kept for offline
  use; no study validates it for SFT [4].

**Curve.** Three forms are fit by weighted least squares, each point
weighted by ``1 / (seed variance / k + item variance / test_n)`` [12]:

* ``rectified``: ``base + (a - base) * n**c / (d + n**c)``, the rectified
  scaling law of [5] rewritten for a score, anchored at the untuned
  model's score at ``n = 0``. It has the flat "pre-power" phase that small
  SFT sets show and a plain power law cannot [5].
* ``power``: ``a - b * n**-c`` with ``a <= 1`` [6][7].
* ``logit``: ``logit(score) = alpha + beta * ln n``; math SFT accuracy is
  close to linear in ``log n`` [8], with no plateau below 1.

The form that best predicts the largest size from the others wins: forms
are compared by extrapolation, not in-sample fit [7].

**Stop.** ``N*`` is the smallest ``n`` past which one more doubling gains
less than ``min_gain`` (default 0.01, one point): a value you set by what
a point is worth, not the test's error bar, which would make ``N*`` grow
with the test's size. The rule is progressive sampling's [13] applied to
a fitted curve. With seeds or per-item scores, a bootstrap that
resamples test items (the same items at every size, a paired design
[12]) and seeds gives a 5 to 95% range on ``N*``; the verdict is
"saturated" only when ``N*`` is inside the measured sizes in at least
90% of draws. A paired minimum detectable effect between the two largest
sizes, ``2.8 * sqrt(2 * seed_var / k + var(item diffs) / test_n)``
[12], says whether ``min_gain`` is visible to this test at all. An
``N*`` past twice the largest size is a forecast [14].

References: [1] Xia et al. 2024, arXiv:2410.09335. [2] Diddee & Ippolito
2024, arXiv:2410.15225. [3] Ivison et al. 2025, arXiv:2503.01807.
[4] NovelSum 2025, arXiv:2502.17184. [5] Lin et al. 2024,
arXiv:2402.02314. [6] Hestness et al. 2017, arXiv:1712.00409.
[7] Alabdulmohsin et al. 2022, arXiv:2209.06640. [8] Yuan et al. 2023,
arXiv:2308.01825. [9] Abbas et al. 2023 (SemDeDup), arXiv:2303.09540.
[10] Lin & Bilmes 2010, NAACL "Multi-document summarization via budgeted
maximization of submodular functions". [11] Leskovec et al. 2007, KDD
"Cost-effective outbreak detection in networks" (lazy greedy).
[12] Miller 2024, arXiv:2411.00640. [13] Provost, Jensen & Oates 1999,
KDD "Efficient progressive sampling". [14] Lourie et al. 2025,
arXiv:2507.00885.

Stdlib only; embeddings come from your ``embed=`` callable.
"""

from __future__ import annotations

import heapq
import math
import random
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SaturationFit", "fit_saturation", "order", "trim_sizes"]

Embed = Callable[[list[str]], Sequence[Sequence[float]]]

#: RANKERS: the ``by=`` values ``order`` accepts.
RANKERS = ("random", "diverse", "target", "bigram")

#: MIN_GAIN = 0.01: the smallest gain per doubling worth more rows, one
#: point on a 0-1 score. A cost judgement, so it is yours to set; one
#: point is the size of the random-vs-selector gap in [1] (convention,
#: untested).
MIN_GAIN = 0.01

#: MIN_SIZES = 5: measured sizes (n >= 1) the fit needs. Up to four
#: parameters, one size held out to pick the form (convention, untested).
MIN_SIZES = 5

#: FIRST_SIZE = 64: the smallest size ``trim_sizes`` proposes; [5] starts
#: its SFT grids at 200 and LIMA (Zhou et al. 2023, arXiv:2305.11206)
#: trains on 1,000, so 64 sits in the pre-power phase on purpose
#: (convention, untested).
FIRST_SIZE = 64

#: GROWTH = 2.0: ratio between proposed sizes; learning curves are read on
#: a log scale [6][13].
GROWTH = 2.0

#: NEAR_DUP = 0.95: embedding cosine at or above which a row is a near
#: duplicate of an earlier one. SemDeDup [9] tunes its threshold per
#: corpus; 0.95 is a conservative default (convention, untested).
NEAR_DUP = 0.95

#: KMEANS_ITERS = 25: Lloyd iterations for ``by="diverse"``, seeded with
#: k-means++ (Arthur & Vassilvitskii 2007, SODA). Clusters k =
#: ceil(sqrt(n / 2)), the rule of thumb in Mardia et al. 1979
#: (convention, untested).
KMEANS_ITERS = 25

#: SATURATED_SHARE = 0.9: share of bootstrap draws whose ``N*`` must sit
#: inside the measured sizes to call the test saturated (convention,
#: untested).
SATURATED_SHARE = 0.9

#: FAILED_SHARE = 0.1: share of failed bootstrap refits above which ``N*``
#: is "not identifiable" (convention, untested).
FAILED_SHARE = 0.1

#: MIN_EXPONENT = 0.1: a fitted power exponent below this makes ``N*``
#: swing by orders of magnitude for a small change in the data; fine-tuning
#: exponents of 0.015 to 0.15 are reported in loss by Zhang et al. 2024,
#: arXiv:2402.17193.
MIN_EXPONENT = 0.1

#: FORECAST_RATIO = 2.0: an ``N*`` past this multiple of the largest
#: measured size is a forecast; downstream curves extrapolate badly [14].
FORECAST_RATIO = 2.0

#: DRAWS = 200: bootstrap draws behind the ``N*`` range; a 5-95% percentile
#: interval is usable from about 200 (Efron & Tibshirani 1993, ch. 13).
DRAWS = 200

#: INTERVAL = (0.05, 0.95): percentiles of the reported ``N*`` range.
INTERVAL = (0.05, 0.95)

#: MDE_Z = 2.8: z(0.975) + z(0.80), the minimum detectable effect factor at
#: 5% two-sided and 80% power [12].
MDE_Z = 2.8

#: VAR_FLOOR = 1e-6: smallest variance a point may carry, so a size scored
#: identically on every seed does not take infinite weight.
VAR_FLOOR = 1e-6

#: LOGIT_EPS = 1e-3: scores are clipped to [eps, 1 - eps] before a logit.
LOGIT_EPS = 1e-3

#: NGRAM = 2: word n-gram of a prompt feature for ``by="bigram"``.
NGRAM = 2

# grids the least-squares fits search; closed form inside each cell
_C_GRID = tuple(0.05 * (60 ** (i / 59)) for i in range(60))  # 0.05 .. 3.0
_D_GRID = tuple(10 ** (i / 6) for i in range(31))  # 1 .. 1e5
_N_GRID = tuple(10 ** (i / 300) for i in range(2701))  # 1 .. 1e9

_WORD = re.compile(r"\w+")


# ---------------------------------------------------------------------
# order
# ---------------------------------------------------------------------


def _prompt(row: Mapping[str, Any]) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    msgs = row.get("messages") or row.get("conversation") or []
    return " ".join(
        str(m.get("content") or "")
        for m in msgs
        if isinstance(m, Mapping) and m.get("role") == "user"
    )


def _response(row: Mapping[str, Any]) -> str:
    for key in ("response", "completion", "answer"):
        if row.get(key):
            return str(row[key])
    msgs = row.get("messages") or row.get("conversation") or []
    said = [
        str(m.get("content") or "")
        for m in msgs
        if isinstance(m, Mapping) and m.get("role") == "assistant"
    ]
    return said[-1] if said else ""


def _text(row: Mapping[str, Any]) -> str:
    return f"{_prompt(row)}\n{_response(row)}"


def _cos(u: Sequence[float], v: Sequence[float]) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    return 0.0 if nu == 0 or nv == 0 else sum(a * b for a, b in zip(u, v, strict=True)) / (nu * nv)


def _dedupe(
    rows: Sequence[Mapping[str, Any]], vecs: Sequence[Sequence[float]] | None
) -> tuple[list[int], list[int]]:
    """Split indices into (unique, duplicates) in input order."""
    seen: set[str] = set()
    unique: list[int] = []
    dups: list[int] = []
    for i, r in enumerate(rows):
        key = " ".join(_WORD.findall(_text(r).lower()))
        near = vecs is not None and any(_cos(vecs[i], vecs[j]) >= NEAR_DUP for j in unique)
        if key in seen or near:
            dups.append(i)
        else:
            seen.add(key)
            unique.append(i)
    return unique, dups


def _d2(u: Sequence[float], v: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(u, v, strict=True))


def _kmeans(
    vecs: Sequence[Sequence[float]], idx: list[int], k: int, rng: random.Random
) -> list[list[int]]:
    """k-means++ seeding, then Lloyd; returns member lists, largest first."""
    centers = [list(vecs[rng.choice(idx)])]
    while len(centers) < k:
        dist = [min(_d2(vecs[i], c) for c in centers) for i in idx]
        if sum(dist) == 0:
            break
        centers.append(list(vecs[rng.choices(idx, weights=dist)[0]]))
    groups: list[list[int]] = []
    for _ in range(KMEANS_ITERS):
        groups = [[] for _ in centers]
        for i in idx:
            groups[min(range(len(centers)), key=lambda c: _d2(vecs[i], centers[c]))].append(i)
        moved = False
        for c, members in enumerate(groups):
            if members:
                mean = [
                    sum(col) / len(members) for col in zip(*(vecs[i] for i in members), strict=True)
                ]
                moved |= mean != centers[c]
                centers[c] = mean
        if not moved:
            break
    return sorted((g for g in groups if g), key=len, reverse=True)


def _bigrams(text: str) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < NGRAM:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def _bigram_order(rows: Sequence[Mapping[str, Any]], idx: list[int]) -> list[int]:
    """Cost-benefit lazy greedy [10][11]: new bigrams per prompt word."""
    feats = {i: _bigrams(_prompt(rows[i])) for i in idx}
    cost = {i: max(len(_WORD.findall(_prompt(rows[i]))), 1) for i in idx}
    covered: set[tuple[str, ...]] = set()
    heap = [(-len(feats[i]) / cost[i], i) for i in idx]
    heapq.heapify(heap)
    out: list[int] = []
    while heap:
        neg, i = heapq.heappop(heap)
        ratio = len(feats[i] - covered) / cost[i]
        if ratio < -neg and heap and (-ratio, i) > heap[0]:
            heapq.heappush(heap, (-ratio, i))
            continue
        out.append(i)
        covered |= feats[i]
    return out


def order(
    rows: Sequence[Mapping[str, Any]],
    by: str = "random",
    *,
    seed: int = 0,
    embed: Embed | None = None,
    dev: Sequence[Any] | None = None,
) -> list[int]:
    """Row indices, first to keep first; duplicates last. See the module
    docstring for what each ``by`` does and the evidence behind it."""
    if by not in RANKERS:
        raise ValueError(f"order(by={by!r}): one of {', '.join(RANKERS)}")
    if by in ("diverse", "target") and embed is None:
        raise ValueError(
            f'order(by="{by}") needs embed=, a function from a list of texts to vectors'
        )
    if by == "target" and not dev:
        raise ValueError('order(by="target") needs dev=, examples of what the test asks')
    vecs = [list(v) for v in embed([_text(r) for r in rows])] if embed is not None else None
    unique, dups = _dedupe(rows, vecs)
    rng = random.Random(seed)
    if by == "random":
        rng.shuffle(unique)
        return unique + dups
    if by == "bigram":
        return _bigram_order(rows, unique) + dups
    assert vecs is not None and embed is not None
    if by == "target":
        dev_texts = [_text(d) if isinstance(d, Mapping) else str(d) for d in dev or ()]
        dvecs = [list(v) for v in embed(dev_texts)]
        best = {i: max(_cos(vecs[i], d) for d in dvecs) for i in unique}
        return sorted(unique, key=lambda i: (-best[i], i)) + dups
    k = max(1, math.ceil(math.sqrt(len(unique) / 2)))
    clusters = [
        sorted(g, key=lambda i: (-len(_response(rows[i])), i))
        for g in _kmeans(vecs, unique, k, rng)
    ]
    out: list[int] = []
    for depth in range(max((len(g) for g in clusters), default=0)):
        out.extend(g[depth] for g in clusters if depth < len(g))
    return out + dups


def trim_sizes(n_rows: int, *, first: int = FIRST_SIZE, growth: float = GROWTH) -> list[int]:
    """Nested sizes to train at: ``first``, ``first * growth``, ... up to
    ``n_rows``. Train every size with the same epochs (or the same steps)
    and say which: at a fixed step budget, many epochs on few rows can
    beat one pass on many (arXiv:2602.11149)."""
    if n_rows < 1:
        raise ValueError(f"trim_sizes(n_rows={n_rows}): need at least one row")
    if growth <= 1:
        raise ValueError(f"trim_sizes(growth={growth}): sizes must grow, so growth > 1")
    sizes, n = [], float(min(first, n_rows))
    while round(n) < n_rows:
        sizes.append(round(n))
        n *= growth
    return [*sizes, n_rows]


# ---------------------------------------------------------------------
# curve
# ---------------------------------------------------------------------

Score = float | Sequence[float] | Sequence[Sequence[float]]


def _as_matrix(value: Score) -> list[list[float]]:
    """A size's scores as seeds x items. A number is one seed, one item; a
    list of numbers is one score per seed; a list of lists is per item."""
    if isinstance(value, int | float):
        return [[float(value)]]
    seq: list[Any] = list(value)
    if all(isinstance(v, int | float) for v in seq):
        return [[float(v)] for v in seq]
    return [[float(x) for x in s] for s in seq]


@dataclass(frozen=True)
class _Point:
    n: int
    y: float
    var: float | None
    seed_means: list[float]
    item_means: list[float] | None


def _point(n: int, m: list[list[float]], paired: bool) -> _Point:
    seed_means = [sum(s) / len(s) for s in m]
    y = sum(seed_means) / len(seed_means)
    parts: list[float] = []
    if len(seed_means) > 1:
        parts.append(statistics.variance(seed_means) / len(seed_means))
    items = [sum(col) / len(col) for col in zip(*m, strict=True)] if paired else None
    if items is not None and len(items) > 1:
        parts.append(statistics.variance(items) / len(items))
    var = max(sum(parts), VAR_FLOOR) if parts else None
    return _Point(n, y, var, seed_means, items)


def _wls1(gs: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> tuple[float, float]:
    """``y = s * g``; returns (s, weighted sse)."""
    den = sum(w * g * g for g, w in zip(gs, ws, strict=True))
    s = 0.0 if den == 0 else sum(w * g * y for g, y, w in zip(gs, ys, ws, strict=True)) / den
    return s, sum(w * (y - s * g) ** 2 for g, y, w in zip(gs, ys, ws, strict=True))


def _wls2(
    xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]
) -> tuple[float, float, float]:
    """``y = a + s * x``; returns (a, s, weighted sse)."""
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws, strict=True)) / sw
    my = sum(w * y for y, w in zip(ys, ws, strict=True)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws, strict=True))
    s = (
        0.0
        if sxx == 0
        else sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws, strict=True)) / sxx
    )
    a = my - s * mx
    return a, s, sum(w * (y - a - s * x) ** 2 for x, y, w in zip(xs, ys, ws, strict=True))


def _predict(form: str, p: tuple[float, ...], n: float) -> float:
    if form == "rectified":
        base, a, c, d = p
        return base + (a - base) * n**c / (d + n**c)
    if form == "power":
        a, b, c = p
        return a - b * n**-c
    alpha, beta = p
    return 1 / (1 + math.exp(-(alpha + beta * math.log(n))))


def _fit(form: str, pts: Sequence[_Point], base: float | None) -> tuple[float, ...] | None:
    ns = [p.n for p in pts]
    ys = [p.y for p in pts]
    ws = [1.0 / p.var for p in pts if p.var] if all(p.var for p in pts) else [1.0] * len(pts)
    best: tuple[float, tuple[float, ...]] | None = None
    if form == "power":
        for c in _C_GRID:
            a, s, sse = _wls2([n**-c for n in ns], ys, ws)
            if -s > 0 and a <= 1 and (best is None or sse < best[0]):
                best = (sse, (a, -s, c))
    elif form == "rectified":
        for c in _C_GRID:
            for d in _D_GRID:
                gs = [n**c / (d + n**c) for n in ns]
                if base is not None:
                    s, sse = _wls1(gs, [y - base for y in ys], ws)
                    b0 = base
                else:
                    b0, s, sse = _wls2(gs, ys, ws)
                if s > 0 and b0 + s <= 1 and (best is None or sse < best[0]):
                    best = (sse, (b0, b0 + s, c, d))
    else:
        zs, zw = [], []
        for y, w in zip(ys, ws, strict=True):
            q = min(max(y, LOGIT_EPS), 1 - LOGIT_EPS)
            zs.append(math.log(q / (1 - q)))
            zw.append(w * (q * (1 - q)) ** 2)  # delta method
        alpha, beta, _ = _wls2([math.log(n) for n in ns], zs, zw)
        return (alpha, beta) if beta > 0 else None
    return best[1] if best else None


def _n_star(form: str, p: tuple[float, ...], min_gain: float, floor: int) -> float:
    """Smallest n past which every doubling gains under ``min_gain``;
    ``inf`` when that is past 1e9."""
    last = -1
    for i, n in enumerate(_N_GRID):
        if _predict(form, p, 2 * n) - _predict(form, p, n) > min_gain:
            last = i
    if last == len(_N_GRID) - 1:
        return math.inf
    return float(max(floor, math.ceil(_N_GRID[last + 1]))) if last >= 0 else float(floor)


@dataclass(frozen=True)
class SaturationFit:
    """The fitted curve, ``N*``, and how sure the data is about it."""

    form: str
    params: tuple[float, ...]
    min_gain: float
    n_saturated: float
    verdict: str
    curve: dict[int, float]
    base: float | None = None
    interval: tuple[float, float] | None = None
    share_saturated: float | None = None
    heldout_error: dict[str, float] = field(default_factory=dict)
    mde: float | None = None
    last_gain: float | None = None
    last_gain_interval: tuple[float, float] | None = None
    notes: tuple[str, ...] = ()

    def predict(self, n: float) -> float:
        if n == 0 and self.base is not None:
            return self.base
        return _predict(self.form, self.params, n)

    @property
    def saturated(self) -> bool:
        return self.verdict in ("saturated", "flat")

    def __str__(self) -> str:
        top = max(self.curve)
        fmt = {
            "rectified": "base + (a - base) * n^c / (d + n^c); base {:.3f}, a {:.3f}, c {:.2f}, d {:.3g}",
            "power": "a - b * n^-c; a {:.3f}, b {:.3g}, c {:.3f}",
            "logit": "logit(score) = {:.3f} + {:.3f} * ln n",
        }[self.form]
        held = ", ".join(
            f"{k} {v:.3f}" for k, v in sorted(self.heldout_error.items(), key=lambda kv: kv[1])
        )
        lines = [
            f"verdict: {self.verdict}; N* = {_rows(self.n_saturated)} (next doubling gains < {self.min_gain:.3f})",
            f"  fit: {fmt.format(*self.params)}",
            f"  error predicting n={top} from the rest: {held}",
        ]
        if self.interval is not None:
            lo, hi = self.interval
            lines.append(
                f"  N* 90% range {_rows(lo)} .. {_rows(hi)}; "
                f"inside the sizes trained in {self.share_saturated:.0%} of draws"
            )
        if self.last_gain is not None:
            gap = ""
            if self.last_gain_interval is not None:
                gap = f" [{self.last_gain_interval[0]:+.3f}, {self.last_gain_interval[1]:+.3f}]"
            lines.append(f"  last doubling gained {self.last_gain:+.3f}{gap}")
        if self.mde is not None:
            lines.append(f"  smallest gain this test can detect: {self.mde:.3f}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def _rows(n: float) -> str:
    return "past 1e9 rows" if math.isinf(n) else f"{n:,.0f} rows"


def _percentile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def _mde(a1: _Point, a0: _Point) -> float | None:
    """Paired, seed-aware minimum detectable gain between two sizes [12]."""
    parts = []
    k = min(len(a1.seed_means), len(a0.seed_means))
    if k > 1:
        sv = (statistics.variance(a1.seed_means) + statistics.variance(a0.seed_means)) / 2
        parts.append(2 * sv / k)
    if a1.item_means is not None and a0.item_means is not None:
        diffs = [x - y for x, y in zip(a1.item_means, a0.item_means, strict=True)]
        parts.append(statistics.variance(diffs) / len(diffs))
    # no seed spread and no item that changed: nothing to measure noise from
    return MDE_Z * math.sqrt(sum(parts)) if sum(parts) > 0 else None


def fit_saturation(
    curve: Mapping[int, Score],
    *,
    base: Score | None = None,
    min_gain: float = MIN_GAIN,
    draws: int = DRAWS,
    seed: int = 0,
) -> SaturationFit:
    """Fit the learning curve ``{n_rows: scores}`` and find ``N*``.

    Each value is one score, one score per training seed, or per-item
    scores per seed (seeds x items, the same items in the same order at
    every size). ``base`` is the untuned model's score(s) on the same
    test, the ``n = 0`` anchor. Seeds or items turn on the weights, the
    bootstrap range and the detectable-gain check.
    """
    if min_gain <= 0:
        raise ValueError(f"fit_saturation(min_gain={min_gain}): a gain per doubling, above 0")
    mats = {int(n): _as_matrix(v) for n, v in curve.items()}
    if any(n < 1 for n in mats):
        raise ValueError(
            "fit_saturation: every curve key is a row count >= 1; pass the untuned model as base="
        )
    if len(mats) < MIN_SIZES:
        raise ValueError(
            f"fit_saturation: {len(mats)} sizes; need at least {MIN_SIZES} (rows.trim_sizes() lists them)"
        )
    widths = {len(s) for m in mats.values() for s in m}
    paired = len(widths) == 1 and next(iter(widths)) > 1
    ns = sorted(mats)
    base_m = _as_matrix(base) if base is not None else None
    base_y = None if base_m is None else sum(sum(s) / len(s) for s in base_m) / len(base_m)

    def points(ms: Mapping[int, list[list[float]]]) -> list[_Point]:
        return [_point(n, ms[n], paired) for n in ns]

    pts = points(mats)
    curve_y = {p.n: p.y for p in pts}
    notes: list[str] = []
    if any(p.var is None for p in pts):
        notes.append(
            "one score per size: equal weights and no range on N*; pass seeds or per-item scores"
        )
    if base_y is None:
        notes.append("no base=: the rectified form fits its own n=0 score")

    # the form that best predicts the largest size from the rest wins [7]
    held: dict[str, float] = {}
    for f in ("rectified", "power", "logit"):
        p = _fit(f, pts[:-1], base_y)
        if p is not None:
            held[f] = abs(_predict(f, p, ns[-1]) - pts[-1].y)
    form, params = "", None
    for form in sorted(held, key=held.__getitem__):
        params = _fit(form, pts, base_y)
        if params is not None:
            break
    if params is None:
        return SaturationFit(
            form="power",
            params=(curve_y[ns[-1]], 0.0, 1.0),
            min_gain=min_gain,
            n_saturated=float(ns[0]),
            verdict="flat",
            curve=curve_y,
            base=base_y,
            heldout_error=held,
            notes=(*notes, "no rising curve fits: more rows did not move the test"),
        )
    if form == "logit":
        notes.append("log-linear won: the data cannot tell a plateau from a steady climb [8]")
    n_star = _n_star(form, params, min_gain, ns[0])

    # bootstrap: test items resampled once per draw for every size (paired),
    # seeds resampled within each size [12]
    interval = share = gain_iv = None
    failed = 0
    if all(p.var is not None for p in pts) and draws > 0:
        rng = random.Random(seed)
        stars: list[float] = []
        gains: list[float] = []
        n_items = len(mats[ns[0]][0])
        for _ in range(draws):
            cols = [rng.randrange(n_items) for _ in range(n_items)] if paired else None
            ms = {}
            for n, m in mats.items():
                picked = [m[rng.randrange(len(m))] for _ in m]
                ms[n] = [[s[c] for c in cols] for s in picked] if cols else picked
            bp = points(ms)
            gains.append(bp[-1].y - bp[-2].y)
            fp = _fit(form, bp, base_y)
            if fp is None:
                failed += 1
            else:
                stars.append(_n_star(form, fp, min_gain, ns[0]))
        if stars:
            interval = (_percentile(stars, INTERVAL[0]), _percentile(stars, INTERVAL[1]))
            share = sum(s <= ns[-1] for s in stars) / draws
        gain_iv = (_percentile(gains, INTERVAL[0]), _percentile(gains, INTERVAL[1]))

    mde = _mde(pts[-1], pts[-2])
    if mde is not None and min_gain < mde:
        notes.append(
            f"min_gain {min_gain:.3f} is under what this test can detect ({mde:.3f}): add items or seeds"
        )

    if failed > FAILED_SHARE * draws or (form == "power" and params[2] < MIN_EXPONENT):
        verdict = "not identifiable"
    elif share is not None:
        verdict = "saturated" if share >= SATURATED_SHARE else "not saturated"
    else:
        verdict = "saturated" if n_star <= ns[-1] else "not saturated"
    if verdict == "not saturated" and n_star > FORECAST_RATIO * ns[-1]:
        verdict = "forecast"
        notes.append(
            f"N* is past {FORECAST_RATIO:g}x the largest size trained: a forecast, not a measurement [14]"
        )
    return SaturationFit(
        form=form,
        params=params,
        min_gain=min_gain,
        n_saturated=n_star,
        verdict=verdict,
        curve=curve_y,
        base=base_y,
        interval=interval,
        share_saturated=share,
        heldout_error=held,
        mde=mde,
        last_gain=pts[-1].y - pts[-2].y,
        last_gain_interval=gain_iv,
        notes=tuple(notes),
    )
