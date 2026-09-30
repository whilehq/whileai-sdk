"""Classifier metrics with bootstrap intervals over rows, per slice. Pure Python.

``wai.compare`` and ``pass_at`` are per-prompt pass rates; a classifier report
(AUROC, recall at 1% FPR, precision/recall/F1 at a threshold, FPR on benign)
with a row bootstrap does not exist in the SDK yet, so it lives here. The
bootstrap is Efron's percentile interval (1979), 1000 resamples, seed 0. No
numpy, so the offline path (``smoke.sh``) runs in the SDK's own environment.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

NAN = float("nan")


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    """Mann-Whitney AUROC: P(score of a positive > score of a negative), ties count half."""
    n_pos = sum(1 for v in y if v == 1)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return NAN
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, v in zip(ranks, y) if v == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _fpr_threshold(neg_scores: Sequence[float], fpr: float) -> float:
    """The score strictly above which at most ``fpr`` of the negatives lie."""
    neg = sorted(neg_scores)
    k = math.floor(fpr * len(neg))  # allowed false positives
    return neg[len(neg) - k - 1] if k < len(neg) else -math.inf


def recall_at_fpr(y: Sequence[int], s: Sequence[float], fpr: float = 0.01) -> float:
    """Recall at the threshold whose FPR on the negatives is at most ``fpr``."""
    neg = [v for v, lab in zip(s, y) if lab == 0]
    pos = [v for v, lab in zip(s, y) if lab == 1]
    if not neg or not pos:
        return NAN
    thr = _fpr_threshold(neg, fpr)
    return sum(1 for v in pos if v > thr) / len(pos)


def at_threshold(y: Sequence[int], s: Sequence[float], thr: float) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for lab, v in zip(y, s):
        pred = v > thr
        if pred and lab == 1:
            tp += 1
        elif pred:
            fp += 1
        elif lab == 1:
            fn += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else NAN
    r = tp / (tp + fn) if tp + fn else NAN
    f1 = 2 * p * r / (p + r) if (p + r) and not (math.isnan(p) or math.isnan(r)) else NAN
    fpr = fp / (fp + tn) if fp + tn else NAN
    return {"precision": p, "recall": r, "f1": f1, "fpr": fpr, "accuracy": (tp + tn) / len(y)}


def _boot(
    fn: Callable[[list[int], list[float]], float],
    y: Sequence[int],
    s: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> list[float]:
    rng = random.Random(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([y[i] for i in idx], [s[i] for i in idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return [NAN, NAN]
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[math.ceil(0.975 * (len(vals) - 1))]
    return [lo, hi]


def report(y: Sequence[int], s: Sequence[float], thr: float, *, n_boot: int = 1000) -> dict:
    """Point estimates with 95% bootstrap intervals for one slice."""
    y = [int(v) for v in y]
    s = [float(v) for v in s]
    n_pos = sum(y)
    out: dict = {"n": len(y), "n_pos": n_pos, "n_neg": len(y) - n_pos, "threshold": thr}
    if n_pos and len(y) - n_pos:
        out["auroc"] = auroc(y, s)
        out["auroc_ci95"] = _boot(auroc, y, s, n_boot)
        out["recall_at_1pct_fpr"] = recall_at_fpr(y, s)
        out["recall_at_1pct_fpr_ci95"] = _boot(recall_at_fpr, y, s, n_boot)
    at = at_threshold(y, s, thr)
    out.update(at)
    for k in ("recall", "fpr"):
        if not math.isnan(at[k]):
            out[f"{k}_ci95"] = _boot(lambda a, b, k=k: at_threshold(a, b, thr)[k], y, s, n_boot)
    return out


def choose_threshold(y: Sequence[int], s: Sequence[float], fpr: float = 0.01) -> float:
    """The score above which ``fpr`` of the benign side of ``(y, s)`` would be flagged."""
    neg = [v for v, lab in zip(s, y) if lab == 0]
    return _fpr_threshold(neg, fpr) if neg else 0.5


def by_slice(
    rows: list[dict], scores: Sequence[float], thr: float, *, n_boot: int = 1000
) -> dict[str, dict]:
    y = [int(r["label"]) for r in rows]
    s = [float(v) for v in scores]
    sl = [r["slice"] for r in rows]
    out = {"all": report(y, s, thr, n_boot=n_boot)}
    for name in sorted(set(sl)):
        idx = [i for i, v in enumerate(sl) if v == name]
        out[name] = report([y[i] for i in idx], [s[i] for i in idx], thr, n_boot=n_boot)
    # the indirect held-out slices together: the number the claim is about
    held = {"indirect_heldout_family", "indirect_heldout_carrier", "sim_tool"}
    idx = [i for i, v in enumerate(sl) if v in held]
    if idx:
        out["indirect_heldout"] = report(
            [y[i] for i in idx], [s[i] for i in idx], thr, n_boot=n_boot
        )
    return out


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _cell(r: dict, key: str) -> str:
    if key not in r:
        return "n/a"
    ci = r.get(f"{key}_ci95", [None, None])
    return f"{fmt(r[key])} [{fmt(ci[0])}, {fmt(ci[1])}]"


def table(per_slice: dict[str, dict], model: str) -> str:
    lines = [
        f"| slice ({model}) | n | AUROC [95%] | recall@1%FPR [95%] | recall@thr [95%] | FPR@thr [95%] |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in per_slice.items():
        lines.append(
            f"| {name} | {r['n']} | {_cell(r, 'auroc')} | {_cell(r, 'recall_at_1pct_fpr')} "
            f"| {_cell(r, 'recall')} | {_cell(r, 'fpr')} |"
        )
    return "\n".join(lines)
