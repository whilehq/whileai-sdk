"""Does decontaminate() actually protect a held-out set?

Ground truth is human-labelled, not written by me:
  QQP  label=1 -> a genuine paraphrase leak, across a wide range of lexical overlap
  PAWS label=0 -> high word overlap, DIFFERENT meaning: a row that must NOT be dropped
  PAWS label=1 -> high word overlap, same meaning: a leak that should be dropped

A contaminated train row carries a task id that is NOT the holdout's, which is the
case whenever rows arrive from another team, a vendor or the Hub: the same_task rule
has nothing to match on and the text rules are all there is.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import whileai as wai
from whileai.config import provenance

WORD = re.compile(r"[a-z0-9]+")


def toks(s: str) -> list[str]:
    return WORD.findall(s.lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(toks(a)), set(toks(b))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def _synthetic_pairs(n_pairs: int, rng: random.Random):
    """Stand-in pairs for --dry-run, so the code path runs with no download.

    These are templates, not human-labelled data: they exercise the harness
    and the controls, and their recall numbers mean nothing. Use the real
    run for any number you intend to quote.
    """
    subj = ["order", "refund", "invoice", "shipment", "account", "booking"]
    verb = ["cancel", "track", "update", "split", "escalate", "reissue"]
    dup, non = [], []
    for _ in range(n_pairs):
        s, v = rng.choice(subj), rng.choice(verb)
        n = rng.randint(1000, 9999)
        dup.append(
            (
                f"How do I {v} the {s} numbered {n} for this customer?",
                f"What is the way to {v} {s} {n} on behalf of a customer?",
            )
        )
        non.append(
            (
                f"How do I {v} the {s} numbered {n} for this customer?",
                f"How do I {rng.choice(verb)} the {rng.choice(subj)} numbered "
                f"{rng.randint(1000, 9999)} for this customer?",
            )
        )
    return dup, non


def build(n_pairs: int, seed: int = 0, dry_run: bool = False):
    rng = random.Random(seed)

    if dry_run:
        n_ctl = max(40, n_pairs // 4)
        dup, non = _synthetic_pairs(n_pairs + 3 * n_ctl, rng)
        reserve = [q for q, _ in non[n_pairs : n_pairs + 3 * n_ctl]]
        return _assemble(
            dup[:n_pairs],
            non[:n_pairs],
            dup[n_pairs : n_pairs + n_pairs // 2],
            non[n_pairs : n_pairs + n_pairs // 2],
            reserve,
            n_ctl,
        )

    from datasets import load_dataset

    qqp = load_dataset("nyu-mll/glue", "qqp", split="train[:60000]")
    paws = load_dataset("google-research-datasets/paws", "labeled_final", split="train[:30000]")

    def clean(s):
        return " ".join((s or "").split())

    qdup = [
        (clean(r["question1"]), clean(r["question2"]))
        for r in qqp
        if r["label"] == 1 and len(toks(r["question1"])) >= 6
    ]
    qnon = [
        (clean(r["question1"]), clean(r["question2"]))
        for r in qqp
        if r["label"] == 0 and len(toks(r["question1"])) >= 6
    ]
    pdup = [(clean(r["sentence1"]), clean(r["sentence2"])) for r in paws if r["label"] == 1]
    pnon = [(clean(r["sentence1"]), clean(r["sentence2"])) for r in paws if r["label"] == 0]

    rng.shuffle(qdup)
    rng.shuffle(qnon)
    rng.shuffle(pdup)
    rng.shuffle(pnon)
    n_ctl = max(40, n_pairs // 4)
    # reserve control text from a slice that never enters the main holdout pool
    reserve = [q for q, _ in qnon[n_pairs : n_pairs + 3 * n_ctl]]
    return _assemble(
        qdup[:n_pairs],
        qnon[:n_pairs],
        pdup[: n_pairs // 2],
        pnon[: n_pairs // 2],
        reserve,
        n_ctl,
    )


def _assemble(qdup, qnon, pdup, pnon, reserve, n_ctl):
    holdout, train = [], []
    tid = 0

    def add(h_text, t_text, kind, leak):
        nonlocal tid
        holdout.append({"prompt": h_text, "task_id": f"hold-{tid}", "answer": ""})
        train.append(
            {
                "prompt": t_text,
                "task_id": f"train-{tid}",  # deliberately a different namespace
                "kind": kind,
                "leak": leak,
                "jac": jaccard(h_text, t_text),
                "pair_of": f"hold-{tid}",
            }
        )
        tid += 1

    # true leaks: the holdout question, re-asked in other words
    for a, b in qdup:
        add(a, b, "qqp_dup", True)
    for a, b in pdup:
        add(a, b, "paws_dup", True)
    # true negatives: a DIFFERENT question that happens to share words
    for a, b in qnon:
        add(a, b, "qqp_nondup", False)
    for a, b in pnon:
        add(a, b, "paws_nondup", False)

    # --- controls, so a reviewer can tell a real miss from a broken harness.
    # Control text comes from `reserve`, which is disjoint from every pair
    # above, so the only way a control row matches the holdout is the one
    # the control is testing.
    a_ctl = reserve[:n_ctl]
    b_ctl = reserve[n_ctl : 2 * n_ctl]
    c_ctl = reserve[2 * n_ctl :]

    # positive control A: a byte-identical copy MUST be caught (recall 1.0)
    for q in a_ctl:
        add(q, q, "ctl_identical", True)
    # positive control B: case + whitespace only; the docstring says exact
    # matches after normalization, so this must also be caught
    for q in b_ctl:
        add(q, "  " + q.upper() + " ", "ctl_case", True)
    # negative control: holdout and train are unrelated questions -> FP ~ 0.
    # The two halves are disjoint, so no control train text is ever a holdout
    # text; anything dropped here is the rule firing on nothing.
    half = len(c_ctl) // 2
    for q, u in zip(c_ctl[:half], c_ctl[half:]):
        add(q, u, "ctl_unrelated", False)

    return holdout, train


def dropped_ids(train, kept):
    kept_ids = {r["task_id"] for r in kept}
    return {r["task_id"] for r in train if r["task_id"] not in kept_ids}


def rate_with_interval(flags: list[bool]):
    """Proportion + 95% interval from the SDK's own pass_at, cross-checked
    against an independent bootstrap. 'caught' is the reward."""
    if not flags:
        return {"rate": None, "lo": None, "hi": None, "n": 0, "via": "empty"}
    rows = [
        {"task_key": f"t{i}", "task_id": f"t{i}", "reward": 1.0 if f else 0.0}
        for i, f in enumerate(flags)
    ]
    p = wai.pass_at(rows, k=1)
    b = _boot(flags)
    # pass_at().ci95 is None on small or degenerate samples (e.g. every row
    # the same), so the bootstrap is both the cross-check and the fallback.
    if p.ci95 is None or p.pass_at_1 is None:
        return {**b, "via": "bootstrap (pass_at ci95 was None)"}
    return {
        "rate": p.pass_at_1,
        "lo": p.ci95[0],
        "hi": p.ci95[1],
        "n": len(flags),
        "via": "wai.pass_at",
        "boot_lo": b["lo"],
        "boot_hi": b["hi"],
    }


def _boot(flags, note=""):
    """Independent cross-check of the SDK's ci95, from the standard library.

    Resampling a Bernoulli sample is a Binomial draw, so the bootstrap
    distribution of the proportion is exactly Binomial(n, p_hat) / n. That
    is computed here rather than sampled: no numpy, no resampling noise,
    and the same answer every run.
    """
    n = len(flags)
    if n == 0:
        return {"rate": None, "lo": None, "hi": None, "n": 0, "via": "boot", "note": note}
    point = sum(flags) / n
    if point in (0.0, 1.0):  # degenerate: the draw is a point mass
        return {"rate": point, "lo": point, "hi": point, "n": n, "via": "boot", "note": note}

    # log pmf in closed form, so large n does not underflow
    log_p, log_q = math.log(point), math.log1p(-point)
    base = math.lgamma(n + 1)

    def log_pmf(k):
        return base - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * log_p + (n - k) * log_q

    lo = hi = None
    cdf = 0.0
    for k in range(n + 1):
        cdf += math.exp(log_pmf(k))
        if lo is None and cdf >= 0.025:
            lo = k / n
        if cdf >= 0.975:
            hi = k / n
            break
    return {
        "rate": point,
        "lo": lo if lo is not None else 0.0,
        "hi": hi if hi is not None else 1.0,
        "n": n,
        "via": "boot",
        "note": note,
    }


BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]


MAIN_KINDS = ("qqp_dup", "paws_dup", "qqp_nondup", "paws_nondup")


def summarize(train, dropped, label):
    out = {"arm": label, "overall": {}, "by_bucket": {}, "by_kind": {}, "controls": {}}
    # headline numbers use the human-labelled pairs only; controls are
    # harness checks and would flatter the recall if mixed in
    main = [r for r in train if r["kind"] in MAIN_KINDS]
    leaks = [r for r in main if r["leak"]]
    cleans = [r for r in main if not r["leak"]]
    out["overall"]["recall"] = rate_with_interval([r["task_id"] in dropped for r in leaks])
    out["overall"]["false_positive"] = rate_with_interval([r["task_id"] in dropped for r in cleans])
    for lo, hi in BUCKETS:
        sel = [r for r in leaks if lo <= r["jac"] < hi]
        selc = [r for r in cleans if lo <= r["jac"] < hi]
        key = f"{lo:.2f}-{hi:.2f}"
        out["by_bucket"][key] = {
            "recall": rate_with_interval([r["task_id"] in dropped for r in sel]),
            "false_positive": rate_with_interval([r["task_id"] in dropped for r in selc]),
        }
    by = defaultdict(list)
    for r in train:
        by[r["kind"]].append(r["task_id"] in dropped)
    for k, v in sorted(by.items()):
        tgt = out["controls"] if k.startswith("ctl_") else out["by_kind"]
        tgt[k] = rate_with_interval(v)
    return out


def run_all(
    pairs: int,
    seed: int,
    semantic: bool,
    verbose: bool = True,
    dry_run: bool = False,
) -> dict:
    holdout, train = build(pairs, seed, dry_run=dry_run)
    if verbose:
        print(
            f"holdout={len(holdout)} train={len(train)} "
            f"leaks={sum(r['leak'] for r in train)} "
            f"clean={sum(not r['leak'] for r in train)}"
        )

    arms = {}

    def run(label, **kw):
        kept, rep = wai.decontaminate(train, against=[holdout], **kw)
        d = dropped_ids(train, kept)
        arms[label] = summarize(train, d, label)
        arms[label]["report"] = {k: v for k, v in rep.items() if isinstance(v, (int, float, str))}
        r = arms[label]["overall"]
        c = arms[label]["controls"]
        if not verbose:
            return
        print(
            f"{label:28s} recall={r['recall']['rate']:.3f} "
            f"[{r['recall']['lo']:.3f},{r['recall']['hi']:.3f}]  "
            f"FP={r['false_positive']['rate']:.3f} "
            f"[{r['false_positive']['lo']:.3f},{r['false_positive']['hi']:.3f}]"
            f"   ctl id/case/unrel="
            f"{c['ctl_identical']['rate']:.2f}/{c['ctl_case']['rate']:.2f}/"
            f"{c['ctl_unrelated']['rate']:.2f}"
        )

    run("default (n=8, overlap=0.8)")
    run("overlap=0 (any 8-gram)", overlap=0.0)
    run("n=5, overlap=0.8", n=5)
    run("n=5, overlap=0.0", n=5, overlap=0.0)
    run("n=3, overlap=0.0", n=3, overlap=0.0)

    if semantic:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer("BAAI/bge-small-en-v1.5")
        cache: dict[str, list[float]] = {}

        def emb(texts):
            texts = list(texts)
            missing = [t for t in texts if t not in cache]
            if missing:
                vecs = m.encode(
                    missing,
                    normalize_embeddings=True,
                    batch_size=256,
                    show_progress_bar=False,
                ).tolist()
                cache.update(zip(missing, vecs))
            return [cache[t] for t in texts]

        # the thresholds the README's sweep reports, 0.85 being the SDK default
        for thr in (0.95, 0.90, 0.85, 0.80, 0.75):
            run(f"semantic bge@{thr:.2f}", embedder=emb, similarity=thr)

    return {
        "n_pairs": pairs,
        "seed": seed,
        "whileai": wai.__version__,
        "n_holdout": len(holdout),
        "n_train": len(train),
        "arms": arms,
    }


def main():
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="template pairs instead of the labelled sets: no download, no key. "
        "Exercises the harness; its recall numbers are not results.",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap --pairs (smoke runs)")
    ap.add_argument(
        "--out",
        default=None,
        help="where to write the arm-by-arm numbers (default out/results.seed<seed>.json; "
        "out/ is gitignored, so a run never overwrites the published results.json)",
    )
    a = ap.parse_args()
    pairs = min(a.pairs, a.limit) if a.limit else a.pairs
    res = run_all(pairs, a.seed, a.semantic, dry_run=a.dry_run)
    out = Path(a.out) if a.out else Path("out") / f"results.seed{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
