"""Train a model router over twelve frontier models, the way the 2026 papers do, and score it honestly.

A router reads a question and picks one model from a pool, trading accuracy
against cost with one knob. Three routers from the literature are trained on
LLMRouterBench's graded answers and scored on held-out questions against the
best single model, a random pick, the per-question oracle, and OpenRouter's
own auto router, each paired and with a 95% interval.

Run: python run.py   (after prepare.py; seconds)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import whileai as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent

HOLDOUT_SHARE = 0.2  # LLMRouterBench's own split is 80/20 (seed 42); ours is by id hash
VAL_SHARE = 0.25  # of train, held back to pick each router's operating point
KNOBS = np.linspace(0.0, 1.0, 41)  # the cost weight each router sweeps
KNN_K = 50  # neighbours; Li 2025 tunes k per benchmark, 50 is mid-range there: untested here
AP_CLUSTERS = 25  # Avengers-Pro: LLMRouterBench's balance_config.json
AP_TOP_K = 3  # nearest clusters a query reads, same config
AP_BETA = 9.0  # softmax temperature over cluster similarity, same config
RIDGE = 1.0  # linear router's L2 penalty: convention, untested
KMEANS_ITERS = 50
SEED = 0
SEED_RUNS = 5
BUDGETS = (
    5,
    10,
    20,
    35,
    50,
)  # USD per 1,000 questions, for the equal-budget comparison  # k-means restarts read for the Avengers-Pro spread


# --- data ---------------------------------------------------------------------


def load(out: Path):
    rows = [json.loads(line) for line in (out / "table.jsonl").open(encoding="utf-8")]
    emb = np.load(out / "embeddings.npy")
    return rows, emb


def stand_in(seed: int = 0, n: int = 600, d: int = 32):
    """Seeded stand-in for --dry-run: 8 topics, 5 models, each strong on different topics."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(8, d))
    topic = rng.integers(0, 8, size=n)
    emb = (centers[topic] + 0.6 * rng.normal(size=(n, d))).astype(np.float32)
    models = [f"model-{i}" for i in range(5)]
    skill = rng.uniform(0.2, 0.9, size=(8, 5))
    price = np.array([0.5, 1.0, 2.0, 5.0, 20.0]) / 1000
    rows = []
    for i in range(n):
        p = skill[topic[i]]
        rows.append(
            {
                "id": f"q{i}",
                "dataset": f"topic-{topic[i]}",
                "score": {m: float(rng.random() < p[j]) for j, m in enumerate(models)},
                "cost": {m: float(price[j] * rng.uniform(0.5, 1.5)) for j, m in enumerate(models)},
            }
        )
    return rows, emb, models


def split(rows):
    h = [int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16) / 16**8 for r in rows]
    test = np.array([x < HOLDOUT_SHARE for x in h])
    val = np.array(
        [HOLDOUT_SHARE <= x < HOLDOUT_SHARE + VAL_SHARE * (1 - HOLDOUT_SHARE) for x in h]
    )
    return ~test & ~val, val, test


# --- routers ------------------------------------------------------------------
# Each is fit on (embeddings, scores, costs) and returns, for new embeddings and
# a cost weight `lam` in [0, 1], the index of the model to call.


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


class KNN:
    """k nearest neighbours in embedding space (Li 2025, arXiv:2505.12601).

    Predicted accuracy and cost of each model = their mean over the k most
    similar training questions; pick the model with the best utility.
    """

    def fit(self, x, s, c):
        self.x, self.s, self.c = unit(x), s, c
        self.c_ref = c.mean(0).max()  # the priciest model's mean cost: lam=1 weighs it as 1 point
        return self

    def predict(self, x):
        sim = unit(x) @ self.x.T
        nn = np.argpartition(-sim, KNN_K, axis=1)[:, :KNN_K]
        return self.s[nn].mean(1), self.c[nn].mean(1)

    def route(self, pred, lam):
        acc, cost = pred
        return np.argmax(acc - lam * cost / self.c_ref, axis=1)


class Linear:
    """One ridge regression per model from embedding to score: the parametric
    baseline (RouterBench, Hu et al. 2024, arXiv:2403.12031), which predicts each
    model's score separately and so can flip rankings on small errors (Lai and
    Ye 2026, arXiv:2602.03478)."""

    def fit(self, x, s, c):
        x1 = np.hstack([x, np.ones((len(x), 1))])
        self.w = np.linalg.solve(x1.T @ x1 + RIDGE * np.eye(x1.shape[1]), x1.T @ s)
        self.cost = c.mean(0)
        self.c_ref = self.cost.max()
        return self

    def predict(self, x):
        x1 = np.hstack([x, np.ones((len(x), 1))])
        return x1 @ self.w, np.broadcast_to(self.cost, (len(x), len(self.cost)))

    route = KNN.route


class AvengersPro:
    """Cluster the questions, score every model per cluster on accuracy and cost,
    and route each new question by its nearest clusters (Zhang et al. 2025,
    arXiv:2508.12631; the leader on LLMRouterBench's cost-quality frontier).

    Per cluster, accuracy is min-max normalised across models and cost becomes
    1 - cost / max cost; the knob `lam` weighs the two, as the reference
    implementation's cost_sensitivity does with performance_weight = 1 - lam.
    """

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit(self, x, s, c):
        x = unit(x)
        rng = np.random.default_rng(self.seed)
        k = min(AP_CLUSTERS, len(x))
        cent = x[rng.choice(len(x), k, replace=False)]
        for _ in range(KMEANS_ITERS):  # spherical k-means
            lab = np.argmax(x @ cent.T, axis=1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
            cent = unit(new)
        self.cent = cent
        acc = np.array([s[lab == j].mean(0) if (lab == j).any() else s.mean(0) for j in range(k)])
        cost = np.array([c[lab == j].mean(0) if (lab == j).any() else c.mean(0) for j in range(k)])
        lo, hi = acc.min(1, keepdims=True), acc.max(1, keepdims=True)
        self.nacc = np.where(hi > lo, (acc - lo) / np.where(hi > lo, hi - lo, 1), 1.0)
        self.cscore = 1 - cost / cost.max(1, keepdims=True).clip(1e-12)
        return self

    def predict(self, x):
        sim = unit(x) @ self.cent.T
        top = np.argsort(-sim, axis=1)[:, :AP_TOP_K]
        w = np.exp(AP_BETA * np.take_along_axis(sim, top, 1))
        w /= w.sum(1, keepdims=True)
        return np.einsum("qt,qtm->qm", w, self.nacc[top]), np.einsum(
            "qt,qtm->qm", w, self.cscore[top]
        )

    def route(self, pred, lam):
        nacc, cscore = pred
        return np.argmax((1 - lam) * nacc + lam * cscore, axis=1)


class JevTask:
    """TypeSafe's Jev names the kind of question (jev.py); the training table
    says how each model does on that kind. Fit reads the true kind of every
    training question, as one-hot rows; predict reads Jev's probabilities, so
    each model's expected accuracy and cost are averaged over the kinds Jev
    thinks the question could be. A task-type router, the way OpenRouter's
    auto router picks [6]."""

    def fit(self, x, s, c):
        w = x / x.sum(0).clip(1)
        self.acc, self.cost = w.T @ s, w.T @ c
        self.c_ref = c.mean(0).max()
        return self

    def predict(self, x):
        return x @ self.acc, x @ self.cost

    route = KNN.route


class JevPick:
    """Jev picks the model directly: it reads the question and each model's
    training accuracy by kind of question (jev.py), and returns a probability
    per model. The knob weighs that probability against the model's mean cost."""

    def fit(self, x, s, c):
        self.cost = c.mean(0)
        self.c_ref = self.cost.max()
        return self

    def predict(self, x):
        return x, np.broadcast_to(self.cost, x.shape)

    route = KNN.route


ROUTERS = {
    "knn": KNN,
    "linear": Linear,
    "avengers-pro": AvengersPro,
    "jev-task": JevTask,
    "jev-pick": JevPick,
}
JEV_ROUTERS = ("jev-task", "jev-pick")


def jev_features(out: Path, rows, models, datasets):
    """(fit features, predict features) for the two Jev routers from out/jev.jsonl,
    or None when jev.py has not run. Rows Jev was not asked about are zeros."""
    path = out / "jev.jsonl"
    if not path.exists():
        return None
    cache = {}
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        cache[rec["id"]] = rec
    onehot = np.array([[float(r["dataset"] == d) for d in datasets] for r in rows])
    task = np.zeros_like(onehot)
    pick = np.zeros((len(rows), len(models)))
    for i, r in enumerate(rows):
        rec = cache.get(r["id"])
        if rec:
            task[i] = [rec["task"].get(d, 0.0) for d in datasets]
            pick[i] = [rec["model"].get(m, 0.0) for m in models]
    stats = {
        "model": sorted({rec["jev"] for rec in cache.values()}),
        "questions": len(cache),
        "median_seconds": float(np.median([rec["seconds"] for rec in cache.values()])),
        "input_tokens": int(sum(rec["input_tokens"] or 0 for rec in cache.values())),
        "task_accuracy": float(
            np.mean(
                [
                    task[i].argmax() == onehot[i].argmax()
                    for i, r in enumerate(rows)
                    if r["id"] in cache
                ]
            )
        ),
    }
    return {"jev-task": (onehot, task), "jev-pick": (onehot, pick)}, stats, set(cache)


# --- measuring ----------------------------------------------------------------


def as_rows(ids, scores, prompts=None):
    """Graded rows for wai.compare. Scores are 0, 0.5 or 1 (ArenaHard ties), so
    each question is written as two rollouts: 1 -> (1, 1), 0.5 -> (1, 0), 0 -> (0, 0)."""
    out = []
    for i, s in zip(ids, scores):
        for j in range(2):
            out.append(
                {
                    "task_id": i,
                    "prompt": i,
                    "final_text": "routed",
                    "markers": {},
                    "reward": float(s >= 0.5 * (j + 1)),
                }
            )
    return out


def random_rows(ids, s):
    """A uniformly random model per question, in expectation: every model's two rollouts."""
    return [r for m in range(s.shape[1]) for r in as_rows(ids, s[:, m])]


def delta(before, after):
    m = wai.compare(before, after, target="pass_at_1")["metrics"]["pass_at_1"]
    return {"delta": m["delta"], "ci95": list(m["ci95"]), "n": m["n_paired"]}


def fmt(d):
    return f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(HERE / "out"), help="where prepare.py wrote the table")
    p.add_argument("--dry-run", action="store_true", help="seeded stand-in table: no download")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rows, emb, models = stand_in()
    else:
        from prepare import DATASETS, MODELS, OPENROUTER

        rows, emb = load(out)
        models = list(MODELS)
    S = np.array([[r["score"][m] for m in models] for r in rows], dtype=float)
    C = np.array([[r["cost"][m] for m in models] for r in rows], dtype=float)
    ids = np.array([r["id"] for r in rows])
    fit, val, test = split(rows)
    print(
        f"{len(rows)} questions x {len(models)} models: {fit.sum()} fit, {val.sum()} val, {test.sum()} held out",
        file=sys.stderr,
    )

    # Baselines, all chosen on train (fit + val), read on the held-out questions.
    train = fit | val
    best = int(np.argmax(S[train].mean(0)))
    oracle = np.argmax(
        S[test] - 1e-6 * C[test] / C.max(), axis=1
    )  # best score, cheapest among ties
    base_rows = as_rows(ids[test], S[test, best])
    rand = random_rows(ids[test], S[test])
    per_1k = lambda c: 1000 * c.mean()  # noqa: E731  USD per thousand questions
    res = {
        "models": models,
        "n": {"fit": int(fit.sum()), "val": int(val.sum()), "test": int(test.sum())},
        "best_single": {
            "model": models[best],
            "accuracy": S[test, best].mean(),
            "usd_per_1k": per_1k(C[test, best]),
        },
        "random": {"accuracy": S[test].mean(), "usd_per_1k": per_1k(C[test])},
        "oracle": {
            "accuracy": S[test, oracle].mean(),
            "usd_per_1k": per_1k(C[test, oracle]),
            "vs_best_single": delta(base_rows, as_rows(ids[test], S[test, oracle])),
        },
        "single_models": {
            m: {"accuracy": S[test, j].mean(), "usd_per_1k": per_1k(C[test, j])}
            for j, m in enumerate(models)
        },
        "routers": {},
    }
    feats = {}
    jev = None if args.dry_run else jev_features(out, rows, models, [d for d, _ in DATASETS])
    if jev:
        feats, res["jev"], asked = jev
        if not all(i in asked for i in ids[val | test]):
            print(
                "out/jev.jsonl does not cover val and held out yet: skipping Jev", file=sys.stderr
            )
            feats = {}

    at_budget: dict = {}
    for name, cls in ROUTERS.items():
        if name in JEV_ROUTERS and name not in feats:
            continue
        xf, xp = feats.get(name, (emb, emb))  # fit reads xf, predict reads xp
        # Pick two operating points on val, with the router fit on `fit` only.
        r = cls().fit(xf[fit], S[fit], C[fit])
        pv = r.predict(xp[val])
        curve = []
        for lam in KNOBS:
            pick = r.route(pv, lam)
            curve.append(
                (
                    lam,
                    S[val][np.arange(val.sum()), pick].mean(),
                    C[val][np.arange(val.sum()), pick].mean(),
                )
            )
        lam_quality = max(curve, key=lambda t: (t[1], -t[2]))[0]
        target = S[val, best].mean()
        cheaper = [t for t in curve if t[1] >= target]
        lam_cost = min(cheaper, key=lambda t: t[2])[0] if cheaper else lam_quality
        points = [("max_quality", lam_quality), ("match_best_cheaper", lam_cost)]
        if not args.dry_run:
            # Third point: the most accurate knob whose val cost, on the questions
            # OpenRouter's auto router answered, stays at or under its cost there.
            vi = np.where(val)[0]
            has_v = np.array([rows[i]["score"].get(OPENROUTER) is not None for i in vi])
            or_cost = np.mean([rows[i]["cost"][OPENROUTER] for i in vi[has_v]])
            under = []
            for lam in KNOBS:
                pick = r.route(pv, lam)
                c_or = C[val][np.arange(val.sum()), pick][has_v].mean()
                if c_or <= or_cost:
                    under.append((S[val][np.arange(val.sum()), pick].mean(), -lam, lam))
            if under:
                points.append(("at_openrouter_cost", max(under)[2]))
        # Refit on all of train, then read the held-out questions once per operating point.
        r = cls().fit(xf[train], S[train], C[train])
        pt = r.predict(xp[test])
        arms = {}
        full_curve = []
        for lam in KNOBS:
            pick = r.route(pt, lam)
            full_curve.append(
                {
                    "lam": float(lam),
                    "accuracy": S[test][np.arange(test.sum()), pick].mean(),
                    "usd_per_1k": per_1k(C[test][np.arange(test.sum()), pick]),
                }
            )
        # Equal budgets: the most accurate knob on val at or under each budget, read on held out.
        for b in BUDGETS:
            fits = [t for t in curve if 1000 * t[2] <= b]
            if fits:
                pick = r.route(pt, max(fits, key=lambda t: t[1])[0])
                at_budget.setdefault(name, {})[b] = (
                    S[test][np.arange(test.sum()), pick],
                    C[test][np.arange(test.sum()), pick],
                )
        for point, lam in points:
            pick = r.route(pt, lam)
            sc = S[test][np.arange(test.sum()), pick]
            co = C[test][np.arange(test.sum()), pick]
            rr = as_rows(ids[test], sc)
            arms[point] = {
                "lam": float(lam),
                "accuracy": sc.mean(),
                "usd_per_1k": per_1k(co),
                "cost_vs_best_single": co.mean() / C[test, best].mean(),
                "vs_best_single": delta(base_rows, rr),
                "vs_random": delta(rand, rr),
                "models_used": {models[j]: int((pick == j).sum()) for j in np.unique(pick)},
            }
            if not args.dry_run:
                # OpenRouter's auto router, on the held-out questions it was run on.
                has = np.array(
                    [rows[i]["score"].get(OPENROUTER) is not None for i in np.where(test)[0]]
                )
                if has.any():
                    o_s = np.array([rows[i]["score"][OPENROUTER] for i in np.where(test)[0][has]])
                    o_c = np.array([rows[i]["cost"][OPENROUTER] for i in np.where(test)[0][has]])
                    arms[point]["vs_openrouter"] = {
                        "n": int(has.sum()),
                        "openrouter_accuracy": o_s.mean(),
                        "openrouter_usd_per_1k": per_1k(o_c),
                        "router_accuracy": sc[has].mean(),
                        "router_usd_per_1k": per_1k(co[has]),
                        "delta": delta(
                            as_rows(ids[test][has], o_s), as_rows(ids[test][has], sc[has])
                        ),
                    }
        res["routers"][name] = {"points": arms, "curve": full_curve}
        if name == "avengers-pro":
            # The one random step in these routers is k-means' start; refit under
            # other seeds at the same knob and read the spread on held-out.
            spread = {}
            for point, lam in points:
                accs, costs = [], []
                for seed in range(SEED_RUNS):
                    rs = AvengersPro(seed=seed).fit(emb[train], S[train], C[train])
                    pick = rs.route(rs.predict(emb[test]), lam)
                    accs.append(S[test][np.arange(test.sum()), pick].mean())
                    costs.append(per_1k(C[test][np.arange(test.sum()), pick]))
                spread[point] = {
                    "accuracy": [min(accs), max(accs)],
                    "usd_per_1k": [min(costs), max(costs)],
                }
            res["routers"][name]["seed_spread"] = spread

    if "jev-task" in at_budget:
        # Jev's task router against Avengers-Pro, paired by question, at each budget both reach.
        res["jev_vs_avengers_pro"] = {}
        for b in BUDGETS:
            if b in at_budget["jev-task"] and b in at_budget["avengers-pro"]:
                (sa, ca), (sj, cj) = at_budget["avengers-pro"][b], at_budget["jev-task"][b]
                res["jev_vs_avengers_pro"][b] = {
                    "avengers_pro": {"accuracy": sa.mean(), "usd_per_1k": per_1k(ca)},
                    "jev_task": {"accuracy": sj.mean(), "usd_per_1k": per_1k(cj)},
                    "delta": delta(as_rows(ids[test], sa), as_rows(ids[test], sj)),
                }
    report(res)
    for point, sp in res["routers"]["avengers-pro"]["seed_spread"].items():
        print(
            f"  avengers-pro {point}: over {SEED_RUNS} k-means seeds acc {sp['accuracy'][0]:.3f}-{sp['accuracy'][1]:.3f}, "
            f"${sp['usd_per_1k'][0]:.2f}-{sp['usd_per_1k'][1]:.2f}/1k"
        )
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out / 'results.json'}")
    return 0


def report(res):
    b, rnd, o = res["best_single"], res["random"], res["oracle"]
    print(f"\nheld out: {res['n']['test']} questions")
    print(f"  best single ({b['model']})  acc {b['accuracy']:.3f}  ${b['usd_per_1k']:.2f}/1k")
    print(f"  random model            acc {rnd['accuracy']:.3f}  ${rnd['usd_per_1k']:.2f}/1k")
    print(
        f"  oracle                  acc {o['accuracy']:.3f}  ${o['usd_per_1k']:.2f}/1k  vs best {fmt(o['vs_best_single'])}"
    )
    print(
        f"\n  {'router':<13}{'point':<20}{'acc':>6}{'$/1k':>8}{'cost':>7}  {'vs best single':<26}{'vs random':<26}vs OpenRouter auto"
    )
    for name, r in res["routers"].items():
        for point, a in r["points"].items():
            o = a.get("vs_openrouter")
            otxt = (
                f"{fmt(o['delta'])} (n={o['n']}; OR ${o['openrouter_usd_per_1k']:.2f} vs ${o['router_usd_per_1k']:.2f}/1k)"
                if o
                else ""
            )
            print(
                f"  {name:<13}{point:<20}{a['accuracy']:>6.3f}{a['usd_per_1k']:>8.2f}{a['cost_vs_best_single']:>6.2f}x  "
                f"{fmt(a['vs_best_single']):<26}{fmt(a['vs_random']):<26}{otxt}"
            )
    if res.get("jev"):
        j = res["jev"]
        print(
            f"\n  Jev ({', '.join(j['model'])}): {j['questions']} questions, median {j['median_seconds']:.2f} s, "
            f"{j['input_tokens']:,} input tokens, names the kind of question right {j['task_accuracy']:.1%}"
        )
    for b, h in res.get("jev_vs_avengers_pro", {}).items():
        print(
            f"  at ${b}/1k  avengers-pro {h['avengers_pro']['accuracy']:.3f} (${h['avengers_pro']['usd_per_1k']:.2f})  "
            f"jev-task {h['jev_task']['accuracy']:.3f} (${h['jev_task']['usd_per_1k']:.2f})  jev - ap {fmt(h['delta'])}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
