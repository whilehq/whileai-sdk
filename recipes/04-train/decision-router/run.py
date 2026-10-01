"""Score two small decision models you train yourself against part 1's routers and TypeSafe's Jev.

Part 1 (recipes/04-train/model-router) trained routers that read a question's
embedding, and asked Jev to route untrained. This part trains two small
models that read the question itself and give each of the twelve models a
probability of answering it correctly (train_modal.py), then puts every
router through the same test: the knob picked on val at a fixed budget, the
held-out questions read once, each pair compared question by question.

Run: python run.py --train   (Modal L40S, six runs, about $5), then python run.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PART1 = HERE.parent / "model-router"
sys.path.insert(0, str(PART1))

import whileai as wai
from whileai.config import provenance


def _part1():
    """Part 1's run.py, loaded by path: both files are called run.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("model_router_part1", PART1 / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p1 = _part1()
KNN, KNOBS, AvengersPro, JevTask = _p1.KNN, _p1.KNOBS, _p1.AvengersPro, _p1.JevTask
as_rows, jev_features, split, load_table = _p1.as_rows, _p1.jev_features, _p1.split, _p1.load

ARMS = ("pointer", "encoder")
SEEDS = (0, 1, 2)
BUDGETS = (5, 10, 20, 35, 50)  # USD per 1,000 questions, as in part 1
ECE_BINS = 10


def delta(before, after):
    m = wai.compare(before, after, target="pass_at_1")["metrics"]["pass_at_1"]
    return {"delta": m["delta"], "ci95": list(m["ci95"]), "n": m["n_paired"]}


def fmt(d):
    return f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


class Trained:
    """A trained decision model as a router: its probability that each model is
    right, weighed against that model's mean training cost by the knob."""

    def fit(self, x, s, c):
        self.cost = c.mean(0)
        self.c_ref = self.cost.max()
        return self

    def predict(self, x):
        return x, np.broadcast_to(self.cost, x.shape)

    route = KNN.route


def calibration(p, y):
    """Brier score and expected calibration error of P(model answers correctly)
    over every (question, model) pair; y may be 0.5 for an ArenaHard tie."""
    p, y = p.ravel(), y.ravel()
    bins = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
    ece = sum(
        abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in np.unique(bins)
    )
    return {"brier": float(((p - y) ** 2).mean()), "ece": float(ece)}


def stand_in(seed=0, n=400, k=5):
    """Seeded rows and predictions for --dry-run: no table, no GPU."""
    rng = np.random.default_rng(seed)
    S = (rng.random((n, k)) < rng.uniform(0.3, 0.8, k)).astype(float)
    C = np.tile(np.array([0.5, 1, 2, 5, 20]) / 1000, (n, 1))
    noisy = np.clip(0.6 * S + 0.2 + 0.1 * rng.normal(size=S.shape), 0.01, 0.99)
    return (
        np.array([f"q{i}" for i in range(n)]),
        S,
        C,
        {"pointer": noisy, "encoder": noisy[:, ::-1] * 0 + noisy.mean(0)},
    )


def build_rows(out1: Path, dest: Path):
    from prepare import MODELS

    rows, _ = load_table(out1)
    queries = {}
    for line in (out1 / "queries.jsonl").open(encoding="utf-8"):
        q = json.loads(line)
        queries[q["id"]] = q["query"]
    fit, val, _ = split(rows)
    tag = np.where(fit, "fit", np.where(val, "val", "test"))
    payload = {
        "models": list(MODELS),
        "rows": [
            {
                "id": r["id"],
                "query": queries[r["id"]],
                "labels": [float(r["score"][m]) for m in MODELS] if t != "test" else None,
                "split": str(t),
            }
            for r, t in zip(rows, tag)
        ],
    }
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def launch(data: Path, out: Path):
    """Every arm and seed in parallel on Modal; each result lands in out/<arm>-s<seed>.json."""
    sys.path.insert(0, str(HERE))
    import train_modal

    payload = json.loads(data.read_text(encoding="utf-8"))
    with train_modal.app.run():
        calls = {
            (a, s): train_modal.train.spawn(a, s, payload["models"], payload["rows"])
            for a in ARMS
            for s in SEEDS
            if not (out / f"{a}-s{s}.json").exists()
        }
        for (a, s), call in calls.items():
            try:
                res = call.get()
            except Exception as e:  # one failed run should not lose the others
                print(f"{a} seed {s} failed: {e}", file=sys.stderr)
                continue
            (out / f"{a}-s{s}.json").write_text(json.dumps(res), encoding="utf-8")
            print(
                f"{a} seed {s}: best epoch {res['best_epoch']}, T={res['temperature']:.2f}",
                file=sys.stderr,
            )


def at_budgets(pv, pt, route, S, C, val, test):
    """The most accurate knob on val at or under each budget, read on held out."""
    curve = []
    for lam in KNOBS:
        pick = route(pv, lam)
        curve.append(
            (
                lam,
                S[val][np.arange(val.sum()), pick].mean(),
                1000 * C[val][np.arange(val.sum()), pick].mean(),
            )
        )
    out = {}
    for b in BUDGETS:
        fits = [t for t in curve if t[2] <= b]
        if fits:
            pick = route(pt, max(fits, key=lambda t: t[1])[0])
            out[b] = (S[test][np.arange(test.sum()), pick], C[test][np.arange(test.sum()), pick])
    full = []
    for lam in KNOBS:
        pick = route(pt, lam)
        full.append(
            {
                "lam": float(lam),
                "accuracy": float(S[test][np.arange(test.sum()), pick].mean()),
                "usd_per_1k": float(1000 * C[test][np.arange(test.sum()), pick].mean()),
            }
        )
    return out, full


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--part1", default=str(PART1 / "out"), help="where part 1's prepare.py and jev.py wrote"
    )
    p.add_argument("--out", default=str(HERE / "out"))
    p.add_argument(
        "--train", action="store_true", help="build the rows and train every arm and seed on Modal"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="seeded stand-in predictions: no table, no GPU"
    )
    args = p.parse_args(argv)
    out, out1 = Path(args.out), Path(args.part1)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        ids, S, C, stand = stand_in()
        n = len(ids)
        val = np.arange(n) % 2 == 0
        test = ~val
        preds = {a: {0: (stand[a][val], stand[a][test])} for a in ARMS}
        refs = {}
        emb = None
    else:
        if args.train:
            build_rows(out1, out / "train_rows.json")
            launch(out / "train_rows.json", out)
        from prepare import DATASETS, MODELS

        rows, emb = load_table(out1)
        models = list(MODELS)
        S = np.array([[r["score"][m] for m in models] for r in rows], dtype=float)
        C = np.array([[r["cost"][m] for m in models] for r in rows], dtype=float)
        ids = np.array([r["id"] for r in rows])
        _, val, test = split(rows)
        preds = {}
        for a in ARMS:
            for s in SEEDS:
                f = out / f"{a}-s{s}.json"
                if f.exists():
                    r = json.loads(f.read_text(encoding="utf-8"))
                    preds.setdefault(a, {})[s] = (np.array(r["val"]), np.array(r["test"]), r)
        refs = {"avengers-pro": AvengersPro, "knn": KNN}
        jev = jev_features(out1, rows, models, [d for d, _ in DATASETS])
        if jev:
            refs["jev-task"] = JevTask
            # A control: the same table read through each question's true dataset, as if
            # Jev always named the kind right. It says how much of Jev's result is the kinds.
            refs["true-kind"] = JevTask

    train = ~(val | test) | val if not args.dry_run else val
    fit_mask = ~(val | test) if not args.dry_run else val
    res = {"budgets": {}, "arms": {}, "curves": {}, "calibration": {}}
    by_budget = {}

    # Reference routers: part 1's, refit exactly as there (fit for the knob, train for held out).
    for name, cls in refs.items():
        if name == "jev-task":
            xf, xp = jev[0]["jev-task"]
        elif name == "true-kind":
            xf = xp = jev[0]["jev-task"][0]
        else:
            xf = xp = emb
        rv = cls().fit(xf[fit_mask], S[fit_mask], C[fit_mask])
        rt = cls().fit(xf[train], S[train], C[train])
        by_budget[name], res["curves"][name] = at_budgets(
            rv.predict(xp[val]), rt.predict(xp[test]), rt.route, S, C, val, test
        )
        if name == "knn":
            res["calibration"]["knn"] = calibration(rt.predict(xp[test])[0], S[test])

    # Trained arms: the seed with the best val log loss is the one reported; the others give the spread.
    for a, seeds in preds.items():
        r = Trained().fit(None, S[train], C[train])
        per_seed = {}
        for s, v in seeds.items():
            pv, pt = v[0], v[1]
            per_seed[s], curve = at_budgets(r.predict(pv), r.predict(pt), r.route, S, C, val, test)
            res["curves"].setdefault(a, {})[s] = curve
            res["calibration"][f"{a}-s{s}"] = calibration(pt, S[test])
            if len(v) > 2:
                meta = v[2]
                res["arms"].setdefault(a, {})[s] = {
                    k: meta[k]
                    for k in (
                        "base",
                        "best_epoch",
                        "temperature",
                        "latency_ms_median",
                        "gpu",
                        "history",
                    )
                }

        def val_loss(s, seeds=seeds):
            pv = np.clip(seeds[s][0], 1e-6, 1 - 1e-6)
            yv = S[val]
            return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)).mean()

        chosen = min(seeds, key=val_loss)
        by_budget[a] = per_seed[chosen]
        res.setdefault("chosen_seed", {})[a] = chosen
        res.setdefault("seed_spread", {})[a] = {
            b: [float(per_seed[s][b][0].mean()) for s in per_seed if b in per_seed[s]]
            for b in BUDGETS
        }

    tid = ids[test]
    for b in BUDGETS:
        row = {}
        for name, d in by_budget.items():
            if b in d:
                sc, co = d[b]
                row[name] = {"accuracy": float(sc.mean()), "usd_per_1k": float(1000 * co.mean())}
        for a in ARMS:
            for ref in ("avengers-pro", "jev-task"):
                if (
                    a in by_budget
                    and ref in by_budget
                    and b in by_budget[a]
                    and b in by_budget[ref]
                ):
                    row[f"{a}_vs_{ref}"] = delta(
                        as_rows(tid, by_budget[ref][b][0]), as_rows(tid, by_budget[a][b][0])
                    )
        res["budgets"][b] = row

    report(res)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out / 'results.json'}")
    return 0


def report(res):
    names = sorted({n for row in res["budgets"].values() for n in row if "_vs_" not in n})
    print("\naccuracy (USD per 1k) at each budget, knob picked on val, read on held out")
    print(f"  {'budget':<8}" + "".join(f"{n:<22}" for n in names))
    for b, row in res["budgets"].items():
        cells = "".join(
            f"{row[n]['accuracy']:.3f} (${row[n]['usd_per_1k']:.2f})".ljust(22)
            if n in row
            else " " * 22
            for n in names
        )
        print(f"  ${b:<7}{cells}")
    print("\npaired, trained arm minus reference")
    for b, row in res["budgets"].items():
        for k, d in row.items():
            if "_vs_" in k:
                print(f"  ${b:<4} {k:<28}{fmt(d)}")
    print("\ncalibration of P(correct) on held out (lower is better)")
    for k, c in res["calibration"].items():
        print(f"  {k:<14} brier {c['brier']:.4f}  ece {c['ece']:.4f}")
    for a, seeds in res.get("arms", {}).items():
        for s, m in seeds.items():
            print(
                f"  {a} seed {s}: {m['base']}, best epoch {m['best_epoch']}, T={m['temperature']:.2f}, "
                f"{m['latency_ms_median']:.1f} ms per question on {m['gpu']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
