"""Paired numbers with intervals, from the graded rows the eval wrote.

    python analyse.py --selftest     # the maths on synthetic rows, offline
    python analyse.py                # the real run, from out/

The noise floor is the spread of the three base runs. Every delta is paired on
the same held-out tasks and reported with a 95% interval; a delta inside the
noise band is flat, however large it looks.

`words` is reported for information only. `wai.compare` treats every marker as
"up is good", and for a voice agent the words are meant to go down, so the
direction-correct marker is `short_enough`.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import whileai as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TARGET = "pass_at_1"  # reward = concise_and_covered
PROXY = "marker:shaped_reward"

CONCISE_WORDS = 120


def to_rows(raw: list[dict]) -> list[dict]:
    """The eval's flat rows as rows `compare` and `pass_at` read."""
    rows = []
    for r in raw:
        rows.append(
            {
                "task_id": r["task_id"],
                "reward": float(r["reward"]),
                "probe": bool(r["probe"]),
                "markers": {
                    "covered_all": float(bool(r["covered_all"])),
                    "short_enough": float(r["words"] <= CONCISE_WORDS and not r["truncated"]),
                    "words": float(r["words"]),
                    "truncated": float(bool(r["truncated"])),
                    "shaped_reward": float(r["shaped_reward"]),
                },
            }
        )
    return rows


def load(name: str) -> list[dict]:
    p = OUT / f"{name}.jsonl"
    if not p.exists():
        raise SystemExit(f"missing {p}; run `modal volume get voice-filter-runs eval {OUT}`")
    return to_rows([json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()])


def mean_reward(rows: list[dict]) -> float:
    return sum(r["reward"] for r in rows) / len(rows)


def summarise(name: str, rows: list[dict]) -> dict:
    pa = wai.pass_at(rows)

    def m(k):
        return sum(r["markers"][k] for r in rows) / len(rows)

    return {
        "arm": name,
        "n_rows": len(rows),
        "pass_at_1": float(pa.pass_at_1),
        "covered_all": m("covered_all"),
        "short_enough": m("short_enough"),
        "words_mean": m("words"),
        "truncated": m("truncated"),
        "shaped_reward": m("shaped_reward"),
    }


MARKERS = ("covered_all", "short_enough", "words", "truncated", "shaped_reward")


def analyse(base_runs: list[list[dict]], arms: dict[str, list[dict]]) -> dict:
    means = [mean_reward(r) for r in base_runs]
    run_std = st.stdev(means) if len(means) > 1 else 0.0

    # One noise floor per metric, each from the same three base re-runs.
    # A single scalar is applied to every metric whatever its scale, which
    # prints "noise<0.142" against a word count; `words` re-runs at 117, 112,
    # 114, so its own floor is ~2.5 words and a 19-word delta clears it.
    per_metric: dict[str, float] = {TARGET: run_std}
    marker_means: dict[str, list[float]] = {}
    for name in MARKERS:
        vals = [sum(r["markers"][name] for r in br) / len(br) for br in base_runs]
        marker_means[name] = [round(v, 4) for v in vals]
        per_metric[f"marker:{name}"] = st.stdev(vals) if len(vals) > 1 else 0.0

    noise = {
        "metric": TARGET,
        "n_runs": len(means),
        "means": [round(m, 4) for m in means],
        "mean": round(sum(means) / len(means), 4),
        "run_std": round(run_std, 4),
        "per_metric_run_std": {k: round(v, 4) for k, v in per_metric.items()},
        "per_metric_base_means": marker_means,
    }
    print(f"noise floor: base x{len(means)} means={noise['means']} run_std={run_std:.4f}")
    for k, v in per_metric.items():
        if k != TARGET:
            print(f"  {k:28s} base re-runs={marker_means[k.split(':', 1)[1]]} run_std={v:.4f}")

    base = base_runs[0]
    res = {"noise": noise, "summaries": {}, "deltas": {}}
    res["summaries"]["base"] = summarise("base", base)
    for i, br in enumerate(base_runs):
        res["summaries"][f"base_run{i + 1}"] = summarise(f"base_run{i + 1}", br)
    for name, rows in arms.items():
        res["summaries"][name] = summarise(name, rows)

    def rep(before, after, label):
        r = wai.compare(
            before,
            after,
            target=TARGET,
            proxy=PROXY,
            must_not_regress=["covered_all"],
            by="probe",
            run_std=per_metric,
            run_std_runs=len(means),
        )
        print(f"\n===== {label} =====")
        print(r)
        return r

    for name, rows in arms.items():
        res["deltas"][f"{name}_vs_base"] = dict(rep(base, rows, f"{name} vs base"))
    if "baseline" in arms and "method" in arms:
        res["deltas"]["method_vs_baseline"] = dict(
            rep(arms["baseline"], arms["method"], "method vs baseline (the one that decides)")
        )
    return res


def selftest() -> None:
    import random

    rng = random.Random(7)

    def mk(p: float, words: int, n_tasks: int = 40) -> list[dict]:
        raw = []
        for t in range(n_tasks):
            for _ in range(4):
                w = max(5, words + rng.randint(-25, 25))
                cov = rng.random() < 0.92
                good = cov and w <= CONCISE_WORDS and rng.random() < p
                raw.append(
                    {
                        "task_id": f"t{t}",
                        "probe": t % 5 == 0,
                        "words": w,
                        "truncated": False,
                        "covered_all": cov,
                        "shaped_reward": (1.0 if cov else 0.0) - 0.3 * min(w / 120, 1),
                        "reward": 1.0 if good else 0.0,
                    }
                )
        return to_rows(raw)

    base_runs = [mk(0.9, 200), mk(0.9, 205), mk(0.9, 196)]
    arms = {"baseline": mk(0.9, 70), "method": mk(0.9, 150)}
    out = analyse(base_runs, arms)
    assert out["noise"]["n_runs"] == 3
    assert "method_vs_baseline" in out["deltas"]
    assert out["summaries"]["baseline"]["short_enough"] > out["summaries"]["method"]["short_enough"]
    print("\nanalyse selftest: ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base-runs", type=int, default=3)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    base_runs = [load(f"base_run{i + 1}") for i in range(args.base_runs)]
    arms = {}
    for a in ("baseline", "method"):
        if (OUT / f"{a}.jsonl").exists():
            arms[a] = load(a)
    res = analyse(base_runs, arms)

    trace = {}
    for a in arms:
        p = OUT / f"{a}_filter_trace.json"
        if p.exists():
            t = json.loads(p.read_text())
            trace[a] = {k: v for k, v in t.items() if k != "trace"}
    res["filter"] = trace
    prep = HERE / "data" / "prep.json"
    if prep.exists():
        res["prep"] = json.loads(prep.read_text())

    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
