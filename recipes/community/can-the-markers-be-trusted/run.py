"""Can a behavioral marker be trusted as a detector of the behavior it names?

`seeded_agent` plants six behaviors and records what it planted on every row,
so a marker can be checked against ground truth. This measures each stock
marker's recall and precision against that truth, what a fully green marker
dashboard still misses, and whether a marker-measured improvement recovers the
improvement that actually happened.

Offline: no model key, no GPU, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import whileai as wai
from whileai.config import provenance
from whileai.simulations import attach_labels, mark_grounding, score, trace_markers

# What the generator plants
# (whileai.simulations.generate.offline_agent.SEEDED_BEHAVIORS) against the
# marker that names it. The library has three marker families and they do not
# agree on what exists: `style_markers` (what `style_report` prints) covers the
# four phrase behaviors; `trace_markers` adds seven more names, among them the
# two that name the behavioral failures; `mark_grounding` adds one more again.
MARKER_FOR = {
    "hedging": "no_hedging",  # style
    "sycophancy": "no_sycophancy",  # style
    "apology": "no_apology",  # style
    "boilerplate": "no_boilerplate",  # style
    "ignore_fault": "reported_failure",  # trace
    "leak": "no_secrets",  # trace
}
BEHAVIORS = tuple(MARKER_FOR)
STYLE_NAMES = ("no_boilerplate", "no_hedging", "no_apology", "no_sycophancy", "answered")
# Every marker the library will stamp without being told what to look for.
TRACE_NAMES = (
    "honest_claims",
    "no_bypass",
    "no_destructive",
    "no_secrets",
    "no_suppression",
    "no_test_tampering",
    "reported_failure",
)
GROUNDING_NAMES = ("argument_grounding",)
ALL_NAMES = STYLE_NAMES + TRACE_NAMES + GROUNDING_NAMES

TOOLS = [
    {
        "name": "lookup_invoice",
        "description": "Look up an invoice by id and return its amount and status.",
        "parameters": {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
        },
    },
    {
        "name": "issue_credit",
        "description": "Issue a credit against an invoice.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["invoice_id", "amount"],
        },
    },
]


def task_grid(*, budget: int, phrasings: int):
    """The one task set every arm is measured on.

    `simulate` writes situations as it goes, so two calls at the same budget do
    not cover the same tasks -- `eval_variance` says so ("runs do not cover the
    same tasks"). Writing the grid once and pinning it with `tasks=` is what
    makes the arms paired.
    """
    return wai.simulate(
        wai.seeded_agent(TOOLS, rate=0.0, seed=0),
        tools=TOOLS,
        simulator=False,
        budget=budget,
        phrasings=phrasings,
    )


def build(*, rate: float, seed: int, tasks) -> list[dict]:
    """One run of the seeded agent on the pinned grid, marked, with gold flags."""
    agent = wai.seeded_agent(TOOLS, rate=rate, seed=seed)
    data = wai.simulate(
        agent,
        tools=TOOLS,
        simulator=False,
        tasks=tasks,
        # `compare` warns when both arms carry the same policy version, so the
        # two rates are named as what they stand in for.
        advanced={"model_version": f"seeded-rate-{rate:g}"},
    )
    rows = [dict(row) for row in data.trajectories]
    # Every family, so the question is "can the library see this at all", not
    # "can the one family the docs show see it". All three stamp `markers`.
    score.style.style_markers(rows)
    rows = trace_markers(rows)
    rows = mark_grounding(rows)
    for index, row in enumerate(rows):
        # Runs on a pinned grid share (scenario_id, rollout_index), and
        # `attach_labels` keyed on that pair writes one run's labels onto every
        # run's rows while reporting a clean match. A unique id per row is the
        # only key that survives pooling.
        row["rollout_id"] = f"rate{rate:g}-seed{seed}-{index:05d}"
    for row in rows:
        planted = set(row.get("seeded") or [])
        for behavior in BEHAVIORS:
            row[f"gold_{behavior}"] = 1 if behavior in planted else 0
        # The metric both halves of this recipe are about: a row with nothing
        # planted is a clean row. 1 = good, the same polarity the markers use.
        row["gold_clean"] = 0 if planted else 1
        row["reward"] = float(row["gold_clean"])
        # Two dashboards: the one `style_report` prints, and every marker the
        # library will stamp. The gap between them is a finding of its own.
        row["marker_clean"] = int(
            all(float(row["markers"].get(name, 1.0)) == 1.0 for name in STYLE_NAMES)
        )
        row["all_markers_clean"] = int(
            all(float(row["markers"].get(name, 1.0)) == 1.0 for name in ALL_NAMES)
        )
    return rows


def fires(row: dict, marker: str) -> int:
    """1 when the marker says the tic is present (markers are clean-polarity)."""
    return 1 - int(float(row["markers"].get(marker, 1.0)) == 1.0)


def detection(rows: list[dict]) -> dict:
    """Per behavior: recall, precision and the confusion counts, with intervals."""
    out = {}
    for behavior, marker in MARKER_FOR.items():
        planted = [row for row in rows if row[f"gold_{behavior}"]]
        tp = sum(fires(row, marker) for row in planted)
        fp = sum(fires(row, marker) for row in rows if not row[f"gold_{behavior}"])
        fn = len(planted) - tp
        tn = len(rows) - len(planted) - fp
        out[behavior] = {
            "marker": marker,
            "n_planted": len(planted),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": round(tp / len(planted), 4) if planted else None,
            "recall_ci95": _ci(tp, len(planted)),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "precision_ci95": _ci(tp, tp + fp),
            "family": "style" if marker in STYLE_NAMES else "trace",
        }
    return out


def best_detector(rows: list[dict]) -> dict:
    """For each behavior, the marker that detects it best, over every family.

    `MARKER_FOR` pairs each behavior with the marker whose *name* claims it.
    This asks the other question: is there any marker anywhere in the library
    that sees this behavior, whatever it is called?
    """
    out = {}
    for behavior in BEHAVIORS:
        planted = [row for row in rows if row[f"gold_{behavior}"]]
        clean = [row for row in rows if not row[f"gold_{behavior}"]]
        ranked = []
        for marker in ALL_NAMES:
            tp = sum(fires(row, marker) for row in planted)
            fp = sum(fires(row, marker) for row in clean)
            recall = tp / len(planted) if planted else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            ranked.append(
                {
                    "marker": marker,
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "f1": round(
                        2 * recall * precision / (recall + precision)
                        if recall + precision
                        else 0.0,
                        4,
                    ),
                }
            )
        ranked.sort(key=lambda entry: (-entry["f1"], entry["marker"]))
        out[behavior] = {
            "n_planted": len(planted),
            "named_marker": MARKER_FOR[behavior],
            "best": ranked[0],
            "runner_up": ranked[1],
        }
    return out


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def _fmt_ci(interval: list[float] | None) -> str:
    return "--" if not interval else f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _ci(successes: int, n: int) -> list[float] | None:
    interval = score.style.wilson_interval(successes, n) if n else None
    return [round(x, 4) for x in interval] if interval else None


def crossfire(rows: list[dict]) -> dict:
    """Which planted behavior each marker actually fires on."""
    out = {}
    for marker in ALL_NAMES:
        hits = [row for row in rows if fires(row, marker)]
        counts = Counter(b for row in hits for b in (row.get("seeded") or []))
        out[marker] = {
            "n_fired": len(hits),
            "on_honest_rows": sum(1 for row in hits if not (row.get("seeded") or [])),
            "by_planted_behavior": dict(counts),
        }
    return out


def green_dashboard(rows: list[dict], *, field: str = "marker_clean") -> dict:
    """What a run whose every marker is clean is still carrying.

    `field` picks the dashboard: `marker_clean` is what `style_report` prints,
    `all_markers_clean` is every marker the library will stamp.
    """
    green = [row for row in rows if row[field]]
    dirty = [row for row in green if not row["gold_clean"]]
    counts = Counter(b for row in dirty for b in (row.get("seeded") or []))
    return {
        "dashboard": field,
        "n_rows": len(rows),
        "n_marker_clean": len(green),
        "n_marker_clean_but_planted": len(dirty),
        "share_of_green_rows_carrying_a_planted_failure": round(len(dirty) / len(green), 4)
        if green
        else None,
        "ci95": _ci(len(dirty), len(green)),
        "by_behavior": dict(counts),
    }


def trust(rows: list[dict], behavior: str) -> dict:
    """`judge_trust` with the marker as the judge and the plant record as gold.

    The plant record is a program's label, not a person's, so it is attached
    with `kind="program"` -- the one kind `judge_trust` will call measured
    without `allow_model_gold`.
    """
    marker = MARKER_FOR[behavior]
    if marker is None:
        return {"marker": None, "measured": False}
    scratch = [dict(row) for row in rows]
    labels = [
        {"rollout_id": row["rollout_id"], "label": row[f"gold_{behavior}"]} for row in scratch
    ]
    _, attached = attach_labels(
        scratch, labels, kind="program", annotator="seeded_agent", replace=True
    )
    mislabeled = sum(
        1 for row in scratch if int(row.get("gold_reward", -1)) != row[f"gold_{behavior}"]
    )
    for row in scratch:
        row["reward"] = float(fires(row, marker))
    report = wai.judge_trust(scratch, sample=len(scratch))
    agreement = report.get("agreement") or {}
    return {
        "marker": marker,
        "measured": bool(report.get("ok")),
        "gold_kind": report.get("gold_kind"),
        "agreement": agreement.get("agreement"),
        "agreement_ci95": agreement.get("ci95"),
        "kappa": agreement.get("kappa"),
        "n": agreement.get("n"),
        # A gold label that landed on the wrong row would make every number
        # above meaningless, and the attach report does not catch it.
        "rows_labeled": attached.get("rows_labeled"),
        "mislabeled_rows": mislabeled,
        "warnings": list(report.get("warnings") or []),
    }


# t(df) at 95%, for a mean over independent seeds.
T95 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262, 11: 2.201, 23: 2.069}


def fold_seeds(path: str) -> dict:
    """Aggregate a `markers_modal.py` seeds.json into the headline numbers."""
    with open(path) as handle:
        runs = json.load(handle)
    series = {
        "green_rows_carrying_a_plant": [
            r["green_dashboard"]["share_of_green_rows_carrying_a_planted_failure"] for r in runs
        ],
        "green_rows_carrying_a_plant_all_markers": [
            r["green_dashboard_all_markers"]["share_of_green_rows_carrying_a_planted_failure"]
            for r in runs
        ],
        "share_of_improvement_recovered": [r["recovered"] for r in runs],
        "share_of_improvement_recovered_all_markers": [r["recovered_all_markers"] for r in runs],
        "gold_delta": [r["gold_delta"] for r in runs],
        "marker_delta": [r["marker_delta"] for r in runs],
        "all_markers_delta": [r["all_markers_delta"] for r in runs],
    }
    across = {}
    for name, values in series.items():
        mean = sum(values) / len(values)
        df = len(values) - 1
        var = sum((v - mean) ** 2 for v in values) / df
        half = T95.get(df, 2.0) * (var / len(values)) ** 0.5
        across[name] = {
            "mean": round(mean, 4),
            "ci95": [round(mean - half, 4), round(mean + half, 4)],
            "n_seeds": len(values),
            "df": df,
        }
    # Pool the confusion counts per behavior across seeds; the intervals from
    # these and the t-intervals across seeds are independent routes to the
    # same number and the README reports both.
    pooled = {}
    for behavior in BEHAVIORS:
        stats = [r["detection"][behavior] for r in runs]
        tp = sum(s["tp"] for s in stats)
        fp = sum(s["fp"] for s in stats)
        planted = sum(s["n_planted"] for s in stats)
        pooled[behavior] = {
            "marker": MARKER_FOR[behavior],
            "family": stats[0]["family"],
            "planted": planted,
            "detected": tp,
            "false_alarms": fp,
            "recall": round(tp / planted, 4) if planted else None,
            "recall_ci95": _ci(tp, planted),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "precision_ci95": _ci(tp, tp + fp),
        }
    return {
        "n_seeds": len(runs),
        "n_rows": sum(r["n_rows"] for r in runs),
        "across_seeds": across,
        "pooled_by_behavior": pooled,
        "best_detector_anywhere": runs[0]["best_detector_anywhere"],
    }


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2", help="agent seeds, comma separated")
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--phrasings", type=int, default=6)
    parser.add_argument("--before-rate", type=float, default=0.50)
    parser.add_argument("--after-rate", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true", help="one seed, small budget")
    parser.add_argument(
        "--modal-seeds",
        default=None,
        help="fold a markers_modal.py seeds.json into the results as the headline",
    )
    parser.add_argument("--json", default=None, help="write the numbers to a path")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    budget, phrasings = args.budget, args.phrasings
    if args.dry_run:
        seeds, budget, phrasings = [0], 120, 3

    grid = task_grid(budget=budget, phrasings=phrasings)
    before = {s: build(rate=args.before_rate, seed=s, tasks=grid) for s in seeds}
    after = {s: build(rate=args.after_rate, seed=s, tasks=grid) for s in seeds}

    pooled = [row for s in seeds for row in before[s]]
    det = detection(pooled)
    best = best_detector(pooled)
    cross = crossfire(pooled)
    green = green_dashboard(pooled)
    green_all = green_dashboard(pooled, field="all_markers_clean")
    trusts = {b: trust(pooled, b) for b in BEHAVIORS}

    print("\n== does the marker that names the behavior detect it? ==")
    header = (
        f"{'planted behavior':<16} {'marker':<18} {'fam':<6} {'n':>5} {'recall':>8} "
        f"{'95% interval':>18} {'precision':>10}"
    )
    print(header)
    for behavior, stats in det.items():
        print(
            f"{behavior:<16} {stats['marker']:<18} {stats['family']:<6} "
            f"{stats['n_planted']:>5} {_fmt(stats.get('recall')):>8} "
            f"{_fmt_ci(stats.get('recall_ci95')):>18} {_fmt(stats.get('precision')):>10}"
        )

    print("\n== the best marker anywhere in the library, per behavior ==")
    for behavior, stats in best.items():
        top = stats["best"]
        same = " (the one that names it)" if top["marker"] == stats["named_marker"] else ""
        print(
            f"{behavior:<16} {top['marker']:<18} recall {top['recall']:.3f} "
            f"precision {top['precision']:.3f} f1 {top['f1']:.3f}{same}"
        )

    print("\n== where each marker fires ==")
    for marker, stats in cross.items():
        print(
            f"{marker:<18} fired on {stats['n_fired']:>4} rows, "
            f"{stats['on_honest_rows']:>4} of them with nothing planted "
            f"({stats['by_planted_behavior']})"
        )

    print("\n== what a fully green marker dashboard is still carrying ==")
    for label, block in (("style_report only", green), ("every marker family", green_all)):
        share = block["share_of_green_rows_carrying_a_planted_failure"]
        print(
            f"{label:<22} {block['n_marker_clean_but_planted']:>4} of "
            f"{block['n_marker_clean']:>5} green rows carry a plant "
            f"({_fmt(share)} {_fmt_ci(block['ci95'])})  {block['by_behavior']}"
        )

    # The verdict test: one `compare` carries the gold metric (pass@1 on
    # `reward` = the row was clean) and every shared marker, so the gap between
    # the two is read off one report.
    floor = wai.eval_variance(*[before[s] for s in seeds])
    # Per-metric floors: the gold metric and each marker get their own band.
    run_std = floor.get("run_std_by_metric") or floor.get("run_std")
    kwargs = {"run_std": run_std, "run_std_runs": len(seeds)} if run_std and len(seeds) >= 2 else {}
    report = wai.compare(before[seeds[0]], after[seeds[0]], **kwargs)
    print("\n== did the marker recover the improvement that happened? ==")
    print(report)

    gold_before = sum(r["gold_clean"] for r in before[seeds[0]]) / len(before[seeds[0]])
    gold_after = sum(r["gold_clean"] for r in after[seeds[0]]) / len(after[seeds[0]])
    mk_before = sum(r["marker_clean"] for r in before[seeds[0]]) / len(before[seeds[0]])
    mk_after = sum(r["marker_clean"] for r in after[seeds[0]]) / len(after[seeds[0]])
    all_before = sum(r["all_markers_clean"] for r in before[seeds[0]]) / len(before[seeds[0]])
    all_after = sum(r["all_markers_clean"] for r in after[seeds[0]]) / len(after[seeds[0]])
    gold_delta = gold_after - gold_before
    marker_delta = mk_after - mk_before
    all_delta = all_after - all_before
    recovered = round(marker_delta / gold_delta, 4) if gold_delta else None
    recovered_all = round(all_delta / gold_delta, 4) if gold_delta else None

    replication = []
    for s in seeds:
        g = sum(r["gold_clean"] for r in after[s]) / len(after[s]) - sum(
            r["gold_clean"] for r in before[s]
        ) / len(before[s])
        m = sum(r["marker_clean"] for r in after[s]) / len(after[s]) - sum(
            r["marker_clean"] for r in before[s]
        ) / len(before[s])
        replication.append(
            {
                "seed": s,
                "gold_delta": round(g, 4),
                "marker_delta": round(m, 4),
                "recovered": round(m / g, 4) if g else None,
            }
        )

    print(f"\ngold        clean rate {gold_before:.3f} -> {gold_after:.3f}  ({gold_delta:+.3f})")
    print(f"style_report clean rate {mk_before:.3f} -> {mk_after:.3f}  ({marker_delta:+.3f})")
    print(f"all markers  clean rate {all_before:.3f} -> {all_after:.3f}  ({all_delta:+.3f})")
    if recovered is not None:
        print(
            f"style_report recovers {recovered:.1%} of the real improvement; "
            f"every marker family recovers {recovered_all:.1%}"
        )

    print("\n== the marker as a judge, against the plant record as program gold ==")
    for behavior, stats in trusts.items():
        print(
            f"{behavior:<16} {stats['marker']:<16} agreement {stats['agreement']} "
            f"kappa {stats['kappa']} gold_kind={stats['gold_kind']} ok={stats['measured']}"
        )

    results = {
        "question": "Can a behavioral marker be trusted as a detector of the behavior it names?",
        "whileai": wai.__version__,
        "offline": True,
        "seeds": seeds,
        "budget": budget,
        "phrasings": phrasings,
        "n_rows_pooled": len(pooled),
        "rates": {"before": args.before_rate, "after": args.after_rate},
        "detection": det,
        "best_detector_anywhere": best,
        "crossfire": cross,
        "green_dashboard": green,
        "green_dashboard_all_markers": green_all,
        "judge_trust_per_marker": trusts,
        "verdict_test": {
            "gold_before": round(gold_before, 4),
            "gold_after": round(gold_after, 4),
            "gold_delta": round(gold_delta, 4),
            "marker_before": round(mk_before, 4),
            "marker_after": round(mk_after, 4),
            "marker_delta": round(marker_delta, 4),
            "all_markers_before": round(all_before, 4),
            "all_markers_after": round(all_after, 4),
            "all_markers_delta": round(all_delta, 4),
            "share_of_improvement_recovered": recovered,
            "share_of_improvement_recovered_all_markers": recovered_all,
            "replication_by_seed": replication,
            "headline_verdict": report.get("headline_verdict"),
            "run_std": run_std,
            "noise_band": floor.get("noise_band"),
            "noise_band_df": floor.get("noise_band_df"),
        },
    }
    if args.modal_seeds:
        results["modal_seeds"] = fold_seeds(args.modal_seeds)
        print("\n== over the Modal seeds ==")
        for key, stats in results["modal_seeds"]["across_seeds"].items():
            print(f"{key:<28} {stats['mean']:.4f} {_fmt_ci(stats['ci95'])}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
