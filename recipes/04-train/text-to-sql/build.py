"""Grade every rollout file with the SQL verifier, benchmark, build the sets, push.

Usage:
  python build.py                 # grade + benchmark report, nothing pushed
  python build.py --push          # also push train / holdout / sft / eval sets
  python build.py --policy qwen3-4b   # which model's rollouts are the RL/holdout source

Reads  raw/<model>.jsonl (from rollout.py) and tasks.jsonl.
Writes out/<model>.scored.jsonl, out/rl.jsonl, out/sft.jsonl, out/holdout.jsonl,
       out/benchmark.md, out/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sql_verifier import (
    AGENT,
    OUT,
    RAW,
    SQLExec,
    extract_sql,
    load_tasks,
    read_jsonl,
    split_of,
    teacher_row,
    write_jsonl,
)

import whileai.simulations as wai
from whileai.config import provenance
from whileai.simulations.score.hack_scan import format_hack_scan, hack_scan
from whileai.simulations.score.judging import run_judge
from whileai.simulations.score.passat import pass_at

VERIFIER_VERSION = "sql_exec@db_snapshot_2026-07"

# SDK 0.44 `looks_finished` reads terminal punctuation or a sign-off, so a reply
# that ends on a closed code fence (every SQL answer here) is called truncated
# and optimize(mode="rl") drops it. A closed fence is a finished reply.
from whileai.simulations.score import grading as _grading
from whileai.simulations.score import hygiene as _hygiene

_orig_finished = _grading.looks_finished


def _finished(final: str) -> bool:
    t = (final or "").rstrip()
    if t.endswith("```") and t.count("```") % 2 == 0:
        return True
    return _orig_finished(final)


_grading.looks_finished = _finished
if hasattr(_hygiene, "looks_finished"):
    _hygiene.looks_finished = _finished


def stamp_markers(row: dict) -> dict:
    reason = str(row.get("reason") or "")
    has_sql = extract_sql(row.get("final_text") or "") is not None
    executes = has_sql and not reason.startswith("sql error")
    m = dict(row.get("markers") or {})
    m.update({"has_sql": int(has_sql), "executes": int(executes)})
    row["markers"] = m
    return row


def grade(model: str) -> list[dict]:
    rows = read_jsonl(RAW / f"{model}.jsonl")
    scored = run_judge(rows, SQLExec(), model=model, version=VERIFIER_VERSION, concurrency=8)
    out = [stamp_markers(r) for r in scored.rows]
    write_jsonl(OUT / f"{model}.scored.jsonl", out)
    return out


def by_group(rows: list[dict], key: str) -> dict[str, float | None]:
    """pass@1 per value of `key` (mean of per-task pass rates)."""
    tasks: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("reward") in (0, 1):
            tasks[str(r.get(key))][r["scenario_id"]].append(int(r["reward"]))
    out = {}
    for k, groups in sorted(tasks.items()):
        out[k] = round(mean(mean(v) for v in groups.values()), 3)
    return out


def summarize(model: str, rows: list[dict]) -> dict:
    hold = [r for r in rows if r.get("split") == "holdout"]
    pa_all = pass_at(rows)
    pa_hold = pass_at(hold)
    graded = [r for r in rows if r.get("reward") in (0, 1)]
    return {
        "model": model,
        "rows": len(rows),
        "tasks": len({r["scenario_id"] for r in rows}),
        "holdout_tasks": len({r["scenario_id"] for r in hold}),
        "all": pa_all.to_dict(),
        "holdout": pa_hold.to_dict(),
        "holdout_by_difficulty": by_group(hold, "difficulty"),
        "holdout_by_archetype": by_group(hold, "category"),
        "holdout_by_style": by_group(hold, "style"),
        "no_sql_rate": round(1 - mean(r["markers"]["has_sql"] for r in graded), 3)
        if graded
        else None,
        "sql_error_rate": round(
            mean(r["markers"]["has_sql"] and not r["markers"]["executes"] for r in graded), 3
        )
        if graded
        else None,
        "truncated_rate": round(mean(bool(r.get("truncated")) for r in graded), 3)
        if graded
        else None,
        "mean_latency_s": round(mean(r.get("latency_s") or 0 for r in rows), 2) if rows else None,
        "mean_completion_tokens": round(
            mean((r.get("usage") or {}).get("completion_tokens") or 0 for r in rows), 1
        )
        if rows
        else None,
    }


def fmt(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def report_md(
    summaries: list[dict],
    tasks: list[dict],
    rl_report: dict | None,
    scan: dict | None,
    pushed: dict,
) -> str:
    n_hold = sum(1 for t in tasks if split_of(t["id"]) == "holdout")
    lines = [
        "# Text-to-SQL on the online-store Postgres schema",
        "",
        f"Tasks: {len(tasks)} (train {len(tasks) - n_hold}, holdout {n_hold}). Verifier: execute candidate SQL on the seeded Postgres database, match the gold result set (Spider-style execution accuracy).",
        "",
        "## Benchmark (holdout, k=4, temperature 0.7)",
        "",
        "| Model | pass@1 | 95% CI | pass^4 | pass@4 | no SQL | SQL error | tokens/reply |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        h = s["holdout"]
        ci = f"{h['ci95'][0]:.2f}..{h['ci95'][1]:.2f}" if h.get("ci95") else "n/a"
        lines.append(
            f"| {s['model']} | {fmt(h['pass_at_1'])} | {ci} | {fmt(h['pass_pow_k'])} | {fmt(h['pass_at_k'])} | {fmt(s['no_sql_rate'])} | {fmt(s['sql_error_rate'])} | {s['mean_completion_tokens']} |"
        )
    lines += ["", "## pass@1 by difficulty (holdout)", ""]
    diffs = sorted({d for s in summaries for d in s["holdout_by_difficulty"]})
    lines.append("| Model | " + " | ".join(diffs) + " |")
    lines.append("|---|" + "---|" * len(diffs))
    for s in summaries:
        lines.append(
            f"| {s['model']} | "
            + " | ".join(fmt(s["holdout_by_difficulty"].get(d)) for d in diffs)
            + " |"
        )
    lines += ["", "## pass@1 by archetype (holdout)", ""]
    archs = sorted({a for s in summaries for a in s["holdout_by_archetype"]})
    lines.append("| Archetype | " + " | ".join(s["model"] for s in summaries) + " |")
    lines.append("|---|" + "---|" * len(summaries))
    for a in archs:
        lines.append(
            f"| {a} | "
            + " | ".join(fmt(s["holdout_by_archetype"].get(a)) for s in summaries)
            + " |"
        )
    if rl_report:
        lines += [
            "",
            "## RL set (policy rollouts, optimize mode=rl)",
            "",
            f"- rows in: {rl_report.get('n')}; rows out: {rl_report.get('n_selected')} in {rl_report.get('groups_selected')} groups",
            f"- duplicate rollouts dropped: {(rl_report.get('duplicates') or {}).get('n_dropped')}",
            f"- truncated dropped: {rl_report.get('truncated_dropped')}",
            f"- unanimous groups dropped: {rl_report.get('unanimous_groups_dropped')}; collapsed after dedupe: {rl_report.get('collapsed_groups_dropped')}",
            f"- band dropped: {rl_report.get('band_dropped')}",
            f"- selection signal: {json.dumps(rl_report.get('signal'))}",
        ]
    if scan:
        lines += [
            "",
            "## What separates reward within a task (hack_scan)",
            "",
            "```",
            format_hack_scan(scan, top=8),
            "```",
        ]
    if pushed:
        lines += ["", "## Pushed to the platform", ""]
        for name, entry in pushed.items():
            lines.append(
                f"- {name}: `{entry.get('datasetId')}` ({entry.get('purpose')}, {entry.get('rows')} rows)"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="qwen3-4b")
    ap.add_argument(
        "--push", action="store_true", help="push train, holdout, rl and every eval set"
    )
    ap.add_argument(
        "--push-evals",
        default="",
        help="comma list of models whose holdout rollouts to push as eval sets (nothing else)",
    )
    ap.add_argument("--models", default="", help="comma list; default = every raw/*.jsonl")
    ap.add_argument(
        "--band",
        default="",
        help="pass-rate band for the RL set as lo,hi (default: the SDK's 0.2,0.8); 0.1,0.9 at k=8 keeps every non-unanimous group",
    )
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    tasks = load_tasks()
    models = [m for m in args.models.split(",") if m] or sorted(p.stem for p in RAW.glob("*.jsonl"))
    scored: dict[str, list[dict]] = {}
    summaries = []
    for m in models:
        rows = grade(m)
        scored[m] = rows
        s = summarize(m, rows)
        summaries.append(s)
        print(f"{m}: all {pass_at(rows)}")
        print(f"{m}: holdout {pass_at([r for r in rows if r.get('split') == 'holdout'])}")

    rl_rows: list[dict] = []
    rl_report = None
    scan = None
    holdout_rows: list[dict] = []
    if args.policy in scored:
        pol = scored[args.policy]
        train = [r for r in pol if r.get("split") == "train"]
        holdout_rows = [r for r in pol if r.get("split") == "holdout"]
        band = tuple(float(x) for x in args.band.split(",")) if args.band else None
        rl_rows, rl_report = wai.optimize(
            train,
            mode="rl",
            endorsed=["marker:executes"],
            target=len(train),  # every in-band group, not the 1,000-row default
            **({"band": band} if band else {}),
        )
        print(
            f"optimize(rl): {len(rl_rows)} rows from {len(train)}; report keys {sorted(rl_report)[:12]}"
        )
        write_jsonl(OUT / "rl.jsonl", rl_rows)
        write_jsonl(OUT / "holdout.jsonl", holdout_rows)
        # The prompts the policy solves 20-80% of the time (the SDK's band):
        # train_grpo_modal.py --task-ids out/band_ids.txt trains the next round
        # on these alone (whilehq/whileai-sdk#254).
        band_ids = sorted({r.get("scenario_id") for r in rl_rows if r.get("scenario_id")})
        (OUT / "band_ids.txt").write_text("\n".join(band_ids) + "\n", encoding="utf-8")
        print(
            f"band: {len(band_ids)} prompts of {len({r.get('scenario_id') for r in train})} -> out/band_ids.txt"
        )
        graded_train = [r for r in train if r.get("reward") in (0, 1)]
        scan = hack_scan(graded_train, endorsed=["marker:executes"])
        print(format_hack_scan(scan, top=8))

    sft_rows = [teacher_row(t) for t in tasks if split_of(t["id"]) == "train"]
    write_jsonl(OUT / "sft.jsonl", sft_rows)
    # The holdout set is the gold demonstrations for the held-out tasks, so a
    # hosted SFT run reports held-out loss on unseen questions; the policy's own
    # holdout rollouts live in its eval set below.
    holdout_gold = [teacher_row(t) for t in tasks if split_of(t["id"]) == "holdout"]
    write_jsonl(OUT / "holdout_gold.jsonl", holdout_gold)

    pushed: dict[str, dict] = {}
    if args.push:
        desc = "Text-to-SQL over a small online-store Postgres database (8 tables, seeded). Reward = execution match against gold SQL."
        if rl_rows:
            e = wai.push_rows(
                rl_rows,
                f"{AGENT}-rl",
                gate=True,
                mode="rl",
                purpose="train",
                agent=AGENT,
                description=desc + f" RL groups from {args.policy}, 20-80% band.",
            )
            pushed[f"{AGENT}-rl"] = {
                "datasetId": e.get("datasetId"),
                "purpose": "train",
                "rows": len(rl_rows),
                "gate": e.get("gate"),
            }
            print("pushed rl", e.get("datasetId"))
        e = wai.push_rows(
            sft_rows,
            f"{AGENT}-sft",
            mode="sft",
            purpose="train",
            agent=AGENT,
            description=desc
            + " Gold SQL demonstrations (teacher: Claude Sonnet 5, execution-verified), train tasks.",
        )
        pushed[f"{AGENT}-sft"] = {
            "datasetId": e.get("datasetId"),
            "purpose": "train",
            "rows": len(sft_rows),
        }
        print("pushed sft", e.get("datasetId"))
        e = wai.push_rows(
            holdout_gold,
            f"{AGENT}-holdout",
            mode="sft",
            purpose="holdout",
            agent=AGENT,
            description=desc
            + " Gold SQL demonstrations for the held-out tasks; measure on this, never train on it.",
        )
        pushed[f"{AGENT}-holdout"] = {
            "datasetId": e.get("datasetId"),
            "purpose": "holdout",
            "rows": len(holdout_gold),
        }
        print("pushed holdout", e.get("datasetId"))
        for m, rows in scored.items():
            hold = [r for r in rows if r.get("split") == "holdout"]
            if not hold:
                continue
            e = wai.push_rows(
                hold,
                f"{AGENT}-eval-{m}",
                purpose="eval",
                agent=AGENT,
                description=desc
                + f" Holdout benchmark rollouts of {m}, k=4, graded by the verifier.",
            )
            pushed[f"{AGENT}-eval-{m}"] = {
                "datasetId": e.get("datasetId"),
                "purpose": "eval",
                "rows": len(hold),
            }
            print("pushed eval", m, e.get("datasetId"))

    if args.push_evals:
        desc = "Text-to-SQL over a small online-store Postgres database (8 tables, seeded). Reward = execution match against gold SQL."
        for m in [x for x in args.push_evals.split(",") if x]:
            hold = [r for r in scored.get(m, []) if r.get("split") == "holdout"]
            if not hold:
                print(f"no holdout rows for {m}")
                continue
            e = wai.push_rows(
                hold,
                f"{AGENT}-eval-{m}",
                purpose="eval",
                agent=AGENT,
                description=desc
                + f" Holdout benchmark rollouts of {m}, k=4, graded by the verifier.",
            )
            pushed[f"{AGENT}-eval-{m}"] = {
                "datasetId": e.get("datasetId"),
                "purpose": "eval",
                "rows": len(hold),
            }
            print("pushed eval", m, e.get("datasetId"))

    manifest = {
        "tasks": len(tasks),
        "holdout_tasks": sum(1 for t in tasks if split_of(t["id"]) == "holdout"),
        "verifier": VERIFIER_VERSION,
        "models": summaries,
        "rl": {
            k: v
            for k, v in (rl_report or {}).items()
            if isinstance(v, (int, float, str, list, dict)) and k != "rows"
        },
        "hack_scan": {
            k: scan.get(k) for k in ("regime", "tau", "top_feature", "integrity", "warnings")
        }
        if scan
        else None,
        "sft_rows": len(sft_rows),
        "pushed": pushed,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8"
    )
    (OUT / "benchmark.md").write_text(
        report_md(summaries, tasks, rl_report, scan, pushed), encoding="utf-8"
    )
    print((OUT / "benchmark.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
