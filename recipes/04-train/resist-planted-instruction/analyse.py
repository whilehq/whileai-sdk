"""Paired base, control and trained arms over the pinned holdout.

pass@1 per arm with a bootstrap over PROMPTS, every pairwise delta with its
interval, an exact sign test over discordant prompts, graded count per arm,
attack rows and clean control rows apart, the false-flag rate, and the effect
this eval can resolve at its own measured spread (rlhf-book ch. 16).

Run: python analyse.py --out out
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rubric import criterion_failures


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def task_means(rows: list[dict]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        v = r.get("reward")
        if isinstance(v, bool) or not isinstance(v, int | float):
            continue
        key = str(r.get("scenario_id") or r.get("task_id") or r.get("prompt") or "")
        groups.setdefault(key, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in groups.items()}


def boot(values: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def sign_test(diffs: list[float]) -> dict:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n_discordant": 0, "p_value": None}
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {"n_discordant": n, "improved": pos, "worsened": neg, "p_value": min(1.0, 2 * tail)}


def split_attack(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Attacked rows and clean control rows, kept apart. A policy that flags
    every order scores perfectly on one half and fails the other."""
    attacked, clean = [], []
    for r in rows:
        meta = r.get("judge_meta") or {}
        (clean if str(meta.get("attack_kind")) == "none" else attacked).append(r)
    return attacked, clean


def flag_rates(rows: list[dict]) -> dict:
    clean_n = clean_flagged = att_n = att_flagged = 0
    for r in rows:
        meta = r.get("judge_meta") or {}
        if "filed_security_review" not in meta:
            continue
        filed = bool(meta.get("filed_security_review"))
        if meta.get("had_planted_text"):
            att_n += 1
            att_flagged += int(filed)
        else:
            clean_n += 1
            clean_flagged += int(filed)
    return {
        "false_flag_rate": round(clean_flagged / clean_n, 4) if clean_n else None,
        "flag_recall": round(att_flagged / att_n, 4) if att_n else None,
        "clean_rows": clean_n,
        "attack_rows": att_n,
    }


def arm(name: str, rows: list[dict]) -> dict:
    means = task_means(rows)
    values = list(means.values())
    graded = sum(
        1
        for r in rows
        if isinstance(r.get("reward"), int | float) and not isinstance(r.get("reward"), bool)
    )
    lo, hi = boot(values) if len(values) >= 3 else (None, None)
    return {
        "arm": name,
        "rows": len(rows),
        "graded_rows": graded,
        "n_prompts": len(means),
        "pass_at_1": round(sum(values) / len(values), 4) if values else None,
        "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        "per_prompt_sd": round(statistics.pstdev(values), 4) if len(values) > 1 else None,
        "flag_rates": flag_rates(rows),
        "criterion_failures": criterion_failures(rows)["failures"],
        "final_text_chars_mean": round(
            statistics.mean(len(r.get("final_text") or "") for r in rows)
        )
        if rows
        else None,
    }


def paired(a: list[dict], b: list[dict], seed: int = 0) -> dict:
    ma, mb = task_means(a), task_means(b)
    shared = sorted(set(ma) & set(mb))
    diffs = [mb[t] - ma[t] for t in shared]
    lo, hi = boot(diffs, seed=seed) if len(diffs) >= 3 else (None, None)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    # two-sided alpha 0.05 at 80% power, normal approximation on the paired differences
    resolvable = round((1.959964 + 0.841621) * sd / math.sqrt(len(diffs)), 4) if diffs else None
    return {
        "n_paired_prompts": len(shared),
        "n_only_a": len(set(ma) - set(mb)),
        "n_only_b": len(set(mb) - set(ma)),
        "delta_pass_at_1": round(sum(diffs) / len(diffs), 4) if diffs else None,
        "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        "paired_diff_sd": round(sd, 4),
        "improved": sum(1 for d in diffs if d > 0),
        "worsened": sum(1 for d in diffs if d < 0),
        "unchanged": sum(1 for d in diffs if d == 0),
        "sign_test": sign_test(diffs),
        "resolvable_effect_80pct_power": resolvable,
        "verdict": "b_better"
        if lo is not None and lo > 0
        else "a_better"
        if hi is not None and hi < 0
        else "no_difference_detected",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="out", help="folder holding eval_<arm>.jsonl")
    args = p.parse_args(argv)
    out = Path(args.out)
    arms = {name: load(out / f"eval_{name}.jsonl") for name in ("base", "random", "trained")}
    present = {k: v for k, v in arms.items() if v}
    if "base" not in present:
        print("no eval_base.jsonl under", out)
        return 1
    report: dict = {name: arm(name, rows) for name, rows in present.items()}
    report["paired"] = {}
    report["paired_by_half"] = {}
    for a, b in (("base", "random"), ("base", "trained"), ("random", "trained")):
        if a in present and b in present:
            report["paired"][f"{a}_vs_{b}"] = paired(present[a], present[b])
            a_att, a_cln = split_attack(present[a])
            b_att, b_cln = split_attack(present[b])
            report["paired_by_half"][f"{a}_vs_{b}"] = {
                "attack_rows": paired(a_att, b_att, seed=1),
                "clean_control_rows": paired(a_cln, b_cln, seed=2),
            }
    (out / "analysis.json").write_text(json.dumps(report, indent=1))
    for name, a in report.items():
        if name in present:
            print(
                f"{name:8s} rows {a['rows']} graded {a['graded_rows']} prompts {a['n_prompts']} "
                f"pass@1 {a['pass_at_1']} {a['ci95']} false-flag {a['flag_rates']['false_flag_rate']}"
            )
    for k, v in report["paired"].items():
        print(
            f"{k:18s} n {v['n_paired_prompts']} delta {v['delta_pass_at_1']:+.4f} {v['ci95']} "
            f"up/down {v['improved']}/{v['worsened']} sign p {v['sign_test']['p_value']} "
            f"resolves {v['resolvable_effect_80pct_power']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
