"""Read the rows the Modal runs wrote and apply PREREGISTRATION.md.

    modal run experiment_modal.py --step fetch    # volume -> out/
    python analyze.py                             # prints the verdict, writes results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import whileai as wai
from whileai.config import provenance

OUT = Path(__file__).resolve().parent / "out"
ARMS = ("authors", "whileai")
SEEDS = (1, 2)


def load(name: str) -> dict:
    return json.loads((OUT / "eval" / f"{name}.json").read_text(encoding="utf-8"))


def rows_of(run: dict, key: str = "sampled") -> list[dict]:
    return [r for r in run[key] if r.get("reward") is not None]


def line(name: str, rows: list[dict]) -> dict:
    pa = wai.pass_at(rows)
    ci = f"[{pa.ci95[0]:.3f}, {pa.ci95[1]:.3f}]" if pa.ci95 else "n/a"
    print(f"  {name:<14} pass@1 {pa.pass_at_1:.3f} {ci}  tasks {pa.n_groups}")
    return {"pass_at_1": pa.pass_at_1, "ci95": pa.ci95, "tasks": pa.n_groups}


def greedy(run: dict) -> float:
    g = rows_of(run, "greedy")
    return sum(r["reward"] for r in g) / len(g)


def main() -> int:
    print(provenance(), file=sys.stderr)
    base = load("base")
    runs = {(a, s): load(f"{a}-s{s}") for a in ARMS for s in SEEDS if (OUT / "eval" / f"{a}-s{s}.json").exists()}
    missing = [f"{a}-s{s}" for a in ARMS for s in SEEDS if (a, s) not in runs]
    if missing:
        print(f"not finished: {', '.join(missing)}")

    print("\n== held-out test, pass@1 from 4 samples at T=0.8 (primary)")
    result: dict = {"base": line("base", rows_of(base))}
    for (a, s), run in sorted(runs.items()):
        result[f"{a}-s{s}"] = {
            **line(f"{a} seed {s}", rows_of(run)),
            "greedy": greedy(run),
            "steps": run["steps"],
            "train_seconds": run["train_seconds"],
            "zero_spread_share": run["reward_stats"]["zero_spread_groups"]
            / max(run["reward_stats"]["groups"], 1),
        }

    print("\n== greedy pass@1 (the authors' eval_pass1.py metric)")
    print(f"  {'base':<14} {greedy(base):.3f}")
    for (a, s), run in sorted(runs.items()):
        print(f"  {a + ' seed ' + str(s):<14} {greedy(run):.3f}")

    print("\n== share of training groups with no reward spread (no gradient)")
    for (a, s), run in sorted(runs.items()):
        print(f"  {a + ' seed ' + str(s):<14} {result[f'{a}-s{s}']['zero_spread_share']:.2f}")

    if all((a, s) in runs for a in ARMS for s in SEEDS):
        seeds = {a: [rows_of(runs[(a, s)]) for s in SEEDS] for a in ARMS}
        print("\n== primary: with whileai minus authors, both seeds pooled, paired by task")
        report = wai.compare(
            [r for rs in seeds["authors"] for r in rs],
            [r for rs in seeds["whileai"] for r in rs],
            target="pass_at_1",
            train_runs={"before": seeds["authors"], "after": seeds["whileai"]},
        )
        print(report)
        result["primary"] = json.loads(json.dumps(report, default=str))
        for a in ARMS:
            print(f"\n== {a} minus base")
            vs = wai.compare(
                rows_of(base),
                [r for rs in seeds[a] for r in rs],
                target="pass_at_1",
                train_runs={"before": None, "after": seeds[a]},
            )
            print(str(vs).splitlines()[0])
            result[f"{a}_vs_base"] = json.loads(json.dumps(vs, default=str))

    (Path(__file__).resolve().parent / "results.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
