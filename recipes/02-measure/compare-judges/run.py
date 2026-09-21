"""Six judges, the same 300 labeled rollouts, one ranked table.

The judge is where a training set gets its labels, so before you trust one,
put several on the same rows and score each against an answer key. This
recipe ships 300 real rollouts with a rule-computed answer key and runs
``compare_judges`` over any judges you name: a decision model (Jev), the
hosted chat judge, Claude, or your own callable.

    python run.py                       # grade with every judge you have a key for
    python run.py --judges jev-latest   # one judge
    python run.py --dry-run             # offline: three toy judges, no key
    python run.py report                # reprint the published table, no network

Needs TYPESAFE_API_KEY for the Jev judges, a whileai login (or
WHILEAI_API_KEY) for the hosted judge, ANTHROPIC_API_KEY for Claude. Judges
without a key are skipped and the report says so. No training run, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import whileai as wai
from whileai.config import provenance
from whileai.judge_comparison import compare_judges

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows" / "labeled.jsonl"
PUBLISHED = HERE / "rows" / "results.json"
OUT = HERE / "out"
SEED = 0

# name -> (what it needs, how to build it). The order is the order the table
# is graded in; the table itself ranks by kappa.
JUDGES: dict[str, tuple[str, str]] = {
    "jev-latest": ("TYPESAFE_API_KEY", "typesafe:jev-latest"),
    "jev-preview": ("TYPESAFE_API_KEY", "typesafe:jev-preview"),
    "hosted": ("whileai login", "hosted"),
    # the policy that wrote the rows, judging itself: the self-preference control
    "qwen3-4b": (
        "whileai login",
        "vllm:Qwen/Qwen3-4B@https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1",
    ),
    "haiku-4.5": ("ANTHROPIC_API_KEY", "anthropic:claude-haiku-4-5"),
    "sonnet-5": ("ANTHROPIC_API_KEY", "anthropic:claude-sonnet-5"),
}


def load_rows(limit: int | None = None) -> list[dict]:
    """The 300 checked-in rows. ``limit`` takes a seeded shuffle first, so a
    smoke run sees passes and failures from every domain, not the first
    file's first passes."""
    rows = [
        json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not limit or limit >= len(rows):
        return rows
    random.Random(SEED).shuffle(rows)
    return rows[:limit]


def has_key(need: str) -> bool:
    if need == "whileai login":
        try:
            return bool(wai.resolve_api_key())
        except Exception:
            return False
    return bool(os.environ.get(need, "").strip())


def build_judges(names: list[str], *, bedrock: bool) -> dict[str, object]:
    judges: dict[str, object] = {}
    for name in names:
        if name not in JUDGES:
            sys.exit(f"unknown judge {name!r}; choose from {', '.join(JUDGES)}")
        need, spec = JUDGES[name]
        if bedrock and spec.startswith("anthropic:"):
            from bedrock_judge import MODEL_IDS, BedrockJudge

            judges[name + " (bedrock)"] = BedrockJudge(MODEL_IDS[spec])
            continue
        if not has_key(need):
            print(f"skip {name}: needs {need}")
            continue
        judges[name] = wai.Hosted() if spec == "hosted" else spec
    if not judges:
        sys.exit(
            "no judge has a key; set one of TYPESAFE_API_KEY, ANTHROPIC_API_KEY, or run `whileai login`"
        )
    return judges


def toy_judges(rows: list[dict]) -> dict[str, object]:
    """Offline judges for --dry-run: the answer key itself, a yes-machine,
    and one that grades by length. They exist to show the table's shape."""
    gold = {r["rollout_id"]: int(r["gold_reward"]) for r in rows}
    median = sorted(len(str(r.get("final_text") or "")) for r in rows)[len(rows) // 2]
    return {
        "the rule itself": lambda r: {"reward": gold[r["rollout_id"]], "reason": "answer key"},
        "always pass": lambda r: {"reward": 1, "reason": "yes"},
        "longer is better": lambda r: {
            "reward": int(len(str(r.get("final_text") or "")) >= median),
            "reason": "length",
        },
    }


def reason_category(row: dict) -> str:
    s = str(row.get("rule_reason") or "")
    s = re.sub(r"\(.*", "", s)
    s = re.sub(r":.*", "", s)
    s = re.sub(r"\b[a-z_]+_[a-z_]+\b.*", "", s)
    return s.strip()[:45] or "unlabeled"


def by_reason(rows: list[dict], table) -> dict[str, dict]:
    """Share of rows each judge matched the rule on, per rule reason."""
    cats = sorted({(int(r["gold_reward"]), reason_category(r)) for r in rows})
    out: dict[str, dict] = {}
    for gold, cat in cats:
        ids = [
            r["rollout_id"]
            for r in rows
            if int(r["gold_reward"]) == gold and reason_category(r) == cat
        ]
        cell: dict[str, float | None] = {}
        for score in table:
            got = {r["rollout_id"]: r.get("reward") for r in score.rows}
            judged = [got[i] for i in ids if got.get(i) in (0, 1)]
            cell[score.name] = (
                round(sum(int(x) == gold for x in judged) / len(judged), 3) if judged else None
            )
        out[f"{cat} (gold {gold}, n={len(ids)})"] = cell
    return out


def print_by_reason(breakdown: dict[str, dict]) -> None:
    names = list(next(iter(breakdown.values())).keys()) if breakdown else []
    print("\nagreement with the rule by rule reason:")
    print(f"{'reason':45} " + " ".join(f"{n[:12]:>12}" for n in names))
    for reason, cell in breakdown.items():
        print(
            f"{reason:45} "
            + " ".join(
                f"{(cell[n] if cell[n] is not None else float('nan')):>12.2f}" for n in names
            )
        )


def print_results(results: dict) -> None:
    print(results["table_text"])
    print_by_reason(results["by_reason"])
    print(f"\nrows: {results['n_rows']}, gold: {results['gold']}, run: {results['ran_at']}")


def stage_run(args) -> int:
    rows = load_rows(args.limit)
    judges = (
        toy_judges(rows)
        if args.dry_run
        else build_judges(args.judges.split(","), bedrock=args.bedrock)
    )
    started = time.time()
    # The answer key is a program, so ok= can never be true without saying so.
    table = compare_judges(rows, judges, allow_model_gold=True, concurrency=args.concurrency)
    results = {
        "n_rows": len(rows),
        "gold": "conduct rule (kind=program), see README",
        "ran_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "seconds": round(time.time() - started, 1),
        "dry_run": bool(args.dry_run),
        "table": table.to_dict(),
        "table_text": str(table),
        "by_reason": by_reason(rows, table),
    }
    print_results(results)
    if not args.dry_run:
        OUT.mkdir(exist_ok=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        for score in table:
            safe = re.sub(r"[^a-z0-9]+", "-", score.name.lower()).strip("-")
            with (OUT / f"graded-{safe}.jsonl").open("w", encoding="utf-8") as f:
                for r in score.rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"saved out/results.json and one graded-*.jsonl per judge under {OUT}")
    return 0


def stage_report(args) -> int:
    path = (
        OUT / "results.json"
        if (OUT / "results.json").exists() and not args.published
        else PUBLISHED
    )
    results = json.loads(path.read_text(encoding="utf-8"))
    print(f"from {path.relative_to(HERE)}")
    print_results(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", nargs="?", default="run", choices=["run", "report"])
    p.add_argument(
        "--judges", default=",".join(JUDGES), help="comma-separated names: " + ", ".join(JUDGES)
    )
    p.add_argument("--limit", type=int, default=None, help="rows to grade (default: all 300)")
    p.add_argument("--concurrency", type=int, default=8, help="parallel calls per judge")
    p.add_argument(
        "--bedrock",
        action="store_true",
        help="reach Claude through AWS Bedrock (boto3) instead of ANTHROPIC_API_KEY",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="offline: three toy judges, no key, no network"
    )
    p.add_argument(
        "--published",
        action="store_true",
        help="report: print the checked-in run even if out/ exists",
    )
    args = p.parse_args(argv)
    return stage_report(args) if args.stage == "report" else stage_run(args)


if __name__ == "__main__":
    sys.exit(main())
