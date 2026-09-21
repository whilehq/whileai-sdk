"""Did training land? Base policy vs served adapter on the holdout, same verifier.

Usage: python delta.py [--before qwen3-4b] [--after hosted-text-to-sql-shop-v1]
Reads out/<model>.scored.jsonl (from build.py), prints wai.delta_report by
difficulty and archetype, and attaches the delta to the training run in
out/train_state.json so the run page shows it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sql_verifier import OUT, read_jsonl

import whileai.simulations as wai
from whileai.config import provenance


def main() -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="qwen3-4b")
    ap.add_argument("--after", default="")
    args = ap.parse_args()
    state = (
        json.loads((OUT / "train_state.json").read_text(encoding="utf-8"))
        if (OUT / "train_state.json").exists()
        else {}
    )
    after_name = args.after or f"hosted-{state.get('name', '')}"
    before = [
        r for r in read_jsonl(OUT / f"{args.before}.scored.jsonl") if r.get("split") == "holdout"
    ]
    after = [
        r for r in read_jsonl(OUT / f"{after_name}.scored.jsonl") if r.get("split") == "holdout"
    ]
    if not before or not after:
        print(
            f"need graded holdout rows for {args.before} ({len(before)}) and {after_name} ({len(after)}); run build.py first"
        )
        return 1
    reports = {}
    for by in ("difficulty", "category"):
        rep = wai.delta_report(before=before, after=after, target="pass_at_1", by=by)
        reports[by] = rep
        print(f"== by {by}")
        print(
            wai.format_delta_report(rep)
            if hasattr(wai, "format_delta_report")
            else json.dumps(
                {
                    k: rep.get(k)
                    for k in (
                        "before",
                        "after",
                        "delta",
                        "interval",
                        "verdict",
                        "moved",
                        "groups",
                        "groups_down",
                    )
                },
                indent=1,
                default=str,
            )
        )
    (OUT / "delta.json").write_text(json.dumps(reports, indent=1, default=str), encoding="utf-8")
    run_id = state.get("run_id")
    if run_id:
        try:
            wai.attach_delta(run_id, before, after, target="pass_at_1", by="difficulty")
            print(f"delta attached to {run_id}")
        except Exception as exc:
            print(f"attach_delta failed: {type(exc).__name__}: {str(exc)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
