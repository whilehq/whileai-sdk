"""Self-distillation: the model's own verified thinking traces as SFT data.

The base solves most tasks in one of k samples (pass@4 0.84 here) but not
reliably (pass@1 0.61). Rejection sampling keeps the traces that executed
and matched the gold, `optimize(mode="sft")` picks one per prompt, the
hosted trainer fine-tunes the same base on them, and the served adapter is
measured on the untouched holdout. Train prompts only; the holdout never
feeds this.

    python rollout.py --hosted qwen3-4b-think --split train --k 4   # the samples
    python build.py                                                 # grades them
    python distill.py --source hosted-qwen3-4b-think [--epochs 2] [--name text-to-sql-shop-sft-think-v1]
    python rollout.py --hosted text-to-sql-shop-sft-think-v1 --split holdout --k 4
    python build.py && python delta.py --before hosted-qwen3-4b-think --after hosted-text-to-sql-shop-sft-think-v1

The hosted SFT trainer truncates at 2048 tokens (prompt + reply), so traces
longer than --max-chars are dropped rather than cut mid-answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sql_verifier import AGENT, OUT, extract_sql, read_jsonl

import whileai.simulations as wai
from whileai.config import provenance

STATE = OUT / "distill_state.json"


def main() -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="hosted-qwen3-4b-think",
        help="graded rollout file (out/<source>.scored.jsonl)",
    )
    ap.add_argument(
        "--holdout",
        default="",
        help="holdout dataset id for held-out loss (gold demos); from out/manifest.json when empty",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="drop traces longer than this (hosted SFT caps at 2048 tokens with the prompt)",
    )
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--name", default="text-to-sql-shop-sft-think-v1")
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--dry-run", action="store_true", help="select and report, push nothing")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(OUT / f"{args.source}.scored.jsonl") if r.get("split") == "train"]
    if not rows:
        print(
            f"no graded train rows in out/{args.source}.scored.jsonl; run rollout.py --split train and build.py first"
        )
        return 1
    correct = [r for r in rows if r.get("reward") == 1]
    complete = [
        r
        for r in correct
        if "</think>" in str(r.get("final_text") or "") and extract_sql(r["final_text"])
    ]
    short = [r for r in complete if len(str(r["final_text"])) <= args.max_chars]
    picked, _report = wai.optimize(short, mode="sft", select="top_per_prompt", min_reward=1.0)
    tasks_all = {r["scenario_id"] for r in rows}
    print(
        f"train rows {len(rows)} on {len(tasks_all)} tasks; correct {len(correct)}; complete traces {len(complete)}; "
        f"within {args.max_chars} chars {len(short)}; selected {len(picked)} (one per prompt) covering "
        f"{len({r['scenario_id'] for r in picked})}/{len(tasks_all)} tasks"
    )
    if args.dry_run:
        return 0

    holdout = args.holdout
    if not holdout and (OUT / "manifest.json").exists():
        pushed = json.loads((OUT / "manifest.json").read_text(encoding="utf-8")).get("pushed") or {}
        holdout = (pushed.get(f"{AGENT}-holdout") or {}).get("datasetId") or ""
    desc = "Self-distillation: the base model's own thinking traces that executed and matched the gold, one per train prompt (rejection sampling)."
    entry = wai.push_rows(
        picked, f"{AGENT}-sft-think", mode="sft", purpose="train", agent=AGENT, description=desc
    )
    sft_id = entry.get("datasetId")
    print(f"pushed {sft_id} ({len(picked)} rows); holdout {holdout or 'none'}")

    t0 = time.time()
    run = wai.train(
        sft_id,
        method="sft",
        epochs=args.epochs,
        holdout=holdout or None,
        base_model=args.base,
        wait=True,
        timeout=3600,
    )
    run_id = getattr(run, "run_id", None)
    training = getattr(run, "training", None) or {}
    print(
        f"run {run_id} {run.status}: loss {training.get('before')} -> {training.get('after')}, {training.get('seconds')} s ({time.time() - t0:.0f} s wall)"
    )
    state = {
        "sft": sft_id,
        "holdout": holdout,
        "run_id": run_id,
        "status": run.status,
        "training": training,
        "rows": len(picked),
        "source": args.source,
        "max_chars": args.max_chars,
        "epochs": args.epochs,
    }
    STATE.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    if run.status != "done":
        return 1
    model = wai.serve(args.name, run)
    state["model"] = model
    STATE.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    print(
        f"served {args.name} v{model.get('version')}; benchmark: python rollout.py --hosted {args.name} --split holdout --k 4"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
