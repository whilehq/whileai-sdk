"""Hosted SFT on the gold demonstrations, then serve the adapter.

Usage: python train.py --sft ds_... --holdout ds_... [--epochs 3] [--name text-to-sql-shop-v1]
State (run id, model name, endpoint) lands in out/train_state.json.

Then benchmark the served model with the verifier:
  python rollout.py --model qwen3-4b --hosted <name> --split holdout --k 4
  python build.py
  python delta.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sql_verifier import OUT

import whileai.simulations as wai
from whileai.config import provenance

STATE = OUT / "train_state.json"


def main() -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--name", default="text-to-sql-shop-v1")
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--timeout", type=float, default=2400)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    run = wai.train(
        args.sft,
        method="sft",
        epochs=args.epochs,
        holdout=args.holdout,
        base_model=args.base,
        wait=True,
        timeout=args.timeout,
    )
    training = getattr(run, "training", None) or {}
    run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
    print(
        f"run {run_id} status={run.status}: {json.dumps(training, default=str)[:600]}  ({time.time() - t0:.0f}s)"
    )
    state = {
        "run_id": run_id,
        "status": run.status,
        "training": training,
        "sft": args.sft,
        "holdout": args.holdout,
        "epochs": args.epochs,
        "base": args.base,
    }
    STATE.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    if run.status != "done":
        print("training did not finish cleanly; not serving")
        return 1
    model = wai.serve(args.name, run)
    state.update({"model": model, "name": args.name})
    STATE.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    print(f"served {args.name} v{model.get('version')} at {model.get('endpoint')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
