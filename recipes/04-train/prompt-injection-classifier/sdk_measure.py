"""The SDK's own measurement over the scored rows, and where it does not fit a classifier.

    python sdk_measure.py

Reads the score files under ``out/`` and writes ``out/sdk_measure.json``:

* ``decontaminate`` train against the frozen tests (8-gram over ``text``), count dropped;
* ``hack_scan`` on the matched training pairs: the pair is the ask, the label is the reward,
  the surface features it ranks are the shortcut detector;
* ``eval_variance`` across the three training seeds on the correctness marker (a
  training-seed floor, not an eval re-run floor: the scorer is deterministic);
* ``wai.compare`` per slice, ProtectAI as ``before`` and each seed as ``after``, on correctness;
* ``holdout_size`` per slice for a five-point gain over the baseline's accuracy;
* ``wai.methods.route`` on the final seed's graded predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from data import read_jsonl

import whileai as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def graded(rows: list[dict], scores: list[float], thr: float, model: str, run: str) -> list[dict]:
    out = []
    for i, (r, s) in enumerate(zip(rows, scores)):
        out.append(
            {
                "prompt": f"row-{i}",
                "scenario_id": f"row-{i}",
                "task_key": f"row-{i}",
                "reward": float(int(s > thr) == r["label"]),
                "slice": r["slice"],
                "final_text": r["text"],
                "model_version": model,
                "label_source": "program",
                "finish_reason": "stop",
                "rollout_index": 0,
                "lineage": {"eval_run": run},
            }
        )
    return out


def main() -> None:
    print(provenance(), file=sys.stderr)
    test = read_jsonl(HERE / "test.jsonl")
    hard = read_jsonl(HERE / "test_hard.jsonl")
    train = read_jsonl(OUT / "train.jsonl")
    result: dict = {}

    # 1. decontaminate: the SDK's 8-gram over a "prompt" field, so text rides as prompt
    tr = [{"prompt": r["text"], **{k: v for k, v in r.items() if k != "text"}} for r in train]
    against = [{"prompt": r["text"]} for r in test + hard]
    clean, rep = wai.decontaminate(tr, against=against, n=8)
    result["decontaminate"] = {
        "train_rows": len(train),
        "kept": len(clean),
        "dropped": len(train) - len(clean),
        "report": dict(rep),
    }
    print(
        f"decontaminate: {len(train) - len(clean)} of {len(train)} training rows share an 8-gram with a test"
    )

    # 2. hack_scan on the matched pairs: the pair is the ask
    pairs = [r for r in train if r.get("pair")]
    scan_rows = [
        {
            "prompt": r["pair"],
            "scenario_id": r["pair"],
            "reward": float(r["label"]),
            "final_text": r["text"],
            "messages": [{"role": "assistant", "content": r["text"]}],
        }
        for r in pairs
    ]
    try:
        scan = wai.hack_scan(scan_rows)
        result["hack_scan"] = {
            k: v
            for k, v in dict(scan).items()
            if k in ("regime", "top", "features", "verdict", "note", "warning", "n_groups")
        }
        print(scan)
    except Exception as e:  # the SDK's shape may not take these rows; that is the finding
        result["hack_scan"] = {"error": f"{type(e).__name__}: {e}"}
        print("hack_scan failed:", e)

    # 3. scores
    base = json.loads((OUT / "scores_protectai.json").read_text())
    seeds = {
        p.stem.rsplit("seed", 1)[1]: json.loads(p.read_text())
        for p in sorted(OUT.glob("scores_minilm_seed*.json"))
    }
    before = graded(test, base["scores"], base["threshold"], base["model"], "protectai")
    runs = {
        s: graded(test, r["scores"], r["threshold"], r["model"], f"seed{s}")
        for s, r in seeds.items()
    }

    # eval_variance across seeds
    try:
        ev = wai.eval_variance(*runs.values(), metric="pass_at_1")
        result["eval_variance"] = dict(ev)
        print(ev)
        evs = wai.eval_variance(*runs.values(), metric="pass_at_1", by="slice")
        result["eval_variance_by_slice"] = dict(evs)
    except Exception as e:
        result["eval_variance"] = {"error": f"{type(e).__name__}: {e}"}
        print("eval_variance failed:", e)

    # compare per slice, baseline vs each seed
    result["compare"] = {}
    for s, after in runs.items():
        try:
            rep = wai.compare(before, after, by="slice")
            result["compare"][f"seed{s}"] = json.loads(json.dumps(dict(rep), default=str))
            print(f"--- compare protectai -> seed {s}")
            print(rep)
        except Exception as e:
            result["compare"][f"seed{s}"] = {"error": f"{type(e).__name__}: {e}"}
            print("compare failed:", e)

    # holdout_size per slice: can the slice resolve five points over the baseline's accuracy?
    result["holdout_size"] = {}
    slices = sorted({r["slice"] for r in test})
    for name in slices:
        rows_b = [r for r in before if r["slice"] == name]
        acc = sum(r["reward"] for r in rows_b) / len(rows_b)
        try:
            hs = wai.holdout_size(0.05, base=min(acc, 0.89), k=1, rows=rows_b)
            d = dict(hs)
            result["holdout_size"][name] = {
                "n_have": len(rows_b),
                "base_accuracy": acc,
                **json.loads(json.dumps(d, default=str)),
            }
            print(f"holdout_size {name}: have {len(rows_b)}; {hs}")
        except Exception as e:
            result["holdout_size"][name] = {"error": f"{type(e).__name__}: {e}"}

    # route on the final seed's graded predictions
    last = runs[sorted(runs)[-1]]
    try:
        from whileai.routing import route as route_fn  # wai.methods.route in newer checkouts
    except ImportError:
        route_fn = None
    if route_fn is None:
        result["route"] = (
            "whileai.routing not in this checkout; see out/route.json from route_probe.py"
        )
    else:
        route = route_fn(last, model=seeds[sorted(runs)[-1]]["model"], size_b=0.022)
        result["route"] = str(route)
        print(route)

    (OUT / "sdk_measure.json").write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
