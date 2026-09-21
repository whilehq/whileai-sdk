"""Identity/spec agent: does less of volunteering its maker when nobody asked.

Three stages, none of which needs a model API key -- every grade in this
recipe is a program, because the behaviour is a string-level property of the
reply and a judge would only add a trust problem to check.

    python run.py --dry-run      # offline end to end, no key, no GPU, no network
    python run.py prep           # build the pool and the two held-out sets
    python run.py audit          # ADA: what each selector keeps, before any GPU
    python run.py analyse        # intervals, noise floor, verdicts -> results.json

Method: Zeng, "Online Data Selection Is Implicit Alignment", arXiv:2607.07023.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import spec

HERE = Path(__file__).parent
OUT = HERE / "out"
DATASET = "while-ai/identity-behavior"
BASE_MODEL = "Qwen/Qwen3-1.7B"
# Share of the pool's tokens each arm is allowed to train on. Every arm gets
# the same number, so the arms differ only in which rows they spend it on.
BUDGET_FRACTION = 0.15
SELECTORS = ("random", "loss", "aas")


def _spec_rewrite(text: str) -> str:
    """Point an identity answer at the spec's maker instead of the retired one.

    The published rows answer with "ZeroProof AI", retired on 2026-09-19.
    The agent's written spec, not the training set, decides what it says, so
    the rows are rewritten before anything trains on them. Longest form
    first, so "ZeroProof AI" becomes the maker rather than the name plus a
    stray "AI".
    """
    text = re.sub(r"zero\s*proof\s*ai", spec.MAKER, text, flags=re.IGNORECASE)
    return re.sub(r"zero\s*proof", spec.NAME, text, flags=re.IGNORECASE)


def prep(limit: int | None = None) -> dict:
    from datasets import load_dataset

    import whileai as wai

    OUT.mkdir(exist_ok=True)
    train = load_dataset(DATASET, "train")["train"]
    ident = load_dataset(DATASET, "eval_identity")["test"]
    leak = load_dataset(DATASET, "eval_leak")["test"]

    rewritten = 0
    pool = []
    for i, row in enumerate(train):
        msgs = []
        for m in row["messages"]:
            new = _spec_rewrite(m["content"])
            rewritten += new != m["content"]
            msgs.append({"role": m["role"], "content": new})
        pool.append(
            {
                "idx": i,
                "kind": row["kind"],
                "messages": msgs,
                "prompt": msgs[0]["content"] if msgs else "",
            }
        )
    if limit:
        pool = pool[:limit]

    holdout_identity = [{"task_id": f"id-{i}", "prompt": p} for i, p in enumerate(ident["prompt"])]
    holdout_leak = [
        {"task_id": f"leak-{i}", "prompt": p, "tier": t, "ask_family": f}
        for i, (p, t, f) in enumerate(zip(leak["prompt"], leak["tier"], leak["ask_family"]))
    ]

    # The holdout must never train. Both held-out sets go in at once: an
    # identity ask and an ordinary request are different distributions, and a
    # row that leaks either one is equally fatal to the comparison.
    clean, report = wai.decontaminate(pool, holdout_identity + holdout_leak, fields=("prompt",))
    kept_idx = {r["idx"] for r in clean}
    dropped = [r for r in pool if r["idx"] not in kept_idx]

    # The detector's precision, measured on replies written by an agent that
    # never had this identity, so every hit would be a false positive. This
    # is the no-key stand-in for judge_trust.
    fp = spec.detector_false_positive_rate([t or "" for t in leak["final_text"]])

    (OUT / "pool.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean)
    )
    (OUT / "holdout_identity.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in holdout_identity)
    )
    (OUT / "holdout_leak.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in holdout_leak)
    )

    summary = {
        "pool_rows_in": len(pool),
        "pool_rows_kept": len(clean),
        "dropped_by_decontaminate": len(dropped),
        "dropped_focused": sum(1 for r in dropped if r["kind"] == "focused"),
        "dropped_control": sum(1 for r in dropped if r["kind"] == "control"),
        "contamination_rate": report.get("contamination_rate"),
        "identity_rows_kept": sum(1 for r in clean if r["kind"] == "focused"),
        "control_rows_kept": sum(1 for r in clean if r["kind"] == "control"),
        "spec_rewrites": rewritten,
        "holdout_identity": len(holdout_identity),
        "holdout_leak": len(holdout_leak),
        "detector_false_positive_rate": fp["false_positive_rate"],
        "detector_checked_on": fp["n"],
        "detector_replies_with_bare_maker_word": fp["replies_containing_the_bare_word"],
        "base_model": BASE_MODEL,
    }
    (OUT / "prep.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def audit() -> dict:
    """ADA over the real pool, using the losses the scoring pass measured."""
    import arm_selectors as sel

    meta = json.loads((OUT / "pool_meta.json").read_text())
    budget = int(sum(r["n_tokens"] for r in meta) * BUDGET_FRACTION)
    table = {}
    for name in SELECTORS:
        kept = sel.select(meta, name, budget)
        table[name] = sel.audit(kept, meta)
    (OUT / "ada.json").write_text(json.dumps({"budget_tokens": budget, "arms": table}, indent=2))
    _print_ada(budget, table)
    return table


def _print_ada(budget: int, table: dict) -> None:
    print(f"\nADA -- attribute mixture of what each selector keeps ({budget:,} tokens each)")
    print(f"{'selector':<10} {'rows':>6} {'identity tokens':>16} {'mean loss':>10} {'tok/row':>8}")
    for name, a in table.items():
        print(
            f"{name:<10} {a['rows']:>6} {a['identity_token_share']:>15.1%} "
            f"{a['mean_loss']:>10.3f} {a['mean_tokens_per_row']:>8.0f}"
        )
    pool_share = next(iter(table.values()))["pool_identity_token_share"]
    print(f"{'(pool)':<10} {'':>6} {pool_share:>15.1%}")


def _rows_for_compare(records: list[dict], metric: str) -> list[dict]:
    """Graded rows in the shape wai.compare reads."""
    out = []
    for r in records:
        out.append(
            {
                "task_id": r["task_id"],
                "reward": float(r[metric]),
                "markers": {
                    "leak": float(r.get("leak", 0.0)),
                    "names_maker": float(r.get("names_maker", 0.0)),
                    "words": float(r.get("words", 0.0)),
                    "names_retired": float(r.get("names_retired", 0.0)),
                },
                "tier": r.get("tier", "n/a"),
            }
        )
    return out


def analyse() -> dict:
    import whileai as wai

    evals = json.loads((OUT / "eval.json").read_text())
    ada = json.loads((OUT / "ada.json").read_text())
    prep_summary = json.loads((OUT / "prep.json").read_text())

    results: dict = {
        "method": {
            "paper": "Online Data Selection Is Implicit Alignment",
            "first_author": "Aoxiong Zeng",
            "arxiv": "2607.07023",
        },
        "base_model": BASE_MODEL,
        "model": "offline",
        "prep": prep_summary,
        "ada": ada,
        "noise_floor": {},
        "arms": {},
    }

    # Noise floor: three passes of the untrained base over the same holdout.
    for split in ("identity", "leak"):
        runs = [evals[f"base_run{i}"][split] for i in (1, 2, 3)]
        floors = {}
        for metric in ("target", "leak", "names_maker", "words"):
            vals = [sum(float(r[metric]) for r in run) / len(run) for run in runs]
            floors[metric] = {
                "runs": vals,
                "run_std": (sum((v - sum(vals) / 3) ** 2 for v in vals) / 2) ** 0.5,
            }
        results["noise_floor"][split] = floors

    # Paired deltas. The base's first pass is the `before` for every arm, so
    # all arms are compared against the same rows in the same order.
    for split, target in (("identity", "names_maker"), ("leak", "target")):
        run_std = {
            m: results["noise_floor"][split][m]["run_std"] for m in ("leak", "names_maker", "words")
        }
        before = _rows_for_compare(evals["base_run1"][split], target)
        for arm in SELECTORS:
            after = _rows_for_compare(evals[arm][split], target)
            rep = wai.compare(
                before,
                after,
                target="pass_at_1",
                markers=["leak", "names_maker", "words"],
                run_std=run_std,
                run_std_runs=3,
            )
            results["arms"].setdefault(arm, {})[split] = json.loads(json.dumps(rep, default=str))
            print(f"\n=== {arm} vs base -- {split} (target {target}) ===")
            print(rep)

    # The line the run exists for: method against the engineer's default.
    for split, target in (("identity", "names_maker"), ("leak", "target")):
        rep = wai.compare(
            _rows_for_compare(evals["loss"][split], target),
            _rows_for_compare(evals["aas"][split], target),
            target="pass_at_1",
            markers=["leak", "names_maker", "words"],
            run_std={
                m: results["noise_floor"][split][m]["run_std"]
                for m in ("leak", "names_maker", "words")
            },
            run_std_runs=3,
        )
        results.setdefault("method_vs_baseline", {})[split] = json.loads(
            json.dumps(rep, default=str)
        )
        print(f"\n=== aas (method) vs loss (baseline) -- {split} ===")
        print(rep)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    return results


def dry_run() -> None:
    """Every offline part, on synthetic rows: no key, no GPU, no network."""
    import arm_selectors as sel

    print("spec:", spec.NAME, "/", spec.MAKER)
    assert spec.leaked("I am a model developed by While.")
    assert not spec.leaked("Let me check that while the sync finishes.")
    assert spec.names_retired("made by ZeroProof AI")
    assert _spec_rewrite("I am ZeroProof, made by ZeroProof AI.") == (
        f"I am {spec.NAME}, made by {spec.MAKER}."
    )
    print("detector and spec rewrite: ok")

    meta = [
        {
            "idx": i,
            "kind": "focused" if i % 5 == 0 else "control",
            # identity rows are short and surprising, control rows long and familiar
            "loss": 3.0 - 0.001 * i if i % 5 == 0 else 1.0 - 0.0001 * i,
            "n_tokens": 40 if i % 5 == 0 else 600,
        }
        for i in range(500)
    ]
    budget = int(sum(r["n_tokens"] for r in meta) * BUDGET_FRACTION)
    table = {n: sel.audit(sel.select(meta, n, budget), meta) for n in SELECTORS}
    _print_ada(budget, table)
    for n in SELECTORS:
        spent = table[n]["tokens"]
        assert spent <= budget, f"{n} overspent the budget"
    assert table["loss"]["identity_token_share"] > table["random"]["identity_token_share"]
    assert (
        abs(table["aas"]["identity_token_share"] - table["aas"]["pool_identity_token_share"]) < 0.02
    ), "aas should hold the pool's mixture"
    print("\nselectors: budget held equal, loss over-selects identity, aas holds the mixture")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", nargs="?", choices=["prep", "audit", "analyse"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.dry_run or not args.stage:
        dry_run()
        return
    if args.stage == "prep":
        prep(args.limit)
    elif args.stage == "audit":
        audit()
    else:
        analyse()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
