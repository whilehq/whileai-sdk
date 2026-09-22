"""Harness and weights: co-optimize the program around the model and the
model, and measure whether both levers together beat either alone.

    python recipe.py --dry-run     # offline: tasks, harness search on scripted candidates, the grid
    python recipe.py --smoke       # Modal: 2 GRPO steps, 16 tasks a split, the live path end to end
    python recipe.py               # the full live run, writes results.json

Three 2026 papers put the harness (instructions, tools, skills, the loop)
and the weights under the same optimizer. HASE (Luo et al., arXiv:2607.03935)
has one Qwen3-8B model either solve the task or edit the harness, in one
multi-turn action space. SIA (Hebbar et al., arXiv:2605.27276) has a
feedback agent update both the scaffold and the weights of a task agent,
and finds that combining the levers beats scaffold iteration alone on all
three of its benchmarks. Prime Agent (Karten et al., arXiv:2608.23552)
moves only the harness: a persistent REPL, and prompts, skills and memory
the agent edits at run time; it trains no weights.

This recipe is the smallest honest version of that question on a small open
coding model and one seeded task set:

  lever 1  the harness: ``candidates/*.py``, each ``harness(model) ->
           wai.Harness`` with a SKILLS.md text in its instructions, an
           optional ``run_python`` tool and a turn cap; searched with the
           Meta-Harness loop (ledger, proposal, gate) imported from
           ``../meta-harness/run.py``
  lever 2  the weights: GRPO with TRL and vLLM colocate on one H100, the
           rollouts under the current harness's instructions, the reward
           ``CodeExec`` on hidden tests

Four arms on the same held-out tasks, one rollout budget each:

  neither  base weights, baseline harness
  harness  base weights, the searched harness
  weights  weights trained under the baseline harness, baseline harness
  both     weights trained under the searched harness, that harness

The claim under test: ``both`` beats ``harness`` and ``weights`` on held-out
pass@1 with a paired interval that excludes zero (``compare_runs``).
``wai.harness.attribute`` over the 2x2 grid says which lever moved the
score. One training seed per arm is "unresolved", never "moved"
(``recipes/papers/check.py``).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import statistics
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import whileai as wai
from whileai.config import provenance
from whileai.simulations.score.stats import compare_runs, task_key

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harnesses
import tasks as task_mod

# The Meta-Harness loop this recipe reuses: ledger shape, proposal, gate.
META_HARNESS = HERE.parent / "meta-harness" / "run.py"

# Defaults. Each is a flag; the README's table names the same numbers.
# The coder variant of the 1.5B Instruct model: the first smoke, on
# Qwen/Qwen2.5-1.5B-Instruct, passed 0, 4 and 2 of 128 rollouts under the
# three harnesses at temperature 1.0, a floor GRPO gets almost no signal from.
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PAPER = "https://arxiv.org/abs/2607.03935"
BOOK = "Tool Use"
METRIC = "pass@1"
K = 4  # rollouts per task at eval: pass@1 averages them, the interval is over tasks
STEPS = 40  # optimizer steps per trained arm
GENERATIONS = 8  # rollouts per prompt: the GRPO group
PROMPTS_PER_STEP = 6  # prompts per optimizer step: 48 rollouts a step
LR = 1e-4  # LoRA GRPO on this base, the filter-metric recipe's setting
BETA = 1e-4  # KL coefficient, SimpleRL-Zoo's for models up to 14B
MAX_COMPLETION = 1024  # completion tokens
LORA_RANK = 32
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)
SEED = 0  # the table, the split, and the scripted stand-ins
TRAIN_SEED = 17  # the adapter init and the sampler, both trained arms
DRY_LIMIT = 16  # tasks per split the dry run plays; the build is always the whole set
SMOKE_LIMIT = 16  # tasks per split in --smoke
SMOKE_STEPS = 2
USD_PER_HOUR = 3.95  # H100 on Modal, modal.com/pricing read 2026-09-20 (recipes/README.md)
ARMS = ("neither", "harness", "weights", "both")
NOT_RUN_DRY = "not run (dry run: no GPU)"


def _meta_harness() -> ModuleType:
    """``recipes/papers/meta-harness/run.py`` as a module: its ``candidates``
    loader, ``score``, ``fmt``, ``worst_rows``, ``propose`` and ``select``
    are the science this recipe reuses rather than forks."""
    if not META_HARNESS.exists():
        raise SystemExit(f"{META_HARNESS} is missing; this recipe imports the Meta-Harness loop")
    spec = importlib.util.spec_from_file_location("meta_harness_run", META_HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["meta_harness_run"] = module
    spec.loader.exec_module(module)
    module.NEXT_NOTE = (  # type: ignore[attr-defined]
        "Then edit the SKILLS text into candidates/{next}.py and run: python recipe.py{flags}"
    )
    return module


def _limit(tasks: list[dict], n: int | None, seed: int) -> list[dict]:
    """The first ``n`` tasks in a seeded order, whole families mixed."""
    if n is None or n >= len(tasks):
        return tasks
    import random

    order = list(tasks)
    random.Random(seed).shuffle(order)
    return sorted(order[:n], key=lambda t: t["scenario_id"])


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def summarize(rows: list[dict]) -> dict[str, Any]:
    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
        "n": p.n_groups,
    }


def _holdout(rows: list[dict], hold_ids: set[str]) -> list[dict]:
    return [r for r in rows if task_key(r) in hold_ids]


# --------------------------------------------------------------------------
# Lever 1: the harness search, in the Meta-Harness ledger shape
# --------------------------------------------------------------------------


def ledger_from_rows(
    mh: ModuleType,
    entries: list[tuple[Path, ModuleType]],
    rows_by_label: dict[str, list[dict]],
    *,
    train_ids: set[str],
    hold_ids: set[str],
    out: Path,
    k: int,
    model: str,
) -> list[dict[str, Any]]:
    """One ledger line per candidate, and ``out/traces/<candidate>/``, the
    way ``meta-harness/run.py`` writes them, so its ``propose`` and
    ``select`` read this recipe's rows unchanged."""
    ledger: list[dict[str, Any]] = []
    for path, module in entries:
        harness = module.harness(model)
        rows = rows_by_label[harness.version]
        for r in rows:
            r["split"] = "holdout" if task_key(r) in hold_ids else "train"
        train = [r for r in rows if r["split"] == "train"]
        hold = [r for r in rows if r["split"] == "holdout"]
        trace_dir = out / "traces" / path.stem
        trace_dir.mkdir(parents=True, exist_ok=True)
        with open(trace_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        worst = mh.worst_rows(train, mh.WORST)
        with open(trace_dir / "worst.jsonl", "w", encoding="utf-8") as fh:
            for r in worst:
                fh.write(json.dumps(r, default=str) + "\n")
        entry = {
            "candidate": path.name,
            "label": harness.version,
            "fingerprint": harness.fingerprint,
            "model": model,
            "k": k,
            "n_tasks": len(train_ids) + len(hold_ids),
            "train": mh.score(train),
            "holdout": mh.score(hold),
            "held_out_models": {},
            "tool_calls": sum(len(r.get("steps") or []) for r in rows),
            "worst": (trace_dir / "worst.jsonl").relative_to(out).as_posix(),
            "rows": (trace_dir / "rows.jsonl").relative_to(out).as_posix(),
        }
        ledger.append(entry)
        print(
            f"{path.stem:<18} train {mh.fmt(entry['train'])}  holdout {mh.fmt(entry['holdout'])}"
            f"  tool calls {entry['tool_calls']}"
        )
    with open(out / "ledger.jsonl", "w", encoding="utf-8") as fh:
        for entry in ledger:
            fh.write(json.dumps(entry, default=str) + "\n")
    print(f"ledger: out/ledger.jsonl ({len(ledger)} candidates, 1 model)")
    return ledger


def search_offline(
    entries: list[tuple[Path, ModuleType]],
    all_tasks: list[dict],
    hold: list[dict],
    *,
    k: int,
    seed: int,
    eval_runs: int,
) -> tuple[dict[str, list[dict]], list[list[dict]]]:
    """Every candidate on the scripted stand-in over every task; then the
    baseline harness on the holdout ``eval_runs`` times for the noise floor."""
    rows_by_label: dict[str, list[dict]] = {}
    for _, module in entries:
        harness = module.harness("scripted")
        assert harness.agent is not None
        rows = harnesses.play(harness.agent, harnesses.spec_of(harness), all_tasks, k=k, seed=seed)
        rows_by_label[harness.version] = harnesses.grade(rows)
    base = entries[0][1].harness("scripted")
    assert base.agent is not None
    hold_ids = {t["scenario_id"] for t in hold}
    base_runs = [_holdout(rows_by_label[base.version], hold_ids)]
    for i in range(1, eval_runs):
        rows = harnesses.play(base.agent, harnesses.spec_of(base), hold, k=k, seed=seed + 1000 + i)
        base_runs.append(harnesses.grade(rows))
    return rows_by_label, base_runs


# --------------------------------------------------------------------------
# The grid: four arms, one holdout, attribution and paired deltas
# --------------------------------------------------------------------------


def report_grid(cells: dict[str, Any], out: Path) -> dict[str, Any]:
    """Print the four cells, attribute over the full grid (or say why not),
    and pair the arms the claim compares. Writes ``out/grid.json``."""
    grid: dict[str, Any] = {}
    for arm in ARMS:
        cell = cells[arm]
        if isinstance(cell, str):
            grid[arm] = {"status": cell}
            print(f"{arm:<8} {cell}")
            continue
        s = summarize(cell["rows"])
        grid[arm] = {
            **s,
            "harness": cell["rows"][0]["harness"]["label"] if cell["rows"] else None,
            "model": cell["rows"][0]["harness"]["model"] if cell["rows"] else None,
            "length_chars": mean_length(cell["rows"]),
            "gpu_minutes": cell.get("gpu_minutes", 0.0),
            "steps": cell.get("steps", 0),
        }
        lo, hi = s["ci"]
        print(
            f"{arm:<8} pass@1 {s['score']:.2f} [{lo:.2f}..{hi:.2f}] over {s['n']} held-out tasks"
            f"  harness={grid[arm]['harness']} model={grid[arm]['model']}"
        )
    rows = [r for arm in ARMS if not isinstance(cells[arm], str) for r in cells[arm]["rows"]]
    try:
        report = wai.harness.attribute(rows)
        print(report)
        grid["attribution"] = {
            "verdict": report["verdict"],
            "sentence": report["sentence"],
            "share_harness": report["share_harness"],
            "share_model": report["share_model"],
            "ci_share_harness": report.get("ci_share_harness"),
            "ci_share_model": report.get("ci_share_model"),
            "cells": report["cells"],
        }
        (out / "attribute.txt").write_text(str(report) + "\n", encoding="utf-8")
    except ValueError as exc:
        reason = str(exc).splitlines()[0]
        print(f"attribution skipped: {reason}")
        grid["attribution"] = {"skipped": reason}
    pairs = [
        ("neither", "harness"),
        ("neither", "weights"),
        ("harness", "both"),
        ("weights", "both"),
    ]
    grid["pairs"] = {}
    for a, b in pairs:
        if isinstance(cells[a], str) or isinstance(cells[b], str):
            grid["pairs"][f"{b}_vs_{a}"] = {"status": "not run"}
            continue
        cmp = compare_runs(cells[a]["rows"], cells[b]["rows"])
        lo, hi = cmp["ci95"]
        grid["pairs"][f"{b}_vs_{a}"] = {
            "delta": cmp["delta"],
            "ci95": [lo, hi],
            "n_paired": cmp["n_paired"],
            "clears_zero": lo > 0,
        }
        print(
            f"{b} vs {a}: {cmp['delta']:+.2f} [{lo:+.2f}, {hi:+.2f}] over {cmp['n_paired']} paired"
            f" tasks -> {'clears zero' if lo > 0 else 'could be chance'}"
        )
    (out / "grid.json").write_text(json.dumps(grid, indent=2, default=str), encoding="utf-8")
    return grid


# --------------------------------------------------------------------------
# results.json in the shape recipes/papers/check.py reads
# --------------------------------------------------------------------------


def write_results(
    cells: dict[str, Any],
    grid: dict[str, Any],
    *,
    base: str,
    base_runs: list[list[dict]],
    k: int,
    decon_dropped: int,
    seed: int,
    pins: dict[str, str],
    gpu: str,
    gpu_minutes: float,
    selected: str,
    gate_passed: bool,
    run_url: str,
) -> dict[str, Any]:
    """``base`` is the untrained model under the baseline harness,
    ``baseline`` the trained-weights arm under the baseline harness, and
    ``recipe`` the arm trained and evaluated under the searched harness: the
    one row check.py's table shows is weights -> both. The full grid and the
    both-vs-harness pair ride along under ``grid``."""
    noise = wai.eval_variance(*base_runs)
    run_std = float(noise["run_std"])
    weights_rows, both_rows = cells["weights"]["rows"], cells["both"]["rows"]
    d = wai.compare(
        weights_rows,
        both_rows,
        target="pass_at_1",
        run_std=run_std,
        run_std_runs=int(noise["n_runs"]),
        train_runs={"before": [weights_rows], "after": [both_rows]},
    )
    print(d)
    results = {
        "recipe": HERE.name,
        "title": "Harness and weights: both levers against either alone",
        "paper": PAPER,
        "book": BOOK,
        "base_model": base,
        "metric": METRIC,
        "n_holdout": len({task_key(r) for r in both_rows}),
        "k": k,
        "arms": {
            "base": {**summarize(base_runs[0]), "steps": 0, "gpu_minutes": 0},
            "baseline": {
                **summarize(weights_rows),
                "steps": cells["weights"]["steps"],
                "gpu_minutes": round(cells["weights"]["gpu_minutes"], 1),
            },
            "recipe": {
                **summarize(both_rows),
                "steps": cells["both"]["steps"],
                "gpu_minutes": round(cells["both"]["gpu_minutes"], 1),
            },
        },
        "delta": {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": "unresolved" if d["target_verdict"] == "unresolved" else d["target_verdict"],
            "noise_band": d.get("noise_band"),
        },
        "checks": {
            "run_std": run_std,
            "run_std_runs": int(noise["n_runs"]),
            "train_seeds": {"baseline": 1, "recipe": 1},
            "decontaminated_dropped": decon_dropped,
            "over_optimized": bool(d.get("over_optimized")),
            "length_before": mean_length(base_runs[0]),
            "length_after": {
                "baseline": mean_length(weights_rows),
                "recipe": mean_length(both_rows),
            },
            "hack_scan_top": cells["both"].get("hack_scan_top", ""),
            "seed": seed,
            "train_seed": TRAIN_SEED,
            "pins": ", ".join(f"{k_} {v}" for k_, v in pins.items()),
            "split": task_mod.SPLIT,
        },
        "grid": {arm: grid[arm] for arm in ARMS},
        "pairs": grid["pairs"],
        "attribution": grid.get("attribution"),
        "searched_harness": selected,
        "gate_passed": gate_passed,
        "gpu": gpu,
        "usd": round(gpu_minutes / 60.0 * USD_PER_HOUR, 2),
        "gpu_minutes": round(gpu_minutes, 1),
        "verified": date.today().isoformat(),
        "whileai": version("whileai"),
        "run_url": run_url,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote results.json")
    return results


# --------------------------------------------------------------------------
# The platform: four versions of one agent, one behavior, the run record
# --------------------------------------------------------------------------


def _examples(rows: list[dict], *, harness: str, trained: bool) -> list[dict[str, Any]]:
    """Every graded row as the platform's row shape: the task, the reply,
    whether the hidden tests passed, and tags the rows page groups by
    (harness, weights, task family)."""
    from whileai.platform import EXAMPLE_TEXT_MAX, EXAMPLE_WHY_MAX

    out: list[dict[str, Any]] = []
    for r in rows:
        ok = float(r.get("reward") or 0.0) >= 1.0
        why = str(r.get("reason") or (r.get("judgment") or {}).get("reason") or "")
        family = str(r.get("family") or str(task_key(r)).split("-")[0])
        out.append(
            {
                "prompt": str(r.get("prompt") or "")[:EXAMPLE_TEXT_MAX],
                "reply": str(r.get("final_text") or "")[:EXAMPLE_TEXT_MAX],
                "ok": ok,
                "why": (why or ("hidden tests passed" if ok else "a hidden test failed"))[
                    :EXAMPLE_WHY_MAX
                ],
                "score": 1.0 if ok else 0.0,
                "tags": {
                    "harness": harness,
                    "weights": "trained" if trained else "base",
                    "task": str(task_key(r)),
                    "family": family,
                },
            }
        )
    return out


def _changed(arm: str, harness: wai.Harness, *, trained: bool, steps: int) -> str:
    """The one line a colleague would write under the iteration."""
    what_h = f"harness `{harness.version}`" + (
        " (bare instructions)"
        if harness.version.startswith("00_")
        else " (a skills text in the instructions)"
    )
    what_w = f"weights trained {steps} GRPO steps on the train split" if trained else "base weights"
    return f"Changed: {what_h}, {what_w}. Arm `{arm}` of the harness x weights grid."


def post_platform(
    cells: dict[str, Any],
    entries_by_label: dict[str, wai.Harness],
    *,
    base: str,
    base_runs: list[list[dict]],
    k: int,
    n_train: int,
    decon_dropped: int,
    pins: dict[str, str],
    gpu: str,
    steps: int,
) -> str:
    """Post the four arms as versions of one tracked agent when
    ``WHILEAI_API_KEY`` is set. Never fails the recipe."""
    if not os.environ.get("WHILEAI_API_KEY"):
        print("platform: WHILEAI_API_KEY not set, nothing posted")
        return ""
    try:
        from whileai.platform import Data, EvalSetup, Optimizer, Provenance, RunRecord, track

        noise = wai.eval_variance(*base_runs)
        tracked = track("harness-and-weights", model=base)
        tracked.behavior(
            "quant_code",
            n=len({task_key(r) for r in base_runs[0]}),
            contamination=decon_dropped,
            reward_is_judge=False,
            graded_by="program",
            test_version=f"quant-code-{task_mod.SPLIT}-{len(base_runs[0])}",
            description="pass@1 on held-out quant coding tasks, CodeExec on hidden tests",
            rubric=(
                "A task passes when the model's Python function, run on the seeded price "
                "table in a fresh interpreter, satisfies every hidden assert (CodeExec). "
                "No judge: the tests are the grader."
            ),
        )
        tracked.noise_floor("quant_code", *base_runs)
        url = ""
        for arm in ARMS:
            cell = cells[arm]
            harness = entries_by_label[cell["rows"][0]["harness"]["label"]]
            trained = arm in ("weights", "both")
            record = RunRecord(
                data=Data(
                    train="seeded quant tasks (tasks.py), train split",
                    n_train=n_train,
                    holdout="seeded quant tasks, held-out families",
                    n_holdout=len({task_key(r) for r in cell["rows"]}),
                    decontaminated_dropped=decon_dropped,
                ),
                optimizer=Optimizer(
                    loss_type="dapo",
                    lr=LR,
                    beta=BETA,
                    epsilon=0.2,
                    num_generations=GENERATIONS,
                    seed=TRAIN_SEED,
                )
                if trained
                else None,
                eval=EvalSetup(
                    metric=METRIC,
                    k=k,
                    run_std=float(noise["run_std"]),
                    run_std_runs=int(noise["n_runs"]),
                    reader="CodeExec",
                    gpu=gpu,
                    gpu_hours=round(cell.get("gpu_minutes", 0.0) / 60, 3),
                    replies=len(cell["rows"]),
                ),
                provenance=Provenance(
                    # the grid axes first: attribute() and the platform read
                    # pins.harness and pins.model off every run
                    pins={
                        **{k_: str(v) for k_, v in pins.items()},
                        "harness": harness.fingerprint,
                        "model": "trained" if trained else "base",
                    },
                    recipe="recipes/papers/harness-and-weights",
                    paper="2607.03935",
                ),
            )
            prun = tracked.run(
                arm,
                base=base,
                method="grpo" if trained else "none",
                targets=["quant_code"],
                trained_on=["quant tasks train split"] if trained else [],
                gpu=gpu,
                harness=harness.pin(),
                record=record,
            )
            p = wai.pass_at(cell["rows"])
            lo, hi = p.ci95 or (p.pass_at_1, p.pass_at_1)
            prun.score(
                "quant_code",
                round(100 * p.pass_at_1, 1),
                ci=round(50 * (hi - lo), 1),
                n=p.n_groups,
                rows=_examples(cell["rows"], harness=harness.version, trained=trained),
            )
            prun.note(_changed(arm, harness, trained=trained, steps=steps))
            minutes = float(cell.get("gpu_minutes", 0.0))
            prun.finish(
                "evaluated",
                hours=round(minutes / 60, 3),
                gpu=gpu,
                cost_usd=round(minutes / 60 * USD_PER_HOUR, 2),
                steps=steps if trained else 0,
            )
            url = prun.url
            print(f"platform: {arm} -> {prun.url}")
        return url
    except Exception as exc:  # the platform is not the experiment
        print(f"platform: skipped ({type(exc).__name__}: {exc})")
        return ""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="offline: scripted candidates, no GPU")
    p.add_argument("--smoke", action="store_true", help="Modal: 2 steps, 16 tasks a split")
    p.add_argument("--k", type=int, default=K, help="rollouts per task at eval")
    p.add_argument("--steps", type=int, default=STEPS, help="optimizer steps per trained arm")
    p.add_argument("--generations", type=int, default=GENERATIONS, help="rollouts per prompt")
    p.add_argument("--prompts-per-step", type=int, default=PROMPTS_PER_STEP)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA, help="KL coefficient")
    p.add_argument("--max-completion", type=int, default=MAX_COMPLETION)
    p.add_argument("--lora-rank", type=int, default=LORA_RANK)
    p.add_argument("--base", default=BASE_MODEL, help="the base model, a Hub id")
    p.add_argument("--eval-runs", type=int, default=EVAL_RUNS, help="base re-runs, noise floor")
    p.add_argument("--seed", type=int, default=SEED, help="the table, the split, the stand-ins")
    p.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    p.add_argument("--holdout", type=float, default=task_mod.HOLDOUT, help="share held out")
    p.add_argument("--split", choices=["family", "task"], default=task_mod.SPLIT)
    p.add_argument("--limit", type=int, default=None, help="tasks per split to play")
    p.add_argument("--candidates", default="candidates", help="folder of candidate files")
    p.add_argument("--out", default="out", help="tasks, ledger, traces, grid")
    p.add_argument("--fresh", action="store_true", help="drop out/ first")
    p.add_argument("--reuse", action="store_true", help="take stages from .cache/ when present")
    return p


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    args = build_parser().parse_args(argv)
    if args.dry_run and args.smoke:
        raise SystemExit("--dry-run is offline and --smoke is Modal; pick one")
    out = HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    mh = _meta_harness()

    # 1. Tasks: build, split, decontaminate, freeze.
    all_tasks = task_mod.build()
    task_mod.SPLIT = args.split
    train, hold = task_mod.split(all_tasks, args.holdout, args.seed, by=args.split)
    train, decon = wai.decontaminate(train, against=hold)
    task_mod.write(all_tasks, out / "tasks.jsonl")
    print(
        f"tasks: {len(all_tasks)} in {len(task_mod.TEMPLATES)} families, split by {args.split}: "
        f"{len(train)} train, {len(hold)} holdout; decontaminate dropped "
        f"{decon['n_contaminated']} train rows"
    )
    limit = args.limit
    if limit is None and args.dry_run:
        limit = DRY_LIMIT
    if limit is None and args.smoke:
        limit = SMOKE_LIMIT
    train_play = _limit(train, limit, args.seed)
    hold_play = _limit(hold, limit, args.seed)
    played = train_play + hold_play
    train_ids = {t["scenario_id"] for t in train_play}
    hold_ids = {t["scenario_id"] for t in hold_play}
    if limit is not None:
        print(f"playing {len(train_play)} train and {len(hold_play)} holdout tasks (--limit)")

    # 2. Candidates.
    folder = (
        HERE / args.candidates if not Path(args.candidates).is_absolute() else Path(args.candidates)
    )
    entries = mh.candidates(folder)
    by_label = {m.harness("base").version: m.harness("base") for _, m in entries}
    baseline_label = entries[0][1].harness("base").version

    steps = SMOKE_STEPS if args.smoke else args.steps
    gpu = os.environ.get("WAI_RECIPE_GPU", "H100")
    gpu_minutes = 0.0
    pins: dict[str, str] = {}
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)

    # 3. Lever 1: the harness search on the base weights. The Modal app stays
    # open across the search, the gate and the training arms.
    stack = contextlib.ExitStack()
    weights_call: Any = None
    specs = [harnesses.spec_of(by_label[label]) for label in by_label]
    with stack:
        if args.dry_run:
            rows_by_label, base_runs = search_offline(
                entries, played, hold_play, k=args.k, seed=args.seed, eval_runs=args.eval_runs
            )
        else:
            all_cached = args.reuse and all(
                (cache / name).exists() for name in ("search.json", "weights.json", "both.json")
            )
            if all_cached:
                # every paid stage is on disk: no Modal app, no image, no GPU;
                # the grid and the platform post read the cache
                print("reuse: every stage cached, Modal not opened")
                modal_run = None  # the lambdas below are never called
            else:
                import modal
                import modal_run

                stack.enter_context(modal.enable_output())
                stack.enter_context(modal_run.app.run())
            cached_w = cache / "weights.json"
            if not (args.reuse and cached_w.exists()):
                # The weights arm needs only the baseline harness, so it
                # trains while the search runs.
                weights_call = modal_run.train_arm.spawn(
                    "weights",
                    specs[0],
                    train_play,
                    hold_play,
                    f"haw-weights-{date.today().isoformat()}",
                    **_train_kwargs(args, steps),
                )
            search_out = _stage(
                cache / "search.json",
                args.reuse,
                lambda: modal_run.search.remote(
                    specs,
                    played,
                    sorted(hold_ids),
                    k=args.k,
                    seed=args.seed,
                    eval_runs=args.eval_runs,
                    max_tokens=args.max_completion,
                    base_model=args.base,
                ),
            )
            rows_by_label = search_out["rows"]
            base_runs = search_out["base_runs"]
            gpu_minutes += search_out["gpu_minutes"]
            pins = search_out["pins"]
            print(f"search: {search_out['gpu_minutes']:.1f} GPU minutes")

        ledger = ledger_from_rows(
            mh,
            entries,
            rows_by_label,
            train_ids=train_ids,
            hold_ids=hold_ids,
            out=out,
            k=args.k,
            model="base",
        )
        flags = " --dry-run" if args.dry_run else " --smoke" if args.smoke else ""
        mh.propose(ledger, entries, out, flags=flags)
        # cost_margin=None: the cost per rollout is reported, not gated; this
        # recipe asks which lever moved the score, not whether the pick is cheap
        verdict = mh.select(ledger, models=["base"], out=out, cost_margin=None)
        picked = verdict.get("selected") or verdict.get("best_on_train")
        selected_label = by_label[Path(picked).stem].version if picked else baseline_label
        gate_passed = verdict.get("selected") is not None
        print(
            f"searched harness: {selected_label}"
            + ("" if gate_passed else " (best on train; the gate did not pass on the holdout)")
        )

        # 4. Lever 2: the weights, under each harness.
        cells: dict[str, Any] = {
            "neither": {"rows": base_runs[0], "gpu_minutes": 0.0, "steps": 0},
            "harness": {
                "rows": _holdout(rows_by_label[selected_label], hold_ids),
                "gpu_minutes": 0.0,
                "steps": 0,
            },
        }
        if args.dry_run:
            cells["weights"] = NOT_RUN_DRY
            cells["both"] = NOT_RUN_DRY
        else:
            cells["weights"] = _stage(
                cache / "weights.json", args.reuse, lambda: weights_call.get()
            )
            cells["both"] = _stage(
                cache / "both.json",
                args.reuse,
                lambda: modal_run.train_arm.remote(
                    "both",
                    harnesses.spec_of(by_label[selected_label]),
                    train_play,
                    hold_play,
                    f"haw-both-{date.today().isoformat()}",
                    **_train_kwargs(args, steps),
                ),
            )
            gpu_minutes += cells["weights"]["gpu_minutes"] + cells["both"]["gpu_minutes"]
            pins = pins or cells["weights"]["pins"]

    # 5. The grid.
    grid = report_grid(cells, out)

    if args.dry_run:
        dry = {
            "tasks": {
                "n": len(all_tasks),
                "families": len(task_mod.TEMPLATES),
                "split": args.split,
                "train": len(train),
                "holdout": len(hold),
                "decontaminated_dropped": int(decon["n_contaminated"]),
                "played": {"train": len(train_play), "holdout": len(hold_play)},
            },
            "candidates": [e["candidate"] for e in ledger],
            "searched_harness": selected_label,
            "gate_passed": gate_passed,
            "grid": {arm: grid[arm] for arm in ARMS},
            "attribution": grid["attribution"],
            "pairs": grid["pairs"],
        }
        (out / "dry_run.json").write_text(json.dumps(dry, indent=2, default=str), encoding="utf-8")
        print("wrote out/dry_run.json; no results.json from a dry run")
        return 0

    usd = gpu_minutes / 60.0 * USD_PER_HOUR
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${usd:.2f} on {gpu}")
    if args.smoke:
        smoke = {
            "steps": steps,
            "tasks_played": {"train": len(train_play), "holdout": len(hold_play)},
            "grid": {arm: grid[arm] for arm in ARMS},
            "gpu_minutes": round(gpu_minutes, 1),
            "usd": round(usd, 2),
            "pins": pins,
            "date": date.today().isoformat(),
        }
        (out / "smoke.json").write_text(json.dumps(smoke, indent=2, default=str), encoding="utf-8")
        print("wrote out/smoke.json; a smoke run claims no number and writes no results.json")
        return 0

    run_url = post_platform(
        cells,
        by_label,
        base=args.base,
        base_runs=base_runs,
        k=args.k,
        n_train=len(train_play),
        decon_dropped=int(decon["n_contaminated"]),
        pins=pins,
        gpu=gpu,
        steps=steps,
    )
    write_results(
        cells,
        grid,
        base=args.base,
        base_runs=base_runs,
        k=args.k,
        decon_dropped=int(decon["n_contaminated"]),
        seed=args.seed,
        pins=pins,
        gpu=gpu,
        gpu_minutes=gpu_minutes,
        selected=selected_label,
        gate_passed=gate_passed,
        run_url=run_url,
    )
    return 0


def _stage(cached: Path, reuse: bool, run: Any) -> dict[str, Any]:
    """One paid stage: from ``.cache/<stage>.json`` under ``--reuse`` when
    it is there, else run it and write it the moment it returns, so a crash
    in a later stage never costs an earlier one."""
    if reuse and cached.exists():
        print(f"{cached.stem}: reused .cache/{cached.name}")
        return json.loads(cached.read_text(encoding="utf-8"))
    out = run()
    cached.write_text(json.dumps(out), encoding="utf-8")
    return out


def _train_kwargs(args: argparse.Namespace, steps: int) -> dict[str, Any]:
    return {
        "base_model": args.base,
        "steps": steps,
        "num_generations": args.generations,
        "prompts_per_step": args.prompts_per_step,
        "learning_rate": args.lr,
        "beta": args.beta,
        "max_completion_length": args.max_completion,
        "lora_rank": args.lora_rank,
        "k": args.k,
        "seed": args.train_seed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
