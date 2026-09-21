"""Meta-Harness: an outer loop that searches over harness code, scores every
candidate on one frozen task set, and gates the pick on held-out tasks and
held-out models.

    python run.py --dry-run                  # offline: scripted candidates, no key
    python run.py --dry-run --propose        # and write out/proposal.md for the proposer
    python run.py --dry-run --select         # and apply the gate
    python run.py --models openai:gpt-4.1-mini,anthropic:claude-haiku-4-5  # the live run

Lee, Nair, Zhang, Lee, Khattab and Finn 2026 (Meta-Harness, arXiv:2603.28052)
put the harness, the code around the model, under search: an agentic
proposer reads the source, the scores and the execution traces of every
prior candidate through a filesystem and writes the next one; candidates
are scored on a train split, and the pick is checked on held-out tasks and
held-out models. This file is the loop's inner half. ``candidates/`` holds
one Python file per candidate, each defining ``harness(model) ->
wai.Harness``; the proposer (you, or the coding agent running
``skills/harness-search``) writes the next file.

What one run does:

1. Load every ``candidates/*.py``, in name order. The first is the baseline.
2. Freeze the task set once: the baseline draws it (``wai.simulate(...,
   mode="rl", repeats=k)``) and saves ``out/tasks.jsonl``; every other
   candidate and every model replays it with ``tasks=`` so the asks match.
3. Grade every row with one judge (a program by default, ``--judge`` names
   a model), split tasks into train and holdout by seed, and score each
   split with ``pass_at``: pass@1 with its interval, over tasks.
4. Write ``out/ledger.jsonl`` (one line per candidate: file, fingerprint,
   model, train and holdout pass@1 with intervals, task counts, the path of
   its worst rows) and ``out/traces/<candidate>/`` (every row, and the
   five worst on the train split).
5. ``--propose`` writes ``out/proposal.md``: the paper's filesystem
   interface, every candidate's source, score and worst rows, and the one
   instruction to write the next file.
6. ``--select`` applies the gate: the best candidate on the train split
   must beat the baseline on the holdout with an interval that excludes
   zero (``compare_runs``), and on at least one held-out model when
   ``--models`` names more than one. ``wai.harness.attribute`` on the
   candidate x model grid says whether the gain is the harness or the model.

Lambert 2025, chapter Evaluation: the train split picks, the holdout
decides, and one number without its interval is not a result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import whileai as wai
from whileai.config import provenance
from whileai.simulations.score.stats import compare_runs, task_key

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common

# Defaults. Each is a flag; the README's table names the same numbers.
K = 4  # rollouts per task: pass@1 averages them, the interval is over tasks
BUDGET = 24  # tasks in the frozen set; half train, half holdout at HOLDOUT
HOLDOUT = 0.5  # share of tasks held out; the train split picks, the holdout decides
MODELS = "scripted,scripted-b"  # the search model first, then the held-out models
JUDGE = "program"  # common.judge; a provider:model string builds wai.Judge(RUBRIC)
SEED = 0  # the draw of the frozen set and of the split
WORST = 5  # rows per candidate in proposal.md, the paper's trace window
NEXT_NOTE = "Then write candidates/{next}.py and run: python run.py{flags} --propose --select"


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import candidate {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "harness", None)):
        raise SystemExit(f"{path.name}: a candidate defines harness(model) -> wai.Harness")
    return module


def candidates(folder: Path) -> list[tuple[Path, ModuleType]]:
    files = sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"no candidates in {folder}; the baseline is candidates/00_baseline.py")
    return [(p, _load(p)) for p in files]


def _safe(name: str) -> str:
    return name.replace(":", "-").replace("/", "-")


def _judge(spec: str) -> Any:
    if spec == "program":
        return common.judge
    return wai.Judge(common.RUBRIC, model=spec)


def simulate(harness: wai.Harness, *, tasks: Path | None, k: int, budget: int, seed: int) -> Any:
    """One candidate on one model. The first call draws the frozen set from
    the seeds with the offline writer; every later call replays it."""
    kw: dict[str, Any] = {}
    if tasks is not None and tasks.exists():
        kw["tasks"] = str(tasks)
    else:
        kw.update(seeds=common.SEEDS, situations=budget, mode="rl", budget=budget * k)
    return wai.simulate(
        harness,
        simulator=False,  # the template writer: offline, deterministic, no key
        repeats=k,
        repeat_policy="fixed",
        reproducible=True,
        seed=seed,
        concurrency=1,
        **kw,
    )


def split(keys: list[str], holdout: float, seed: int) -> tuple[set[str], set[str]]:
    order = sorted(keys)
    random.Random(seed).shuffle(order)
    n_hold = round(len(order) * holdout)
    return set(order[n_hold:]), set(order[:n_hold])


def score(rows: list[dict]) -> dict[str, Any]:
    pa = wai.pass_at(rows)
    return {
        "pass_at_1": pa.pass_at_1,
        "ci95": list(pa.ci95) if pa.ci95 else None,
        "n_tasks": pa.n_groups,
        "note": pa.note or None,
    }


def fmt(s: dict[str, Any]) -> str:
    if s["pass_at_1"] is None:
        return "n/a"
    ci = f" [{s['ci95'][0]:.2f}..{s['ci95'][1]:.2f}]" if s["ci95"] else ""
    return f"{s['pass_at_1']:.2f}{ci}"


def worst_rows(rows: list[dict], n: int) -> list[dict]:
    failed = [r for r in rows if not r.get("reward")]
    return [
        {
            "task": task_key(r),
            "prompt": r.get("prompt"),
            "reply": r.get("final_text"),
            "why": r.get("reason"),
            "steps": r.get("steps"),
        }
        for r in failed[:n]
    ]


def evaluate(
    entries: list[tuple[Path, ModuleType]],
    *,
    models: list[str],
    judge: Any,
    out: Path,
    k: int,
    budget: int,
    holdout: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Every candidate on every model over the same frozen tasks. Returns the
    ledger, one entry per candidate, and writes the traces."""
    tasks = out / "tasks.jsonl"
    search_model = models[0]
    ledger: list[dict[str, Any]] = []
    train_keys: set[str] = set()
    hold_keys: set[str] = set()
    for path, module in entries:
        rows_all: list[dict] = []
        per_model: dict[str, list[dict]] = {}
        for model in models:
            harness = module.harness(model)
            data = simulate(harness, tasks=tasks, k=k, budget=budget, seed=seed)
            if not tasks.exists():
                data.save(str(tasks))
                train_keys, hold_keys = split(
                    sorted({task_key(r) for r in data.rows()}), holdout, seed
                )
            scored = data.grade(judge=judge)
            rows = [dict(r) for r in scored.rows]
            for r in rows:
                r["split"] = "holdout" if task_key(r) in hold_keys else "train"
            per_model[model] = rows
            rows_all.extend(rows)
        if not train_keys and not hold_keys:  # tasks.jsonl came from an earlier run
            train_keys, hold_keys = split(sorted({task_key(r) for r in rows_all}), holdout, seed)
            for r in rows_all:
                r["split"] = "holdout" if task_key(r) in hold_keys else "train"
        first = per_model[search_model]
        train = [r for r in first if r["split"] == "train"]
        hold = [r for r in first if r["split"] == "holdout"]
        trace_dir = out / "traces" / path.stem
        trace_dir.mkdir(parents=True, exist_ok=True)
        with open(trace_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
            for r in rows_all:
                fh.write(json.dumps(r, default=str) + "\n")
        worst = worst_rows(train, WORST)
        with open(trace_dir / "worst.jsonl", "w", encoding="utf-8") as fh:
            for r in worst:
                fh.write(json.dumps(r, default=str) + "\n")
        harness = module.harness(search_model)
        entry = {
            "candidate": path.name,
            "label": harness.version,
            "fingerprint": harness.fingerprint,
            "model": search_model,
            "k": k,
            "n_tasks": len(train_keys) + len(hold_keys),
            "train": score(train),
            "holdout": score(hold),
            "held_out_models": {
                m: score([r for r in per_model[m] if r["split"] == "holdout"]) for m in models[1:]
            },
            "worst": (trace_dir / "worst.jsonl").relative_to(out).as_posix(),
            "rows": (trace_dir / "rows.jsonl").relative_to(out).as_posix(),
        }
        ledger.append(entry)
        extra = "".join(f"  {m} {fmt(s)}" for m, s in entry["held_out_models"].items())
        print(
            f"{path.stem:<18} train {fmt(entry['train'])}  holdout {fmt(entry['holdout'])}{extra}"
        )
    with open(out / "ledger.jsonl", "w", encoding="utf-8") as fh:
        for entry in ledger:
            fh.write(json.dumps(entry, default=str) + "\n")
    print(f"ledger: {_show(out / 'ledger.jsonl')} ({len(ledger)} candidates, {len(models)} models)")
    return ledger


def _show(path: Path) -> str:
    """A path as the README quotes it: relative to the recipe when inside it."""
    try:
        return path.relative_to(HERE).as_posix()
    except ValueError:
        return str(path)


def _rows(out: Path, entry: dict[str, Any], *, model: str, split_name: str) -> list[dict]:
    with open(out / entry["rows"], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r["harness"]["model"] == model and r.get("split") == split_name]


def propose(
    ledger: list[dict[str, Any]],
    entries: list[tuple[Path, ModuleType]],
    out: Path,
    *,
    flags: str = "",
) -> Path:
    """The proposer's view of the loop, as one file: every candidate's
    source, its train score with interval, its worst rows, and the one
    instruction. Nothing here is compressed, which is the paper's point."""
    next_n = len(ledger)
    lines = [
        "# Proposal",
        "",
        "You are the proposer in a Meta-Harness loop (Lee et al. 2026, arXiv:2603.28052).",
        "Below is every candidate so far: its source, its pass@1 on the train split with a",
        "95% interval over tasks, and its worst rows. Change what the worst rows say is",
        f"wrong, and only that. {NEXT_NOTE.format(next=f'{next_n:02d}_<name>', flags=flags)}",
        "",
    ]
    for (path, _), entry in zip(entries, ledger):
        lines += [
            f"## {entry['candidate']}",
            "",
            f"train pass@1 {fmt(entry['train'])} on {entry['train']['n_tasks']} tasks; "
            f"holdout {fmt(entry['holdout'])}; fingerprint {entry['fingerprint']}",
            "",
            "```python",
            path.read_text(encoding="utf-8").rstrip(),
            "```",
            "",
            "Worst rows:",
            "",
        ]
        with open(out / entry["worst"], encoding="utf-8") as fh:
            worst = [json.loads(line) for line in fh if line.strip()]
        if not worst:
            lines.append("- none failed on the train split")
        for w in worst:
            lines.append(f"- ask: {w['prompt']}")
            lines.append(f"  reply: {w['reply']}")
            lines.append(f"  why: {w['why']}")
        lines.append("")
    dest = out / "proposal.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"proposal: {_show(dest)}")
    return dest


def select(ledger: list[dict[str, Any]], *, models: list[str], out: Path) -> dict[str, Any]:
    """The gate. The train split picks the candidate; the holdout on the
    search model, and on at least one held-out model, has to agree with an
    interval that excludes zero. Then attribution over the grid."""
    baseline = ledger[0]
    best = max(ledger[1:], key=lambda e: e["train"]["pass_at_1"] or 0.0, default=None)
    verdict: dict[str, Any] = {"baseline": baseline["candidate"], "selected": None}
    if best is None:
        verdict["reason"] = "only the baseline has run; write a candidate"
        print("select:", verdict["reason"])
        (out / "selected.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        return verdict
    verdict["best_on_train"] = best["candidate"]
    checks: dict[str, Any] = {}
    for model in models:
        cmp = compare_runs(
            _rows(out, baseline, model=model, split_name="holdout"),
            _rows(out, best, model=model, split_name="holdout"),
        )
        lo, hi = cmp["ci95"]
        checks[model] = {
            "delta": cmp["delta"],
            "ci95": [lo, hi],
            "n_paired": cmp["n_paired"],
            "clears": lo > 0,
        }
        print(
            f"holdout on {model}: {best['candidate']} vs {baseline['candidate']} "
            f"{cmp['delta']:+.2f} [{lo:+.2f}, {hi:+.2f}] over {cmp['n_paired']} paired tasks"
            f" -> {'clears zero' if lo > 0 else 'could be chance'}"
        )
    verdict["checks"] = checks
    on_search = checks[models[0]]["clears"]
    on_held_out = any(c["clears"] for m, c in checks.items() if m != models[0])
    passed = on_search and (on_held_out or len(models) == 1)
    if passed:
        verdict["selected"] = best["candidate"]
        verdict["reason"] = f"{best['candidate']} beats the baseline on the holdout" + (
            " and on a held-out model" if len(models) > 1 else ""
        )
    else:
        verdict["reason"] = (
            f"{best['candidate']} is best on train but does not clear the baseline "
            + ("on the holdout" if not on_search else "on any held-out model")
            + "; write the next candidate"
        )
    if len(models) > 1 and len(ledger) > 1:
        grid = [
            r for e in ledger for m in models for r in _rows(out, e, model=m, split_name="holdout")
        ]
        report = wai.harness.attribute(grid)
        print(report)
        verdict["attribution"] = {
            "verdict": report["verdict"],
            "share_harness": report["share_harness"],
            "share_model": report["share_model"],
        }
    print("select:", verdict["reason"])
    (out / "selected.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=int, default=K, help="rollouts per task")
    p.add_argument("--budget", type=int, default=BUDGET, help="tasks in the frozen set")
    p.add_argument("--holdout", type=float, default=HOLDOUT, help="share of tasks held out")
    p.add_argument(
        "--models",
        default=MODELS,
        help="comma list: the search model first, then held-out models (provider:model)",
    )
    p.add_argument("--judge", default=JUDGE, help="'program' or a provider:model for wai.Judge")
    p.add_argument("--seed", type=int, default=SEED, help="the draw of the tasks and the split")
    p.add_argument("--candidates", default="candidates", help="folder of candidate files")
    p.add_argument("--out", default="out", help="ledger, traces, proposal, selection")
    p.add_argument("--propose", action="store_true", help="write out/proposal.md")
    p.add_argument("--select", action="store_true", help="apply the gate, write selected.json")
    p.add_argument("--fresh", action="store_true", help="drop out/ first: a new frozen set")
    p.add_argument("--dry-run", action="store_true", help="offline: scripted models, no key")
    return p


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    args = build_parser().parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    scripted = [m for m in models if m.startswith("scripted")]
    if args.dry_run:
        models = [m if m.startswith("scripted") else f"scripted-{_safe(m)}" for m in models]
    elif scripted:
        print(
            f"--models names {', '.join(scripted)}: scripted models run offline only. "
            "Pass --dry-run, or --models provider:model,... for the live run "
            "(the provider's key from the environment).",
            file=sys.stderr,
        )
        return 2
    out = (HERE / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    folder = (
        (HERE / args.candidates)
        if not Path(args.candidates).is_absolute()
        else Path(args.candidates)
    )
    entries = candidates(folder)
    ledger = evaluate(
        entries,
        models=models,
        judge=_judge(args.judge),
        out=out,
        k=args.k,
        budget=args.budget,
        holdout=args.holdout,
        seed=args.seed,
    )
    if args.propose:
        propose(ledger, entries, out, flags=" --dry-run" if args.dry_run else "")
    if args.select:
        select(ledger, models=models, out=out)
    if not args.propose and not args.select:
        print(f"Next: python run.py{' --dry-run' if args.dry_run else ''} --propose --select")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
