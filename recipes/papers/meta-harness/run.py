"""Meta-Harness: an outer loop that searches over harness code, scores every
candidate on one frozen task set, and gates the pick on held-out tasks and
held-out models.

    python run.py --dry-run                  # offline: scripted candidates, no key
    python run.py --dry-run --propose        # and write out/proposal.md for the proposer
    python run.py --dry-run --select         # and apply the gate
    python run.py --models openai:gpt-4.1-mini,anthropic:claude-haiku-4-5  # the live run
    python run.py --traces traces.jsonl --propose --select    # the frozen set is production traffic

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
   With ``--traces``, the asks are production traffic instead: one task
   per distinct prompt in the file (a JSONL ``wai.load_traces`` reads, or
   an OTLP JSON batch ``wai.rows_from_otel`` reads).
3. Grade every row with one judge (a program by default, ``--judge`` names
   a model), split tasks into train and holdout, and score each split
   with ``pass_at``: pass@1 with its interval, over tasks. The split is by
   seed for a drawn set and by day for traces: the latest days are the
   holdout, so the proposer never reads a row from the days that decide.
   Train prompts that overlap a holdout prompt (``wai.decontaminate``) are
   dropped from the proposer's window and counted.
4. Write ``out/ledger.jsonl`` (one line per candidate: file, fingerprint,
   model, train and holdout pass@1 with intervals, task counts, cost per
   rollout in tokens and tool calls, the path of its worst rows) and
   ``out/traces/<candidate>/`` (every row, and the five worst on the
   train split).
5. ``--propose`` writes ``out/proposal.md``: the paper's filesystem
   interface, every candidate's source, score and worst rows, and the one
   instruction to write the next file.
6. ``--select`` applies the gate. The pick is the candidate that leads
   the most train tasks (per task, the best pass rate across candidates;
   Agrawal et al. 2025 (GEPA), arXiv:2507.19457, select on the per-task
   frontier so a mean gained by regressing a subset does not win), ties
   by train mean. It must beat the baseline on the holdout with an
   interval that excludes zero (``compare_runs``), on at least one
   held-out model when ``--models`` names more than one, and at no more
   than ``--cost-margin`` above the baseline's cost per rollout (Wang et
   al. 2026, arXiv:2607.12227: at a matched budget, harness evolution
   lost to spending the same compute on more samples of the baseline, so
   a candidate that spends more per task is a frontier point, not a pick,
   until the cost is accepted). The verdict also counts the holdout tasks
   the baseline passed and the pick failed. ``wai.harness.attribute`` on
   the candidate x model grid says whether the gain is the harness or the
   model.

Lambert 2025, chapter Evaluation: the train split picks, the holdout
decides, and one number without its interval is not a result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import whileai as wai
from whileai.config import provenance
from whileai.harness import _model_name
from whileai.simulations import load_traces, rows_from_otel
from whileai.simulations.score.hygiene import tool_calls
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
COST_MARGIN = 0.0  # how much more per rollout than the baseline a pick may cost; 0 = matched
CONCURRENCY = 8  # rollouts in flight on a live model; scripted models run one at a time
COST_EPS = 1e-9  # float slack so a pick at exactly the baseline's cost passes margin 0
NANOS_PER_SECOND = 1e9  # an OTLP span's start_time_unix_nano is in nanoseconds
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


def load_production(path: Path, *, budget: int, seed: int) -> list[dict]:
    """Production traffic as tasks: one per distinct prompt, ``task_id``
    from the prompt so every candidate and every model answer the same
    asks, ``ts`` kept for the day split. A JSONL of traces, or an OTLP JSON
    batch (``resourceSpans``), which ``wai.rows_from_otel`` groups into
    conversations. At most ``budget`` tasks, drawn by ``seed``."""
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{") and "resourceSpans" in text[:4096]:
        rows = rows_from_otel(json.loads(text))
    else:
        rows = load_traces(str(path))
    by_prompt: dict[str, dict] = {}
    for r in rows:
        prompt = str(r.get("prompt") or "").strip()
        if prompt and prompt not in by_prompt:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            task = {"prompt": prompt, "task_id": "task_" + digest}
            if r.get("ts") is not None:
                task["ts"] = r["ts"]
            by_prompt[prompt] = task
    tasks = list(by_prompt.values())
    if not tasks:
        raise SystemExit(f"{path}: no prompts; the file needs rows wai.load_traces reads")
    if len(tasks) > budget:
        tasks = random.Random(seed).sample(tasks, budget)
    return sorted(tasks, key=lambda t: t["task_id"])


def _day(ts: Any) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):  # unix nanos from an OTLP span
        return (
            datetime.fromtimestamp(float(ts) / NANOS_PER_SECOND, tz=timezone.utc).date().isoformat()
        )
    return str(ts)[:10]


def split_by_day(tasks: list[dict], holdout: float, seed: int) -> tuple[set[str], set[str], str]:
    """The holdout is the latest days, whole days, until it holds at least
    ``holdout`` of the tasks; the proposer never reads a row from them. With
    no timestamp on every task the split falls back to the seed."""
    days = {t["task_id"]: _day(t.get("ts")) for t in tasks}
    if any(d is None for d in days.values()):
        train, hold = split(sorted(days), holdout, seed)
        return train, hold, f"no timestamp on every task; split by seed {seed}"
    hold: set[str] = set()
    picked: list[str] = []
    for day in sorted(set(days.values()), reverse=True):
        if picked and len(hold) >= round(len(tasks) * holdout):
            break
        picked.append(day)
        hold |= {key for key, d in days.items() if d == day}
    train = set(days) - hold
    return train, hold, f"holdout is {min(picked)} and later: {len(hold)} of {len(tasks)} tasks"


def cost(rows: list[dict]) -> dict[str, float | None]:
    """Cost per rollout: tokens when every row carries ``usage`` (a real
    model's rows do; production rows from OTLP do), and model calls always
    (the reply plus one per tool call, so a harness with no tools costs 1)."""
    n = len(rows)
    if not n:
        return {"tokens": None, "calls": None}
    usage = [r.get("usage") for r in rows if isinstance(r.get("usage"), dict)]
    tokens = None
    if len(usage) == n:
        tokens = (
            sum(
                float(u.get("input_tokens") or 0) + float(u.get("output_tokens") or 0)
                for u in usage
            )
            / n
        )
    return {"tokens": tokens, "calls": sum(1 + tool_calls(r) for r in rows) / n}


def simulate(
    harness: wai.Harness,
    *,
    tasks: Path | None,
    k: int,
    budget: int,
    seed: int,
    production: list[dict] | None = None,
    concurrency: int = 1,
) -> Any:
    """One candidate on one model. The first call draws the frozen set from
    the seeds with the offline writer, or takes the production tasks; every
    later call replays it."""
    kw: dict[str, Any] = {}
    if tasks is not None and tasks.exists():
        kw["tasks"] = str(tasks)
    elif production:
        kw.update(tasks=production, mode="rl", budget=len(production) * k)
    else:
        kw.update(seeds=common.SEEDS, situations=budget, mode="rl", budget=budget * k)
    return wai.simulate(
        harness,
        simulator=False,  # the template writer: offline, deterministic, no key
        repeats=k,
        repeat_policy="fixed",
        reproducible=True,  # round-synchronous: the same rows at any concurrency
        seed=seed,
        concurrency=concurrency,
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


def freeze_split(
    rows: list[dict], *, production: list[dict] | None, holdout: float, seed: int
) -> tuple[set[str], set[str], str, dict[str, Any]]:
    """Train and holdout keys, how they were drawn, and the contamination
    count. A drawn set splits by seed; production traffic splits by day and
    drops train prompts that overlap a holdout prompt (``wai.decontaminate``,
    the 8-gram rule) from the proposer's window."""
    if not production:
        train, hold = split(sorted({task_key(r) for r in rows}), holdout, seed)
        return train, hold, f"split by seed {seed}", {"n_train": len(train), "n_dropped": 0}
    train, hold, how = split_by_day(production, holdout, seed)
    by_key = {t["task_id"]: t for t in production}
    clean, report = wai.decontaminate(
        [by_key[key] for key in sorted(train)], against=[by_key[key] for key in sorted(hold)]
    )
    dropped = train - {t["task_id"] for t in clean}
    contamination = {"n_train": report["n"], "n_dropped": len(dropped), "dropped": sorted(dropped)}
    return train - dropped, hold, how, contamination


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
    production: list[dict] | None = None,
    concurrency: int = CONCURRENCY,
) -> list[dict[str, Any]]:
    """Every candidate on every model over the same frozen tasks. Returns the
    ledger, one entry per candidate, and writes the traces. With
    ``production`` the tasks are the traffic, the holdout is its latest
    days, and train prompts overlapping a holdout prompt leave the
    proposer's window; ``out/split.json`` records the split."""
    tasks = out / "tasks.jsonl"
    split_file = out / "split.json"
    search_model = models[0]
    ledger: list[dict[str, Any]] = []
    train_keys: set[str] = set()
    hold_keys: set[str] = set()
    if split_file.exists():
        saved = json.loads(split_file.read_text(encoding="utf-8"))
        train_keys, hold_keys = set(saved["train"]), set(saved["holdout"])
    for path, module in entries:
        rows_all: list[dict] = []
        per_model: dict[str, list[dict]] = {}
        for model in models:
            harness = module.harness(model)
            data = simulate(
                harness,
                tasks=tasks,
                k=k,
                budget=budget,
                seed=seed,
                production=production,
                concurrency=concurrency if not model.startswith("scripted") else 1,
            )
            if not tasks.exists():
                data.save(str(tasks))
                train_keys, hold_keys, how, contamination = freeze_split(
                    data.rows(), production=production, holdout=holdout, seed=seed
                )
                split_file.write_text(
                    json.dumps(
                        {
                            "how": how,
                            "train": sorted(train_keys),
                            "holdout": sorted(hold_keys),
                            "contamination": contamination,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"split: {how}; {contamination['n_dropped']} train prompt(s) overlapped "
                    "the holdout and left the proposer's window"
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
            "cost": cost(first),
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
    names = {model, _model_name(model)}  # rows carry the bare model name, not the spec
    return [r for r in rows if r["harness"]["model"] in names and r.get("split") == split_name]


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
    split_note = ""
    if (out / "split.json").exists():
        saved = json.loads((out / "split.json").read_text(encoding="utf-8"))
        split_note = (
            f"The {saved['how']}; {saved['contamination']['n_dropped']} train prompt(s) that "
            "overlapped a holdout prompt are not shown. "
        )
    lines = [
        "# Proposal",
        "",
        "You are the proposer in a Meta-Harness loop (Lee et al. 2026, arXiv:2603.28052).",
        "Below is every candidate so far: its source, its pass@1 on the train split with a",
        "95% interval over tasks, and its worst rows. Change what the worst rows say is",
        f"wrong, and only that. {split_note}"
        f"{NEXT_NOTE.format(next=f'{next_n:02d}_<name>', flags=flags)}",
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


def tasks_led(ledger: list[dict[str, Any]], *, model: str, out: Path) -> dict[str, int]:
    """How many train tasks each candidate leads: per task, the best pass
    rate across every candidate, ties shared (the per-task frontier of
    Agrawal et al. 2025, GEPA, arXiv:2507.19457)."""
    per: dict[str, dict[str, float]] = {}
    for entry in ledger:
        rows = _rows(out, entry, model=model, split_name="train")
        per[entry["candidate"]] = dict(wai.pass_at(rows).per_task) if rows else {}
    led = dict.fromkeys(per, 0)
    for task in sorted({t for rates in per.values() for t in rates}):
        top = max(rates.get(task, 0.0) for rates in per.values())
        for name, rates in per.items():
            if rates.get(task, 0.0) == top:
                led[name] += 1
    return led


def regressed(out: Path, baseline: dict[str, Any], pick: dict[str, Any], *, model: str) -> int:
    """Holdout tasks the baseline passed every time and the pick failed
    every time: the subset a mean can hide."""
    base = wai.pass_at(_rows(out, baseline, model=model, split_name="holdout")).per_task
    new = wai.pass_at(_rows(out, pick, model=model, split_name="holdout")).per_task
    return sum(1 for t, rate in base.items() if rate == 1.0 and new.get(t) == 0.0)


def _cost_of(out: Path, entry: dict[str, Any], *, model: str) -> dict[str, float | None]:
    """The ledger's cost per rollout, or, for a ledger another recipe wrote
    without one, the same number read off the entry's rows."""
    if isinstance(entry.get("cost"), dict):
        return entry["cost"]
    with open(out / entry["rows"], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    names = {model, _model_name(model)}
    return cost([r for r in rows if r.get("harness", {}).get("model") in names] or rows)


def cost_ratio(
    baseline: dict[str, Any], pick: dict[str, Any], *, out: Path, model: str
) -> tuple[float, str]:
    """The pick's cost per rollout over the baseline's: tokens when both
    carry them, else tool calls."""
    base_cost, pick_cost = _cost_of(out, baseline, model=model), _cost_of(out, pick, model=model)
    unit = "tokens" if base_cost.get("tokens") and pick_cost.get("tokens") else "calls"
    base, new = base_cost.get(unit) or 0.0, pick_cost.get(unit) or 0.0
    return (new / base if base else 1.0), unit


def select(
    ledger: list[dict[str, Any]],
    *,
    models: list[str],
    out: Path,
    cost_margin: float | None = COST_MARGIN,
) -> dict[str, Any]:
    """The gate. The candidate that leads the most train tasks is the pick;
    the holdout on the search model, and on at least one held-out model,
    has to agree with an interval that excludes zero, at a cost per rollout
    within ``cost_margin`` of the baseline's (``None``: cost is reported,
    not gated). Then attribution over the grid."""
    baseline = ledger[0]
    verdict: dict[str, Any] = {"baseline": baseline["candidate"], "selected": None}
    if len(ledger) == 1:
        verdict["reason"] = "only the baseline has run; write a candidate"
        print("select:", verdict["reason"])
        (out / "selected.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        return verdict
    led = tasks_led(ledger, model=models[0], out=out)
    best = max(ledger[1:], key=lambda e: (led[e["candidate"]], e["train"]["pass_at_1"] or 0.0))
    verdict["tasks_led"] = led
    verdict["best_on_train"] = best["candidate"]
    print(
        "train tasks led: "
        + ", ".join(f"{e['candidate']} {led[e['candidate']]}" for e in ledger)
        + f" -> pick {best['candidate']}"
    )
    checks: dict[str, Any] = {}
    for model in models:
        cmp = compare_runs(
            _rows(out, baseline, model=model, split_name="holdout"),
            _rows(out, best, model=model, split_name="holdout"),
        )
        lo, hi = cmp["ci95"] or (None, None)  # None: too few paired tasks for an interval
        lost = regressed(out, baseline, best, model=model)
        clears = lo is not None and lo > 0
        checks[model] = {
            "delta": cmp["delta"],
            "ci95": [lo, hi] if lo is not None else None,
            "n_paired": cmp["n_paired"],
            "regressed": lost,
            "clears": clears,
        }
        interval = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "[no interval]"
        delta = f"{cmp['delta']:+.2f}" if cmp["delta"] is not None else "n/a"
        print(
            f"holdout on {model}: {best['candidate']} vs {baseline['candidate']} "
            f"{delta} {interval} over {cmp['n_paired']} paired tasks, "
            f"{lost} the baseline passed and the pick failed"
            f" -> {'clears zero' if clears else 'could be chance'}"
        )
    ratio, unit = cost_ratio(baseline, best, out=out, model=models[0])
    gated = cost_margin is not None
    within = not gated or ratio <= 1.0 + cost_margin + COST_EPS
    checks["cost"] = {"ratio": ratio, "unit": unit, "margin": cost_margin, "clears": within}
    print(
        f"cost per rollout: {best['candidate']} at {ratio:.2f}x the baseline in {unit} -> "
        + (
            "not gated"
            if not gated
            else "within the margin"
            if within
            else f"over --cost-margin {cost_margin}"
        )
    )
    verdict["checks"] = checks
    on_search = checks[models[0]]["clears"]
    on_held_out = any(c["clears"] for m, c in checks.items() if m not in (models[0], "cost"))
    on_score = on_search and (on_held_out or len(models) == 1)
    if on_score and within:
        verdict["selected"] = best["candidate"]
        verdict["reason"] = (
            f"{best['candidate']} beats the baseline on the holdout"
            + (" and on a held-out model" if len(models) > 1 else "")
            + (f" at {ratio:.2f}x its cost" if gated else "")
        )
    elif on_score:
        verdict["reason"] = (
            f"{best['candidate']} beats the baseline on the holdout at {ratio:.2f}x its cost "
            f"per rollout in {unit}; not selected at --cost-margin {cost_margin} (Wang et al. "
            "2026, arXiv:2607.12227: at a matched budget the baseline may do as well). "
            f"Pass --cost-margin {max(0.0, ratio - 1.0):.2f} to accept the cost, or write a "
            "cheaper candidate"
        )
    else:
        verdict["reason"] = (
            f"{best['candidate']} leads the most train tasks but does not clear the baseline "
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
    p.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help="rollouts in flight on a live model (scripted models run one at a time)",
    )
    p.add_argument(
        "--traces",
        default=None,
        help="production traffic as the frozen set: a JSONL of traces or an OTLP JSON batch",
    )
    p.add_argument(
        "--cost-margin",
        type=float,
        default=COST_MARGIN,
        help="how much more per rollout than the baseline a pick may cost (0 = matched)",
    )
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
    production = None
    if args.traces:
        production = load_production(Path(args.traces), budget=args.budget, seed=args.seed)
        print(f"traces: {len(production)} tasks from {args.traces}")
    ledger = evaluate(
        entries,
        models=models,
        judge=_judge(args.judge),
        out=out,
        k=args.k,
        budget=args.budget,
        holdout=args.holdout,
        seed=args.seed,
        production=production,
        concurrency=args.concurrency,
    )
    flags = (" --dry-run" if args.dry_run else "") + (
        f" --traces {args.traces}" if args.traces else ""
    )
    if args.propose:
        propose(ledger, entries, out, flags=flags)
    if args.select:
        select(ledger, models=models, out=out, cost_margin=args.cost_margin)
    if not args.propose and not args.select:
        print(f"Next: python run.py{' --dry-run' if args.dry_run else ''} --propose --select")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
