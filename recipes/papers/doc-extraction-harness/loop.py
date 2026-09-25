# Adapted from recipes/papers/meta-harness/run.py (Meta-Harness, Lee et al. 2026,
# arXiv:2603.28052), copied at origin/main 25b9fac9 so this recipe does not
# change another recipe's files. The method is unchanged. What differs, all
# additive: --task (a task module supplies the frozen set and the judge; a
# declared split is kept), --metric (marker:<name> for a per-task score in
# [0, 1]), --blind (holdout scores left out of proposal.md), and a pruned-file
# writer for candidates that are not meta-harness's. meta-harness's own
# common.py (its support-bot task) is not used: --task is required.
"""Meta-Harness: an outer loop that searches over harness code, scores every
candidate on one frozen task set, and gates the pick on held-out tasks and
held-out models.

    python run.py --dry-run                  # offline: scripted candidates, no key
    python run.py --dry-run --propose        # and write out/proposal.md for the proposer
    python run.py --dry-run --select         # and apply the gate
    python run.py --models openai:gpt-4.1-mini,anthropic:claude-haiku-4-5  # the live run
    python run.py --traces traces.jsonl --propose --select    # the frozen set is production traffic
    python run.py --dry-run --select --prune # and drop the pick's edits that buy nothing

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
7. ``--prune`` takes the pick's named edits (``EDITS``, built with
   ``common.from_edits``) out one at a time and drops every edit whose
   removal costs no train score and no cost per rollout (Xia et al. 2026,
   RRSI, arXiv:2609.24972: the pruner removes changes that are too small,
   too expensive or no longer useful, so the harness keeps reusable
   mechanisms and not benchmark-specific noise). The decisions read the
   train split; the pruned harness then faces the same gate, and
   ``out/<pick>_pruned.py`` is the file to copy into ``candidates/``.

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
from whileai.simulations.score.stats import compare_runs, metric_summary, task_key, task_means

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


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
PRUNE_TOL = 0.0  # train pass@1 an edit may cost and still be pruned; 0 = any loss keeps it
COST_EPS = 1e-9  # float slack so a pick at exactly the baseline's cost passes margin 0
NANOS_PER_SECOND = 1e9  # an OTLP span's start_time_unix_nano is in nanoseconds
# METRIC: what score, the tasks-led count, the holdout comparison and the
# regressed count read. "pass_at_1" (a binary reward, the default) or
# "marker:<name>" for a per-task score in [0, 1] the judge writes as a marker
# (--metric; a task with several graded fields scores its share right).
METRIC = "pass_at_1"
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


# REUSE: --reuse keeps a candidate's graded rows from an earlier run when its
# harness fingerprint and model are unchanged, instead of replaying it every
# round (added in this copy; the source recipe replays every candidate).
REUSE = False
# BLIND: --blind also keeps the holdout score out of the ledger line the loop
# prints per candidate (added in this copy): the proposer reads stdout too.
BLIND = False


def _reuse(out: Path, path: Path, harness: Any, model: str) -> list[dict] | None:
    rows_file = out / "traces" / path.stem / "rows.jsonl"
    if not rows_file.exists():
        return None
    names = {model, _model_name(model)}
    with open(rows_file, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = [r for r in rows if (r.get("harness") or {}).get("model") in names]
    if not rows or any(r["harness"].get("hash") != harness.fingerprint for r in rows):
        return None
    return rows


def _load_task(path: Path) -> ModuleType:
    """A task module: ``tasks()`` returns the frozen set as ``{prompt,
    task_id}`` dicts (``split`` of ``train`` or ``holdout`` to declare the
    split), ``judge(row)`` grades a row. The loop does not change per task."""
    spec = importlib.util.spec_from_file_location(f"task_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import task module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.resolve().parent))
    spec.loader.exec_module(module)
    for name in ("tasks", "judge"):
        if not callable(getattr(module, name, None)):
            raise SystemExit(f"{path.name}: a task module defines tasks() and judge(row)")
    return module


def _safe(name: str) -> str:
    return name.replace(":", "-").replace("/", "-")


def _judge(spec: str) -> Any:
    raise SystemExit(
        f"--judge {spec}: this copy takes its judge from the --task module; drop --judge"
    )


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
        raise SystemExit("no frozen set: pass --task (this copy draws no support-bot tasks)")
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
    if METRIC != "pass_at_1":
        ms = metric_summary(rows, METRIC)
        return {
            "pass_at_1": ms["mean"],
            "ci95": list(ms["ci95"]) if ms["ci95"] else None,
            "n_tasks": ms["n_tasks"],
            "note": ms.get("warning"),
            "metric": METRIC,
        }
    pa = wai.pass_at(rows)
    return {
        "pass_at_1": pa.pass_at_1,
        "ci95": list(pa.ci95) if pa.ci95 else None,
        "n_tasks": pa.n_groups,
        "note": pa.note or None,
    }


def per_task(rows: list[dict]) -> dict[str, float]:
    """Each task's score under ``METRIC``: pass rate, or the marker's mean."""
    if METRIC == "pass_at_1":
        return dict(wai.pass_at(rows).per_task)
    return task_means(rows, METRIC)


def fmt(s: dict[str, Any]) -> str:
    if s["pass_at_1"] is None:
        return "n/a"
    ci = f" [{s['ci95'][0]:.2f}..{s['ci95'][1]:.2f}]" if s["ci95"] else ""
    return f"{s['pass_at_1']:.2f}{ci}"


def worst_rows(rows: list[dict], n: int) -> list[dict]:
    failed = [r for r in rows if not r.get("reward")]
    if METRIC != "pass_at_1":  # the lowest-scoring rows first
        key = METRIC.split(":", 1)[1]
        failed.sort(key=lambda r: float((r.get("markers") or {}).get(key, 0.0) or 0.0))
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
    declared = {t["task_id"]: t.get("split") for t in production}
    if all(v in ("train", "holdout") for v in declared.values()):
        # The task module drew the split itself (a hash of the task id, say):
        # keep it. Templated tasks share text by design, so the 8-gram rule
        # would drop every train task; the module owns that decision.
        train = {key for key, v in declared.items() if v == "train"}
        hold = set(declared) - train
        how = f"split declared by the task module: {len(hold)} of {len(declared)} tasks held out"
        return train, hold, how, {"n_train": len(train), "n_dropped": 0, "declared": True}
    train, hold, how = split_by_day(production, holdout, seed)
    by_key = {t["task_id"]: t for t in production}
    clean, report = wai.decontaminate(
        [by_key[key] for key in sorted(train)], against=[by_key[key] for key in sorted(hold)]
    )
    dropped = train - {t["task_id"] for t in clean}
    contamination = {"n_train": report["n"], "n_dropped": len(dropped), "dropped": sorted(dropped)}
    return train - dropped, hold, how, contamination


def ledger_entry(
    candidate: str,
    harness: wai.Harness,
    per_model: dict[str, list[dict]],
    *,
    models: list[str],
    out: Path,
    k: int,
    n_tasks: int,
) -> dict[str, Any]:
    """One ledger line from a candidate's graded rows, and its traces on
    disk: every row, and the worst on the train split."""
    first = per_model[models[0]]
    train = [r for r in first if r["split"] == "train"]
    hold = [r for r in first if r["split"] == "holdout"]
    trace_dir = out / "traces" / Path(candidate).stem
    trace_dir.mkdir(parents=True, exist_ok=True)
    with open(trace_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in (r for m in models for r in per_model[m]):
            fh.write(json.dumps(r, default=str) + "\n")
    with open(trace_dir / "worst.jsonl", "w", encoding="utf-8") as fh:
        for r in worst_rows(train, WORST):
            fh.write(json.dumps(r, default=str) + "\n")
    entry = {
        "candidate": candidate,
        "label": harness.version,
        "fingerprint": harness.fingerprint,
        "model": models[0],
        "k": k,
        "n_tasks": n_tasks,
        "train": score(train),
        "holdout": score(hold),
        "cost": cost(first),
        "held_out_models": {
            m: score([r for r in per_model[m] if r["split"] == "holdout"]) for m in models[1:]
        },
        "worst": (trace_dir / "worst.jsonl").relative_to(out).as_posix(),
        "rows": (trace_dir / "rows.jsonl").relative_to(out).as_posix(),
    }
    extra = "".join(f"  {m} {fmt(s)}" for m, s in entry["held_out_models"].items())
    print(
        f"{Path(candidate).stem:<18} train {fmt(entry['train'])}  "
        + ("holdout withheld (--blind)" if BLIND else f"holdout {fmt(entry['holdout'])}{extra}")
    )
    return entry


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
            cached = _reuse(out, path, harness, model) if REUSE else None
            if cached is not None:
                for r in cached:
                    r["split"] = "holdout" if task_key(r) in hold_keys else "train"
                per_model[model] = cached
                rows_all.extend(cached)
                print(f"{path.stem} on {model}: {len(cached)} rows reused (same fingerprint)")
                continue
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
        entry = ledger_entry(
            path.name,
            module.harness(search_model),
            per_model,
            models=models,
            out=out,
            k=k,
            n_tasks=len(train_keys) + len(hold_keys),
        )
        ledger.append(entry)
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
    blind: bool = False,
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
        f"Below is every candidate so far: its source, its {METRIC} on the train split with a",
        "95% interval over tasks, and its worst rows. Change what the worst rows say is",
        f"wrong, and only that. {split_note}"
        f"{NEXT_NOTE.format(next=f'{next_n:02d}_<name>', flags=flags)}",
        "",
    ]
    for (path, _), entry in zip(entries, ledger):
        lines += [
            f"## {entry['candidate']}",
            "",
            f"train {METRIC} {fmt(entry['train'])} on {entry['train']['n_tasks']} tasks; "
            + ("" if blind else f"holdout {fmt(entry['holdout'])}; ")
            + f"fingerprint {entry['fingerprint']}",
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
        per[entry["candidate"]] = per_task(rows) if rows else {}
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
    base = per_task(_rows(out, baseline, model=model, split_name="holdout"))
    new = per_task(_rows(out, pick, model=model, split_name="holdout"))
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


def _ratio(base_cost: dict[str, Any], new_cost: dict[str, Any]) -> tuple[float, str]:
    unit = "tokens" if base_cost.get("tokens") and new_cost.get("tokens") else "calls"
    base, new = base_cost.get(unit) or 0.0, new_cost.get(unit) or 0.0
    return (new / base if base else 1.0), unit


def cost_ratio(
    baseline: dict[str, Any], pick: dict[str, Any], *, out: Path, model: str
) -> tuple[float, str]:
    """The pick's cost per rollout over the baseline's: tokens when both
    carry them, else tool calls."""
    return _ratio(_cost_of(out, baseline, model=model), _cost_of(out, pick, model=model))


def select(
    ledger: list[dict[str, Any]],
    *,
    models: list[str],
    out: Path,
    cost_margin: float | None = COST_MARGIN,
    dest: str = "selected.json",
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
        (out / dest).write_text(json.dumps(verdict, indent=2), encoding="utf-8")
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
            metric=METRIC,
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
        report = wai.harness.attribute(grid, metric=METRIC)
        print(report)
        verdict["attribution"] = {
            "verdict": report["verdict"],
            "share_harness": report["share_harness"],
            "share_model": report["share_model"],
        }
    print("select:", verdict["reason"])
    (out / dest).write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def replay(
    harness: wai.Harness,
    *,
    model: str,
    judge: Any,
    out: Path,
    k: int,
    seed: int,
    concurrency: int,
) -> list[dict]:
    """One harness on the frozen tasks, graded, each row marked with its
    split from ``out/split.json``."""
    saved = json.loads((out / "split.json").read_text(encoding="utf-8"))
    hold_keys = set(saved["holdout"])
    data = simulate(
        harness,
        tasks=out / "tasks.jsonl",
        k=k,
        budget=0,
        seed=seed,
        concurrency=1 if model.startswith("scripted") else concurrency,
    )
    rows = [dict(r) for r in data.grade(judge=judge).rows]
    for r in rows:
        r["split"] = "holdout" if task_key(r) in hold_keys else "train"
    return rows


def prune(
    verdict: dict[str, Any],
    ledger: list[dict[str, Any]],
    entries: list[tuple[Path, ModuleType]],
    *,
    models: list[str],
    judge: Any,
    out: Path,
    k: int,
    seed: int,
    concurrency: int = CONCURRENCY,
    tol: float = PRUNE_TOL,
    cost_margin: float | None = COST_MARGIN,
) -> dict[str, Any] | None:
    """The pruner of Xia et al. 2026 (RRSI, arXiv:2609.24972): take the
    pick's edits out one at a time and keep out every edit whose removal
    costs at most ``tol`` of train pass@1 and no cost per rollout. The
    decisions read the train split on the search model only, so the
    holdout still decides: the pruned harness then faces the same gate as
    the pick, and its holdout is compared with the pick's."""
    name = verdict.get("selected")
    if not name:
        print("prune: nothing was selected; pruning a pick the holdout rejected only fits train")
        return None
    module = next(m for p, m in entries if p.name == name)
    edits = getattr(module, "EDITS", None)
    if not edits:
        print(f"prune: {name} declares no EDITS; write it with common.from_edits to prune it")
        return None
    pick = next(e for e in ledger if e["candidate"] == name)
    search = models[0]
    rows = _rows(out, pick, model=search, split_name="train")
    current = (score(rows)["pass_at_1"] or 0.0, cost(rows))
    dropped: list[str] = []
    decisions: list[dict[str, Any]] = []
    for edit in edits:
        trial = (*dropped, edit)
        trial_rows = [
            r
            for r in replay(
                module.harness(search, drop=trial),
                model=search,
                judge=judge,
                out=out,
                k=k,
                seed=seed,
                concurrency=concurrency,
            )
            if r["split"] == "train"
        ]
        without = (score(trial_rows)["pass_at_1"] or 0.0, cost(trial_rows))
        ratio, unit = _ratio(current[1], without[1])
        useless = without[0] >= current[0] - tol - COST_EPS and ratio <= 1.0 + COST_EPS
        decisions.append(
            {
                "edit": edit,
                "train_with": current[0],
                "train_without": without[0],
                "cost_without": ratio,
                "unit": unit,
                "dropped": useless,
            }
        )
        print(
            f"prune {edit}: train {without[0]:.2f} without it vs {current[0]:.2f}, "
            f"{ratio:.2f}x the cost in {unit} -> "
            + ("dropped" if useless else "kept: removing it loses score or raises cost")
        )
        if useless:
            dropped.append(edit)
            current = without
    report: dict[str, Any] = {"pick": name, "tol": tol, "decisions": decisions}
    if not dropped:
        report["pruned"] = None
        report["reason"] = f"every edit in {name} earns its place on train"
        print("prune:", report["reason"])
        (out / "pruned.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    pruned_name = f"{Path(name).stem}_pruned.py"
    per_model = {
        m: replay(
            module.harness(m, drop=tuple(dropped)),
            model=m,
            judge=judge,
            out=out,
            k=k,
            seed=seed,
            concurrency=concurrency,
        )
        for m in models
    }
    entry = ledger_entry(
        pruned_name,
        module.harness(search, drop=tuple(dropped)),
        per_model,
        models=models,
        out=out,
        k=k,
        n_tasks=pick["n_tasks"],
    )
    gate = select(
        [ledger[0], entry], models=models, out=out, cost_margin=cost_margin, dest="pruned_gate.json"
    )
    cmp = compare_runs(
        _rows(out, pick, model=search, split_name="holdout"),
        _rows(out, entry, model=search, split_name="holdout"),
        metric=METRIC,
    )
    ratio, unit = cost_ratio(pick, entry, out=out, model=search)
    lo, hi = cmp["ci95"] or (None, None)
    interval = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "[no interval]"
    delta = f"{cmp['delta']:+.2f}" if cmp["delta"] is not None else "n/a"
    print(
        f"pruned vs pick on the holdout: {delta} {interval} over {cmp['n_paired']} paired "
        f"tasks, at {ratio:.2f}x the pick's cost in {unit}"
    )
    kept = {e: edits[e] for e in edits if e not in dropped}
    source = out / pruned_name
    listed = "".join(f"    {e!r}: {edit!r},\n" for e, edit in kept.items())
    if not hasattr(module, "SCRIPTED_RATE"):
        # A candidate from another recipe (its own Edit type, its own
        # helpers): the pruned file loads the pick and fixes the drop.
        pick_path = next(p for p, m in entries if p.name == name)
        source.write_text(
            f'"""{name} with {", ".join(dropped)} pruned (run.py --prune)."""\n\n'
            "from __future__ import annotations\n\n"
            "import runpy\n\n"
            "import whileai as wai\n\n"
            f"PICK = {str(pick_path.resolve())!r}\n"
            f"DROPPED = {tuple(dropped)!r}\n\n\n"
            "def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:\n"
            '    return runpy.run_path(PICK)["harness"](model, drop=DROPPED + tuple(drop))\n',
            encoding="utf-8",
        )
    else:
        source.write_text(
            f'"""{name} with {", ".join(dropped)} pruned (run.py --prune): the edits left are the\n'
            'ones whose removal cost train score or raised cost per rollout."""\n\n'
            "from __future__ import annotations\n\n"
            "from common import Edit, from_edits\n\n"
            "import whileai as wai\n\n"
            f"EDITS = {{\n{listed}}}\n\n"
            f"SCRIPTED_RATE = {module.SCRIPTED_RATE!r}\n"
            f"SCRIPTED_BEHAVIORS: tuple[str, ...] = {tuple(module.SCRIPTED_BEHAVIORS)!r}\n\n\n"
            "def harness(model: str, drop: tuple[str, ...] = ()) -> wai.Harness:\n"
            "    return from_edits(\n"
            "        model,\n"
            "        EDITS,\n"
            f'        label="{Path(pruned_name).stem}",\n'
            "        scripted_rate=SCRIPTED_RATE,\n"
            "        scripted_behaviors=SCRIPTED_BEHAVIORS,\n"
            "        drop=drop,\n"
            "    )\n",
            encoding="utf-8",
        )
    report.update(
        pruned=pruned_name,
        dropped=dropped,
        kept=list(kept),
        selected=gate["selected"],
        vs_pick={"delta": cmp["delta"], "ci95": cmp["ci95"], "cost_ratio": ratio, "unit": unit},
        source=_show(source),
    )
    print(
        f"prune: {len(dropped)} of {len(edits)} edits dropped ({', '.join(dropped)}); "
        + (
            f"the pruned harness clears the gate; copy {_show(source)} into candidates/ to keep it"
            if gate["selected"]
            else "the pruned harness does not clear the gate; keep the pick"
        )
    )
    (out / "pruned.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=int, default=K, help="rollouts per task")
    p.add_argument(
        "--budget",
        type=int,
        default=None,
        help=f"tasks in the frozen set (default {BUDGET}); with --task, at most this many per "
        "declared split, 0 or unset = every task the module declares",
    )
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
    p.add_argument(
        "--prune",
        action="store_true",
        help="after --select, drop each edit of the pick that buys nothing on train (RRSI)",
    )
    p.add_argument(
        "--prune-tol",
        type=float,
        default=PRUNE_TOL,
        help="train pass@1 an edit may cost when removed and still be pruned",
    )
    p.add_argument(
        "--task",
        default=None,
        help="a task module: tasks() -> [{prompt, task_id, split?}] and judge(row); "
        "the frozen set and the judge come from it instead of common.py",
    )
    p.add_argument(
        "--metric",
        default="pass_at_1",
        help="pass_at_1, or marker:<name> for a per-task score in [0, 1] the judge writes",
    )
    p.add_argument(
        "--reuse",
        action="store_true",
        help="keep a candidate's rows from an earlier run when its fingerprint is unchanged",
    )
    p.add_argument(
        "--blind", action="store_true", help="leave holdout scores out of out/proposal.md"
    )
    p.add_argument("--candidates", default="candidates", help="folder of candidate files")
    p.add_argument("--out", default="out", help="ledger, traces, proposal, selection")
    p.add_argument("--propose", action="store_true", help="write out/proposal.md")
    p.add_argument("--select", action="store_true", help="apply the gate, write selected.json")
    p.add_argument("--fresh", action="store_true", help="drop out/ first: a new frozen set")
    p.add_argument("--dry-run", action="store_true", help="offline: scripted models, no key")
    return p


def main(argv: list[str] | None = None) -> int:
    global METRIC, REUSE, BLIND
    print(provenance(), file=sys.stderr)
    args = build_parser().parse_args(argv)
    METRIC = args.metric
    REUSE = args.reuse
    BLIND = args.blind
    if not args.task:
        print("--task is required: python loop.py --task task_docs.py ...", file=sys.stderr)
        return 2
    task = _load_task(Path(args.task))
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
    if task is not None:
        production = list(task.tasks())
        if args.budget:
            # the smoke run: the first --budget tasks of each declared split,
            # in the module's order, so a dry run stays under a minute
            kept: list[dict] = []
            seen: dict[str, int] = {}
            for t in production:
                s = str(t.get("split"))
                if seen.get(s, 0) < args.budget:
                    seen[s] = seen.get(s, 0) + 1
                    kept.append(t)
            production = kept
        print(f"task: {len(production)} tasks from {args.task}")
    if args.budget is None:
        args.budget = BUDGET
    if args.traces:
        production = load_production(Path(args.traces), budget=args.budget, seed=args.seed)
        print(f"traces: {len(production)} tasks from {args.traces}")
    judge = task.judge if task is not None and args.judge == "program" else _judge(args.judge)
    ledger = evaluate(
        entries,
        models=models,
        judge=judge,
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
    if args.task:
        flags += f" --task {args.task} --metric {args.metric}" + (" --blind" if args.blind else "")
    if args.propose:
        propose(ledger, entries, out, flags=flags, blind=args.blind)
    if args.select or args.prune:
        verdict = select(ledger, models=models, out=out, cost_margin=args.cost_margin)
        if args.prune:
            prune(
                verdict,
                ledger,
                entries,
                models=models,
                judge=judge,
                out=out,
                k=args.k,
                seed=args.seed,
                concurrency=args.concurrency,
                tol=args.prune_tol,
                cost_margin=args.cost_margin,
            )
    if not args.propose and not args.select and not args.prune:
        print(f"Next: python run.py{' --dry-run' if args.dry_run else ''} --propose --select")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
