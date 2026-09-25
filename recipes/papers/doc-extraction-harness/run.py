"""Drive the Meta-Harness recipe on the document-extraction task, measure the
noise floor, audit the judge, and write and post the report.

    python run.py search --dry-run     # one round of the recipe's loop, scripted model and judge
    python run.py noise --dry-run      # the baseline twice more: the three-run floor
    python run.py gate --dry-run       # baseline and pick on every model: --select, attribute, --prune
    python run.py audit --dry-run      # judge vs program gold; 40 rows for a person to label
    python run.py report --dry-run     # results.json, chart.html, ledger.md (+ --post live)
    python run.py search --models "vllm:nvidia/Llama-3.1-Nemotron-Nano-8B-v1@$NEMO,vllm:Qwen/Qwen3-8B@$QWEN,..."

``search`` is ``loop.py --task task_docs.py --metric
marker:field_f1 --blind --k 4 --reuse --propose --select`` on the search model (the
first in ``--models``) with this recipe's candidates and out/ (Lee et al.
2026, Meta-Harness, arXiv:2603.28052). ``gate`` runs the same file on the
baseline and the pick over every model. This file adds only what the recipe
does not do: the three-run noise floor, the minimum detectable effect, the
judge audit, per-field tables, spend, the report. A second task is another
``task_<name>.py`` and another candidates folder (``--task``, ``--candidates``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import runpy
import shutil
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LOOP = HERE / "loop.py"  # adapted from ../meta-harness/run.py
sys.path.insert(0, str(HERE))

import docs
import whileai as wai

DRY_MODELS = "scripted,scripted-b,scripted-c"
# K = 4: rollouts per document, the Meta-Harness recipe's default --k.
K = 4
# RERUNS = 3: two runs are a difference, not a spread: t(df=1) is 12.71
# (skills/strengthen-your-evals, "The noise floor").
RERUNS = 3
# MDE_Z = 2.8: z(0.975) + z(0.8), the minimum detectable effect at 80% power
# (skills/strengthen-your-evals, "Gates, with the math").
MDE_Z = 2.8
# N_BOOT = 2000, BOOT_SEED = 0: the bootstrap over documents, wai.compare's defaults.
N_BOOT, BOOT_SEED = 2000, 0
# GPU_USD_PER_HOUR: Modal's list prices (`modal billing rates`, 2026-09-25). The
# serve app takes the first free GPU of a fallback list; out/gpus.json records
# which one each model's container landed on (read off its vLLM log).
GPU_USD_PER_HOUR = {"L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}
L40S_USD_PER_HOUR = GPU_USD_PER_HOUR["L40S"]
# SCALEDOWN_S = 600: serve_modal.py's scaledown_window; each idle tail is billed.
SCALEDOWN_S = 600
# Claude Haiku 4.5 list prices, USD per million tokens (input, output).
HAIKU_USD_PER_MTOK = (1.0, 5.0)
# JUDGE_SYSTEM_TOKENS = 330, JUDGE_OUT_TOKENS = 110: the rubric judge's system
# prompt and a criteria object plus one sentence, measured off test calls;
# CHARS_PER_TOKEN = 3.6 for JSON and English (convention). Spend is estimated.
JUDGE_SYSTEM_TOKENS, JUDGE_OUT_TOKENS, CHARS_PER_TOKEN = 330, 110, 3.6
# The headline is field F1 (a program vs the gold, SROIE/CORD; ANLS on names). The Haiku rubric
# judge's quality score sits beside it: on the baseline's 380 distinct answers
# its agreement with program gold on the four checkable criteria was 0.42-0.71,
# kappa 0.11-0.17, under the 0.8 / 0.6 floors (skills/audit-your-judge), so it
# does not steer the search. out/audit.json has the numbers.
HEADLINE, SECOND = "field_f1", "quality"
# EXTRA: reported beside the headline, never steering: precision, recall, ANLS
# on name fields, null accuracy, and exact field accuracy for continuity.
EXTRA = ("field_precision", "field_recall", "anls_names", "null_acc", "field_acc")
# Behavior names as the page lists them (letters, digits, . _ - only, so "--"
# stands for a dash): numbered so the headline sorts and registers first.
# Names are fixed once posted (the API has no behavior delete), so the
# readable wording lives in each behavior's description; the headline is
# registered first and every search run targets it.
NAMES = {"field_f1": "field_f1_selection", "quality": "response_quality_selection"}
HOLDOUT_NAMES = {"field_f1": "field_f1", "quality": "response_quality"}
SLICE_TITLES = {
    "easy": "Easy docs F1",
    "medium": "Medium docs F1",
    "hard": "Hard docs F1",
    "invoice": "Invoices F1",
    "receipt": "Receipts F1",
    "purchase_order": "Purchase orders F1",
    "bank_statement": "Bank statements F1",
    "claim_form": "Claim forms F1",
}
SLICE_NAMES = {k: f"field_f1_{k}_selection" for k in SLICE_TITLES}
HOLDOUT_SLICE_NAMES = {k: f"field_f1_{k}" for k in SLICE_TITLES}


def bname(mk: str, split: str) -> str:
    return NAMES[mk] if split == "train" else HOLDOUT_NAMES[mk]


F1_RUBRIC = (
    "Program: per-field match against the generator's gold. Money, date, id, digits and enum "
    "exact after normalization; names ANLS ≥ 0.5. Micro F1 over fields: a wrong value is FP+FN, a "
    "missing value FN, an invented value FP, a correct null counts in neither. Mean over k=4 "
    "rollouts, then over documents."
)


def quality_rubric() -> str:
    task = runpy.run_path(str(HERE / "task_docs.py"))
    crit = "; ".join(f"{c.slug}: {c.description}" for c in task["RUBRIC"].criteria)
    return (
        "Haiku 4.5 rubric judge (temperature 0, sees the document and the final reply, never the "
        f"gold); score = share of criteria met. Criteria: {crit}. Failed its audit against program "
        "gold (agreement 0.47-0.79, kappa 0.16-0.39); reference only."
    )[:4000]


BASELINE, PLACEBO = "00_baseline.py", "01_placebo.py"
LABELS_N = 40


def out_dir(dry: bool) -> Path:
    return HERE / ("out-dry" if dry else "out")


def _name(model: str) -> str:
    """The model as rows carry it (``whileai.harness._model_name``)."""
    if "://" in model:
        return model
    return model.split(":", 1)[1] if ":" in model else model


def short(model: str) -> str:
    return _name(model).split("@")[0]


def _models(a: argparse.Namespace) -> list[str]:
    return [m.strip() for m in (DRY_MODELS if a.dry_run else a.models).split(",") if m.strip()]


# ---------------------------------------------------------------- running the recipe


def loop(
    a: argparse.Namespace,
    *,
    out: Path,
    candidates: Path,
    models: list[str],
    salt: int = 0,
    extra: tuple[str, ...] = (),
) -> None:
    cmd = [
        sys.executable,
        str(LOOP),
        "--task",
        str(HERE / a.task),
        "--metric",
        f"marker:{HEADLINE}",
        "--blind",
        "--reuse",
        "--candidates",
        str(candidates),
        "--out",
        str(out),
        "--k",
        str(K),
        "--budget",
        str(a.limit),
        "--concurrency",
        str(a.concurrency),
        "--models",
        ",".join(models),
        *extra,
    ]
    env = {
        **os.environ,
        "DOCX_SALT": str(salt),
        "PYTHONIOENCODING": "utf-8",
        "DOCX_JUDGE_CACHE": str(out_dir(a.dry_run) / "judge_cache.jsonl"),
        # the loop grades with the program only; `rejudge` adds the quality
        # judge afterwards, so judging never holds the GPU (same cache)
        "DOCX_JUDGE": "off",
    }
    if a.dry_run:
        cmd.append("--dry-run")
        env = {
            k: v for k, v in env.items() if not k.startswith(("WHILEAI_", "VLLM_", "ANTHROPIC_"))
        }
        env["DOCX_JUDGE"] = "scripted"
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=LOOP.parent, capture_output=True, text=True)
    t1 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    print("\n".join(ln for ln in proc.stdout.splitlines() if "rollouts," not in ln)[-6000:])
    with (out_dir(a.dry_run) / "spend.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "start": t0,
                    "end": t1,
                    "models": models,
                    "out": out.name,
                    "dry_run": a.dry_run,
                    "rc": proc.returncode,
                }
            )
            + "\n"
        )
    if proc.returncode:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(proc.returncode)


def search(a: argparse.Namespace) -> None:
    """A round. With ``--blind`` (every round until the pick) the loop only
    proposes: ``--select`` would print the pick's holdout comparison, and
    the proposer must not see it (Lee et al. 2026; the recipe's --blind
    covers proposal.md, this covers stdout). ``led`` then counts the train
    tasks each candidate leads; the one ``search`` without ``--blind`` at
    the pick writes selected.json."""
    o = out_dir(a.dry_run)
    o.mkdir(exist_ok=True)
    loop(
        a,
        out=o,
        candidates=HERE / a.candidates,
        models=_models(a)[:1],
        extra=("--propose",)
        + (() if a.blind else ("--select",))
        + (("--fresh",) if a.fresh else ()),
    )
    if a.blind:
        led(a)


def led(a: argparse.Namespace) -> dict[str, int]:
    """Train tasks led per candidate (GEPA's per-task frontier, as the
    loop's ``select`` counts them), read off the stored train rows only;
    written to out/led.json. No holdout row is opened."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("docx_loop", LOOP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.METRIC = f"marker:{HEADLINE}"
    o = out_dir(a.dry_run)
    ledger = [json.loads(line) for line in (o / "ledger.jsonl").read_text().splitlines() if line]
    counts = mod.tasks_led(ledger, model=_models(a)[0], out=o)
    best = (
        max(
            (e for e in ledger[1:]),
            key=lambda e: (counts[e["candidate"]], e["train"]["pass_at_1"] or 0),
        )["candidate"]
        if len(ledger) > 1
        else ledger[0]["candidate"]
    )
    (o / "led.json").write_text(json.dumps({"tasks_led": counts, "leader": best}, indent=2))
    print("train tasks led: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f" -> {best}")
    return counts


def _replay_dir(o: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("tasks.jsonl", "split.json"):
        shutil.copy(o / f, dest / f)


def noise(a: argparse.Namespace) -> None:
    """The baseline twice more, alone, on the same frozen tasks, on the
    search model, with the sampling seeds moved (``DOCX_SALT``)."""
    o = out_dir(a.dry_run)
    only = o / "noise-candidates"
    only.mkdir(exist_ok=True)
    shutil.copy(HERE / a.candidates / BASELINE, only / BASELINE)
    for salt in range(1, RERUNS):
        dest = o / f"noise-{salt}"
        if (dest / "ledger.jsonl").exists() and not a.force:
            print(f"noise run {salt}: cached")
            continue
        _replay_dir(o, dest)
        loop(a, out=dest, candidates=only, models=_models(a)[:1], salt=salt)


def pick_of(o: Path) -> str:
    if (o / "selected.json").exists():
        sel = json.loads((o / "selected.json").read_text())
        return sel.get("selected") or sel.get("best_on_train") or BASELINE
    if (o / "led.json").exists():
        return json.loads((o / "led.json").read_text())["leader"]
    return BASELINE


def gate(a: argparse.Namespace) -> None:
    """The recipe's gate on the baseline and the pick, over every model:
    --select (holdout and held-out models, intervals above zero, matched
    cost), attribution, then --prune."""
    o = out_dir(a.dry_run)
    pick = pick_of(o)
    folder = o / "gate-candidates"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    for f in (BASELINE, pick):
        shutil.copy(HERE / a.candidates / f, folder / f)
    dest = o / "gate"
    _replay_dir(o, dest)
    # the search model's rows for the baseline and the pick are already paid
    # for: copy them under the gate's traces so --reuse keeps them (the
    # fingerprint and model match) and only the held-out models run live
    for f in (BASELINE, pick):
        src = o / "traces" / Path(f).stem / "rows.jsonl"
        if src.exists():
            (dest / "traces" / Path(f).stem).mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest / "traces" / Path(f).stem / "rows.jsonl")
    loop(
        a,
        out=dest,
        candidates=folder,
        models=_models(a),
        extra=("--select", "--prune", "--cost-margin", "0"),
    )


# ---------------------------------------------------------------- reading what the recipe wrote


def rows_of(o: Path, cand: str, model: str, split: str | None = None) -> list[dict[str, Any]]:
    p = o / "traces" / Path(cand).stem / "rows.jsonl"
    if not p.exists():
        return []
    names = {model, _name(model)}
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    return [
        r for r in rows if r["harness"]["model"] in names and (split is None or r["split"] == split)
    ]


def per_task(rows: list[dict[str, Any]], marker: str) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in rows:
        v = (r.get("markers") or {}).get(marker)
        if v is not None:
            by.setdefault(r["scenario_id"], []).append(float(v))
    return {k: sum(v) / len(v) for k, v in by.items()}


def boot_ci(values: list[float]) -> list[float]:
    """Percentile bootstrap over documents; when every document agrees, the
    exact one-sided bound 1 - 0.025**(1/n) (Clopper and Pearson 1934), never
    a zero-width interval."""
    n = len(values)
    rng = random.Random(BOOT_SEED)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    lo, hi = means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1]
    if hi <= lo:
        m = sum(values) / n
        b = 1 - 0.025 ** (1 / n)
        return [max(0.0, m - b), m] if m >= 1 else [m, min(1.0, m + b)]
    return [lo, hi]


def stat(rows: list[dict[str, Any]], marker: str) -> dict[str, Any] | None:
    t = per_task(rows, marker)
    if not t:
        return None
    v = list(t.values())
    return {"mean": sum(v) / len(v), "ci": boot_ci(v), "n": len(v)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    out: dict[str, Any] = {k: stat(rows, k) for k in (HEADLINE, SECOND, *EXTRA)}
    rw = {}
    for r in rows:
        rw.setdefault(r["scenario_id"], []).append(float(r.get("reward") or 0))
    ex = [sum(v) / len(v) for v in rw.values()]
    out["doc_exact"] = {"mean": sum(ex) / len(ex), "ci": boot_ci(ex), "n": len(ex)}
    out["criteria"] = {
        k.split(":", 1)[1]: stat(rows, k)
        for k in sorted(
            {m for r in rows for m in (r.get("markers") or {}) if m.startswith("rubric:")}
        )
    }
    out["fields"] = {f"{t}.{f}": stat(rows, f"{t}.{f}") for t, f in docs.FIELDS}
    out["fields"] = {k: v for k, v in out["fields"].items() if v}
    out["slices"] = {}
    for dim in ("difficulty", "doc_type"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(str((r.get("scenario_dimensions") or {}).get(dim)), []).append(r)
        out["slices"][dim] = {
            g: {mk: stat(rs, mk) for mk in (HEADLINE, SECOND)} for g, rs in sorted(groups.items())
        }
    usage = [r.get("usage") or {} for r in rows]
    toks = [float(u.get("input_tokens") or 0) + float(u.get("output_tokens") or 0) for u in usage]
    out.update(
        rollouts=len(rows),
        judged=sum(1 for r in rows if SECOND in (r.get("markers") or {})),
        tokens_per_rollout=sum(toks) / len(toks),
        tool_calls_per_rollout=sum(
            sum(1 for s in r["steps"] if isinstance(s, dict) and s.get("tool")) for r in rows
        )
        / len(rows),
        parse_fail=sum(1 for r in rows if not (r.get("markers") or {}).get("parsed")) / len(rows),
        truncated=sum(1 for r in rows if r.get("finish_reason") == "length"),
    )
    return out


def paired(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    marker: str,
    nf: dict[str, Any] | None,
) -> dict[str, Any]:
    """``wai.compare`` on one metric, paired by document, with the floor;
    plus the difference spread the MDE needs (MDE = 2.8 sd / sqrt(n))."""
    run_std = (nf or {}).get("run_std", {}).get(marker) if nf else None
    rep = wai.compare(
        before,
        after,
        target=f"marker:{marker}",
        run_std=run_std,
        run_std_runs=RERUNS if run_std is not None else None,
    )
    m = rep["metrics"][f"marker:{marker}"]
    a, b = per_task(before, marker), per_task(after, marker)
    diffs = [b[k] - a[k] for k in b if k in a]
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    band = (nf or {}).get("noise_band", {}).get(marker) if nf else None
    ci = list(m["ci95"]) if m.get("ci95") else [None, None]
    return {
        "delta": m["delta"],
        "ci": ci,
        "p": m.get("p_value"),
        "n": m.get("n_paired"),
        "sd_diff": sd,
        "mde": MDE_Z * sd / math.sqrt(len(diffs)) if diffs else None,
        "noise_band": band,
        "docs_up": sum(1 for d in diffs if d > 0),
        "docs_down": sum(1 for d in diffs if d < 0),
        "clears": ci[0] is not None and ci[0] > 0 and (band is None or m["delta"] > band),
    }


def noise_floor(o: Path, model: str, split: str) -> dict[str, Any] | None:
    runs = [rows_of(o, BASELINE, model, split)] + [
        rows_of(o / f"noise-{s}", BASELINE, model, split) for s in range(1, RERUNS)
    ]
    if any(not r for r in runs):
        return None
    out: dict[str, Any] = {
        "split": split,
        "runs": RERUNS,
        "rule": "t(df=2) x run_std x sqrt(2) (wai.eval_variance)",
        "means": {},
        "run_std": {},
        "noise_band": {},
    }
    for mk in (HEADLINE, SECOND):
        v = wai.eval_variance(*runs, metric=f"marker:{mk}")
        out["means"][mk] = list(v["means"].values())
        out["run_std"][mk] = v["run_std"]
        out["noise_band"][mk] = v["noise_band"]
    return out


def spend(o: Path) -> dict[str, Any]:
    """Modal: each live loop call is one container from its start (the cold
    start inside it) to its end, concurrent calls side by side, plus one
    idle tail per run of calls that sit closer than the tail. Anthropic: every judge call actually made (cache
    misses), tokens estimated from its characters. The bills are the check."""
    p = o / "spend.jsonl"
    calls = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    per_model: dict[str, list[list[float]]] = {}
    busy: dict[str, float] = {}
    for c in sorted((c for c in calls if not c.get("dry_run")), key=lambda c: c["start"]):
        for m in c["models"]:
            # every loop call holds its own container (64 rollouts in flight fill
            # one), so concurrent calls are billed side by side, not merged
            busy[short(m)] = busy.get(short(m), 0.0) + (c["end"] - c["start"])
            sess = per_model.setdefault(short(m), [])
            if sess and c["start"] - sess[-1][1] < SCALEDOWN_S:
                sess[-1][1] = max(sess[-1][1], c["end"])
            else:
                sess.append([c["start"], c["end"]])
    gpus = json.loads((o / "gpus.json").read_text()) if (o / "gpus.json").exists() else {}
    modal = {}
    for m, sess in per_model.items():
        billed = busy[m] + SCALEDOWN_S * len(sess)
        gpu = gpus.get(m, "L40S")
        modal[m] = {
            "gpu": gpu,
            "sessions": len(sess),
            "billed_minutes_est": round(billed / 60, 1),
            "usd_est": round(billed / 3600 * GPU_USD_PER_HOUR[gpu], 2),
        }
    jc = o / "judge_cache.jsonl"
    judged = (
        [json.loads(line) for line in jc.read_text().splitlines() if line.strip()]
        if jc.exists()
        else []
    )
    tin = sum(JUDGE_SYSTEM_TOKENS + j.get("chars_in", 0) / CHARS_PER_TOKEN for j in judged)
    tout = JUDGE_OUT_TOKENS * len(judged)
    anth = tin / 1e6 * HAIKU_USD_PER_MTOK[0] + tout / 1e6 * HAIKU_USD_PER_MTOK[1]
    manual = (
        json.loads((o / "spend_manual.json").read_text())
        if (o / "spend_manual.json").exists()
        else []
    )
    if manual:
        modal["before the loop"] = {
            "usd_est": round(sum(x["usd_est"] for x in manual), 2),
            "what": "; ".join(x["what"] for x in manual),
        }
    m_usd = sum(v["usd_est"] for v in modal.values())
    return {
        "modal": modal,
        "modal_usd_est": round(m_usd, 2),
        "anthropic": {
            "judge_calls": len(judged),
            "input_tokens_est": int(tin),
            "output_tokens_est": tout,
            "usd_est": round(anth, 2),
        },
        "total_usd_est": round(m_usd + anth, 2),
        "gpu": "L40S",
        "usd_per_hour": L40S_USD_PER_HOUR,
    }


# ---------------------------------------------------------------- regrade and rejudge


def _all_row_files(o: Path) -> list[Path]:
    return sorted(o.glob("traces/*/rows.jsonl")) + sorted(o.glob("*/traces/*/rows.jsonl"))


def _task(a: argparse.Namespace) -> dict[str, Any]:
    return runpy.run_path(str(HERE / a.task))


def regrade(a: argparse.Namespace) -> None:
    """Re-run the program grader over every stored row (no model call): a
    grader change is applied to rows already paid for. Judge markers stay."""
    task = _task(a)
    n = 0
    for f in _all_row_files(out_dir(a.dry_run)):
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        for r in rows:
            g = task["grade_row"](r, with_quality=False)
            keep = {
                k: v
                for k, v in (r.get("markers") or {}).items()
                if k.startswith("rubric:") or k == "quality"
            }
            r["markers"] = {**g["markers"], **keep}
            r["reward"], r["reason"] = g["reward"], g["reason"]
        f.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        n += len(rows)
    print(f"regraded {n} rows with the program grader")


def rejudge(a: argparse.Namespace) -> None:
    """Add the quality judge's markers to every stored row that lacks them
    (verdicts cached by rubric version, prompt and reply), off the GPU path."""
    from concurrent.futures import ThreadPoolExecutor

    task = _task(a)
    version = task["RUBRIC"].version
    if a.dry_run:
        os.environ["DOCX_JUDGE"] = "scripted"
    os.environ.setdefault("DOCX_JUDGE_CACHE", str(out_dir(a.dry_run) / "judge_cache.jsonl"))
    full = [x.strip() for x in (a.full or "").split(",") if x.strip()]
    for f in _all_row_files(out_dir(a.dry_run)):
        rel = f.relative_to(out_dir(a.dry_run)).as_posix()
        if a.only and not any(x in rel for x in a.only.split(",")):
            continue
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        todo = [r for r in rows if (r.get("markers") or {}).get("rubric_version") != version]
        if a.per_doc and not any(rel.startswith(x) for x in full):
            # the judge budget: the first --per-doc rollouts of each document
            # (by rollout index, never by score); the candidates in --full get all k
            seen: dict[str, int] = {}
            keep = []
            for r in todo:
                n = seen.get(r["scenario_id"], 0)
                seen[r["scenario_id"]] = n + 1
                if n < a.per_doc:
                    keep.append(r)
            todo = keep

        def one(r: dict[str, Any]) -> None:
            ask = task["BY_PROMPT"].get(r["prompt"])
            if ask is None:
                return
            answer, _ = task["docs"].parse_answer(r["final_text"])
            q = task["quality"](r, ask, answer)
            if q.get("reward") is None:
                return
            m = {k: v for k, v in r["markers"].items() if not k.startswith("rubric:")}
            m.update(q.get("markers") or {})
            m["quality"] = float(q["reward"])
            m["rubric_version"] = version
            r["markers"] = m

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(one, todo))
        f.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        print(f"{rel}: judged {len(todo)} rows (rubric {version})")


# ---------------------------------------------------------------- the judge audit


def _wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    """Wilson 95% lower bound on agreement (Wilson 1927), the bound compare_judges gates on."""
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (
        1 + z * z / n
    )


def _kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def audit(a: argparse.Namespace) -> dict[str, Any]:
    """The judge against program gold on the criteria a program can check
    (labels ``kind="program"``, never human), and the rows a person must
    label for the one it cannot (skills/audit-your-judge)."""
    task = runpy.run_path(str(HERE / a.task))
    o = out_dir(a.dry_run)
    m = _models(a)[0]
    rows = [
        r
        for p in sorted((o / "traces").glob("*/rows.jsonl"))
        for r in rows_of(o, p.parent.name + ".py", m)
        if SECOND in (r.get("markers") or {})
    ]
    uniq: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        uniq.setdefault((r["prompt"], r["final_text"]), r)
    rows = list(uniq.values())  # one verdict per distinct answer: the cache made the rest copies
    report: dict[str, Any] = {
        "judge": task["JUDGE_MODEL"] if not a.dry_run else "scripted",
        "rubric_version": task["RUBRIC"].version,
        "n_distinct_answers": len(rows),
        "criteria": {},
    }
    for slug in task["PROGRAM_CHECKABLE"]:
        pairs = [
            (int(r["markers"][f"rubric:{slug}"]), int(r["markers"][f"program:{slug}"]))
            for r in rows
            if f"rubric:{slug}" in r["markers"] and f"program:{slug}" in r["markers"]
        ]
        if not pairs:
            continue
        j, g = [p[0] for p in pairs], [p[1] for p in pairs]
        # the library's own reading: judge verdict as reward, program verdict as gold of kind program
        lib_rows = [{"rollout_id": f"a{i}", "reward": float(x)} for i, x in enumerate(j)]
        lib_rows, _ = wai.simulations.attach_labels(
            lib_rows,
            [{"key": f"a{i}", "label": y} for i, y in enumerate(g)],
            annotator="program",
            kind="program",
        )
        agree = wai.simulations.judge_agreement(lib_rows)
        neg = [x for x, y in zip(j, g) if y == 0]
        report["criteria"][slug] = {
            "n": len(pairs),
            "agreement": sum(x == y for x, y in pairs) / len(pairs),
            "agreement_wilson_lb": _wilson_lb(sum(x == y for x, y in pairs), len(pairs)),
            "kappa": _kappa(j, g),
            "false_pass_rate": (sum(neg) / len(neg)) if neg else None,
            "n_program_fail": len(neg),
            "judge_pass_rate": sum(j) / len(j),
            "program_pass_rate": sum(g) / len(g),
            "judge_agreement_lib": {
                k: agree.get(k) for k in ("agreement", "kappa", "n") if k in agree
            },
            "gold_kind": "program",
            "ok": None,
        }
    for c in report["criteria"].values():
        if "agreement" in c:
            c["ok"] = c["agreement_wilson_lb"] >= 0.8 and (c["kappa"] or 0) >= 0.6
    for slug in task["QUALITY_ONLY"]:
        report["criteria"][slug] = {
            "status": f"judge unaudited on quality: {LABELS_N} rows awaiting labels"
        }
    ordered = sorted(rows, key=lambda r: (r["markers"][SECOND], r["scenario_id"]))
    step = max(1, len(ordered) // LABELS_N)
    chosen = ordered[::step][:LABELS_N]
    with (HERE / "labels_needed.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(chosen):
            fh.write(
                json.dumps(
                    {
                        "key": f"L{i:02d}",
                        "doc_id": r["scenario_id"],
                        "prompt": r["prompt"],
                        "final_text": r["final_text"],
                        "criteria": {c.slug: c.description for c in task["RUBRIC"].criteria},
                        "question": "For each criterion, does the final reply meet it? Label 1 or 0; "
                        "the judge's verdict is withheld.",
                        "labels": {c.slug: None for c in task["RUBRIC"].criteria},
                        "annotator": None,
                    }
                )
                + "\n"
            )
    report["labels_needed"] = {
        "file": "labels_needed.jsonl",
        "n": len(chosen),
        "spread": "every k-th distinct answer sorted by the judge's quality score",
    }
    (o / "audit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for slug, c in report["criteria"].items():
        if "agreement" in c:
            print(
                f"audit {slug:<26} agreement {c['agreement']:.2f}  kappa "
                f"{c['kappa'] if c['kappa'] is None else round(c['kappa'], 2)}  false-pass "
                f"{c['false_pass_rate'] if c['false_pass_rate'] is None else round(c['false_pass_rate'], 2)}"
                f" (n={c['n']}, program fails {c['n_program_fail']})"
            )
        else:
            print(f"audit {slug:<26} {c['status']}")
    return report


# ---------------------------------------------------------------- the report


def sdk_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The SDK's own readings of one candidate's train rows: doc-exact pass@1
    with its interval over tasks (``pass_at``), reply length (``length_report``)
    and the feature most tied to the reward (``hack_scan``): is a gain a
    behavior, or length or format?"""
    out: dict[str, Any] = {}
    try:
        pa = wai.pass_at(rows)
        out["doc_exact_pass_at_1"] = {
            "mean": pa.pass_at_1,
            "ci95": list(pa.ci95) if pa.ci95 else None,
            "n_tasks": pa.n_groups,
        }
    except Exception as exc:
        out["doc_exact_pass_at_1"] = {"error": str(exc)[:200]}
    try:
        lr = wai.simulations.length_report(rows)
        out["length_report"] = {k: v for k, v in lr.items() if not isinstance(v, (list, dict))}
    except Exception as exc:
        out["length_report"] = {"error": str(exc)[:200]}
    try:
        # the grader's own markers are the reward's parts, not a hack: scan the text features only
        hs = wai.hack_scan([{**r, "markers": {}} for r in rows], n_perm=50)
        top = (hs.get("top") or hs.get("features") or [None])[0] if isinstance(hs, dict) else None
        out["hack_scan"] = {
            "top": top,
            "verdict": hs.get("verdict") if isinstance(hs, dict) else None,
            "note": hs.get("note") if isinstance(hs, dict) else None,
        }
    except Exception as exc:
        out["hack_scan"] = {"error": str(exc)[:200]}
    return out


def pts(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def _ci(s: dict[str, Any] | None) -> str:
    return "n/a" if not s else f"{pts(s['mean'])} [{pts(s['ci'][0])}, {pts(s['ci'][1])}]"


def _d(d: dict[str, Any] | None) -> str:
    if not d or d["ci"][0] is None:
        return "n/a"
    return f"{100 * d['delta']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]"


def build_results(a: argparse.Namespace) -> dict[str, Any]:
    o = out_dir(a.dry_run)
    models = _models(a)
    sm = models[0]
    ledger = [json.loads(line) for line in (o / "ledger.jsonl").read_text().splitlines() if line]
    if (o / "selected.json").exists() and not a.blind:
        selected = json.loads((o / "selected.json").read_text())
    else:
        selected = json.loads((o / "led.json").read_text()) if (o / "led.json").exists() else {}
        selected.setdefault(
            "reason", "search in progress: holdout withheld from the proposer (--blind)"
        )
    split = json.loads((o / "split.json").read_text())
    frozen = json.loads((HERE / "frozen.json").read_text())
    learned = (
        json.loads((HERE / "learned.json").read_text()) if (HERE / "learned.json").exists() else {}
    )
    # --blind: no holdout row is opened until the pick; every holdout field is None
    splits = ("train",) if a.blind else ("train", "holdout")
    nf = {s: (noise_floor(o, sm, s) if s in splits else None) for s in ("train", "holdout")}
    base = {s: rows_of(o, BASELINE, sm, s) for s in splits}
    cands = []
    for e in ledger:
        c = e["candidate"]
        entry: dict[str, Any] = {
            "candidate": Path(c).stem,
            "fingerprint": e["fingerprint"],
            "led_train_tasks": (selected.get("tasks_led") or {}).get(c),
            "learned": learned.get(Path(c).stem),
            "holdout": None,
        }
        for s in splits:
            rows = rows_of(o, c, sm, s)
            entry[s] = summarize(rows)
            if c != BASELINE and rows:
                entry[s]["vs_baseline"] = {
                    mk: paired(base[s], rows, mk, nf[s]) for mk in (HEADLINE, SECOND)
                }
        tr = rows_of(o, c, sm, "train")
        if tr:
            entry["sdk_checks"] = sdk_checks(tr)
        entry["cost_ratio_tokens"] = (
            entry["train"]["tokens_per_rollout"] / cands[0]["train"]["tokens_per_rollout"]
            if cands
            else 1.0
        )
        cands.append(entry)
    pick = Path(pick_of(o)).stem
    pk = next(x for x in cands if x["candidate"] == pick)
    mde_gate = None
    if pick != Path(BASELINE).stem:
        d = pk["train"]["vs_baseline"][HEADLINE]
        mde_gate = {
            "pick": pick,
            "train_delta": d["delta"],
            "mde": d["mde"],
            "noise_band": d["noise_band"],
            "passes": d["delta"] >= (d["mde"] or 0)
            and (d["noise_band"] is None or d["delta"] > d["noise_band"]),
        }
    sizing = None
    if pick != Path(BASELINE).stem and pk.get("holdout"):
        d = (
            pk["holdout"]["vs_baseline"][HEADLINE]
            if pk.get("holdout", {}).get("vs_baseline")
            else None
        )
        if d and d["sd_diff"]:
            hs = wai.holdout_size(max(abs(d["delta"]), 0.01), task_std=d["sd_diff"], k=K)
            sizing = {
                "effect": d["delta"],
                "task_std": d["sd_diff"],
                "report": {k: v for k, v in dict(hs).items() if not isinstance(v, (list, dict))},
            }
    tasks = [
        json.loads(line) for line in (o / "tasks.jsonl").read_text().splitlines() if line.strip()
    ]
    tkey = {t.get("task_id") or t.get("scenario_id"): t for t in tasks}
    tr_t = [dict(tkey[k]) for k in split["train"] if k in tkey]
    ho_t = [dict(tkey[k]) for k in split["holdout"] if k in tkey]
    try:
        _, dc = wai.decontaminate(tr_t, against=ho_t)
        decon = {
            "n_train": len(tr_t),
            "contaminated_by_8gram_rule": dc.get("n_contaminated"),
            "note": "templated documents share the schema text by design; the split is by ask-id hash "
            "and the proposer reads train rows only",
        }
    except Exception as exc:
        decon = {"error": str(exc)[:200]}
    g = o / "gate"
    gate_res = None
    if (g / "selected.json").exists():
        gsel = json.loads((g / "selected.json").read_text())
        gate_res = {"selected": gsel, "models": {}}
        for m in models:
            b, p = rows_of(g, BASELINE, m, "holdout"), rows_of(g, pick + ".py", m, "holdout")
            gate_res["models"][short(m)] = {
                "baseline": summarize(b),
                "pick": summarize(p),
                "vs_baseline": {
                    mk: paired(b, p, mk, nf["holdout"] if m == sm else None)
                    for mk in (HEADLINE, SECOND)
                }
                if b and p
                else None,
            }
        for f in ("pruned.json", "pruned_gate.json"):
            if (g / f).exists():
                gate_res[f.split(".")[0]] = json.loads((g / f).read_text())
    audit_res = json.loads((o / "audit.json").read_text()) if (o / "audit.json").exists() else None
    return {
        "recipe": "doc-extraction-harness",
        "paper": "https://arxiv.org/abs/2603.28052",
        "method": "Meta-Harness loop (loop.py, adapted from recipes/papers/meta-harness/run.py; --task task_docs.py --metric marker:field_f1 "
        "--blind --k 4), coding agent as proposer; gate: --select --prune on baseline and pick over every model",
        "search_model": short(sm) + " (no adapter)",
        "held_out_models": [short(m) + " (no adapter)" for m in models[1:]],
        "served_by": "vLLM 0.10.0, one Modal L40S per model, apps docx-serve-<slug> (serve_modal.py)",
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens_per_turn": 1024,
            "seed": "sha256(prompt:DOCX_SALT:n)[:8] + 7919*turn per request, n = the rollout's index "
            "for that prompt (0..k-1)",
        },
        "headline": "field_f1: micro field-level F1 per document (SROIE, Huang et al. 2019; CORD, Park et al. "
        "2019), typed comparison after normalization, ANLS on name fields (DocVQA, Biten et al. 2019, "
        "arXiv:1907.00490); mean over k=4 rollouts, then over documents; 95% bootstrap interval over documents",
        "beside_headline": [
            "field_precision, field_recall, anls_names, null_acc, field_acc (exact, nulls included; continuity)",
            "response_quality: the Haiku 4.5 rubric judge (rubric v2) score, mean of five criteria met; "
            "see audit for which criteria clear the floors",
            "doc_exact: every field right (a conjunction, reported, never the headline)",
        ],
        "judge": {
            "model": "anthropic:claude-haiku-4-5-20251001",
            "temperature": 0.0,
            "rubric_version": runpy.run_path(str(HERE / a.task))["RUBRIC"].version,
            "sees": "the document and the final reply, never the gold",
        },
        "audit": audit_res,
        "test_versions": {
            "train": frozen["selection"]["test_version"],
            "holdout": frozen["holdout"]["test_version"],
        },
        "n": {"train": frozen["selection"]["n"], "holdout": frozen["holdout"]["n"]},
        "k": K,
        "unit": "documents (asks); k=4 rollouts each, averaged per document before any interval",
        "split": split["how"],
        "seeds": {
            "data": frozen["data_seed"],
            "bootstrap": BOOT_SEED,
            "noise_salts": list(range(RERUNS)),
        },
        "noise_floor": nf,
        "candidates": cands,
        "slice_n": {
            f"{dim}.{key}": v[HEADLINE]["n"]
            for dim, d in ((cands[0]["train"] or {}).get("slices") or {}).items()
            for key, v in d.items()
            if v.get(HEADLINE)
        },
        "holdout_slice_n": {
            f"{dim}.{key}": v[HEADLINE]["n"]
            for dim, d in ((cands[0].get("holdout") or {}).get("slices") or {}).items()
            for key, v in d.items()
            if v.get(HEADLINE)
        },
        "search_gate": selected,
        "pick": pick,
        "mde_gate": mde_gate,
        "holdout_size": sizing,
        "decontaminate": decon,
        "final_gate": gate_res,
        "spend": spend(o),
        "whileai": wai.__version__,
        "python": sys.version.split()[0],
        "verified": date.today().isoformat(),
        "dry_run": a.dry_run,
        "blind": a.blind,
        "blind_note": BLIND_NOTE,
    }


def report(a: argparse.Namespace) -> None:
    o = out_dir(a.dry_run)
    res = build_results(a)
    dest = o if a.dry_run else HERE
    (dest / "results.json").write_text(
        json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8"
    )
    chart(res, dest / "chart.html")
    ledger_md(res, dest / "ledger.md")
    print(f"wrote {dest.name}/results.json, chart.html, ledger.md")
    if a.post or a.dry_run:
        calls = post(res, o, dry=a.dry_run, models=_models(a))
        if a.dry_run:
            (o / "platform_calls.json").write_text(
                json.dumps(calls, indent=1, default=str), encoding="utf-8"
            )
            print(f"track() built {len(calls)} calls on a fake transport")
    for c in res["candidates"]:
        t, h = c["train"], c["holdout"]
        vb = (t.get("vs_baseline") or {}).get(HEADLINE)
        print(
            f"{c['candidate']:<24} field F1 train {_ci(t[HEADLINE])} holdout "
            f"{_ci(h[HEADLINE]) if h else 'withheld'} | quality train "
            f"{_ci(t[SECOND])} | tok/rollout {t['tokens_per_rollout']:.0f} | led {c['led_train_tasks']}"
            + (f" | vs v0 {_d(vb)} MDE {pts(vb['mde'])}" if vb else "")
        )
    for s, f in res["noise_floor"].items():
        if f:
            print(
                f"noise floor {s}: field F1 {pts(f['noise_band'][HEADLINE])}, quality {pts(f['noise_band'][SECOND])} points"
            )
    print("search gate:", res["search_gate"].get("reason"))
    if res["final_gate"]:
        print("final gate:", res["final_gate"]["selected"].get("reason"))
    print("spend:", json.dumps(res["spend"]))


def ledger_md(res: dict[str, Any], dest: Path) -> None:
    """Five lines per candidate (skills/manage-experiments)."""
    nf = res["noise_floor"]

    def band(s: str, mk: str) -> str:
        return pts(nf[s]["noise_band"][mk]) if nf.get(s) else "n/a"

    out = [
        "# Ledger\n",
        f"Train (selection) {res['test_versions']['train']}, n={res['n']['train']} documents; holdout "
        f"{res['test_versions']['holdout']}, n={res['n']['holdout']}; k={res['k']} rollouts each. Headline "
        f"field F1 (program vs gold, SROIE/CORD convention, ANLS on names), response quality (Haiku 4.5 rubric judge) beside it; points, 95% interval over "
        f"documents. Noise floor (t(df=2) x run_std x sqrt(2), three runs of 00_baseline): train field F1 "
        f"{band('train', HEADLINE)}, quality {band('train', SECOND)}; holdout field F1 {band('holdout', HEADLINE)}, "
        f"quality {band('holdout', SECOND)} points. {BLIND_NOTE}\n",
    ]
    for c in res["candidates"]:
        path = HERE / "candidates" / f"{c['candidate']}.py"
        mod = runpy.run_path(str(path))
        doc = (mod.get("__doc__") or "").strip().replace("\n", " ")
        t, h = c["train"], c["holdout"]
        vb = t.get("vs_baseline") or {}
        hold = (
            f"holdout field F1 {_ci(h[HEADLINE])}, quality {_ci(h[SECOND])}"
            if h
            else "holdout withheld until the pick (--blind)"
        )
        moved = (
            f"field F1 train {_ci(t[HEADLINE])}"
            + (
                f", {_d(vb.get(HEADLINE))} vs 00_baseline (MDE {pts(vb[HEADLINE]['mde'])}; {vb[HEADLINE]['docs_up']} "
                f"documents up, {vb[HEADLINE]['docs_down']} down)"
                if vb
                else ""
            )
            + f"; quality train {_ci(t[SECOND])}"
            + (f", {_d(vb.get(SECOND))}" if vb else "")
            + f"; {hold}"
            + f"; {c['cost_ratio_tokens']:.2f}x the baseline's tokens per rollout; leads {c['led_train_tasks']} "
            "train documents"
        )
        out += [
            f"## {c['candidate']} (`{c['fingerprint']}`)\n",
            f"- **Changed:** {doc}",
            f"- **Moved:** {moved}.",
            f"- **Why:** {mod.get('WHY', '')}.",
            f"- **Learned:** {c.get('learned') or 'not written yet'}",
            "- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "
            f'"$MODELS"` with `candidates/{c["candidate"]}.py` in place (tests {res["test_versions"]["train"]}/'
            f"{res['test_versions']['holdout']}, k={res['k']}, DOCX_SALT=0)\n",
        ]
    dest.write_text("\n".join(out), encoding="utf-8")


BLIND_NOTE = (
    "Blinding: the proposer (the coding agent) read train rows only. Round 1's holdout "
    "comparison (02_doc_is_defined vs 00_baseline) was printed by the loop's --select and read "
    "once, before candidate 03 was written. From round 2 the loop ran --propose only, but its "
    "per-candidate ledger line still printed the holdout mean (loop.py, fixed after round 2): "
    "03's holdout mean was seen once, after 04 was written and before 05. From round 3 no "
    "holdout number was printed, opened or scored until the pick, when the gate ran once."
)


def _panel(res: dict[str, Any], mk: str, title: str) -> str:
    cs = res["candidates"]
    W, H, L, R, T, B = 760, 330, 56, 16, 28, 96
    keys = [("train", "c1", "train (selection)"), ("holdout", "c2", "holdout")]
    vals = [v for c in cs for k, *_ in keys if c.get(k) and c[k][mk] for v in c[k][mk]["ci"]]
    lo_y = max(0.0, math.floor((min(vals) - 0.03) * 20) / 20)
    hi_y = min(1.0, math.ceil((max(vals) + 0.03) * 20) / 20)
    step = (W - L - R) / len(cs)

    def x(i: int, j: int) -> float:
        return L + (i + 0.5) * step + (j - 0.5) * min(14, step / 4)

    def y(v: float) -> float:
        return T + (hi_y - v) / (hi_y - lo_y) * (H - T - B)

    svg = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}" style="width:100%;max-width:{W}px">',
        f'<text x="{L}" y="16" class="ttl">{title}</text>',
    ]
    t = lo_y
    while t <= hi_y + 1e-9:
        svg.append(
            f'<line x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
            f'<text x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end" class="axis">{100 * t:.0f}</text>'
        )
        t += 0.05
    for k, col, label in keys:
        f, b0 = res["noise_floor"].get(k), cs[0].get(k)
        if f and b0 and b0[mk]:
            m, band = b0[mk]["mean"], f["noise_band"][mk]
            y1, y2 = y(min(hi_y, m + band)), y(max(lo_y, m - band))
            svg.append(
                f'<rect x="{L}" width="{W - L - R}" y="{y1:.1f}" height="{y2 - y1:.1f}" class="band {col}">'
                f"<title>{label}: baseline {100 * m:.1f} ± noise floor {100 * band:.1f}</title></rect>"
            )
    for i, c in enumerate(cs):
        for j, (k, col, label) in enumerate(keys):
            s = (c.get(k) or {}).get(mk)
            if not s:
                continue
            xi, yi = x(i, j), y(s["mean"])
            tip = f"{c['candidate']}, {label}: {100 * s['mean']:.1f} [{100 * s['ci'][0]:.1f}, {100 * s['ci'][1]:.1f}], n={s['n']}"
            svg.append(
                f'<line x1="{xi:.1f}" x2="{xi:.1f}" y1="{y(s["ci"][0]):.1f}" y2="{y(s["ci"][1]):.1f}" class="bar {col}"/>'
            )
            svg.append(
                f'<circle cx="{xi:.1f}" cy="{yi:.1f}" r="5" class="pt {col}"><title>{tip}</title></circle>'
                if j == 0
                else f'<rect x="{xi - 4.5:.1f}" y="{yi - 4.5:.1f}" width="9" height="9" class="pt {col}" '
                f'transform="rotate(45 {xi:.1f} {yi:.1f})"><title>{tip}</title></rect>'
            )
        mark = " ★" if c["candidate"] == res["pick"] else ""
        svg.append(
            f'<text x="{x(i, 0.5):.1f}" y="{H - B + 16}" class="axis" text-anchor="end" '
            f'transform="rotate(-30 {x(i, 0.5):.1f} {H - B + 16})">{c["candidate"]}{mark}</text>'
        )
    svg.append("</svg>")
    return "".join(svg)


def chart(res: dict[str, Any], dest: Path) -> None:
    """Candidate index against the train and holdout scores with 95% bars and
    the baseline's noise band, one panel per metric (one axis each)."""
    nf = res["noise_floor"]
    rows = "".join(
        f"<tr><td>{c['candidate']}{' ★' if c['candidate'] == res['pick'] else ''}</td>"
        f"<td>{_ci(c['train'][HEADLINE])}</td><td>{_ci(c['holdout'][HEADLINE]) if c['holdout'] else 'withheld'}</td>"
        f"<td>{_ci(c['train'][SECOND])}</td><td>{_ci(c['holdout'][SECOND]) if c['holdout'] else 'withheld'}</td>"
        f"<td>{c['cost_ratio_tokens']:.2f}x</td><td>{c['led_train_tasks']}</td></tr>"
        for c in res["candidates"]
    )
    fg = res.get("final_gate")
    gate_rows = ""
    if fg:
        for m, v in fg["models"].items():
            vb = (v.get("vs_baseline") or {}).get(HEADLINE)
            gate_rows += (
                f"<tr><td>{m}</td><td>{_ci(v['baseline'][HEADLINE]) if v['baseline'] else 'n/a'}</td>"
                f"<td>{_ci(v['pick'][HEADLINE]) if v['pick'] else 'n/a'}</td><td>{_d(vb)}</td></tr>"
            )
    fb = (
        f"train field F1 {pts(nf['train']['noise_band'][HEADLINE])}, holdout field F1 "
        f"{pts(nf['holdout']['noise_band'][HEADLINE])} points"
        if nf.get("train") and nf.get("holdout")
        else "not measured"
    )
    html = f"""<title>Doc extraction climb</title>
<style>
.viz {{ --s:#fcfcfb; --t1:#0b0b0b; --t2:#52514e; --grid:#e4e3df; --c1:#2a78d6; --c2:#eb6834;
  --b1:rgba(42,120,214,.10); --b2:rgba(235,104,52,.10); background:var(--s); color:var(--t1); font:14px system-ui,sans-serif; padding:16px; }}
@media (prefers-color-scheme: dark) {{ .viz {{ --s:#1a1a19; --t1:#fff; --t2:#c3c2b7; --grid:#33322f; --c1:#3987e5; --c2:#d95926;
  --b1:rgba(57,135,229,.16); --b2:rgba(217,89,38,.16); }} }}
.grid {{ stroke:var(--grid) }} .axis {{ fill:var(--t2); font-size:11px }} .ttl {{ fill:var(--t1); font-size:13px; font-weight:600 }}
.band.c1 {{ fill:var(--b1) }} .band.c2 {{ fill:var(--b2) }} .bar {{ stroke-width:2 }}
.bar.c1 {{ stroke:var(--c1) }} .bar.c2 {{ stroke:var(--c2) }} .pt.c1 {{ fill:var(--c1) }} .pt.c2 {{ fill:var(--c2) }}
table {{ border-collapse:collapse; margin-top:12px }} td,th {{ padding:3px 10px; border-bottom:1px solid var(--grid); text-align:left }}
.legend span {{ margin-right:16px; color:var(--t2) }}
</style>
<div class="viz">
<h3 style="margin:0 0 4px">Document extraction on {res["search_model"]}: Meta-Harness climb</h3>
<div class="legend"><span style="color:var(--c1)">● train (selection)</span><span style="color:var(--c2)">◆ holdout</span>
<span>shaded: baseline ± noise floor ({fb})</span><span>★ pick</span></div>
{_panel(res, HEADLINE, "Field F1 (program vs gold; SROIE/CORD convention, ANLS on names), points: the headline")}
{_panel(res, SECOND, "Response quality (Haiku 4.5 rubric judge, failed its audit), points")}
<table><tr><th>candidate</th><th>field F1 train</th><th>field F1 holdout</th><th>quality train</th><th>quality holdout</th><th>tokens vs v0</th><th>train docs led</th></tr>{rows}</table>
{"<h4>Final gate: holdout field F1, baseline vs pick, per model</h4><table><tr><th>model</th><th>baseline</th><th>pick</th><th>paired delta [95%]</th></tr>" + gate_rows + "</table>" if gate_rows else ""}
<p style="color:var(--t2)">n = {res["n"]["train"]} train and {res["n"]["holdout"]} holdout documents ({res["test_versions"]["train"]}, {res["test_versions"]["holdout"]}), k={res["k"]}. Search gate: {res["search_gate"].get("reason")}</p>
</div>"""
    dest.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- the platform


SETTING = re.compile(r"(lr\d|\de-0\d|-s\d+\b|_s\d+$|_seed\d+|^h-[0-9a-f]{12}$|^v\d+$)")


def readback(tracked: Any) -> list[str]:
    """skills/manage-experiments, section 5, verbatim: what a teammate
    opening the page could not read. Empty means clean."""
    out = []
    if tracked.experiment() is None:
        out.append("no question posted: tracked.experiment(question=...)")
    for b in tracked.behaviors():
        if SETTING.search(b.name):
            out.append(f"behavior {b.name!r} names a seed or setting; put it in Optimizer(seed=)")
        if not (b.test_version or "").startswith("t-"):
            out.append(f"behavior {b.name!r}: test_version is not the asks' hash")
    pictured = {f.run for f in tracked.figures()}
    for r in tracked.runs():
        v, note = r["version"], r.get("notes") or ""
        trained = r.get("method") not in (None, "none", "eval")
        if SETTING.search(v):
            out.append(f"version {v!r} encodes settings; say the arm in words, numbers in record")
        if not ((r.get("record") or {}).get("data") or {}):
            out.append(f"run {v!r}: no data posted; RunRecord(data=Data(...))")
        for word in ("Changed", "Moved", "Why", "Learned", "Reproduce"):
            if trained and f"{word}:" not in note:
                out.append(f"run {v!r}: note has no '{word}:' line; run.note(...)")
        if trained and r["id"] not in pictured:
            out.append(f"run {v!r}: no picture; tracked.figure(name, fig, run=run)")
        for e in r.get("evals") or []:
            if 0 < e["score"] < 1:
                out.append(
                    f"{v} {e['behavior']}: {e['score']} reads as a fraction; post points, or fraction=True"
                )
    return out


class FakeTransport:
    """Records every call and answers the shapes ``track`` reads, as the
    skills' check.py files do; the dry run posts here."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.runs: dict[str, Any] = {}

    def __call__(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path, body))
        if path == "/agents":
            return {"id": body["id"], "name": body.get("name") or body["id"]}
        if path == "/runs" and method == "POST":
            rid = f"run_{len(self.runs):03d}"
            self.runs[rid] = body
            return {"id": rid, "version": body["version"]}
        if path.endswith("/evals"):
            return {"evals": body}
        if method == "GET":
            return {"behaviors": [], "runs": [], "figures": []}
        return {"ok": True}


def short_change(mod: dict[str, Any]) -> str:
    """The candidate's change in plain words: its docstring's first clause."""
    doc = (mod.get("__doc__") or "").strip().replace("\n", " ")
    head = doc.split(".")[0]
    return head.split(":", 1)[1].strip() if ":" in head else head


def results_table(res: dict[str, Any], split: str = "train") -> str:
    """Overall and per-type and per-difficulty field F1 by round, in points
    (markdown), so the numbers read even where the chart misbehaves."""
    cols = (
        [("all", None)]
        + [(t, ("doc_type", t)) for t in docs.DOC_TYPES]
        + [(d, ("difficulty", d)) for d in ("easy", "medium", "hard")]
    )
    short_names = {"purchase_order": "PO", "bank_statement": "bank", "claim_form": "claim"}
    head = "| round | " + " | ".join(short_names.get(k, k) for k, _ in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for c in res["candidates"]:
        t = c.get(split)
        if not t:
            continue
        cells = []
        for _, sl in cols:
            st = (
                t[HEADLINE]
                if sl is None
                else ((t.get("slices") or {}).get(sl[0]) or {}).get(sl[1], {}).get(HEADLINE)
            )
            cells.append(f"{100 * st['mean']:.1f}" if st else "-")
        tag = " (placebo)" if c["candidate"] == Path(PLACEBO).stem else ""
        lines.append(f"| {c['candidate']}{tag} | " + " | ".join(cells) + " |")
    n = {k: (res.get("slice_n") or {}).get(f"{sl[0]}.{sl[1]}") for k, sl in cols if sl}
    where = "search set" if split == "train" else "held-out set"
    return (
        f"## Field F1 by round, {where}, points (mean over documents)\n\n"
        + "\n".join(lines)
        + "\n\nDocuments per column: "
        + ", ".join(f"{short_names.get(k, k)} {v}" for k, v in n.items() if v)
        + "; 95% intervals are on each run's scores.\n\n"
    )


def dataset_notes(res: dict[str, Any]) -> str:
    """The dataset as the page describes it (markdown, under 4096 chars):
    the numbers by round first, then what the documents are, how hard, how
    they were split, one example."""
    asks = docs.build(json.loads((HERE / "frozen.json").read_text())["data_seed"])
    by_type = {t: sum(a["doc_type"] == t for a in asks) for t in docs.DOC_TYPES}
    diff = {d: sum(docs.difficulty(a) == d for a in asks) for d in ("easy", "medium", "hard")}
    ex = min(
        (a for a in asks if a["doc_type"] == "receipt" and a["split"] == "selection"),
        key=lambda a: len(a["text"]),
    )
    gold = json.dumps(ex["gold"])
    table = results_table(res)
    if res.get("final_gate"):
        table += results_table(res, "holdout")
    text = (
        table + "## The dataset\n\n"
        f"**{len(asks)} generated documents**, rendered from ground-truth records by a seeded program "
        f"(`docs.py`, data seed {res['seeds']['data']}), so every gold value is known exactly and the "
        "grader is a program: "
        + ", ".join(f"{n} {t.replace('_', ' ')}s" for t, n in by_type.items())
        + ".\n\n"
        "- **Layouts:** two to four per type (letter-style and tabular invoices, thermal receipts, "
        "multi-page purchase orders with page subtotals, bank statements with and without printed "
        "totals, claim forms with Last, First names and day-first dates).\n"
        "- **OCR noise** on labels and boilerplate (T0TAL, 5UBTOTAL, lnvoice, RECElPT, Bolance).\n"
        "- **Distractor numbers:** balance due, amount paid, previous balance, page subtotals, "
        "change due, quote refs, agent phone numbers.\n"
        "- **Missing fields** (gold null: no tax, no tip, no email, no delivery date) and "
        "**computed fields** the document does not print (due date from Net terms, bank totals from "
        "the transaction table, purchase-order line counts across pages).\n"
        f"- **Difficulty** from the generator's knobs (OCR damage, distractors, missing, computed, "
        f"layout): easy {diff['easy']}, medium {diff['medium']}, hard {diff['hard']}.\n"
        f"- **Split** by a hash of the ask id, never its position: {res['n']['train']} search "
        f"(selection) documents, test version `{res['test_versions']['train']}`; "
        f"{res['n']['holdout']} held-out, `{res['test_versions']['holdout']}` (frozen in "
        "`frozen.json` before any model answered). k=4 rollouts per document, per-rollout seeds.\n"
        "- **Headline:** field-level micro F1 per document (SROIE/CORD convention: exact match after "
        "normalization for money, dates, ids, digits, enums; ANLS ≥ 0.5 for names), averaged over "
        "rollouts then documents; 95% bootstrap interval over documents.\n\n"
        f"**Example** ({ex['id']}, {docs.difficulty(ex)}):\n\n```\n{ex['text'].strip()}\n```\n\n"
        f"Gold: `{gold}`\n"
    )
    if len(text) > 4096:  # the field's cap: shorten the example before anything else
        cut = len(text) - 4096 + 40
        text = text.replace(
            ex["text"].strip(),
            ex["text"].strip()[: max(200, len(ex["text"].strip()) - cut)] + "\n[...]",
        )
    return text[:4096]


_TASK_CACHE: dict[str, Any] = {}


def examples(
    o: Path,
    cand: str,
    model: str,
    split: str = "train",
    *,
    dim: str | None = None,
    key: str | None = None,
    metric: str = HEADLINE,
    n_fail: int = 14,
    n_pass: int = 6,
) -> list[Any]:
    """Up to 20 rollouts of one split (and one slice) for a score card,
    failures first (the lowest score, one per document), then passes: the
    document excerpt, the reply, the gold and the fields wrong, so a reader
    sees what a change fixed (skills/manage-experiments). ``metric`` is the
    headline (F1) or the judge's ``quality``."""
    from whileai.platform import Example

    rows = rows_of(o, cand, model, split)
    if dim and key:
        rows = [r for r in rows if str((r.get("scenario_dimensions") or {}).get(dim)) == key]
    if metric != HEADLINE:
        rows = [r for r in rows if metric in (r.get("markers") or {})]
    seen: set[str] = set()
    ordered = []
    for r in sorted(rows, key=lambda r: (r["markers"].get(metric, 0.0), r["scenario_id"])):
        if r["scenario_id"] in seen:
            continue
        seen.add(r["scenario_id"])
        ordered.append(r)
    fails = [r for r in ordered if r["markers"].get(metric, 0.0) < 1.0][:n_fail]
    passes = [r for r in ordered if r["markers"].get(metric, 0.0) >= 1.0][-n_pass:]
    if "task" not in _TASK_CACHE:
        _TASK_CACHE["task"] = runpy.run_path(str(HERE / "task_docs.py"))
    task = _TASK_CACHE["task"]
    out = []
    for r in fails + passes:
        ask = task["BY_PROMPT"].get(r["prompt"])
        if ask is None:
            continue
        f1 = r["markers"].get(metric, 0.0)
        wrong = [k.split(".", 1)[1] for k, v in r["markers"].items() if "." in k and v == 0.0]
        if metric != HEADLINE:
            missed = [
                k.split(":", 1)[1]
                for k, v in r["markers"].items()
                if k.startswith("rubric:") and v == 0.0
            ]
            why = (
                f"judge quality {100 * f1:.0f}: "
                + ("every criterion met" if not missed else "missed " + ", ".join(missed))
                + (f" | {(r.get('judge_reason') or '')[:200]}" if r.get("judge_reason") else "")
            )
        else:
            why = (
                "every field right"
                if f1 >= 1.0
                else (
                    ("no JSON in the final reply; " if not r["markers"].get("parsed") else "")
                    + ("cut at the token cap; " if r.get("finish_reason") == "length" else "")
                    + f"wrong: {', '.join(wrong)}"
                )
            )
        out.append(
            Example(
                prompt=f"[{ask['id']}, {ask['doc_type']}, {docs.difficulty(ask)}]\n"
                + ask["text"].strip()[:1000],
                reply=(r.get("final_text") or "")[-1150:],
                ok=f1 >= 1.0,
                why=why[:390],
                score=round(100 * f1, 1),
                tags={"doc_type": ask["doc_type"], "difficulty": docs.difficulty(ask)},
                reference=json.dumps(ask["gold"])[:1150],
            )
        )
    return out


def figure_by_type(res: dict[str, Any]) -> dict[str, Any]:
    """Field F1 per document type across rounds on the search set."""
    cs = res["candidates"]
    names = [
        c["candidate"] + (" (placebo)" if c["candidate"] == Path(PLACEBO).stem else "") for c in cs
    ]
    data = []
    for t in docs.DOC_TYPES:
        p = [
            ((c.get("train") or {}).get("slices") or {})
            .get("doc_type", {})
            .get(t, {})
            .get(HEADLINE)
            for c in cs
        ]
        data.append(
            {
                "type": "scatter",
                "mode": "markers+lines",
                "name": t.replace("_", " "),
                "x": names,
                "y": [round(100 * s["mean"], 1) if s else None for s in p],
                "error_y": {
                    "type": "data",
                    "symmetric": False,
                    "array": [round(100 * (s["ci"][1] - s["mean"]), 1) if s else None for s in p],
                    "arrayminus": [
                        round(100 * (s["mean"] - s["ci"][0]), 1) if s else None for s in p
                    ],
                },
            }
        )
    return {
        "data": data,
        "layout": {
            "title": "Field F1 by harness round and document type (search set)",
            "xaxis": {"title": "harness round"},
            "yaxis": {"title": "field F1, points (95% interval over documents)", "range": [0, 100]},
        },
    }


def figure(res: dict[str, Any]) -> dict[str, Any]:
    """The hill climb: train F1 with its interval per candidate in order, the
    placebo marked, the baseline's three-run noise band shaded; the holdout
    traces only once the gate has run (blind until then)."""
    cs = res["candidates"]
    names = [
        c["candidate"]
        + (" (placebo: rewording only)" if c["candidate"] == Path(PLACEBO).stem else "")
        + (" ★ pick" if c["candidate"] == res["pick"] and not res.get("blind") else "")
        for c in cs
    ]
    data = []
    for mk, label in ((HEADLINE, "field F1"), (SECOND, "judge quality (reference)")):
        for k in ("train",) if res.get("blind") else ("train", "holdout"):
            p = [(c.get(k) or {}).get(mk) for c in cs]
            if not any(p):
                continue
            data.append(
                {
                    "type": "scatter",
                    "mode": "markers+lines",
                    "name": f"{label}, {k}",
                    "x": names,
                    "y": [round(100 * s["mean"], 1) if s else None for s in p],
                    "error_y": {
                        "type": "data",
                        "symmetric": False,
                        "array": [
                            round(100 * (s["ci"][1] - s["mean"]), 1) if s else None for s in p
                        ],
                        "arrayminus": [
                            round(100 * (s["mean"] - s["ci"][0]), 1) if s else None for s in p
                        ],
                    },
                }
            )
    shapes, annotations = [], []
    f, b0 = res["noise_floor"].get("train"), cs[0]["train"][HEADLINE]
    if f:
        m, band = 100 * b0["mean"], 100 * f["noise_band"][HEADLINE]
        shapes.append(
            {
                "type": "rect",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": m - band,
                "y1": m + band,
                "opacity": 0.12,
                "line": {"width": 0},
            }
        )
        annotations.append(
            {
                "xref": "paper",
                "x": 0.01,
                "y": m + band,
                "text": f"baseline run-to-run noise (±{band:.1f} points, three runs)",
                "showarrow": False,
                "yanchor": "bottom",
                "xanchor": "left",
            }
        )
    return {
        "data": data,
        "layout": {
            "title": "Field F1 by harness round (search set)"
            + ("" if res.get("blind") else " and holdout"),
            "xaxis": {"title": "harness round"},
            "yaxis": {"title": "field F1, points (95% interval over documents)", "range": [0, 100]},
            "shapes": shapes,
            "annotations": annotations,
        },
    }


def post(res: dict[str, Any], o: Path, *, dry: bool, models: list[str]) -> list[Any]:
    """Every candidate as a run, once, as soon as it is scored; later calls
    refresh notes, behaviors and the figure. Dry runs post to a fake."""
    from whileai.platform import Behavior, Data, EvalSetup, Judge, RunRecord, track

    fake = FakeTransport() if dry else None
    tracked = track(
        "doc-extraction",
        model=res["search_model"].split(" ")[0],
        **({"transport": fake} if fake else {}),
    )
    tracked.experiment(
        question="Meta-Harness harness optimization: document extraction on Nemotron-8B. Can Meta-Harness "
        "(Lee et al. 2026) raise Nemotron-8B's document-extraction field F1 by changing only the "
        "harness? 200 generated invoices, receipts, POs, bank statements and claim forms; 100 to "
        "search on, 100 held out; weights untouched.",
        hypothesis="The model's extraction errors follow a few nameable rules (computed totals, distractor "
        "amounts, name order, missing fields); one prompt or loop change each fixes them.",
        method="Meta-Harness (Lee, Nair, Zhang, Lee, Khattab & Finn 2026, arXiv:2603.28052): an outer loop "
        "over harness code; a proposer reads every prior candidate's source, scores and worst traces, "
        "writes one change per candidate, and the pick must beat the baseline on held-out tasks and "
        "held-out models at no higher cost. Run with the whileai SDK (wai.Harness, the meta-harness "
        "loop, pass_at, eval_variance, attribute). One change per candidate, k=4 rollouts per document, "
        "field F1 against program gold; a placebo rewording as the control; the proposer blind to the "
        "holdout; the Haiku 4.5 rubric judge beside the headline for reference.",
        measure="field_f1 (SROIE/CORD micro F1, ANLS on names) on the frozen holdout, points, 95% interval over documents; response_quality (Haiku 4.5 rubric judge, audited per criterion, below the floors) beside it.",
        decide="The pick leads the most train documents and beats the baseline on the holdout and on held-out "
        "models with an interval above zero, at no more tokens per rollout.",
        notes=dataset_notes(res),
    )
    au = res.get("audit") or {}
    checked = [c for c in (au.get("criteria") or {}).values() if "agreement" in c]
    agreement = round(sum(c["agreement"] for c in checked) / len(checked), 3) if checked else None
    passing = [k for k, c in (au.get("criteria") or {}).items() if c.get("ok")]
    quality_judge = Judge(
        name=f"Haiku 4.5 rubric judge (v2); failed its audit against program gold: agreement {agreement}, "
        f"clears the floors on {', '.join(passing) or 'none'}; shown for reference",
        agreement=agreement,
    )
    program_judge = Judge(
        name="a program: field F1 against the generator's gold (SROIE/CORD; ANLS on names)"
    )
    # the headline first: the page opens on the first behavior
    for mk, split in (
        (HEADLINE, "train"),
        (HEADLINE, "holdout"),
        (SECOND, "train"),
        (SECOND, "holdout"),
    ):
        f = res["noise_floor"].get(split)
        n = res["n"][split]
        desc = (
            (
                f"Field F1 — search set ({n} docs). The headline: field-level F1 against the gold, a program (SROIE/CORD standard)."
                if split == "train"
                else f"Field F1 — held out ({n} docs, scored at the gate for the baseline and the pick only)."
            )
            if mk == HEADLINE
            else (
                f"Judge quality (Haiku) — failed audit, reference only. Search set ({n} docs)."
                if split == "train"
                else f"Judge quality (Haiku) — failed audit, reference only. Held out ({n} docs), scored at the gate."
            )
        )
        where = "Search set (101 docs)." if split == "train" else "Held out (99 docs)."
        tracked.behavior(
            Behavior(
                name=bname(mk, split),
                test_version=res["test_versions"][split],
                n=n,
                judge=program_judge if mk == HEADLINE else quality_judge,
                noise_floor=round(100 * f["noise_band"][mk], 1) if f else None,
                contamination=0,
                reward_is_judge=False,
                graded_by="program" if mk == HEADLINE else "judge",
                description=desc,
                rubric=(F1_RUBRIC if mk == HEADLINE else quality_rubric()) + " " + where,
            )
        )
    # the search-set slices: per difficulty and per document type (train rows only until the gate)
    slice_n = res.get("slice_n") or {}
    for dim, keys in (("difficulty", ("easy", "medium", "hard")), ("doc_type", docs.DOC_TYPES)):
        for key in keys:
            n = slice_n.get(f"{dim}.{key}")
            if not n:
                continue
            tracked.behavior(
                Behavior(
                    name=SLICE_NAMES[key],
                    test_version=res["test_versions"]["train"],
                    n=n,
                    judge=program_judge,
                    contamination=0,
                    reward_is_judge=False,
                    graded_by="program",
                    description=f"{SLICE_TITLES[key]} — search set ({n} docs); a slice of the headline"
                    + (", difficulty from the generator's knobs" if dim == "difficulty" else "")
                    + ".",
                    rubric=F1_RUBRIC
                    + f" {SLICE_TITLES[key].replace(' F1', '')} only; search set ({n} docs).",
                )
            )
    # the held-out slices: empty while the search is blind (the names posted
    # earlier cannot be deleted through the API), scored for every candidate
    # from the stored rows once the search has stopped
    hold_n = res.get("holdout_slice_n") or {}
    for dim, keys in (("difficulty", ("easy", "medium", "hard")), ("doc_type", docs.DOC_TYPES)):
        for key in keys:
            n = hold_n.get(f"{dim}.{key}")
            if not n and key in ("easy", "medium", "hard"):
                n = res["n"]["holdout"]
            if not n:
                continue
            tracked.behavior(
                Behavior(
                    name=HOLDOUT_SLICE_NAMES[key],
                    test_version=res["test_versions"]["holdout"],
                    n=n,
                    judge=program_judge,
                    contamination=0,
                    reward_is_judge=False,
                    graded_by="program",
                    description=(
                        f"{SLICE_TITLES[key]} — held out ({n} docs); a slice of the held-out headline"
                        + ("" if not res.get("blind") else "; EMPTY until the search stops")
                        + "."
                    ),
                    rubric=F1_RUBRIC
                    + f" {SLICE_TITLES[key].replace(' F1', '')} only; held out ({n} docs).",
                )
            )
    ids_path = o / "posted.json"
    ids = json.loads(ids_path.read_text()) if ids_path.exists() and not dry else {}

    def _tokens(c: dict[str, Any]) -> float:
        return sum(
            c[s]["tokens_per_rollout"] * c[s]["rollouts"] for s in ("train", "holdout") if c.get(s)
        )

    total = sum(_tokens(c) for c in res["candidates"]) or 1.0
    last = None
    for c in res["candidates"]:
        mod = runpy.run_path(str(HERE / "candidates" / f"{c['candidate']}.py"))
        h = mod["harness"](models[0])
        t, s = c["train"], c["holdout"]
        vb = t.get("vs_baseline") or {}
        edits = mod.get("EDITS") or {}
        if edits:
            last = list(edits.values())[-1]
            added = last.instructions or (
                f"loop settings: turns {last.max_turns}, retries {last.retries}, validate {last.validate}"
            )
            changed = f'added one rule to the system prompt: "{added}"'
        else:
            changed = (mod.get("__doc__") or "").strip().split(". ")[0].replace("\n", " ")
        base_t = res["candidates"][0]["train"][HEADLINE]
        rnd = int(c["candidate"][:2])
        heading = (
            "starting harness (v0)"
            if c["candidate"] == Path(BASELINE).stem
            else "control: rewording only"
            if c["candidate"] == Path(PLACEBO).stem
            else f"Meta-Harness round {rnd - 1}: " + short_change(mod)
        )
        note = (
            f"**{heading}**\n\n"
            f"Changed: {changed} (candidates/{c['candidate']}.py, fingerprint {c['fingerprint']}).\n"
            f"Moved: train F1 from {pts(base_t['mean'])} (baseline) to {_ci(t[HEADLINE])}"
            + (f", paired {_d(vb.get(HEADLINE))}" if vb else "")
            + f"; leads {c['led_train_tasks']} of {res['n']['train']} train documents; "
            f"{c['cost_ratio_tokens']:.2f}x the baseline's tokens per document"
            + (f"; holdout F1 {_ci(s[HEADLINE])}" if s else "; holdout withheld until the gate")
            + ".\n"
            f"Why: {mod.get('WHY', '')}.\n"
            f"Learned: {c.get('learned') or 'pending the round'}\n"
            f"Reproduce: cd recipes/papers/doc-extraction-harness && python run.py search --blind "
            f'--models "$MODELS" with candidates/00..{c["candidate"][:2]} in place '
            f"(tests {res['test_versions']['train']}/{res['test_versions']['holdout']}, k=4, DOCX_SALT=0)"
        )
        if c["candidate"] in ids:
            run = tracked.open(ids[c["candidate"]])
            try:  # point an existing run's chart at the scored headline
                tracked._call("PATCH", f"/runs/{run.id}", {"targets": [bname(HEADLINE, "train")]})
            except Exception as exc:
                print(f"targets not updated on {c['candidate']}: {str(exc)[:120]}")
        else:
            # the run targets the search-set headline until the gate, so the
            # page's chart plots the climb and not a fallback slice
            run = tracked.run(
                c["candidate"], method="eval", harness=h.pin(), targets=[bname(HEADLINE, "train")]
            )
        try:
            for split in ("train", "holdout"):
                for mk in (HEADLINE, SECOND):
                    sc = c[split][mk] if c.get(split) else None
                    if sc:
                        run.score(
                            bname(mk, split),
                            round(100 * sc["mean"], 1),
                            ci=round(max(50 * (sc["ci"][1] - sc["ci"][0]), 0.1), 1),
                            n=sc["n"],
                            test_version=res["test_versions"][split],
                            examples=examples(o, c["candidate"], models[0], split, metric=mk),
                        )
            # the slices, per difficulty and per document type: search set, and
            # the holdout once the search has stopped
            for split, names in (("train", SLICE_NAMES), ("holdout", HOLDOUT_SLICE_NAMES)):
                for dim in ("difficulty", "doc_type"):
                    for key, v in (
                        ((c.get(split) or {}).get("slices") or {}).get(dim) or {}
                    ).items():
                        sc = v.get(HEADLINE)
                        if sc and key in names:
                            run.score(
                                names[key],
                                round(100 * sc["mean"], 1),
                                ci=round(max(50 * (sc["ci"][1] - sc["ci"][0]), 0.1), 1),
                                n=sc["n"],
                                test_version=res["test_versions"][split],
                                examples=examples(
                                    o, c["candidate"], models[0], split, dim=dim, key=key
                                ),
                            )
            run.note(note)
        finally:
            fh = res["noise_floor"].get("holdout")
            share = _tokens(c) / total
            if c["candidate"] in ids:
                ids[c["candidate"]] = run.id
                last = run
                continue  # noqa: B012  # the run stays open for the gate's scores
            run.finish(
                record=RunRecord(
                    data=Data(
                        train=res["test_versions"]["train"],
                        n_train=t[HEADLINE]["n"],
                        holdout=res["test_versions"]["holdout"],
                        n_holdout=s[HEADLINE]["n"] if s else res["n"]["holdout"],
                    ),
                    eval=EvalSetup(
                        metric="field_f1",
                        k=K,
                        run_std=fh["run_std"][HEADLINE] if fh else None,
                        run_std_runs=RERUNS,
                        reader="Haiku 4.5 rubric judge",
                    ),
                ),
                gpu="L40S",
                cost_usd=round(res["spend"]["total_usd_est"] * share, 2),
                say=False,
            )
        ids[c["candidate"]] = run.id
        last = run
    # the baseline is what production runs today: the served version the
    # verdict compares every candidate against (Sahana, 2026-09-25); nothing
    # else is promoted or archived
    if not dry:
        tracked.promote(Path(BASELINE).stem)
    tracked.figure(
        "climb-by-doc-type",
        figure_by_type(res),
        caption="Field F1 by harness round on the search set, one line per document type (n = 20 to 21 "
        "documents each, 95% interval over documents): bank statements, the longest documents, show "
        "the climb most clearly.",
        run=last,
    )
    tracked.figure(
        "climb",
        figure(res),
        caption="Field F1 by harness round on the 101 search documents, 95% interval over documents; "
        "shaded: the baseline's run-to-run noise (three runs); the placebo is a rewording with no new rule.",
        run=last,
    )
    if not dry:
        ids_path.write_text(json.dumps(ids, indent=1))
        print("posted: https://while.ai/platform/runs?agent=doc-extraction")
        problems = readback(tracked)
        for problem in problems:
            print("readback:", problem)
        if not problems:
            print("readback: clean (no problems)")
        if not res.get("blind"):
            # the page keys its chart and table off the behavior named field_f1
            # (the holdout): read back that every run now has a score on it
            missing = [
                r["version"]
                for r in tracked.runs()
                if not any(
                    e.get("behavior") == HOLDOUT_NAMES[HEADLINE] for e in r.get("evals") or []
                )
            ]
            print(
                "field_f1 scored on every run" if not missing else f"field_f1 missing on {missing}"
            )
            try:
                print("verdict:", tracked.verdict(HOLDOUT_NAMES[HEADLINE], version=res["pick"]))
            except Exception as exc:
                print("verdict:", str(exc)[:200])
    return [list(x) for x in fake.calls] if fake else []


# ---------------------------------------------------------------- judges and routing

# RUBRIC_V1 = the first rubric's version hash: its verdicts on the baseline and
# placebo rows are in the judge cache (the 12:32 loop judged live), so the
# rubric ablation v1 vs v2 costs nothing more.
RUBRIC_V1 = "ee2a503a8072"


def judges(a: argparse.Namespace) -> dict[str, Any]:
    """``wai.compare_judges`` (skills/audit-your-judge): the Haiku 4.5 rubric
    judge, v1 and v2, against program gold on each program-checkable
    criterion, over the distinct answers on the search model. Both judges
    read the verdict cache (no new calls); a row a version never judged is
    left unjudged and counted."""
    from whileai.judge_comparison import compare_judges

    task = _task(a)
    o = out_dir(a.dry_run)
    m = _models(a)[0]
    cache: dict[str, dict[str, Any]] = {}
    jc = o / "judge_cache.jsonl"
    if jc.exists():
        for line in jc.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cache[item["key"]] = item["verdict"]
    uniq: dict[tuple[str, str], dict[str, Any]] = {}
    for p in sorted((o / "traces").glob("*/rows.jsonl")):
        for r in rows_of(o, p.parent.name + ".py", m):
            uniq.setdefault((r["prompt"], r["final_text"]), r)
    answers = list(uniq.values())

    def cached(version: str, slug: str):
        def judge(row: dict[str, Any]) -> dict[str, Any]:
            key = hashlib.sha256(
                json.dumps(
                    [version, task["JUDGE_MODEL"], row["prompt"], row["final_text"]]
                ).encode()
            ).hexdigest()
            v = cache.get(key)
            if v is None or f"rubric:{slug}" not in (v.get("markers") or {}):
                return {"reward": None, "reason": f"not judged under rubric {version}"}
            return {
                "reward": float(v["markers"][f"rubric:{slug}"]),
                "reason": v.get("reason") or "",
            }

        judge.__name__ = f"haiku45_rubric_{version[:6]}"
        return judge

    report: dict[str, Any] = {
        "tool": "wai.compare_judges (whileai.judge_comparison)",
        "gold": "program verdict on the criterion (docs.grade vs the generator's gold), kind=program",
        "n_distinct_answers": len(answers),
        "floors": {"agreement_wilson_lb": 0.8, "kappa": 0.6},
        "criteria": {},
    }
    v2 = task["RUBRIC"].version
    for slug in task["PROGRAM_CHECKABLE"]:
        lib_rows = [
            {"rollout_id": f"a{i}", "prompt": r["prompt"], "final_text": r["final_text"]}
            for i, r in enumerate(answers)
        ]
        labels = [
            {"key": f"a{i}", "label": int(r["markers"][f"program:{slug}"])}
            for i, r in enumerate(answers)
            if f"program:{slug}" in r["markers"]
        ]
        lib_rows, _ = wai.simulations.attach_labels(
            lib_rows, labels, annotator="program", kind="program"
        )
        table = compare_judges(
            lib_rows,
            {"rubric v1": cached(RUBRIC_V1, slug), "rubric v2": cached(v2, slug)},
            concurrency=8,
        )
        print(f"== {slug}")
        print(table)
        report["criteria"][slug] = {
            s.name: {
                k: getattr(s, k, None)
                for k in ("agreement", "ci95", "kappa", "leak", "n", "unjudged", "ok", "seconds")
            }
            for s in table.scores
        }
        report["criteria"][slug]["best"] = table.best.name if table.best else None
    (o / "compare_judges.json").write_text(json.dumps(report, indent=1, default=str))
    return report


ROUTE_DIR = (
    HERE.parents[3] / "route"
)  # the sibling worktree that carries whileai.routing (read-only)

_ROUTE_SNIPPET = r"""
import json, sys
sys.path.insert(0, ".")
import whileai.routing as routing
args = json.loads(sys.argv[1])
rows = [json.loads(l) for l in open(args["rows"], encoding="utf-8") if l.strip()]
out = {}
for level, keys in args["slices"].items():
    sl = []
    for r in rows:
        if r["scenario_id"] in keys:
            r = dict(r)
            # the reward is the program's doc-exact check against the gold, a
            # rule, not a model judge: say so, or route asks for a judge audit
            r["judge_meta"] = {"scorer_kind": "rule"}
            r.pop("judge_name", None)
            sl.append(r)
    # two passes: the pool as graded, and only the replies that ended on
    # their own (routing's own remedy for a truncated pass: score only replies that ended)
    ended = [r for r in sl if r.get("finish_reason") != "length"
             and not any(isinstance(s, dict) and s.get("truncated") for s in r.get("steps") or [])]
    for tag, pool in (("as_graded", sl), ("ended_only", ended)):
        rep = routing.route(pool, model=args["model"], size_b=8.0)
        out[f"{level}/{tag}"] = dict(rep)
        out[f"{level}/{tag}"]["_n_rows"] = len(pool)
        print(f"[{level} {tag}] n={len(pool)} method={rep.method} plan={rep.plan} why={rep.get('why')}")
        for name, m in (rep.methods or {}).items():
            print(f"   {name:<20} ok={m.get('ok')} {m.get('why')}")
        for line in (rep.get('notes') or []):
            print("   note:", line)
json.dump(out, open(args["dest"], "w"), indent=1, default=str)
"""


def route(a: argparse.Namespace) -> None:
    """``wai.methods.route`` per difficulty slice on the pick's train rows on
    the search model: which training method the graded pool supports next
    (GRPO band, SFT on passes, OPSD on floors), from the pool the harness
    search leaves behind. Runs inside the route worktree, read-only."""
    o = out_dir(a.dry_run)
    cand = a.candidate or Path(pick_of(o)).stem
    rows = rows_of(o, cand + ".py", _models(a)[0], "train")
    slices: dict[str, list[str]] = {}
    for r in rows:
        slices.setdefault(str((r.get("scenario_dimensions") or {}).get("difficulty")), []).append(
            r["scenario_id"]
        )
    slices = {k: sorted(set(v)) for k, v in slices.items()}
    slices["all"] = sorted({r["scenario_id"] for r in rows})
    src = o / f"route_rows_{cand}.jsonl"
    src.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    args = {
        "rows": str(src),
        "slices": slices,
        "model": short(_models(a)[0]),
        "dest": str(o / "route.json"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _ROUTE_SNIPPET, json.dumps(args)],
        cwd=ROUTE_DIR,
        capture_output=True,
        text=True,
    )
    print(proc.stdout[-6000:])
    if proc.returncode:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(proc.returncode)
    src.unlink()
    print(f"wrote {o.name}/route.json for {cand}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "command",
        choices=[
            "search",
            "noise",
            "gate",
            "regrade",
            "rejudge",
            "audit",
            "judges",
            "route",
            "led",
            "report",
        ],
    )
    ap.add_argument("--candidate", default="", help="route: the candidate (default: the pick)")
    ap.add_argument(
        "--models",
        default=os.environ.get("DOCX_MODELS", ""),
        help="the search model first, then held-out models: vllm:<hub id>@<url>,...",
    )
    ap.add_argument("--task", default="task_docs.py", help="the task module (tasks(), judge(row))")
    ap.add_argument("--candidates", default="candidates", help="the candidates folder")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="rollouts in flight per model (the serve app admits 64)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="offline: scripted models and judge, out-dry/"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="documents per split to play (0 = all 101 + 99); smoke.sh passes 16 for CI",
    )
    ap.add_argument("--fresh", action="store_true", help="search: drop out/ first")
    ap.add_argument(
        "--blind",
        action="store_true",
        help="search/report: no holdout row is opened or scored; the search's rounds until the pick",
    )
    ap.add_argument("--force", action="store_true", help="noise: re-run cached runs")
    ap.add_argument(
        "--post", action="store_true", help="report: post to while.ai/platform with your saved key"
    )
    ap.add_argument(
        "--per-doc",
        type=int,
        default=0,
        help="rejudge: judge only the first N rollouts of each document (0 = all k); the judge budget",
    )
    ap.add_argument(
        "--full",
        default="",
        help="rejudge: row files (comma list of path prefixes under out/) that get every rollout judged",
    )
    ap.add_argument(
        "--only", default="", help="rejudge: only row files whose path contains one of these"
    )
    a = ap.parse_args()
    if not a.dry_run and not a.models:
        raise SystemExit(
            "live runs need --models vllm:<id>@<url>,... (and VLLM_API_KEY), or --dry-run"
        )
    {
        "search": search,
        "noise": noise,
        "gate": gate,
        "regrade": regrade,
        "rejudge": rejudge,
        "audit": audit,
        "judges": judges,
        "route": route,
        "led": led,
        "report": report,
    }[a.command](a)


if __name__ == "__main__":
    main()
