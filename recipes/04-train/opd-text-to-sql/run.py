"""On-policy distillation on text-to-SQL, with the teacher gate measured before any training.

    python run.py --dry-run          # offline: the configs, the selftest, the plan, no key, no GPU
    python run.py --stage gate       # score teacher and student on the holdout (about $1)
    python run.py --stage train      # OPD x 2 seeds and sequence-KD x 2 seeds on Modal
    python run.py --stage eval       # every adapter on the same holdout
    python run.py --stage report     # results.json and the table
    python run.py                    # all of it, in order, stopping at a failed gate

The student is Qwen2.5-1.5B-Instruct, the teacher Qwen2.5-7B-Instruct (one
tokenizer). The holdout, the verifier and the tasks are the ones in
`../text-to-sql`; nothing about the eval is rebuilt here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
T2S = HERE.parent / "text-to-sql"
sys.path.insert(0, str(T2S))
from sql_verifier import load_tasks, read_jsonl, split_of, write_jsonl

import whileai as wai
from whileai.config import provenance
from whileai.simulations.score.stats import decontaminate, eval_variance

OUT = HERE / "out"
CONFIGS = HERE / "configs"
STUDENT = "Qwen/Qwen2.5-1.5B-Instruct"
TEACHER = "Qwen/Qwen2.5-7B-Instruct"
AGENT = "opd-text-to-sql"
METRIC = "sql_execution_match"
K = 4
MAX_TOKENS = 512  # the student's cap: both models are scored under it, and OPD samples under it
TEMPERATURE = 0.7  # the text-to-sql benchmark's sampling temperature
STUDENT_RERUNS = 3  # the noise floor needs three independent draws (strengthen-your-evals 5b.3)
SEEDS = (17, 23)  # two training seeds per arm, MIN_TRAIN_SEEDS
STEPS = 100
ROWS_PER_STEP = 8
N_PROMPTS = 200  # x OPD_SAMPLES (4) = 800 rows = STEPS x ROWS_PER_STEP
PROMPT_SEED = 0
MDE_SD = 0.376  # per-task paired sd of a binary pass rate near 0.5-0.8, measured across five lanes
TEACHER_TRUNCATION_MAX = 0.05
GPU_USD_PER_HOUR = 1.95  # L40S, modal.com/pricing read 2026-09-20
ARMS = {
    "opd": "on-policy distillation, reverse KL on the teacher's top-32 tokens",
    "opd-full": "on-policy distillation, reverse KL over the full vocabulary",
    "seqkd": "sequence KD (SFT on teacher completions)",
}


# ----------------------------------------------------------------- statistics


def per_task(rows: list[dict]) -> dict[str, float]:
    """Per-task pass rate from k rollouts; the unit of every interval is the task."""
    tot: dict[str, list[int]] = {}
    for r in rows:
        tot.setdefault(str(r["scenario_id"]), []).append(int(r.get("reward") or 0))
    return {t: sum(v) / len(v) for t, v in tot.items()}


def paired(before: list[dict], after: list[dict]) -> dict:
    """Per-task deltas, ties by kind, the exact sign test, and the MDE at n.

    Ties split three ways because they mean three different things
    (strengthen-your-evals, section 4): both arms at 1 is saturation, both
    at 0 is a floor (a real shared failure, reported as such), anything else
    tied is an equal partial rate. The sign test is the two-sided exact
    binomial over the discordant tasks at equal k; the bootstrap is
    `wai.compare`'s and is reported beside it.
    """
    a, b = per_task(before), per_task(after)
    keys = sorted(set(a) & set(b))
    diffs = [b[t] - a[t] for t in keys]
    ups = sum(1 for d in diffs if d > 0)
    downs = sum(1 for d in diffs if d < 0)
    both_one = sum(1 for t in keys if a[t] == 1.0 and b[t] == 1.0)
    both_zero = sum(1 for t in keys if a[t] == 0.0 and b[t] == 0.0)
    ties = sum(1 for d in diffs if d == 0)
    n = len(keys)
    sd = statistics.pstdev(diffs) if n > 1 else 0.0
    return {
        "n_tasks": n,
        "mean_delta": sum(diffs) / n if n else 0.0,
        "ups": ups,
        "downs": downs,
        "ties": ties,
        "ties_saturated": both_one,
        "ties_floor": both_zero,
        "ties_partial": ties - both_one - both_zero,
        "sign_p": sign_test(ups, downs),
        "paired_sd": sd,
        "mde80_measured": 2.8 * sd / math.sqrt(n) if n else None,
        "mde80_default": 2.8 * MDE_SD / math.sqrt(n) if n else None,
    }


def sign_test(ups: int, downs: int) -> float | None:
    """Two-sided exact binomial test that ups and downs are equally likely; None with no discordant task."""
    m = ups + downs
    if m == 0:
        return None
    lo = min(ups, downs)
    p = sum(math.comb(m, i) for i in range(lo + 1)) / 2**m
    return min(1.0, 2 * p)


def expected_mixed(p: dict[str, float], k: int = 8) -> float:
    """E[mixed groups] / N at k rollouts from per-task pass rates (strengthen-your-evals 5b.7)."""
    if not p:
        return 0.0
    return sum(1 - q**k - (1 - q) ** k for q in p.values()) / len(p)


def truncation(rows: list[dict]) -> float:
    return sum(1 for r in rows if r.get("truncated")) / max(len(rows), 1)


def mean_tokens(rows: list[dict]) -> float:
    return sum(float(r.get("tokens") or 0) for r in rows) / max(len(rows), 1)


def gate_verdict(delta: float, mde: float, teacher_trunc: float) -> tuple[bool, str]:
    """Gate 9: the teacher beats the student by more than the MDE, and finishes its replies."""
    if teacher_trunc >= TEACHER_TRUNCATION_MAX:
        return (
            False,
            f"teacher truncation {teacher_trunc:.1%} is not under {TEACHER_TRUNCATION_MAX:.0%}",
        )
    if delta <= mde:
        return False, f"teacher - student {delta:+.3f} does not exceed the MDE {mde:.3f}"
    return (
        True,
        f"teacher - student {delta:+.3f} exceeds the MDE {mde:.3f}; teacher truncation {teacher_trunc:.1%}",
    )


def pin_rows(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def holdout_tasks() -> list[dict]:
    return [t for t in load_tasks() if split_of(t["id"]) == "holdout"]


def check_pin() -> str:
    """The holdout is frozen by content: the pin in git must match the rows."""
    digest = pin_rows(holdout_tasks())
    pinned = (HERE / "holdout.sha256").read_text(encoding="utf-8").strip()
    if digest != pinned:
        raise SystemExit(f"holdout: sha256 {digest[:12]} does not match the pin {pinned[:12]}")
    return digest


# ----------------------------------------------------------------- configs and the offline path


def write_configs() -> None:
    """The prime-rl TOML for the same two methods, from the method objects.

    The runs here go through TRL (both models fit one GPU there); these are
    the same knobs in prime-rl's own words, written so a reader with two
    GPUs and a served teacher can run the method the other way.
    """
    CONFIGS.mkdir(exist_ok=True)
    opd = wai.prime_rl_config(
        "text-to-sql-shop",
        wai.OPD(wai.Endpoint(url="http://localhost:8001/v1", model=TEACHER), max_tokens=MAX_TOKENS),
        model=STUDENT,
        steps=STEPS,
        batch=ROWS_PER_STEP,
        out=CONFIGS / "opd.toml",
    )
    opsd = wai.prime_rl_config(
        "text-to-sql-shop",
        wai.OPSD(privileged="reference", anchor="live", max_tokens=MAX_TOKENS),
        model=STUDENT,
        steps=STEPS,
        batch=ROWS_PER_STEP,
        out=CONFIGS / "opsd.toml",
    )
    print(opd)
    print(opsd)


def selftest() -> None:
    """The gate and the paired statistics on rows with known answers; each check has a failing twin."""
    rows_a = [{"scenario_id": f"t{i}", "reward": int(i % 4 == 0), "tokens": 10} for i in range(40)]
    rows_b = [{"scenario_id": f"t{i}", "reward": int(i % 2 == 0), "tokens": 12} for i in range(40)]
    p = paired(rows_a, rows_b)
    assert p["n_tasks"] == 40 and p["ups"] == 10 and p["downs"] == 0, p
    assert p["ties"] == 30 and p["ties_saturated"] == 10 and p["ties_floor"] == 20, p
    assert abs(p["sign_p"] - 2 * 0.5**10) < 1e-12, p["sign_p"]
    assert sign_test(0, 0) is None and sign_test(5, 5) == 1.0
    assert sign_test(6, 0) < 0.05 < sign_test(4, 1)  # a gate that can fail
    assert abs(expected_mixed({"a": 0.5}, k=8) - (1 - 2 * 0.5**8)) < 1e-12
    assert expected_mixed({"a": 0.0, "b": 1.0}, k=8) == 0.0
    ok, why = gate_verdict(0.20, 0.05, 0.01)
    assert ok, why
    assert not gate_verdict(0.04, 0.05, 0.01)[0]  # too small a gap
    assert not gate_verdict(0.20, 0.05, 0.08)[0]  # teacher cut off
    rows_t = [
        {"scenario_id": "x", "reward": 0, "truncated": True},
        {"scenario_id": "x", "reward": 1},
    ]
    assert truncation(rows_t) == 0.5
    assert pin_rows([{"a": 1}]) != pin_rows([{"a": 2}]) and pin_rows(
        [{"a": 1, "b": 2}]
    ) == pin_rows([{"b": 2, "a": 1}])
    check_pin()
    print("selftest: gate, sign test, ties, expected mixed groups, pin: ok")


# ----------------------------------------------------------------- the paid stages


def _modal():
    import modal

    return (
        modal.Function.from_name("opd-text-to-sql", "sample"),
        modal.Function.from_name("opd-text-to-sql", "train_opd"),
        modal.Function.from_name("opd-text-to-sql", "train_seqkd"),
    )


def _save(name: str, result: dict) -> list[dict]:
    OUT.mkdir(exist_ok=True)
    write_jsonl(OUT / f"{name}.jsonl", result["rows"])
    (OUT / f"{name}.summary.json").write_text(
        json.dumps(result["summary"], indent=1), encoding="utf-8"
    )
    print(json.dumps(result["summary"]))
    return result["rows"]


def _rows(name: str) -> list[dict]:
    rows = read_jsonl(OUT / f"{name}.jsonl")
    if not rows:
        raise SystemExit(f"out/{name}.jsonl is missing; run the stage that writes it first")
    return rows


def _train_summary(name: str) -> dict:
    """A trained arm's summary: out/<name>.train.json, else the volume's summary.json."""
    path = OUT / f"{name}.train.json"
    if not path.exists():
        import subprocess

        OUT.mkdir(exist_ok=True)
        subprocess.run(
            ["modal", "volume", "get", "opd-runs", f"{name}/summary.json", str(path), "--force"],
            capture_output=True,
        )
    if not path.exists():
        raise SystemExit(f"no training summary for {name}; run the train stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_ids() -> list[str]:
    """The same 200 train prompts every arm trains on (opd_modal._prompt_set's rule)."""
    import random

    train = [t for t in load_tasks() if split_of(t["id"]) == "train"]
    random.Random(PROMPT_SEED).shuffle(train)
    return [t["id"] for t in train[:N_PROMPTS]]


def _have(name: str, expect: int) -> list[dict] | None:
    """Rows already sampled (out/, else the volume): a stage resumes rather than re-spends."""
    rows = read_jsonl(OUT / f"{name}.jsonl")
    if len(rows) != expect:
        import subprocess

        OUT.mkdir(exist_ok=True)
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "opd-runs",
                f"rows/{name}.jsonl",
                str(OUT / f"{name}.jsonl"),
                "--force",
            ],
            capture_output=True,
        )
        rows = read_jsonl(OUT / f"{name}.jsonl")
    return rows if len(rows) == expect else None


def stage_gate(limit: int = 0) -> dict:
    sample, _, _ = _modal()
    n_hold = min(len(holdout_tasks()), limit) if limit else len(holdout_tasks())
    n_train = min(N_PROMPTS, limit) if limit else N_PROMPTS
    common = dict(k=K, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, limit=limit)
    want = {"teacher": (TEACHER, dict(seed=0, **common), n_hold * K)}
    for i in range(STUDENT_RERUNS):
        want[f"student-{i}"] = (STUDENT, dict(seed=i, **common), n_hold * K)
    # the teacher's completions on the train prompts: the sequence-KD control's rows
    want["teacher-train"] = (
        TEACHER,
        dict(
            split="train",
            task_ids=_prompt_ids(),
            seed=0,
            k=4,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            limit=limit,
        ),
        n_train * 4,
    )
    rows: dict[str, list[dict]] = {}
    calls = {}
    for name, (model, kw, expect) in want.items():
        have = _have(name, expect)
        if have is not None:
            rows[name] = have
            print(f"{name}: {len(have)} rows already sampled")
        else:
            calls[name] = sample.spawn(model, name, **kw)
    for name, call in calls.items():
        rows[name] = _save(name, call.get())
    return gate_report(rows["teacher"], [rows[f"student-{i}"] for i in range(STUDENT_RERUNS)])


def gate_report(teacher: list[dict], students: list[list[dict]]) -> dict:
    var = eval_variance(*students)
    floor = float(var["noise_band"])
    student = students[0]
    pt, ps = wai.pass_at(teacher), wai.pass_at(student)
    p = paired(student, teacher)
    rep = wai.compare(student, teacher, run_std=var["run_std"], run_std_runs=len(students))
    mde = max(p["mde80_measured"] or 0.0, p["mde80_default"] or 0.0)
    delta = pt.pass_at_1 - ps.pass_at_1
    ok, why = gate_verdict(delta, mde, truncation(teacher))
    student_p = per_task(student)
    gate = {
        "teacher": {
            "model": TEACHER,
            "pass_at_1": pt.pass_at_1,
            "ci95": list(pt.ci95),
            "pass_at_k": pt.pass_at_k,
            "truncated": truncation(teacher),
            "mean_tokens": mean_tokens(teacher),
        },
        "student": {
            "model": STUDENT,
            "pass_at_1": ps.pass_at_1,
            "ci95": list(ps.ci95),
            "pass_at_k": ps.pass_at_k,
            "truncated": truncation(student),
            "mean_tokens": mean_tokens(student),
            "reruns": [wai.pass_at(s).pass_at_1 for s in students],
        },
        "noise": {
            "run_std": var["run_std"],
            "n_runs": var["n_runs"],
            "noise_band": floor,
            "points": round(floor * 100, 2),
        },
        "paired": p,
        "compare": {
            "delta": rep["metrics"]["pass_at_1"]["delta"],
            "ci95": rep["metrics"]["pass_at_1"]["ci95"],
            "verdict": rep["headline_verdict"],
        },
        "mde": mde,
        "pool": {
            "expected_mixed_k8": expected_mixed(student_p, 8),
            "all_fail_k4": sum(1 for v in student_p.values() if v == 0) / max(len(student_p), 1),
            "all_pass_k4": sum(1 for v in student_p.values() if v == 1) / max(len(student_p), 1),
        },
        "k": K,
        "max_tokens": MAX_TOKENS,
        "n_holdout": len(student_p),
        "passed": ok,
        "why": why,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    print(rep)
    print(
        f"GATE {'PASS' if ok else 'FAIL'}: teacher {pt.pass_at_1:.3f} {list(pt.ci95)} vs student "
        f"{ps.pass_at_1:.3f} {list(ps.ci95)}; {why}; noise floor {floor * 100:.2f} points from "
        f"{var['n_runs']} re-runs (run_std {var['run_std']}); ties {p['ties']} (floor {p['ties_floor']}, "
        f"saturated {p['ties_saturated']}); sign test p={p['sign_p']}; student pool E[mixed]@8 "
        f"{gate['pool']['expected_mixed_k8']:.2f}, all-fail@4 {gate['pool']['all_fail_k4']:.2f}"
    )
    return gate


def stage_train(arms: list[str], steps: int = STEPS) -> dict:
    _, train_opd, train_seqkd = _modal()
    calls = {}
    for seed in SEEDS:
        for arm, support in (("opd", "top_k"), ("opd-full", "full")):
            if arm in arms:
                calls[f"{arm}-s{seed}"] = train_opd.spawn(
                    f"{arm}-s{seed}",
                    student=STUDENT,
                    teacher=TEACHER,
                    seed=seed,
                    steps=steps,
                    rows_per_step=ROWS_PER_STEP,
                    n_prompts=N_PROMPTS,
                    prompt_seed=PROMPT_SEED,
                    max_new_tokens=MAX_TOKENS,
                    support=support,
                )
        if "seqkd" in arms:
            calls[f"seqkd-s{seed}"] = train_seqkd.spawn(
                f"seqkd-s{seed}",
                student=STUDENT,
                teacher_rows="teacher-train",
                seed=seed,
                steps=steps,
                rows_per_step=ROWS_PER_STEP,
            )
    out = {}
    OUT.mkdir(exist_ok=True)
    for name, call in calls.items():
        out[name] = call.get()
        (OUT / f"{name}.train.json").write_text(json.dumps(out[name], indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in out[name].items() if k != "curve"}))
    return out


def stage_eval(arms: list[str], limit: int = 0) -> None:
    sample, _, _ = _modal()
    calls = {}
    for arm in arms:
        for seed in SEEDS:
            name = f"{arm}-s{seed}"
            calls[name] = sample.spawn(
                STUDENT,
                name,
                adapter=name,
                seed=0,
                k=K,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                limit=limit,
            )
    for name, call in calls.items():
        _save(name, call.get())


def stage_report(arms: list[str]) -> dict:
    gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
    students = [_rows(f"student-{i}") for i in range(STUDENT_RERUNS)]
    base = students[0]
    var = eval_variance(*students)
    holdout = holdout_tasks()
    pin = check_pin()
    # the training prompts against the frozen holdout, 8-gram (Lambert 2025, chapter Evaluation)
    keep = set(_prompt_ids())
    train_rows = [
        {"prompt": t["question"], "scenario_id": t["id"]} for t in load_tasks() if t["id"] in keep
    ]
    _, decon = decontaminate(
        train_rows, against=[{"prompt": t["question"], "scenario_id": t["id"]} for t in holdout]
    )
    pb = wai.pass_at(base)
    results = {
        "recipe": "opd-text-to-sql",
        "base_model": STUDENT,
        "teacher": TEACHER,
        "metric": METRIC,
        "test_version": "t-" + pin[:8],
        "n_holdout": len(holdout),
        "k": K,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "steps": STEPS,
        "rows_per_step": ROWS_PER_STEP,
        "train_prompts": N_PROMPTS,
        "seeds": list(SEEDS),
        "gate": gate,
        "checks": {
            "run_std": var["run_std"],
            "run_std_runs": var["n_runs"],
            "noise_band": var["noise_band"],
            "noise_band_points": round(float(var["noise_band"]) * 100, 2),
            "decontaminated_dropped": int(decon.get("dropped", decon.get("n_dropped", 0)) or 0),
            "decontaminate_rules_skipped": decon.get("rules_skipped"),
            "train_seeds": len(SEEDS),
            "grader": "program (sql_verifier.SQLExec, execution match on the seeded store)",
            "truncation_rule": "a reply cut at the cap scores 0",
        },
        "arms": {
            "base": {
                "method": "none",
                "score": pb.pass_at_1,
                "ci": list(pb.ci95),
                "ci_half": (pb.ci95[1] - pb.ci95[0]) / 2,
                "reruns": [wai.pass_at(s).pass_at_1 for s in students],
                "pass_at_k": pb.pass_at_k,
                "truncated": truncation(base),
                "mean_tokens": mean_tokens(base),
                "steps": 0,
                "note": "Changed: nothing.\nMoved: the base, three re-runs.\nWhy: the noise floor.\nLearned: what a re-run moves.\nReproduce: python run.py --stage gate",
            }
        },
        "verdicts": {},
    }
    for arm in arms:
        seeds_rows = {s: _rows(f"{arm}-s{s}") for s in SEEDS}
        trains = {s: _train_summary(f"{arm}-s{s}") for s in SEEDS}
        pooled = [r for rows in seeds_rows.values() for r in rows]
        rep = wai.compare(
            base,
            seeds_rows[SEEDS[0]],
            run_std=var["run_std"],
            run_std_runs=var["n_runs"],
            train_runs={"before": None, "after": list(seeds_rows.values())},
        )
        print(f"== {arm} vs base")
        print(rep)
        head = rep["metrics"]["pass_at_1"]
        per_seed = {}
        for s, rows in seeds_rows.items():
            pa = wai.pass_at(rows)
            pr = paired(base, rows)
            per_seed[str(s)] = {
                "score": pa.pass_at_1,
                "ci": list(pa.ci95),
                "pass_at_k": pa.pass_at_k,
                "delta": pr["mean_delta"],
                "ups": pr["ups"],
                "downs": pr["downs"],
                "ties": pr["ties"],
                "ties_floor": pr["ties_floor"],
                "ties_saturated": pr["ties_saturated"],
                "sign_p": pr["sign_p"],
                "truncated": truncation(rows),
                "mean_tokens": mean_tokens(rows),
                "gpu_minutes": round(trains[s]["seconds"] / 60, 1),
            }
        scores = [v["score"] for v in per_seed.values()]
        mean = sum(scores) / len(scores)
        results["arms"][arm] = {
            "method": ARMS[arm],
            "score": mean,
            "ci": [
                min(v["ci"][0] for v in per_seed.values()),
                max(v["ci"][1] for v in per_seed.values()),
            ],
            "ci_half": max((v["ci"][1] - v["ci"][0]) / 2 for v in per_seed.values()),
            "delta": head.get("delta"),
            "delta_ci": head.get("ci95"),
            "delta_p": head.get("p_value"),
            "within_noise": head.get("within_noise"),
            "seed_delta": rep.get("train_delta"),
            "seed_delta_ci": rep.get("train_ci95"),
            "seed_std": rep.get("train_std"),
            "verdict": rep["headline_verdict"],
            "markers": {
                m: {
                    "before": r.get("mean_a"),
                    "after": r.get("mean_b"),
                    "delta": r.get("delta"),
                    "ci95": r.get("ci95"),
                }
                for m, r in rep["metrics"].items()
                if m.startswith("marker:")
            },
            "per_seed": per_seed,
            "pooled": paired(base, pooled),
            "truncated": truncation(pooled),
            "mean_tokens": mean_tokens(pooled),
            "steps": STEPS,
            "gpu_minutes": round(sum(t["seconds"] for t in trains.values()) / 60, 1),
            "config": trains[SEEDS[0]]["config"],
        }
        results["verdicts"][arm] = rep["headline_verdict"]
    trained = [a for a in arms if a in results["arms"]]
    results["pairs"] = {}
    for i, a in enumerate(trained):
        for b in trained[i + 1 :]:
            rows_a, rows_b = _rows(f"{a}-s{SEEDS[0]}"), _rows(f"{b}-s{SEEDS[0]}")
            rep = wai.compare(rows_a, rows_b, run_std=var["run_std"], run_std_runs=var["n_runs"])
            head = rep["metrics"]["pass_at_1"]
            results["pairs"][f"{b}_vs_{a}"] = {
                "delta": head.get("delta"),
                "ci95": head.get("ci95"),
                "p_value": head.get("p_value"),
                "verdict": rep["headline_verdict"],
                "paired": paired(rows_a, rows_b),
            }
    (HERE / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print_table(results)
    return results


def print_table(results: dict) -> None:
    b = results["arms"]["base"]
    print(
        "| Arm | seed | pass@1 (95% CI) | vs base, paired | ties (floor/sat) | sign p | cut | tokens | GPU min |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    print(
        f"| base | 3 re-runs | {b['score']:.3f} ({b['ci'][0]:.3f}..{b['ci'][1]:.3f}); re-runs "
        f"{', '.join(f'{x:.3f}' for x in b['reruns'])} | - | - | - | {b['truncated']:.1%} | {b['mean_tokens']:.0f} | - |"
    )
    for name, arm in results["arms"].items():
        if name == "base":
            continue
        for seed, v in arm["per_seed"].items():
            print(
                f"| {name} | {seed} | {v['score']:.3f} ({v['ci'][0]:.3f}..{v['ci'][1]:.3f}) | {v['delta']:+.3f} "
                f"| {v['ties']} ({v['ties_floor']}/{v['ties_saturated']}) | {v['sign_p']:.3g} | {v['truncated']:.1%} "
                f"| {v['mean_tokens']:.0f} | {v['gpu_minutes']} |"
            )
        d, ci = arm["delta"], arm["delta_ci"]
        sd, sci = arm.get("seed_delta"), arm.get("seed_delta_ci")
        print(
            f"| {name} | across seeds | mean {arm['score']:.3f} | seed {next(iter(arm['per_seed']))}: {d:+.3f} "
            f"({ci[0]:+.3f}..{ci[1]:+.3f}); across seeds {sd:+.3f} ({sci[0]:+.3f}..{sci[1]:+.3f}) | "
            f"{arm['verdict']} | | | | |"
        )


# ----------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=["all", "gate", "train", "eval", "report"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--limit", type=int, default=0, help="first N holdout tasks only (a smoke run)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="offline: configs, selftest, the plan; no key, no GPU",
    )
    args = ap.parse_args(argv)
    if args.dry_run:
        write_configs()
        selftest()
        n = len(holdout_tasks())
        print(
            f"plan: gate = teacher x1 + student x{STUDENT_RERUNS} on {n} holdout tasks at k={K}, cap {MAX_TOKENS}; "
            f"MDE80 at n={n} with sd {MDE_SD}: {2.8 * MDE_SD / math.sqrt(n):.3f}; then "
            f"{' and '.join(ARMS.values())}, {len(SEEDS)} seeds each, {STEPS} steps x {ROWS_PER_STEP} rows on one L40S; "
            f"about 2.5 L40S hours in all, about ${2.5 * GPU_USD_PER_HOUR:.0f} at ${GPU_USD_PER_HOUR}/h"
        )
        return 0
    t0 = time.time()
    if args.stage in ("all", "gate"):
        gate = stage_gate(limit=args.limit)
        if not gate["passed"]:
            print("the gate is the result; nothing is trained")
            return 0 if args.stage == "gate" else 1
    if args.stage in ("all", "train"):
        stage_train(args.arms, steps=args.steps)
    if args.stage in ("all", "eval"):
        stage_eval(args.arms, limit=args.limit)
    if args.stage in ("all", "report"):
        stage_report(args.arms)
    print(f"{args.stage}: {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
