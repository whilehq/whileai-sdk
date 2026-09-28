"""Post every finished arm to the While platform: one experiment, two behaviors, a run per arm.

    python post.py --dry-run   # print what would be posted, no key
    python post.py             # needs WHILEAI_API_KEY

Reads rows/<arm>.holdout.jsonl. The noise floor is `wai.eval_variance` over the
three base passes (base, base-r2, base-r3); every arm's score carries its own
95% interval over holdout tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import data
import report

import whileai as wai
from whileai.config import provenance

HERE = pathlib.Path(__file__).resolve().parent
AGENT = "deepagents-reviewer"
BASE_MODEL = "Qwen/Qwen3.8-27B"
TESTS = {"holdout": "t-swe-review-holdout-v1", "public": "t-swebench-verified-review-v1"}
BUDGET = 8  # model calls; the stock agent's median on the pilot was 7

# arm file stem -> (label, method, what changed)
ARMS = {
    "srv-harness": (
        "harness: searched Deep Agents profile",
        "harness",
        "the HarnessProfile the dev-split search picked; weights untouched",
    ),
    "srv-sft-without": (
        "SFT: smithtune's pick",
        "SFT",
        "LoRA SFT on the 258 trajectories smithtune's council kept",
    ),
    "srv-sft-with": (
        "SFT: While's pick",
        "SFT",
        "LoRA SFT on the 173 trajectories wai.simulations.optimize kept: correct, within 8 calls, spread over tool patterns",
    ),
}


def graded(rows: list[dict]) -> list[dict]:
    """Both behaviors as 0/1 per review."""
    return [{**r, "in_budget": int(r["correct"] and r["model_calls"] <= BUDGET)} for r in rows]


def points(rows: list[dict], key: str) -> tuple[float, float, int]:
    """Points out of 100 and the 95% half-width, over tasks."""
    vals = [float(r[key]) for r in rows]
    n = len(vals)
    p = sum(vals) / n
    return round(100 * p, 1), round(100 * 1.96 * math.sqrt(p * (1 - p) / n), 1), n


def examples(rows: list[dict], tasks: dict[str, dict]) -> list[dict]:
    out = []
    for r in rows:
        t = tasks[r["task_id"]]
        out.append(
            {
                "prompt": f"{t['repo']} {t['instance_id']}: review a patch the hidden tests "
                f"{'pass' if t['label'] == 'approve' else 'fail'}",
                "reply": f"VERDICT: {r['verdict']} after {r['model_calls']} model calls",
                "ok": bool(r["correct"]),
                "why": "matches the hidden tests"
                if r["correct"]
                else (
                    "no verdict"
                    if r["verdict"] is None
                    else f"said {r['verdict']}, tests say {r['label']}"
                ),
                "tags": {"label": r["label"], "repo": t["repo"]},
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--split", default="public", choices=["holdout", "public"])
    args = p.parse_args(argv)

    tasks = {t["id"]: t for t in data.load(args.split)}
    dropped = json.loads(data.DECONTAM.read_text(encoding="utf-8"))["n_dropped"]
    passes = [
        graded(report.load(a, args.split))
        for a in ("srv-base", "srv-base-r2", "srv-base-r3")
        if (report.ROWS / f"{a}.{args.split}.jsonl").exists()
    ]
    noise = wai.eval_variance(
        *[[{**r, "reward": int(r["correct"])} for r in rows] for rows in passes]
    )
    floor = round(100 * (noise["noise_band"] or 0.0), 1) if len(passes) >= 3 else None
    arms = {
        a: graded(report.load(a, args.split))
        for a in ARMS
        if (report.ROWS / f"{a}.{args.split}.jsonl").exists()
    }

    print(f"noise floor over {len(passes)} base passes: {floor} points")
    for name, rows in [("base", passes[0]), *arms.items()]:
        c, ch, n = points(rows, "correct")
        b, bh, _ = points(rows, "in_budget")
        print(f"{name:16s} review_correct {c} ±{ch}   correct_in_budget {b} ±{bh}   n={n}")
    if args.dry_run:
        return 0

    from whileai.platform import Behavior, Data, Judge, Provenance, RunRecord, track

    tracked = track(AGENT, model=BASE_MODEL, harness="deepagents 0.7 stock, read-only filesystem")
    tracked.experiment(
        question=(
            "A Deep Agents code reviewer on Qwen3.8-27B: which lever makes it more accurate "
            "and cheaper on unseen repos, a tuned harness, SFT on its own traces, or RL?"
        ),
        hypothesis=(
            "SFT on the traces LangSmith already has copies the stock agent's long reads; "
            "picking lean correct traces, or RL with a call cost, cuts calls without "
            "losing accuracy."
        ),
        method=(
            "Public SWE-agent patches with hidden-test outcomes as review tasks. Traces from "
            "the stock agent in LangSmith; smithtune for the smithtune SFT arm; wai.simulations.optimize "
            "for the While SFT arm; the harness searched on dev repos only."
        ),
        measure=(
            f"{len(tasks)} reviews on {len({t['repo'] for t in tasks.values()})} repos no "
            f"training trace touched, 50/50 approve/reject, points out of 100 with 95% "
            f"intervals, against a floor from three base passes."
        ),
        decide="Promote an arm whose interval clears the base and the floor on both behaviors.",
    )
    judge = Judge(name="hidden tests (program)", agreement=1.0)
    common = dict(
        test_version=TESTS[args.split],
        n=len(tasks),
        judge=judge,
        noise_floor=floor,
        contamination=dropped,
        reward_is_judge=False,
        graded_by="program",
    )
    tracked.behavior(
        Behavior(
            name="review_correct",
            **common,
            description="The reviewer approves a patch the hidden tests pass and rejects one they fail.",
            rubric="PASS: verdict equals the hidden-test outcome. FAIL: the other verdict, or none.",
        )
    )
    tracked.behavior(
        Behavior(
            name="correct_in_budget",
            **common,
            description=f"The review is correct and used at most {BUDGET} model calls.",
            rubric=f"PASS: correct verdict within {BUDGET} model calls. FAIL: wrong, none, or over budget.",
        )
    )

    def post(label, method, changed, rows):
        run = tracked.run(
            label,
            method=method,
            base=BASE_MODEL,
            targets=[] if method == "none" else ["correct_in_budget"],
            trained_on=[] if method in ("none", "harness") else ["deepagents-review train traces"],
            record=RunRecord(
                data=Data(
                    holdout=TESTS[args.split], n_holdout=len(tasks), decontaminated_dropped=dropped
                ),
                provenance=Provenance(recipe="recipes/community/deepagents-review-four-arms"),
            ),
        )
        c, ch, n = points(rows, "correct")
        b, bh, _ = points(rows, "in_budget")
        run.score("review_correct", c, ci=ch, n=n, rows=examples(rows, tasks))
        run.score("correct_in_budget", b, ci=bh, n=n)
        calls = sum(r["model_calls"] for r in rows) / len(rows)
        run.note(
            f"Changed: {changed}.\nMoved: review_correct {c}, correct_in_budget {b}, "
            f"{calls:.1f} model calls per review.\n"
            f"Reproduce: see recipes/community/deepagents-review-four-arms/README.md"
        )
        run.finish(say=False)

    post("base", "none", "nothing; stock Deep Agents on the base model, three passes", passes[0])
    for arm, rows in arms.items():
        post(*ARMS[arm], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
