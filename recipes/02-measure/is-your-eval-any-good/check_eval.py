"""Six checks on an evaluation set, before you trust a number it produced.

Run it on your own graded rows, or with no arguments for an offline demo on a
deliberately broken eval.

    python check_eval.py                          # offline demo, no key, no GPU
    python check_eval.py holdout.jsonl            # one eval set
    python check_eval.py base.jsonl tuned.jsonl   # two arms, plus arm hygiene
    python check_eval.py base.jsonl tuned.jsonl --train train.jsonl

Every check answers one question and names the fix. Nothing here needs a
network or a key.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from whileai.config import provenance
from whileai.simulations import (
    compare_runs,
    decontaminate,
    eval_variance,
    group_signal,
    pass_at,
    reward_correlations,
)


def marker_names(rows: list[dict]) -> list[str]:
    """Every per-criterion marker name the rubric put on these rows."""
    names: set[str] = set()
    for row in rows:
        if isinstance(row.get("markers"), dict):
            names.update(str(k) for k in row["markers"])
    return sorted(names)


CEILING = 0.9
MIN_HEADROOM = 0.05
OK, WARN, BAD = "ok  ", "WARN", "BAD "


def load(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def say(level: str, title: str, detail: str, fix: str = "") -> None:
    print(f"  [{level}] {title}")
    for line in detail.splitlines():
        print(f"         {line}")
    if fix:
        print(f"         fix: {fix}")
    print()


# ---------------------------------------------------------------- the checks


def check_room_to_move(rows: list[dict]) -> None:
    """1. Is the eval hard enough to show an improvement at all?"""
    p = pass_at(rows)
    if p.pass_at_1 is None:
        say(BAD, "Is there room to move?", f"Nothing scored. {p.note}", "grade the rows first")
        return
    band = f" [{p.ci95[0]:.2f}..{p.ci95[1]:.2f}]" if p.ci95 else ""
    detail = f"base passes {p.pass_at_1:.2f}{band} of {p.n_groups} situations"
    if p.pass_at_1 >= CEILING:
        say(
            BAD,
            "Is there room to move?",
            detail + "\nthe model already passes nearly everything, so training cannot show a gain",
            "harder situations, or a stricter rubric",
        )
    elif p.pass_at_1 <= 0.05:
        say(
            BAD,
            "Is there room to move?",
            detail + "\nthe model fails nearly everything; this measures nothing either",
            "easier situations, or demonstrations before RL",
        )
    else:
        say(OK, "Is there room to move?", detail)


def check_headroom(rows: list[dict]) -> None:
    """2. Is there anything a grouped update could learn? (Lambert 2025, chapter Reasoning)"""
    p = pass_at(rows)
    if p.pass_at_k is None or p.headroom is None:
        say(
            WARN,
            "Anything to learn from repeats?",
            f"not measurable. {p.note or 'need repeats>=4 per situation'}",
            "re-run the eval with repeats=4",
        )
        return
    detail = (
        f"pass@1 {p.pass_at_1:.2f} -> pass@{p.k} {p.pass_at_k:.2f}, headroom {p.headroom:.2f}\n"
        f"pass^{p.k} {p.pass_pow_k:.2f} is the share it gets right every single time"
    )
    if p.headroom < MIN_HEADROOM:
        say(
            WARN,
            "Anything to learn from repeats?",
            detail
            + "\nrepeats find almost nothing one run missed, so a grouped update has no signal",
            "harder situations, or train with SFT/demonstrations instead of RL",
        )
    else:
        say(OK, "Anything to learn from repeats?", detail)


def check_contains_the_behavior(rows: list[dict]) -> None:
    """3. Does the eval actually exercise what you claim to test?

    A criterion that never fails is not being tested. This is the contrast
    question, asked of the eval instead of the training set.
    """
    names = marker_names(rows)
    if not names:
        say(
            WARN,
            "Does it contain the behavior?",
            "no per-criterion markers on these rows, so only the overall pass rate is visible",
            "have the rubric emit one marker per criterion",
        )
        return
    lines, dead = [], []
    for name in names:
        vals = [
            float(r["markers"][name])
            for r in rows
            if isinstance(r.get("markers"), dict) and r["markers"].get(name) is not None
        ]
        if not vals:
            continue
        fail = sum(1 for v in vals if v == 0.0) / len(vals)
        lines.append(f"{name:<28} fails {fail:6.1%} of {len(vals)} rows")
        if fail < 0.02:
            dead.append(f"{name} ({fail:.1%})")
    detail = "\n".join(lines)
    if dead:
        say(
            BAD,
            "Does it contain the behavior?",
            detail
            + f"\n{len(dead)} criterion(s) almost never fail, so the eval barely tests them: "
            + ", ".join(dead)
            + "\na criterion that cannot fail here cannot show an improvement here either",
            "write situations that can fail that criterion, or drop it from the claim",
        )
    else:
        say(OK, "Does it contain the behavior?", detail)


def check_self_noise(rows: list[dict]) -> None:
    """4. How much does the number move when nothing changes? (chapter Evaluation)

    The book puts most post-training evals at 0.25 to 1.5 points of run-to-run
    standard deviation with the setup held constant. Any claim under twice
    that is indistinguishable from re-running the eval.
    """
    runs = {
        str((r.get("lineage") or {}).get("eval_run"))
        for r in rows
        if isinstance(r.get("lineage"), dict) and (r["lineage"] or {}).get("eval_run") is not None
    }
    if len(runs) >= 2:
        v = eval_variance(rows, by="eval_run")
        say(
            OK if v["n_runs"] >= 3 else WARN,
            "How much does it move on its own?",
            f"{v['n_runs']} runs, run_std {v['run_std']}, "
            f"a delta under {v['noise_band']} is noise ({v['stability']})",
        )
        return
    say(
        WARN,
        "How much does it move on its own?",
        "one run, so the eval's own noise is unmeasured\n"
        "a single run is a draw, not a distribution",
        "simulate(tasks=..., runs=3) on each arm",
    )


def check_the_judge(rows: list[dict]) -> None:
    """5. Did the grader actually work, and is reward tracking the behavior?"""
    statuses: dict[str, int] = {}
    for r in rows:
        s = str(r.get("judge_status") or "none")
        statuses[s] = statuses.get(s, 0) + 1
    bad = sum(v for k, v in statuses.items() if k not in ("ok", "none"))
    if bad:
        errs = {
            str((r.get("judge_meta") or {}).get("error"))
            for r in rows
            if isinstance(r.get("judge_meta"), dict) and r["judge_meta"].get("error")
        }
        first = next(iter(errs), "")
        say(
            BAD,
            "Did the judge work?",
            f"{bad} of {len(rows)} rows failed to grade: {statuses}\n{first[:160]}",
            "re-run the judge; an ungraded eval scores nothing",
        )
        return
    corr = reward_correlations(rows)
    flagged = corr.get("flagged") or {}
    proxies = {
        k: v for k, v in flagged.items() if k in ("reply_length", "tool_calls", "assistant_turns")
    }
    if proxies:
        say(
            BAD,
            "Did the judge work?",
            f"graded cleanly, but reward tracks a cheap feature: {proxies}\n"
            "the rubric may be rewarding length or tool spam, not the behavior",
            "length-match the rubric, or check hack_scan(rows)",
        )
    else:
        say(
            OK,
            "Did the judge work?",
            f"all {len(rows)} rows graded, no cheap-feature correlation flagged",
        )


def check_contamination(rows: list[dict], train_path: str | None) -> None:
    """6. Did the training data already see the eval? (chapter Evaluation, 8-gram)"""
    if not train_path:
        say(
            WARN,
            "Did training see the eval?",
            "no training set given, so overlap is unchecked",
            "pass --train train.jsonl",
        )
        return
    train = load(train_path)
    _, rep = decontaminate(train, rows)
    level = OK if rep["n_contaminated"] == 0 else BAD
    say(
        level,
        "Did training see the eval?",
        f"{rep['n_contaminated']} of {rep['n']} training rows overlap the eval "
        f"({rep['n_exact']} verbatim, {rep['n_near']} near copies) at {rep['ngram']}-gram",
        ""
        if rep["n_contaminated"] == 0
        else "drop them before training; the gain is otherwise memorised",
    )


def check_arm_hygiene(base: list[dict], tuned: list[dict]) -> None:
    """The one that invalidates a result outright: did the two arms differ in
    anything other than the weights under test? (chapter Evaluation: every layer of an
    agentic eval changes the score, so document all of them.)"""
    print("Two arms, so the comparison itself gets checked:\n")
    broken: list[str] = []

    def one(rows: list[dict], key: str) -> Any:
        vals = {str(r.get(key)) for r in rows if r.get(key) is not None}
        return next(iter(vals)) if len(vals) == 1 else (None if not vals else sorted(vals))

    pb, pt = one(base, "policy_version"), one(tuned, "policy_version")
    if pb is not None and pb == pt:
        broken.append("the arms are not distinguishable")
        say(
            BAD,
            "Are the two arms different models?",
            f"both arms stamped {pb}\nnothing in the rows says which weights produced which arm",
            'pass advanced={"model_version": "...-base"} and "...-sft" so the arms are distinguishable',
        )
    else:
        say(OK, "Are the two arms different models?", f"base {pb}\ntuned {pt}")

    unverified: list[str] = []
    ub, ut = one(base, "user_model"), one(tuned, "user_model")
    agent_of = lambda v: str(v).split("@", 1)[0] if v else None  # noqa: E731
    moved = (ub == agent_of(pb)) and (ut == agent_of(pt))
    if ub is None and ut is None:
        unverified.append("who played the user is unrecorded")
        say(
            WARN,
            "Did both arms face the same user?",
            "no user_model on the rows, so this cannot be checked\n"
            "(rows from an older SDK do not carry it, and that is most of the risk)",
            "regenerate on a current SDK, or pin user_model= explicitly",
        )
    elif ub != ut or moved:
        broken.append("the environment moved with the arm")
        say(
            BAD,
            "Did both arms face the same user?",
            f"base arm's user was {ub}\ntuned arm's user was {ut}\n"
            "the simulated user ran on the model under test, so the environment moved with the arm\n"
            "the delta measures the pair, not the policy",
            "pin user_model= to one fixed model on both arms and re-run",
        )
    else:
        say(OK, "Did both arms face the same user?", f"both arms: {ub}")

    g_b, g_t = group_signal(base), group_signal(tuned)
    r = compare_runs(base, tuned)
    if r["delta"] is None:
        say(BAD, "The comparison", f"not computable: {r['note']}")
        return
    lo, hi = r["ci95"]
    clears = lo > 0 or hi < 0
    body = (
        f"delta {r['delta']:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  over {r['n_paired']} paired situations\n"
        f"mixed verdicts: base {g_b.get('n_mixed')}/{g_b.get('n_groups')}, "
        f"tuned {g_t.get('n_mixed')}/{g_t.get('n_groups')}"
    )
    if broken:
        # An interval that excludes zero is not a result when the two arms
        # differed in something other than the weights. Report the number and
        # refuse to read it, rather than letting the interval carry the claim.
        say(
            BAD,
            "The comparison",
            body + "\ndo not read this number: " + ", and ".join(broken),
            "fix the checks above, then re-run both arms",
        )
    elif unverified:
        say(
            WARN,
            "The comparison",
            body
            + "\nthis number may be sound, but the rows cannot prove it: "
            + ", and ".join(unverified),
            "re-run both arms on a current SDK so the rows record it",
        )
    elif clears:
        say(OK, "The comparison", body)
    else:
        say(
            WARN,
            "The comparison",
            body,
            "the interval covers zero: add situations, not repeats per situation",
        )


# ---------------------------------------------------------------- the demo


def demo_rows(n_tasks: int = 60, k: int = 4, seed: int = 0, easy: bool = True) -> list[dict]:
    """A deliberately broken eval: too easy, one criterion that never fails,
    reward that tracks reply length, and one ungradeable row."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_tasks):
        p = 0.95 if easy else 0.6
        for _j in range(k):
            ok = rng.random() < p
            rows.append(
                {
                    "scenario_id": f"task-{i}",
                    "prompt": f"situation {i}: the customer asks for a refund on order {1000 + i}",
                    "final_text": "sure, done. " * (12 if ok else 3),
                    "reward": float(ok),
                    "judge_status": "ok",
                    "policy_version": "Qwen/Qwen3-4B@aaaaaaaaaaaaaaaa",
                    "user_model": "Qwen/Qwen3-4B",
                    "sampling": {"temperature": 0.8},
                    "markers": {
                        "did_the_job": float(ok),
                        "never_fails": 1.0,
                        "confirmed_first": float(rng.random() < 0.5),
                    },
                }
            )
    return rows


def main() -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", nargs="*", help="one graded eval file, or base and tuned")
    ap.add_argument("--train", help="training rows, to check contamination")
    args = ap.parse_args()

    if not args.rows:
        print(
            "No files given, so this is the offline demo on a deliberately broken eval.\n"
            "Both arms are the SAME model on two different draws, so the true delta is zero.\n"
            "Watch the interval exclude zero anyway.\n"
        )
        base = demo_rows(easy=True, seed=0)
        tuned = demo_rows(easy=True, seed=1)
        pairs = [("demo eval", base)]
        two = (base, tuned)
    else:
        loaded = [load(p) for p in args.rows]
        pairs = list(zip(args.rows, loaded))
        two = (loaded[0], loaded[1]) if len(loaded) == 2 else None

    for name, rows in pairs[:1]:
        print(f"=== {name}: {len(rows)} rows ===\n")
        check_room_to_move(rows)
        check_headroom(rows)
        check_contains_the_behavior(rows)
        check_self_noise(rows)
        check_the_judge(rows)
        check_contamination(rows, args.train)

    if two:
        check_arm_hygiene(*two)

    print("Nothing above is a score. It is whether the score means anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
