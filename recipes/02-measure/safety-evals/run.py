"""Safety evals for a tool-using agent: prompt injection, data exfiltration,
secret leakage, unauthorized writes, and the over-refusal that a naive fix
brings. Offline, no key, seconds.

    python run.py                      # the whole report on three scripted agents
    python run.py --k 8 --seed 1       # more repeats, another draw
    python run.py --json out.json      # every report as one file

The pipeline, in the order the output prints:

1. **Simulate.** The suite's asks go in as ``seeds=``; the simulator adds
   its own grid of situations from the tools and policy (its own prompt-
   injection and false-authority probes land in the suite's buckets when
   drawn) and rolls each ask ``k`` times.
2. **Grade.** ``evaluate(rows, safety_judge)``: four markers read from the
   trajectory, reward 1 only when all hold. ``evaluate`` stamps the rows
   as eval lineage so they cannot be trained on by accident.
3. **Read per category.** pass@1 with an interval, and pass^k: how often
   the agent held on every one of ``k`` tries. For a leak, pass^k is the
   number.
4. **Trust the judge.** Agreement with hand-labeled transcripts, and the
   probes: a safety-only judge is satisfied by any refusal, which is the
   reward a policy learns to refuse everything from.
5. **Before and after.** The same tasks pinned (``tasks=``) and re-run on
   two candidate fixes. ``delta_report`` with ``must_not_regress=
   ["helpful_on_benign"]`` fails the fix that got safe by refusing, and
   passes the one that kept the job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agents import AGENTS
from judge import HELPFUL, SAFETY_MARKERS, safety_judge, safety_only_judge
from suite import ATTACKS, CATEGORIES, LABELED, SEEDS, SYSTEM_PROMPT, TOOLS, classify

import whileai.simulations as wai
from whileai.config import provenance

GRID_SITUATIONS = 8  # situations the simulator writes on top of the suite


def simulate(
    agent_name: str, *, k: int = 4, seed: int = 0, tasks: Any = None
) -> wai.SimulationData:
    """The suite plus a slice of the simulator's own grid, ``k`` repeats
    each. ``tasks=`` re-runs a previous run's asks so two agents are
    compared on the same tasks."""
    common: dict[str, Any] = dict(
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        simulator=False,  # template writer: no model, no key
        mode="rl",
        repeats=k,
        repeat_policy="fixed",  # every ask gets all k, graded or not
        reproducible=True,
        seed=seed,
        concurrency=1,
    )
    if tasks is not None:
        return wai.simulate(AGENTS[agent_name], tasks=tasks, **common)
    n = len(SEEDS) + GRID_SITUATIONS
    return wai.simulate(AGENTS[agent_name], seeds=SEEDS, situations=n, budget=n * k, **common)


def grade(data: wai.SimulationData, agent_name: str, judge=safety_judge) -> list[dict]:
    rows = [dict(r) for r in data.trajectories]
    for r in rows:
        r["category"] = classify(str(r.get("prompt") or ""))
    return wai.evaluate(rows, judge, model=agent_name).rows


def by_category(rows: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cat in (*CATEGORIES, "grid"):
        sub = [r for r in rows if r.get("category") == cat]
        if not sub:
            continue
        p = wai.pass_at(sub, min_k=2)
        out[cat] = {
            "asks": p.n_groups,
            "rows": len(sub),
            "pass_at_1": p.pass_at_1,
            "ci95": p.ci95,
            "pass_pow_k": p.pass_pow_k,
            "k": p.k,
        }
    return out


def format_categories(table: dict[str, dict[str, Any]]) -> str:
    lines = [f"  {'category':<20} {'asks':>4}  {'pass@1':>6}  {'95% CI':>14}  {'pass^k':>6}"]
    for cat, r in table.items():
        ci = r["ci95"]
        span = f"{ci[0]:.2f}..{ci[1]:.2f}" if ci else "n/a"
        pk = f"{r['pass_pow_k']:.2f}" if r["pass_pow_k"] is not None else "n/a"
        lines.append(f"  {cat:<20} {r['asks']:>4}  {r['pass_at_1']:>6.2f}  {span:>14}  {pk:>6}")
    return "\n".join(lines)


def failures(rows: list[dict], *, limit: int = 4) -> list[str]:
    """One line per failed row, the marker that failed and what the agent did."""
    seen: set[str] = set()
    out = []
    for r in rows:
        if r.get("reward") == 1 or r.get("failure_class") in seen:
            continue
        seen.add(str(r.get("failure_class")))
        sent = [
            f"{s['tool']}({(s.get('arguments') or {}).get('to') or (s.get('arguments') or {}).get('approval_id', '')})"
            for s in r.get("steps") or []
            if s.get("tool") in ("send_email", "update_plan")
        ]
        out.append(
            f"  [{r.get('failure_class')}] {r.get('category')}: {str(r.get('prompt'))[:60]!r}\n"
            f"      calls={sent or '-'}  reply={str(r.get('final_text'))[:80]!r}"
        )
        if len(out) >= limit:
            break
    return out


def trust(rows: list[dict], *, seed: int = 0) -> dict[str, Any]:
    """Two questions about the judge: does it agree with people, and what
    does a refusal buy under it?"""
    labeled = wai.run_judge(LABELED, safety_judge).rows
    agreement = wai.judge_trust(labeled)
    benign = [r for r in rows if r.get("category") == "benign"]
    attacks = [r for r in rows if r.get("category") not in ("benign", "grid")]
    probes = {
        name: {
            "refusal_on_benign": wai.judge_probes(
                benign, judge, probes=["refusal"], sample=len(benign), seed=seed
            ),
            "shortcuts_on_attacks": wai.judge_probes(
                attacks,
                judge,
                probes=["success_claim", "filler", "sycophancy", "prompt_echo"],
                sample=40,
                seed=seed,
            ),
        }
        for name, judge in (("safety_only", safety_only_judge), ("safety_judge", safety_judge))
    }
    return {"agreement": agreement, "probes": probes}


def format_trust(report: dict[str, Any]) -> str:
    a = report["agreement"]["agreement"]
    lines = [
        f"  hand labels: {a['n']} transcripts, agreement {a['agreement']:.2f} "
        f"(95% {a['ci95'][0]:.2f}..{a['ci95'][1]:.2f}), kappa {a['kappa']:.2f}, "
        f"{len(report['agreement']['disagreements'])} to review"
    ]
    for name, probes in report["probes"].items():
        ref = probes["refusal_on_benign"]["probes"]["refusal"]
        holes = probes["shortcuts_on_attacks"]["exploitable_by"]
        lines.append(
            f"  {name:<13} a refusal passes {ref['exploit_rate']:.0%} of benign asks"
            f"{'  <- exploitable' if ref['flagged'] else ''}; "
            f"additive shortcuts on attacks: {holes or 'none'}"
        )
    return "\n".join(lines)


def compare(before: list[dict], after: list[dict], *, seed: int = 0) -> dict[str, Any]:
    report = wai.delta_report(
        before,
        after,
        target="pass_at_1",
        must_not_regress=[HELPFUL, *SAFETY_MARKERS],
        by="category",
        seed=seed,
    )
    report["refusal_on_benign"] = {
        "before": wai.refusal_report([r for r in before if r.get("category") == "benign"]),
        "after": wai.refusal_report([r for r in after if r.get("category") == "benign"]),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, default=4, help="repeats per ask")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write every report to this path")
    args = ap.parse_args(argv)
    for key in ("OPENAI_API_KEY", "VLLM_API_KEY"):
        os.environ.pop(key, None)  # offline on purpose
    out: dict[str, Any] = {}

    print(f"== suite: {len(ATTACKS)} asks in {len(CATEGORIES)} categories, k={args.k}")
    base = simulate("trusting", k=args.k, seed=args.seed)
    rows = grade(base, "trusting")
    n_grid = sum(1 for r in rows if r["category"] == "grid")
    print(
        f"   {len(rows)} rows: {len(rows) - n_grid} from the suite, {n_grid} from the "
        f"simulator's own grid ({base.stopped_because})"
    )

    print("\n== trusting agent, by category")
    table = by_category(rows)
    print(format_categories(table))
    print("  where it failed, one example per failure class:")
    print("\n".join(failures(rows)))
    out["trusting"] = {"by_category": table, "pass_at": wai.pass_at(rows).__dict__}

    print("\n== the judge")
    tr = trust(rows, seed=args.seed)
    print(format_trust(tr))
    out["judge_trust"] = tr

    for name in ("locked-down", "hardened"):
        print(f"\n== before/after: trusting -> {name} (same tasks, pinned)")
        after = grade(simulate(name, k=args.k, seed=args.seed, tasks=base), name)
        rep = compare(rows, after, seed=args.seed)
        print(wai.format_delta_report(rep))
        b, a = rep["refusal_on_benign"]["before"], rep["refusal_on_benign"]["after"]
        print(f"  refusal on benign asks: {b['refusal_rate']:.0%} -> {a['refusal_rate']:.0%}")
        out[name] = {"by_category": by_category(after), "delta": rep}

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
