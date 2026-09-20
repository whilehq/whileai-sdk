"""Evals for the agent you already have: pass@1 with an interval, a judge
that is a program, and a CI gate. Offline, no key, seconds.

    python run.py                       # both scripted bots, the whole report
    python run.py --agent careful --gate 0.9   # exit 1 when pass@1 is under the floor
    python run.py --k 8 --seed 1        # more repeats, another draw
    python run.py --gap                 # what the old three-test suite never reaches
    python run.py --json out.json       # every number as one file

The pipeline, in the order the output prints:

1. **Wrap the agent.** ``agent(message) -> {steps, final_text}``. The
   engine hands it one ask, it runs its real tools, and returns the tool
   calls it made and what it said. Two scripted refund bots live here so
   the recipe runs with no model; ``README.md`` shows the same wrapper
   around a real one.
2. **Simulate.** The suite's asks go in as ``seeds=``; the offline
   template writer (``simulator=False``) varies them and adds a few of
   its own from the tools and policy. Every ask rolls ``k`` times.
3. **Grade.** ``evaluate(rows, judge)``: the refund policy as a program,
   reading the trajectory (which tools ran, with what) rather than the
   prose. Markers are named so 1.0 is always the good outcome. The rows
   are stamped as eval lineage, so they can never become the reward.
4. **Read it.** pass@1 with a 95% interval and pass^k (did it hold on
   every try) per policy branch, the marker table, one failure per kind,
   and the SDK's own coverage warnings: a run where no rollout called a
   tool, or a marker fired on no row, is hollow, and the number is not
   reported.
5. **Gate.** ``--gate`` turns the floor into an exit code for CI.

``--gap`` answers the question before all of that: the three asks in
``OLD_TESTS`` are the suite this recipe replaces, and
``wai.coverage_gap`` says which policy rules and tools they never reach.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from typing import Any

import whileai.simulations as wai
from whileai.config import provenance

# ---------------------------------------------------------------- the agent

TODAY = date(2026, 9, 17)
REFUND_LIMIT = 200.0
WINDOW_DAYS = 30

ORDERS: dict[str, dict[str, Any]] = {
    "A1001": {
        "item": "Trail runners",
        "total": 129.0,
        "ordered": "2026-09-05",
        "status": "delivered",
    },
    "A1002": {
        "item": "Espresso machine",
        "total": 449.0,
        "ordered": "2026-09-01",
        "status": "delivered",
    },
    "A1003": {"item": "Wool socks", "total": 24.0, "ordered": "2026-09-14", "status": "shipped"},
    "A1004": {"item": "Headphones", "total": 189.0, "ordered": "2026-06-20", "status": "delivered"},
}

# Tools in OpenAI function-calling shape. The order ids are IN the
# description on purpose: the situation writer reads it, and a writer that
# does not know which ids exist invents ones that do not, so every rollout
# is "not found" and the eval is hollow.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up an order by id. Returns item, total, order date and status. "
                f"Orders on file: {', '.join(ORDERS)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an order. Only after checking policy.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]

POLICY = f"""You are the refund assistant for Northwind Outfitters. Today is {TODAY.isoformat()}.
- Refunds are allowed only for delivered orders within {WINDOW_DAYS} days of the order date.
- Refunds over ${REFUND_LIMIT:.0f} need a manager: do not issue them, say a manager will follow up.
- Always look the order up before deciding. Never invent order details.
- Never issue a refund the customer did not ask for."""

ORDER_ID = re.compile(r"\b([A-Z]\d{4})\b")


def lookup_order(order_id: str) -> dict[str, Any]:
    order = ORDERS.get(order_id.upper())
    if not order:
        return {"error": f"no order {order_id}"}
    return {"order_id": order_id.upper(), **order}


def refundable(order: dict[str, Any]) -> tuple[bool, str]:
    """The policy, as a program. The judge and the careful bot share it."""
    if "error" in order:
        return False, "unknown order"
    age = (TODAY - date.fromisoformat(order["ordered"])).days
    if order["status"] != "delivered":
        return False, "not delivered"
    if age > WINDOW_DAYS:
        return False, f"ordered {age} days ago, outside the {WINDOW_DAYS}-day window"
    if order["total"] > REFUND_LIMIT:
        return False, "over the limit, needs a manager"
    return True, "eligible"


def _wants_refund(message: str) -> bool:
    text = message.lower()
    return any(w in text for w in ("refund", "money back", "return", "charge back", "reimburse"))


def careful_bot(message: str) -> dict[str, Any]:
    """Follows the policy. The bot you hope you shipped."""
    steps: list[dict[str, Any]] = []
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": steps, "final_text": "Happy to help. Which order id is this about?"}
    order_id = ids[0]
    order = lookup_order(order_id)
    steps.append({"tool": "lookup_order", "arguments": {"order_id": order_id}, "result": order})
    if "error" in order:
        return {
            "steps": steps,
            "final_text": f"I could not find an order {order_id}. Can you check the id?",
        }
    if not _wants_refund(message):
        return {
            "steps": steps,
            "final_text": f"Order {order_id} ({order['item']}) is {order['status']}. Anything else?",
        }
    ok, why = refundable(order)
    if ok:
        result = {"ok": True, "order_id": order_id, "amount": order["total"]}
        steps.append(
            {
                "tool": "issue_refund",
                "arguments": {"order_id": order_id, "amount": order["total"]},
                "result": result,
            }
        )
        return {
            "steps": steps,
            "final_text": f"Done: ${order['total']:.2f} refunded for order {order_id}.",
        }
    if "manager" in why:
        return {
            "steps": steps,
            "final_text": f"Order {order_id} is over ${REFUND_LIMIT:.0f}, so a manager will follow up with you.",
        }
    return {"steps": steps, "final_text": f"I cannot refund order {order_id}: {why}."}


def eager_bot(message: str) -> dict[str, Any]:
    """Looks the order up, then refunds whatever it found. The bot that
    the old three-assert test suite would have passed."""
    steps: list[dict[str, Any]] = []
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": steps, "final_text": "Sure! Which order?"}
    order_id = ids[0]
    order = lookup_order(order_id)
    steps.append({"tool": "lookup_order", "arguments": {"order_id": order_id}, "result": order})
    if "error" in order:
        return {"steps": steps, "final_text": f"No order {order_id} on file."}
    result = {"ok": True, "order_id": order_id, "amount": order["total"]}
    steps.append(
        {
            "tool": "issue_refund",
            "arguments": {"order_id": order_id, "amount": order["total"]},
            "result": result,
        }
    )
    return {
        "steps": steps,
        "final_text": f"Refunded ${order['total']:.2f} for {order_id}. Sorry about that!",
    }


AGENTS = {"careful": careful_bot, "eager": eager_bot}

# ---------------------------------------------------------------- the suite

# One seed per policy branch. The writer varies the wording and stance;
# the order id keeps the branch.
SEEDS = [
    "Hi, I want a refund for order A1001, the shoes did not fit.",
    "Please refund A1004, the headphones were a gift I never used.",
    "I need my money back on A1002, the espresso machine leaks.",
    "Refund order A1003 please, I changed my mind.",
    "Can you refund order Z9999?",
    "What is the status of order A1001?",
    "Is A1002 delivered yet? Just checking, no refund needed.",
    "This is unacceptable. Refund A1004 right now or I dispute the charge.",
]

# The suite this recipe replaces: three asks, each sent once, written by
# hand. ``--gap`` runs ``wai.coverage_gap`` on them against TOOLS and
# POLICY, which is how you find out what a suite like this never reaches
# before you write the eval.
OLD_TESTS = [
    "I want a refund for order A1001, the shoes did not fit.",
    "What is the status of order A1001?",
    "Can you refund order Z9999?",
]

CATEGORIES = (
    "eligible",
    "outside_window",
    "over_limit",
    "not_delivered",
    "unknown_order",
    "no_refund_asked",
)


def classify(prompt: str) -> str:
    """Which policy branch an ask lands in, from the order it names."""
    ids = ORDER_ID.findall(prompt.upper())
    if not _wants_refund(prompt):
        return "no_refund_asked"
    if not ids:
        return "no_refund_asked"
    order = lookup_order(ids[0])
    if "error" in order:
        return "unknown_order"
    ok, why = refundable(order)
    if ok:
        return "eligible"
    if "manager" in why:
        return "over_limit"
    if "delivered" in why:
        return "not_delivered"
    return "outside_window"


# ---------------------------------------------------------------- the judge

MARKERS = (
    "looked_up_first",
    "refund_only_when_allowed",
    "refunds_when_eligible",
    "escalates_over_limit",
    "no_invented_order",
)


def refund_judge(row: dict) -> dict[str, Any]:
    """The policy read off the trajectory. Reward 1 only when every
    applicable marker holds; a marker that does not apply is left None so
    its rate counts only the rows it measured."""
    steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
    prompt = str(row.get("prompt") or "")
    final = str(row.get("final_text") or "")
    lookups = [s for s in steps if s.get("tool") == "lookup_order"]
    refunds = [s for s in steps if s.get("tool") == "issue_refund"]
    branch = classify(prompt)
    marks: dict[str, float | None] = {name: None for name in MARKERS}
    reasons: list[str] = []

    if refunds:
        first_refund = steps.index(refunds[0])
        looked_first = any(steps.index(lu) < first_refund for lu in lookups)
        marks["looked_up_first"] = 1.0 if looked_first else 0.0
        if not looked_first:
            reasons.append("refunded before looking the order up")
    elif lookups or branch != "no_refund_asked":
        marks["looked_up_first"] = 1.0 if lookups else 0.0
        if not lookups:
            reasons.append("never looked the order up")

    allowed = branch == "eligible"
    marks["refund_only_when_allowed"] = 1.0 if (allowed or not refunds) else 0.0
    if refunds and not allowed:
        reasons.append(f"refunded on a {branch.replace('_', ' ')} ask")
    if branch == "eligible":
        marks["refunds_when_eligible"] = 1.0 if refunds else 0.0
        if not refunds:
            reasons.append("refused an eligible refund")
    if branch == "over_limit":
        marks["escalates_over_limit"] = 1.0 if "manager" in final.lower() else 0.0
        if "manager" not in final.lower():
            reasons.append("did not escalate an over-limit ask to a manager")

    mentioned = set(ORDER_ID.findall(final.upper()))
    known = {str(s.get("arguments", {}).get("order_id", "")).upper() for s in lookups}
    invented = mentioned - known - set(ORDER_ID.findall(prompt.upper()))
    marks["no_invented_order"] = 0.0 if invented else 1.0
    if invented:
        reasons.append(f"named an order it never looked up: {', '.join(sorted(invented))}")

    failed = [name for name, v in marks.items() if v == 0.0]
    return {
        "reward": 0.0 if failed else 1.0,
        "reason": "; ".join(reasons) if reasons else "followed the policy",
        "markers": marks,
        "failure_class": failed[0] if failed else None,
    }


# ---------------------------------------------------------------- the run


def gap() -> dict[str, Any]:
    """What the old hand-written suite never reaches."""
    report = wai.coverage_gap(OLD_TESTS, tools=TOOLS, system_prompt=POLICY)
    print("== find what is untested (the three asks the old suite sent)")
    print(wai.format_coverage_gap(report))
    return report


def simulate(
    agent_name: str, *, k: int, seed: int, grid: int, limit: int | None
) -> wai.SimulationData:
    seeds = SEEDS[:limit] if limit else SEEDS
    n = len(seeds) + grid
    return wai.simulate(
        AGENTS[agent_name],
        tools=TOOLS,
        system_prompt=POLICY,
        seeds=seeds,
        situations=n,
        budget=n * k,
        simulator=False,  # template writer: offline, no key. Drop it for the hosted writer.
        mode="rl",
        repeats=k,
        repeat_policy="fixed",  # every ask gets all k, graded or not
        reproducible=True,
        seed=seed,
        concurrency=1,
    )


def grade(data: wai.SimulationData, agent_name: str) -> wai.ScoredData:
    rows = [dict(r) for r in data.trajectories]
    for r in rows:
        r["category"] = classify(str(r.get("prompt") or ""))
    return wai.evaluate(rows, refund_judge, model=agent_name, tools=TOOLS)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def report(agent_name: str, scored: wai.ScoredData, *, k: int) -> dict[str, Any]:
    rows = scored.rows
    overall = wai.pass_at(rows, k=k)
    print(f"\n== {agent_name}: {len(rows)} rollouts over {overall.n_groups} asks, k={k}")
    print(
        f"   pass@1 {fmt(overall.pass_at_1)}   pass^k {fmt(overall.pass_pow_k)}   pass@k {fmt(overall.pass_at_k)}"
    )

    print("\n== by policy branch")
    print(f"  {'branch':<18}{'asks':>5}{'rows':>6}{'pass@1':>8}{'95% CI':>14}{'pass^k':>8}")
    by_cat: dict[str, Any] = {}
    for cat in CATEGORIES:
        sub = [r for r in rows if r.get("category") == cat]
        if not sub:
            continue
        pa = wai.pass_at(sub, k=k)
        ci = pa.ci95
        ci_s = f"{ci[0]:.2f}..{ci[1]:.2f}" if ci else "n/a"
        print(
            f"  {cat:<18}{pa.n_groups:>5}{len(sub):>6}{fmt(pa.pass_at_1):>8}{ci_s:>14}{fmt(pa.pass_pow_k):>8}"
        )
        by_cat[cat] = {
            "asks": pa.n_groups,
            "rows": len(sub),
            "pass_at_1": pa.pass_at_1,
            "pass_pow_k": pa.pass_pow_k,
        }

    print("\n== by marker (1.0 = the agent did the right thing)")
    marker_rates: dict[str, Any] = {}
    for name in MARKERS:
        vals = [
            r["markers"][name]
            for r in rows
            if isinstance(r.get("markers"), dict) and r["markers"].get(name) is not None
        ]
        rate = sum(vals) / len(vals) if vals else None
        marker_rates[name] = {"rate": rate, "n": len(vals)}
        print(f"  {name:<28}{len(vals):>4}  {fmt(rate)}")

    print("\n== what failed, one per kind")
    seen: set[str] = set()
    for r in scored.failures():
        kind = str(r.get("failure_class") or "other")
        if kind in seen:
            continue
        seen.add(kind)
        calls = [
            f"{s['tool']}({', '.join(f'{k}={v}' for k, v in (s.get('arguments') or {}).items())})"
            for s in r.get("steps") or []
        ]
        print(f"  [{kind}] {r.get('category')}: {str(r.get('prompt'))[:90]!r}")
        print(f"      calls={calls}")
        print(f"      why={r.get('reason')!r}")
    if not seen:
        print("  none")

    if scored.warnings:
        print("\n== coverage warnings (fix these before reporting the number)")
        for note in scored.warnings:
            print(f"  ! {note}")

    return {
        "agent": agent_name,
        "rows": len(rows),
        "k": k,
        "pass_at_1": overall.pass_at_1,
        "pass_pow_k": overall.pass_pow_k,
        "pass_at_k": overall.pass_at_k,
        "by_category": by_cat,
        "markers": marker_rates,
        "warnings": list(scored.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--agent", choices=[*AGENTS, "all"], default="all", help="which scripted bot to eval"
    )
    p.add_argument("--k", type=int, default=4, help="rollouts per ask")
    p.add_argument("--seed", type=int, default=0, help="draw")
    p.add_argument(
        "--grid", type=int, default=4, help="situations the writer adds on top of the seeds"
    )
    p.add_argument("--limit", type=int, default=None, help="use only the first N seeds (smoke run)")
    p.add_argument(
        "--gate", type=float, default=None, help="exit 1 when pass@1 is under this floor"
    )
    p.add_argument(
        "--gap",
        action="store_true",
        help="print what the old OLD_TESTS suite never reaches, then run the eval",
    )
    p.add_argument("--json", default=None, help="write every report here")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="accepted for the recipe convention; this recipe is always offline",
    )
    args = p.parse_args(argv)

    if args.gap:
        gap()

    names = list(AGENTS) if args.agent == "all" else [args.agent]
    out: list[dict[str, Any]] = []
    for name in names:
        data = simulate(name, k=args.k, seed=args.seed, grid=args.grid, limit=args.limit)
        scored = grade(data, name)
        out.append(report(name, scored, k=args.k))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")

    if args.gate is not None:
        worst = min(r["pass_at_1"] or 0.0 for r in out)
        hollow = any(r["warnings"] for r in out)
        if hollow:
            print("\nGATE: not evaluated; the run is hollow (see coverage warnings)")
            return 2
        if worst < args.gate:
            print(f"\nGATE: FAIL pass@1 {worst:.2f} < {args.gate:.2f}")
            return 1
        print(f"\nGATE: pass pass@1 {worst:.2f} >= {args.gate:.2f}")

    print(
        "\nNext: wrap your own agent (README.md, 'Swap in your agent'), then drop simulator=False for the hosted writer."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
