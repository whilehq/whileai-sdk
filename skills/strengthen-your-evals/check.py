"""skills/strengthen-your-evals, runnable: every ```python block of SKILL.md,
verbatim, around a scripted refund agent. Offline, no key, no model, seconds.

The setup (this file only): a four-order shop, two tools, the refund
POLICY, the three-ask suite the team already has (OLD_TESTS), SEEDS (one
ask per policy branch), ``refund_judge`` (the policy as a program),
VERSIONS (``v1`` is the agent as shipped, ``v2`` the same agent after a
prompt fix), LABELS (hand labels for sixty rows, scripted here), MODEL,
and ``fake``, a recording transport that answers like the platform API.
Everything below the setup is the playbook.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from datetime import date
from typing import Any

import whileai.simulations as wai
from whileai.platform import (
    Behavior,
    Data,
    EvalSetup,
    Harness,
    Judge,
    RunRecord,
    track,
)

T0 = time.monotonic()

# ------------------------------------------------------------------ the world

TODAY = date(2026, 9, 17)
REFUND_LIMIT = 200.0
WINDOW_DAYS = 30
MODEL = "claude-haiku-4-5"

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

# The order ids are in the description on purpose: the situation writer
# reads it, and a writer that does not know the ids invents ones that do
# not exist, so every rollout is "not found" and the eval is hollow.
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
    """The policy, as a program. The judge and the fixed agent share it."""
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


def _lookup_step(order_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    order = lookup_order(order_id)
    return order, {"tool": "lookup_order", "arguments": {"order_id": order_id}, "result": order}


def _refund_step(order_id: str, order: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": "issue_refund",
        "arguments": {"order_id": order_id, "amount": order["total"]},
        "result": {"ok": True, "order_id": order_id, "amount": order["total"]},
    }


def shipped_agent(message: str) -> dict[str, Any]:
    """v1, the agent in production: looks the order up, then refunds
    whatever it found. The old three-ask suite passes it."""
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": [], "final_text": "Sure! Which order?"}
    order_id = ids[0]
    order, step = _lookup_step(order_id)
    if "error" in order:
        return {"steps": [step], "final_text": f"No order {order_id} on file."}
    return {
        "steps": [step, _refund_step(order_id, order)],
        "final_text": f"Refunded ${order['total']:.2f} for {order_id}. Sorry about that!",
    }


def fixed_agent(message: str) -> dict[str, Any]:
    """v2, after the prompt fix: follows the policy."""
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": [], "final_text": "Happy to help. Which order id is this about?"}
    order_id = ids[0]
    order, step = _lookup_step(order_id)
    if "error" in order:
        return {"steps": [step], "final_text": f"I could not find an order {order_id}."}
    if not _wants_refund(message):
        return {
            "steps": [step],
            "final_text": f"Order {order_id} ({order['item']}) is {order['status']}.",
        }
    ok, why = refundable(order)
    if ok:
        return {
            "steps": [step, _refund_step(order_id, order)],
            "final_text": f"Done: ${order['total']:.2f} refunded for order {order_id}.",
        }
    if "manager" in why:
        return {
            "steps": [step],
            "final_text": f"Order {order_id} is over ${REFUND_LIMIT:.0f}, a manager will follow up.",
        }
    return {"steps": [step], "final_text": f"I cannot refund order {order_id}: {why}."}


VERSIONS = {"v1": shipped_agent, "v2": fixed_agent}

# ------------------------------------------------------------------ the suite

OLD_TESTS = [
    "I want a refund for order A1001, the shoes did not fit.",
    "What is the status of order A1001?",
    "Can you refund order Z9999?",
]

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


def classify(prompt: str) -> str:
    """Which policy branch an ask lands in, from the order it names."""
    ids = ORDER_ID.findall(prompt.upper())
    if not _wants_refund(prompt) or not ids:
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


# ------------------------------------------------------------------ the judge

MARKERS = (
    "looked_up_first",
    "refund_only_when_allowed",
    "refunds_when_eligible",
    "escalates_over_limit",
    "no_invented_order",
)


def refund_judge(row: dict) -> dict[str, Any]:
    """The policy read off the trajectory, not the prose. Reward 1 only when
    every applicable marker holds; a marker that does not apply stays None."""
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
    }


def hand_labels(rows: list[dict]) -> list[dict[str, Any]]:
    """Stands in for a person reading sixty replies. The reader sees only
    the ask and the reply: was a refund claimed, and should one have been?
    It is not the judge; it reads the prose, the judge reads the calls."""
    out = []
    for row in rows:
        claimed = "refunded" in str(row.get("final_text") or "").lower()
        should = classify(str(row.get("prompt") or "")) == "eligible"
        out.append(
            {
                "scenario_id": row["scenario_id"],
                "rollout_index": row["rollout_index"],
                "label": 1 if claimed == should else 0,
                "annotator": "you",
            }
        )
    return out


# --------------------------------------------------------------- the platform


class FakePlatform:
    """Records every call and answers like the API, including the verdict
    rule: delta +- sqrt(ci_a^2 + ci_b^2) must exclude zero, and every other
    behavior whose candidate point estimate is lower counts as a regression."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.evals: dict[tuple[str, str], dict[str, Any]] = {}
        self.serving: str | None = None

    def __call__(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path, body))
        if path == "/agents":
            return {"id": body["id"], "name": body["name"]}
        if "/behaviors/" in path and method == "PUT":
            name = path.rsplit("/", 1)[1]
            self.behaviors[name] = {"name": name, **(body or {})}
            return self.behaviors[name]
        if path == "/runs":
            run_id = f"{body['agent']}-{body['version']}"
            self.runs[run_id] = body
            return {"id": run_id, "version": body["version"]}
        if path.endswith("/evals"):
            version = self.runs[path.split("/")[2]]["version"]
            for item in body:
                self.evals[(item["behavior"], version)] = item
            return {"evals": body}
        if path.endswith("/promote"):
            self.serving = body["version"]
            return {"serving": self.serving}
        if "/dashboard" in path:
            return self._dashboard(path)
        return {"ok": True}

    def _dashboard(self, path: str) -> dict[str, Any]:
        agent_id = path.split("/")[2]
        name = path.split("behavior=")[1] if "behavior=" in path else next(iter(self.behaviors))
        versions = [v for (b, v) in self.evals if b == name]
        serving = self.serving
        candidate = next((v for v in reversed(versions) if v != serving), None)
        verdict: dict[str, Any] = {"candidate": candidate, "serving": serving}
        if candidate and serving:
            cand, serv = self.evals[(name, candidate)], self.evals[(name, serving)]
            delta = round(cand["score"] - serv["score"], 1)
            width = (
                None
                if cand.get("ci") is None or serv.get("ci") is None
                else math.sqrt(cand["ci"] ** 2 + serv["ci"] ** 2)
            )
            lower = 0
            for other in self.behaviors:
                pair = self.evals.get((other, candidate)), self.evals.get((other, serving))
                if other != name and all(pair) and pair[0]["score"] < pair[1]["score"]:
                    lower += 1
            verdict.update(
                delta=delta,
                excludesZero=None if width is None else abs(delta) > width,
                regressions=lower,
            )
        return {
            "agent": {"id": agent_id, "name": agent_id},
            "behavior": self.behaviors.get(name),
            "behaviors": list(self.behaviors),
            "verdict": verdict,
        }


fake = FakePlatform()

# =========================================================== the playbook

# ---- 1. what the suite you have never reaches
gap = wai.coverage_gap(OLD_TESTS, tools=TOOLS, system_prompt=POLICY)
print(wai.format_coverage_gap(gap))

assert gap["untested_rules"], "the three-ask suite should leave a rule untested"
assert gap["single_shot"], "each old ask runs once"

# ---- 2. the frozen test
K, N = 4, 64  # rollouts per ask, asks


def holdout(agent, seed=0):
    """The same N asks for every version, each rolled K times."""
    return wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        seeds=SEEDS,
        situations=N,
        budget=N * K,
        simulator=False,  # offline writer, no key; drop it for the hosted writer
        mode="rl",
        repeats=K,
        repeat_policy="fixed",
        reproducible=True,
        seed=seed,
        fault_rate=0.0,
        avg_turns=1,
    )


data = {v: holdout(agent) for v, agent in VERSIONS.items()}
asks = sorted({r["prompt"] for r in data["v1"].rows()})
for d in data.values():
    assert sorted({r["prompt"] for r in d.rows()}) == asks, "every version must face the same asks"
TEST_VERSION = "t-" + hashlib.sha256("\n".join(asks).encode()).hexdigest()[:8]
print(f"held-out test {TEST_VERSION}: {len(asks)} asks x {K} rollouts")

assert len(asks) == N, f"{len(asks)} asks, wanted {N}"
assert all(len(d.rows()) == N * K for d in data.values())

# ---- 3. the judge, checked against people
scored = {v: wai.evaluate(d.rows(), refund_judge, tools=TOOLS) for v, d in data.items()}
LABELS = hand_labels(scored["v1"].rows[:60])

labeled, _ = wai.attach_labels(scored["v1"].rows[:60], LABELS, kind="human")
trust = wai.judge_trust(labeled, refund_judge)
print(trust)
JUDGE = Judge(
    name="refund policy, as a program",
    agreement=trust["agreement"]["agreement"],
    human_n=trust["agreement"]["n"],
)

assert trust["gold_kind"] == "human", trust["gold_kind"]
assert trust["agreement"]["n"] == 60
assert JUDGE.agreement is not None and JUDGE.agreement >= 0.8, JUDGE.agreement


# ---- 4. the number, per behavior, with what it rests on
def score(rows):
    """pass@1 in points, the half-width of its 95% interval, and the asks it rests on."""
    pa = wai.pass_at(rows, k=K)
    lo, hi = pa.ci95
    return round(100 * pa.pass_at_1, 1), round(100 * (hi - lo) / 2, 1), pa.n_groups


def behaviors(rows):
    """The headline and every policy branch as its own behavior."""
    out = {"refund_policy": score(rows)}
    for name, m in wai.marker_summary(rows).items():
        if m["n_tasks"] < 3:
            continue  # unmeasured: under three asks reach it, and the card says so
        lo, hi = m["ci95"] or (m["mean"], m["mean"])  # no interval when every row agrees
        out[name] = (round(100 * m["mean"], 1), round(100 * (hi - lo) / 2, 1), m["n_tasks"])
    return out


for v, s in scored.items():
    pa = wai.pass_at(s.rows, k=K)
    capable = sum(1 for p in pa.per_task.values() if p < 1)
    print(f"{v}: pass@1 {score(s.rows)}  failure-capable asks {capable}/{pa.n_groups}")
    for name, (pts, ci, n) in behaviors(s.rows).items():
        print(f"   {name:<26} {pts:>5} +- {ci:<5} n={n}")

v1, v2 = score(scored["v1"].rows), score(scored["v2"].rows)
assert v2[0] > v1[0], (v1, v2)
assert N - 2 <= v1[2] <= N, v1  # two wordings of one opener share a scenario id
first_capable = sum(1 for p in wai.pass_at(scored["v1"].rows, k=K).per_task.values() if p < 1)
assert first_capable > 0, "the shipped agent should fail some asks"
assert len(behaviors(scored["v1"].rows)) >= 3, behaviors(scored["v1"].rows)

# ---- 5. the noise floor
first = score(scored["v1"].rows)
again = score(wai.evaluate(holdout(VERSIONS["v1"]).rows(), refund_judge, tools=TOOLS).rows)
NOISE = round(abs(first[0] - again[0]), 1)  # points; a scripted agent gives 0, a model 1 to 3
print(f"noise floor {NOISE} points (same test, rolled twice)")

assert NOISE == 0.0, NOISE  # scripted agents are deterministic

# ---- 6. how big the test has to be
need = wai.holdout_size(0.05, before=scored["v1"].rows, after=scored["v2"].rows, k=K)
print(f"asks to prove 5 points: {need['n_tasks']} (you have {need['n_paired']})")

assert need["n_paired"] == v1[2], need
assert need["sd_source"] == "rows", need["sd_source"]

# ---- 7. report, so the platform can tell a fix from a fluctuation
tracked = track(
    "refund-agent",
    model=MODEL,
    harness=Harness(label="v1", instructions=POLICY, tools=TOOLS),
    transport=fake,  # drop this line to talk to the platform
)
for name, (_pts, _ci, n) in behaviors(scored["v1"].rows).items():
    tracked.behavior(
        Behavior(
            name=name,
            test_version=TEST_VERSION,
            n=n,
            judge=JUDGE,
            noise_floor=NOISE,
            contamination=0,  # nothing trained, nothing to leak
            reward_is_judge=False,
        )
    )
for version, s in scored.items():
    run = tracked.run(version, method="eval", targets=["refund_policy"], harness=version)
    for name, (pts, ci, n) in behaviors(s.rows).items():
        run.score(name, pts, ci=ci, n=n, test_version=TEST_VERSION)
    run.finish(
        record=RunRecord(
            data=Data(holdout=TEST_VERSION, n_holdout=len(asks)),
            eval=EvalSetup(metric="pass@1", k=K, run_std=NOISE, run_std_runs=2, reader=JUDGE.name),
        )
    )
tracked.promote("v1")  # what is in production today; the next version is the candidate
print(tracked.verdict())

# ------------------------------------------------------------------ the checks
verdict = str(tracked.verdict())
assert "beats" in verdict, verdict
assert "v2 beats v1" in verdict, verdict
paths = [p for _, p, _ in fake.calls]
assert sum("/behaviors/" in p for p in paths) >= 3, paths
assert paths.count("/runs") == 2, paths
assert len(fake.evals) >= 6, len(fake.evals)
elapsed = time.monotonic() - T0
assert elapsed < 60, f"{elapsed:.1f}s"
print(f"\nok: {elapsed:.1f}s, {len(fake.behaviors)} behaviors, {len(fake.runs)} runs")
sys.exit(0)
