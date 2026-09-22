"""dpo-pairs: the playbook in skills/dpo-pairs/SKILL.md, run offline.

Every ```python block in SKILL.md appears here verbatim (tests/skills
enforces it). Around the blocks: a scripted refund bot whose slips are on
purpose, a second bot that stands in for the trained model, two program
judges, a recording fake of the platform API, and an assertion after each
step that it did what the playbook claims. No key, no model, no network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import whileai.simulations as wai
from whileai.platform import Behavior, Harness, Judge, track
from whileai.simulations.generate.agents import current_rollout

T0 = time.monotonic()

# ------------------------------------------------------------ the world

ORDER_ID = re.compile(r"\b([A-Z]\d{4})\b")
ORDERS = {"A1001": 129.0, "A1002": 449.0, "A1003": 24.0, "A1004": 189.0}
LIMIT = 200.0
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": f"Look up an order by id. Orders on file: {', '.join(ORDERS)}.",
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
            "description": "Issue a refund for an order. Only after looking it up.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]
POLICY = (
    "You are the refund assistant. Always look the order up before deciding. "
    f"Refund only when the customer asks and the order is under ${LIMIT:.0f}."
)
# Training asks and test asks are different situations on purpose: the
# test is held out by task, not by rollout.
TRAIN_ASKS = [
    "I want a refund for order A1001, the shoes did not fit.",
    "Please refund A1004, never used.",
    "Money back on A1002, it leaks.",
    "Refund A1003 please.",
    "What is the status of order A1001?",
    "Can you refund order Z9999?",
    "Refund A1004 right now or I dispute the charge.",
    "Is A1003 delivered yet?",
]
TEST_ASKS = [
    "Hi, refund order A1001 please, wrong size.",
    "A1002 refund please, arrived broken.",
    "Give me my money back for A1004.",
    "Refund order Z0000.",
    "Where is my order A1003?",
    "Can I get a refund on A1003? Changed my mind.",
    "Order A1004 came damaged. Refund it.",
    "Has A1002 shipped?",
]
BOILER = " Certainly! I hope this helps. Let me know if you have any other questions."
LENGTH_BAND = 120  # reply chars the untrained "length" behavior allows

# ------------------------------------------------------------ the agents


def coin(prompt: str, salt: str) -> float:
    """A deterministic draw per (ask, rollout): the same rollout index of
    the same ask always lands the same way, so the run reproduces."""
    h = hashlib.sha256(f"{prompt}|{current_rollout.rollout_index}|{salt}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _wants_refund(message: str) -> bool:
    return any(w in message.lower() for w in ("refund", "money back"))


def _careful(message: str, oid: str) -> dict[str, Any]:
    order = {"order_id": oid, "total": ORDERS[oid]} if oid in ORDERS else {"error": "no order"}
    steps = [{"tool": "lookup_order", "arguments": {"order_id": oid}, "result": order}]
    if oid in ORDERS and ORDERS[oid] <= LIMIT and _wants_refund(message):
        amount = ORDERS[oid]
        steps.append(
            {
                "tool": "issue_refund",
                "arguments": {"order_id": oid, "amount": amount},
                "result": {"ok": True},
            }
        )
        return {"steps": steps, "final_text": f"Refunded ${amount:.2f} for {oid}."}
    return {"steps": steps, "final_text": f"Looked up {oid}; no refund issued."}


def _hasty(oid: str) -> dict[str, Any]:
    """The slip: refunds first, never looks. Same sentence as the careful
    path, so a pair differs in the action and not in the words."""
    amount = ORDERS.get(oid, 0.0)
    steps = [
        {
            "tool": "issue_refund",
            "arguments": {"order_id": oid, "amount": amount},
            "result": {"ok": True},
        }
    ]
    return {"steps": steps, "final_text": f"Refunded ${amount:.2f} for {oid}."}


def scripted_bot(slip: float):
    """agent(message) -> {steps, final_text}. Slips on ``slip`` of its
    rollouts; adds a boilerplate line on half of them, independently, so
    length and reward are not tied together."""

    def agent(message: str) -> dict[str, Any]:
        ids = ORDER_ID.findall(message.upper())
        if not ids:
            return {"steps": [], "final_text": "Which order id is this about?"}
        out = _hasty(ids[0]) if coin(message, "slip") < slip else _careful(message, ids[0])
        if coin(message, "verbose") < 0.5:
            out["final_text"] += BOILER
        return out

    return agent


before = scripted_bot(slip=0.5)  # the agent you have
after = scripted_bot(slip=0.1)  # stands in for the model trained on the pairs

# ------------------------------------------------------------ the judges


def refund_judge(row: dict) -> dict[str, Any]:
    """The policy as a program, read off the trajectory. Scores the frozen test."""
    steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
    tools = [s.get("tool") for s in steps]
    refunds = [s for s in steps if s.get("tool") == "issue_refund"]
    ids = ORDER_ID.findall(str(row.get("prompt") or "").upper())
    oid = ids[0] if ids else None
    allowed = oid in ORDERS and ORDERS[oid] <= LIMIT and _wants_refund(str(row.get("prompt")))
    looked_first = not refunds or "lookup_order" in tools[: tools.index("issue_refund")]
    marks = {
        "looked_up_first": 1.0 if looked_first else 0.0,
        "refund_only_when_allowed": 1.0 if (allowed or not refunds) else 0.0,
        "refunds_when_eligible": (1.0 if refunds else 0.0) if allowed else None,
    }
    failed = [k for k, v in marks.items() if v == 0.0]
    return {
        "reward": 0.0 if failed else 1.0,
        "reason": ", ".join(failed) or "followed the policy",
        "markers": marks,
    }


def train_reward(row: dict) -> dict[str, Any]:
    """The training reward: one rule, looked up before refunding. Narrower
    than the judge on purpose, so the held-out scorer is not the reward."""
    mark = refund_judge(row)["markers"]["looked_up_first"]
    return {"reward": mark, "reason": "looked up first" if mark else "refunded blind"}


def length_judge(row: dict) -> dict[str, Any]:
    """An untrained behavior: the reply stays in band."""
    n = len(str(row.get("final_text") or ""))
    return {"reward": 1.0 if n <= LENGTH_BAND else 0.0, "reason": f"{n} chars"}


# ------------------------------------------------------------ the platform


class FakePlatform:
    """Records every call and answers like the API, including the verdict
    rule (delta +- sqrt(ci_a^2 + ci_b^2)). Drop transport= for the real one."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.behaviors: dict[str, dict] = {}
        self.versions: dict[str, str] = {}
        self.evals: dict[str, dict[str, dict]] = {}
        self.serving: str | None = None

    def __call__(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path, body))
        if path == "/runs":
            run_id = f"{body['agent']}-{body['version']}"
            self.versions[run_id] = body["version"]
            return {"id": run_id, "version": body["version"]}
        if path.endswith("/evals"):
            version = self.versions[path.split("/")[2]]
            for item in body:
                self.evals.setdefault(version, {})[item["behavior"]] = item
            return {"evals": body}
        if "/behaviors/" in path:
            name = path.rsplit("/", 1)[-1]
            self.behaviors[name] = {**body, "name": name}
            return self.behaviors[name]
        if path.endswith("/promote"):
            self.serving = body["version"]
            return {"ok": True}
        if "/dashboard" in path:
            return self.dashboard(path.split("behavior=")[-1] if "behavior=" in path else None)
        return {"id": (body or {}).get("id", "")} if isinstance(body, dict) else {"ok": True}

    def dashboard(self, target: str | None) -> dict[str, Any]:
        target = target or next(iter(self.behaviors))
        serving = self.serving
        others = [v for v in self.evals if v != serving]
        verdict: dict[str, Any] = {"serving": serving, "candidate": others[-1] if others else None}
        if serving in self.evals and others:
            cand, serv = self.evals[others[-1]], self.evals[serving]
            c, s = cand[target], serv[target]
            delta = round(c["score"] - s["score"], 1)
            verdict.update(
                delta=delta,
                excludesZero=abs(delta) > math.sqrt(c.get("ci", 0) ** 2 + s.get("ci", 0) ** 2),
                regressions=sum(
                    1
                    for b in cand
                    if b != target and b in serv and cand[b]["score"] < serv[b]["score"]
                ),
            )
        return {
            "agent": {"id": "refund-bot", "name": "refund-bot"},
            "behavior": self.behaviors[target],
            "behaviors": list(self.behaviors),
            "verdict": verdict,
        }


fake = FakePlatform()
OUT = Path(tempfile.mkdtemp(prefix="dpo-pairs-"))

# ============================================================ the playbook

# -- 1. frozen held-out test first, held out by task, with its noise floor
K = 4  # rollouts per ask
SIM = dict(
    tools=TOOLS,
    system_prompt=POLICY,
    simulator=False,
    mode="rl",
    repeats=K,
    repeat_policy="fixed",
    reproducible=True,
    seed=0,
    concurrency=1,
)
test = wai.simulate(
    before, seeds=TEST_ASKS, situations=len(TEST_ASKS), budget=len(TEST_ASKS) * K, **SIM
)
test_tasks = {r["scenario_id"] for r in test.trajectories}
base = wai.simulate(before, tasks=test, runs=3, advanced={"model_version": "v0"}, **SIM)
base_scored = wai.evaluate(base, refund_judge, tools=TOOLS)
floor = wai.eval_variance(base_scored.rows)
noise_floor = 100 * floor["run_std"]
print(f"frozen test: {len(test_tasks)} tasks, k={K}, noise floor {noise_floor:.1f} points")

assert len(test_tasks) == len(TEST_ASKS), "every test ask must be its own task"
assert not test.warnings and not base_scored.warnings, (test.warnings, base_scored.warnings)
assert floor["n_runs"] == 3 and floor["run_std"] is not None, floor
assert {r["model_version"] for r in base_scored.rows} == {"v0"}

# -- 2. roll the agent several times per training ask, graded in the loop
train = wai.simulate(
    before,
    seeds=TRAIN_ASKS,
    situations=len(TRAIN_ASKS),
    budget=len(TRAIN_ASKS) * K,
    grader=train_reward,
    **SIM,
)
rows = [dict(r) for r in train.trajectories]
assert not train.warnings, train.warnings
assert test_tasks.isdisjoint(r["scenario_id"] for r in rows), "test tasks leaked into training"

assert len(rows) == len(TRAIN_ASKS) * K
assert {r["lineage"]["source"] for r in rows} == {"grade"}, "grader= rows carry grade lineage"
assert {r["reward"] for r in rows} == {0, 1}, "the bot must both pass and fail"
assert {r["prompt"] for r in rows}.isdisjoint(r["prompt"] for r in test.trajectories)

# -- 3. pairs: same ask, rewards differ, closest in length
pairs, pair_report = wai.build_preference_pairs(rows, length_match=True)
print(
    f"pairs {pair_report['pairs']} from {pair_report['prompts_with_contrast']}"
    f"/{pair_report['prompts_seen']} asks, chosen longer in "
    f"{pair_report['length']['chosen_longer_frac']:.0%}"
)
for note in pair_report["warnings"]:
    print("!", note)
assert pairs, "no ask had both a pass and a fail; raise repeats or use a harder ask set"
assert pair_report["length"]["chosen_longer_frac"] < 0.8, "pairs would teach length"

for p in pairs:
    assert p["prompt"] == p["chosen"]["prompt"] == p["rejected"]["prompt"]
    assert p["chosen_score"] > p["rejected_score"] and p["margin"] >= 1.0
    assert p["first_turn_differs"], "the contrast must be in the first action"
assert 0 <= pair_report["length"]["chosen_longer_frac"] <= 1
assert pair_report["eval_sourced"] == 0 and pair_report["same_policy_pairs"] == len(pairs)
_, unmatched = wai.build_preference_pairs(rows, length_match=False)
assert "chosen_longer_frac" in unmatched["length"]

# -- 4. the pairs are not teaching a tic
sides = [p["chosen"] for p in pairs] + [p["rejected"] for p in pairs]
style = wai.style_report(sides)
length = wai.length_report(sides)
assert not style["warnings"], style["warnings"]
assert length["n_truncated"] == 0, length
print(style)  # the report prints itself: every marker, and the ones it did not stamp

assert set(style["markers"]) >= {"no_boilerplate", "no_hedging", "no_apology", "no_sycophancy"}
assert all(abs(v["reward_corr"] or 0) < style["threshold"] for v in style["markers"].values())
assert length["n"] == 2 * len(pairs)

# -- 5. nothing from the frozen test in the pairs
clean, decon = wai.decontaminate(pairs, against=[base_scored.rows])
assert decon["n_contaminated"] == 0, decon["examples"]
print(f"decontaminate: {decon['n_kept']} pairs kept against {decon['n_eval_texts']} test asks")

_, leak = wai.decontaminate(
    [*pairs, {**pairs[0], "prompt": TEST_ASKS[0]}], against=[base_scored.rows]
)
assert leak["n_contaminated"] == 1 and leak["n_exact"] == 1, leak

# -- 6. write the JSONL a DPO trainer loads
export = wai.export_preference(
    clean, str(OUT / "pairs.jsonl"), system_prompt=POLICY, tools=TOOLS, format="trl"
)
print(f"wrote {export['path']}: {export['pairs']} pairs")

lines = (OUT / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
assert len(lines) == len(clean) == export["pairs"]
for line in lines:
    item = json.loads(line)
    assert {"prompt", "chosen", "rejected"} <= set(item)
    assert item["prompt"][-1]["role"] == "user"
    assert item["chosen"][0]["role"] == "assistant" and item["rejected"][0]["role"] == "assistant"
    assert item["chosen"] != item["rejected"]
assert export["tool_call_roundtrip"]["invalid"] == 0, export["tool_call_roundtrip"]
assert not export.get("warnings"), export["warnings"]

# -- 7. score before and after on the frozen test, then report
cand = wai.simulate(after, tasks=test, runs=3, advanced={"model_version": "v1"}, **SIM)
cand_scored = wai.evaluate(cand, refund_judge, tools=TOOLS)
delta = wai.delta_report(base_scored.rows, cand_scored.rows, target="pass_at_1")
print(wai.format_delta_report(delta))
assert delta["ok"] and delta["target_verdict"] == "moved", delta["warnings"]

assert delta["replicated"] and delta["eval_runs"] == {"before": 3, "after": 3}
assert not delta["not_comparable"], delta["not_comparable"]
assert delta["n_paired_tasks"] == len(test_tasks)
assert delta["target_delta"] > 0 and delta["target_ci95"][0] > 0


def score(data: wai.SimulationData, judge) -> dict[str, float | int]:
    """One behavior on the frozen test: pass@1 in points, 95% half-width, tasks."""
    graded = wai.evaluate([dict(r) for r in data.trajectories], judge, tools=TOOLS)
    pa = wai.pass_at(graded.rows, k=K)
    lo, hi = pa.ci95
    return {"score": 100 * pa.pass_at_1, "ci": 100 * (hi - lo) / 2, "n": pa.n_groups}


JUDGES = {"refunds": refund_judge, "length": length_judge}
tracked = track(
    "refund-bot",
    model="Qwen/Qwen3-4B",
    harness=Harness(instructions=POLICY, tools=TOOLS),
    transport=fake,
)
tracked.behavior(
    Behavior(
        name="refunds",
        test_version="t1",
        n=len(test_tasks),
        judge=Judge(name="refund_judge, a program"),
        noise_floor=noise_floor,
        contamination=decon["n_contaminated"],
        reward_is_judge=False,
    )
)
tracked.behavior(Behavior(name="length", test_version="t1", n=len(test_tasks)))
v0 = tracked.run("v0", method="none")
for name, judge in JUDGES.items():
    v0.score(name, test_version="t1", **score(base, judge))
v0.finish()
tracked.promote("v0")
run = tracked.run("v1", method="DPO", targets=["refunds"], trained_on=["refund-pairs"])
for name, judge in JUDGES.items():
    run.score(name, test_version="t1", **score(cand, judge))
run.finish()
verdict = str(tracked.verdict("refunds"))
print(verdict)

# ============================================================ end of playbook

assert "v1" in verdict and "v0" in verdict and "beats" in verdict, verdict
assert set(run.scores) == set(v0.scores) == {"refunds", "length"}
assert run.scores["refunds"].score > v0.scores["refunds"].score
assert run.spec.method == "DPO" and run.spec.targets == ["refunds"]
paths = [(m, p.split("?")[0]) for m, p, _ in fake.calls]
assert ("POST", "/agents") in paths and ("POST", "/runs") in paths
assert ("PUT", "/agents/refund-bot/behaviors/refunds") in paths
assert ("PATCH", "/runs/refund-bot-v1") in paths
assert ("GET", "/agents/refund-bot/dashboard") in paths
harness_body = next(b for m, p, b in fake.calls if p == "/agents")["harness"]
assert harness_body["tools"] == ["lookup_order", "issue_refund"]

elapsed = time.monotonic() - T0
print(f"ok: dpo-pairs check passed in {elapsed:.1f}s")
sys.exit(0 if elapsed < 60 else 1)
