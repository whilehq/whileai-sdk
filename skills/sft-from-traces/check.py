"""sft-from-traces: the playbook in SKILL.md, run offline.

Every ```python block in SKILL.md appears here verbatim (tests/skills
enforces it). No key, no model, no network, seconds: the traces come from
a scripted refund agent, the "after" agent is a second script that slips
less, and the platform is a recording fake that answers like the API.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

import whileai.simulations as wai
from whileai.platform import Behavior, Harness, Judge, track

# ------------------------------------------------------------ the agent's world

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

# The order ids are in the description on purpose: a situation writer that
# does not know which ids exist invents ones, and every rollout is "not found".
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
    """The policy as a program. The judge and the agents share it."""
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


def wants_refund(message: str) -> bool:
    text = message.lower()
    return any(w in text for w in ("refund", "money back", "return", "charge", "reimburse"))


RAMBLE = (
    " I have also noted this on the account for you, and if anything else comes up about this "
    "order or any other order please do not hesitate to reach out again at any time and one of "
    "us will be glad to take another careful look for you."
)


def scripted_agent(*, slip: float, ramble: float, seed: int):
    """``agent(message) -> {"steps", "final_text"}``, the wrapper shape the SDK plays.

    ``slip`` is the chance it refunds an order the policy forbids (the flaw
    the traces show). ``ramble`` is the chance a reply runs long (the
    behavior nobody trains, scored so a regression would show).
    """
    rng = random.Random(seed)

    def agent(message: str) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        tail = RAMBLE if rng.random() < ramble else ""
        ids = ORDER_ID.findall(message.upper())
        if not ids:
            return {
                "steps": steps,
                "final_text": "Happy to help. Which order id is this about?" + tail,
            }
        order_id = ids[0]
        order = lookup_order(order_id)
        steps.append({"tool": "lookup_order", "arguments": {"order_id": order_id}, "result": order})
        if "error" in order:
            text = f"I could not find an order {order_id}. Can you check the id?"
            return {"steps": steps, "final_text": text + tail}
        if not wants_refund(message):
            text = f"Order {order_id} ({order['item']}) is {order['status']}. Anything else?"
            return {"steps": steps, "final_text": text + tail}
        ok, why = refundable(order)
        if ok or rng.random() < slip:
            result = {"ok": True, "order_id": order_id, "amount": order["total"]}
            steps.append(
                {
                    "tool": "issue_refund",
                    "arguments": {"order_id": order_id, "amount": order["total"]},
                    "result": result,
                }
            )
            text = f"Done: ${order['total']:.2f} refunded for order {order_id}."
            return {"steps": steps, "final_text": text + tail}
        if "manager" in why:
            text = f"Order {order_id} is over ${REFUND_LIMIT:.0f}, so a manager will follow up."
            return {"steps": steps, "final_text": text + tail}
        return {"steps": steps, "final_text": f"I cannot refund order {order_id}: {why}." + tail}

    return agent


# The agent in production today, and the stand-in for the fine-tuned one.
# With a real model, ``after_agent`` wraps the adapter the same way.
before_agent = scripted_agent(slip=0.5, ramble=0.3, seed=1)
after_agent = scripted_agent(slip=0.05, ramble=0.3, seed=2)

# ------------------------------------------------------------------ the judges


def branch_of(prompt: str) -> str:
    """Which policy branch an ask lands in, from the order it names."""
    ids = ORDER_ID.findall(prompt.upper())
    if not wants_refund(prompt) or not ids:
        return "no_refund_asked"
    order = lookup_order(ids[0])
    if "error" in order:
        return "unknown_order"
    ok, why = refundable(order)
    if ok:
        return "eligible"
    if "manager" in why:
        return "over_limit"
    return "not_delivered" if "delivered" in why else "outside_window"


def refund_judge(row: dict) -> dict[str, Any]:
    """The refund policy read off the trajectory: which tools ran, with what."""
    steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
    prompt = str(row.get("prompt") or "")
    final = str(row.get("final_text") or "").lower()
    lookups = [s for s in steps if s.get("tool") == "lookup_order"]
    refunds = [s for s in steps if s.get("tool") == "issue_refund"]
    branch = branch_of(prompt)
    marks: dict[str, float | None] = {
        "looked_up_first": None,
        "refund_only_when_allowed": 1.0 if (branch == "eligible" or not refunds) else 0.0,
        "refunds_when_eligible": None,
        "escalates_over_limit": None,
    }
    if refunds:
        marks["looked_up_first"] = 1.0 if steps.index(lookups[0]) < steps.index(refunds[0]) else 0.0
    if branch == "eligible":
        marks["refunds_when_eligible"] = 1.0 if refunds else 0.0
    if branch == "over_limit":
        marks["escalates_over_limit"] = 1.0 if "manager" in final else 0.0
    failed = [name for name, v in marks.items() if v == 0.0]
    return {
        "reward": 0.0 if failed else 1.0,
        "reason": f"{branch}: " + (", ".join(failed) if failed else "followed the policy"),
        "markers": marks,
        "failure_class": failed[0] if failed else None,
    }


def length_judge(row: dict) -> dict[str, Any]:
    """The behavior nobody trains: a reply stays under 40 words."""
    words = len(str(row.get("final_text") or "").split())
    return {"reward": 1.0 if words <= 40 else 0.0, "reason": f"{words} words"}


# ------------------------------------------------------------ the trace fixture

ASK_TEMPLATES = [
    "I want a refund for order {o}, it did not fit.",
    "Please refund {o}.",
    "Money back on {o} please, it arrived broken.",
    "Can I return {o}? I changed my mind.",
    "Refund {o} right now or I dispute the charge.",
    "Hi, I would like to be reimbursed for {o}.",
    "What is the status of order {o}?",
    "Is {o} delivered yet? No refund needed, just checking.",
]
EXTRA_ASKS = [
    "Can you refund order Z9999?",
    "Refund B7777 please, it never arrived.",
    "I want my money back.",
    "Where is my package?",
    "Please refund order Q1234.",
    "Status of order X0001?",
    "I need a refund but I lost the order number.",
    "Return request for order M5555, wrong size.",
]


def write_traces(path: Path, agent) -> int:
    """Forty production rows: one ask per task, the agent's steps and reply,
    and the label production already has (here the policy program plays the
    reviewer; in real traces it is a person or the grader you accept)."""
    asks = [t.format(o=o) for o in ORDERS for t in ASK_TEMPLATES] + EXTRA_ASKS
    rows = []
    for i, ask in enumerate(asks):
        row = {"task_id": f"task-{i:03d}", "prompt": ask, **agent(ask)}
        row["reward"] = refund_judge(row)["reward"]
        rows.append(row)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


# ------------------------------------------------------- the platform, offline


class FakePlatform:
    """Records every call and answers like the API, including the verdict rule:
    delta +- sqrt(ci_candidate^2 + ci_served^2) must exclude zero."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.scores: dict[str, dict[str, dict[str, Any]]] = {}
        self.serving: str | None = None

    def __call__(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path, body))
        parts = path.strip("/").split("/")
        if path == "/agents":
            return {"id": body["id"], "name": body["name"]}
        if method == "PUT" and "behaviors" in parts:
            self.behaviors[parts[-1]] = {"name": parts[-1], **(body or {})}
            return self.behaviors[parts[-1]]
        if path == "/runs":
            run_id = f"{body['agent']}-{body['version']}"
            self.runs[run_id] = body
            return {"id": run_id, "version": body["version"]}
        if path.endswith("/evals"):
            version = self.runs[parts[1]]["version"]
            for item in body:
                self.scores.setdefault(version, {})[item["behavior"]] = item
            return {"evals": body}
        if path.endswith("/promote"):
            self.serving = body["version"]
            return {"serving": self.serving}
        if "dashboard" in parts[-1]:
            return self.dashboard(path.split("behavior=")[1] if "behavior=" in path else None)
        return {"ok": True}

    def dashboard(self, name: str | None) -> dict[str, Any]:
        name = name or next(iter(self.behaviors))
        serving = self.serving
        candidates = [v for v in self.scores if v != serving]
        cand = candidates[-1] if candidates else None
        verdict: dict[str, Any] = {"candidate": cand, "serving": serving}
        if cand and serving and name in self.scores[cand] and name in self.scores[serving]:
            a, b = self.scores[cand][name], self.scores[serving][name]
            delta = round(a["score"] - b["score"], 1)
            verdict["delta"] = delta
            if a.get("ci") is not None and b.get("ci") is not None:
                verdict["excludesZero"] = abs(delta) > math.sqrt(a["ci"] ** 2 + b["ci"] ** 2)
            verdict["regressions"] = sum(
                1
                for other, s in self.scores[cand].items()
                if other != name
                and other in self.scores[serving]
                and s["score"] < self.scores[serving][other]["score"]
            )
        return {
            "agent": {
                "id": "refund-bot",
                "name": "refund-bot",
                "serving": serving,
                "candidate": cand,
            },
            "behavior": self.behaviors.get(name),
            "behaviors": list(self.behaviors),
            "verdict": verdict,
        }


fake = FakePlatform()

# ------------------------------------------------------------------- the steps

started = time.monotonic()
OUT_DIR = Path(tempfile.mkdtemp(prefix="sft-from-traces-"))
TRACES_PATH = OUT_DIR / "traces.jsonl"
n_traces = write_traces(TRACES_PATH, before_agent)
assert n_traces == 40
K = 4  # rollouts per held-out task, per run
TEST_VERSION = "heldout-v1"

# 1. Load the traces and read what they say.
traces = wai.load_traces(str(TRACES_PATH))
report = wai.trace_report(traces, tools=TOOLS, policy=POLICY)
print(wai.format_trace_report(report))

assert len(traces) == 40, report
assert sum(1 for t in traces if t.get("reward") == 0) >= 5, (
    "the fixture agent should fail some traces"
)

# 2. Freeze the held-out test first, by task.
tasks = sorted({t["task_id"] for t in traces})
held = set(tasks[::3])
heldout_traces = [t for t in traces if t["task_id"] in held]
train_traces = [t for t in traces if t["task_id"] not in held]


def roll_heldout(agent, version: str) -> wai.SimulationData:
    """The frozen test, rolled three times so the re-run spread is measurable."""
    return wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        tasks=heldout_traces,  # the held-out asks, keyed by task_id
        simulator=False,
        mode="rl",
        repeats=K,
        repeat_policy="fixed",
        runs=3,
        reproducible=True,
        seed=0,
        concurrency=1,
        advanced={"model_version": version},  # tells the two arms apart
    )


before_data = roll_heldout(before_agent, "refund-bot-base")
heldout_asks = [t["prompt"] for t in heldout_traces]
before = wai.evaluate(
    before_data.rows(), refund_judge, model="base", tools=TOOLS, eval_set=heldout_asks
)
before_len = wai.evaluate(before_data.rows(), length_judge, model="base", eval_set=heldout_asks)
noise = wai.eval_variance(before.rows)
noise_len = wai.eval_variance(before_len.rows)
print(f"held out {len(held)} of {len(tasks)} tasks; base pass@1 per run {noise['means']}")
print(f"noise floor {noise['run_std_points']} points ({noise['stability']})")

assert not held & {t["task_id"] for t in train_traces}
assert not before.warnings, before.warnings
assert before.eval_coverage["missing"] == [], before.eval_coverage
assert {r["scenario_id"] for r in before.rows} == held
assert noise["n_runs"] == 3 and noise_len["n_runs"] == 3

tracked = track(
    "refund-bot",
    model="Qwen/Qwen3-4B",
    harness=Harness(instructions=POLICY, tools=TOOLS),
    transport=fake,  # drop transport= to talk to the real platform
)
tracked.behavior(
    Behavior(
        name="refund_policy",
        test_version=TEST_VERSION,
        n=len(held),
        judge=Judge(name="refund policy as a program"),
        noise_floor=noise["run_std_points"],
        reward_is_judge=False,  # a program reading the tool calls is a verifier
        description="Refund only when the policy allows",
    )
)
tracked.behavior(
    Behavior(
        name="length",
        test_version=TEST_VERSION,
        n=len(held),
        noise_floor=noise_len["run_std_points"],
        description="Reply under 40 words; not trained",
    )
)

assert set(fake.behaviors) == {"refund_policy", "length"}
assert fake.behaviors["refund_policy"]["testVersion"] == TEST_VERSION

# 3. More situations around the failures, offline.
mined = wai.mine_traces(train_traces)
failing_asks = sorted({train_traces[i]["prompt"] for i in mined["flaw_rows"]})
pool = wai.simulate(
    before_agent,
    tools=TOOLS,
    system_prompt=POLICY,
    seeds=failing_asks,  # the asks the agent failed, real ids and all
    grader=refund_judge,  # graded in the loop; the search re-rolls failures
    simulator=False,  # offline writer; drop for the hosted one
    mode="rl",
    repeats=10,  # rejection sampling wants 10 to 30 per ask
    repeat_policy="fixed",
    situations=len(failing_asks),
    budget=len(failing_asks) * 10,
    reproducible=True,
    seed=1,
    concurrency=1,
    advanced={"model_version": "refund-bot-base"},
)
candidates = pool.rows() + [t for t in train_traces if t.get("reward") == 1]
print(
    f"{len(failing_asks)} failing asks -> {len(pool.rows())} rollouts, {len(candidates)} candidates"
)

assert len(failing_asks) >= 4, mined
assert not pool.warnings, pool.warnings
assert all(r.get("steps") for r in pool.rows()), "the writer lost the order ids"
assert {r["prompt"] for r in pool.rows()} == set(failing_asks)
assert not {r["prompt"] for r in candidates} & set(heldout_asks)

# 4. Keep the demonstrations the judge approved, one distinct way of being right per ask.
selected, selection = wai.select_for_sft(candidates, target=200)
print(f"selected {len(selected)} of {selection['n']} (eligible {selection['n_eligible']})")

assert selected and all(r["reward"] == 1.0 for r in selected)
assert selection["eval_sourced"] == 0, "training rows must not come from evaluate()"
assert len({r["prompt"] for r in selected}) == len(selected), "one demo per ask"

# 5. Nothing from the held-out tasks may reach the file.
clean, contamination = wai.decontaminate(selected, against=[heldout_traces])
print(
    f"decontaminate: dropped {contamination['n_contaminated']} of {contamination['n']} "
    f"({contamination['n_near']} near copies of a held-out ask, {contamination['n_same_task']} same task)"
)

assert contamination["n_same_task"] == 0, "held out by task, so no task id may match"
assert len(clean) == contamination["n_kept"] == len(selected) - contamination["n_contaminated"]
assert not {r.get("scenario_id") or r.get("task_id") for r in clean} & held
assert not {r["prompt"] for r in clean} & set(heldout_asks)
# the gate bites when it should: the held-out passes themselves are all caught by task id
_, leak = wai.decontaminate(
    [t for t in heldout_traces if t["reward"] == 1], against=[heldout_traces]
)
assert leak["n_same_task"] == leak["n"] > 0, leak

# 6. Export what TRL loads, in the shape it trains on.
SFT_PATH = OUT_DIR / "refund-sft.trl.jsonl"
export = wai.export_dataset(clean, str(SFT_PATH), system_prompt=POLICY, tools=TOOLS, format="trl")
print(f"wrote {export['n_written']} rows to {export['path']}; mask {export['mask_mode']}")

sft_rows = [json.loads(line) for line in SFT_PATH.read_text(encoding="utf-8").splitlines()]
assert len(sft_rows) == len(clean) == export["n_written"]
assert all(r["messages"][0]["role"] == "system" for r in sft_rows)
assert all("loss_mask" not in r for r in sft_rows), "TRL reads no per-message mask (#507)"
assert export["mask_mode"].startswith("TRL trains on every token"), export["mask_mode"]
assert all("prompt" not in r and r.get("prompt_text") for r in sft_rows), "TRL shape"
assert export["rewards"]["n_fail"] == 0 and export["tool_call_roundtrip"]["invalid"] == 0

# 7. Score the after agent on the same frozen test, then compare.
after_data = roll_heldout(after_agent, "refund-bot-sft")
after = wai.evaluate(
    after_data.rows(), refund_judge, model="sft", tools=TOOLS, eval_set=heldout_asks
)
after_len = wai.evaluate(after_data.rows(), length_judge, model="sft", eval_set=heldout_asks)
delta = wai.delta_report(before.rows, after.rows, target="pass_at_1")
delta_len = wai.delta_report(
    before_len.rows, after_len.rows, target="pass_at_1", must_not_regress=["pass_at_1"]
)
print(
    f"refund_policy: {delta['target_verdict']} {delta['target_delta']:+.3f} "
    f"[{delta['target_ci95'][0]:.3f}, {delta['target_ci95'][1]:.3f}] "
    f"on {delta['n_paired_tasks']} tasks, replicated={delta['replicated']}"
)
print(f"length: {delta_len['target_verdict']} {delta_len['target_delta']:+.3f}")

assert {r["scenario_id"] for r in after.rows} == held
assert delta["n_paired_tasks"] == len(held) and delta["replicated"]
assert delta["target_verdict"] == "moved" and delta["target_ci95"][0] > 0, delta
assert not delta["not_comparable"], delta["not_comparable"]
assert delta_len["ok"], delta_len["warnings"]


# 8. Report every behavior for both versions, then read the verdict.
def points(scored: wai.ScoredData) -> dict[str, Any]:
    """pass@1 on the 0-100 scale the platform draws, ci as the 95% half-width."""
    pa = wai.pass_at(scored.rows, k=K)
    lo, hi = pa.ci95
    return {"score": round(100 * pa.pass_at_1, 1), "ci": round(50 * (hi - lo), 1), "n": pa.n_groups}


base = tracked.run("base", method="none")
base.score("refund_policy", test_version=TEST_VERSION, **points(before))
base.score("length", test_version=TEST_VERSION, **points(before_len))
base.finish()
tracked.promote("base")  # what production serves today

run = tracked.run("v1", method="SFT", targets=["refund_policy"], trained_on=["refund-sft"])
run.score("refund_policy", test_version=TEST_VERSION, **points(after))
run.score("length", test_version=TEST_VERSION, **points(after_len))  # the untrained one too
run.finish(hours=0.4, gpu="1xH100", cost_usd=6)
verdict = str(tracked.verdict("refund_policy"))
print("verdict:", verdict)

assert fake.serving == "base" and set(fake.scores) == {"base", "v1"}
assert set(fake.scores["v1"]) == {"refund_policy", "length"}, "score every behavior"
assert "v1" in verdict and "base" in verdict, verdict
assert "beats" in verdict, verdict
assert fake.scores["v1"]["refund_policy"]["score"] > fake.scores["base"]["refund_policy"]["score"]

elapsed = time.monotonic() - started
print(f"ok in {elapsed:.1f}s")
assert elapsed < 60, elapsed
sys.exit(0)
