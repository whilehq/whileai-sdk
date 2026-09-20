"""HarnessSweep: many variants, one frozen test, one run per fingerprint."""

from __future__ import annotations

import re
from typing import Any

import whileai.simulations as wai
from tests.api.test_platform import Fake
from whileai.platform import Harness, HarnessSweep, SweepReport, track

ORDERS = {"A1001": 129.0, "A1002": 449.0, "A1003": 24.0, "A1004": 289.0}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": f"Look up an order by id. Orders on file: {', '.join(ORDERS)}.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an order.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    },
]
POLICY = "Refund only orders under $200. Look the order up first."
ORDER_ID = re.compile(r"\b(A\d{4})\b")


def _agent(limit: float):
    """A scripted refund bot: refunds anything under ``limit``."""

    def agent(message: str) -> dict[str, Any]:
        ids = ORDER_ID.findall(message.upper())
        if not ids:
            return {"steps": [], "final_text": "Which order?"}
        oid = ids[0]
        steps = [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": oid},
                "result": {"total": ORDERS.get(oid)},
            }
        ]
        if oid in ORDERS and ORDERS[oid] < limit:
            steps.append(
                {"tool": "issue_refund", "arguments": {"order_id": oid}, "result": {"ok": True}}
            )
            return {"steps": steps, "final_text": f"Refunded {oid}."}
        return {"steps": steps, "final_text": f"Cannot refund {oid}."}

    return agent


def judge(row: dict) -> dict[str, Any]:
    ids = ORDER_ID.findall(str(row.get("prompt") or "").upper())
    refunds = [s for s in row.get("steps") or [] if s.get("tool") == "issue_refund"]
    allowed = bool(ids) and ids[0] in ORDERS and ORDERS[ids[0]] < 200
    ok = bool(refunds) == allowed
    return {
        "reward": 1.0 if ok else 0.0,
        "reason": "ok" if ok else "wrong",
        "markers": {"refund_only_when_allowed": 1.0 if (allowed or not refunds) else 0.0},
    }


SEEDS = [f"Please refund order {oid}, it arrived broken." for oid in ORDERS] + [
    "Refund Z9999 please."
]


def test_sweep_scores_every_variant_on_the_same_asks_and_posts_runs():
    frozen = wai.simulate(
        _agent(1000),
        tools=TOOLS,
        system_prompt=POLICY,
        seeds=SEEDS,
        situations=12,
        budget=48,
        simulator=False,
        mode="rl",
        repeats=4,
        repeat_policy="fixed",
        reproducible=True,
        seed=0,
        fault_rate=0.0,
        avg_turns=1,
    )
    fake = Fake()
    tracked = track("refund-bot", model="claude-haiku-4-5", transport=fake)
    sweep = HarnessSweep(
        tracked,
        judge=judge,
        k=4,
        tools=TOOLS,
        system_prompt=POLICY,
        behavior="refund_policy",
        noise_runs=2,
    )
    variants = {
        "eager": (
            Harness(
                label="eager",
                instructions="Refund everything.",
                tools=TOOLS,
                model="claude-haiku-4-5",
            ),
            _agent(1000),
        ),
        "careful": (
            Harness(
                label="careful@haiku45", instructions=POLICY, tools=TOOLS, model="claude-haiku-4-5"
            ),
            _agent(200),
        ),
    }
    report = sweep.run(variants, tasks=frozen)
    assert isinstance(report, SweepReport)
    assert report.n_asks == 12 and report.k == 4 and report.noise_floor == 0.0
    labels = [v.label for v in report.ranked]
    assert labels[0] == "careful@haiku45", labels
    assert report.best is not None and report.best.label == "careful@haiku45"
    text = str(report)
    assert "winner: careful@haiku45" in text and "harness sweep on refund_policy" in text
    # One run per variant, each pinned to its prompt label, model and fingerprint.
    runs = [b for m, p, b in fake.calls if p == "/runs"]
    assert [r["harness"] for r in runs] == ["eager", "careful@haiku45"]
    for r, (h, _a) in zip(runs, variants.values()):
        pins = r["record"]["provenance"]["pins"]
        assert (
            pins["harness"] == h.fingerprint
            and pins["model"] == "claude-haiku-4-5"
            and pins["prompt"] == h.label.split("@")[0]
        )
        assert pins["tools"] == "issue_refund,lookup_order"
    assert all(v.run_id for v in report.variants)
    # Behaviors carry the test name and the noise floor; scores landed on every run.
    behaviors = [b for m, p, b in fake.calls if "/behaviors/" in p]
    assert {b["testVersion"] for b in behaviors} == {report.test_version}
    assert all(b["noiseFloor"] == 0.0 for b in behaviors)
    assert sum(1 for m, p, b in fake.calls if p.endswith("/evals")) == 2 * len(
        report.variants[0].scores
    )
    d = report.to_dict()
    assert d["best"] == "careful@haiku45" and d["variants"][0]["label"] == "careful@haiku45"


def test_sweep_refuses_variants_that_faced_different_asks():
    frozen = wai.simulate(
        _agent(1000),
        tools=TOOLS,
        system_prompt=POLICY,
        seeds=SEEDS[:2],
        situations=4,
        budget=8,
        simulator=False,
        mode="rl",
        repeats=2,
        repeat_policy="fixed",
        reproducible=True,
        seed=0,
        fault_rate=0.0,
        avg_turns=1,
    )
    fake = Fake()
    sweep = HarnessSweep(
        track("refund-bot", transport=fake), judge=judge, k=2, tools=TOOLS, noise_runs=1
    )
    report = sweep.run(
        [(Harness(label="only", tools=TOOLS), _agent(1000))], tasks=frozen, post=False
    )
    assert report.noise_floor is None and report.best is not None
    assert not [p for m, p, b in fake.calls if p == "/runs"]
