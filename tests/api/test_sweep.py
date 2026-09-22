"""HarnessSweep: many variants, one frozen test, one run per fingerprint."""

from __future__ import annotations

import re
from typing import Any

import pytest

import whileai.simulations as wai
from tests.api.test_platform import Fake
from whileai.platform import Harness, HarnessSweep, SweepReport, track
from whileai.simulations.score.stats import _t_quantile
from whileai.sweep import VariantResult

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


# ------------------------------------------------- the floor is the platform's t-band


def _fixed_rollouts(passing_by_seed: dict[int, int], tasks: int = 100, k: int = 10):
    """Rollouts whose pass count per seed is chosen: ``tasks`` asks x ``k``
    rows, the first ``passing_by_seed[seed]`` rows marked to pass."""

    def rollouts(self, agent, tasks_arg, seed=0):
        n_pass = passing_by_seed[seed]
        return [
            {
                "prompt": f"ask {i // k}",
                "task_id": f"t{i // k}",
                "steps": [],
                "final_text": agent(f"ask {i // k}")["final_text"],
                "expected": 1.0 if i < n_pass else 0.0,
            }
            for i in range(tasks * k)
        ]

    return rollouts


def _expected_judge(row: dict) -> dict[str, Any]:
    return {"reward": row["expected"], "reason": ""}


def _ok_agent(message: str) -> dict[str, Any]:
    return {"steps": [], "final_text": "ok"}


def test_noise_floor_is_the_platform_t_band_not_the_raw_difference(monkeypatch):
    # Two re-runs at 89.1 and 90.6: the old floor was |delta| = 1.5. The platform's
    # ``Tracked.noise_floor`` reads the same pair as t(df=1) x run_std x sqrt(2).
    monkeypatch.setattr(HarnessSweep, "_rollouts", _fixed_rollouts({0: 891, 1: 906}))
    fake = Fake()
    sweep = HarnessSweep(
        track("bot", transport=fake), judge=_expected_judge, k=10, behavior="policy", noise_runs=2
    )
    report = sweep.run([(Harness(label="only"), _ok_agent)], tasks=[])
    run_std = ((0.891 - 0.8985) ** 2 + (0.906 - 0.8985) ** 2) ** 0.5  # sample sd, n-1 = 1
    band = _t_quantile(1) * run_std * 2**0.5 * 100
    assert report.noise_floor == pytest.approx(band, abs=0.01) and band > 19
    assert report.noise_floor != 1.5
    assert report.run_std == pytest.approx(run_std, abs=1e-4) and report.noise_runs == 2
    text = str(report)
    assert "2 re-runs" in text and "wide" in text
    # The run's EvalSetup carries the sd, not the band; the behavior carries the band.
    run = next(b for m, p, b in fake.calls if p == "/runs")
    assert run["record"]["eval"]["runStd"] == pytest.approx(run_std, abs=1e-4)
    assert run["record"]["eval"]["runStdRuns"] == 2
    beh = next(b for m, p, b in fake.calls if "/behaviors/" in p)
    assert beh["noiseFloor"] == pytest.approx(band, abs=0.01)
    # Nothing was measured for these, so nothing is posted.
    assert "contamination" not in beh and "rewardIsJudge" not in beh


# --------------------------------------------------------- same denominators per arm


def test_an_arm_that_lost_a_graded_row_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(HarnessSweep, "_rollouts", _fixed_rollouts({0: 50}, tasks=5, k=4))

    def judge_that_breaks_on_boom(row: dict) -> dict[str, Any]:
        if row["final_text"] == "boom":
            raise RuntimeError("judge fell over")  # evaluate marks the row reward=None
        return {"reward": row["expected"]}

    def boom_agent(message: str) -> dict[str, Any]:
        return {"steps": [], "final_text": "boom" if message == "ask 2" else "ok"}

    sweep = HarnessSweep(
        track("bot", transport=Fake()), judge=judge_that_breaks_on_boom, k=4, noise_runs=1
    )
    with pytest.raises(ValueError, match=r"faulty.*graded 16 rows on 4 asks.*sound.*20 rows on 5"):
        sweep.run(
            [(Harness(label="sound"), _ok_agent), (Harness(label="faulty"), boom_agent)],
            tasks=[],
            post=False,
        )


def test_clears_is_false_when_the_arms_have_different_denominators():
    a = VariantResult("a", None, "aaaa", {"policy": (90.0, 1.0, 12)}, graded_rows=48)
    b = VariantResult("b", None, "bbbb", {"policy": (60.0, 1.0, 11)}, graded_rows=44)
    report = SweepReport("policy", "t-1", 12, 4, 0.0, None, [a, b])
    assert not report.clears(a, b) and report.best is None
    text = str(report)
    assert "48" in text and "44" in text and "12" in text and "11" in text
    assert "different asks" in text
    b.scores["policy"] = (60.0, 1.0, 12)
    assert report.clears(a, b)


# -------------------------------------------------- names refused before any rollout


def _never_called(message: str) -> dict:
    raise AssertionError("a rollout ran before the names were checked")


def test_invalid_behavior_name_is_refused_before_any_rollout():
    with pytest.raises(ValueError, match=r"behavior name 'refund policy'.*Behavior"):
        HarnessSweep(track("bot", transport=Fake()), judge=judge, behavior="refund policy")


def test_invalid_marker_name_is_refused_when_first_seen(monkeypatch):
    monkeypatch.setattr(HarnessSweep, "_rollouts", _fixed_rollouts({0: 20}, tasks=5, k=4))

    def judge_with_spaced_marker(row: dict) -> dict[str, Any]:
        return {"reward": row["expected"], "markers": {"refund only": 1.0}}

    sweep = HarnessSweep(track("bot", transport=Fake()), judge=judge_with_spaced_marker, k=4)
    with pytest.raises(ValueError, match=r"marker name 'refund only'"):
        sweep.run([(Harness(label="only"), _ok_agent)], tasks=[], post=False)
