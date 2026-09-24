"""skills/audit-your-judge, runnable: every ```python block of SKILL.md,
verbatim, over a fixed set of graded transcripts with known truth. Offline,
no key, no model, no network, seconds.

The setup (this file only): a returns-support domain (``WINDOW_DAYS``),
60 fixed transcripts (``ROWS``) whose true correctness (``truth``) is known
by construction, three scripted judges over those rows (``always_pass``,
``generous``, ``accurate``), ``BLIND_LABELS`` (a codebook read, standing in
for a person who never sees ``truth``), and ``fake``, a recording transport
that answers like the platform API. Everything below the setup line is the
playbook.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from typing import Any

import whileai.simulations as wai
from whileai.judge_comparison import compare_judges
from whileai.platform import Behavior, Data, EvalSetup, Harness, Judge, RunRecord, track

T0 = time.monotonic()

# ------------------------------------------------------------------ the world

WINDOW_DAYS = 30  # a refund is allowed only for an order inside this many days


def _row(
    rid: str, prompt: str, order_id: str, days: int, refunded: bool, final_text: str, truth: int
) -> dict:
    steps = [
        {
            "tool": "lookup_order",
            "arguments": {"order_id": order_id},
            "result": {"status": "delivered", "days_since_order": days},
        }
    ]
    if refunded:
        steps.append(
            {
                "tool": "issue_refund",
                "arguments": {"order_id": order_id, "amount": 40.0},
                "result": {"ok": True},
            }
        )
    return {
        "rollout_id": rid,
        "prompt": prompt,
        "steps": steps,
        "final_text": final_text,
        "truth": truth,
    }


def _fixture_rows() -> list[dict]:
    """60 transcripts: 24 the agent handled correctly, 36 it did not. The
    36 split into three failure shapes on purpose, so the three judges below
    disagree on them in three different ways."""
    rows: list[dict] = []
    for i in range(24):  # correct: eligible, refunded, said so plainly
        oid = f"E{100 + i}"
        rows.append(
            _row(
                f"ok-{i}",
                f"Please refund order {oid}, it arrived damaged.",
                oid,
                5 + (i % 20),
                True,
                f"Done: refunded $40 for order {oid}.",
                1,
            )
        )
    for i in range(15):  # over-refunded: outside the window, refunded anyway,
        oid = f"O{200 + i}"  # same reassuring text as a correct refund
        rows.append(
            _row(
                f"over-{i}",
                f"Refund order {oid} please.",
                oid,
                WINDOW_DAYS + 5 + i,
                True,
                f"Done: refunded $40 for order {oid}.",
                0,
            )
        )
    for i in range(3):  # over-refunded, but the lookup never reported an age:
        oid = f"G{300 + i}"  # a payload gap, not a judge failure
        steps = [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": oid},
                "result": {"status": "delivered"},
            },
            {
                "tool": "issue_refund",
                "arguments": {"order_id": oid, "amount": 40.0},
                "result": {"ok": True},
            },
        ]
        rows.append(
            {
                "rollout_id": f"gap-{i}",
                "prompt": f"Refund order {oid} please.",
                "steps": steps,
                "final_text": f"Done: refunded $40 for order {oid}.",
                "truth": 0,
            }
        )
    for i in range(18):  # wrongly refused: eligible, but the agent declined
        oid = f"D{400 + i}"
        rows.append(
            _row(
                f"refuse-{i}",
                f"I'd like a refund for order {oid}.",
                oid,
                3 + (i % 10),
                False,
                f"Sorry, I can't process a refund for order {oid}.",
                0,
            )
        )
    return rows


ROWS = _fixture_rows()
assert len(ROWS) == 60
assert sum(r["truth"] for r in ROWS) == 24

# --------------------------------------------------------------- the judges


def always_pass(row: dict) -> dict[str, Any]:
    """Reads nothing, says pass. The default a training run silently falls back to."""
    return {"reward": 1.0, "reason": "pass"}


def generous(row: dict) -> dict[str, Any]:
    """Reads only the reply's tone: a dollar figure or 'refunded' reads as done."""
    text = str(row.get("final_text") or "").lower()
    ok = "refunded" in text or "$" in text
    return {
        "reward": 1.0 if ok else 0.0,
        "reason": "reply reads as resolved" if ok else "reply reads as unresolved",
    }


def accurate(row: dict) -> dict[str, Any]:
    """Reads the tool calls: a refund is correct only when the lookup's own
    fields said the order was still inside the window."""
    steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
    lookups = [s for s in steps if s.get("tool") == "lookup_order"]
    refunds = [s for s in steps if s.get("tool") == "issue_refund"]
    eligible = True
    for s in lookups:
        result = s.get("result") or {}
        days = result.get("days_since_order")
        if days is None:
            continue  # the payload carries no age; this judge cannot see the violation
        if result.get("status") != "delivered" or days > WINDOW_DAYS:
            eligible = False
    ok = bool(refunds) == eligible
    markers = {
        "looked_up_first": 1.0 if lookups else 0.0,
        "no_refund_when_ineligible": 0.0 if (refunds and not eligible) else 1.0,
    }
    return {
        "reward": 1.0 if ok else 0.0,
        "reason": "matches the eligibility rule"
        if ok
        else "refund decision contradicts the lookup",
        "markers": markers,
    }


def generous_with_clause(row: dict) -> dict[str, Any]:
    """generous, plus a rubric clause requiring a literal 'confirmed:' prefix."""
    if "confirmed:" not in str(row.get("final_text") or "").lower():
        return {"reward": 0.0, "reason": "missing the required 'confirmed:' prefix"}
    return generous(row)


def accurate_with_clause(row: dict) -> dict[str, Any]:
    """accurate, plus the same clause: unaffected, since it never read the wording."""
    if "confirmed:" not in str(row.get("final_text") or "").lower():
        return dict(accurate(row), reason="prefix missing, eligibility check unaffected")
    return accurate(row)


def prefers(a: dict, b: dict) -> str:
    """A position-biased pairwise pick over `generous`: real disagreement wins;
    a genuine tie goes to whichever side is handed first."""
    ra, rb = generous(a)["reward"], generous(b)["reward"]
    if ra != rb:
        return "a" if ra > rb else "b"
    return "a"


# =========================================================== the playbook

# ---- 1/2. sample and blind-label. BLIND_LABELS applies the codebook (the
# eligibility rule read off the tool calls) independently of any judge, the
# way a person would if handed the same transcripts with no verdict attached.
BLIND_LABELS = [
    {"key": r["rollout_id"], "label": r["truth"], "annotator": "reviewer"} for r in ROWS
]

# ---- 3. attach the labels, then compare judges
labeled, label_report = wai.attach_labels(ROWS, BLIND_LABELS, annotator="reviewer", kind="human")
table = compare_judges(
    labeled, {"always pass": always_pass, "generous": generous, "accurate": accurate}
)
print(table)

assert label_report["rows_labeled"] == 60, label_report
assert table.n_rows == 60
always_pass_score, generous_score, accurate_score = (
    table["always pass"],
    table["generous"],
    table["accurate"],
)

assert always_pass_score.agreement == 0.4, always_pass_score.agreement
assert always_pass_score.kappa is not None and abs(always_pass_score.kappa) < 1e-9, (
    always_pass_score.kappa  # "just says pass": kappa is 0, exactly
)
assert always_pass_score.leak == 1.0, always_pass_score.leak
assert generous_score.agreement == 0.7, generous_score.agreement  # looks like a real judge...
assert generous_score.kappa is not None and generous_score.kappa < 0.5, generous_score.kappa
assert generous_score.leak == 0.5, generous_score.leak  # ...but passes half of what really failed
assert not generous_score.ok
assert not always_pass_score.ok
assert accurate_score.ok, accurate_score
assert accurate_score.agreement is not None and accurate_score.agreement >= 0.8
assert accurate_score.kappa is not None and accurate_score.kappa >= 0.6
assert table.best is not None and table.best.name == "accurate", table.best

# ---- 5b. length bias: correlate reply length with reward on every judged arm


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5


def length_bias(judge_score) -> float:
    """Pearson correlation between reply length and judge reward: the
    check AlpacaEval's length control exists because a judge skips."""
    graded = [r for r in judge_score.rows if r.get("reward") is not None]
    lengths = [float(len(str(r.get("final_text") or ""))) for r in graded]
    rewards = [float(r["reward"]) for r in graded]
    return pearson(lengths, rewards)


generous_length_bias = length_bias(generous_score)
accurate_length_bias = length_bias(accurate_score)
print(
    f"length x reward correlation: generous {generous_length_bias:+.2f}, "
    f"accurate {accurate_length_bias:+.2f}"
)

assert generous_length_bias < -0.9, generous_length_bias  # shorter replies read as "resolved"
assert abs(accurate_length_bias) < abs(generous_length_bias), (
    accurate_length_bias,
    generous_length_bias,
)

# ---- 6. ablate: add one clause, re-run, read the delta per judge


def mean_reward(judge_score) -> float:
    """Mean judge reward across a JudgeScore's own graded rows."""
    graded = [r["reward"] for r in judge_score.rows if r.get("reward") is not None]
    return sum(graded) / len(graded)


baseline = compare_judges(labeled, {"generous": generous, "accurate": accurate})
with_clause = compare_judges(
    labeled, {"generous": generous_with_clause, "accurate": accurate_with_clause}
)
for name in ("generous", "accurate"):
    delta = mean_reward(with_clause[name]) - mean_reward(baseline[name])
    print(f"clause delta, {name}: {delta:+.3f}")

generous_delta = mean_reward(with_clause["generous"]) - mean_reward(baseline["generous"])
accurate_delta = mean_reward(with_clause["accurate"]) - mean_reward(baseline["accurate"])
assert generous_delta < -0.5, generous_delta  # the weak judge collapses
assert accurate_delta == 0.0, accurate_delta  # the judge that reads substance does not move

# ---- 7. reverse the presentation order


def flip_rate(pairs: list[tuple[dict, dict]]) -> float:
    flips = 0
    for a, b in pairs:
        winner_forward = a if prefers(a, b) == "a" else b
        winner_backward = b if prefers(b, a) == "a" else a
        if winner_forward is not winner_backward:
            flips += 1
    return flips / len(pairs)


ok_rows = [r for r in ROWS if r["rollout_id"].startswith("ok-")]
refuse_rows = [r for r in ROWS if r["rollout_id"].startswith("refuse-")]
tied_pairs = list(zip(ok_rows[::2], ok_rows[1::2]))  # both score 1 under `generous`: a real tie
decisive_pairs = list(zip(ok_rows, refuse_rows))  # one scores 1, one scores 0: real signal

tied_flips = flip_rate(tied_pairs)
decisive_flips = flip_rate(decisive_pairs)
print(f"order reversal: {tied_flips:.0%} of ties flip, {decisive_flips:.0%} of decisive pairs flip")

assert tied_flips == 1.0, tied_flips  # every tie is decided by position alone
assert decisive_flips == 0.0, decisive_flips  # real disagreement does not move

# ---- 8. only then train: grade a fresh held-out set with the judge that
# cleared the floors, score every behavior, report

LIVE_ORDERS: dict[str, dict[str, Any]] = {
    "L1001": {"status": "delivered", "days_since_order": 4},
    "L1002": {"status": "delivered", "days_since_order": 50},
    "L1003": {"status": "delivered", "days_since_order": 12},
    "L1004": {"status": "shipped", "days_since_order": 1},
}

LIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": f"Look up an order by id. Orders on file: {', '.join(LIVE_ORDERS)}.",
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
            "description": "Issue a refund for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]

LIVE_POLICY = (
    f"You are the returns assistant. Refunds are allowed only for delivered orders "
    f"within {WINDOW_DAYS} days of the order date. Always look the order up first."
)

ORDER_ID = re.compile(r"\b([A-Z]\d{4})\b")


def _lookup(order_id: str) -> dict[str, Any]:
    order = LIVE_ORDERS.get(order_id.upper())
    return (
        {"error": f"no order {order_id}"}
        if order is None
        else {"order_id": order_id.upper(), **order}
    )


def _eligible_live(order: dict[str, Any]) -> bool:
    return (
        "error" not in order
        and order["status"] == "delivered"
        and order["days_since_order"] <= WINDOW_DAYS
    )


def shipped(message: str) -> dict[str, Any]:
    """v1, the agent in production: refunds whatever order it finds."""
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": [], "final_text": "Which order is this about?"}
    oid = ids[0]
    order = _lookup(oid)
    step = {"tool": "lookup_order", "arguments": {"order_id": oid}, "result": order}
    if "error" in order:
        return {"steps": [step], "final_text": f"No order {oid} on file."}
    refund = {
        "tool": "issue_refund",
        "arguments": {"order_id": oid, "amount": 40.0},
        "result": {"ok": True},
    }
    return {"steps": [step, refund], "final_text": f"Done: refunded $40 for order {oid}."}


def fixed(message: str) -> dict[str, Any]:
    """v2, after the prompt fix: checks the tool's own eligibility fields first."""
    ids = ORDER_ID.findall(message.upper())
    if not ids:
        return {"steps": [], "final_text": "Which order is this about?"}
    oid = ids[0]
    order = _lookup(oid)
    step = {"tool": "lookup_order", "arguments": {"order_id": oid}, "result": order}
    if "error" in order:
        return {"steps": [step], "final_text": f"No order {oid} on file."}
    if not _eligible_live(order):
        return {
            "steps": [step],
            "final_text": f"I can't refund order {oid}: outside the return window.",
        }
    refund = {
        "tool": "issue_refund",
        "arguments": {"order_id": oid, "amount": 40.0},
        "result": {"ok": True},
    }
    return {"steps": [step, refund], "final_text": f"Done: refunded $40 for order {oid}."}


LIVE_SEEDS = [
    "Please refund order L1001, it arrived broken.",
    "Refund L1002 please, I changed my mind.",
    "I want my money back on L1003.",
    "Is L1004 delivered yet?",
    "Refund order L1004 right away.",
]

VERSIONS = {"v1": shipped, "v2": fixed}
K, N = 2, 8  # rollouts per ask, asks


def holdout(agent, tasks=None, seed=0):
    """Write the N asks once (tasks=None), then replay that run's asks for the
    other version, so both are graded on the same held-out set."""
    where = {"tasks": tasks} if tasks is not None else {"seeds": LIVE_SEEDS, "situations": N}
    return wai.simulate(
        agent,
        tools=LIVE_TOOLS,
        system_prompt=LIVE_POLICY,
        budget=N * K,
        simulator=False,  # offline writer, no key
        mode="rl",
        repeats=K,
        repeat_policy="fixed",
        reproducible=True,
        seed=seed,
        fault_rate=0.0,
        avg_turns=1,
        **where,
    )


frozen = holdout(VERSIONS["v1"])
data = {"v1": frozen, "v2": holdout(VERSIONS["v2"], tasks=frozen)}
asks = sorted({r["prompt"] for r in frozen.rows()})
for d in data.values():
    assert sorted({r["prompt"] for r in d.rows()}) == asks, "every version must face the same asks"
TEST_VERSION = "t-" + hashlib.sha256("\n".join(asks).encode()).hexdigest()[:8]
print(f"held-out test {TEST_VERSION}: {len(asks)} asks x {K} rollouts")

# the judge that cleared the floors above is the one used to grade the held-out
# test; a judge that did not clear them never reaches this line
JUDGE = Judge(
    name="accurate (tool-call eligibility check)",
    agreement=table["accurate"].agreement,
    human_n=table["accurate"].n,
)
scored = {v: wai.evaluate(d.rows(), accurate, tools=LIVE_TOOLS) for v, d in data.items()}


def score(rows):
    """pass@1 in points, the half-width of its 95% interval, and the asks it rests on."""
    pa = wai.pass_at(rows, k=K)
    lo, hi = pa.ci95
    return round(100 * pa.pass_at_1, 1), round(100 * (hi - lo) / 2, 1), pa.n_groups


def behaviors(rows):
    """The headline, plus every marker the judge above reports, as its own behavior."""
    out = {"refund_policy": score(rows)}
    for name, m in wai.marker_summary(rows).items():
        if m["n_tasks"] < 3:
            continue
        lo, hi = m["ci95"] or (m["mean"], m["mean"])
        out[name] = (round(100 * m["mean"], 1), round(100 * (hi - lo) / 2, 1), m["n_tasks"])
    return out


for v, s in scored.items():
    print(f"{v}: pass@1 {score(s.rows)}")
    for name, (pts, ci, n) in behaviors(s.rows).items():
        print(f"   {name:<26} {pts:>5} +- {ci:<5} n={n}")

v1, v2 = score(scored["v1"].rows), score(scored["v2"].rows)
assert v2[0] > v1[0], (v1, v2)  # the fixed version should score higher under the validated judge

first = score(scored["v1"].rows)
again = score(
    wai.evaluate(holdout(VERSIONS["v1"], tasks=frozen).rows(), accurate, tools=LIVE_TOOLS).rows
)
NOISE = round(abs(first[0] - again[0]), 1)  # points; a scripted agent gives 0
assert NOISE == 0.0, NOISE
print(f"noise floor {NOISE} points (same test, rolled twice)")


class FakePlatform:
    """Records every call and answers like the API, including the verdict
    rule: delta +- sqrt(ci_a^2 + ci_b^2) must exclude zero."""

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

tracked = track(
    "returns-agent",
    model="scripted-agent",
    harness=Harness(label="v1", instructions=LIVE_POLICY, tools=LIVE_TOOLS),
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
            contamination=0,
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
tracked.promote("v1")
print(tracked.verdict())

# ------------------------------------------------------------------ the checks
verdict = str(tracked.verdict())
assert "beats" in verdict, verdict
assert "v2 beats v1" in verdict, verdict
paths = [p for _, p, _ in fake.calls]
assert sum("/behaviors/" in p for p in paths) >= 3, paths
assert paths.count("/runs") == 2, paths
elapsed = time.monotonic() - T0
assert elapsed < 60, f"{elapsed:.1f}s"
print(f"\nok: {elapsed:.1f}s, {len(fake.behaviors)} behaviors, {len(fake.runs)} runs")
sys.exit(0)
