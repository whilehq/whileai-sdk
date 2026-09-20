"""Offline check for skills/manage-experiments/SKILL.md: the bookkeeping
that makes an agent's runs readable to the people who open the page.

A scripted checkout bot, two prompts, three seeds each, scored by a judge
that is a program, posted to a recording fake platform that answers like
the API. Every block in SKILL.md is below, verbatim, and ``readback``
(the block an agent runs against a real account) is checked both ways:
clean on this account, loud on an account named the way real coding
agents named theirs on 2026-09-20 (``dapo-lr5e-05-s17-30st``,
``looks_up_before_answering_seed1``, a score of 0.75). No key, no network.

    uv run python skills/manage-experiments/check.py
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from typing import Any

from whileai.platform import (
    Behavior,
    Data,
    Example,
    Harness,
    Judge,
    Optimizer,
    PlatformError,
    RunRecord,
    track,
)

T0 = time.monotonic()

# ---------------------------------------------------------------- the checkout bot

MODEL = "claude-haiku-4-5"
TOOLS = ["get_order", "issue_refund", "escalate"]
POLICY = (
    "Refund a delivered order within 30 days of the order date. "
    "Over $200 goes to a manager. Never refund an undelivered order."
)
PROMPT_POLICY = "You are checkout support for Northwind. " + POLICY
PROMPT_DATES = PROMPT_POLICY + " Today is 2026-09-20; compare dates before deciding."
ORDERS = {
    "A1001": {"total": 129.0, "ordered": "2026-09-05", "status": "delivered"},
    "A1002": {"total": 449.0, "ordered": "2026-09-01", "status": "delivered"},
    "A1003": {"total": 24.0, "ordered": "2026-09-14", "status": "shipped"},
    "A1004": {"total": 189.0, "ordered": "2026-06-20", "status": "delivered"},
}
TEMPLATES = [
    "Refund order {oid}, please.",
    "I want my money back on {oid}.",
    "Can you refund {oid}? It was not what I ordered.",
    "Refund {oid} right now or I dispute the charge.",
    "Is {oid} still refundable?",
    "Please process a refund for {oid}.",
    "Return and refund {oid}.",
    "{oid} arrived damaged, refund it.",
    "Cancel {oid} and refund me.",
    "How do I get a refund on {oid}?",
    "Refund {oid} to my card.",
    "I changed my mind about {oid}, refund please.",
    "Give me a refund for {oid}, it is faulty.",
]
ASKS = [t.format(oid=oid) for t in TEMPLATES for oid in ORDERS]  # 52 asks, one per branch
NOISE = 2.0  # points, from scoring the served prompt twice on this test

OID = re.compile(r"\b(A\d{4})\b")


def expected(ask: str) -> str:
    order = ORDERS[OID.search(ask).group(1)]
    if order["status"] != "delivered":
        return "deny"
    if order["total"] > 200:
        return "escalate"
    return "refund" if order["ordered"] >= "2026-08-21" else "deny"


def bot(prompt: str, ask: str, rng: random.Random) -> str:
    """The prompt without the dates line guesses on old orders; with it, it
    reads the date. A little seed noise either way."""
    truth = expected(ask)
    order = ORDERS[OID.search(ask).group(1)]
    if "compare dates" not in prompt and order["ordered"] < "2026-08-21":
        return "refund" if rng.random() < 0.8 else truth
    return truth if rng.random() < 0.95 else "refund"


def score_on(prompt: str, seed: int) -> tuple[float, float, list[Example]]:
    """pass@1 in points with a 95% half-width, and every row as an Example."""
    rng = random.Random(seed)
    rows = [(ask, bot(prompt, ask, rng), expected(ask)) for ask in ASKS]
    examples = [
        Example(prompt=ask, reply=got, ok=got == want, why=f"policy says {want}")
        for ask, got, want in rows
    ]
    p = sum(e.ok for e in examples) / len(examples)
    return round(100 * p, 1), round(100 * 1.96 * (p * (1 - p) / len(examples)) ** 0.5, 1), examples


# ---------------------------------------------------------------- the fake platform


class FakePlatform:
    """Records every call and answers like the API: runs keep their evals,
    notes, record and archived flag; the dashboard is built from them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.experiment: dict[str, Any] | None = None
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}

    def __call__(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path, body))
        if path == "/agents":
            return {**body, "name": body.get("name") or body["id"]}
        if path.endswith("/experiment"):
            if method == "GET":
                if self.experiment is None:
                    raise PlatformError(404, "no experiment")
                return self.experiment
            self.experiment = dict(body)
            return self.experiment
        if "/behaviors" in path:
            if method == "GET":
                return {"behaviors": list(self.behaviors.values())}
            name = path.rsplit("/", 1)[1]
            self.behaviors[name] = {"name": name, **body}
            return self.behaviors[name]
        if path == "/runs" and method == "POST":
            rid = f"run_{len(self.runs):03d}"
            self.runs[rid] = {"id": rid, "status": "running", "evals": [], **body}
            return self.runs[rid]
        if path.startswith("/runs?"):
            return {"runs": list(self.runs.values())}
        if path.endswith("/evals"):
            run = self.runs[path.split("/")[2]]
            run["evals"].extend({**e, "version": run["version"]} for e in body)
            return {"evals": body}
        if path.startswith("/runs/") and method == "PATCH":
            self.runs[path.split("/")[2]].update(body)
            return self.runs[path.split("/")[2]]
        if "/dashboard" in path:
            agent = path.split("/")[2]
            evals = [e for r in self.runs.values() if not r.get("archived") for e in r["evals"]]
            return {
                "agent": {"id": agent, "name": agent, "serving": "policy"},
                "behavior": next(iter(self.behaviors.values()), None),
                "behaviors": list(self.behaviors),
                "versions": [
                    {"v": e["version"], "score": e["score"], "ci": e.get("ci"), "n": e.get("n")}
                    for e in evals
                ],
            }
        return {"ok": True}


fake = FakePlatform()

# ---------------------------------------------------------------- SKILL.md blocks

tracked = track("checkout-support", model=MODEL, transport=fake)  # drop transport= for real
tracked.experiment(
    question="Does telling the prompt today's date cut wrong refunds on old orders?",
    hypothesis="Most wrong refunds are orders past 30 days; a dated rule removes them.",
    method="Same asks, same judge, two prompts on one model, three seeds each.",
    measure="refunds_when_eligible on the frozen test, points out of 100, 95% interval.",
    decide="Promote policy+dates when its interval clears policy and the noise floor.",
)

TEST = "t-" + hashlib.sha256("\n".join(ASKS).encode()).hexdigest()[:8]
tracked.behavior(
    Behavior(
        name="refunds_when_eligible",
        test_version=TEST,
        n=len(ASKS),
        judge=Judge(name="refund policy as a program", agreement=0.93, human_n=60),
        noise_floor=NOISE,
        contamination=0,
        reward_is_judge=False,
        rubric=POLICY,
        description="Refunds delivered orders within 30 days; escalates over $200; else denies.",
    )
)

ARMS = {"policy": PROMPT_POLICY, "policy+dates": PROMPT_DATES}
for arm, prompt in ARMS.items():
    harness = Harness(label=f"{arm}@{MODEL}", model=MODEL, instructions=prompt, tools=TOOLS)
    for seed in (1, 2, 3):
        run = tracked.run(
            arm,
            method="eval",
            harness=harness,
            targets=["refunds_when_eligible"],
            record=RunRecord(
                data=Data(holdout=TEST, n_holdout=len(ASKS)),
                optimizer=Optimizer(seed=seed),
            ),
        )
        score, ci, examples = score_on(prompt, seed)
        failed = [e for e in examples if not e.ok]
        worst = (failed + [e for e in examples if e.ok])[:20]  # 20 rows, failures first
        run.score("refunds_when_eligible", score, ci=ci, n=len(ASKS), examples=worst)
        run.note(
            f"seed {seed}: {len(failed)} of {len(ASKS)} wrong; every miss is an old order refunded."
        )
        run.finish(say=False)

SETTING = re.compile(r"(lr\d|\de-0\d|-s\d+\b|_s\d+$|_seed\d+|^h-[0-9a-f]{12}$|^v\d+$)")


def readback(tracked) -> list[str]:
    """What a teammate opening the page could not read. Empty means clean."""
    out = []
    if tracked.experiment() is None:
        out.append("no question posted: tracked.experiment(question=...)")
    for b in tracked.behaviors():
        if SETTING.search(b.name):
            out.append(f"behavior {b.name!r} names a seed or setting; put it in Optimizer(seed=)")
        if not (b.test_version or "").startswith("t-"):
            out.append(f"behavior {b.name!r}: test_version is not the asks' hash")
    for r in tracked.runs():
        v = r["version"]
        if SETTING.search(v):
            out.append(f"version {v!r} encodes settings; say the arm in words, numbers in record")
        if not ((r.get("record") or {}).get("data") or {}):
            out.append(f"run {v!r}: no data posted; RunRecord(data=Data(...))")
        if not r.get("notes"):
            out.append(f"run {v!r}: no note; run.note(what happened)")
        for e in r.get("evals") or []:
            if e["score"] <= 1:
                out.append(f"{v} {e['behavior']}: {e['score']} reads as a fraction; post points")
    return out


for problem in readback(tracked):
    print("fix:", problem)

assert not readback(tracked), readback(tracked)
print("readable: 0 problems")

print(tracked.brief())
print(tracked.verdict())

dead = [r for r in tracked.runs() if r["version"] == "policy"][-1]
tracked.archive(dead["id"])  # a wrong launch or a dead arm: archive, never delete

# ---------------------------------------------------------------- assertions

calls = fake.calls
first_run = next(i for i, c in enumerate(calls) if c[0] == "POST" and c[1] == "/runs")
first_exp = next(i for i, c in enumerate(calls) if c[0] == "PUT" and c[1].endswith("/experiment"))
assert first_exp < first_run, "the question is posted before the first run"
assert fake.experiment and fake.experiment["decide"].startswith("Promote")

posted = [c[2] for c in calls if c[0] == "POST" and c[1] == "/runs"]
assert len(posted) == 6 and {p["version"] for p in posted} == {"policy", "policy+dates"}
assert all(p["harness"].endswith(f"@{MODEL}") for p in posted), "harness label is prompt@model"
seeds = sorted(p["record"]["optimizer"]["seed"] for p in posted)
assert seeds == [1, 1, 2, 2, 3, 3], "seed lives in the record, not the version"
assert all(p["record"]["provenance"]["pins"].get("harness") for p in posted), "harness pinned"

evals = [e for c in calls if c[1].endswith("/evals") for e in c[2]]
assert len(evals) == 6 and all(e["score"] > 1 and e["ci"] is not None and e["n"] for e in evals)
assert all(1 <= len(e["examples"]) <= 20 for e in evals), "every score carries its rows"
assert fake.behaviors["refunds_when_eligible"]["testVersion"] == TEST
assert re.fullmatch(r"t-[0-9a-f]{8}", TEST), "the test is named by its content"

notes = [c for c in calls if c[0] == "PATCH" and "notes" in (c[2] or {})]
assert len(notes) == 6, "every run says what happened"
archived = [c for c in calls if c[0] == "PATCH" and (c[2] or {}).get("archived") is True]
assert len(archived) == 1 and len(tracked.runs()) == 5 and len(tracked.runs(archived=True)) == 6

# the block that reads an account back must be loud on names real agents posted
bad = FakePlatform()
t_bad = track("zero-rl-sweep-qwen3.5-4b", model="Qwen/Qwen3.5-4B-Base", transport=bad)
t_bad.behavior(Behavior(name="looks_up_before_answering_seed1", test_version="v1"))
r_bad = t_bad.run("dapo-lr5e-05-s17-30st", method="grpo", harness="h-5c820b09b7ed")
r_bad.score("looks_up_before_answering_seed1", 0.75, ci=0.09, n=40)
r_bad.finish(say=False)
loud = readback(t_bad)
for word in (
    "no question",
    "names a seed",
    "not the asks' hash",
    "encodes settings",
    "no data",
    "no note",
    "reads as a fraction",
):
    assert any(word in p for p in loud), (word, loud)
print(f"readback on a badly named account: {len(loud)} problems, as expected")

# the verdict names the arm that won, from the same rows the page reads
line = str(tracked.verdict())
assert "policy+dates" in line, line
assert time.monotonic() - T0 < 60
print(f"ok in {time.monotonic() - T0:.1f}s")
