"""Offline check for skills/manage-experiments/SKILL.md: how a coding agent
says what it did, why the number moved, and how to run it again, so the
person opening withwhile.com/platform reads it in one look.

A scripted checkout bot, two prompts (harness), one short SFT run (training)
with a reward curve and held-out checkpoints, posted to a recording fake
platform that answers like the API. Every block in SKILL.md is below,
verbatim, and ``readback`` is checked both ways: clean on this account,
loud on one posted the way real coding agents posted theirs on 2026-09-20
(``dapo-lr5e-05-s17-30st``, ``looks_up_before_answering_seed1``, ``0.75``,
no note, no picture). No key, no network.

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
    Provenance,
    RunRecord,
    track,
)

T0 = time.monotonic()

# ---------------------------------------------------------------- the checkout bot

MODEL = "claude-haiku-4-5"
OPEN_MODEL = "Qwen/Qwen3-4B"
TOOLS = ["get_order", "issue_refund", "escalate"]
POLICY = (
    "Refund a delivered order within 30 days of the order date. "
    "Over $200 goes to a manager. Never refund an undelivered order."
)
PROMPT_POLICY = "You are checkout support for Northwind. " + POLICY
DATES_LINE = "Today is 2026-09-20; compare dates before deciding."
PROMPT_DATES = PROMPT_POLICY + " " + DATES_LINE
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
STEPS = 100
CHECKPOINT_EVERY = 25

OID = re.compile(r"\b(A\d{4})\b")


def expected(ask: str) -> str:
    order = ORDERS[OID.search(ask).group(1)]
    if order["status"] != "delivered":
        return "deny"
    if order["total"] > 200:
        return "escalate"
    return "refund" if order["ordered"] >= "2026-08-21" else "deny"


def bot(ask: str, rng: random.Random, *, old_order_miss: float) -> str:
    """The bot's one weakness is refunding old orders; ``old_order_miss`` is
    how often it does. The dates line takes it from 0.8 to 0.05; training
    walks it down step by step."""
    truth = expected(ask)
    order = ORDERS[OID.search(ask).group(1)]
    if order["ordered"] < "2026-08-21" and rng.random() < old_order_miss:
        return "refund"
    return truth if rng.random() < 0.97 else "refund"


def score_rows(seed: int, *, old_order_miss: float) -> tuple[float, float, list[Example]]:
    """pass@1 in points with a 95% half-width, and every row as an Example."""
    rng = random.Random(seed)
    examples = []
    for ask in ASKS:
        got, want = bot(ask, rng, old_order_miss=old_order_miss), expected(ask)
        examples.append(Example(prompt=ask, reply=got, ok=got == want, why=f"policy says {want}"))
    p = sum(e.ok for e in examples) / len(examples)
    return round(100 * p, 1), round(100 * 1.96 * (p * (1 - p) / len(examples)) ** 0.5, 1), examples


def score_on(prompt: str, seed: int) -> tuple[float, float, list[Example]]:
    """Score one prompt on the frozen test."""
    return score_rows(seed, old_order_miss=0.05 if DATES_LINE in prompt else 0.8)


def score_checkpoint(step: int) -> tuple[float, float, list[Example]]:
    """Score the weights at ``step`` on the frozen test."""
    return score_rows(step, old_order_miss=0.8 * (1 - step / STEPS))


def train_reward(step: int) -> float:
    """The reward the trainer logs at ``step``, 0 to 1, rising with noise."""
    return round(min(1.0, 0.3 + 0.55 * step / STEPS + random.Random(step).uniform(-0.05, 0.05)), 3)


def worst_of(examples: list[Example]) -> list[Example]:
    """The rows the page shows under a score: at most 20, failures first."""
    return sorted(examples, key=lambda e: e.ok)[:20]


# ---------------------------------------------------------------- the fake platform


class FakePlatform:
    """Records every call and answers like the API: runs keep their evals,
    train points, notes, record and archived flag; figures by name; the
    dashboard is built from the evals."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.experiment: dict[str, Any] | None = None
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.figures: dict[str, dict[str, Any]] = {}

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
        if "/figures" in path:
            if method == "GET":
                return {"figures": list(self.figures.values())}
            name = path.rsplit("/", 1)[1]
            self.figures[name] = {"name": name, **body}
            return self.figures[name]
        if path == "/runs" and method == "POST":
            rid = f"run_{len(self.runs):03d}"
            self.runs[rid] = {"id": rid, "status": "running", "evals": [], "train": [], **body}
            return self.runs[rid]
        if path.startswith("/runs?"):
            return {"runs": list(self.runs.values())}
        if path.endswith("/train"):
            self.runs[path.split("/")[2]]["train"].extend(body)
            return {"written": len(body)}
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
    method="Same asks, same judge. Harness: two prompts on one model. Training: SFT on traces.",
    measure="refunds_when_eligible on the frozen test, points out of 100, 95% interval.",
    decide="Promote the version whose interval clears the served one and the noise floor.",
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


def flips(before: list[Example], after: list[Example]) -> tuple[list[str], list[str]]:
    """Which asks changed answer between two scorings of the same test."""
    was = {e.prompt: e.ok for e in before}
    fixed = [e.prompt for e in after if e.ok and not was[e.prompt]]
    broke = [e.prompt for e in after if not e.ok and was[e.prompt]]
    return fixed, broke


ARMS = {"policy": PROMPT_POLICY, "policy+dates": PROMPT_DATES}
scored = {}
for arm, prompt in ARMS.items():
    run = tracked.run(
        arm,
        method="eval",
        harness=Harness(label=f"{arm}@{MODEL}", model=MODEL, instructions=prompt, tools=TOOLS),
        targets=["refunds_when_eligible"],
        record=RunRecord(data=Data(holdout=TEST, n_holdout=len(ASKS)), optimizer=Optimizer(seed=1)),
    )
    score, ci, examples = score_on(prompt, seed=1)
    run.score("refunds_when_eligible", score, ci=ci, n=len(ASKS), examples=worst_of(examples))
    scored[arm] = (run, score, ci, examples)

(base_run, base, base_ci, base_rows), (new_run, new, new_ci, new_rows) = scored.values()
fixed, broke = flips(base_rows, new_rows)
base_run.note("Served prompt, seed 1. Misses are refunds of orders older than 30 days.")
new_run.note(
    f"Changed: one line, '{DATES_LINE}'\n"
    f"Moved: {base} to {new} points (±{new_ci}) on {len(ASKS)} asks.\n"
    f"Why: {len(fixed)} asks now pass, every one an order older than 30 days; {len(broke)} broke.\n"
    f"Reproduce: seed 1, test {TEST}, uv run python evals/run.py --prompt policy+dates"
)
tracked.figure(
    "harness",
    {
        "data": [
            {
                "type": "bar",
                "x": list(scored),
                "y": [s[1] for s in scored.values()],
                "error_y": {"type": "data", "array": [s[2] for s in scored.values()]},
            }
        ],
        "layout": {
            "title": "refunds_when_eligible, points out of 100",
            "yaxis": {"range": [0, 100]},
        },
    },
    caption=f"The dates line fixes {len(fixed)} asks and breaks {len(broke)}.",
    run=new_run,
)
for r in (base_run, new_run):
    r.finish(say=False)

run = tracked.run(
    "dates-sft",
    method="SFT",
    base=OPEN_MODEL,
    targets=["refunds_when_eligible"],
    trained_on=["refund traces 2026-09"],
    record=RunRecord(
        data=Data(train="refund traces 2026-09", n_train=800, holdout=TEST, n_holdout=len(ASKS)),
        optimizer=Optimizer(lr=1e-5, seed=17),
        provenance=Provenance(pins={"trl": "1.13.0", "transformers": "5.17.0"}),
    ),
)
steps, rewards, held = [], [], []
for step in range(1, STEPS + 1):
    reward = train_reward(step)  # what the trainer logged at this step
    run.log(step, reward=reward)
    steps.append(step)
    rewards.append(reward)
    if step % CHECKPOINT_EVERY == 0:
        score, ci, rows = score_checkpoint(step)  # the frozen test on this checkpoint
        held.append((step, score, ci, rows))
        run.score("refunds_when_eligible", score, ci=ci, n=len(ASKS), examples=worst_of(rows))
first, last = held[0], held[-1]
fixed, broke = flips(base_rows, last[3])
together = last[1] - first[1] > NOISE  # held-out moved with the reward, or only the reward did
run.note(
    f"Changed: SFT on 800 refund traces, lr 1e-5, seed 17, {STEPS} steps.\n"
    f"Moved: reward {rewards[0]} to {rewards[-1]}; held-out {base} to {last[1]} points (±{last[2]}).\n"
    + (
        f"Why: {len(fixed)} asks now pass, every one an old order; {len(broke)} broke. "
        "Reward and held-out rose together, so the judge is not being gamed.\n"
        if together
        else "Why: reward rose but held-out did not. Treat as reward hacking until shown otherwise.\n"
    )
    + "Reproduce: uv run python train.py --seed 17 (pins in the record)"
)
tracked.figure(
    "training",
    {
        "data": [
            {"type": "scatter", "name": "reward", "x": steps, "y": rewards},
            {
                "type": "scatter",
                "name": "held-out, points",
                "x": [h[0] for h in held],
                "y": [h[1] for h in held],
                "yaxis": "y2",
                "mode": "lines+markers",
            },
        ],
        "layout": {
            "title": "dates-sft: reward per step and the frozen test at checkpoints",
            "yaxis": {"title": "reward"},
            "yaxis2": {"title": "held-out", "overlaying": "y", "side": "right", "range": [0, 100]},
        },
    },
    caption="Held-out rises with the reward; a flat held-out line under a rising reward is hacking.",
    run=run,
)
run.finish(hours=0.4, gpu="A10G", cost_usd=0.5, say=False)

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
    pictured = {f.run for f in tracked.figures()}
    for r in tracked.runs():
        v, note = r["version"], r.get("notes") or ""
        trained = r.get("method") not in (None, "none", "eval")
        if SETTING.search(v):
            out.append(f"version {v!r} encodes settings; say the arm in words, numbers in record")
        if not ((r.get("record") or {}).get("data") or {}):
            out.append(f"run {v!r}: no data posted; RunRecord(data=Data(...))")
        for word in ("Changed", "Moved", "Why", "Reproduce"):
            if trained and f"{word}:" not in note:
                out.append(f"run {v!r}: note has no '{word}:' line; run.note(...)")
        if trained and r["id"] not in pictured:
            out.append(f"run {v!r}: no picture; tracked.figure(name, fig, run=run)")
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

dup = tracked.run("dates-sft", method="SFT", base=OPEN_MODEL)  # launched twice by mistake
tracked.archive(dup.id)  # a wrong launch or a dead arm: archive, never delete

# ---------------------------------------------------------------- assertions

calls = fake.calls
first_run = next(i for i, c in enumerate(calls) if c[0] == "POST" and c[1] == "/runs")
first_exp = next(i for i, c in enumerate(calls) if c[0] == "PUT" and c[1].endswith("/experiment"))
assert first_exp < first_run, "the question is posted before the first run"

posted = [c[2] for c in calls if c[0] == "POST" and c[1] == "/runs"]
assert [p["version"] for p in posted] == ["policy", "policy+dates", "dates-sft", "dates-sft"]
assert all(p["record"]["data"]["holdout"] == TEST for p in posted[:3]), "every run names its test"
assert posted[2]["record"]["provenance"]["pins"]["trl"] == "1.13.0", "pins travel with the run"
assert posted[2]["record"]["optimizer"]["seed"] == 17, "seed lives in the record"

evals = [e for c in calls if c[1].endswith("/evals") for e in c[2]]
assert len(evals) == 2 + STEPS // CHECKPOINT_EVERY
assert all(e["score"] > 1 and e["ci"] is not None and e["n"] == len(ASKS) for e in evals)
assert all(1 <= len(e["examples"]) <= 20 and not e["examples"][0]["ok"] for e in evals[:2]), (
    "failures first"
)
assert fake.behaviors["refunds_when_eligible"]["testVersion"] == TEST
assert re.fullmatch(r"t-[0-9a-f]{8}", TEST), "the test is named by its content"

points = [p for c in calls if c[1].endswith("/train") for p in c[2]]
assert len(points) == STEPS and points[-1]["step"] == STEPS, "every step logged"
assert base < last[1] and rewards[-1] > rewards[0] and together, "the demo run really improved"
assert fixed and not broke, (fixed, broke)

notes = {c[1].split("/")[2]: c[2]["notes"] for c in calls if c[0] == "PATCH" and "notes" in c[2]}
assert len(notes) == 3, "every run says what happened"
assert all(w in notes[run.id] for w in ("Changed:", "Moved:", "Why:", "Reproduce:"))
assert "so the judge is not being gamed" in notes[run.id]
assert set(fake.figures) == {"harness", "training"}
assert fake.figures["training"]["run"] == run.id
assert len(fake.figures["training"]["figure"]["data"]) == 2

archived = [c for c in calls if c[0] == "PATCH" and (c[2] or {}).get("archived") is True]
assert len(archived) == 1 and len(tracked.runs()) == 3 and len(tracked.runs(archived=True)) == 4

# the block that reads an account back must be loud on what real agents posted
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
    "no 'Why:' line",
    "no picture",
    "reads as a fraction",
):
    assert any(word in p for p in loud), (word, loud)
print(f"readback on an account posted the old way: {len(loud)} problems, as expected")

line = str(tracked.verdict())
assert "dates-sft" in line, line
assert time.monotonic() - T0 < 60
print(f"ok in {time.monotonic() - T0:.1f}s")
