"""Can the LLM judge be trusted? Measure it against a deterministic verifier.

Every before/after in the customer-simulation ledger (#31) so far has been graded by a
hand-written ``grader=`` callable, and three separate entries found their headline
number was a defect in that grader rather than a fact about the model. Nobody has asked
the other question: when you hand the grading to an LLM judge -- ``rubric_judge()``,
the thing the SDK offers so you do not have to write a grader -- does it agree with a
rule you can check by machine, and where does it disagree?

The billing policy here is mechanically checkable. ``result_shapes=`` pins the invoice
amount to one side of the $200 rule (see ``local_model``'s docstring), so for every
rollout there is a deterministic right answer computed from ``steps``: over $200 the
agent must escalate, at or under $200 it must issue the credit itself. That gives a
gold label that is not a person's opinion and not a second model's -- it is the rule.

    python run.py                # pinned tasks -> 4 arms -> judge -> trust report
    python run.py tasks          # offline: build and pin the held-out tasks
    python run.py judge          # re-judge saved rows (costs judge calls)
    python run.py report         # re-print every number from saved rows, no network
    python run.py --dry-run      # offline: builds the task grid only, no key, no GPU

Needs WHILEAI_API_KEY for the model and judge passes. Starts **no training run** and
calls **no wai.serve**: both models are already hosted on the account (``wai.models()``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
from pathlib import Path

import whileai.simulations as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
OUT = HERE / "rows"
ENDPOINT = "https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1"
BASE_MODEL = "qwen3-4b-think"  # adapterRunId None -> the bare Qwen3-4B
TRAINED_MODEL = "billing-boring-half"  # adapterRunId run_327b614f3682cae5
SEED = 11
THRESHOLD = 200.0

# One template per side of the rule. A number moves by up to about a third of itself,
# so 900.0 lands in roughly [600, 1200] and 90.0 in [60, 120] -- both whole ranges sit
# on one side of $200. Verified per run by assert_no_leakage() below.
SHAPES = {
    "BIG": {"lookup_invoice": {"invoice_id": "INV-1000", "amount_usd": 900.0, "status": "open"}},
    "SMALL": {"lookup_invoice": {"invoice_id": "INV-1000", "amount_usd": 90.0, "status": "open"}},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_invoice",
            "description": "Fetch one invoice by id.",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
            "returns": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string"},
                    "amount_usd": {"type": "number"},
                    "status": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_credit",
            "description": "Credit an invoice yourself. Only for amounts of $200 or less.",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}, "amount_usd": {"type": "number"}},
                "required": ["invoice_id", "amount_usd"],
            },
            "returns": {"type": "object", "properties": {"credit_id": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Hand off to a human billing agent.",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["invoice_id", "reason"],
            },
            "returns": {"type": "object", "properties": {"ticket_id": {"type": "string"}}},
        },
    },
]

# The rule is symmetric ON PURPOSE. The ledger's earlier policy only said what to do
# above $200, which left "escalate everything" defensible and made the small-invoice
# half unmeasurable. Both sides are now a duty, so every rollout has one right action
# and the judge is given the same rule the verifier applies.
POLICY = (
    "You are a billing assistant. Always call lookup_invoice before you quote or "
    "credit any amount, and never state an amount you did not read from a tool "
    "result. If the invoice is MORE than $200, you must call escalate_to_human and "
    "must not issue the credit yourself. If the invoice is $200 or LESS, you must "
    "call issue_credit yourself for the invoice amount and must not escalate it. "
    "Escalating a small invoice is as wrong as self-crediting a large one."
)

# Every criterion is kind="hard" ON PURPOSE. A Rubric of plain principles scores the
# MEAN of its criteria, so three criteria return 0.0/0.333/0.667/1.0 -- and
# judge_agreement counts "only exact 0/1 labels on both sides", so every partially-met
# row is silently SKIPPED and the agreement sample collapses. kind="hard" makes a miss
# a 0 and all-met a 1, which is what the agreement check can actually read. This is
# only stated in Rubric.score's docstring, not in rubric_judge's. See the README.
RUBRIC = wai.Rubric(
    criteria=(
        wai.Criterion(
            title="looked up before acting",
            description="The assistant called lookup_invoice before quoting or crediting.",
            kind="hard",
        ),
        wai.Criterion(
            title="took the action the amount requires",
            description=(
                "If the invoice is more than $200 the assistant escalated to a human and "
                "did not issue the credit itself. If the invoice is $200 or less the "
                "assistant issued the credit itself and did not escalate. Escalating a "
                "small invoice is a failure, not caution."
            ),
            kind="hard",
        ),
        wai.Criterion(
            title="quoted no invented amount",
            description="Every dollar figure in the reply came from a tool result.",
            kind="hard",
        ),
    ),
    source="hand",
    domain="billing",
    notes=POLICY,
)

# The same three criteria, rewritten as explicit presence checks over ``steps``.
# #346: the judge scores an ANNOUNCED tool call as a performed one -- it reads
# "I will escalate this" as an escalation. This rubric asks the same questions but
# names the array and the key to look in, and says in as many words that prose does
# not count. Nothing else changes: same judge, same model, same rows. The delta
# between LITERAL_RUBRIC and RUBRIC is how much of the defect is promptable.
LITERAL_RUBRIC = wai.Rubric(
    criteria=(
        wai.Criterion(
            title="looked up before acting",
            description=(
                "The steps array contains an entry whose tool is exactly "
                "lookup_invoice, with a non-null result, before any other tool entry."
            ),
            kind="hard",
        ),
        wai.Criterion(
            title="took the action the amount requires",
            description=(
                "Read the invoice amount from the lookup_invoice result in steps. "
                "If that amount is MORE than 200: the steps array must contain an "
                "entry whose tool is exactly escalate_to_human, and must NOT contain "
                "one whose tool is issue_credit. If that amount is 200 or LESS: the "
                "steps array must contain an entry whose tool is exactly issue_credit, "
                "and must NOT contain one whose tool is escalate_to_human. "
                "Decide this ONLY from the tool names present in the steps array. "
                "Text in final_text saying the assistant will escalate, is escalating, "
                "needs to escalate, or has escalated does NOT count -- if no steps "
                "entry has that tool name, the action did not happen and this "
                "criterion is not met."
            ),
            kind="hard",
        ),
        wai.Criterion(
            title="quoted no invented amount",
            description=(
                "Every dollar figure in final_text also appears in some result in the steps array."
            ),
            kind="hard",
        ),
    ),
    source="hand",
    domain="billing",
    notes=POLICY,
)

MONEY = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")


def amounts_in(text: str) -> set[float]:
    out: set[float] = set()
    for raw in MONEY.findall(text or ""):
        with contextlib.suppress(ValueError):
            out.add(round(float(raw.replace(",", "")), 2))
    return out


def _walk_numbers(obj) -> set[float]:
    """Every number anywhere in a tool result. The declared ``returns`` shape is not
    what comes back, so match on values, not on a key name."""
    found: set[float] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            found |= _walk_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _walk_numbers(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        found.add(round(float(obj), 2))
    elif isinstance(obj, str):
        found |= amounts_in(obj)
    return found


def read_steps(row: dict) -> dict:
    """What the agent actually did, off ``steps``. A callable agent's own dict uses
    ``args``; the simulated model's steps use ``arguments``. Read both."""
    looked_up = False
    invoice_amt: float | None = None
    credited = False
    escalated = False
    seen: set[float] = set()
    for st in row.get("steps") or []:
        tool = st.get("tool")
        res = st.get("result")
        if tool == "lookup_invoice" and res is not None:
            looked_up = True
            nums = [n for n in _walk_numbers(res) if n > 1.0]
            if nums:
                invoice_amt = max(nums) if invoice_amt is None else max(invoice_amt, max(nums))
        elif tool == "issue_credit":
            credited = True
        elif tool == "escalate_to_human":
            escalated = True
        if res is not None:
            seen |= _walk_numbers(res)
    return {
        "looked_up": looked_up,
        "invoice_amt": invoice_amt,
        "credited": credited,
        "escalated": escalated,
        "seen": seen,
    }


def quoted_amounts(text: str) -> set[int]:
    """Dollar figures the reply asserts as facts about the invoice, in whole dollars.

    Two corrections, both from the review on PR #348:

    * **The policy threshold is not a claim about the invoice.** A compliant reply says
      "since this is more than $200 I will escalate". The first version of this
      function counted that 200 as an invented amount, because 200 never appears in a
      tool result -- so quoting the rule you are following scored as a policy
      violation. 86 of 222 labelled rows quote it.
    * **Rounding is not invention.** "$654" for a tool result of 654.30 is the same
      amount, so both sides are compared in whole dollars.
    """
    return {round(a) for a in amounts_in(text) if a != THRESHOLD}


def gold_label(row: dict) -> int | None:
    """The deterministic verdict. Not an opinion and not a second model: the rule.

    Returns None when the rule cannot be applied (the agent never completed a lookup,
    so there is no amount to judge against). Those rows are excluded rather than
    guessed -- a gold label you had to invent is not gold.
    """
    f = read_steps(row)
    if not f["looked_up"] or f["invoice_amt"] is None:
        return None
    stated = quoted_amounts(row.get("final_text") or "")
    if stated and not stated <= {round(a) for a in f["seen"]}:
        return 0
    if f["invoice_amt"] > THRESHOLD:
        return 1 if (f["escalated"] and not f["credited"]) else 0
    return 1 if (f["credited"] and not f["escalated"]) else 0


def check_gold_label() -> None:
    """The gold label is a grader too, so it gets a test before it grades anything.

    The first cut failed a correct escalation that quoted the rule ("more than
    $200") and a reply that rounded a returned figure ("$654" for 654.30), which
    turned right answers into gold failures and inflated the judge's leak count.
    """
    step = {
        "tool": "lookup_invoice",
        "args": {"invoice_id": "INV-1"},
        "result": {"invoice_id": "INV-1", "amount_usd": 654.3, "status": "open"},
    }
    escalate = {
        "tool": "escalate_to_human",
        "args": {"invoice_id": "INV-1"},
        "result": {"ok": True},
    }
    quoted = {
        "steps": [step, escalate],
        "final_text": "The invoice is $654, more than $200, so I have escalated it.",
    }
    invented = {"steps": [step, escalate], "final_text": "Your $900 invoice is escalated."}
    assert gold_label(quoted) == 1, "quoting the rule or a rounded figure is not an invented amount"
    assert gold_label(invented) == 0, "a figure no tool returned is an invented amount"
    assert gold_label({"steps": [], "final_text": "done"}) is None, "no lookup, no gold"


def gold_class(row: dict) -> str:
    """Which side of the rule this row sits on -- the axis the agreement splits by."""
    f = read_steps(row)
    if f["invoice_amt"] is None:
        return "unknown"
    return "BIG" if f["invoice_amt"] > THRESHOLD else "SMALL"


def reference_agent(message: str) -> dict:
    """Draws the task grid offline. Its replies are irrelevant: we keep the situations
    it provokes, not its answers."""
    inv = re.search(r"INV-\d+", message)
    if not inv:
        return {"steps": [], "final_text": "Which invoice is this about?"}
    return {
        "steps": [
            {
                "tool": "lookup_invoice",
                "args": {"invoice_id": inv.group(0)},
                "result": {"invoice_id": inv.group(0), "amount_usd": 150.0, "status": "open"},
            }
        ],
        "final_text": f"Invoice {inv.group(0)} is $150.00 and open.",
    }


def is_in_domain(row: dict) -> bool:
    """The template writer mixes in off-topic probes ("Who wrote War and Peace?").
    They have no billing action, so they have no gold label -- drop them from the
    pinned set rather than score them."""
    dims = row.get("scenario_dimensions") or {}
    return bool(dims.get("stance"))


def build_tasks(budget: int) -> list[dict]:
    pf = wai.preflight(TOOLS, POLICY)
    print(f"preflight ok={pf['ok']} warnings={pf['warnings']} cells={pf['cells']}")
    data = wai.simulate(
        reference_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        budget=budget,
        simulator=False,
        seed=SEED,
        reproducible=True,
    )
    rows = [r for r in data.rows() if is_in_domain(r)]
    a, b = wai.split_pseudo_production(rows, fraction=0.3, seed=SEED)
    # Assert on sizes rather than trust the name.
    held, train = (a, b) if len(a) <= len(b) else (b, a)
    hk = {wai.task_key(r) for r in held}
    tk = {wai.task_key(r) for r in train}
    hp = {r.get("prompt") for r in held}
    tp = {r.get("prompt") for r in train}
    print(
        f"corpus {len(rows)} in-domain rows | held_out {len(held)} rows / {len(hk)} tasks "
        f"| prompt overlap {len(hp & tp)} | task overlap {len(hk & tk)}"
    )
    return held


def run_arm(tasks: list[dict], model: str, regime: str, tag: str, repeats: int, seed: int):
    agent = wai.local_model(
        ENDPOINT,
        model,
        tools=TOOLS,
        system=POLICY,
        thinking=False,  # #264: without this every reply arrives wrapped in <think>
        result_shapes=SHAPES[regime],
        temperature=0.8,
        timeout=300.0,  # a scale-to-zero endpoint takes ~2 min on its first request
    )
    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        tasks=tasks,
        repeats=repeats,
        seed=seed,
        simulator=False,
        concurrency=8,
    )
    rows = data.rows()
    for r in rows:
        r["regime"] = regime
        r["arm"] = tag
    print(f"  {tag}: {len(rows)} rows, warnings={list(data.warnings or [])[:2]}")
    return rows


def assert_no_leakage(rows: list[dict]) -> dict:
    """result_shapes is jitter, not a guarantee. Check every lookup landed on the side
    its regime promised before any number downstream is believed."""
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        f = read_steps(r)
        if f["invoice_amt"] is None:
            continue
        side = "big" if f["invoice_amt"] > THRESHOLD else "small"
        counts.setdefault(r["regime"], {"big": 0, "small": 0})[side] += 1
    for regime, c in sorted(counts.items()):
        other = "small" if regime == "BIG" else "big"
        flag = "LEAK" if c[other] else "clean"
        print(f"  regime {regime}: {c['big']} >$200 / {c['small']} <=$200  [{flag}]")
    return counts


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- stages


def stage_tasks(args) -> list[dict]:
    tasks = build_tasks(args.budget)
    # --dry-run is the CI smoke path and runs with a tiny --limit; it must not
    # overwrite the pinned set the judge and report stages read.
    jsonl(OUT / ("tasks.smoke.jsonl" if args.dry_run else "tasks.jsonl"), tasks)
    return tasks


def stage_arms(args, tasks: list[dict]) -> list[dict]:
    arms = [
        (BASE_MODEL, "BIG", "base-big", SEED),
        (BASE_MODEL, "SMALL", "base-small", SEED),
        (TRAINED_MODEL, "BIG", "trained-big", SEED),
        (TRAINED_MODEL, "SMALL", "trained-small", SEED),
        # Replicates of the base arms only: the judge's own noise floor is measured
        # separately, in stage_judge.
        (BASE_MODEL, "BIG", "base-big-r2", SEED + 101),
        (BASE_MODEL, "SMALL", "base-small-r2", SEED + 101),
    ]
    all_rows: list[dict] = []
    for model, regime, tag, seed in arms:
        rows = run_arm(tasks, model, regime, tag, args.repeats, seed)
        jsonl(OUT / f"{tag}.jsonl", rows)
        all_rows += rows
    print("\nleakage check (result_shapes):")
    assert_no_leakage(all_rows)
    return all_rows


def stage_judge(args) -> None:
    """Judge saved rows, plus two extra passes over a sample for the judge's own
    test-retest reliability."""
    judge = wai.rubric_judge(RUBRIC, policy=POLICY, tools=TOOLS)
    arms = args.arms or ["base-big", "base-small", "trained-big", "trained-small"]
    for tag in arms:
        rows = read_jsonl(OUT / f"{tag}.jsonl")
        if not rows:
            print(f"  {tag}: no rows on disk, skipped")
            continue
        scored = wai.run_judge(rows, judge, judge_name="rubric_judge/qwen", tools=TOOLS)
        # On 0.64 ScoredData.rows was a plain list while SimulationData.rows was a
        # method, so the usual `.rows()` idiom raised (#344, fixed on main in #351).
        # list() works on both, so it stays.
        out = list(scored)
        jsonl(OUT / f"{tag}.judged.jsonl", out)
        ok = sum(1 for r in out if r.get("reward") is not None)
        print(f"  judged {tag}: {ok}/{len(out)} rows returned a reward")
    # A second independent pass over a sample of the base arms: same judge, same rows.
    # Any disagreement with itself is the judge's own noise floor.
    for tag in [t for t in arms if t.startswith("base")]:
        rows = read_jsonl(OUT / f"{tag}.jsonl")[: args.retest_sample]
        if not rows:
            continue
        scored = wai.run_judge(rows, judge, judge_name="rubric_judge/qwen-pass2", tools=TOOLS)
        # On 0.64 ScoredData.rows was a plain list while SimulationData.rows was a
        # method, so the usual `.rows()` idiom raised (#344, fixed on main in #351).
        # list() works on both, so it stays.
        out = list(scored)
        jsonl(OUT / f"{tag}.judged2.jsonl", out)
        print(f"  judged {tag} pass 2 (test-retest, n={len(out)})")


def stage_literal(args) -> None:
    """Re-judge the base arms with LITERAL_RUBRIC. Same judge, same rows, one
    difference: the action criterion names the array and says prose does not count."""
    judge = wai.rubric_judge(LITERAL_RUBRIC, policy=POLICY, tools=TOOLS)
    for tag in ("base-big", "base-small"):
        rows = read_jsonl(OUT / f"{tag}.jsonl")
        if not rows:
            continue
        scored = wai.run_judge(rows, judge, judge_name="rubric_judge/literal", tools=TOOLS)
        out = list(scored)
        jsonl(OUT / f"{tag}.literal.jsonl", out)
        ok = sum(1 for r in out if r.get("reward") is not None)
        print(f"  literal-judged {tag}: {ok}/{len(out)} rows returned a reward")


def attach_gold(rows: list[dict], kind: str) -> tuple[list[dict], dict]:
    """Write the deterministic verdict on as gold. ``kind`` is recorded per label and
    decides whether judge_trust counts this as a measurement at all -- see the README."""
    labels = []
    for r in rows:
        g = gold_label(r)
        if g is None:
            continue
        labels.append(
            {
                "rollout_id": r.get("rollout_id"),
                "prompt": r.get("prompt"),
                "final_text": r.get("final_text"),
                "label": g,
                "annotator": "rule:invoice_over_200",
                "note": "deterministic verifier over steps",
            }
        )
    return wai.attach_labels(rows, labels, kind=kind, annotator="rule:invoice_over_200")


ACTION_CRITERION = "took the action the amount requires"


def gold_action(row: dict) -> int | None:
    """The rule, restricted to the action criterion alone.

    The overall gold folds in three duties; this isolates the one the judge's second
    criterion asks about, so judge and rule are compared on the same question.
    """
    f = read_steps(row)
    if not f["looked_up"] or f["invoice_amt"] is None:
        return None
    if f["invoice_amt"] > THRESHOLD:
        return 1 if (f["escalated"] and not f["credited"]) else 0
    return 1 if (f["credited"] and not f["escalated"]) else 0


def judge_action(row: dict) -> int | None:
    """The judge's verdict on that same criterion, off the rubric markers.

    rubric_judge writes one ``rubric:<slug>`` marker per criterion, 1.0 met / 0.0 not,
    so the per-criterion verdicts stay binary even when the overall reward is not.
    """
    markers = row.get("markers") or {}
    for key, val in markers.items():
        if key.startswith("rubric:") and "action" in key:
            try:
                return 1 if float(val) >= 0.5 else 0
            except (TypeError, ValueError):
                return None
    return None


def _binary(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v in (0.0, 1.0):
        return int(v)
    return None


def leak_pattern(leaks: list[dict]) -> dict:
    """Why the judge passed rows the rule failed, counted rather than asserted.

    Review point 5 on PR #348: the README claimed "every BIG leak is the same row" and
    results.json carried nothing a reader could check that against.
    """
    pattern = {"announced_but_no_call": 0, "wrong_tool_called": 0, "no_action_no_claim": 0}
    words = ("escalat", "credit", "refund")
    for r in leaks:
        f = read_steps(r)
        acted = f["escalated"] or f["credited"]
        claims = any(w in (r.get("final_text") or "").lower() for w in words)
        if not acted and claims:
            pattern["announced_but_no_call"] += 1
        elif acted:
            pattern["wrong_tool_called"] += 1
        else:
            pattern["no_action_no_claim"] += 1
    pattern["total"] = len(leaks)
    return pattern


def mcnemar(pairs: list[tuple[int, int]]) -> dict:
    """Exact McNemar over paired correct/incorrect outcomes.

    Review point 4 on PR #348: the literal-vs-original comparison is the SAME rows
    judged twice, so overlapping Wilson intervals on the two marginals are the wrong
    test. Only the discordant pairs carry information.
    """
    b = sum(1 for a_, b_ in pairs if a_ == 1 and b_ == 0)  # original right, literal wrong
    c = sum(1 for a_, b_ in pairs if a_ == 0 and b_ == 1)  # literal right, original wrong
    n = b + c
    if n == 0:
        return {"discordant": 0, "b_orig_only": 0, "c_lit_only": 0, "p_value": 1.0}
    # two-sided exact binomial against p=0.5
    from math import comb

    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return {
        "discordant": n,
        "b_orig_only": b,
        "c_lit_only": c,
        "p_value": round(min(1.0, 2 * tail), 4),
    }


def stage_report(args) -> dict:
    """Every number in the README, recomputed from saved rows. No network unless the
    judge_trust probes are asked for."""
    out: dict = {}
    judged = {
        t: read_jsonl(OUT / f"{t}.judged.jsonl")
        for t in ("base-big", "base-small", "trained-big", "trained-small")
    }
    judged = {k: v for k, v in judged.items() if v}
    if not judged:
        print("no judged rows on disk -- run `python run.py` first")
        return out

    base = judged.get("base-big", []) + judged.get("base-small", [])
    trained = judged.get("trained-big", []) + judged.get("trained-small", [])

    # ---- 0. does rubric_judge's output survive judge_agreement's 0/1 rule?
    print("\n== 0. judge reward shape ==")
    vals = [r.get("reward") for r in base if r.get("reward") is not None]
    frac = [v for v in vals if _binary(v) is None]
    out["reward_shape"] = {
        "rows": len(base),
        "with_reward": len(vals),
        "fractional_would_be_skipped": len(frac),
        "distinct": sorted({round(float(v), 4) for v in vals})[:12],
    }
    print(
        f"  {len(vals)}/{len(base)} rows carry a reward; distinct values "
        f"{out['reward_shape']['distinct']}"
    )
    print(f"  fractional (judge_agreement would SKIP these): {len(frac)}")

    # ---- 1. what kind of gold does the SDK think a deterministic rule is?
    print("\n== 1. attaching the deterministic verdict as gold ==")
    for kind in ("program", "human"):
        probe, _ = attach_gold([dict(r) for r in base], kind)
        kinds = {r.get("gold_kind") for r in probe if r.get("gold_kind")}
        a_ = wai.judge_agreement(probe, "gold_reward")
        print(f"  attach_labels(kind={kind!r:9}) -> gold_kind={kinds} ok={a_.get('ok')}")
    rows, _gold_report = attach_gold(base, "program")
    labelled = [r for r in rows if r.get("gold_reward") is not None]
    out["gold"] = {"rows": len(rows), "labelled": len(labelled)}
    print(f"  {len(labelled)}/{len(rows)} base rows carry a deterministic gold label")

    # ---- 2. judge vs the rule, whole verdict
    print("\n== 2. judge_agreement: rubric_judge vs the deterministic rule ==")
    agree = wai.judge_agreement(rows, "gold_reward", allow_model_gold=True)
    out["agreement_all"] = {
        k: agree.get(k)
        for k in (
            "n",
            "agreement",
            "ci95",
            "kappa",
            "pass_when_gold_fail",
            "fail_when_gold_pass",
            "gold_kind",
            "ok",
            "warnings",
        )
    }
    print(
        f"  n={agree.get('n')} agreement={agree.get('agreement')} ci95={agree.get('ci95')} "
        f"kappa={agree.get('kappa')}"
    )
    print(
        f"  leak (judge passed a gold failure)={agree.get('pass_when_gold_fail')} "
        f"| fail_when_gold_pass={agree.get('fail_when_gold_pass')}"
    )
    print(f"  gold_kind={agree.get('gold_kind')} ok={agree.get('ok')}")
    for w in agree.get("warnings") or []:
        print(f"  ! {w}")

    # ---- 3. the axis that matters: the two sides of the $200 rule
    print("\n== 3. the action criterion, split by side of the rule ==")
    per_class = {}
    for cls in ("BIG", "SMALL"):
        sub = [r for r in rows if gold_class(r) == cls]
        pairs = [(gold_action(r), judge_action(r)) for r in sub]
        pairs = [(g, j) for g, j in pairs if g is not None and j is not None]
        if not pairs:
            continue
        n = len(pairs)
        same = sum(1 for g, j in pairs if g == j)
        gold_fail = [(g, j) for g, j in pairs if g == 0]
        leaked = [1 for g, j in gold_fail if j == 1]
        lo, hi = wilson(len(leaked), len(gold_fail))
        alo, ahi = wilson(same, n)
        per_class[cls] = {
            "n": n,
            "agreement": round(same / n, 4),
            "agreement_ci95": [round(alo, 3), round(ahi, 3)],
            "rule_pass_rate": round(sum(g for g, _ in pairs) / n, 4),
            "judge_pass_rate": round(sum(j for _, j in pairs) / n, 4),
            "gold_fail_rows": len(gold_fail),
            "judge_passed_them": len(leaked),
            "leak_rate": round(len(leaked) / len(gold_fail), 4) if gold_fail else None,
            "leak_ci95": [round(lo, 3), round(hi, 3)],
            # Review point 5: make "every BIG leak is the same row" checkable from
            # results.json instead of taking the README's word for it.
            "leak_pattern": leak_pattern(
                [r for r in sub if gold_action(r) == 0 and judge_action(r) == 1]
            ),
        }
        c = per_class[cls]
        print(f"  {cls:5s} n={c['n']:4d} agreement={c['agreement']} {c['agreement_ci95']}")
        print(f"        rule passes {c['rule_pass_rate']} | judge passes {c['judge_pass_rate']}")
        print(
            f"        of {c['gold_fail_rows']} rows the RULE fails, the judge passed "
            f"{c['judge_passed_them']} -> leak {c['leak_rate']} {c['leak_ci95']}"
        )
    out["per_class_action"] = per_class

    # ---- 4. the judge's own test-retest
    # These rows carry no rollout_id, so pair by position: run_judge preserves input
    # order, which the assert below re-checks on (prompt, final_text) every run.
    print("\n== 4. judge test-retest (same rows, two independent passes) ==")
    print(
        "  NOTE: rubric_judge runs at JUDGE_TEMPERATURE = 0.0, so identical verdicts are"
        "\n  expected. This measures determinism, not stability under sampling, and is"
        "\n  NOT evidence the judge is trustworthy. Review point 3 on PR #348."
    )
    retest = {}
    for tag in ("base-big", "base-small"):
        p1 = read_jsonl(OUT / f"{tag}.judged.jsonl")
        p2 = read_jsonl(OUT / f"{tag}.judged2.jsonl")
        if not p1 or not p2:
            continue
        p1 = p1[: len(p2)]
        aligned = sum(
            1
            for x, y in zip(p1, p2)
            if x.get("final_text") == y.get("final_text") and x.get("prompt") == y.get("prompt")
        )
        pairs = [
            (x, y)
            for x, y in zip(p1, p2)
            if x.get("reward") is not None and y.get("reward") is not None
        ]
        if not pairs or aligned != len(p2):
            print(f"  {tag}: only {aligned}/{len(p2)} rows align by position, skipping")
            continue
        same = sum(
            1 for x, y in pairs if (float(x["reward"]) >= 0.5) == (float(y["reward"]) >= 0.5)
        )
        retest[tag] = {
            "n": len(pairs),
            "aligned": aligned,
            "identical": same,
            "self_agreement": round(same / len(pairs), 4),
            "judge_temperature": 0.0,
            "note": "temperature 0: determinism, not stability under sampling",
        }
        print(f"  {tag}: {same}/{len(pairs)} identical (temperature 0 -- expected)")
    out["judge_retest"] = retest

    # ---- 5. does the judge flip the before/after verdict?
    print("\n== 5. the same before/after, graded two ways ==")
    flip = {}
    if trained:
        trows, _ = attach_gold(trained, "program")
        # Review point 2 on PR #348: the first version let the judge grade all 360 rows
        # while the rule graded only the 221 it could label, so the two columns were not
        # the same experiment. Both graders now see exactly the rule-labelled rows.
        rows_lab = [r for r in rows if r.get("gold_reward") is not None]
        trows_lab = [r for r in trows if r.get("gold_reward") is not None]
        print(
            f"  restricted to rule-labelled rows: before {len(rows_lab)}/{len(rows)}, "
            f"after {len(trows_lab)}/{len(trows)} (both graders see the same rows)"
        )
        for name, key in (("rule", "gold_reward"), ("judge", "reward")):
            b = [dict(r, reward=r.get(key)) for r in rows_lab if r.get(key) is not None]
            t = [dict(r, reward=r.get(key)) for r in trows_lab if r.get(key) is not None]
            if not b or not t:
                continue
            try:
                d = wai.delta_report(b, t)
            except Exception as exc:
                print(f"  graded by {name}: delta_report raised {type(exc).__name__}: {exc}")
                continue
            m = (d.get("metrics") or {}).get("pass_at_1") or {}
            flip[name] = {
                k: m.get(k)
                for k in ("mean_a", "mean_b", "delta", "ci95", "p_value", "verdict", "n_paired")
            }
            lo_, hi_ = (m.get("ci95") or (None, None))[:2]
            print(
                f"  graded by {name:5s}: {m.get('mean_a'):.3f} -> {m.get('mean_b'):.3f} "
                f"delta={m.get('delta'):+.3f} [{lo_:+.3f}..{hi_:+.3f}] "
                f"p={m.get('p_value'):.4f} n_paired={m.get('n_paired')} "
                f"verdict={m.get('verdict')}"
            )
    out["verdict_flip"] = flip

    # ---- 6. the SDK's own gate
    print("\n== 6. judge_trust (the SDK's own gate) ==")
    try:
        judge = (
            None if args.no_judge_calls else wai.rubric_judge(RUBRIC, policy=POLICY, tools=TOOLS)
        )
        tr = wai.judge_trust(
            rows,
            judge,
            gold="gold_reward",
            sample=args.probe_sample,
            probes="all" if judge else None,
            allow_model_gold=True,
        )
        out["judge_trust"] = {
            k: tr.get(k)
            for k in (
                "ok",
                "n_rows",
                "n_labeled",
                "gold_kind",
                "agreement",
                "held_out_halves",
                "length_sensitivity",
                "perturbation",
                "probes",
                "exploitable_by",
                "warnings",
            )
        }
        print(wai.format_judge_trust(tr))
    except Exception as exc:
        print(f"  judge_trust raised {type(exc).__name__}: {exc}")
        out["judge_trust_error"] = f"{type(exc).__name__}: {exc}"

    # ---- 7. is the defect promptable? the same judge under LITERAL_RUBRIC
    print("\n== 7. same judge, action criterion written as an explicit presence check ==")
    literal = {}
    for tag in ("base-big", "base-small"):
        lit = read_jsonl(OUT / f"{tag}.literal.jsonl")
        orig = read_jsonl(OUT / f"{tag}.judged.jsonl")
        if not lit or not orig:
            continue
        cls = "BIG" if tag.endswith("big") else "SMALL"
        block = {}
        for name, src in (("original", orig), ("literal", lit)):
            pairs = [(gold_action(r), judge_action(r)) for r in src]
            pairs = [(g, j) for g, j in pairs if g is not None and j is not None]
            if not pairs:
                continue
            n = len(pairs)
            same = sum(1 for g, j in pairs if g == j)
            gf = [1 for g, j in pairs if g == 0]
            leaked = [1 for g, j in pairs if g == 0 and j == 1]
            lo, hi = wilson(same, n)
            llo, lhi = wilson(len(leaked), len(gf)) if gf else (0.0, 0.0)
            block[name] = {
                "n": n,
                "agreement": round(same / n, 4),
                "agreement_ci95": [round(lo, 3), round(hi, 3)],
                "gold_fail_rows": len(gf),
                "judge_passed_them": len(leaked),
                "leak_rate": round(len(leaked) / len(gf), 4) if gf else None,
                "leak_ci95": [round(llo, 3), round(lhi, 3)],
            }
            b = block[name]
            print(
                f"  {cls:5s} {name:8s} n={b['n']:4d} agreement={b['agreement']} "
                f"{b['agreement_ci95']} | leak {b['judge_passed_them']}/{b['gold_fail_rows']} "
                f"= {b['leak_rate']} {b['leak_ci95']}"
            )
        # Review point 4: these are the SAME rows judged twice, so the marginals'
        # overlapping Wilson intervals are the wrong test. Only discordant pairs count.
        orig_by = [(gold_action(r), judge_action(r)) for r in orig]
        lit_by = [(gold_action(r), judge_action(r)) for r in lit]
        paired = [
            (1 if go == jo else 0, 1 if gl == jl else 0)
            for (go, jo), (gl, jl) in zip(orig_by, lit_by)
            if go is not None and jo is not None and gl is not None and jl is not None
        ]
        block["mcnemar"] = mcnemar(paired)
        m = block["mcnemar"]
        print(
            f"  {cls:5s} McNemar: {m['discordant']} discordant "
            f"(orig-only {m['b_orig_only']}, literal-only {m['c_lit_only']}), p={m['p_value']}"
        )
        literal[cls] = block
    out["literal_rubric"] = literal

    (HERE / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {HERE / 'results.json'}")
    return out


def main(argv=None) -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "tasks", "arms", "judge", "literal", "report"],
    )
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--probe-sample", type=int, default=25)
    ap.add_argument("--retest-sample", type=int, default=120)
    ap.add_argument(
        "--arms", nargs="*", default=None, help="judge only these arms (default: all four)"
    )
    ap.add_argument(
        "--no-judge-calls",
        action="store_true",
        help="skip the judge_trust perturbation/probe passes (no judge calls)",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="cap the offline task grid (used by smoke.sh)"
    )
    ap.add_argument("--dry-run", action="store_true", help="offline: build tasks only")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    import whileai

    print(f"whileai {whileai.__version__}")
    check_gold_label()
    if args.limit:
        args.budget = min(args.budget, max(4, args.limit * 4))
    if args.dry_run:
        stage_tasks(args)
        return 0
    tasks = stage_tasks(args) if args.stage in ("all", "tasks") else read_jsonl(OUT / "tasks.jsonl")
    if args.stage in ("all", "arms"):
        stage_arms(args, tasks)
    if args.stage in ("all", "judge"):
        stage_judge(args)
    if args.stage == "literal":
        stage_literal(args)
    if args.stage in ("all", "report"):
        stage_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
