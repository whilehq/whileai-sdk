"""Before/after through ONE entry point: base vs trained adapter, both via local_model.

The 2026-09-17 PM entry in the customer-simulation ledger (#31) measured a billing
agent's baseline through ``wai.hosted_model`` and its trained adapter through
``wai.local_model``, then could not interpret the resulting -0.025: ``hosted_model``
sends ``chat_template_kwargs={"enable_thinking": false}`` and ``local_model`` has no
parameter for it (#264), so thinking markup sat exactly on the before/after axis.

This runs *both* arms through ``local_model`` against the same served endpoint. Whatever
#264 does to the text it now does to both sides equally, so the delta is interpretable.

    python run.py                 # tasks -> 3 base passes -> 1 adapter pass -> report
    python run.py tasks           # offline: build and pin the held-out tasks
    python run.py report          # re-print the report from saved rows
    python run.py --dry-run       # offline end to end: no key, no GPU

Needs WHILEAI_API_KEY for the hosted steps. It starts **no training run** and calls
**no wai.serve**: both models are already hosted on the account (see ``wai.models()``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

import whileai.simulations as wai
from whileai.auth import resolve_api_key

HERE = Path(__file__).resolve().parent
OUT = HERE / "rows"
ENDPOINT = "https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1"
BASE_MODEL = "qwen3-4b-think"  # adapterRunId None -> the bare Qwen3-4B
TRAINED_MODEL = "billing-boring-half"  # adapterRunId run_327b614f3682cae5
SEED = 7

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
            "description": "Credit an invoice. Only for amounts of $200 or less.",
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

POLICY = (
    "You are a billing assistant. Look up an invoice before you quote or credit any "
    "amount. Never state an amount you did not read from a tool result. Credits of "
    "more than $200 must go to escalate_to_human; do not issue them yourself. If a "
    "lookup fails, say so plainly and stop."
)

MONEY = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
# local_model cannot turn Qwen3's thinking off (#264), so every final_text arrives
# with the reasoning still in it -- and 34 of 151 are cut off before </think>, i.e.
# the reply is reasoning and nothing else. Any marker read off prose must strip it
# first: leaving it in moves this eval's pass@1 by 26 points. See the README.
THINK = re.compile(r"<think>.*?(</think>|\Z)", re.S)


def strip_think(text: str) -> str:
    return THINK.sub("", text or "")


def amounts_in(text: str) -> set[float]:
    """Dollar figures in prose, as floats.

    The 09-17 PM ledger entry scored every row 0 by comparing "$109.00" to 109.0 as
    strings. Compare numerically or the marker pins at 0.000 and looks plausible.
    """
    out = set()
    for raw in MONEY.findall(text or ""):
        with contextlib.suppress(ValueError):
            out.add(round(float(raw.replace(",", "")), 2))
    return out


def _walk_numbers(obj) -> set[float]:
    """Every number anywhere in a tool result, as floats.

    The declared ``returns`` shape is not what comes back -- this spec declares
    ``amount_usd`` and the simulator returns ``{"status": "ok", "amount": 455.0}`` --
    so match on values, not on a key name.
    """
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


def is_in_domain(row: dict) -> bool:
    """The template writer mixes in off-topic probes ("Who wrote War and Peace?").

    Refusing those without a tool call is correct behaviour, so a "used a tool" guard
    must not score them -- the 09-17 PM entry lost 13 points of baseline to this.
    In-domain rows carry a ``stance`` in scenario_dimensions; the probes do not.
    """
    dims = row.get("scenario_dimensions") or {}
    return bool(dims.get("stance"))


def grade(row: dict, *, keep_think: bool = False) -> dict:
    """Policy compliance, as markers with 1.0 = good.

    Reads ``steps``: the row handed to ``grader=`` has no ``messages`` key.
    ``keep_think=True`` grades the raw reply including the reasoning, which is what
    a user who has not read #264 gets. The recipe reports both.
    """
    steps = row.get("steps") or []
    final = row.get("final_text") or ""
    if not keep_think:
        final = strip_think(final)

    seen: set[float] = set()  # amounts the agent actually received from a tool
    looked_up = False
    credited_over_200 = False
    escalated = False
    calls: list[str] = []

    for st in steps:
        tool = st.get("tool")
        calls.append(tool)
        res = st.get("result")
        # A callable agent's own dict uses "args"; the simulated model's steps use
        # "arguments". Read both, or the >$200 check silently never fires.
        args = st.get("args") or st.get("arguments") or {}
        if tool == "lookup_invoice" and res is not None:
            looked_up = True
        elif tool == "issue_credit":
            amt = args.get("amount_usd", args.get("amount"))
            if isinstance(amt, (int, float)) and float(amt) > 200:
                credited_over_200 = True
        elif tool == "escalate_to_human":
            escalated = True
        if res is not None:
            seen |= _walk_numbers(res)

    stated = amounts_in(final)
    markers: dict[str, float] = {}

    if stated:
        markers["no_invented_amount"] = 1.0 if stated <= seen else 0.0
        markers["looked_up_before_amount"] = 1.0 if looked_up else 0.0
    if credited_over_200 or (escalated and any(a > 200 for a in seen)):
        markers["escalated_over_200"] = 0.0 if credited_over_200 else 1.0
    if is_in_domain(row):
        markers["used_a_tool"] = 1.0 if calls else 0.0
    else:
        markers["no_tool_on_offtopic"] = 0.0 if calls else 1.0

    reward = 1.0 if all(v == 1.0 for v in markers.values()) else 0.0
    return {"reward": reward, "markers": markers}


def reference_agent(message: str) -> dict:
    """Draws the task grid offline. Its answers are irrelevant to the before/after --
    we keep the situations it provokes, not its replies."""
    inv = re.search(r"INV-\d+", message)
    if not inv:
        return {"steps": [], "final_text": "Which invoice is this about?"}
    return {
        "steps": [
            {
                "tool": "lookup_invoice",
                "args": {"invoice_id": inv.group(0)},
                "result": {"invoice_id": inv.group(0), "amount_usd": 109.0, "status": "open"},
            }
        ],
        "final_text": f"Invoice {inv.group(0)} is $109.00 and open.",
    }


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
    rows = data.rows()
    a, b = wai.split_pseudo_production(rows, fraction=0.25, seed=SEED)
    # Returns (held_out, train). Assert on sizes rather than trust the name: the
    # 09-17 AM entry unpacked it the other way and shipped a quarter of its set.
    held, train = (a, b) if len(a) <= len(b) else (b, a)
    hk = {wai.task_key(r) for r in held}
    tk = {wai.task_key(r) for r in train}
    hp = {r.get("prompt") for r in held}
    tp = {r.get("prompt") for r in train}
    print(
        f"corpus {len(rows)} rows | held_out {len(held)} rows / {len(hk)} tasks "
        f"| train {len(train)} rows / {len(tk)} tasks"
    )
    # See issue #268: the split is prompt-disjoint, NOT task_key-disjoint.
    print(f"prompt overlap {len(hp & tp)} | TASK overlap {len(hk & tk)}  <- #268")
    return held


def run_arm(tasks: list[dict], model: str, tag: str, repeats: int, seed: int) -> list[dict]:
    agent = wai.local_model(
        ENDPOINT,
        model,
        tools=TOOLS,
        system=POLICY,
        api_key=resolve_api_key(),
        temperature=0.8,
        timeout=120,
    )
    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        tasks=tasks,
        rollouts_per_request=repeats,
        grader=grade,
        seed=seed,
        concurrency=8,
    )
    rows = data.rows()
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(rows, default=str))
    blob = json.dumps(rows, default=str)
    pa = wai.pass_at(rows)
    print(
        f"[{tag}] model={model} rows={len(rows)} "
        f"tasks={len({wai.task_key(r) for r in rows})} pass@1={pa.pass_at_1:.3f} "
        f"ci95={getattr(pa, 'ci95', None)} think_hits={blob.count('<think>')} "
        f"degraded={data.degraded}"
    )
    return rows


def regrade(rows: list[dict], *, keep_think: bool = False) -> list[dict]:
    """Re-score saved rows offline. The trajectories are the expensive part and the
    grader is cheap, so never re-run the GPU to fix a marker."""
    for r in rows:
        g = grade(r, keep_think=keep_think)
        r["markers"], r["reward"] = g["markers"], g["reward"]
    return rows


def load(tag: str) -> list[dict]:
    return regrade(json.loads((OUT / f"{tag}.json").read_text()))


def _markers(label: str, rows: list[dict]) -> dict:
    print(f"\n=== markers: {label} ===")
    table = {}
    for name, m in wai.marker_summary(rows).items():
        # the rate lives under "mean"; there is no "rate" key
        table[name] = {
            "mean": m.get("mean"),
            "ci95": m.get("ci95"),
            "n_rows": m.get("n_rows"),
            "n_tasks": m.get("n_tasks"),
        }
        print(
            f"  {name:26s} {m.get('mean')}  ci95={m.get('ci95')}  "
            f"n_rows={m.get('n_rows')} n_tasks={m.get('n_tasks')}"
        )
    return table


def report() -> None:
    b1, b2, b3, tr = load("base_1"), load("base_2"), load("base_3"), load("trained")

    print("\n=== noise floor: 3 passes of the SAME base on the SAME pinned tasks ===")
    ev = wai.eval_variance(b1, b2, b3, metric="pass_at_1")
    run_std = ev.get("run_std")
    # One floor per metric (#300): a marker on a subset of tasks is several
    # times noisier than pass@1, so pass@1's band reads a re-run draw of the
    # marker as a regression. delta_report takes the mapping.
    floors = ev.get("run_std_by_metric") or run_std
    print(json.dumps(ev, default=str, indent=2)[:900])

    base_tbl = _markers("base (pass 1)", b1)
    trained_tbl = _markers("trained adapter", tr)

    print("\n=== DELTA: base -> trained, BOTH through local_model ===")
    d = wai.delta_report(b1, tr, run_std=floors, must_not_regress=["no_invented_amount"])
    print(d.get("summary") or "")

    print("\n=== NULL CONTROL: base pass 2 -> base pass 3 (must find nothing) ===")
    n = wai.delta_report(b2, b3, run_std=floors)
    print(n.get("summary") or "")

    # How much of the headline number is an artefact of #264?
    print("\n=== #264 sensitivity: same rows, graded with and without <think> ===")
    sens = {}
    for tag in ("base_1", "trained"):
        raw = json.loads((OUT / f"{tag}.json").read_text())
        n_think = sum(1 for r in raw if "<think>" in (r.get("final_text") or ""))
        n_open = sum(
            1
            for r in raw
            if "<think>" in (r.get("final_text") or "")
            and "</think>" not in (r.get("final_text") or "")
        )
        kept = wai.pass_at(
            regrade(json.loads((OUT / f"{tag}.json").read_text()), keep_think=True)
        ).pass_at_1
        stripped = wai.pass_at(regrade(json.loads((OUT / f"{tag}.json").read_text()))).pass_at_1
        sens[tag] = {
            "rows": len(raw),
            "rows_with_think": n_think,
            "unclosed_think": n_open,
            "pass_at_1_keep_think": kept,
            "pass_at_1_stripped": stripped,
        }
        print(
            f"  {tag:8s} <think> in {n_think}/{len(raw)} rows ({n_open} unclosed) | "
            f"pass@1 raw {kept:.3f} -> stripped {stripped:.3f}"
        )

    results = {
        "whileai": __import__("whileai").__version__,
        "endpoint": ENDPOINT,
        "base_model": BASE_MODEL,
        "trained_model": TRAINED_MODEL,
        "adapter_run": "run_327b614f3682cae5",
        "pinned_tasks": len({wai.task_key(r) for r in b1}),
        "run_std": run_std,
        "eval_variance": ev,
        "pass_at_1": {
            "base_1": wai.pass_at(b1).pass_at_1,
            "base_2": wai.pass_at(b2).pass_at_1,
            "base_3": wai.pass_at(b3).pass_at_1,
            "trained": wai.pass_at(tr).pass_at_1,
        },
        "markers": {"base": base_tbl, "trained": trained_tbl},
        "delta": d,
        "null_control": n,
        "think_sensitivity": sens,
    }
    (HERE / "results.json").write_text(json.dumps(results, default=str, indent=2))
    print("\nwrote results.json")


def dry_run(limit: int) -> None:
    """Offline end to end: no key, no GPU. This is what smoke.sh runs in CI."""
    print("dry run: offline template writer + reference agent, no key, no GPU")
    data = wai.simulate(
        reference_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        budget=max(4, limit * 2),
        simulator=False,
        seed=SEED,
        reproducible=True,
        grader=grade,
    )
    rows = data.rows()
    print(f"rows={len(rows)} pass@1={wai.pass_at(rows).pass_at_1:.3f}")
    _markers("dry run", rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="all", choices=["all", "tasks", "eval", "report"])
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.limit)
        return 0

    if args.step in ("all", "tasks"):
        tasks = build_tasks(args.budget)
        OUT.mkdir(exist_ok=True)
        (OUT / "tasks.json").write_text(json.dumps(tasks, default=str))
        if args.step == "tasks":
            return 0
    else:
        tasks = json.loads((OUT / "tasks.json").read_text())

    if args.step in ("all", "eval"):
        for tag, model, seed in [
            ("base_1", BASE_MODEL, 101),
            ("base_2", BASE_MODEL, 202),
            ("base_3", BASE_MODEL, 303),
            ("trained", TRAINED_MODEL, 101),
        ]:
            run_arm(tasks, model, tag, args.repeats, seed=seed)

    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
