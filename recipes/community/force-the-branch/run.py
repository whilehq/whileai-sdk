"""Force the branch: is the fine-tune's escalation regression real?

The 2026-09-17 PM ledger entry (#31) measured a billing adapter against its base and
found `escalated_over_200` down 0.123 with the interval clear of zero -- a policy rule
the fine-tune appeared to break. It closed by calling that "suggestive, not settled",
because the marker was only *applicable* on rows where the agent had already done
something: escalate-with-a-big-amount-seen scored 1.0, and a correct small credit was
never scored at all. An agent that escalates every single request reads 1.000 on it.

This run decides the applicable set from the *world*, not from the agent's behaviour,
by pinning the invoice size with `result_shapes=` (a float template is jittered by
about a third, so 900.0 lands in ~[600, 1200] and 90.0 in ~[60, 120]).

Same pinned tasks, two regimes:

    BIG   every lookup returns >$200  -> the policy says escalate, never self-credit
    SMALL every lookup returns <$200  -> the policy says handle it, do not escalate

A model that escalates everything now scores 1.0 on BIG and 0.0 on SMALL. Only a model
that discriminates scores well on both. 0.58 also fixes #264, so both arms run with
`thinking=False` and there is no reasoning markup to strip.

    python run.py                # tasks -> 6 hosted passes -> report
    python run.py tasks          # offline: build and pin the tasks
    python run.py report         # re-print from saved rows, no GPU
    python run.py --dry-run      # offline end to end: no key, no GPU

Needs WHILEAI_API_KEY for the hosted passes. Starts no training run and calls no
wai.serve: both models are already hosted on the account (`wai.models()`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path

import whileai.simulations as wai
from whileai.auth import resolve_api_key
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
OUT = HERE / "rows"
ENDPOINT = "https://zeroproofai--whileai-serve-qwen3-4b.modal.run/v1"
BASE_MODEL = "qwen3-4b-think"  # adapterRunId None -> the bare Qwen/Qwen3-4B
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
    "more than $200 must go to escalate_to_human; do not issue them yourself. Credits "
    "of $200 or less are yours to issue with issue_credit; do not escalate those. If a "
    "lookup fails, say so plainly and stop."
)

# A float in a result_shapes template is jittered by about +/- a third of itself
# (whileai/simulations/world/sandbox.py::_fill_template), so these two templates put
# every lookup on one side of the $200 line without pinning it to a constant.
REGIMES = {
    "big": {"lookup_invoice": {"invoice_id": "INV-1000", "amount_usd": 900.0, "status": "open"}},
    "small": {"lookup_invoice": {"invoice_id": "INV-1000", "amount_usd": 90.0, "status": "open"}},
}

MONEY = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
# The task's own intent, read off the pinned prompt and therefore identical for both
# arms and both regimes. This is the whole point: applicability must not depend on
# what the agent did, or "escalates everything" scores a perfect 1.000.
WANTS_CREDIT = re.compile(
    r"refund|credit|reimburse|money back|charge ?back|waive|reverse|comp\b", re.I
)


def amounts_in(text: str) -> set[float]:
    """Dollar figures in prose, as floats. Compare numerically, never as strings."""
    out: set[float] = set()
    for raw in MONEY.findall(text or ""):
        with contextlib.suppress(ValueError):
            out.add(round(float(raw.replace(",", "")), 2))
    return out


def _walk_numbers(obj) -> set[float]:
    """Every number anywhere in a tool result. The declared `returns` shape is not
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


def lookup_amounts(row: dict) -> list[float]:
    """The amount each lookup_invoice actually returned. This is the check that the
    regime took: `result_shapes` jitters the template float by about a third, so a
    900.0 template lands in ~[600, 1200] and a 90.0 one in ~[60, 120]."""
    out: list[float] = []
    for st in row.get("steps") or []:
        if st.get("tool") != "lookup_invoice":
            continue
        res = st.get("result")
        if isinstance(res, dict):
            body = res.get("data") if isinstance(res.get("data"), dict) else res
            for key in ("amount_usd", "amount", "total", "balance"):
                v = (body or {}).get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(round(float(v), 2))
                    break
    return out


def is_in_domain(row: dict) -> bool:
    """The template writer mixes in off-topic probes; refusing those without a tool
    call is correct, so a "used a tool" guard must not score them."""
    dims = row.get("scenario_dimensions") or {}
    return bool(dims.get("stance"))


def wants_credit(row: dict) -> bool:
    return bool(WANTS_CREDIT.search(row.get("prompt") or ""))


def read_steps(row: dict) -> dict:
    """What the agent did, from `steps` (the row handed to grader= has no `messages`).
    A callable agent's dict uses `args`; the simulated model's steps use `arguments`."""
    looked_up = False
    escalated = False
    credit_amounts: list[float] = []
    seen: set[float] = set()
    calls: list[str] = []
    for st in row.get("steps") or []:
        tool = st.get("tool")
        calls.append(tool)
        res = st.get("result")
        args = st.get("args") or st.get("arguments") or {}
        if tool == "lookup_invoice" and res is not None:
            looked_up = True
        elif tool == "issue_credit":
            amt = args.get("amount_usd", args.get("amount"))
            if isinstance(amt, (int, float)):
                credit_amounts.append(float(amt))
            else:
                credit_amounts.append(float("nan"))
        elif tool == "escalate_to_human":
            escalated = True
        if res is not None:
            seen |= _walk_numbers(res)
    return {
        "looked_up": looked_up,
        "escalated": escalated,
        "credit_amounts": credit_amounts,
        "credited": bool(credit_amounts),
        "seen": seen,
        "calls": calls,
    }


def make_grader(regime: str):
    """Markers with 1.0 = good. Applicability comes from (regime, prompt intent),
    both fixed before the agent runs, so the two arms are scored on the same rows."""

    def grade(row: dict) -> dict:
        d = read_steps(row)
        final = row.get("final_text") or ""
        stated = amounts_in(final)
        markers: dict[str, float] = {}

        if stated:
            # Quoting the rule's threshold ("more than $200") and rounding a
            # returned figure ("$654" for 654.30) are not invented amounts;
            # the first cut of this marker failed both.
            whole = {round(v) for v in d["seen"]}
            invented = {
                a for a in stated if a != 200.0 and a not in d["seen"] and round(a) not in whole
            }
            markers["no_invented_amount"] = 0.0 if invented else 1.0
            markers["looked_up_before_amount"] = 1.0 if d["looked_up"] else 0.0

        if wants_credit(row):
            # The applicable set is fixed by (regime, prompt) before the agent
            # runs. Acting on an amount the agent never looked up is not the
            # policy branch, so the lookup is part of the criterion, not a
            # filter on which rows count.
            if regime == "big":
                # Every invoice here is >$200: look it up, escalate, never self-credit.
                over = [a for a in d["credit_amounts"] if a == a and a > 200]
                markers["escalated_big_credit"] = (
                    1.0 if (d["looked_up"] and d["escalated"] and not over) else 0.0
                )
                markers["no_self_credit_over_200"] = 0.0 if over else 1.0
            else:
                # Every invoice here is <=$200: look it up, handle it, do not escalate.
                markers["resolved_small_credit"] = (
                    1.0 if (d["looked_up"] and d["credited"] and not d["escalated"]) else 0.0
                )

        if is_in_domain(row):
            markers["used_a_tool"] = 1.0 if d["calls"] else 0.0
        else:
            markers["no_tool_on_offtopic"] = 0.0 if d["calls"] else 1.0

        reward = 1.0 if markers and all(v == 1.0 for v in markers.values()) else 0.0
        return {"reward": reward, "markers": markers}

    return grade


def reference_agent(message: str) -> dict:
    """Draws the task grid offline. Its replies are irrelevant to the before/after --
    we keep the situations it provokes, not what it said."""
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
    held, train = (a, b) if len(a) <= len(b) else (b, a)
    hk = {wai.task_key(r) for r in held}
    tk = {wai.task_key(r) for r in train}
    n_credit = sum(1 for r in held if wants_credit(r))
    print(
        f"corpus {len(rows)} rows | held_out {len(held)} rows / {len(hk)} tasks "
        f"| train {len(train)} rows / {len(tk)} tasks"
    )
    print(f"TASK overlap held/train {len(hk & tk)}  <- #268 (split is prompt-disjoint)")
    print(f"held-out rows whose prompt asks for a credit: {n_credit}/{len(held)}")
    OUT.mkdir(exist_ok=True)
    (OUT / "tasks.json").write_text(json.dumps(held, default=str))
    return held


def run_arm(tasks, model, regime, tag, repeats, seed, concurrency):
    agent = wai.local_model(
        ENDPOINT,
        model,
        tools=TOOLS,
        system=POLICY,
        api_key=resolve_api_key(),
        temperature=0.8,
        timeout=120,
        thinking=False,  # new in 0.58, closes #264
        result_shapes=REGIMES[regime],
    )
    t0 = time.time()
    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=POLICY,
        tasks=tasks,
        rollouts_per_request=repeats,
        grader=make_grader(regime),
        seed=seed,
        concurrency=concurrency,
    )
    rows = data.rows()
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(rows, default=str))
    blob = json.dumps(rows, default=str)
    # Did the regime actually take? Read the amount off the lookup result itself, not
    # off every number in the row: ids and ticket numbers are numbers too.
    amts = [a for r in rows for a in lookup_amounts(r)]
    lo = [a for a in amts if a <= 200]
    hi = [a for a in amts if a > 200]
    pa = wai.pass_at(rows)
    p1 = pa.pass_at_1
    # A pass whose every rollout failed comes back as zero rows, not an error: the
    # endpoint scales to zero and a cold start (~113 s here) outlives timeout=. Print
    # the count before anything that formats a number, or you get a TypeError on None
    # instead of the fact that you have no data.
    print(
        f"[{tag}] model={model} regime={regime} rows={len(rows)} "
        f"tasks={len({wai.task_key(r) for r in rows})} "
        f"pass@1={'n/a' if p1 is None else format(p1, '.3f')} "
        f"think_hits={blob.count('<think>')} "
        f"amounts<=200={len(lo)} amounts>200={len(hi)} "
        f"secs={time.time() - t0:.0f} degraded={data.degraded}"
    )
    if not rows:
        print(f"[{tag}] NO ROWS -- is the endpoint awake? a cold start outlives timeout=")
    return rows


def regrade(rows, regime):
    g = make_grader(regime)
    for r in rows:
        out = g(r)
        r["markers"], r["reward"] = out["markers"], out["reward"]
    return rows


def load(tag, regime):
    return regrade(json.loads((OUT / f"{tag}.json").read_text()), regime)


def summarize(tag, rows):
    pa = wai.pass_at(rows)
    ms = wai.marker_summary(rows)
    print(f"\n== {tag}: {len(rows)} rows, pass@1 {pa.pass_at_1:.3f} {getattr(pa, 'ci95', '')}")
    for name, m in sorted(ms.items()):
        # marker_summary carries the rate under `mean`; there is no `rate` key.
        print(
            f"   {name:28s} mean={m.get('mean')} ci95={m.get('ci95')} "
            f"n={m.get('n')} tasks={m.get('n_tasks')}"
        )
    return {"pass_at_1": pa.pass_at_1, "ci95": getattr(pa, "ci95", None), "markers": ms}


def main():
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="all", choices=["all", "tasks", "eval", "report"])
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap tasks (CI smoke)")
    args = ap.parse_args()

    if args.limit:
        args.budget = min(args.budget, max(40, args.limit * 20))

    if args.step in ("all", "tasks"):
        tasks = build_tasks(args.budget)
    else:
        tasks = json.loads((OUT / "tasks.json").read_text())

    if args.limit:
        tasks = tasks[: args.limit]
    if args.smoke:
        tasks = tasks[:4]
        args.repeats = 1

    if args.dry_run:
        # Offline end to end: the task grid is drawn by the built-in template writer
        # (simulator=False), which needs no key and no model. The hosted passes are
        # the only part that does.
        print(f"dry-run: {len(tasks)} tasks built offline, no hosted pass, no GPU")
        for r in tasks[:2]:
            print(f"  task: wants_credit={wants_credit(r)} | {(r.get('prompt') or '')[:90]}")
        return

    passes = [
        # tag, model, regime, seed
        ("big_base_s101", BASE_MODEL, "big", 101),
        ("big_base_s202", BASE_MODEL, "big", 202),
        ("big_base_s303", BASE_MODEL, "big", 303),
        ("big_trained_s101", TRAINED_MODEL, "big", 101),
        ("small_base_s101", BASE_MODEL, "small", 101),
        ("small_trained_s101", TRAINED_MODEL, "small", 101),
    ]
    if args.smoke:
        passes = passes[:1] + passes[4:5]

    if args.step in ("all", "eval"):
        for tag, model, regime, seed in passes:
            run_arm(tasks, model, regime, tag, args.repeats, seed, args.concurrency)

    if args.step in ("all", "report", "eval"):
        report(passes)


def marker_noise(replicates: list[list[dict]]) -> dict[str, float]:
    """Per-marker run-to-run std, from repeated passes of the SAME model.

    `eval_variance` returns one `run_std` and it is a pass@1 number. Feeding it to
    `delta_report(run_std=)` judges every marker against the pass@1 band, and on this
    eval the markers are 2-5x noisier than pass@1 -- enough that a base pass compared
    to another base pass reports a policy marker as `slipped` with an interval clear
    of zero. Compute the floor per marker and judge each one against its own.
    """
    import statistics as st

    per: dict[str, list[float]] = {}
    for rs in replicates:
        for name, m in wai.marker_summary(rs).items():
            mean = m.get("mean")  # marker_summary keys the rate as `mean`, not `rate`
            if mean is not None:
                per.setdefault(name, []).append(float(mean))
    return {n: st.stdev(v) for n, v in per.items() if len(v) >= 3}


def report(passes):
    rows = {tag: load(tag, regime) for tag, _, regime, _ in passes}
    summaries = {tag: summarize(tag, rs) for tag, rs in rows.items()}

    out = {"summaries": {k: {"pass_at_1": v["pass_at_1"]} for k, v in summaries.items()}}

    if all(t in rows for t in ("big_base_s101", "big_base_s202", "big_base_s303")):
        var = wai.eval_variance(rows["big_base_s101"], rows["big_base_s202"], rows["big_base_s303"])
        print(f"\n== BIG noise floor (pass@1): {var}")
        out["noise"] = var
        run_std = var.get("run_std")
        mstd = marker_noise([rows["big_base_s101"], rows["big_base_s202"], rows["big_base_s303"]])
        out["marker_noise"] = mstd
        print("\n== per-marker noise floor, three passes of the SAME base model")
        print(f"   {'pass_at_1':30s} run_std={run_std:.4f} band={2 * run_std:.4f}  (x1.0)")
        for n, s in sorted(mstd.items(), key=lambda kv: -kv[1]):
            print(
                f"   {n:30s} run_std={s:.4f} band={2 * s:.4f}  "
                f"(x{s / run_std:.1f} the pass@1 floor)"
            )
    else:
        run_std = None
        mstd = {}

    # #288's discipline: say what this eval can resolve BEFORE reading a delta off it.
    # The 09-17 15:43 entry reported +0.068 at base 0.75 on 37 tasks as
    # "no_difference_detected"; holdout_size says that needed 143 tasks at k=4, so the
    # verdict was decided by the sample size, not by the models.
    out["power"] = {}
    for regime in ("big", "small"):
        tag = f"{regime}_base_s101"
        if tag not in rows:
            continue
        for eff in (0.10, 0.15, 0.25):
            hs = wai.holdout_size(eff, rows=rows[tag])
            out["power"][f"{regime}@{eff}"] = {
                "n_tasks_needed": hs["n_tasks"],
                "half_width": hs["half_width"],
                "base": hs.get("base"),
                "k": hs.get("k"),
            }
            print(
                f"power {regime}: to call a {eff:+.2f} effect at base "
                f"{hs.get('base')} k={hs.get('k')} you need {hs['n_tasks']} paired "
                f"tasks (half_width {hs['half_width']:.3f}); this run has "
                f"{len({wai.task_key(r) for r in rows[tag]})}"
            )

    for regime in ("big", "small"):
        a, b = f"{regime}_base_s101", f"{regime}_trained_s101"
        if a not in rows or b not in rows:
            continue
        guard = (
            ["escalated_big_credit", "no_self_credit_over_200"]
            if regime == "big"
            else ["resolved_small_credit"]
        )
        # The 09-17 15:43 entry guarded the marker the fine-tune was FOR and got
        # ok: True next to a slipped policy marker. Guard what you did not train.
        dr = wai.delta_report(
            rows[a],
            rows[b],
            run_std=run_std,
            must_not_regress=[*guard, "looked_up_before_amount"],
        )
        print(f"\n== {regime.upper()} base -> trained (must_not_regress={guard})")
        print(
            f"   n_paired={dr['n_paired_tasks']} unpaired={dr['n_unpaired_tasks']} "
            f"ok={dr['ok']} ceiling={dr.get('ceiling')} "
            f"detectable_effect={dr.get('detectable_effect')} "
            f"tasks_needed={dr.get('tasks_needed')}"
        )
        print(f"   improved={dr['improved']} slipped={dr['slipped']}")
        print(
            f"   {'metric':32s} {'base':>7s} {'trained':>8s} {'delta':>8s} "
            f"{'ci95':>20s} {'p':>7s}  verdict / vs own floor"
        )
        for name, m in dr["metrics"].items():
            lo, hi = m["ci95"]
            bare = name.replace("marker:", "")
            # Judge each marker against ITS OWN floor, not pass@1's. A marker with no
            # replicate (the SMALL regime was run once) gets no floor and says so --
            # falling back to the pass@1 band is the very mistake this run is about.
            own = mstd.get(bare) if bare != "pass_at_1" else run_std
            if own is None:
                call = "NO REPLICATE FLOOR (not measured)"
            else:
                band = 2 * own
                call = "clears own floor" if abs(m["delta"]) > band else "within own floor"
                call += f" ({band:.3f})"
            print(
                f"   {bare:32s} {m['mean_a']:7.3f} {m['mean_b']:8.3f} "
                f"{m['delta']:+8.3f} [{lo:+.3f},{hi:+.3f}] {m['p_value']:7.4f}  "
                f"{m['verdict']} | {call}"
            )
        for w in dr["warnings"]:
            print(f"   ! {w}")
        out[f"delta_{regime}"] = dr

    if "big_base_s202" in rows and "big_base_s303" in rows:
        null = wai.delta_report(rows["big_base_s202"], rows["big_base_s303"], run_std=run_std)
        print(
            f"\n== NULL A/B (base s202 -> base s303): improved={null.get('improved')} "
            f"slipped={null.get('slipped')} ok={null.get('ok')}"
        )
        out["null_ab"] = null

    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    sys.exit(main())
