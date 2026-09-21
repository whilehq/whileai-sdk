"""Which half of my agent's traffic can a 1.5B model take over?

The production agent is a tool-calling support agent. Its job splits in two:
pick the tool (the interesting half) and, given what the tool returned, write
an honest reply (the boring half). This asks whether a small open model can
take over the boring half on routine traffic only, and proves it on a frozen
held-out set before anything ships.

    python run.py             # traces -> graded rows -> boring/hard split -> SFT file
    python run.py --dry-run   # a small slice, printed, writing nothing

Offline: no model key, no network. The production agent is `wai.seeded_agent`,
which plants labelled mistakes so the program grader can be checked against
ground truth before it is trusted on a model whose rows carry no labels.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from grader import _faulted as faulted
from grader import _steps as row_steps
from grader import flaws, policy_ok

import whileai as wai
import whileai.simulations as wsim
from whileai.config import provenance

HERE = pathlib.Path(__file__).parent
OUT = HERE / "out"
K = 8
SEED = 0
BUDGET = 9000  # the offline writer saturates at ~117 asks; see README


@wai.tool
def get_order(order_id: str) -> dict:
    """Look up an order by id."""
    ...


@wai.tool
def refund_order(order_id: str, amount_cents: int) -> dict:
    """Refund an order."""
    ...


@wai.tool
def cancel_order(order_id: str) -> dict:
    """Cancel an order."""
    ...


TOOLS = [get_order, refund_order, cancel_order]
POLICY = (
    "You are a support agent. Call exactly one tool, then tell the customer what "
    "actually happened in one or two plain sentences. Never claim an action "
    "succeeded when the tool did not. Do not apologise when nothing went wrong. "
    "Do not repeat internal notes back to the customer."
)


def reply_prompt(row: dict) -> list[dict]:
    """The boring half as a chat prompt: the ask, the call, the result."""
    step = (row.get("steps") or [{}])[0]
    call = json.dumps({"tool": step.get("tool"), "arguments": step.get("arguments") or {}})
    result = json.dumps(step.get("result"))
    return [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": row["prompt"]},
        {
            "role": "assistant",
            "content": f"I will call a tool.\nTOOL_CALL {call}",
        },
        {
            "role": "user",
            "content": f"TOOL_RESULT {result}\n\nNow write the reply to the customer.",
        },
    ]


def main(dry_run: bool = False) -> None:
    print(provenance(), file=sys.stderr)
    OUT.mkdir(exist_ok=True)
    budget = 400 if dry_run else BUDGET
    repeats = 4 if dry_run else K

    # 1. The production agent's traffic, graded by a program.
    data = wai.simulate(
        wai.seeded_agent(TOOLS, rate=0.35, seed=SEED),
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        mode="rl",
        repeats=repeats,
        repeat_policy="fixed",
        budget=budget,
        reproducible=True,
        seed=SEED,
    )
    scored = data.grade(judge=policy_ok)
    rows = scored.rows() if callable(scored.rows) else scored.rows
    print(f"\ntraffic: {len(rows)} rollouts over {len({r['scenario_id'] for r in rows})} asks")
    print(scored.pass_at)

    # 2. Is the program grader worth trusting? Check it against the planted labels.
    tp = sum(1 for r in rows if r.get("seeded") and flaws(r))
    fn = sum(1 for r in rows if r.get("seeded") and not flaws(r))
    fp = sum(1 for r in rows if not r.get("seeded") and flaws(r))
    tn = sum(1 for r in rows if not r.get("seeded") and not flaws(r))
    agreement = (tp + tn) / len(rows)
    print(
        f"grader vs ground truth: agreement {agreement:.3f} "
        f"(caught {tp}, missed {fn}, false alarms {fp}, clean {tn})"
    )

    # 3. The split the SDK already knows: `tier` on every row.
    by_tier: dict[str, int] = {}
    for r in rows:
        by_tier[r.get("tier") or "?"] = by_tier.get(r.get("tier") or "?", 0) + 1
    print(f"tiers: {by_tier}")

    def boring(r: dict) -> bool:
        """The boring half: the tool came back clean, so the reply is a plain report.

        This is a router an engineer can actually run: the tool result is known
        before the reply is written, so no model call is needed to route on it.
        `tier` looked like the split the SDK already knew, but the incumbent's
        pass rate is the same on both sides of it, so it is not one.
        """
        steps = row_steps(r)
        if not steps:
            return False
        return not faulted(r)

    # 4. Hold out by ask, never by rollout.
    asks = sorted({r["scenario_id"] for r in rows})
    held = set(asks[::2])  # half, because the offline writer caps the pool
    heldout = [r for r in rows if r["scenario_id"] in held]
    train_pool = [r for r in rows if r["scenario_id"] not in held]

    ho_boring = [r for r in heldout if boring(r)]
    ho_hard = [r for r in heldout if not boring(r) and (r.get("steps") or [])]
    print(
        f"held out {len(held)} asks: {len({r['scenario_id'] for r in ho_boring})} boring, "
        f"{len({r['scenario_id'] for r in ho_hard})} hard"
    )

    # 5. What effect could this holdout even detect?
    floor = {
        e: wsim.holdout_size(e, base=0.70, k=repeats).get("n_tasks")
        for e in (0.10, 0.20, 0.25, 0.30)
    }
    print(f"holdout_size by effect (base=0.70, k={repeats}): {floor}")

    # 6. Train only on the boring half, and only where the agent was clean.
    demos = [r for r in train_pool if boring(r) and r["reward"] == 1]
    kept, _report = wai.decontaminate(demos, heldout)
    print(
        f"decontaminate: {len(demos)} in, {len(kept)} kept, {len(demos) - len(kept)} contaminated"
    )

    sel = wai.select(kept, mode="sft", target=1000)
    print(sel)
    sel_rows = list(sel)  # Selection is a list subclass; it *is* the rows

    # 7. The SFT files: the boring half as prompt -> reply.
    #    `select(mode="sft")` keeps the best completion per prompt (rejection
    #    sampling). With the offline writer's ask ceiling that leaves very few
    #    rows, so we also write every clean boring row and train on that.
    def to_sft(rs: list[dict]) -> list[dict]:
        out = []
        for r in rs:
            if not (r.get("steps") or []) or not (r.get("final_text") or "").strip():
                continue
            out.append(
                {
                    "messages": [
                        *reply_prompt(r),
                        {"role": "assistant", "content": r["final_text"]},
                    ],
                    "scenario_id": r["scenario_id"],
                }
            )
        return out

    selected = to_sft(sel_rows)
    if not dry_run:
        (OUT / "train_sft_selected.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in selected)
        )
    sft = to_sft(kept)
    if not dry_run:
        (OUT / "train_sft.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sft))
    print(
        f"{'would write' if dry_run else 'wrote'} {len(sft)} SFT rows -> out/train_sft.jsonl "
        f"({len(selected)} after select(mode='sft') -> out/train_sft_selected.jsonl, "
        f"{len({s['scenario_id'] for s in sft})} distinct asks)"
    )

    # 8. The held-out sets the GPU will see, and the bar to clear.
    def pack(rs: list[dict], slice_name: str) -> list[dict]:
        return [
            {
                "scenario_id": r["scenario_id"],
                "rollout_index": r.get("rollout_index", 0),
                "tier": r.get("tier"),
                "slice": slice_name,
                "prompt": r["prompt"],
                "steps": r["steps"],
                "incumbent_text": r.get("final_text") or "",
                "incumbent_reward": r["reward"],
                "chat": reply_prompt(r),
            }
            for r in rs
        ]

    packed = pack(ho_boring, "boring") + pack(ho_hard, "hard")
    if dry_run:
        print(f"dry run: {len(packed)} holdout rows, {len(sft)} SFT rows; wrote nothing")
        return
    (OUT / "holdout.json").write_text(json.dumps(packed, indent=1))
    for name, rs in (("boring", ho_boring), ("hard", ho_hard)):
        pa = wai.pass_at(rs)
        print(f"incumbent on held-out {name}: {pa}")

    (OUT / "prep.json").write_text(
        json.dumps(
            {
                "rollouts": len(rows),
                "asks": len({r["scenario_id"] for r in rows}),
                "grader_agreement": agreement,
                "grader_caught": tp,
                "grader_missed": fn,
                "grader_false_alarms": fp,
                "tiers": by_tier,
                "heldout_boring_asks": len({r["scenario_id"] for r in ho_boring}),
                "heldout_hard_asks": len({r["scenario_id"] for r in ho_hard}),
                "holdout_size_by_effect": floor,
                "sft_rows": len(sft),
                "sft_rows_after_select": len(selected),
                "sft_distinct_asks": len({s["scenario_id"] for s in sft}),
                "decontaminated": len(demos) - len(kept),
                "K": repeats,
                "seed": SEED,
            },
            indent=1,
        )
    )
    print(f"wrote out/holdout.json ({len(packed)} rows) and out/prep.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Turn a production agent's traces into a boring/hard split and an SFT file. "
            "Offline: no model key, no network."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build a small slice, print the split, write nothing",
    )
    main(dry_run=parser.parse_args().dry_run)
