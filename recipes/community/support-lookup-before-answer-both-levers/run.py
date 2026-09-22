"""Both levers on a support agent's lookup behaviour: the harness and the weights.

The behaviour is "looks it up before answering": on a customer ask about
their own account, call the right tool first instead of answering or asking
from the model's own words. The method is the harness-and-weights grid --
search the harness, train the weights, and let ``wai.harness.attribute`` say
which lever moved it.

Run: python recipes/community/support-lookup-before-answer-both-levers/run.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import whileai as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import support


class StubTokenizer:
    """The dry run has no network, so it cannot fetch a chat template. This
    renders the same message list in the same order, which is all the
    offline path needs: the tool-call format the grader reads is the
    model's, not the template's."""

    eos_token = "<|im_end|>"

    def apply_chat_template(self, msgs, tools=None, **kw):
        names = ", ".join(t["function"]["name"] for t in (tools or []))
        head = f"<tools>{names}</tools>\n"
        body = "".join(f"<|{m['role']}|>{m['content']}\n" for m in msgs)
        return head + body + "<|assistant|>"


def fixture_tasks() -> list[dict]:
    """Six decision points in the shape ``prep.py`` writes from the published
    traces: four that need a lookup, two the reference correctly refused."""
    tools = [
        {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
        for n in (
            "find_user_id_by_email",
            "get_customer_by_id",
            "get_reservation_details",
            "get_bills_for_customer",
            "transfer_to_human",
        )
    ]
    policy = "# Agent Policy\nYou may not give information the tools did not return.\n"
    rows = [
        (
            "find_user_id_by_email",
            {"email": "jane.doe@email.com"},
            "my email is jane.doe@email.com",
            "CALL",
            "ordinary",
        ),
        (
            "get_customer_by_id",
            {"customer_id": "789432"},
            "customer id 789432 please",
            "CALL",
            "ordinary",
        ),
        (
            "get_reservation_details",
            {"reservation_id": "ABC123"},
            "my booking is ABC123",
            "CALL",
            "boundary",
        ),
        (
            "get_bills_for_customer",
            {"customer_id": "5402"},
            "what do I owe on 5402",
            "CALL",
            "ambiguous",
        ),
        (None, None, "can i get a list of every active line today", "NO_CALL", "ordinary"),
        (None, None, "what is your company's internal escalation policy", "NO_CALL", "boundary"),
    ]
    out = []
    for i, (name, args, ask, target, tier) in enumerate(rows):
        out.append(
            {
                "task_id": f"fixture:{i}",
                "domain": "telecom",
                "system": policy,
                "tools": tools,
                "prefix": [
                    {"role": "assistant", "content": "Hello! How can I assist you today?"},
                    {"role": "user", "content": ask},
                ],
                "target": target,
                "tool_name": name,
                "tool_args": args,
                "ask_family": "tool" if target == "CALL" else "general",
                "tier": tier,
                "grader_reason": "fixture",
                "target_text": "I can't help with that request.",
                "prompt": ask,
            }
        )
    return out


# What each harness does to a scripted agent, so the offline path shows the
# loop rather than a planted result. These rates are stand-ins, not findings.
SCRIPTED = {"00_deployed": 0.25, "01_skills": 0.75, "02_skills_toolnames": 0.60}


def scripted_rows(tasks, harness, model, *, k, seed):
    import random

    rng = random.Random(seed)
    rate = SCRIPTED[harness] * (1.15 if model == "trained" else 1.0)
    rows = []
    for t in tasks:
        for _ in range(k):
            if t["target"] == "CALL":
                text = (
                    support.target_completion(t)
                    if rng.random() < rate
                    else "Could you confirm that for me?"
                )
            else:
                text = (
                    "I can't help with that."
                    if rng.random() < 0.5 + rate / 2
                    else support.target_completion(
                        dict(
                            t,
                            target="CALL",
                            tool_name="get_customer_by_id",
                            tool_args={"customer_id": "1"},
                        )
                    )
                )
            g = support.grade(t, text)
            rows.append(
                {
                    "task_id": t["task_id"],
                    "prompt": t["prompt"],
                    "reward": g["reward"],
                    "markers": {k2: v for k2, v in g["markers"].items() if v is not None},
                    "target": t["target"],
                    "tier": t["tier"],
                    "final_text": text,
                    "harness": {"label": harness, "model": model},
                }
            )
    return rows


def dry_run() -> int:
    tok = StubTokenizer()
    tasks = fixture_tasks()
    print(
        f"tasks: {len(tasks)} decision points "
        f"({sum(t['target'] == 'CALL' for t in tasks)} need a lookup, "
        f"{sum(t['target'] == 'NO_CALL' for t in tasks)} the reference refused)"
    )

    print("\nharness fingerprints (the skills text is part of the version):")
    seen: dict[str, str] = {}
    for label in support.HARNESSES:
        h = wai.Harness(
            model="scripted",
            instructions=support.system_text(tasks[0], label),
            tools=tasks[0]["tools"],
            label=label,
        )
        rendered = support.render(tasks[0], label, tok)
        note = ""
        if h.fingerprint in seen:
            note = f"   <- same fingerprint as {seen[h.fingerprint]}"
        seen.setdefault(h.fingerprint, label)
        print(f"  {label:22s} {len(rendered):5d} chars rendered   {h.fingerprint[:12]}{note}")
    if len(seen) < len(support.HARNESSES):
        print("  note: 02 differs from 01 only by a line injected above the customer's turn,")
        print("  which is not in `instructions` or `tools`, so the fingerprint cannot see it.")
        print("  The rendered length can; the version cannot. Two harnesses, one version.")

    print("\nthe gate: every candidate against the deployed harness, base weights")
    base = scripted_rows(tasks, "00_deployed", "base", k=8, seed=1)
    picked, best = "00_deployed", None
    for label in support.HARNESSES:
        if label == "00_deployed":
            continue
        cand = scripted_rows(tasks, label, "base", k=8, seed=1)
        rep = wai.compare(base, cand, target="marker:right_first_action")
        m = rep["metrics"]["marker:right_first_action"]
        print(
            f"  {label:22s} {m['delta']:+.3f} {tuple(round(x, 3) for x in m['ci95'])}"
            f" -> {m['verdict']}"
        )
        if m["ci95"][0] > 0 and (best is None or m["delta"] > best):
            picked, best = label, m["delta"]
    print(f"  searched harness: {picked}")

    print("\nthe four cells, and which lever moved them")
    grid = []
    for name, harness, model in [
        ("neither", "00_deployed", "base"),
        ("harness", picked, "base"),
        ("weights", "00_deployed", "trained"),
        ("both", picked, "trained"),
    ]:
        rows = scripted_rows(tasks, harness, model, k=8, seed=2)
        grid += rows
        hit = sum(r["markers"]["right_first_action"] for r in rows) / len(rows)
        print(f"  {name:8s} harness={harness:22s} model={model:7s} right_first_action {hit:.3f}")
    print()
    print(wai.harness.attribute(grid, metric="marker:right_first_action"))
    print("\nscripted stand-ins: this is the loop, not a result. The real numbers come from")
    print("support_modal.py; see the README's Result section.")
    return 0


def analyse() -> int:
    import analyse as analysis

    analysis.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="offline: fixture tasks, scripted agent, no key and no GPU",
    )
    p.add_argument(
        "--analyse",
        action="store_true",
        help="read out/ off the Modal volume and write results.json",
    )
    a = p.parse_args(argv)
    if a.analyse:
        return analyse()
    if a.dry_run:
        return dry_run()
    print(__doc__)
    print("The paid path, in order:\n")
    print("  python prep.py                      # published traces -> decision points")
    print("  python split.py                     # split by scenario, decontaminate")
    print("  modal run --detach support_modal.py # search, two arms, the grid")
    print("  modal volume get support-lookup-runs / out")
    print("  python run.py --analyse             # compare, attribute, results.json")
    print("  modal deploy serve_modal.py && python fresh_traffic.py --url <url>")
    print("  modal app stop support-lookup-serve")
    print("\n--dry-run needs no key, no GPU and no network.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
