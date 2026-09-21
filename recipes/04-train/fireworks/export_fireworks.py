"""Write what a Fireworks managed training job reads, and print the commands.

Offline end to end: the rows come from the stand-in agent, the grade is a
rule, and the two files land in ``out/``. Nothing here needs a key. The
``firectl`` lines printed at the end are the training compute and the
serving step; ``prove.py`` is the before/after through ``wai.Fireworks``.

    python export_fireworks.py --n 32 --account my-account --name refunds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import whileai as wai
from whileai.simulations import conversation
from whileai.simulations.export import export_preference

# The smallest Qwen3 in Fireworks' managed-training list (SFT, DPO and RFT
# all take it, and it serves on an on-demand deployment). The hosted-loop
# recipe trains the same base, so a reader can compare the two routes.
BASE_MODEL = "accounts/fireworks/models/qwen3-4b"

# Two rollouts per ask is the least that gives DPO a pass and a fail on the
# same prompt; four is what the dpo recipe uses when the policy is a model.
REPEATS = 2

POLICY = (
    "You help customers with orders. Look the order up before you promise "
    "anything. Refund only after a lookup shows the order is eligible."
)

RUBRIC = (
    "Pass when the agent looked the order up before promising or issuing a "
    "refund and its final answer matches what the lookup returned. Fail when "
    "it refunded without a lookup, invented an order, or contradicted the tool."
)


@wai.tool
def lookup_order(order_id: str) -> dict:
    """Look up an order: status, total and whether it is eligible for a refund."""
    return {"order_id": order_id, "status": "delivered", "total": 42.0, "eligible": True}


@wai.tool
def create_refund(order_id: str, amount: float) -> dict:
    """Issue a refund for an eligible order."""
    return {"order_id": order_id, "amount": amount, "refunded": True}


TOOLS = [lookup_order, create_refund]


def rule(row: dict) -> dict:
    """The stand-in grade: the seeded fault is the failure. Replace with
    ``wai.Judge(rubric=RUBRIC)`` or a verifier when the agent is a model."""
    return {"reward": int(not row.get("seeded"))}


def constructed_negative(row: dict) -> dict | None:
    """A rejected side for a pass: the failure the policy names, a refund
    with no lookup first. The stand-in agent seeds its faults per situation,
    so both rollouts of an ask agree and ``select_for_preference`` finds no
    on-policy contrast; with a model agent it pairs the agent's own passes
    and fails and this is not used (the dpo recipe calls the same fallback
    constructed negatives)."""
    messages = list(conversation(row))
    first = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), None)
    if first is None:
        return None
    bad_turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_no_lookup",
                "type": "function",
                "function": {
                    "name": "create_refund",
                    "arguments": json.dumps({"order_id": "ORD-0000", "amount": 42.0}),
                },
            }
        ],
    }
    return {**row, "messages": [*messages[:first], bad_turn], "reward": 0}


def preference_pairs(scored, selection, out: Path) -> tuple[dict, str]:
    """Write ``dpo.jsonl``: on-policy pairs when the run has a one-turn
    contrast, constructed negatives when it does not. Returns the export
    report and a line saying which."""
    pairs, report = scored.select_for_preference()
    kwargs = dict(system_prompt=selection.system_prompt, tools=selection.tools, format="fireworks")
    if pairs:
        dpo = export_preference(pairs, str(out / "dpo.jsonl"), validate=False, **kwargs)
        if dpo["pairs"]:
            return dpo, f"on-policy, {report['prompts_with_contrast']} prompts with a contrast"
    built = []
    for row in selection:
        negative = constructed_negative(row)
        if negative is not None:
            built.append({"prompt": row.get("prompt"), "chosen": row, "rejected": negative})
    dpo = export_preference(built, str(out / "dpo.jsonl"), **kwargs)
    return dpo, "constructed negatives (the run's own contrasts sit on tool results, not turns)"


def check_shapes(sft: Path, dpo: Path) -> dict:
    """The two files against the shapes docs.fireworks.ai/fine-tuning names.
    Pure Python, so the smoke run can hold the recipe to it offline."""
    sft_lines = [json.loads(s) for s in sft.read_text(encoding="utf-8").splitlines()]
    for line in sft_lines:
        assert set(line) <= {"messages", "tools"}, sorted(line)
        assert all(m["role"] in ("system", "user", "assistant", "tool") for m in line["messages"])
        assert all("weight" in m for m in line["messages"] if m["role"] == "assistant")
    dpo_lines = [json.loads(s) for s in dpo.read_text(encoding="utf-8").splitlines()]
    for line in dpo_lines:
        assert set(line) == {"input", "preferred_output", "non_preferred_output"}, sorted(line)
        assert len(line["preferred_output"]) == len(line["non_preferred_output"]) == 1
        assert line["preferred_output"][0]["role"] == "assistant"
    return {"sft_rows": len(sft_lines), "dpo_pairs": len(dpo_lines)}


def commands(account: str, name: str, base_model: str, out: Path) -> str:
    """The Fireworks side, with the ids filled in. Dataset, job, deployment,
    then the model id the served result answers to."""
    sft_model = f"{name}-sft"
    dpo_model = f"{name}-dpo"
    return "\n".join(
        [
            "# 1. upload the two datasets",
            f"firectl dataset create {name}-sft {(out / 'sft.jsonl').as_posix()}",
            f"firectl dataset create {name}-dpo {(out / 'dpo.jsonl').as_posix()}",
            "",
            "# 2. train on Fireworks GPUs: SFT from the passes, then DPO warm-started from it",
            f"firectl sftj create --base-model {base_model} --dataset {name}-sft --output-model {sft_model}",
            f"firectl dpo-job create --loss-method DPO --warm-start-from accounts/{account}/models/{sft_model} "
            f"--dataset accounts/{account}/datasets/{name}-dpo --output-model {dpo_model}",
            "firectl sftj get <JOB_ID>          # until State: COMPLETED",
            "",
            "# 3. serve the result (trained LoRAs deploy on-demand; live merge = no inference overhead)",
            f"firectl deployment create accounts/{account}/models/{dpo_model} --deployment-shape default",
            "",
            "# 4. prove it on held-out tasks, base against trained, through the same API",
            f"python prove.py --base {base_model} --tuned accounts/{account}/models/{dpo_model}",
        ]
    )


def main() -> None:
    from whileai.config import provenance

    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32, help="rollout budget for the stand-in run")
    ap.add_argument("--out", default="out", help="where sft.jsonl and dpo.jsonl land")
    ap.add_argument("--account", default="<ACCOUNT_ID>", help="your Fireworks account id")
    ap.add_argument("--name", default="refunds", help="dataset and model id prefix on Fireworks")
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = wai.simulate(
        wai.seeded_agent(TOOLS),  # or your agent: wai.Fireworks(args.base_model)
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,  # the offline situation writer; a model writer needs a key
        mode="rl",
        repeats=REPEATS,
        repeat_policy="fixed",
        budget=args.n,
    )
    scored = data.grade(judge=rule)

    selection = scored.select(
        mode="sft"
    )  # the passes, with the run's system prompt and tool schemas
    sft = selection.export(str(out / "sft.jsonl"), format="fireworks")
    dpo, pairs_from = preference_pairs(scored, selection, out)
    print(
        f"sft: {sft['n_written']} rows, mask {sft['mask_mode']}, "
        f"{sft['trained_messages']} assistant turns at weight 1"
    )
    print(
        f"dpo: {dpo['pairs']} pairs, {pairs_from}; "
        f"{dpo.get('fireworks_turns_cut', 0)} pairs lost turns after the first"
    )
    print("shapes:", check_shapes(out / "sft.jsonl", out / "dpo.jsonl"))
    print()
    print(commands(args.account, args.name, args.base_model, out))


if __name__ == "__main__":
    main()
