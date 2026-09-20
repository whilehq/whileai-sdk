"""Bring your own agent: the callable contract, what a broken one looks like,
and keeping an eval score out of the training reward. Offline, no key.

    python run.py                 # all three parts
    python run.py --part contract
    python run.py --part broken
    python run.py --part eval

The engine calls ``agent(message)`` once per rollout and expects
``{"steps": [...], "final_text": str}`` back. Part one is a working agent.
Part two shows what the run reports when the agent raises or returns the
wrong shape (``stopped_because == "agent_failed"`` with the first error, not
a silent zero-row run). Part three grades the same rows twice, once as a
training grade and once as a held-out eval, and shows the selectors
counting the eval rows so they are not trained on by accident.
"""

from __future__ import annotations

import argparse
import os
import sys

import whileai.simulations as wai
from whileai.config import provenance
from whileai.simulations.score.judging import evaluate, run_judge

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_runbook",
            "description": "Fetch the runbook entry for a service.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restart a service.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    },
]
POLICY = "You are an on-call ops assistant. Read the runbook before restarting anything."


# ----------------------------------------------------------------- part one
def ops_agent(message: str) -> dict:
    """The contract: one string in, steps and a final reply out.

    ``message`` is the situation the simulator wrote (a user turn).
    ``steps`` is the ordered list of tool calls the agent made, each with
    the arguments it sent and the result it got back. ``final_text`` is
    what the agent said at the end. Nothing else is required; extra keys
    are kept on the row.
    """
    from whileai.simulations.generate.agents import current_rollout

    service = "checkout-api" if "checkout" in message.lower() else "search-api"
    # Every third repeat this agent skips the runbook: a behavior a judge
    # can catch, and a spread of rewards the eval part below needs.
    careless = (current_rollout.rollout_index or 0) % 3 == 2
    steps = []
    if not careless:
        steps.append(
            {
                "tool": "get_runbook",
                "arguments": {"service": service},
                "result": {"status": "ok", "restart_allowed": True},
            }
        )
    steps.append(
        {
            "tool": "restart_service",
            "arguments": {"service": service},
            "result": {"status": "ok"},
        }
    )
    return {"steps": steps, "final_text": f"Restarted {service}."}


def part_contract() -> wai.SimulationData:
    print("== contract: a working callable")
    data = wai.simulate(
        ops_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,  # template writer, no model key needed
        budget=16,
        mode="rl",
        situations=4,
        repeats=4,
        # mode="rl" defaults to repeat_policy="successive": two probe
        # rollouts per ask, the rest only where the graded probes split.
        # This run is not graded, so without "fixed" no ask ever reaches
        # its third repeat and the careless branch above never fires.
        repeat_policy="fixed",
        seed=0,
    )
    print(f"rows={len(data.trajectories)} stopped_because={data.stopped_because!r}")
    row = data.trajectories[0]
    print(f"first row: {len(row['steps'])} step(s), final_text={row['final_text']!r}")
    careless = sum(1 for r in data.trajectories if len(r["steps"]) == 1)
    print(f"{careless} of {len(data.trajectories)} rows skipped the runbook (rollout_index 2)")
    return data


# ----------------------------------------------------------------- part two
def agent_that_raises(message: str) -> dict:
    raise RuntimeError("model endpoint returned 502")


def agent_wrong_shape(message: str) -> dict:
    # An OpenAI-style message is not the contract.
    return {"role": "assistant", "content": "Restarted."}


def part_broken() -> None:
    print("== broken: what the run says when the agent is the problem")
    for name, fn in (("raises", agent_that_raises), ("wrong shape", agent_wrong_shape)):
        data = wai.simulate(
            fn, tools=TOOLS, system_prompt=POLICY, simulator=False, budget=8, seed=0
        )
        print(
            f"{name:12s} rows={len(data.trajectories)} "
            f"stopped_because={data.stopped_because!r} "
            f"agent_errors={data.search.get('agent_errors')}"
        )
        print(f"{'':12s} first error: {data.search.get('first_agent_error')}")
    print("A zero-row run names the agent, not the writer; the first error says which bug.")


# --------------------------------------------------------------- part three
def runbook_judge(row: dict) -> int:
    """1 when the agent read the runbook before restarting."""
    tools = [s.get("tool") for s in row.get("steps") or []]
    return 1 if "get_runbook" in tools else 0


def part_eval(data: wai.SimulationData) -> None:
    print("== eval: a held-out score is not a reward")
    train = run_judge(data.trajectories, runbook_judge)  # lineage.source == "grade"
    held_out = evaluate(data.trajectories, runbook_judge)  # lineage.source == "eval"
    print(f"pass@1 on both: {train.pass_at.pass_at_1:.2f}")
    for label, scored in (("run_judge", train), ("evaluate ", held_out)):
        kept, report = scored.select_for_rl()
        flag = report["hygiene_warnings"][-1] if report["eval_sourced"] else "clean"
        print(f"{label}: selected={len(kept)} eval_sourced={report['eval_sourced']}  {flag[:72]}")
    print("Same rows, same judge, same numbers. The count is the only thing that differs,")
    print("and it is the thing that stops the eval scorer becoming the reward model.")


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=["contract", "broken", "eval", "all"], default="all")
    args = parser.parse_args(argv)
    for key in ("OPENAI_API_KEY", "VLLM_API_KEY"):
        os.environ.pop(key, None)  # this example is offline on purpose
    data = None
    if args.part in ("contract", "eval", "all"):
        data = part_contract()
    if args.part in ("broken", "all"):
        if data is not None:
            print()
        part_broken()
    if args.part in ("eval", "all"):
        print()
        assert data is not None
        part_eval(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
