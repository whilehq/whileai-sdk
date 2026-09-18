"""#285: the search loop aims at graded failures, not only at tool faults.

``mutation_worthy`` fills the mutation parents from sandbox faults and
ignores any score column, so a grader that fails reply-form rules (an
unsupported claim, two questions where one was asked for) never steered
the loop. With ``grader=`` set, a row the grader fails is now a mutation
parent too, and ``search["mutation_aims"]`` says which aim produced what."""

from __future__ import annotations

from tests.helpers import POLICY, TOOLS, scripted_agent, simulate_offline
from whileai.simulations.generate.generator import ModelSimulator


def _clean_agent(message: str) -> dict:
    """Every tool call succeeds and the reply is fine by the world's
    lights; it just never asks the one question the rule wants."""
    return {
        "steps": [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": "ord_1"},
                "result": {"status": "ok", "order_id": "ord_1"},
            }
        ],
        "final_text": "Found it. Refund on the way and I have also emailed you a receipt.",
    }


def _one_question(row: dict) -> dict:
    """A reply-form rule: the agent has to ask exactly one question."""
    asked = str(row.get("final_text") or "").count("?")
    return {"reward": int(asked == 1), "reason": f"one_question: asked {asked}, wanted 1"}


def _run(agent, **kw):
    kw.setdefault("budget", 12)
    kw.setdefault("per_round", 4)
    # fault_rate=0: no scheduled faults, so the only failures are graded ones
    return simulate_offline(agent, fault_rate=0, concurrency=1, seed=1, **kw)


def test_graded_failures_become_mutation_parents_when_a_grader_runs():
    data = _run(_clean_agent, grader=_one_question, mutate_failures=True)
    rows = data.trajectories
    assert rows and all(r["reward"] == 0 for r in rows)
    assert not any(r.get("faults") for r in rows), "the world raised no fault"
    aims = data.search["mutation_aims"]
    assert aims["world_fault"] == {"parents": 0, "rows": 0}
    assert aims["graded_failure"]["parents"] > 0
    # and the loop acted on them: mutated asks were generated from graded parents
    mutated = [r for r in rows if r["arm"] == "failure_mutation"]
    assert mutated and aims["graded_failure"]["rows"] == len(mutated)
    assert all(r["parent_failure_id"] for r in mutated)


def test_the_knob_turns_it_off_and_nothing_else_steers():
    data = _run(
        _clean_agent, grader=_one_question, mutate_failures=True, mutate_graded_failures=False
    )
    assert data.trajectories and all(r["reward"] == 0 for r in data.trajectories)
    assert data.search["mutation_aims"]["graded_failure"] == {"parents": 0, "rows": 0}
    assert not [r for r in data.trajectories if r["arm"] == "failure_mutation"]


def test_without_a_grader_only_world_faults_steer_as_before():
    # scripted_agent returns not_found on some asks: world faults, no grader
    data = _run(scripted_agent, mutate_failures=True, budget=16, per_round=8)
    aims = data.search["mutation_aims"]
    assert aims["graded_failure"] == {"parents": 0, "rows": 0}
    assert aims["world_fault"]["parents"] > 0


def test_a_faulted_row_is_counted_once_as_a_world_fault():
    def faulted_and_failed(message: str) -> dict:
        return {
            "steps": [
                {
                    "tool": "lookup_order",
                    "arguments": {"order_id": "x"},
                    "result": {"status": "error"},
                }
            ],
            "final_text": "All good, refund sent.",
        }

    data = _run(faulted_and_failed, grader=_one_question, mutate_failures=True)
    aims = data.search["mutation_aims"]
    assert aims["world_fault"]["parents"] > 0
    assert aims["graded_failure"]["parents"] == 0


def test_retry_card_names_what_the_grader_found_wrong():
    gen = ModelSimulator(tools=TOOLS, policy=POLICY)
    detail = gen._parent_detail(
        {
            "scenario_id": "s1",
            "prompt": "where is my refund",
            "reward": 0,
            "grader_reason": "one_question: asked 0, wanted 1",
        }
    )
    assert detail["failed"] == "one_question: asked 0, wanted 1"
    # an ungraded parent says nothing about a verdict
    assert "failed" not in gen._parent_detail({"scenario_id": "s2", "prompt": "hi"})
