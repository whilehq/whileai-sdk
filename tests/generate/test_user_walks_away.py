"""The simulated person can give up on the agent's question (#289).

Before this, ``_want_followup`` answered every question until the depth cap
and the user-sim prompt said to always answer, so whether a thread ended was
decided by the turn budget and never by what the agent said. No rubric
criterion about asking could fail.
"""

from __future__ import annotations

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS
from whileai.simulations.generate.agents import (
    USER_LEFT,
    _user_walks_away,
    _want_followup,
    local_model,
    user_sim_system,
)

ASK = "Which order is it, and what was the amount?"


def test_first_question_is_always_tried_and_later_ones_can_lose_the_person():
    msgs = [f"msg-{i}" for i in range(400)]
    # The first question always gets an attempt, at every level.
    for level in ("short", "normal", "endless"):
        assert all(
            _want_followup(m, 1, user_turns=1, budget=12, agent_text=ASK, patience=level)
            for m in msgs
        )
    # From the second question on, the default walks away at a real rate.
    second = sum(
        not _want_followup(m, 3, user_turns=2, budget=12, agent_text=ASK, questions=1) for m in msgs
    )
    third = sum(
        not _want_followup(m, 5, user_turns=3, budget=12, agent_text=ASK, questions=2) for m in msgs
    )
    assert 100 <= second <= 180, second  # 35% of 400
    assert 200 <= third <= 280, third  # 60% of 400
    short = sum(
        not _want_followup(
            m, 3, user_turns=2, budget=12, agent_text=ASK, questions=1, patience="short"
        )
        for m in msgs
    )
    assert short > second
    # "endless" is the old behaviour: every question is answered to the cap.
    assert all(
        _want_followup(
            m, 5, user_turns=3, budget=12, agent_text=ASK, questions=4, patience="endless"
        )
        for m in msgs
    )
    # The first question never loses the person, whatever the level.
    assert not any(_user_walks_away(m, 3, questions=0, patience="short") for m in msgs)
    # The hazard is pure patience: it knows nothing about the budget. The
    # depth cap is the loop's rule, applied before the hazard is consulted.
    assert any(_user_walks_away(m, 9, questions=3, patience="short") for m in msgs)
    assert not any(
        _want_followup(m, 9, user_turns=6, budget=12, agent_text=ASK, questions=3, patience="short")
        for m in msgs
    )
    # Same message and turn, same draw: a seeded run reproduces.
    assert _want_followup("m", 3, user_turns=2, budget=12, agent_text=ASK, questions=1) == (
        _want_followup("m", 3, user_turns=2, budget=12, agent_text=ASK, questions=1)
    )


ANSWERS = [
    "it was order ORD-7, twenty dollars",
    "the receipt says ORD-7 and forty two dollars",
    "order ORD-7, I paid nineteen fifty for it",
    "ORD-7 for the blue lamp, thirty dollars",
    "the number is ORD-7, total came to sixty",
    "ORD-7 again, the amount was eighty five",
]


QUESTIONS = [
    ASK,
    "Was it the blue lamp or the desk?",
    "What date did it arrive?",
    "Do you have the receipt number?",
    "Which card did you pay with?",
    "Should I refund to the same card?",
    "Anything else I should note on the order?",
]


def _asking_agent():
    """The agent asks a new question every turn; the person keeps answering
    with a fresh detail, so only patience can end the thread early."""

    def fake_complete(_url, _model, messages, **kwargs):
        if kwargs.get("tools"):
            asked = sum(1 for m in messages if m.get("role") == "assistant")
            return {"content": QUESTIONS[asked % len(QUESTIONS)]}
        # The user-sim sees the whole thread in its one user message; the
        # number of questions in it says which answer comes next.
        body = str(messages[-1].get("content", ""))
        answered = sum(1 for q in QUESTIONS if q in body)
        return {"content": ANSWERS[answered % len(ANSWERS)]}

    return fake_complete


def _threads(monkeypatch, fake_complete, n: int = 40, **kw) -> list[dict]:
    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 12
    )
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=12, **kw)
    return [agent(f"refund order ORD-{i}, it arrived broken") for i in range(n)]


def test_default_patience_ends_some_threads_on_the_agents_question(monkeypatch):
    rows = _threads(monkeypatch, _asking_agent())
    left = [r for r in rows if r.get("ended_by") == "user_left"]
    assert 5 <= len(left) < len(rows), len(left)
    for row in left:
        # The row ends on the question nobody came back to.
        assert row["final_text"] in QUESTIONS
        assert row["steps"][-1].get("text") == row["final_text"]
        # ...after the person answered at least once.
        assert sum(1 for s in row["steps"] if s.get("user")) >= 1
    # The rest ran to the depth cap and are not labelled as the person leaving.
    assert all("ended_by" not in r for r in rows if r not in left)


def test_endless_patience_never_walks_away(monkeypatch):
    rows = _threads(monkeypatch, _asking_agent(), patience="endless")
    assert rows and not any(r.get("ended_by") for r in rows)
    # Threads run to the depth cap (6 user turns at budget 12) or the turn
    # cap; nothing shorter, because the person never leaves.
    depths = [sum(1 for s in r["steps"] if s.get("user")) for r in rows]
    assert max(depths) >= 5 and min(depths) >= 4, depths


def test_user_sim_may_decline_and_the_parser_ends_the_thread(monkeypatch):
    seen: list[str] = []

    def fake_complete(_url, _model, messages, **kwargs):
        if kwargs.get("tools"):
            return {"content": "What is the internal SKU reference for that item?"}
        seen.append(messages[0]["content"])
        return {"content": "<think>I would not know that.</think>\n[leaves]"}

    rows = _threads(monkeypatch, fake_complete, n=3)
    assert all(r.get("ended_by") == "user_left" for r in rows)
    assert all(r["final_text"].endswith("?") for r in rows)
    assert all(not any(s.get("user") for s in r["steps"]) for r in rows)
    assert seen and all("[leaves]" in prompt for prompt in seen)
    # With patience="endless" the prompt does not offer the way out, and a
    # leave mark is not an answer: the writer is retried and the row is a
    # follow-up miss, not a person leaving.
    seen.clear()
    rows = _threads(monkeypatch, fake_complete, n=3, patience="endless")
    assert seen and all("[leaves]" not in prompt for prompt in seen)
    assert not any(r.get("ended_by") for r in rows)


def test_user_sim_prompt_offers_the_way_out_only_when_asked():
    assert USER_LEFT in user_sim_system(TOOLS, may_leave=True)
    assert USER_LEFT not in user_sim_system(TOOLS)
    assert "answer it with a concrete detail" in user_sim_system(TOOLS, may_leave=True)


def test_simulate_patience_reaches_the_runner_and_the_run_counts_rows(monkeypatch):
    levels: list[str] = []

    def fake_local(url, model, **kwargs):
        levels.append(kwargs.get("patience"))

        def agent(message):
            return {
                "steps": [{"text": ASK}],
                "final_text": ASK,
                "ended_by": "user_left",
            }

        return agent

    monkeypatch.setattr("whileai.simulations.run.engine.local_model", fake_local)
    common = dict(
        tools=TOOLS,
        policy=POLICY,
        backend="vllm:fake@http://127.0.0.1:9",
        budget=3,
        repeats=1,
        grade=False,
        concurrency=2,
        simulator=False,
        seed=0,
        time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    data = wai.simulate(**common)
    assert levels == ["normal"]
    n = len(data.trajectories)
    assert n and all(r["ended_by"] == "user_left" for r in data.trajectories)
    counted = data.search["ended_on_question"]
    assert counted == {"share": 1.0, "n": n, "user_left": n}
    wai.simulate(patience="endless", **common)
    assert levels == ["normal", "endless"]
    try:
        wai.simulate(patience="forever", **common)
    except ValueError as exc:
        assert "'endless'" in str(exc)
    else:
        raise AssertionError("an unknown patience level must be refused, not ignored")
