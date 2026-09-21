"""``simulate(seed=)`` reproduces the task set at any concurrency by default (#645).

Before 0.111 the default was ``reproducible=False``: rows landed in completion
order, and which situations were written next depended on which rollouts had
finished, so one seed at the default concurrency drew a different task set on
every machine (59, 58 and 58 tasks from one ``budget=160`` in the measurement
that changed the default). A lesson that wanted the same file everywhere had to
pin ``concurrency=1``. Now ``reproducible=None`` resolves to ``True`` unless a
clock is set, since a clock stop lands wherever the run happens to be.

The agent here sleeps a random few milliseconds so that only scheduling differs
between the two runs; the answer echoes the prompt so every rollout is usable.
"""

from __future__ import annotations

import random
import time

import whileai as wai
from whileai.simulations.run.config import resolve_run_config as make


def _agent(message: str) -> str:
    time.sleep(random.random() * 0.01)
    return "ok: " + str(message)[:40]


def _task_ids(**kw) -> list[str]:
    data = wai.simulate(_agent, seed=0, simulator=False, phrasings=2, budget=48, **kw)
    return [str(r.get("scenario_id", r.get("prompt", ""))) for r in data.rows()]


def test_default_gives_the_same_task_set_twice_at_the_default_concurrency() -> None:
    """The contract is same seed, same concurrency, same agent: same rows.
    Concurrency is the batch size, so it stays in the contract; what no
    longer varies is thread timing, which is all that differs here."""
    first = _task_ids()
    second = _task_ids()
    assert len(first) == 48
    assert first == second, "same ids in the same order, so a positional cut never splits an ask"
    assert _task_ids(reproducible=True) == first


def test_default_resolves_to_true_without_a_clock_and_false_with_one() -> None:
    assert make(agent=_agent).reproducible is True
    assert make(agent=_agent, time_budget=5).reproducible is False
    assert make(agent=_agent, reproducible=False).reproducible is False
    assert make(agent=_agent, reproducible=True, time_budget=5).reproducible is True
