"""wai.methods.Swarm: the loop, the topologies, the result and the calibration check."""

from __future__ import annotations

import hashlib
import re

import pytest

import whileai as wai
from whileai.swarm import Calibration, Swarm, SwarmResult, neighbours


def _chat(messages, seed):
    """Wrong on a bare ask; right more often once it has seen feedback or a teammate."""
    user = messages[-1]["content"]
    h = int(hashlib.sha256(f"{seed}:{user[:400]}".encode()).hexdigest(), 16)
    p = 0.05 + 0.3 * ("What failed" in user) + 0.4 * ("teammate" in user)
    return "print(42)" if (h % 1000) / 1000 < p else "print(41)"


def _fitness(text):
    m = re.search(r"print\((\d+)\)", text)
    got = int(m.group(1)) if m else 0
    return {"fitness": 1 - abs(42 - got) / 42, "correct": got == 42, "feedback": f"printed {got}"}


def test_reachable_one_dot_down():
    assert wai.methods.Swarm is Swarm


def test_neighbours_by_topology():
    assert neighbours("solo", 3, 8) == []
    assert neighbours("ring", 0, 8) == [7, 1]
    assert neighbours("star", 2, 4) == [0, 1, 3]
    with pytest.raises(ValueError, match="topology must be"):
        neighbours("mesh", 0, 8)


def test_bad_configuration_is_refused_on_construction():
    with pytest.raises(ValueError):
        Swarm(_chat, topology="mesh")
    with pytest.raises(ValueError, match="at least 2"):
        Swarm(_chat, particles=1)
    with pytest.raises(ValueError, match="at least 1"):
        Swarm(_chat, rounds=0)
    with pytest.raises(TypeError, match="backend"):
        Swarm(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("topology", ["solo", "ring", "star"])
def test_run_returns_a_result_that_prints(topology):
    out = Swarm(_chat, topology=topology, particles=4, rounds=3)("print 42", _fitness, seed=1)
    assert isinstance(out, SwarmResult)
    assert {s["round"] for s in out.samples} <= {0, 1, 2}
    assert all(set(s["grade"]) == {"fitness", "correct", "feedback"} for s in out.samples)
    if out.rescued:
        assert out.best["grade"]["correct"] and out.round is not None
    assert f"{topology} swarm, 4 particles x 3 rounds" in str(out)


def test_stop_on_correct_ends_the_run():
    def always(messages, seed):
        return "print(42)"

    out = Swarm(always, particles=3, rounds=3)("print 42", _fitness)
    assert out.rescued and out.round == 0 and len(out.samples) == 3
    full = Swarm(always, particles=3, rounds=3, stop_on_correct=False)("print 42", _fitness)
    assert len(full.samples) == 9


def test_a_bare_number_is_a_fitness_with_no_target():
    out = Swarm(_chat, particles=2, rounds=2)("x", lambda t: 0.5)
    assert not out.rescued and out.best["grade"]["fitness"] == 0.5


def test_a_bad_fitness_says_what_it_must_return():
    with pytest.raises(TypeError, match="fitness must return"):
        Swarm(_chat, particles=2, rounds=1)("x", lambda t: "yes")


def test_refinement_prompt_carries_feedback_and_a_teammate_only_when_sharing():
    seen: list[str] = []

    def spy(messages, seed):
        seen.append(messages[-1]["content"])
        return "print(41)"

    Swarm(spy, topology="solo", particles=2, rounds=2)("x", _fitness)
    assert any("What failed" in u for u in seen) and not any("teammate" in u for u in seen)
    seen.clear()
    Swarm(spy, topology="star", particles=2, rounds=2)("x", _fitness)
    assert any("teammate" in u for u in seen)


def test_calibration_names_a_cliff_and_a_hill():
    cliff = [{"fitness": f, "correct": f == 1.0} for f in (0, 0.2, 0.6, 0.8, 1.0, 1.0)]
    cal = Swarm.calibration(cliff)
    assert isinstance(cal, Calibration) and not cal.partial_signal
    assert "a cliff, not a hill" in str(cal) and cal.n == 6
    hill = [
        {"grade": {"fitness": 0.8, "correct": True}},
        {"grade": {"fitness": 0.2, "correct": False}},
    ]
    assert Swarm.calibration(hill).partial_signal
    labels = [r[0] for r in cal.rows]
    assert labels[0] == "0" and labels[-1] == "1.0"


def test_a_backend_is_called_through_complete(monkeypatch):
    calls = []

    def fake_complete(base_url, model, messages, **kw):
        calls.append((base_url, model, kw.get("extra"), kw.get("temperature")))
        return {"content": "print(42)"}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    swarm = Swarm(
        wai.Endpoint("Qwen/Qwen3-4B", url="http://localhost:8000/v1"), particles=2, rounds=1
    )
    out = swarm("print 42", _fitness, seed=3)
    assert out.rescued
    assert calls[0][0] == "http://localhost:8000/v1" and calls[0][1] == "Qwen/Qwen3-4B"
    assert calls[0][2] == {"seed": 3000} and calls[0][3] == 1.0
    assert repr(swarm) == "Swarm(topology='ring', particles=2, rounds=1)"
