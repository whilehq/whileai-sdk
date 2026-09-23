"""Particle swarm optimization over rollouts, and the check to run first.

    import whileai as wai

    def fitness(text: str) -> dict:
        # graded 0..1 score the swarm climbs, the real target, feedback it reads
        return {"fitness": 0.5, "correct": False, "feedback": "failed test 3"}

    swarm = wai.methods.Swarm(wai.Endpoint(url="http://localhost:8000/v1", model="Qwen/Qwen3-4B"))
    result = swarm("Write is_prime(n) ...", fitness)
    print(result)          # rescued in round 1 of 3, best fitness 1.00, 12 samples
    print(wai.methods.Swarm.calibration(result.samples))   # is the fitness a hill

Kennedy and Eberhart 1995 in prose. A particle is one attempt; its
position is the text it wrote; its personal best is its highest-fitness
attempt. Round 0 is ``particles`` fresh samples. Each later round a
particle sees its own best with the fitness's feedback and, by topology,
a neighbour's best, and writes a new attempt. The model is the velocity
update. ``solo`` shows no neighbour (the social term off), ``ring`` the
better of two ring neighbours (lbest), ``star`` the best in the swarm
(gbest); Kennedy and Mendes 2002 on the contrast.

Read this before trusting it. On every task family in
``recipes/01-simulate/swarm-rescue`` the swarm tied plain resampling at
the same budget: partial credit on tests predicts a pass 0 times below
three quarters of the tests, a model judge ranks passes at chance, and on
a task built so fitness is a real hill the swarm doubled the programs that
pass the shown tests while the hidden pass did not follow. A swarm needs
a fitness that is graded, predicts the target, and cannot be met by
rewriting to the example. ``Swarm.calibration`` measures the first two on
any graded rows; run it on your fitness before a swarm.
"""

from __future__ import annotations

import concurrent.futures as cf
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .models import Backend

#: Particles per round. Eight is the population the swarm-rescue recipe ran
#: at, the same count as the GRPO group whose all-fail prompts it rescues.
PARTICLES = 8
#: Rounds including round 0. Three gives two refinement rounds; in the
#: recipe most rescues in every arm came from round 0, so more buys little.
ROUNDS = 3
#: Sampling temperature. 1.0 keeps round 0 as diverse as plain resampling,
#: the diversity the recipe found was the resource that rescued tasks.
TEMPERATURE = 1.0
#: Reply budget. 2,048 tokens fits a program with thinking off; the recipe
#: saw 8% of replies at that cap on competition problems.
MAX_TOKENS = 2048
#: Where a fitness bucket starts, for ``calibration``. The top bucket is an
#: exact 1.0, the step every calibration curve in the recipe jumped at.
BUCKETS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
#: The fewest particles that can share anything; one particle is resampling
#: with feedback.
MIN_PARTICLES = 2
TOPOLOGIES = ("solo", "ring", "star")

SYSTEM = "Solve the problem. Think briefly, then give the whole answer."
REFINE = (
    "Write an improved answer. Keep what works, fix what fails, and if the approach is "
    "wrong change it."
)

Chat = Callable[[list[dict], int], str]
Fitness = Callable[[str], Any]


def neighbours(topology: str, i: int, n: int) -> list[int]:
    """Who particle ``i`` of ``n`` sees under ``topology``."""
    if topology == "solo":
        return []
    if topology == "ring":
        return [(i - 1) % n, (i + 1) % n]
    if topology == "star":
        return [j for j in range(n) if j != i]
    raise ValueError(f"topology must be one of {', '.join(TOPOLOGIES)}, not {topology!r}")


def _grade(fitness: Fitness, text: str) -> dict[str, Any]:
    """The fitness's verdict as ``{fitness, correct, feedback}``. A bare
    number is a fitness with no target (``correct`` stays False)."""
    out = fitness(text)
    if isinstance(out, (int, float)):
        return {"fitness": float(out), "correct": False, "feedback": ""}
    if not isinstance(out, Mapping) or "fitness" not in out:
        raise TypeError(
            "fitness must return a number or a mapping with 'fitness' (and optionally "
            f"'correct', 'feedback'); got {type(out).__name__}"
        )
    return {
        "fitness": float(out["fitness"]),
        "correct": bool(out.get("correct", False)),
        "feedback": str(out.get("feedback") or ""),
    }


def _chat_for(model: Backend | Chat, temperature: float, max_tokens: int) -> Chat:
    """A ``(messages, seed) -> text`` call from a backend object or a callable."""
    if callable(model) and not isinstance(model, Backend):
        return model
    if not isinstance(model, Backend):
        raise TypeError(
            "model: pass a backend (wai.Endpoint, wai.OpenAI, wai.Hosted, ...) or a callable "
            f"(messages, seed) -> str; got {type(model).__name__}"
        )
    from .simulations.generate.agents import complete, default_agent_spec, parse_backend_spec

    base_url, name = parse_backend_spec(model.spec or default_agent_spec())

    def chat(messages: list[dict], seed: int) -> str:
        reply = complete(
            base_url,
            name,
            messages,
            api_key=model.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra={"seed": seed},
        )
        return str(reply.get("content") or "")

    return chat


def _describe(label: str, attempt: Mapping[str, Any]) -> str:
    g = attempt["grade"]
    lines = [f"{label} (fitness {g['fitness']:.2f}):", str(attempt["text"])]
    if g["feedback"]:
        lines += ["What failed:", g["feedback"]]
    return "\n".join(lines)


@dataclass
class SwarmResult:
    """One swarm on one problem. ``samples`` holds every attempt as
    ``{round, particle, text, grade}``; ``best`` is the highest fitness, a
    correct attempt first. Prints itself."""

    topology: str
    particles: int
    rounds: int
    rescued: bool
    round: int | None
    best: dict[str, Any]
    samples: list[dict[str, Any]] = field(repr=False)

    def __str__(self) -> str:
        where = f"rescued in round {self.round}" if self.rescued else "not rescued"
        return (
            f"{self.topology} swarm, {self.particles} particles x {self.rounds} rounds: {where} "
            f"of {self.rounds}, best fitness {self.best['grade']['fitness']:.2f}, "
            f"{len(self.samples)} samples"
        )


@dataclass
class Calibration:
    """P(correct | fitness bucket) over graded attempts. ``rows`` pairs each
    bucket's label with (correct, count). A hill rises with the bucket; a
    cliff is zero until the top. Prints itself."""

    rows: list[tuple[str, int, int]]
    n: int

    @property
    def partial_signal(self) -> bool:
        """Any correct attempt below full fitness: the least a hill needs."""
        return any(c for label, c, _ in self.rows if label != "1.0")

    def __str__(self) -> str:
        lines = [f"P(correct | fitness), {self.n} attempts"]
        for label, c, n in self.rows:
            lines.append(f"  fitness {label:>9}: {c:5d}/{n:<6d} = {c / n:.3f}")
        if not self.partial_signal:
            lines.append(
                "  no correct attempt below full fitness: a cliff, not a hill; a swarm has "
                "nothing to climb on this fitness"
            )
        return "\n".join(lines)


class Swarm:
    """A particle swarm over rollouts of ``model``.

    ``model`` is a backend object or a callable ``(messages, seed) -> str``.
    ``topology`` is ``solo``, ``ring`` or ``star``. ``particles`` attempts
    a round for ``rounds`` rounds, round 0 fresh. ``stop_on_correct`` ends
    at the first correct attempt. ``system`` is the system prompt;
    ``temperature`` and ``max_tokens`` go to a backend. Call it with the
    problem text and a fitness; see the module docstring for the result
    the recipe got, and ``calibration`` for the check to run first.
    """

    def __init__(
        self,
        model: Backend | Chat,
        *,
        topology: str = "ring",
        particles: int = PARTICLES,
        rounds: int = ROUNDS,
        stop_on_correct: bool = True,
        system: str = SYSTEM,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ):
        neighbours(topology, 0, 2)
        if particles < MIN_PARTICLES:
            raise ValueError(f"particles={particles}: a swarm needs at least {MIN_PARTICLES}")
        if rounds < 1:
            raise ValueError(f"rounds={rounds}: at least 1 (round 0 is the fresh samples)")
        self.topology = topology
        self.particles = int(particles)
        self.rounds = int(rounds)
        self.stop_on_correct = stop_on_correct
        self.system = system
        self._chat = _chat_for(model, temperature, max_tokens)

    def __repr__(self) -> str:
        return (
            f"Swarm(topology={self.topology!r}, particles={self.particles}, rounds={self.rounds})"
        )

    def _sample(self, fitness: Fitness, messages: list[dict], seed: int) -> dict[str, Any]:
        text = self._chat(messages, seed)
        return {"text": text, "grade": _grade(fitness, text)}

    def _fan(self, fn: Callable[[int], dict[str, Any]]) -> list[dict[str, Any]]:
        with cf.ThreadPoolExecutor(max_workers=self.particles) as ex:
            return list(ex.map(fn, range(self.particles)))

    def __call__(self, problem: str, fitness: Fitness, *, seed: int = 0) -> SwarmResult:
        """Run the swarm on ``problem``. ``fitness(text)`` returns a number,
        or ``{fitness, correct, feedback}``: the graded score the swarm
        climbs, the target it never sees as a number, and the text a
        particle reads about its own attempt."""
        rng = random.Random(seed)
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": problem}]
        samples: list[dict[str, Any]] = []
        pbest = self._fan(lambda i: self._sample(fitness, msgs, seed * 1000 + i))
        for i, s in enumerate(pbest):
            samples.append({"round": 0, "particle": i, **s})
        stop = self.stop_on_correct and any(s["grade"]["correct"] for s in pbest)
        for r in range(1, self.rounds):
            if stop:
                break

            def step(i: int, r: int = r) -> dict[str, Any]:
                parts = [problem, "", _describe("Your best attempt so far", pbest[i])]
                nb = neighbours(self.topology, i, self.particles)
                if nb:
                    top = max(pbest[j]["grade"]["fitness"] for j in nb)
                    pick = rng.choice([j for j in nb if pbest[j]["grade"]["fitness"] == top])
                    parts += ["", _describe("Another attempt, from a teammate", pbest[pick])]
                parts += ["", REFINE]
                m = [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": "\n".join(parts)},
                ]
                return self._sample(fitness, m, seed * 1000 + r * self.particles + i)

            batch = self._fan(step)
            for i, s in enumerate(batch):
                samples.append({"round": r, "particle": i, **s})
                # a tie goes to the newer attempt, so the particle keeps moving
                if s["grade"]["fitness"] >= pbest[i]["grade"]["fitness"]:
                    pbest[i] = s
            stop = self.stop_on_correct and any(s["grade"]["correct"] for s in batch)
        hits = [s["round"] for s in samples if s["grade"]["correct"]]
        best = max(samples, key=lambda s: (s["grade"]["correct"], s["grade"]["fitness"]))
        return SwarmResult(
            topology=self.topology,
            particles=self.particles,
            rounds=self.rounds,
            rescued=bool(hits),
            round=min(hits) if hits else None,
            best=best,
            samples=samples,
        )

    @staticmethod
    def calibration(samples: Sequence[Mapping[str, Any]]) -> Calibration:
        """P(correct | fitness bucket) over graded attempts: rows carrying
        ``fitness`` and ``correct``, either at the top level or under
        ``grade`` (a ``SwarmResult.samples`` list works as is). Run it on
        base samples of your task before a swarm: a fitness whose partial
        buckets never hold a correct attempt is a cliff."""
        counts: dict[str, list[int]] = {}
        order: list[str] = []
        for s in samples:
            g = s.get("grade", s)
            f = float(g["fitness"])
            if f >= 1.0:
                label = "1.0"
            else:
                lo = max(b for b in BUCKETS if b <= f)
                label = "0" if f == 0 else f"{lo:.2f}+"
            if label not in counts:
                counts[label] = [0, 0]
                order.append(label)
            counts[label][0] += int(bool(g["correct"]))
            counts[label][1] += 1
        order.sort(key=lambda k: (k != "0", k == "1.0", k))
        rows = [(k, counts[k][0], counts[k][1]) for k in order]
        return Calibration(rows=rows, n=sum(n for _, _, n in rows))


__all__ = ["BUCKETS", "TOPOLOGIES", "Calibration", "Swarm", "SwarmResult", "neighbours"]
