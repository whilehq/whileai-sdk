"""A harness sweep: many prompt, tool and model variants of one agent, scored
on the same frozen asks, one run per harness fingerprint on the platform.

Today a team changes a prompt by hand and scores it. This runs every
variant at once on the same held-out asks, grades them with the same
judge, measures the noise floor by scoring one variant twice, and posts
each as a run whose record pins the prompt label, the model and the
harness fingerprint, so the Runs page groups the dots by prompt or by
model and the verdict says which win is real. The winner is the variant
whose interval clears every other and the noise floor; anything closer
is "about the same", not a result.

rlhfbook.com, "Evaluation": scores move with the prompt and sampling
setup, not only the weights, so a comparison is only fair with the setup
held constant. Miller 2024 (arXiv 2411.00640): the interval is the result.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .platform import (
    Behavior,
    Data,
    EvalSetup,
    Harness,
    Judge,
    Provenance,
    RunRecord,
    Tracked,
    _pinned,
)

Agent = Callable[[str], dict[str, Any]]
Score = tuple[float, float, int]  # points, half-width of the 95% interval, asks
MIN_MARKER_ASKS = 3  # under this a marker has no interval and is left off the card
# Miller 2024 (arXiv 2411.00640): under ~50 items the interval is too wide to
# show a gain of a few points; the platform verdict says "unproven" below it.
MIN_ASKS = 50
LABEL_MAX = 40  # RunSpec.version max_length; tests pin the two together


def check_labels(harnesses: Sequence[Harness]) -> None:
    """Refuse before the first rollout what the platform would refuse after
    the last: a label over the run-version cap, or two variants with one
    label. A sweep is minutes of model calls, and the rows live in memory
    until they post."""
    seen: set[str] = set()
    for h in harnesses:
        label = h.version
        if len(label) > LABEL_MAX:
            raise ValueError(
                f"variant label {label!r} is {len(label)} characters; a run version holds "
                f"{LABEL_MAX}. Use the release tag or prompt name before '@' "
                f"(2026.09.2@claude-sonnet-5), not a sentence"
            )
        if label in seen:
            raise ValueError(f"two variants are labelled {label!r}; every variant needs its own")
        seen.add(label)


@dataclass
class VariantResult:
    """One harness variant after scoring."""

    label: str
    model: str | None
    fingerprint: str
    scores: dict[str, Score]
    run_id: str | None = None

    @property
    def headline(self) -> Score:
        return next(iter(self.scores.values()))


@dataclass
class SweepReport:
    """What a sweep found. ``print()`` it; ``best`` is the winner or None
    when no variant's interval clears the rest."""

    behavior: str
    test_version: str
    n_asks: int
    k: int
    noise_floor: float | None
    judge: Judge | None
    variants: list[VariantResult] = field(default_factory=list)

    @property
    def ranked(self) -> list[VariantResult]:
        return sorted(self.variants, key=lambda v: v.headline[0], reverse=True)

    def clears(self, a: VariantResult, b: VariantResult) -> bool:
        """``a`` beats ``b`` for real: the difference interval excludes zero
        and the difference clears the noise floor."""
        sa, ca, _ = a.headline
        sb, cb, _ = b.headline
        delta = sa - sb
        return delta > math.sqrt(ca**2 + cb**2) and delta > (self.noise_floor or 0.0)

    @property
    def best(self) -> VariantResult | None:
        ranked = self.ranked
        if not ranked:
            return None
        top = ranked[0]
        return top if all(self.clears(top, other) for other in ranked[1:]) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "behavior": self.behavior,
            "test_version": self.test_version,
            "n_asks": self.n_asks,
            "k": self.k,
            "noise_floor": self.noise_floor,
            "judge": self.judge.wire() if self.judge else None,
            "best": self.best.label if self.best else None,
            "variants": [
                {
                    "label": v.label,
                    "model": v.model,
                    "harness": v.fingerprint,
                    "run": v.run_id,
                    "scores": {k: list(s) for k, s in v.scores.items()},
                }
                for v in self.ranked
            ],
        }

    def __str__(self) -> str:
        head = (
            f"harness sweep on {self.behavior}: {len(self.variants)} variants, "
            f"{self.n_asks} asks x {self.k}, test {self.test_version}"
        )
        if self.noise_floor is not None:
            head += f", noise floor {self.noise_floor:g}"
        if self.judge and self.judge.agreement is not None:
            head += f", judge agreement {self.judge.agreement:.2f} on {self.judge.human_n}"
        w_label = max(7, *(len(v.label) for v in self.variants)) + 2
        w_model = max(5, *(len(v.model or "-") for v in self.variants)) + 2
        lines = [
            head,
            f"  {'variant':<{w_label}}{'model':<{w_model}}{'harness':<14}{'pass@1':>7}{'±':>6}",
        ]
        for v in self.ranked:
            s, ci, _ = v.headline
            lines.append(
                f"  {v.label:<{w_label}}{(v.model or '-'):<{w_model}}{v.fingerprint:<14}{s:>7}{ci:>6}"
            )
        if self.n_asks < MIN_ASKS:
            lines.append(
                f"  {self.n_asks} asks is under {MIN_ASKS}: the platform verdict says unproven "
                "until the frozen test has that many (a new set is a new test name)"
            )
        best = self.best
        if best is None and self.variants:
            top = self.ranked[0]
            lines.append(
                f"  no winner: {top.label} leads but its interval does not clear every "
                "other variant and the noise floor; more asks or a bigger change"
            )
        elif best is not None:
            lines.append(f"  winner: {best.label} clears every other variant and the noise floor")
        return "\n".join(lines)


def prompt_of(harness: Harness) -> str:
    """The prompt name a variant's label carries: name the variants
    ``prompt@model`` (``policy+act@sonnet5``) and the page groups by prompt
    and by model as two axes; a label with no ``@`` is the prompt itself."""
    return harness.version.split("@", 1)[0]


class HarnessSweep:
    """Score many harness variants of one agent on the same frozen asks and
    post one run per variant, its record pinned to the prompt, model and
    fingerprint that produced the score.

    ``tracked`` is the platform handle from ``track()``. ``judge`` is the
    grader (a program or a checked model judge) applied to every
    variant's rows. ``k`` is rollouts per ask. ``behavior`` names the
    headline score; every marker the judge sets becomes a behavior beside
    it. ``noise_runs`` scores the first variant that many times and takes
    the spread as the noise floor. ``labels`` are hand labels for
    ``judge_trust`` on the first variant's fresh rollouts, or a ``Judge``
    already measured on the frozen run (hand labels attach to the replies
    a person read, and a sweep rolls new ones), posted as the behaviors'
    ``Judge``. ``concurrency`` is parallel rollouts per variant (the
    library default is 32; a small provider key wants 4 to 8). Name
    variants ``prompt@model`` and the Runs page groups by each; a label
    is a run version on the platform, so it is checked against that cap
    before any rollout runs.
    """

    def __init__(
        self,
        tracked: Tracked,
        *,
        judge: Callable[[dict], Any],
        k: int = 4,
        tools: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        behavior: str = "policy",
        noise_runs: int = 2,
        labels: Any = None,
        concurrency: int | None = None,
    ):
        self.tracked = tracked
        self.judge = judge
        self.k = int(k)
        self.tools = list(tools or [])
        self.system_prompt = system_prompt
        self.behavior = behavior
        self.noise_runs = max(1, int(noise_runs))
        self.labels = labels
        self.concurrency = concurrency

    # ---------------------------------------------------------------- scoring

    def _rollouts(self, agent: Agent, tasks: Any, seed: int = 0) -> list[dict]:
        from .simulations import simulate

        data = simulate(
            agent,
            tools=self.tools or None,
            system_prompt=self.system_prompt,
            tasks=tasks,
            budget=None,
            simulator=False,
            mode="rl",
            repeats=self.k,
            repeat_policy="fixed",
            reproducible=True,
            seed=seed,
            fault_rate=0.0,
            avg_turns=1,
            concurrency=self.concurrency,
        )
        return [dict(r) for r in data.rows()]

    def _scored(self, rows: Sequence[dict]) -> list[dict]:
        from .simulations import evaluate

        return list(evaluate(rows, self.judge, tools=self.tools or None).rows)

    def _scores(self, rows: Sequence[dict]) -> dict[str, Score]:
        from .simulations import marker_summary, pass_at

        pa = pass_at(rows, k=self.k)
        mean = float(pa.pass_at_1 or 0.0)
        lo, hi = pa.ci95 or (mean, mean)
        out: dict[str, Score] = {
            self.behavior: (round(100 * mean, 1), round(100 * (hi - lo) / 2, 1), pa.n_groups)
        }
        for name, m in marker_summary(rows).items():
            if m["n_tasks"] < MIN_MARKER_ASKS:
                continue
            mlo, mhi = m["ci95"] or (m["mean"], m["mean"])
            out[name] = (round(100 * m["mean"], 1), round(100 * (mhi - mlo) / 2, 1), m["n_tasks"])
        return out

    def _judge(self, rows: Sequence[dict]) -> Judge | None:
        if self.labels is None:
            return None
        if isinstance(self.labels, Judge):
            return self.labels
        from .simulations import attach_labels, judge_trust

        labeled, _ = attach_labels([dict(r) for r in rows], self.labels, kind="human")
        trust = judge_trust(labeled, self.judge)
        agreement = trust["agreement"]
        return Judge(
            name=getattr(self.judge, "__name__", "judge"),
            agreement=agreement["agreement"],
            human_n=agreement["n"],
        )

    # ------------------------------------------------------------------- run

    def run(
        self,
        variants: Mapping[str, tuple[Harness, Agent]] | Sequence[tuple[Harness, Agent]],
        tasks: Any,
        *,
        test_version: str | None = None,
        post: bool = True,
    ) -> SweepReport:
        """Score every ``(Harness, agent)`` on ``tasks`` (a previous run, its
        rows, or a JSONL path: the frozen asks) and post one run per variant.
        ``post=False`` scores without touching the platform."""
        pairs = list(variants.values()) if isinstance(variants, Mapping) else list(variants)
        # a ``whileai.Harness`` (runnable) is accepted next to the platform record
        pairs = [(_pinned(h), a) for h, a in pairs]
        if not pairs:
            raise ValueError("variants is empty; pass at least one (Harness, agent) pair")
        check_labels([h for h, _ in pairs])
        results: list[VariantResult] = []
        first_rows: list[dict] | None = None
        asks: list[str] | None = None
        for harness, agent in pairs:
            rows = self._scored(self._rollouts(agent, tasks))
            got = sorted({str(r.get("prompt") or "") for r in rows})
            if asks is None:
                asks = got
                first_rows = rows
            elif got != asks:
                raise ValueError(
                    f"{harness.version} faced different asks from {pairs[0][0].version}; "
                    "pass the same frozen run as tasks= for every variant"
                )
            results.append(
                VariantResult(
                    label=harness.version,
                    model=harness.model,
                    fingerprint=harness.fingerprint,
                    scores=self._scores(rows),
                )
            )
        assert asks is not None and first_rows is not None
        version = test_version or "t-" + hashlib.sha256("\n".join(asks).encode()).hexdigest()[:8]

        noise: float | None = None
        if self.noise_runs > 1:
            first = results[0].headline[0]
            again = [
                self._scores(self._scored(self._rollouts(pairs[0][1], tasks, seed=i)))[
                    self.behavior
                ][0]
                for i in range(1, self.noise_runs)
            ]
            noise = round(max(abs(first - a) for a in again), 1)
        judge = self._judge(first_rows)

        report = SweepReport(
            behavior=self.behavior,
            test_version=version,
            n_asks=len(asks),
            k=self.k,
            noise_floor=noise,
            judge=judge,
            variants=results,
        )
        if post:
            self._post(report, pairs)
        return report

    def _post(self, report: SweepReport, pairs: Sequence[tuple[Harness, Agent]]) -> None:
        first = report.variants[0]
        for name, (_s, _ci, n) in first.scores.items():
            self.tracked.behavior(
                Behavior(
                    name=name,
                    test_version=report.test_version,
                    n=n,
                    judge=report.judge,
                    noise_floor=report.noise_floor,
                    contamination=0,
                    reward_is_judge=False,
                )
            )
        for (harness, _agent), result in zip(pairs, report.variants):
            run = self.tracked.run(
                harness.version,
                method="eval",
                targets=[report.behavior],
                harness=harness,
                record=RunRecord(
                    data=Data(holdout=report.test_version, n_holdout=report.n_asks),
                    eval=EvalSetup(
                        metric="pass@1",
                        k=self.k,
                        run_std=report.noise_floor,
                        run_std_runs=self.noise_runs,
                        reader=report.judge.name if report.judge else None,
                    ),
                    provenance=Provenance(
                        pins={
                            "prompt": prompt_of(harness),
                            **({"tools": ",".join(sorted(harness.tools))} if harness.tools else {}),
                        }
                    ),
                ),
            )
            for name, (pts, ci, n) in result.scores.items():
                run.score(name, pts, ci=ci, n=n, test_version=report.test_version)
            run.finish()
            result.run_id = run.id


__all__ = ["HarnessSweep", "SweepReport", "VariantResult"]
