"""A harness sweep: many prompt, tool and model variants of one agent, scored
on the same frozen asks, one run per harness fingerprint on the platform.

Today a team changes a prompt by hand and scores it. This runs every
variant at once on the same held-out asks, grades them with the same
judge, measures the noise floor by scoring one variant again, and posts
each as a run whose record pins the prompt label, the model and the
harness fingerprint, so the Runs page groups the dots by prompt or by
model and the verdict says which win is real. The winner is the variant
whose interval clears every other and the noise floor; anything closer
is "about the same", not a result.

rlhfbook.com, "Evaluation": scores move with the prompt and sampling
setup, not only the weights, so a comparison is only fair with the setup
held constant. Miller 2024 (arXiv 2411.00640): the interval is the result.
The floor is the platform's rule, ``Tracked.noise_floor``: ``eval_variance``
over the re-runs gives ``run_std``, and the band a delta has to clear is
t(df=runs-1) x run_std x sqrt(2), in points. Two re-runs make df=1 and a t
of 12.7, so the band is wide; three or more narrow it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .platform import (
    BEHAVIOR_NAME_MAX,
    BEHAVIOR_NAME_PATTERN,
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
Score = tuple[float, float, int]  # points, half-width of the 95% interval, asks with a graded row
MIN_MARKER_ASKS = 3  # under this a marker has no interval and is left off the card
# Miller 2024 (arXiv 2411.00640): under ~50 items the interval is too wide to
# show a gain of a few points; the platform verdict says "unproven" below it.
MIN_ASKS = 50
LABEL_MAX = 40  # RunSpec.version max_length; tests pin the two together
_NAME = re.compile(BEHAVIOR_NAME_PATTERN)


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


def check_name(name: Any, what: str = "behavior") -> str:
    """Refuse a behavior or marker name the platform would refuse at post
    time, with the rule ``Behavior(name=)`` applies (``BEHAVIOR_NAME_PATTERN``,
    at most ``BEHAVIOR_NAME_MAX`` characters). The headline behavior is
    checked when the sweep is built, before any rollout; a marker the judge
    sets is checked the first time it is scored, before the next variant
    rolls."""
    if not isinstance(name, str) or len(name) > BEHAVIOR_NAME_MAX or not _NAME.match(name):
        where = "HarnessSweep(behavior=)" if what == "behavior" else "the judge's markers dict"
        raise ValueError(
            f"{what} name {name!r} is not a platform behavior name: Behavior(name=) takes "
            f"{BEHAVIOR_NAME_PATTERN} up to {BEHAVIOR_NAME_MAX} characters (refund_policy, "
            f"not 'refund policy'). Rename it in {where}"
        )
    return name


def _graded_per_ask(rows: Sequence[dict]) -> Counter[str]:
    """Graded rows (``reward`` not None) per ask, keyed the way ``pass_at``
    groups them. A judge error leaves ``reward=None`` on its row and the
    engine drops an agent-error rollout, so this is the denominator each
    arm's mean rests on."""
    from .simulations.score.stats import task_key

    return Counter(task_key(r) for r in rows if r.get("reward") is not None)


def _refuse_unequal(label: str, graded: Counter[str], first: str, expected: Counter[str]) -> None:
    """Refuse an arm whose graded rows per ask differ from the first arm's:
    the two means would rest on different denominators."""
    if graded == expected:
        return
    differing = sorted(k for k in set(graded) | set(expected) if graded.get(k) != expected.get(k))
    raise ValueError(
        f"{label} was graded {sum(graded.values())} rows on {len(graded)} asks; {first} "
        f"{sum(expected.values())} rows on {len(expected)} asks ({len(differing)} ask(s) differ, "
        f"first {differing[0]!r}). Every arm must be scored on the same rows per ask or the "
        "means rest on different denominators: a judge error marks its row reward=None "
        "(judge_status 'error') and an agent error drops its rollout. Fix the judge or the "
        "agent for that arm and re-run, with the same frozen run as tasks= for every variant"
    )


@dataclass
class VariantResult:
    """One harness variant after scoring. ``graded_rows`` is the count of
    rows that carried a reward, the denominator behind ``scores``."""

    label: str
    model: str | None
    fingerprint: str
    scores: dict[str, Score]
    run_id: str | None = None
    graded_rows: int | None = None

    @property
    def headline(self) -> Score:
        return next(iter(self.scores.values()))


@dataclass
class SweepReport:
    """What a sweep found. ``print()`` it; ``best`` is the winner or None
    when no variant's interval clears the rest. ``noise_floor`` is the
    re-run band in points, t(df=noise_runs-1) x ``run_std`` x sqrt(2) x 100,
    the rule ``Tracked.noise_floor`` applies; ``run_std`` is the sample
    standard deviation of the first variant's pass@1 over ``noise_runs``
    scorings, in 0-1 units."""

    behavior: str
    test_version: str
    n_asks: int
    k: int
    noise_floor: float | None
    judge: Judge | None
    variants: list[VariantResult] = field(default_factory=list)
    run_std: float | None = None
    noise_runs: int | None = None

    @property
    def ranked(self) -> list[VariantResult]:
        return sorted(self.variants, key=lambda v: v.headline[0], reverse=True)

    def clears(self, a: VariantResult, b: VariantResult) -> bool:
        """``a`` beats ``b`` for real: both were scored on the same number of
        asks, the difference interval excludes zero and the difference
        clears the noise floor. Two arms on different denominators are never
        compared; a judge error or a dropped rollout on one side moves that
        mean on its own."""
        sa, ca, na = a.headline
        sb, cb, nb = b.headline
        if na != nb:
            return False
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
            "run_std": self.run_std,
            "noise_runs": self.noise_runs,
            "judge": self.judge.wire() if self.judge else None,
            "best": self.best.label if self.best else None,
            "variants": [
                {
                    "label": v.label,
                    "model": v.model,
                    "harness": v.fingerprint,
                    "run": v.run_id,
                    "graded_rows": v.graded_rows,
                    "scores": {k: list(s) for k, s in v.scores.items()},
                }
                for v in self.ranked
            ],
        }

    def _noise_line(self) -> str | None:
        if self.noise_floor is None or not self.noise_runs or self.run_std is None:
            return None
        from .simulations.defaults import MIN_RERUNS
        from .simulations.score.stats import POINTS_PER_UNIT, _t_quantile

        df = self.noise_runs - 1
        line = (
            f"  noise floor {self.noise_floor:g} = t(df={df})={_t_quantile(df):.2f} x run_std "
            f"{self.run_std:.4f} x sqrt(2) x {POINTS_PER_UNIT}, from {self.noise_runs} re-runs"
        )
        if self.variants:
            line += f" of {self.variants[0].label}"
        if self.noise_runs < MIN_RERUNS:
            line += (
                f": a wide band from {self.noise_runs} runs; noise_runs={MIN_RERUNS} or more "
                "narrows it"
            )
        return line

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
            f"  {'variant':<{w_label}}{'model':<{w_model}}{'harness':<14}{'pass@1':>7}{'±':>6}"
            f"{'asks':>6}{'graded':>8}",
        ]
        for v in self.ranked:
            s, ci, n = v.headline
            graded = "-" if v.graded_rows is None else str(v.graded_rows)
            lines.append(
                f"  {v.label:<{w_label}}{(v.model or '-'):<{w_model}}{v.fingerprint:<14}{s:>7}"
                f"{ci:>6}{n:>6}{graded:>8}"
            )
        noise = self._noise_line()
        if noise:
            lines.append(noise)
        if len({v.headline[2] for v in self.variants}) > 1:
            counts = ", ".join(f"{v.label} {v.headline[2]}" for v in self.ranked)
            lines.append(
                f"  variants were scored on different asks ({counts}): a judge error or a "
                "dropped rollout on one side, so no comparison between them is a result"
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
    headline score, under the platform's behavior-name rule (checked here,
    before any rollout); every marker the judge sets becomes a behavior
    beside it. ``noise_runs`` scores the first variant that many times in
    all; the floor is the platform's rule (``Tracked.noise_floor``):
    ``eval_variance`` over the scorings gives ``run_std``, and the band a
    delta has to clear is t(df=runs-1) x run_std x sqrt(2), in points. The
    default 2 gives df=1 and a t of 12.7, a wide band the report says so
    about; 3 or more narrow it. ``labels`` are hand labels for
    ``judge_trust`` on the first variant's fresh rollouts, or a ``Judge``
    already measured on the frozen run (hand labels attach to the replies
    a person read, and a sweep rolls new ones), posted as the behaviors'
    ``Judge``. ``concurrency`` is parallel rollouts per variant (the
    library default is 32; a small provider key wants 4 to 8). Name
    variants ``prompt@model`` and the Runs page groups by each; a label
    is a run version on the platform, so it is checked against that cap
    before any rollout runs. Every arm must carry the same graded rows per
    ask as the first; an arm that lost rows to a judge error or a dropped
    rollout is refused by name, since its mean would rest on a different
    denominator.
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
        self.behavior = check_name(behavior)
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
            check_name(name, "marker")
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
        ``post=False`` scores without touching the platform. An arm graded on
        different rows per ask from the first is refused by name."""
        pairs = list(variants.values()) if isinstance(variants, Mapping) else list(variants)
        # a ``whileai.Harness`` (runnable) is accepted next to the platform record
        pairs = [(_pinned(h), a) for h, a in pairs]
        if not pairs:
            raise ValueError("variants is empty; pass at least one (Harness, agent) pair")
        check_labels([h for h, _ in pairs])
        first_label = pairs[0][0].version
        results: list[VariantResult] = []
        first_rows: list[dict] | None = None
        first_graded: Counter[str] | None = None
        for harness, agent in pairs:
            rows = self._scored(self._rollouts(agent, tasks))
            graded = _graded_per_ask(rows)
            if first_graded is None:
                first_graded = graded
                first_rows = rows
            else:
                _refuse_unequal(harness.version, graded, first_label, first_graded)
            results.append(
                VariantResult(
                    label=harness.version,
                    model=harness.model,
                    fingerprint=harness.fingerprint,
                    scores=self._scores(rows),
                    graded_rows=sum(graded.values()),
                )
            )
        assert first_rows is not None and first_graded is not None
        asks = sorted({str(r.get("prompt") or "") for r in first_rows})
        version = test_version or "t-" + hashlib.sha256("\n".join(asks).encode()).hexdigest()[:8]

        noise: float | None = None
        run_std: float | None = None
        noise_runs: int | None = None
        if self.noise_runs > 1:
            from .simulations.score.stats import POINTS_PER_UNIT, eval_variance

            reruns: list[list[dict]] = []
            for i in range(1, self.noise_runs):
                again = self._scored(self._rollouts(pairs[0][1], tasks, seed=i))
                _refuse_unequal(
                    f"re-run {i + 1} of {first_label}",
                    _graded_per_ask(again),
                    first_label,
                    first_graded,
                )
                reruns.append(again)
            # The platform's rule (``Tracked.noise_floor``): the band carries the
            # estimate's own degrees of freedom, runs - 1, and is posted in points.
            variance = eval_variance(first_rows, *reruns, metric="pass_at_1")
            if variance["run_std"] is not None:
                run_std = float(variance["run_std"])
                noise = round(float(variance["noise_band"]) * POINTS_PER_UNIT, 2)
                noise_runs = int(variance["n_runs"])
        judge = self._judge(first_rows)

        report = SweepReport(
            behavior=self.behavior,
            test_version=version,
            n_asks=len(asks),
            k=self.k,
            noise_floor=noise,
            judge=judge,
            variants=results,
            run_std=run_std,
            noise_runs=noise_runs,
        )
        if post:
            self._post(report, pairs)
        return report

    def _post(self, report: SweepReport, pairs: Sequence[tuple[Harness, Agent]]) -> None:
        first = report.variants[0]
        for name, (_s, _ci, n) in first.scores.items():
            # ``contamination`` and ``reward_is_judge`` are not measured here,
            # so they are not posted; a zero would read as a measurement.
            self.tracked.behavior(
                Behavior(
                    name=name,
                    test_version=report.test_version,
                    n=n,
                    judge=report.judge,
                    noise_floor=report.noise_floor,
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
                        run_std=report.run_std,
                        run_std_runs=report.noise_runs,
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
