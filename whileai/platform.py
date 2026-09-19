"""Report what you trained to the While platform, so a person can decide.

The platform draws one screen per tracked agent: the held-out score by
version with the frontier model as the line to beat, the training curve,
what moved on the behaviors you did not train, the judge checks, live
traffic on the served version, and cost. This module is how a coding
agent fills that screen. The person reads it and presses Promote.

Your agent framework stays yours. ``track`` takes the agent object you
already have (OpenAI Agents SDK, Pydantic AI, LangGraph, Claude Agent SDK,
or anything with a name, a model and tools) and reads the model, the
instructions and the tool list off it to fingerprint the harness. Or
describe it by hand::

    from whileai.platform import Behavior, Frontier, Harness, Judge, track

    tracked = track(
        "refund-bot",
        model="Qwen/Qwen3-4B",
        harness=Harness(instructions=SYSTEM_PROMPT, tools=["lookup_order", "issue_refund"]),
        frontier=Frontier(name="Sonnet 5", score=81, cost_per_1k=18.0),
    )
    tracked.behavior(
        Behavior(
            name="refunds",
            test_version="v2",
            n=240,
            judge=Judge(agreement=0.86, human_n=60, length_bias=0.08),
            noise_floor=2.4,
            reward_is_judge=False,
        )
    )

    run = tracked.run("v4", method="GRPO", targets=["refunds"], trained_on=["refunds-grpo"])
    run.log(10, reward=0.41, kl=0.01)       # or trainer.add_callback(wai.TrainerCallback(run))
    run.score("refunds", 83, ci=2.7, n=240)  # every behavior, not only the targets
    run.finish(hours=2.1, gpu="1xH100", cost_usd=31, record=RunRecord(  # drawn as one table
        data=Data(train="refunds-grpo", n_train=1024, holdout="refunds-test-v2", n_holdout=240),
        optimizer=Optimizer(loss_type="dapo", lr=5e-5, beta=1e-4, num_generations=8, seed=17),
        eval=EvalSetup(metric="pass@1", k=4, run_std=0.02, run_std_runs=3),
        provenance=Provenance(pins={"trl": "1.13.0"}, paper="2503.18892"),
    ))

    print(tracked.verdict())  # refunds: v4 beats v3 by 5 (interval excludes zero); 1 regression

Say what the runs are for, and show your working. The experiment block
sits at the top of the Runs page, a figure grid follows the run table,
and a note sits under its run. Figures are illustration; the verdict
above comes from the scored evals::

    tracked.experiment(
        question="Does GRPO on refunds-grpo lift refunds without moving length?",
        measure="pass@1 on refunds-test-v2, n=240, 95% interval",
        decide="promote when the interval clears the 2.4 noise floor",
    )
    tracked.figure("reward-by-step", fig, caption="Training reward, v4")  # plotly Figure or dict
    run.note("Reward flattened at step 300; the last 100 steps bought nothing.")

Every object is a pydantic model, validated before it leaves the process, and
each one says which chapter of Lambert 2025 (arXiv:2504.12501) it comes from.
Chapters are cited by title because the numbering has moved between editions.
Logging never raises into a training loop: points are buffered, sent in
batches, and a failed send is retried on the next flush.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from datetime import date
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from whileai._env import getenv
from whileai.auth import resolve_api_key

log = logging.getLogger("whileai.platform")

DEFAULT_PLATFORM_URL = "https://mbxp83jd48.execute-api.us-east-1.amazonaws.com"
PLATFORM_URL_ENV = "WHILEAI_PLATFORM_URL"

FLUSH_EVERY = 25
FLUSH_SECONDS = 15.0
MAX_BATCH = 2000

#: What the platform accepts for the experiment block, figures and run
#: notes. Each mirrors the check the API makes, so a bad value fails on
#: the line that wrote it instead of as a 4xx from the server.
EXPERIMENT_FIELD_MAX = 4096  # chars per experiment field, markdown allowed
FIGURE_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,39}$"  # one figure per name per agent
FIGURE_MAX_BYTES = 200_000  # JSON bytes of {data, layout}; the API says 413 past it
FIGURE_MAX_TRACES = 50  # traces per figure; the API says 422 past it
FIGURE_TRACE_TYPES = ("scatter", "bar", "pie")  # the page ships plotly.js-basic
FIGURE_CAPTION_MAX = 1024  # chars
FIGURE_DROPPED_LAYOUT_KEYS = ("images", "updatemenus", "sliders", "template")  # never stored
NOTES_MAX = 8192  # chars of markdown on one run

Transport = Callable[..., Any]


class PlatformError(RuntimeError):
    """The platform said no. ``status`` is the HTTP code, 0 when unreachable."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------- wire


class _Wire(BaseModel):
    """Base for everything that crosses to the API: snake_case in Python,
    camelCase on the wire, unknown fields from newer servers ignored."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
        validate_assignment=True,
    )

    def wire(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")


class Frontier(_Wire):
    """The model you pay for today, drawn on every screen as the line to beat.

    Lambert 2025, chapter Synthetic Data and Distillation: a stronger model's
    outputs are the usual teacher for a smaller open one, so the
    comparison the platform makes is teacher vs student on the same
    held-out test. ``score`` is on the same test as the versions;
    ``cost_per_1k`` replies and ``p50_s`` latency are the deployment
    numbers that chapter does not cover and a buyer asks about first.
    """

    name: str
    score: float | None = None
    cost_per_1k: float | None = Field(default=None, alias="costPer1k", ge=0)
    p50_s: float | None = Field(default=None, alias="p50s", ge=0)


class Harness(_Wire):
    """Everything around the weights: the instructions, the tools, the model
    name. Changing any of it changes what the agent does, so it is
    versioned like weights and the fingerprint is the version.

    Lambert 2025, chapter Evaluation: scores move with the prompt and sampling
    setup, not only the weights, so a result is only comparable with its
    setup held constant. The fingerprint is how the platform knows two
    versions were measured under the same harness.
    """

    label: str | None = None
    instructions: str | None = None
    tools: list[str] = Field(default_factory=list)
    model: str | None = None

    @field_validator("tools", mode="before")
    @classmethod
    def _tool_names(cls, value: Any) -> list[str]:
        return [_tool_name(t) for t in (value or [])]

    @property
    def fingerprint(self) -> str:
        """Twelve hex characters over model, instructions and sorted tools."""
        blob = json.dumps(
            {"model": self.model, "instructions": self.instructions, "tools": sorted(self.tools)},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def version(self) -> str:
        return self.label or f"h-{self.fingerprint}"

    def wire(self) -> dict[str, Any]:
        out = super().wire()
        out["label"] = self.version
        out["hash"] = self.fingerprint
        return out


class Judge(_Wire):
    """How the scores on a behavior were produced, and whether to trust them.

    Lambert 2025, chapter Evaluation: LLM-as-a-judge replaced human raters for
    most post-training evals; a judge is only as good as its agreement
    with people on a labeled slice (``agreement`` over ``human_n`` items,
    after Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench and
    Chatbot Arena"). ``length_bias`` is the correlation of score with
    reply length, the bias that length-controlled AlpacaEval (Dubois et
    al. 2024) was built to remove.
    """

    name: str | None = None
    agreement: float | None = Field(default=None, ge=0, le=1)
    human_n: int | None = Field(default=None, ge=0)
    length_bias: float | None = Field(default=None, ge=-1, le=1)


class Behavior(_Wire):
    """One thing you measure, with its own frozen held-out test and judge.

    Lambert 2025, chapter Evaluation: labs keep train, dev and held-out sets
    apart, and post-training evals move 0.25 to 1.5 points between runs
    of the same setup. So ``test_version`` names a frozen set (bump it
    when the set changes), ``noise_floor`` is that run-to-run spread
    measured on your own test, ``contamination`` is how many test items
    were found in the training data, and ``n`` is the size the interval
    comes from.

    Gao et al. 2022, arXiv:2210.10760, and Lambert 2025, chapter Reward
    Modeling: a judge that is also the reward gets exploited and cannot see it
    happen, which is what ``reward_is_judge`` records and warns about.
    """

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    test_version: str | None = None
    n: int | None = Field(default=None, ge=1)
    judge: Judge | None = None
    noise_floor: float | None = Field(default=None, ge=0)
    contamination: int | None = Field(default=None, ge=0)
    reward_is_judge: bool | None = None
    description: str | None = Field(default=None, max_length=400)


class Experiment(_Wire):
    """What this agent's training is trying to find out, in the agent's
    own words: one block of markdown fields, one per tracked agent.

    The Runs page renders it at the top, above the run table, as five
    labeled rows (Question, Hypothesis, Method, Measure, Decide) plus
    Notes when set, so the person reading the dashboard knows what the
    runs below are for before they read a number. ``question`` is
    required; the rest are optional; every field is markdown of at most
    4096 chars. Sent with ``tracked.experiment(...)``, read back with
    ``tracked.experiment()``.

    Convention, untested: the five headings are the pre-registration a
    lab writes before a run, so the dashboard shows the claim next to
    the evidence.
    """

    question: str = Field(min_length=1, max_length=EXPERIMENT_FIELD_MAX)
    hypothesis: str | None = Field(default=None, max_length=EXPERIMENT_FIELD_MAX)
    method: str | None = Field(default=None, max_length=EXPERIMENT_FIELD_MAX)
    measure: str | None = Field(default=None, max_length=EXPERIMENT_FIELD_MAX)
    decide: str | None = Field(default=None, max_length=EXPERIMENT_FIELD_MAX)
    notes: str | None = Field(default=None, max_length=EXPERIMENT_FIELD_MAX)


class Figure(_Wire):
    """One Plotly figure the agent posted, as JSON (``{data, layout}``),
    never as code.

    The Runs page draws every figure on the agent in a grid after the
    run table, caption (or name) above each, with the house layout under
    the agent's layout. Figures are illustration: the verdict on the page
    comes from the platform-scored evals (``run.score``), not from
    anything drawn here. ``figure`` holds ``data`` (1 to 50 traces of
    type scatter, bar or pie) and ``layout``; ``run`` is the id of the
    run it belongs to, when it belongs to one. Sent with
    ``tracked.figure(name, fig)``, listed with ``tracked.figures()``.
    """

    name: str = Field(pattern=FIGURE_NAME_PATTERN)
    caption: str | None = Field(default=None, max_length=FIGURE_CAPTION_MAX)
    run: str | None = None
    figure: dict[str, Any]
    updated_at: str | None = None


class Data(_Wire):
    """What the run trained on and what it was scored against, as ids and
    counts a reader can check, not as a sentence.

    Lambert 2025, chapter Evaluation: a score means nothing without the
    held-out set it came from and proof the training data did not contain it.
    ``decontaminated_dropped`` is that proof as a count; ``hash`` fields pin
    the exact rows (sha256 of the ordered ids, or a dataset revision).
    """

    train: str | None = None
    train_hash: str | None = None
    n_train: int | None = Field(default=None, ge=0)
    difficulty: str | None = None
    holdout: str | None = None
    holdout_hash: str | None = None
    n_holdout: int | None = Field(default=None, ge=0)
    decontaminated_dropped: int | None = Field(default=None, ge=0)


class Optimizer(_Wire):
    """The knobs that decide what the policy gradient does, named the way
    the papers name them so two runs can be compared knob by knob.

    Schulman et al. 2017, arXiv:1707.06347, and Yu et al. 2025 (DAPO),
    arXiv:2503.14476: ``epsilon`` and ``epsilon_high`` are
    the clip range (DAPO's clip-higher when they differ), ``beta`` the KL
    coefficient, ``loss_type`` which normalisation the objective uses (GRPO's
    per-sequence mean, Dr. GRPO / DAPO's token sum). ``num_generations`` is
    the group size the advantage is relative to. ``seed`` is the one number
    a re-run needs first.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")

    loss_type: str | None = None
    lr: float | None = None
    beta: float | None = Field(default=None, ge=0)
    epsilon: float | None = Field(default=None, ge=0)
    epsilon_high: float | None = Field(default=None, ge=0)
    num_generations: int | None = Field(default=None, ge=1)
    prompts_per_step: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    lora_rank: int | None = Field(default=None, ge=0)
    seed: int | None = None


class EvalSetup(_Wire):
    """How the held-out score was produced: the metric, samples per task,
    and the eval's own re-run noise.

    Lambert 2025, chapter Evaluation: one evaluation is one draw. ``run_std``
    is the standard deviation of the untrained base's score over
    ``run_std_runs`` re-runs of the same eval, the floor a delta has to clear
    before it is a result. ``reader`` names how the answer span was read
    (``boxed``, ``lenient``, a judge name).
    """

    metric: str | None = None
    k: int | None = Field(default=None, ge=1)
    run_std: float | None = Field(default=None, ge=0)
    run_std_runs: int | None = Field(default=None, ge=1)
    reader: str | None = None


class Provenance(_Wire):
    """What it takes to run this again: library versions, the image, the
    recipe and commit, the paper, where the weights landed.

    Lambert 2025, evaluation-variance appendix: pin the seed and the versions;
    a run that cannot be repeated is not a result.
    """

    pins: dict[str, str] = Field(default_factory=dict)
    image: str | None = None
    recipe: str | None = None
    commit: str | None = None
    paper: str | None = None
    adapter: str | None = None


class RunRecord(_Wire):
    """The scientific record of one run: data, optimizer, eval setup and
    provenance, each a typed block. Attach it when the run opens
    (``tracked.run(..., record=)``) or when it closes (``run.finish(record=)``).
    The Runs page draws it as one table next to the curve.
    """

    data: Data | None = None
    optimizer: Optimizer | None = None
    eval: EvalSetup | None = None
    provenance: Provenance | None = None


class RunSpec(_Wire):
    """What one training job is: the version it produces, the base it starts
    from, the method, which behaviors it aims at, and which datasets it
    trained on.

    GRPO is Shao et al. 2024, arXiv:2402.03300, and DPO is Rafailov et al.
    2023, arXiv:2305.18290; ``method`` is recorded, not interpreted.
    ``targets`` are the claim, every other behavior is the check (see
    ``Score``).
    """

    version: str = Field(min_length=1, max_length=40)
    base: str | None = None
    method: str | None = Field(default=None, max_length=20)
    targets: list[str] = Field(default_factory=list)
    trained_on: list[str] = Field(default_factory=list)
    harness: str | None = None
    gpu: str | None = None
    record: RunRecord | None = None
    id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class TrainPoint(_Wire):
    """One point on the training curve.

    The group-normalised reward is what GRPO climbs (Shao et al. 2024,
    arXiv:2402.03300), and the KL penalty to the reference model is the brake
    (Schulman et al. 2017, arXiv:1707.06347). KL distance from the start is
    the measure of how far the policy has moved, and a run whose reward climbs
    while KL runs away is the picture of a proxy being gamed (Gao et al. 2022,
    arXiv:2210.10760). ``loss`` is the SFT and DPO curve. Other finite numbers
    are kept under their own names.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")

    step: int = Field(ge=0)
    reward: float | None = None
    kl: float | None = Field(default=None, ge=0)
    loss: float | None = None


class Score(_Wire):
    """One version scored on one behavior's frozen held-out test.

    Lambert 2025, chapter Evaluation: one evaluation is one draw; the interval
    is the result. ``ci`` is the half-width of the 95% interval and ``n`` the
    number of held-out items it came from. Score every behavior the agent has,
    not only the ones the run trained: the untrained ones move too (verbosity,
    sycophancy, refusals; Lambert 2025, chapter Over-optimization) and the
    platform's regressions tile is that check.
    """

    behavior: str = Field(min_length=1, max_length=64)
    score: float = Field(allow_inf_nan=False)
    ci: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    n: int | None = Field(default=None, ge=1)
    test_version: str | None = None
    version: str | None = None


class LiveDay(_Wire):
    """One day of traffic on the served version.

    ``flagged`` is how many replies failed a check in production; the
    flagged rate over time is the only number that says whether a
    held-out improvement survived contact with real traffic. The served
    endpoint writes these itself when the platform serves the model.
    """

    day: date
    version: str
    replies: int = Field(ge=0)
    flagged: int = Field(default=0, ge=0)
    p50_s: float | None = Field(default=None, alias="p50s", ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


# --------------------------------------------------------------- read models


class VersionScore(_Wire):
    v: str
    score: float
    ci: float | None = None
    n: int | None = None
    run: str | None = None


class TrainCurve(_Wire):
    run: str
    method: str | None = None
    base: str | None = None
    hours: float | None = None
    gpu: str | None = None
    cost_usd: float | None = None
    status: str | None = None
    trained_on: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    steps: list[int] = Field(default_factory=list)
    reward: list[float | None] = Field(default_factory=list)
    kl: list[float | None] = Field(default_factory=list)
    loss: list[float | None] = Field(default_factory=list)
    #: mean completion length per step, and the share of rollouts that hit
    #: the length cap: the two curves that show a reasoning run growing or
    #: collapsing before the score does ("Over-Optimization").
    completion_length: list[float | None] = Field(default_factory=list)
    clip_ratio: list[float | None] = Field(default_factory=list)
    record: RunRecord | None = None


class Delta(_Wire):
    """Candidate minus served on one behavior; ``target`` marks the claim."""

    name: str
    candidate: float | None = None
    serving: float | None = None
    delta: float | None = None
    target: bool = False


class LiveSeries(_Wire):
    days: list[str] = Field(default_factory=list)
    version: list[str | None] = Field(default_factory=list)
    flagged_pct: list[float | None] = Field(default_factory=list)
    p50s: list[float | None] = Field(default_factory=list)
    replies: list[int] = Field(default_factory=list)
    served_at_index: int | None = None
    replies_7d: int = Field(default=0, alias="replies7d")
    new_failures: int = 0
    cost_per_1k: float | None = Field(default=None, alias="costPer1k")


class Verdict(_Wire):
    """The one line a person reads before Promote.

    Lambert 2025, chapter Evaluation: a difference inside the run-to-run
    spread is not a result. ``excludes_zero`` is whether the difference
    interval, ``delta +- sqrt(ci_candidate^2 + ci_served^2)``, excludes
    zero. "beats" or "trails" is said only when it does and the delta
    also clears the behavior's declared ``noise_floor``; a missing
    interval, an interval that includes zero, or a delta inside the
    re-run band is said in those words. ``regressions`` counts the other
    behaviors whose point estimate came out lower, with no interval on
    that check yet. The line ends with what the number rests on (judge
    agreement, n) and starts with "unproven:" when one is missing or short.
    """

    candidate: str | None = None
    serving: str | None = None
    delta: float | None = None
    excludes_zero: bool | None = None
    regressions: int = 0
    behavior: str | None = None
    # Filled from the behavior block of the dashboard.
    noise_floor: float | None = None
    n: int | None = None
    judge_agreement: float | None = None
    judge_human_n: int | None = None
    reward_is_judge: bool | None = None
    contamination: int | None = None

    def __str__(self) -> str:
        b = self.behavior or "?"
        if self.delta is None:
            if self.serving and not self.candidate:
                return f"{b}: {self.serving} is serving; no newer candidate yet"
            return f"{b}: no candidate scored against {self.serving or 'a served version'} yet"
        if self.candidate == self.serving:
            return f"{b}: {self.candidate} is the served version; no candidate to compare"
        delta = float(self.delta)
        signed = f"{delta:+g}"
        floor = self.noise_floor
        claim = False
        if self.excludes_zero is None:
            head = (
                f"{b}: {self.candidate} scored {signed} vs {self.serving}, no interval on one "
                "side, not a result (pass ci= on both)"
            )
        elif not self.excludes_zero:
            head = (
                f"{b}: {self.candidate} about the same as {self.serving} "
                f"({signed}, interval includes zero)"
            )
        elif floor is not None and abs(delta) <= floor:
            head = (
                f"{b}: {self.candidate} scored {signed} vs {self.serving}, interval excludes "
                f"zero but inside the eval's re-run band ({floor:g}), not a result"
            )
        else:
            claim = True
            word = "beats" if delta > 0 else "trails"
            note = (
                f", clears the noise floor of {floor:g}"
                if floor is not None
                else ", no noise floor declared"
            )
            head = (
                f"{b}: {self.candidate} {word} {self.serving} by {abs(delta):g} "
                f"(interval excludes zero{note})"
            )
        if self.regressions:
            head += (
                f"; {self.regressions} behavior{'' if self.regressions == 1 else 's'} lower "
                "(point estimates, no interval on that check)"
            )
        gaps: list[str] = []
        if self.n is None:
            gaps.append("n not declared")
        elif self.n < 50:
            gaps.append(f"n={self.n} under 50")
        if self.judge_agreement is None:
            gaps.append("judge agreement unmeasured")
        elif self.judge_agreement < 0.8:
            gaps.append(f"judge agreement {self.judge_agreement:g} under 0.8")
        if self.reward_is_judge:
            gaps.append("the training reward is the judge")
        if self.contamination:
            gaps.append(f"contamination {self.contamination}")
        rests: list[str] = []
        if self.judge_agreement is not None:
            rests.append(
                f"judge agreement {self.judge_agreement:g}"
                + (f" on {self.judge_human_n}" if self.judge_human_n else "")
            )
        if self.n is not None:
            rests.append(f"n={self.n}")
        if rests:
            head += "; " + ", ".join(rests)
        if claim and gaps:
            head = "unproven: " + head + " (" + "; ".join(gaps) + ")"
        return head


class TrackedInfo(_Wire):
    id: str
    name: str
    model: str | None = None
    harness: dict[str, Any] | None = None
    serving: str | None = None
    candidate: str | None = None
    frontier: Frontier | None = None


class Dashboard(_Wire):
    """The Runs screen as data. One per tracked agent and behavior."""

    agent: TrackedInfo
    behavior: Behavior | None = None
    behaviors: list[str] = Field(default_factory=list)
    versions: list[VersionScore] = Field(default_factory=list)
    train: TrainCurve | None = None
    deltas: list[Delta] = Field(default_factory=list)
    live: LiveSeries = Field(default_factory=LiveSeries)
    verdict: Verdict = Field(default_factory=Verdict)

    def model_post_init(self, __context: Any) -> None:
        beh = self.behavior
        if beh is None:
            return
        v = self.verdict
        if v.behavior is None:
            v.behavior = beh.name
        # What the verdict rests on travels with it, so str() can say it.
        v.noise_floor = beh.noise_floor if v.noise_floor is None else v.noise_floor
        v.n = beh.n if v.n is None else v.n
        if beh.judge is not None:
            if v.judge_agreement is None:
                v.judge_agreement = beh.judge.agreement
            if v.judge_human_n is None:
                v.judge_human_n = beh.judge.human_n
        if v.reward_is_judge is None:
            v.reward_is_judge = beh.reward_is_judge
        if v.contamination is None:
            v.contamination = beh.contamination


# ----------------------------------------------------------------- transport


def platform_url() -> str:
    return (getenv("PLATFORM_URL", DEFAULT_PLATFORM_URL) or DEFAULT_PLATFORM_URL).rstrip("/")


def _key(explicit: str | None) -> str:
    key = resolve_api_key(explicit)
    if not key:
        raise PlatformError(
            401,
            "No API key. Run `whileai login`, or set WHILEAI_API_KEY, or pass api_key=... "
            "(keys are on while.ai under Account).",
        )
    return key


def _request(
    method: str, path: str, *, api_key: str, body: Any = None, timeout: float = 30.0
) -> Any:
    """One call. Returns the parsed JSON; raises PlatformError on 4xx/5xx."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(platform_url() + path, data=data, method=method)
    req.add_header("X-Api-Key", api_key)
    req.add_header("User-Agent", "whileai-sdk")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            message = json.loads(raw or b"{}").get("error") or e.reason
        except (ValueError, AttributeError):
            message = e.reason
        raise PlatformError(e.code, f"{method} {path}: {message}") from None
    except urllib.error.URLError as e:
        raise PlatformError(
            0, f"{method} {path}: could not reach {platform_url()} ({e.reason})"
        ) from None
    return json.loads(raw or b"{}")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _json_default(value: Any) -> Any:
    """What ``json.dumps`` does with the values plotly puts in a figure
    without plotly's own encoder: numpy arrays and scalars, dates,
    bytes-free objects with ``tolist``/``item``/``isoformat``."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"figure holds a value JSON cannot carry: {type(value).__name__}")


def _figure_json(name: str, fig: Any) -> dict[str, Any]:
    """Turn a plotly ``Figure`` (duck-typed) or a ``{data, layout}``
    mapping into the wire shape, and check what the API checks. Raises
    ``ValueError`` on the line that took the bad value, before any
    network call. Returns the figure as plain JSON values."""
    if not isinstance(name, str) or not re.match(FIGURE_NAME_PATTERN, name):
        raise ValueError(
            f"figure name {name!r} must match {FIGURE_NAME_PATTERN} "
            "(lowercase letters, digits and dashes, up to 40 chars)"
        )
    if hasattr(fig, "to_plotly_json"):
        raw = fig.to_plotly_json()
    elif hasattr(fig, "to_dict"):
        raw = fig.to_dict()
    else:
        raw = fig
    if not isinstance(raw, Mapping) or "data" not in raw:
        raise ValueError(
            f"figure {name!r}: fig must be a plotly Figure or a dict with a 'data' list of "
            f"traces (got {type(fig).__name__})"
        )
    layout = raw.get("layout") or {}
    if not isinstance(layout, Mapping):
        raise ValueError(f"figure {name!r}: layout must be a dict (got {type(layout).__name__})")
    figure: dict[str, Any] = {
        "data": raw["data"],
        "layout": {k: v for k, v in layout.items() if k not in FIGURE_DROPPED_LAYOUT_KEYS},
    }
    # Compact, like JSON.stringify, so the byte count is the one the API sees.
    encoded = json.dumps(
        figure, default=_json_default, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(encoded) > FIGURE_MAX_BYTES:
        raise ValueError(
            f"figure {name!r} is {len(encoded):,} bytes as JSON; the limit is "
            f"{FIGURE_MAX_BYTES:,} bytes. Downsample the traces or split it into several figures."
        )
    figure = json.loads(encoded)
    data = figure["data"]
    if not isinstance(data, list) or not 1 <= len(data) <= FIGURE_MAX_TRACES:
        count = len(data) if isinstance(data, list) else type(data).__name__
        raise ValueError(
            f"figure {name!r}: data must be a list of 1..{FIGURE_MAX_TRACES} traces (got {count})"
        )
    allowed = ", ".join(FIGURE_TRACE_TYPES)
    for i, trace in enumerate(data):
        if not isinstance(trace, Mapping):
            raise ValueError(
                f"figure {name!r}: trace {i} must be a dict (got {type(trace).__name__}); "
                f"allowed types are {allowed}"
            )
        kind = trace.get("type", "scatter")
        if kind not in FIGURE_TRACE_TYPES:
            raise ValueError(
                f"figure {name!r}: trace {i} has type {kind!r}; allowed types are {allowed} "
                "(a missing type means scatter)"
            )
    return figure


# ------------------------------------------------------------------ handles


class Run:
    """One training job on one tracked agent, producing one version.

    ``log`` buffers and ``flush`` sends, so a training loop never waits on
    the network and never sees an exception from it. ``score`` records an
    eval on one behavior. ``finish`` flushes and closes the run.

    Has the ``progress``, ``log`` and ``finish`` shape that
    ``wai.TrainerCallback(run)`` calls on a Transformers or TRL trainer.
    """

    def __init__(
        self,
        tracked: Tracked,
        run_id: str,
        spec: RunSpec,
        *,
        flush_every: int = FLUSH_EVERY,
        flush_seconds: float = FLUSH_SECONDS,
    ):
        self.tracked = tracked
        self.id = run_id
        self.spec = spec
        self.version = spec.version
        self.status = "running"
        self.step = 0
        self.total_steps: int | None = None
        self.errors = 0
        self.scores: dict[str, Score] = {}
        self.notes: str | None = None
        self._flush_every = max(1, int(flush_every))
        self._flush_seconds = float(flush_seconds)
        self._buffer: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self._warned = False
        self._ci_warned: set[tuple[str, str]] = set()

    # ------------------------------------------------------------ logging

    def log(self, step: int | TrainPoint, **metrics: Any) -> None:
        """Record one point of the training curve: ``reward`` and ``kl`` for
        RL, ``loss`` for SFT and DPO; any other finite number is kept."""
        if isinstance(step, TrainPoint):
            point = step
        else:
            clean = {k: n for k, v in metrics.items() if (n := _number(v)) is not None}
            if not clean:
                return
            point = TrainPoint(step=int(step), **clean)
        with self._lock:
            self._buffer.append(point.wire())
            self.step = max(self.step, point.step)
            due = len(self._buffer) >= self._flush_every or (
                time.monotonic() - self._last_flush >= self._flush_seconds
            )
        if due:
            self.flush()

    def progress(self, step: int, total: int | None = None) -> None:
        """Where the run is; the trainer callback reads this from the trainer."""
        self.step = max(self.step, int(step))
        if total:
            self.total_steps = int(total)

    def flush(self) -> None:
        """Send buffered points now. Failures are counted, never raised."""
        with self._lock:
            batch, self._buffer = self._buffer[:MAX_BATCH], self._buffer[MAX_BATCH:]
            self._last_flush = time.monotonic()
        if not batch:
            return
        try:
            self.tracked._call("POST", f"/runs/{self.id}/train", batch)
        except PlatformError as e:
            self.errors += 1
            with self._lock:
                self._buffer = batch + self._buffer
            if not self._warned:
                self._warned = True
                log.warning(
                    "Could not send %d training points (%s); will retry on the next flush.",
                    len(batch),
                    e,
                )
            return
        if self._buffer:
            self.flush()

    # ------------------------------------------------------------ evals

    def score(self, behavior: str | Score, score: float | None = None, **fields: Any) -> Score:
        """Record this version's score on one behavior's held-out test.

        ``ci`` is the half-width of the 95% interval, ``n`` the number of
        held-out items. Score every behavior, not only the ones this run
        trained: the ones you did not train are the check.
        """
        if isinstance(behavior, Score):
            item = behavior
        else:
            if score is None:
                raise TypeError("score(behavior, score, ci=..., n=...) needs the score")
            item = Score(behavior=behavior, score=score, **fields)
        if item.ci is None and ("ci", item.behavior) not in self._ci_warned:
            self._ci_warned.add(("ci", item.behavior))
            log.warning(
                "score(%r) has no interval; pass ci=<half-width of the 95%% interval> so the "
                "platform can say whether the change is real.",
                item.behavior,
            )
        if ("n", item.behavior) not in self._ci_warned:
            if item.n is None:
                self._ci_warned.add(("n", item.behavior))
                log.warning(
                    "score(%r) has no n; pass n=<held-out items> so the verdict can say what "
                    "the number rests on.",
                    item.behavior,
                )
            elif item.n < 50:
                self._ci_warned.add(("n", item.behavior))
                log.warning(
                    "score(%r) rests on n=%d held-out items; under 50 the interval is too wide "
                    "to prove a gain of a few points (holdout_size() says how many you need).",
                    item.behavior,
                    item.n,
                )
        out = self.tracked._call("POST", f"/runs/{self.id}/evals", [item.wire()])
        recorded = (out.get("evals") or [item.wire()])[0]
        self.scores[item.behavior] = Score.model_validate(recorded)
        return self.scores[item.behavior]

    # ------------------------------------------------------------ lifecycle

    def finish(
        self,
        status: str = "evaluated",
        *,
        hours: float | None = None,
        gpu: str | None = None,
        cost_usd: float | None = None,
        steps: int | None = None,
        summary: Mapping[str, Any] | None = None,
        record: RunRecord | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Flush, then close the run with a status, what it cost, and the
        ``RunRecord`` (data, optimizer, eval, provenance) if it was not
        given when the run opened."""
        self.flush()
        patch: dict[str, Any] = {"status": status}
        if record is not None:
            item = record if isinstance(record, RunRecord) else RunRecord.model_validate(record)
            patch["record"] = item.wire()
        if hours is not None:
            patch["hours"] = float(hours)
        if gpu is not None:
            patch["gpu"] = gpu
        if cost_usd is not None:
            patch["costUsd"] = float(cost_usd)
        total = steps if steps is not None else (self.total_steps or self.step or None)
        if total is not None:
            patch["steps"] = int(total)
        if summary and _number(summary.get("train_loss")) is not None:
            patch["summary"] = {"train_loss": summary["train_loss"]}
        self.status = status
        return self.tracked._call("PATCH", f"/runs/{self.id}", patch)

    def note(self, markdown: str) -> None:
        """Put free-form markdown on this run. The Runs page renders it
        under the run record when the run is selected, so the reader gets
        what happened in the agent's words (what surprised you, what you
        would change) next to the numbers. At most 8192 chars; a second
        call replaces the first. Kept on ``run.notes``."""
        text = str(markdown)
        if len(text) > NOTES_MAX:
            raise ValueError(
                f"note() takes at most {NOTES_MAX:,} chars of markdown (got {len(text):,})"
            )
        self.tracked._call("PATCH", f"/runs/{self.id}", {"notes": text})
        self.notes = text

    def fail(self, error: str) -> dict[str, Any]:
        self.flush()
        self.status = "failed"
        return self.tracked._call(
            "PATCH", f"/runs/{self.id}", {"status": "failed", "error": str(error)[:400]}
        )

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None and self.status == "running":
            self.fail(f"{exc_type.__name__}: {exc}")
        elif self.status == "running":
            self.flush()

    @property
    def url(self) -> str:
        return f"https://withwhile.com/platform/runs?agent={self.tracked.id}"

    def __repr__(self) -> str:
        return (
            f"Run({self.id!r}, version={self.version!r}, status={self.status!r}, step={self.step})"
        )


class Tracked:
    """A handle to one agent as the platform tracks it. Made by ``track``.

    Not an agent: your framework runs the agent. This object reports what
    the agent is (model + harness), what you measure it on (``behavior``),
    what you trained (``run``) and what it does in production (``live``),
    and reads the decision back (``dashboard``, ``verdict``).
    """

    def __init__(
        self,
        id: str,
        *,
        name: str | None = None,
        model: str | None = None,
        harness: Harness | None = None,
        frontier: Frontier | None = None,
        api_key: str | None = None,
        transport: Transport | None = None,
    ):
        self.id = id
        self.name = name or id
        self.model = model
        self.harness = harness
        self.frontier = frontier
        self._api_key = api_key
        self._transport = transport
        self.record: dict[str, Any] | None = None

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        if self._transport is not None:
            return self._transport(method, path, body)
        return _request(method, path, api_key=_key(self._api_key), body=body)

    def register(self) -> dict[str, Any]:
        """Create or update the record. Safe to call every run."""
        body: dict[str, Any] = {"id": self.id, "name": self.name}
        if self.model:
            body["model"] = self.model
        if self.harness is not None:
            body["harness"] = self.harness.wire()
        if self.frontier is not None:
            body["frontier"] = self.frontier.wire()
        self.record = self._call("POST", "/agents", body)
        return self.record

    def behavior(self, behavior: str | Behavior, **fields: Any) -> Behavior:
        """Declare one thing you measure. See ``Behavior`` for the fields."""
        item = behavior if isinstance(behavior, Behavior) else Behavior(name=behavior, **fields)
        if item.reward_is_judge:
            log.warning(
                "behavior(%r): the judge is also the reward, so it cannot catch reward hacking; "
                "score with a different model or a verifier (reward_is_judge=False).",
                item.name,
            )
        body = item.wire()
        body.pop("name", None)
        out = self._call("PUT", f"/agents/{self.id}/behaviors/{item.name}", body)
        return Behavior.model_validate(out) if isinstance(out, dict) and out.get("name") else item

    def behaviors(self) -> list[Behavior]:
        rows = self._call("GET", f"/agents/{self.id}/behaviors").get("behaviors") or []
        return [Behavior.model_validate(r) for r in rows]

    def experiment(
        self,
        question: str | None = None,
        *,
        hypothesis: str | None = None,
        method: str | None = None,
        measure: str | None = None,
        decide: str | None = None,
        notes: str | None = None,
    ) -> Experiment | None:
        """Say what the runs on this agent are for, or read it back.

        With a ``question``, stores the experiment block (one per agent;
        a second call replaces it) and the Runs page renders it at the
        top as five labeled rows plus Notes: what you are asking,
        what you expect, how you train, how you measure and what result
        makes you promote. Every field is markdown of at most 4096 chars.
        With no arguments, returns the stored ``Experiment``, or ``None``
        when none was posted.
        """
        if question is None:
            if any(v is not None for v in (hypothesis, method, measure, decide, notes)):
                raise TypeError(
                    "experiment(question=..., ...) needs the question; "
                    "experiment() with no arguments reads the stored block"
                )
            try:
                out = self._call("GET", f"/agents/{self.id}/experiment")
            except PlatformError as e:
                if e.status == 404:  # literal: no experiment posted on this agent yet
                    return None
                raise
            return Experiment.model_validate(out)
        item = Experiment(
            question=question,
            hypothesis=hypothesis,
            method=method,
            measure=measure,
            decide=decide,
            notes=notes,
        )
        out = self._call("PUT", f"/agents/{self.id}/experiment", item.wire())
        if isinstance(out, dict) and out.get("question"):
            return Experiment.model_validate(out)
        return item

    def figure(
        self,
        name: str,
        fig: Any,
        *,
        caption: str | None = None,
        run: str | Run | None = None,
    ) -> Figure:
        """Post one Plotly figure as JSON, for the figures grid on the
        Runs page. Figures are illustration: the verdict on the page
        comes from the platform-scored evals (``run.score``), never from
        a figure.

        ``fig`` is a plotly ``Figure`` (read through ``to_plotly_json()``
        or ``to_dict()``, so plotly is never imported here) or a dict
        with ``data`` (a list of traces) and ``layout``. ``name`` is
        lowercase letters, digits and dashes, up to 40 chars, and names
        the slot: posting the same name again replaces the figure.
        ``caption`` (at most 1024 chars) is drawn above it; ``run`` ties
        it to one run on this agent. Checked here before any network
        call, with the API's limits: at most 200 KB of JSON, 1 to 50
        traces, trace types in scatter, bar and pie (the page ships
        plotly.js-basic). ``layout.images``, ``updatemenus``, ``sliders``
        and ``template`` are dropped, as the API drops them; titles,
        annotations, shapes and axes are kept.
        """
        figure = _figure_json(name, fig)
        if caption is not None and len(caption) > FIGURE_CAPTION_MAX:
            raise ValueError(
                f"figure {name!r}: caption is at most {FIGURE_CAPTION_MAX:,} chars "
                f"(got {len(caption):,})"
            )
        run_id = run.id if isinstance(run, Run) else run
        body: dict[str, Any] = {"figure": figure}
        if caption is not None:
            body["caption"] = caption
        if run_id is not None:
            body["run"] = run_id
        out = self._call("PUT", f"/agents/{self.id}/figures/{name}", body)
        if isinstance(out, dict) and out.get("name"):
            return Figure.model_validate(out)
        return Figure(name=name, caption=caption, run=run_id, figure=figure)

    def figures(self) -> list[Figure]:
        """Every figure posted on this agent, sorted by name."""
        rows = self._call("GET", f"/agents/{self.id}/figures").get("figures") or []
        return [Figure.model_validate(r) for r in rows]

    def run(
        self,
        version: str | RunSpec,
        *,
        flush_every: int = FLUSH_EVERY,
        flush_seconds: float = FLUSH_SECONDS,
        **fields: Any,
    ) -> Run:
        """Open a training run that will produce ``version``. See ``RunSpec``."""
        spec = version if isinstance(version, RunSpec) else RunSpec(version=version, **fields)
        if spec.base is None and self.model:
            spec.base = self.model
        if spec.harness is None and self.harness is not None:
            spec.harness = self.harness.version
        body = {"agent": self.id, **spec.wire()}
        out = self._call("POST", "/runs", body)
        return Run(self, out["id"], spec, flush_every=flush_every, flush_seconds=flush_seconds)

    def runs(self) -> list[dict[str, Any]]:
        return list(self._call("GET", f"/runs?agent={self.id}").get("runs") or [])

    def promote(self, version: str) -> dict[str, Any]:
        """Make ``version`` the served one. Usually the person's button."""
        return self._call("POST", f"/agents/{self.id}/promote", {"version": version})

    def live(self, day: str | date | LiveDay, **fields: Any) -> dict[str, Any]:
        """Report one day of traffic on the served version. See ``LiveDay``."""
        item = day if isinstance(day, LiveDay) else LiveDay.model_validate({"day": day, **fields})
        return self._call("POST", "/live", [{"agent": self.id, **item.wire()}])

    def dashboard(self, behavior: str | None = None) -> Dashboard:
        """The Runs screen as data: versions, train, deltas, judge, live, verdict."""
        q = f"?behavior={behavior}" if behavior else ""
        return Dashboard.model_validate(self._call("GET", f"/agents/{self.id}/dashboard{q}"))

    def verdict(self, behavior: str | None = None) -> Verdict:
        """Does the candidate beat the served version, and is it real? ``str()`` it."""
        return self.dashboard(behavior).verdict

    def __repr__(self) -> str:
        return f"Tracked({self.id!r}, model={self.model!r})"


# --------------------------------------------------------- your framework


class Described(BaseModel):
    """What ``describe`` reads off an agent object."""

    name: str
    model: str | None = None
    harness: Harness


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    for attr in ("name", "__name__"):
        value = getattr(tool, attr, None)
        if isinstance(value, str) and value:
            return value
    fn = getattr(tool, "function", None) or getattr(tool, "func", None)
    if fn is not None and fn is not tool:
        return _tool_name(fn)
    if isinstance(tool, Mapping):
        inner = tool.get("function")
        if isinstance(inner, Mapping) and inner.get("name"):
            return str(inner["name"])
        if tool.get("name"):
            return str(tool["name"])
    return type(tool).__name__


def _model_name(model: Any) -> str | None:
    if model is None:
        return None
    if isinstance(model, str):
        return model
    for attr in ("model_name", "name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(model)


def _text(value: Any) -> str | None:
    if value is None or callable(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        parts = [p for p in value if isinstance(p, str)]
        return "\n".join(parts) if parts else None
    return str(value)


def describe(agent: Any, *, name: str | None = None) -> Described:
    """Read name, model, instructions and tools off the agent object you
    already have, by attribute name, so no framework is imported here.

    Covers OpenAI Agents SDK (``name``, ``instructions``, ``tools``,
    ``model``), Pydantic AI (``name``, ``model``, system prompts), LangGraph
    prebuilt agents and Claude Agent SDK options (``system_prompt``,
    ``allowed_tools`` / ``tools``, ``model``), plus any mapping with those
    keys. Anything else gets a name from ``name=`` or the class.
    """
    if isinstance(agent, Mapping):
        mapping = agent

        def get(k: str) -> Any:
            return mapping.get(k)
    else:

        def get(k: str) -> Any:
            return getattr(agent, k, None)

    found_name = name or _text(get("name")) or type(agent).__name__
    instructions = None
    for key in ("instructions", "system_prompt", "system", "_system_prompts", "prompt"):
        instructions = _text(get(key))
        if instructions:
            break
    tools: list[Any] = []
    for key in ("tools", "allowed_tools", "_function_tools"):
        value = get(key)
        if value:
            tools = list(value.values()) if isinstance(value, Mapping) else list(value)
            break
    model = _model_name(get("model"))
    harness = Harness(instructions=instructions, tools=tools, model=model)
    return Described(name=found_name, model=model, harness=harness)


def track(
    agent: str | Any,
    *,
    name: str | None = None,
    model: str | None = None,
    harness: Harness | str | Mapping[str, Any] | None = None,
    frontier: Frontier | Mapping[str, Any] | None = None,
    api_key: str | None = None,
    transport: Transport | None = None,
    register: bool = True,
) -> Tracked:
    """Start tracking an agent on the platform and return its handle.

    Pass a name and describe it yourself, or pass the agent object from
    your framework and let ``describe`` read the model, instructions and
    tools off it. The harness fingerprint becomes the harness version, so
    a prompt edit shows up as a new version without anyone naming it.
    """
    if isinstance(agent, str):
        found = Described(name=agent, model=model, harness=Harness())
    else:
        found = describe(agent, name=name)
        model = model or found.model

    h: Harness | None
    if isinstance(harness, Harness):
        h = harness
    elif isinstance(harness, str):
        h = Harness(label=harness)
    elif isinstance(harness, Mapping):
        h = Harness.model_validate(dict(harness))
    elif isinstance(agent, str):
        h = None  # nothing known about the harness; leave the record's harness alone
    else:
        h = found.harness
    if h is not None and h.model is None and model:
        h.model = model

    f: Frontier | None
    if isinstance(frontier, Frontier) or frontier is None:
        f = frontier
    else:
        f = Frontier.model_validate(dict(frontier))

    tracked = Tracked(
        _slug(name or found.name),
        name=name or found.name,
        model=model,
        harness=h,
        frontier=f,
        api_key=api_key,
        transport=transport,
    )
    if register:
        tracked.register()
    return tracked


def _slug(name: str) -> str:
    out = "".join(c if c.isalnum() or c in "._-" else "-" for c in name.strip())
    out = out.strip("-._") or "agent"
    return out[:64]


def tracked_agents(api_key: str | None = None) -> list[TrackedInfo]:
    """Every tracked agent on the account."""
    out = _request("GET", "/agents", api_key=_key(api_key))
    return [TrackedInfo.model_validate(a) for a in out.get("agents") or []]


# ---------------------------------------------------------------------
# The rest of the platform: sign in, datasets, hosted training and
# serving. Loaded on first use so ``from whileai import platform``
# stays cheap and the engine is not imported for a login.
# ---------------------------------------------------------------------

_LAZY: dict[str, tuple[str, str]] = {
    "login": ("whileai.auth", "login"),
    "logout": ("whileai.auth", "logout"),
    "signup": ("whileai.auth", "signup"),
    "account": ("whileai.auth", "account"),
    "LoginError": ("whileai.auth", "LoginError"),
    "push": ("whileai.simulations.ingest.platform", "push_rows"),
    "push_file": ("whileai.simulations.ingest.platform", "push_file"),
    "pull": ("whileai.simulations.ingest.platform", "pull"),
    "datasets": ("whileai.simulations.ingest.platform", "datasets"),
    "catalog": ("whileai.simulations.ingest.platform", "catalog"),
    "preview": ("whileai.simulations.ingest.platform", "preview"),
    "publish": ("whileai.simulations.ingest.platform", "publish"),
    "unpublish": ("whileai.simulations.ingest.platform", "unpublish"),
    "agents": ("whileai.simulations.ingest.platform", "agents"),
    "register_agent": ("whileai.simulations.ingest.platform", "register_agent"),
    "hf_publish": ("whileai.simulations.ingest.platform", "hf_publish"),
    "import_hf": ("whileai.simulations.ingest.platform", "import_hf"),
    "train": ("whileai.simulations.training", "train"),
    "training_run": ("whileai.simulations.training", "training_run"),
    "runs": ("whileai.simulations.training", "list_runs"),
    "get_run": ("whileai.simulations.training", "get_run"),
    "serve": ("whileai.simulations.training", "serve"),
    "unserve": ("whileai.simulations.training", "unserve"),
    "models": ("whileai.simulations.training", "models"),
    "reward_model": ("whileai.simulations.training", "reward_model"),
    "TrainerCallback": ("whileai.simulations.training", "TrainerCallback"),
    "TrainingRun": ("whileai.simulations.training", "TrainingRun"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'whileai.platform' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # the lazy names above, visible to editors and mypy
    from .auth import LoginError, account, login, logout, signup
    from .simulations.ingest.platform import (
        agents,
        catalog,
        datasets,
        hf_publish,
        import_hf,
        preview,
        publish,
        pull,
        push_file,
        register_agent,
        unpublish,
    )
    from .simulations.ingest.platform import push_rows as push
    from .simulations.training import (
        TrainerCallback,
        TrainingRun,
        get_run,
        models,
        reward_model,
        serve,
        train,
        training_run,
        unserve,
    )
    from .simulations.training import list_runs as runs


__all__ = [
    "DEFAULT_PLATFORM_URL",
    "Behavior",
    "Dashboard",
    "Data",
    "Delta",
    "Described",
    "EvalSetup",
    "Experiment",
    "Figure",
    "Frontier",
    "Harness",
    "Judge",
    "LiveDay",
    "LiveSeries",
    "LoginError",
    "Optimizer",
    "PlatformError",
    "Provenance",
    "Run",
    "RunRecord",
    "RunSpec",
    "Score",
    "Tracked",
    "TrackedInfo",
    "TrainCurve",
    "TrainPoint",
    "TrainerCallback",
    "TrainingRun",
    "Verdict",
    "VersionScore",
    "account",
    "agents",
    "catalog",
    "datasets",
    "describe",
    "get_run",
    "hf_publish",
    "import_hf",
    "login",
    "logout",
    "models",
    "platform_url",
    "preview",
    "publish",
    "pull",
    "push",
    "push_file",
    "register_agent",
    "reward_model",
    "runs",
    "serve",
    "signup",
    "track",
    "tracked_agents",
    "train",
    "training_run",
    "unpublish",
    "unserve",
]
