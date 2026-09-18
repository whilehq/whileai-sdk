"""``simulate()``: the public entry point.

Knob resolution lives in :mod:`.run.config`, spec loading in
:mod:`.run.spec`, row helpers in :mod:`.run.rows`, and the engine (inputs,
build, scheduler loop, finish) in :mod:`.run.engine`. This module keeps
the signature, the docstring, and the names older code imported from here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .data import SimulationData
from .generate.scenarios import SEARCH_ARMS, reallocate_search_arms
from .run.config import HUNG_SLOT_S as _HUNG_SLOT_S  # noqa: F401
from .run.config import SATURATION_CAP as _SATURATION_CAP  # noqa: F401
from .run.config import (  # noqa: F401
    _merge_advanced,
    _parse_situations_arg,
    resolve_run_config,
    resolve_topology,
    writer_spec_for,
)
from .run.engine import Run
from .run.rows import (  # noqa: F401
    _collect_finished,
    _prompt_arm,
    _row_conversation,
    _situation_key_from_meta,
    _stratified_prompts,
    _usable_rollout,
    mutation_worthy,
    record_coverage,
    row_cell_key,
)
from .run.spec import (  # noqa: F401
    _looks_like_spec_path,
    _read_spec_file,
    _spec_extra_text,
    _spec_from_path,
    apply_spec,
    backend_spec,
    kind_from_spec,
)

_SEARCH_ARMS = dict(SEARCH_ARMS)
_reallocate = reallocate_search_arms
# Underscore spellings older imports and tests still use.
_apply_spec = apply_spec
_backend_spec = backend_spec
_cell_key = row_cell_key
_kind_from_spec = kind_from_spec
_mutation_worthy = mutation_worthy
_record_coverage = record_coverage

__all__ = ["resolve_topology", "simulate", "writer_spec_for"]


def simulate(
    agent: Any = None,
    *,
    spec: Any = None,
    tools: list[dict] | None = None,
    system_prompt: str | None = None,
    budget: int | None = 1000,
    time_budget: float | None = None,
    until: str = "compute",
    mode: str = "explore",
    situations: int | None = None,
    requests_per_situation: int | None = None,
    rollouts_per_request: int | None = None,
    unique_situations: bool = False,
    reproducible: bool = False,
    grade: bool = False,
    llm_grade: bool = False,
    traces: Any = None,
    grader: Any = None,
    rubric: str | None = None,
    strategy: str = "auto",
    seeds: list | None = None,
    scaffold: str | None = None,
    execute: Callable | None = None,
    output: str | None = None,
    tasks: Any = None,
    runs: int = 1,
    advanced: dict | None = None,
    repeats: int | None = None,
    phrasings: int | None = None,
    repeat_policy: str | None = None,
    concurrency: int | None = None,
    simulator: Any = None,
    user_model: Any = None,
    backend: Any = None,
    seed: int | None = None,
    sampling: dict | None = None,
    max_turns: int | None = None,
    avg_turns: float | None = None,
    fault_rate: float | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    logprobs: bool | None = None,
    patience: str | None = None,
    **passed: Any,
) -> SimulationData:
    """Inspect an agent, generate situations, and roll them out.

    The knobs most runs touch, in the signature so an editor shows them:
    ``repeats`` (rollouts per ask, k), ``phrasings`` (wordings per
    situation, n), ``repeat_policy`` (``"fixed"`` gives every ask all k
    repeats; ``mode="rl"`` defaults to ``"successive"``, which stops
    early on unanimous asks), ``concurrency`` (parallel rollouts, 32),
    ``simulator`` (the situation writer; ``False`` is the offline
    template writer, no key, and ``"hosted"`` is the default written out,
    the same as leaving it unset), ``user_model``, ``backend`` (the agent's
    model when ``agent=`` is not a callable), ``seed``, ``sampling``,
    ``max_turns`` / ``avg_turns`` (model-backed agents only; a callable
    agent is played single-turn: one message in, one trajectory out),
    ``fault_rate``, ``temperature``, ``timeout``, ``logprobs``. Each
    is ``None`` unless you set it, and a misspelled keyword is a
    ``TypeError``, never silently ignored. ``seeds=`` is a list of
    opening asks the writer keeps and varies; with a callable agent
    whose world has real ids (order numbers, account names), put those
    ids in the seeds or the tool descriptions, or the writer invents
    ids and every rollout is "not found".

    Every seed is run. Each one becomes at least one situation:
    ``situations`` is sized up to ``len(seeds)`` when you pass a smaller
    number, and the search never spends a seed's slot on an ask it wrote
    itself. The one thing that can still drop a seed is ``budget``,
    which pays for ``len(seeds) * repeats`` rows before anything else;
    when it cannot, the run says which seeds it dropped in ``warnings``
    and lists them in ``search["seeds_dropped"]`` before rolling out.
    Seeds are asks to build a run around, not the eval set: to check
    that a fixed list of asks all ran and how each scored, use
    ``evaluate(eval_set=asks)``.

    Input is an intent or an agent: ``system_prompt`` alone, ``tools``
    plus a prompt, or ``spec=``. Search writes a grid of human requests
    (ordinary, vague, complex, adversarial) and a spread of agent replies.
    It spends ``budget`` rows and ``time_budget`` seconds on new coverage.
    No default ``reward``. Pass ``grade=True`` for the deterministic
    conduct grade, or grade later with ``grade()``. A callable
    ``grader=`` is the only in-simulate score hook.

    ``traces=`` (rows or a JSONL path of production traces) aims the
    covering grid at the behaviors those traces show instead of the whole
    space, and drops any generated row that near-copies a source trace,
    so held-out traces stay out of training. Without it the grid comes
    from the agent's tools and policy alone (cold start).

    ``execute=`` is the caller's world: a function ``(tool, arguments) ->
    result`` that answers every tool call for real, against their repo,
    database, or service. Without it the mock world answers, which fits
    record-shaped tools and not code. Scheduled faults still apply first.
    ``whileai.simulations.generate.agents.current_rollout`` is a
    thread-local set before each rollout with ``prompt``, ``rollout_index``
    and ``seed``, so ``execute`` can tell which run it is answering.

    Three models can take part: the agent (``agent=`` / ``backend=``), the
    situation writer (``simulator=``), and the simulated user
    (``user_model=``, a backend spec; ``None`` means the writer's model, the
    agent's own by default). Every row records all three next to
    ``model_version``: ``writer_model``, ``user_model``, and
    ``judge_meta.model`` once graded. When the agent model also wrote the
    situations or played the user, the run's ``degraded`` list carries
    ``same_model`` and ``warnings`` says which call separates them.
    Training on a model's own unfiltered output teaches it its own habits
    (rlhf-book ch. 12), so the row says who wrote what.

    ``patience=`` is how long the simulated person keeps answering the
    agent's questions. ``"normal"`` (the default) always tries to answer
    the first question, and from the second question on may walk away
    (35% on the second, 60% on each after that, drawn per thread so a
    seeded run reproduces); at any question the person may also leave
    when it asks for something they could not or would not know.
    ``"short"`` walks away sooner (60% then 90%); ``"endless"`` never
    walks away, which is what every run did before this knob: the
    person answered every question until the depth cap, so no rubric
    criterion about asking could fail. The odds are a default, not a
    measurement: to ground them, fit a Kaplan-Meier hazard per question
    index on source traces and set the levels from it. A row the person
    left carries ``ended_by="user_left"`` and ends on the agent's
    question; ``search["ended_on_question"]`` is ``{"share", "n",
    "user_left"}``: of ``n`` rows, the share that ended on a question and
    how many of those the person left.

    ``scaffold=`` is generation-only guidance appended to the system prompt
    of the MODEL-BACKED teacher during rollout (and to the scene writer).
    It never enters ``profile.policy``, so exports and evals stay on the
    plain policy; it is ignored for user-supplied callable agents. Measured
    to help some agents and hurt others. Configure per agent, no default.

    Every row says how it was sampled under ``sampling``: ``temperature``,
    ``max_tokens`` and ``model``, as the model backend resolved them
    (rlhf-book ch. 16: a result is only comparable with its sampling
    settings on record). A callable agent samples however it samples, so
    its rows carry ``sampling: None`` unless you pass ``sampling={...}``,
    which is recorded on every row as given.

    Variation is three independent counts. Do not collapse them.
    ``situations`` (N) is distinct worlds. ``requests_per_situation`` /
    ``phrasings`` (n) is different human wordings of one world.
    ``rollouts_per_request`` / ``repeats`` (k) is independent agent runs of
    the same wording. Follow-ups branch on that run.
    ``unique_situations=True`` keeps picking new worlds (n=1, k=1 unless you
    set them). Silent aliases: ``n`` / ``phrasings`` for n, ``repeats`` /
    ``rollouts_per_prompt`` for k, ``unique`` for unique_situations,
    ``policy`` for system_prompt. Writer completions are
    ``advanced["completions_per_request"]``. Seed openers are
    ``advanced["seed_prompts"]``.

    ``tasks=`` re-runs a previous run's task set instead of drawing a new
    one: pass that run (``SimulationData``), its rows, or its JSONL path.
    Every distinct prompt is rolled out again, ``repeat_count`` times,
    on its own ``scenario_id`` and ``scenario_dimensions`` and under the
    same faults and world state, and nothing else is generated.

    ``repeat_count`` is k as *this* call resolves it — from this call's
    ``repeats`` / ``rollouts_per_request`` when given, otherwise from the
    pinned run (the most rollouts any of its prompts has), never from
    this call's ``mode`` preset. So a base run made with ``repeats=4``
    and re-run as ``simulate(..., tasks=base)`` comes back at k=4 and
    ``pass_at`` reports the same k on both sides; pass ``repeats=`` to
    re-run at a different k on purpose::

        base = wai.simulate(agent, tools=TOOLS, system_prompt=P,
                            mode="rl", repeats=4)
        rerun = wai.simulate(agent, tools=TOOLS, system_prompt=EDITED,
                             tasks=base, mode="rl")  # k=4, inherited

    ``runs=`` replays the same task set that many times in one call and
    stamps ``lineage.eval_run`` (0, 1, 2, ...) on every row, which is
    what ``delta_report`` needs before it will call a change real
    (rlhf-book ch. 16 and appendix C: one evaluation is a draw, three
    give a standard deviation). ``simulate(tasks=base, runs=3)`` is the
    usual form; without ``tasks=`` the first run draws the task set and
    the rest replay it. Between runs nothing changes but the agent's own
    sampling: same tasks, same faults, same world state, same seed, so a
    deterministic agent gives identical runs and a zero re-run band. The
    rows of every run come back in one ``SimulationData`` (``output=``
    holds them all); ``search["eval_runs"]`` lists the rows and stop
    reason per run, and ``eval_variance(data.rows())`` splits by
    ``eval_run`` on its own.

    A run
    otherwise draws its tasks from the grid by seed and, above
    ``concurrency: 1``, by completion order, so a re-run shares only part
    of its tasks with the first and ``compare_runs`` drops the rest;
    pinning is how an A/B (a prompt edit, a model swap, another seed)
    keeps every pair. The run stops when every pinned prompt has its
    rollouts (``stopped_because="tasks_done"``) or the budget is spent.

    A seeded run is reproducible bit-for-bit at ``concurrency: 1``,
    apart from timing fields and per-invocation identity: with
    ``grader=`` every row's ``lineage.scoring_run_id`` names that one
    scoring pass (a fresh id per call, like a timestamp), so two runs
    differ on that key and on nothing else. Pass ``run_id=`` to
    ``run_judge`` to pin it.
    ``reproducible=True`` makes it so at any concurrency: each batch of
    rollouts finishes, and every verdict of the batch lands, before the
    next is chosen, so results are consumed in submission order and
    every round sees the same state. ``concurrency: 1`` schedules the
    same way without the flag, so two same-seed serial runs in one
    process return the same rows. Same seed, same concurrency, same
    agent gives the same rows; a slow rollout holds its batch, so
    uneven latency costs throughput. It needs the clock off, since a
    clock stop lands wherever the run happens to be.
    Without the flag, which rows land before the cap depends on thread
    timing.
    """
    n_runs = int(runs)
    if n_runs < 1:
        raise ValueError("runs= is how many times to replay the task set, 1 or more")
    # Named knobs travel the same road as before (``advanced`` / aliases),
    # so nothing downstream changes; they are in the signature to be seen.
    for _name, _val in (
        ("repeats", repeats),
        ("phrasings", phrasings),
        ("repeat_policy", repeat_policy),
        ("concurrency", concurrency),
        ("simulator", simulator),
        ("user_model", user_model),
        ("backend", backend),
        ("seed", seed),
        ("sampling", sampling),
        ("max_turns", max_turns),
        ("avg_turns", avg_turns),
        ("fault_rate", fault_rate),
        ("temperature", temperature),
        ("timeout", timeout),
        ("logprobs", logprobs),
        ("patience", patience),
    ):
        if _val is not None:
            passed[_name] = _val
    kwargs: dict[str, Any] = dict(
        spec=spec,
        tools=tools,
        system_prompt=system_prompt,
        budget=budget,
        time_budget=time_budget,
        until=until,
        mode=mode,
        situations=situations,
        requests_per_situation=requests_per_situation,
        rollouts_per_request=rollouts_per_request,
        unique_situations=unique_situations,
        reproducible=reproducible,
        grade=grade,
        llm_grade=llm_grade,
        traces=traces,
        grader=grader,
        rubric=rubric,
        strategy=strategy,
        seeds=seeds,
        scaffold=scaffold,
        execute=execute,
        output=output,
        tasks=tasks,
        advanced=advanced,
        passed=passed,
    )
    if n_runs == 1:
        return Run(resolve_run_config(agent, **kwargs)).run()
    return _repeat_runs(agent, n_runs, kwargs)


def _stamp_eval_run(rows: list[dict], index: int) -> None:
    for row in rows:
        lineage = row.get("lineage")
        if not isinstance(lineage, dict):
            lineage = {}
        lineage["eval_run"] = index
        row["lineage"] = lineage


def _repeat_runs(agent: Any, n_runs: int, kwargs: dict[str, Any]) -> SimulationData:
    """``simulate(runs=N)``: the same task set N times, one result.

    Each run is a full ``Run`` on the same config; runs after the first
    replay the first run's task set when none was pinned. Rows are
    stamped ``lineage.eval_run`` and gathered on the first run's
    ``SimulationData``, which is written to ``output=`` once, at the end,
    so the file holds every run.
    """
    output = kwargs.pop("output", None)
    first: SimulationData | None = None
    per_run: list[dict[str, Any]] = []
    for index in range(n_runs):
        run_kwargs = dict(kwargs)
        if first is not None and run_kwargs.get("tasks") is None:
            run_kwargs["tasks"] = first
        data = Run(resolve_run_config(agent, **run_kwargs)).run()
        _stamp_eval_run(data.trajectories, index)
        per_run.append({"rows": len(data.trajectories), "stopped_because": data.stopped_because})
        if first is None:
            first = data
            continue
        first.trajectories.extend(data.trajectories)
        first.elapsed_seconds += data.elapsed_seconds
        first.rollout_seconds += data.rollout_seconds
        first.scenario_generation_seconds += data.scenario_generation_seconds
        first.row_seconds.extend(data.row_seconds)
        first.degraded.extend(d for d in data.degraded if d not in first.degraded)
    assert first is not None
    first.search["eval_runs"] = {"runs": n_runs, "per_run": per_run}
    if output:
        first.save(str(output), meta=True)
    return first
