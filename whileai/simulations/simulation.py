"""``simulate()``: the public entry point.

Knob resolution lives in :mod:`.run.config`, spec loading in
:mod:`.run.spec`, row helpers in :mod:`.run.rows`, and the engine (inputs,
build, scheduler loop, finish) in :mod:`.run.engine`. This module keeps
the signature, the docstring, and the names older code imported from here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import resolve_backend as _resolve_backend
from .data import SimulationData
from .defaults import DEFAULT_BUDGET
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
from .run.engine import Run, tier_mix_of
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
from .tools import schemas as _tool_schemas

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
    budget: int | None = DEFAULT_BUDGET,
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
    hard_share: float | None = None,
    patience: str | None = None,
    **passed: Any,
) -> SimulationData:
    """Generate situations for an agent, roll them out, and return the rows.

    Reach for it first: it is the run everything else reads. Give it the
    agent and it writes a grid of human asks (ordinary, vague, complex,
    adversarial), plays each one against the agent, and returns a
    ``SimulationData``: ``rows()`` (one row per rollout, with the prompt,
    the tool calls, the final reply and its lineage), ``warnings``
    (plain-words notes, each naming the call that changes it),
    ``report()``, ``search`` (how the budget was spent), ``pass_at``
    once graded, and ``save(path)``. Nothing is graded unless you ask:
    pass ``grade=True`` for the deterministic conduct grade, a callable
    ``grader=`` to score inside the loop (the only in-simulate score
    hook), or grade later with ``data.grade(...)`` or ``grade()``.

    The agent and the budget:

    * ``agent``: a callable ``message -> trajectory``, played single-turn
      (one message in, one trajectory out); or a backend object
      (``wai.OpenAI("gpt-4.1-mini")``) or spec string
      (``"openai:gpt-4.1-mini"``), which the SDK plays multi-turn from
      ``tools`` and ``system_prompt``. Leave it ``None`` to play the
      model ``configure(agent=)`` set, else the model While hosts, from
      ``tools``, ``system_prompt`` (alias ``policy``) and ``backend``
      (its model); ``spec=`` is the third way in.
    * ``budget``: rows the run may produce, 1000 by default, a per-run cap
      when ``runs`` is above 1 (``runs=3, budget=100`` returns up to 300
      rows, and ``report()["budget_per_run"]`` carries the cap).
      ``time_budget`` is the same cap in seconds.
    * ``mode``: ``"explore"`` (default) spends the budget on new coverage.
      ``"rl"`` gives every ask several repeats, so pass rates and RL
      groups exist, and defaults ``repeat_policy`` to ``"successive"``,
      which stops early on unanimous asks; ``"fixed"`` gives every ask all
      k repeats.
    * ``situations`` (N, distinct worlds), ``phrasings`` (n, wordings of
      one world; alias ``requests_per_situation``) and ``repeats`` (k,
      independent runs of one wording; alias ``rollouts_per_request``)
      are three independent counts. Do not collapse them. Under
      ``mode="rl"`` ``repeats`` is a floor, not a count: dynamic sampling
      (Yu et al. 2025 (DAPO), arXiv:2503.14476) re-rolls uneven groups, so
      some asks end with more than k rollouts and ``pass_at`` reports the
      smallest k. Follow-ups branch on the run. ``unique_situations=True``
      (alias ``unique``) keeps picking new worlds (n=1, k=1 unless you set
      them). With
      ``situations=N`` the run stops once all N have their rollouts
      (``stopped_because="situations_exhausted"``), whatever ``budget``
      still allows; a budget above ``situations x phrasings x repeats`` is
      not spent.
    * ``concurrency`` (32 parallel rollouts), ``seed``, ``sampling``,
      ``temperature``, ``timeout``, ``logprobs``, ``fault_rate``,
      ``max_turns`` / ``avg_turns`` (model-backed agents only): each is
      ``None`` unless you set it, and a misspelled keyword is a
      ``TypeError``, never silently ignored. Writer completions are
      ``advanced["completions_per_request"]``; seed openers are
      ``advanced["seed_prompts"]``.

    What the situations come from:

    * ``simulator``: the situation writer. ``"hosted"`` is the default
      written out, the same as leaving it unset; ``False`` is the offline
      template writer, no key needed. ``user_model`` plays the simulated
      person (a backend spec; ``None`` means the writer's model, the
      agent's own by default).
    * ``seeds``: opening asks the writer keeps and varies. Every seed is
      run and becomes at least one situation: ``situations`` is sized up
      to ``len(seeds)`` when you pass a smaller number, and the search
      never spends a seed's slot on an ask it wrote itself. The one thing
      that can still drop a seed is ``budget``, which pays for
      ``len(seeds) * repeats`` rows before anything else; when it cannot,
      ``warnings`` says which seeds were dropped and
      ``search["seeds_dropped"]`` lists them before rolling out. With a
      callable agent whose world has real ids (order numbers, account
      names), put those ids in the seeds or the tool descriptions, or the
      writer invents ids and every rollout is "not found". Seeds are asks
      to build a run around, not the eval set: to check that a fixed list
      of asks all ran and how each scored, use ``evaluate(eval_set=asks)``.
    * ``traces``: rows or a JSONL path of production traces. The grid
      then aims at the tools, faults and world states those traces show
      instead of the whole space (without it the grid comes from the
      agent's tools and policy alone, a cold start), and any generated
      row that near-copies a source trace is dropped, so held-out traces
      stay out of training. Traces reproduce world-visible situations. A
      failure that lives in how a reply is worded (an unsupported claim,
      an estimate not labelled as one, two questions where one was asked
      for) has no world-visible trigger, so traces alone cannot aim at
      it; put a grader in the loop for those.
    * ``hard_share``: the difficulty dial, the share of situations drawn
      from the ambiguous, boundary and adversarial tiers, 0.40 by
      default, where a base fails most often. ``search["tier_mix"]``
      reports the share asked for and the share drawn;
      ``dimensions={"stance": [...]}`` pins one axis and keeps the other
      axes of the grid.
    * ``tasks``: a previous run (a ``SimulationData``, its rows, or its
      JSONL path) whose task set is replayed instead of drawing a new
      one. Every distinct prompt is rolled out again, on its own
      ``scenario_id`` and ``scenario_dimensions``, under the same faults
      and world state, and nothing else is generated. An unpinned re-run
      draws by seed and, above ``concurrency: 1``, by completion order, so
      it shares only part of its tasks with the first and ``compare_runs``
      drops the rest; pinning is how an A/B (a prompt edit, a model swap,
      another seed) keeps every pair. k comes from this call's
      ``repeats`` when given, otherwise from the pinned run (the most
      rollouts any of its prompts has), never from this call's ``mode``,
      so ``pass_at`` reports the same k on both sides. The run stops when
      every pinned prompt has its rollouts
      (``stopped_because="tasks_done"``) or the budget is spent.
    * ``runs``: replay the same task set that many times in one call and
      stamp ``lineage.eval_run`` (0, 1, 2, ...) on every row, which is what
      ``delta_report`` needs before it will call a change real (Lambert 2025,
      chapter Evaluation and its evaluation-variance appendix: one evaluation
      is a draw, three give a standard deviation). ``simulate(tasks=base,
      runs=3)`` is the usual form; without ``tasks=`` the first run draws the
      task set (from ``seeds=`` when given) and the rest replay it. Between
      runs nothing changes but the agent's own sampling (same tasks, faults,
      world state and seed), so a deterministic agent gives identical runs and
      a zero re-run band. All rows come back in one ``SimulationData``
      (``output=`` holds them all); ``search["eval_runs"]`` lists the rows and
      stop reason per run, and ``eval_variance(data.rows())`` splits by
      ``eval_run`` on its own. Replayed rows keep the writer of the run they
      replay on ``writer_model`` and say ``lineage.replayed_from_run``, so
      ``delta_report`` on two runs of one call sees one writer.

    What steers the search and answers the tools:

    * ``grader``: a callable judge. Its verdict steers the search the way
      a tool fault already does: a row the grader failed is re-rolled and
      its ask is mutated into new ones, so the budget moves toward what
      the grader catches, not only toward broken tools. A grader that
      fails reply-form rules is exactly the signal traces cannot give
      (measured with a 12-rule grader: every rule with a tool-result
      trigger was reproduced and every rule about the reply's wording was
      not), and without a grader there is no verdict to steer by, so
      there is no switch to set: the grader is the switch. A graded
      failure is a reward under 0.5 (a 0 from a 0/1 judge, a failed
      verifier, a rubric below half); markers ride along on the row but
      do not aim on their own, since their direction differs per marker.
      ``search["mutation_aims"]`` counts the parents and the mutated rows
      per aim, ``world_fault`` and ``graded_failure``. To grade beside the
      loop and still steer by tool faults alone, pass
      ``advanced={"mutate_graded_failures": False}``.
    * ``execute``: your world, a function ``(tool, arguments) -> result``
      that answers every tool call for real, against your repo, database
      or service. Without it the mock world answers, which fits
      record-shaped tools and not code. Scheduled faults still apply
      first. ``whileai.simulations.generate.agents.current_rollout`` is a
      thread-local set before each rollout with ``prompt``,
      ``rollout_index`` and ``seed``, so ``execute`` can tell which run it
      is answering.
    * ``patience``: how long the simulated person keeps answering the
      agent's questions. ``"normal"`` (the default) always tries to answer
      the first question, and from the second on may walk away (35% on
      the second, 60% on each after that, drawn per thread so a seeded
      run reproduces); at any question the person may also leave when it
      asks for something they could not or would not know. ``"short"``
      walks away sooner (60% then 90%); ``"endless"`` never walks away, so
      the person answers every question until the depth cap and no rubric
      criterion about asking can fail. The odds are a default, not a
      measurement: to ground them, fit a Kaplan-Meier hazard per question
      index on source traces and set the levels from it. A row the person
      left carries ``ended_by="user_left"`` and ends on the agent's
      question; ``search["ended_on_question"]`` is ``{"share", "n",
      "user_left"}``: of ``n`` rows, the share that ended on a question
      and how many of those the person left.
    * ``scaffold``: generation-only guidance appended to the system prompt
      of the model-backed teacher during rollout (and to the scene
      writer). It never enters ``profile.policy``, so exports and evals
      stay on the plain policy; it is ignored for callable agents.
      Measured to help some agents and hurt others, so there is no
      default: configure it per agent.
    * ``reproducible``: a seeded run is already reproducible bit-for-bit
      at ``concurrency: 1``, apart from timing fields and per-invocation
      identity (with ``grader=`` every row's ``lineage.scoring_run_id``
      names that one scoring pass, a fresh id per call; pass ``run_id=``
      to ``run_judge`` to pin it). ``True`` makes it so at any
      concurrency: each batch of rollouts finishes, and every verdict of
      the batch lands, before the next is chosen, so every round sees
      the same state. A slow rollout holds its batch, so uneven latency
      costs throughput, and it needs the clock off, since a clock stop
      lands wherever the run happens to be. Without the flag, which rows
      land before the cap depends on thread timing.

    What every row records. ``sampling`` (``temperature``, ``max_tokens`` and
    ``model`` as the model backend resolved them), because a result is only
    comparable with its sampling settings on record (Lambert 2025, chapter
    Evaluation); a callable agent samples however it samples, so its rows
    carry ``sampling: None`` unless you pass ``sampling={...}``, which is
    recorded as given. Three models can take part, the agent (``agent=`` or
    ``backend=``), the situation writer (``simulator=``) and the simulated
    user (``user_model=``), and every row names all three next to
    ``model_version``: ``writer_model``, ``user_model``, and
    ``judge_meta.model`` once graded. When the agent model also wrote the
    situations or played the user, the run's ``degraded`` list carries
    ``same_model`` and ``warnings`` says which call separates them, since
    training on a model's own unfiltered output teaches it its own habits
    (Lambert 2025, chapter Synthetic Data and Distillation).

    ```python
    import whileai.simulations as wai

    TOOLS = [{"type": "function", "function": {
        "name": "lookup_order",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}}]
    agent = wai.seeded_agent(TOOLS)  # an offline demo agent, no key
    base = wai.simulate(agent, tools=TOOLS, simulator=False, seed=0,
                        mode="rl", repeats=4, budget=32)
    rerun = wai.simulate(agent, tools=TOOLS, simulator=False, seed=0,
                         tasks=base, mode="rl")  # same tasks, k=4 inherited
    print(base.stopped_because, len(base.rows()), base.warnings)
    ```
    """
    n_runs = int(runs)
    if n_runs < 1:
        raise ValueError("runs= is how many times to replay the task set, 1 or more")
    # A backend object (``wai.OpenAI("gpt-4.1-mini")``) is resolved here the
    # way ``configure(agent=...)`` resolves it: its spec string goes down
    # the same road as ``agent="openai:gpt-4.1-mini"``, and a key given on
    # it is kept for its provider (#472).
    agent = _resolve_backend(agent, kwarg="agent")
    backend = _resolve_backend(backend, kwarg="backend")
    # Named knobs travel the same road as before (``advanced`` / aliases),
    # so nothing downstream changes; they are in the signature to be seen.
    tools = _tool_schemas(tools)
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
        ("hard_share", hard_share),
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


def _stamp_eval_run(rows: list[dict], index: int, *, replayed_from: int | None = None) -> None:
    for row in rows:
        lineage = row.get("lineage")
        if not isinstance(lineage, dict):
            lineage = {}
        lineage["eval_run"] = index
        if replayed_from is not None:
            # the run whose task set this run replayed; writer_model on
            # the row stays the writer of that run
            lineage["replayed_from_run"] = replayed_from
        row["lineage"] = lineage


def _repeat_runs(agent: Any, n_runs: int, kwargs: dict[str, Any]) -> SimulationData:
    """``simulate(runs=N)``: the same task set N times, one result.

    Each run is a full ``Run`` on the same config, ``budget`` included
    (it is a per-run cap, so the call returns up to ``N x budget`` rows
    and ``report()["budget_per_run"]`` says so); runs after the first
    replay the first run's task set when none was pinned. Rows are
    stamped ``lineage.eval_run`` (and ``lineage.replayed_from_run = 0``
    on the replays) and gathered on the first run's ``SimulationData``,
    which is written to ``output=`` once, at the end, so the file holds
    every run. ``degraded`` and ``warnings`` are the union over runs;
    ``search["tier_mix"]`` is recounted over every run's rows with a
    ``per_run`` breakdown, so it describes what the call returned and
    not run 0 alone.
    """
    output = kwargs.pop("output", None)
    first: SimulationData | None = None
    per_run: list[dict[str, Any]] = []
    for index in range(n_runs):
        run_kwargs = dict(kwargs)
        replayed_from: int | None = None
        if first is not None and run_kwargs.get("tasks") is None:
            run_kwargs["tasks"] = first
            replayed_from = 0
            # The first run drew the task set from these; a replay takes the
            # set as drawn, so the seeds must not be handed over again (#375).
            for drawn in ("seeds", "seed_prompts", "extra_situations", "situations"):
                run_kwargs.pop(drawn, None)
        data = Run(resolve_run_config(agent, **run_kwargs)).run()
        _stamp_eval_run(data.trajectories, index, replayed_from=replayed_from)
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
        # a flag without its line would be a flag nobody can act on
        first.warnings.extend(w for w in data.warnings if w not in first.warnings)
    assert first is not None
    first.search["eval_runs"] = {"runs": n_runs, "per_run": per_run}
    first.coverage["runs"] = n_runs
    mix = first.search.get("tier_mix")
    if isinstance(mix, dict) and n_runs > 1:
        # run 0 counted its own rows; the call returns every run's
        requested = float(mix.get("hard_share_requested") or 0.0)
        by_run: dict[int, list[dict]] = {}
        for row in first.trajectories:
            by_run.setdefault(int((row.get("lineage") or {}).get("eval_run", 0)), []).append(row)
        total = tier_mix_of(first.trajectories, requested)
        mix.update(total)
        mix["per_run"] = [
            {"eval_run": i, **tier_mix_of(rows, requested)} for i, rows in sorted(by_run.items())
        ]
    if output:
        first.save(str(output), meta=True)
    return first
