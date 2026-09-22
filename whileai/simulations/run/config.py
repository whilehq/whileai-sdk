"""Knob resolution for ``simulate()``.

Every public parameter, silent alias and ``advanced`` key is read here,
validated here, and lands on one :class:`RunConfig`. The engine never
touches ``**kwargs`` again; whatever is left in ``RunConfig.advanced``
goes to the situation writer as keyword arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from whileai._env import getenv

from ..defaults import (
    AGENT_MAX_TOKENS_FLOOR,
    DEAD_AGENT_MIN_ERRORS,
    DEFAULT_AVG_TURNS,
    DEFAULT_BUDGET,
    DEFAULT_CARDS_PER_WAVE,
    DEFAULT_COMPLETIONS_PER_REQUEST,
    DEFAULT_CONCURRENCY,
    DEFAULT_EXTRA_CARDS,
    DEFAULT_FAULT_RATE,
    DEFAULT_MIN_USER_TURNS,
    DEFAULT_POOL_SIZE,
    DEFAULT_PROBE,
    DEFAULT_SEED,
    DEFAULT_WRITER_FLIGHT,
    HUNG_SLOT_S,
    MAX_COMPLETIONS_PER_REQUEST,
    RL_FAULT_RATE,
    RL_ROLLOUTS_PER_PROMPT,
    SAMPLING_TEMPERATURE_MAX,
    SATURATION_CAP,
    SFT_COMPLETIONS_PER_PROMPT,
    SFT_PHRASINGS_PER_SITUATION,
    STOP_GRACE_S,
    TIMEOUT_TOKENS_PER_SECOND,
    RunKnobs,
    resolve_knobs,
)
from ..generate.adapters import resolve_system_prompt
from ..generate.agents import LOCAL_MODEL_TIMEOUT, Patience, patience_hazards, reply_budget
from ..generate.diversity import adaptive_allocator
from ..generate.scenarios import SEARCH_ARMS, check_dimensions
from .spec import spec_rubric

if TYPE_CHECKING:
    from ..world.sandbox import WorldOptions

# The values themselves, with the reason for each, live in
# ``whileai.simulations.defaults``; these names stay importable from here.
__all__ = [
    "DEAD_AGENT_MIN_ERRORS",
    "HUNG_SLOT_S",
    "SATURATION_CAP",
    "STOP_GRACE_S",
    "RunConfig",
    "resolve_run_config",
    "resolve_topology",
    "writer_spec_for",
]

_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "explore": {"n_req": 1, "k": 1, "repeat_policy": "none"},
    # n_req is distinct phrasings, k is completions per phrasing: only k is
    # what select_for_sft's top_per_prompt chooses among, and at k=1 the
    # mode could not do the selection it is named for (see
    # SFT_COMPLETIONS_PER_PROMPT in defaults.py; repeats= moves it).
    "sft": {
        "n_req": SFT_PHRASINGS_PER_SITUATION,
        "k": SFT_COMPLETIONS_PER_PROMPT,
        "repeat_policy": "adaptive",
    },
    "rl": {"n_req": 1, "k": RL_ROLLOUTS_PER_PROMPT, "repeat_policy": "successive"},
    "adaptive": {"n_req": 1, "k": 1, "repeat_policy": "adaptive"},
}

_ALIAS_NAMES = {
    "unique",
    "repeats",
    "rollouts_per_prompt",
    "n",
    "phrasings",
    "repeat_policy",
    "policy",
}
_MOVED_NAMES = {
    "concurrency",
    "hard_share",
    "dimensions",
    "arm_weights",
    "simulator",
    "user_model",
    "backend",
    "fault_rate",
    "risk",
    "texture",
    "max_turns",
    "avg_turns",
    "min_user_turns",
    "patience",
    "temperature",
    "agent_max_tokens",
    "sampling",
    "timeout",
    "logprobs",
    "seed",
    "grader",
    "llm_spec",
    "embedder",
    "seed_prompts",
    "extra_situations",
    "prefer_success",
    # steering_weight is an advanced knob, not a named parameter.
    "steering_weight",
    # conversation topology: who opens. "user" (default), "agent",
    # "auto" (share observed in traces), or a 0..1 rate.
    "opening",
}


def _parse_situations_arg(
    situations: Any, extra_situations: list | None
) -> tuple[int | None, list[str]]:
    """Public ``situations=`` is N (int). Seed openers come from advanced."""
    seeds: list[str] = []
    for item in extra_situations or []:
        text = str(item or "").strip()
        if text:
            seeds.append(text)
    if situations is None:
        return None, seeds
    if isinstance(situations, bool) or not isinstance(situations, int):
        raise ValueError("situations= is N, an int. Pass seed openers in advanced['seed_prompts']")
    return max(1, int(situations)), seeds


def _pinned_tasks(tasks: Any) -> list[dict]:
    """The task set a run replays instead of drawing its own (#98).

    ``tasks`` is a previous run (``SimulationData``), its rows, or a JSONL
    path. One entry per distinct prompt, in first-seen order, carrying
    what the engine needs to land the row on the same task: the prompt,
    its ``scenario_id`` and ``scenario_dimensions``, the arm, and the
    fault plan and world state the rollout ran under.
    """
    if tasks is None:
        return []
    source = tasks
    if hasattr(source, "trajectories"):
        source = source.trajectories
    elif isinstance(source, (str, Path)):
        from ..score.quality import load_jsonl

        source = load_jsonl(str(source))
    out: list[dict] = []
    seen: set[str] = set()
    rollouts: dict[str, int] = {}
    for row in source if isinstance(source, (list, tuple)) else list(source):
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue
        rollouts[prompt] = rollouts.get(prompt, 0) + 1
        if prompt in seen:
            continue
        seen.add(prompt)
        dims = row.get("scenario_dimensions")
        dims = dict(dims) if isinstance(dims, dict) and dims else None
        faults = row.get("faults")
        plan: dict[str, Any] = {
            k: dict(v) for k, v in (faults or {}).items() if isinstance(v, dict)
        }
        world = row.get("world_state") or (dims or {}).get("world_state")
        if world and str(world) != "unspecified":
            plan["world_state"] = str(world)
        stance = (dims or {}).get("stance")
        if stance:
            plan["stance"] = str(stance)
        out.append(
            {
                "prompt": prompt,
                "scenario_id": str(row.get("scenario_id") or row.get("task_id") or "") or None,
                "assignment": dims,
                "arm": str(row.get("arm") or "") or "pinned",
                "plan": plan,
                # who wrote the situation, so the replayed row keeps the
                # writer's name and delta_report sees one writer, not
                # "pinned" against a model name
                "writer_model": str(row.get("writer_model") or "") or None,
            }
        )
    if not out:
        raise ValueError(
            "tasks= has no rows with a prompt; pass a previous run, its rows, or a JSONL path"
        )
    # The base run's k: the most rollouts any one prompt has. A re-run that
    # names no repeats inherits it, so a before/after stays paired at the
    # same k instead of silently comparing k=4 against k=1.
    out[0]["base_k"] = max(rollouts.values())
    return out


def default_rollout_timeout(agent_max_tokens: int | None = None) -> float:
    """Seconds one agent call may take when ``timeout=`` is not given.

    ``max(LOCAL_MODEL_TIMEOUT, reply budget / TIMEOUT_TOKENS_PER_SECOND)``:
    300 s, or the reply budget (``agent_max_tokens=``, else the model
    default) at 4 tokens a second, whichever is longer. A flat 300 s
    re-rolled every 4,096-token reply of a loaded 9B server and turned a
    two-hour run into six and a half (#470); the budget the caller set
    is the time the call needs.
    """
    return max(
        float(LOCAL_MODEL_TIMEOUT), reply_budget(agent_max_tokens) / TIMEOUT_TOKENS_PER_SECOND
    )


def _merge_advanced(advanced: dict | None, passed: dict) -> tuple[dict, dict]:
    """Split silent aliases from advanced knobs. Unknown names error."""
    cfg = dict(advanced or {})
    aliases: dict[str, Any] = {}
    # repeat_policy written into advanced= is topology, not a writer knob
    if "repeat_policy" in cfg:
        aliases["repeat_policy"] = cfg.pop("repeat_policy")
    for key, val in passed.items():
        if key in _ALIAS_NAMES:
            aliases[key] = val
        elif key in _MOVED_NAMES:
            cfg[key] = val
        else:
            raise TypeError(f"simulate() got unexpected keyword argument {key!r}")
    return cfg, aliases


def writer_spec_for(agent: Any, simulator: Any) -> Any:
    """The situation writer's backend when none was named.

    A string agent spec (``openai:<model>``, ``vllm:<model>@<url>``) is a
    bring-your-own model; the writer runs on it too, so no While key
    is involved. ``WHILEAI_SURROGATE`` and an explicit ``simulator``
    still win.

    ``simulator="hosted"`` is the default written out, so the way back
    from the offline writer is a value, not "delete the argument".
    """
    if isinstance(simulator, str) and simulator.strip().lower() == "hosted":
        simulator = None
    if simulator is not None or getenv("SURROGATE"):
        return simulator
    if isinstance(agent, str) and ":" in agent and not agent.startswith(("http://", "https://")):
        return agent
    return simulator


def resolve_topology(
    *,
    mode: str | None = None,
    repeat_policy: str | None = None,
    unique: bool = False,
    unique_situations: bool = False,
    requests_per_situation: int | None = None,
    n: int | None = None,
    phrasings: int | None = None,
    rollouts_per_request: int | None = None,
    repeats: int | None = None,
    rollouts_per_prompt: int | None = None,
) -> dict[str, Any]:
    """Map public knobs. Phrasings (n) are wordings per situation; repeats (k) are reruns per phrasing."""
    k_vals = [int(x) for x in (rollouts_per_request, repeats, rollouts_per_prompt) if x is not None]
    if len(set(k_vals)) > 1:
        raise ValueError("pass rollouts_per_request= or repeats=, not both")
    n_vals = [int(x) for x in (requests_per_situation, n, phrasings) if x is not None]
    if len(set(n_vals)) > 1:
        raise ValueError("pass requests_per_situation=, phrasings=, or n=, not both")
    k_explicit = bool(k_vals)
    n_explicit = bool(n_vals)
    new_cards = bool(unique_situations or unique)
    mode_name = str(mode or "").strip().lower() or None
    policy_name = str(repeat_policy or "").strip().lower() or None
    if policy_name in {"unique"}:
        policy_name = "none"
        new_cards = True
    if policy_name is not None and policy_name not in {"none", "adaptive", "successive", "fixed"}:
        raise ValueError("repeat_policy= must be none, adaptive, successive, or fixed")
    if mode_name in _MODE_PRESETS and policy_name is None:
        policy_name = _MODE_PRESETS[mode_name]["repeat_policy"]
    if policy_name == "none":
        mode_name = mode_name or "explore"
        new_cards = True
    elif mode_name is None and policy_name == "adaptive":
        mode_name = "adaptive"
    elif mode_name is None:
        mode_name = "explore"
        policy_name = "none"
        new_cards = True
    if mode_name not in _MODE_PRESETS:
        raise ValueError("mode= must be explore, sft, rl, or adaptive")
    preset = _MODE_PRESETS[mode_name]
    if policy_name is None:
        policy_name = preset["repeat_policy"]
    n_req = int(preset["n_req"])
    k = int(preset["k"])
    if n_explicit:
        n_req = max(1, n_vals[0])
    if k_explicit:
        k = max(1, k_vals[0])
    if new_cards:
        if not n_explicit:
            n_req = 1
        if not k_explicit:
            k = 1
        if policy_name == "none":
            policy_name = "none"
        elif n_req == 1:
            policy_name = policy_name or "adaptive"
    return {
        "mode": mode_name,
        "repeat_policy": policy_name,
        "n_req": n_req,
        "k": k,
        "k_explicit": k_explicit,
        "n_explicit": n_explicit,
        "unique_situations": new_cards,
    }


def _model_version_tag(agent: Any, advanced: dict) -> str:
    """Which weights produced each row.

    Rounds of the continual loop are indistinguishable without it (base
    and every adapter can share a model name). ``advanced["model_version"]``
    overrides; the resolved backend model is the default; callable agents
    record their name.
    """
    tag = str(advanced.pop("model_version", "") or "").strip()
    if tag:
        return tag
    if agent is not None and callable(agent) and not isinstance(agent, str):
        return getattr(agent, "__name__", "callable-agent")
    try:
        from ..generate.agents import default_simulator_spec, parse_backend_spec

        spec = agent if isinstance(agent, str) else default_simulator_spec()
        return parse_backend_spec(spec)[1]
    except Exception:
        return "unknown"


@dataclass
class RunConfig:
    """Everything ``simulate()`` was asked for, resolved and validated."""

    # what to simulate
    agent: Any
    spec: Any
    tools: list[dict] | None
    system_prompt: str | None
    scaffold_text: str
    traces: Any
    execute: Callable | None
    # seeds= as given (amplification only fires when the caller passed it)
    # and the full opener list: seeds + advanced seed_prompts + extra_situations
    seeds: list | None
    seed_prompts: list[str]
    n_situations_target: int | None
    # tasks= : replay exactly these prompts on their scenario ids, draw none
    pinned_tasks: list[dict]
    # topology
    topo: dict
    repeat_count: int
    n_req: int
    unique_cards: bool
    k_immediate: bool
    # successive allocation (rl): rollouts a prompt gets before the run
    # decides whether it splits; the structural minimum is 2
    probe: int
    # Round-synchronous scheduling: same seed, same concurrency, same
    # agent gives the same rows. Costs throughput under uneven latency.
    reproducible: bool
    # budget and stop rule
    budget: int | None
    cap: int
    time_budget: float | None
    until_key: str
    until_sat: bool
    # coverage stance
    strategy: str
    resolved_strategy: str
    steering_weight: float | None
    opening_req: Any
    targeted_regions: list[str]
    # output and grading
    output: str | None
    out_path: Path | None
    # checkpoint= : every row appended here as it lands; a re-run with the
    # same path and tasks= loads them and rolls out only what is missing
    checkpoint_path: Path | None
    # on_progress= : called with the progress dict on every progress line
    on_progress: Callable[[dict], None] | None
    # grade= : True for the rubric judge, "conduct" for the deterministic
    # conduct check by name, False for ungraded rows.
    grade: bool | str
    grader: Any
    llm_grade: bool
    llm_spec: Any
    # what doing the job means, for the judge: rubric= or the spec's rubric.md
    rubric: str | None
    # engine knobs
    concurrency: int
    # the difficulty dial: share of situations drawn from the hard tiers
    # (ambiguous, boundary, adversarial); None when the caller left it at
    # the default. Travels to the writer and mixer as a plain argument.
    hard_share: float | None
    dimensions: Any
    # how the situation search splits across arms: structured, llm_guided,
    # open_ended, behavior_targeted, failure_mutation. None uses SEARCH_ARMS.
    # Raising the structured share makes a harder eval set; open_ended stays
    # capped in the 5-10% band whatever is asked for.
    arm_weights: Any
    simulator: Any
    # the simulated user's model: a backend spec, or None for the agent's own
    user_model: str | None
    backend: Any
    fault_rate: float
    max_turns: Any
    # target thread length; at or under 1 it is one user line and one
    # reply for every rollout, and the follow-up branch never runs
    avg_turns: float
    # the generation knobs the caller named (``fault_rate`` or ``risk``,
    # ``avg_turns``). The two above always carry a value, so this is the
    # only record of which were set and which are defaults: a default that
    # the rows miss is not a setting that failed, and is not warned about.
    set_by_caller: frozenset[str]
    min_user_turns: int
    # a level name, or (second, later) walk-away chances (see patience_hazards)
    patience: Patience
    # one temperature for every simulated-user line, or None for the defaults
    user_temperature: float | None
    temperature: Any
    agent_max_tokens: int | None
    sampling: dict | None
    logprobs: Any
    seed: int
    embedder: Any
    mutate_failures: bool
    mutate_graded_failures: bool
    pool_size: int
    scenario_concurrency: int
    writer_flight: int
    scenarios_per_request: int
    distinct_cards: bool
    completions_per_request: int
    extra_cards: int
    hung_slot_s: float
    stop_grace_s: float
    rollout_timeout: float
    model_version_tag: str
    # every other engine number, each an ``advanced`` key of the same name
    knobs: RunKnobs = field(default_factory=RunKnobs)
    # the mock world's dials (sandbox WorldOptions), or None for the defaults
    world_options: WorldOptions | None = None
    # what is left goes to the situation writer as keyword arguments
    advanced: dict = field(default_factory=dict)


def resolve_run_config(
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
    reproducible: bool | None = None,
    grade: bool | str = False,
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
    checkpoint: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
    advanced: dict | None = None,
    passed: dict | None = None,
) -> RunConfig:
    """Turn the ``simulate()`` call into a :class:`RunConfig`.

    Validation errors surface here, before any model or file is touched.
    """
    cfg, aliases = _merge_advanced(advanced, dict(passed or {}))
    # Engine knobs first, so none of them rides ``**advanced`` into the
    # situation writer as an unknown keyword.
    knobs = resolve_knobs(cfg)
    policy = resolve_system_prompt(system_prompt, aliases.get("policy"))
    scaffold_text = str(scaffold or "").strip()
    unique_flag = bool(unique_situations or aliases.get("unique", False))
    repeats = aliases.get("repeats")
    rollouts_per_prompt = aliases.get("rollouts_per_prompt")
    k_arg = rollouts_per_request
    if k_arg == 1 and (repeats is not None or rollouts_per_prompt is not None):
        # A leftover default 1 plus an alias means the alias wins.
        k_arg = None
    topo = resolve_topology(
        mode=mode,
        repeat_policy=aliases.get("repeat_policy"),
        unique=unique_flag,
        unique_situations=unique_flag,
        requests_per_situation=requests_per_situation,
        n=aliases.get("n"),
        phrasings=aliases.get("phrasings"),
        rollouts_per_request=k_arg,
        repeats=repeats,
        rollouts_per_prompt=rollouts_per_prompt,
    )

    seed_prompts: list[str] = []
    for item in list(seeds or []) + list(cfg.pop("seed_prompts", None) or []):
        text = str(item or "").strip()
        if text:
            seed_prompts.append(text)
    for item in cfg.pop("extra_situations", None) or []:
        text = str(item or "").strip()
        if text:
            seed_prompts.append(text)
    n_situations_target, seed_prompts = _parse_situations_arg(situations, seed_prompts)
    pinned_tasks = _pinned_tasks(tasks)
    if pinned_tasks and seed_prompts:
        raise ValueError(
            "tasks= replays a fixed task set; it cannot be combined with seeds= or seed_prompts"
        )
    if pinned_tasks and not topo["k_explicit"]:
        # tasks= copies the prompts; without repeats= it also keeps the
        # base run's k, so the re-run pairs at the same k (#31).
        topo["k"] = max(1, int(pinned_tasks[0].pop("base_k", 1) or 1))
        topo["k_explicit"] = True
    elif pinned_tasks:
        pinned_tasks[0].pop("base_k", None)

    concurrency = int(cfg.pop("concurrency", DEFAULT_CONCURRENCY))
    dimensions = cfg.pop("dimensions", None)
    check_dimensions(dimensions)
    arm_weights = cfg.pop("arm_weights", None)
    if arm_weights is not None:
        if not isinstance(arm_weights, dict) or not arm_weights:
            raise ValueError("arm_weights= must be a non-empty dict of arm name -> weight")
        bad = {k for k, v in arm_weights.items() if not isinstance(v, (int, float)) or v < 0}
        if bad:
            raise ValueError(
                f"arm_weights= values must be non-negative numbers; bad: {sorted(bad)}"
            )
        unknown = set(arm_weights) - set(SEARCH_ARMS)
        if unknown:
            raise ValueError(
                f"arm_weights= has unknown arm(s) {sorted(unknown)}; "
                f"known arms are {sorted(SEARCH_ARMS)}"
            )
        if sum(arm_weights.values()) <= 0:
            raise ValueError("arm_weights= must have a positive total")
    simulator = writer_spec_for(agent, cfg.pop("simulator", None))
    hard_share = cfg.pop("hard_share", None)
    if hard_share is not None:
        try:
            hard_share = float(hard_share)
        except (TypeError, ValueError):
            raise ValueError("hard_share= is a fraction between 0 and 1") from None
        if not 0.0 <= hard_share <= 1.0:
            raise ValueError(
                f"hard_share={hard_share} is outside 0..1; it is the share of "
                "situations drawn from the hard tiers, not a count"
            )
    user_model = cfg.pop("user_model", None)
    if user_model is not None:
        if not isinstance(user_model, str):
            raise TypeError(
                "user_model= is a backend spec string ('openai:<model>' or "
                "'vllm:<model>@<url>') or None for the agent's own model"
            )
        user_model = user_model.strip() or None
    # A decision model (typesafe:) answers questions and writes nothing,
    # so it can judge but not play a role; say so before any call is made.
    from ..generate.typesafe_backend import no_chat_error

    for role, value in (("agent", agent), ("simulator", simulator), ("user_model", user_model)):
        refusal = no_chat_error(value)
        if refusal:
            raise ValueError(f"{role}= {refusal}")
    backend = cfg.pop("backend", None)
    explicit_fault = "fault_rate" in cfg or "risk" in cfg
    fault_rate = float(cfg.pop("fault_rate", DEFAULT_FAULT_RATE))
    risk = cfg.pop("risk", None)
    if risk is not None:
        fault_rate = float(risk)
    elif str(topo["mode"]) == "rl" and not explicit_fault:
        fault_rate = RL_FAULT_RATE
    texture = cfg.pop("texture", None)
    max_turns = cfg.pop("max_turns", None)
    explicit_turns = "avg_turns" in cfg
    avg_turns = float(cfg.pop("avg_turns", DEFAULT_AVG_TURNS))
    set_by_caller = frozenset(
        name
        for name, was_set in (("fault_rate", explicit_fault), ("avg_turns", explicit_turns))
        if was_set
    )
    min_user_turns = max(1, int(cfg.pop("min_user_turns", DEFAULT_MIN_USER_TURNS)))
    # patience: a level name ("normal", "short", "endless") or a table
    # {"second": p, "later": q} / (p, q) of walk-away chances fitted from
    # your own traces. patience_hazards() is the one validator, so a bad
    # value fails here with the fix before any model is touched.
    raw_patience = cfg.pop("patience", None)
    patience: Patience
    if raw_patience is None or isinstance(raw_patience, str):
        patience = str(raw_patience or "normal").strip().lower()
        patience_hazards(patience)
    else:
        patience = patience_hazards(raw_patience)
    # user_temperature: the sampling temperature of every simulated-user
    # line (follow-ups and human-tool answers alike); None keeps the two
    # named defaults in generate/agents.py.
    raw_user_temperature = cfg.pop("user_temperature", None)
    user_temperature = None if raw_user_temperature is None else float(raw_user_temperature)
    if user_temperature is not None and not 0.0 <= user_temperature <= SAMPLING_TEMPERATURE_MAX:
        raise ValueError(
            f"user_temperature={user_temperature!r} is outside 0..2; it is a sampling "
            "temperature for the simulated user's lines (None keeps the defaults)"
        )
    temperature = cfg.pop("temperature", None)
    # The model agent's reply budget. Default: 768 tokens, or 2048 above an
    # 8k context (ZP_CONTEXT_TOKENS). A reasoning model that thinks before it
    # answers needs more, or its replies are cut mid-thought and score 0.
    raw_max_tokens = cfg.pop("agent_max_tokens", None)
    agent_max_tokens = int(raw_max_tokens) if raw_max_tokens else None
    if agent_max_tokens is not None and agent_max_tokens < AGENT_MAX_TOKENS_FLOOR:
        raise ValueError(
            f"agent_max_tokens is a reply budget in tokens ({AGENT_MAX_TOKENS_FLOOR} or more)"
        )
    logprobs = cfg.pop("logprobs", False)
    if logprobs not in (False, True, "tokens"):
        raise ValueError('logprobs must be False, True, or "tokens"')
    # How a callable agent samples is the caller's to say; it is recorded
    # on every row as given. A model backend records its own instead.
    sampling = cfg.pop("sampling", None)
    if sampling is not None and not isinstance(sampling, dict):
        raise ValueError(
            'sampling= is a dict of how your agent samples, like {"temperature": 0.7, '
            '"max_tokens": 1024, "model": "my-model"}'
        )
    seed = int(cfg.pop("seed", DEFAULT_SEED))
    # The named grader= parameter wins; advanced={"grader": ...} stays as
    # the legacy spelling. Both route to one application path at the end.
    grader = grader if grader is not None else cfg.pop("grader", None)
    cfg.pop("grader", None)
    if grade not in (True, False, "conduct"):
        raise ValueError(
            f"grade= is True (the rubric judge), False, or 'conduct' (the deterministic "
            f"conduct check); got {grade!r}. A callable goes in grader=."
        )
    resolved_rubric = (str(rubric).strip() or None) if rubric else spec_rubric(spec)
    if grade is True and grader is None:
        # grade=True is the rubric judge, the same one data.grade(wai.Judge(
        # rubric=...)) runs, applied through the grader path so every row
        # carries reward, judge_status, judge_name and lineage. The advisory
        # llm_grade pass writes llm_reward and never reward, so it is not
        # this. The conduct check answers a different question (what the
        # agent did, not whether it did the job) and is only reachable by
        # name, because a reward nobody chose reads exactly like one they
        # did (Lambert 2025, chapter Reward Models). With no key the judge
        # stops here, before any budget is spent; it never substitutes.
        from whileai.judge import Judge

        from ..score.llm_judge import MISSING_JUDGE_KEY, resolve_judge_key

        judge = Judge(resolved_rubric, policy=policy or "", tools=list(tools or []))
        if not resolve_judge_key(None, judge.spec):
            raise RuntimeError(MISSING_JUDGE_KEY)
        grader = judge
    if grader is not None and not callable(grader):
        # A string here ran every rollout through run_judge as an error:
        # 150 rows "judged", none with a reward, and nothing said so.
        raise TypeError(
            f"grader= takes a callable row -> {{'reward': 0 or 1, ...}}, got {type(grader).__name__} "
            f"{grader!r}. For the hosted judge, leave grader= off and call data.grade() after "
            "the run, or pass whileai.simulations.score.grade_llm.grade_one."
        )
    llm_grade = bool(llm_grade or cfg.pop("llm_grade", False))
    llm_spec = cfg.pop("llm_spec", None)
    embedder = cfg.pop("embedder", "hash")

    until_key = str(until or "compute").strip().lower()
    if until_key in {"first", "saturation"}:
        until_sat = True
        until_key = "saturation"
    elif until_key in {"compute", "budget_only", "budget", "time"}:
        until_sat = False
        until_key = "compute"
    else:
        raise ValueError("until= must be compute, saturation, first, or budget_only")
    time_budget = None if time_budget is None or float(time_budget) <= 0 else float(time_budget)

    repeat_count = int(topo["k"])
    n_req = int(topo["n_req"])
    unique_cards = bool(topo["unique_situations"])
    if topo["mode"] == "adaptive" and not unique_cards:
        adapt = adaptive_allocator(time_budget, until_key)
        if not topo["n_explicit"]:
            n_req = int(adapt["n_req"])
        if not topo["k_explicit"]:
            repeat_count = int(adapt["k"])
    # Adaptive defers extra k so verify can react to behavior. Successive
    # (the rl default) probes every prompt and spends the rest of k on the
    # prompts that split; ``repeat_policy="fixed"`` is the old all-k-at-once.
    repeat_policy_name = topo["repeat_policy"]
    if repeat_policy_name == "fixed":
        k_immediate = True
    elif repeat_policy_name == "successive":
        k_immediate = False
    else:
        k_immediate = bool(topo["k_explicit"] or topo["mode"] == "rl")
    probe = max(1, int(cfg.pop("probe", DEFAULT_PROBE)))

    out_path = Path(output).expanduser() if output else None
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
    if on_progress is not None and not callable(on_progress):
        raise TypeError(
            "on_progress= takes a callable progress_dict -> None, called on every progress "
            f"line; got {type(on_progress).__name__}"
        )
    if texture is not None:
        cfg["texture_rate"] = float(texture)
    mutate_failures = bool(cfg.pop("mutate_failures", True))
    # On whenever a grader runs beside the loop: the grader's verdict is
    # then the definition of failure the search steers by, not only the
    # sandbox's (#285). Off without a grader, since there is no verdict.
    # Not a named parameter: the grader is the switch, and the only
    # thing left to say is "grade but do not steer", which is an
    # advanced key like mutate_failures beside it.
    graded_raw = cfg.pop("mutate_graded_failures", None)
    mutate_graded_failures = (grader is not None) if graded_raw is None else bool(graded_raw)
    if mutate_graded_failures and grader is None:
        raise ValueError(
            "advanced={'mutate_graded_failures': True} needs grader=; without a grader "
            "there is no verdict to steer by"
        )
    pool_size = int(cfg.pop("per_round", DEFAULT_POOL_SIZE))
    writer_raw = cfg.pop("scenario_concurrency", None)
    # Writer flight is a scheduler internal. Topology (unique / explore)
    # does not change it. Too many starves rollouts.
    scenario_concurrency = DEFAULT_WRITER_FLIGHT if writer_raw is None else max(1, int(writer_raw))
    writer_flight = max(1, scenario_concurrency)
    scenarios_per_request = max(1, int(cfg.pop("scenarios_per_request", DEFAULT_CARDS_PER_WAVE)))
    # A unique-situation run must walk the planned grid. Previously the
    # public unique=True knob still left the model writer in weighted
    # resampling mode unless callers also knew about this private switch.
    distinct_cards = bool(cfg.pop("distinct_cards", unique_cards))
    if "completions_per_request" in cfg:
        completions_per_request = max(
            1, min(MAX_COMPLETIONS_PER_REQUEST, int(cfg["completions_per_request"]))
        )
    else:
        completions_per_request = DEFAULT_COMPLETIONS_PER_REQUEST
    cfg.pop("completions_per_request", None)
    extra_cards = max(0, int(cfg.pop("extra_cards", DEFAULT_EXTRA_CARDS)))
    hung_slot_s = float(cfg.pop("hung_slot", HUNG_SLOT_S))
    stop_grace_s = max(0.0, float(cfg.pop("stop_grace", STOP_GRACE_S)))
    cfg.pop("scene_brief", None)
    model_version_tag = _model_version_tag(agent, cfg)
    # Popped unconditionally: on a non-trace run the key must not ride
    # **advanced into the generator, where it is an unknown kwarg.
    targeted_regions = [str(x) for x in (cfg.pop("targeted_regions", None) or [])]

    # strategy= names the coverage stance explicitly; traces change the
    # coverage distribution, not the space. auto resolves descriptively
    # and records its reason; targeted requires an explicit user choice.
    if strategy not in ("auto", "broad", "trace", "targeted"):
        raise ValueError("strategy= must be auto, broad, trace, or targeted")
    resolved_strategy = strategy
    if strategy == "auto":
        resolved_strategy = "trace" if traces is not None else "broad"
    if resolved_strategy in ("trace", "targeted") and traces is None:
        raise ValueError(f"strategy='{resolved_strategy}' needs traces=")
    # steering_weight controls how much the trace-aimed distribution
    # outweighs background coverage; accepted via steering_weight= or
    # advanced=. Over a trace-focused grid, weight w sends each structured
    # card draw to the front (trace-mined) half of the steered axes with
    # probability w; rows drawn that way carry row["steering"] =
    # {"origin": "targeted"}. None applies no bias; a number overrides it.
    steering_weight = cfg.pop("steering_weight", None)
    opening_req = cfg.pop("opening", None)
    if opening_req is not None and opening_req not in ("user", "agent", "auto"):
        try:
            opening_req = float(opening_req)
        except (TypeError, ValueError):
            raise ValueError(
                'opening= must be "user", "agent", "auto", or a rate in [0, 1]'
            ) from None
        if not 0.0 <= opening_req <= 1.0:
            raise ValueError("opening= rate must be in [0, 1]")
    if steering_weight is not None:
        try:
            steering_weight = float(steering_weight)
        except (TypeError, ValueError):
            raise ValueError("steering_weight= must be a number in [0, 1]") from None
        if not 0.0 <= steering_weight <= 1.0:
            raise ValueError("steering_weight= must be a number in [0, 1]")
        if traces is None:
            raise ValueError("steering_weight= needs traces=")
    # Seconds per completion. The default survives a served model's cold
    # start (two to three minutes) and scales with the reply budget, so a
    # long reply is not re-rolled for taking the time it was allowed.
    raw_timeout = cfg.pop("timeout", None)
    rollout_timeout = (
        float(raw_timeout) if raw_timeout else default_rollout_timeout(agent_max_tokens)
    )

    # advanced={"world": {...}}: the mock world's dials (search_hits,
    # exists_share, default_fault_mode, name pools, ...). Validated here so a
    # typo fails before any model is touched; reaches MockEnvironment(options=).
    world_options = cfg.pop("world", None)
    if world_options is not None:
        from ..world.sandbox import WorldOptions

        world_options = WorldOptions.coerce(world_options)
    # A tool_condition the world cannot answer would steer cells at a fault
    # that never fires. "success", a condition in the world's condition_modes,
    # or a fault mode the world knows (shipped or added through
    # advanced={"world": {"fault_modes": ...}}) are the values that land.
    if isinstance(dimensions, dict) and dimensions.get("tool_condition"):
        from ..world.sandbox import WorldOptions

        world = WorldOptions.coerce(world_options)
        unknown_conditions = [
            str(v)
            for v in dimensions["tool_condition"]
            if str(v) != "success" and world.fault_mode_for(str(v)) is None
        ]
        if unknown_conditions:
            raise ValueError(
                f"dimensions= tool_condition values {unknown_conditions} name no fault mode "
                "the mock "
                f"world knows; use success, {', '.join(sorted(world.condition_modes))} "
                f"or a key of fault_modes ({', '.join(sorted(world.fault_modes))}). Add a "
                'builder with advanced={"world": {"fault_modes": {**FAULT_MODES, name: fn}}}.'
            )

    cap = budget if budget is not None else SATURATION_CAP
    return RunConfig(
        agent=agent,
        spec=spec,
        tools=tools,
        system_prompt=policy,
        scaffold_text=scaffold_text,
        traces=traces,
        execute=execute,
        seeds=seeds,
        seed_prompts=seed_prompts,
        n_situations_target=n_situations_target,
        pinned_tasks=pinned_tasks,
        topo=topo,
        repeat_count=repeat_count,
        n_req=n_req,
        unique_cards=unique_cards,
        k_immediate=k_immediate,
        probe=probe,
        # None: reproducible unless a clock is set, since a clock stop lands
        # wherever the run happens to be (#645).
        reproducible=(time_budget is None) if reproducible is None else bool(reproducible),
        budget=budget,
        cap=int(cap),
        time_budget=time_budget,
        until_key=until_key,
        until_sat=until_sat,
        strategy=strategy,
        resolved_strategy=resolved_strategy,
        steering_weight=steering_weight,
        opening_req=opening_req,
        targeted_regions=targeted_regions,
        output=output,
        out_path=out_path,
        checkpoint_path=checkpoint_path,
        on_progress=on_progress,
        grade=grade,
        grader=grader,
        llm_grade=llm_grade,
        llm_spec=llm_spec,
        rubric=resolved_rubric,
        concurrency=concurrency,
        hard_share=hard_share,
        dimensions=dimensions,
        arm_weights=arm_weights,
        simulator=simulator,
        user_model=user_model,
        backend=backend,
        fault_rate=fault_rate,
        max_turns=max_turns,
        avg_turns=avg_turns,
        set_by_caller=set_by_caller,
        min_user_turns=min_user_turns,
        patience=patience,
        user_temperature=user_temperature,
        temperature=temperature,
        agent_max_tokens=agent_max_tokens,
        sampling=sampling,
        logprobs=logprobs,
        seed=seed,
        embedder=embedder,
        mutate_failures=mutate_failures,
        mutate_graded_failures=mutate_graded_failures,
        pool_size=pool_size,
        scenario_concurrency=scenario_concurrency,
        writer_flight=writer_flight,
        scenarios_per_request=scenarios_per_request,
        distinct_cards=distinct_cards,
        completions_per_request=completions_per_request,
        extra_cards=extra_cards,
        hung_slot_s=hung_slot_s,
        stop_grace_s=stop_grace_s,
        rollout_timeout=rollout_timeout,
        model_version_tag=model_version_tag,
        knobs=knobs,
        world_options=world_options,
        advanced=cfg,
    )
