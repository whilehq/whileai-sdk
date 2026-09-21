"""The run engine behind ``simulate()``.

One :class:`Run` per call, driven by a :class:`RunConfig`. Phases, in
order:

1. **inputs**: load the spec, inspect the agent, draft tools for a
   prompt-only agent, amplify seeds, read traces into a grid emphasis.
2. **build**: the result object, the scene-brief thread, the rollout
   runner, the situation writer and its coverage grid.
3. **loop**: writer waves fill a prompt pool; a selector picks a diverse
   batch; rollouts run in a thread pool; every batch of results updates
   the search state (arm weights, region retargeting, the verify queue).
4. **finish**: leakage pruning, coverage summary, grading, save.

The loop state lives on the instance so each phase is a method with a
small local scope. Nothing here is public; ``simulate()`` is the door.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import re
import sys
import threading
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...auth import trial_prerun_note
from ..data import SimulationData, clean_faults, conversation, export_row, note_stage, row_world
from ..defaults import (
    DELIVERED_FAULT_LEAK,
    DELIVERED_FAULT_SHORTFALL,
    DELIVERED_LONG_CONVERSATION_TURNS,
    DELIVERED_STANCE_MIN_SHARE,
    DELIVERED_TURNS_MIN_REQUEST,
    DELIVERED_TURNS_SHORTFALL,
    FINGERPRINT_STEM_MIN_LEN,
    MAX_COMPLETIONS_PER_REQUEST,
    OK_STATUSES,
    PARENT_HEAD_CHARS,
    PASS_THRESHOLD,
    PROGRESS_MIN_BUDGET,
    PROGRESS_MIN_ROWS_FOR_ESTIMATE,
    REPORT_LIST_ITEMS,
    SCENARIO_ID_CHARS,
    SHORT_HASH_CHARS,
    laplace,
)
from ..generate.actionspace import (
    action_space_targets,
    induced_keys_from_trajectory,
    render_target_situation,
    shape_as_tags,
    shape_from_trajectory,
    uncovered_action_shapes,
)
from ..generate.adapters import inspect, resolve
from ..generate.agents import (
    _account_url,
    current_rollout,
    default_agent_spec,
    default_max_turns,
    ended_on_question,
    hosted_model,
    local_model,
    missing_hosted_key,
    parse_backend_spec,
    public_llm_error,
    touch_hosted,
)
from ..generate.coverage import (
    NEW_SIGNATURE_FLOOR,
    SATURATION_COPIES,
    build_coverage_summary,
    copies_remaining,
    pairwise_coverage,
    space_saturated,
)
from ..generate.diversity import (
    HARD_SHARE,
    MAX_NOVELTY_RESTARTS,
    NOVELTY_RESTART_FLOOR,
    adaptive_allocator,
    allocator_slot_counts,
    behavior_tier,
    cap_scenario_families,
    new_turn_stats,
    record_turns,
    sampling_plan,
    scenario_family,
)
from ..generate.embeddings import (
    EmbeddingArchive,
    is_semantic,
    resolve_embedder,
    select_execution_batch,
)
from ..generate.explore import mutate_pool
from ..generate.generator import (
    amplify_seeds,
    draft_tools,
    make_default_generator,
    write_result_shapes,
    write_scene_brief,
)
from ..generate.scenarios import (
    RULE_CAP,
    SEARCH_ARMS,
    complete_yields,
    intent_for_tool,
    keep_fault_plan,
    reallocate_search_arms,
    retarget_regions,
    rule_axis,
)
from ..ingest.traces import (
    behavior_state,
    dimensions_from_traces,
    drop_leaky_rows,
    exemplar_result_shapes,
    load_traces,
    mine_result_exemplars,
    mine_traces,
    opening_share,
    region_progress,
)
from ..schema import SCHEMA_KEY, SCHEMA_VERSION, Judgment, ScorerRef, attach
from ..score.checklist import privileged_context
from ..score.grading import (
    behavior_signature,
    conduct_grade,
    dead_tools,
    dead_tools_note,
    tool_outcomes,
)
from ..world.sandbox import WorldOptions
from .config import RunConfig
from .rows import (
    LOST_REASONS,
    _row_conversation,
    _situation_key_from_meta,
    _stratified_prompts,
    _unusable_reason,
    _usable_rollout,
    failed_criteria,
    mutation_worthy,
    record_coverage,
    row_cell_key,
    system_prompt_stamp,
)
from .spec import apply_spec, backend_spec, kind_from_spec

_AUTH_ERROR_MARKS = (
    "rejected the API key",
    "Hosted models need a key",
    "No API key for",
    "quota exceeded",
)


def _auth_error(message: str) -> str | None:
    """The auth error inside a writer or agent failure, or None.

    A key the endpoint rejects (401/403) is a configuration error, not a
    transient one: no wave and no rollout after it can succeed. The
    no-key case fails at setup (``missing_hosted_key``); this is the
    present-but-wrong-key case, which otherwise spent the whole time
    budget on 401s and returned zero rows with the reason buried in
    ``search["writer_errors"]``. A spent daily allowance (the account
    proxy's 429) is the same shape: every later call today answers 429.
    """
    text = str(message or "")
    for mark in _AUTH_ERROR_MARKS:
        if mark.lower() in text.lower():
            start = max(text.find("Hosted Qwen"), text.find("Hosted model"))
            return text[start:] if start >= 0 else text
    return None


_TIMEOUT_MARKS = ("timed out", "timeout")


def _timeout_error(message: str) -> bool:
    """Did the agent call time out? ``socket.timeout`` is ``TimeoutError``
    on 3.10+ and reads ``timed out``; a wrapper may say ``timeout``."""
    text = str(message or "").lower()
    return any(mark in text for mark in _TIMEOUT_MARKS)


def _graded_failure(row: dict, threshold: float = PASS_THRESHOLD) -> bool:
    """Did the grader fail this row? A reward under ``threshold``
    (``PASS_THRESHOLD``, 0.5) -- a 0 from a 0/1 judge, a failed verifier, a
    rubric below half -- OR any single rubric criterion failed.

    The scalar alone is not enough. ``rubric_judge`` scores a rubric as the
    mean of its criteria, so a row that breaks one rule of three still scores
    0.667 and passes a 0.5 threshold. Those rows are exactly the ones a rubric
    cares about, and reading only the mean made them invisible to the search:
    the situation was never re-rolled, so the rule was never aimed at (#285,
    where the four missed rules failed 4 to 10% of the time in the source
    traces and 0 to 2% in what was generated from them).

    A criterion that never varies contributes no advantage to a grouped update
    (Shao et al. 2024, arXiv:2402.03300), and difficulty filtering has to read
    the band on the criterion being trained rather than on a mean that spans
    several (Lambert 2025, chapter Reasoning). ``None`` (not judged, judge
    error) is still not a failure.
    """
    if failed_criteria(row):
        return True
    reward = row.get("reward")
    if reward is None or isinstance(reward, bool):
        return False
    try:
        return float(reward) < threshold
    except (TypeError, ValueError):
        return False


def _stop_reason(side: str, message: str) -> str:
    return f"{side}_quota_exceeded" if "quota" in message.lower() else f"{side}_auth_failed"


log = logging.getLogger("whileai.simulations")


def someone_listens(logger: logging.Logger = log) -> bool:
    """Is any handler other than the library's ``NullHandler`` attached to
    ``logger`` or an ancestor it propagates to? ``logging.basicConfig()``,
    a caplog, a root ``StreamHandler``: any of them counts."""
    current: logging.Logger | None = logger
    while current is not None:
        if any(not isinstance(h, logging.NullHandler) for h in current.handlers):
            return True
        if not current.propagate:
            return False
        current = current.parent
    return False


def _say(message: str) -> None:
    """A progress line goes to the ``whileai.simulations`` logger at INFO.
    When nothing is listening it also goes to stderr, so a script with no
    logging setup can tell a working run from a stuck one (#400); attach
    any handler (``logging.basicConfig()``) to take the stream over."""
    log.info("%s", message)
    if not someone_listens():
        print(message, file=sys.stderr, flush=True)


# Every number this module reads lives in ``whileai.simulations.defaults``
# with the reason for its value; the per-run ones are ``advanced`` keys on
# ``RunConfig.knobs``.
_FINGERPRINT_STOPWORDS = {"the", "a", "an", "to", "for", "and", "of", "on"}


def _agent_error_text(exc: BaseException) -> str:
    """The dropped-row sentinel. Names the exception type so a TypeError
    from a wrong signature reads differently from a RuntimeError inside."""
    return f"<agent error: {type(exc).__name__}: {public_llm_error(exc)}>"


FINISH_REASONS = ("stop", "length", "tool", "error")

__all__ = ["FINISH_REASONS", "Run", "progress_line"]

#: A queued job is ``(prompt, plan, meta, selection)``; older tuples are
#: shorter, so the two trailing slots are read by index with a length check.
_JOB_META = 2
_JOB_SELECTION = 3


def _finish_reason(raw: dict, steps: list, final_text: str) -> str:
    """Why the rollout ended, on the row where a trainer can read it.

    ``length``: a turn was cut by the reply token cap (the backend said
    so). ``error``: the agent raised. ``tool``: the last thing the agent
    did was call a tool and no final reply followed, so the turn budget
    ran out. ``stop``: the agent finished on its own. A callable agent may
    say it outright with ``finish_reason`` in what it returns. A length
    cut scored 0 teaches the cheapest fix, shorter thinking, before it
    teaches the task (#253), so the trainer masks these by default.
    """
    told = raw.get("finish_reason")
    if isinstance(told, str) and told in FINISH_REASONS:
        return told
    if final_text.startswith("<agent error:"):
        return "error"
    if any(isinstance(s, dict) and s.get("truncated") for s in steps):
        return "length"
    if not final_text.strip() and steps and isinstance(steps[-1], dict) and steps[-1].get("tool"):
        return "tool"
    return "stop"


#: Seconds in a minute and in an hour, for the clock text.
_MINUTE_S = 60
_HOUR_S = 60 * _MINUTE_S


def _clock_text(seconds: float) -> str:
    """Seconds as a short human span: ``45s``, ``1m40s``, ``1h4m``."""
    total = max(0, int(seconds))
    if total < _MINUTE_S:
        return f"{total}s"
    if total < _HOUR_S:
        minutes, rest = divmod(total, _MINUTE_S)
        return f"{minutes}m{rest}s" if rest else f"{minutes}m"
    hours, rest = divmod(total, _HOUR_S)
    minutes = rest // _MINUTE_S
    return f"{hours}h{minutes}m" if minutes else f"{hours}h"


def _left_text(seconds: float) -> str:
    """The same span, rounded, for an estimate nobody should read to the
    second: whole minutes over a minute, whole seconds under it."""
    if seconds < _MINUTE_S:
        return f"{max(1, round(seconds))}s"
    if seconds < _HOUR_S:
        return f"{max(1, round(seconds / _MINUTE_S))}m"
    return _clock_text(seconds)


def progress_line(
    rows: int,
    cap: int,
    situations: int,
    elapsed: float,
    *,
    min_rows_for_estimate: int = PROGRESS_MIN_ROWS_FOR_ESTIMATE,
    rerolled: int = 0,
    lost: int = 0,
    lost_by: Mapping[str, int] | None = None,
    resumed: int = 0,
) -> str:
    """One line of run progress, in the words a waiting person wants:

    ``12/64 rollouts, 3 situations written, 1m40s elapsed, ~5m left``

    and, once a rollout has been re-rolled or lost, why the run is slower
    than its rows say (#470):

    ``12/64 rollouts (4 resumed), ..., 6 re-rolled, 1 lost (1 agent error)``

    The estimate is the finished rate carried forward over the rows this
    call landed (resumed rows took no time here), and it is left off
    until ``PROGRESS_MIN_ROWS_FOR_ESTIMATE`` rollouts have landed, because
    before that it is the first rollout's latency dressed up as a forecast.
    """
    head = f"{rows}/{cap} rollouts"
    if resumed:
        head += f" ({resumed} resumed)"
    parts = [head, f"{situations} situations written", f"{_clock_text(elapsed)} elapsed"]
    landed = rows - resumed
    if landed >= min_rows_for_estimate and rows < cap and elapsed > 0:
        rate = landed / elapsed
        if rate > 0:
            parts.append(f"~{_left_text((cap - rows) / rate)} left")
    if rerolled:
        parts.append(f"{rerolled} re-rolled")
    if lost:
        by = ", ".join(f"{n} {r.replace('_', ' ')}" for r, n in (lost_by or {}).items() if n)
        parts.append(f"{lost} lost ({by})" if by else f"{lost} lost")
    return ", ".join(parts)


def _hit_length_cap(row: dict) -> bool:
    """A step the backend flagged as cut by its token cap, or a reply that
    ends mid-sentence by the hygiene rule."""
    from ..score.hygiene import is_truncated

    steps = row.get("steps") or []
    if any(isinstance(s, dict) and s.get("truncated") for s in steps):
        return True
    return is_truncated(row)


HARD_TIERS = ("ambiguous", "boundary", "adversarial")


def tier_mix_of(rows: Sequence[dict], requested: float) -> dict[str, Any]:
    """The difficulty mixture a set of rows carries against the share
    asked for: ``counts`` per tier, ``rows``, ``hard_share_requested``
    and ``hard_share_realized`` (the share of rows from ``HARD_TIERS``).
    One function so a single run and ``simulate(runs=N)`` (every run's
    rows together) count the same way."""
    counts: dict[str, int] = {}
    for t in rows:
        dims = t.get("scenario_dimensions")
        tier = str(t.get("tier") or "") or behavior_tier(dims if isinstance(dims, dict) else {})
        counts[tier] = counts.get(tier, 0) + 1
    n = sum(counts.values())
    hard = sum(counts.get(tier, 0) for tier in HARD_TIERS)
    realized = hard / n if n else None
    return {
        "hard_share_requested": round(float(requested), 4),
        "hard_share_realized": None if realized is None else round(realized, 4),
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "rows": n,
    }


class Run:
    """One ``simulate()`` call: inputs, build, loop, finish."""

    def __init__(self, cfg: RunConfig) -> None:
        self.c = cfg
        self.started = time.monotonic()
        # Set when the run stops; work that has not begun checks it and
        # returns at once, closing the window between a future being
        # marked running and its body actually starting.
        self.stopping = False
        # Read by the progress flush before the loop exists.
        self.inflight: dict = {}
        self.scenario_futs: list = []
        self.generated_pool: list[str] = []
        # The caller's own asks, filled by _seed_pool; never evicted by
        # the situations quota. Seeds the budget cannot pay for are
        # dropped before the run and named here.
        self.seed_prompt_set: set[str] = set()
        self.seeds_dropped: list[str] = []
        self.seed_budget_note: str = ""
        # Loop state the seed reservation reads; _init_loop_state resets both.
        self.prompt_rollouts: dict[str, int] = {}
        self.used: set[str] = set()
        # prompt -> (id(meta), situation key): the key is a pure function
        # of the prompt's meta, and every pool scan asked for it again
        self._situation_key_cache: dict[str, tuple[int, str]] = {}

    # ------------------------------------------------------------ driver

    def run(self) -> SimulationData:
        c = self.c
        if c.out_path is not None:
            # Writes the progress file before inspection or backend
            # construction so callers see it immediately.
            c.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_progress({"stage": "setup", "rows": 0, "scenario_s": 0, "rollout_s": 0})
            log.info("simulate setup rows=0")
        self._resolve_inputs()
        self._resolve_traces()
        self._amplify_seeds()  # after traces: failing asks seed the run too
        self._build_data()
        self._start_scene_thread()
        self._build_runner()
        self._build_generator()
        self._note_rule_axis_cap()
        self._init_loop_state()
        self._seed_pool()
        self._load_checkpoint()
        if c.out_path is not None:
            self._write_progress({"stage": "start", "rows": 0, "scenario_s": 0, "rollout_s": 0})
        self._start_writers()
        try:
            self._loop()
        finally:
            self._shutdown()
        return self._finish()

    # ------------------------------------------------------------ inputs

    def _resolve_inputs(self) -> None:
        c = self.c
        tools, policy, spec_sits = apply_spec(c.spec, c.tools, c.system_prompt, [])
        self.seed_prompts: list[str] = list(c.seed_prompts)
        self.seed_prompts.extend(str(s).strip() for s in spec_sits if str(s).strip())
        # simulate-from-seeds: a few example asks are a behavior request,
        # not the situation list. With an explicit situations=N target the
        # engine runs the coordinate search itself, amplifying the examples
        # across phrasing/stance/language axes to N distinct situations
        # before generation. Disclosed in search["seed_amplification"].
        self.seed_amp_report: dict | None = None
        self.profile = inspect(c.agent, tools=tools, system_prompt=policy)
        self.profile.rubric = str(c.rubric or "")
        self.tools: list[dict] = list(self.profile.tools or [])
        self.policy: str = str(self.profile.policy or "")
        # Generation-only teacher guidance. profile.policy and export stay plain.
        self.gen_policy = f"{self.policy}\n\n{c.scaffold_text}" if c.scaffold_text else self.policy
        # The deploy prompt the rows are generated under, as a short hash
        # and a head on every row and the full text once per run (#296):
        # a base rate measured under a full policy is not the base rate
        # under a bare prompt, and policy_version alone reads as a model id.
        self.system_prompt_sha, self.system_prompt_head, self.system_prompt_chars = (
            system_prompt_stamp(self.gen_policy)
        )
        self.writer_kind = kind_from_spec(c.spec, self.policy)
        # May be replaced by the backend spec once the runner is built.
        self.simulator = c.simulator
        self.drafted_tools: list[str] = []
        self.tool_draft_failed = False
        if (
            not self.tools
            and self.policy
            and self.simulator is not False
            and (c.agent is None or isinstance(c.agent, str))
            and not c.pinned_tasks
        ):
            # A description with no tools gives the writer and the world no
            # domain; draft the tool surface the described agent would have.
            # Pinned tasks (tasks=) bring their own prompts, so there is no
            # situation to anchor; a drafted tool surface there only invites
            # the policy to call tools that do not exist (Nemotron-8B answered
            # every text-to-SQL task with a tool call instead of a query).
            drafted = draft_tools(
                self.policy,
                backend_spec=self.simulator if isinstance(self.simulator, str) else None,
                kind=self.writer_kind,
            )
            if drafted:
                self.tools = drafted
                self.profile.tools = drafted
                self.drafted_tools = [d["function"]["name"] for d in drafted]
            else:
                # The run goes on with no tools, which is a different
                # dataset from the one asked for. Say so instead of
                # letting a tool-free run pass for the described agent.
                self.tool_draft_failed = True
        if c.agent is None and not self.tools and not self.policy:
            raise ValueError("simulate needs an agent, tools=, or a system prompt.")

    def _amplify_seeds(self) -> None:
        c = self.c
        # The asks the caller handed in (plus a spec's situations and any
        # failing asks mined from traces), before the writer mints
        # variants of them. Every one of these is run: the situations
        # quota never evicts them and a writer ask never takes their
        # rows. Only the budget can drop one, and _fit_seeds_to_budget
        # says which before the run starts.
        self.given_seeds: list[str] = list(dict.fromkeys(self.seed_prompts))
        self._fit_seeds_to_budget()
        if c.n_situations_target and len(self.given_seeds) > int(c.n_situations_target):
            # situations= below the number of seeds used to evict the
            # extra seeds. The seeds win: they are the asks the caller
            # wrote, so the target grows to hold them.
            c.n_situations_target = len(self.given_seeds)
        # Amplifies seed prompts only when seeds= is given; advanced["seed_prompts"]
        # stays literal. Offline runs (simulator=False) make no network calls.
        # Runs after inspect() so the writer hint carries the resolved policy.
        if (
            (c.seeds or getattr(self, "failure_seeds", 0))
            and self.seed_prompts
            and c.n_situations_target
            and len(self.seed_prompts) < int(c.n_situations_target)
            and self.simulator is not False
        ):
            given = len(self.seed_prompts)
            self.seed_prompts = amplify_seeds(
                self.seed_prompts,
                int(c.n_situations_target),
                policy=self.policy,
                backend_spec=self.simulator if isinstance(self.simulator, str) else None,
            )
            self.seed_amp_report = {
                "given": given,
                "target": int(c.n_situations_target),
                "total": len(self.seed_prompts),
                "minted": len(self.seed_prompts) - given,
            }

    def _resolve_traces(self) -> None:
        c = self.c
        self.trace_rows: list[dict] = []
        self.trace_focused = False
        self.optimizer_state: dict | None = None
        self.allocation_hits = {"n": 0}
        self.dimensions = c.dimensions
        self.arm_weights = c.arm_weights
        if c.traces is not None:
            # same normalization as trace_report: a messages-only export
            # otherwise mines as zero tools and the grid is never aimed
            self.trace_rows = load_traces(c.traces)
            # A capability failure carries no tool, fault or world-state
            # signal: the same two tools, no fault, the wrong SQL. Aiming
            # the grid at axes cannot see it (SQL dogfood, 2026-09-16:
            # 41 failures, empty emphasis). The failing asks themselves
            # are the target, so they seed the run and are amplified into
            # variants; the leakage rule keeps the originals out.
            failing = []
            seen = set(self.seed_prompts)
            for row in self.trace_rows:
                if row.get("reward") not in (0, 0.0, False):
                    continue
                ask = str(row.get("prompt") or "").strip()
                if ask and ask not in seen:
                    seen.add(ask)
                    failing.append(ask)
                if len(failing) >= c.knobs.failing_seeds_cap:
                    break
            self.failure_seeds = len(failing)
            self.seed_prompts.extend(failing)
            # The optimizer's memory feeds the run it aims: regions from
            # the whole trace history, budget shares from their lifecycle.
            # Cells whose coordinates intersect a hot region's expansion
            # recipe draw extra weight proportional to its share; cells
            # outside every recipe keep base weight - that is the
            # exploration reserve in action.
            self.optimizer_state = behavior_state(self.trace_rows, targeted=c.targeted_regions)
            if self.trace_rows and self.dimensions is None and c.resolved_strategy != "broad":
                # trace: denser near observed behaviors, background kept.
                # targeted: drops tools the traces did not touch, narrowing the space.
                self.dimensions = dimensions_from_traces(
                    self.trace_rows,
                    self.tools,
                    self.policy,
                    broaden=c.resolved_strategy != "targeted",
                )
                self.trace_focused = True
                # the grid flips 90% of cells to success for cold starts;
                # traces that show faults are asking for the fault cells
                if mine_traces(self.trace_rows)["faults"]:
                    c.advanced.setdefault("prefer_success", False)
        # The weight only applies over a trace-focused grid (its front-half
        # ordering is what the bias aims at) and only when nonzero; anything
        # else is exactly the unsteered draw and records no applied weight.
        self.applied_steering = (
            float(c.steering_weight) if c.steering_weight and self.trace_focused else None
        )

    def _allocation_boost(self, assignment: dict) -> float:
        knobs = self.c.knobs
        factor = 1.0
        for region in self.optimizer_state["regions"]:
            share = region.get("budget_share") or 0.0
            if share <= 0:
                continue
            recipe = region.get("recipe") or {}
            match = 0.0
            tools_r = recipe.get("tool") or []
            if tools_r and str(assignment.get("tool")) in tools_r:
                match += knobs.allocation_tool_weight
            conds = recipe.get("tool_condition") or []
            if conds and str(assignment.get("tool_condition")) in conds:
                match += knobs.allocation_condition_weight
            if match:
                factor += knobs.allocation_gain * share * match
        return factor

    def _apply_allocation(self, region_list) -> None:
        """Boost grid cells that sit inside a hot trace region. No-op cold."""
        if self.optimizer_state is None:
            return
        for region in region_list or []:
            boost = self._allocation_boost(region.get("assignment") or {})
            if boost != 1.0:
                self.allocation_hits["n"] += 1
                region["weight"] = round(float(region.get("weight") or 0.0) * boost, 6)

    # ------------------------------------------------------------- build

    def _build_data(self) -> None:
        c = self.c
        data = SimulationData(profile=self.profile, arm_weights=dict(SEARCH_ARMS))
        data.scaffold_chars = len(c.scaffold_text)
        data.system_prompts = {self.system_prompt_sha: self.gen_policy}
        data.mode = c.topo["mode"]
        data.repeat_policy = c.topo["repeat_policy"]
        data.n_situations = c.n_situations_target
        data.requests_per_situation = c.n_req
        data.rollouts_per_request = c.repeat_count
        data.unique_situations = c.unique_cards
        note_stage(data, "agent ingestion")
        if self.tool_draft_failed:
            data.degraded.append("tool_draft_unavailable")
        self.data = data
        if self.seed_budget_note:
            data.warnings.append(self.seed_budget_note)
            data.search["seeds_dropped"] = list(self.seeds_dropped)
        self.scene_box: dict[str, Any] = {"brief": ""}
        self.shape_box: dict[str, dict] = {}
        self.trace_exemplars: dict[str, list] = {}
        if self.trace_rows:
            # Result shapes mined from traces become the templates first;
            # the model-written pass below fills only tools the traces
            # did not show.
            self.trace_exemplars = mine_result_exemplars(self.trace_rows)
            self.shape_box.update(exemplar_result_shapes(self.trace_exemplars))

    def _fit_seeds_to_budget(self) -> None:
        """Work out which seeds the budget cannot pay for, before the run.

        Every seed costs ``repeats`` rows. When ``budget`` is smaller
        than that bill the run used to spend it on whichever seeds came
        first, and the caller had to count asks in the output to notice
        the rest were gone. The seeds that do not fit are dropped here,
        named in ``warnings`` and listed in ``search["seeds_dropped"]``
        once the run object exists.
        """
        c = self.c
        seeds = list(self.given_seeds)
        if not seeds or c.budget is None:
            return
        k = max(1, int(c.repeat_count))
        covered = max(1, int(c.budget) // k)
        if covered >= len(seeds):
            return
        dropped = seeds[covered:]
        gone = set(dropped)
        self.seed_prompts = [p for p in self.seed_prompts if p not in gone]
        self.given_seeds = seeds[:covered]
        self.seeds_dropped = dropped
        self.seed_budget_note = (
            f"budget={int(c.budget)} covers {covered} of {len(seeds)} seeds at "
            f"repeats={k}; raise budget to {len(seeds) * k}+ or drop seeds"
        )

    def _seeds_waiting(self) -> int:
        """Seeds the run has not started yet. Their situation slots and
        their rows are held for them: a writer ask never takes one."""
        if not self.seed_prompt_set:
            return 0
        return sum(1 for p in self.seed_prompt_set if p not in self.used)

    def _room_for_a_new_ask(self) -> bool:
        """False when every row left in the budget is owed to a seed."""
        c = self.c
        waiting = self._seeds_waiting()
        if not waiting or c.budget is None:
            return True
        scheduled = sum(self.prompt_rollouts.values())
        return int(c.cap) - scheduled - waiting * max(1, int(c.repeat_count)) > 0

    def _start_scene_thread(self) -> None:
        self.scene_thread: threading.Thread | None = None
        use_model_writer = not (
            self.simulator is False
            or (callable(self.simulator) and not isinstance(self.simulator, str))
        )
        if not use_model_writer:
            return
        scene_spec = self.simulator if isinstance(self.simulator, str) else None
        self.scene_thread = threading.Thread(
            target=self._fill_scene, args=(scene_spec, time.monotonic()), daemon=True
        )
        self.scene_thread.start()

    def _fill_scene(self, scene_spec: str | None, scene_t0: float) -> None:
        data = self.data
        # Shapes first: every rollout benefits, and the writer can run
        # its first waves without the scene brief.
        shapes = write_result_shapes(self.tools, backend_spec=scene_spec)
        if shapes:
            # setdefault, not update: a template mined from a real
            # trace outranks a model-written guess for that tool.
            for shape_name, shape in shapes.items():
                self.shape_box.setdefault(shape_name, shape)
        elif "result_shapes_unavailable" not in data.degraded:
            data.degraded.append("result_shapes_unavailable")
        brief = write_scene_brief(
            self.tools, self.gen_policy, backend_spec=scene_spec, kind=self.writer_kind
        )
        self.scene_box["brief"] = brief
        data.scene_brief = brief
        data.scene_brief_seconds = time.monotonic() - scene_t0
        if not brief and "scene_brief_unavailable" not in data.degraded:
            data.degraded.append("scene_brief_unavailable")

    def _build_runner(self) -> None:
        c = self.c
        self.fault_plans: dict = {}
        kind = self.profile.transport
        self.turns = (
            default_max_turns(n_tools=len(self.tools))
            if c.max_turns is None
            else max(1, int(c.max_turns))
        )
        self.turn_stats = new_turn_stats()
        # Resolve the opening-side topology axis: explicit value, or the
        # share observed in this run's traces ("auto"). Model-backed
        # runners only; callable agents cannot be asked for an opener.
        opening_req = c.opening_req
        if opening_req == "auto":
            self.opening_rate = opening_share(self.trace_rows) if self.trace_rows else 0.0
            self.opening_source = "traces"
        elif opening_req == "agent":
            self.opening_rate, self.opening_source = 1.0, "explicit"
        elif isinstance(opening_req, float):
            self.opening_rate, self.opening_source = opening_req, "explicit"
        else:
            self.opening_rate, self.opening_source = (
                0.0,
                ("explicit" if opening_req == "user" else "default"),
            )
        runner_kw: dict[str, Any] = {
            "fault_plans": self.fault_plans,
            "max_turns": self.turns,
            "avg_turns": float(c.avg_turns),
            "min_user_turns": c.min_user_turns,
            "patience": c.patience,
            "turn_stats": self.turn_stats,
        }
        if c.temperature is not None:
            runner_kw["temperature"] = float(c.temperature)
        if c.agent_max_tokens:
            runner_kw["max_tokens"] = int(c.agent_max_tokens)
        if c.logprobs:
            runner_kw["logprobs"] = c.logprobs
        if c.user_model:
            runner_kw["user_model"] = c.user_model
        if c.user_temperature is not None:
            runner_kw["user_temperature"] = float(c.user_temperature)
        if c.world_options is not None:
            runner_kw["world_options"] = c.world_options
        # Who plays the user: the agent's own model unless user_model= names
        # another. Callable and HTTP agents take one message and never get a
        # simulated user, so they carry no tag.
        self.agent_model: str | None = None
        self.user_model: str | None = None
        # <model>@<hash>: the suffix is lineage.system_prompt_sha, not a second hash.
        self.policy_version = f"{c.model_version_tag}@{self.system_prompt_sha}"
        if c.execute is not None:
            runner_kw["execute"] = c.execute
        if c.backend:
            spec_backend = backend_spec(c.backend)
            url, model_name = parse_backend_spec(spec_backend)
            self.agent_model = model_name
            self.runner = local_model(
                url,
                model_name,
                tools=self.tools,
                system=self.gen_policy,
                timeout=c.rollout_timeout,
                opening_rate=self.opening_rate,
                result_shapes=self.shape_box,
                **runner_kw,
            )
            self.simulator = self.simulator if self.simulator is not None else spec_backend
        elif c.agent is None or kind not in {"callable", "backend_spec", "http"}:
            self.agent_model = parse_backend_spec(default_agent_spec())[1]
            self.runner = hosted_model(
                self.tools,
                system=self.gen_policy,
                timeout=c.rollout_timeout,
                opening_rate=self.opening_rate,
                result_shapes=self.shape_box,
                **runner_kw,
            )
        else:
            self.runner, kind = resolve(
                c.agent,
                tools=self.tools,
                policy=self.policy,
                opening_rate=self.opening_rate,
                result_shapes=self.shape_box,
                timeout=c.rollout_timeout,
                **runner_kw,
            )
            if kind == "backend_spec":
                self.agent_model = parse_backend_spec(c.agent)[1]
        self.kind = kind
        # How the rollouts were sampled, read off the runner that samples
        # them: a model backend knows its temperature, reply budget and
        # model (the defaults it resolved, not the knobs as passed). A
        # callable agent is the caller's, so the row says only what they
        # told simulate(sampling=), else None.
        self.sampling: dict[str, Any] | None = getattr(self.runner, "sampling", None)
        if self.sampling is None and c.sampling is not None:
            self.sampling = dict(c.sampling)
        if self.agent_model is not None:
            self.user_model = (
                parse_backend_spec(c.user_model)[1] if c.user_model else self.agent_model
            )

    def _note_rule_axis_cap(self) -> None:
        """Say, once per run, when the policy has more clauses than the
        grid's rule axis holds: the rows cover the first ``RULE_CAP``
        clauses and none of the rest, and nothing else in the run says so
        (#391). A caller who set ``dimensions={"rule": [...]}`` chose the
        axis, so the note is theirs to skip."""
        dims = self.c.dimensions
        if isinstance(dims, Mapping) and dims.get("rule"):
            return
        rules, total = rule_axis(self.policy, cap=RULE_CAP)
        if total <= len(rules):
            return
        note = (
            f"The policy has {total} clauses and the grid's rule axis holds {RULE_CAP} "
            f"(RULE_AXIS_CAP_GRID; ZP_RULE_CAP overrides), so the rows cover the first "
            f"{RULE_CAP} clauses in document order and none of the other {total - len(rules)}. "
            "Pass dimensions={'rule': [...]} with the clauses that matter, or split the "
            "policy and run each part."
        )
        self.data.warnings.append(note)
        log.warning(note)

    def _build_generator(self) -> None:
        c = self.c
        self.generator = make_default_generator(
            self.tools,
            policy=self.policy,
            per_round=c.pool_size,
            seed=c.seed,
            dimensions=self.dimensions,
            arm_weights=self.arm_weights,
            simulator=self.simulator,
            hard_share=c.hard_share,
            world=c.world_options,
            kind=self.writer_kind,
            scenarios_per_request=c.scenarios_per_request,
            completions_per_request=c.completions_per_request,
            distinct_cards=c.distinct_cards,
            extra_cards=c.extra_cards,
            scene_brief=self.scene_box["brief"],
            time_budget=c.time_budget,
            run_started=self.started,
            mode=c.topo["mode"],
            steering_weight=self.applied_steering,
            **c.advanced,
        )
        gen = self.generator
        self._apply_allocation(getattr(gen, "regions", None))
        self.planned_cell_keys = {
            json.dumps(region["assignment"], sort_keys=True, default=str)
            for region in (getattr(gen, "regions", None) or [])
            if isinstance(region, dict)
            and isinstance(region.get("assignment"), dict)
            and region["assignment"]
        }
        # Search-arm weights, reallocated after every batch by yield.
        self.search: dict[str, float] = dict(SEARCH_ARMS)
        gen.arm_weights = dict(SEARCH_ARMS)
        model_obj = getattr(gen, "model", None)
        if model_obj is not None and hasattr(model_obj, "arm_weights"):
            model_obj.arm_weights = dict(SEARCH_ARMS)
        # The model arm keeps its own region list; applies the allocation
        # boost to it directly so round-1 cards and short runs get it too.
        self._apply_allocation(getattr(model_obj, "regions", None))
        model_backend = getattr(getattr(gen, "model", None), "backend_spec", None)
        # Who writes the situations: the writer model's tag, a callable
        # writer's name, or "template" when no model writes (simulator=False).
        self.writer_model = "template"
        if isinstance(model_backend, str):
            try:
                self.writer_model = parse_backend_spec(model_backend)[1]
            except ValueError:
                self.writer_model = model_backend
        elif model_obj is not None and callable(model_obj):
            self.writer_model = getattr(model_obj, "__name__", "callable-writer")
        if isinstance(model_backend, str):
            try:
                hosted_url, _ = parse_backend_spec(model_backend)
            except ValueError:
                hosted_url = None
            else:
                auth_err = missing_hosted_key(hosted_url)
                if auth_err:
                    raise RuntimeError(
                        auth_err + " The situation writer runs on hosted Qwen "
                        "by default, even with your own agent=. Alternatives: "
                        "agent='openai:<model>' with OPENAI_API_KEY runs writer "
                        "and agent on your key; simulator=False uses the "
                        "built-in template writer with no model at all."
                    )
                threading.Thread(
                    target=touch_hosted,
                    args=(hosted_url,),
                    kwargs={"timeout": c.knobs.hosted_touch_s},
                    daemon=True,
                ).start()
                # A trial key buys about a dozen hosted situations a day.
                # Saying so after the run has spent them is no use, so the
                # note lands before the first writer wave. Only for the
                # writer this run picked for itself (simulator= brings its
                # own model, and no quota of ours) on the account route,
                # which is what the allowance meters (VLLM_API_KEY goes to
                # the shared pool and spends no trial), and only for a
                # saved key whose tier the credentials file recorded:
                # reading it costs no network call.
                if c.simulator is None and _account_url(hosted_url):
                    trial = trial_prerun_note()
                    if trial:
                        self.data.warnings.append(trial)
                        log.warning(trial)
        self.fault_plans.update(gen.fault_plans)
        self.declared = {
            str((t.get("function", t) or {}).get("name", "")) for t in self.tools or []
        } - {""}
        self.data.budget = int(c.cap)
        self.search_plan = sampling_plan(c.time_budget)
        self.action_shapes, _ = action_space_targets(
            self.tools,
            max_len=int(self.search_plan["max_shape_len"]),
            cap=int(self.search_plan["enum_cap"]),
        )
        self.induced_shape_keys: set[str] = set()
        self.resolved_embedder = resolve_embedder(c.embedder)
        self.data.embedder_name = str(getattr(self.resolved_embedder, "name", "unknown"))
        self.data.semantic = is_semantic(self.resolved_embedder)
        # The default embedder is the hash; a run that never asked for a
        # semantic one is not degraded, so the note only lands when the
        # requested embedder fell back.
        if not self.data.semantic and c.embedder not in (None, "hash"):
            self.data.degraded.append("semantic_embedding_unavailable")
        self.archive = EmbeddingArchive(self.data.embedder_name, self.data.semantic)

    # ---------------------------------------------------------- rollouts

    def _record_shapes(self, rows: list[dict]) -> None:
        known = {shape.key() for shape in self.action_shapes}
        for row in rows:
            self.induced_shape_keys.update(
                induced_keys_from_trajectory(row, self.action_shapes, self.tools)
            )
            observed = shape_from_trajectory(row, self.tools)
            if observed is not None and observed.key() not in known:
                self.action_shapes.append(observed)
                known.add(observed.key())
            if isinstance(row, dict):
                row["tools_used"] = [
                    step.get("tool")
                    for step in (row.get("steps") or [])
                    if isinstance(step, dict) and step.get("tool")
                ]

    def _scaled(self, plan: dict | None, key: str = "") -> dict | None:
        """The fault plan this prompt keeps at ``fault_rate``, or nothing."""
        if not plan or self.c.fault_rate <= 0:
            return None
        if not keep_fault_plan(key, self.c.fault_rate, self.c.seed):
            return None
        return plan

    @staticmethod
    def _realized_dims(steps: list, faults: list) -> dict:
        # Rows without a grid cell (seeds, open-ended, behavior cards)
        # still get auditable coordinates - realized from what actually
        # happened, marked so audits can tell assigned from observed.
        tools_called = [
            str(s.get("tool")) for s in steps or [] if isinstance(s, dict) and s.get("tool")
        ]
        condition = "success"
        for s in steps or []:
            result = s.get("result") if isinstance(s, dict) else None
            status = str(result.get("status")) if isinstance(result, dict) else ""
            if status and status not in OK_STATUSES:
                condition = status
                break
        else:
            # A plan is {tool: {"mode": ...}} (clean_faults); older callers
            # passed a list of {"fault": ...}. Both name the condition.
            plans = list(faults.values()) if isinstance(faults, dict) else list(faults or [])
            for plan in plans:
                if not isinstance(plan, dict):
                    continue
                kind = str(plan.get("mode") or plan.get("fault") or "")
                if kind:
                    condition = kind
                    break
        return {
            "tool": tools_called[0] if tools_called else "unrelated",
            "tool_condition": condition,
            "origin": "realized",
        }

    def _writer_of(self, meta: dict) -> str:
        """The model tag that wrote one prompt. Rows a model did not write
        say so: ``seed`` is the caller's own opener, ``template`` the
        built-in writer and its mutations. A replay from ``tasks=`` keeps
        the writer of the run it replays (the prompt was written once, by
        that model); ``lineage.replayed`` says it is a replay. ``pinned``
        only when the source rows carried no writer at all."""
        origin = str(meta.get("generator") or "")
        if origin == "model":
            return self.writer_model
        if origin == "user":
            return "seed"
        if origin == "pinned":
            return str(meta.get("writer_model") or "pinned")
        return "template"

    def _build_row(self, job: tuple) -> dict:
        """Run one rollout on a worker thread and shape it as a row."""
        c = self.c
        prompt, rollout, meta, selection = job
        if self.stopping:
            return {
                "_skipped": True,
                "prompt": prompt,
                "steps": [],
                "final_text": "",
                "reward": None,
            }
        meta = dict(meta or {})
        assignment = meta.get("assignment") or meta.get("scenario_dimensions")
        if prompt in self.pinned_plans:
            faults = self.pinned_plans[prompt] or None
        else:
            faults = self._scaled(
                self.generator.fault_plans.get(prompt) or self.fault_plans.get(prompt), prompt
            )
        # the caller's execute= world reads this to know which rollout it answers
        current_rollout.prompt = prompt
        current_rollout.rollout_index = rollout
        current_rollout.seed = meta.get("seed", c.seed)
        plan = clean_faults(faults)
        current_rollout.faults = plan
        current_rollout.world_state = row_world(assignment) or ""
        current_rollout.tools = list(self.tools or [])
        current_rollout.privileged = privileged_context(assignment, plan) if assignment else None
        try:
            raw = self.runner(prompt)
        except Exception as exc:
            raw = {"steps": [], "final_text": _agent_error_text(exc)}
        raw = raw if isinstance(raw, dict) else {"steps": [], "final_text": str(raw)}
        if "steps" not in raw and "final_text" not in raw:
            # An OpenAI-style message or some other shape: the callable
            # contract is {"steps": [...], "final_text": str}. Reported
            # like a raise so the run does not end blaming the writer.
            keys = sorted(str(k) for k in raw)
            raw = {
                "steps": [],
                "final_text": _agent_error_text(
                    TypeError(f"agent returned keys {keys} instead of steps and final_text")
                ),
            }
        if not assignment:
            assignment = self._realized_dims(raw.get("steps") or [], clean_faults(faults))
        semantic = self.data.semantic
        t = {
            # Born stamped: the streamed file and a later save() must agree.
            SCHEMA_KEY: SCHEMA_VERSION,
            "scenario_id": meta.get("region_id")
            or "probe_" + hashlib.sha256(str(prompt).encode()).hexdigest()[:SCENARIO_ID_CHARS],
            "scenario_dimensions": assignment,
            "arm": meta.get("arm") or "unattributed",
            "prompt": prompt,
            "world_state": row_world(assignment),
            "faults": clean_faults(faults),
            "steps": raw.get("steps") or [],
            "final_text": str(raw.get("final_text", "")),
            # topology axis: which side opened this conversation
            **({"opener": str(raw["opener"]), "opening": "agent"} if raw.get("opener") else {}),
            "behavior_signature": None,
            "reward": None,
            "grader_reason": None,
            "reason": None,
            "rollout_index": rollout,
            "model_version": c.model_version_tag,
            # Who wrote this prompt and who played the user, next to who
            # answered: a row that cannot say is a row nobody can audit.
            "writer_model": self._writer_of(meta),
            "user_model": self.user_model,
            # Which deploy prompt, so a base rate can be audited later
            # (#296). The text is in data.system_prompts under the hash.
            "lineage": {
                "system_prompt_sha": self.system_prompt_sha,
                "system_prompt_head": self.system_prompt_head,
                "system_prompt_chars": self.system_prompt_chars,
                # A replay from tasks= keeps the original writer's name on
                # writer_model (the situation was written once); this is
                # where the fact that it is a replay lives.
                **({"replayed": True} if meta.get("generator") == "pinned" else {}),
            },
            "seed": meta.get("seed", c.seed),
            "semantic_cluster": None if not semantic else selection.get("cluster"),
            "semantic_novelty": None if not semantic else selection.get("novelty"),
            "parent_failure_id": meta.get("parent_failure_id") or meta.get("parent"),
            "selection_reason": selection.get("reason"),
        }
        if t["arm"] == "failure_mutation":
            aim = self.failure_aim.get(str(t["parent_failure_id"] or ""), "unattributed")
            self.mutation_aims[aim]["rows"] += 1
        # Cards drawn the steered way carry the mark onto the row, so
        # metadata's targeted/background split counts real draws.
        steering = meta.get("steering")
        if isinstance(steering, dict) and steering.get("origin"):
            t["steering"] = dict(steering)
        # The teacher's block, born with the row: what the world knows and
        # what the checklist expects. Exporters scrub it (_EXPORT_NEVER);
        # leak_report reads it. Before this it was empty on every run that
        # did not attach a rubric, so a leak check on it passed vacuously.
        privileged = privileged_context(assignment, t["faults"])
        if privileged:
            t["privileged"] = privileged
        # A seeded_agent says what it did wrong on purpose; the row keeps it.
        seeded = raw.get("seeded")
        if isinstance(seeded, list):
            t["seeded"] = [str(x) for x in seeded]
        t.update(_row_conversation(meta, prompt, c.seed))
        t["behavior_signature"] = behavior_signature(t)
        t["finish_reason"] = _finish_reason(raw, t["steps"], t["final_text"])
        # The simulated person gave up on the agent's question (#289): a
        # rubric about asking reads it here, next to the question itself.
        if raw.get("ended_by"):
            t["ended_by"] = str(raw["ended_by"])
        # Sampling facts roll up from the agent turns: the summed logprob
        # and token count a trainer needs for an importance ratio or a KL.
        lp_steps = [
            s
            for s in t["steps"]
            if isinstance(s, dict)
            and isinstance(s.get("logprob"), (int, float))
            and not isinstance(s.get("logprob"), bool)
        ]
        if lp_steps:
            t["logprob"] = round(sum(float(s["logprob"]) for s in lp_steps), 6)
            t["n_tokens"] = sum(int(s.get("n_tokens") or 0) for s in lp_steps)
            tokens = [
                float(x)
                for s in lp_steps
                for x in (s.get("token_logprobs") or [])
                if isinstance(x, (int, float)) and not isinstance(x, bool)
            ]
            if tokens:
                # per-token, in generation order across the agent's turns:
                # what a truncated-importance-sampling ratio is built from
                t["token_logprobs"] = tokens
        # Which policy, exactly, and how it was sampled (async RL,
        # Noukhovitch et al. 2024, arXiv:2410.18252): a later update needs
        # the sampler's version and temperature on the row, not in a notebook.
        t["policy_version"] = self.policy_version
        t["sampling"] = dict(self.sampling) if self.sampling is not None else None
        # Token usage rolls up the same way, so a row says what it cost and a
        # trace built from it can carry gen_ai.usage.* on every model turn.
        used = [
            s for s in t["steps"] if isinstance(s, dict) and isinstance(s.get("input_tokens"), int)
        ]
        if used:
            t["usage"] = {
                "input_tokens": sum(int(s.get("input_tokens") or 0) for s in used),
                "output_tokens": sum(int(s.get("output_tokens") or 0) for s in used),
            }

        return t

    @staticmethod
    def _error_row(job: tuple, exc: Exception) -> dict:
        t = {
            SCHEMA_KEY: SCHEMA_VERSION,
            "steps": [],
            "final_text": _agent_error_text(exc),
            "arm": (job[2] or {}).get("arm") or "unattributed",
            "prompt": job[0],
            "reward": None,
        }
        t["behavior_signature"] = behavior_signature(t)
        return t

    def _note_lost(self, t: dict) -> None:
        """A rollout that did not become a row: an agent error is counted.

        A dead agent is called off early. Every lost rollout is re-rolled
        up to repeat_count times and a fresh situation then fills the
        slot, so an agent that raises on every call burned ~15 calls per
        budgeted row before the writer ran dry (#88). Once no row has
        landed and the errors pass the allowance, the run stops as
        agent_failed. A run with any surviving row keeps re-rolling.
        """
        final = str((t or {}).get("final_text") or "")
        if final.lower().startswith("<agent error"):
            self.agent_errors += 1
            if not self.first_agent_error:
                self.first_agent_error = final[len("<agent error: ") :].rstrip(">")
            if _timeout_error(final):
                self.timed_out += 1
            if _timeout_error(final) and not self.timeout_noted:
                # A served model that scaled to zero outlives a short
                # timeout on its first request (#302); the rollout is
                # dropped and, without this, the run looks finished.
                self.timeout_noted = True
                note = (
                    f"An agent call timed out after {self.c.rollout_timeout:.0f} s and the "
                    "rollout was dropped. A served model that scaled to zero takes two to "
                    "three minutes to answer its first request: raise timeout= on "
                    "local_model (or simulate(timeout=)), or send one throwaway request "
                    "first so the endpoint is warm."
                )
                self.data.warnings.append(note)
                log.warning(note)
            auth = _auth_error(final[len("<agent error: ") :].rstrip(">"))
            if auth and not self.stopping:
                # a rejected key fails every rollout the same way; no
                # allowance, no re-roll, stop on the first one
                self.stopping = True
                self.agent_dead = True
                self.auth_error = auth
            if (
                not self.stopping
                and not self.data.trajectories
                and self.agent_errors >= self.agent_error_allowance
            ):
                self.stopping = True
                self.agent_dead = True

    def _discard_lost(self, t: dict) -> None:
        """The re-roll allowance is spent (or there was none): the rollout
        is lost for good, under the reason it was unusable."""
        self.cap_lifted["lost"] += 1
        reason = _unusable_reason(t) or "empty_reply"
        self.lost_by[reason] = self.lost_by.get(reason, 0) + 1
        note_stage(self.data, "rollout failure discarded")

    def _record_experiment_knobs(self) -> None:
        """Every other ``simulate()`` parameter that makes one run a
        different experiment from the next, as the run resolved it, so
        ``report()`` is the whole record and nothing lives only in the
        caller's notebook. ``mode``, ``repeats``, ``budget``, ``until``
        and the topology counts are written by the caller of this method.

        ``hard_share`` is the dial as resolved (the default when unset);
        what the rows actually drew stays in ``search["tier_mix"]``.
        ``fault_rate`` is this run's rate. ``world["default_fault_rate"]``
        is not: it is the rate a fault plan with no rate of its own fires
        at, the world's default, so ``world_note`` says so next to it.
        """
        c = self.c
        cov = self.data.coverage
        cov["world_note"] = (
            "world.default_fault_rate is the rate a fault plan with no rate of its "
            "own fires at (the world default); this run's fault rate is fault_rate."
        )
        cov["hard_share"] = HARD_SHARE if c.hard_share is None else float(c.hard_share)
        cov["fault_rate"] = float(c.fault_rate)
        cov["seed"] = int(c.seed)
        cov["runs"] = 1
        # ``budget`` is the row cap of one Run. simulate(runs=N) repeats the
        # Run N times with the same kwargs, so the cap applies per run and
        # the rows returned are up to N x budget; the key says so.
        cov["budget_per_run"] = int(c.cap)
        cov["strategy"] = c.resolved_strategy
        cov["time_budget"] = c.time_budget
        cov["reproducible"] = bool(c.reproducible)
        cov["concurrency"] = int(c.concurrency)
        dims = c.dimensions
        cov["dimensions"] = (
            {str(k): list(v) for k, v in dims.items()} if isinstance(dims, Mapping) else dims
        )
        # the search-arm weights the run started from; ``data.arm_weights``
        # is where the reallocation by yield left them
        cov["arm_weights"] = dict(c.arm_weights) if c.arm_weights else dict(SEARCH_ARMS)
        cov["tasks"] = len(c.pinned_tasks)
        cov["traces"] = len(self.trace_rows)
        cov["seeds"] = len(c.seeds or [])
        grader = c.grader
        cov["grader"] = (
            None if grader is None else getattr(grader, "__name__", type(grader).__name__)
        )
        cov["grade"] = bool(c.grade)
        cov["llm_grade"] = bool(c.llm_grade)
        # who did which job: the situation writer ("template" offline),
        # the agent's model (a callable agent's name), the simulated user
        cov["simulator"] = self.writer_model
        # the same value under the name every row carries, so a reader who
        # knows the row key finds it on the report too
        cov["writer_model"] = self.writer_model
        cov["agent_model"] = c.model_version_tag
        cov["user_model"] = self.user_model
        cov["max_turns"] = c.max_turns
        cov["avg_turns"] = c.avg_turns
        cov["temperature"] = c.temperature
        cov["sampling"] = None if c.sampling is None else dict(c.sampling)
        cov["timeout"] = c.rollout_timeout
        cov["logprobs"] = c.logprobs

    def _delivered(self) -> dict:
        """What the generation knobs actually produced, beside what was asked.

        A knob that does not deliver is invisible from the data alone, and
        every one of these has been measured missing its setting: fault_rate
        0.5 reaching 29% of a pool and 0.0 still firing on 34 of 504 rows,
        avg_turns 6 measuring 0.44 mean user turns, and a stance request
        arriving as the easiest tier because ordinary is floored at half the
        pool. A caller reading only the setting describes an intention, not
        their data, and an agent driving these knobs cannot correct what it
        cannot see.

        Rows carry the truth: ``faults`` per row, ``tier`` per row, the
        stance in ``scenario_dimensions``, and the user turns in
        ``messages``. This reports both numbers so the gap is a fact rather
        than an inference.
        """
        rows = list(self.data.trajectories)
        n = len(rows)
        if not n:
            return {}

        def share(pred) -> float:
            return round(sum(1 for t in rows if pred(t)) / n, 4)

        def mix(get) -> dict:
            out: dict[str, int] = {}
            for t in rows:
                key = str(get(t) or "") or "unlabelled"
                out[key] = out.get(key, 0) + 1
            return dict(sorted(out.items(), key=lambda kv: -kv[1]))

        def dim(t: dict, name: str) -> Any:
            d = t.get("scenario_dimensions")
            return d.get(name) if isinstance(d, dict) else None

        turns = [
            sum(
                1
                for m in (t.get("messages") or [])
                if isinstance(m, dict) and m.get("role") == "user"
            )
            for t in rows
        ]
        return {
            "rows": n,
            "fault_share": share(lambda t: bool(t.get("faults"))),
            "tier_mix": mix(lambda t: t.get("tier")),
            "stance_mix": mix(lambda t: dim(t, "stance")),
            "mean_user_turns": round(sum(turns) / n, 2),
            "user_turns_3plus_share": round(
                sum(1 for x in turns if x >= DELIVERED_LONG_CONVERSATION_TURNS) / n, 4
            ),
        }

    def _warn_on_undelivered(self, requested: dict, delivered: dict) -> None:
        """Say so when a knob missed its setting by enough to change a result.

        ``requested`` here is what the caller set, not the resolved config:
        the call site strips the knobs that carry a default and the turn
        knob when the agent was played single-turn (#476).
        """
        if not delivered:
            return
        gaps = []
        want_fault = requested.get("fault_rate")
        got_fault = delivered.get("fault_share")
        if isinstance(want_fault, (int, float)) and isinstance(got_fault, (int, float)):
            if want_fault > 0 and got_fault < want_fault * DELIVERED_FAULT_SHORTFALL:
                gaps.append(
                    f"fault_rate={want_fault} but {100 * got_fault:.0f}% of rows carry a fault"
                )
            elif want_fault == 0 and got_fault > DELIVERED_FAULT_LEAK:
                gaps.append(f"fault_rate=0 but {100 * got_fault:.0f}% of rows carry a fault")
        want_turns = requested.get("avg_turns")
        got_turns = delivered.get("mean_user_turns")
        if (
            isinstance(want_turns, (int, float))
            and isinstance(got_turns, (int, float))
            and want_turns >= DELIVERED_TURNS_MIN_REQUEST
            and got_turns < want_turns * DELIVERED_TURNS_SHORTFALL
        ):
            gaps.append(f"avg_turns={want_turns:g} but the mean is {got_turns} user turns")
        asked = requested.get("stance")
        if asked:
            got = delivered.get("stance_mix") or {}
            total = max(1, sum(got.values()))
            hit = sum(v for k, v in got.items() if k in set(asked))
            if hit / total < DELIVERED_STANCE_MIN_SHARE:
                gaps.append(
                    f"stance={sorted(asked)} but {100 * hit / total:.0f}% of rows carry one of them"
                )
        if gaps:
            warnings.warn(
                "the generation knobs did not deliver what was set: "
                + "; ".join(gaps)
                + ". data.report()['delivered'] carries the measured values beside "
                "report()['requested']. Report the delivered numbers, not the settings: "
                "a card that quotes the setting describes an intention rather than the data.",
                stacklevel=2,
            )

    def _rollouts_requested(self) -> int | None:
        """How many rows the run was asked for: pinned prompts times k
        under ``tasks=``, else situations x requests x k inside the row
        budget, else the budget itself. None when nothing bounded it.
        Successive allocation may stop a unanimous group short of k on
        purpose; that gap is ``rollouts_saved`` in the group summary, not
        a loss."""
        c = self.c
        k = max(1, int(c.repeat_count or 1))
        if self.pinned_prompts:
            return len(self.pinned_prompts) * k
        if c.n_situations_target:
            return min(int(c.cap), int(c.n_situations_target) * max(1, int(c.n_req or 1)) * k)
        return int(c.budget) if c.budget is not None else None

    def _lost_note(self, lost: int) -> str:
        """The run-level warning for lost rollouts: the count against what
        was asked for, the reasons, and the fix for each reason."""
        requested = self._rollouts_requested()
        asked = f" of {requested} asked for" if requested else ""
        by = ", ".join(f"{n} {reason.replace('_', ' ')}" for reason, n in self.lost_by.items() if n)
        fixes: list[str] = []
        if self.lost_by.get("agent_error"):
            fixes.append(
                f"agent errors (first: {self.first_agent_error or 'n/a'}): warm a "
                "scale-to-zero endpoint before the run, or raise timeout= if it answers slowly"
            )
        if self.lost_by.get("empty_reply"):
            fixes.append(
                "empty replies: the agent returned no final text; check the wrapper returns "
                "final_text, or raise agent_max_tokens= so the last turn is not cut"
            )
        if self.lost_by.get("tool_markup"):
            fixes.append(
                "tool markup: raw <tool_call> tags or a tool schema reached the visible text; "
                "fix the agent's tool-call format before grading it"
            )
        return (
            f"{lost} rollout(s){asked} never became rows ({by}), so every rate in this run "
            f"is over the {len(self.data.trajectories)} that did, and the missing ones are not "
            f"missing at random. Fix: {'; '.join(fixes)}. data.report()['rollouts_lost_by'] "
            "has the breakdown."
        )

    # -------------------------------------------------------- loop state

    def _init_loop_state(self) -> None:
        c = self.c
        self.signatures: set[str] = set()
        self.cells: set[str] = set()
        self.cell_counts: dict[str, int] = {}
        self.shape_counts: dict[str, int] = {}
        self.flat = 0
        self.round_index = 0
        self.empty_streak = 0
        # Rollouts the agent callable failed to produce. Counted here so a
        # run that drops every rollout says so instead of blaming the writer.
        self.agent_errors = 0
        self.first_agent_error = ""
        self.timeout_noted = False
        self.agent_error_allowance = max(
            c.knobs.dead_agent_errors, c.knobs.dead_agent_budget_multiple * int(c.cap or 0)
        )
        self.agent_dead = False
        self.auth_error: str | None = None
        self.writer_idle = 0
        self.restart_count = 0
        # Starvation relief: when every situation slot is used but rows are
        # still owed because rollouts were discarded, lifts the situations
        # cap once so fresh situations fill the lost slots. Stays unlifted
        # when nothing was lost.
        self.cap_lifted = {"lifted": False, "lost": 0}
        # Why each lost rollout was lost (LOST_REASONS), and how many
        # usable rollouts finished after the row budget was already met.
        # Over-cap rollouts are not lost: the run asked for cap rows and
        # got them; these were in flight when the last one landed.
        self.lost_by: dict[str, int] = {reason: 0 for reason in LOST_REASONS}
        # Re-rolls by the same reasons, rows this call landed, rows loaded
        # from checkpoint=, and agent errors that were call timeouts: the
        # numbers a four-hour run owes the person watching it (#470).
        self.rerolled_by: dict[str, int] = {reason: 0 for reason in LOST_REASONS}
        self.landed = 0
        self.resumed = 0
        self.timed_out = 0
        self.over_cap = 0
        # Restarts scale with the job: a 10k-row budget cannot live on the
        # same retry allowance as a smoke run.
        self.max_restarts = max(
            MAX_NOVELTY_RESTARTS, int(c.cap or 0) // c.knobs.rows_per_extra_restart
        )
        # A dormant switch: nothing sets it, so the region and fingerprint
        # dedup branches below never fire. Kept so the paths stay readable
        # next to the code that would flip it.
        self.explore_only = False
        self.failing_regions: list[dict] = []
        self.failing_rows: list[dict] = []
        # Why each failing row is a mutation parent, by its prompt head and
        # situation id, and how many parents and mutated rows each aim
        # produced (#285): a fault the world raised, or a grade the judge
        # gave. Reported as search["mutation_aims"].
        self.failure_aim: dict[str, str] = {}
        # Which rubric criterion drove each mutation, by name. Reported as
        # search["failure_criteria"], so a run can say what it aimed at
        # rather than only how many rows it re-rolled.
        self.failure_criteria: dict[str, int] = {}
        self.mutation_aims: dict[str, dict[str, int]] = {
            "world_fault": {"parents": 0, "rows": 0},
            "graded_failure": {"parents": 0, "rows": 0},
            "unattributed": {"parents": 0, "rows": 0},
        }
        self.used = set()
        self.rerolls: dict[str, int] = {}
        self.discarded: set[str] = set()
        self.used_situations: set[str] = set()
        self.used_scenario_ids: set[str] = set()
        self.region_counts: dict[str, int] = {}
        self.region_sigs: dict[str, set] = {}
        self.region_fails: dict[str, int] = {}
        self.region_novelty: dict[str, float] = {}
        self.behavior_gap_prompts: list[str] = []
        self.generated_pool = []
        # tasks= : the prompts to replay, in order, and the plan (faults,
        # world state, stance) each ran under. The pool is exactly these.
        self.pinned_prompts: list[str] = []
        self.pinned_plans: dict[str, dict] = {}
        self.scenario_families: list[tuple[str, frozenset[str]]] = []
        self.situation_prompts: dict[str, list[str]] = {}
        self.prompt_rollouts = {}
        self.verify_queue: list[tuple] = []
        self.allocator_counts: dict[str, int] = {"explore": 0, "expand": 0, "verify": 0}
        # Successive allocation (rl): one label per finished rollout of a
        # prompt (reward when the row carries one, behavior signature
        # otherwise), the prompt's state, and the run's own mixed rate.
        self.group_labels: dict[str, list] = {}
        self.group_state: dict[str, str] = {}
        self.group_job: dict[str, tuple] = {}
        self.groups_probed = 0
        self.groups_mixed = 0
        # Empirical hazard: of the groups that were unanimous after n
        # rollouts and got another, how many split on it. Laplace's
        # 1/(n+2) is only the prior before the run has seen any.
        self.hazard_seen: dict[int, int] = {}
        self.hazard_split: dict[int, int] = {}
        self.rollout_durations: list[float] = []
        self.writer_fallback_error = ""
        # Judge in the loop: verdicts run beside the rollouts, never in
        # front of them. A row's allocation decision waits for its verdict.
        self.judge_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(c.concurrency))
        )
        self.judge_inflight: dict = {}
        self.judged_in_loop = 0
        # seconds the rollout pool sat empty with nothing to do but wait
        # for verdicts: the run had too few situations for its concurrency
        self.idle_on_judge_s = 0.0
        # rollouts that hit the length cap: never judged, counted as done
        # without evidence in their group (Yu et al. 2025 (DAPO), arXiv:2503.14476)
        self.group_truncated: dict[str, int] = {}
        self.skipped_truncated = 0
        self._successive = c.topo["repeat_policy"] == "successive" and not c.k_immediate
        # Set when the clock can no longer fit a fresh group: only verify
        # jobs are scheduled so in-flight groups finish before the whistle.
        self.closing = False
        # streaming output
        self.written = 0
        self.stream_started = False
        self.reported_rows = 0
        self.reported_at = self.started
        # plain-words progress on the logger, for a run long enough that
        # silence reads as a hang. Small budgets stay quiet.
        self.progress_on = int(c.cap or 0) >= PROGRESS_MIN_BUDGET
        self.progress_every_s = c.knobs.progress_every_s
        self.progress_every_rows = c.knobs.progress_every_rows
        self.progress_rows = 0
        self.progress_at = self.started
        # named so a test can hand the throttle a clock of its own
        self.progress_clock = time.monotonic
        # the caller's own listener, called on every line whatever the budget
        self.on_progress = c.on_progress
        # writer waves
        self.walked_ids: set[str] = set()
        self.walked_lock = threading.Lock()
        self.seen_prints: set[str] = set()
        self.generation_started = self.started
        self.scenario_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, c.scenario_concurrency)
        )
        self.scenario_futs = []
        self.next_producer_round = 0
        self.last_batch_size = 1
        # One slot per requested rollout. Refill writers before the unused
        # pool hits zero: keep about two waves of prompts in the pipe.
        self.flight = max(1, int(c.concurrency))
        # Round-synchronous scheduling: every rollout, writer wave and
        # verdict of a round lands before the next round is chosen, so
        # the loop runs the same number of rounds and folds the same
        # batches whatever the thread timing. ``reproducible=True`` asks
        # for it; a single slot gets it for free. Without it the 0.35 s
        # collect window decided how many of one batch's rollouts a round
        # saw, which drifted the round counter that seeds selection and
        # made two same-seed runs in one process draw different situations.
        self.sync = bool(c.reproducible) or self.flight == 1
        typical_n = min(c.completions_per_request, c.knobs.writer_typical_completions)
        self.writer_batch = max(1, c.scenarios_per_request * typical_n)
        waves = c.knobs.writer_buffer_waves
        self.writer_buffer = max(
            self.writer_batch * waves, min(self.flight * waves, c.knobs.writer_buffer_cap)
        )
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.flight)
        self.inflight = {}
        self.inflight_started: dict = {}

    def _seed_pool(self) -> None:
        c = self.c
        for task in c.pinned_tasks:
            prompt = task["prompt"]
            self.pinned_prompts.append(prompt)
            self.generated_pool.append(prompt)
            meta: dict[str, Any] = {"arm": task["arm"], "generator": "pinned", "seed": c.seed}
            if task.get("writer_model"):
                meta["writer_model"] = task["writer_model"]
            if task.get("scenario_id"):
                meta["region_id"] = task["scenario_id"]
            if task.get("assignment"):
                meta["assignment"] = task["assignment"]
            self.generator.meta[prompt] = meta
            self.generator.provenance[prompt] = task["arm"]
            plan = dict(task.get("plan") or {})
            self.pinned_plans[prompt] = plan
            if plan:
                # The runner reads this dict by prompt: the mock world
                # answers under the same faults and world state as before.
                self.fault_plans[prompt] = plan
        self.seed_prompt_set = {str(t).strip() for t in self.given_seeds if str(t).strip()}
        for text in self.seed_prompts:
            text = str(text).strip()
            if not text:
                continue
            self.generated_pool.append(text)
            self.generator.meta[text] = {
                "arm": "open_ended",
                "generator": "user",
                "seed": self.c.seed,
            }
            self.generator.provenance[text] = "open_ended"

    @staticmethod
    def _mean_novelty(rows: list[dict]) -> float:
        vals = [float(r["novelty"]) for r in rows if r.get("novelty") is not None]
        return math.fsum(vals) / len(vals) if vals else 1.0

    def _novelty_parents(self) -> list[dict]:
        """The failing rows the writer mutates from: the newest
        ``writer_context_parents`` of them, or none with mutation off."""
        if not self.c.mutate_failures:
            return []
        n = self.c.knobs.writer_context_parents
        return self.failing_rows[-n:] if n else []

    def _behavior_gaps(self) -> list[str]:
        """The newest ``writer_context_items`` behavior-gap prompts."""
        n = self.c.knobs.writer_context_items
        return list(self.behavior_gap_prompts[-n:]) if n else []

    def _novelty_restart(self, round_id: int, info: dict, *, clear_avoid: bool) -> int:
        gen = self.generator
        if self.restart_count >= self.max_restarts or gen.model is None:
            return round_id
        self.restart_count += 1
        bump = round_id + self.restart_count * self.c.knobs.restart_seed_stride
        ctx_kwargs = {
            "novelty_parents": self._novelty_parents(),
            "avoid": [] if clear_avoid else (info.get("concentrated") or []),
            "underexplored": info.get("sparse") or [],
            "behavior_gaps": self._behavior_gaps(),
        }
        if hasattr(gen, "set_search_context"):
            missing = uncovered_action_shapes(
                self.action_shapes,
                self.induced_shape_keys,
                limit=int(self.search_plan["shape_limit"]),
            )
            ctx_kwargs["action_targets"] = [shape_as_tags(s, self.tools) for s in missing]
            ctx_kwargs["arm_weights"] = self.search
            gen.set_search_context(**ctx_kwargs)
        return bump

    def _situation_key(self, prompt: str, meta: dict) -> str:
        hit = self._situation_key_cache.get(prompt)
        if hit is not None and hit[0] == id(meta):
            return hit[1]
        sk = _situation_key_from_meta(meta, prompt)
        self._situation_key_cache[prompt] = (id(meta), sk)
        return sk

    def _prompt_available(
        self, prompt: str, *, room: bool | None = None, waiting: int | None = None
    ) -> bool:
        """Whether a pool prompt may be scheduled now.

        ``room`` and ``waiting`` are ``_room_for_a_new_ask()`` and
        ``_seeds_waiting()``; a scan over the pool computes them once
        and passes them in, since neither depends on the prompt.
        """
        c = self.c
        if prompt in self.used or prompt in self.discarded:
            return False
        if prompt in self.pinned_plans:
            # A pinned prompt is owed its rollouts whatever the situation
            # caps say; the caller fixed the task set.
            return True
        if prompt in self.seed_prompt_set:
            # So is a seed: it is the ask the caller wrote, not a
            # situation the search picked, so the situations= quota
            # never evicts it. Only the budget can, and the run says so
            # up front when it does.
            return True
        meta = self.generator.meta.get(prompt) or {}
        sk = self._situation_key(prompt, meta)
        if self.explore_only:
            rid = str(meta.get("region_id") or "")
            if rid and rid in self.used_scenario_ids:
                return False
            if sk and sk in self.used_situations:
                return False
        elif sk:
            if room is None:
                room = self._room_for_a_new_ask()
            if not room:
                # Every row left is owed to a seed that has not gone out.
                return False
            if (
                c.n_situations_target
                and not self.cap_lifted["lifted"]
                and sk not in self.used_situations
                and len(self.used_situations)
                + (self._seeds_waiting() if waiting is None else waiting)
                >= c.n_situations_target
            ):
                return False
            if len(self.situation_prompts.get(sk, [])) >= c.n_req and prompt not in (
                self.situation_prompts.get(sk) or []
            ):
                return False
        return True

    def _available(self) -> list[str]:
        room, waiting = self._room_for_a_new_ask(), self._seeds_waiting()
        unused = [
            p for p in self.generated_pool if self._prompt_available(p, room=room, waiting=waiting)
        ]
        if self.seed_prompts:
            seed_set = set(self.seed_prompts)
            unused.sort(key=lambda p: 0 if p in seed_set else 1)
        return unused

    def _schedule_prompt(self, jobs: list, prompt: str, meta: dict, row: dict, action: str) -> None:
        c = self.c
        meta = dict(meta or {})
        meta["allocator"] = action
        if action == "verify":
            idx = self.prompt_rollouts.get(prompt, 0)
            if idx >= c.repeat_count:
                return
            jobs.append((prompt, idx, meta, row))
            self.prompt_rollouts[prompt] = idx + 1
            self.allocator_counts["verify"] = self.allocator_counts.get("verify", 0) + 1
            return
        if prompt in self.used:
            return
        # A prompt resumed from checkpoint= already has ``start`` rows; it
        # is owed the rest, not another k.
        start = self.prompt_rollouts.get(prompt, 0)
        owed = max(0, c.repeat_count - start)
        if c.k_immediate:
            k_now = owed
        elif c.topo["repeat_policy"] == "successive":
            k_now = min(owed, c.probe)
        else:
            k_now = min(1, owed)
        if k_now <= 0:
            self.used.add(prompt)
            return
        for i in range(k_now):
            jobs.append((prompt, start + i, meta, row))
        self.prompt_rollouts[prompt] = start + k_now
        self.used.add(prompt)
        sk = _situation_key_from_meta(meta, prompt)
        if sk:
            self.situation_prompts.setdefault(sk, []).append(prompt)
            self.used_situations.add(sk)
        rid = str((meta or {}).get("region_id") or "")
        if rid:
            self.used_scenario_ids.add(rid)
        self.allocator_counts[action] = self.allocator_counts.get(action, 0) + k_now

    def _append_jobs(self, jobs: list, prompt: str, meta: dict, row: dict) -> None:
        action = (
            "expand"
            if _situation_key_from_meta(meta, prompt) in self.used_situations
            else "explore"
        )
        self._schedule_prompt(jobs, prompt, meta, row, action)

    # ------------------------------------------------------------ output

    def _progress(self, elapsed: float) -> dict[str, Any]:
        """The run's counters at one moment: what ``on_progress=`` receives,
        what the progress line prints, and what ``search["rollouts"]``
        keeps at the end (#470). Rows landed against rollouts re-rolled
        and lost, each by reason, so a run that is slower than its rows
        explain says where the time went."""
        lost_by = {r: n for r, n in self.lost_by.items() if n}
        rerolled_by = {r: n for r, n in self.rerolled_by.items() if n}
        return {
            "rows": len(self.data.trajectories),
            "cap": int(self.c.cap),
            "landed": self.landed,
            "resumed": self.resumed,
            "rerolled": sum(rerolled_by.values()),
            "rerolled_by": rerolled_by,
            "timed_out": self.timed_out,
            "lost": int(self.cap_lifted.get("lost", 0)),
            "lost_by": lost_by,
            "inflight": len(self.inflight),
            "situations": len(self.generated_pool),
            "elapsed_s": round(elapsed, 1),
        }

    def _note_progress(self, *, force: bool = False) -> None:
        """Say where the run is, on the logger, at most ten seconds and at
        most ten events apart, an event being a row landed, a rollout
        re-rolled or one lost: a run that only re-rolls still speaks. A
        hosted run can spend minutes between rows, and a tester with no
        output assumes it hung. ``on_progress=`` gets the same numbers as
        a dict on every line, whatever the budget."""
        if not (self.progress_on or self.on_progress is not None):
            return
        now = self.progress_clock()
        progress = self._progress(now - self.started)
        events = progress["landed"] + progress["rerolled"] + progress["lost"]
        stale = now - self.progress_at >= self.progress_every_s
        many = events - self.progress_rows >= self.progress_every_rows
        if not (force or stale or many):
            return
        if self.progress_on:
            _say(
                progress_line(
                    progress["rows"],
                    progress["cap"],
                    progress["situations"],
                    now - self.started,
                    rerolled=progress["rerolled"],
                    lost=progress["lost"],
                    lost_by=progress["lost_by"],
                    resumed=progress["resumed"],
                )
            )
        if self.on_progress is not None:
            self.on_progress(dict(progress))
        self.progress_rows, self.progress_at = events, now

    def _land(self, t: dict) -> None:
        """Store one usable rollout as a row: on the run, on ``checkpoint=``
        at once, and on the streamed ``output=``. The one place a row
        lands, so a kill after this line loses nothing (#470)."""
        data = self.data
        now = time.monotonic() - self.started
        if not data.first_row_seconds:
            data.first_row_seconds = now
        data.trajectories.append(t)
        data.row_seconds.append(now)
        record_turns(self.turn_stats, t)
        self.landed += 1
        path = self.c.checkpoint_path
        if path is not None:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(export_row(t), default=str) + "\n")
                fh.flush()
        self._flush_output("rollout")

    def _load_checkpoint(self) -> None:
        """Resume from ``checkpoint=``: the rows already on disk come back
        as rows of this run, and with ``tasks=`` each pinned prompt is
        credited its rows, so a prompt with all k is never scheduled and
        one with fewer gets only what it is owed (#470). Rows whose
        prompt is not in ``tasks=`` belong to another task set and are
        left in the file but out of the run; ``warnings`` says how many."""
        c = self.c
        path = c.checkpoint_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return
        rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict) and str(row.get("prompt") or "").strip():
                        rows.append(row)
        if not rows:
            return
        pinned = set(self.pinned_prompts)
        skipped = 0
        by_prompt: dict[str, list[dict]] = {}
        for row in rows:
            prompt = str(row.get("prompt") or "")
            if pinned and prompt not in pinned:
                skipped += 1
                continue
            by_prompt.setdefault(prompt, []).append(row)
        for prompt, group in by_prompt.items():
            for row in group:
                lineage = row.get("lineage")
                if not isinstance(lineage, dict):
                    lineage = {}
                lineage["resumed"] = True
                row["lineage"] = lineage
                self.data.trajectories.append(row)
                self.data.row_seconds.append(0.0)
                record_turns(self.turn_stats, row)
                self.resumed += 1
            if prompt not in pinned:
                continue
            # credit the pinned prompt its rows, the way _schedule_prompt
            # would have counted them
            self.prompt_rollouts[prompt] = len(group)
            self.group_labels[prompt] = [self._row_label(r) for r in group]
            if len(group) >= c.repeat_count:
                self.used.add(prompt)
                self.group_state[prompt] = "complete"
                meta = self.generator.meta.get(prompt) or {}
                sk = _situation_key_from_meta(meta, prompt)
                if sk:
                    self.situation_prompts.setdefault(sk, []).append(prompt)
                    self.used_situations.add(sk)
                rid = str(meta.get("region_id") or "")
                if rid:
                    self.used_scenario_ids.add(rid)
        done = sum(1 for p in by_prompt if p in pinned and p in self.used)
        note = (
            f"resumed {self.resumed} rows from {path}"
            + (f": {done} of {len(pinned)} tasks already finished" if pinned else "")
            + (f"; {skipped} rows on disk are not in tasks= and were left out" if skipped else "")
        )
        _say(note)
        if skipped:
            self.data.warnings.append(note)

    def _note_writer_start(self) -> None:
        """One line when the situation writer starts, because the first
        rows cannot land until it has written something."""
        if not self.progress_on:
            return
        if isinstance(self.simulator, str) and self.simulator not in ("hosted", "default"):
            _say(f"writing situations with {self.simulator}; first rows in about a minute")
            return
        _say("writing situations with the hosted writer; first rows in about a minute")

    def _write_progress(self, payload: dict) -> None:
        Path(str(self.c.out_path) + ".progress.json").write_text(json.dumps(payload, default=str))

    def _flush_output(self, stage: str) -> None:
        c = self.c
        if c.out_path is None:
            return
        data = self.data
        rows = data.trajectories
        now = time.monotonic()
        room, waiting = self._room_for_a_new_ask(), self._seeds_waiting()
        unused_n = sum(
            1 for p in self.generated_pool if self._prompt_available(p, room=room, waiting=waiting)
        )
        inflight_n = len(self.inflight)
        writers_n = len(self.scenario_futs)
        self._write_progress(
            {
                "stage": stage,
                "rows": len(rows),
                "scenario_s": round(data.scenario_generation_seconds, 3),
                "rollout_s": round(data.rollout_seconds, 3),
                "scene_s": round(data.scene_brief_seconds, 3),
                "first_row_s": round(data.first_row_seconds, 3),
                "total_s": round(now - self.started, 3),
                "unused": unused_n,
                "inflight": inflight_n,
                "writers": writers_n,
                "search": data.search or None,
            }
        )
        should_report = (
            stage != "rollout"
            or len(rows) >= c.cap
            or len(rows) - self.reported_rows >= c.knobs.flush_report_rows
            or now - self.reported_at >= c.knobs.flush_report_s
        )
        if should_report:
            elapsed = now - self.started
            rate = len(rows) / elapsed if elapsed else 0.0
            log.info(
                "simulate %s rows=%d/%d elapsed=%.1fs rate=%.1f/s unused=%d inflight=%d writers=%d",
                stage,
                len(rows),
                c.cap,
                elapsed,
                rate,
                unused_n,
                inflight_n,
                writers_n,
            )
            self.reported_rows, self.reported_at = len(rows), now
        if not rows:
            return
        c.out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.stream_started:
            with open(c.out_path, "w") as fh:
                for row in rows:
                    fh.write(json.dumps(export_row(row), default=str) + "\n")
            self.stream_started = True
            self.written = len(rows)
            return
        with open(c.out_path, "a") as fh:
            for row in rows[self.written :]:
                fh.write(json.dumps(export_row(row), default=str) + "\n")
        self.written = len(rows)

    # ----------------------------------------------------------- writers

    @staticmethod
    def _fingerprint(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", str(text).lower())
        norm = []
        for word in words:
            if word in _FINGERPRINT_STOPWORDS:
                continue
            if len(word) > FINGERPRINT_STEM_MIN_LEN and word.endswith("s"):
                word = word[:-1]
            norm.append(word)
        return hashlib.sha256(" ".join(norm).encode()).hexdigest()[:SHORT_HASH_CHARS]

    def _produce(
        self,
        round_id: int,
        cards: int | None = None,
        completions: int | None = None,
        out_tokens: int | None = None,
    ) -> tuple[list[str], dict, dict, dict]:
        """One writer wave on a worker thread: a fresh local generator
        that shares the run's regions, walked ids and search context."""
        c = self.c
        gen = self.generator
        if self.stopping:
            return [], {}, {}, {}
        knobs = c.knobs
        n_cards = max(knobs.min_cards_per_wave, int(cards or c.scenarios_per_request))
        n_comp = max(
            1,
            min(
                MAX_COMPLETIONS_PER_REQUEST,
                int(completions if completions is not None else c.completions_per_request),
            ),
        )
        # About tokens_per_card per card: long-prompt cards must be
        # realizable (the old 768 ceiling gave 12-card batches 64 tokens
        # per message).
        token_cap = (
            out_tokens
            if out_tokens is not None
            else max(
                knobs.wave_tokens_floor,
                min(
                    knobs.wave_tokens_cap, knobs.tokens_per_card * n_cards + knobs.wave_tokens_base
                ),
            )
        )
        local = make_default_generator(
            self.tools,
            policy=self.policy,
            per_round=c.pool_size,
            seed=c.seed,
            dimensions=self.dimensions,
            simulator=self.simulator,
            hard_share=c.hard_share,
            world=c.world_options,
            kind=self.writer_kind,
            scenarios_per_request=n_cards,
            completions_per_request=n_comp,
            distinct_cards=c.distinct_cards,
            extra_cards=c.extra_cards,
            scene_brief=self.scene_box["brief"],
            out_tokens=token_cap,
            time_budget=c.time_budget,
            run_started=self.started,
            steering_weight=self.applied_steering,
            **c.advanced,
        )
        src_model = getattr(gen, "model", None)
        loc_model = getattr(local, "model", None)
        if src_model is not None and loc_model is not None:
            if getattr(src_model, "regions", None):
                loc_model.regions = src_model.regions
                loc_model.region_index = {r["id"]: r for r in loc_model.regions}
            loc_model.walked_ids = self.walked_ids
            loc_model.walked_lock = self.walked_lock
            if hasattr(loc_model, "arm_weights"):
                loc_model.arm_weights = dict(getattr(gen, "arm_weights", None) or self.search)
        if hasattr(local, "arm_weights"):
            local.arm_weights = dict(getattr(gen, "arm_weights", None) or self.search)
        if hasattr(local, "set_search_context"):
            local.set_search_context(
                novelty_parents=getattr(gen, "novelty_parents", []),
                avoid=getattr(gen, "avoid", []),
                underexplored=getattr(gen, "underexplored", []),
                behavior_gaps=getattr(gen, "behavior_gaps", []),
                action_targets=getattr(gen, "action_targets", []),
                arm_weights=getattr(gen, "arm_weights", None) or self.search,
            )
        texts = list(local(None, round_id, include_model=True) or [])
        return (texts, dict(local.meta), dict(local.fault_plans), dict(local.last_errors))

    def _ingest_producer(self, fut: concurrent.futures.Future) -> int:
        gen = self.generator
        try:
            more, metas, plans, errors = fut.result()
        except Exception as exc:
            msg = str(exc)
            if "Hosted Qwen" in msg:
                gen.last_errors["llm_guided"] = msg
            else:
                gen.last_errors["llm_guided"] = f"{type(exc).__name__}: {exc}"
            auth = _auth_error(msg)
            if auth:
                self.data.stopped_because = _stop_reason("writer", auth)
                raise RuntimeError(auth) from None
            return 0
        gen.meta.update(metas)
        gen.fault_plans.update(plans)
        gen.last_errors.update(errors)
        for err in (errors or {}).values():
            auth = _auth_error(err)
            if auth:
                self.data.stopped_because = _stop_reason("writer", auth)
                raise RuntimeError(auth) from None
        # sticky: the writer wrote at least once, so a later wave that
        # errors is a writer error, not a template fallback
        if any((m or {}).get("generator") == "model" for m in metas.values()):
            gen.model_produced = True
        added = 0
        for prompt in more:
            if not prompt or prompt in self.generated_pool:
                continue
            fp = self._fingerprint(prompt)
            if self.explore_only and fp in self.seen_prints:
                continue
            meta = metas.get(prompt) or {}
            rid = str(meta.get("region_id") or "")
            if self.explore_only and rid and rid in self.used_scenario_ids:
                continue
            if self.explore_only:
                self.seen_prints.add(fp)
            self.generated_pool.append(prompt)
            added += 1
        return added

    def _ingest_finished_writers(self) -> None:
        """Fold every finished writer wave into the pool."""
        done = [f for f in self.scenario_futs if f.done()]
        self.scenario_futs[:] = [f for f in self.scenario_futs if not f.done()]
        for fut in done:
            self._ingest_producer(fut)

    def _launch_writers(self, n: int) -> None:
        """Queue another writer wave. New round_id draws new temp and tags."""
        n = max(0, int(n))
        if n <= 0:
            return
        self.scenario_futs.extend(
            self.scenario_pool.submit(self._produce, self.next_producer_round + i) for i in range(n)
        )
        self.next_producer_round += n

    def _start_writers(self) -> None:
        gen = self.generator
        data = self.data
        self.generation_started = time.monotonic()
        if self.pinned_prompts:
            pass  # tasks=: the pool is the task set; nothing is written
        elif gen.model is None:
            texts = list(gen(None, 0, include_model=False) or [])
            self.generated_pool.extend(texts)
        else:
            self._note_writer_start()
            # Tiny batches first so the first rollouts start early.
            knobs = self.c.knobs
            initial_writers = min(knobs.first_wave_writers, max(1, self.c.writer_flight))
            for i in range(initial_writers):
                self.scenario_futs.append(
                    self.scenario_pool.submit(
                        self._produce, i, knobs.first_wave_cards, None, knobs.first_wave_tokens
                    )
                )
            self.next_producer_round = initial_writers
            self._ingest_finished_writers()
        data.scenario_generation_seconds = time.monotonic() - self.generation_started
        self._flush_output("seed")
        gen.model_produced = any(
            (gen.meta.get(prompt) or {}).get("generator") == "model"
            for prompt in self.generated_pool
        )
        if self.generated_pool and any(
            (gen.meta.get(p) or {}).get("arm") for p in self.generated_pool
        ):
            note_stage(data, "generated candidate with arm provenance")

    def _select(
        self, candidates: list[str], *, batch_size: int, selection_seed: int, selection_round: int
    ):
        tick = time.monotonic()
        try:
            return select_execution_batch(
                candidates,
                embedder=self.resolved_embedder,
                archive=self.archive,
                batch_size=batch_size,
                seed=selection_seed,
                round_index=selection_round,
            )
        finally:
            self.data.embedding_selection_seconds += time.monotonic() - tick

    # -------------------------------------------------------------- loop

    def _clock_left(self) -> float | None:
        if self.c.time_budget is None:
            return None
        return self.c.time_budget - (time.monotonic() - self.started)

    def _loop(self) -> None:
        c = self.c
        data = self.data
        gen = self.generator
        while len(data.trajectories) < c.cap:
            left = self._clock_left()
            if left is not None and left <= 0:
                data.stopped_because = "time_budget"
                break
            gen.novelty_parents = self._novelty_parents()
            self._update_closing(left)
            remaining = c.cap - len(data.trajectories) - len(self.inflight)
            take = min(max(0, self.flight - len(self.inflight)), max(0, remaining))
            unused = self._refill_pool(remaining, take)
            selected, info = self._select_batch(unused, take)
            batch = self._build_batch(selected, take)
            self.round_index += 1
            if not batch:
                verdict = self._on_empty_batch(remaining)
                if verdict == "break":
                    break
                if verdict == "continue":
                    continue
            else:
                self.empty_streak = 0
                self._submit(batch[:remaining])
                if self.sync and self.inflight:
                    # Round-synchronous: the batch finishes (or hits the
                    # hung-slot limit) before anything is collected, so
                    # results are consumed in submission order and each
                    # round's selection seed sees the same state.
                    concurrent.futures.wait(list(self.inflight), timeout=c.hung_slot_s)
            results, jobs_for = self._collect()
            if self.auth_error:
                data.stopped_because = _stop_reason("agent", self.auth_error)
                raise RuntimeError(self.auth_error) from None
            if self.agent_dead:
                data.stopped_because = "agent_failed"
                break
            if self._update_search(results, jobs_for, info, selected):
                break

    def _settle_inflight(self) -> None:
        """The run is over. Cancel rollouts and writer waves that never
        started, wait up to the stop grace for the ones running, keep
        rollouts that finish while there is room under the cap, and
        report whatever is abandoned.

        Without this a clock stop returned with worker threads still
        calling the caller's agent and threw away every row they made,
        and writer waves kept talking to the model after return.
        Abandoned work holds its thread until its own timeout fires.
        """
        c = self.c
        data = self.data
        self.stopping = True
        for fut in list(self.inflight):
            if fut.cancel():
                self.inflight.pop(fut, None)
                self.inflight_started.pop(fut, None)
        # Writer waves too: a wave still talking to the model after
        # return is work the caller did not ask for and cannot see.
        self.scenario_futs[:] = [f for f in self.scenario_futs if not f.cancel()]
        pending = list(self.inflight) + list(self.scenario_futs)
        if pending and c.stop_grace_s > 0:
            concurrent.futures.wait(pending, timeout=c.stop_grace_s)
        still_writing = [f for f in self.scenario_futs if not f.done()]
        if still_writing:
            data.search["abandoned_writer_waves"] = len(still_writing)
            if "writer_waves_abandoned" not in data.degraded:
                data.degraded.append("writer_waves_abandoned")
                n = len(still_writing)
                note = (
                    f"{n} writer wave{'s were' if n != 1 else ' was'} still talking to the "
                    f"writer model when the run stopped ({data.stopped_because}) and "
                    f"{'were' if n != 1 else 'was'} abandoned after the {c.stop_grace_s:g}s "
                    "stop grace; the situations it was writing were never rolled out and "
                    "cost writer tokens. Raise advanced={'stop_grace': <seconds>} to wait "
                    "for them, or lower advanced={'scenario_concurrency': <n>} so fewer "
                    "waves are in flight when the run stops."
                )
                data.warnings.append(note)
                log.warning(note)
        self.scenario_futs[:] = []
        for fut in [f for f in list(self.inflight) if f.done()]:
            job = self.inflight.pop(fut)
            self.inflight_started.pop(fut, None)
            try:
                t = fut.result()
            except Exception as exc:
                t = self._error_row(job, exc)
            if isinstance(t, dict) and t.get("_skipped"):
                continue
            if len(data.trajectories) >= c.cap:
                self.over_cap += 1
                continue
            if not _usable_rollout(t):
                self._note_lost(t)
                self._discard_lost(t)
                continue
            self._land(t)
        if self.judge_inflight and c.stop_grace_s > 0:
            concurrent.futures.wait(list(self.judge_inflight), timeout=c.stop_grace_s)
        self._drain_judgments()
        if self.judge_inflight:
            data.search["abandoned_judgments"] = len(self.judge_inflight)
            self.judge_inflight.clear()
        abandoned = len(self.inflight)
        if abandoned:
            # Threads cannot be killed; the pool is told to start nothing
            # new and these results are dropped when they arrive.
            data.search["abandoned_rollouts"] = abandoned
            if "rollouts_abandoned" not in data.degraded:
                data.degraded.append("rollouts_abandoned")
            self.inflight.clear()
            self.inflight_started.clear()

    def _refill_pool(self, remaining: int, take: int) -> list[str]:
        """Fold in finished writer waves, launch more if the pool runs
        low, top up offline templates. Returns the eligible prompts."""
        c = self.c
        data = self.data
        gen = self.generator
        if self.sync and self.scenario_futs:
            # Every launched writer wave lands before selection, in launch
            # order, so the pool does not depend on which wave returned first.
            concurrent.futures.wait(list(self.scenario_futs))
        self._ingest_finished_writers()
        data.scenario_generation_seconds = time.monotonic() - self.generation_started
        if self.pinned_prompts:
            return self._available()
        unused = self._available()
        if unused or self.inflight:
            self.writer_idle = 0
        elif self.generated_pool:
            self.writer_idle += 1
        knobs = c.knobs
        if gen.model is not None:
            slots = max(0, c.writer_flight - len(self.scenario_futs))
            pipeline = len(unused) + len(self.scenario_futs) * self.writer_batch
            need = min(max(0, remaining), self.writer_buffer)
            refill = 0
            if remaining > 0 and slots:
                # Keeps a prompt buffer; explore/unique changes which
                # situations are eligible, not whether the buffer
                # refills. Writer flight stays small so rollouts
                # share the GPU.
                low = len(unused) < max(
                    knobs.pool_low_floor,
                    min(knobs.pool_low_cap, self.flight // knobs.pool_low_flight_divisor),
                )
                exhausted = (
                    not unused
                    and self.generated_pool
                    and self.writer_idle >= knobs.writer_idle_rounds_to_rest
                    and not c.unique_cards
                    and c.time_budget is None
                )
                if exhausted or self._situations_complete():
                    # nothing a new wave writes can be rolled out
                    refill = 0
                elif low or pipeline < need:
                    refill = min(slots, max(0, c.writer_flight - len(self.scenario_futs)))
            self._launch_writers(refill)
        unused = self._available()
        if gen.model is None and len(unused) < take * knobs.offline_topup_multiple:
            for prompt in gen(None, self.round_index, include_model=False) or []:
                if prompt and prompt not in self.generated_pool:
                    self.generated_pool.append(prompt)
                    gen.meta.update(getattr(gen, "last_candidate_provenance", {}))
        unused = self._available()
        if gen.model is None and not gen.model_produced:
            bounce = 0
            while len(unused) < take:
                bounce += 1
                stride = bounce * knobs.offline_bounce_stride
                for prompt in gen(None, self.round_index + stride, include_model=False) or []:
                    if prompt and prompt not in self.generated_pool:
                        self.generated_pool.append(prompt)
                unused = self._available()
                if bounce >= knobs.offline_bounce_limit:
                    break
        if gen.model is not None and not unused and self.scenario_futs and not self.inflight:
            wait_s = knobs.writer_wait_s
            left = self._clock_left()
            if left is not None:
                wait_s = max(knobs.writer_wait_floor_s, min(wait_s, left))
            done, _ = concurrent.futures.wait(
                self.scenario_futs, timeout=wait_s, return_when=concurrent.futures.FIRST_COMPLETED
            )
            self.scenario_futs[:] = [f for f in self.scenario_futs if f not in done]
            for fut in done:
                self._ingest_producer(fut)
            unused = self._available()
        self.fault_plans.update(gen.fault_plans)
        if (
            gen.last_errors.get("llm_guided")
            and not gen.model_produced
            and "generator_fallback" not in data.degraded
        ):
            data.degraded.append("generator_fallback")
            # keep the hosted writer's last error: the template writer that
            # takes over knows nothing about a custom spec, and a run that
            # ends with no rows must be able to say why
            self.writer_fallback_error = str(gen.last_errors.get("llm_guided") or "")
        if unused and any((gen.meta.get(p) or {}).get("arm") for p in unused):
            note_stage(data, "generated candidate with arm provenance")
        return unused

    def _select_batch(self, unused: list[str], take: int) -> tuple[list[dict], dict]:
        """Pick a diverse batch from the pool; restart the writer if the
        batch is stale; cap near-copy scenario families."""
        c = self.c
        data = self.data
        gen = self.generator
        selected: list[dict] = []
        info: dict = {}
        if self.pinned_prompts:
            # No novelty pick, no family cap: every pinned prompt runs.
            return [
                {"text": p, "cluster": None, "novelty": None, "reason": "pinned"}
                for p in unused[: max(0, take)]
            ], info
        if unused:
            pick_n = take if self.explore_only else max(take * c.knobs.select_oversample, take)
            selected, info = self._select(
                unused,
                batch_size=min(len(unused), pick_n),
                selection_seed=c.seed + self.round_index,
                selection_round=self.round_index,
            )
        if (
            selected
            and self._mean_novelty(selected) < NOVELTY_RESTART_FLOOR
            and self.restart_count < self.max_restarts
        ):
            bump = self._novelty_restart(self.round_index, info, clear_avoid=True)
            unused = self._available()
            if unused:
                selected, info = self._select(
                    unused,
                    batch_size=max(1, take),
                    selection_seed=c.seed + bump,
                    selection_round=bump,
                )
        family_rejected: list[dict] = []
        if not data.semantic and not c.unique_cards:
            family_batch = list(self.scenario_families)
            selected, family_rejected = cap_scenario_families(
                selected,
                family_batch,
                cap=max(c.knobs.family_cap_floor, c.n_req * c.knobs.family_cap_per_phrasing),
            )
            if gen.model is None:
                fill_to = min(take, len(selected) + len(family_rejected))
                backfill_n = max(0, fill_to - len(selected))
                selected.extend(family_rejected[:backfill_n])
                family_rejected = family_rejected[backfill_n:]
        for row in family_rejected:
            self.discarded.add(row["text"])
        if family_rejected:
            info["family_rejected"] = len(family_rejected)
        if info.get("mixed_spaces_refused") and "embedding_space_mismatch" not in data.degraded:
            data.degraded.append("embedding_space_mismatch")
        if selected:
            if data.semantic:
                note_stage(data, "semantic embedding produced")
            note_stage(data, "selection reason / novelty")
            if self.archive.compatible(self.resolved_embedder):
                self.archive.add(row["vector"] for row in selected)
            missing = uncovered_action_shapes(
                self.action_shapes,
                self.induced_shape_keys,
                limit=int(self.search_plan["shape_limit"]),
            )
            if hasattr(gen, "set_search_context"):
                items = c.knobs.writer_context_items
                gen.set_search_context(
                    novelty_parents=self._novelty_parents(),
                    avoid=(
                        [row["text"] for row in family_rejected[: c.knobs.family_avoid_items]]
                        + list(info.get("concentrated") or [])
                    )[:items],
                    underexplored=info.get("sparse") or [],
                    behavior_gaps=self._behavior_gaps(),
                    action_targets=[shape_as_tags(s, self.tools) for s in missing],
                    arm_weights=self.search,
                )
        return selected, info

    def _card_action(self, meta: dict, prompt: str, slots: dict | None) -> str:
        sk = _situation_key_from_meta(meta, prompt)
        rid = str((meta or {}).get("region_id") or "")
        if sk and sk in self.used_situations:
            return "expand"
        if slots is not None and rid and rid in self.used_scenario_ids:
            return "expand"
        return "explore"

    def _add_job(
        self, batch: list, filled: dict, prompt: str, meta: dict, row: dict, action: str
    ) -> int:
        c = self.c
        sk = _situation_key_from_meta(meta, prompt)
        if prompt not in self.seed_prompt_set:
            # Seeds are served first: a writer ask waits for the
            # situation slots and the rows the seeds have not spent yet.
            if action in ("explore", "expand") and not self._room_for_a_new_ask():
                return 0
            if (
                action == "explore"
                and c.n_situations_target
                and not self.cap_lifted["lifted"]
                and sk not in self.used_situations
                and len(self.used_situations) + self._seeds_waiting() >= c.n_situations_target
            ):
                return 0
        before = len(batch)
        self._schedule_prompt(batch, prompt, meta, row, action)
        added = len(batch) - before
        if added:
            filled[action] = filled.get(action, 0) + added
        return added

    def _build_batch(self, selected: list[dict], take: int) -> list:
        """Turn the selected prompts into rollout jobs: seeds first, then
        the verify queue, then stratified picks under the allocator's
        slot plan, then one failure mutation and one behavior gap for
        the offline writer."""
        c = self.c
        gen = self.generator
        batch: list = []
        row_by_text = {row["text"]: row for row in selected}
        filled = {"explore": 0, "expand": 0, "verify": 0}
        slots = None
        if c.topo["mode"] == "adaptive" and not c.unique_cards and take:
            live_plan = adaptive_allocator(
                c.time_budget, c.until_key, elapsed=time.monotonic() - self.started
            )
            slots = allocator_slot_counts(take, live_plan)

        # Named seed openers are extra requests, not writer completions.
        # Schedule them before hash selection can bury them.
        openers = list(dict.fromkeys([*self.pinned_prompts, *self.seed_prompts]))
        if openers and take:
            for prompt in openers:
                if len(batch) >= take:
                    break
                if prompt not in self.generated_pool or not self._prompt_available(prompt):
                    continue
                meta = dict(
                    gen.meta.get(prompt)
                    or {"arm": "open_ended", "generator": "user", "seed": c.seed}
                )
                row = row_by_text.get(prompt) or {
                    "text": prompt,
                    "cluster": None,
                    "novelty": None,
                    "reason": "seed",
                }
                action = self._card_action(meta, prompt, slots)
                self._add_job(batch, filled, prompt, meta, row, action)
        verify_cap = take if slots is None else slots["verify"]
        if not c.k_immediate and c.repeat_count > 1 and self.verify_queue:
            still: list[tuple] = []
            for prompt, meta, row in self.verify_queue:
                if len(batch) >= take or filled["verify"] >= verify_cap:
                    still.append((prompt, meta, row))
                    continue
                if self.prompt_rollouts.get(prompt, 0) >= c.repeat_count:
                    continue
                self._add_job(batch, filled, prompt, dict(meta), row, "verify")
                if self.prompt_rollouts.get(prompt, 0) < c.repeat_count:
                    still.append((prompt, meta, row))
            self.verify_queue[:] = still
        if self.closing:
            # The clock is nearly out: finish the groups in flight, open
            # none. A fresh group started now would be cut mid-group.
            return batch
        stratified = _stratified_prompts(
            [row["text"] for row in selected], take, gen, used_situations=self.used_situations
        )

        jobs = []
        for prompt in stratified:
            row = row_by_text.get(prompt)
            if not row:
                continue
            meta = dict(gen.meta.get(prompt) or gen.last_candidate_provenance.get(prompt) or {})
            meta.setdefault("arm", gen.provenance.get(prompt, "unattributed"))
            meta.setdefault("seed", c.seed)
            jobs.append((prompt, meta, row, self._card_action(meta, prompt, slots)))
        if slots is None:
            for prompt, meta, row, action in jobs:
                if len(batch) >= take:
                    break
                self._add_job(batch, filled, prompt, meta, row, action)
                self.scenario_families.append(scenario_family(prompt))
        else:
            scheduled: set[str] = set()
            expand_cands = [j for j in jobs if j[3] == "expand"]
            explore_cands = [j for j in jobs if j[3] == "explore"]

            def _fill(cands: list, action: str, limit: int) -> None:
                for prompt, meta, row, _act in cands:
                    if len(batch) >= take or filled[action] >= limit:
                        return
                    if prompt in scheduled:
                        continue
                    if self._add_job(batch, filled, prompt, meta, row, action):
                        scheduled.add(prompt)
                        self.scenario_families.append(scenario_family(prompt))

            _fill(expand_cands, "expand", slots["expand"])
            _fill(explore_cands, "explore", slots["explore"])
            for prompt, meta, row, action in explore_cands + expand_cands:
                if len(batch) >= take:
                    break
                if prompt in scheduled:
                    continue
                if self._add_job(batch, filled, prompt, meta, row, action):
                    scheduled.add(prompt)
                    self.scenario_families.append(scenario_family(prompt))

        # Offline templates only. Live writer steers via cards and
        # search context. At most one mutation and one gap per batch.
        mutation_slots = 0
        gap_slots = 0
        if gen.model is None and not self.pinned_prompts:
            if c.mutate_failures:
                mutation_slots = 1 if self.failing_rows else 0
            gap_slots = 1
        parents = [str(t.get("prompt") or "") for t in self.failing_rows if t.get("prompt")]
        if mutation_slots and parents:
            for name, prompt in mutate_pool(parents, rounds=1, limit=mutation_slots):
                if not self._prompt_available(prompt):
                    continue
                meta = {
                    "arm": "failure_mutation",
                    "seed": c.seed,
                    "generator": name,
                    "parent": parents[0][:PARENT_HEAD_CHARS],
                }
                self._append_jobs(
                    batch,
                    prompt,
                    meta,
                    {"cluster": None, "novelty": None, "reason": "failure_mutation"},
                )
            self.failing_regions = []

        missing = uncovered_action_shapes(
            self.action_shapes, self.induced_shape_keys, limit=max(gap_slots, 1)
        )
        if gen.model is None and gap_slots:
            for shape in missing[: max(0, gap_slots)]:
                prompt = render_target_situation(shape, self.tools)
                if not self._prompt_available(prompt):
                    continue
                meta = {
                    "arm": "behavior_targeted",
                    "seed": c.seed,
                    "generator": "actionspace",
                    "action_key": shape.key(),
                }
                self._append_jobs(
                    batch,
                    prompt,
                    meta,
                    {"cluster": None, "novelty": None, "reason": "behavior_targeted"},
                )
                self.behavior_gap_prompts.append(prompt)
        return batch

    def _on_empty_batch(self, remaining: int) -> str:
        """Nothing to schedule this round. Returns ``"proceed"`` when
        rollouts or writers are still in flight (collect them),
        ``"continue"`` after relaunching or lifting a cap, ``"break"``
        when the run is over."""
        c = self.c
        data = self.data
        gen = self.generator
        if not self.inflight and self._situations_complete():
            # Every situation the run was asked for exists and has all
            # its rollouts; a bigger budget cannot be met. Writer waves
            # still in flight do not change that: the hosted writer kept
            # this branch from firing (a wave was always in flight, so the
            # loop waited on it, then launched another) and a runs=2 call
            # at budget=192 wrote 1,900 situations it never rolled out.
            # _settle_inflight cancels or drains the waves.
            data.stopped_because = "situations_exhausted"
            return "break"
        if self.inflight or self.scenario_futs:
            self.empty_streak = 0
            return "proceed"
        if self.judge_inflight:
            # nothing to roll out until a verdict lands; wait on the judge
            waited = time.monotonic()
            self._drain_judgments(wait_s=c.knobs.collect_wait_s)
            self.idle_on_judge_s += time.monotonic() - waited
            self.empty_streak = 0
            return "continue"
        if self.closing:
            data.stopped_because = "time_budget"
            return "break"
        if (
            c.topo["repeat_policy"] == "successive"
            and not c.k_immediate
            and self._resume_stopped(remaining)
        ):
            self.empty_streak = 0
            note_stage(data, "resumed stopped groups: nothing fresh to open")
            return "continue"
        self.empty_streak += 1
        if self.pinned_prompts and not self._available():
            # tasks=: every pinned prompt has its rollouts (or was lost
            # and re-rolled); there is nothing else this run may draw.
            data.stopped_because = "tasks_done"
            return "break"
        if (
            not self.cap_lifted["lifted"]
            and c.n_situations_target
            and remaining > 0
            and self.cap_lifted["lost"] > 0
            and len(self.used_situations) >= c.n_situations_target
        ):
            self.cap_lifted["lifted"] = True
            self.empty_streak = 0
            note_stage(data, "situation cap lifted to fill lost rollouts")
            return "continue"
        # Unique ingest may drop exact/near-dupe cards. That is
        # not a run stop: the writer can invent another situation.
        if (
            gen.model is not None
            and not self.generated_pool
            and self.empty_streak >= c.knobs.empty_rounds_to_stop
        ):
            err = gen.last_errors.get("llm_guided") or "empty response"
            if "Hosted Qwen" in err:
                raise RuntimeError(err[err.find("Hosted Qwen") :]) from None
            raise RuntimeError(f"hosted Qwen produced no situations: {err}")
        if gen.model is not None and remaining > 0:
            if self.writer_idle >= c.knobs.writer_idle_rounds_to_restart and not c.unique_cards:
                # Writer stalled on duplicates. Restart it
                # with a rotated seed AND a rotating window
                # of already-used asks as avoid pressure:
                # reseeding alone reconverges to the same
                # asks (measured: 26 vs the old ceiling 28).
                if self.restart_count < self.max_restarts:
                    seen = sorted(self.used)
                    width = c.knobs.restart_avoid_window
                    lo_i = (self.restart_count * width) % max(1, len(seen))
                    window = seen[lo_i : lo_i + width] or seen[:width]
                    self.round_index = self._novelty_restart(
                        self.round_index, {"concentrated": window}, clear_avoid=False
                    )
                    self.writer_idle = 0
                    self.empty_streak = 0
                    note_stage(data, "writer restart after ask starvation")
                else:
                    data.stopped_because = "ask_exhausted"
                    return "break"
            slots = max(0, c.writer_flight - len(self.scenario_futs))
            self._launch_writers(min(slots, c.writer_flight))
        elif gen.model is None:
            # The offline writer ran dry. Leaving the default
            # stopped_because="budget" here claimed a 300-row
            # budget was met by 106 rows.
            data.stopped_because = "writer_exhausted"
            return "break"
        return "continue"

    def _submit(self, batch: list) -> None:
        data = self.data
        for prompt, _, meta, _ in batch:
            plan = self.generator.fault_plans.get(prompt) or self.fault_plans.get(prompt)
            assignment = meta.get("assignment") or meta.get("scenario_dimensions") or {}
            if plan or (isinstance(assignment, dict) and assignment.get("world_state")):
                note_stage(data, "world/fault instantiated")
                break
        now = time.monotonic()
        for job in batch:
            fut = self.pool.submit(self._build_row, job)
            self.inflight[fut] = job
            self.inflight_started[fut] = now

    def _collect(self) -> tuple[list[dict], list]:
        """Wait briefly, take every finished rollout, re-roll lost ones,
        and store the usable rows. Hung slots stay in flight."""
        c = self.c
        data = self.data
        rollout_started = time.monotonic()
        wait_s = c.knobs.collect_wait_s
        floor = c.knobs.collect_wait_floor_s
        left = self._clock_left()
        if left is not None:
            wait_s = max(floor, min(wait_s, left))
        if self.inflight:
            if self.sync:
                # Re-rolled rollouts land here too: take the whole set,
                # bounded by the hung-slot limit and the clock.
                wait_s = c.hung_slot_s if left is None else max(floor, min(c.hung_slot_s, left))
                concurrent.futures.wait(self.inflight, timeout=wait_s)
            else:
                concurrent.futures.wait(
                    self.inflight, timeout=wait_s, return_when=concurrent.futures.FIRST_COMPLETED
                )
        results, jobs_for = [], []
        now = time.monotonic()
        for fut in list(self.inflight):
            job = self.inflight[fut]
            if fut.done():
                self.inflight.pop(fut, None)
                started_at = self.inflight_started.pop(fut, now)
                self.rollout_durations.append(max(0.0, now - started_at))
                try:
                    results.append(fut.result())
                    jobs_for.append(job)
                except Exception as exc:
                    results.append(self._error_row(job, exc))
                    jobs_for.append(job)
            elif now - self.inflight_started.get(fut, now) >= c.hung_slot_s:
                # Leaves the future in inflight to avoid launching a
                # replacement on top of a still-running request.
                continue
        if self.stopping:
            # Rollouts that started after the stop returned empty; they
            # are not lost work and must not be re-rolled.
            kept = [(t, job) for t, job in zip(results, jobs_for) if not t.get("_skipped")]
            results = [t for t, _ in kept]
            jobs_for = [job for _, job in kept]
        paired = [(t, job) for t, job in zip(results, jobs_for) if _usable_rollout(t)]
        if len(paired) != len(results):
            # a lost rollout is re-rolled for the same prompt so a
            # repeat group keeps all k members; after the retry cap
            # it counts as lost and a fresh situation fills the slot
            now = time.monotonic()
            for t, job in zip(results, jobs_for):
                if _usable_rollout(t):
                    continue
                self._note_lost(t)
                key = str(job[0])
                if not self.stopping and self.rerolls.get(key, 0) < c.repeat_count:
                    self.rerolls[key] = self.rerolls.get(key, 0) + 1
                    reason = _unusable_reason(t) or "empty_reply"
                    self.rerolled_by[reason] = self.rerolled_by.get(reason, 0) + 1
                    fut = self.pool.submit(self._build_row, job)
                    self.inflight[fut] = job
                    self.inflight_started[fut] = now
                    note_stage(data, "rollout re-rolled")
                else:
                    self._discard_lost(t)
        results = [t for t, _ in paired]
        jobs_for = [job for _, job in paired]
        room = c.cap - len(data.trajectories)
        self.over_cap += max(0, len(results) - max(0, room))
        results, jobs_for = results[:room], jobs_for[:room]
        for t in results:
            note_stage(data, "model rollout")
            if t.get("steps"):
                note_stage(data, "full tool trajectory")
            if t.get("behavior_signature"):
                note_stage(data, "behavior signature")
            self._land(t)
            note_stage(data, "row stored")
        data.rollout_seconds += time.monotonic() - rollout_started
        self._note_progress()
        return results, jobs_for

    def _update_search(
        self, results: list[dict], jobs_for: list, info: dict, selected: list[dict]
    ) -> bool:
        """Fold a batch of results into the search state. Returns True
        when the run reached saturation and should stop."""
        c = self.c
        data = self.data
        gen = self.generator
        executed: dict[str, int] = {}
        new_sig: dict[str, int] = {}
        new_cell: dict[str, int] = {}
        new_shape = 0
        fresh = 0
        for t in results:
            arm = t["arm"]
            executed[arm] = executed.get(arm, 0) + 1
            if t["behavior_signature"] not in self.signatures:
                new_sig[arm] = new_sig.get(arm, 0) + 1
                fresh += 1
            key = row_cell_key(t)
            if key:
                self.cells.add(key)
            assignment = t.get("scenario_dimensions")
            if isinstance(assignment, dict) and assignment:
                grid = json.dumps(assignment, sort_keys=True, default=str)
                prev = self.cell_counts.get(grid, 0)
                self.cell_counts[grid] = prev + 1
                if prev == 0:
                    new_cell[arm] = new_cell.get(arm, 0) + 1
        self.signatures.update(t["behavior_signature"] for t in results)
        # Yield is new signatures plus new cells per row, for arms that
        # ran. Idle arms are filled with the mean so they carry no vote.
        yields = complete_yields(
            {
                arm: (new_sig.get(arm, 0) + new_cell.get(arm, 0)) / n_arm
                for arm, n_arm in executed.items()
                if n_arm
            },
            self.search,
        )
        self.search = reallocate_search_arms(self.search, yields)
        data.arm_weights = dict(self.search)
        if hasattr(gen, "reallocate"):
            gen.reallocate(yields)
        gen.arm_weights = dict(self.search)
        model_live = getattr(gen, "model", None)
        if model_live is not None and hasattr(model_live, "arm_weights"):
            model_live.arm_weights = dict(self.search)

        self._record_shapes(results)
        for t in results:
            observed = shape_from_trajectory(t, self.tools)
            if observed is None:
                continue
            sk = observed.key()
            prev = self.shape_counts.get(sk, 0)
            self.shape_counts[sk] = prev + 1
            if prev == 0:
                new_shape += 1
        for t, job in zip(results, jobs_for):
            rid = t.get("scenario_id")
            if not rid:
                continue
            self.region_counts[rid] = self.region_counts.get(rid, 0) + 1
            self.region_sigs.setdefault(rid, set()).add(t["behavior_signature"])
            if mutation_worthy(t):
                self.region_fails[rid] = self.region_fails.get(rid, 0) + 1
            sel = job[_JOB_SELECTION] if len(job) > _JOB_SELECTION else {}
            nov = sel.get("novelty") if isinstance(sel, dict) else None
            if nov is not None:
                prev = self.region_novelty.get(rid, float(nov))
                w = c.knobs.region_novelty_smoothing
                self.region_novelty[rid] = (1.0 - w) * prev + w * float(nov)

        assign_id = {
            json.dumps(r["assignment"], sort_keys=True, default=str): r["id"] for r in gen.regions
        }

        knobs = c.knobs

        def novelty_fn(assignment):
            rid = assign_id.get(json.dumps(assignment, sort_keys=True, default=str), "")
            return float(self.region_novelty.get(rid, knobs.gap_value_unknown))

        def behavior_fn(assignment):
            rid = assign_id.get(json.dumps(assignment, sort_keys=True, default=str), "")
            count = self.region_counts.get(rid, 0)
            nsig = len(self.region_sigs.get(rid, ()))
            fails = self.region_fails.get(rid, 0)
            if count >= knobs.gap_min_rows and nsig <= 1:
                gap = knobs.gap_value_stuck
            elif nsig >= knobs.gap_rich_signatures:
                gap = knobs.gap_value_rich
            else:
                gap = knobs.gap_value_unknown
            fault_rate = fails / (count + 1.0)
            return min(1.0, knobs.gap_weight * gap + (1.0 - knobs.gap_weight) * fault_rate)

        axis_counts: dict[str, dict[str, int]] = {}
        for t in data.trajectories:
            dims = t.get("scenario_dimensions")
            if not isinstance(dims, dict):
                continue
            for axis in ("tool_condition", "history", "world_state"):
                value = str(dims.get(axis) or "")
                if value:
                    slot = axis_counts.setdefault(axis, {})
                    slot[value] = slot.get(value, 0) + 1

        self._apply_allocation(
            retarget_regions(
                gen.regions,
                self.tools,
                counts=self.region_counts,
                novelty=novelty_fn,
                behavior_value=behavior_fn,
                axis_counts=axis_counts,
                mode=c.topo["mode"],
            )
        )
        model_obj = getattr(gen, "model", None)
        if model_obj is not None and getattr(model_obj, "regions", None):
            self._apply_allocation(
                retarget_regions(
                    model_obj.regions,
                    self.tools,
                    counts=self.region_counts,
                    novelty=novelty_fn,
                    behavior_value=behavior_fn,
                    axis_counts=axis_counts,
                    mode=c.topo["mode"],
                )
            )
        templates = getattr(gen, "templates", None)
        if templates is not None:
            templates.regions = gen.regions

        region_index = {r["id"]: r for r in gen.regions}
        self.failing_rows = [t for t in results if mutation_worthy(t)]
        for t in self.failing_rows:
            self._aim_failure(t, "world_fault")
        self.failing_regions = [
            region_index[t["scenario_id"]]
            for t in self.failing_rows
            if t["scenario_id"] in region_index
        ]
        successive = c.topo["repeat_policy"] == "successive" and not c.k_immediate
        # Any mode with a grader judges beside the loop: rows are judged as
        # they land instead of all at once after the clock, which on a
        # 120 s explore run added a minute of judging past the budget.
        judged_async = c.grader is not None
        for t, job in zip(results, jobs_for):
            prompt = str(t.get("prompt") or job[0] or "")
            if not prompt:
                continue
            if _hit_length_cap(t):
                # A completion cut by the cap is not a completion. Scoring
                # it runs the judge out of distribution; it carries no
                # label in its group and the pruner drops it later.
                attach(
                    t,
                    Judgment(
                        rollout_id=str(t.get("rollout_id") or ""),
                        scorer=ScorerRef(name="length_cap", kind="rule"),
                        reward=None,
                        status="missing_reward",
                        reason="truncated: hit the length cap, not judged",
                    ),
                )
                self.skipped_truncated += 1
                if successive:
                    self._successive_update(prompt, t, job, truncated=True)
                continue
            if judged_async:
                self._submit_judgment(t, job)
                continue
            if successive:
                self._successive_update(prompt, t, job)
                continue
            if self.prompt_rollouts.get(prompt, 0) >= c.repeat_count:
                continue
            want_verify = mutation_worthy(t)
            if not want_verify and c.topo["mode"] == "adaptive" and not c.k_immediate:
                nsig = len(self.region_sigs.get(t.get("scenario_id"), ()))
                live = adaptive_allocator(
                    c.time_budget, c.until_key, elapsed=time.monotonic() - self.started
                )
                # Short/messy clocks peek for different outcomes.
                # Long/saturation only re-rolls when behavior already differs.
                if nsig > 1 or live["explore"] < knobs.adaptive_verify_explore_floor:
                    want_verify = True
            if not want_verify:
                continue
            meta = job[_JOB_META] if len(job) > _JOB_META else {}
            sel = job[_JOB_SELECTION] if len(job) > _JOB_SELECTION else {}
            self.verify_queue.append(
                (prompt, dict(meta or {}), sel if isinstance(sel, dict) else {})
            )
        missing = uncovered_action_shapes(
            self.action_shapes, self.induced_shape_keys, limit=int(self.search_plan["shape_limit"])
        )
        deficit = copies_remaining(self.cell_counts)
        if self.induced_shape_keys:
            deficit += copies_remaining(self.shape_counts)
        axis_gaps = self._axis_gaps()
        if hasattr(gen, "set_search_context"):
            gen.set_search_context(
                novelty_parents=self._novelty_parents(),
                avoid=list(getattr(gen, "avoid", []) or []),
                underexplored=(list(info.get("sparse") or []) + axis_gaps)[
                    : knobs.writer_context_items
                ],
                behavior_gaps=self._behavior_gaps(),
                action_targets=[shape_as_tags(s, self.tools) for s in missing],
                arm_weights=self.search,
            )
        data.search = {
            "cell_counts": dict(self.cell_counts),
            "shape_counts": dict(self.shape_counts),
            "region_counts": dict(self.region_counts),
            "arm_weights": dict(self.search),
            "avoid": list(getattr(gen, "avoid", []) or []),
            "underexplored": list(getattr(gen, "underexplored", []) or []),
            "axis_gaps": axis_gaps,
            "min_cell_copies": min(self.cell_counts.values()) if self.cell_counts else 0,
            "copy_deficit": deficit,
            "uncovered_shapes": len(missing),
            "plateau_batches": self.flat,
            "copies_needed": SATURATION_COPIES,
            "allocator": dict(self.allocator_counts),
            "mode": c.topo["mode"],
            "repeat_policy": c.topo["repeat_policy"],
            "n_req": c.n_req,
            "k": c.repeat_count,
        }
        if judged_async:
            if self.sync:
                self._settle_judgments()
            self._drain_judgments()
        if successive:
            data.search["groups"] = self._group_summary()
        space_rate = (fresh + sum(new_cell.values()) + new_shape) / max(1, len(results))
        self.last_batch_size = len(results)
        record_coverage(
            data,
            data.trajectories,
            cells=self.cells,
            shape_keys=self.induced_shape_keys,
            arm_weights=self.search,
            batch_fresh_rate=space_rate,
            mean_batch_novelty=self._mean_novelty(selected) if selected else None,
        )
        self.flat = self.flat + 1 if space_rate < NEW_SIGNATURE_FLOOR else 0
        data.search["plateau_batches"] = self.flat
        data.search["mutation_aims"] = {k: dict(v) for k, v in self.mutation_aims.items()}
        data.search["failure_criteria"] = dict(
            sorted(self.failure_criteria.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        if c.until_sat and space_saturated(
            self.cell_counts, self.shape_counts, expected_cells=self.planned_cell_keys
        ):
            data.stopped_because = "saturation"
            return True
        return False

    # ------------------------------------------- successive allocation

    def _successive_update(
        self, prompt: str, t: dict, job: tuple, *, truncated: bool = False
    ) -> None:
        """Fold one finished rollout into its group and decide whether the
        prompt gets another.

        The label is the row's binary ``reward`` when a grader ran, else
        its behavior signature. A group with two labels has split: it is
        filled to k, that is where a grouped update's gradient lives. A
        group that is still unanimous after n rollouts gets one more only
        while the chance the next rollout differs beats what a fresh
        prompt offers per rollout. Both sides are measured on this run:
        the hazard is how often a group unanimous after n split on its
        next rollout (Laplace's 1/(n+2) only as the prior), and the fresh
        side is the run's own mixed rate over its probe size. Nothing here
        is a tuned constant: the probe (2, the least that can show a
        split), k, and the run's measurements decide. Dynamic sampling (Yu
        et al. 2025 (DAPO), arXiv:2503.14476) and difficulty filtering
        (Lambert 2025, chapter Reasoning), applied at generation time.
        """
        c = self.c
        k = c.repeat_count
        reward = t.get("reward")
        label = (
            int(reward)
            if reward in (0, 1) and not isinstance(reward, bool)
            else str(t.get("behavior_signature") or "")
        )
        labels = self.group_labels.setdefault(prompt, [])
        self.group_job[prompt] = job
        if truncated:
            self.group_truncated[prompt] = self.group_truncated.get(prompt, 0) + 1
        was_unanimous = len(labels) >= 1 and len(set(labels)) == 1
        n_prev = len(labels)
        if not truncated:
            labels.append(label)
        n_done = len(labels) + self.group_truncated.get(prompt, 0)
        if not truncated and was_unanimous and n_prev >= max(1, min(k, c.probe)):
            # this rollout was a continuation of a unanimous group: it is
            # one observation of the hazard at n_prev
            self.hazard_seen[n_prev] = self.hazard_seen.get(n_prev, 0) + 1
            if len(set(labels)) > 1:
                self.hazard_split[n_prev] = self.hazard_split.get(n_prev, 0) + 1
        n_sched = self.prompt_rollouts.get(prompt, 0)
        probe = max(1, min(k, c.probe))
        state = self.group_state.get(prompt, "probing")
        if n_done == probe:
            self.groups_probed += 1
        if len(set(labels)) > 1:
            if state != "mixed":
                self.groups_mixed += 1
                self.group_state[prompt] = "mixed"
            self._queue_verify(prompt, job, k - n_sched)
            if n_done >= k:
                self.group_state[prompt] = "mixed"
            return
        if n_sched >= k:
            self.group_state[prompt] = "complete" if n_done >= k else state
            return
        if n_done < n_sched:
            # more of this prompt's probe is still in flight; decide when
            # it lands
            self.group_state[prompt] = state
            return
        p_next = self._hazard(n_done)
        mixed_rate = self._mixed_rate()
        if not self._fresh_available() or p_next > mixed_rate / probe:
            self.group_state[prompt] = "probing"
            self._queue_verify(prompt, job, 1)
        else:
            self.group_state[prompt] = "stopped_unanimous"

    _JUDGE_KEYS = (
        "reward",
        "reason",
        "judge_status",
        "judge_name",
        "failure_class",
        "markers",
        "lineage",
    )

    def _submit_judgment(self, row: dict, job: tuple) -> None:
        """Hand a landed rollout to the caller's grader on the judge pool.
        The successive allocator then reads its reward, the signal a
        grouped update trains on, instead of a behavior signature. The
        rollout scheduler never waits on the judge: a slow judge delays
        one prompt's decision, not the run. Rows judged here are not
        judged again at the end."""
        from ..score.judging import run_judge

        grader = self.c.grader

        def one() -> dict:
            scored = run_judge([row], grader, source="grade")
            return scored.rows[0] if scored.rows else {}

        fut = self.judge_pool.submit(one)
        self.judge_inflight[fut] = (row, job)

    def _settle_judgments(self) -> None:
        """Round-synchronous runs: every verdict of the round lands before
        the allocator reads any, so a group's next rollout is decided on
        the same evidence in every run. Otherwise a verdict that landed a
        few milliseconds later was folded a round later, and the verify
        queue, the fresh-prompt check and the round counter all moved.
        Bounded by the hung-slot limit and the clock, like a rollout."""
        if not self.judge_inflight:
            return
        c = self.c
        left = self._clock_left()
        wait_s = c.hung_slot_s if left is None else max(0.1, min(c.hung_slot_s, left))
        concurrent.futures.wait(list(self.judge_inflight), timeout=wait_s)

    def _drain_judgments(self, *, wait_s: float = 0.0) -> int:
        """Fold every verdict that has landed into its row and let the
        allocator decide on it. Returns how many landed."""
        if not self.judge_inflight:
            return 0
        if wait_s > 0:
            concurrent.futures.wait(
                list(self.judge_inflight),
                timeout=wait_s,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
        landed = 0
        for fut in [f for f in list(self.judge_inflight) if f.done()]:
            row, job = self.judge_inflight.pop(fut)
            try:
                verdict = fut.result()
            except Exception:
                verdict = {}
            for key in self._JUDGE_KEYS:
                if key in verdict:
                    row[key] = verdict[key]
            prompt = str(row.get("prompt") or job[0] or "")
            if (
                self.c.mutate_graded_failures
                and _graded_failure(row, self.c.knobs.pass_threshold)
                and not mutation_worthy(row)
            ):
                # The grader's verdict steers like a tool fault (#285): the
                # row becomes a mutation parent, its situation counts as
                # failing, and it is re-rolled under the same gate a
                # faulted row gets. Rows the world already faulted are
                # parents already.
                self._aim_failure(row, "graded_failure")
                self.failing_rows.append(row)
                rid = row.get("scenario_id")
                if rid:
                    self.region_fails[rid] = self.region_fails.get(rid, 0) + 1
                if (
                    prompt
                    and not self._successive
                    and self.prompt_rollouts.get(prompt, 0) < self.c.repeat_count
                ):
                    self._queue_verify(prompt, job, 1)
            if prompt and self._successive:
                self._successive_update(prompt, row, job)
            landed += 1
        self.judged_in_loop += landed
        return landed

    def _aim_failure(self, row: dict, aim: str) -> None:
        """Record why ``row`` is a mutation parent, under the keys a
        mutated row's ``parent_failure_id`` can carry: the offline writer
        names the parent by its prompt head, the live writer by its
        situation id.

        ``aim`` says the row failed; ``failure_criteria`` says WHICH rule it
        broke. A generic "this row failed" cannot steer toward a specific
        behaviour, which is what a rubric is made of."""
        self.mutation_aims[aim]["parents"] += 1
        for name in failed_criteria(row):
            self.failure_criteria[name] = self.failure_criteria.get(name, 0) + 1
        prompt = str(row.get("prompt") or "")
        if prompt:
            self.failure_aim[prompt[:PARENT_HEAD_CHARS]] = aim
        rid = row.get("scenario_id")
        if rid:
            self.failure_aim[str(rid)] = aim

    def _resume_stopped(self, remaining: int) -> int:
        """Nothing fresh can be opened and rows are still owed: the
        unanimous groups the search stopped earlier are the best use of
        what is left, so each gets its next rollout back. Returns how
        many were resumed."""
        if remaining <= 0:
            return 0
        k = self.c.repeat_count
        resumed = 0
        for prompt, state in list(self.group_state.items()):
            if state != "stopped_unanimous" or self.prompt_rollouts.get(prompt, 0) >= k:
                continue
            job = self.group_job.get(prompt)
            if job is None:
                continue
            self.group_state[prompt] = "probing"
            self._queue_verify(prompt, job, 1)
            resumed += 1
        return resumed

    def _hazard(self, n: int) -> float:
        """Chance the next rollout of a group unanimous after ``n`` splits
        it: the run's own count at that n, with Laplace's rule on n
        unanimous draws (``1/(n+2)`` at alpha 1) as a one-observation
        prior. A run where continuations never split learns quickly that
        a unanimous group is a consistent cell."""
        prior = laplace(0, n, self.c.knobs.smoothing_alpha)
        seen = self.hazard_seen.get(n, 0)
        split = self.hazard_split.get(n, 0)
        return (split + prior) / (seen + 1.0)

    def _mixed_rate(self) -> float:
        """Share of probed groups that split, Laplace-smoothed."""
        return laplace(self.groups_mixed, self.groups_probed, self.c.knobs.smoothing_alpha)

    def _queue_verify(self, prompt: str, job: tuple, count: int) -> None:
        meta = job[_JOB_META] if len(job) > _JOB_META else {}
        sel = job[_JOB_SELECTION] if len(job) > _JOB_SELECTION else {}
        for _ in range(max(0, int(count))):
            self.verify_queue.append(
                (prompt, dict(meta or {}), sel if isinstance(sel, dict) else {})
            )

    def _situations_complete(self) -> bool:
        """Every situation the run was asked for (``situations=N``) has
        been drawn and every prompt drawn has all its rollouts, with no
        lost rollout owed. Nothing the writer adds can be rolled out, so
        the run stops launching waves and takes ``situations_exhausted``.
        Under ``tasks=`` the pinned set is the target, not this."""
        c = self.c
        return bool(
            c.n_situations_target
            and not self.pinned_prompts
            and len(self.used_situations) >= c.n_situations_target
            and self.cap_lifted["lost"] == 0
            and all(self.prompt_rollouts.get(p, 0) >= c.repeat_count for p in self.used)
        )

    def _fresh_available(self) -> bool:
        """Can the run still open a new prompt? When it cannot, finishing
        a unanimous group costs nothing else."""
        c = self.c
        if self.closing:
            return False
        if (
            c.n_situations_target
            and not self.cap_lifted["lifted"]
            and len(self.used_situations) >= c.n_situations_target
        ):
            return False
        return bool(self._available()) or self.generator.model is not None

    def _update_closing(self, left: float | None) -> None:
        """Stop opening groups when the clock cannot fit one more rollout
        round trip after the ones in flight. The estimate is the median of
        recent rollout durations, so it is the agent's own pace."""
        if self.closing or left is None or not self.rollout_durations:
            return
        if self.c.topo["repeat_policy"] != "successive" or self.c.k_immediate:
            return
        recent = sorted(self.rollout_durations[-self.c.knobs.closing_window_rollouts :])
        est = recent[len(recent) // 2]
        if left < self.c.knobs.closing_margin * est:
            self.closing = True
            note_stage(self.data, "closing: finishing groups before the clock")

    def _group_rows(self) -> dict[str, list[dict]]:
        by_prompt: dict[str, list[dict]] = {}
        for row in self.data.trajectories:
            prompt = str(row.get("prompt") or "")
            if prompt:
                by_prompt.setdefault(prompt, []).append(row)
        return by_prompt

    @staticmethod
    def _row_label(row: dict) -> Any:
        reward = row.get("reward")
        if reward in (0, 1) and not isinstance(reward, bool):
            return int(reward)
        return str(row.get("behavior_signature") or "")

    def _group_verdict(self, prompt: str, rows: list[dict]) -> str:
        """mixed, stopped_unanimous, complete, or partial, read from the
        stored rows so the summary and the stamp cannot disagree. Rows that
        land during shutdown never pass through ``_successive_update``."""
        k = self.c.repeat_count
        n = len(rows)
        if len({self._row_label(r) for r in rows}) > 1:
            return "mixed" if n >= k else "partial"
        if self.group_state.get(prompt) == "stopped_unanimous":
            return "stopped_unanimous"
        return "complete" if n >= k else "partial"

    def _group_summary(self) -> dict:
        c = self.c
        k = c.repeat_count
        counts = {"mixed": 0, "stopped_unanimous": 0, "complete": 0, "partial": 0}
        saved = 0
        groups = self._group_rows()
        for prompt, rows in groups.items():
            verdict = self._group_verdict(prompt, rows)
            counts[verdict] += 1
            if verdict == "stopped_unanimous":
                saved += max(0, k - len(rows))
        return {
            "k": k,
            "probe": max(1, min(k, c.probe)),
            "groups": len(groups),
            **counts,
            "rollouts_saved": saved,
            "mixed_rate": round(self._mixed_rate(), 4),
            "hazard": {str(n): round(self._hazard(n), 4) for n in sorted(self.hazard_seen)},
            "idle_on_judge_s": round(self.idle_on_judge_s, 1),
            "truncated_skipped": self.skipped_truncated,
            "closing": self.closing,
        }

    def _stamp_groups(self) -> None:
        """Mark rows of groups the budget or clock cut short, so a reader
        can tell a cut group from one the search stopped on purpose."""
        for prompt, rows in self._group_rows().items():
            if self._group_verdict(prompt, rows) == "partial":
                for row in rows:
                    row["group_cut"] = True

    def _axis_gaps(self) -> list[str]:
        """Prose nudges for the writer about axes the rows so far miss."""
        data = self.data
        n_rows = max(1, len(data.trajectories))
        short_n = sum(1 for t in data.trajectories if "short" in str(t.get("length") or ""))
        long_n = sum(1 for t in data.trajectories if "long" in str(t.get("length") or ""))
        tones = {str(t.get("tone") or "") for t in data.trajectories}
        tiers = {str(t.get("tier") or "") for t in data.trajectories}
        tools_hit = {
            str((t.get("scenario_dimensions") or {}).get("tool") or "")
            for t in data.trajectories
            if isinstance(t.get("scenario_dimensions"), dict)
        }
        knobs = self.c.knobs
        axis_gaps: list[str] = []
        if short_n / n_rows < knobs.short_share_floor:
            axis_gaps.append("You keep it brief.")
        if long_n / n_rows < knobs.long_share_floor:
            axis_gaps.append("You use more words.")
        for tone, line in (
            ("frustrated", "You are frustrated."),
            ("curt", "You are curt."),
            ("polite", "You are being nice."),
        ):
            if tone not in tones:
                axis_gaps.append(line)
        if "adversarial" not in tiers:
            axis_gaps.append("You are pushing a constraint.")
        if "ambiguous" not in tiers:
            axis_gaps.append("You are confused.")
        for name in sorted(self.declared)[: knobs.writer_context_items]:
            if name and name not in tools_hit:
                intent = intent_for_tool(name)
                if intent:
                    axis_gaps.append(f"You want to {intent}.")
        return axis_gaps[: knobs.writer_context_items]

    def _shutdown(self) -> None:
        data = self.data
        self._settle_inflight()
        self._flush_output("stopped")
        self.scenario_pool.shutdown(wait=False, cancel_futures=True)
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.judge_pool.shutdown(wait=False, cancel_futures=True)
        if self.scene_thread is not None:
            knobs = self.c.knobs
            left = knobs.scene_join_s
            clock = self._clock_left()
            if clock is not None:
                left = max(knobs.writer_wait_floor_s, min(knobs.scene_join_clocked_s, clock))
            self.scene_thread.join(timeout=left)
            data.scene_brief = self.scene_box["brief"]
            if not data.scene_brief and "scene_brief_unavailable" not in data.degraded:
                data.degraded.append("scene_brief_unavailable")

    # ------------------------------------------------------------ finish

    def _finish(self) -> SimulationData:
        c = self.c
        data = self.data
        gen = self.generator
        self._note_progress(force=True)
        data.declared_tools = self.declared
        # The README promises messages on every row. The JSONL writer built
        # them lazily; a caller reading data.trajectories saw only steps.
        for row in data.trajectories:
            if not row.get("messages"):
                row["messages"] = conversation(row)
        # grader application happens once, at the end of simulate, through
        # run_judge: full judge contract (judge_status, lineage, no silent
        # zeros) instead of the legacy data.grade() write-back.
        if c.llm_grade:
            data.llm_grade(spec=c.llm_spec)
        if self.trace_rows:
            self._finish_traces()
        if gen.model is not None and gen.last_errors:
            data.search["writer_errors"] = dict(gen.last_errors)
        lost = int(self.cap_lifted.get("lost", 0))
        if lost:
            # Not missing at random: the tasks that failed are the ones a
            # cold endpoint or a leaky reply format failed on, so every
            # rate over the survivors is biased (the eval's composition decides
            # what a pass rate means, Lambert 2025, chapter Evaluation). Say so
            # on the run, with the fix for each way a rollout is lost.
            note = self._lost_note(lost)
            if "rollouts_lost" not in data.degraded:
                data.degraded.append("rollouts_lost")
            data.warnings.append(note)
            log.warning(note)
        empty = int(self.lost_by.get("empty_reply", 0))
        if not data.trajectories and empty and empty >= sum(self.lost_by.values()) * 0.9:
            # Every rollout came back without a reply, so the run spent its
            # budget on nothing. Name the cause and the one fix (#375).
            self._all_replies_empty = True
            note = (
                f"no rows: the agent returned an empty reply on all {empty} rollouts; "
                "return {'final_text': <what it said>, 'steps': [...]} from the agent "
                "callable (or check the endpoint answers) and run again"
            )
            if note not in data.warnings:
                data.warnings.append(note)
            log.warning(note)
        if self.agent_errors:
            # The callable raised (or returned nothing usable). The rows
            # were built and dropped; without this the run reports zero
            # rows and a stop reason that names the writer.
            data.search["agent_errors"] = self.agent_errors
            data.search["first_agent_error"] = self.first_agent_error
            if "agent_errors" not in data.degraded:
                data.degraded.append("agent_errors")
            if not data.trajectories:
                data.stopped_because = "agent_failed"
                log.warning(
                    "every rollout failed inside the agent (%d errors); first: %s",
                    self.agent_errors,
                    self.first_agent_error,
                )
        if "generator_fallback" in data.degraded and getattr(gen, "model_produced", False):
            # the note was set while the first waves were still in flight;
            # every prompt in the model path is model-written or nothing
            data.degraded.remove("generator_fallback")
        cut_in_flight = (
            "abandoned_rollouts" in data.search or "abandoned_writer_waves" in data.search
        )
        if not data.trajectories and data.stopped_because != "agent_failed" and not cut_in_flight:
            # No rows, the agent is not to blame, and nothing was still
            # running when the run stopped: the writer produced no situation
            # the run could use. "time_budget" here hid a hosted writer that
            # failed cold for the whole clock (dogfood, 2026-09-15). A clock
            # stop with work still in flight keeps its own reason.
            data.stopped_because = "writer_failed"
            errors = dict(getattr(gen, "last_errors", {}) or {})
            if getattr(self, "writer_fallback_error", ""):
                errors.setdefault("llm_guided", self.writer_fallback_error)
            data.search["writer_errors"] = errors
            log.warning(
                "no rows: the writer produced no usable situation in %.0fs (degraded=%s; %s)",
                time.monotonic() - self.started,
                ",".join(data.degraded) or "none",
                errors.get("llm_guided") or "no error text",
            )
        misses = int(self.turn_stats.get("followup_misses", 0) or 0)
        if misses:
            data.search["followup_misses"] = misses
            if (
                misses >= c.knobs.followup_starved_min
                and misses >= len(data.trajectories) // c.knobs.followup_starved_divisor
                and "followups_starved" not in data.degraded
            ):
                data.degraded.append("followups_starved")
        # The simulated user reasoned out loud. The reasoning was stripped
        # before it became a user turn; the counts and their shares of the
        # user turns are on every run, zeros included, so two arms can be
        # compared in the same unit as config["unclosed_think_share"]
        # (#284: 18 unclosed on one arm, 0 on the other).
        user_turns = int(self.turn_stats.get("user_turns", 0) or 0)
        stripped = int(self.turn_stats.get("user_think_stripped", 0) or 0)
        unclosed = int(self.turn_stats.get("user_think_unclosed", 0) or 0)
        data.search["user_think"] = {
            "user_turns": user_turns,
            "stripped": stripped,
            "unclosed": unclosed,
            "stripped_share": round(stripped / user_turns, 4) if user_turns else 0.0,
            "unclosed_share": round(unclosed / user_turns, 4) if user_turns else 0.0,
        }
        if stripped:
            note = (
                f"{stripped} simulated-user turns came back as <think> reasoning ({unclosed} cut "
                "off inside the block by the user model's token cap); the reasoning was stripped "
                "before it became a user turn, and a turn with no spoken line was dropped. Pass "
                "thinking=False to wai.local_model(...) so the user model does not reason, or "
                "put the user on a non-reasoning model with user_model=."
            )
            data.warnings.append(note)
            log.warning(note)
        data.elapsed_seconds = time.monotonic() - self.started
        n = len(data.trajectories)
        data.rows_per_second = (n / data.elapsed_seconds) if data.elapsed_seconds else 0.0
        data.unique_prompts = len({t["prompt"] for t in data.trajectories})
        data.unique_behavior_signatures = len({t["behavior_signature"] for t in data.trajectories})
        if data.semantic and data.trajectories:
            floor = c.knobs.semantic_duplicate_novelty
            duplicate = sum(
                1 for t in data.trajectories if float(t.get("semantic_novelty") or 0.0) < floor
            )
            data.semantic_duplicate_rate = duplicate / len(data.trajectories)
        if data.coverage_curve:
            data.coverage_curve[-1]["stopped_because"] = data.stopped_because
        else:
            record_coverage(
                data,
                data.trajectories,
                cells=self.cells,
                shape_keys=self.induced_shape_keys,
                arm_weights=data.arm_weights or self.search,
                stopped_because=data.stopped_because,
            )
        data.coverage = build_coverage_summary(
            data.coverage_curve,
            budget=c.cap,
            stopped_because=data.stopped_because,
            flat_streak=self.flat,
            last_batch_size=self.last_batch_size,
            copy_deficit=int((data.search or {}).get("copy_deficit") or 0),
        )
        data.coverage["min_cell_copies"] = (data.search or {}).get("min_cell_copies", 0)
        data.coverage["pairwise"] = pairwise_coverage(
            [json.loads(key) for key in self.planned_cell_keys],
            [t.get("scenario_dimensions") for t in data.trajectories],
        )
        data.coverage["copies_needed"] = SATURATION_COPIES
        data.coverage["unique"] = c.unique_cards
        data.coverage["unique_situations"] = c.unique_cards
        data.coverage["repeats"] = c.repeat_count
        # The knobs the run resolved, so a report says what it ran under:
        # every RunKnobs field, and the three named advanced keys that
        # steer the simulated user and the world.
        data.coverage["knobs"] = asdict(c.knobs)
        data.coverage["patience"] = c.patience
        data.coverage["user_temperature"] = c.user_temperature
        data.coverage["world"] = WorldOptions.coerce(c.world_options).summary()
        self._record_experiment_knobs()
        data.coverage["mode"] = c.topo["mode"]
        data.coverage["repeat_policy"] = c.topo["repeat_policy"]
        data.coverage["until"] = c.until_key
        # What was asked for against what came back. ``rollouts_lost`` are
        # the rollouts that finished but could not be rows (by reason in
        # ``rollouts_lost_by``); ``rollouts_over_cap`` finished after the
        # row budget was already full and were not needed. Agent error
        # counts stay in data.search["agent_errors"]. A run that asked for
        # 34 tasks x 4 and got 129 rows says so here instead of reporting
        # a clean pass rate over the 129 (#303).
        # What the generation knobs were set to, and what they produced.
        requested = {
            "fault_rate": c.fault_rate,
            "avg_turns": c.avg_turns,
            "stance": (c.dimensions or {}).get("stance")
            if isinstance(c.dimensions, dict)
            else None,
        }
        delivered = self._delivered()
        data.coverage["requested"] = requested
        data.coverage["delivered"] = delivered
        # The warning is about a setting the caller made and the rows missed.
        # ``requested`` carries a value for every knob, default or not, so
        # the check is narrowed to the ones the call named (#476): an unset
        # fault_rate arriving as 0.8 under mode="rl" is not an intention the
        # run failed. ``avg_turns`` is dropped when no simulated user was
        # played (callable and HTTP agents take one message in, one
        # trajectory out), since a turn count that does not apply cannot be
        # missed at any setting. ``stance`` only exists when the caller
        # passed it inside dimensions=.
        asked = {k: v for k, v in requested.items() if k in c.set_by_caller or k == "stance"}
        if self.user_model is None:
            asked.pop("avg_turns", None)
        self._warn_on_undelivered(asked, delivered)
        data.coverage["rollouts_requested"] = self._rollouts_requested()
        data.coverage["rollouts_completed"] = len(data.trajectories)
        data.coverage["rollouts_lost"] = int(self.cap_lifted.get("lost", 0))
        data.coverage["rollouts_lost_by"] = dict(self.lost_by)
        rollouts = self._progress(time.monotonic() - self.started)
        rollouts.pop("inflight", None)
        data.search["rollouts"] = rollouts
        if rollouts["rerolled"] > rollouts["landed"]:
            # More calls were thrown away than kept: the run spent most of
            # its time on rollouts that never became rows, and its wall
            # clock says nothing about its rows (#470: a 300 s timeout on
            # 4,096-token replies re-rolled each one up to k times).
            by = ", ".join(f"{n} {r.replace('_', ' ')}" for r, n in rollouts["rerolled_by"].items())
            fix = (
                f"raise timeout= (now {c.rollout_timeout:.0f} s; {rollouts['timed_out']} of the "
                "agent errors were call timeouts) or lower agent_max_tokens="
                if rollouts["timed_out"]
                else "data.search['rollouts']['rerolled_by'] says why; data.warnings has the fix per reason"
            )
            note = (
                f"{rollouts['rerolled']} rollouts were re-rolled against {rollouts['landed']} "
                f"rows landed ({by}); most of this run's calls never became rows. Fix: {fix}."
            )
            if note not in data.warnings:
                data.warnings.append(note)
            log.warning(note)
            warnings.warn(note, UserWarning, stacklevel=2)
        data.coverage["rollouts_over_cap"] = int(self.over_cap)
        data.coverage["n_situations"] = c.n_situations_target
        data.coverage["requests_per_situation"] = c.n_req
        data.coverage["rollouts_per_request"] = c.repeat_count
        data.allocator = dict(self.allocator_counts)
        if self.seed_amp_report:
            data.search["seed_amplification"] = self.seed_amp_report
        if self.seeds_dropped:
            # search is rebuilt every batch, so the note written before
            # the run is put back here, on the run the caller reads.
            data.search["seeds_dropped"] = list(self.seeds_dropped)
        if c.pinned_tasks:
            ran = {str(t.get("prompt") or "") for t in data.trajectories}
            pinned = set(self.pinned_prompts)
            data.search["pinned_tasks"] = {
                "prompts": len(pinned),
                "tasks": len({t["scenario_id"] for t in c.pinned_tasks if t.get("scenario_id")}),
                "ran": len(pinned & ran),
                "missing": sorted(pinned - ran)[:REPORT_LIST_ITEMS],
            }
        if self.drafted_tools:
            data.search["drafted_tools"] = self.drafted_tools
        data.search["strategy"] = {
            "requested": c.strategy,
            "resolved": c.resolved_strategy,
            "broaden": c.resolved_strategy != "targeted",
            "reason": (
                "traces supplied -> aimed distribution"
                if c.resolved_strategy == "trace" and c.strategy == "auto"
                else "no traces -> broad exploration"
                if c.strategy == "auto"
                else "explicit"
            ),
            "opening": {
                "requested": c.opening_req,
                "rate": round(self.opening_rate, 4),
                "source": self.opening_source,
            },
            "steering_weight": {
                "requested": c.steering_weight,
                "applied": self.applied_steering,
                "source": ("override" if c.steering_weight is not None else "rule"),
            },
        }
        self._finish_grading()
        if c.topo["repeat_policy"] == "successive" and not c.k_immediate:
            # after grading, so a group is read by its rewards, not by a
            # signature standing in for a verdict that had not landed
            self._stamp_groups()
            data.search["groups"] = self._group_summary()
            elapsed = max(1e-9, time.monotonic() - self.started)
            if self.idle_on_judge_s > c.knobs.idle_judge_share * elapsed:
                # every asked situation was probed and the pool waited on
                # verdicts; breadth, not the judge, is the fix
                need = -(-max(1, int(c.concurrency)) // max(1, min(c.repeat_count, c.probe)))
                note_stage(
                    data,
                    f"rl pool idle on judge verdicts {self.idle_on_judge_s:.0f}s of "
                    f"{elapsed:.0f}s; situations>={need} keeps {c.concurrency} rollouts busy",
                )
        if self.trace_rows and "behavior_state" in data.search:
            # Close the loop on the rows that ship: same region predicates
            # as the traces, measured after grading and leak-pruning.
            data.search["behavior_state"]["region_progress"] = region_progress(
                data.search["behavior_state"], data.trajectories
            )
        self._record_tier_mix()
        data.writer_model = self.writer_model
        if not data.trajectories and getattr(self, "_all_replies_empty", False):
            # Set last: earlier wrap-up names the writer, but the writer did
            # its job; the agent never answered (#375).
            data.stopped_because = "empty_replies"
        data.user_model = self.user_model
        # One model writing the exam, sitting it, and playing the examiner's
        # stand-in is the regime Lambert 2025 (chapter Synthetic Data and
        # Distillation) warns about: a model trained on its own unfiltered
        # output learns its own habits. The default still does it; the run
        # says so, once, and names the fix.
        roles = [
            role
            for role, tag in (
                ("wrote the situations", self.writer_model),
                ("played the user", self.user_model),
            )
            if self.agent_model and tag == self.agent_model
        ]
        if roles:
            fix = " and ".join(
                {"wrote the situations": "simulator=", "played the user": "user_model="}[r]
                for r in roles
            )
            note = (
                f"The agent model ({self.agent_model}) also {' and '.join(roles)}. "
                f"Pass {fix} to use a different model."
            )
            if "same_model" not in data.degraded:
                data.degraded.append("same_model")
            data.warnings.append(note)
            log.warning(note)
        # A run whose rollouts never called a tool is hollow: the writer
        # asked about things the world does not have, or the wrapper did
        # not record steps. Grading it gives a number that means nothing.
        rows = data.trajectories
        if rows and data.declared_tools:
            with_calls = sum(
                1
                for r in rows
                if any(isinstance(s, dict) and s.get("tool") for s in (r.get("steps") or []))
            )
            if with_calls == 0:
                note = (
                    f"0 of {len(rows)} rollouts called a tool, so this run says nothing "
                    "about tool use. Put the ids your world has (order numbers, account "
                    "names) in the tool descriptions or in seeds=, and check the agent "
                    "wrapper records its steps, before grading it."
                )
                if "no_tool_calls" not in data.degraded:
                    data.degraded.append("no_tool_calls")
                data.warnings.append(note)
                log.warning(note)
        # A declared tool the world cannot answer fails exactly like a world
        # fault: the agent reports the miss, an honesty rubric rewards it,
        # and the behaviour behind the tool never happens. Per-tool calls
        # and successes on every run, and the ones that never work named
        # with the fix (#287). Same fault rule as trace_mining's fault_n.
        # Written once, to coverage: data.report() reads coverage.
        if rows:
            outcomes = tool_outcomes(rows)
            if outcomes:
                data.coverage["tools"] = outcomes
            dead = dead_tools(outcomes)
            data.coverage["dead_tools"] = dead
            if dead:
                note = dead_tools_note(outcomes, dead, execute=c.execute is not None)
                if "dead_tools" not in data.degraded:
                    data.degraded.append("dead_tools")
                data.warnings.append(note)
                log.warning(note)
        if rows:
            # How many threads end on the agent's question, and how many of
            # those because the person walked away. Zero here on a run with
            # questions in it means asking was free (#289).
            data.search["ended_on_question"] = ended_on_question(rows)
        if c.out_path is not None and data.trajectories:
            data.save(str(c.out_path), meta=True)
        return data

    def _record_tier_mix(self) -> None:
        """What difficulty mixture the shipped rows carry, next to the ask.

        Rows the mixer never sees (seeds, open asks, the per-arm quota,
        cells with no stance) carry the ordinary label, so a run lands
        below the hard share it asked for; a small run more so. The gap is
        recorded, and when the caller set the dial and the gap passes ten
        points the run says so and names the pin.
        """
        c = self.c
        data = self.data
        requested = HARD_SHARE if c.hard_share is None else float(c.hard_share)
        mix = tier_mix_of(data.trajectories, requested)
        rows = int(mix["rows"])
        hard = sum(mix["counts"].get(tier, 0) for tier in HARD_TIERS)
        realized = hard / rows if rows else None
        asked = c.hard_share is not None and rows >= c.knobs.tier_mix_min_rows
        if asked and realized is not None and requested - realized > c.knobs.tier_mix_tolerance:
            mix["note"] = (
                f"hard_share={requested:g} asked, {realized:.2f} drawn "
                f"({hard} of {rows} rows from the hard tiers). Open asks and cells "
                "with no stance count as ordinary, and the grid holds a fixed number of "
                "hard cells. For a set that is hard throughout, pin the axis: "
                "dimensions={'stance': ['boundary', 'ambiguous', 'adversarial']}."
            )
            data.warnings.append(mix["note"])
            log.warning(mix["note"])
        data.search["tier_mix"] = mix

    def _finish_traces(self) -> None:
        """Record what the traces did to the run and drop generated rows
        that near-copy a source trace."""
        data = self.data
        # Source traces shaped the grid; they must not shape the rows.
        # A generated near-copy of a held-out trace is training leakage.
        mined = mine_traces(self.trace_rows)
        # The optimizer's map rides every trace-fed run: the trace
        # history classifies into behavior regions (new / persistent /
        # improving / uncertain / passing) with budget shares. Recorded
        # for callers and the platform UI; allocation is disclosure
        # until the steering calibration sets how hard to apply it.
        state_record = dict(self.optimizer_state or behavior_state(self.trace_rows))
        # applied means a cell weight actually changed, not merely that
        # regions existed; the gain reads the one constant that steers.
        state_record["applied"] = self.allocation_hits["n"] > 0
        state_record["allocation_gain"] = self.c.knobs.allocation_gain
        # region_progress is attached at the very end of simulate(), so
        # it measures the rows that ship: graded, leak-pruned.
        data.search["behavior_state"] = state_record
        kept_rows, leak = drop_leaky_rows(
            data.trajectories, self.trace_rows, embedder=self.resolved_embedder
        )
        data.trajectories[:] = kept_rows
        data.search["trace_mining"] = {
            "n_traces": mined["n"],
            "n_flaw_rows": len(mined["flaw_rows"]),
            "failure_seeds": getattr(self, "failure_seeds", 0),
            "faults": mined["faults"],
            "tools": {name: dict(slot) for name, slot in mined["tools"].items()},
            # Observed result payloads reused as shape templates for
            # invented results.
            "result_exemplars": {
                name: len(values) for name, values in self.trace_exemplars.items()
            },
            "focused_dimensions": {
                axis: list(values) for axis, values in (self.dimensions or {}).items()
            },
        }
        data.search["trace_leakage"] = {
            key: leak[key]
            for key in ("n", "n_sources", "threshold", "n_leaky", "n_dropped", "max_similarity")
        }
        # Dropped rows are not refilled (the loop has already ended), so a
        # 39%-short dataset must say why instead of standing next to
        # stopped_because="budget" as if the budget were met.
        if leak.get("n_dropped") and "trace_leakage_dropped" not in data.degraded:
            data.degraded.append("trace_leakage_dropped")

    def _finish_grading(self) -> None:
        c = self.c
        data = self.data
        if c.grader is not None:
            # Scores generated rows with the caller's grader, writes the
            # verdicts onto the trajectories, and discloses the split.
            # Judge failures mark rows unjudged instead of silent zeros.
            from ..score.judging import ScoredData, run_judge

            pending = [row for row in data.trajectories if "judge_status" not in row]
            if pending:
                fresh = run_judge(pending, c.grader, source="grade")
                for row, verdict in zip(pending, fresh.rows):
                    for key in self._JUDGE_KEYS:
                        if key in verdict:
                            row[key] = verdict[key]
            judge_name = next(
                (str(r.get("judge_name")) for r in data.trajectories if r.get("judge_name")),
                getattr(c.grader, "__name__", "grader"),
            )
            run_id = next(
                (
                    str((r.get("lineage") or {}).get("scoring_run_id"))
                    for r in data.trajectories
                    if isinstance(r.get("lineage"), dict) and r["lineage"].get("scoring_run_id")
                ),
                "",
            )
            scored = ScoredData(
                list(data.trajectories), run_id=run_id, source="grade", judge_name=judge_name
            )
            data.search["grader"] = {
                "judge": scored.judge_name,
                "judged_in_loop": self.judged_in_loop,
                "judged_after": len(pending),
                "errors": sum(
                    1 for r in data.trajectories if r.get("judge_status") not in (None, "ok")
                ),
                "scored": len(scored),
                "passes": len(scored.passes()),
                "failures": len(scored.failures()),
                "partials": len(scored.partials()),
                "unjudged": len(scored.unjudged()),
            }
        if c.grade and c.grader is None and not c.llm_grade and data.trajectories:
            # The deterministic conduct check, offline and free. It reports
            # what the agent did, not whether it did the job, so its rows carry
            # label_source="conduct" and must not be read as a rubric grade.
            # "grade" reads as "grade against my rubric" and this checked no
            # rubric, so say so rather than let a conduct score be mistaken for
            # one. On a run with no tools it has nothing to check and returns
            # conforms for every row: measured 212 of 212 at reward 1.0 on a
            # tau2 airline spec, a reply of "Sure, cancelled." among them, and
            # select_for_sft(min_reward=1.0) then took all 212 as gold. A
            # reward nobody chose is worse than no reward (Lambert 2025,
            # chapter Reward Models).
            warnings.warn(
                "grade=True scored these rows with the conduct check, not "
                "against a rubric: it reports what the agent did, not "
                "whether it did the job, and on a run with no tools it "
                "returns conforms for every row. Rows carry "
                'label_source="conduct". For a grade against a rubric '
                "pass llm_grade=True, grader=, or grade() afterwards.",
                stacklevel=2,
            )
            declared = {
                str((t.get("function") or t).get("name") or "")
                for t in (data.profile.tools or [])
                if isinstance(t, dict)
            }
            for row in data.trajectories:
                if row.get("reward") is not None:
                    continue
                verdict = conduct_grade(row, declared or None)
                row["reward"] = verdict.get("reward")
                # the row template pre-seeds reason=None, so setdefault kept
                # every conduct reason off the export
                if verdict.get("reason") is not None and not row.get("reason"):
                    row["reason"] = verdict["reason"]
                row["label_source"] = "conduct"
