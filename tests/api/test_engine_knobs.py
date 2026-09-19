"""Every number the run engine reads is named in ``defaults.py`` and
tunable through ``advanced={...}``.

Each knob is checked two ways: it lands on ``RunConfig.knobs`` under its
own name (and never leaks into the writer's kwargs), and the engine reads
it from there, so setting it changes what the engine does. The reference
table is checked against the defaults so the two cannot drift.
"""

from __future__ import annotations

import ast
import re
from dataclasses import fields
from pathlib import Path

import pytest

from tests.helpers import POLICY, TOOLS, simulate_offline
from whileai.simulations import defaults
from whileai.simulations.data import SimulationData
from whileai.simulations.defaults import RunKnobs, knob_bounds, knob_default, knob_names, laplace
from whileai.simulations.run import engine
from whileai.simulations.run.config import resolve_run_config
from whileai.simulations.run.engine import Run, _graded_failure

REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "reference" / "parameters.md"


def _cfg(**advanced):
    return resolve_run_config(None, tools=TOOLS, system_prompt=POLICY, advanced=advanced)


def _other_value(name: str):
    """A legal value for ``name`` that differs from its default."""
    default = knob_default(name)
    lo, hi = knob_bounds(name)
    kind = int if isinstance(default, int) else float
    candidate = default + 1 if kind is int else default + 0.05
    if hi is not None and candidate > hi:
        candidate = default - (1 if kind is int else 0.05)
    assert lo is None or candidate >= lo, name
    assert candidate != default
    return candidate


# ------------------------------------------------------ the mechanism


@pytest.mark.parametrize("name", knob_names())
def test_every_knob_lands_on_run_config_under_its_own_name(name):
    value = _other_value(name)
    cfg = _cfg(**{name: value})
    assert getattr(cfg.knobs, name) == value
    # never rides **advanced into the situation writer
    assert name not in cfg.advanced
    # and the default is what defaults.py says
    assert getattr(_cfg().knobs, name) == knob_default(name)


def test_every_knob_has_a_reason_next_to_it():
    """The comment above each field carries its name and a source."""
    source = Path(defaults.__file__).read_text(encoding="utf-8")
    body = source[source.index("class RunKnobs") :]
    for name in knob_names():
        line = re.search(rf"^\s+{name}: (int|float) = knob\(", body, re.M)
        assert line, name
        above = body[: line.start()].rstrip().splitlines()[-12:]
        comments = "\n".join(x.strip() for x in above if x.strip().startswith("#"))
        assert f"{name} =" in comments, f"{name} has no `# {name} = <value>:` comment"
        assert (
            "arXiv" in comments
            or "Lambert 2025" in comments
            or "measured" in comments
            or "observed" in comments
            or "convention, untested" in comments
            or "Laplace" in comments
            or re.search(r"\bsee\b", comments)
        ), f"{name}: the comment names no source"


def test_a_knob_below_its_floor_names_the_floor_and_the_default():
    with pytest.raises(ValueError) as err:
        _cfg(empty_rounds_to_stop=0)
    assert "empty_rounds_to_stop" in str(err.value)
    assert "floor 1" in str(err.value)
    assert "default is 8" in str(err.value)


def test_a_knob_above_its_ceiling_is_refused():
    with pytest.raises(ValueError, match=r"ceiling 1.0"):
        _cfg(gap_weight=1.5)


def test_a_knob_of_the_wrong_type_is_refused():
    with pytest.raises(ValueError, match="is an integer"):
        _cfg(empty_rounds_to_stop="eight")
    with pytest.raises(ValueError, match="is an integer"):
        _cfg(empty_rounds_to_stop=2.5)
    with pytest.raises(ValueError, match="is a number"):
        _cfg(collect_wait_s=True)
    # an integral float is fine for an int knob, and an int for a float one
    assert _cfg(empty_rounds_to_stop=3.0).knobs.empty_rounds_to_stop == 3
    assert _cfg(collect_wait_s=1).knobs.collect_wait_s == 1.0


def test_knob_names_are_plain_english_and_unique():
    names = knob_names()
    assert len(names) == len(set(names))
    for name in names:
        assert re.fullmatch(r"[a-z][a-z0-9_]+", name), name
        assert not re.search(r"\d", name.split("_")[0]), name


def test_readme_lists_every_knob_with_its_default():
    text = REFERENCE.read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| `([a-z_]+)` \| `([^`]*)` \|", text, re.M))
    missing = [n for n in knob_names() if n not in rows]
    assert not missing, f"docs/reference/parameters.md advanced table lacks {missing}"
    wrong = {}
    for name in knob_names():
        documented = ast.literal_eval(rows[name])
        if documented != knob_default(name):
            wrong[name] = (documented, knob_default(name))
    assert not wrong, f"docs/reference/parameters.md says / defaults.py says: {wrong}"


# ------------------------------------------------- the engine reads it


def _run(**advanced) -> Run:
    run = Run(_cfg(**advanced))
    run._init_loop_state()
    return run


def _close(run: Run) -> None:
    run.pool.shutdown(wait=False)
    run.scenario_pool.shutdown(wait=False)
    run.judge_pool.shutdown(wait=False)


def test_writer_buffer_and_allowances_follow_the_knobs():
    base = _run()
    try:
        assert base.writer_buffer == max(8 * 1 * 2, min(32 * 2, 96))
        assert base.agent_error_allowance == max(16, 2 * 1000)
        assert base.max_restarts == max(engine.MAX_NOVELTY_RESTARTS, 1000 // 100)
        assert (base.progress_every_s, base.progress_every_rows) == (10.0, 10)
    finally:
        _close(base)
    run = _run(
        writer_buffer_waves=3,
        writer_buffer_cap=40,
        dead_agent_errors=5,
        dead_agent_budget_multiple=0,
        rows_per_extra_restart=10,
        progress_every_s=1.0,
        progress_every_rows=2,
    )
    try:
        assert run.writer_buffer == max(8 * 1 * 3, min(32 * 3, 40))
        assert run.agent_error_allowance == 5
        assert run.max_restarts == max(engine.MAX_NOVELTY_RESTARTS, 1000 // 10)
        assert (run.progress_every_s, run.progress_every_rows) == (1.0, 2)
    finally:
        _close(run)


def test_empty_rounds_to_stop_ends_a_run_whose_writer_wrote_nothing():
    """A model writer, an empty pool, N empty rounds: the run raises."""

    def check(knob: int, streak: int) -> str:
        run = _run(empty_rounds_to_stop=knob)
        try:
            run.data = SimulationData(profile=run.c.spec)
            run.generator = type(
                "Gen", (), {"model": object(), "last_errors": {"llm_guided": "empty response"}}
            )()
            run.empty_streak = streak - 1
            run.generated_pool = []
            try:
                return run._on_empty_batch(remaining=10)
            except RuntimeError as exc:
                return f"raised: {exc}"
        finally:
            _close(run)

    assert check(knob=3, streak=3).startswith("raised: hosted Qwen produced no situations")
    assert check(knob=8, streak=3) == "continue"


def test_writer_idle_rounds_to_restart_and_the_avoid_window():
    seen = {}

    def check(**knobs) -> None:
        run = _run(**knobs)
        try:
            run.data = SimulationData(profile=run.c.spec)
            run.generator = type(
                "Gen", (), {"model": object(), "last_errors": {}, "fault_plans": {}}
            )()
            run.generated_pool = ["a"]
            run.used = {f"ask {i}" for i in range(20)}
            run.writer_idle = 2
            run.c.unique_cards = False  # a unique-situation run never restarts
            run.max_restarts = 5

            def restart(round_id, info, *, clear_avoid):
                seen["window"] = list(info["concentrated"])
                return round_id

            run._novelty_restart = restart  # type: ignore[method-assign]
            run._launch_writers = lambda n: None  # type: ignore[method-assign]
            run._on_empty_batch(remaining=10)
            seen["restarted"] = run.writer_idle == 0
        finally:
            _close(run)

    check(writer_idle_rounds_to_restart=2, restart_avoid_window=3)
    assert seen["restarted"] and len(seen["window"]) == 3
    seen.clear()
    check(writer_idle_rounds_to_restart=4)
    assert not seen.get("restarted")


def test_smoothing_alpha_changes_the_hazard_and_the_mixed_rate():
    assert laplace(0, 2) == 0.25
    assert laplace(3, 3, alpha=0.5) == 0.875
    run = _run()
    try:
        assert run._hazard(2) == 1.0 / 4.0
        run.groups_mixed, run.groups_probed = 1, 3
        assert run._mixed_rate() == 2.0 / 5.0
    finally:
        _close(run)
    run = _run(smoothing_alpha=0.5)
    try:
        assert run._hazard(2) == 0.5 / 3.0
        run.groups_mixed, run.groups_probed = 1, 3
        assert run._mixed_rate() == 1.5 / 4.0
    finally:
        _close(run)


def test_allocation_gain_and_match_weights_scale_the_boost():
    region = {"budget_share": 0.5, "recipe": {"tool": ["lookup_order"], "tool_condition": ["ok"]}}
    cell = {"tool": "lookup_order", "tool_condition": "ok"}

    def boost(**knobs) -> float:
        run = Run(_cfg(**knobs))
        run.optimizer_state = {"regions": [region]}
        return run._allocation_boost(cell)

    assert boost() == 1.0 + 4.0 * 0.5 * (0.6 + 0.4)
    assert boost(allocation_gain=1.0) == 1.0 + 1.0 * 0.5 * 1.0
    assert boost(allocation_tool_weight=0.2, allocation_condition_weight=0.1) == pytest.approx(
        1.0 + 4.0 * 0.5 * 0.3
    )


def test_closing_margin_reads_the_median_of_the_window():
    def closes(left: float, **knobs) -> bool:
        run = _run(**knobs)
        try:
            run.data = SimulationData(profile=run.c.spec)
            run.c.topo["repeat_policy"] = "successive"
            run.c.k_immediate = False
            run.rollout_durations = [1.0] * 30 + [10.0] * 5
            run._update_closing(left)
            return run.closing
        finally:
            _close(run)

    # default: last 20 -> median 1.0 (15 ones, 5 tens); closes under 2 s
    assert closes(1.9)
    assert not closes(2.1)
    assert closes(2.9, closing_margin=3.0)
    # a window of 5 sees only the tens: median 10, closes under 20 s
    assert closes(19.0, closing_window_rollouts=5)
    assert not closes(21.0, closing_window_rollouts=5)


def test_axis_gap_floors_and_writer_context_items():
    def gaps(**knobs) -> list[str]:
        run = _run(**knobs)
        try:
            run.data = SimulationData(profile=run.c.spec)
            run.data.trajectories = [{"length": "short"}] + [{"length": "long"}] * 9
            run.declared = set()
            return run._axis_gaps()
        finally:
            _close(run)

    default = gaps()
    assert "You keep it brief." not in default  # 10% short clears the 8% floor
    assert "You keep it brief." in gaps(short_share_floor=0.5)
    assert "You use more words." not in default
    assert len(default) == 5  # tones and tiers
    assert len(gaps(writer_context_items=2)) == 2


def test_tier_mix_note_waits_for_min_rows_and_respects_the_tolerance():
    def note(rows: int, **knobs) -> str | None:
        cfg = resolve_run_config(
            None, tools=TOOLS, system_prompt=POLICY, advanced={"hard_share": 0.6, **knobs}
        )
        run = Run(cfg)
        run.data = SimulationData(profile=cfg.spec)
        run.data.trajectories = [{"tier": "ordinary"}] * rows
        run._record_tier_mix()
        return run.data.search["tier_mix"].get("note")

    assert note(19) is None and note(20)
    assert note(19, tier_mix_min_rows=5)
    assert note(20, tier_mix_tolerance=0.7) is None


def test_pass_threshold_decides_what_a_graded_failure_is():
    assert not _graded_failure({"reward": 0.7})
    assert _graded_failure({"reward": 0.7}, 0.8)
    assert not _graded_failure({"reward": None}, 0.8)


def test_pass_threshold_reaches_the_search_loop():
    def aims(**knobs) -> int:
        data = simulate_offline(
            budget=8,
            per_round=8,
            concurrency=1,
            grader=lambda row: {"reward": 0.7},
            advanced=knobs,
        )
        return data.search["mutation_aims"]["graded_failure"]["parents"]

    assert aims() == 0
    assert aims(pass_threshold=0.8) > 0


def test_failing_seeds_cap_bounds_the_asks_mined_from_traces():
    traces = [
        {"prompt": f"refund order 8{i}", "reward": 0, "final_text": "no", "steps": []}
        for i in range(6)
    ]
    data = simulate_offline(budget=6, per_round=6, concurrency=1, traces=traces)
    assert data.search["trace_mining"]["failure_seeds"] == 6
    data = simulate_offline(
        budget=6, per_round=6, concurrency=1, traces=traces, advanced={"failing_seeds_cap": 2}
    )
    assert data.search["trace_mining"]["failure_seeds"] == 2


def test_dead_agent_errors_calls_off_a_raising_agent_that_soon():
    def broken(message):
        raise RuntimeError("boom")

    def errors(allowance: int) -> int:
        data = simulate_offline(
            broken,
            budget=50,
            per_round=8,
            concurrency=1,
            advanced={"dead_agent_errors": allowance, "dead_agent_budget_multiple": 0},
        )
        assert data.stopped_because == "agent_failed"
        return int(data.search["agent_errors"])

    # the default allowance is max(16, 2 x budget) = 100 calls
    assert 3 <= errors(3) < errors(6) < 16


def test_module_constants_are_the_defaults_the_engine_reads():
    """Values that mean the same thing in two files come from one name."""
    assert resolve_run_config(None, tools=TOOLS, system_prompt=POLICY).budget == (
        defaults.DEFAULT_BUDGET
    )
    cfg = resolve_run_config(None, tools=TOOLS, system_prompt=POLICY, mode="rl")
    assert cfg.repeat_count == defaults.RL_ROLLOUTS_PER_PROMPT
    assert cfg.fault_rate == defaults.RL_FAULT_RATE
    assert cfg.probe == defaults.DEFAULT_PROBE
    assert cfg.concurrency == defaults.DEFAULT_CONCURRENCY
    assert cfg.hung_slot_s == defaults.HUNG_SLOT_S
    assert cfg.knobs.pass_threshold == defaults.PASS_THRESHOLD
    assert cfg.knobs.allocation_gain == defaults.ALLOC_GAIN
    assert {f.name for f in fields(RunKnobs)} == set(knob_names())
