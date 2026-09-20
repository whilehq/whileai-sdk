"""A long run can be watched, resumed, and read afterwards (#470).

A 2,400-row ``simulate(tasks=...)`` through a vLLM agent ran four hours
as a black box: nothing said the rows were being re-rolled, no row was
on disk until the call returned, and a kill lost every one. These pin
the four asks: progress with re-roll and loss counts (logger and
``on_progress=``), ``checkpoint=`` that lands each row at once and
resumes, the counters on ``data.search["rollouts"]`` with a warning when
re-rolls outnumber rows, and a call timeout that scales with the reply
budget.
"""

from __future__ import annotations

import json
import logging

import pytest

import whileai.simulations as wai
from tests.helpers import POLICY, TOOLS, simulate_offline
from whileai.simulations.defaults import TIMEOUT_TOKENS_PER_SECOND
from whileai.simulations.generate.agents import LOCAL_MODEL_TIMEOUT, reply_budget
from whileai.simulations.run.config import default_rollout_timeout, resolve_run_config


def _ok(message: str) -> dict:
    return {"final_text": "Done.", "steps": [{"role": "assistant", "text": "Done."}]}


def _counting(calls: list) -> callable:
    def agent(message: str) -> dict:
        calls.append(message)
        return _ok(message)

    return agent


def _lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "whileai.simulations"]


# ------------------------------------------------------------- progress


def test_on_progress_fires_with_monotone_counters():
    seen: list[dict] = []
    data = simulate_offline(_ok, budget=30, concurrency=2, on_progress=seen.append)
    assert seen, "on_progress never fired"
    keys = {
        "rows",
        "cap",
        "landed",
        "resumed",
        "rerolled",
        "rerolled_by",
        "timed_out",
        "lost",
        "lost_by",
        "inflight",
        "situations",
        "elapsed_s",
    }
    assert all(keys <= set(p) for p in seen)
    for key in ("rows", "landed", "rerolled", "lost", "elapsed_s"):
        values = [p[key] for p in seen]
        assert values == sorted(values), (key, values)
    assert seen[-1]["rows"] == len(data.rows()) == 30
    assert seen[-1]["landed"] == 30
    assert seen[-1]["resumed"] == 0
    assert seen[-1]["cap"] == 30


def test_on_progress_fires_even_on_a_budget_too_small_for_the_log_line(caplog):
    seen: list[dict] = []
    with caplog.at_level(logging.INFO, logger="whileai.simulations"):
        simulate_offline(_ok, budget=4, on_progress=seen.append)
    assert seen and seen[-1]["rows"] == 4
    assert not [line for line in _lines(caplog) if "rollouts," in line]


def test_progress_line_says_rerolled_and_lost():
    from whileai.simulations.run import engine

    line = engine.progress_line(
        12, 64, 3, 100.0, rerolled=6, lost=1, lost_by={"agent_error": 1}, resumed=4
    )
    assert line.startswith("12/64 rollouts (4 resumed), 3 situations written, 1m40s elapsed")
    assert line.endswith(", 6 re-rolled, 1 lost (1 agent error)")
    # the estimate is over the rows this call landed, not the resumed ones
    assert "~" in line


def test_on_progress_must_be_callable():
    with pytest.raises(TypeError, match="on_progress"):
        simulate_offline(_ok, budget=4, on_progress="print")


# ----------------------------------------------------------- checkpoint


def test_checkpoint_has_one_line_per_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    data = simulate_offline(_ok, budget=12, checkpoint=str(path))
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == len(data.rows()) == 12
    assert [row["prompt"] for row in lines] == [row["prompt"] for row in data.rows()]
    assert all("messages" in row and "scenario_id" in row for row in lines)


def _pinned(agent, tasks, budget, path):
    return simulate_offline(
        agent,
        tasks=tasks,
        mode="sft",
        repeats=2,
        per_round=40,
        concurrency=1,
        budget=budget,
        checkpoint=str(path),
    )


def test_a_rerun_with_the_same_checkpoint_and_tasks_skips_finished_tasks(tmp_path):
    base = simulate_offline(_ok, mode="sft", repeats=2, per_round=40, concurrency=1, budget=12)
    assert len(base.rows()) == 12
    prompts = {r["prompt"] for r in base.rows()}
    assert len(prompts) == 6
    path = tmp_path / "ckpt.jsonl"

    # a run cut short: one task finished (2 rows) and one half done (1 row)
    first_calls: list[str] = []
    first = _pinned(_counting(first_calls), base, 3, path)
    assert len(first.rows()) == 3
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert first.search["rollouts"]["resumed"] == 0
    assert first.search["rollouts"]["landed"] == 3

    # the resume: finished tasks skipped, the half-done one topped up
    second_calls: list[str] = []
    second = _pinned(_counting(second_calls), base, 12, path)
    rows = second.rows()
    assert len(rows) == 12
    assert {r["prompt"] for r in rows} == prompts
    by_prompt: dict[str, int] = {}
    for r in rows:
        by_prompt[r["prompt"]] = by_prompt.get(r["prompt"], 0) + 1
    assert set(by_prompt.values()) == {2}, by_prompt
    assert len(second_calls) == 9, "resumed tasks were rolled out again"
    assert second.search["rollouts"]["resumed"] == 3
    assert second.search["rollouts"]["landed"] == 9
    assert sum(1 for r in rows if (r.get("lineage") or {}).get("resumed")) == 3
    # the file now holds the whole run, one line per row
    assert len(path.read_text(encoding="utf-8").splitlines()) == 12
    assert second.stopped_because in {"tasks_done", "situations_exhausted", "budget"}, (
        second.stopped_because
    )
    # the shape is the shape: rows() reads the same either way
    assert set(rows[0]) >= {"prompt", "final_text", "steps", "scenario_id"}

    # nothing left to do: a third call rolls out nothing and returns the union
    third_calls: list[str] = []
    third = _pinned(_counting(third_calls), base, 12, path)
    assert len(third.rows()) == 12
    assert third_calls == []
    assert third.search["rollouts"]["resumed"] == 12


def test_checkpoint_rows_outside_the_task_set_stay_out(tmp_path):
    base = simulate_offline(_ok, mode="sft", repeats=2, per_round=40, concurrency=1, budget=8)
    path = tmp_path / "ckpt.jsonl"
    path.write_text(
        json.dumps({"prompt": "not one of the tasks", "final_text": "x", "steps": []}) + "\n",
        encoding="utf-8",
    )
    data = _pinned(_ok, base, 8, path)
    assert len(data.rows()) == 8
    assert not any(r["prompt"] == "not one of the tasks" for r in data.rows())
    assert any("not in tasks=" in w for w in data.warnings)


def test_checkpoint_and_runs_do_not_combine(tmp_path):
    with pytest.raises(ValueError, match="checkpoint="):
        simulate_offline(_ok, budget=4, runs=2, checkpoint=str(tmp_path / "x.jsonl"))


# ------------------------------------------------------ counters and warning


def test_rerolls_are_counted_by_reason_on_the_run():
    seen: set[str] = set()

    def flaky(message: str) -> dict:
        # the first call on every prompt fails; the re-roll lands
        if message not in seen:
            seen.add(message)
            raise RuntimeError("boom")
        return _ok(message)

    data = simulate_offline(flaky, budget=8, concurrency=1)
    counts = data.search["rollouts"]
    assert len(data.rows()) == 8
    assert counts["landed"] == 8
    assert counts["rerolled"] >= 8
    assert counts["rerolled_by"].get("agent_error", 0) >= 8
    assert counts["lost"] == 0
    assert counts["timed_out"] == 0
    assert "inflight" not in counts


def test_more_rerolls_than_rows_is_a_user_warning():
    def silent(message: str) -> dict:
        return {"final_text": "", "steps": []}

    with pytest.warns(UserWarning, match="re-rolled against 0 rows landed"):
        data = simulate_offline(silent, budget=6, concurrency=1)
    counts = data.search["rollouts"]
    assert counts["landed"] == 0
    assert counts["rerolled"] > 0
    assert counts["rerolled_by"] == {"empty_reply": counts["rerolled"]}
    assert counts["lost"] > 0
    assert any("re-rolled against 0 rows landed" in w for w in data.warnings)


# ------------------------------------------------------------ timeout rule


def test_default_timeout_scales_with_the_reply_budget():
    assert default_rollout_timeout(None) == max(
        LOCAL_MODEL_TIMEOUT, reply_budget(None) / TIMEOUT_TOKENS_PER_SECOND
    )
    assert default_rollout_timeout(4096) == 1024.0
    assert default_rollout_timeout(256) == LOCAL_MODEL_TIMEOUT
    cfg = resolve_run_config(
        None, tools=TOOLS, system_prompt=POLICY, passed={"agent_max_tokens": 4096}
    )
    assert cfg.rollout_timeout == 1024.0
    assert cfg.checkpoint_path is None and cfg.on_progress is None
    # an explicit timeout= still wins
    cfg = resolve_run_config(
        None,
        tools=TOOLS,
        system_prompt=POLICY,
        passed={"agent_max_tokens": 4096, "timeout": 90},
    )
    assert cfg.rollout_timeout == 90.0


def test_the_rule_is_in_the_docstring():
    doc = wai.simulate.__doc__ or ""
    assert "max(300, agent_max_tokens / 4)" in doc
    assert "checkpoint" in doc and "on_progress" in doc
