"""One task key everywhere (score/stats.task_key), and the config every
report reads off the rows (score/passat.run_config, delta_report config)."""

from __future__ import annotations

import whileai.simulations as wai
from tests.generate.test_logprobs import TOOLS
from whileai.simulations.export import _stamp_groups
from whileai.simulations.generate.adapters import HTTP_REPLY_TOKENS, resolve
from whileai.simulations.generate.agents import (
    LOCAL_MODEL_TEMPERATURE,
    default_agent_spec,
    hosted_model,
    parse_backend_spec,
    reply_budget,
)
from whileai.simulations.score.curriculum import curriculum, retire_solved
from whileai.simulations.score.delta import delta_report
from whileai.simulations.score.optimize import group_signal, trim_unanimous_groups
from whileai.simulations.score.passat import pass_at, run_config
from whileai.simulations.score.stats import compare_runs, task_key


def test_task_key_is_the_situation_id_then_task_id_then_prompt():
    assert task_key({"scenario_id": "s1", "task_id": "t1", "prompt": "p"}) == "s1"
    assert task_key({"task_id": "t1", "prompt": "p"}) == "t1"
    assert task_key({"prompt": "p"}) == "p"
    assert task_key({}) == ""
    assert wai.task_key is task_key


def _situation_rows(n_tasks: int = 6, phrasings: int = 2, repeats: int = 2) -> list[dict]:
    """rl-shaped rows: each situation has ``phrasings`` wordings, each wording
    ``repeats`` rollouts, every situation half pass."""
    rows = []
    for t in range(n_tasks):
        for p in range(phrasings):
            for r in range(repeats):
                rows.append(
                    {
                        "scenario_id": f"scn_{t}",
                        "prompt": f"situation {t}, wording {p}",
                        "reward": (t + p + r) % 2,
                        "steps": [],
                        "final_text": "x",
                        "sampling": {"temperature": 0.8, "max_tokens": 768, "model": "m"},
                        "policy_version": "m@abc123",
                        "lineage": {"judge_version": "judge@v1"},
                    }
                )
    return rows


def test_every_report_counts_the_same_tasks():
    before = _situation_rows()
    after = [dict(r, reward=1) for r in before]
    rates = pass_at(before)
    assert rates.n_groups == 6  # six situations, not twelve wordings
    assert rates.n_rows == 24
    assert set(rates.per_task) == {f"scn_{t}" for t in range(6)}
    assert delta_report(before, after, target="pass_at_1")["n_paired_tasks"] == rates.n_groups
    assert compare_runs(before, after)["n_paired"] == rates.n_groups
    assert group_signal(before)["n_groups"] == rates.n_groups
    assert curriculum(before)["n_tasks"] == rates.n_groups
    assert wai.metric_summary(before)["n_tasks"] == rates.n_groups
    assert wai.eval_variance(before, after)["tasks_in_every_run"] == rates.n_groups


def test_phrasings_of_one_situation_pool_into_one_task():
    rows = [
        {"scenario_id": "s", "prompt": "where is my order", "reward": 1},
        {"scenario_id": "s", "prompt": "order status please", "reward": 0},
    ]
    out = pass_at(rows)
    assert out.n_groups == 1 and out.per_task == {"s": 0.5}
    assert group_signal(rows)["n_mixed"] == 1
    # rows from elsewhere, with no situation id: the prompt text is the task
    bare = [dict(r, scenario_id=None) for r in rows]
    assert pass_at(bare).n_groups == 2 and set(pass_at(bare).per_task) == {
        "where is my order",
        "order status please",
    }


def test_pruners_and_exports_group_by_the_same_key():
    rows = _situation_rows(n_tasks=2, phrasings=2, repeats=1)
    for r in rows:
        if r["scenario_id"] == "scn_0":
            r["reward"] = 1  # unanimous across its two wordings
    kept, report = trim_unanimous_groups(rows)
    assert report["n_groups_dropped"] == 1
    assert {task_key(r) for r in kept} == {"scn_1"}
    entries = [
        {"prompt": r["prompt"], "scenario_id": r["scenario_id"], "reward": r["reward"]}
        for r in rows
    ]
    _stamp_groups(entries)
    assert len({e["group_id"] for e in entries}) == 2
    assert all(e["k"] == 2 for e in entries)


def test_curriculum_task_id_is_the_situation_and_keeps_a_prompt():
    rows = _situation_rows(n_tasks=2, phrasings=2, repeats=2)
    for r in rows:
        if r["scenario_id"] == "scn_1":
            r["reward"] = 1  # solved
    cur = curriculum(rows)
    assert cur["n_tasks"] == 2
    assert [s["task_id"] for s in cur["trainable"]] == ["scn_0"]
    assert [s["task_id"] for s in cur["retired"]] == ["scn_1"]
    assert cur["trainable"][0]["prompt"] == "situation 0, wording 0"
    assert {task_key(r) for r in retire_solved(rows)} == {"scn_0"}


def test_pass_at_config_reads_the_rows():
    rows = _situation_rows(n_tasks=3, phrasings=1, repeats=4)
    cfg = pass_at(rows).config
    assert cfg == {
        "n_tasks": 3,
        "k": 4,
        "temperature": 0.8,
        "max_tokens": 768,
        "policy_version": "m@abc123",
        "judge_version": "judge@v1",
        "prompt_hash": "abc123",
        "mixed": [],
        "truncated_share": None,
        "answered_share": 1.0,
        "unclosed_think_share": 0.0,
    }
    assert pass_at(rows).to_dict()["config"] == cfg
    # rows that disagree: the field is None and named
    rows[0]["sampling"] = {"temperature": 1.0, "max_tokens": 768, "model": "m"}
    cfg = pass_at(rows).config
    assert cfg["temperature"] is None and cfg["mixed"] == ["temperature"]
    assert cfg["max_tokens"] == 768
    # rows that carry nothing: None, and not called mixed
    cfg = pass_at([{"prompt": "p", "reward": 1}]).config
    assert cfg["temperature"] is None and cfg["policy_version"] is None and cfg["mixed"] == []
    assert cfg["n_tasks"] == 1 and cfg["k"] == 1
    assert run_config([])["n_tasks"] is None
    assert pass_at([]).config["n_tasks"] == 0


def test_delta_report_names_every_setting_that_differs():
    before = _situation_rows()
    same = [dict(r, reward=1) for r in before]
    report = delta_report(before, same, target="pass_at_1")
    assert report["config"]["before"]["policy_version"] == "m@abc123"
    assert report["config"]["after"] == report["config"]["before"]
    assert (
        "Before and after are the same policy version; this compares a model to itself."
        in report["warnings"]
    )

    after = [
        dict(
            r,
            reward=1,
            policy_version="m2@def456",
            sampling={"temperature": 1.0, "max_tokens": 2048, "model": "m2"},
            lineage={"judge_version": "judge@v2"},
        )
        for r in before
    ]
    report = delta_report(before, after, target="pass_at_1")
    joined = " ".join(report["warnings"])
    assert "different judges (judge@v1 vs judge@v2)" in joined
    assert "temperature 0.8 and after at 1.0" in joined and "same temperature=" in joined
    assert "768 reply tokens and after 2048" in joined and "agent_max_tokens=" in joined
    assert "compares a model to itself" not in joined
    assert report["config"]["after"]["prompt_hash"] == "def456"

    # a side that says nothing is not accused of differing
    quiet = [
        {k: v for k, v in r.items() if k not in ("sampling", "policy_version", "lineage")}
        for r in after
    ]
    report = delta_report(before, quiet, target="pass_at_1")
    assert report["config"]["after"]["temperature"] is None
    assert not any("temperature" in w or "judges" in w or "itself" in w for w in report["warnings"])


def test_every_model_backed_runner_says_how_it_samples():
    _, hosted_name = parse_backend_spec(default_agent_spec())
    assert hosted_model(TOOLS).sampling == {
        "temperature": LOCAL_MODEL_TEMPERATURE,
        "max_tokens": reply_budget(),
        "model": hosted_name,
    }
    byo, kind = resolve("vllm:m@http://127.0.0.1:9", tools=TOOLS, max_tokens=4096)
    assert kind == "backend_spec"
    assert byo.sampling == {
        "temperature": LOCAL_MODEL_TEMPERATURE,
        "max_tokens": 4096,
        "model": "m",
    }
    http, kind = resolve("http://127.0.0.1:9/v1/chat/completions", tools=TOOLS, temperature=0.5)
    assert kind == "http"
    assert http.sampling == {
        "temperature": 0.5,
        "max_tokens": HTTP_REPLY_TOKENS,
        "model": "gpt-4o-mini",
    }
    # a callable is the caller's; nothing is claimed for it
    assert (
        getattr(
            resolve(lambda m: {"steps": [], "final_text": ""}, tools=TOOLS)[0], "sampling", None
        )
        is None
    )
