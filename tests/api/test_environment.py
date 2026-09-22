"""export_environment: a simulation becomes an installable RL environment.

Offline. The verifiers-backed tests skip when the package is not
installed (``uv sync --extra rl``); the export itself needs only the SDK.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

import whileai.simulations as wai
from whileai.harness import Disclosure, Harness
from whileai.simulations import environment as environment_mod
from whileai.simulations.environment import (
    _ref_of,
    _row_from_state,
    build_tasks,
    resolve_ref,
)
from whileai.simulations.score.grading import conduct_grade

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund",
            "description": "Refund an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]
POLICY = "Look up an order before refunding it."


def _rows() -> list[dict]:
    """Four prompts, k=4: one always solved, one never, two mixed, one ungraded."""
    plan = {
        "refund ORD-1 please": [1, 1, 1, 1],
        "refund ORD-2 please": [0, 0, 0, 0],
        "where is ORD-3": [1, 0, 1, 0],
        "cancel ORD-4 today": [1, 1, 0, 1],
        "hi, need help with ORD-5": [None],
    }
    rows = []
    for i, (prompt, labels) in enumerate(plan.items()):
        for k, label in enumerate(labels):
            rows.append(
                {
                    "prompt": prompt,
                    "scenario_id": f"s{i}",
                    "rollout_index": k,
                    "reward": label,
                    "seed": 7,
                    "world_state": "entity exists" if i % 2 else "",
                    "faults": {"lookup_order": {"mode": "timeout", "rate": 1.0}} if i == 2 else {},
                    "steps": [],
                    "final_text": "done",
                    "judge_status": "ok" if label is not None else "missing_reward",
                }
            )
    return rows


def outcome_reward(row: dict) -> dict:
    """A module-level judge: reward 1 when the agent looked the order up."""
    called = any(s.get("tool") == "lookup_order" for s in row.get("steps") or [])
    return {"reward": 1 if called else 0, "reason": "looked up" if called else "no lookup"}


def test_build_tasks_applies_band_split_and_decontamination():
    train, held, report = build_tasks(_rows(), holdout=0.5, band=(0.2, 0.8))
    tasks = train + held
    prompts = {t["prompt"] for t in tasks}
    assert "refund ORD-1 please" not in prompts and "refund ORD-2 please" not in prompts
    assert "hi, need help with ORD-5" in prompts  # ungraded prompts are kept
    assert report["band_dropped"] == 2 and report["tasks"] == 3
    mixed = next(t for t in tasks if t["prompt"] == "where is ORD-3")
    assert mixed["info"]["calibration"] == {"pass_rate": 0.5, "n": 4}
    assert mixed["info"]["faults"]["lookup_order"]["mode"] == "timeout"
    assert all(t["info"]["split"] in ("train", "holdout") for t in tasks)
    assert {t["example_id"] for t in tasks} == {t["info"]["task_id"] for t in tasks}


def test_partial_credit_counts_toward_the_band_and_the_contrast():
    # the checklist scores 0.5 when conduct is half; a prompt graded
    # [0.5, 0.5, 0.5, 1] used to get no calibration at all and skip the band
    rows = _rows()
    for r in rows:
        if r["prompt"] == "refund ORD-1 please":
            r["reward"] = 1 if r["rollout_index"] == 3 else 0.5
        if r["prompt"] == "refund ORD-2 please":
            r["reward"] = 0.5
    train, held, report = build_tasks(rows, holdout=0.5, band=(0.2, 0.8))
    tasks = {t["prompt"]: t for t in train + held}
    assert tasks["refund ORD-1 please"]["info"]["calibration"] == {"pass_rate": 0.625, "n": 4}
    assert tasks["refund ORD-2 please"]["info"]["calibration"] == {"pass_rate": 0.5, "n": 4}
    assert report["band_dropped"] == 0 and report["graded_prompts"] == 4
    assert report["graded_mixed"] == 3  # ORD-2 at a flat 0.5 has no contrast


def test_prompts_from_one_scenario_get_their_own_task_id_and_one_split():
    rows = _rows()
    for r in rows:
        r["scenario_id"] = "shared"
    train, held, _ = build_tasks(rows, holdout=0.5, band=None)
    tasks = train + held
    assert len({t["example_id"] for t in tasks}) == len(tasks) == 5
    assert len({t["info"]["split"] for t in tasks}) == 1


def test_build_tasks_explicit_holdout_and_no_band():
    train, held, _ = build_tasks(_rows(), holdout=["where is ORD-3"], band=None)
    assert [t["prompt"] for t in held] == ["where is ORD-3"]
    assert len(train) == 4  # no band: unanimous prompts stay


def test_refs_round_trip_and_reject_locals():
    ref = _ref_of(conduct_grade)
    assert ref == "whileai.simulations.score.grading:conduct_grade"
    assert resolve_ref(ref) is conduct_grade
    assert resolve_ref(_ref_of(outcome_reward)) is outcome_reward
    with pytest.raises(ValueError, match="importable"):
        _ref_of(lambda row: 1)
    with pytest.raises(ValueError, match="module:attr"):
        _ref_of("conduct_grade")
    # A Verifier instance with no module-level name is referenced by its
    # class, which the trainer instantiates bare: fine for a default one,
    # refused when it carries configuration a bare one would silently lose.
    from whileai.simulations.verify.code import CodeExec
    from whileai.simulations.verify.text import ExactMatch

    assert type(resolve_ref(_ref_of(ExactMatch()))) is ExactMatch
    with pytest.raises(ValueError, match="configured but not bound"):
        _ref_of(CodeExec(tests="assert candidate == '4'"))
    with pytest.raises(ValueError, match="configured but not bound"):
        _ref_of(ExactMatch(whole=True))


def test_export_environment_writes_an_installable_package(tmp_path):
    out = tmp_path / "refund-agent"
    report = wai.export_environment(
        _rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5, reward=outcome_reward
    )
    assert report["name"] == "refund_agent" and report["warnings"] == []
    assert report["reward"].endswith(":outcome_reward")
    assert resolve_ref(report["reward"]) is outcome_reward
    pkg = out / "refund_agent"
    spec = json.loads((pkg / "spec.json").read_text())
    assert spec["system_prompt"] == POLICY
    assert [t["name"] for t in spec["tools"]] == ["lookup_order", "create_refund"]
    assert spec["tools"][0]["parameters"]["required"] == ["order_id"]
    assert spec["max_turns"] >= 1 and spec["execute"] is None
    train = [json.loads(line) for line in (pkg / "data" / "train.jsonl").read_text().splitlines()]
    held = [json.loads(line) for line in (pkg / "data" / "holdout.jsonl").read_text().splitlines()]
    assert len(train) + len(held) == report["tasks"] == 3
    assert set(train[0]) == {"prompt", "info", "example_id"}
    assert (
        "prompt" in (pkg / "__init__.py").read_text()
        or "load_environment" in (pkg / "__init__.py").read_text()
    )
    pyproject = (out / "pyproject.toml").read_text()
    assert 'name = "refund-agent"' in pyproject and "verifiers>=0.3" in pyproject
    assert 'build-backend = "hatchling.build"' in pyproject
    assert 'include = ["refund_agent/**", "pyproject.toml", "README.md"]' in pyproject
    assert "[tool.verifiers.eval]" in pyproject
    readme = (out / "README.md").read_text()
    assert ":outcome_reward" in readme and "prime eval run refund-agent" in readme
    assert "2 train / 1 holdout" in readme or "1 train / 2 holdout" in readme
    assert "whileai>=" in pyproject  # the SDK carries the environment module
    assert not (pkg / "_zp_env.py").exists() and not (pkg / "_zp_checklist.py").exists()


def test_export_without_reward_defaults_to_the_checklist_and_warns_without_metadata(tmp_path):
    rows = _rows()
    for r in rows:  # the writer's assignment names the target tool
        r["scenario_dimensions"] = {"tool": "lookup_order", "stance": "ordinary"}
    report = wai.export_environment(rows, tmp_path / "env", tools=TOOLS, system_prompt=POLICY)
    assert report["reward"].endswith(":task_checklist")
    assert report["outcome_checkable"] == report["tasks"]
    # four prompts at the default holdout of 0.2 round to none held out, which
    # is a package that trains and cannot be measured; that is the only warning
    assert [w.split(":")[0] for w in report["warnings"]] == ["empty_holdout"]
    task = json.loads(
        (tmp_path / "env" / "env" / "data" / "train.jsonl").read_text().splitlines()[0]
    )
    assert task["info"]["scenario_dimensions"]["tool"] == "lookup_order"
    bare = [{"prompt": f"p{i}", "steps": [], "final_text": ""} for i in range(4)]
    report = wai.export_environment(
        bare, tmp_path / "bare", tools=TOOLS, system_prompt=POLICY, holdout=0.5
    )
    assert report["outcome_checkable"] == 0
    assert any("conduct_grade" in w and "call nothing" in w for w in report["warnings"])
    assert "call nothing" in (tmp_path / "bare" / "README.md").read_text()


def test_export_needs_tools(tmp_path):
    with pytest.raises(ValueError, match="tools"):
        wai.export_environment(_rows(), tmp_path / "env", system_prompt=POLICY)


def test_task_file_is_not_a_training_file():
    """The answer key rides in ``info`` for the server; nothing in the
    student-visible ``prompt`` field carries it."""
    rows = _rows()
    for r in rows:
        r["privileged"] = {"reference": "zzteachersecretzz"}
    train, held, _ = build_tasks(rows, holdout=0.5, band=None)
    for t in train + held:
        assert "zzteachersecretzz" not in t["prompt"]
        assert t["info"]["privileged"]["reference"] == "zzteachersecretzz"


# The tests below drive an exported package through verifiers itself. A
# module-level importorskip would skip the export tests above too, and CI
# installs the dev extra only, so the gate is per test.
needs_verifiers = pytest.mark.skipif(
    importlib.util.find_spec("verifiers") is None,
    reason="verifiers not installed (uv sync --extra rl)",
)


def _fake_state(task: dict, spec: dict) -> dict:
    return {
        "info": task["info"],
        "prompt": [
            {"role": "system", "content": spec["system_prompt"]},
            {"role": "user", "content": task["prompt"]},
        ],
        "completion": [],
    }


@needs_verifiers
def test_load_environment_drives_a_rollout_through_the_mock_world(tmp_path):
    out = tmp_path / "refund-agent"
    wai.export_environment(
        _rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5, reward=outcome_reward
    )
    env = wai.load_environment(out / "refund_agent" / "spec.json")
    assert [t.name for t in env.tool_defs] == ["lookup_order", "create_refund"]
    assert len(env.dataset) >= 1
    spec = json.loads((out / "refund_agent" / "spec.json").read_text())
    task = json.loads((out / "refund_agent" / "data" / "train.jsonl").read_text().splitlines()[0])
    state = asyncio.run(env.setup_state(_fake_state(task, spec)))
    assert state["zp_steps"] == [] and state["zp_world"] is not None

    args = env.update_tool_args("lookup_order", {"order_id": "ORD-9"}, [], state)
    msg = asyncio.run(env.call_tool("lookup_order", args, "call_1"))
    assert msg.role == "tool" and msg.tool_call_id == "call_1"
    result = json.loads(msg.content)
    assert "status" in result
    assert state["zp_steps"][0]["tool"] == "lookup_order"
    assert state["zp_steps"][0]["arguments"] == {"order_id": "ORD-9"}

    state["completion"] = [{"role": "assistant", "content": "Looked it up; refunded."}]
    assert env.reward_func(state) == 1.0
    assert env.n_calls(state) == 1.0 and env.judge_ok(state) == 1.0
    rubrics = getattr(env.rubric, "rubrics", None) or [env.rubric]
    rubric_funcs = {f.__name__ for r in rubrics for f in getattr(r, "funcs", [])}
    assert {"reward_func", "n_calls", "judge_ok", "lookup_order_calls"} <= rubric_funcs


@needs_verifiers
def test_load_environment_world_is_seeded_per_task(tmp_path):
    out = tmp_path / "env"
    wai.export_environment(_rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5)
    env = wai.load_environment(out / "env" / "spec.json")
    spec = json.loads((out / "env" / "spec.json").read_text())
    assert env.reward.__name__ == "task_checklist"
    faulty = next(
        json.loads(line)
        for line in (out / "env" / "data" / "train.jsonl").read_text().splitlines()
        + (out / "env" / "data" / "holdout.jsonl").read_text().splitlines()
        if json.loads(line)["info"]["faults"]
    )
    s1 = asyncio.run(env.setup_state(_fake_state(faulty, spec)))
    s2 = asyncio.run(env.setup_state(_fake_state(faulty, spec)))
    a1 = env.update_tool_args("lookup_order", {"order_id": "ORD-3"}, [], s1)
    a2 = env.update_tool_args("lookup_order", {"order_id": "ORD-3"}, [], s2)
    r1 = json.loads(asyncio.run(env.call_tool("lookup_order", a1, "c")).content)
    r2 = json.loads(asyncio.run(env.call_tool("lookup_order", a2, "c")).content)
    assert r1 == r2 == {"status": "timeout", "error": "request timed out"}


@needs_verifiers
def test_load_environment_with_a_live_world(tmp_path):
    calls: list[tuple[str, dict]] = []

    def world(tool: str, arguments: dict) -> dict:
        calls.append((tool, arguments))
        return {"status": "ok", "order": arguments.get("order_id")}

    out = tmp_path / "env"
    wai.export_environment(_rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5)
    env = wai.load_environment(out / "env" / "spec.json", execute=world, reward=outcome_reward)
    spec = json.loads((out / "env" / "spec.json").read_text())
    task = json.loads((out / "env" / "data" / "train.jsonl").read_text().splitlines()[0])
    state = asyncio.run(env.setup_state(_fake_state(task, spec)))
    assert "zp_world" not in state
    args = env.update_tool_args("lookup_order", {"order_id": "ORD-1"}, [], state)
    msg = asyncio.run(env.call_tool("lookup_order", args, "c1"))
    assert json.loads(msg.content) == {"status": "ok", "order": "ORD-1"}
    assert calls == [("lookup_order", {"order_id": "ORD-1"})]


@needs_verifiers
def test_truncated_rollouts_score_zero_and_trace_monitor_runs(tmp_path):
    out = tmp_path / "env"
    wai.export_environment(
        _rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5, reward=outcome_reward
    )
    env = wai.load_environment(out / "env" / "spec.json")
    spec = json.loads((out / "env" / "spec.json").read_text())
    task = json.loads((out / "env" / "data" / "train.jsonl").read_text().splitlines()[0])
    state = asyncio.run(env.setup_state(_fake_state(task, spec)))
    args = env.update_tool_args("lookup_order", {"order_id": "ORD-9"}, [], state)
    asyncio.run(env.call_tool("lookup_order", args, "c1"))
    state["completion"] = [{"role": "assistant", "content": "Looked it up."}]
    assert env.reward_func(state) == 1.0 and env.truncated(state) == 0.0
    state["stop_condition"] = "max_turns_reached"
    assert env.reward_func(state) == 0.0 and env.truncated(state) == 1.0
    assert env.trace_clean(state) in (0.0, 1.0)
    rubrics = getattr(env.rubric, "rubrics", None) or [env.rubric]
    names = {f.__name__ for r in rubrics for f in getattr(r, "funcs", [])}
    assert {"truncated", "trace_clean"} <= names


def test_build_tasks_counts_mixed_groups():
    _, _, report = build_tasks(_rows(), holdout=0.5, band=None)
    assert report["graded_mixed"] == 2  # ORD-3 and ORD-4 were both solved and failed


# --------------------------------------------------------------------------
# harnesses=: the trainer rolls out under several harnesses (#712, step 4;
# Kim et al. 2026, arXiv:2606.25447)
# --------------------------------------------------------------------------

EAGER = "Refund first, ask questions later."


def _harnesses() -> list:
    careful = Harness(
        instructions=POLICY, tools=TOOLS, label="careful", disclosure=Disclosure(max_turns=4)
    )
    eager = {"label": "eager", "instructions": EAGER, "tools": TOOLS[:1]}
    return [careful, eager]


def _many_rows(n: int = 40) -> list[dict]:
    return [
        {"prompt": f"please help with order ORD-{i} today", "reward": i % 2, "steps": []}
        for i in range(n)
    ]


def test_export_with_two_harnesses_writes_them_to_the_spec_and_the_readme(tmp_path):
    out = tmp_path / "env"
    careful, eager = _harnesses()
    report = wai.export_environment(
        _rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5, harnesses=[careful, eager]
    )
    spec = json.loads((out / "env" / "spec.json").read_text())
    assert spec["system_prompt"] == POLICY and len(spec["tools"]) == 2  # unchanged
    assert [h["label"] for h in spec["harnesses"]] == ["careful", "eager"]
    assert spec["harnesses"][0]["hash"] == careful.fingerprint
    assert spec["harnesses"][0]["instructions"] == POLICY
    assert [t["name"] for t in spec["harnesses"][0]["tools"]] == ["lookup_order", "create_refund"]
    assert spec["harnesses"][0]["disclosure"] == {"max_turns": 4}
    assert spec["harnesses"][1]["instructions"] == EAGER
    assert [t["name"] for t in spec["harnesses"][1]["tools"]] == ["lookup_order"]
    assert (
        spec["harnesses"][1]["hash"]
        == Harness(instructions=EAGER, tools=TOOLS[:1], label="eager").fingerprint
    )
    assert report["harnesses"] == [
        {"label": "careful", "hash": careful.fingerprint},
        {"label": "eager", "hash": spec["harnesses"][1]["hash"]},
    ]
    readme = (out / "README.md").read_text()
    assert "### Harnesses" in readme and "arXiv:2606.25447" in readme
    assert f"- **careful** (`{careful.fingerprint}`): 2 tools, turn cap 4" in readme
    assert (
        f"- **eager** (`{spec['harnesses'][1]['hash']}`): 1 tools, turn cap {spec['max_turns']}"
        in readme
    )


def test_export_refuses_two_harnesses_with_one_label(tmp_path):
    same = [
        Harness(instructions=POLICY, tools=TOOLS, label="v1"),
        {"label": "v1", "instructions": EAGER},
    ]
    with pytest.raises(ValueError, match="same label twice"):
        wai.export_environment(_rows(), tmp_path / "env", tools=TOOLS, holdout=0.5, harnesses=same)
    with pytest.raises(TypeError, match="harnesses\\[0\\]"):
        wai.export_environment(_rows(), tmp_path / "env2", tools=TOOLS, holdout=0.5, harnesses=[3])


def test_export_without_harnesses_leaves_the_spec_and_readme_as_they_were(tmp_path):
    out = tmp_path / "env"
    wai.export_environment(_rows(), out, tools=TOOLS, system_prompt=POLICY, holdout=0.5)
    spec = json.loads((out / "env" / "spec.json").read_text())
    assert "harnesses" not in spec
    assert set(spec) == {
        "name",
        "system_prompt",
        "tools",
        "max_turns",
        "reward",
        "execute",
        "sdk_version",
    }
    assert "### Harnesses" not in (out / "README.md").read_text()


def _harness_state(task: dict, spec: dict) -> dict:
    state = _fake_state(task, spec)
    state["example_id"] = task["example_id"]
    return state


@needs_verifiers
def test_load_environment_draws_a_harness_per_task_and_runs_under_it(tmp_path):
    out = tmp_path / "env"
    wai.export_environment(
        _many_rows(),
        out,
        tools=TOOLS,
        system_prompt=POLICY,
        holdout=0.2,
        band=None,
        reward=outcome_reward,
        harnesses=_harnesses(),
    )
    env = wai.load_environment(out / "env" / "spec.json")
    spec = json.loads((out / "env" / "spec.json").read_text())
    by_label = {h["label"]: h for h in spec["harnesses"]}
    tasks = [
        json.loads(line) for line in (out / "env" / "data" / "train.jsonl").read_text().splitlines()
    ]
    assert len(tasks) >= 20

    seen: dict[str, str] = {}
    for task in tasks:
        first = asyncio.run(env.setup_state(_harness_state(task, spec)))
        again = asyncio.run(env.setup_state(_harness_state(task, spec)))
        assert first["harness"] == again["harness"]  # the same task draws the same harness
        label = first["harness"]["label"]
        assert first["harness"] == {"label": label, "hash": by_label[label]["hash"]}
        # the rollout's system prompt and tools are that harness's
        assert first["prompt"][0] == {"role": "system", "content": by_label[label]["instructions"]}
        assert first["prompt"][1]["role"] == "user"
        assert [t.name for t in first["tool_defs"]] == [t["name"] for t in by_label[label]["tools"]]
        seen[task["example_id"]] = label
    assert set(seen.values()) == {"careful", "eager"}  # across many tasks both appear

    # a fresh environment from the same spec draws the same map: a re-run is a re-run
    env2 = wai.load_environment(out / "env" / "spec.json")
    for task in tasks:
        assert (
            asyncio.run(env2.setup_state(_harness_state(task, spec)))["harness"]["label"]
            == seen[task["example_id"]]
        )

    # the trace and the reward see which harness the rollout ran under
    task = tasks[0]
    state = asyncio.run(env.setup_state(_harness_state(task, spec)))
    args = env.update_tool_args("lookup_order", {"order_id": "ORD-1"}, [], state)
    asyncio.run(env.call_tool("lookup_order", args, "c1"))
    state["completion"] = [{"role": "assistant", "content": "Looked it up."}]
    assert _row_from_state(state, state["zp_info"])["harness"] == state["harness"]
    assert env.reward_func(state) == 1.0
    # the world answers every tool any harness carries, and the monitor counts them
    rubrics = getattr(env.rubric, "rubrics", None) or [env.rubric]
    names = {f.__name__ for r in rubrics for f in getattr(r, "funcs", [])}
    assert {"lookup_order_calls", "create_refund_calls"} <= names


@needs_verifiers
def test_load_environment_harness_mix_weights_and_message_objects(tmp_path):
    import verifiers as vf

    out = tmp_path / "env"
    wai.export_environment(
        _many_rows(), out, tools=TOOLS, holdout=0.2, band=None, harnesses=_harnesses()
    )
    spec = json.loads((out / "env" / "spec.json").read_text())
    tasks = [
        json.loads(line) for line in (out / "env" / "data" / "train.jsonl").read_text().splitlines()
    ]

    only_eager = wai.load_environment(out / "env" / "spec.json", harness_mix=[0, 1])
    labels = {
        asyncio.run(only_eager.setup_state(_harness_state(t, spec)))["harness"]["label"]
        for t in tasks
    }
    assert labels == {"eager"}

    other_seed = wai.load_environment(out / "env" / "spec.json", harness_seed=1)
    base = wai.load_environment(out / "env" / "spec.json")
    draws = [
        asyncio.run(base.setup_state(_harness_state(t, spec)))["harness"]["label"]
        == asyncio.run(other_seed.setup_state(_harness_state(t, spec)))["harness"]["label"]
        for t in tasks
    ]
    assert not all(draws)  # the seed is part of the draw

    with pytest.raises(ValueError, match="harness_mix="):
        wai.load_environment(out / "env" / "spec.json", harness_mix=[1])
    with pytest.raises(ValueError, match="harness_mix="):
        wai.load_environment(out / "env" / "spec.json", harness_mix="random")

    # verifiers hands setup_state message objects, not dicts; the harness's
    # instructions replace the system message in the same shape
    state = _harness_state(tasks[0], spec)
    state["prompt"] = [vf.SystemMessage(content=POLICY), vf.UserMessage(content=tasks[0]["prompt"])]
    state = asyncio.run(base.setup_state(state))
    expected = {h["label"]: h for h in spec["harnesses"]}[state["harness"]["label"]]["instructions"]
    assert isinstance(state["prompt"][0], vf.SystemMessage)
    assert state["prompt"][0].content == expected and state["prompt"][1].role == "user"


def test_load_environment_without_harnesses_draws_none():
    from whileai.simulations.environment import _draw_index, _harness_weights

    assert _harness_weights("uniform", 0) == []
    assert _harness_weights("uniform", 2) == [0.5, 0.5]
    assert _harness_weights([1, 3], 2) == [0.25, 0.75]
    assert _draw_index("t1", 0, [1.0, 0.0]) == 0 and _draw_index("t1", 0, [0.0, 1.0]) == 1
    assert _draw_index("t1", 0, [0.5, 0.5]) == _draw_index("t1", 0, [0.5, 0.5])


def test_export_says_when_no_group_is_mixed(tmp_path):
    """A package with ``graded_mixed`` 0 used to ship with ``warnings`` empty,
    while ``select_for_rl`` refuses the same rows (#684)."""
    ungraded = [
        {
            "prompt": f"ask {i}",
            "scenario_id": f"s{i}",
            "rollout_index": k,
            "steps": [],
            "final_text": "",
        }
        for i in range(4)
        for k in range(3)
    ]
    report = wai.export_environment(
        ungraded, tmp_path / "blind", tools=TOOLS, system_prompt=POLICY, holdout=0.0
    )
    assert report["train"] == 4 and report["graded_mixed"] == 0
    (line,) = [w for w in report["warnings"] if w.startswith("no_mixed_groups")]
    assert "no prompt has two graded rollouts" in line
    assert "no_mixed_groups" in (tmp_path / "blind" / "README.md").read_text()
    unanimous = [dict(r, reward=1) for r in ungraded]
    report = wai.export_environment(
        unanimous, tmp_path / "flat", tools=TOOLS, system_prompt=POLICY, holdout=0.0, band=None
    )
    assert report["graded_prompts"] == 4 and report["graded_mixed"] == 0
    (line,) = [w for w in report["warnings"] if w.startswith("no_mixed_groups")]
    assert "every one of the 4 graded prompts" in line
    # a mixed group anywhere: nothing to say
    report = wai.export_environment(
        _rows(), tmp_path / "mixed", tools=TOOLS, system_prompt=POLICY, holdout=0.0, band=None
    )
    assert report["graded_mixed"] == 2
    assert not any(w.startswith("no_mixed_groups") for w in report["warnings"])


def test_an_export_with_no_holdout_says_so_and_does_not_advertise_one(tmp_path):
    """The band drops most prompts on a small run, so the first export a
    reader makes can land every surviving task in train. The package still
    installs; it just cannot be measured, and nothing said so until the
    trainer machine called load_environment(split="holdout") and it raised."""
    rows = [{"prompt": f"p{i}", "steps": [], "final_text": "done"} for i in range(3)]
    report = wai.export_environment(
        rows, tmp_path / "env", tools=TOOLS, system_prompt=POLICY, holdout=0.01
    )
    assert report["holdout"] == 0 and report["train"]
    warning = next(w for w in report["warnings"] if w.startswith("empty_holdout"))
    assert "cannot be measured" in warning and "decontamination" in warning
    readme = (tmp_path / "env" / "README.md").read_text()
    assert '"split": "train"' in readme and '"split": "holdout"' not in readme
    assert "not a held-out result" in readme
    # and the same export with a holdout keeps the holdout quickstart
    ok = wai.export_environment(
        _rows(), tmp_path / "ok", tools=TOOLS, system_prompt=POLICY, holdout=0.5, band=None
    )
    assert ok["holdout"] and not any(w.startswith("empty_holdout") for w in ok["warnings"])
    assert '"split": "holdout"' in (tmp_path / "ok" / "README.md").read_text()
    # holdout=0 is a choice, not an accident: no warning
    none_asked = wai.export_environment(
        rows, tmp_path / "none", tools=TOOLS, system_prompt=POLICY, holdout=0.0
    )
    assert not any(w.startswith("empty_holdout") for w in none_asked["warnings"])


def test_the_rl_extra_hint_names_the_python_range_it_is_marked_for():
    """`pip install 'whileai[rl]'` on an interpreter outside the marker
    resolves to nothing and exits 0. An error that repeats that line sends
    the reader round the loop again, so it names the range instead."""
    import re

    from whileai.simulations.defaults import RL_EXTRA_PYTHON_MAX, RL_EXTRA_PYTHON_MIN
    from whileai.simulations.environment import _rl_extra_install

    # the constants are the marker in pyproject.toml, so the two cannot drift
    # (read with a regex, not tomllib: the package supports Python 3.10)
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    marker = re.search(r'^rl = \["verifiers[^;]*;([^"]*)"\]', pyproject, re.M).group(1)
    lo = ".".join(str(p) for p in RL_EXTRA_PYTHON_MIN)
    hi = ".".join(str(p) for p in RL_EXTRA_PYTHON_MAX)
    assert f"python_version >= '{lo}'" in marker and f"python_version < '{hi}'" in marker

    inside = _rl_extra_install()
    assert "pip install 'whileai[rl]'" in inside
    with mock.patch.object(environment_mod.sys, "version_info", (3, 99, 0)):
        outside = _rl_extra_install()
    assert "Python 3.99" in outside and f"python_version < '{hi}'" in outside
    assert "resolves to nothing" in outside
