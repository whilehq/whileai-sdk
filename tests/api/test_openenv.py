"""export_environment(runtime="openenv"): the OpenEnv package and its runtime.

The export needs only the SDK. The tests that drive the environment skip
when ``openenv`` is not installed (``uv sync --extra openenv``); CI installs
the dev extra only, so the gate is per test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import whileai.simulations as wai
from whileai.simulations.openenv import SUBMIT_TOOL, read_spec, task_view

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
    plan = {
        "refund ORD-1 please": [1, 0, 1, 0],
        "refund ORD-2 please": [0, 1, 0, 1],
        "where is ORD-3": [1, 0, 1, 0],
        "cancel ORD-4 today": [1, 1, 0, 1],
        "hi, need help with ORD-5": [1, 0],
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
                    "faults": {},
                    "steps": [],
                    "final_text": "done",
                    "judge_status": "ok",
                    "privileged": {"reference": "zzteachersecretzz"},
                }
            )
    return rows


def outcome_reward(row: dict) -> dict:
    """Reward 1 when the agent looked the order up before answering."""
    called = any(s.get("tool") == "lookup_order" for s in row.get("steps") or [])
    return {"reward": 1 if called else 0, "reason": "looked up" if called else "no lookup"}


def _export(tmp_path: Path) -> tuple[Path, dict]:
    out = tmp_path / "refund_agent"
    report = wai.export_environment(
        _rows(),
        out,
        system_prompt=POLICY,
        tools=TOOLS,
        holdout=0.2,
        reward=outcome_reward,
        runtime="openenv",
    )
    return out, report


def test_openenv_export_writes_the_package(tmp_path: Path) -> None:
    out, report = _export(tmp_path)
    assert report["runtime"] == "openenv" and report["path"] == str(out)
    for rel in (
        "openenv.yaml",
        "pyproject.toml",
        "README.md",
        "__init__.py",
        "client.py",
        "spec.json",
        "data/train.jsonl",
        "data/holdout.jsonl",
        "server/__init__.py",
        "server/app.py",
        "server/refund_agent_environment.py",
        "server/Dockerfile",
        "server/requirements.txt",
    ):
        assert (out / rel).is_file(), rel
    manifest = (out / "openenv.yaml").read_text(encoding="utf-8")
    assert "name: refund_agent" in manifest
    assert "declared_tools: [lookup_order, create_refund, submit]" in manifest
    pyproject = (out / "pyproject.toml").read_text(encoding="utf-8")
    assert 'server = "refund_agent.server.app:main"' in pyproject
    assert '"openenv>=0.5"' in pyproject and '"whileai>=' in pyproject
    app = (out / "server" / "app.py").read_text(encoding="utf-8")
    assert "def main(" in app and '__name__ == "__main__"' in app
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\ntitle: RefundAgent Environment")
    assert "sdk: docker" in readme and "RefundAgentEnv" in readme
    spec, base = read_spec(out)
    assert base == out and spec["name"] == "refund_agent"
    assert spec["reward"].endswith(":outcome_reward")
    assert [t["name"] for t in spec["tools"]] == ["lookup_order", "create_refund"]
    train = [json.loads(line) for line in (out / "data" / "train.jsonl").read_text().splitlines()]
    assert len(train) == report["train"] == 4 and report["holdout"] == 1
    # the verifiers layout is not written: no package module beside the spec
    assert not (out / "refund_agent").exists()


def test_runtime_is_checked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runtime="):
        wai.export_environment(
            _rows(), tmp_path / "x", system_prompt=POLICY, tools=TOOLS, runtime="gym"
        )


def test_task_view_never_shows_info() -> None:
    task = {
        "example_id": "t1",
        "prompt": "refund ORD-1 please",
        "info": {
            "privileged": {"reference": "zzteachersecretzz"},
            "faults": {"lookup_order": {"mode": "timeout"}},
            "calibration": {"pass_rate": 0.5, "n": 4},
        },
    }
    view = task_view(task, "train", 0)
    assert view == {
        "id": "t1",
        "prompt": "refund ORD-1 please",
        "split": "train",
        "index": 0,
        "calibration": {"pass_rate": 0.5, "n": 4},
    }
    assert "zzteachersecretzz" not in json.dumps(view)


needs_openenv = pytest.mark.skipif(
    importlib.util.find_spec("openenv") is None,
    reason="openenv not installed (uv sync --extra openenv)",
)


@needs_openenv
def test_environment_runs_an_episode_and_grades_on_submit(tmp_path: Path) -> None:
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

    from whileai.simulations.openenv import environment_class

    out, _ = _export(tmp_path)
    env_cls = environment_class(out)
    assert env_cls.__name__ == "RefundAgentEnvironment"
    assert env_cls.SUPPORTS_CONCURRENT_SESSIONS is True
    env = env_cls()

    # task API: splits, counts, one task, never the answer key
    assert env.list_splits() == [
        {"name": "train", "type": "train"},
        {"name": "holdout", "type": "test"},
    ]
    assert env.num_tasks("train") == 4 and env.num_tasks("holdout") == 1
    assert "zzteachersecretzz" not in json.dumps(env.list_tasks("train"))
    with pytest.raises(IndexError):
        env.get_task("holdout", 5)

    first = env.reset(split="train", index=1)
    assert first.done is False and first.reward is None
    meta = first.metadata
    assert meta["task"] == env.get_task("train", 1)
    assert meta["messages"][0] == {"role": "system", "content": POLICY}
    assert meta["messages"][1]["role"] == "user"
    assert [t["function"]["name"] for t in meta["tools"]] == ["lookup_order", "create_refund"]
    assert meta["submit"] == SUBMIT_TOOL
    assert "zzteachersecretzz" not in json.dumps(meta)
    assert env.state.split == "train" and env.state.index == 1 and env.state.n_calls == 0

    listed = env.step(ListToolsAction())
    assert [t.name for t in listed.tools] == ["lookup_order", "create_refund", SUBMIT_TOOL]

    step = env.step(CallToolAction(tool_name="lookup_order", arguments={"order_id": "ORD-3"}))
    assert step.done is False and step.reward is None and step.error is None
    assert step.result["status"] == "ok"
    assert env.state.n_calls == 1 and env.state.step_count == 1

    final = env.step(CallToolAction(tool_name=SUBMIT_TOOL, arguments={"answer": "It shipped."}))
    assert final.done is True and final.reward == 1.0
    assert final.result["reason"] == "looked up" and final.result["judge_status"] == "ok"
    assert final.metadata["n_calls"] == 1 and final.metadata["trace_clean"] is True
    assert env.state.done is True

    after = env.step(CallToolAction(tool_name="lookup_order", arguments={"order_id": "ORD-3"}))
    assert after.error is not None and after.done is True

    # no lookup, no reward
    env.reset(seed=3)
    assert env.step(CallToolAction(tool_name=SUBMIT_TOOL, arguments={"answer": "no"})).reward == 0.0


@needs_openenv
def test_turn_cap_truncates_with_reward_zero(tmp_path: Path) -> None:
    from openenv.core.env_server.mcp_types import CallToolAction

    from whileai.simulations.openenv import environment_class

    out, _ = _export(tmp_path)
    env = environment_class(out)()
    env.reset(split="holdout", index=0)
    obs = None
    for _ in range(env.max_turns):
        obs = env.step(CallToolAction(tool_name="lookup_order", arguments={"order_id": "ORD-9"}))
    assert obs is not None and obs.done is True and obs.reward == 0.0
    assert obs.metadata["truncated"] is True and obs.metadata["n_calls"] == env.max_turns


@needs_openenv
def test_execute_override_and_reset_cursor(tmp_path: Path) -> None:
    from openenv.core.env_server.mcp_types import CallToolAction

    from whileai.simulations.openenv import environment_class

    out, _ = _export(tmp_path)
    seen: list[tuple[str, dict]] = []

    def execute(tool: str, arguments: dict) -> dict:
        seen.append((tool, arguments))
        return {"status": "ok", "echo": arguments}

    env = environment_class(out, execute=execute)()
    ids = [env.reset().metadata["task"]["id"] for _ in range(4)]
    assert len(set(ids)) == 4  # no split, no index, no seed: the next task in order
    env.step(
        CallToolAction(tool_name="create_refund", arguments={"order_id": "ORD-2", "amount": 3})
    )
    assert seen == [("create_refund", {"order_id": "ORD-2", "amount": 3})]


@needs_openenv
def test_make_app_serves_health_and_the_task_api(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from whileai.simulations.openenv import make_app

    out, _ = _export(tmp_path)
    app = make_app(out, max_concurrent_envs=2)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "healthy"
        meta = client.get("/metadata").json()
        assert meta["name"] == "refund_agent" and meta["description"] == POLICY
        splits = client.get("/refund_agent/splits").json()
        assert [s["name"] for s in splits] == ["train", "holdout"]
