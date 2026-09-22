"""The OpenEnv runtime for an exported While environment.

OpenEnv (Meta PyTorch, ``pip install openenv``) is the Gymnasium-style
contract TRL, torchforge, SkyRL and Unsloth drive: an ``Environment`` with
``reset()``, ``step(action)`` and ``state``, served over HTTP or WebSocket
by ``create_app`` and packaged as a Docker image or a Hugging Face Space.
``export_environment(..., runtime="openenv")`` writes that package; this
module is the environment behind it, built once and tested once, the way
``environment.py`` holds the verifiers class.

The episode is the SDK rollout, one tool call per step:

* ``reset(split=, index=)`` picks a task (``seed=`` picks one at random,
  neither picks the next one in order), seeds the mock world from the
  task's fault plan and world state (or hands calls to the caller's
  ``execute=``), draws the task's harness when the spec carries any, and
  returns the chat messages and the tool schemas the policy prompts with.
* ``step(CallToolAction(tool_name, arguments))`` runs the tool in the
  world and returns its result; ``step(ListToolsAction())`` lists the
  tools. The turn cap ends the episode with reward 0 (DAPO's overlong
  filtering, arXiv:2503.14476: a truncated rollout earns no signal).
* ``step(CallToolAction("submit", {"answer": ...}))`` ends the episode:
  the reward grades the trajectory through the SDK judge contract, and
  the observation carries the verdict and the trace flags.

Task ``info`` (fault plan, world state, privileged reference) stays on
the server; the task API and the observations show a task's id and
prompt only, so a client cannot read the answer key.
"""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .defaults import ENV_HARNESS_MIX, ENV_HARNESS_SEED, ENV_MAX_TURNS_FALLBACK
from .environment import (
    _TASK_META,
    DEFAULT_REWARD,
    SPEC_FILE,
    _draw_index,
    _harness_weights,
    _sdk_version,
    resolve_ref,
)
from .score.judging import normalize_judge_result

__all__ = ["SUBMIT_TOOL", "environment_class", "make_app", "read_spec", "task_view"]

#: The tool that ends an episode: the policy's final answer, graded.
SUBMIT_TOOL = "submit"
SUBMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The final reply to the user, once the task is done.",
        }
    },
    "required": ["answer"],
}
#: The splits an export writes, in the order the task API lists them.
SPLITS = ("train", "holdout")
_SPLIT_TYPES = {"train": "train", "holdout": "test"}


def read_spec(spec: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    """The spec dict and the directory its ``data/`` sits in.

    ``spec`` is a path to ``spec.json``, the directory holding it, or the
    dict itself (then ``_dir`` names the directory, default ``.``).
    """
    if isinstance(spec, Mapping):
        spec_dict = dict(spec)
        return spec_dict, Path(spec_dict.get("_dir") or ".")
    path = Path(spec)
    if path.is_dir():
        path = path / SPEC_FILE
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def read_tasks(base: Path, split: str) -> list[dict[str, Any]]:
    """The tasks of one split, or none when the file is absent."""
    file = base / "data" / f"{split}.jsonl"
    if not file.exists():
        return []
    return [
        json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def task_view(task: Mapping[str, Any], split: str | None = None, index: int | None = None) -> dict:
    """What a client may see of a task: id, prompt, split, index and the
    calibration (solve rate when graded). Never ``info``."""
    info = task.get("info") or {}
    view: dict[str, Any] = {
        "id": task.get("example_id") or info.get("task_id"),
        "prompt": task.get("prompt"),
    }
    if split is not None:
        view["split"] = split
    if index is not None:
        view["index"] = index
    if info.get("calibration"):
        view["calibration"] = dict(info["calibration"])
    return view


def _row(
    task: Mapping[str, Any], steps: Sequence[dict], answer: str, harness: dict | None
) -> dict[str, Any]:
    """The SDK row shape the reward reads, from one episode."""
    info = task.get("info") or {}
    row: dict[str, Any] = {
        "prompt": str(task.get("prompt") or ""),
        "steps": list(steps),
        "final_text": answer,
        "scenario_id": info.get("scenario_id"),
        "world_state": info.get("world_state") or "",
        "faults": info.get("faults") or {},
    }
    if info.get("privileged"):
        row["privileged"] = info["privileged"]
    if harness:
        row["harness"] = dict(harness)
    for key in _TASK_META:
        if info.get(key) is not None:
            row[key] = info[key]
    return row


def _class_name(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in str(name).split("_") if part)


def _make_classes() -> tuple[type, type]:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
        ListToolsAction,
        ListToolsObservation,
        Tool,
        ToolError,
        ToolErrorType,
    )
    from openenv.core.env_server.types import EnvironmentMetadata, Observation, State

    from .score.trace import trace_flags
    from .world.sandbox import MockEnvironment, WorldOptions

    class WhileState(State):
        """Episode state: the OpenEnv fields plus which task runs and how far."""

        task_id: str | None = None
        split: str | None = None
        index: int | None = None
        n_calls: int = 0
        done: bool = False
        harness: dict[str, Any] | None = None

    class WhileEnvironment(Environment):
        """One SDK world per episode; the spec's tools; the reward on submit."""

        SUPPORTS_CONCURRENT_SESSIONS = True
        State = WhileState

        def __init__(
            self,
            spec: Mapping[str, Any],
            *,
            tasks: Mapping[str, Sequence[dict]],
            reward: Callable[[dict], Any],
            execute: Callable[[str, dict], Any] | None = None,
            world: WorldOptions | Mapping[str, Any] | None = None,
            harness_mix: Any = ENV_HARNESS_MIX,
            harness_seed: int = ENV_HARNESS_SEED,
        ) -> None:
            super().__init__()
            self.spec = dict(spec)
            self.tasks = {s: list(tasks.get(s) or []) for s in SPLITS}
            self.reward = reward
            self.execute = execute
            self.world = WorldOptions.coerce(world if world is not None else spec.get("world"))
            self.max_turns = int(spec.get("max_turns") or ENV_MAX_TURNS_FALLBACK)
            self.tool_defs: list[dict] = list(spec.get("tools") or [])
            self.harnesses: list[dict] = list(spec.get("harnesses") or [])
            self.harness_weights = _harness_weights(harness_mix, len(self.harnesses))
            self.harness_seed = int(harness_seed)
            # the world answers every tool any harness carries
            self._world_tool_defs = list(self.tool_defs)
            known = {t["name"] for t in self._world_tool_defs}
            for harness in self.harnesses:
                for tool in harness.get("tools") or []:
                    if tool["name"] not in known:
                        known.add(tool["name"])
                        self._world_tool_defs.append(tool)
            self._cursor = dict.fromkeys(SPLITS, 0)
            self._state = WhileState(episode_id=str(uuid.uuid4()), step_count=0)
            self._task: dict | None = None
            self._episode_tools: list[dict] = list(self.tool_defs)
            self._harness: dict | None = None
            self._world: MockEnvironment | None = None
            self._steps: list[dict] = []
            self._messages: list[dict] = []

        # -- task API (openenv TaskProvider) ---------------------------------

        def list_splits(self) -> list[dict[str, str]]:
            return [{"name": s, "type": _SPLIT_TYPES[s]} for s in SPLITS if self.tasks[s]]

        def list_tasks(self, split: str) -> list[dict]:
            return [task_view(t, split, i) for i, t in enumerate(self._split(split))]

        def num_tasks(self, split: str) -> int:
            return len(self._split(split))

        def get_task(self, split: str, index: int) -> dict:
            tasks = self._split(split)
            if not 0 <= int(index) < len(tasks):
                raise IndexError(f"{split!r} has {len(tasks)} tasks; index {index} is out of range")
            return task_view(tasks[int(index)], split, int(index))

        def get_task_range(
            self, split: str, start: int | None = None, stop: int | None = None
        ) -> list[dict]:
            return self.list_tasks(split)[slice(start, stop)]

        def _split(self, split: str) -> list[dict]:
            if split not in self.tasks:
                raise ValueError(f"unknown split {split!r}; one of {list(SPLITS)}")
            return self.tasks[split]

        # -- episode ---------------------------------------------------------

        def _pick(self, split: str | None, index: int | None, seed: int | None) -> tuple[str, int]:
            split = split or "train"
            tasks = self._split(split)
            if not tasks:
                raise ValueError(f"no tasks in split {split!r}")
            if index is not None:
                if not 0 <= int(index) < len(tasks):
                    raise IndexError(
                        f"{split!r} has {len(tasks)} tasks; index {index} is out of range"
                    )
                return split, int(index)
            if seed is not None:
                return split, random.Random(int(seed)).randrange(len(tasks))
            i = self._cursor[split] % len(tasks)
            self._cursor[split] = i + 1
            return split, i

        def _draw_harness(self, task: Mapping[str, Any]) -> dict | None:
            if not self.harnesses:
                return None
            info = task.get("info") or {}
            key = info.get("task_id") or task.get("example_id") or task.get("prompt")
            return self.harnesses[_draw_index(str(key), self.harness_seed, self.harness_weights)]

        def reset(
            self,
            seed: int | None = None,
            episode_id: str | None = None,
            split: str | None = None,
            index: int | None = None,
            **kwargs: Any,
        ) -> Observation:
            split, index = self._pick(split, index, seed)
            task = self._split(split)[index]
            info = dict(task.get("info") or {})
            self._task = task
            self._harness = self._draw_harness(task)
            instructions = str(self.spec.get("system_prompt") or "")
            self._episode_tools = list(self.tool_defs)
            harness_state = None
            if self._harness is not None:
                instructions = str(self._harness.get("instructions") or instructions)
                self._episode_tools = list(self._harness.get("tools") or self.tool_defs)
                harness_state = {"label": self._harness["label"], "hash": self._harness["hash"]}
            self._steps = []
            self._world = None
            if self.execute is None:
                world_seed = info.get("seed")
                self._world = MockEnvironment(
                    [{"type": "function", "function": t} for t in self._world_tool_defs],
                    seed=int(world_seed) if isinstance(world_seed, int) else 0,
                    faults=dict(info.get("faults") or {}),
                    world_state=str(info.get("world_state") or ""),
                    options=self.world,
                )
            self._messages = (
                [{"role": "system", "content": instructions}] if instructions else []
            ) + [{"role": "user", "content": str(task.get("prompt") or "")}]
            self._state = WhileState(
                episode_id=episode_id or str(uuid.uuid4()),
                step_count=0,
                task_id=str(task.get("example_id") or info.get("task_id") or index),
                split=split,
                index=index,
                n_calls=0,
                done=False,
                harness=harness_state,
            )
            self._reset_rubric()
            return Observation(
                done=False,
                reward=None,
                metadata={
                    "task": task_view(task, split, index),
                    "messages": list(self._messages),
                    "tools": [{"type": "function", "function": t} for t in self._episode_tools],
                    "submit": SUBMIT_TOOL,
                    "max_turns": self.max_turns,
                    "harness": harness_state,
                },
            )

        def _tools(self) -> list[Tool]:
            tools = [
                Tool(
                    name=t["name"],
                    description=str(t.get("description") or ""),
                    input_schema=t.get("parameters") or {"type": "object", "properties": {}},
                )
                for t in self._episode_tools
            ]
            tools.append(
                Tool(
                    name=SUBMIT_TOOL,
                    description="End the episode with the final reply; the reward grades it.",
                    input_schema=SUBMIT_SCHEMA,
                )
            )
            return tools

        def step(
            self,
            action: Any,
            timeout_s: float | None = None,
            **kwargs: Any,
        ) -> Observation:
            if isinstance(action, ListToolsAction):
                return ListToolsObservation(tools=self._tools())
            if not isinstance(action, CallToolAction):
                return CallToolObservation(
                    tool_name=str(getattr(action, "tool_name", "") or type(action).__name__),
                    error=ToolError(
                        error_type=ToolErrorType.INVALID_ARGS,
                        message="expected CallToolAction or ListToolsAction",
                    ),
                )
            if self._task is None:
                return CallToolObservation(
                    tool_name=action.tool_name,
                    error=ToolError(
                        error_type=ToolErrorType.EXECUTION_ERROR,
                        message="call reset() before step()",
                    ),
                )
            if self._state.done:
                return CallToolObservation(
                    tool_name=action.tool_name,
                    done=True,
                    error=ToolError(
                        error_type=ToolErrorType.EXECUTION_ERROR,
                        message="the episode is over; call reset()",
                    ),
                )
            if action.tool_name == SUBMIT_TOOL:
                return self._submit(action.arguments or {})
            return self._call(action.tool_name, dict(action.arguments or {}))

        def _call(self, tool: str, arguments: dict) -> Observation:
            if self.execute is not None:
                try:
                    result = self.execute(tool, arguments)
                except Exception as exc:
                    result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            else:
                assert self._world is not None
                result = self._world.call(tool, arguments)
            if not isinstance(result, dict):
                result = {"status": "ok", "result": result}
            self._steps.append({"tool": tool, "arguments": arguments, "result": result})
            self._state.step_count += 1
            self._state.n_calls += 1
            truncated = self._state.n_calls >= self.max_turns
            if truncated:
                self._state.done = True
            return CallToolObservation(
                tool_name=tool,
                result=result,
                done=truncated,
                reward=0.0 if truncated else None,
                metadata={"n_calls": self._state.n_calls, "truncated": truncated},
            )

        def _submit(self, arguments: Mapping[str, Any]) -> Observation:
            assert self._task is not None
            answer = arguments.get("answer")
            if answer is None:
                answer = arguments.get("final_text", "")
            if not isinstance(answer, str):
                answer = json.dumps(answer, default=str)
            row = _row(self._task, self._steps, answer, self._state.harness)
            try:
                verdict = normalize_judge_result(self.reward(row))
            except Exception as exc:
                verdict = {
                    "reward": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "judge_status": "error",
                    "judge_meta": {},
                }
            flags = trace_flags(row) or {}
            value = verdict.get("reward")
            reward = float(value) if isinstance(value, (int, float)) else 0.0
            meta = verdict.get("judge_meta")
            markers = meta.get("markers") if isinstance(meta, dict) else None
            self._state.step_count += 1
            self._state.done = True
            return CallToolObservation(
                tool_name=SUBMIT_TOOL,
                result={
                    "reward": reward,
                    "reason": verdict.get("reason"),
                    "judge_status": verdict.get("judge_status"),
                    "markers": markers,
                },
                reward=reward,
                done=True,
                metadata={
                    "n_calls": self._state.n_calls,
                    "truncated": False,
                    "judge_ok": verdict.get("judge_status") == "ok",
                    "trace_flags": flags,
                    "trace_clean": not any(str(k).startswith(("lie.", "hack.")) for k in flags),
                },
            )

        @property
        def state(self) -> WhileState:
            return self._state

        def get_metadata(self) -> EnvironmentMetadata:
            return EnvironmentMetadata(
                name=str(self.spec.get("name") or "while_env"),
                description=str(self.spec.get("system_prompt") or "")[:200]
                or "A While RL environment.",
                version=str(self.spec.get("sdk_version") or _sdk_version()),
            )

        def close(self) -> None:
            self._world = None
            self._task = None

    return WhileEnvironment, WhileState


def environment_class(
    spec: str | Path | Mapping[str, Any],
    *,
    reward: Any = None,
    execute: Any = None,
    world: Any = None,
    harness_mix: str | Sequence[float] = ENV_HARNESS_MIX,
    harness_seed: int = ENV_HARNESS_SEED,
) -> type[Any]:
    """The OpenEnv ``Environment`` for an exported spec, as a no-argument
    class ``create_app`` can call.

    ``reward`` and ``execute`` override the spec's references (a callable
    or ``'module:attr'``); ``world`` overrides the mock world's dials.
    The class carries its ``State`` type for ``create_app(state_cls=)``.
    """
    try:
        base, state_cls = _make_classes()
    except ImportError as exc:  # pragma: no cover - the extra is named for the user
        raise ImportError(
            "the OpenEnv runtime needs openenv: pip install 'whileai[openenv]'"
        ) from exc
    spec_dict, base_dir = read_spec(spec)
    tasks = {s: read_tasks(base_dir, s) for s in SPLITS}
    if not any(tasks.values()):
        raise ValueError(f"no tasks under {base_dir / 'data'}")
    reward_obj = resolve_ref(reward) if isinstance(reward, str) else reward
    if reward_obj is None:
        reward_obj = resolve_ref(str(spec_dict.get("reward") or DEFAULT_REWARD))
    execute_obj = resolve_ref(execute) if isinstance(execute, str) else execute
    if execute_obj is None and spec_dict.get("execute"):
        execute_obj = resolve_ref(str(spec_dict["execute"]))

    class Bound(base):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__(
                spec_dict,
                tasks=tasks,
                reward=reward_obj,
                execute=execute_obj,
                world=world,
                harness_mix=harness_mix,
                harness_seed=harness_seed,
            )

    name = f"{_class_name(str(spec_dict.get('name') or 'while'))}Environment"
    Bound.__name__ = Bound.__qualname__ = name
    Bound.State = state_cls
    return Bound


def make_app(
    spec: str | Path | Mapping[str, Any],
    *,
    env_name: str | None = None,
    max_concurrent_envs: int | None = None,
    **options: Any,
) -> Any:
    """The FastAPI app that serves the environment: ``uvicorn`` runs it,
    ``openenv build`` packages it, the OpenEnv clients drive it. ``options``
    go to ``environment_class``."""
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    env_cls = environment_class(spec, **options)
    spec_dict, _ = read_spec(spec)
    return create_app(
        env_cls,
        CallToolAction,
        CallToolObservation,
        env_name=env_name or str(spec_dict.get("name") or "while_env"),
        max_concurrent_envs=max_concurrent_envs,
        state_cls=env_cls.State,
    )


# --------------------------------------------------------------------------
# The package export_environment(runtime="openenv") writes
# --------------------------------------------------------------------------

_MANIFEST = """spec_version: 1
name: {name}
version: 0.1.0
type: space
runtime: fastapi
app: server.app:app
port: 8000
validation:
  reward:
    range: [0.0, 1.0]
    oracle_tolerance: 0.0
    floor_margin: 0.1
  resources:
    cpu: 1.0
    memory_mb: 1024
    disk_mb: 512
    episode_timeout_s: 120.0
  capabilities:
    verifier:
      kind: reward_channel
    declared_tools: [{tools}]
  types:
    tags: [whileai, agents, tool-use]
"""

_PYPROJECT = """[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-{dist}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.10"
dependencies = [
    "openenv>=0.5",
    "whileai>={sdk_version}",
    "fastapi>=0.115.0",
    "uvicorn>=0.24.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[project.scripts]
server = "{name}.server.app:main"

[tool.setuptools]
include-package-data = true
packages = ["{name}", "{name}.server"]
package-dir = {{ "{name}" = ".", "{name}.server" = "server" }}

[tool.setuptools.package-data]
{name} = ["spec.json", "openenv.yaml", "data/*.jsonl"]
"""

_INIT = '''"""{name}: a While RL environment on the OpenEnv runtime. See README.md."""

from .client import {cls}Env

__all__ = ["{cls}Env"]
'''

_CLIENT = '''"""Client for the {name} environment: an OpenEnv MCP tool client.

    from {name} import {cls}Env

    with {cls}Env(base_url="http://localhost:8000").sync() as env:
        first = env.reset(split="train", index=0)
        messages = first.observation.metadata["messages"]
        tools = first.observation.metadata["tools"]
        ...                                  # the policy picks a tool call
        result = env.call_tool("lookup_order", order_id="ORD-1")
        final = env.call_tool("submit", answer="Refunded.")   # reward in final["reward"]
"""

from openenv.core.mcp_client import MCPToolClient


class {cls}Env(MCPToolClient):
    """``reset(split=, index=)`` picks a task; ``call_tool`` runs one step;
    ``call_tool("submit", answer=...)`` ends the episode with the reward."""
'''

_SERVER_INIT = '''"""{name} server: the environment class and the FastAPI app."""

from .{name}_environment import {cls}Environment

__all__ = ["{cls}Environment"]
'''

_SERVER_ENV = '''"""The OpenEnv Environment for {name}, built by the While SDK from spec.json."""

from pathlib import Path

from whileai.simulations.openenv import environment_class

SPEC = Path(__file__).resolve().parents[1] / "spec.json"

{cls}Environment = environment_class(SPEC)
'''

_SERVER_APP = '''"""FastAPI app for the {name} environment.

    uv run --project . server            # or: uvicorn server.app:app --port 8000
    openenv build && docker run -p 8000:8000 openenv-{dist}:latest
"""

import os
from pathlib import Path

from whileai.simulations.openenv import make_app

SPEC = Path(__file__).resolve().parents[1] / "spec.json"

app = make_app(
    SPEC,
    env_name="{name}",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
'''

_REQUIREMENTS = """openenv>=0.5
whileai>={sdk_version}
fastapi>=0.115.0
uvicorn>=0.24.0
"""

_DOCKERFILE = """ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \\
    apt-get install -y --no-install-recommends git && \\
    rm -rf /var/lib/apt/lists/*

COPY . /app/env

WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \\
        curl -LsSf https://astral.sh/uv/install.sh | sh && \\
        mv /root/.local/bin/uv /usr/local/bin/uv && \\
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \\
    fi

RUN --mount=type=cache,target=/root/.cache/uv \\
    if [ -f uv.lock ]; then \\
        uv sync --frozen --no-install-project --no-editable; \\
    else \\
        uv sync --no-install-project --no-editable; \\
    fi

RUN --mount=type=cache,target=/root/.cache/uv \\
    if [ -f uv.lock ]; then \\
        uv sync --frozen --no-editable; \\
    else \\
        uv sync --no-editable; \\
    fi

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv

COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
"""


def _readme(name: str, spec: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    dist = name.replace("_", "-")
    cls = _class_name(name)
    tools = ", ".join("`" + t["name"] + "`" for t in spec.get("tools") or [])
    world = spec.get("execute") or "whileai mock world, seeded per task"
    lines = [
        "---",
        f"title: {cls} Environment",
        "emoji: \U0001f433",
        "colorFrom: green",
        "colorTo: gray",
        "sdk: docker",
        "pinned: false",
        "app_port: 8000",
        "base_path: /web",
        "tags:",
        "  - openenv",
        "  - whileai",
        "---",
        "",
        f"# {dist}",
        "",
        "A While RL environment on the OpenEnv runtime: the task set, the world",
        "that answers tool calls, and the reward that grades a finished",
        "trajectory. The trainer samples its own rollouts from the policy under",
        "training, so nothing here is off-policy.",
        "",
        "### Overview",
        f"- **Environment ID**: `{dist}`",
        f"- **Short description**: {str(spec.get('system_prompt') or '')[:160].strip() or 'tool-using agent'}",
        "- **Tags**: whileai, agents, tool-use",
        "",
        "### Datasets",
        "- **Splits**: `data/train.jsonl`, `data/holdout.jsonl` (one task per prompt, written by the While simulator)",
        f"- **Split sizes**: {report.get('train', 0)} train / {report.get('holdout', 0)} holdout",
        "",
        "### Task",
        "- **Type**: multi-turn tool use, one tool call per step",
        f"- **Tools**: {tools}, plus `{SUBMIT_TOOL}` to end the episode",
        f"- **Turn cap**: {spec.get('max_turns')} tool calls; a truncated episode earns 0",
        f"- **World**: `{world}`",
        f"- **Reward**: `{spec.get('reward')}` through the While judge contract, on `{SUBMIT_TOOL}`",
    ]
    if report.get("band"):
        lines.append(
            f"- **Difficulty band**: {report['band'][0]:.0%} to {report['band'][1]:.0%} solve rate; "
            f"{report.get('band_dropped', 0)} of {report.get('graded_prompts', 0)} graded prompts dropped"
        )
    decon = report.get("decontamination") or {}
    if decon:
        lines.append(
            f"- **Train vs holdout {decon.get('ngram')}-gram overlap**: "
            f"{decon.get('n_contaminated', 0)} tasks ({(decon.get('contamination_rate') or 0):.1%})"
        )
    if spec.get("harnesses"):
        lines += ["", "### Harnesses"]
        for h in spec["harnesses"]:
            lines.append(f"- **{h['label']}** (`{h['hash']}`): {len(h.get('tools') or [])} tools")
        lines.append("")
        lines.append(
            "Each task draws one from its id and a seed; the episode runs under that "
            "harness's instructions and tools, and `state.harness` says which."
        )
    lines += [
        "",
        "### Quickstart",
        "",
        "```bash",
        "uv run --project . server                # serves on :8000, web UI at /web",
        "openenv validate .                       # the OpenEnv quality bar",
        "openenv build && docker run -p 8000:8000 openenv-" + dist + ":latest",
        "openenv push                             # a Hugging Face Space",
        "```",
        "",
        "```python",
        f"from {name} import {cls}Env",
        "",
        f'with {cls}Env(base_url="http://localhost:8000").sync() as env:',
        '    first = env.reset(split="train", index=0)',
        '    messages = first.observation.metadata["messages"]   # system + user',
        '    tools = first.observation.metadata["tools"]         # OpenAI function schemas',
        "    # ... the policy picks a tool call from (messages, tools) ...",
        '    result = env.call_tool("<tool>", **arguments)       # one step in the world',
        f'    final = env.call_tool("{SUBMIT_TOOL}", answer="...")     # ends the episode',
        '    reward = final["reward"]',
        "```",
        "",
    ]
    for warning in report.get("warnings") or []:
        lines += [f"**Warning.** {warning}", ""]
    lines += [
        "`info` on every task carries its fault plan, world state, privileged",
        "reference and calibration. The world and the reward read it on the",
        "server; the task API and the observations never show it.",
        "",
    ]
    return "\n".join(lines)


def write_package(
    out_dir: Path,
    *,
    name: str,
    spec: Mapping[str, Any],
    train: Sequence[dict],
    held: Sequence[dict],
    report: Mapping[str, Any],
    description: str = "",
) -> None:
    """Write the OpenEnv package: manifest, server, client, spec and data."""
    from .environment import _write_jsonl

    cls = _class_name(name)
    dist = name.replace("_", "-")
    sdk_version = str(spec.get("sdk_version") or _sdk_version())
    fmt = {
        "name": name,
        "cls": cls,
        "dist": dist,
        "sdk_version": sdk_version,
        "description": (description or f"While RL environment: {name}").replace('"', "'"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    server = out_dir / "server"
    server.mkdir(exist_ok=True)
    tools = ", ".join([t["name"] for t in spec.get("tools") or []] + [SUBMIT_TOOL])
    files = {
        out_dir / "openenv.yaml": _MANIFEST.format(name=name, tools=tools),
        out_dir / "pyproject.toml": _PYPROJECT.format(**fmt),
        out_dir / "README.md": _readme(name, spec, report),
        out_dir / "__init__.py": _INIT.format(**fmt),
        out_dir / "client.py": _CLIENT.format(**fmt),
        out_dir / SPEC_FILE: json.dumps(dict(spec), indent=2, default=str),
        server / "__init__.py": _SERVER_INIT.format(**fmt),
        server / f"{name}_environment.py": _SERVER_ENV.format(**fmt),
        server / "app.py": _SERVER_APP.format(**fmt),
        server / "requirements.txt": _REQUIREMENTS.format(**fmt),
        server / "Dockerfile": _DOCKERFILE,
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    _write_jsonl(out_dir / "data" / "train.jsonl", list(train))
    _write_jsonl(out_dir / "data" / "holdout.jsonl", list(held))
