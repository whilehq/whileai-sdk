"""What every candidate shares: the instructions, the ``run_python`` tool,
the loop that plays a harness over the tasks, the grader, and the offline
stand-in.

A candidate file changes the harness (instructions, the skills text it
carries, whether it gets the tool, the turn cap) and nothing here.
``build()`` turns it into a ``wai.Harness``. ``play()`` runs any harness
over the tasks through one ``chat`` function, so the same loop serves the
scripted stand-in (dry run), a vLLM engine in a Modal container (live), and
the trained adapter after GRPO. ``grade()`` is ``wai.verify.CodeExec``
against the hidden tests with the table builder prepended.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import whileai as wai
from whileai.harness import Disclosure
from whileai.simulations.verify import CodeExec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tasks as task_mod

# The instructions every candidate starts from. The baseline is these and
# nothing else: no skills text, no tool, one turn.
BASE_INSTRUCTIONS = (
    "You write Python functions over a table of daily price bars. Read the task, then reply "
    "with one ```python block that defines the requested function. Standard library only."
)

# TOOL_CHARS = 1500: the tool result is cut here so a print loop cannot
# fill the context; the step says when it was cut (convention, untested).
TOOL_CHARS = 1500
# TOOL_TIMEOUT_S = 10.0: the same wall clock CodeExec gives the tests.
TOOL_TIMEOUT_S = 10.0
# GRADE_WORKERS = 8: sandboxes graded in parallel; each is one interpreter
# start, so threads are enough (convention, untested).
GRADE_WORKERS = 8
# TURNS_WITH_TOOL = 3: the turn cap a tool harness gets: run, read, answer.
TURNS_WITH_TOOL = 3

Chat = Callable[[list[dict[str, Any]]], list[str]]


# --------------------------------------------------------------------------
# The tool: run the candidate's code on the seeded table, return what it printed
# --------------------------------------------------------------------------


def run_python(code: str) -> str:
    """Run a Python snippet with the price table defined as ``BARS`` and
    return what it printed (stdout, then stderr), cut at TOOL_CHARS."""
    # The table under both spellings: the instructions say BARS, and the
    # smoke run showed the model reaching for `bars`, the prompt's name.
    program = task_mod.SETUP + "\nbars = BARS\n" + code
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=TOOL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {TOOL_TIMEOUT_S:.0f}s"
    text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    text = text.strip() or f"(no output, exit {proc.returncode})"
    if len(text) > TOOL_CHARS:
        text = text[-TOOL_CHARS:] + f"\n[cut to the last {TOOL_CHARS} characters]"
    return text


TOOL_INSTRUCTIONS = (
    " You may test before you answer: send a ```python block that defines the function and "
    "prints its value on the table (in the sandbox the table is the global `bars`, so write "
    "`print(fn(bars, ...))`); the output comes back as the next message. Then reply with your "
    "final ```python block, which defines the function and prints nothing: a block with a "
    "print is run as a test, a block without one is your answer and is graded."
)


# --------------------------------------------------------------------------
# Building a harness from a candidate's pieces
# --------------------------------------------------------------------------


def instructions_for(skills: str | None, tool: bool) -> str:
    text = BASE_INSTRUCTIONS
    if tool:
        text += TOOL_INSTRUCTIONS
    if skills:
        text += "\n\n# SKILLS.md\n\n" + skills.strip()
    return text


def build(
    model: str,
    *,
    skills: str | None,
    tool: bool,
    label: str,
    disclosure: Disclosure | None = None,
    scripted_rate: float,
) -> wai.Harness:
    """The candidate as a ``wai.Harness``. ``model`` is ``base`` (the
    untrained weights), ``trained``, or a ``scripted*`` stand-in; the
    fingerprint hashes the instructions, the tool name and the disclosure,
    so a skills edit is a new version without anyone naming it."""
    if disclosure is None:
        disclosure = Disclosure(max_turns=TURNS_WITH_TOOL) if tool else Disclosure()
    text = instructions_for(skills, tool)
    agent = None
    if model.startswith("scripted"):
        agent = scripted(model, rate=scripted_rate)
    harness = wai.Harness(
        "base" if model.startswith("scripted") else model,
        instructions=text,
        tools=[run_python] if tool else None,
        agent=agent,
        label=label,
        disclosure=disclosure,
    )
    harness.scripted_rate = scripted_rate  # type: ignore[attr-defined]
    return harness


def spec_of(harness: wai.Harness) -> dict[str, Any]:
    """The harness as plain data a container can rebuild the loop from."""
    return {
        "label": harness.version,
        "hash": harness.fingerprint,
        "model": harness.model_name,
        "kind": harness.kind,
        "instructions": harness.instructions or "",
        "tool": "run_python" in harness.tool_names,
        "max_turns": harness.disclosure.max_turns or 1,
    }


# --------------------------------------------------------------------------
# The loop: one harness over the tasks, k rollouts each, through one chat
# --------------------------------------------------------------------------


def _fence(text: str) -> str | None:
    from whileai.simulations.verify.code import _CODE_FENCE

    blocks = _CODE_FENCE.findall(text or "")
    return blocks[-1].strip() if blocks else None


def play(
    chat: Chat,
    spec: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    k: int,
    seed: int,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Every task ``k`` times under one harness. ``chat`` takes a batch of
    conversations (``{task, rollout_index, turn, messages}``) and returns
    one reply per conversation. A harness with the tool gets up to
    ``max_turns`` assistant turns: a reply with a ```python block is run
    with ``run_python`` and the output goes back as the next message; the
    last reply is ``final_text``. Rows carry ``steps`` (one per tool
    call), ``privileged`` (the tests) and the harness stamp."""
    model = model or spec["model"]
    stamp = {"label": spec["label"], "hash": spec["hash"], "model": model, "kind": spec["kind"]}
    active = [
        {
            "task": t,
            "rollout_index": i,
            "turn": 0,
            "seed": seed,
            "tool": bool(spec.get("tool")),
            "messages": [
                {"role": "system", "content": spec["instructions"]},
                {"role": "user", "content": t["prompt"]},
            ],
            "steps": [],
        }
        for t in tasks
        for i in range(k)
    ]
    rows: list[dict[str, Any]] = []
    max_turns = int(spec.get("max_turns") or 1)
    while active:
        replies = chat(active)
        still: list[dict[str, Any]] = []
        to_run: list[tuple[dict[str, Any], str]] = []
        for convo, reply in zip(active, replies):
            convo["messages"].append({"role": "assistant", "content": reply})
            convo["turn"] += 1
            code = _fence(reply) if spec.get("tool") else None
            # A block that prints is a test and goes to the tool; a block
            # without a print is the answer, whatever turns are left.
            if code and "print(" in code and convo["turn"] < max_turns:
                to_run.append((convo, code))
                still.append(convo)
                continue
            t = convo["task"]
            rows.append(
                {
                    "prompt": t["prompt"],
                    "final_text": reply,
                    "steps": convo["steps"],
                    "scenario_id": t["scenario_id"],
                    "rollout_index": convo["rollout_index"],
                    "family": t["family"],
                    "privileged": dict(t["privileged"]),
                    "harness": dict(stamp),
                    "turns": convo["turn"],
                }
            )
        if to_run:
            with ThreadPoolExecutor(max_workers=GRADE_WORKERS) as pool:
                results = list(pool.map(lambda pair: run_python(pair[1]), to_run))
            for (convo, code), result in zip(to_run, results):
                convo["steps"].append(
                    {"tool": "run_python", "arguments": {"code": code}, "result": result}
                )
                convo["messages"].append(
                    {"role": "user", "content": f"run_python output:\n{result}"}
                )
        active = still
    rows.sort(key=lambda r: (r["scenario_id"], r["rollout_index"]))
    return rows


def grade(rows: list[dict[str, Any]], *, workers: int = GRADE_WORKERS) -> list[dict[str, Any]]:
    """``CodeExec`` on every row: the last ```python block plus the hidden
    tests, with the table builder prepended. Sets ``reward`` and ``reason``."""
    verifier = CodeExec(setup=task_mod.SETUP, timeout=TOOL_TIMEOUT_S)

    def one(row: dict[str, Any]) -> dict[str, Any]:
        out = verifier(row)
        row["reward"] = out.get("reward")
        row["reason"] = out.get("reason")
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, rows))


# --------------------------------------------------------------------------
# The offline stand-in: a seeded agent that solves a task at a planted rate
# --------------------------------------------------------------------------


def _stub(task: dict[str, Any]) -> str:
    names = ", ".join(["bars", *task["params"]])
    return f"```python\ndef {task['function']}({names}):\n    return 0.0\n```"


def scripted(model: str, *, rate: float) -> Chat:
    """A ``chat`` that answers with the reference implementation at
    ``rate`` and a stub otherwise. Which rollouts it gets right is drawn
    from the model name, the task id and the rollout index, so the same
    candidate gives the same rows every run and two stand-ins differ the
    way two models would: in where they fail. Under a tool harness it sends
    its code once to be run, then sends it again as the answer."""

    def chat(convos: list[dict[str, Any]]) -> list[str]:
        out = []
        for c in convos:
            key = f"{model}:{c['seed']}:{c['task']['scenario_id']}:{c['rollout_index']}"
            draw = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
            code = task_mod.reference_source(c["task"]) if draw < rate else _stub(c["task"])
            if c["turn"] == 0 and c.get("tool"):
                fn = c["task"]["function"]
                args = ", ".join(["bars", *(repr(v) for v in c["task"]["params"].values())])
                code = code.rstrip("`").rstrip() + f"\nprint({fn}({args}))\n```"
            out.append(code)
        return out

    chat.__name__ = f"scripted[{model}]"
    return chat


def shuffle_seed(*parts: Any) -> int:
    """One integer from any parts: a per-request sampling seed."""
    key = ":".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


__all__ = [
    "BASE_INSTRUCTIONS",
    "TURNS_WITH_TOOL",
    "build",
    "grade",
    "instructions_for",
    "play",
    "run_python",
    "scripted",
    "shuffle_seed",
    "spec_of",
]
