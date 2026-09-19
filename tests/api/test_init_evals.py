"""``whileai init-evals`` writes an eval harness that runs on the project it found.

A coding agent told to build evals for a bot spent ten minutes copying
the recipe by hand. This command writes the same four files, wired to the
bot's own tools, policy and callable, and the files have to work: they
lint, the judge test passes, and the run script goes end to end offline
with no key.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from whileai.cli import main
from whileai.init_evals import scan

REPO = Path(__file__).resolve().parents[2]

BOT = '''"""A tiny refund bot, in the shape a tester's project has it."""

from __future__ import annotations

import re
from typing import Any

SYSTEM = """You are the refund assistant for Northwind Outfitters.
- Look the order up before you decide anything.
- Refunds over $200 need a manager: say a manager will follow up.
- Never invent order details."""

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up an order by id. Orders on file: A1001, A1002, A1003.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Refund an order. Only after checking the policy.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["order_id", "amount"],
        },
    },
]

ORDERS = {
    "A1001": {"item": "Trail runners", "total": 129.0, "status": "delivered"},
    "A1002": {"item": "Espresso machine", "total": 449.0, "status": "delivered"},
    "A1003": {"item": "Wool socks", "total": 24.0, "status": "shipped"},
}


def _run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "lookup_order":
        order_id = str(arguments.get("order_id", "")).upper()
        return ORDERS.get(order_id) or {"error": f"no order {order_id}"}
    if name == "issue_refund":
        return {"ok": True, **arguments}
    return {"error": f"no tool {name}"}


def answer(message: str) -> str:
    ids = re.findall(r"\\b[A-Z]\\d{4}\\b", message.upper())
    if not ids:
        return "Happy to help. Which order id is this about?"
    order = _run_tool("lookup_order", {"order_id": ids[0]})
    if "error" in order:
        return f"I could not find order {ids[0]}."
    if "refund" not in message.lower():
        return f"Order {ids[0]} ({order['item']}) is {order['status']}."
    if order["total"] > 200:
        return f"Order {ids[0]} is over $200, so a manager will follow up."
    _run_tool("issue_refund", {"order_id": ids[0], "amount": order["total"]})
    return f"Done: ${order['total']:.2f} refunded for order {ids[0]}."
'''


def _offline_env(project: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "WHILEAI_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    env["WHILEAI_HOME"] = str(project / "whileai-home")
    env["PYTHONPATH"] = str(REPO)
    return env


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "bot.py").write_text(BOT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def generated(project, capsys):
    assert main(["init-evals"]) == 0
    return project, capsys.readouterr().out


def test_it_writes_four_files_and_says_what_it_wired_up(generated):
    project, out = generated
    for name in ("agent.py", "judge.py", "run.py", "test_judge.py", "README.md"):
        assert (project / "evals" / name).exists(), name
    assert "bot:answer" in out
    assert "bot:TOOLS" in out
    assert "bot:SYSTEM" in out
    assert "bot:_run_tool" in out
    # the ids in the tool descriptions end up in the seed asks
    assert "A1001" in (project / "evals" / "run.py").read_text(encoding="utf-8")


def test_the_wrapper_records_tool_calls_through_the_thread_local(generated, monkeypatch):
    project, _ = generated
    module = _load(project / "evals" / "agent.py", "generated_evals_agent")
    try:
        bot = sys.modules["bot"]
        monkeypatch.setattr(
            bot,
            "answer",
            lambda message: str(bot._run_tool("lookup_order", {"order_id": "A1001"})),
        )
        result = module.agent("hi")
        assert [step["tool"] for step in result["steps"]] == ["lookup_order"]
        assert result["steps"][0]["arguments"] == {"order_id": "A1001"}
        assert result["steps"][0]["result"]["item"] == "Trail runners"
        assert "Trail runners" in result["final_text"]
        # the Anthropic input_schema came out in OpenAI function shape
        assert module.TOOLS[0]["type"] == "function"
        assert module.TOOLS[0]["function"]["name"] == "lookup_order"
        assert module.TOOLS[0]["function"]["parameters"]["required"] == ["order_id"]
        assert "Northwind" in module.SYSTEM_PROMPT
    finally:
        sys.modules.pop("generated_evals_agent", None)
        sys.modules.pop("bot", None)


def test_the_generated_files_lint(generated):
    project, _ = generated
    probe = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("ruff is not importable here")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "E,W,F,I,UP,B,SIM,RUF",
            "--target-version",
            "py310",
            "--line-length",
            "100",
            str(project / "evals"),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_generated_judge_test_passes(generated):
    project, _ = generated
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "evals/test_judge.py", "-q", "-p", "no:cacheprovider"],
        cwd=project,
        env=_offline_env(project),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_run_script_goes_end_to_end_offline(generated):
    project, _ = generated
    done = subprocess.run(
        [sys.executable, "evals/run.py", "--offline", "--k", "1", "--gap"],
        cwd=project,
        env=_offline_env(project),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pass@1" in done.stdout
    assert "what these asks never reach" in done.stdout


def test_it_refuses_to_overwrite_without_force(generated, capsys):
    project, _ = generated
    (project / "evals" / "judge.py").write_text("# mine\n", encoding="utf-8")
    assert main(["init-evals"]) == 1
    assert "--force" in capsys.readouterr().err
    assert (project / "evals" / "judge.py").read_text(encoding="utf-8") == "# mine\n"

    assert main(["init-evals", "--force"]) == 0
    assert "MARKERS" in (project / "evals" / "judge.py").read_text(encoding="utf-8")


def test_the_scan_reads_other_spellings(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "service.py").write_text(
        "TOOL_DEFS = [{'type': 'function', 'function': {'name': 'ship_it'}}]\n"
        'POLICY = "Ship nothing twice."\n'
        "def respond(message):\n"
        "    return 'ok'\n"
        "def call_tool(name, arguments):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    found = scan(tmp_path, tmp_path / "evals")
    assert found.tools.ref == "pkg.service:TOOL_DEFS"
    assert found.system_prompt.ref == "pkg.service:POLICY"
    assert found.agent.ref == "pkg.service:respond"
    assert found.recorder.ref == "pkg.service:call_tool"


def test_flags_override_the_scan(project, capsys):
    (project / "other.py").write_text(
        "HANDLERS = 1\n\n\ndef talk(message: str) -> str:\n    return 'hi'\n", encoding="utf-8"
    )
    assert main(["init-evals", "--agent", "other:talk", "--dir", "checks"]) == 0
    out = capsys.readouterr().out
    assert "other:talk" in out
    assert "other.talk(message)" in (project / "checks" / "agent.py").read_text(encoding="utf-8")


def test_nothing_found_still_writes_files_marked_todo(tmp_path, monkeypatch, capsys):
    (tmp_path / "notes.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["init-evals"]) == 0
    out = capsys.readouterr().out
    assert "not found" in out
    assert "TODO" in out
    agent_py = (tmp_path / "evals" / "agent.py").read_text(encoding="utf-8")
    assert "TODO: import the module your agent lives in" in agent_py
    assert "TODO: your tool definitions" in agent_py
    assert "TODO: record the tool calls" in agent_py
    assert "TODO" in (tmp_path / "evals" / "run.py").read_text(encoding="utf-8")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_an_agent_with_defaulted_extra_parameters_is_found(tmp_path):
    src = "\n".join(
        [
            "TOOLS = [{'name': 'lookup', 'description': 'x', 'input_schema': {'type': 'object'}}]",
            "SYSTEM = 'Refund desk.'",
            "def _run_tool(name, args):",
            "    return {}",
            "def answer(user_message: str, history: list | None = None) -> str:",
            "    return 'ok'",
            "",
        ]
    )
    (tmp_path / "bot.py").write_text(src, encoding="utf-8")
    from whileai.init_evals import scan

    found = scan(tmp_path, tmp_path / "evals")
    assert found.agent is not None and found.agent.ref == "bot:answer"
    assert found.recorder is not None and found.recorder.ref == "bot:_run_tool"


def test_an_agent_with_a_second_required_parameter_is_not_the_agent(tmp_path):
    src = "\n".join(
        [
            "TOOLS = [{'name': 'lookup', 'description': 'x', 'parameters': {'type': 'object'}}]",
            "def answer(message, session):",
            "    return 'ok'",
            "",
        ]
    )
    (tmp_path / "bot.py").write_text(src, encoding="utf-8")
    from whileai.init_evals import scan

    assert scan(tmp_path, tmp_path / "evals").agent is None
