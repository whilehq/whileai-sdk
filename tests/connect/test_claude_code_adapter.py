"""claude_code wraps the Claude Code CLI as an agent: one subprocess per rollout."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import whileai.simulations as wai
from whileai.simulations.generate.agents import USER_TURN_MARK

TOOL_CALL = {
    "type": "assistant",
    "message": {
        "content": [
            {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "README.md"}}
        ]
    },
}
TOOL_RESULT = {
    "type": "user",
    "message": {"content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "# title"}]},
}
FINAL = {"type": "result", "result": "The README starts with a title."}


def _stream(*events):
    return "".join(json.dumps(e) + "\n" for e in events)


class Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _capture_run(monkeypatch, proc):
    seen = {}

    def run(command, **kw):
        seen["command"] = list(command)
        seen.update(kw)
        return proc

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_prompt_and_flags_reach_the_cli(monkeypatch):
    seen = _capture_run(monkeypatch, Proc(_stream(FINAL)))
    agent = wai.claude_code(["--model", "sonnet"], cwd="/repo", max_turns=3, timeout=7)
    out = agent("summarize the README")
    # the program is looked up on PATH first (claude.cmd on Windows, #712)
    assert Path(seen["command"][0]).stem.lower() == "claude"
    assert seen["command"][1:3] == ["-p", "summarize the README"]
    assert seen["command"][3:6] == ["--output-format", "stream-json", "--verbose"]
    assert seen["command"][-4:] == ["--max-turns", "3", "--model", "sonnet"]
    assert seen["cwd"] == "/repo" and seen["timeout"] == 7
    assert seen["capture_output"] is True and seen["text"] is True
    assert out == {"steps": [], "final_text": "The README starts with a title."}
    assert agent.__name__ == "claude_code"


def test_no_max_turns_means_no_flag(monkeypatch):
    seen = _capture_run(monkeypatch, Proc(_stream(FINAL)))
    wai.claude_code()("hi")
    assert "--max-turns" not in seen["command"]


def test_tool_use_and_results_pair_into_steps(monkeypatch):
    _capture_run(monkeypatch, Proc(_stream(TOOL_CALL, TOOL_RESULT, FINAL)))
    out = wai.claude_code()("read it")
    assert out["steps"] == [
        {"tool": "Read", "arguments": {"file_path": "README.md"}, "result": "# title"}
    ]
    assert out["final_text"] == "The README starts with a title."


def test_multi_turn_situations_are_joined_into_one_prompt(monkeypatch):
    seen = _capture_run(monkeypatch, Proc(_stream(FINAL)))
    wai.claude_code()(f"first ask{USER_TURN_MARK}second ask")
    assert seen["command"][2] == "first ask\n\nsecond ask"


def test_a_crash_with_no_output_raises_with_the_stderr(monkeypatch):
    _capture_run(monkeypatch, Proc("", returncode=1, stderr="not logged in"))
    with pytest.raises(RuntimeError, match="claude exited 1: not logged in"):
        wai.claude_code()("hi")


def test_a_nonzero_exit_that_still_wrote_a_result_is_parsed(monkeypatch):
    _capture_run(monkeypatch, Proc(_stream(TOOL_CALL, FINAL), returncode=1))
    out = wai.claude_code()("hi")
    assert out["final_text"] == FINAL["result"]
    # a call the CLI never answered is kept as a step with an empty result
    assert out["steps"] == [{"tool": "Read", "arguments": {"file_path": "README.md"}, "result": ""}]
