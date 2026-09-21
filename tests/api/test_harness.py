"""``wai.Harness``: the program around the model, run like an agent,
versioned like weights; ``wai.harness.attribute``: which lever moved the
score (#712)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import whileai as wai
from tests.api.test_platform import Fake
from tests.helpers import POLICY, TOOLS
from whileai.harness import Disclosure, Harness, attribute, parse_codex_stream, parse_pi_stream
from whileai.platform import Harness as PlatformHarness
from whileai.platform import track

SEEDS = [f"Please refund order A100{i}, it arrived broken." for i in range(1, 5)]


def _scripted(limit: float):
    """A refund bot that refunds anything under ``limit``."""

    def agent(message: str) -> dict:
        oid = next((w.strip(",.") for w in message.split() if w.startswith("A100")), "A1001")
        total = 100.0 * (int(oid[-1]) if oid[-1].isdigit() else 1)
        steps = [
            {"tool": "lookup_order", "arguments": {"order_id": oid}, "result": {"total": total}}
        ]
        if total < limit:
            steps.append(
                {"tool": "create_refund", "arguments": {"order_id": oid}, "result": {"ok": True}}
            )
            return {"steps": steps, "final_text": f"Refunded {oid}."}
        return {"steps": steps, "final_text": f"Cannot refund {oid}."}

    return agent


# ------------------------------------------------------------------ identity


def test_front_door_has_the_harness_and_keeps_selection_reachable():
    assert wai.Harness is Harness
    assert "Harness" in wai.__all__ and "Selection" not in wai.__all__
    assert wai.Selection.__name__ == "Selection"
    assert callable(wai.harness.attribute)


def test_fingerprint_is_the_disclosure_and_nothing_else():
    a = Harness("openai:gpt-4.1-mini", instructions=POLICY, tools=TOOLS)
    b = Harness(wai.OpenAI("gpt-4.1-mini"), instructions=POLICY, tools=TOOLS)
    assert a.fingerprint == b.fingerprint, "a backend object and its spec string are one harness"
    assert a.kind == "prompted" and a.model_name == "gpt-4.1-mini"
    assert a.version == f"h-{a.fingerprint}" and len(a.fingerprint) == 12
    assert (
        Harness("openai:gpt-4.1-mini", instructions=POLICY + " Be brief.", tools=TOOLS).fingerprint
        != a.fingerprint
    )
    assert (
        Harness("openai:gpt-4.1-mini", instructions=POLICY, tools=TOOLS[:1]).fingerprint
        != a.fingerprint
    )
    capped = Harness(
        "openai:gpt-4.1-mini", instructions=POLICY, tools=TOOLS, disclosure=Disclosure(max_turns=3)
    )
    assert capped.fingerprint != a.fingerprint, "the turn cap is part of the setup"
    assert (
        Harness("openai:gpt-4.1-mini", instructions=POLICY, tools=TOOLS, label="v2").version == "v2"
    )
    assert a.tool_names == sorted(t["function"]["name"] for t in TOOLS)
    assert a.stamp() == {
        "label": a.version,
        "hash": a.fingerprint,
        "model": "gpt-4.1-mini",
        "kind": "prompted",
    }
    assert "prompted" in repr(a) and "gpt-4.1-mini" in repr(a)


def test_pin_is_the_platform_record_with_the_same_hash():
    h = Harness(
        "anthropic:claude-haiku-4-5", instructions=POLICY, tools=TOOLS, label="careful@haiku"
    )
    pinned = h.pin()
    assert isinstance(pinned, PlatformHarness)
    assert pinned.fingerprint == h.fingerprint
    assert pinned.version == "careful@haiku" and pinned.tools == h.tool_names
    wire = pinned.wire()
    assert wire["hash"] == h.fingerprint and "disclosure" not in wire, "the wire does not change"
    # a platform Harness with no disclosure keeps the hash it always had
    plain = PlatformHarness(instructions=POLICY, tools=TOOLS, model="claude-haiku-4-5")
    assert plain.fingerprint != pinned.fingerprint
    assert (
        plain.fingerprint
        == PlatformHarness(instructions=POLICY, tools=TOOLS, model="claude-haiku-4-5").fingerprint
    )


def test_tracked_run_accepts_the_runnable_harness():
    fake = Fake()
    h = Harness(
        "anthropic:claude-haiku-4-5", instructions=POLICY, tools=TOOLS, label="careful@haiku"
    )
    tracked = track("refund-bot", model="claude-haiku-4-5", harness=h, transport=fake)
    assert isinstance(tracked.harness, PlatformHarness)
    run = tracked.run("careful@haiku", method="eval", targets=["refund_policy"], harness=h)
    run.finish()
    posted = next(b for m, p, b in fake.calls if p == "/runs")
    assert posted["harness"] == "careful@haiku"
    assert posted["record"]["provenance"]["pins"]["harness"] == h.fingerprint


# ------------------------------------------------------------------- running


def test_prompted_harness_hands_the_engine_its_setup():
    h = Harness(
        "openai:gpt-4.1-mini",
        instructions=POLICY,
        tools=TOOLS,
        disclosure=Disclosure(max_turns=4),
    )
    agent, tools, prompt, cap = h.into_simulate(None, None, None)
    assert agent == "openai:gpt-4.1-mini" and tools == TOOLS and prompt == POLICY and cap == 4
    # what the call names wins over the harness
    agent, tools, prompt, cap = h.into_simulate(TOOLS[:1], "Other.", 2)
    assert tools == TOOLS[:1] and prompt == "Other." and cap == 2
    with pytest.raises(ValueError, match="no model"):
        Harness(instructions=POLICY, tools=TOOLS)("hi")


def test_callable_harness_runs_through_simulate_and_stamps_every_row():
    h = Harness(
        agent=_scripted(250), instructions=POLICY, tools=TOOLS, label="eager", model="scripted"
    )
    assert h.kind == "callable"
    data = wai.simulate(
        h,
        seeds=SEEDS,
        situations=4,
        budget=8,
        simulator=False,
        mode="rl",
        repeats=2,
        repeat_policy="fixed",
        reproducible=True,
        seed=0,
        fault_rate=0.0,
        avg_turns=1,
    )
    rows = data.rows()
    assert rows, "the harness supplied tools and prompt; the run produced rows"
    for row in rows:
        assert row["harness"] == {
            "label": "eager",
            "hash": h.fingerprint,
            "model": "scripted",
            "kind": "callable",
        }
        assert row["steps"][0]["tool"] == "lookup_order"


def test_command_harness_runs_a_program_per_task(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "import json, sys\n"
        "prompt = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()\n"
        "print(json.dumps({'steps': [{'tool': 'echo', 'arguments': {'text': prompt}, 'result': 'ok'}], "
        "'final_text': prompt.upper()}))\n",
        encoding="utf-8",
    )
    by_argv = Harness.command([sys.executable, str(script), "{prompt}"], label="echo@argv")
    out = by_argv("refund A1001")
    assert out["final_text"] == "REFUND A1001" and out["steps"][0]["arguments"] == {
        "text": "refund A1001"
    }
    assert by_argv.kind == "command"
    by_stdin = Harness.command([sys.executable, str(script)], label="echo@stdin")
    assert by_stdin("hello")["final_text"] == "HELLO"
    assert by_argv.fingerprint != by_stdin.fingerprint, "the command line is part of the setup"
    with pytest.raises(ValueError, match="argv is empty"):
        Harness.command([])
    bad = Harness.command([sys.executable, "-c", "import sys; sys.exit(3)"])
    with pytest.raises(RuntimeError, match="exited 3"):
        bad("x")


class _Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_claude_code_preset_composes_the_cli_and_discloses_the_context(monkeypatch, tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# rules\n", encoding="utf-8")
    seen = {}

    def run(command, **kw):
        seen["command"] = list(command)
        seen.update(kw)
        return _Proc(json.dumps({"type": "result", "result": "done"}) + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    h = Harness.claude_code(
        "sonnet", cwd=tmp_path, max_turns=3, tools=["Read", "Bash"], instructions="Be terse."
    )
    out = h("fix the test")
    assert out == {"steps": [], "final_text": "done"}
    # the program is resolved on PATH (claude.cmd on Windows); the flags follow
    assert Path(seen["command"][0]).stem.lower() == "claude"
    assert seen["command"][1:3] == ["-p", "fix the test"]
    assert "--max-turns" in seen["command"] and "3" in seen["command"]
    assert seen["command"][seen["command"].index("--allowedTools") + 1] == "Read,Bash"
    assert seen["command"][seen["command"].index("--append-system-prompt") + 1] == "Be terse."
    assert seen["cwd"] == str(tmp_path)
    assert h.disclosure.context == ("CLAUDE.md",) and h.disclosure.max_turns == 3
    assert h.tool_names == ["Bash", "Read"] and h.model_name == "sonnet" and h.kind == "command"
    # the same preset in a directory with no context file is a different harness
    other = Harness.claude_code(
        "sonnet",
        cwd=tmp_path / "empty",
        max_turns=3,
        tools=["Read", "Bash"],
        instructions="Be terse.",
    )
    assert other.fingerprint != h.fingerprint


def test_codex_and_pi_presets_prepend_instructions_and_say_so(monkeypatch, tmp_path):
    (tmp_path / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    seen = {}

    def run(command, **kw):
        seen["command"] = list(command)
        return _Proc(
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}})
            + "\n"
        )

    monkeypatch.setattr(subprocess, "run", run)
    codex = Harness.codex(
        "gpt-5-codex", cwd=tmp_path, sandbox="read-only", instructions="Be terse."
    )
    assert codex("list files")["final_text"] == "ok"
    assert Path(seen["command"][0]).stem.lower() == "codex"
    assert seen["command"][1:4] == ["exec", "--json", "--skip-git-repo-check"]
    assert seen["command"][-1] == "Be terse.\n\nlist files"
    assert codex.disclosure.context == ("AGENTS.md",)
    assert (
        "instructions prepended" in (codex.disclosure.notes or "")
        and "sandbox read-only" in codex.disclosure.notes
    )
    pi = Harness.pi("claude-sonnet-4-5", provider="anthropic", cwd=tmp_path, extensions=False)
    assert pi.tool_names == ["bash", "edit", "read", "write"]
    assert pi.disclosure.subagents is False and "no extensions" in (pi.disclosure.notes or "")
    argv = pi.agent._harness_command
    assert argv[:4] == ["pi", "--mode", "json", "--no-session"] and "--no-extensions" in argv


def test_codex_stream_becomes_steps_and_final_text():
    events = [
        {"type": "thread.started", "thread_id": "t1"},
        {
            "type": "item.started",
            "item": {"id": "i1", "type": "command_execution", "command": "bash -lc ls"},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "i1",
                "type": "command_execution",
                "command": "bash -lc ls",
                "aggregated_output": "README.md\nsrc\n",
                "exit_code": 0,
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "i2",
                "type": "file_change",
                "changes": [{"path": "a.py", "kind": "update"}],
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "i3",
                "type": "mcp_tool_call",
                "server": "db",
                "tool": "query",
                "arguments": {"sql": "select 1"},
                "result": {"rows": 1},
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "i4", "type": "agent_message", "text": "Two entries."},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
    ]
    out = parse_codex_stream("".join(json.dumps(e) + "\n" for e in events))
    assert [s["tool"] for s in out["steps"]] == ["shell", "apply_patch", "db.query"]
    assert (
        out["steps"][0]["arguments"] == {"command": "bash -lc ls"}
        and out["steps"][0]["exit_code"] == 0
    )
    assert out["steps"][0]["result"] == "README.md\nsrc\n"
    assert out["final_text"] == "Two entries."
    long = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "cat big",
            "aggregated_output": "x" * 5000,
        },
    }
    cut = parse_codex_stream(json.dumps(long))["steps"][0]
    assert (
        cut["result_truncated"] is True
        and cut["result_chars"] == 5000
        and len(cut["result"]) == 2000
    )
    with pytest.raises(RuntimeError, match="codex reported an error"):
        parse_codex_stream(json.dumps({"type": "error", "message": "no auth"}))


def test_pi_stream_pairs_tool_calls_with_results():
    events = [
        {"type": "session", "version": "0.73", "cwd": "/repo"},
        {"type": "agent_start"},
        {
            "type": "tool_execution_start",
            "toolCallId": "c1",
            "toolName": "bash",
            "args": {"command": "ls"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "c1",
            "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "a\nb\n"}], "isError": False},
        },
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Two files."}]},
        },
        {"type": "agent_end", "messages": []},
    ]
    out = parse_pi_stream("".join(json.dumps(e) + "\n" for e in events))
    assert out["steps"] == [{"tool": "bash", "arguments": {"command": "ls"}, "result": "a\nb\n"}]
    assert out["final_text"] == "Two files."
    # without execution events, the assistant's toolCall blocks pair with turn_end results
    fallback = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "c2", "name": "read", "arguments": {"path": "x"}}
                ],
            },
        },
        {
            "type": "turn_end",
            "toolResults": [
                {"role": "tool", "toolCallId": "c2", "content": "text of x", "isError": True}
            ],
        },
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Read it."}]},
        },
    ]
    out = parse_pi_stream("".join(json.dumps(e) + "\n" for e in fallback))
    assert out["steps"] == [
        {"tool": "read", "arguments": {"path": "x"}, "result": "text of x", "is_error": True}
    ]
    assert out["final_text"] == "Read it."


# --------------------------------------------------------------- attribution


def _grid(cell_pass, tasks=30, models=("x", "y"), harnesses=("a", "b")):
    rows = []
    for h in harnesses:
        for m in models:
            for t in range(tasks):
                rows.append(
                    {
                        "task_id": f"t{t}",
                        "reward": cell_pass(h, m, t),
                        "harness": {"label": h, "model": m},
                    }
                )
    return rows


def test_attribute_names_the_harness_when_only_the_harness_moves_the_score():
    rows = _grid(lambda h, m, t: 1.0 if (t % 3 != 0) == (h == "a") else 0.0)
    report = attribute(rows)
    assert report["harnesses"] == ["a", "b"] and report["models"] == ["x", "y"]
    assert report["n_tasks"] == 30
    assert report["cells"]["a@x"] == 66.7 and report["cells"]["b@x"] == 33.3
    assert report["share_harness"] == 1.0 and report["share_model"] == 0.0
    assert report["verdict"] == "harness" and report["ranking_reversal"] is False
    assert report["ci_difference"][0] > 0
    text = str(report)
    assert "harness moved the score more than the model" in text
    assert "the same model leads under every harness" in text
    assert "spread explained: harness 100%" in text


def test_attribute_sees_a_ranking_reversal_and_an_unresolved_difference():
    # under harness a model x wins, under b model y wins: pure interaction
    rows = _grid(lambda h, m, t: 1.0 if ((h == "a") == (m == "x")) and t % 2 == 0 else 0.0)
    report = attribute(rows)
    assert report["ranking_reversal"] is True
    assert report["top_model_by_harness"] == {"a": "x", "b": "y"}
    assert report["share_interaction"] == 1.0 and report["verdict"] == "unresolved"
    assert "leading model changes with the harness" in str(report)


def test_attribute_refuses_a_grid_it_cannot_read():
    with pytest.raises(ValueError, match="no row carries a harness"):
        attribute([{"task_id": "t", "reward": 1.0}])
    one = _grid(lambda h, m, t: 1.0, harnesses=("a",))
    with pytest.raises(ValueError, match="at least 2 harnesses"):
        attribute(one)
    hole = [r for r in _grid(lambda h, m, t: 1.0) if r["harness"] != {"label": "b", "model": "y"}]
    with pytest.raises(ValueError, match=r"no rows for \['b@y'\]"):
        attribute(hole)


def test_attribute_without_enough_tasks_says_so_instead_of_an_interval():
    from whileai.simulations.score.stats import MIN_CI_TASKS

    rows = _grid(lambda h, m, t: 1.0 if h == "a" else 0.0, tasks=MIN_CI_TASKS - 1)
    report = attribute(rows)
    assert report["ci_share_harness"] is None and report["verdict"] == "unresolved"
    assert "no interval" in (report["note"] or "")
