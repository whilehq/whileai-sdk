"""The harness: the program around the model, as one object you can run,
fingerprint and compare.

A model never meets a task alone. Something builds its context, hands it
tools, decides when to stop, retries, compacts, delegates. That program
is the harness, and on the agents most teams run (a closed model behind
Claude Code, Codex, pi, or their own prompt-and-tools loop) it is the
only part they can change. Zhang et al. 2026 (arXiv:2605.23950) measured
that among comparable frontier models the harness explains more of the
score spread than the model does, and can reverse which model ranks
first; Lee et al. 2026 (Meta-Harness, arXiv:2603.28052) beat hand-built
harnesses on TerminalBench-2 by searching over harness code with the
scores and traces of earlier candidates in view; Kim et al. 2026
(arXiv:2606.25447) found a policy trained under one fixed harness
collapses when the tool environment shifts. So a harness is versioned
like weights, run like an agent, and compared like a training arm::

    import whileai as wai

    careful = wai.Harness(wai.OpenAI("gpt-4.1-mini"), instructions=POLICY, tools=TOOLS)
    coder = wai.Harness.claude_code("sonnet", cwd="repo/", max_turns=8)

    data = wai.simulate(careful, mode="rl", repeats=4)      # rows carry the harness
    scored = data.grade(judge)
    print(wai.harness.attribute(rows))   # which lever moved the score: harness or model

Three ways to make one:

* ``Harness(model, instructions=, tools=)``: the prompted loop the SDK
  plays itself, tools answered by the mock world or your ``execute=``.
* ``Harness.command([...])`` and the presets ``Harness.claude_code``,
  ``Harness.codex``, ``Harness.pi``: a coding agent driven through its
  JSON event stream, one subprocess per task, the trace normalized to the
  same ``{steps, final_text}`` every other adapter returns.
* ``Harness(agent=callable)``: any ``message -> trajectory`` you already
  have, given a fingerprint.

The fingerprint hashes the disclosure fields Zhang et al. ask a comparison
to state (context construction, tool interaction, orchestration,
verification): model, instructions, tool names, context files, turn cap,
compaction, retries, subagents, sampling. Two harnesses with one
fingerprint were measured under the same setup; a prompt edit is a new
version without anyone naming it (Lambert 2025, chapter Evaluation).

``attribute(rows)`` reads rows from a harness x model grid on one task set
and says which lever moved the score: the two-way decomposition of the
cell means into a harness part, a model part and their interaction, with
bootstrap intervals over tasks, and whether the model ranking flips
across harnesses. That is the variance decomposition protocol of Zhang et
al. 2026, run on your own tasks.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .report import Report

# COMMAND_TIMEOUT_S = 300.0: seconds one coding-agent subprocess may take
# per task, the same five minutes CLAUDE_CODE_TIMEOUT_S in
# generate/adapters.py gives a `claude -p` run: a coding agent reads and
# edits files, so it gets the rollout timeout, not the chat one
# (convention, untested).
COMMAND_TIMEOUT_S = 300.0
# RESULT_CHARS = 2000: a tool result longer than this is cut on the step
# and the step says so (result_truncated, result_chars), the same cap
# CLAUDE_CODE_RESULT_CHARS applies; a shell dump is not a training row
# (convention, untested).
RESULT_CHARS = 2000
# FINGERPRINT_CHARS = 12: hex characters of the sha256 a harness is named
# by, the length whileai.platform.Harness.fingerprint already uses so the
# two agree byte for byte (convention).
FINGERPRINT_CHARS = 12
# MIN_LEVELS = 2: harnesses and models an attribution needs on each axis;
# one level has no spread to attribute (structural minimum).
MIN_LEVELS = 2
#: The argv token that stands for the task prompt in ``Harness.command``.
PROMPT = "{prompt}"

#: Built-in tools of each coding agent, as its own docs name them, so the
#: fingerprint of a preset says what the agent could call. Claude Code:
#: docs.anthropic.com/claude-code/settings, tools available to Claude.
#: Codex: `codex exec` runs shell commands and applies patches. pi: the
#: four tools its README ships (read, write, edit, bash).
CLAUDE_CODE_TOOLS = ("Read", "Write", "Edit", "Bash", "Grep", "Glob", "WebFetch", "WebSearch")
CODEX_TOOLS = ("shell", "apply_patch")
PI_TOOLS = ("read", "write", "edit", "bash")
#: Context files each agent loads into its prompt from the working
#: directory; the disclosure lists the ones that exist there.
CLAUDE_CODE_CONTEXT = ("CLAUDE.md", ".claude/CLAUDE.md", "AGENTS.md")
CODEX_CONTEXT = ("AGENTS.md",)
PI_CONTEXT = ("AGENTS.md", "CLAUDE.md", ".pi/SYSTEM.md", ".pi/APPEND_SYSTEM.md", ".pi/AGENTS.md")

Trajectory = dict[str, Any]
Parser = Callable[[str], Trajectory]


# --------------------------------------------------------------------------
# Disclosure: what a comparison has to say about the harness
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Disclosure:
    """What a comparison must state about the harness for a reader to
    know what was measured (Zhang et al. 2026, arXiv:2605.23950: context
    construction, tool interaction, orchestration and verification are
    the four places harnesses differ). Every field is optional; the ones
    set enter the fingerprint.

    * ``context``: files the harness loads into the prompt (``AGENTS.md``,
      ``CLAUDE.md``, ``.pi/SYSTEM.md``), as found in the working directory.
    * ``max_turns``: the turn cap.
    * ``compaction``: how a long context is shortened (``"none"``,
      ``"summarize"``, a name your harness uses).
    * ``retries``: retries or verification loops around a step.
    * ``subagents``: whether the harness delegates to subagents.
    * ``sampling``: temperature and the rest, when the harness fixes them.
    * ``notes``: anything else a reader needs to re-run it.
    """

    context: tuple[str, ...] = ()
    max_turns: int | None = None
    compaction: str | None = None
    retries: int | None = None
    subagents: bool | None = None
    sampling: Mapping[str, Any] | None = None
    notes: str | None = None

    def items(self) -> dict[str, Any]:
        """The fields that are set, in a stable order, JSON-ready."""
        out: dict[str, Any] = {}
        if self.context:
            out["context"] = sorted(str(c) for c in self.context)
        for name in ("max_turns", "compaction", "retries", "subagents", "notes"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.sampling:
            out["sampling"] = {str(k): self.sampling[k] for k in sorted(self.sampling)}
        return out


def _present(cwd: str | os.PathLike[str] | None, names: Sequence[str]) -> tuple[str, ...]:
    base = Path(cwd) if cwd is not None else Path.cwd()
    return tuple(n for n in names if (base / n).is_file())


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    if isinstance(tool, Mapping):
        inner = tool.get("function")
        if isinstance(inner, Mapping) and inner.get("name"):
            return str(inner["name"])
        if tool.get("name"):
            return str(tool["name"])
    for attr in ("name", "__name__"):
        value = getattr(tool, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(tool).__name__


def _model_spec(model: Any) -> str | None:
    """The model as the string the engine reads: ``provider:model``, a
    served URL, or a bare name. A backend object gives its ``spec``."""
    if model is None:
        return None
    if isinstance(model, str):
        return model
    spec = getattr(model, "spec", None)
    if isinstance(spec, str):
        return spec
    name = getattr(model, "model", None)
    return str(name) if name else repr(model)


def _model_name(spec: str | None) -> str | None:
    """The model's own name, without the provider prefix, for a label."""
    if spec is None:
        return None
    if "://" in spec:
        return spec
    return spec.split(":", 1)[1] if ":" in spec else spec


class Harness:
    """The program around the model: instructions, tools, the loop that
    runs them, and the facts a reader needs to compare two of them.

    Reach for it when the thing you are changing is not the weights. Pass
    it wherever an agent goes (``simulate(harness)``), and every row
    carries ``harness = {label, hash, model, kind}``, so ``attribute``
    can later say which lever moved the score. ``pin()`` is the platform
    record for ``tracked.run(harness=)``.

    * ``model``: a backend (``wai.OpenAI("gpt-4.1-mini")``), a
      ``provider:model`` string, or ``None`` for the configured or hosted
      model. On a command harness it is the name the CLI is told.
    * ``instructions``: the system prompt (a command harness appends or
      prepends it the way its CLI allows, and the disclosure says which).
    * ``tools``: ``@wai.tool`` functions, plain functions, or OpenAI
      function schemas; a command harness lists the tools the CLI ships.
    * ``agent``: a callable ``message -> trajectory`` to fingerprint, or
      the runner ``Harness.command`` builds. ``None`` is the prompted
      loop the SDK plays itself.
    * ``label``: the version name the platform shows; ``h-<fingerprint>``
      when left out. Name variants ``prompt@model`` and the Runs page
      groups by both.
    * ``disclosure``: a ``Disclosure``; presets fill it from the CLI flags
      and the context files present in ``cwd``.

    Lambert 2025, chapter Evaluation: a score is only comparable with the
    setup held constant, so the setup is what the fingerprint hashes.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        instructions: str | None = None,
        tools: Sequence[Any] | None = None,
        agent: Callable[[str], Trajectory] | None = None,
        label: str | None = None,
        disclosure: Disclosure | None = None,
    ):
        self.model = model
        self.instructions = str(instructions).strip() if instructions else None
        self.tools: list[Any] = list(tools or [])
        self.agent = agent
        self.label = label
        self.disclosure = disclosure or Disclosure()
        self._runner: Callable[[str], Trajectory] | None = None
        if agent is not None and not callable(agent):
            raise TypeError("agent= must be a callable message -> trajectory, or None")

    # ------------------------------------------------------------ identity

    @property
    def kind(self) -> str:
        """``"prompted"`` (the SDK plays the loop), ``"command"`` (a CLI
        driven through its JSON stream) or ``"callable"`` (yours)."""
        if self.agent is None:
            return "prompted"
        return "command" if getattr(self.agent, "_harness_command", None) else "callable"

    @property
    def model_spec(self) -> str | None:
        return _model_spec(self.model)

    @property
    def model_name(self) -> str | None:
        return _model_name(self.model_spec)

    @property
    def tool_names(self) -> list[str]:
        return sorted(_tool_name(t) for t in self.tools)

    def _disclosure_blob(self) -> dict[str, Any]:
        blob: dict[str, Any] = {"kind": self.kind, **self.disclosure.items()}
        command = getattr(self.agent, "_harness_command", None)
        if command:
            blob["command"] = [str(a) for a in command]
        return blob

    @property
    def fingerprint(self) -> str:
        """Hex over the disclosure: model, instructions, tool names, and
        every ``Disclosure`` field set. The same fields, the same hash; the
        formula is ``whileai.platform.Harness.fingerprint``, so ``pin()``
        carries this exact value."""
        blob = json.dumps(
            {
                "model": self.model_name,
                "instructions": self.instructions,
                "tools": self.tool_names,
                "disclosure": self._disclosure_blob(),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:FINGERPRINT_CHARS]

    @property
    def version(self) -> str:
        return self.label or f"h-{self.fingerprint}"

    def stamp(self) -> dict[str, Any]:
        """What every row from this harness carries under ``harness``."""
        return {
            "label": self.version,
            "hash": self.fingerprint,
            "model": self.model_name,
            "kind": self.kind,
        }

    def pin(self) -> Any:
        """The platform record (``whileai.platform.Harness``) for
        ``tracked.run(harness=)``: same label, same fingerprint."""
        from .platform import Harness as PlatformHarness

        return PlatformHarness(
            label=self.label,
            instructions=self.instructions,
            tools=self.tool_names,
            model=self.model_name,
            disclosure=self._disclosure_blob(),
        )

    def wire(self) -> dict[str, Any]:
        return self.pin().wire()

    def __repr__(self) -> str:
        parts = [f"kind={self.kind!r}"]
        if self.model_name:
            parts.append(f"model={self.model_name!r}")
        if self.tools:
            parts.append(f"tools={len(self.tools)}")
        parts.append(f"version={self.version!r}")
        return f"Harness({', '.join(parts)})"

    # ------------------------------------------------------------- running

    def tool_schemas(self) -> list[dict[str, Any]] | None:
        """The tools as OpenAI function schemas, or ``None`` when there are
        none (the engine keeps its "no tools given" branch)."""
        if not self.tools:
            return None
        from .simulations.tools import schemas

        return schemas(self.tools)

    def into_simulate(
        self, tools: Any, system_prompt: str | None, max_turns: int | None
    ) -> tuple[Any, Any, str | None, int | None]:
        """What ``simulate(harness)`` hands the engine: the agent to play,
        and the tools, system prompt and turn cap the call did not name
        taken from the harness. A prompted harness becomes its model spec
        so the engine plays the world and the scheduled faults itself; a
        command or callable harness is played as it is."""
        agent: Any = self.model if self.kind == "prompted" else self
        return (
            agent,
            tools if tools is not None else (self.tools or None),
            system_prompt if system_prompt is not None else self.instructions,
            max_turns if max_turns is not None else self.disclosure.max_turns,
        )

    def stamp_rows(self, rows: Sequence[dict]) -> None:
        mark = self.stamp()
        for row in rows:
            if isinstance(row, dict):
                row["harness"] = dict(mark)

    def __call__(self, message: str) -> Trajectory:
        """Play one task. A prompted harness, called directly, builds its
        loop on first use and answers tools from the mock world; inside
        ``simulate`` the engine plays it instead."""
        if self.agent is not None:
            return self.agent(message)
        if self._runner is None:
            from .simulations.generate.adapters import resolve

            spec = self.model_spec
            if spec is None:
                raise ValueError(
                    "a prompted Harness with no model cannot be called on its own; give it "
                    "wai.OpenAI(...) or a provider:model string, or pass it to simulate()"
                )
            kw: dict[str, Any] = {}
            if self.disclosure.max_turns is not None:
                kw["max_turns"] = int(self.disclosure.max_turns)
            self._runner, _ = resolve(
                spec, tools=self.tool_schemas() or [], policy=self.instructions or "", **kw
            )
        return self._runner(message)

    # ------------------------------------------------------------ presets

    @classmethod
    def command(
        cls,
        argv: Sequence[str],
        *,
        parse: Parser | None = None,
        model: Any = None,
        label: str | None = None,
        cwd: str | os.PathLike[str] | None = None,
        timeout: float = COMMAND_TIMEOUT_S,
        disclosure: Disclosure | None = None,
        **rest: Any,
    ) -> Harness:
        """A harness that is a program: one subprocess per task.

        ``argv`` is the command; the token ``"{prompt}"`` in any entry is
        replaced by the task text, or the text goes to stdin when no entry
        carries the token.
        ``parse`` turns stdout into ``{steps, final_text}``; the default
        reads stdout as that JSON. ``instructions`` and ``tools`` (in
        ``rest``) are recorded for the fingerprint; the presets set them
        from what each CLI ships.
        """
        command = [str(a) for a in argv]
        if not command:
            raise ValueError("argv is empty; the first entry is the program to run")
        reader = parse or _parse_json_trajectory
        workdir = os.fspath(cwd) if cwd is not None else None

        def run(message: str) -> Trajectory:
            text = str(message)
            # The program is looked up on PATH here, not by CreateProcess:
            # on Windows an npm-installed CLI is `claude.cmd`, which a bare
            # `["claude", ...]` cannot find (WinError 2) while a shell can.
            program = shutil.which(command[0]) or command[0]
            args = [program, *command[1:]]
            try:
                if any(PROMPT in a for a in command):
                    proc = subprocess.run(
                        [a.replace(PROMPT, text) for a in args],
                        cwd=workdir,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                else:
                    proc = subprocess.run(
                        args,
                        cwd=workdir,
                        input=text,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"{command[0]} is not installed or not on PATH; install it, or pass the "
                    "full path as the first argv entry"
                ) from exc
            if proc.returncode != 0 and not (proc.stdout or "").strip():
                raise RuntimeError(
                    f"{command[0]} exited {proc.returncode}: {(proc.stderr or '')[:300]}"
                )
            return reader(proc.stdout or "")

        run.__name__ = f"command[{command[0]}]"
        run._harness_command = command  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        return cls(
            model,
            instructions=rest.get("instructions"),
            tools=rest.get("tools"),
            agent=run,
            label=label,
            disclosure=disclosure,
        )

    @classmethod
    def claude_code(
        cls,
        model: str | None = None,
        *,
        cwd: str | os.PathLike[str] | None = None,
        max_turns: int | None = None,
        tools: Sequence[str] | None = None,
        instructions: str | None = None,
        label: str | None = None,
        timeout: float = COMMAND_TIMEOUT_S,
        extra_args: Sequence[str] = (),
    ) -> Harness:
        """Claude Code as the harness: ``claude -p <task> --output-format
        stream-json``. ``tools`` becomes ``--allowedTools``;
        ``instructions`` is appended to its system prompt
        (``--append-system-prompt``); ``max_turns`` is ``--max-turns``.
        The disclosure lists the ``CLAUDE.md`` / ``AGENTS.md`` files in
        ``cwd`` that it will read."""
        from .simulations.generate.adapters import parse_claude_stream

        argv: list[str] = ["claude", "-p", PROMPT, "--output-format", "stream-json", "--verbose"]
        if model:
            argv += ["--model", str(model)]
        if max_turns is not None:
            argv += ["--max-turns", str(int(max_turns))]
        if tools:
            argv += ["--allowedTools", ",".join(str(t) for t in tools)]
        if instructions:
            argv += ["--append-system-prompt", str(instructions)]
        argv += [str(a) for a in extra_args]
        return cls.command(
            argv,
            parse=parse_claude_stream,
            model=model,
            label=label,
            cwd=cwd,
            timeout=timeout,
            instructions=instructions,
            tools=list(tools) if tools else list(CLAUDE_CODE_TOOLS),
            disclosure=Disclosure(
                context=_present(cwd, CLAUDE_CODE_CONTEXT),
                max_turns=max_turns,
                compaction="auto",
                subagents=True,
                notes="instructions appended to the system prompt" if instructions else None,
            ),
        )

    @classmethod
    def codex(
        cls,
        model: str | None = None,
        *,
        cwd: str | os.PathLike[str] | None = None,
        sandbox: str | None = None,
        instructions: str | None = None,
        label: str | None = None,
        timeout: float = COMMAND_TIMEOUT_S,
        extra_args: Sequence[str] = (),
    ) -> Harness:
        """Codex CLI as the harness: ``codex exec --json <task>``.
        ``sandbox`` is ``--sandbox`` (``read-only``, ``workspace-write``,
        ``danger-full-access``). Codex has no system-prompt flag in exec
        mode, so ``instructions`` are prepended to the task text and the
        disclosure says so. The disclosure lists the ``AGENTS.md`` in
        ``cwd``."""
        argv: list[str] = ["codex", "exec", "--json", "--skip-git-repo-check"]
        if model:
            argv += ["--model", str(model)]
        if sandbox:
            argv += ["--sandbox", str(sandbox)]
        argv += [str(a) for a in extra_args]
        argv.append(PROMPT)
        return cls.command(
            _prepend(argv, instructions),
            parse=parse_codex_stream,
            model=model,
            label=label,
            cwd=cwd,
            timeout=timeout,
            instructions=instructions,
            tools=list(CODEX_TOOLS),
            disclosure=Disclosure(
                context=_present(cwd, CODEX_CONTEXT),
                compaction="auto",
                notes=_prepend_note(instructions, sandbox and f"sandbox {sandbox}"),
            ),
        )

    @classmethod
    def pi(
        cls,
        model: str | None = None,
        *,
        provider: str | None = None,
        cwd: str | os.PathLike[str] | None = None,
        instructions: str | None = None,
        extensions: bool = True,
        label: str | None = None,
        timeout: float = COMMAND_TIMEOUT_S,
        extra_args: Sequence[str] = (),
    ) -> Harness:
        """pi (pi.dev) as the harness: ``pi --mode json --no-session
        <task>``. ``provider`` and ``model`` are its ``--provider`` and
        ``--model``; ``extensions=False`` adds ``--no-extensions`` so the
        four built-in tools are the whole tool set. ``instructions`` are
        prepended to the task text (pi reads its system prompt from
        ``.pi/SYSTEM.md``; put it there to change the prompt itself) and
        the disclosure says so."""
        argv: list[str] = ["pi", "--mode", "json", "--no-session"]
        if provider:
            argv += ["--provider", str(provider)]
        if model:
            argv += ["--model", str(model)]
        if not extensions:
            argv.append("--no-extensions")
        argv += [str(a) for a in extra_args]
        argv.append(PROMPT)
        return cls.command(
            _prepend(argv, instructions),
            parse=parse_pi_stream,
            model=model,
            label=label,
            cwd=cwd,
            timeout=timeout,
            instructions=instructions,
            tools=list(PI_TOOLS),
            disclosure=Disclosure(
                context=_present(cwd, PI_CONTEXT),
                compaction="auto",
                subagents=extensions,
                notes=_prepend_note(instructions, None if extensions else "no extensions"),
            ),
        )


def _prepend(argv: list[str], instructions: str | None) -> list[str]:
    """Bake ``instructions`` in front of the ``{prompt}`` token, for a CLI
    with no system-prompt flag. The token stays a token."""
    if not instructions:
        return argv
    marker = f"{str(instructions).strip()}\n\n{PROMPT}"
    return [marker if a == PROMPT else a for a in argv]


def _prepend_note(instructions: str | None, more: str | None) -> str | None:
    parts = []
    if instructions:
        parts.append("instructions prepended to the task text")
    if more:
        parts.append(more)
    return "; ".join(parts) or None


# --------------------------------------------------------------------------
# Parsers: one JSON event stream -> {steps, final_text}
# --------------------------------------------------------------------------


def _parse_json_trajectory(stdout: str) -> Trajectory:
    out = json.loads(stdout)
    if not isinstance(out, dict) or "steps" not in out:
        raise ValueError("the command must print one JSON object with 'steps' and 'final_text'")
    return out


def _events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _cut(result: Any) -> dict[str, Any]:
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    step: dict[str, Any] = {"result": text[:RESULT_CHARS]}
    if len(text) > RESULT_CHARS:
        step["result_truncated"] = True
        step["result_chars"] = len(text)
    return step


def parse_codex_stream(stdout: str) -> Trajectory:
    """``codex exec --json``: one event per line, ``item.completed``
    carries the finished item. ``command_execution`` becomes a ``shell``
    step (command, aggregated output, exit code), ``file_change`` an
    ``apply_patch`` step, ``mcp_tool_call`` and ``web_search`` their own
    tool steps, and the last ``agent_message`` is ``final_text``."""
    steps: list[dict[str, Any]] = []
    final_text = ""
    errors: list[str] = []
    for event in _events(stdout):
        kind = str(event.get("type") or "")
        if kind == "error":
            errors.append(str(event.get("message") or event))
            continue
        if kind != "item.completed":
            continue
        item = event.get("item") or {}
        itype = str(item.get("type") or "")
        if itype == "command_execution":
            step = {
                "tool": "shell",
                "arguments": {"command": item.get("command")},
                **_cut(item.get("aggregated_output") or ""),
            }
            if item.get("exit_code") is not None:
                step["exit_code"] = item["exit_code"]
            steps.append(step)
        elif itype == "file_change":
            steps.append(
                {
                    "tool": "apply_patch",
                    "arguments": {"changes": item.get("changes") or []},
                    "result": str(item.get("status") or "completed"),
                }
            )
        elif itype == "mcp_tool_call":
            name = item.get("tool") or item.get("name") or "mcp_tool"
            server = item.get("server")
            steps.append(
                {
                    "tool": f"{server}.{name}" if server else str(name),
                    "arguments": item.get("arguments") or {},
                    **_cut(item.get("result") or item.get("error") or ""),
                }
            )
        elif itype == "web_search":
            steps.append(
                {"tool": "web_search", "arguments": {"query": item.get("query")}, "result": ""}
            )
        elif itype == "agent_message":
            final_text = str(item.get("text") or "")
    if errors and not steps and not final_text:
        raise RuntimeError(f"codex reported an error and produced nothing: {errors[0][:300]}")
    return {"steps": steps, "final_text": final_text}


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return default


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, Mapping):
                if block.get("type") in (None, "text") and block.get("text") is not None:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if content is None:
        return ""
    return json.dumps(content, default=str)


def parse_pi_stream(stdout: str) -> Trajectory:
    """``pi --mode json``: one event per line. Tool calls are read from
    ``tool_execution_start`` / ``tool_execution_end`` (tool name,
    arguments, result, error flag) and, when those are absent, from the
    assistant message's ``toolCall`` blocks paired with ``toolResults``
    on ``turn_end``. ``final_text`` is the text of the last assistant
    ``message_end``. Written from pi's ``docs/json.md``; field names are
    read leniently (``toolName`` or ``name``, ``args`` or ``arguments``)."""
    steps: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    final_text = ""
    for event in _events(stdout):
        kind = str(event.get("type") or "")
        if kind == "tool_execution_start":
            call_id = str(_first(event, "toolCallId", "id", default=len(pending)))
            pending[call_id] = {
                "tool": str(_first(event, "toolName", "name", default="")),
                "arguments": _first(event, "args", "arguments", "input", default={}) or {},
            }
        elif kind == "tool_execution_end":
            call_id = str(_first(event, "toolCallId", "id", default=""))
            call = pending.pop(
                call_id, {"tool": str(_first(event, "toolName", "name", default=""))}
            )
            result = _first(event, "result", "output", "content", default="")
            if isinstance(result, Mapping):
                is_error = bool(result.get("isError"))
                result = _first(result, "content", "output", "text", default=result)
            else:
                is_error = bool(event.get("isError"))
            step = {**call, **_cut(_text_of(result))}
            if is_error:
                step["is_error"] = True
            steps.append(step)
            seen_ids.add(call_id)
        elif kind == "message_end":
            message = event.get("message") or {}
            if str(message.get("role") or "assistant") != "assistant":
                continue
            content = message.get("content")
            text = _text_of(content)
            if text:
                final_text = text
            for block in content if isinstance(content, list) else []:
                if isinstance(block, Mapping) and str(block.get("type") or "") in (
                    "toolCall",
                    "tool_call",
                    "toolUse",
                    "tool_use",
                ):
                    call_id = str(_first(block, "id", "toolCallId", default=len(pending)))
                    if call_id not in seen_ids and call_id not in pending:
                        pending[call_id] = {
                            "tool": str(_first(block, "name", "toolName", default="")),
                            "arguments": _first(block, "arguments", "args", "input", default={})
                            or {},
                        }
        elif kind == "turn_end":
            for result in event.get("toolResults") or []:
                if not isinstance(result, Mapping):
                    continue
                call_id = str(_first(result, "toolCallId", "id", default=""))
                if call_id in seen_ids:
                    continue
                call = pending.pop(call_id, {"tool": str(_first(result, "toolName", default=""))})
                step = {**call, **_cut(_text_of(_first(result, "content", "output", default="")))}
                if result.get("isError"):
                    step["is_error"] = True
                steps.append(step)
                seen_ids.add(call_id)
    steps.extend({**call, "result": ""} for call in pending.values())
    return {"steps": steps, "final_text": final_text}


# --------------------------------------------------------------------------
# Attribution: which lever moved the score
# --------------------------------------------------------------------------


class AttributionReport(Report):
    """Which lever moved the score across a harness x model grid. Prints
    the grid in points, the share of the spread each lever explains with
    its interval, whether the model ranking flips across harnesses, and
    the verdict in words."""

    _summary_keys = ("verdict", "share_harness", "share_model")

    def __str__(self) -> str:
        harnesses: list[str] = list(self["harnesses"])
        models: list[str] = list(self["models"])
        cells: dict[str, float] = self["cells"]
        w = max(8, *(len(h) for h in harnesses)) + 2
        cw = max(8, *(len(m) for m in models)) + 2
        lines = [
            f"attribution on {self['metric']}: {len(harnesses)} harnesses x {len(models)} models, "
            f"{self['n_tasks']} tasks each cell",
            f"  {'harness':<{w}}" + "".join(f"{m:>{cw}}" for m in models),
        ]
        for h in harnesses:
            lines.append(f"  {h:<{w}}" + "".join(f"{cells[f'{h}@{m}']:>{cw}.1f}" for m in models))

        def share(name: str) -> str:
            value = self[f"share_{name}"]
            ci = self.get(f"ci_share_{name}")
            text = f"{100 * value:.0f}%"
            if ci:
                text += f" [{100 * ci[0]:.0f}..{100 * ci[1]:.0f}]"
            return text

        lines.append(
            f"  spread explained: harness {share('harness')}, model {share('model')}, "
            f"interaction {100 * self['share_interaction']:.0f}%"
        )
        lines.append(
            f"  harness moves the score by up to {self['spread_harness_pts']:.1f} points, "
            f"the model by up to {self['spread_model_pts']:.1f}"
        )
        lines.append(
            "  model ranking flips across harnesses"
            if self["ranking_reversal"]
            else "  the same model leads under every harness"
        )
        lines.append(f"  {self['sentence']}")
        if self.get("note"):
            lines.append(f"  {self['note']}")
        return "\n".join(lines)


def _harness_and_model(row: Mapping[str, Any]) -> tuple[str, str] | None:
    mark = row.get("harness")
    if isinstance(mark, Mapping) and mark.get("label"):
        model = mark.get("model")
        if not model:
            sampling = row.get("sampling")
            model = sampling.get("model") if isinstance(sampling, Mapping) else None
        return str(mark["label"]), str(model or "-")
    return None


def _value(row: Mapping[str, Any], metric: str) -> float | None:
    if metric == "pass_at_1":
        v = row.get("reward")
    else:
        markers = row.get("markers")
        v = markers.get(metric.split(":", 1)[1]) if isinstance(markers, Mapping) else None
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class _Decomposition:
    cells: dict[tuple[str, str], float]
    share_harness: float
    share_model: float
    share_interaction: float
    spread_harness: float
    spread_model: float
    top_models: list[str] = field(default_factory=list)


def _decompose(
    means: Mapping[tuple[str, str], Mapping[str, float]],
    harnesses: Sequence[str],
    models: Sequence[str],
    tasks: Sequence[str],
) -> _Decomposition:
    """Two-way additive decomposition of the cell means over ``tasks``:
    grand mean, harness effects, model effects, the interaction, and the
    share of the total sum of squares each takes."""
    cells = {
        (h, m): sum(means[(h, m)][t] for t in tasks) / len(tasks) for h in harnesses for m in models
    }
    grand = sum(cells.values()) / len(cells)
    a = {h: sum(cells[(h, m)] for m in models) / len(models) - grand for h in harnesses}
    b = {m: sum(cells[(h, m)] for h in harnesses) / len(harnesses) - grand for m in models}
    ss_h = len(models) * sum(v * v for v in a.values())
    ss_m = len(harnesses) * sum(v * v for v in b.values())
    ss_i = sum((cells[(h, m)] - grand - a[h] - b[m]) ** 2 for h in harnesses for m in models)
    total = ss_h + ss_m + ss_i
    shares = (0.0, 0.0, 0.0) if total <= 0 else (ss_h / total, ss_m / total, ss_i / total)
    top = [max(models, key=lambda m: cells[(h, m)]) for h in harnesses]
    return _Decomposition(
        cells=cells,
        share_harness=shares[0],
        share_model=shares[1],
        share_interaction=shares[2],
        spread_harness=max(a.values()) - min(a.values()),
        spread_model=max(b.values()) - min(b.values()),
        top_models=top,
    )


def attribute(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str = "pass_at_1",
    n_boot: int | None = None,
    seed: int = 0,
    level: float | None = None,
) -> AttributionReport:
    """Which lever moved the score: the harness, the model, or neither for sure.

    Reach for it after running every harness in a set on every model in a
    set over the same tasks (``simulate(harness, tasks=frozen)`` per
    pair, rows stamped ``harness = {label, model}``). The report is the
    grid of cell means in points, the share of the spread across cells
    that the harness axis, the model axis and their interaction each
    explain (two-way decomposition of the cell means, Zhang et al. 2026,
    arXiv:2605.23950, the variance decomposition protocol), a percentile
    bootstrap interval over tasks on each share (Miller 2024,
    arXiv:2411.00640: the unit is the task), whether the top model
    changes from one harness to another (their ranking reversal), and a
    verdict: the harness moved the score more than the model, the model
    more than the harness, or the difference could be chance.

    * ``metric``: ``"pass_at_1"`` (binary reward) or ``"marker:<name>"``.
    * ``n_boot`` (``DEFAULT_BOOT``), ``seed`` (0), ``level``
      (``CI_LEVEL``): the bootstrap.

    Every cell must have rows on the same tasks; a missing cell or a task
    seen in only some cells is named, not skipped in silence.
    """
    from .simulations.defaults import CI_LEVEL
    from .simulations.score.stats import DEFAULT_BOOT, MIN_CI_TASKS, no_interval_note, task_key

    draws = int(n_boot if n_boot is not None else DEFAULT_BOOT)
    cover = float(level if level is not None else CI_LEVEL)
    if not 0 < cover < 1:
        raise ValueError("level is the interval's coverage, strictly between 0 and 1")

    per_cell: dict[tuple[str, str], dict[str, list[float]]] = {}
    unstamped = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = _harness_and_model(row)
        if key is None:
            unstamped += 1
            continue
        v = _value(row, metric)
        if v is None:
            continue
        per_cell.setdefault(key, {}).setdefault(task_key(dict(row)), []).append(v)
    if not per_cell:
        raise ValueError(
            "no row carries a harness; run them through simulate(wai.Harness(...)) or set "
            "row['harness'] = {'label': ..., 'model': ...} on each"
        )
    harnesses = sorted({h for h, _ in per_cell})
    models = sorted({m for _, m in per_cell})
    if len(harnesses) < MIN_LEVELS or len(models) < MIN_LEVELS:
        raise ValueError(
            f"attribution needs at least {MIN_LEVELS} harnesses and {MIN_LEVELS} models; got "
            f"{len(harnesses)} harness(es) {harnesses} and {len(models)} model(s) {models}. "
            "One level has no spread to attribute; run the missing arm"
        )
    missing = [f"{h}@{m}" for h in harnesses for m in models if (h, m) not in per_cell]
    if missing:
        raise ValueError(
            f"the grid is not full: no rows for {missing}. Every harness has to run on every "
            "model over the same tasks for the shares to mean anything"
        )
    task_sets = [set(per_cell[(h, m)]) for h in harnesses for m in models]
    shared = sorted(set.intersection(*task_sets))
    dropped = len(set.union(*task_sets)) - len(shared)
    if not shared:
        raise ValueError(
            "no task appears in every cell; pass the same frozen tasks (tasks=) to every run"
        )
    means = {
        cell: {t: sum(vals[t]) / len(vals[t]) for t in shared} for cell, vals in per_cell.items()
    }
    point = _decompose(means, harnesses, models, shared)

    ci_h = ci_m = ci_diff = None
    note_parts: list[str] = []
    if len(shared) >= MIN_CI_TASKS:
        rng = random.Random(seed)
        n = len(shared)
        boots = []
        for _ in range(draws):
            sample = [shared[rng.randrange(n)] for _ in range(n)]
            d = _decompose(means, harnesses, models, sample)
            boots.append((d.share_harness, d.share_model, d.share_harness - d.share_model))
        lo_i = max(0, int((1 - cover) / 2 * draws))
        hi_i = min(draws - 1, int((1 + cover) / 2 * draws) - 1)

        def interval(i: int) -> tuple[float, float]:
            ordered = sorted(b[i] for b in boots)
            return (ordered[lo_i], ordered[hi_i])

        ci_h, ci_m, ci_diff = interval(0), interval(1), interval(2)
    else:
        note_parts.append(no_interval_note(len(shared), quantity="the shares"))

    if ci_diff is not None and ci_diff[0] > 0:
        verdict = "harness"
        sentence = "the harness moved the score more than the model did"
    elif ci_diff is not None and ci_diff[1] < 0:
        verdict = "model"
        sentence = "the model moved the score more than the harness did"
    else:
        verdict = "unresolved"
        sentence = (
            "which lever moved the score more could be chance: the interval on the "
            "difference in shares covers zero"
        )
    reversal = len(set(point.top_models)) > 1
    if reversal:
        sentence += "; the leading model changes with the harness, so a model ranking from one harness does not carry"
    if dropped:
        note_parts.append(f"{dropped} task(s) seen in only some cells were left out")
    if unstamped:
        note_parts.append(f"{unstamped} row(s) carried no harness and were left out")

    best = max(point.cells, key=lambda c: point.cells[c])
    return AttributionReport(
        metric=metric,
        harnesses=harnesses,
        models=models,
        n_tasks=len(shared),
        cells={f"{h}@{m}": round(100 * point.cells[(h, m)], 1) for h in harnesses for m in models},
        share_harness=round(point.share_harness, 4),
        share_model=round(point.share_model, 4),
        share_interaction=round(point.share_interaction, 4),
        ci_share_harness=ci_h,
        ci_share_model=ci_m,
        ci_difference=ci_diff,
        spread_harness_pts=round(100 * point.spread_harness, 1),
        spread_model_pts=round(100 * point.spread_model, 1),
        best_cell=f"{best[0]}@{best[1]}",
        top_model_by_harness=dict(zip(harnesses, point.top_models)),
        ranking_reversal=reversal,
        verdict=verdict,
        sentence=sentence,
        level=cover,
        n_boot=draws,
        note="; ".join(note_parts) or None,
    )


def __getattr__(name: str) -> Any:
    # The platform-bound sweep lives in whileai.sweep; reach it from here
    # without importing pydantic on `import whileai.harness`.
    if name in ("HarnessSweep", "SweepReport", "VariantResult"):
        from . import sweep

        return getattr(sweep, name)
    raise AttributeError(f"module 'whileai.harness' has no attribute {name!r}")


__all__ = [
    "AttributionReport",
    "Disclosure",
    "Harness",
    "attribute",
    "parse_codex_stream",
    "parse_pi_stream",
]
