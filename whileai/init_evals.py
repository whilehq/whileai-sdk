"""``wai init-evals``: write an eval harness wired to the agent in this project.

The four files every cold-start tester wrote by hand before this command
existed: the wrapper that puts an agent in ``agent(message) -> {steps,
final_text}`` shape, the policy as a judge that reads the trajectory, the
run script with pass@1 and a CI gate, and a pytest file that checks the
judge on hand-written rows.

The scan is deliberately dumb: it reads the project's Python with ``ast``,
never imports it, and prints what it picked so a wrong guess is one flag
away. Nothing found means the files are still written, with the missing
parts marked TODO.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .templates import evals as templates

SKIP_DIRS = {
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "site-packages",
    "test",
    "tests",
    "venv",
}
TOOL_NAMES = {
    "TOOLS",
    "tools",
    "TOOL_DEFS",
    "tool_defs",
    "TOOL_SPECS",
    "TOOL_SCHEMAS",
    "TOOLSET",
    "FUNCTIONS",
}
PROMPT_NAMES = {
    "SYSTEM",
    "SYSTEM_PROMPT",
    "POLICY",
    "PROMPT",
    "system_prompt",
    "policy",
    "INSTRUCTIONS",
}
AGENT_NAMES = ("answer", "respond", "chat", "run", "handle", "reply", "ask")
RECORDER_NAMES = ("_run_tool", "run_tool", "call_tool", "execute_tool", "_call_tool")
FILES = ("agent.py", "judge.py", "run.py", "test_judge.py", "README.md")
ID_PATTERN = re.compile(r"\b[A-Z]{1,4}[-_]?\d{3,6}\b")


@dataclass
class Pick:
    """One thing the scan found, as ``module:name``."""

    module: str
    name: str

    @property
    def ref(self) -> str:
        return f"{self.module}:{self.name}"

    @property
    def dotted(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class ModuleFacts:
    """What one Python file offers, read off its syntax tree."""

    module: str
    tools: str | None = None
    system_prompt: str | None = None
    agent: str | None = None
    recorder: str | None = None
    tool_count: int = 0
    tool_strings: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(1 for item in (self.tools, self.system_prompt, self.agent) if item)


@dataclass
class Scan:
    """What the whole project offers."""

    root: Path
    modules: list[ModuleFacts] = field(default_factory=list)
    agent: Pick | None = None
    tools: Pick | None = None
    system_prompt: Pick | None = None
    recorder: Pick | None = None
    tool_count: int = 0
    ids: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)

    @property
    def found_nothing(self) -> bool:
        return not (self.agent or self.tools or self.system_prompt)


# ------------------------------------------------------------------ scanning


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _candidate_files(root: Path, out_dir: Path) -> list[Path]:
    """Top level and one level down, skipping the noise directories."""
    files = sorted(p for p in root.glob("*.py") if p.is_file())
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in SKIP_DIRS or child.name.startswith("."):
            continue
        if child.resolve() == out_dir.resolve():
            continue
        files.extend(sorted(p for p in child.glob("*.py") if p.is_file()))
    return [p for p in files if p.name != "setup.py"]


def _is_tool_list(node: ast.AST) -> tuple[bool, int, list[str]]:
    """A list of dicts that look like tool definitions, and its strings."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return False, 0, []
    dicts = [item for item in node.elts if isinstance(item, ast.Dict)]
    if not dicts or len(dicts) != len(node.elts):
        return False, 0, []
    keys = {
        key.value
        for item in dicts
        for key in item.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    if not ({"name", "function"} & keys):
        return False, 0, []
    strings = [
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    ]
    return True, len(node.elts), strings


def _returns_str(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool | None:
    if node.returns is None:
        return None
    return isinstance(node.returns, ast.Name) and node.returns.id == "str"


def _takes_one_string(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """One required positional (the message), anything after it defaulted.

    ``answer(message)``, ``answer(user_message: str, history=None)`` and
    ``chat(text, *, session=None)`` all qualify: the wrapper calls the
    function with one positional argument and lets the defaults stand.
    """
    args = node.args
    positional = args.posonlyargs + args.args
    if args.vararg or not positional:
        return False
    required = len(positional) - len(args.defaults)
    if required != 1:
        return False
    if any(d is None for d in args.kw_defaults):
        return False
    first = positional[0]
    if first.annotation is None:
        return True
    ann = first.annotation
    if isinstance(ann, ast.Name):
        return ann.id == "str"
    if isinstance(ann, ast.Constant):
        return ann.value == "str"
    return True


def _read_module(path: Path, root: Path) -> ModuleFacts | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (SyntaxError, OSError):
        return None
    facts = ModuleFacts(module=_module_name(path, root))
    agent_rank = 99
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in targets:
                if facts.tools is None and name in TOOL_NAMES:
                    ok, count, strings = _is_tool_list(node.value)
                    if ok:
                        facts.tools = name
                        facts.tool_count = count
                        facts.tool_strings = strings
                if (
                    facts.system_prompt is None
                    and name in PROMPT_NAMES
                    and _looks_like_text(node.value)
                ):
                    facts.system_prompt = name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.lower() in RECORDER_NAMES and len(node.args.args) >= 2:
                facts.recorder = facts.recorder or node.name
            if node.name.startswith("_") or not _takes_one_string(node):
                continue
            returns = _returns_str(node)
            if returns is False:
                continue
            preferred = node.name.lower() in AGENT_NAMES
            if returns is None and not preferred:
                continue
            rank = (0 if preferred else 1) + (0 if returns else 2)
            if rank < agent_rank:
                facts.agent, agent_rank = node.name, rank
    return facts


def _looks_like_text(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):  # an f-string policy
        return True
    if isinstance(node, ast.BinOp):  # "a" + "b", or a %-format
        return _looks_like_text(node.left)
    if isinstance(node, ast.Call):  # "...".format(...), "\n".join(...)
        return isinstance(node.func, ast.Attribute) and _looks_like_text(node.func.value)
    return False


def _parse_ref(value: str, flag: str) -> Pick:
    module, _, name = value.partition(":")
    if not module or not name:
        raise ValueError(f"{flag} wants module:name, for example bot:answer (got {value!r})")
    return Pick(module=module, name=name)


def scan(root: Path, out_dir: Path) -> Scan:
    """Read the project's Python and pick the agent, its tools and its policy."""
    result = Scan(root=root)
    for path in _candidate_files(root, out_dir):
        facts = _read_module(path, root)
        if facts and facts.module:
            result.modules.append(facts)
    best = sorted(result.modules, key=lambda f: (-f.score, f.module.count("."), f.module))
    for facts in best:
        if result.agent is None and facts.agent:
            result.agent = Pick(facts.module, facts.agent)
        if result.tools is None and facts.tools:
            result.tools = Pick(facts.module, facts.tools)
            result.tool_count = facts.tool_count
            result.ids = _ids_in(facts.tool_strings)
            result.tool_names = _tool_names_in(facts.tool_strings)
        if result.system_prompt is None and facts.system_prompt:
            result.system_prompt = Pick(facts.module, facts.system_prompt)
    result.recorder = _recorder_for(result.agent, result.modules)
    return result


def _recorder_for(agent: Pick | None, modules: list[ModuleFacts]) -> Pick | None:
    """The tool runner to wrap, in the module the agent lives in."""
    if agent is None:
        return None
    for facts in modules:
        if facts.module == agent.module and facts.recorder:
            return Pick(facts.module, facts.recorder)
    return None


def _ids_in(strings: list[str]) -> list[str]:
    found: list[str] = []
    for text in strings:
        for match in ID_PATTERN.findall(text):
            if match not in found:
                found.append(match)
    return found[:3]


def _tool_names_in(strings: list[str]) -> list[str]:
    return [s for s in strings if re.fullmatch(r"[a-z][a-z0-9_]{2,40}", s)]


# ------------------------------------------------------------------ rendering


def _render_agent(scan_result: Scan) -> str:
    modules = sorted(
        {
            pick.module
            for pick in (scan_result.agent, scan_result.tools, scan_result.system_prompt)
            if pick
        }
    )
    if modules:
        imports = "\n".join(f"import {name}" for name in modules)
    else:
        imports = "# TODO: import the module your agent lives in, then use it below.\n# import bot"

    if scan_result.tools:
        tools_block = (
            f"# Your tools, in the shape the writer and the judge read.\n"
            f"TOOLS = [_openai_tool(tool) for tool in {scan_result.tools.dotted}]"
        )
    else:
        tools_block = (
            "# TODO: your tool definitions. The writer reads the names and\n"
            "# descriptions to write asks, and the judge reads the calls. Put your\n"
            '# real ids in a description ("Orders on file: A1001, A1002") or the\n'
            '# writer invents ids and every rollout comes back "not found".\n'
            "TOOLS = [\n"
            "    _openai_tool(\n"
            "        {\n"
            '            "name": "lookup_order",\n'
            '            "description": "Look up an order by id. Orders on file: A1001, A1002.",\n'
            '            "parameters": {\n'
            '                "type": "object",\n'
            '                "properties": {"order_id": {"type": "string"}},\n'
            '                "required": ["order_id"],\n'
            "            },\n"
            "        }\n"
            "    )\n"
            "]"
        )

    if scan_result.system_prompt:
        system_block = (
            "# The policy the writer reads for the branches an ask can land in.\n"
            f"SYSTEM_PROMPT = {scan_result.system_prompt.dotted}"
        )
    else:
        system_block = (
            "# TODO: the system prompt your agent ships with, the real one. The\n"
            "# writer reads it for the policy branches an ask can land in.\n"
            'SYSTEM_PROMPT = "TODO: paste your agent\'s system prompt here."'
        )

    if scan_result.agent and scan_result.recorder:
        agent_block = templates.RECORDING_AGENT.replace(
            "__WAI_RECORDER_REF__", scan_result.recorder.dotted
        ).replace("__WAI_AGENT_REF__", scan_result.agent.dotted)
    else:
        call = (
            f"{scan_result.agent.dotted}(message)"
            if scan_result.agent
            else '"TODO: call your agent here and return what it said."'
        )
        agent_block = templates.TODO_RECORDING_AGENT.replace("__WAI_AGENT_CALL__", call)

    return (
        templates.AGENT_PY.replace("__WAI_IMPORTS__", imports)
        .replace("__WAI_TOOLS_BLOCK__", tools_block)
        .replace("__WAI_SYSTEM_BLOCK__", system_block)
        .replace("__WAI_AGENT_BLOCK__", agent_block.rstrip("\n"))
    )


def _render_seeds(scan_result: Scan) -> str:
    if scan_result.ids:
        ids = (scan_result.ids * 3)[:3]
        asks = [
            f"Hi, I need help with {ids[0]}.",
            f"Something is wrong with {ids[1]}. What can you do about it?",
            f"{ids[2]} is not what I expected, and I want this sorted today.",
        ]
        note = (
            "# One ask per branch of your policy. The writer varies the wording\n"
            "# and the stance; the id is what keeps the branch. These ids came\n"
            "# out of your tool descriptions: replace the asks with real ones."
        )
    else:
        asks = [
            "Hi, I need help with my last order.",
            "Something is wrong with what I received. What can you do about it?",
            "I want this sorted today, it is the second time I am asking.",
        ]
        note = (
            "# TODO: one ask per branch of your policy, in the words a real\n"
            "# person uses, naming a real id from your world (an order number, an\n"
            "# account). An ask that names no id lands on an id the writer\n"
            '# invents, and every rollout comes back "not found".'
        )
    lines = "\n".join(f"    {json.dumps(ask)}," for ask in asks)
    return f"{note}\nSEEDS = [\n{lines}\n]"


def render(scan_result: Scan) -> dict[str, str]:
    """The five files, as text."""
    example_tool = scan_result.tool_names[0] if scan_result.tool_names else "lookup_order"
    label = scan_result.agent.dotted if scan_result.agent else "your agent"
    return {
        "agent.py": _render_agent(scan_result),
        "judge.py": templates.JUDGE_PY,
        "run.py": templates.RUN_PY.replace("__WAI_SEEDS_BLOCK__", _render_seeds(scan_result)),
        "test_judge.py": templates.TEST_JUDGE_PY.replace("__WAI_EXAMPLE_TOOL__", example_tool),
        "README.md": templates.README_MD.replace("__WAI_AGENT_LABEL__", label),
    }


# ------------------------------------------------------------------ the command


def _line(label: str, value: str) -> str:
    return f"  {label:<14}{value}"


def init_evals(
    *,
    root: str | Path = ".",
    out: str | Path = "evals",
    agent: str | None = None,
    tools: str | None = None,
    system_prompt: str | None = None,
    force: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Write the eval files. Returns the exit code the CLI hands back."""
    out_stream = stream or sys.stdout
    root_path = Path(root).resolve()
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = root_path / out_path

    try:
        overrides = {
            "agent": _parse_ref(agent, "--agent") if agent else None,
            "tools": _parse_ref(tools, "--tools") if tools else None,
            "system_prompt": (
                _parse_ref(system_prompt, "--system-prompt") if system_prompt else None
            ),
        }
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    found = scan(root_path, out_path)
    picked_by_flag = set()
    for field_name, pick in overrides.items():
        if pick is not None:
            setattr(found, field_name, pick)
            picked_by_flag.add(field_name)
    if "agent" in picked_by_flag:
        found.recorder = _recorder_for(found.agent, found.modules)

    existing = [name for name in FILES if (out_path / name).exists()]
    if existing and not force:
        print(
            f"error: {out_path} already has {', '.join(existing)}. "
            "Move them, or pass --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    files = render(found)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out_path / name).write_text(text, encoding="utf-8")

    where = out_path.name if out_path.parent == root_path else str(out_path)
    print(f"init-evals: read {len(found.modules)} module(s) under {root_path}", file=out_stream)

    def show(label: str, pick: Pick | None, extra: str = "", missing: str = "") -> None:
        if pick is None:
            print(_line(label, f"not found, TODO in {where}/agent.py: {missing}"), file=out_stream)
        else:
            flag = " (--flag)" if label.replace(" ", "_") in picked_by_flag else ""
            print(_line(label, f"{pick.ref}{extra}{flag}"), file=out_stream)

    show("agent", found.agent, missing="call your agent in agent()")
    show(
        "tools",
        found.tools,
        extra=f" ({found.tool_count} tools)" if found.tool_count else "",
        missing="paste your tool definitions",
    )
    show("system prompt", found.system_prompt, missing="paste your system prompt")
    if found.recorder:
        print(
            _line(
                "tool calls",
                f"{found.recorder.ref} wrapped in a thread-local recorder",
            ),
            file=out_stream,
        )
    else:
        print(
            _line("tool calls", f"TODO in {where}/agent.py: record them in _local.calls"),
            file=out_stream,
        )
    if found.ids:
        print(
            _line("seed ids", ", ".join(found.ids) + " (read off your tool descriptions)"),
            file=out_stream,
        )
    else:
        print(
            _line(
                "seed ids",
                f"none found: put the ids your world has (order numbers, account names) in "
                f"SEEDS in {where}/run.py or in the tool descriptions, or every ask stops at "
                "'which order?'",
            ),
            file=out_stream,
        )

    print(f"\nwrote {', '.join(f'{where}/{name}' for name in files)}", file=out_stream)
    if found.found_nothing:
        print(
            "\nNothing to wire up was found, so the files are stubs: every place\n"
            "that needs your code is marked TODO. Point the command at it with\n"
            "--agent module:callable --tools module:NAME --system-prompt module:NAME.",
            file=out_stream,
        )
    print(
        f"\nnext:\n"
        f"  python {where}/run.py --gap    what these asks never reach\n"
        f"  python {where}/run.py          pass@1 with an interval, offline, no key\n"
        f"  pytest {where}/test_judge.py   the judge on hand-written rows",
        file=out_stream,
    )
    return 0


def add_arguments(parser: Any) -> None:
    """The flags, kept next to the command they belong to."""
    parser.add_argument("--agent", help="the callable to wrap, as module:callable")
    parser.add_argument("--tools", help="the tool definitions, as module:NAME")
    parser.add_argument("--system-prompt", help="the system prompt, as module:NAME")
    parser.add_argument("--dir", default="evals", help="where to write the files (default: evals)")
    parser.add_argument("--force", action="store_true", help="overwrite files that are there")
