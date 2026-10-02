"""One harness, many spellings: a per-session rewrite of tool names, argument names and tool order.

Phase 1's arm C. The capture proxy hands every request to the engine in chat-completions shape and
every response back the same way. This module rewrites the request on the way in (tool names,
argument names, tool order, the names as they appear in the system prompt and in earlier assistant
turns) and maps the model's tool calls back to the harness's real names on the way out. The harness
never sees a fake name, and the model never sees a real one.

One mapping per capture session, drawn from a seed derived from the session id, so every turn of a
rollout sees the same spelling and the token prefix of turn n+1 still extends turn n. This covers
the format and context-name modes of KAT-Coder's three overfitting modes (arXiv:2607.05471); it
does not touch control flow (retries, stop rules).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

# Synonyms a real harness uses for the same tool (Claude Code, OpenCode, Codex, Mini-SWE-Agent, Pi).
SYNONYMS: dict[str, list[str]] = {
    "bash": ["bash", "shell", "run_command", "exec", "terminal", "execute_bash"],
    "read": ["read", "read_file", "view", "cat_file", "open_file"],
    "write": ["write", "write_file", "create_file", "save_file"],
    "edit": ["edit", "edit_file", "str_replace", "apply_edit", "replace_in_file"],
    "glob": ["glob", "find_files", "list_files", "file_search"],
    "grep": ["grep", "search", "search_files", "ripgrep"],
    "ls": ["ls", "list_dir", "list_directory"],
}
CASES = ("lower", "title", "pascal", "camel", "snake")


def _words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [w.lower() for w in re.split(r"[\s_\-]+", spaced) if w]


def _case(words: list[str], style: str) -> str:
    if style == "lower":
        return "".join(words)
    if style == "title":
        return "_".join(w.capitalize() for w in words)
    if style == "pascal":
        return "".join(w.capitalize() for w in words)
    if style == "camel":
        return words[0] + "".join(w.capitalize() for w in words[1:])
    return "_".join(words)


@dataclass
class Spelling:
    """One session's mapping. `tools` and `args` map real -> shown; the reverse maps undo them."""

    tools: dict[str, str] = field(default_factory=dict)
    args: dict[str, str] = field(default_factory=dict)
    order_seed: int = 0

    @property
    def tools_back(self) -> dict[str, str]:
        return {v: k for k, v in self.tools.items()}

    @property
    def args_back(self) -> dict[str, str]:
        return {v: k for k, v in self.args.items()}


def draw(session_id: str, tools: list[dict[str, Any]]) -> Spelling:
    """A deterministic spelling for one session over the tools its harness offers."""
    seed = int(hashlib.sha256(session_id.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    tool_case, arg_case = rng.choice(CASES), rng.choice(("snake", "camel"))
    spelling = Spelling(order_seed=rng.randrange(2**31))
    taken: set[str] = set()
    for tool in tools:
        fn = tool.get("function", {})
        real = fn.get("name", "")
        if not real:
            continue
        stem = "_".join(_words(real))
        options = SYNONYMS.get(stem) or [stem]
        shown = _case(_words(rng.choice(options)), tool_case)
        while shown in taken or not shown:
            shown = _case(_words(rng.choice(options)) + [str(rng.randrange(10))], tool_case)
        taken.add(shown)
        spelling.tools[real] = shown
        for arg in (fn.get("parameters") or {}).get("properties", {}) or {}:
            if arg not in spelling.args:
                spelling.args[arg] = _case(_words(arg), arg_case) or arg
    # Argument renames must stay one-to-one across every tool.
    if len(set(spelling.args.values())) != len(spelling.args):
        spelling.args = {a: a for a in spelling.args}
    return spelling


def _rename_args(obj: Any, mapping: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {mapping.get(k, k): v for k, v in obj.items()}
    return obj


def _rename_call_args(arguments: str, mapping: dict[str, str]) -> str:
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    return json.dumps(_rename_args(parsed, mapping), ensure_ascii=False)


def _rename_in_text(text: str, mapping: dict[str, str]) -> str:
    # Only names written as code (`Bash`, "Bash") are renamed, so plain words in prose survive.
    for real, shown in mapping.items():
        text = re.sub(rf"([`\"']){re.escape(real)}([`\"'])", rf"\g<1>{shown}\g<2>", text)
    return text


def rewrite_request(chat_request: dict[str, Any], spelling: Spelling) -> None:
    """In place: the request the engine sees uses the session's spelling everywhere."""
    tools = chat_request.get("tools")
    if tools:
        out = []
        for tool in tools:
            tool = json.loads(json.dumps(tool))
            fn = tool.get("function", {})
            fn["name"] = spelling.tools.get(fn.get("name", ""), fn.get("name", ""))
            params = fn.get("parameters") or {}
            if isinstance(params.get("properties"), dict):
                params["properties"] = _rename_args(params["properties"], spelling.args)
            if isinstance(params.get("required"), list):
                params["required"] = [spelling.args.get(a, a) for a in params["required"]]
            out.append(tool)
        random.Random(spelling.order_seed).shuffle(out)
        chat_request["tools"] = out
    choice = chat_request.get("tool_choice")
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        name = choice["function"].get("name", "")
        choice["function"]["name"] = spelling.tools.get(name, name)
    for msg in chat_request.get("messages", []) or []:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            msg["content"] = _rename_in_text(msg["content"], spelling.tools)
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            fn["name"] = spelling.tools.get(fn.get("name", ""), fn.get("name", ""))
            if "arguments" in fn:
                fn["arguments"] = _rename_call_args(fn["arguments"], spelling.args)
        if msg.get("role") == "tool" and msg.get("name"):
            msg["name"] = spelling.tools.get(msg["name"], msg["name"])


def restore_response(response: dict[str, Any], spelling: Spelling) -> None:
    """In place: the model's tool calls go back to the harness under the real names."""
    back_tools, back_args = spelling.tools_back, spelling.args_back
    for choice in response.get("choices", []) or []:
        msg = choice.get("message") or {}
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            fn["name"] = back_tools.get(fn.get("name", ""), fn.get("name", ""))
            if "arguments" in fn:
                fn["arguments"] = _rename_call_args(fn["arguments"], back_args)
