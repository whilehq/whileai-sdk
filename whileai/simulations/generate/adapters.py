"""Framework adapters. Every runner returns {steps, final_text}."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect as _inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .agents import complete, local_model, parse_backend_spec, split_user_turns

#: Claude Code tool results are kept to this many characters on the step;
#: a longer one is cut and the step says so (result_truncated, result_chars).
CLAUDE_CODE_RESULT_CHARS = 2000
#: Reply budget in tokens for an ``http`` agent; ``agent_max_tokens`` does
#: not reach it, so every row from one says this number.
HTTP_REPLY_TOKENS = 1024


def _missing(extra: str, exc: Exception) -> ImportError:
    return ImportError(f"this adapter needs '{extra}' installed (underlying: {exc})")


def _sync(fn: Callable) -> Callable:
    if not _inspect.iscoroutinefunction(fn):
        return fn

    def agent(message: str) -> dict:
        return asyncio.run(fn(message))

    agent.__name__ = getattr(fn, "__name__", "async_agent")
    return agent


def openai_http(
    url: str,
    *,
    model: str,
    tools: list[dict],
    execute: Callable[[str, dict], Any] | None = None,
    system: str = "",
    api_key: str | None = None,
    max_turns: int = 5,
    temperature: float = 0.3,
) -> Callable:
    """Any OpenAI-compatible /v1/chat/completions endpoint with tool support."""
    sim = execute or (lambda tool, args: {"ok": True})
    policy_text = str(system or "").strip()
    reply_tokens = HTTP_REPLY_TOKENS

    def agent(message: str) -> dict:
        # New chat every call. Prior turns do not carry over.
        turns = split_user_turns(message)
        messages = ([{"role": "system", "content": policy_text}] if policy_text else []) + [
            {"role": "user", "content": turns[0]},
        ]
        steps: list[dict] = []
        user_i = 0
        final_text = ""
        for _ in range(max_turns):
            reply = complete(
                url,
                model,
                messages,
                tools=tools,
                api_key=api_key,
                temperature=temperature,
                max_tokens=reply_tokens,
            )
            calls = reply.get("tool_calls") or []
            spoken = (reply.get("content") or "").strip()
            if not calls:
                if spoken:
                    steps.append({"text": spoken})
                    messages.append({"role": "assistant", "content": spoken})
                    final_text = spoken
                if user_i + 1 < len(turns):
                    user_i += 1
                    steps.append({"user": turns[user_i]})
                    messages.append({"role": "user", "content": turns[user_i]})
                    continue
                return {"steps": steps, "final_text": spoken}
            messages.append(reply)
            attached = False
            for call in calls:
                fn = call.get("function", {})
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = sim(fn.get("name", ""), arguments)
                if not isinstance(result, str):
                    stored, content = result, json.dumps(result)
                else:
                    stored, content = result, result
                    with contextlib.suppress(json.JSONDecodeError):
                        stored = json.loads(result)
                step = {"tool": fn.get("name", ""), "arguments": arguments, "result": stored}
                if spoken and not attached:
                    step["text"] = spoken
                    attached = True
                steps.append(step)
                messages.append(
                    {"role": "tool", "tool_call_id": call.get("id", ""), "content": content}
                )
            if spoken:
                final_text = spoken
        return {"steps": steps, "final_text": final_text}

    agent.__name__ = f"openai_http[{model}]"
    # How every reply was sampled, as the engine stamps it on the row.
    agent.sampling = {  # type: ignore[attr-defined]
        "temperature": float(temperature),
        "max_tokens": reply_tokens,
        "model": model,
    }
    agent.system = policy_text  # type: ignore[attr-defined]
    agent.policy = policy_text  # type: ignore[attr-defined]
    return agent


def from_langchain(executor: Any) -> Callable:
    """Classic AgentExecutor: steps come from intermediate_steps."""
    try:
        importlib.import_module("langchain_core.agents")
    except Exception as exc:  # pragma: no cover
        raise _missing("langchain", exc) from exc

    def agent(message: str) -> dict:
        steps = []
        final = ""
        for turn in split_user_turns(message):
            out = executor.invoke({"input": turn}, return_only_outputs=False)
            for action, observation in out.get("intermediate_steps", []):
                args = (
                    action.tool_input
                    if isinstance(action.tool_input, dict)
                    else {"input": action.tool_input}
                )
                steps.append({"tool": action.tool, "arguments": args, "result": str(observation)})
            final = str(out.get("output", ""))
        return {"steps": steps, "final_text": final}

    return agent


def from_langgraph(graph: Any) -> Callable:
    """Compiled LangGraph: AIMessage.tool_calls paired with ToolMessage by id."""
    try:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    except Exception as exc:  # pragma: no cover
        raise _missing("langchain", exc) from exc

    def agent(message: str) -> dict:
        turns = split_user_turns(message)
        conversation: list = []
        state: dict = {"messages": []}
        for turn in turns:
            state = graph.invoke({"messages": [*conversation, HumanMessage(content=turn)]})
            conversation = list(state.get("messages", []))
        messages = state.get("messages", [])
        results = {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}
        steps, final = [], ""
        for m in messages:
            if isinstance(m, AIMessage):
                for call in m.tool_calls or []:
                    steps.append(
                        {
                            "tool": call.get("name", ""),
                            "arguments": call.get("args") or {},
                            "result": results.get(call.get("id", ""), ""),
                        }
                    )
                if not m.tool_calls and m.content:
                    final = str(m.content)
        return {"steps": steps, "final_text": final}

    return agent


def from_openai_agents(agent_obj: Any) -> Callable:
    try:
        from agents import Runner
    except Exception as exc:  # pragma: no cover
        raise _missing("openai-agents", exc) from exc

    async def agent(message: str) -> dict:
        pending: dict[str, dict] = {}
        steps: list[dict] = []
        final = ""
        prior = None
        for turn in split_user_turns(message):
            if prior is None:
                result = await Runner.run(agent_obj, turn)
            else:
                try:
                    nxt: Any = [*list(result.to_input_list()), {"role": "user", "content": turn}]
                except Exception:
                    nxt = turn
                result = await Runner.run(agent_obj, nxt)
            prior = result
            for item in getattr(result, "new_items", ()) or ():
                name = type(item).__name__.lower()
                raw = getattr(item, "raw_item", item)
                if "toolcall" in name and "output" not in name:
                    call_id = str(getattr(raw, "call_id", getattr(raw, "id", len(pending))))
                    arguments = getattr(raw, "arguments", {}) or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"__unparsed__": arguments}
                    pending[call_id] = {
                        "tool": str(getattr(raw, "name", "")),
                        "arguments": arguments,
                    }
                elif "toolcalloutput" in name or "tooloutput" in name:
                    call_id = str(getattr(raw, "call_id", getattr(item, "call_id", "")))
                    call = pending.pop(call_id, {"tool": "", "arguments": {}})
                    steps.append({**call, "result": str(getattr(item, "output", raw))})
            final = str(getattr(result, "final_output", "") or "")
        steps.extend({**call, "result": ""} for call in pending.values())
        return {"steps": steps, "final_text": final}

    return _sync(agent)


def from_claude_sdk(client: Any) -> Callable:
    async def agent(message: str) -> dict:
        pending: dict[str, dict] = {}
        steps: list[dict] = []
        final_parts: list[str] = []
        for turn in split_user_turns(message):
            stream = client.query(prompt=turn)
            if _inspect.isawaitable(stream):
                stream = await stream
            async for event in stream:
                content = getattr(event, "content", None)
                if content is None and isinstance(event, dict):
                    content = event.get("content")
                for block in content or ():
                    kind = str(
                        (
                            block.get("type")
                            if isinstance(block, dict)
                            else getattr(block, "type", type(block).__name__)
                        )
                        or ""
                    ).lower()
                    kind = kind.replace("_", "")
                    get = (
                        block.get
                        if isinstance(block, dict)
                        else lambda key, default=None, _b=block: getattr(_b, key, default)
                    )
                    if "tooluse" in kind:
                        pending[str(get("id", ""))] = {
                            "tool": str(get("name", "")),
                            "arguments": get("input", {}) or {},
                        }
                    elif "toolresult" in kind:
                        call = pending.pop(
                            str(get("tool_use_id", "")), {"tool": "", "arguments": {}}
                        )
                        steps.append({**call, "result": str(get("content", ""))})
                    elif kind in ("text", "textblock"):
                        final_parts.append(str(get("text", "")))
        steps.extend({**call, "result": ""} for call in pending.values())
        return {"steps": steps, "final_text": "".join(final_parts)}

    return _sync(agent)


def subprocess_agent(command: list[str], *, timeout: float = 60) -> Callable:
    import subprocess

    def agent(message: str) -> dict:
        turns = split_user_turns(message)
        payload = turns[0] if len(turns) == 1 else "\n\n".join(turns)
        proc = subprocess.run(command, input=payload.encode(), capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"agent exited {proc.returncode}: {proc.stderr.decode()[:200]}")
        return json.loads(proc.stdout.decode())

    agent.__name__ = f"subprocess[{command[0]}]"
    return agent


SUPPORTED = (
    "callable",
    "http",
    "langchain",
    "langgraph",
    "openai_agents",
    "claude_sdk",
    "subprocess",
)


def detect(target: Any) -> str:
    if isinstance(target, ConnectedAgent):
        return target.transport
    if isinstance(target, str):
        if target.startswith(("http://", "https://")):
            return "http"
        if ":" in target:
            return "backend_spec"
        raise ValueError(
            f"cannot detect a transport for string {target!r}; pass an http(s) URL or a backend spec."
        )
    if isinstance(target, (list, tuple)) and target and isinstance(target[0], str):
        return "subprocess"
    if hasattr(target, "get_graph") and hasattr(target, "invoke"):
        return "langgraph"
    if hasattr(target, "invoke") and hasattr(target, "agent"):
        return "langchain"
    module = type(target).__module__
    class_name = type(target).__name__
    if module == "agents" or module.startswith("agents."):
        return "openai_agents"
    if "ClaudeSDKClient" in class_name or hasattr(target, "query"):
        return "claude_sdk"
    if callable(target) and not _inspect.isclass(target):
        return "callable"
    raise ValueError(
        f"cannot detect a transport for {type(target).__name__}; pass a callable, URL, or tools=."
    )


def resolve(
    target: Any,
    *,
    transport: str | None = None,
    tools: list | None = None,
    policy: str = "",
    execute: Callable | None = None,
    model: str | None = None,
    fault_plans: dict | None = None,
    max_turns: int = 5,
    avg_turns: float = 6,
    min_user_turns: int = 1,
    patience: str = "normal",
    turn_stats: dict | None = None,
    opening_rate: float = 0.0,
    temperature: float | None = None,
    result_shapes: dict | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
    user_model: str | None = None,
) -> tuple[Any, str]:
    if isinstance(target, ConnectedAgent):
        return target.run, target.transport
    kind = transport or detect(target)
    loop_kw: dict[str, Any] = {"max_turns": max_turns}
    if temperature is not None:
        loop_kw["temperature"] = temperature
    if kind == "callable":
        return _sync(target), kind
    if kind == "backend_spec":
        url, spec_model = parse_backend_spec(target)
        # a bring-your-own model gets the same world as hosted qwen:
        # mined result shapes and the caller's rollout timeout
        if timeout is not None:
            loop_kw["timeout"] = timeout
        if max_tokens:
            # only local_model takes a reply budget; openai_http does not
            loop_kw["max_tokens"] = int(max_tokens)
        if user_model:
            loop_kw["user_model"] = user_model
        return local_model(
            url,
            spec_model,
            tools=tools or [],
            system=policy,
            fault_plans=fault_plans,
            avg_turns=avg_turns,
            min_user_turns=min_user_turns,
            patience=patience,
            opening_rate=opening_rate,
            execute=execute,
            result_shapes=result_shapes,
            turn_stats=turn_stats,
            **loop_kw,
        ), kind
    if kind == "http":
        if not tools:
            raise ValueError("an HTTP agent needs tools=[...] (the schemas it may call)")
        return openai_http(
            target,
            model=model or "gpt-4o-mini",
            tools=tools,
            execute=execute,
            system=policy,
            **loop_kw,
        ), kind
    if kind == "langchain":
        return from_langchain(target), kind
    if kind == "langgraph":
        return from_langgraph(target), kind
    if kind == "openai_agents":
        return from_openai_agents(target), kind
    if kind == "claude_sdk":
        return from_claude_sdk(target), kind
    if kind == "subprocess":
        return subprocess_agent(list(target)), kind
    raise ValueError(f"unknown transport {kind!r}; supported: {SUPPORTED}")


def parse_claude_stream(stdout: str) -> dict:
    pending: dict[str, dict] = {}
    steps: list[dict] = []
    final_text = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "assistant":
            for block in (event.get("message") or {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    pending[str(block.get("id"))] = {
                        "tool": str(block.get("name", "")),
                        "arguments": block.get("input") or {},
                    }
        elif kind == "user":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    call = pending.pop(str(block.get("tool_use_id")), {"tool": "", "arguments": {}})
                    result = str(block.get("content"))
                    step = {**call, "result": result[:CLAUDE_CODE_RESULT_CHARS]}
                    if len(result) > CLAUDE_CODE_RESULT_CHARS:
                        # say so on the step; a silent cut is a decision nobody made
                        step["result_truncated"] = True
                        step["result_chars"] = len(result)
                    steps.append(step)
        elif kind == "result":
            final_text = str(event.get("result") or "")
    steps.extend({**call, "result": ""} for call in pending.values())
    return {"steps": steps, "final_text": final_text}


def claude_code(
    extra_args: tuple | list = (),
    *,
    cwd: str | None = None,
    max_turns: int | None = None,
    timeout: float = 300,
) -> Callable:
    import subprocess

    def agent(message: str) -> dict:
        turns = split_user_turns(message)
        prompt = turns[0] if len(turns) == 1 else "\n\n".join(turns)
        command = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose"]
        if max_turns is not None:
            command += ["--max-turns", str(max_turns)]
        command += [str(argument) for argument in extra_args]
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 and not proc.stdout.strip():
            raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[:300]}")
        return parse_claude_stream(proc.stdout)

    agent.__name__ = "claude_code"
    return agent


_READ_PREFIXES = (
    "lookup",
    "get",
    "search",
    "list",
    "read",
    "check",
    "view",
    "find",
    "show",
    "query",
    "fetch",
    "browse",
)
_MUTATION_VERBS = (
    "order",
    "send",
    "delete",
    "pay",
    "write",
    "book",
    "refund",
    "cancel",
    "place",
    "update",
    "create",
    "issue",
    "grant",
    "disable",
    "approve",
    "deny",
    "commit",
    "execute",
    "schedule",
    "charge",
    "transfer",
    "reset",
    "apply",
    "add",
    "remove",
    "export",
    "publish",
    "deploy",
    "revoke",
    "message",
    "notify",
    "submit",
    "start",
    "assign",
    "close",
    "exchange",
    "shell",
)
_USER_FIELDS = (
    "address",
    "email",
    "amount",
    "amount_usd",
    "reference",
    "phone",
    "account",
    "iban",
    "card",
    "code",
    "promo",
    "sku",
    "order_id",
    "invoice_id",
    "claim_id",
    "policy_id",
    "patient_id",
    "employee_id",
    "user_id",
    "recipient",
    "to",
    "date",
)


def _name_tokens(name: str) -> list[str]:
    import re

    return [t for t in re.split(r"[^a-z]+", (name or "").lower()) if t]


def _tool_name(schema: dict) -> str:
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    return str(fn.get("name") or schema.get("name") or "")


def _schema_from_object(tool: Any) -> dict | None:
    if isinstance(tool, dict):
        if tool.get("function") or tool.get("name"):
            return tool
        return None
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    if not name:
        return None
    desc = str(getattr(tool, "description", "") or "")[:1000]
    params = (
        getattr(tool, "params_json_schema", None)
        or getattr(tool, "args_schema", None)
        or getattr(tool, "args", None)
        or {}
    )
    if hasattr(params, "model_json_schema"):
        params = params.model_json_schema()
    elif hasattr(params, "schema") and not isinstance(params, dict):
        try:
            params = params.schema()
        except Exception:
            params = {}
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    elif (
        "properties" not in params and params and all(isinstance(v, dict) for v in params.values())
    ):
        params = {"type": "object", "properties": dict(params)}
    elif "type" not in params and "properties" not in params:
        params = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {"name": str(name), "description": desc, "parameters": params},
    }


def _tools_from_agent(agent: Any) -> list[dict]:
    if isinstance(agent, ConnectedAgent):
        return list(agent.profile.tools)
    raw = None
    for attr in ("tools", "tool_schemas", "declared_tools"):
        value = getattr(agent, attr, None)
        if value:
            raw = value
            break
    if raw is None:
        getter = getattr(agent, "get_tools", None)
        if callable(getter):
            try:
                raw = getter()
            except Exception:
                raw = None
    if not raw:
        return []
    out = []
    for item in raw:
        schema = _schema_from_object(item)
        if schema and _tool_name(schema):
            out.append(schema)
    return out


def _policy_from_agent(agent: Any) -> str:
    if isinstance(agent, ConnectedAgent):
        return agent.profile.policy
    for attr in ("instructions", "system_prompt", "system", "policy"):
        value = getattr(agent, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    inner = getattr(agent, "agent", None)
    if inner is not None and inner is not agent:
        return _policy_from_agent(inner)
    return ""


def _capabilities(tools: list[dict]) -> dict[str, list[str]]:
    caps: dict[str, list[str]] = {}
    for schema in tools:
        name = _tool_name(schema)
        tokens = _name_tokens(name)
        listed = []
        if tokens and tokens[0] in _READ_PREFIXES:
            listed.append("read")
        else:
            token_set = set(tokens)
            if any(v in token_set for v in _MUTATION_VERBS):
                listed.append("mutate")
            if any(
                v in token_set for v in ("pay", "charge", "refund", "transfer", "order", "purchase")
            ):
                listed.append("financial")
            if any(v in token_set for v in ("send", "email", "notify", "message", "publish")):
                listed.append("communication")
            if any(v in token_set for v in ("delete", "remove", "cancel", "revoke")):
                listed.append("irreversible")
        caps[name] = listed or ["other"]
    return caps


def resolve_system_prompt(
    system_prompt: str | None = None, policy: str | None = None
) -> str | None:
    """Public name is system_prompt. policy= is a silent alias."""
    if system_prompt is not None and policy is not None and str(system_prompt) != str(policy):
        raise ValueError("pass system_prompt= or policy=, not both")
    if system_prompt is not None:
        return system_prompt
    return policy


def _constraints(tools: list[dict], policy: str) -> dict[str, Any]:
    required_user = []
    for schema in tools:
        fn = schema.get("function", schema) if isinstance(schema, dict) else {}
        props = (fn.get("parameters") or {}).get("properties") or {}
        for key in props:
            low = str(key).lower()
            if (
                low.endswith("_id")
                or low == "id"
                or any(field in low or low.endswith(field) for field in _USER_FIELDS)
            ) and key not in required_user:
                required_user.append(key)
    mutating = [name for name, caps in _capabilities(tools).items() if "read" not in caps]
    return {
        "mutating_tools": mutating,
        "read_tools": [n for n in _capabilities(tools) if n not in set(mutating)],
        "required_user_fields": required_user,
        "policy_present": bool(str(policy).strip()),
    }


@dataclass
class AgentProfile:
    tools: list[dict] = field(default_factory=list)
    policy: str = ""
    capabilities: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    transport: str = "callable"
    name: str = ""
    # what doing the job means, for the judge; empty means conduct floor only
    rubric: str = ""

    @property
    def system_prompt(self) -> str:
        return self.policy


@dataclass
class ConnectedAgent:
    run: Callable
    profile: AgentProfile
    transport: str = "callable"

    def __call__(self, message: str) -> dict:
        return self.run(message)


def _with_parameters(schema: dict) -> dict:
    """Anthropic's ``input_schema`` read as ``parameters``.

    Tools arrive in three shapes: the OpenAI envelope, a bare
    ``{name, description, parameters}``, and Anthropic's
    ``{name, description, input_schema}``. Everything downstream reads
    ``parameters``, so the Anthropic key is copied onto it once, here,
    instead of being missed one call site at a time (the wire tools built
    for a model-backed agent dropped it, and the agent saw a tool with no
    arguments). The caller's dict is left alone.
    """
    envelope = isinstance(schema.get("function"), dict)
    fn = schema["function"] if envelope else schema
    if not isinstance(fn.get("input_schema"), dict) or fn.get("parameters") is not None:
        return schema
    fn = dict(fn)
    fn["parameters"] = fn["input_schema"]
    if not envelope:
        return fn
    out = dict(schema)
    out["function"] = fn
    return out


def _merge_tool_lists(base: list | None, extra: list | None) -> list[dict]:
    merged: dict[str, dict] = {}
    for schema in list(base or []) + list(extra or []):
        if not isinstance(schema, dict):
            continue
        name = _tool_name(schema)
        if name:
            merged[name] = _with_parameters(schema)
    return list(merged.values())


def inspect(
    agent: Any,
    *,
    tools: list[dict] | None = None,
    system_prompt: str | None = None,
    policy: str | None = None,
    transport: str | None = None,
) -> AgentProfile:
    """Read tools and system prompt off the agent; caller extras are merged in."""
    policy = resolve_system_prompt(system_prompt, policy)
    if agent is None:
        derived_tools = _merge_tool_lists([], tools)
        derived_policy = str(policy or "")
        return AgentProfile(
            tools=derived_tools,
            policy=derived_policy,
            capabilities=_capabilities(derived_tools),
            constraints=_constraints(derived_tools, derived_policy),
            transport="hosted",
            name="hosted-qwen",
        )
    kind = "callable"
    with contextlib.suppress(ValueError):
        kind = transport or detect(agent)
    derived_tools = _merge_tool_lists(_tools_from_agent(agent), tools)
    agent_policy = _policy_from_agent(agent)
    extra_policy = str(policy).strip() if policy is not None else ""
    if extra_policy and extra_policy not in str(agent_policy or ""):
        derived_policy = (str(agent_policy).rstrip() + "\n" + extra_policy).strip()
    elif extra_policy:
        derived_policy = extra_policy
    else:
        derived_policy = str(agent_policy or "")
    name = getattr(agent, "name", None) or getattr(agent, "__name__", None) or type(agent).__name__
    return AgentProfile(
        tools=derived_tools,
        policy=derived_policy,
        capabilities=_capabilities(derived_tools),
        constraints=_constraints(derived_tools, derived_policy),
        transport=kind,
        name=str(name),
    )


def connect(
    agent: Any,
    *,
    tools: list[dict] | None = None,
    system_prompt: str | None = None,
    policy: str | None = None,
    transport: str | None = None,
    execute: Callable | None = None,
    model: str | None = None,
) -> ConnectedAgent:
    profile = inspect(
        agent, tools=tools, system_prompt=system_prompt, policy=policy, transport=transport
    )
    runner, kind = resolve(
        agent,
        transport=transport or profile.transport,
        tools=profile.tools,
        policy=profile.policy,
        execute=execute,
        model=model,
    )
    profile.transport = kind
    return ConnectedAgent(run=runner, profile=profile, transport=kind)
