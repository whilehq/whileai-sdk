"""Production traces to a focused coverage grid. Simulation, version two.

Version one is cold start: ``simulate`` invents diverse situations from the
agent spec alone. Version two starts from traces of the deployed agent (real
production rows, or a slice of simulations set aside as pseudo production):
mine what actually happened, then aim the covering grid at the observed
tools, faults, and worlds instead of the whole space.

Leakage rule: source traces never enter the generated dataset. They shape
the grid and nothing else. ``leakage_report`` / ``drop_leaky_rows`` verify
no generated prompt is a near copy of a source trace, so a trace held out
for evaluation stays out of training.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..defaults import (
    TRACE_EXEMPLAR_DICT_KEYS,
    TRACE_EXEMPLAR_LIST_ITEMS,
    TRACE_EXEMPLAR_MAX_CHARS,
    TRACE_EXEMPLAR_STRING_CHARS,
    TRACE_EXEMPLARS_PER_TOOL,
    TRACE_EXPLORATION_FLOOR,
    TRACE_EXPLORATION_MAX,
    TRACE_LEAK_EXAMPLES,
    TRACE_LEAK_THRESHOLD,
    TRACE_MIN_SUPPORT,
    TRACE_PSEUDO_PRODUCTION_FRACTION,
    TRACE_RATE_FLOOR,
    TRACE_REPORT_LIST_CAP,
    TRACE_STATE_PRIORITY,
    TRACE_SUPPORT_SATURATION,
    TRACE_TASK_HASH_CHARS,
)
from ..generate.embeddings import resolve_embedder
from ..generate.scenarios import build_dimensions
from ..score.grading import NO_FAULT, _fault_from_result, behavior_signature, trace_fault
from ..tools import schemas as _tool_schemas

#: Observed fault chip -> the grid axis and value that reproduces it. The
#: chip names are what ``trace_fault`` reads off a result; the values are
#: the ``tool_condition`` / ``world_state`` axis values ``build_dimensions``
#: emits. ``error`` maps to ``timeout`` because a generic error has no
#: fault of its own on the grid. Extend with
#: ``dimensions_from_traces(fault_to_axis={**FAULT_TO_AXIS, ...})``.
FAULT_TO_AXIS: dict[str, tuple[str, str]] = {
    "timeout": ("tool_condition", "timeout"),
    "malformed": ("tool_condition", "malformed_result"),
    "stale": ("tool_condition", "stale_result"),
    "deny": ("tool_condition", "permission_denied"),
    "error": ("tool_condition", "timeout"),
    "not_found": ("world_state", "entity missing"),
    "already_done": ("world_state", "entity already acted on"),
}
_FAULT_TO_AXIS = FAULT_TO_AXIS

#: Row keys read for a 0/1 label, in order: the SDK's ``reward``, then the
#: advisory judge label an unlabelled row may carry.
REWARD_KEYS = ("reward", "qwen_reward")
#: Axis values that mean "nothing went wrong" and stay in every aimed axis
#: as the contrast (they never gain emphasis).
CLEAN_CONDITION = "success"
CLEAN_WORLD = "entity exists"
#: Tool-axis entries that are not tools and always keep their place.
SPECIAL_TOOLS = frozenset({"unrelated", "multi_tool"})
#: Row keys that do not carry a world state.
UNKNOWN_WORLDS = frozenset({"unspecified", "unknown"})


def _binary_reward(row: dict) -> int | None:
    for key in REWARD_KEYS:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            # External rows label with True/False; a skipped False would
            # hide that trace's flaw signal from mining.
            value = int(value)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == 0.0:
            return 0
        if number == 1.0:
            return 1
    return None


def mine_traces(rows: Sequence[dict]) -> dict[str, Any]:
    """What the deployed agent actually did, counted for grid focusing.

    ``flaw_rows`` is any row with an observed fault or a 0 label. Those are
    the behaviors worth simulating more of.
    """
    tools: dict[str, dict[str, int]] = {}
    faults: dict[str, int] = {}
    worlds: dict[str, int] = {}
    behaviors: set[str] = set()
    flaw_rows: list[int] = []
    asks: list[str] = []
    seen_asks: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        fault = trace_fault(row)
        reward = _binary_reward(row)
        behaviors.add(behavior_signature(row))
        if fault != NO_FAULT:
            faults[fault] = faults.get(fault, 0) + 1
        world = str(row.get("world_state") or "").strip()
        if world and world not in UNKNOWN_WORLDS:
            worlds[world] = worlds.get(world, 0) + 1
        flawed = fault != NO_FAULT or reward == 0
        if flawed:
            flaw_rows.append(i)
        for step in row.get("steps") or []:
            if not isinstance(step, dict) or not step.get("tool"):
                continue
            slot = tools.setdefault(str(step["tool"]), {"n": 0, "fault_n": 0})
            slot["n"] += 1
            result = step.get("result")
            if result is not None and _fault_from_result(result):
                slot["fault_n"] += 1
        prompt = str(row.get("prompt") or "").strip()
        if prompt and prompt not in seen_asks:
            seen_asks.add(prompt)
            asks.append(prompt)
    return {
        "n": len(rows),
        "tools": tools,
        "faults": faults,
        "world_states": worlds,
        "unique_behaviors": len(behaviors),
        "flaw_rows": flaw_rows,
        "asks": asks,
    }


# Trace-grounded result exemplars. Agent specs declare tool inputs but
# almost never result shapes, so invented results drift from the real
# product. Traces carry the real payloads; a few per tool become shape
# templates. The counts and caps live in defaults.py (TRACE_EXEMPLAR_*)
# and are keywords on ``mine_result_exemplars``.
_EXEMPLARS_PER_TOOL = TRACE_EXEMPLARS_PER_TOOL
_EXEMPLAR_MAX_CHARS = TRACE_EXEMPLAR_MAX_CHARS
#: Shape-key detail: record keys listed, and the length band (in hundreds
#: of characters, capped) a text result is bucketed by.
_SHAPE_KEY_KEYS = 10
_SHAPE_TEXT_BANDS = 4


def _exemplar_value(result: Any) -> Any:
    """Structured view of a step result; JSON-in-a-string is parsed."""
    if isinstance(result, str):
        try:
            return json.loads(result)
        except ValueError:
            return result
    return result


def _exemplar_shape_key(value: Any) -> str:
    """Coarse shape identity: keys for records, item shape for lists,
    length band for text. Two results with the same key teach nothing new."""
    if isinstance(value, dict):
        return "dict:" + ",".join(sorted(str(k) for k in value)[:_SHAPE_KEY_KEYS])
    if isinstance(value, list):
        return "list:" + (_exemplar_shape_key(value[0]) if value else "empty")
    if isinstance(value, str):
        return f"str:{min(len(value) // 100, _SHAPE_TEXT_BANDS)}"
    return type(value).__name__


def _trim_exemplar(
    value: Any,
    *,
    string_chars: int = TRACE_EXEMPLAR_STRING_CHARS,
    list_items: int = TRACE_EXEMPLAR_LIST_ITEMS,
    dict_keys: int = TRACE_EXEMPLAR_DICT_KEYS,
) -> Any:
    """Shrink a payload toward the serialized cap without breaking JSON."""
    if isinstance(value, str):
        return value if len(value) <= string_chars else value[: string_chars - 3] + "..."
    if isinstance(value, list):
        return [_trim_exemplar(v) for v in value[:list_items]]
    if isinstance(value, dict):
        return {str(k): _trim_exemplar(v) for k, v in list(value.items())[:dict_keys]}
    return value


def mine_result_exemplars(
    rows: Sequence[dict],
    *,
    per_tool: int = TRACE_EXEMPLARS_PER_TOOL,
    max_chars: int = TRACE_EXEMPLAR_MAX_CHARS,
) -> dict[str, list]:
    """Up to ``per_tool`` real result payloads per tool, shape-diverse.

    Sibling of ``mine_traces``: same rows in, but this collects what the
    tools RETURNED, for grounding invented results. Faulted and empty
    results are skipped (they show the fault axis, not the success
    shape), a result whose shape is already kept is skipped, and each
    exemplar is trimmed to serialize within ``max_chars`` (about a quarter
    of that in tokens; the default keeps three per tool under a few
    hundred tokens of writer prompt).
    """
    out: dict[str, list] = {}
    seen: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for step in row.get("steps") or []:
            if not isinstance(step, dict) or not step.get("tool"):
                continue
            step = _normalize_step(step)
            if "result" not in step:
                continue
            name = str(step["tool"])
            value = _exemplar_value(step["result"])
            if not value:
                continue
            if isinstance(value, dict) and not (set(value) - {"status", "ok"}):
                # A bare status carries no shape worth copying.
                continue
            if trace_fault({"steps": [{"tool": name, "result": value}]}) != NO_FAULT:
                continue
            kept = out.setdefault(name, [])
            if len(kept) >= per_tool:
                continue
            key = _exemplar_shape_key(value)
            if key in seen.setdefault(name, set()):
                continue
            trimmed = _trim_exemplar(value)
            try:
                if len(json.dumps(trimmed, default=str)) > int(max_chars):
                    continue
            except (TypeError, ValueError):
                continue
            seen[name].add(key)
            kept.append(trimmed)
    return {name: kept for name, kept in out.items() if kept}


def exemplar_result_shapes(exemplars: dict[str, list]) -> dict[str, dict]:
    """Exemplars in ``write_result_shapes`` form: one template per tool.

    The sandbox fills one dict template per call (``MockEnvironment``),
    so the first record-shaped exemplar becomes that template; a bare
    list of records is wrapped the way ``_parse_result_shapes`` wraps
    one. Non-record exemplars stay report-only.
    """
    out: dict[str, dict] = {}
    for name, values in (exemplars or {}).items():
        for value in values:
            if isinstance(value, dict) and value:
                out[name] = dict(value)
                break
            if isinstance(value, list) and value and isinstance(value[0], dict):
                out[name] = {"results": [dict(value[0])]}
                break
    return out


def dimensions_from_traces(
    rows: Sequence[dict],
    tools: list[dict],
    policy: str = "",
    *,
    broaden: bool = True,
    fault_to_axis: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, list[str]]:
    """Coverage axes aimed at behaviors seen in ``rows``.

    Starts from ``build_dimensions`` for this agent so every value is one
    the writer and sandbox understand. The tool axis puts observed failing
    tools first; ``broaden=False`` drops tools the traces never touched
    (keeping the base specials such as ``unrelated``), so a run spends its
    budget near the flaws instead of boiling the ocean. Fault and world
    axes always keep their clean value: contrast needs passing rows too.
    ``fault_to_axis`` maps an observed fault chip to the axis value that
    reproduces it (``FAULT_TO_AXIS`` by default).
    """
    mapping = FAULT_TO_AXIS if fault_to_axis is None else dict(fault_to_axis)
    base = build_dimensions(tools, policy)
    mined = mine_traces(rows)
    observed = mined["tools"]

    def _tool_rank(name: str) -> tuple:
        slot = observed.get(name) or {}
        return (-int(slot.get("fault_n", 0)), -int(slot.get("n", 0)), name)

    base_tools = list(base.get("tool") or [])
    specials = [t for t in base_tools if t in SPECIAL_TOOLS]
    real = [t for t in base_tools if t not in specials]
    seen = [t for t in real if t in observed]
    unseen = [t for t in real if t not in observed]
    seen.sort(key=_tool_rank)
    tool_axis = seen + (unseen if broaden else []) + specials if seen else base_tools

    conditions = list(base.get("tool_condition") or [])
    worlds = list(base.get("world_state") or [])
    focus_conditions: list[str] = []
    focus_worlds: list[str] = []
    for name in sorted(mined["faults"], key=mined["faults"].get, reverse=True):
        axis_value = mapping.get(name)
        if not axis_value:
            if name in conditions:
                focus_conditions.append(name)
            continue
        axis, value = axis_value
        if axis == "tool_condition" and value in conditions:
            focus_conditions.append(value)
        elif axis == "world_state" and value in worlds:
            focus_worlds.append(value)
    for world in sorted(mined["world_states"], key=mined["world_states"].get, reverse=True):
        if world in worlds and world not in focus_worlds:
            focus_worlds.append(world)
    if focus_conditions:
        conditions = [CLEAN_CONDITION] + [c for c in focus_conditions if c != CLEAN_CONDITION]
    if focus_worlds:
        clean = [w for w in (CLEAN_WORLD,) if w in worlds]
        worlds = clean + [w for w in focus_worlds if w not in clean]

    out = dict(base)
    out["tool"] = tool_axis
    out["tool_condition"] = conditions
    out["world_state"] = worlds
    return out


def _task_key(row: dict, index: int) -> tuple[str, object]:
    """What makes two rows the same task for splitting purposes.

    The unit every report counts in: ``task_key`` (``scenario_id`` when
    the row has one, else ``task_id``, else the prompt), so repeats and
    rephrasings of one situation land on the same side and the held-out
    slice is disjoint from train in the unit ``pass_at``, ``compare_runs``
    and ``delta_report`` group by (#268). Splitting on the prompt alone
    left 16 of 28 held-out situations in train, and ``decontaminate``
    cannot see that because it compares prompts. A row with no key is its
    own task, so rows that merely lack one are not swept onto one side.
    """
    from ..score.stats import task_key

    key = task_key(row) if isinstance(row, dict) else ""
    if key and str(key).strip():
        return ("task", str(key))
    return ("index", index)


def split_pseudo_production(
    rows: Sequence[dict], *, fraction: float = TRACE_PSEUDO_PRODUCTION_FRACTION, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Set aside a pseudo-production slice; the rest stays for training.

    The split is by task, not by row: every row sharing a ``task_key``
    (the ``scenario_id``, else the prompt) lands on the same side, so the
    held-out slice is disjoint from the training side in the unit every
    report groups by, not just prompt-disjoint. Splitting
    by row is not enough — under ``mode="rl"`` with ``repeats=k`` each
    prompt has k rows, and scattering siblings across the two sides trains
    the student on every prompt it is then evaluated on.

    Every unique flaw signature (fault name plus behavior shape) sends its
    task to the production side first, so the held-out slice contains each
    distinct failure at least once. ``fraction`` is still counted in rows,
    but whole tasks are added, so the slice can overshoot it by up to the
    size of one task. Deterministic in ``seed``.
    """
    items = [row for row in rows if isinstance(row, dict)]
    n = len(items)
    if n == 0:
        return [], []
    target = max(1, min(n, round(max(0.0, float(fraction)) * n))) if fraction > 0 else 0
    tasks: dict[tuple[str, object], list[int]] = {}
    task_of: list[tuple[str, object]] = []
    for i, row in enumerate(items):
        key = _task_key(row, i)
        tasks.setdefault(key, []).append(i)
        task_of.append(key)
    seen_flaws: set[tuple[str, str]] = set()
    chosen: set[tuple[str, object]] = set()
    held = 0
    for i, row in enumerate(items):
        fault = trace_fault(row)
        if fault == NO_FAULT and _binary_reward(row) != 0:
            continue
        key = (fault, behavior_signature(row))
        if key in seen_flaws:
            continue
        seen_flaws.add(key)
        if task_of[i] not in chosen:
            chosen.add(task_of[i])
            held += len(tasks[task_of[i]])
    if held < target:
        rest = [task for task in tasks if task not in chosen]
        rest.sort(
            key=lambda task: hashlib.sha256(
                f"{seed}:{tasks[task][0]}:"
                f"{str(items[tasks[task][0]].get('prompt') or '')[:TRACE_TASK_HASH_CHARS]}".encode()
            ).hexdigest()
        )
        for task in rest:
            if held >= target:
                break
            chosen.add(task)
            held += len(tasks[task])
    production = [items[i] for i in range(n) if task_of[i] in chosen]
    remainder = [items[i] for i in range(n) if task_of[i] not in chosen]
    return production, remainder


def flaw_rows(rows: Sequence[dict]) -> list[dict]:
    """Rows with an observed fault or a 0 label: the next round's traces.

    The hill-climb loop feeds a round's failures back into
    ``simulate(traces=flaw_rows(evaluated))`` so the next batch aims at
    what the agent still gets wrong.
    """
    mined = mine_traces(rows)
    keep = set(mined["flaw_rows"])
    items = [row for row in rows if isinstance(row, dict)]
    return [row for i, row in enumerate(items) if i in keep]


def _prompt_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("prompt") or "").strip()
    return str(item or "").strip()


def _leak_sources(sources: Any) -> list[Any]:
    """The source texts of ``leakage_report`` / ``drop_leaky_rows`` as one
    flat list of rows and strings, whichever documented shape came in.

    ``decontaminate(against=[holdout])`` takes a list of sources, so a
    reader writes ``sources=[holdout]`` here too. Before this normaliser
    that list-of-lists was embedded as the ``repr`` of a row list, scored
    0.83 against a byte-identical holdout row and reported ``n_leaky: 0``
    (#479). A ``list``/``tuple`` element is never a text, so it is flattened
    exactly; a bare row dict is wrapped; dicts and strings pass through.
    """
    if sources is None:
        return []
    if isinstance(sources, (dict, str)):
        return [sources]
    flat: list[Any] = []
    for item in sources:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _leak_flags(
    generated: Sequence[Any],
    sources: Sequence[Any],
    *,
    threshold: float,
    embedder: Any,
    examples: int = TRACE_LEAK_EXAMPLES,
) -> tuple[list[bool], dict[str, Any]]:
    gen_texts = [_prompt_of(item) for item in generated]
    src_texts = [t for t in (_prompt_of(item) for item in _leak_sources(sources)) if t]
    flags = [False] * len(gen_texts)
    report: dict[str, Any] = {
        "n": len(gen_texts),
        "n_sources": len(src_texts),
        "threshold": float(threshold),
        "n_leaky": 0,
        "max_similarity": 0.0,
        "leaky": [],
    }
    if not gen_texts or not src_texts:
        return flags, report
    resolved = resolve_embedder(embedder)
    src_norm = {" ".join(t.lower().split()) for t in src_texts}
    src_vecs = resolved.embed(src_texts)
    gen_vecs = resolved.embed([t or " " for t in gen_texts])
    for i, vec in enumerate(gen_vecs):
        best, best_j = 0.0, -1
        for j, src in enumerate(src_vecs):
            sim = _cosine(vec, src)
            if sim > best:
                best, best_j = sim, j
        if " ".join(gen_texts[i].lower().split()) in src_norm:
            best = 1.0
        # Clamped: float cosine drift printed 1.0000000000000002 in a
        # report a customer reads.
        report["max_similarity"] = min(1.0, max(report["max_similarity"], best))
        if best >= min(float(threshold), 1.0):
            flags[i] = True
            report["n_leaky"] += 1
            if len(report["leaky"]) < int(examples):
                report["leaky"].append({"row": i, "source": best_j, "similarity": round(best, 4)})
    return flags, report


def leakage_report(
    generated: Sequence[Any],
    sources: Sequence[Any],
    *,
    threshold: float = TRACE_LEAK_THRESHOLD,
    embedder: Any = "hash",
    examples: int = TRACE_LEAK_EXAMPLES,
) -> dict[str, Any]:
    """Near-copy check of generated prompts against source traces.

    A generated row whose prompt sits at or above ``threshold`` cosine
    similarity to any source prompt is flagged (0.9 by default: the 8-gram
    exact-overlap test of Lambert 2025, chapter Evaluation, with a small
    paraphrase allowance). Exact matches always flag, whatever the embedder
    thinks. ``leaky`` lists the first ``examples`` offenders; ``n_leaky`` is
    the full count.

    ``sources`` takes the same shapes as ``decontaminate(against=...)``: a
    list of rows, a list of row lists (``[holdout]``, flattened), a list of
    prompt strings, or a single row. Each shape gives the same report.
    """
    return _leak_flags(
        generated, sources, threshold=threshold, embedder=embedder, examples=examples
    )[1]


def drop_leaky_rows(
    rows: Sequence[dict],
    sources: Sequence[Any],
    *,
    threshold: float = TRACE_LEAK_THRESHOLD,
    embedder: Any = "hash",
    examples: int = TRACE_LEAK_EXAMPLES,
) -> tuple[list[dict], dict[str, Any]]:
    """Kept rows plus the report. Flagged rows are removed, not rewritten.

    ``sources`` takes the same shapes as ``decontaminate(against=...)``: a
    list of rows, a list of row lists (``[holdout]``, flattened), a list of
    prompt strings, or a single row. A row byte-identical to a source is
    always dropped, under every shape and whatever the embedder thinks.

    >>> holdout = [{"prompt": "Cancel order 9911.", "task_id": "h1"}]
    >>> train = [{"prompt": "Cancel order 9911.", "task_id": "t1"}]
    >>> kept, report = drop_leaky_rows(train, sources=[holdout])
    >>> len(kept), report["n_leaky"], report["max_similarity"]
    (0, 1, 1.0)
    """
    flags, report = _leak_flags(
        rows, sources, threshold=threshold, embedder=embedder, examples=examples
    )
    kept = [row for row, bad in zip(rows, flags) if not bad]
    report["n_dropped"] = len(rows) - len(kept)
    return kept, report


#: What a tool drafted from traces says about itself, and the JSON type its
#: arguments get. Traces carry values, not schemas, so every argument is a
#: string until the caller edits the draft (``infer_harness`` reads the
#: observed values for a type; this union keeps the looser shape).
OBSERVED_TOOL_DESCRIPTION = "{name}, observed in this agent's traces"
OBSERVED_ARG_TYPE = "string"


def tools_from_traces(traces: Sequence[dict]) -> list[dict]:
    """The agent's tool surface, read off the calls the traces contain.

    Argument names are unioned across every observed call, so a tool called
    with different arguments in different traces ends up with all of them.
    """
    seen: dict[str, set[str]] = {}
    for row in traces or ():
        if not isinstance(row, dict):
            continue
        for step in row.get("steps") or ():
            name = isinstance(step, dict) and step.get("tool")
            if not name:
                continue
            args = step.get("arguments")
            seen.setdefault(str(name), set()).update(
                str(k) for k in (args or {}) if isinstance(args, dict)
            )
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": OBSERVED_TOOL_DESCRIPTION.format(name=name),
                "parameters": {
                    "type": "object",
                    "properties": {arg: {"type": OBSERVED_ARG_TYPE} for arg in sorted(args)},
                },
            },
        }
        for name, args in sorted(seen.items())
    ]


def simulate_from_traces(
    traces: Sequence[dict],
    agent: Any = None,
    *,
    tools: list[dict] | None = None,
    policy: str = "",
    mode: str = "rl",
    **kwargs: Any,
):
    """Alias for ``simulate(agent, traces=...)``: same grid focus and
    leakage gate, for callers who start from the traces.

    With no agent, tools or policy, the tool surface is read from the traces
    themselves, so handing over graded telemetry is enough to start.
    """
    from whileai.simulations import simulate as _simulate

    if agent is None and not tools and not policy:
        # a path is as valid a source here as rows, and the tools live inside
        tools = tools_from_traces(load_traces(traces)) or None
    return _simulate(
        agent, tools=tools, system_prompt=policy or None, mode=mode, traces=traces, **kwargs
    )


# --- canonical trace input --------------------------------------------------
#
# The one public trajectory schema everything trace-shaped normalizes to:
#
#     prompt:     str            first user ask
#     steps:      list of {"user": str}
#                       | {"tool": str, "arguments": dict, "result": Any}
#                       | {"text": str}                  agent turns
#     final_text: str            the agent's last message
#     reward:     0 | 1          OPTIONAL; ungraded traces are first-class
#
# Every other key carries through untouched. ``simulate(traces=...)``
# accepts anything ``load_traces`` accepts: While simulation rows, eval
# rollouts, raw JSONL exports, ``rows_from_otel`` output, graded or not.

#: Row keys read, in order, for the ask, the step list and the final reply.
#: The spellings the exporters we have met use; extend by editing the
#: tuples before calling ``load_traces``.
PROMPT_KEYS = ("prompt", "question", "input", "task", "ask")
STEP_KEYS = ("steps", "tool_trace", "trace")
FINAL_KEYS = ("final_text", "final", "output", "response", "answer")


#: Names a tool step's argument and result fields arrive under. The platform's
#: trace ingest writes `input`/`output`; OpenAI-style exports write
#: `arguments`/`result`. Renaming them here rather than teaching every consumer
#: both spellings, because the consumers that only knew one did not fail
#: loudly: `tool_call_roundtrip` reported "checked: 0" on a whole dataset of
#: ingested traces and every row passed the export gate without being looked at.
ARG_KEYS = ("arguments", "input", "args", "parameters")
RESULT_KEYS = ("result", "output", "response")
_PROMPT_KEYS, _STEP_KEYS, _FINAL_KEYS, _ARG_KEYS, _RESULT_KEYS = (
    PROMPT_KEYS,
    STEP_KEYS,
    FINAL_KEYS,
    ARG_KEYS,
    RESULT_KEYS,
)

#: How a tool result is bound to its call, in order: by call id (the only
#: binding that survives parallel calls answered out of order), then by tool
#: name, then to the first unfilled call (FIFO). Fixed, not a knob: any other
#: order (last-unfilled, name before id) swaps payloads between parallel
#: calls answered in order, and mining then blames the wrong tool. An orphan
#: result becomes its own step so its fault still reaches the miner.
RESULT_BINDING = ("id", "name", "fifo")


def _normalize_step(step: dict) -> dict:
    """One trajectory step in the canonical spelling.

    Only tool steps are touched. A `{"user": ...}` or `{"text": ...}` step has
    no arguments or result, and `input` on a non-tool step is somebody else's
    field.
    """
    if "tool" not in step:
        return step
    out = dict(step)
    for canonical, aliases in (("arguments", ARG_KEYS), ("result", RESULT_KEYS)):
        if canonical in out:
            continue
        source = next((k for k in aliases if k in out), None)
        if source is not None:
            out[canonical] = out.pop(source)
    return out


def _block_text(value: Any) -> str:
    """Flatten a content field that is a plain string or a block list.

    Anthropic-shaped messages carry ``content`` as a list of typed blocks.
    Stringifying that list yields a Python repr, which is what used to reach
    the writer as the agent's turn.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            str(b.get("text") or "")
            for b in value
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return "" if value is None else str(value)


def _steps_from_messages(messages: Sequence[dict]) -> list[dict]:
    steps: list[dict] = []
    by_id: dict[str, dict] = {}

    def _attach(result: Any, name: str, call_id: str = "") -> None:
        # RESULT_BINDING: id, then name, then FIFO. See the constant for why
        # the order is fixed.
        target = by_id.pop(call_id, None) if call_id else None
        if target is not None and "result" in target:
            target = None
        if target is None:
            unfilled = [s for s in steps if "tool" in s and "result" not in s]
            if name:
                target = next((s for s in unfilled if s.get("tool") == name), None)
            if target is None and unfilled:
                target = unfilled[0]
        if target is None:
            steps.append({"tool": name, "arguments": {}, "result": result})
            return
        target["result"] = result
        for key, step in list(by_id.items()):
            if step is target:
                by_id.pop(key)

    def _coerce_result(value: Any) -> Any:
        text = value if isinstance(value, str) else _block_text(value)
        with contextlib.suppress(ValueError, TypeError):
            return json.loads(text)
        return text

    def _errored(result: Any) -> Any:
        # An Anthropic tool_result carries failure as `is_error: true` next to
        # the text, not inside it. Grading keys a fault on `status` / `error`
        # (grading._step_faulted), so a bare string would read as success.
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        if isinstance(result, dict):
            out = dict(result)
            out.setdefault("error", text)
            out.setdefault("status", "error")
            return out
        return {"error": text, "status": "error"}

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        raw_content = message.get("content")
        blocks = raw_content if isinstance(raw_content, list) else []
        # A tool_result block rides on a user-role message in the Anthropic
        # dialect; it is a tool answer, not something a person said.
        results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
        content = _block_text(raw_content)
        if role == "user":
            for block in results:
                result = _coerce_result(block.get("content"))
                if block.get("is_error"):
                    result = _errored(result)
                _attach(
                    result,
                    str(block.get("name") or ""),
                    str(block.get("tool_use_id") or ""),
                )
            if content or not results:
                steps.append({"user": content})
        elif role == "assistant":
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    call_step: dict[str, Any] = {
                        "tool": str(block.get("name") or ""),
                        "arguments": block.get("input")
                        if isinstance(block.get("input"), dict)
                        else {},
                    }
                    steps.append(call_step)
                    if block.get("id"):
                        by_id[str(block["id"])] = call_step
            for call in message.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call.get("function"), dict) else call
                raw = (fn or {}).get("arguments")
                if isinstance(raw, str):
                    with contextlib.suppress(ValueError):
                        raw = json.loads(raw)
                step = {
                    "tool": str((fn or {}).get("name") or ""),
                    "arguments": raw if isinstance(raw, dict) else {},
                }
                steps.append(step)
                if call.get("id"):
                    by_id[str(call["id"])] = step
            if content:
                steps.append({"text": content})
        elif role == "tool":
            tool_result: Any = _coerce_result(raw_content)
            _attach(
                tool_result,
                str(message.get("name") or ""),
                str(message.get("tool_call_id") or message.get("tool_use_id") or ""),
            )
    return steps


def _coerce_reward(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number in (0.0, 1.0) else None


def load_traces(source) -> list[dict]:
    """Normalize any supported trace source to the one trajectory schema the SDK reads.

    Reach for it when you have traces from somewhere else (a production
    log, an eval harness, an OpenAI-style ``messages`` export) and want
    ``simulate(traces=...)``, ``evaluate`` or ``decontaminate`` to read
    them. It returns a list of dicts in the canonical schema: ``prompt``
    (the first user ask), ``steps`` (a list of ``{"user": str}``,
    ``{"tool": str, "arguments": dict, "result": Any}`` and
    ``{"text": str}`` agent turns), ``final_text`` (the agent's last
    message) and, optionally, ``reward`` (0 or 1). Every other key
    carries through untouched, and ungraded traces are first-class.

    * ``source``: a JSONL path or an iterable of dicts. Rows carrying
      ``tool_trace``/``trace`` instead of ``steps``,
      ``final``/``output``/``response`` instead of ``final_text``, or only
      OpenAI-style ``messages`` are converted (``PROMPT_KEYS``,
      ``STEP_KEYS``, ``FINAL_KEYS``, ``ARG_KEYS`` and ``RESULT_KEYS`` list
      the spellings read); ``reward`` is kept only when it coerces cleanly
      to 0 or 1, and its absence is fine. Rows that are not dicts or carry
      neither an ask nor any steps are dropped.

    >>> rows = wai.load_traces([{"question": "Where is order 4473?", "output": "Shipped."}])
    >>> rows[0]["prompt"], rows[0]["final_text"]
    ('Where is order 4473?', 'Shipped.')
    """
    from pathlib import Path as _Path

    if isinstance(source, (str, _Path)):
        from ..score.quality import load_jsonl

        rows = load_jsonl(source)
    else:
        rows = list(source)
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row = dict(row)
        # An empty steps list is absence, not content: real exports emit
        # steps: [] next to a populated messages/tool_trace field.
        steps = next((row[k] for k in STEP_KEYS if isinstance(row.get(k), list) and row[k]), None)
        if steps is None and isinstance(row.get("messages"), list):
            steps = _steps_from_messages(row["messages"])
        row["steps"] = [_normalize_step(s) for s in (steps or []) if isinstance(s, dict)]
        prompt = next((str(row[k]) for k in PROMPT_KEYS if row.get(k)), "")
        if not prompt:
            prompt = next((str(s["user"]) for s in row["steps"] if "user" in s), "")
        row["prompt"] = prompt
        final = next((str(row[k]) for k in FINAL_KEYS if row.get(k)), "")
        if not final:
            final = next((str(s["text"]) for s in reversed(row["steps"]) if "text" in s), "")
        row["final_text"] = final
        if "reward" in row:
            reward = _coerce_reward(row["reward"])
            if reward is None:
                row.pop("reward")
            else:
                row["reward"] = reward
        if row["prompt"] or row["steps"]:
            out.append(row)
    return out


def opening_share(rows: Sequence[dict]) -> float:
    """Share of traces whose conversation opens with the assistant.

    Reads the raw ``messages`` field (kept by ``load_traces``); rows
    without messages count as user-opened. This is the evidence the
    ``opening="auto"`` topology axis resolves against.
    """
    items = [r for r in rows if isinstance(r, dict)]
    if not items:
        return 0.0
    agent_first = 0
    for row in items:
        msgs = row.get("messages")
        if (
            isinstance(msgs, list)
            and msgs
            and isinstance(msgs[0], dict)
            and str(msgs[0].get("role") or "") == "assistant"
        ):
            agent_first += 1
    return agent_first / len(items)


def trace_report(traces, tools: list[dict] | None = None, policy: str = "") -> dict[str, Any]:
    """What these traces contain and what they will aim generation at.

    Run before ``simulate(traces=...)``. With ``tools`` (and optionally
    ``policy``) the report also computes the actual grid emphasis: which
    axis values move forward in the coverage grid because of these traces.
    Reward stays optional; ungraded counts are reported, never required.
    ``advisory_labels`` counts rows carrying only a ``qwen_reward``: those
    labels do steer trace mining, so they are disclosed, not hidden under
    "ungraded". ``dropped`` counts input rows that carried no usable
    signal and were discarded by normalization.
    """
    tools = _tool_schemas(tools)
    from pathlib import Path as _Path

    if isinstance(traces, (str, _Path)):
        from ..score.quality import load_jsonl

        raw = load_jsonl(traces)
    else:
        raw = list(traces)
    rows = load_traces(raw)
    mined = mine_traces(rows)
    graded = [r for r in rows if r.get("reward") in (0, 1)]
    advisory = [
        r
        for r in rows
        if r.get("reward") not in (0, 1) and _coerce_reward(r.get("qwen_reward")) is not None
    ]
    passes = sum(r["reward"] for r in graded)
    report: dict[str, Any] = {
        "traces": len(rows),
        "dropped": len(raw) - len(rows),
        "unique_prompts": len({" ".join(str(r.get("prompt") or "").lower().split()) for r in rows}),
        "tools_observed": mined["tools"],
        "faults_observed": dict(mined["faults"]),
        "world_states_observed": dict(mined.get("world_states") or {}),
        "distinct_behaviors": mined["unique_behaviors"],
        "graded": len(graded),
        "passes": passes,
        "fails": len(graded) - passes,
        # A row is ungraded only when NO label is in play: advisory
        # (qwen_reward-only) rows steer trace mining, so they are counted
        # and disclosed separately, never folded into "ungraded".
        "ungraded": len(rows) - len(graded) - len(advisory),
        "advisory_labels": len(advisory),
    }
    if tools:
        aimed = dimensions_from_traces(rows, tools, policy)
        base = build_dimensions(tools, policy)
        # Aiming works two ways: an axis NARROWS (unobserved values are
        # dropped, so every retained value's budget share rises) or values
        # REORDER toward the front. The clean contrast values stay in every
        # aimed axis by design and receive no extra weight, so they are
        # excluded from the claim.
        clean = {CLEAN_CONDITION, CLEAN_WORLD, NO_FAULT}
        emphasis: dict[str, list[str]] = {}
        for axis in ("tool", "tool_condition", "world_state"):
            base_axis = list(base.get(axis) or [])
            aimed_axis = list(aimed.get(axis) or [])
            base_pos = {v: i for i, v in enumerate(base_axis)}
            narrowed = len(aimed_axis) < len(base_axis)
            gained = [
                v
                for i, v in enumerate(aimed_axis)
                if v in base_pos and v not in clean and (narrowed or i < base_pos[v])
            ]
            if gained:
                emphasis[axis] = gained
        report["emphasis"] = emphasis
        foreign = [
            name
            for name in mined["tools"]
            if name not in {str((t.get("function") or t).get("name") or "") for t in tools}
        ]
        if foreign:
            report["foreign_tools"] = foreign
    return report


def format_trace_report(report: dict[str, Any]) -> str:
    """The trace report as a text block, aiming stated in plain words."""
    lines = [
        f"traces:             {report['traces']}",
        f"unique prompts:     {report['unique_prompts']}",
        f"distinct behaviors: {report['distinct_behaviors']}",
        f"graded:             {report['graded']} "
        f"(pass {report['passes']} / fail {report['fails']}), "
        f"ungraded {report['ungraded']}"
        + (
            f", advisory judge labels {report['advisory_labels']} (these steer aiming)"
            if report.get("advisory_labels")
            else ""
        ),
        "tools observed:     "
        + (
            ", ".join(
                f"{name} x{slot['n']}"
                + (f" ({slot['fault_n']} calls faulted)" if slot.get("fault_n") else "")
                for name, slot in sorted(report["tools_observed"].items())
            )
            or "none"
        ),
        "faults observed:    "
        + (
            ", ".join(f"{name} x{n}" for name, n in sorted(report["faults_observed"].items()))
            or "none"
        ),
    ]
    if report.get("dropped"):
        lines.append(
            f"dropped rows:       {report['dropped']} (no usable signal after normalization)"
        )
    if report.get("foreign_tools"):
        lines.append(
            "warning: observed tools not in this agent's toolset: "
            + ", ".join(report["foreign_tools"][:TRACE_REPORT_LIST_CAP])
        )
    emphasis = report.get("emphasis")
    if emphasis:
        parts = []
        names = {"tool": "tools", "tool_condition": "faults", "world_state": "world states"}
        for axis, values in emphasis.items():
            shown = ", ".join(values[:TRACE_REPORT_LIST_CAP])
            if len(values) > TRACE_REPORT_LIST_CAP:
                shown += f" (+{len(values) - TRACE_REPORT_LIST_CAP} more)"
            parts.append(f"{names.get(axis, axis)} {shown}")
        lines.append("extra generation weight goes to: " + "; ".join(parts))
    else:
        lines.append(
            "grid emphasis: computed against the agent's tools at "
            "simulate time (pass tools= to preview it here)"
        )
    return "\n".join(lines)


def infer_harness(rows: Sequence[dict]) -> dict[str, Any]:
    """Draft a harness from observed trace rows. Mechanical, no model.

    Tool schemas come from what the agent actually sent: every argument key
    seen for a tool becomes a property, its JSON type read off the observed
    values, and a key present on every call becomes required. The policy
    cannot be inferred — exporters do not ship system prompts — so it comes
    back empty for the caller to fill in. A drafted schema is a starting
    point to edit, not a spec to trust: it can only describe arguments the
    traces happened to exercise.
    """

    def _json_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        steps = row.get("steps")
        if not steps and isinstance(row.get("tool_trace"), list):
            steps = [
                {"tool": t.get("tool"), "arguments": t.get("input")}
                for t in row["tool_trace"]
                if isinstance(t, dict)
            ]
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            name = str(step.get("tool") or "").strip()
            if not name:
                continue
            args = step.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            entry = seen.setdefault(name, {"calls": 0, "keys": {}})
            entry["calls"] += 1
            for key, value in args.items():
                types = entry["keys"].setdefault(str(key), {"n": 0, "types": set()})
                types["n"] += 1
                types["types"].add(_json_type(value))

    tools = []
    for name in sorted(seen):
        entry = seen[name]
        properties = {}
        required = []
        for key in sorted(entry["keys"]):
            info = entry["keys"][key]
            types = sorted(info["types"])
            properties[key] = {"type": types[0] if len(types) == 1 else "string"}
            if info["n"] == entry["calls"]:
                required.append(key)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return {
        "tools": tools,
        "policy": "",
        "observed_calls": {name: seen[name]["calls"] for name in sorted(seen)},
    }


__all__ = [
    "dimensions_from_traces",
    "drop_leaky_rows",
    "exemplar_result_shapes",
    "flaw_rows",
    "format_trace_report",
    "infer_harness",
    "leakage_report",
    "load_traces",
    "mine_result_exemplars",
    "mine_traces",
    "simulate_from_traces",
    "split_pseudo_production",
    "trace_report",
]


# --- behavioral state over trace history ------------------------------------

# The allocation numbers live in defaults.py (TRACE_*) with their reasons and
# are keywords on ``behavior_state``. The old private names stay as aliases.
_STATE_PRIORITY = TRACE_STATE_PRIORITY
_EXPLORATION_FLOOR = TRACE_EXPLORATION_FLOOR
_MIN_SUPPORT = TRACE_MIN_SUPPORT
#: Grid axes a failing row's coordinates are remembered under: the region's
#: expansion recipe (how variants are generated around it, never its identity).
RECIPE_AXES = ("tool", "tool_condition", "world_state", "stance", "history")
_RECIPE_AXES = RECIPE_AXES


def _row_regions(row: dict) -> list[tuple[str, str, bool | None]]:
    """(region_id, kind, failed) memberships for one trajectory.

    A region is a named behavioral predicate, never a coordinate tuple:
    markers first (the customer's grader vocabulary, score 0 = fail,
    1 = pass), then fault-response pairs (fault kind x whether the run
    still succeeded), then a capability fallback for plain reward
    failures. Coordinates are recorded separately as the region's
    expansion recipe.
    """
    out: list[tuple[str, str, bool | None]] = []
    scores = row.get("scores")
    if isinstance(scores, dict):
        for name, value in scores.items():
            raw_value = value if not isinstance(value, dict) else value.get("value")
            if raw_value is None:
                continue
            try:
                v = float(raw_value)
            except (TypeError, ValueError):
                continue
            if v in (0.0, 1.0):
                out.append((str(name), "marker", v == 0.0))
    fault = trace_fault(row)
    reward = _binary_reward(row)
    if fault and fault != NO_FAULT:
        # Ungraded is unknown, never failed: reward=None rows count as
        # region support but not toward the fail/pass rates.
        failed = (reward == 0) if reward is not None else None
        out.append((f"recover_after_{fault}", "fault_response", failed))
    if not out:
        tools = [
            str(s.get("tool"))
            for s in row.get("steps") or []
            if isinstance(s, dict) and s.get("tool")
        ]
        surface = tools[0] if tools else "no-tool"
        if reward is not None:
            out.append((f"task:{surface}", "capability", reward == 0))
    return out


def behavior_state(
    rows: Sequence[dict],
    *,
    targeted: Sequence[str] = (),
    exploration: float = TRACE_EXPLORATION_FLOOR,
    min_support: int = TRACE_MIN_SUPPORT,
    priority: Mapping[str, float] | None = None,
    max_exploration: float = TRACE_EXPLORATION_MAX,
) -> dict:
    """The optimizer's memory: evidence in, allocation out.

    Rows are the agent's whole history. History buckets by
    ``model_version`` when rows carry it (rounds of the continual
    loop); otherwise by time-order halves. Each region tracks support,
    per-bucket fail rates, first/last bucket, whether it was previously
    ``targeted``, and the coordinate values it was observed under (its
    expansion recipe - how the simulator generates variants around it,
    never its identity). Classification: new, persistent, improving,
    uncertain, solved, passing; persistent-despite-targeting keeps top
    priority and flags ``rotate_coordinates``. Priority is the status
    weight scaled by support and by a Laplace-shrunk fail rate over the
    region's graded rows, so a region that fails once in a hundred draws
    far less than one that fails half the time. Regions under
    ``min_support`` graded rows (3: the smallest count at which "never
    failed" has a 95% upper bound under two thirds) are flagged
    ``low_support``: reported, but not to be trusted for allocation.
    ``priority`` maps a status to its weight (``TRACE_STATE_PRIORITY``).
    budget_share sums to 1 - exploration, with ``exploration`` clamped to
    ``max_exploration``; broad exploration is always reserved.
    """
    weights = dict(TRACE_STATE_PRIORITY if priority is None else priority)
    missing_status = set(TRACE_STATE_PRIORITY) - set(weights)
    if missing_status:
        raise ValueError(
            f"priority= needs a weight for every status; missing {sorted(missing_status)}"
        )
    items = [r for r in rows if isinstance(r, dict)]
    if any("ts" in r for r in items):
        items.sort(key=lambda r: r.get("ts") or 0)
    n = len(items)
    if n == 0:
        return {"regions": [], "exploration_share": 1.0, "traces": 0, "buckets": []}

    versions: list[str] = []
    for r in items:
        v = str(r.get("model_version") or "")
        if v and v not in versions:
            versions.append(v)
    if len(versions) >= 2:  # noqa: PLR2004  # two versions before a comparison

        def bucket_of(idx, r):
            return str(r.get("model_version") or versions[0])

        buckets = versions
    else:
        cut = max(1, n // 2)

        def bucket_of(idx, r):
            return "recent" if idx >= cut else "old"

        buckets = ["old", "recent"]
    latest = buckets[-1]

    regions: dict[str, dict] = {}
    for idx, row in enumerate(items):
        bucket = bucket_of(idx, row)
        dims = row.get("scenario_dimensions") or {}
        for region_id, kind, failed in _row_regions(row):
            slot = regions.setdefault(
                region_id,
                {
                    "region": region_id,
                    "kind": kind,
                    "by_bucket": {},
                    "recipe": {},
                    "support": 0,
                    "first_bucket": bucket,
                },
            )
            if failed is not None:
                b = slot["by_bucket"].setdefault(bucket, [0, 0])  # [fail, pass]
                b[0 if failed else 1] += 1
            slot["support"] += 1
            slot["last_bucket"] = bucket
            if failed:
                recipe = slot["recipe"]
                for axis in RECIPE_AXES:
                    value = str(dims.get(axis) or "")
                    if value and value != "unspecified":
                        recipe.setdefault(axis, [])
                        if value not in recipe[axis]:
                            recipe[axis].append(value)
                for step in row.get("steps") or []:
                    tool = str(step.get("tool") or "") if isinstance(step, dict) else ""
                    if tool:
                        recipe.setdefault("tool", [])
                        if tool not in recipe["tool"]:
                            recipe["tool"].append(tool)

    out = []
    for slot in regions.values():
        per = slot["by_bucket"]
        latest_stats = per.get(latest)
        earlier = [per[b] for b in buckets[:-1] if b in per]
        old_fail = sum(b[0] for b in earlier)
        old_seen = sum(b[0] + b[1] for b in earlier)
        rec_fail = latest_stats[0] if latest_stats else 0
        rec_seen = (latest_stats[0] + latest_stats[1]) if latest_stats else 0
        was_targeted = slot["region"] in set(targeted)
        if old_fail == 0 and rec_fail == 0:
            status = "passing"
        elif old_seen == 0 and rec_fail > 0:
            status = "new"
        elif old_fail > 0 and rec_fail > 0:
            status = "persistent"
        elif old_fail > 0 and rec_seen > 0 and rec_fail == 0:
            status = "solved" if was_targeted else "improving"
        elif old_fail > 0 and rec_seen == 0:
            status = "uncertain"
        else:
            status = "new" if rec_fail else "passing"
        fails_total = sum(b[0] for b in per.values())
        graded_total = sum(b[0] + b[1] for b in per.values())
        # support: 0.5 with no failures, 1.0 at TRACE_SUPPORT_SATURATION
        support_factor = min(
            1.0, 0.5 + 0.5 * min(fails_total, TRACE_SUPPORT_SATURATION) / TRACE_SUPPORT_SATURATION
        )
        # Laplace-shrunk fail rate over every graded row in the region:
        # one failure in a hundred and fifty in a hundred used to draw the
        # same priority. A region with no graded rows sits at the prior.
        fail_rate = (fails_total + 1.0) / (graded_total + 2.0)
        rate_factor = TRACE_RATE_FLOOR + (1.0 - TRACE_RATE_FLOOR) * fail_rate
        slot["status"] = status
        slot["n_graded"] = graded_total
        slot["fail_rate"] = round(fail_rate, 4)
        slot["low_support"] = graded_total < int(min_support)
        slot["previously_targeted"] = was_targeted
        slot["rotate_coordinates"] = bool(was_targeted and status == "persistent")
        slot["history"] = [
            {
                "bucket": b,
                "n": per[b][0] + per[b][1],
                "fail_rate": round(per[b][0] / (per[b][0] + per[b][1]), 3),
            }
            for b in buckets
            if b in per
        ]
        slot["priority"] = round(float(weights[status]) * support_factor * rate_factor, 4)
        out.append(slot)

    total = sum(s["priority"] for s in out) or 1.0
    pool = 1.0 - max(0.0, min(float(max_exploration), float(exploration)))
    for slot in out:
        slot["budget_share"] = round(pool * slot["priority"] / total, 4)
        del slot["by_bucket"]
    out.sort(key=lambda s: -s["budget_share"])
    return {
        "regions": out,
        "exploration_share": round(1.0 - pool, 4),
        "traces": n,
        "buckets": buckets,
    }


def region_progress(state: dict, generated_rows: Sequence[dict]) -> list[dict]:
    """Same rules on both sides of the loop: re-measure each trace-derived
    region on the generated (and graded) rows. The per-region pair -
    fail rate in the traces vs fail rate in what we generated - is the
    hill-climb readout: searching and fixing, in the same vocabulary the
    customer's grader speaks.
    """
    counts: dict[str, list[int]] = {}
    for row in generated_rows or []:
        if not isinstance(row, dict):
            continue
        for region_id, _kind, failed in _row_regions(row):
            if failed is None:
                continue
            slot = counts.setdefault(region_id, [0, 0])
            slot[0 if failed else 1] += 1
    out = []
    for region in (state or {}).get("regions", []):
        rid = region.get("region")
        hist = region.get("history") or []
        trace_rate = hist[-1]["fail_rate"] if hist else None
        pair = counts.get(rid)
        gen_n = (pair[0] + pair[1]) if pair else 0
        out.append(
            {
                "region": rid,
                "status": region.get("status"),
                "trace_fail_rate": trace_rate,
                "generated_n": gen_n,
                "generated_fail_rate": (round(pair[0] / gen_n, 3) if pair and gen_n else None),
            }
        )
    return out
