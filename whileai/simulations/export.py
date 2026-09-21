"""Training-ready rows: system prompt, tool schemas, standard wire format.

A simulated row stores the conversation without the agent's own system
prompt or tool schemas; the run knows them, the row does not. A trainer
needs all three, in the shape chat templates expect, whether or not the
target model emits thinking tokens. ``training_rows`` closes that gap:

* prepends the ``system`` message,
* attaches the tool schemas on each row,
* converts tool calls to the OpenAI wire format (``id``, ``type``,
  ``function.name``, ``function.arguments`` as a JSON string) and links
  each tool result by ``tool_call_id``,
* strips ``<think>`` blocks from assistant turns, so a thinking rollout
  model never teaches a non-thinking student to emit them.

The strip cuts the other way too. A reasoning base (Qwen3) trained on
think-free targets learns to print an empty ``<think></think>`` and
answer at once. That is fine until the eval, where that adapter meets
the untrained base under one shared ``max_tokens``: the base reasons and
runs out of budget, the adapter answers, and the delta is a win over
replies the base never produced (#297). ``delta_report`` fails that
comparison; the fix is ``thinking=`` set the same on both arms, or
``strip_think=False`` when the student is a reasoning model and should
keep reasoning.

Three wire shapes come out of here, and they are not the same shape:

``format="openai"`` (the default)
    The OpenAI chat-completions wire row. ``messages`` is the whole
    conversation, ``function.arguments`` is a JSON **string**, and the
    row keeps ``prompt`` (the ask, as text) beside it for grouping and
    slicing. This is what an API replay, an eval harness, or a custom
    collator wants, and it is what every previous version emitted.

``format="fireworks"``
    What a Fireworks managed training job reads (``firectl dataset
    create``, then ``firectl sftj create`` or ``firectl dpo-job
    create``). SFT rows are ``{"messages": [...], "tools": [...]}`` in
    the OpenAI wire shape (``function.arguments`` a JSON string, the
    encoding Fireworks' function-calling datasets take), with the
    SDK's per-message ``loss_mask`` carried as Fireworks' per-message
    ``weight`` (0 keeps a message out of the loss, 1 trains on it), so
    ``mask_mode`` survives the trip. Nothing else the row carries is
    written: Fireworks reads the fields it documents. Preference rows
    are Fireworks' one-turn shape, ``{"input": {"messages": <up to the
    first assistant turn>, "tools": [...]}, "preferred_output":
    [<one assistant message>], "non_preferred_output": [<one assistant
    message>]}``: the prefix both sides share (tool turns included) is
    the input, the first assistant turn where they differ is the
    preference, and later turns are cut (the report counts the pairs
    that lost turns as ``fireworks_turns_cut``). Shapes follow
    docs.fireworks.ai/fine-tuning (fine-tuning-models, dpo-fine-tuning).

``format="trl"``
    What ``trl`` (and anything else that calls
    ``maybe_apply_chat_template``) actually accepts. For SFT the row is
    conversational ``{"messages": [...]}`` with **no** ``prompt`` string
    column — TRL's ``is_conversational`` sniffs the column set, and a
    ``prompt`` string next to ``messages`` makes it decide the row is
    not conversational, apply no chat template, and train on the bare
    ask; the ask survives as ``prompt_text``. For preference data the
    row is TRL's conversational DPO triple: ``prompt`` is the message
    list up to the first assistant turn, and ``chosen``/``rejected`` are
    the **completions only**. In both, ``function.arguments`` is a
    **dict**, because HF chat templates render it with ``| tojson`` and
    a pre-encoded string comes out quoted twice.

    The TRL rows carry no ``loss_mask``. trl 0.19.1's ``SFTTrainer``
    tokenizes with the chat template and its
    ``DataCollatorForLanguageModeling`` labels every token; the only two
    columns that take tokens out of the loss are token-level and built by
    the trainer itself: ``completion_mask`` (from ``prompt``/``completion``
    rows) and ``assistant_masks`` (from ``assistant_only_loss=True``, which
    needs a ``{% generation %}`` block in the chat template). A per-message
    ``loss_mask`` is passed through unread (#507). So ``mask_mode="final"``
    and ``unroll=True`` come out as prompt/completion rows, which TRL
    trains exactly as the mask asks, and ``mask_mode="assistant"`` comes
    out as ``messages`` rows, which TRL trains on every token unless
    ``assistant_only_loss=True`` is set; the report's ``mask_mode`` says
    which.

``to_trl`` performs that reshape on rows you already built, and the
round-trip gate reports which of the two encodings it checked.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .defaults import PASS_THRESHOLD
from .schema import check, stamp
from .score.privileged import leak_report
from .score.quality import load_jsonl, write_jsonl
from .score.stats import task_key

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
# The second group are the sampled diversity axes. They cost bytes, but a
# training file that drops them cannot answer "which slice regressed" after
# the run: post-training evals stratify by exactly these, and re-deriving
# them from message text is guesswork.
_CARRY_KEYS = (
    "reward",
    "qwen_reward",
    "reason",
    "prompt",
    "scenario_id",
    "world_state",
    "faults",
    "fault_detected",
    "label_source",
    "failure_class",
    "judge_status",
    "judge_name",
    "lineage",
    "logprob",
    "n_tokens",
    "token_logprobs",
    "sampling",
    "policy_version",
    "writer_model",
    "user_model",
    "usage",
    "tier",
    "ask_family",
    "ask",
    "stance",
    "tone",
    "texture",
    "vagueness",
    "phrasing",
    "pressure",
    "user",
    "history",
    "length",
    "intent_known",
    "tool_known",
)


def _strip_think(text: str) -> str:
    return _THINK_BLOCK.sub("", str(text or "")).strip()


def _decoded_arguments(arguments: Any) -> Any:
    """Unwrap repeated JSON string encoding down to the structured value."""
    value = arguments
    for _ in range(3):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value


def _wire_arguments(arguments: Any) -> str:
    # Chat templates render arguments with |tojson, so a pre-encoded string
    # would be quoted twice and the model learns string-wrapped arguments.
    value = _decoded_arguments(arguments)
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, separators=(",", ": "), default=str)


#: The wire shapes the exporters can emit. See the module docstring.
EXPORT_FORMATS = ("openai", "trl", "fireworks")

#: What ``tool_call_roundtrip`` says it checked, per format. The gate is
#: only as good as the encoding it was pointed at, so the report names it
#: instead of reading as a clean bill of health for every consumer.
_ENCODINGS = {
    "openai": "json_string",
    "trl": "dict",
    "fireworks": "json_string",
}
_ENCODING_NOTES = {
    "json_string": (
        "arguments parse back to a dict from their JSON string (OpenAI wire). "
        "A chat-template trainer needs dicts: export format='trl' for that."
    ),
    "dict": "arguments are dicts, as HF chat templates render them (| tojson).",
}


def tool_call_roundtrip(rows: Sequence[dict], *, format: str = "openai") -> dict[str, Any]:
    """Check every tool call in exported rows carries structured arguments.

    Guards the training run, not the export. Which check that is depends
    on where the rows are going, so the report names the encoding it
    validated:

    * ``format="openai"`` (``encoding: "json_string"``): arguments must
      parse back to a dict. Arguments that survive as un-parseable
      strings get re-quoted by chat templates and teach the model to emit
      string-wrapped arguments, which then spiral on tool rejections.
    * ``format="trl"`` (``encoding: "dict"``): arguments must already
      *be* dicts. A JSON string here is valid OpenAI wire and still wrong
      for a chat template, which would render it quoted twice, so it
      counts as invalid rather than passing on a technicality.
    """
    if format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {EXPORT_FORMATS}, got {format!r}")
    encoding = _ENCODINGS[format]
    checked = invalid = 0
    bad_rows: list[int] = []
    for i, row in enumerate(rows):
        row_bad = False
        for message in _turns(row):
            for call in message.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                checked += 1
                raw = fn.get("arguments")
                if encoding == "dict":
                    parsed = raw
                else:
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                    except ValueError:
                        parsed = None
                if not isinstance(parsed, dict):
                    invalid += 1
                    row_bad = True
        if row_bad:
            bad_rows.append(i)
    return {
        "checked": checked,
        "invalid": invalid,
        "rows": bad_rows[:20],
        "encoding": encoding,
        "checked_for": _ENCODING_NOTES[encoding],
    }


def _turns(row: dict) -> list[dict]:
    """The whole conversation of an exported row, whichever shape it is in:
    ``messages``, or ``prompt`` + ``completion`` when both are message lists."""
    messages = row.get("messages")
    if isinstance(messages, list):
        return messages
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return list(prompt) + list(row.get("completion") or [])
    return []


def _convert_messages(
    messages: Sequence[dict],
    *,
    system: str,
    strip_think: bool,
    max_tool_output_chars: int | None = None,
    stats: dict[str, int] | None = None,
) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    pending_ids: list[str] = []
    call_i = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "assistant":
            if strip_think:
                content = _strip_think(content)
            entry: dict[str, Any] = {"role": "assistant", "content": content}
            calls = message.get("tool_calls") or []
            if calls:
                wire = []
                for call in calls:
                    call = call if isinstance(call, dict) else {}
                    fn = call["function"] if isinstance(call.get("function"), dict) else call
                    call_id = str(call.get("id") or f"call_{call_i:04d}")
                    call_i += 1
                    pending_ids.append(call_id)
                    wire.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": str(fn.get("name") or ""),
                                "arguments": _wire_arguments(fn.get("arguments")),
                            },
                        }
                    )
                entry["tool_calls"] = wire
            out.append(entry)
        elif role == "tool":
            if max_tool_output_chars is not None and len(content) > int(max_tool_output_chars):
                # Tool output eats context faster than anything else the
                # trainer sees; cutting it is a decision the export makes
                # out loud (Lambert 2025, chapter Tool Use), never silently.
                cap = max(0, int(max_tool_output_chars))
                cut = len(content) - cap
                marker = "[... " + str(cut) + " chars of tool output truncated]"
                content = content[:cap] + chr(10) + marker
                if stats is not None:
                    stats["truncated"] = stats.get("truncated", 0) + 1
                    stats["chars_cut"] = stats.get("chars_cut", 0) + cut
            entry = {"role": "tool", "content": content}
            if message.get("name"):
                entry["name"] = str(message["name"])
            call_id = (
                str(message.get("tool_call_id"))
                if message.get("tool_call_id")
                else (pending_ids.pop(0) if pending_ids else "")
            )
            if call_id:
                entry["tool_call_id"] = call_id
            out.append(entry)
        elif role in {"user", "system"}:
            out.append({"role": role, "content": content})
    return out


def _trl_messages(messages: Sequence[dict]) -> list[dict]:
    """The same turns with tool-call ``arguments`` as dicts.

    HF chat templates render arguments with ``| tojson``, so a string
    that is already JSON comes out quoted twice and the student learns to
    emit a string where the template expects an object. An argument blob
    that does not decode to a dict is left exactly as it was, so the
    round-trip gate reports it instead of this quietly dropping it.
    """
    out: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        entry = dict(message)
        calls = entry.get("tool_calls")
        if not calls:
            out.append(entry)
            continue
        wire: list[dict] = []
        for call in calls:
            call = dict(call) if isinstance(call, dict) else {}
            fn = dict(call.get("function") or {})
            fn["arguments"] = _decoded_arguments(fn.get("arguments"))
            call["function"] = fn
            wire.append(call)
        entry["tool_calls"] = wire
        out.append(entry)
    return out


def _first_assistant(messages: Sequence[dict]) -> int:
    for i, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return i
    return -1


def _trl_training_row(row: dict) -> dict:
    """One ``training_rows`` row as TRL conversational SFT.

    The ``prompt`` string moves to ``prompt_text``. TRL's
    ``is_conversational`` decides by column set, and ``prompt`` (a string)
    next to ``messages`` (a list) is not a shape it knows: it returns
    False, ``maybe_apply_chat_template`` hands the row back untouched with
    no ``text`` key and no error, and ``SFTTrainer`` trains on the bare
    ask instead of the conversation. Silently. Hence the rename.
    """
    out = dict(row)
    out["messages"] = _trl_messages(row.get("messages") or [])
    prompt = out.pop("prompt", None)
    if isinstance(prompt, str) and prompt:
        out["prompt_text"] = prompt
    # trl 0.19.1 reads no per-message mask; a column it passes through
    # unread would read as a promise the file does not keep (#507).
    out.pop("loss_mask", None)
    return out


def _trl_completion_row(row: dict) -> dict | None:
    """One ``training_rows`` row as TRL conversational prompt/completion, or None.

    ``prompt`` is the conversation up to the last assistant turn and
    ``completion`` is that turn (and anything after it). trl 0.19.1's
    ``SFTTrainer`` tokenizes both, builds a token-level ``completion_mask``
    from where the prompt ends, and its collator labels only the
    completion (``completion_only_loss`` defaults on when the first row
    has a ``prompt``): the ``mask_mode="final"`` mask, honored by the
    trainer. A row with no assistant turn has no completion and is
    dropped and counted rather than written as an empty target.
    """
    out = _trl_training_row(row)
    messages = out.pop("messages")
    last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    if last < 1:
        return None
    out["prompt"] = messages[:last]
    out["completion"] = messages[last:]
    return out


def _trl_preference_row(row: dict) -> dict | None:
    """One ``export_preference`` row as TRL conversational DPO, or None.

    TRL wants ``prompt`` as the message list up to the first assistant
    turn and ``chosen``/``rejected`` as the completion only; handing it a
    ``prompt`` string with full conversations on both sides raises
    ``TypeError: string indices must be integers``. A side with nothing
    after the split point is no completion at all, so the pair is dropped
    and counted rather than written as an empty preference.
    """
    chosen = _trl_messages(row.get("chosen") or [])
    rejected = _trl_messages(row.get("rejected") or [])
    ci, ri = _first_assistant(chosen), _first_assistant(rejected)
    if ci < 1 or ri < 1:
        return None
    out = dict(row)
    out["prompt"] = chosen[:ci]
    out["chosen"] = chosen[ci:]
    out["rejected"] = rejected[ri:]
    return out


#: A message as Fireworks reads it: the three fields its dataset docs name.
_FIREWORKS_TURN_KEYS = ("role", "content", "tool_calls")


def _fireworks_training_row(row: dict) -> dict:
    """One ``training_rows`` row as a Fireworks SFT line.

    ``messages`` and ``tools`` in the OpenAI wire shape (arguments stay
    JSON strings), and the SDK's per-message ``loss_mask`` carried as
    Fireworks' per-message ``weight`` on the assistant turns: 0 keeps a
    turn out of the loss, 1 trains on it, which is how ``mask_mode``
    survives. ``prompt``, ``reward`` and lineage stay out; Fireworks reads
    the fields it documents.
    """
    mask = list(row.get("loss_mask") or [])
    messages: list[dict] = []
    for i, message in enumerate(row.get("messages") or []):
        if not isinstance(message, dict):
            continue
        entry = {k: message[k] for k in _FIREWORKS_TURN_KEYS if k in message}
        if "tool_call_id" in message:
            entry["tool_call_id"] = message["tool_call_id"]
        if entry.get("role") == "assistant" and i < len(mask):
            entry["weight"] = 1 if mask[i] else 0
        messages.append(entry)
    out: dict[str, Any] = {"messages": messages}
    if row.get("tools"):
        out["tools"] = list(row["tools"])
    return out


def _fireworks_turn(message: dict) -> dict:
    return {k: message[k] for k in _FIREWORKS_TURN_KEYS if k in message}


def _fireworks_preference_rows(rows: Sequence[dict]) -> tuple[list[dict], int, int]:
    """``export_preference`` rows as Fireworks DPO lines: ``(rows, dropped, cut)``.

    Fireworks takes one-turn preferences: ``input.messages`` is a
    conversation, ``preferred_output`` and ``non_preferred_output`` the one
    assistant message that follows it. A pair's sides share a prefix (the
    ask, and often the same opening tool call and its result) and then
    differ, so the shared prefix is the input and the first assistant turn
    where the sides differ is the preference; the turns after it are cut,
    and ``cut`` counts the pairs that lost some. A pair that diverges on a
    tool result rather than an assistant turn, or whose sides never
    differ, has no one-turn contrast and is dropped and counted.
    """
    out: list[dict] = []
    dropped = cut = 0
    for row in rows:
        chosen = [_fireworks_turn(m) for m in row.get("chosen") or [] if isinstance(m, dict)]
        rejected = [_fireworks_turn(m) for m in row.get("rejected") or [] if isinstance(m, dict)]
        k = 0
        while k < len(chosen) and k < len(rejected) and chosen[k] == rejected[k]:
            k += 1
        if k < 1 or k >= len(chosen) or k >= len(rejected):
            dropped += 1
            continue
        preferred, non_preferred = chosen[k], rejected[k]
        if preferred.get("role") != "assistant" or non_preferred.get("role") != "assistant":
            dropped += 1
            continue
        if len(chosen) > k + 1 or len(rejected) > k + 1:
            cut += 1
        out.append(
            {
                "input": {"messages": chosen[:k], "tools": list(row.get("tools") or [])},
                "preferred_output": [preferred],
                "non_preferred_output": [non_preferred],
            }
        )
    return out, dropped, cut


TRL_KINDS = ("training", "completion", "preference")


def to_trl(rows: Sequence[dict], kind: str = "training") -> list[dict]:
    """Rows from ``training_rows`` / ``export_preference`` in TRL's shape.

    ``kind="training"`` reshapes SFT rows as conversational ``messages``
    (no ``loss_mask``: trl 0.19.1 trains on every token of them unless
    ``assistant_only_loss=True``), ``kind="completion"`` as conversational
    ``prompt``/``completion`` with the last assistant turn as the
    completion (the ``mask_mode="final"`` mask, which the trainer honors
    through the ``completion_mask`` it builds), ``kind="preference"`` DPO
    pairs; see the module docstring for what each shape is and why it
    differs from the default OpenAI wire rows. Equivalent to passing
    ``format="trl"`` to the exporters, for callers that already hold
    rows. Completion rows with no assistant turn and preference pairs
    whose chosen or rejected side has no completion after the prompt
    prefix are dropped.
    """
    if kind not in TRL_KINDS:
        raise ValueError(f"kind must be one of {TRL_KINDS}, got {kind!r}")
    if kind == "training":
        return [_trl_training_row(r) for r in rows if isinstance(r, dict)]
    shape = _trl_completion_row if kind == "completion" else _trl_preference_row
    out = [shape(r) for r in rows if isinstance(r, dict)]
    return [r for r in out if r is not None]


MASK_MODES = ("assistant", "final")

#: What trl 0.19.1's ``SFTTrainer`` does with a ``format="trl"`` file, in the
#: report's ``mask_mode`` in place of the SDK's intention (#507). Checked
#: against ``trl/trainer/sft_trainer.py`` at v0.19.1: ``tokenize`` builds
#: ``completion_mask`` for prompt/completion rows and ``assistant_masks``
#: when ``assistant_only_loss=True``; ``DataCollatorForLanguageModeling``
#: unlabels tokens from those two columns and nothing else.
TRL_MASK_EVERY_TOKEN = (
    "TRL trains on every token of every turn (trl 0.19.1 SFTTrainer reads no "
    "per-message mask); set assistant_only_loss=True in SFTConfig to train the "
    "assistant turns only, which needs a chat template with a {% generation %} "
    "block (Qwen2.5-Instruct has none)"
)
TRL_MASK_COMPLETION = (
    "TRL trains on the last assistant turn only: prompt/completion rows, from "
    "which trl 0.19.1 SFTTrainer builds completion_mask (completion_only_loss "
    "defaults on)"
)


def loss_mask(messages: Sequence[dict], *, mode: str = "assistant") -> list[int]:
    """One 0/1 per message: 1 carries loss, 0 is context only.

    ``"assistant"`` trains every assistant turn, the multi-turn default.
    ``"final"`` trains only the last assistant turn, for conversations
    whose earlier agent turns were scripted or came from another policy
    (Lambert 2025, chapter Instruction Tuning, describes both). System,
    user, and tool messages are always 0: tool output is the environment
    speaking, not the policy, and training on it teaches the model to
    invent tool results (Lambert 2025, chapter Tool Use).
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mode!r}")
    mask = [1 if isinstance(m, dict) and m.get("role") == "assistant" else 0 for m in messages]
    if mode == "final" and any(mask):
        last = max(i for i, v in enumerate(mask) if v)
        mask = [1 if i == last else 0 for i in range(len(mask))]
    return mask


def _resolve(source) -> tuple[list[dict], str, list, str]:
    """rows, system prompt, tools, source path (best effort)."""
    if hasattr(source, "trajectories"):
        profile = getattr(source, "profile", None)
        system = str(getattr(profile, "policy", "") or "")
        tools = list(getattr(profile, "tools", None) or [])
        return list(source.trajectories), system, tools, ""
    if isinstance(source, (str, Path)):
        return load_jsonl(source), "", [], str(source)
    # a RowList (scored.rows, a slice, a decontaminate result) carries both
    system = str(getattr(source, "system_prompt", "") or "")
    tools = list(getattr(source, "tools", None) or [])
    return list(source), system, tools, ""


def training_rows(
    source,
    *,
    system_prompt: str | None = None,
    tools: Sequence[dict] | None = None,
    strip_think: bool = True,
    mask_mode: str = "assistant",
    unroll: bool = False,
    max_tool_output_chars: int | None = None,
) -> list[dict]:
    """Build the rows a trainer can consume directly: system prompt in, tool schemas on, wire format fixed.

    Reach for it when you want the rows in memory rather than in a file
    (``export_dataset`` writes these same rows as JSONL, with the gates).
    A simulated row stores the conversation without the agent's own
    system prompt or tool schemas; the run knows them, the row does not.
    This call returns a list of dicts, one per source row, each with
    ``messages`` (the ``system`` message prepended, tool calls in the
    OpenAI wire format with ``id``, ``type``, ``function.name`` and
    ``function.arguments`` as a JSON string, each tool result linked by
    ``tool_call_id``), ``tools`` (the schemas), ``loss_mask`` (which
    assistant turns carry loss), the ask as ``prompt``, and the row's
    grade and lineage. The module docstring has the two wire shapes.

    * ``source``: a ``SimulationData`` (system prompt and tools come from
      its profile), a row list, or a JSONL path. For lists and paths,
      pass ``system_prompt=`` and ``tools=`` explicitly; a row exported
      without its policy trains an agent that never saw its rules.
    * ``strip_think`` (``True``): remove ``<think>`` blocks from the
      assistant turns, so a thinking rollout model never teaches a
      non-thinking student to emit them. On a reasoning base such as
      Qwen3 that teaches the adapter to emit an empty ``<think></think>``
      and answer at once, so at eval it answers while the untrained base
      is still reasoning under the same ``max_tokens``. Pass
      ``strip_think=False`` when the student should keep reasoning, and
      set ``thinking=`` the same on both arms of the eval either way.
    * ``mask_mode`` (``"assistant"``): which assistant turns carry loss
      (see ``loss_mask``): all of them, or ``"final"`` for the last turn
      only.
    * ``unroll`` (``False``): ``True`` turns an N-turn conversation into N
      samples, the k-th ending at the k-th assistant turn with loss on
      that turn only (Lambert 2025, chapter Instruction Tuning). Every earlier
      agent turn then trains once with exactly the context it had,
      instead of only the last one (``mask_mode="final"``) or all of them
      at once (``"assistant"``, where later turns see context the policy
      never produced). Each sample carries ``unroll`` (``turn``,
      ``turns``) and ``lineage.unrolled_from`` (the source row's prompt
      hash and rollout index); ``mask_mode`` is ignored, and no group
      fields are stamped, since samples of one conversation are not a
      GRPO group.
    * ``max_tool_output_chars`` (``None``, cut nothing): cap each tool
      message at that many characters, appending ``[... N chars of tool
      output truncated]`` and counting the cut on the row as
      ``tool_output_truncated`` (messages) and ``tool_output_chars_cut``.
      Tool output is masked from the loss anyway; what it costs is
      context, and the cut is explicit rather than silent (Lambert 2025,
      chapter Tool Use).

    ```python
    rows = wai.training_rows(data, unroll=True)
    print(rows[0]["messages"][0]["role"], rows[0]["loss_mask"])
    ```
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    from whileai.simulations import conversation

    rows, system, resolved_tools, _ = _resolve(source)
    if system_prompt is not None:
        system = str(system_prompt)
    if tools is not None:
        resolved_tools = list(tools)
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Rows pulled from the platform store actions as ``tool_trace``
        # (input/output); the conversation builder reads ``steps``
        # (arguments/result). Without this bridge a pulled trace exports as
        # a tool-less chat and the roundtrip gate has nothing to check —
        # the dataset trains an agent that never calls a tool.
        if (
            not row.get("messages")
            and not row.get("steps")
            and isinstance(row.get("tool_trace"), list)
        ):
            row = dict(row)
            row["steps"] = [
                {
                    "tool": step.get("tool"),
                    "arguments": step.get("input"),
                    "result": step.get("output"),
                }
                for step in row["tool_trace"]
                if isinstance(step, dict)
            ]
        messages = row.get("messages") or conversation(row)
        cut_stats: dict[str, int] = {}
        entry: dict[str, Any] = {
            "messages": _convert_messages(
                messages,
                system=system,
                strip_think=strip_think,
                max_tool_output_chars=max_tool_output_chars,
                stats=cut_stats,
            ),
        }
        if cut_stats.get("truncated"):
            entry["tool_output_truncated"] = cut_stats["truncated"]
            entry["tool_output_chars_cut"] = cut_stats["chars_cut"]
        # Train on the agent's turns only. Tool output is the environment's
        # text, not the policy's, and is masked from the loss (Lambert 2025,
        # chapter Tool Use); system and user turns likewise. One entry per
        # message.
        entry["loss_mask"] = loss_mask(entry["messages"], mode=mask_mode)
        if resolved_tools:
            entry["tools"] = list(resolved_tools)
        for key in _CARRY_KEYS:
            if row.get(key) is not None:
                entry[key] = row[key]
        if unroll:
            out.extend(stamp(sample) for sample in _unroll(entry, row))
        else:
            out.append(stamp(entry))
    if not unroll:
        _stamp_groups(out)
    check(out, "training", where="training_rows")
    return out


def _unroll(entry: dict, row: dict) -> list[dict]:
    """One sample per assistant turn: the conversation up to that turn,
    loss on it alone. A conversation with no assistant turn is one sample
    as it stands."""
    messages = entry["messages"]
    turns = [
        i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    if not turns:
        return [entry]
    source = {
        "prompt_sha1": hashlib.sha1(str(row.get("prompt") or "").encode("utf-8")).hexdigest()[:12],
        "rollout_index": row.get("rollout_index"),
    }
    if row.get("scenario_id"):
        source["scenario_id"] = row["scenario_id"]
    samples: list[dict] = []
    for k, i in enumerate(turns, 1):
        sample = dict(entry)
        sample["messages"] = list(messages[: i + 1])
        sample["loss_mask"] = [1 if j == i else 0 for j in range(i + 1)]
        sample["unroll"] = {"turn": k, "turns": len(turns)}
        lineage = dict(entry.get("lineage") or {})
        lineage["unrolled_from"] = dict(source)
        sample["lineage"] = lineage
        samples.append(sample)
    return samples


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _stamp_groups(rows: list[dict]) -> None:
    """GRPO group identity, stamped whenever any prompt repeats.

    ``select_for_rl`` hands over whole groups, and then the export used to
    flatten them: a GRPO trainer had to re-group by exact prompt string, an
    equality that one whitespace edit silently breaks. ``group_id`` is the
    stable name (sha1 of the ``task_key``: the situation id, else the prompt), ``k`` the
    group size, ``n0``/``n1`` the fail/pass counts so a consumer can drop
    unanimous groups without rescoring, and ``reward_mean``/``reward_std``
    the group's reward statistics (Shao et al. 2024, arXiv:2402.03300,
    GRPO: group-normalized advantages divide by this std, so a trainer can
    see where it is near zero and choose batch-level normalization or Dr.
    GRPO instead). A
    partial-credit reward counts as a pass above 0.5 and a fail below it;
    exactly 0.5 (an advisory verdict) counts as neither. A run with no
    repeated prompt is an SFT/explore export and gets no group fields.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        task = task_key(row)
        if task:
            groups.setdefault(task, []).append(row)
    if not any(len(members) > 1 for members in groups.values()):
        return
    for prompt, members in groups.items():
        gid = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        rewards = [_numeric(m.get("reward")) for m in members]
        numeric = [v for v in rewards if v is not None]
        n0 = sum(1 for v in numeric if v < PASS_THRESHOLD)
        n1 = sum(1 for v in numeric if v > PASS_THRESHOLD)
        mean = sum(numeric) / len(numeric) if numeric else None
        std = (
            (sum((v - mean) ** 2 for v in numeric) / len(numeric)) ** 0.5
            if numeric and mean is not None
            else None
        )
        for m in members:
            m["group_id"] = gid
            m["k"] = len(members)
            m["n0"] = n0
            m["n1"] = n1
            m["reward_mean"] = mean
            m["reward_std"] = std


def export_training(
    source,
    output: str | None = None,
    *,
    system_prompt: str | None = None,
    tools: Sequence[dict] | None = None,
    strip_think: bool = True,
    validate: bool = True,
    mask_mode: str = "assistant",
    unroll: bool = False,
    max_tool_output_chars: int | None = None,
    format: str = "openai",
    push_to: str | None = None,
) -> dict[str, Any]:
    """Write ``training_rows`` as JSONL, gated so a broken row never reaches the trainer.

    Reach for it when graded rows are ready for SFT. It builds the rows
    with ``training_rows`` (system prompt in, tool schemas on, tool calls
    in the wire format, ``<think>`` blocks stripped), checks them, and
    writes one JSON object per line. It returns the report: ``path``,
    ``n`` and ``n_written``, ``rewards``, ``with_system``, ``with_tools``,
    ``format``, ``mask_mode``, ``tool_call_roundtrip``,
    ``privileged_leaks``, ``warnings``, and what was cut or unrolled.
    ``export_dataset``
    and ``export_training`` are one function object under two names
    (``export_dataset is export_training``): same arguments, same file,
    same report. Write ``export_dataset`` in new code (it exports a
    dataset, not a training run); the older spelling is kept so nothing
    written today breaks.

    * ``source``: a ``SimulationData`` (system prompt and tools come from
      its profile), a row list, or a JSONL path; for the last two pass
      ``system_prompt`` and ``tools``.
    * ``output``: the file to write. With a path source and no ``output``
      it writes ``<name>.train.jsonl`` next to it. It never overwrites
      the source.
    * ``validate`` (``True``): refuse to write a dataset whose tool calls
      do not round-trip to structured arguments, or whose assistant turns
      quote the row's own ``privileged`` block (the export scrubs the key,
      not the reply that recited it); ``False`` exports anyway and leaves
      the report to read. The leak check reads the source before the
      scrub, so pass the ``SimulationData`` or its ``trajectories``; rows
      that already came through ``rows()``, ``save()`` or a file carry
      nothing to check, and ``report["privileged_leaks"]`` says so.
    * ``format``: ``"openai"`` (the default) writes the OpenAI
      chat-completions wire row: the full ``messages`` list,
      ``function.arguments`` as a JSON string, and the ask carried
      alongside as ``prompt``. ``"trl"`` writes what ``trl`` can actually
      load: conversational SFT rows (arguments as dicts, the ask under
      ``prompt_text``). TRL decides "is this conversational?" from the
      column set, so a ``prompt`` string beside ``messages`` makes it skip
      the chat template without an error and train on the bare ask, which
      is why the TRL rows do not carry one. The report says which format
      and which argument encoding the round-trip gate checked.
    * ``mask_mode`` with ``format="trl"``: the TRL rows carry no
      ``loss_mask``, because trl 0.19.1's ``SFTTrainer`` never reads one.
      Its collator (``DataCollatorForLanguageModeling``) labels every
      token and unlabels only from two token-level columns the trainer
      builds itself: ``completion_mask`` from ``prompt``/``completion``
      rows, and ``assistant_masks`` from ``assistant_only_loss=True``,
      which needs a ``{% generation %}`` block in the chat template
      (Qwen2.5-Instruct has none). So ``mask_mode="final"`` and
      ``unroll=True`` write prompt/completion rows (prompt up to the last
      assistant turn, that turn as the completion), and TRL trains on
      exactly what the mask asked. ``mask_mode="assistant"`` (the default)
      writes ``messages`` rows, and TRL trains on every token of them
      unless ``assistant_only_loss=True`` is set. The report's
      ``mask_mode`` states which of the two TRL will do, and
      ``trained_messages`` / ``masked_messages`` count what TRL trains,
      not what the SDK intended; a ``format="trl"`` export on 94 rows was
      measured at 9x the tokens its ``loss_mask`` marked (#507).
    * ``push_to``: a Hub repo (``"me/my-set"``) to upload the written
      file to, with your own token: ``HF_TOKEN`` from the environment or
      the login ``hf auth login`` cached, through ``huggingface_hub``
      (``pip install 'whileai[hf]'``). Private by default; no platform
      call. ``wai.hub.push`` is the same upload for a file, a directory
      or rows you already hold, with ``token=`` and ``private=``. The
      report gains ``hub`` (``repo_id``, ``url``, ``commit``).
    * ``strip_think``, ``mask_mode``, ``unroll``, ``max_tool_output_chars``:
      passed through to ``training_rows``, which explains each.

    ```python
    report = wai.export_dataset(data, "train.jsonl", format="trl")
    print(report["n"], report["mask_mode"], report["tool_call_roundtrip"])
    ```
    """
    if format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {EXPORT_FORMATS}, got {format!r}")
    rows = training_rows(
        source,
        system_prompt=system_prompt,
        tools=tools,
        strip_think=strip_think,
        mask_mode=mask_mode,
        unroll=unroll,
        max_tool_output_chars=max_tool_output_chars,
    )
    # The masks the SDK computed, kept for the count: the TRL rows drop the
    # column because the trainer never reads it (#507).
    masks = [list(r["loss_mask"]) for r in rows]
    final_only = unroll or mask_mode == "final"
    no_completion_dropped = 0
    if format == "trl":
        reshaped = to_trl(rows, "completion" if final_only else "training")
        no_completion_dropped = len(rows) - len(reshaped)
        rows = reshaped
    roundtrip = tool_call_roundtrip(rows, format=format)
    if validate and roundtrip["invalid"]:
        raise ValueError(
            f"tool_call_roundtrip_invalid: {roundtrip['invalid']} of "
            f"{roundtrip['checked']} tool calls do not parse back to "
            f"structured arguments (rows {roundtrip['rows']}); training on "
            "them teaches string-wrapped arguments. Fix the rows or pass "
            "validate=False."
        )
    raw, _, _, src = _resolve(source)
    # The scrub drops the ``privileged`` key at any depth and copies the
    # assistant's reply through verbatim, so a reply that recited the block
    # still recites it in the training file. Check the unscrubbed side,
    # which is the only place the needles still exist (#249).
    leaks = leak_report(raw)
    if validate and leaks["n_leaked"]:
        raise ValueError(
            f"privileged_leak: {leaks['n_leaked']} of {leaks['n_checked']} rows quote "
            "their own privileged context (reference, principle or hidden state) in "
            "an assistant turn; the export scrubs the key, not the reply, so training "
            "on them teaches the model to say what only the grader was told. Drop "
            "those rows (leak_report(...)['leaked'] names them) or pass validate=False."
        )
    dest = output
    if not dest and src:
        path = Path(src)
        dest = str(path.with_name(path.stem + ".train" + (path.suffix or ".jsonl")))
    if format == "trl":
        # What TRL does, not what the SDK meant: every token of a messages
        # row, the completion of a prompt/completion row.
        mask_line = TRL_MASK_COMPLETION if final_only else TRL_MASK_EVERY_TOKEN
        trained = sum(len(r["completion"]) if final_only else len(r["messages"]) for r in rows)
        masked = sum(len(r["prompt"]) for r in rows) if final_only else 0
    else:
        mask_line = "final (unrolled)" if unroll else mask_mode
        trained = sum(sum(m) for m in masks)
        masked = sum(len(m) - sum(m) for m in masks)
    report: dict[str, Any] = {
        "n": len(rows),
        "format": format,
        "with_system": sum(1 for r in rows if _turns(r) and _turns(r)[0]["role"] == "system"),
        "with_tools": sum(1 for r in rows if r.get("tools")),
        "groups": len({r["group_id"] for r in rows if "group_id" in r}),
        "tool_call_roundtrip": roundtrip,
        "mask_mode": mask_line,
        "unrolled": unroll,
        "max_tool_output_chars": max_tool_output_chars,
        "tool_output_truncated": sum(int(r.get("tool_output_truncated") or 0) for r in rows),
        "tool_output_chars_cut": sum(int(r.get("tool_output_chars_cut") or 0) for r in rows),
        "trained_messages": trained,
        "masked_messages": masked,
        "privileged_leaks": {
            k: leaks[k] for k in ("checked", "n_checked", "n_leaked", "leaked", "summary")
        },
    }
    # SFT clones every row it is given. A failed rollout in the file
    # teaches the failure, so say how many there are instead of leaving
    # the caller to notice after training (rejection sampling keeps the
    # passes: Lambert 2025, chapter Rejection Sampling).
    rewards = [_numeric(r.get("reward")) for r in rows]
    n_fail = sum(1 for v in rewards if v is not None and v < PASS_THRESHOLD)
    n_pass = sum(1 for v in rewards if v is not None and v >= PASS_THRESHOLD)
    report["rewards"] = {
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_ungraded": len(rows) - n_pass - n_fail,
    }
    warnings: list[str] = []
    if leaks["n_leaked"]:
        warnings.append(
            f"{leaks['n_leaked']} of {leaks['n_checked']} rows quote their own privileged "
            "context in an assistant turn and are exported anyway (validate=False); "
            "report['privileged_leaks']['leaked'] names them."
        )
    n_cut = sum(1 for r in raw if isinstance(r, dict) and r.get("finish_reason") == "length")
    if n_cut:
        warnings.append(
            f"{n_cut} of {len(raw)} rows were cut by the reply token cap (finish_reason "
            "'length') and are exported as SFT targets; a model trained on them learns to "
            "stop mid-thought. Filter on finish_reason == 'stop' or raise agent_max_tokens=."
        )
    if n_fail:
        warnings.append(
            f"{n_fail} of {len(rows)} rows have reward below 0.5 and are exported as "
            "SFT targets; a model trained on them learns the failure. Pass "
            "`scored.passes()` (or filter on reward) unless that is intended."
        )
    if no_completion_dropped:
        report["no_completion_dropped"] = no_completion_dropped
        warnings.append(
            f"{no_completion_dropped} of {len(masks)} rows have no assistant turn and "
            "were dropped: a TRL prompt/completion row needs a completion."
        )
    if warnings:
        report["warnings"] = warnings
    if push_to and not dest:
        raise ValueError("push_to needs an output path: the file that is written is what is pushed")
    if dest:
        written = [_fireworks_training_row(r) for r in rows] if format == "fireworks" else rows
        report["path"] = write_jsonl(dest, written)
        report["n_written"] = len(written)
    if push_to:
        from whileai.hub import push

        report["hub"] = push(report["path"], push_to)
    return report


# Product name: it exports a dataset, not a training run. The old name
# stays as an alias so nothing written today breaks.
export_dataset = export_training


# Pair metadata that survives export. Scores and margin feed a margin-aware
# loss; the two model names and same_policy tell a reviewer whether the pair
# is on-policy; length_delta is the length-exploit check (Lambert 2025,
# chapter Direct Alignment).
_PAIR_KEYS = (
    "tie",
    "pairwise",
    "first_turn_differs",
    "chosen_score",
    "rejected_score",
    "margin",
    "chosen_model",
    "rejected_model",
    "same_policy",
    "length_delta",
    "chosen_reason",
    "rejected_reason",
    "rejected_failure_class",
    "lineage",
)


def export_preference(
    pairs: Sequence[dict],
    output: str | None = None,
    *,
    system_prompt: str | None = None,
    tools: Sequence[dict] | None = None,
    strip_think: bool = True,
    validate: bool = True,
    drop_ties: bool = True,
    format: str = "openai",
) -> dict[str, Any]:
    """Write chosen/rejected pairs as DPO-style JSONL.

    With ``format="openai"`` (the default) each line is ``{"prompt":
    "<the ask, as text>", "chosen": [...messages...], "rejected":
    [...messages...]}`` in the same wire format as ``export_dataset``:
    both sides are the whole conversation, prompt turns included, and
    tool-call arguments are JSON strings.

    With ``format="fireworks"`` each line is Fireworks' one-turn DPO shape
    (``input.messages`` is the prefix both sides share, then one assistant
    message each as ``preferred_output`` and ``non_preferred_output``);
    pairs that lost later turns are counted as ``fireworks_turns_cut``.

    With ``format="trl"`` each line is TRL's conversational preference
    triple: ``prompt`` is the message list up to the first assistant turn
    and ``chosen``/``rejected`` are the **completions only**, with
    arguments as dicts. The default shape is not loadable by
    ``trl.data_utils.maybe_apply_chat_template`` — a ``prompt`` string
    with conversational sides raises ``TypeError: string indices must be
    integers`` — so pass ``format="trl"`` when a TRL trainer is the
    consumer. Pairs whose chosen or rejected side has no completion after
    the prompt prefix are dropped (``no_completion_dropped``).

    The roundtrip gate runs over BOTH sides and names the encoding it
    checked. Pairs come from ``ScoredData.select_for_preference()`` /
    ``build_preference_pairs``. A pair ``judge_pairs`` marked ``tie``
    carries no preference and is left out (``ties_dropped`` in the
    report) unless ``drop_ties=False``.
    """
    from whileai.simulations import conversation

    if format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {EXPORT_FORMATS}, got {format!r}")
    entries = [p for p in pairs if isinstance(p, dict)]
    not_pairs = [
        i
        for i, p in enumerate(entries)
        if not (isinstance(p.get("chosen"), dict) and isinstance(p.get("rejected"), dict))
    ]
    if not_pairs:
        raise ValueError(
            f"export_preference takes chosen/rejected pairs, and {len(not_pairs)} "
            f"of {len(entries)} entries have no chosen/rejected side. Build the "
            "pairs first: export_preference(build_preference_pairs(rows), ...) "
            "or scored.select_for_preference()."
        )
    system = str(system_prompt or "")
    out_rows: list[dict] = []
    ties_dropped = 0
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        if drop_ties and pair.get("tie"):
            ties_dropped += 1
            continue
        entry: dict[str, Any] = {"prompt": pair.get("prompt")}
        for side in ("chosen", "rejected"):
            row = pair.get(side) or {}
            messages = row.get("messages") or conversation(row)
            entry[side] = _convert_messages(messages, system=system, strip_think=strip_think)
        if tools:
            entry["tools"] = list(tools)
        for key in _PAIR_KEYS:
            if pair.get(key) is not None:
                entry[key] = pair[key]
        out_rows.append(stamp(entry))
    no_completion_dropped = 0
    if format == "trl":
        reshaped = to_trl(out_rows, "preference")
        no_completion_dropped = len(out_rows) - len(reshaped)
        out_rows = reshaped
    check(out_rows, "preference", where="export_preference")
    both_sides = [{"messages": r[side]} for r in out_rows for side in ("chosen", "rejected")]
    roundtrip = tool_call_roundtrip(both_sides, format=format)
    if validate and roundtrip["invalid"]:
        raise ValueError(
            f"tool_call_roundtrip_invalid: {roundtrip['invalid']} of "
            f"{roundtrip['checked']} tool calls in the pairs do not parse "
            "back to structured arguments. Fix the rows or pass "
            "validate=False."
        )
    stats_rows = out_rows
    fireworks_cut = 0
    if format == "fireworks":
        out_rows, dropped, fireworks_cut = _fireworks_preference_rows(out_rows)
        no_completion_dropped += dropped
    report: dict[str, Any] = {
        "pairs": len(out_rows),
        "format": format,
        "tool_call_roundtrip": roundtrip,
    }
    if fireworks_cut:
        report["fireworks_turns_cut"] = fireworks_cut
    if ties_dropped:
        report["ties_dropped"] = ties_dropped
    if no_completion_dropped:
        report["no_completion_dropped"] = no_completion_dropped
    deltas = [r["length_delta"] for r in stats_rows if isinstance(r.get("length_delta"), int)]
    if deltas:
        chosen_longer = sum(1 for d in deltas if d > 0)
        report["chosen_longer_frac"] = round(chosen_longer / len(deltas), 3)
        from .score.judging import length_confound_warning

        length_note = length_confound_warning(chosen_longer, len(deltas))
        if length_note:
            report["warnings"] = [length_note]
    identical = sum(1 for r in stats_rows if r.get("first_turn_differs") is False)
    if identical:
        from .score.judging import first_turn_note

        report["first_turn_identical"] = identical
        report.setdefault("warnings", []).append(first_turn_note(identical, len(out_rows)))
    margins = [r["margin"] for r in stats_rows if isinstance(r.get("margin"), (int, float))]
    if margins:
        report["mean_margin"] = round(sum(margins) / len(margins), 4)
    # A 0-byte JSONL is not an empty dataset, it is a crash downstream:
    # datasets raises a bare StopIteration on it and pyarrow refuses the
    # file. No pairs means no file, loudly.
    if not out_rows:
        if validate:
            raise ValueError(
                "no_preference_pairs: nothing had a chosen and a rejected "
                "side. All-pass or all-fail runs produce no contrast; "
                "regrade or raise difficulty, or pass validate=False to "
                "get the empty report without a file."
            )
        report["path"] = None
        return report
    if output:
        report["path"] = write_jsonl(output, out_rows)
    return report


__all__ = [
    "EXPORT_FORMATS",
    "MASK_MODES",
    "export_dataset",
    "export_preference",
    "export_training",
    "loss_mask",
    "to_trl",
    "tool_call_roundtrip",
    "training_rows",
]
