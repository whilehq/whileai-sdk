"""What every document-extraction candidate shares: the code tool, the agent
loop, the chat backends, and ``build``.

A candidate is a file in ``candidates/`` defining ``harness(model) ->
wai.Harness``; it changes the instructions, the turn cap, the retries and the
validator, and nothing here. ``build`` returns a harness whose agent is a
callable (``kind="callable"``), so ``wai.simulate`` inside the Meta-Harness
recipe plays it as it is: the model's code really runs, on the document.

The model string picks the backend:

* ``scripted`` / ``scripted-b`` (the dry run): a stand-in that knows the gold
  record and makes planted mistakes that a rule in the instructions removes.
* ``vllm:<hub id>@<url>``: your own vLLM endpoint (``serve_modal.py``), key in
  ``VLLM_API_KEY``.

Every model turn is a step carrying ``input_tokens`` and ``output_tokens``,
so the row's ``usage`` is real and the recipe's cost gate reads tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import whileai as wai
from whileai.harness import Disclosure

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import docs

# TOOL_TIMEOUT_S = 10: wall clock for one run of the model's code; the
# harness-and-weights recipe's run_python uses the same (convention, untested).
TOOL_TIMEOUT_S = 10.0
# TOOL_CHARS = 2000: the tool output is cut here so a print loop cannot fill
# the context; the message says when it was cut (convention, untested).
TOOL_CHARS = 2000
# SAMPLING: the Nemotron-Nano-8B-v1 model card's recommended temperature 0.6
# and top_p 0.95; 1024 reply tokens a turn. The same for every candidate and
# model, so sampling is not a lever in this search.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 1024}
# REQUEST_TIMEOUT_S = 240: one reply is at most 1024 tokens, well under a
# minute at the slowest rate seen; a request that hangs past this is retried
# instead of holding a round-synchronous batch (and the GPU) idle.
REQUEST_TIMEOUT_S = 240
# SALT: moves every sampling seed, for the three-run noise floor (DOCX_SALT).
SALT = int(os.environ.get("DOCX_SALT", "0"))

RUN_PYTHON = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run Python 3 (standard library, no network, 10 s) with the document text as the "
            "string DOC; returns what it printed."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}

Chat = Callable[[list[dict[str, str]], int], tuple[str, dict[str, Any]]]

# The baseline's instructions (candidates/00_baseline.py). A candidate
# written as named edits appends each edit's text to these.
BASE_INSTRUCTIONS = """detailed thinking off
You extract structured data from business documents. The user gives you the document type, the fields to extract with a short description of each, and the document text, which may contain OCR errors.

You have a Python tool. To use it, reply with exactly one ```python code block and nothing else. It runs in a sandbox (Python 3 standard library, no network, 10 second limit) where the document text is the string variable DOC, and whatever it prints comes back to you as the next message. You may use the tool up to 3 times.

When you are done, reply with one ```json block containing a single JSON object whose keys are exactly the requested field names. Write dates as YYYY-MM-DD and money as a plain number (for example 1234.5). Use null for a field the document does not give."""

# ---------------------------------------------------------------- the code tool

_PRELUDE = """
import socket as _socket
def _no_network(*a, **k):
    raise OSError("network is disabled in this sandbox")
_socket.socket = _no_network
_socket.create_connection = _no_network
_socket.getaddrinfo = _no_network
try:
    import resource as _r
    _r.setrlimit(_r.RLIMIT_CPU, (10, 10))
except Exception:
    pass
with open("doc.txt", encoding="utf-8") as _f:
    DOC = _f.read()
del _f
"""


def run_python(code: str, doc: str) -> str:
    """The code tool: the model's code in a fresh interpreter (``-I``, an empty
    environment, a temp directory, socket calls replaced by an error, CPU
    capped), with the document as the string ``DOC``. Returns what it printed."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "doc.txt").write_text(doc, encoding="utf-8")
        Path(tmp, "snippet.py").write_text(_PRELUDE + "\n" + code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "snippet.py"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=TOOL_TIMEOUT_S,
                env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {TOOL_TIMEOUT_S:.0f}s"
    err = proc.stderr.strip()
    if err:
        err = "\n".join(err.splitlines()[-6:])  # the traceback's tail is the useful part
    text = (proc.stdout or "") + (("\n" + err) if err else "")
    text = text.strip() or f"(no output, exit {proc.returncode})"
    if len(text) > TOOL_CHARS:
        text = text[:TOOL_CHARS] + f"\n[cut to the first {TOOL_CHARS} characters]"
    return text


# ---------------------------------------------------------------- the loop

_PY = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)
_JSONFENCE = re.compile(r"```json\s*\n(.*?)```", re.S)
_DOC = re.compile(r"Document:\n<<<\n(.*)\n>>>\s*$", re.S)
_TYPE = re.compile(r"^Document type: (.+)$", re.M)
LAST_TURN = "That was your last tool call. Reply now with the ```json block."


def _allowed(desc: str) -> list[str]:
    """The values a field's description lists after "one of:" or as ISO codes."""
    m = re.search(r"one of:?\s*([^.;]+)", desc)
    if m:
        return [x.strip() for x in m.group(1).split(",") if x.strip()]
    m = re.search(r"ISO 4217 code:\s*([A-Z, ]+?)(?:\s+or\s+([A-Z]{3}))?$", desc.strip())
    if m:
        codes = [x.strip() for x in m.group(1).split(",") if x.strip()]
        if m.group(2):
            codes.append(m.group(2))
        return codes
    return []


def schema_problems(obj: dict[str, Any], doc_type: str, *, enums: bool = False) -> list[str]:
    """What a validator can see without the gold: the keys and the value
    shapes; with ``enums``, also that an enum or code field holds one of the
    values its description lists."""
    schema = docs.SCHEMAS[doc_type]
    out = []
    missing = [k for k in schema if k not in obj]
    extra = [k for k in obj if k not in schema]
    if missing:
        out.append("missing keys: " + ", ".join(missing))
    if extra:
        out.append("unexpected keys: " + ", ".join(extra))
    for k, (kind, _) in schema.items():
        v = obj.get(k)
        if v is None:
            continue
        if kind == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
            out.append(f"{k} is not YYYY-MM-DD: {v!r}")
        if kind == "money" and (isinstance(v, bool) or not isinstance(v, (int, float))):
            out.append(f"{k} is not a plain number: {v!r}")
        if kind == "int" and (isinstance(v, bool) or not isinstance(v, int)):
            out.append(f"{k} is not an integer: {v!r}")
        if enums and kind in ("enum", "code"):
            allowed = _allowed(schema[k][1])
            if allowed and str(v).strip().lower() not in [a.lower() for a in allowed]:
                out.append(f"{k} is not one of the allowed values ({', '.join(allowed)}): {v!r}")
    return out


def agent(chat: Chat, *, instructions: str, max_turns: int, retries: int, validate: bool | str):
    """The harness's loop as a callable ``prompt -> trajectory``. A reply that
    is a ```python block (with no ```json block) runs in the tool while turns
    remain; the reply with the JSON is final. With ``retries``, an unparseable
    final (or, with ``validate``, a malformed one) is sent back with the reason."""

    seen: dict[str, int] = {}
    seen_lock = threading.Lock()

    def run(prompt: str) -> dict[str, Any]:
        doc_m, type_m = _DOC.search(prompt), _TYPE.search(prompt)
        doc = doc_m.group(1) if doc_m else prompt
        doc_type = type_m.group(1).strip().replace(" ", "_") if type_m else ""
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
        steps: list[dict[str, Any]] = []
        turns, retries_left, final, cut = 0, retries, "", False
        # the n-th time this harness plays this prompt is rollout n: each of
        # the k rollouts gets its own seed (the engine does not pass the index)
        with seen_lock:
            nth = seen.get(prompt, 0)
            seen[prompt] = nth + 1
        base = int(hashlib.sha256(f"{prompt}:{SALT}:{nth}".encode()).hexdigest()[:8], 16)
        while True:
            reply, u = chat(messages, base + 7919 * turns)
            turns += 1
            cut = u.get("finish_reason") == "length"
            steps.append(
                {
                    "model_turn": turns,
                    "input_tokens": int(u.get("prompt_tokens") or 0),
                    "output_tokens": int(u.get("completion_tokens") or 0),
                    "truncated": cut,
                }
            )
            messages.append({"role": "assistant", "content": reply})
            code = _PY.findall(reply)
            if code and not _JSONFENCE.search(reply) and turns < max_turns:
                out = run_python(code[-1], doc)
                steps.append({"tool": "run_python", "arguments": {"code": code[-1]}, "result": out})
                msg = f"run_python output:\n{out}"
                if turns == max_turns - 1:
                    msg += "\n\n" + LAST_TURN
                messages.append({"role": "user", "content": msg})
                continue
            final = reply
            answer, why = docs.parse_answer(reply)
            problems = [why] if answer is None else []
            if answer is not None and validate and doc_type in docs.SCHEMAS:
                problems = schema_problems(answer, doc_type, enums=validate == "schema+enum")
            if problems and retries_left > 0:
                retries_left -= 1
                steps.append({"retry": problems})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your answer cannot be accepted: "
                        + "; ".join(problems)
                        + ". Reply with only the corrected ```json block.",
                    }
                )
                continue
            break
        return {
            "steps": steps,
            "final_text": final,
            "finish_reason": "length" if cut else "stop",
        }

    return run


# ---------------------------------------------------------------- the backends


def endpoint_chat(url: str, model: str, key: str) -> Chat:
    """An OpenAI-compatible chat call against your own vLLM server."""
    import requests

    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    extra: dict[str, Any] = {}
    if "qwen3" in model.lower():
        # a hybrid-thinking checkpoint: served with thinking off, the same way
        # Nemotron is told "detailed thinking off" (a serving setting, not a lever)
        extra["chat_template_kwargs"] = {"enable_thinking": False}

    def chat(messages: list[dict[str, str]], seed: int) -> tuple[str, dict[str, Any]]:
        body = {"model": model, "messages": messages, "seed": seed, **SAMPLING, **extra}
        err = ""
        for attempt in range(6):
            try:
                r = requests.post(
                    f"{base}/chat/completions",
                    json=body,
                    timeout=REQUEST_TIMEOUT_S,
                    headers={"Authorization": f"Bearer {key}"},
                )
                if r.status_code == 200:
                    data = r.json()
                    choice = data["choices"][0]
                    u = dict(data.get("usage") or {})
                    u["finish_reason"] = choice.get("finish_reason")
                    return choice["message"].get("content") or "", u
                if r.status_code == 400:  # context overflow: an honest failed reply
                    return "", {"finish_reason": "length", "error": r.text[:200]}
                err = f"{r.status_code} {r.text[:200]}"
            except requests.RequestException as exc:
                err = str(exc)[:200]
            time.sleep(min(30, 5 * 2**attempt))
        raise RuntimeError(f"endpoint failed six times: {err}")

    return chat


_GOLD: dict[str, dict[str, Any]] = {}


def scripted_chat(name: str) -> Chat:
    """The offline stand-in. It knows the gold record for every document and
    makes the mistakes a small model makes, each at a planted rate that a rule
    in the system prompt removes, so a candidate that states the rule scores
    higher. It sends one ```python block first (so the tool path runs) and the
    JSON after. Its numbers show the mechanics, not a result."""
    if not _GOLD:
        _GOLD.update({docs.task_prompt(a): a for a in docs.build()})

    def draw(*parts: Any) -> float:
        key = ":".join(map(str, (name, *parts)))
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def chat(messages: list[dict[str, str]], seed: int) -> tuple[str, dict[str, Any]]:
        system, user = messages[0]["content"].lower(), messages[1]["content"]
        ask = _GOLD[user]
        turn = sum(1 for m in messages if m["role"] == "assistant")
        u: dict[str, Any] = {
            "prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
            "finish_reason": "stop",
        }
        if turn == 0:
            u["completion_tokens"] = 20
            return "```python\nprint(len(DOC.splitlines()), 'lines')\n```", u
        out = dict(ask["gold"])
        for f, (kind, _) in docs.SCHEMAS[ask["doc_type"]].items():
            r = draw(ask["id"], f, seed, SALT)
            gold = ask["gold"][f]
            if gold is None and r < 0.5 and "never guess" not in system:
                out[f] = "unknown"
            elif kind == "money" and gold is not None and r < 0.3 and "sum" not in system:
                out[f] = round(gold * 1.1, 2)
            elif kind == "date" and gold is not None and r < 0.25 and "dd.mm" not in system:
                out[f] = gold[5:7] + "/" + gold[8:] + "/" + gold[:4]
            elif kind == "name" and r < 0.2 and "first last" not in system:
                out[f] = " ".join(reversed(str(gold).split()))
        u["completion_tokens"] = 80
        if draw(ask["id"], "parse", seed, SALT) < 0.08 and turn < 2:
            return "Here are the fields: " + json.dumps(out)[:-3], u
        return f"```json\n{json.dumps(out)}\n```", u

    return chat


def chat_for(model: str) -> Chat:
    if model.startswith("scripted"):
        return scripted_chat(model)
    if not model.startswith("vllm:") or "@" not in model:
        raise ValueError(f"model {model!r}: use vllm:<hub id>@<url> or scripted")
    name, url = model[len("vllm:") :].split("@", 1)
    key = os.environ.get("VLLM_API_KEY")
    if not key:
        raise SystemExit("set VLLM_API_KEY to the key your docx-serve app was deployed with")
    return endpoint_chat(url, name, key)


# ---------------------------------------------------------------- building a harness


def build(
    model: str,
    *,
    instructions: str,
    label: str,
    max_turns: int = 4,
    retries: int = 0,
    validate: bool | str = False,
) -> wai.Harness:
    """The candidate as a ``wai.Harness`` with a callable agent: the
    fingerprint hashes the instructions, the tool, the turn cap, the retries,
    the validator (``True``: keys and value shapes; ``"schema+enum"``: also
    the listed values of enum and code fields) and the sampling."""
    loop = None
    if model:  # a harness built only to read its fingerprint needs no backend
        loop = agent(
            chat_for(model),
            instructions=instructions,
            max_turns=max_turns,
            retries=retries,
            validate=validate,
        )
    return wai.Harness(
        model,
        instructions=instructions,
        tools=[RUN_PYTHON],
        agent=loop,
        label=label,
        disclosure=Disclosure(
            max_turns=max_turns,
            retries=retries,
            sampling=SAMPLING,
            notes=("validate: schema+enum" if validate == "schema+enum" else "validate: schema")
            if validate
            else None,
        ),
    )


@dataclass(frozen=True)
class Edit:
    """One named change to the baseline, so ``run.py --prune`` can take it
    back out: text appended to the instructions, and loop settings."""

    instructions: str = ""
    max_turns: int | None = None
    retries: int | None = None
    validate: bool | str | None = None


def from_edits(
    model: str, edits: dict[str, Edit], *, label: str, drop: tuple[str, ...] = ()
) -> wai.Harness:
    """A candidate as named edits on the baseline, minus the ones in ``drop``.
    Instructions: the baseline's, then each kept edit's text in order; loop
    settings: the last kept edit that sets each one wins."""
    unknown = sorted(set(drop) - set(edits))
    if unknown:
        raise ValueError(f"no edit named {', '.join(unknown)}; edits are {', '.join(edits)}")
    kept = [e for name, e in edits.items() if name not in drop]
    text = "\n\n".join([BASE_INSTRUCTIONS, *(e.instructions for e in kept if e.instructions)])
    turns, retries, validate = 4, 0, False
    for e in kept:
        turns = e.max_turns if e.max_turns is not None else turns
        retries = e.retries if e.retries is not None else retries
        validate = e.validate if e.validate is not None else validate
    return build(
        model,
        instructions=text,
        label=label if not drop else f"{label} minus {', '.join(drop)}",
        max_turns=turns,
        retries=retries,
        validate=validate,
    )
