"""The reviewer: a stock Deep Agents harness over a read-only checkout.

Every arm runs this same file. What changes between arms is the model endpoint
(base, SFT adapter, RL adapter) or the harness profile (the harness arm). The
agent reads the repository at the issue's base commit with Deep Agents' own
filesystem tools, then ends on one line: `VERDICT: approve` or `VERDICT: reject`.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import re
import shutil
import tarfile
import threading
import urllib.request
import uuid

HERE = pathlib.Path(__file__).resolve().parent
REPOS = HERE / ".cache" / "repos"

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

OPENROUTER = "https://openrouter.ai/api/v1"
BASE_MODEL = "qwen/qwen3.8-27b"

SYSTEM = """You review a proposed patch for a GitHub issue. The repository is checked out \
at the commit the patch applies to; read whatever you need with your file tools.

Decide whether the patch fixes the issue without breaking existing behavior. \
Approve only if you would merge it as is.

End your final message with exactly one line:
VERDICT: approve
or
VERDICT: reject"""

VERDICT = re.compile(r"VERDICT:\s*(approve|reject)", re.I)
NUDGE = "Give your review now. End with the VERDICT line."


def _verdict(messages) -> str | None:
    ai = [m for m in messages if getattr(m, "type", None) == "ai"]
    text = ai[-1].content if ai else ""
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    found = VERDICT.findall(text or "")
    return found[-1].lower() if found else None


def snapshot(repo: str, commit: str) -> pathlib.Path:
    """Repository tree at `commit`, fetched once as a GitHub tarball and cached."""
    dest = REPOS / f"{repo.replace('/', '__')}@{commit[:12]}"
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(dest.name, threading.Lock())
    with lock:
        if not dest.exists():
            _fetch(repo, commit, dest)
    return dest


def _long(path: pathlib.Path) -> str:
    """Windows caps paths at 260 characters without the extended-length prefix;
    deep example trees in a few repos go past it."""
    prefix = "\\\\?\\"
    p = str(path.resolve())
    return prefix + p if os.name == "nt" and not p.startswith(prefix) else p


def _fetch(repo: str, commit: str, dest: pathlib.Path) -> None:
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    data = urllib.request.urlopen(url, timeout=120).read()
    tmp = dest.with_suffix(".part")
    if tmp.exists():  # a failed earlier extract
        shutil.rmtree(_long(tmp))
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        top = tar.getmembers()[0].name.split("/")[0]
        for m in tar.getmembers():
            if not (m.isfile() or m.isdir()):
                continue
            m.name = m.name[len(top) + 1 :] if m.name != top else "."
            if m.name and m.name != ".":
                tar.extract(m, _long(tmp), filter="data")
    tmp.rename(dest)


def use_profile(path: pathlib.Path, model_name: str = BASE_MODEL) -> None:
    """Register a candidate's HarnessProfile for this model, for this process."""
    from importlib.util import module_from_spec, spec_from_file_location

    from deepagents import register_harness_profile

    spec = spec_from_file_location(path.stem, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    # ChatOpenAI reports provider "openai", whatever the base_url.
    register_harness_profile(f"openai:{model_name}", mod.PROFILE)


def prompt(task: dict) -> str:
    return f"## Issue\n\n{task['issue'].strip()}\n\n## Proposed patch\n\n```diff\n{task['patch'].strip()}\n```"


def model(base_url: str = OPENROUTER, name: str = BASE_MODEL, api_key: str | None = None):
    from langchain_openai import ChatOpenAI

    extra_body: dict = {}
    if base_url == OPENROUTER:
        extra_body["reasoning"] = {"max_tokens": 8192}
        # One full-precision host for every base call. OpenRouter otherwise
        # spreads a model over many hosts, some fp8/fp4, and that is noise in a
        # before/after.
        extra_body["provider"] = {"order": ["deepinfra/bf16"], "allow_fallbacks": False}

    return ChatOpenAI(
        base_url=base_url,
        model=name,
        api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        temperature=0.6,
        # Qwen3.8 thinks before every turn. At 4096 it spent the whole budget on
        # reasoning and returned an empty turn on 8 of 20 pilot reviews; the
        # reasoning cap leaves room for the tool call or the verdict.
        max_tokens=16384,
        extra_body=extra_body,
        timeout=300,
        max_retries=3,
    )


def build(chat_model, root: pathlib.Path):
    from deepagents import FilesystemPermission, create_deep_agent
    from deepagents.backends import FilesystemBackend

    return create_deep_agent(
        model=chat_model,
        system_prompt=SYSTEM,
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
    )


def review(task: dict, chat_model, *, arm: str, max_steps: int = 40) -> dict:
    """Run one review. Returns the verdict, the call counts and the root run id."""
    run_id = uuid.uuid4()
    config = {
        "run_id": run_id,
        "run_name": "review",
        "recursion_limit": max_steps * 2 + 1,
        "tags": [arm, task["split"]],
        # One thread per review, so a nudged review stays one trajectory when
        # smithtune pulls it (a pulled root run brings its whole thread).
        "metadata": {
            "thread_id": str(run_id),
            "task_id": task["id"],
            "arm": arm,
            "split": task["split"],
            "repo": task["repo"],
        },
    }
    last_run = run_id
    error = None
    messages = [{"role": "user", "content": prompt(task)}]
    verdict = None
    try:
        agent = build(chat_model, snapshot(task["repo"], task["base_commit"]))
        messages = agent.invoke({"messages": messages}, config=config)["messages"]
        verdict = _verdict(messages)
        if verdict is None:
            # Same nudge in every arm: models often stop on an empty turn after
            # reading. One follow-up, counted in the calls like any other.
            last_run = uuid.uuid4()
            messages = agent.invoke(
                {"messages": [*messages, {"role": "user", "content": NUDGE}]},
                config={**config, "run_id": last_run, "run_name": "review-nudge"},
            )["messages"]
            verdict = _verdict(messages)
    except Exception as e:  # tarball gone, recursion limit, provider error: the review failed
        error = f"{type(e).__name__}: {e}"[:300]
        messages = []

    ai = [m for m in messages if getattr(m, "type", None) == "ai"]
    tool_calls = sum(len(getattr(m, "tool_calls", []) or []) for m in ai)
    return {
        "task_id": task["id"],
        "arm": arm,
        "split": task["split"],
        "label": task["label"],
        "verdict": verdict,
        "correct": verdict == task["label"],
        "model_calls": len(ai),
        "tool_calls": tool_calls,
        # The tool sequence, kept locally so picking training rows never has
        # to read runs back from LangSmith one by one (that endpoint rate-limits).
        "tools": [tc["name"] for m in ai for tc in (getattr(m, "tool_calls", None) or [])],
        "nudged": last_run != run_id,
        "run_id": str(last_run),  # the root run holding the whole conversation
        "error": error,
    }


if __name__ == "__main__":
    import sys

    import data

    task = data.load("dev", limit=1)[0]
    print(json.dumps(review(task, model(), arm="smoke"), indent=2), file=sys.stdout)
