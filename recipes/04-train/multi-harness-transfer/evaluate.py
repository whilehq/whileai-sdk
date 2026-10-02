"""Score one checkpoint on SmolDataEnvs test tasks under many harnesses, k samples each.

A fork of FineEnvs' `05-multi-harness-rl/eval/evaluate.py` (FineEnvs @ 26ab9c6) with three
changes: the harness list is an argument, each task/harness pair is sampled `--samples` times,
and the sandbox backend is an argument. Sampling, step limit, timeout, reward key and the
tool-count rule are theirs, unchanged. One JSON file per (task, harness, sample); a rerun
retries only ungraded cells.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from smoldataenv_harbor.tasks import load_tasks

# FineEnvs' eval protocol (eval/evaluate.py @ 26ab9c6).
SAMPLING = {"temperature": 0.8, "top_p": 1.0, "top_k": -1}
AGENT_STEP_LIMIT = 17
AGENT_TIMEOUT_SEC = 600


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


_local = threading.local()


def _env(args):
    """One client per worker thread, reused across rollouts.

    `HarborEnv.close()` in sync mode never awaits its async close (OpenEnv @ 7ee88d5 warns
    "coroutine 'MCPClientBase._close_async' was never awaited"), so a client per rollout leaks one
    server-side session per cell; near 4,000 cells that filled 32 GB, and on the SFT runs it turned
    into every call failing. A thread-local client opens one connection per worker instead.
    """
    from harbor_env import HarborEnv

    if getattr(_local, "env", None) is None:
        _local.env = HarborEnv(args.server, message_timeout_s=1800)
    return _local.env


def episode(args, split, index, task, harness):
    import httpx
    from harbor_env.harness import HarborSession

    session = HarborSession(
        env=_env(args),
        owns_env=False,
        split=split,
        task_index=index,
        instruction=task["instruction"],
        harness=harness,
        sandbox=args.sandbox,
        llm_url=args.vllm_url,
        model=args.model,
        reward_key="correctness,reward",
        sampling=SAMPLING,
        agent_step_limit=AGENT_STEP_LIMIT,
        agent_timeout_sec=AGENT_TIMEOUT_SEC,
    )
    try:
        session.wait_for_completion(timeout_s=1200)
        correctness = session.verify([]).env_reward
        if correctness not in (0, 1):
            raise RuntimeError("No verifier grade")
        trace = session.fetch_training_trace()
        r = httpx.get(
            args.server.rstrip("/")
            + f"/smoldataenv/trials/{session.result.trial_name}/tool-count",
            timeout=30,
        )
        r.raise_for_status()
        return {
            "correctness": correctness,
            "tool_calls": r.json()["native_tool_calls"],
            "generated_tokens": sum(len(t.completion_token_ids) for t in trace.turns),
            "prompt_tokens": sum(len(t.prompt_token_ids) for t in trace.turns),
            "model_calls": len(trace.turns),
            "trial": session.result.trial_name,
        }
    finally:
        session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="LiquidAI/LFM2.5-2.6B", help="served model name")
    p.add_argument("--checkpoint", required=True, help="repo@revision being scored")
    p.add_argument("--harnesses", required=True, help="comma-separated seam names")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--tasks", type=int, default=250)
    p.add_argument("--sandbox", default="modal")
    p.add_argument("--data", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8200")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    p.add_argument("--output", required=True)
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()

    import httpx

    split = httpx.get(args.server.rstrip("/") + "/smoldataenv/splits", timeout=30).json()["test"]
    tasks = list(enumerate(load_tasks(args.data, "test")))[: args.tasks]
    harnesses = [h.strip() for h in args.harnesses.split(",") if h.strip()]
    out = Path(args.output)
    identity = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "sampling": SAMPLING,
        "agent_step_limit": AGENT_STEP_LIMIT,
        "agent_timeout_sec": AGENT_TIMEOUT_SEC,
        "sandbox": args.sandbox,
        "agent_versions": json.loads(os.environ.get("OPENENV_HARBOR_AGENT_VERSIONS", "{}")),
    }
    ident_path = out / "identity.json"
    if ident_path.exists() and json.loads(ident_path.read_text()) != identity:
        raise ValueError("Output directory belongs to a different evaluation")
    write_json(ident_path, identity)

    cells = [
        (i, t, h, s) for i, t in tasks for h in harnesses for s in range(args.samples)
    ]

    def one(cell):
        index, task, harness, sample = cell
        path = out / "cells" / f"{task['name']}--{harness}--{sample}.json"
        if path.exists():
            prev = json.loads(path.read_text())
            if prev.get("correctness") in (0, 1):
                return prev
        start = time.monotonic()
        row = {
            "task": task["name"],
            "difficulty": task["difficulty"],
            "harness": harness,
            "sample": sample,
            "checkpoint": args.checkpoint,
        }
        try:
            row.update(episode(args, split, index, task, harness))
        except Exception as exc:  # noqa: BLE001 - an ungraded cell is retried on rerun
            _local.env = None  # a failed call may have broken this thread's connection; reopen
            detail = str(exc)
            for k, v in os.environ.items():
                if v and any(w in k for w in ("TOKEN", "SECRET", "API_KEY")):
                    detail = detail.replace(v, "[redacted]")
            row.update(correctness=None, error=type(exc).__name__, detail=detail[:1000])
        row["seconds"] = round(time.monotonic() - start, 1)
        write_json(path, row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for fut in as_completed([pool.submit(one, c) for c in cells]):
            rows.append(fut.result())
            if len(rows) % 25 == 0 or len(rows) == len(cells):
                graded = [r for r in rows if r.get("correctness") in (0, 1)]
                print(
                    f"{len(rows)}/{len(cells)} done, {len(graded)} graded, "
                    f"pass {sum(r['correctness'] for r in graded) / max(1, len(graded)):.3f}",
                    flush=True,
                )

    summary = {}
    for h in harnesses:
        g = [r for r in rows if r["harness"] == h and r.get("correctness") in (0, 1)]
        errs = {}
        for r in rows:
            if r["harness"] == h and r.get("error"):
                errs[r["error"]] = errs.get(r["error"], 0) + 1
        summary[h] = {
            "cells": sum(r["harness"] == h for r in rows),
            "graded": len(g),
            "pass": sum(r["correctness"] for r in g) / len(g) if g else None,
            "errors": errs,
        }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
