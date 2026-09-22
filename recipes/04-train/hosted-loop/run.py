"""The hosted loop from the SDK: push, train, serve, call. One key, one A10G run.

    python run.py                  # all four steps, SFT on Qwen/Qwen3-4B, then a chat call
    python run.py data             # simulate, grade, split, push train + holdout
    python run.py train            # wai.train on the pushed set, wait for it
    python run.py serve            # wai.serve the adapter under --name
    python run.py call             # one chat completion against the endpoint
    python run.py models           # what the account hosts

Needs a key: ``wai login`` or WHILEAI_API_KEY
(https://docs.while.ai/get-started/quickstart). The
rows come from the offline template writer and a scripted agent, so no model
key is needed to build them. Training runs on the platform's A10G (about a
minute for SFT); serving wakes a GPU that bills by the hour and the first
call after idle pays a cold start of a few minutes. State between steps
lives in hosted-loop.json next to this file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

import whileai.simulations as wai
from whileai.auth import resolve_api_key
from whileai.config import provenance
from whileai.simulations.score.judging import run_judge

STATE = Path(__file__).with_name("hosted-loop.json")
DOCS = "https://docs.while.ai/api/training"

# CALL_WINDOW_S = 900: the README's "call waits up to fifteen" minutes; one
# ceiling for a slow first reply and for gateway errors while the container wakes.
CALL_WINDOW_S = 900
# WARMUP_STATUSES: what the gateway answers while the serving container is
# still starting; any other status is a real error and raises at once.
WARMUP_STATUSES = frozenset({502, 503, 504})
# WARMUP_RETRY_S = 5, WARMUP_RETRY_MAX_S = 60: first gap between retries during
# warm-up, doubling to the cap; a cold start takes minutes, so polling faster
# only adds log lines.
WARMUP_RETRY_S = 5
WARMUP_RETRY_MAX_S = 60

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund",
            "description": "Refund an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id", "amount"],
            },
        },
    },
]
POLICY = (
    "You are a refund assistant. Look up an order before refunding it. If the lookup "
    "fails, say so and stop. Never invent an order id."
)


def scripted_agent(message: str) -> dict:
    """Reliability depends on the ask, so the graded set has passes to imitate
    and contrast between repeats: some asks always pass, some never, some
    on alternate repeats."""
    from whileai.simulations.generate.agents import current_rollout

    match = re.search(r"\b(\d{4,})\b", message)
    order = match.group(1) if match else None
    if order is None:
        return {
            "steps": [],
            "final_text": "Which order would you like refunded? I need the order id.",
        }
    k = current_rollout.rollout_index or 0
    bucket = sum(ord(c) for c in message) % 3
    lookup = {
        "tool": "lookup_order",
        "arguments": {"order_id": order},
        "result": {"status": "ok", "total": 42.0},
    }
    refund = {
        "tool": "create_refund",
        "arguments": {"order_id": order, "amount": 42.0},
        "result": {"status": "ok"},
    }
    if bucket == 0 or (bucket == 1 and k % 2 == 0):
        return {"steps": [lookup, refund], "final_text": f"Refunded order {order} for $42.00."}
    return {"steps": [refund], "final_text": f"Refunded order {order}."}


def judge(row: dict) -> int:
    """1 when the refund followed a lookup, or the agent asked for the id."""
    tools = [s.get("tool") for s in row.get("steps") or []]
    if "create_refund" in tools:
        return 1 if tools[0] == "lookup_order" else 0
    return 1 if "order id" in str(row.get("final_text", "")).lower() else 0


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(**fields) -> None:
    state = load()
    state.update(fields)
    STATE.write_text(json.dumps(state, indent=2))


def need(state: dict, *keys: str) -> None:
    missing = [k for k in keys if not state.get(k)]
    if missing:
        sys.exit(f"missing {', '.join(missing)} in {STATE.name}; run the earlier step first")


# ------------------------------------------------------------------ steps


def step_data(args: argparse.Namespace, push: bool = True) -> None:
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,  # template writer: no model key
        mode="rl",
        budget=args.budget,
        repeats=4,
        seed=args.seed,
        time_budget=None,
    )
    scored = run_judge(data.trajectories, judge)
    print(f"simulated {len(data.trajectories)} rows, pass@1 {scored.pass_at.pass_at_1:.2f}")
    train, holdout = wai.split_pseudo_production(scored.rows, fraction=0.25, seed=args.seed)
    print(f"split by task: train {len(train)} rows, holdout {len(holdout)} rows")
    if not push:
        print("dry run: these are the rows `data` would push; nothing sent, no key used")
        return
    pushed = wai.push_rows(
        train,
        f"{args.name}-train",
        purpose="train",
        mode="rl",
        agent=args.name,
        gate=True,
        description="recipes/04-train/hosted-loop: scripted refund agent, template situations",
    )
    train_id = pushed["datasetId"]
    held = wai.push_rows(
        holdout, f"{args.name}-holdout", purpose="holdout", agent=args.name, parent=train_id
    )
    gate = pushed.get("gate") or {}
    print(f"pushed train {train_id} (gate ok={gate.get('ok')}), holdout {held['datasetId']}")
    for line in gate.get("warnings") or []:
        print("  gate:", line)
    save(train_id=train_id, holdout_id=held["datasetId"])


def step_train(args: argparse.Namespace) -> None:
    state = load()
    need(state, "train_id", "holdout_id")
    started = time.time()
    run = wai.train(
        state["train_id"],
        method=args.method,
        holdout=state["holdout_id"],
        base_model=args.base,
        epochs=args.epochs if args.method == "sft" else None,
        steps=args.steps if args.method != "sft" else None,
    )
    print(f"started {run.method} run {run.run_id}: {run.url}")
    save(run_id=run.run_id)
    status = run.wait(timeout=args.timeout, poll=20)
    t = run.training
    print(
        f"{status} in {time.time() - started:.0f}s: {t.get('metric', 'loss')} "
        f"{t.get('before')} -> {t.get('after')} on {t.get('holdoutRows')} held-out rows"
    )
    if status != "done":
        sys.exit(f"run {run.run_id} {status}: {run.error}")
    print(f"adapter: {run.adapter}")


def step_serve(args: argparse.Namespace) -> None:
    state = load()
    need(state, "run_id")
    model = wai.serve(args.name, state["run_id"])
    print(f"serving {model['name']} v{model.get('version')} on {model['baseModel']}")
    print(f"endpoint {model['endpoint']}")
    save(endpoint=model["endpoint"], model_name=model["name"])


def step_call(args: argparse.Namespace) -> None:
    state = load()
    need(state, "endpoint", "model_name")
    body = {
        "model": state["model_name"],
        "messages": [
            {"role": "system", "content": POLICY},
            {"role": "user", "content": "please refund order 88213"},
        ],
        "max_tokens": 200,
        "temperature": 0,
        # Qwen3 reasons first by default; the reply is what the agent would say.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    deadline = started + CALL_WINDOW_S
    delay = WARMUP_RETRY_S
    while True:
        res = requests.post(
            state["endpoint"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {resolve_api_key()}"},
            json=body,
            # the first call after idle pays the cold start: wait out the window
            timeout=max(1.0, deadline - time.time()),
        )
        print(f"HTTP {res.status_code} in {time.time() - started:.0f}s")
        if res.status_code not in WARMUP_STATUSES:
            break
        # a 502/503/504 this early is the gateway answering for a container
        # that is still waking, not the model; retry inside the same window
        if time.time() + delay > deadline:
            sys.exit(
                f"HTTP {res.status_code} after {time.time() - started:.0f}s: the endpoint "
                "is still starting; run `python run.py call` again in a minute."
            )
        print(f"  endpoint still starting; retrying in {delay:.0f}s")
        time.sleep(delay)
        delay = min(delay * 2, WARMUP_RETRY_MAX_S)
    res.raise_for_status()
    print(res.json()["choices"][0]["message"]["content"].strip())


def step_models(args: argparse.Namespace) -> None:
    for m in wai.models():
        print(f"{m['name']:24s} v{m.get('version')}  {m['baseModel']}  run {m.get('adapterRunId')}")


STEPS = {
    "data": step_data,
    "train": step_train,
    "serve": step_serve,
    "call": step_call,
    "models": step_models,
}


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", nargs="?", default="all", choices=["all", *STEPS])
    parser.add_argument("--name", default="hosted-loop", help="dataset, agent and model name")
    parser.add_argument("--method", default="sft", choices=["sft", "grpo", "dpo"])
    parser.add_argument("--base", default="Qwen/Qwen3-4B", help="a served base, or nothing serves")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="the data step offline: simulate, grade, split, push nothing; no key, no GPU",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        print("== data (dry run)")
        step_data(args, push=False)
        return 0
    if not resolve_api_key():
        sys.exit(f"No API key. Run `wai login` or set WHILEAI_API_KEY ({DOCS}).")
    steps = list(STEPS) if args.step == "all" else [args.step]
    if args.step == "all":
        steps.remove("models")
    for name in steps:
        print(f"== {name}")
        STEPS[name](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
