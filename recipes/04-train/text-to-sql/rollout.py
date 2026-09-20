"""Sample a policy on the task set, k times per task, through `wai.simulate`.

    python rollout.py --model qwen3-4b                       # hosted Qwen3-4B, thinking off
    python rollout.py --model sonnet-5 --split holdout       # Claude via OPENAI_BASE_URL or Bedrock
    python rollout.py --hosted t2s-r1 --split holdout        # a model you served with wai.serve
    python rollout.py --agent "openai:gpt-4.1-mini"          # any agent spec the SDK accepts

`simulate(tasks=...)` replays exactly these prompts on their scenario ids,
`repeats` times each, on the agent; the gold SQL is attached to the rows
afterwards (the SDK never carries an answer key through a rollout) and the
rows land in raw/<model>.jsonl for build.py.

The account's Qwen3-4B endpoint runs with thinking off through the SDK. A
model served under its own name (wai.serve, `--hosted`) thinks by default;
to benchmark the *base* with thinking on, serve it under a name:
`wai.serve("qwen3-4b-think", base_model="Qwen/Qwen3-4B")`, then
`--hosted qwen3-4b-think`. The agent's reply budget follows
ZP_CONTEXT_TOKENS (2048 tokens once it is above 8192), which this script
sets when unset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "ZP_CONTEXT_TOKENS", "16384"
)  # before the SDK import: agent replies up to 2048

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sql_verifier import (
    AGENT,
    RAW,
    load_tasks,
    read_jsonl,
    split_of,
    system_prompt,
)

import whileai.simulations as wai
from whileai.config import provenance

SERVE_URL = "https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1"

# name -> agent spec. "vllm:<model>@<url>" and "openai:<model>" are SDK specs;
# a callable is the SDK's bring-your-own-agent contract.
MODELS: dict[str, object] = {
    "qwen3-4b": f"vllm:Qwen/Qwen3-4B@{SERVE_URL}",
    "sonnet-5": "claude-sonnet-5",
    "haiku-4.5": "claude-haiku-4-5-20251001",
}
BEDROCK_IDS = {
    "claude-sonnet-5": "global.anthropic.claude-sonnet-5",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def claude_agent(model: str, max_tokens: int = 800):
    """Claude as a callable agent: the Claude API with ANTHROPIC_API_KEY, else Bedrock."""
    import anthropic

    sys_p = system_prompt()
    if os.environ.get("ANTHROPIC_API_KEY"):
        client: anthropic.Anthropic | anthropic.AnthropicBedrock = anthropic.Anthropic(
            max_retries=4, timeout=120.0
        )
        model_id = model
    else:
        client = anthropic.AnthropicBedrock(
            aws_region=os.environ.get("AWS_REGION") or "us-west-2", max_retries=4, timeout=120.0
        )
        model_id = BEDROCK_IDS.get(model, model)

    def agent(message: str) -> dict:
        resp = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": sys_p, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": message}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return {"steps": [], "final_text": text}

    return agent


def warm(spec: str, minutes: float = 15) -> None:
    """One cheap request so a scale-to-zero endpoint is up before the run.

    simulate() stops after 16 failed calls, which a cold vLLM server produces in
    about a minute of startup; a single blocking call absorbs the cold start.
    """
    from urllib import error, request

    from whileai.simulations.generate.agents import parse_backend_spec, resolve_completion_key

    base_url, model = parse_backend_spec(spec)
    key = resolve_completion_key(base_url)
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1}
    ).encode()
    req = request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    deadline = time.time() + minutes * 60
    while True:
        try:
            with request.urlopen(req, timeout=600):
                return
        except error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                return  # not a cold start; let the run report it
        except Exception:
            pass
        if time.time() > deadline:
            return
        time.sleep(15)


def main() -> int:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(MODELS))
    ap.add_argument(
        "--hosted", default="", help="a model served from your account (wai.serve name)"
    )
    ap.add_argument("--agent", default="", help="any SDK agent spec, e.g. openai:gpt-4.1-mini")
    ap.add_argument(
        "--system-prefix",
        default="",
        help="text placed before the schema prompt (e.g. a Nemotron reasoning switch: 'detailed thinking on')",
    )
    ap.add_argument(
        "--name",
        default="",
        help="file stem for raw/<name>.jsonl (default: derived from the model)",
    )
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--split", default="all", choices=["all", "holdout", "train"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--max-tokens", type=int, default=4096, help="agent reply budget (whileai >= 0.47)"
    )
    ap.add_argument("--timeout", type=float, default=300, help="seconds per agent call")
    ap.add_argument(
        "--no-writer",
        action="store_true",
        help="simulator=False: no scene briefs or result shapes; the prompts are pinned",
    )
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke)")
    args = ap.parse_args()

    name = args.model
    spec: object = MODELS[args.model]
    if args.hosted:
        name, spec = f"hosted-{args.hosted}", f"vllm:{args.hosted}@{SERVE_URL}"
    elif args.agent:
        name, spec = args.agent.split("@")[0].replace(":", "-").replace("/", "-"), args.agent
    if args.name:
        name = args.name
    if isinstance(spec, str) and spec.startswith("claude-"):
        spec = claude_agent(spec)

    tasks = load_tasks()
    if args.split != "all":
        tasks = [t for t in tasks if split_of(t["id"]) == args.split]
    if args.limit:
        tasks = tasks[: args.limit]
    out_path = RAW / f"{name}.jsonl"
    have = read_jsonl(out_path)
    done = {r["scenario_id"] for r in have}
    todo = [t for t in tasks if t["id"] not in done]
    print(
        f"{name}: {len(tasks)} tasks x k={args.k}; {len(have)} rows on disk; {len(todo)} tasks to run",
        flush=True,
    )
    if not todo:
        return 0

    if isinstance(spec, str) and spec.startswith("vllm:"):
        warm(spec)
    t0 = time.time()
    sys_p = (
        (args.system_prefix.strip() + "\n\n" + system_prompt())
        if args.system_prefix
        else system_prompt()
    )
    kw = dict(
        system_prompt=sys_p,
        tasks=[{"prompt": t["question"], "scenario_id": t["id"]} for t in todo],
        repeats=args.k,
        # one user turn, one reply: no simulated follow-ups. avg_turns=1 also
        # keeps whileai 0.44's turn sampler off a division by zero that
        # max_turns=1 alone triggers (fixed on main after 0.44).
        max_turns=1,
        avg_turns=1,
        temperature=args.temperature,
        concurrency=args.concurrency,
        budget=len(todo) * args.k,
        # Pinned prompts need no situation writer. With agent="vllm:..." the
        # engine otherwise drafts scene briefs and result shapes on the same
        # server, which doubled a 2,400-row run (whilehq/whileai-sdk#470).
        **({"simulator": False} if args.no_writer else {}),
    )
    try:
        # a reasoning model thinks for 1-3k tokens before the query (whileai >= 0.47)
        data = wai.simulate(spec, agent_max_tokens=args.max_tokens, timeout=args.timeout, **kw)
    except TypeError as exc:
        if "agent_max_tokens" not in str(exc) and "timeout" not in str(exc):
            raise
        print(
            "  this whileai has no agent_max_tokens/timeout knobs (needs >= 0.47): "
            "replies capped at 2048 tokens, 60 s per call; thinking models lose some rows",
            flush=True,
        )
        data = wai.simulate(spec, **kw)
    by_id = {t["id"]: t for t in todo}
    rows = []
    for r in data.trajectories:
        t = by_id.get(r.get("scenario_id"))
        if t is None:
            continue
        r["privileged"] = {"reference": t["sql"]}
        r["category"] = t["archetype"]
        r["difficulty"] = t["difficulty"]
        r["style"] = t.get("style")
        r["split"] = split_of(t["id"])
        r["agent"] = AGENT
        r["model_version"] = name
        rows.append(r)
    # append, never rewrite: two runs on the same file (a top-up next to a long
    # sampling job) lost rows when the second finished with a stale copy
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(
        f"  {len(rows)} rows in {time.time() - t0:.0f}s ({data.stopped_because}); {len(have) + len(rows)} on disk",
        flush=True,
    )
    if data.search.get("agent_errors"):
        print(
            f"  agent errors {data.search['agent_errors']}; first: {data.search.get('first_agent_error')}",
            flush=True,
        )
    # why a run took longer than rows / throughput: re-rolls and lost rows
    for key in ("lost", "rerolled", "cap_lifted", "degraded"):
        val = data.search.get(key) if isinstance(data.search, dict) else None
        if val is None:
            val = getattr(data, key, None)
        if val:
            print(f"  {key}: {val}", flush=True)
    if getattr(data, "warnings", None):
        for w in list(data.warnings)[:3]:
            print(f"  warning: {str(w)[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
