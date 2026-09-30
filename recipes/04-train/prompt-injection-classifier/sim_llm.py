"""LLM-written agent traffic with wai.simulate: haiku writes the situations, plays the user, and
answers every tool call with a realistic document through ``execute=``.

    python sim_llm.py sim_llm_<domain>.jsonl <domain> 64

Nothing templated: the situation writer is a model (``simulator=``), the tool
results are written by a model for the call that was made (an email body, a
page, a record, a review thread, a ticket), and the agent's replies are the
agent's. Labels are added later by program.
"""

import json
import os
import sys
import time

import requests
from sim_domains import DOMAINS

import whileai as wai
from whileai.config import provenance
from whileai.simulations.generate.agents import current_rollout

OUT, DOMAIN, BUDGET = sys.argv[1], sys.argv[2], int(sys.argv[3])
SYSTEM, TOOLS = DOMAINS[DOMAIN]
API = "https://api.anthropic.com/v1/messages"
KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-haiku-4-5"

WRITER = (
    "You write the raw content a software tool returns to an AI assistant. Given the tool, its "
    "arguments and the business, write ONLY the content: a realistic, specific document of the "
    "kind that tool returns (an email with headers and a body, a web page's text, a JSON record, a "
    "review thread, a ticket with comments, a lab report, a contract clause, a CI log). Invent "
    "plausible names, dates, amounts and ids. 120 to 350 words. No preamble, no commentary, no "
    "mention of being an AI. Vary the register: some formal, some sloppy, some with typos."
)


def execute(tool: str, arguments: dict) -> str:
    seed = getattr(current_rollout, "seed", None)
    prompt = (
        f"Business: {DOMAIN}. Tool: {tool}. Arguments: {json.dumps(arguments)[:400]}. Seed {seed}."
    )
    for attempt in range(3):
        try:
            resp = requests.post(
                API,
                headers={
                    "x-api-key": KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 600,
                    "temperature": 1.0,
                    "system": WRITER,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            resp.raise_for_status()
            return "".join(b.get("text", "") for b in resp.json()["content"])
        except Exception as e:
            err = str(e)[:120]
            time.sleep(2 * (attempt + 1))
    return json.dumps({"status": "error", "message": err})


print(provenance(), file=sys.stderr)
wai.configure(agent="anthropic:claude-haiku-4-5", judge="anthropic:claude-haiku-4-5")
t0 = time.time()
data = wai.simulate(
    tools=TOOLS,
    system_prompt=SYSTEM,
    execute=execute,
    simulator="anthropic:claude-haiku-4-5",
    user_model="anthropic:claude-haiku-4-5",
    mode="explore",
    budget=BUDGET,
    hard_share=0.6,
    seed=200 + list(DOMAINS).index(DOMAIN),
    dimensions={"stance": ["ordinary", "boundary", "ambiguous", "adversarial"]},
    max_turns=6,
    temperature=0.9,
)
rows = list(data.rows())
with open(OUT, "w") as fh:
    for r in rows:
        r["domain"] = DOMAIN
        fh.write(json.dumps(r, default=str) + "\n")
rep = data.report()
print(
    f"{DOMAIN}: {len(rows)} rows in {time.time() - t0:.0f}s; delivered {rep.get('delivered')}; tools {rep.get('tools')}",
    file=sys.stderr,
)
for w in getattr(data, "warnings", []) or []:
    print("WARN", str(w)[:220], file=sys.stderr)
