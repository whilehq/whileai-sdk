"""The probe's broken set: edit an AIME 2025 / HMMT Nov 2025 problem so it has no single answer, then check the edit.

python probe_data.py      # writes data/probe_broken.json (OPENROUTER_API_KEY)
"""

import concurrent.futures as cf
import json
import os
import time
import urllib.request

KEY = os.environ.get("OPENROUTER_API_KEY", "")

BREAK = """Here is a competition math problem with a unique numeric answer ({gold}).

{problem}

Rewrite it with ONE small edit so that it has NO valid unique answer: either add a condition that contradicts the others,
or remove a condition so the answer is not determined. Keep the style and length; the flaw should not be announced and
should take real work to notice. Reply with JSON only: {{"problem": "...", "flaw": "one sentence"}}"""

CHECK = """Is this math problem well-posed with a single numeric answer? Work it out carefully.

{problem}

Reply at the end with JSON only: {{"well_posed": true|false, "why": "one sentence"}}"""


def last_json(txt):
    dec, best = json.JSONDecoder(), None
    for i, ch in enumerate(txt):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(txt[i:])
                if isinstance(obj, dict):
                    best = obj
            except ValueError:
                pass
    return best


def chat(model, prompt, reasoning=True, max_tokens=20000):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "reasoning": {"effort": "medium"} if reasoning else {"enabled": False},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    for a in range(4):
        try:
            txt = (
                json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"][
                    "content"
                ]
                or ""
            )
            return last_json(txt)
        except Exception:
            time.sleep(3 * (a + 1))
    return None


def rows(ds):
    u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split=train&offset=0&length=100"
    return [dict(r["row"], ds=ds) for r in json.load(urllib.request.urlopen(u))["rows"]]


def one(p):
    b = chat("deepseek/deepseek-v4-pro", BREAK.format(problem=p["problem"], gold=p["answer"]))
    if not b or "problem" not in b:
        return None
    c = chat("z-ai/glm-5.3", CHECK.format(problem=b["problem"]))
    return {
        "src": p["ds"],
        "src_idx": p["problem_idx"],
        "orig": p["problem"],
        "problem": b["problem"],
        "flaw": b.get("flaw"),
        "check": c,
    }


if __name__ == "__main__":
    # the committed file came from two passes of this script (the first used a
    # JSON reader that choked on LaTeX braces); a fresh run makes a new set
    src = rows("MathArena/aime_2025") + rows("MathArena/hmmt_nov_2025")
    with cf.ThreadPoolExecutor(20) as ex:
        out = [x for x in ex.map(one, src) if x]
    ok = [x for x in out if x["check"] and x["check"].get("well_posed") is False]
    print(len(src), len(out), "verified broken:", len(ok))
    json.dump(
        ok,
        open(os.path.join(os.path.dirname(__file__), "data", "probe_broken.json"), "w"),
        indent=1,
    )
