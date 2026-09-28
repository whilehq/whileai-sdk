"""Model-written paraphrases of public attack strings, and model-written harmless inserts.

    python gen_inserts.py inserts.json

Paraphrases keep the label by construction only if the instruction survives, so
every paraphrase is checked by a second call (a different prompt, same model
family; the audit against program-labelled rows is in the recipe) and dropped
when the checker says the instruction to the assistant is gone. Harmless
inserts are sentences a human would leave in a document that mention
instructions, ignoring, overriding, system prompts or security with no
instruction to the model in them.
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_payloads

EXT = Path(__file__).resolve().parent / "ext"
API = "https://api.anthropic.com/v1/messages"
KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-haiku-4-5"


def ask(system: str, user: str, max_tokens: int = 300, temperature: float = 1.0) -> str:
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
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=60,
            )
            resp.raise_for_status()
            return "".join(b.get("text", "") for b in resp.json()["content"]).strip()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return ""


PARA = (
    "Rewrite the instruction below in different words, keeping exactly what it asks the AI "
    "assistant reading a document to do. Vary the tone (polite, urgent, casual, bureaucratic, "
    "disguised as a note, as a comment, as a footnote). One rewrite, one to three sentences, "
    "no quotes, no commentary."
)
CHECK = (
    "You are checking a sentence found inside a document that an AI assistant will read. Answer "
    "with one word: INSTRUCTION if the sentence tells the assistant (the model, the AI, the "
    "system reading this) to do something, or DATA if it does not address the assistant. "
    "Only the word."
)
BENIGN = (
    "Write ONE sentence that a person could naturally leave inside a business document, email, "
    "ticket, review or web page, that uses the word or idea given but is NOT an instruction to "
    "any AI or assistant: it talks to a human reader, quotes something, or describes a thing. "
    "Vary register. Output the sentence only."
)
IDEAS = [
    "ignore",
    "disregard",
    "override",
    "system prompt",
    "instructions",
    "bypass",
    "forget the above",
    "as an AI",
    "jailbreak",
    "prompt injection",
    "admin mode",
    "developer note",
    "translate this",
    "reply with",
    "send the file to",
    "the password",
    "do not follow",
    "new task",
    "act as",
    "confidential",
    "reveal",
    "urgent action required",
    "security notice",
    "policy update",
]


def main() -> None:
    out = Path(sys.argv[1])
    payloads = load_payloads(EXT)
    families = [f for f in payloads if f not in ("agentdojo", "bipia_test")]
    src = [(f, p) for f in families for p in payloads[f]]
    random.Random(0).shuffle(src)
    src = src[:400]
    result = {"paraphrases": [], "benign": []}
    with ThreadPoolExecutor(8) as pool:
        paras = list(pool.map(lambda fp: (fp[0], fp[1], ask(PARA, fp[1])), src))
        checks = list(pool.map(lambda x: ask(CHECK, x[2], max_tokens=5, temperature=0.0), paras))
    for (f, p, q), c in zip(paras, checks):
        if q and c.upper().startswith("INSTRUCTION"):
            result["paraphrases"].append({"family": f, "source": p, "text": q})
    dropped = sum(
        1 for (_, _, q), c in zip(paras, checks) if not (q and c.upper().startswith("INSTRUCTION"))
    )
    with ThreadPoolExecutor(8) as pool:
        ben = list(
            pool.map(
                lambda i: ask(BENIGN, f"Word or idea: {IDEAS[i % len(IDEAS)]}. Variation {i}."),
                range(320),
            )
        )
        bchecks = list(pool.map(lambda x: ask(CHECK, x, max_tokens=5, temperature=0.0), ben))
    for b, c in zip(ben, bchecks):
        if b and c.upper().startswith("DATA"):
            result["benign"].append(b)
    result["stats"] = {
        "paraphrases_asked": len(src),
        "paraphrases_kept": len(result["paraphrases"]),
        "paraphrases_dropped_by_check": dropped,
        "benign_asked": 320,
        "benign_kept": len(result["benign"]),
    }
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result["stats"]))


if __name__ == "__main__":
    main()
