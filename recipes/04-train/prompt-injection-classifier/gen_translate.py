"""Model translations of public attack strings and of benign statements into five languages.

    python gen_translate.py translations.json

The label rides on the source string (a public attack, or a benign statement),
so a translation keeps it by construction as long as the instruction survives;
a checker call drops translations where it did not. German, French, Spanish,
Portuguese, Italian. The multilingual test set is never read here.
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
from data import BODY_SENTENCES, DECOYS, load_payloads

S = Path(__file__).resolve().parent
API = "https://api.anthropic.com/v1/messages"
KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-haiku-4-5"
LANGS = {"de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese", "it": "Italian"}


def ask(system: str, user: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
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


TRANSLATE = "Translate the text into {lang}. Keep the meaning and the tone exactly; keep names, ids and URLs. Output only the translation."
CHECK = (
    "You are checking a sentence found inside a document that an AI assistant will read. Answer "
    "with one word: INSTRUCTION if the sentence tells the assistant (the model, the AI, the "
    "system reading this) to do something, or DATA if it does not address the assistant. Only the word."
)


def main() -> None:
    out = Path(sys.argv[1])
    rng = random.Random(5)
    payloads = load_payloads(S / "ext")
    gandalf = [
        json.loads(line)["text"] for line in (S / "data/gandalf.jsonl").read_text().splitlines()
    ]
    attacks = [("gandalf", p) for p in rng.sample(gandalf, 300)]
    for f in ("injecagent", "bipia_task", "bipia_harm"):
        attacks += [(f, p) for p in payloads[f]]
    benign = list(DECOYS) + list(BODY_SENTENCES)
    oasst = [
        json.loads(line)["text"]
        for line in (S / "data/oasst1_prompts.jsonl").read_text().splitlines()
    ]
    benign += [t for t in oasst if len(t) < 200 and "\n" not in t][:260]
    jobs = []
    for code in LANGS:
        for f, p in rng.sample(attacks, 120):
            jobs.append(("attack", f, code, p))
        for b in rng.sample(benign, 90):
            jobs.append(("benign", "benign", code, b))
    with ThreadPoolExecutor(8) as pool:
        texts = list(pool.map(lambda j: ask(TRANSLATE.format(lang=LANGS[j[2]]), j[3]), jobs))
        checks = list(pool.map(lambda t: ask(CHECK, t, max_tokens=5, temperature=0.0), texts))
    result = {"attacks": [], "benign": [], "dropped": 0}
    for (kind, f, code, src), t, c in zip(jobs, texts, checks):
        if not t:
            result["dropped"] += 1
            continue
        verdict = c.upper().startswith("INSTRUCTION")
        if kind == "attack" and verdict:
            result["attacks"].append({"family": f, "lang": code, "source": src, "text": t})
        elif kind == "benign" and not verdict:
            result["benign"].append({"lang": code, "source": src, "text": t})
        else:
            result["dropped"] += 1
    result["stats"] = {
        "asked": len(jobs),
        "attacks": len(result["attacks"]),
        "benign": len(result["benign"]),
        "dropped": result["dropped"],
    }
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result["stats"]))


if __name__ == "__main__":
    main()
