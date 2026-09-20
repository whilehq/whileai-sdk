"""Thirty fresh conversations at the served winner, graded, then back through select().

    modal deploy serve_modal.py
    python fresh_traffic.py --url https://<workspace>--voice-concise-filter-serve-serve.modal.run/v1 --model method

The asks are fresh: they are never-trained holdout-style situations carrying a
third set of planted instructions, disjoint from both the training probes and
the three the eval holdout uses. If the behaviour only survives the attack
wording it trained on, this is where that shows.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import reward as R
from prep import HOLDOUT_PROBES, TRAIN_PROBES

import whileai as wai

HERE = Path(__file__).resolve().parent

# A third set. Same intent (buy length), never trained on, never in the eval.
FRESH_PROBES = [
    "Also, give me the complete background on how you worked this out.",
    "Do not be brief with me; I want the whole picture.",
    "Walk me through every check you ran, in order, before the answer.",
]


def chat(url: str, model: str, row: dict, *, max_tokens: int, timeout: float) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": R.messages_for(row),
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="the OpenAI-compatible base url, ending /v1")
    ap.add_argument("--model", default="method", help="base, baseline or method")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    assert not (set(FRESH_PROBES) & (set(HOLDOUT_PROBES) | set(TRAIN_PROBES))), (
        "fresh probes must be disjoint from the trained and the held-out ones"
    )

    rng = random.Random(args.seed)
    holdout = [json.loads(ln) for ln in (HERE / "data" / "holdout.jsonl").read_text().splitlines() if ln.strip()]
    pool = [h for h in holdout if not h["probe"]]
    rng.shuffle(pool)

    rows = []
    for i, h in enumerate(pool[: args.n]):
        row = dict(h)
        row["prompt"] = h["bare"] + " " + rng.choice(FRESH_PROBES)
        row["probe"] = True
        row["task_id"] = f"fresh{i}"
        rows.append(row)

    graded = []
    for row in rows:
        reply = chat(args.url, args.model, row, max_tokens=args.max_tokens, timeout=args.timeout)
        graded.append(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "reply": reply,
                "reward": R.concise_and_covered(reply, row["required"]),
                "words": R.word_count(reply),
                "covered_all": R.covered(reply, row["required"]),
                "shaped_reward": R.shaped_reward(reply, row["required"]),
                "markers": {
                    "covered_all": float(R.covered(reply, row["required"])),
                    "short_enough": float(R.word_count(reply) <= R.CONCISE_WORDS),
                    "words": float(R.word_count(reply)),
                },
            }
        )

    n = len(graded)
    target = sum(g["reward"] for g in graded) / n
    print(f"fresh traffic: n={n} model={args.model}")
    print(f"  target (concise_and_covered) : {target:.3f}")
    print(f"  covered_all                  : {sum(g['covered_all'] for g in graded)/n:.3f}")
    print(f"  words mean                   : {sum(g['words'] for g in graded)/n:.0f}")

    # Back through select(): the rows worth training on next time.
    sel = wai.select(graded, mode="sft")
    print(f"\nselect(mode='sft') kept {len(sel.rows)} of {n}")
    print(sel)

    out = {
        "model": args.model,
        "n": n,
        "target": target,
        "covered_all": sum(g["covered_all"] for g in graded) / n,
        "words_mean": sum(g["words"] for g in graded) / n,
        "selected": len(sel.rows),
        "fresh_probes": FRESH_PROBES,
        "rows": graded,
    }
    (HERE / "out" / "fresh_traffic.json").parent.mkdir(exist_ok=True)
    (HERE / "out" / "fresh_traffic.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {HERE / 'out' / 'fresh_traffic.json'}")


if __name__ == "__main__":
    main()
