"""Thirty fresh conversations at the served adapter, over HTTP.

Fresh means never played and never trained on: these are held-out scenarios
the capped eval sample left on the floor. Same domains, same environment --
this is a serving check, not a distribution-shift test, and it says so.

    python fresh_traffic.py --url https://<workspace>--support-lookup-serve-serve.modal.run --arm both
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import requests
from transformers import AutoTokenizer

import whileai as wai
from whileai.config import provenance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support

BASE_MODEL = "Qwen/Qwen3-4B"


def stratified(tasks, n, seed=0):
    import random

    rng = random.Random(seed)
    buckets = {}
    for t in tasks:
        buckets.setdefault((t["target"], t["tier"]), []).append(t)
    for v in buckets.values():
        rng.shuffle(v)
    out, i = [], 0
    keys = sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(out) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out


def main():
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--arm", default="both")
    ap.add_argument("--harness", default="01_skills")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="fresh.json")
    a = ap.parse_args()

    split = json.load(open("split.json"))
    played = {r["task_id"] for r in json.load(open("out/cell_harness.json"))}
    unplayed = [t for t in split["holdout"] if t["task_id"] not in played]
    fresh = stratified(unplayed, a.n, seed=7)
    print(f"{len(unplayed)} held-out tasks the eval never played; taking {len(fresh)}")

    # The frozen holdout was built to discriminate, so its stratified sample
    # took every rare row it could find. What it left behind is whatever the
    # rare tiers were drawn from -- here, ordinary lookup asks only. Say what
    # this check can and cannot see rather than letting a pass@1 imply more.
    mix = Counter((t["target"], t["tier"]) for t in fresh)
    print("composition:", ", ".join(f"{k[0]}/{k[1]} x{v}" for k, v in mix.most_common()))
    if not any(k[0] == "NO_CALL" for k in mix):
        print("no NO_CALL rows left in the unplayed pool: this check measures the behaviour")
        print("only, and cannot see a regression in the capability twin (stays_in_scope).")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = []
    for t in fresh:
        prompt = support.render(t, a.harness, tok)
        r = requests.post(
            a.url.rstrip("/") + "/v1/completions",
            json={
                "model": a.arm,
                "prompt": prompt,
                "max_tokens": 160,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            timeout=600,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["text"]
        g = support.grade(t, text)
        rows.append(
            {
                "task_id": t["task_id"],
                "prompt": t["prompt"],
                "reward": g["reward"],
                "markers": {k: v for k, v in g["markers"].items() if v is not None},
                "target": t["target"],
                "tier": t["tier"],
                "domain": t["domain"],
                "final_text": text[:2000],
                "harness": {"label": a.harness, "model": a.arm},
            }
        )

    n = len(rows)
    hit = sum(r["reward"] for r in rows)
    print(f"\nserved arm {a.arm} under {a.harness}: {hit}/{n} right first action")
    print(wai.pass_at(rows, k=1))

    # push the fresh rows back through select, the way the loop is meant to close
    sel = wai.select(rows)
    print("\n--- select on the fresh rows")
    print(sel)

    Path(a.out).write_text(json.dumps({"rows": rows, "n": n, "hit": hit}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
