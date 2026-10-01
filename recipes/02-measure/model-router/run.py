"""Route each request to a cheap model or a frontier model, and measure what it keeps and saves.

A router is worth its complexity only if it beats sending the same share of
traffic to the frontier model at random. Every router here is scored against
that control on held-out questions, paired, with a 95% interval, and against
the all-frontier baseline it is trying to replace.

Run: python recipes/02-measure/model-router/run.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import whileai as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent

# Defaults. Every one is a flag; the why is next to each.
CHEAP = "google/gemma-3-12b-it"  # open weights, self-hostable on one GPU
STRONG = "anthropic/claude-sonnet-5.5"  # the frontier model the traffic runs on today
# USD per million tokens (input, output), OpenRouter list price on 2026-09-30.
CHEAP_PRICE = (0.05, 0.15)
STRONG_PRICE = (2.0, 10.0)
BASE_URL = "https://openrouter.ai/api/v1"
HOLDOUT_SHARE = 1 / 3  # a third of each subject held out: convention, powers ~0.07 at k=1
SHARES = (0.2, 0.4, 0.6)  # frontier-call budgets reported in the table
MARGIN = 0.02  # "keeps quality" = interval's lower end above -2 points: convention, untested
HASH_DIM = 2**16  # hashed word n-grams; a 4x smaller table costs <0.01 AUC on similar text
L2 = 1.0  # logistic-regression ridge, per example: convention, untested
TEMPERATURE = 0.7  # >0 so the two cheap draws can disagree, which the cascade reads
MAX_TOKENS = 1200
RANDOM_ROLLOUTS = 40  # the random control's share resolves to 1/40 = 2.5 points

PROMPT = (
    "Answer the multiple-choice question. Think it through briefly, then end "
    'with "The answer is (X)" where X is the letter.\n\n'
    "Question: {question}\n\nOptions:\n{options}"
)
LETTERS = "ABCDEFGHIJ"


def load_tasks(limit: int | None) -> list[dict]:
    tasks = [json.loads(line) for line in (HERE / "tasks.jsonl").open(encoding="utf-8")]
    for t in tasks:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(t["options"]))
        t["prompt"] = PROMPT.format(question=t["question"], options=opts)
        # A stable per-question draw, so the split never depends on file order.
        h = int(hashlib.sha256(t["id"].encode()).hexdigest()[:8], 16) / 16**8
        t["split"] = "holdout" if h < HOLDOUT_SHARE else "train"
    if limit:
        tasks = tasks[:: max(1, len(tasks) // limit)][:limit]
    return tasks


# --- sampling ---------------------------------------------------------------


def draws_wanted(task: dict, role: str) -> int:
    # Cheap: two draws everywhere (the cascade compares them). Strong: one on
    # train, two on holdout, so the headline can be re-read on a fresh draw.
    return 2 if role == "cheap" or task["split"] == "holdout" else 1


async def sample_all(tasks, models, cache_path: Path, concurrency: int, base_url: str):
    from openai import APIStatusError, AsyncOpenAI  # only the live path needs it

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENAI_API_KEY (an OpenRouter key works with the default --base-url)")
    client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=120)
    cache = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            r = json.loads(line)
            cache[(r["task_id"], r["model"], r["draw"])] = r
    todo = [
        (t, role, models[role], d)
        for t in tasks
        for role in ("cheap", "strong")
        for d in range(draws_wanted(t, role))
        if (t["id"], models[role], d) not in cache
    ]
    print(f"sampling {len(todo)} calls ({len(cache)} cached)", file=sys.stderr)
    sem = asyncio.Semaphore(concurrency)
    out = cache_path.open("a", encoding="utf-8")

    async def one(t, role, model, d):
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": t["prompt"]}],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    break
                except APIStatusError as e:
                    if e.status_code in (400, 401, 402, 403, 404):  # a retry will not fix these
                        raise RuntimeError(f"{model}: {e.status_code} {e.message}") from None
                    if attempt == 3:
                        print(f"  gave up on {t['id']} {model}: {e}", file=sys.stderr)
                        return
                    await asyncio.sleep(2**attempt)
                except Exception as e:  # timeouts and dropped connections
                    if attempt == 3:
                        print(f"  gave up on {t['id']} {model}: {e}", file=sys.stderr)
                        return
                    await asyncio.sleep(2**attempt)
            u = r.usage
            rec = {
                "task_id": t["id"],
                "model": model,
                "draw": d,
                "text": r.choices[0].message.content or "",
                "in_tokens": u.prompt_tokens if u else 0,
                "out_tokens": u.completion_tokens if u else 0,
                "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            cache[(t["id"], model, d)] = rec
            out.write(json.dumps(rec) + "\n")
            out.flush()

    jobs = [asyncio.create_task(one(*x)) for x in todo]
    try:
        await asyncio.gather(*jobs)
    except RuntimeError as e:  # stop every call still in flight, keep what landed
        for j in jobs:
            j.cancel()
        raise SystemExit(str(e)) from None
    finally:
        out.close()
    return cache


def stand_in_samples(tasks, models, seed: int):
    """Seeded stand-in for --dry-run: per-subject skill, so routing has something to find."""
    rng = random.Random(seed)
    subjects = sorted({t["subject"] for t in tasks})
    skill = {s: {"cheap": rng.uniform(0.3, 0.8), "strong": rng.uniform(0.75, 0.95)} for s in subjects}
    cache = {}
    for t in tasks:
        hard = random.Random(t["id"]).random()  # a per-question difficulty both models share
        for role in ("cheap", "strong"):
            for d in range(draws_wanted(t, role)):
                p = skill[t["subject"]][role] * (1.3 - 0.6 * hard)
                letter = t["answer"] if rng.random() < p else rng.choice(LETTERS[: len(t["options"])])
                cache[(t["id"], models[role], d)] = {
                    "text": f"... The answer is ({letter})",
                    "in_tokens": 350,
                    "out_tokens": 60 if role == "cheap" else 250,
                    "sampled_at": "dry-run",
                }
    return cache


# --- grading and cost ---------------------------------------------------------


def grade(tasks, samples, models, prices):
    """Per task: the reward of each model's draw d (1/0), its cost, and its letter."""
    verify = wai.verify.MultipleChoice()
    for t in tasks:
        t["r"], t["cost"], t["letter"] = {}, {}, {}
    for role in ("cheap", "strong"):
        for d in (0, 1):
            batch = [t for t in tasks if d < draws_wanted(t, role)]
            got = [samples.get((t["id"], models[role], d)) for t in batch]
            # One grading call per (model, draw): a call that never came back scores 0.
            rows = wai.rows(
                [t["prompt"] for t in batch],
                [s["text"] if s else "" for s in got],
                verify,
                references=[t["answer"] for t in batch],
            )
            pin, pout = prices[role]
            for t, s, row in zip(batch, got, rows):
                t["r"][role, d] = float(row["reward"]) if s else 0.0
                t["cost"][role, d] = (s["in_tokens"] * pin + s["out_tokens"] * pout) / 1e6 if s else 0.0
                m = re.findall(r"answer is \(?([A-J])\)?", s["text"]) if s else []
                t["letter"][role, d] = m[-1] if m else None


# --- routers ------------------------------------------------------------------
# Each returns a score per holdout task; higher = send to the frontier model.
# A budget `share` escalates the top `share` of scores.


def features(text: str) -> list[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams = words + [a + " " + b for a, b in zip(words, words[1:])]
    return sorted({int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % HASH_DIM for g in grams})


def fit_text_router(train: list[dict], epochs: int = 30, lr: float = 0.5):
    """Logistic regression on hashed n-grams of the prompt: P(cheap model gets it wrong).

    The label is the cheap model's miss rate over its two draws, a soft label
    that halves the noise of a single draw (RouteLLM, Ong et al. 2024,
    arXiv:2406.18665, trains its routers on the same kind of win label).
    """
    import numpy as np

    w, b = np.zeros(HASH_DIM), 0.0
    xs = [features(t["question"]) for t in train]
    ys = [1 - (t["r"]["cheap", 0] + t["r"]["cheap", 1]) / 2 for t in train]
    rng = random.Random(0)
    order = list(range(len(train)))
    for epoch in range(epochs):
        rng.shuffle(order)
        step = lr / (1 + epoch)
        for i in order:
            z = b + w[xs[i]].sum()
            g = 1 / (1 + math.exp(-z)) - ys[i]
            w[xs[i]] -= step * (g + L2 * w[xs[i]] / len(train))
            b -= step * g
    return lambda t: float(b + w[features(t["question"])].sum())


def subject_router(train: list[dict]):
    """The free rule: escalate the subjects where the frontier model gained most on train."""
    gain: dict[str, list[float]] = {}
    for t in train:
        gain.setdefault(t["subject"], []).append(t["r"]["strong", 0] - t["r"]["cheap", 0])
    mean = {s: sum(v) / len(v) for s, v in gain.items()}
    # Tie-break inside a subject by question length, a stand-in for "harder".
    return lambda t: mean.get(t["subject"], 0.0) + 1e-6 * len(t["question"])


def escalate(holdout, score, share):
    ranked = sorted(holdout, key=score, reverse=True)
    n = round(share * len(holdout))
    return {t["id"] for t in ranked[:n]}


# --- measuring ---------------------------------------------------------------


def arm_rows(holdout, reward_of):
    return [
        {"task_id": t["id"], "prompt": t["prompt"], "final_text": "routed", "reward": reward_of(t), "markers": {}}
        for t in holdout
    ]


def routed(holdout, sent, d, cascade=False):
    """Rows and total cost when the tasks in `sent` go to the frontier model, on draw d."""
    rows = arm_rows(holdout, lambda t: t["r"]["strong", d] if t["id"] in sent else t["r"]["cheap", d])
    cost = 0.0
    for t in holdout:
        if cascade:  # the cascade always pays for both cheap draws first
            cost += t["cost"]["cheap", 0] + t["cost"]["cheap", 1]
            cost += t["cost"]["strong", d] if t["id"] in sent else 0.0
        else:
            cost += t["cost"]["strong", d] if t["id"] in sent else t["cost"]["cheap", d]
    return rows, cost


def at_random(holdout, share, d):
    """Random routing at `share`, in expectation: RANDOM_ROLLOUTS rows per task, that share of them frontier.

    Graded rows carry 0/1 rewards, so the expectation is written as rollouts:
    the per-task mean is exact to 1/RANDOM_ROLLOUTS, and the control carries no
    luck of its own draw.
    """
    n_strong = round(share * RANDOM_ROLLOUTS)
    return [
        {"task_id": t["id"], "prompt": t["prompt"], "final_text": "routed", "markers": {},
         "reward": t["r"]["strong", d] if j < n_strong else t["r"]["cheap", d]}
        for t in holdout
        for j in range(RANDOM_ROLLOUTS)
    ]


def mean(rows):
    return sum(r["reward"] for r in rows) / len(rows)


def delta(before, after):
    m = wai.compare(before, after, target="pass_at_1")["metrics"]["pass_at_1"]
    return {"delta": m["delta"], "ci95": list(m["ci95"]), "verdict": m["verdict"], "n": m["n_paired"]}


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def measure(holdout, routers, cascade_sent, d):
    strong_rows, strong_cost = routed(holdout, {t["id"] for t in holdout}, d)
    cheap_rows, cheap_cost = routed(holdout, set(), d)
    acc_s, acc_c = mean(strong_rows), mean(cheap_rows)
    out = {
        "all_frontier": {"accuracy": acc_s, "cost_usd": strong_cost},
        "all_cheap": {"accuracy": acc_c, "cost_usd": cheap_cost, "vs_frontier": delta(strong_rows, cheap_rows)},
        "arms": [],
    }

    def arm(name, sent, share, cascade=False):
        rows, cost = routed(holdout, sent, d, cascade)
        acc = mean(rows)
        gap = acc_s - acc_c
        out["arms"].append(
            {
                "router": name,
                "frontier_share": share,
                "accuracy": acc,
                "cost_vs_frontier": cost / strong_cost,
                "gap_recovered": (acc - acc_c) / gap if gap else float("nan"),
                "vs_random": delta(at_random(holdout, share, d), rows),
                "vs_frontier": delta(strong_rows, rows),
            }
        )

    for share in SHARES:
        for name, score in routers.items():
            arm(name, escalate(holdout, score, share), share)
    c_share = len(cascade_sent) / len(holdout)
    arm("cascade", cascade_sent, c_share, cascade=True)
    arm("text", escalate(holdout, routers["text"], c_share), c_share)
    oracle = {t["id"] for t in holdout if t["r"]["strong", d] > t["r"]["cheap", d]}
    arm("oracle", oracle, len(oracle) / len(holdout))
    return out


def smallest_share_that_keeps(holdout, score, d, margin):
    """The lowest frontier share whose interval against all-frontier stays above -margin."""
    strong_rows, _ = routed(holdout, {t["id"] for t in holdout}, d)
    for pct in range(5, 101, 5):
        rows, _ = routed(holdout, escalate(holdout, score, pct / 100), d)
        if delta(strong_rows, rows)["ci95"][0] >= -margin:
            return pct / 100
    return 1.0


def fmt(x):
    return f"{x['delta']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def report(res, title):
    print(f"\n{title}")
    print(f"  all frontier  {res['all_frontier']['accuracy']:.3f}   cost 1.00x")
    ac = res["all_cheap"]
    print(f"  all cheap     {ac['accuracy']:.3f}   cost {ac['cost_usd'] / res['all_frontier']['cost_usd']:.2f}x   "
          f"vs frontier {fmt(ac['vs_frontier'])}")
    print(f"  {'router':<8} {'share':>5} {'acc':>6} {'cost':>6} {'gap':>5}  {'vs random at same share':<30} vs all frontier")
    for a in res["arms"]:
        print(f"  {a['router']:<8} {a['frontier_share']:>5.2f} {a['accuracy']:>6.3f} {a['cost_vs_frontier']:>5.2f}x "
              f"{a['gap_recovered']:>5.2f}  {fmt(a['vs_random']):<30} {fmt(a['vs_frontier'])}")


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cheap", default=CHEAP, help="the model most traffic should land on")
    p.add_argument("--strong", default=STRONG, help="the frontier model it replaces")
    p.add_argument("--cheap-price", default=",".join(map(str, CHEAP_PRICE)), help="USD per M tokens: in,out")
    p.add_argument("--strong-price", default=",".join(map(str, STRONG_PRICE)), help="USD per M tokens: in,out")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", BASE_URL))
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="fewer questions, for a smoke run")
    p.add_argument("--margin", type=float, default=MARGIN, help="how far below all-frontier still counts as kept")
    p.add_argument("--out", default=str(HERE / "out"))
    p.add_argument("--fresh", action="store_true", help="ignore cached samples and call again")
    p.add_argument("--dry-run", action="store_true", help="seeded stand-in models: no key, no calls")
    args = p.parse_args(argv)

    models = {"cheap": args.cheap, "strong": args.strong}
    prices = {
        "cheap": tuple(map(float, args.cheap_price.split(","))),
        "strong": tuple(map(float, args.strong_price.split(","))),
    }
    tasks = load_tasks(args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "samples.jsonl"
    if args.fresh and cache_path.exists():
        cache_path.unlink()
    if args.dry_run:
        samples = stand_in_samples(tasks, models, seed=0)
    else:
        samples = asyncio.run(sample_all(tasks, models, cache_path, args.concurrency, args.base_url))
    grade(tasks, samples, models, prices)

    train = [t for t in tasks if t["split"] == "train"]
    holdout = [t for t in tasks if t["split"] == "holdout"]
    print(f"{len(train)} train, {len(holdout)} held out; {args.cheap} vs {args.strong}", file=sys.stderr)

    routers = {"subject": subject_router(train), "text": fit_text_router(train)}
    # The cascade: answer with the cheap model twice; escalate when the two disagree.
    cascade_sent = {
        t["id"] for t in holdout
        if t["letter"]["cheap", 0] is None or t["letter"]["cheap", 0] != t["letter"]["cheap", 1]
    }

    results = {
        "models": models,
        "prices_usd_per_mtok": prices,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "dry_run": args.dry_run,
        "sampled": sorted({s["sampled_at"][:10] for s in samples.values()}),
        "text_router_auc": auc(
            [routers["text"](t) for t in holdout], [t["r"]["cheap", 0] == 0 for t in holdout]
        ),
        "draw_1": measure(holdout, routers, cascade_sent, d=0),
        "draw_2": measure(holdout, routers, cascade_sent, d=1),
        "kept_share": {
            name: smallest_share_that_keeps(holdout, score, 0, args.margin) for name, score in routers.items()
        },
        "margin": args.margin,
    }
    report(results["draw_1"], "held out, draw 1 (the headline)")
    report(results["draw_2"], "held out, draw 2 (same routers, fresh answers from both models)")
    print(f"\ntext router AUC for 'cheap model misses': {results['text_router_auc']:.3f}")
    for name, share in results["kept_share"].items():
        print(f"{name}: smallest frontier share within {args.margin:.0%} of all-frontier = {share:.0%}")
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'results.json'}")
    if not args.dry_run:
        print("\nNext: swap tasks.jsonl for a sample of your own traffic and a grader for it; the routers do not change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
