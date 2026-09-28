"""Round-7 training rows: model-written carriers only, matched twins, public and paraphrased payloads.

    python data_llm.py --sim sim_llm_clinic.jsonl --sim sim_llm_bank.jsonl ... --inserts inserts.json --out out --tag v7

No carrier template is used here. Carriers are what ``wai.simulate`` produced
with a model as situation writer, user and world (``execute=``): tool results
(emails, records, pages, threads, reports), the agent's own replies (drafts,
summaries) and the user's asks. Each carrier over 60 words yields a matched
pair: the payload (a public attack string, or a model paraphrase of one that a
checker confirmed still addresses the assistant) planted at a random line
boundary, and the same carrier with a model-written harmless insert at the
same boundary. Chunks are cut to at most 512 tokens by lines, the planted
line always inside. Direct side: Gandalf and the paraphrases as user turns
against oasst1 and the simulated asks. The frozen tests are untouched; every
payload sharing an 8-gram with any test is dropped and counted.

Why this shape (Lambert 2025, chapter Synthetic Data and Distillation): a
stronger model writes the carriers, prompts are diverse (six businesses, four
stances), outputs are deduplicated, the paraphrases are filtered by a checker,
and real data (public attack strings, Gandalf, oasst1) is mixed in.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from data import (
    HELDOUT_FAMILIES,
    drop_overlap,
    load_payloads,
    load_public_rows,
    read_jsonl,
    sha256_rows,
    write_jsonl,
)

HERE = Path(__file__).resolve().parent


WORLD_MARK = "\u2063"  # sim_seeds.py appends this to a tool result the world planted into


def harvest_world_labelled(paths: list[Path]) -> list[dict]:
    """Rows whose label the world set: seeded runs where ``execute`` planted (or did not) and marked."""
    rows: list[dict] = []
    seen: set[str] = set()
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for m in r.get("messages", []):
                c = m.get("content")
                if m["role"] != "tool" or not isinstance(c, str):
                    continue
                try:
                    obj = json.loads(c)
                    if isinstance(obj, dict) and isinstance(obj.get("result"), str):
                        c = obj["result"]
                    elif isinstance(obj, dict):
                        continue
                except json.JSONDecodeError:
                    pass
                planted = c.endswith(WORLD_MARK)
                c = c.rstrip(WORLD_MARK)
                if len(c.split()) < 40 or c in seen:
                    continue
                seen.add(c)
                segs = [x for x in c.split("\n") if x.strip()]
                rows.append(
                    {
                        "text": "\n".join(cut(segs, 0)),
                        "label": int(planted),
                        "slice": "train",
                        "carrier": "tool:" + str(m.get("name") or ""),
                        "domain": r.get("domain", ""),
                        "pair": None,
                        "clean": None,
                        "family": "world" if planted else "benign",
                        "source": "world:planted" if planted else "world:clean",
                    }
                )
    return rows


def harvest(paths: list[Path]) -> tuple[list[tuple[str, str, str]], list[str]]:
    """(carriers as (text, kind, domain), user asks) from simulate rows."""
    carriers: list[tuple[str, str, str]] = []
    asks: set[str] = set()
    seen: set[str] = set()
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            domain = r.get("domain", "")
            if r.get("prompt"):
                asks.add(r["prompt"])
            for m in r.get("messages", []):
                c = m.get("content")
                if not isinstance(c, str):
                    continue
                if m["role"] == "tool":
                    try:  # the execute= hook's result rides inside {"status", "result"}
                        obj = json.loads(c)
                        if isinstance(obj, dict) and isinstance(obj.get("result"), str):
                            c = obj["result"]
                        elif isinstance(obj, dict) and obj.get("status") in ("error", "timeout"):
                            continue
                    except json.JSONDecodeError:
                        pass
                    kind = "tool:" + str(m.get("name") or "")
                elif m["role"] == "assistant":
                    kind = "reply"
                elif m["role"] == "user":
                    asks.add(c)
                    continue
                else:
                    continue
                if len(c.split()) < 60 or c in seen:
                    continue
                seen.add(c)
                carriers.append((c, kind, domain))
    return carriers, sorted(asks)


def cut(segments: list[str], at: int, max_words: int = 330) -> list[str]:
    """Trim a segment list to about ``max_words`` words, keeping index ``at`` inside."""
    total = sum(len(s.split()) for s in segments)
    if total <= max_words:
        return segments
    lo, hi = at, at + 1
    words = len(segments[at].split())
    while words < max_words and (lo > 0 or hi < len(segments)):
        if lo > 0:
            lo -= 1
            words += len(segments[lo].split())
        if hi < len(segments) and words < max_words:
            words += len(segments[hi].split())
            hi += 1
    return segments[lo:hi]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="append", required=True)
    ap.add_argument("--inserts", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--holdout-domain",
        default=None,
        help="a business whose carriers never train; they become test_llm.jsonl with held-out families",
    )
    ap.add_argument(
        "--merge", action="append", default=[], help="training files to append (round 8: v5 + v7)"
    )
    ap.add_argument(
        "--seeded",
        action="append",
        default=[],
        help="sim_seeds.py runs: tool results labelled by the world",
    )
    ap.add_argument(
        "--translations",
        default=None,
        help="gen_translate.py output: attacks and benign in five languages",
    )
    a = ap.parse_args()
    out = Path(a.out)
    rng = random.Random(a.seed)
    tests = [
        r["text"]
        for t in ("test", "test_hard", "test_paste")
        for r in read_jsonl(HERE / f"{t}.jsonl")
    ]
    test_set = set(tests)

    payloads = load_payloads(Path(a.ext))
    ins = json.loads(Path(a.inserts).read_text())
    fam_pay = {f: v for f, v in payloads.items() if f not in HELDOUT_FAMILIES}
    fam_pay["paraphrase"] = [
        p["text"] for p in ins["paraphrases"] if p["family"] not in HELDOUT_FAMILIES
    ]
    public = load_public_rows(Path(a.ext), Path(a.data))
    fam_pay["gandalf"] = [r["text"] for r in public["gandalf"]]
    fam_pay, dropped = drop_overlap(fam_pay, tests)
    fams = [f for f in fam_pay if fam_pay[f]]

    carriers, asks = harvest([Path(p) for p in a.sim])
    benign_inserts = list(ins["benign"])
    if a.holdout_domain:
        held = [c for c in carriers if c[2] == a.holdout_domain]
        carriers = [c for c in carriers if c[2] != a.holdout_domain]
        trng = random.Random(a.seed + 11)
        hold_pay = {f: payloads[f] for f in ("agentdojo", "bipia_test")}
        hold_ins = benign_inserts[
            -60:
        ]  # the last sixty inserts are the test's; training keeps the rest
        benign_inserts = benign_inserts[:-60]
        test_rows: list[dict] = []
        for k, (text, kind, domain) in enumerate(held):
            segs = [x for x in text.split("\n") if x.strip()]
            if len(segs) < 2:
                continue
            f = trng.choice(list(hold_pay))
            p, d = trng.choice(hold_pay[f]), trng.choice(hold_ins)
            at = trng.randint(1, len(segs))
            base = {
                "slice": "llm_heldout_domain",
                "carrier": kind,
                "domain": domain,
                "pair": f"llm-{k}",
                "clean": None,
            }
            test_rows.append(
                {
                    **base,
                    "text": "\n".join(cut([*segs[:at], p, *segs[at:]], at)),
                    "label": 1,
                    "family": f,
                    "source": "planted:llm",
                }
            )
            test_rows.append(
                {
                    **base,
                    "text": "\n".join(cut([*segs[:at], d, *segs[at:]], at)),
                    "label": 0,
                    "family": "benign",
                    "source": "twin:llm",
                }
            )
        write_jsonl(out / "test_llm.jsonl", test_rows)
        (out / "test_llm.sha256").write_text(sha256_rows(test_rows) + "\n")
        print(
            f"test_llm {len(test_rows)} rows from {a.holdout_domain} sha256 {sha256_rows(test_rows)}"
        )
        tests += [r["text"] for r in test_rows]
        test_set = set(tests)
    rows: list[dict] = []
    k = 0
    for text, kind, domain in carriers:
        segs = [s for s in text.split("\n") if s.strip()]
        if len(segs) < 2:
            segs = [s.strip() for s in text.replace(". ", ".\n").split("\n") if s.strip()]
        if len(segs) < 2:
            continue
        for _ in range(2):  # two pairs per carrier, different payloads and positions
            f = rng.choice(fams)
            p = rng.choice(fam_pay[f])
            d = rng.choice(benign_inserts)
            at = rng.randint(1, len(segs))
            pair = f"{a.tag}-{k}"
            k += 1
            base = {
                "slice": "train",
                "carrier": kind,
                "domain": domain,
                "pair": pair,
                "clean": None,
            }
            pos = cut([*segs[:at], p, *segs[at:]], at)
            neg = cut([*segs[:at], d, *segs[at:]], at)
            rows.append(
                {**base, "text": "\n".join(pos), "label": 1, "family": f, "source": "planted:llm"}
            )
            rows.append(
                {
                    **base,
                    "text": "\n".join(neg),
                    "label": 0,
                    "family": "benign",
                    "source": "twin:llm",
                }
            )
        # and the carrier as it came, benign
        if rng.random() < 0.5:
            rows.append(
                {
                    "text": "\n".join(cut(segs, 0)),
                    "label": 0,
                    "slice": "train",
                    "carrier": kind,
                    "domain": domain,
                    "pair": None,
                    "clean": None,
                    "family": "benign",
                    "source": "benign:llm_clean",
                }
            )
    # direct side
    for f in ("gandalf", "paraphrase"):
        for p in fam_pay.get(f, []):
            rows.append(
                {
                    "text": p,
                    "label": 1,
                    "slice": "train",
                    "carrier": "user_turn",
                    "domain": "",
                    "pair": None,
                    "clean": None,
                    "family": f,
                    "source": "public:payload",
                }
            )
    for f in fams:
        if f in ("gandalf", "paraphrase"):
            continue
        for p in fam_pay[f]:
            rows.append(
                {
                    "text": p,
                    "label": 1,
                    "slice": "train",
                    "carrier": "user_turn",
                    "domain": "",
                    "pair": None,
                    "clean": None,
                    "family": f,
                    "source": "public:payload",
                }
            )
    for r in public["injecagent_user"] + public["oasst1"]:
        rows.append(
            {
                **r,
                "slice": "train",
                "carrier": "user_turn",
                "domain": "",
                "pair": None,
                "clean": None,
                "family": "benign",
            }
        )
    for t in asks + benign_inserts:
        rows.append(
            {
                "text": t,
                "label": 0,
                "slice": "train",
                "carrier": "user_turn",
                "domain": "",
                "pair": None,
                "clean": None,
                "family": "benign",
                "source": "benign:llm",
            }
        )
    world = harvest_world_labelled([Path(p) for p in a.seeded])
    rows += world
    n_tr = 0
    if a.translations:
        tr = json.loads(Path(a.translations).read_text())
        tr_attacks = [t for t in tr["attacks"] if t["text"] not in test_set]
        tr_benign = [t for t in tr["benign"] if t["text"] not in test_set]
        for t in tr_attacks:  # direct, as a user turn
            rows.append(
                {
                    "text": t["text"],
                    "label": 1,
                    "slice": "train",
                    "carrier": "user_turn",
                    "domain": t["lang"],
                    "pair": None,
                    "clean": None,
                    "family": f"{t['family']}:{t['lang']}",
                    "source": "translated:attack",
                }
            )
        for t in tr_benign:
            rows.append(
                {
                    "text": t["text"],
                    "label": 0,
                    "slice": "train",
                    "carrier": "user_turn",
                    "domain": t["lang"],
                    "pair": None,
                    "clean": None,
                    "family": "benign",
                    "source": "translated:benign",
                }
            )
        # and planted into model-written carriers as matched twins, one pair per carrier
        for text, kind, domain in carriers[: min(len(carriers), 2 * len(tr_attacks))]:
            segs = [x for x in text.split("\n") if x.strip()]
            if len(segs) < 2:
                continue
            at_ = rng.randint(1, len(segs))
            p_ = rng.choice(tr_attacks)
            d_ = rng.choice(tr_benign)
            pair = f"{a.tag}-tr-{n_tr}"
            n_tr += 1
            base = {
                "slice": "train",
                "carrier": kind,
                "domain": domain,
                "pair": pair,
                "clean": None,
            }
            rows.append(
                {
                    **base,
                    "text": "\n".join(cut([*segs[:at_], p_["text"], *segs[at_:]], at_)),
                    "label": 1,
                    "family": f"{p_['family']}:{p_['lang']}",
                    "source": "planted:translated",
                }
            )
            rows.append(
                {
                    **base,
                    "text": "\n".join(cut([*segs[:at_], d_["text"], *segs[at_:]], at_)),
                    "label": 0,
                    "family": "benign",
                    "source": "twin:translated",
                }
            )
    for m in a.merge:
        rows += [r for r in read_jsonl(Path(m)) if r.get("domain") != a.holdout_domain]
    rows = [r for r in rows if r["text"] not in test_set]
    keys = sorted({r.get("pair") or r["text"] for r in rows})
    val_keys = set(random.Random(a.seed + 1).sample(keys, len(keys) // 10))
    val = [r for r in rows if (r.get("pair") or r["text"]) in val_keys]
    train = [r for r in rows if (r.get("pair") or r["text"]) not in val_keys]
    rng.shuffle(train)
    write_jsonl(out / f"train_{a.tag}.jsonl", train)
    write_jsonl(out / f"val_{a.tag}.jsonl", val)
    stats = {
        "carriers": len(carriers),
        "carrier_kinds": dict(Counter(k for _, k, _ in carriers)),
        "domains": dict(Counter(d for _, _, d in carriers)),
        "asks": len(asks),
        "payloads_dropped_8gram": dropped,
        "payloads_kept": {f: len(v) for f, v in fam_pay.items()},
        "benign_inserts": len(benign_inserts),
        "world_labelled": {"rows": len(world), "planted": sum(r["label"] for r in world)},
        "translated_pairs": n_tr,
        "train_rows": len(train),
        "val_rows": len(val),
        "label_1": sum(r["label"] for r in train),
        "pairs": sum(1 for r in train if r.get("pair") and r["label"] == 1),
        "train_sha256": sha256_rows(train),
    }
    (out / f"build_stats_{a.tag}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
