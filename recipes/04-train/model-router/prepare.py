"""Turn LLMRouterBench into one table: each question, how every model scored on it, what it cost.

LLMRouterBench (Li et al. 2026, arXiv:2601.07206) ran 33 models over 21
datasets and kept every answer, its grade and its cost. A router is trained
on exactly that table, so this script downloads the release (1.3 GB, once),
keeps the twelve flagship models and the twelve datasets all of them
answered, caps each dataset so no single one dominates, and embeds every
question on your CPU.

Run: python prepare.py   (needs `huggingface_hub`, `numpy`, `fastembed`; ~10 minutes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The flagship pool: every model LLMRouterBench ran on all twelve datasets below.
MODELS = (
    "gpt-5",
    "gpt-5-chat",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "deepseek-v3-0324",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-235b-a22b-2507",
    "kimi-k2-0905",
    "glm-4.6",
    "intern-s1",
)
# OpenRouter's own auto router, run by the benchmark on nine of the datasets:
# kept as a comparison, never as a model a router here can pick.
OPENROUTER = "openrouter"
DATASETS = (
    ("aime", "hybrid"),
    ("arc-agi", "v1"),
    ("arenahard_coding", "test"),
    ("arenahard_creative_writing", "test"),
    ("arenahard_math", "test"),
    ("gpqa", "test"),
    ("hle", "subset_500"),
    ("livecodebench", "test"),
    ("livemathbench", "test"),
    ("mmlupro", "test_3000"),
    ("simpleqa", "test"),
    ("swe-bench", "verified"),
)
PER_DATASET = 1000  # cap: SimpleQA alone has 4,326; convention, untested
EMBEDDER = (
    "BAAI/bge-small-en-v1.5"  # 33M params, CPU; the benchmark finds the encoder matters little
)
MAX_CHARS = 2000  # the encoder reads 512 tokens; longer questions are cut before embedding
SEED = 0


def download(cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    tar = hf_hub_download(
        "NPULH/LLMRouterBench", "bench-release.tar.gz", repo_type="dataset", local_dir=cache
    )
    root = cache / "bench-release"
    if not root.exists():
        print("extracting (7.8 GB on disk)...")
        with tarfile.open(tar) as t:
            t.extractall(cache, filter="data")
    return root


def load(root: Path) -> list[dict]:
    keep = set(MODELS) | {OPENROUTER}
    cells: dict[tuple, dict] = {}
    for ds, split in DATASETS:
        for dirpath, _, files in os.walk(root / ds):
            for f in files:
                if not f.endswith(".json"):
                    continue
                try:
                    d = json.loads((Path(dirpath) / f).read_text(encoding="utf-8"))
                except OSError:  # a path past Windows' 260 characters, for a model not in the pool
                    continue
                if d.get("split") != split or d.get("model_name") not in keep:
                    continue
                for r in d["records"]:
                    q = r.get("origin_query") or r.get("prompt") or ""
                    q = q if isinstance(q, str) else json.dumps(q)
                    key = (ds, hashlib.sha1(q.encode()).hexdigest()[:16])
                    c = cells.setdefault(key, {"id": f"{ds}-{key[1]}", "dataset": ds, "query": q})
                    c.setdefault("score", {})[d["model_name"]] = r.get("score")
                    c.setdefault("cost", {})[d["model_name"]] = r.get("cost")
    rows = [
        c
        for c in cells.values()
        if all(c["score"].get(m) is not None and c["cost"].get(m) is not None for m in MODELS)
    ]
    rng = random.Random(SEED)
    out = []
    for ds, _ in DATASETS:
        sub = sorted((c for c in rows if c["dataset"] == ds), key=lambda c: c["id"])
        out += rng.sample(sub, min(PER_DATASET, len(sub)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", default=str(HERE / "raw"), help="where the 1.3 GB release lands")
    p.add_argument("--out", default=str(HERE / "out"))
    args = p.parse_args()

    import numpy as np
    from fastembed import TextEmbedding

    rows = load(download(Path(args.cache)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} questions x {len(MODELS)} models; embedding with {EMBEDDER}...")
    emb = np.array(
        list(TextEmbedding(EMBEDDER).embed([r["query"][:MAX_CHARS] for r in rows])),
        dtype=np.float32,
    )
    np.save(out / "embeddings.npy", emb)
    with (out / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:  # the question text, for a router that reads it (jev.py)
            fh.write(json.dumps({"id": r["id"], "query": r["query"]}) + "\n")
    with (out / "table.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            r.pop("query")  # the table carries grades and costs
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out / 'table.jsonl'}, {out / 'queries.jsonl'} and {out / 'embeddings.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
