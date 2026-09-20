"""Scenario-embedding selection. Lexical and semantic vectors are never mixed."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from whileai._env import getenv

from .diversity import apply_annealing_explore
from .scenarios import novelty as min_cosine_distance

# HASH_DIM = 256: buckets of the word+bigram hash embedder; enough that
# two short asks rarely collide, small enough that the O(n^2) novelty
# scan stays cheap in pure Python (convention, untested).
HASH_DIM = 256
_DIM = HASH_DIM
#: EMBED_BATCH = 128 texts per HTTP embedding call, EMBED_TIMEOUT_S = 30,
# EMBED_RETRIES = 3 with EMBED_BACKOFF_S * attempt between them: the
# OpenAI and Ollama embedders share these (convention).
EMBED_BATCH = 128
EMBED_TIMEOUT_S = 30
EMBED_RETRIES = 3
EMBED_BACKOFF_S = 1.5
# BGE_ENCODE_BATCH = 32: sentence-transformers batch on CPU or MPS (convention).
BGE_ENCODE_BATCH = 32
# MODAL_EMBED_TIMEOUT_S = 20 / MODAL_PROBE_TIMEOUT_S = 1.5: the hosted
# embedder's call timeout, and the quick probe that decides whether it is
# up before a run commits to it (convention).
MODAL_EMBED_TIMEOUT_S = 20.0
MODAL_PROBE_TIMEOUT_S = 1.5


# Every float sum on the batch-selection path goes through ``math.fsum``,
# never the builtin ``sum``. ``fsum`` is correctly rounded, so its result
# is fixed by IEEE 754 alone; builtin ``sum`` of floats changed algorithm
# in CPython 3.12 (compensated summation, gh-100425), and the last-ulp
# differences flipped the sign of near-zero novelty scores and with it the
# sort order that picks the batch. Same seed then drew different rows on
# 3.11 and 3.12 (issue #410).
def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cos(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def _hash_vector(text: str, dim: int = _DIM) -> list[float]:
    vec = [0.0] * dim
    words = re.findall(r"[a-z]+|\d+", str(text).lower())
    words = ["<num>" if word.isdigit() else word for word in words]
    tokens = [f"w:{w}" for w in words] + [f"b:{a}>{b}" for a, b in itertools.pairwise(words)]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "big") % dim] += 1.0
    return _normalize(vec)


class HashEmbedder:
    name = "hash-word-bigram-v2"
    semantic = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


class CallableEmbedder:
    def __init__(
        self, fn: Callable[[Sequence[str]], Any], name: str = "callable", semantic: bool = True
    ):
        self.fn = fn
        self.name = name
        self.semantic = semantic

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raw = self.fn(list(texts))
        return [_normalize([float(x) for x in row]) for row in raw]


class OllamaEmbedder:
    semantic = True

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"ollama:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            request = urllib.request.Request(
                f"{self.host}/api/embed",
                data=json.dumps(
                    {"model": self.model, "input": list(texts[start : start + EMBED_BATCH])}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=EMBED_TIMEOUT_S) as response:
                output.extend(json.loads(response.read()).get("embeddings") or [])
        if len(output) != len(texts):
            raise RuntimeError(
                f"{self.name} returned {len(output)} embeddings for {len(texts)} texts"
            )
        return [_normalize([float(x) for x in row]) for row in output]


class OpenAIEmbedder:
    semantic = True

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = os.environ.get("OPENAI_BASE_URL", base_url).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or getenv("API_KEY", "")
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps(
                    {"model": self.model, "input": list(texts[start : start + EMBED_BATCH])}
                ).encode(),
                headers=headers,
            )
            for attempt in range(EMBED_RETRIES):
                try:
                    with urllib.request.urlopen(request, timeout=EMBED_TIMEOUT_S) as response:
                        rows = sorted(
                            json.loads(response.read()).get("data") or [],
                            key=lambda row: row["index"],
                        )
                        output.extend(row["embedding"] for row in rows)
                    break
                except (OSError, urllib.error.URLError):
                    if attempt == EMBED_RETRIES - 1:
                        raise
                    time.sleep(EMBED_BACKOFF_S * (attempt + 1))
        if len(output) != len(texts):
            raise RuntimeError(
                f"{self.name} returned {len(output)} embeddings for {len(texts)} texts"
            )
        return [_normalize([float(x) for x in row]) for row in output]


DEFAULT_EMBED_URL = "https://zeroproofai--zeroproof-embed-embedder-embed.modal.run"


class ModalEmbedder:
    """The hosted While embedding endpoint: batched, semantic."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = MODAL_EMBED_TIMEOUT_S,
    ):
        self.url = url or getenv("EMBED") or DEFAULT_EMBED_URL
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "")
        self.timeout = timeout
        self.name = "modal:bge-small"
        self.semantic = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"texts": list(map(str, texts))}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["embeddings"]


def _live_modal(url: str | None = None) -> ModalEmbedder | None:
    candidate = ModalEmbedder(url, timeout=MODAL_PROBE_TIMEOUT_S)
    try:
        candidate.embed(["probe"])
    except Exception:
        return None
    candidate.timeout = MODAL_EMBED_TIMEOUT_S
    return candidate


_BGE_MODEL = "BAAI/bge-small-en-v1.5"


def _bge_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def bge_embedder(model: str = _BGE_MODEL, device: str | None = None) -> CallableEmbedder:
    """Local BGE-small via sentence-transformers. MPS, else CPU. No HTTP."""
    from sentence_transformers import SentenceTransformer

    chosen = _bge_device(device)
    try:
        st = SentenceTransformer(model, device=chosen)
    except Exception:
        if chosen == "cpu":
            raise
        chosen = "cpu"
        st = SentenceTransformer(model, device=chosen)

    def encode(texts: Sequence[str]):
        cleaned = [str(t) if str(t).strip() else " " for t in texts]
        return st.encode(
            cleaned, batch_size=BGE_ENCODE_BATCH, normalize_embeddings=True, show_progress_bar=False
        )

    embedder = CallableEmbedder(encode, name=f"sentence-transformers:{model}")
    embedder.device = chosen  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return embedder


def resolve_embedder(embedder: Any = "hash") -> Any:
    if embedder is None or embedder == "hash":
        return HashEmbedder()
    if embedder == "modal":
        return _live_modal() or HashEmbedder()
    if embedder in {"bge", "sentence-transformers"}:
        return bge_embedder()
    if isinstance(embedder, str):
        kind, _, rest = embedder.partition(":")
        if kind == "ollama":
            return OllamaEmbedder(rest or "nomic-embed-text")
        if kind == "openai":
            return OpenAIEmbedder(rest or "text-embedding-3-small")
        if kind == "modal":
            return _live_modal(rest or None) or HashEmbedder()
        if kind in {"bge", "sentence-transformers"}:
            return bge_embedder(rest or _BGE_MODEL)
        raise ValueError(
            "embedder must be hash, bge, ollama:<model>, openai:<model>, or a callable"
        )
    if hasattr(embedder, "embed"):
        return embedder
    if callable(embedder):
        return CallableEmbedder(embedder)
    raise TypeError("invalid embedder")


def is_semantic(embedder: Any) -> bool:
    return bool(getattr(embedder, "semantic", False)) and not str(
        getattr(embedder, "name", "")
    ).startswith("hash")


#: Batch selection: candidates are k-means clustered (at most MAX_CLUSTERS
#: = 8 clusters, sqrt of the pool, KMEANS_ITERS = 8 Lloyd steps), then a
#: batch is half "typical" picks spread across clusters and half "fill"
#: picks by novelty against everything already run (FILL_SHARE = 0.5),
#: drawn from a k-center spread over CANDIDATE_MULTIPLE = 2 times the
#: batch. Cluster-then-compare-within-cluster is SemDeDup's recipe
#: (2303.09540, k-means then pairwise inside each cluster); the split and
#: the caps are conventions, untested. KMEANS_DEFAULT_ITERS = 40 is the
#: standalone default.
MAX_CLUSTERS = 8
KMEANS_ITERS = 8
KMEANS_DEFAULT_ITERS = 40
FILL_SHARE = 0.5
CANDIDATE_MULTIPLE = 2
# SUMMARY_CLUSTERS = 8: how many concentrated and sparse cluster examples
# the batch info reports (convention).
SUMMARY_CLUSTERS = 8


def kmeans(
    vectors: list[list[float]], k: int, seed: int, iterations: int = KMEANS_DEFAULT_ITERS
) -> tuple[list[int], list[list[float]]]:
    n = len(vectors)
    if not n:
        return [], []
    dim = len(vectors[0])
    k = max(1, min(k, n))
    rng = random.Random(seed)
    centers = [list(vectors[rng.randrange(n)])]
    while len(centers) < k:
        distances = []
        for vec in vectors:
            distances.append(min(math.fsum((a - b) ** 2 for a, b in zip(vec, c)) for c in centers))
        total = math.fsum(distances)
        if total <= 0:
            centers.append(list(vectors[len(centers) % n]))
            continue
        pick = rng.random() * total
        acc = 0.0
        index = n - 1
        for i, dist in enumerate(distances):
            acc += dist
            if acc >= pick:
                index = i
                break
        centers.append(list(vectors[index]))
    labels = [-1] * n
    for _ in range(iterations):
        new_labels = []
        for vec in vectors:
            best, best_d = 0, None
            for ci, center in enumerate(centers):
                d = math.fsum((a - b) ** 2 for a, b in zip(vec, center))
                if best_d is None or d < best_d:
                    best, best_d = ci, d
            new_labels.append(best)
        if new_labels == labels:
            break
        labels = new_labels
        for cluster in range(k):
            members = [vectors[i] for i, lab in enumerate(labels) if lab == cluster]
            if members:
                centers[cluster] = [
                    math.fsum(row[j] for row in members) / len(members) for j in range(dim)
                ]
    return labels, centers


def select_diverse(vectors: list[list[float]], k: int) -> list[int]:
    if k >= len(vectors):
        return list(range(len(vectors)))
    if not vectors:
        return []
    dim = len(vectors[0])
    centroid = [math.fsum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    centroid = _normalize(centroid)
    first = min(range(len(vectors)), key=lambda i: _cos(vectors[i], centroid))
    chosen = [first]
    nearest = [1.0 - _cos(v, vectors[first]) for v in vectors]
    chosen_set = {first}
    while len(chosen) < k:
        nxt = max(
            (i for i in range(len(vectors)) if i not in chosen_set),
            key=lambda i: nearest[i],
            default=None,
        )
        if nxt is None:
            break
        chosen.append(nxt)
        chosen_set.add(nxt)
        for i, v in enumerate(vectors):
            d = 1.0 - _cos(v, vectors[nxt])
            if d < nearest[i]:
                nearest[i] = d
    return chosen


class EmbeddingArchive:
    """Tested scenario embeddings. Refuses to mix lexical and semantic rows."""

    def __init__(self, embedder_name: str, semantic: bool):
        self.embedder_name = embedder_name
        self.semantic = semantic
        self.vectors: list[list[float]] = []

    def compatible(self, embedder: Any) -> bool:
        return (
            str(getattr(embedder, "name", "")) == self.embedder_name
            and bool(is_semantic(embedder)) == self.semantic
        )

    def add(self, vectors: Iterable[list[float]]) -> None:
        self.vectors.extend(vectors)


def select_execution_batch(
    candidates: list[str],
    *,
    embedder: Any,
    archive: EmbeddingArchive,
    batch_size: int,
    seed: int = 0,
    round_index: int = 0,
) -> tuple[list[dict], dict]:
    """Embed, cluster/stratify, k-center, novelty-filter, return batch metas.

    Each meta: text, vector, cluster, novelty, reason.
    """
    texts = list(dict.fromkeys(str(c) for c in candidates if str(c)))
    info: dict[str, Any] = {
        "embedder": getattr(embedder, "name", "unknown"),
        "semantic": is_semantic(embedder),
        "degraded": [],
        "mixed_spaces_refused": False,
    }
    if not texts:
        return [], info
    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise RuntimeError("embedder returned a different count than inputs")

    n_clusters = max(1, min(MAX_CLUSTERS, round(math.sqrt(len(texts)))))
    labels, _ = kmeans(vectors, n_clusters, seed, iterations=KMEANS_ITERS)
    by_cluster: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_cluster.setdefault(int(label), []).append(i)

    need = max(1, int(batch_size))
    fill_n = max(1, int(need * FILL_SHARE))
    typical_n = max(0, need - fill_n)

    archive_ok = archive.compatible(embedder)
    if archive.vectors and not archive_ok:
        info["mixed_spaces_refused"] = True
        info["degraded"].append("embedding_space_mismatch")
        tested = []
    else:
        tested = archive.vectors

    # Per-round stream: seeding on ``seed`` alone replayed the same draws
    # every round against a changing pool.
    rng = random.Random(int(seed) + int(round_index) * 7919)
    typical_idx: list[int] = []
    cluster_order = sorted(by_cluster, key=lambda c: (-len(by_cluster[c]), c))
    while len(typical_idx) < typical_n and cluster_order:
        progressed = False
        for cluster in cluster_order:
            remaining = [i for i in by_cluster[cluster] if i not in typical_idx]
            if not remaining:
                continue
            typical_idx.append(rng.choice(remaining))
            progressed = True
            if len(typical_idx) >= typical_n:
                break
        if not progressed:
            break

    per_cluster = max(1, math.ceil((need * CANDIDATE_MULTIPLE) / max(1, len(by_cluster))))
    stratified: list[int] = []
    for indices in by_cluster.values():
        sub = [vectors[i] for i in indices]
        picks = select_diverse(sub, min(per_cluster, len(indices)))
        stratified.extend(indices[p] for p in picks)
    stratified = list(dict.fromkeys(stratified))
    spread_vecs = [vectors[i] for i in stratified]
    spread_picks = select_diverse(
        spread_vecs, min(max(need * CANDIDATE_MULTIPLE, need), len(stratified))
    )
    ranked = [stratified[p] for p in spread_picks]

    taken = set(typical_idx)
    fill_idx: list[int] = []
    novelty_of = {}
    for index in ranked:
        novelty_of[index] = min_cosine_distance(vectors[index], tested)
    for index in sorted(ranked, key=lambda i: (-novelty_of.get(i, 0.0), texts[i])):
        if index in taken:
            continue
        fill_idx.append(index)
        taken.add(index)
        if len(fill_idx) >= fill_n:
            break
    for index in range(len(texts)):
        if len(typical_idx) + len(fill_idx) >= need:
            break
        if index not in taken:
            fill_idx.append(index)
            taken.add(index)

    batch_idx = list(dict.fromkeys(typical_idx + fill_idx))[:need]
    batch_idx = apply_annealing_explore(
        batch_idx, len(texts), novelty_of, need=need, round_index=round_index, seed=seed
    )
    selected: list[dict] = []
    for index in batch_idx:
        score = novelty_of.get(index)
        if score is None:
            score = min_cosine_distance(vectors[index], tested)
        kind = "typical" if index in typical_idx else "fill"
        selected.append(
            {
                "text": texts[index],
                "vector": vectors[index],
                "cluster": int(labels[index]),
                "novelty": score,
                "reason": (
                    f"{kind} cluster={labels[index]} novelty={score:.3f} "
                    f"embedder={info['embedder']}"
                ),
            }
        )
    info["clusters"] = len(by_cluster)
    info["candidate_pool"] = len(texts)
    info["batch_design"] = "typical_kcenter_fill_plus_annealing_explore"
    info["round_index"] = int(round_index)
    dense = cluster_order[: max(1, len(cluster_order) // 3)] if cluster_order else []
    sparse = (
        list(reversed(cluster_order))[: max(1, len(cluster_order) // 3)] if cluster_order else []
    )
    info["concentrated"] = [texts[by_cluster[c][0]] for c in dense if by_cluster[c]][
        :SUMMARY_CLUSTERS
    ]
    info["sparse"] = [texts[by_cluster[c][0]] for c in sparse if by_cluster[c]][:SUMMARY_CLUSTERS]
    return selected, info
