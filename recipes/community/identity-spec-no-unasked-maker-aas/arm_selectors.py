"""The four online selectors, and the audit that predicts what they do.

Zeng (arXiv:2607.07023) frames online data selection as a reweighted SFT
objective: the scorer that decides which rows survive is playing the part
usually given to a reward model, so the *attribute mixture* of what it keeps
is an implicit preference over behaviour. Every selector here spends the
same token budget; they differ only in the order they spend it.

``audit`` is the paper's Alignment Drift Auditing (ADA) diagnostic, and it
costs no GPU: it reports the attribute mixture of the kept rows, from which
the paper claims the direction of the behavioural shift is predictable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def _stable_rand(seed: int, idx: int) -> float:
    """Deterministic uniform in [0, 1). Reproducible across machines."""
    h = hashlib.sha256(f"{seed}:{idx}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _take_to_budget(order: Sequence[dict], budget_tokens: int) -> list[dict]:
    """Spend the token budget in the given order. The budget, not the row
    count, is what is held equal across arms -- a selector that prefers short
    rows would otherwise get more gradient steps for the same tokens."""
    kept, spent = [], 0
    for row in order:
        if spent + row["n_tokens"] > budget_tokens:
            continue
        kept.append(row)
        spent += row["n_tokens"]
    return kept


def select(meta: Sequence[dict], selector: str, budget_tokens: int, seed: int = 0) -> list[dict]:
    """Return the rows one selector keeps for ``budget_tokens`` of training.

    ``meta`` rows carry ``idx``, ``loss`` (mean NLL of the assistant tokens
    under the untrained base), ``n_tokens`` and ``kind`` ("focused" for an
    identity row, "control" otherwise).
    """
    rows = list(meta)
    if selector == "random":
        order = sorted(rows, key=lambda r: _stable_rand(seed, r["idx"]))
        return _take_to_budget(order, budget_tokens)

    if selector == "loss":
        # "Train on what the model gets wrong." The default a busy engineer
        # reaches for, and the paper's loss-based online selector.
        order = sorted(rows, key=lambda r: -r["loss"])
        return _take_to_budget(order, budget_tokens)

    if selector == "aas":
        # Alignment-Aware Selection: the same loss ranking, but the share of
        # the budget that identity rows may take is capped at the share they
        # hold in the pool. The selector stays loss-greedy inside each
        # stratum, so it keeps the efficiency and gives up only the freedom
        # to change the mixture.
        pool_tokens = sum(r["n_tokens"] for r in rows) or 1
        focused_share = sum(r["n_tokens"] for r in rows if r["kind"] == "focused") / pool_tokens
        caps = {
            "focused": int(budget_tokens * focused_share),
            "control": budget_tokens - int(budget_tokens * focused_share),
        }
        kept, spent = [], {"focused": 0, "control": 0}
        for row in sorted(rows, key=lambda r: -r["loss"]):
            k = row["kind"] if row["kind"] in caps else "control"
            if spent[k] + row["n_tokens"] > caps[k]:
                continue
            kept.append(row)
            spent[k] += row["n_tokens"]
        return kept

    raise ValueError(f"unknown selector {selector!r}; use random, loss or aas")


def audit(kept: Sequence[dict], pool: Sequence[dict]) -> dict:
    """ADA: the attribute mixture of the selected rows (Zeng, arXiv:2607.07023).

    No GPU and no model. The paper's second claim is that the direction of
    the behavioural shift is readable from these numbers before training, so
    this is the cheap half of the method and the half worth running first.
    """
    kept = list(kept)
    tok = sum(r["n_tokens"] for r in kept) or 1
    pool_tok = sum(r["n_tokens"] for r in pool) or 1
    foc_tok = sum(r["n_tokens"] for r in kept if r["kind"] == "focused")
    return {
        "rows": len(kept),
        "tokens": tok,
        "identity_row_share": (
            sum(1 for r in kept if r["kind"] == "focused") / len(kept) if kept else 0.0
        ),
        "identity_token_share": foc_tok / tok,
        "pool_identity_token_share": (
            sum(r["n_tokens"] for r in pool if r["kind"] == "focused") / pool_tok
        ),
        "mean_loss": sum(r["loss"] for r in kept) / len(kept) if kept else 0.0,
        "mean_tokens_per_row": tok / len(kept) if kept else 0.0,
    }
