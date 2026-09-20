"""The seeded draw depends on the seed and the inputs, not the interpreter.

Issue #410: ``simulate(reproducible=True, seed=0)`` drew one different
row on CPython 3.11 and 3.12. The batch picker summed floats with the
builtin ``sum``, whose algorithm changed in 3.12 (compensated summation),
and a novelty score of ``0.0`` on one interpreter was ``-2.2e-16`` on the
other, which reordered the novelty sort. Every float sum on the selection
path is now ``math.fsum``, which is correctly rounded and so fixed by
IEEE 754 alone.

The literals below were recorded once and are asserted on every
interpreter CI runs (3.10 to 3.13). They are the guarantee: if one of
these tests fails after a deliberate change to the writer or the picker,
the draw changed for every existing seed, and the CHANGELOG bullet for
that change must say so before the literals are updated.
"""

from __future__ import annotations

import hashlib

import whileai.simulations as wai
from whileai.simulations.generate.embeddings import (
    EmbeddingArchive,
    HashEmbedder,
    _cos,
    _normalize,
    select_execution_batch,
)
from whileai.simulations.generate.scenarios import novelty

# Ten copies of 0.1 sum to 0.9999999999999999 with the builtin ``sum`` on
# CPython 3.11 and to 1.0 on 3.12; ``fsum`` gives the correctly rounded
# value everywhere.
_TENTH = [0.1] * 10


def test_vector_arithmetic_is_correctly_rounded():
    assert repr(_normalize(_TENTH)[0]) == "0.31622776601683794"
    assert repr(_cos(_TENTH, _TENTH)) == "0.10000000000000002"


def test_novelty_of_a_duplicate_is_exactly_zero():
    # Before the fix this was 0.0 on one interpreter and -2.2e-16 on
    # another, and the sign decided the sort.
    assert novelty(_TENTH, [_TENTH]) == 0.0
    assert (
        repr(novelty([0.1] * 9 + [0.2], [_TENTH, [0.2] * 5 + [0.1] * 5])) == "0.03523617876226781"
    )


_TESTED = [
    "Look up order ORD-1001 and tell me where it is.",
    "Cancel my order, it never arrived.",
    "Where is ORD-2002?",
]
_CANDIDATES = [
    "Look up order ORD-1001 and tell me where it is.",
    "Where is ORD-2002?",
    "Can you check on ORD-3003 for me, it is late.",
    "hi",
    "I need a refund for ORD-4004, the item was broken on arrival and support has not answered.",
    "Track ORD-5005 please.",
    "What is your return policy for orders placed last month?",
    "Cancel my order, it never arrived.",
    "ORD-6006 shows delivered but nothing came.",
    "Change the address on ORD-7007 before it ships.",
]
# (texts in batch order, repr of each novelty score) per round.
_GOLDEN_BATCHES = {
    0: (
        [
            "hi",
            "What is your return policy for orders placed last month?",
            "I need a refund for ORD-4004, the item was broken on arrival and support has not answered.",
            "Change the address on ORD-7007 before it ships.",
        ],
        ["1.0", "0.8998747651356482", "0.8229155991697134", "0.7249904508915366"],
    ),
    1: (
        [
            "Cancel my order, it never arrived.",
            "hi",
            "What is your return policy for orders placed last month?",
            "I need a refund for ORD-4004, the item was broken on arrival and support has not answered.",
        ],
        ["0.0", "1.0", "0.8998747651356482", "0.8229155991697134"],
    ),
    2: (
        [
            "Look up order ORD-1001 and tell me where it is.",
            "hi",
            "What is your return policy for orders placed last month?",
            "I need a refund for ORD-4004, the item was broken on arrival and support has not answered.",
        ],
        ["0.0", "1.0", "0.8998747651356482", "0.8229155991697134"],
    ),
}


def test_batch_picker_golden_draws():
    """Seed the picker, take three rounds, compare to recorded literals.

    The candidate list repeats two tested rows on purpose: their novelty
    is exactly ``0.0`` and they sort last, which is the case that flipped
    between interpreters.
    """
    emb = HashEmbedder()
    archive = EmbeddingArchive(emb.name, emb.semantic)
    archive.add(emb.embed(_TESTED))
    for round_index, (texts, scores) in _GOLDEN_BATCHES.items():
        batch, _info = select_execution_batch(
            _CANDIDATES,
            embedder=emb,
            archive=archive,
            batch_size=4,
            seed=0,
            round_index=round_index,
        )
        assert [m["text"] for m in batch] == texts, round_index
        assert [repr(m["novelty"]) for m in batch] == scores, round_index


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]
_GOLDEN_SCENARIO_IDS = [
    "probe_e55c48",
    "sc-1adc2ab6c1",
    "sc-2f784fb823",
    "sc-349121ce62",
    "sc-38e4c114a7",
    "sc-4743a51f7d",
    "sc-7e880e0655",
    "sc-b5b93f1da9",
]
_GOLDEN_PROMPT_SHA = "406076192baa83b0"


def test_simulate_seed_zero_golden_draw():
    """The issue's script at budget 16: same seed, same rows, any CPython."""
    data = wai.simulate(
        wai.seeded_agent(_TOOLS),
        tools=_TOOLS,
        system_prompt="Help customers with orders.",
        simulator=False,
        mode="rl",
        repeats=2,
        repeat_policy="fixed",
        budget=16,
        reproducible=True,
        seed=0,
    )
    rows = data.rows()
    assert len(rows) == 16
    assert sorted({str(r.get("scenario_id")) for r in rows}) == _GOLDEN_SCENARIO_IDS
    digest = hashlib.sha256("\n".join(str(r.get("prompt")) for r in rows).encode()).hexdigest()
    assert digest[:16] == _GOLDEN_PROMPT_SHA
