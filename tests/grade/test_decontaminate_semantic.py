"""#286: word overlap does not see a paraphrase. ``decontaminate(embedder=)``
adds a semantic pass on top of the n-gram rule, counted on its own, and
task identity decides before similarity does."""

from __future__ import annotations

import math

import pytest

import whileai.simulations as wai
from whileai.simulations.score.stats import decontaminate

# Deterministic "embeddings": a direction per topic, so similarity is a
# fact of the test and not of any model.
_TOPIC = {
    "refund": (1.0, 0.0, 0.0),
    "money back": (0.97, 0.24, 0.0),  # the paraphrase: 0.97 cosine to refund
    "cancel": (0.0, 1.0, 0.0),
    "cancel three": (0.0, 0.97, 0.24),  # reads like cancel, a different task
    "weather": (0.0, 0.0, 1.0),
}


def fake_embedder(texts):
    out = []
    for text in texts:
        low = text.lower()
        vec = (0.0, 0.0, 0.0)
        for key, direction in _TOPIC.items():  # longest key wins
            if key in low and (vec == (0.0, 0.0, 0.0) or len(key) > 6):
                vec = direction
        out.append([3.0 * x for x in vec])  # un-normalized on purpose
    return out


EVAL = [
    {
        "scenario_id": "s-refund",
        "prompt": "please look up order ORD-4017 and tell me whether the refund has been issued yet",
    },
    {"scenario_id": "s-cancel", "prompt": "cancel one reservation for customer 88 and confirm"},
]

ROWS = [
    # 0: verbatim eval prompt -> exact
    {"prompt": "please look up order ORD-4017 and tell me whether the refund has been issued yet"},
    # 1: a paraphrase, no shared 8-gram -> invisible to words, 0.97 cosine
    {"prompt": "has my money back for order ORD-4017 gone through, can you check"},
    # 2: reads like the cancel task but is a different task with a different answer
    {"scenario_id": "s-other", "prompt": "cancel three reservations for customer 12 and 13"},
    # 3: unrelated
    {"prompt": "what is the weather in Lisbon"},
]


def test_semantic_pass_sees_the_paraphrase_the_ngram_rule_misses():
    lexical = decontaminate(ROWS, EVAL)[1]
    assert lexical["n_contaminated"] == 1 and lexical["n_exact"] == 1
    assert lexical["n_semantic"] == 0 and lexical["cosine"] is None

    kept, report = wai.decontaminate(ROWS, EVAL, embedder=fake_embedder)
    assert report["n_semantic"] == 2 and report["n_exact"] == 1 and report["n_near"] == 0
    assert report["n_contaminated"] == 3 == len(ROWS) - len(kept)
    assert kept == [ROWS[3]]
    semantic = [e for e in report["examples"] if e["match"] == "semantic"]
    assert [e["index"] for e in semantic] == [1, 2]
    assert semantic[0]["similarity"] == pytest.approx(0.97, abs=1e-3)
    assert semantic[0]["eval_prompt"].startswith("please look up order ORD-4017")
    assert report["cosine"] == 0.85 and report["by_field"] == {"prompt": 3}
    # the note says what a flag means: reads alike, not the same task
    joined = " ".join(report["notes"])
    assert "read alike" in joined and "not a verdict" in joined


def test_a_row_caught_by_words_is_not_counted_again_by_the_embedder():
    rows = [ROWS[0], ROWS[1]]
    _kept, report = decontaminate(rows, EVAL, embedder=fake_embedder)
    assert report["n_exact"] == 1 and report["n_semantic"] == 1
    assert report["n_contaminated"] == 2  # not 3
    matches = [e["match"] for e in report["examples"]]
    assert matches == ["exact", "semantic"]


def test_threshold_is_the_callers_and_the_default_stays_lexical():
    assert decontaminate(ROWS, EVAL, embedder=fake_embedder, cosine=0.99)[1]["n_semantic"] == 0
    assert decontaminate(ROWS, EVAL, embedder=fake_embedder, cosine=0.9)[1]["n_semantic"] == 2
    with pytest.raises(ValueError, match="between 0 and 1"):
        decontaminate(ROWS, EVAL, embedder=fake_embedder, cosine=1.5)
    with pytest.raises(TypeError, match="callable"):
        decontaminate(ROWS, EVAL, embedder="bge")
    with pytest.raises(ValueError, match="one vector per text"):
        decontaminate(ROWS, EVAL, embedder=lambda texts: [[1.0]])


def test_task_identity_decides_before_similarity_does():
    rows = [
        # the eval situation rephrased, same scenario_id: same task whatever the words
        {"scenario_id": "s-refund", "prompt": "what is the weather like where my refund is"},
        # a different recorded task that reads like the cancel eval prompt
        {"scenario_id": "s-other", "prompt": "cancel three reservations for customer 12"},
        # the same recorded task as an eval row is dropped even with no wording in common
        {"task_id": "s-cancel", "prompt": "weather"},
    ]
    kept, report = decontaminate(rows, EVAL, embedder=fake_embedder)
    assert kept == []
    assert report["n_same_task"] == 2 and report["n_semantic"] == 1
    assert report["n_contaminated"] == 3 and report["n_exact"] == 0 and report["n_near"] == 0
    by_index = {e["index"]: e for e in report["examples"]}
    assert by_index[0]["match"] == "same_task" and by_index[0]["task"] == "s-refund"
    assert by_index[1]["match"] == "semantic"
    assert by_index[2]["match"] == "same_task" and report["by_field"]["task"] == 2
    # without an embedder the same-task rule still holds on its own
    assert decontaminate(rows, EVAL)[1]["n_same_task"] == 2
    # the semantic pass never pairs a row with an eval prompt of its own task id:
    # with cosine=0 everything reads alike, and the s-refund eval prompt is skipped
    only_same = [{"scenario_id": "s-refund", "prompt": "x"}]
    assert (
        decontaminate(only_same, EVAL[:1], embedder=fake_embedder, cosine=0)[1]["n_same_task"] == 1
    )
    different = [{"scenario_id": "s-z", "prompt": "x"}]
    assert (
        decontaminate(different, EVAL[:1], embedder=fake_embedder, cosine=0)[1]["n_semantic"] == 1
    )


def test_embedder_is_called_once_per_side_and_vectors_are_normalized():
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return fake_embedder(texts)

    _kept, report = decontaminate(ROWS, EVAL, embedder=counting)
    assert len(calls) == 2  # eval prompts once, candidate rows once
    assert calls[0] == [e["prompt"] for e in EVAL]
    assert len(calls[1]) == 3  # the exact hit was not embedded
    for example in report["examples"]:
        if example["match"] == "semantic":
            assert 0 <= example["similarity"] <= 1 and not math.isnan(example["similarity"])
