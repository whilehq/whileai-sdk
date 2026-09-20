"""``wai.rows``: a public benchmark's prompts and completions become the rows
every measurement reads (#613).

The issue's comment measured the cost of the missing road: an outcome-reward
GRPO arm spent 37% of its rollout budget on groups where all five rollouts
scored the same, which is exactly what ``select(mode="rl", band=(0.2, 0.8))``
exists to drop and could not, because hand-built rows had no documented way
in. These tests pin the road: five completions per prompt, some all-pass and
all-fail groups, and ``select`` dropping them.
"""

from __future__ import annotations

import pytest

import whileai as wai
from whileai.simulations.schema import from_row, version_of

QUESTIONS = [
    "Janet has 3 ducks. Each lays 4 eggs a day. How many eggs a day?",
    "A shirt costs $20 and is 25% off. What does it cost?",
    "Tom reads 12 pages an hour for 3 hours. How many pages?",
    "A train goes 60 miles in an hour. How far in 2.5 hours?",
    "There are 8 boxes of 6 apples. How many apples?",
    "Sara had 50 dollars and spent 18. How much is left?",
]
GOLD = ["12", "15", "36", "150", "48", "32"]

# Five completions per prompt: two groups unanimous (all pass, all fail),
# four mixed. The unanimous ones carry no gradient and must go.
PATTERN = {
    0: [1, 1, 1, 1, 1],
    1: [0, 0, 0, 0, 0],
    2: [1, 0, 1, 0, 1],
    3: [1, 1, 0, 1, 1],
    4: [0, 1, 0, 0, 1],
    5: [1, 0, 0, 1, 1],
}


def _completions(long: bool = False) -> list[list[str]]:
    out = []
    for i, gold in enumerate(GOLD):
        group = []
        for j, right in enumerate(PATTERN[i]):
            number = gold if right else str(int(gold) + j + 1)
            if long:
                # A GSM8K chain of thought: over the 600-char cap, ending on
                # ``#### N`` with no punctuation.
                body = f"Step {j}: work it out carefully, line by line, as the model does. " * 10
                group.append(f"{body}\n#### {number}")
            else:
                # distinct per rollout: ``select`` dedupes identical replies
                group.append(f"Try {j}: the answer is {number}.")
        out.append(group)
    return out


def test_rows_from_a_verifier_are_typed_and_graded():
    rows = wai.rows(QUESTIONS, _completions(), wai.verify.MathEqual(), references=GOLD)
    assert len(rows) == 30
    first = rows[0]
    for key in ("task_id", "prompt", "final_text", "reward"):
        assert key in first, key
    assert version_of(first) == "1"
    assert first["scenario_id"] == first["task_id"]
    assert first["judge_name"] == "MathEqual"
    assert first["privileged"] == {"reference": "12"}
    rewards = [[r["reward"] for r in rows[i * 5 : i * 5 + 5]] for i in range(6)]
    assert rewards == [PATTERN[i] for i in range(6)]
    task, rollout, judgments, _ = from_row(first)
    assert task.task_id == first["task_id"]
    assert rollout.index == 0 and judgments[0].reward == 1


def test_task_ids_default_to_a_stable_hash_of_the_prompt():
    a = wai.rows(QUESTIONS[:1], ["12"], [1])
    b = wai.rows(QUESTIONS[:1], ["13"], [0])
    assert a[0]["task_id"] == b[0]["task_id"]
    named = wai.rows(QUESTIONS[:1], ["12"], [1], task_ids=["gsm8k-test-0"])
    assert named[0]["task_id"] == "gsm8k-test-0"


def test_precomputed_rewards_and_markers_ride_in():
    scores = [PATTERN[i] for i in range(6)]
    markers = [[{"chars": float(len(c))} for c in group] for group in _completions()]
    rows = wai.rows(QUESTIONS, _completions(), scores, markers=markers)
    assert rows[7]["reward"] == 0 and isinstance(rows[7]["reward"], int)
    assert rows[7]["markers"] == {"chars": float(len(_completions()[1][2]))}
    assert rows[7]["label_source"] == "given"
    flat = wai.rows(QUESTIONS, _completions(), [v for i in range(6) for v in PATTERN[i]])
    assert [r["reward"] for r in flat] == [r["reward"] for r in rows]
    # a bool is a verdict, not a label
    assert wai.rows(QUESTIONS[:1], [["12", "13"]], [[True, False]])[1]["reward"] == 0


def test_callable_rewards_take_prompt_completion_and_reference():
    def two(prompt: str, completion: str) -> float:
        return 1.0 if "12" in completion else 0.0

    def three(prompt: str, completion: str, reference: str) -> float:
        return float(reference in completion)

    assert [r["reward"] for r in wai.rows(QUESTIONS[:1], [["12", "13"]], two)] == [1, 0]
    rows = wai.rows(QUESTIONS[:1], [["12", "13"]], three, references=GOLD[:1])
    assert [r["reward"] for r in rows] == [1, 0]
    assert rows[0]["judge_name"] == "three"

    def contract(row: dict) -> dict:
        return {"reward": 1.0, "reason": "seen", "markers": {"seen": 1.0}}

    graded = wai.rows(QUESTIONS[:1], ["12"], contract)
    assert graded[0]["reward"] == 1 and graded[0]["markers"] == {"seen": 1.0}


def test_message_list_prompts_keep_the_conversation():
    prompt = [
        {"role": "system", "content": "Answer with a number."},
        {"role": "user", "content": QUESTIONS[0]},
    ]
    rows = wai.rows([prompt], ["12"], [1])
    assert rows[0]["prompt"] == QUESTIONS[0]
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]


def test_bad_shapes_name_the_kwarg():
    with pytest.raises(ValueError, match="completions="):
        wai.rows(QUESTIONS, ["12"], [1])
    with pytest.raises(ValueError, match="references="):
        wai.rows(QUESTIONS, _completions(), wai.verify.MathEqual(), references=GOLD[:2])
    with pytest.raises(ValueError, match=r"reward\[0\]\[1\]"):
        wai.rows(QUESTIONS[:1], [["12", "13"]], [[1, 2]])
    with pytest.raises(ValueError, match="reward="):
        wai.rows(QUESTIONS[:1], [["12", "13"]], [1, 0, 1])
    with pytest.raises(TypeError, match="reward="):
        wai.rows(QUESTIONS[:1], ["12"], object())


def test_rows_are_the_rows_every_measurement_reads():
    before = wai.rows(QUESTIONS, _completions(), wai.verify.MathEqual(), references=GOLD)
    after = wai.rows(QUESTIONS, [[g] * 5 for g in GOLD], wai.verify.MathEqual(), references=GOLD)
    passed = wai.pass_at(before)
    assert passed.n_groups == 6 and passed.pass_at_1 == pytest.approx(17 / 30)
    report = wai.compare(before, after)
    assert report["n_paired_tasks"] == 6
    clean, dec = wai.decontaminate(before, against=after[:5])
    assert dec["n_contaminated"] == 5 and len(clean) == 25
    sized = wai.simulations.holdout_size(0.1, before=before)
    assert sized["n_tasks"] > 0
    var = wai.simulations.eval_variance(before, before, before)
    assert var["run_std"] == 0.0


@pytest.mark.parametrize("long", [False, True])
def test_select_rl_drops_the_unanimous_groups(long: bool):
    """The issue comment: 5 rollouts per prompt, some all-pass or all-fail,
    ``band=(0.2, 0.8)`` drops them and keeps the four mixed groups. The
    ``long`` case is a GSM8K chain of thought over the char cap that ends
    on ``#### N``; it used to be dropped as truncated before it reached the
    band, and the report then said only "kept 0"."""
    rows = wai.rows(QUESTIONS, _completions(long), wai.verify.MathEqual(), references=GOLD)
    picked = wai.select(rows, mode="rl", band=(0.2, 0.8))
    kept = sorted({r["task_id"] for r in picked})
    expected = sorted({rows[i * 5]["task_id"] for i in (2, 3, 4, 5)})
    assert kept == expected, str(picked)
    assert len(picked) == 20
    assert picked.report["unanimous_groups_dropped"] == 2
    assert picked.report["gates"] == {"do_nothing": 0, "incomplete_junk": 0, "unusable_label": 0}
    assert "rows dropped before grouping" not in str(picked)


def test_select_report_names_a_gate_that_dropped_rows():
    rows = wai.rows(QUESTIONS, _completions(), [[0.5] * 5 for _ in GOLD])
    picked = wai.select(rows, mode="rl")
    assert len(picked) == 0
    assert picked.report["gates"]["unusable_label"] == 30
    assert "rows dropped before grouping: unusable label 30" in str(picked)
