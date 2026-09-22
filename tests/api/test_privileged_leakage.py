"""Teacher-only context never reaches a student-visible field.

``Privileged`` (``principle``, ``hidden_state``, ``reference``) is context
the teacher sees and the student never does. Today the projection
``to_row`` simply does not read it and every exporter rebuilds the
student's view from ``prompt``/``steps``/``final_text``, so nothing leaks.
That is a property worth pinning: the failure mode is silent, it lands in
a training file, and it is discovered only after a model has memorised the
answer key.

The guard is depth-independent since #149. ``export_row`` used to be an
allowlist, which made nesting unreachable by accident -- no carrier for a
nested privileged block was copied out -- and every test here was written
flat because of it. The export now carries the whole row, so the tests
below pin the nested case too.

Each test names the boundary it guards. ``from_row``/``to_row``
passthrough is deliberately excluded: carrying a source row's unknown keys
back out is the wire round-trip identity, not a student-visible export,
and it is pinned as such in ``test_schema_v1``.

The last section is the RLVR path, where the block is not decoration: a
verifier reads ``privileged.reference`` and writes a ``reason`` about it,
and ``reason`` is on the export carry list. Every guard above passes
vacuously on an offline run, because nothing there populates the block at
all (#31); these do not.
"""

import json

import pytest

from whileai.simulations.data import export_row
from whileai.simulations.export import (
    _CARRY_KEYS,
    export_preference,
    export_training,
    training_rows,
)
from whileai.simulations.schema import (
    Privileged,
    Rollout,
    Step,
    Task,
    to_row,
)
from whileai.simulations.score.judging import (
    build_preference_pairs,
    evaluate,
    run_judge,
)
from whileai.simulations.verify import (
    CodeExec,
    ExactMatch,
    Includes,
    MathEqual,
    MultipleChoice,
    Numeric,
)

SECRET = "zzteachersecretzz"
PRIVILEGED_KEYS = ("principle", "hidden_state", "reference")


def _privileged_row(
    prompt: str = "refund order 4412", reward: float = 1.0, final_text: str = "Refunded."
) -> dict:
    """A source row that carries every privileged key, as a hostile
    upstream (a hand-written trace, a platform pull) might."""
    return {
        "prompt": prompt,
        "steps": [
            {"tool": "lookup_order", "arguments": {"order_id": "4412"}, "result": {"status": "ok"}}
        ],
        "final_text": final_text,
        "scenario_id": "sc-1",
        "reward": reward,
        "principle": f"{SECRET}_principle",
        "hidden_state": {"answer": f"{SECRET}_hidden"},
        "reference": f"{SECRET}_reference",
    }


def _nested_privileged_row() -> dict:
    """The same hostile upstream, one level down.

    Written for #149. While ``export_row`` was an allowlist these keys
    were unreachable: nothing that could hold a nested privileged block
    (``lineage``, ``scenario_dimensions``, ``judge_meta``, a tool
    ``result``) was copied out at all, so a top-level-only guard was
    enough and every test here only ever built flat rows. The export now
    carries the whole row, so the guard has to be as deep as the row is.
    """
    return {
        "prompt": "refund order 4412",
        "steps": [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": "4412"},
                # a tool whose result quotes the teacher's answer key
                "result": {"status": "ok", "rubric": f"{SECRET}_in_a_tool_result"},
            }
        ],
        "final_text": "Refunded.",
        "scenario_id": "sc-1",
        "reward": 1.0,
        "world_state": "exists",
        "markers": {"grounded": 1.0},
        "scenario_dimensions": {"tool": "lookup_order", "privileged": {"principle": SECRET}},
        "lineage": {"source": "grade", "reference": f"{SECRET}_in_lineage"},
        "judge_meta": {"trials": [{"n": 1, "hidden_state": {"answer": SECRET}}]},
    }


def _dump(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _privileged_keys_in(value) -> list[str]:
    """Every privileged key name reachable anywhere inside ``value``."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in (*PRIVILEGED_KEYS, "privileged", "rubric"):
                found.append(str(key))
            found.extend(_privileged_keys_in(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_privileged_keys_in(item))
    return found


def test_to_row_never_projects_the_privileged_block():
    """The teacher's own ``Task.privileged`` has no wire projection."""
    task = Task(
        task_id="t-1",
        prompt="refund order 4412",
        privileged=Privileged(
            principle=f"{SECRET}_principle",
            hidden_state={"answer": f"{SECRET}_hidden"},
            reference=f"{SECRET}_reference",
        ),
    )
    rollout = Rollout(
        rollout_id="r-1",
        task_id="t-1",
        final_text="Refunded.",
        steps=[Step(tool="lookup_order", arguments={"order_id": "4412"}, result={"status": "ok"})],
    )
    row = to_row(task, rollout)
    assert SECRET not in _dump(row)
    for key in PRIVILEGED_KEYS:
        assert key not in row


def test_engine_export_row_drops_privileged_keys():
    """``export_row`` is what ``simulate()`` writes to JSONL."""
    out = export_row(_privileged_row())
    assert SECRET not in _dump(out)
    for key in PRIVILEGED_KEYS:
        assert key not in out


def test_engine_export_row_drops_privileged_keys_at_any_depth():
    """The export carries the whole row now (#149), so the block list is
    applied to the whole row, not just its top level. A top-level-only
    guard let the teacher's answer key ride out inside a carried
    ``scenario_dimensions``, ``lineage``, ``judge_meta`` or tool result."""
    row = _nested_privileged_row()
    assert _privileged_keys_in(row), "fixture must actually nest something"
    out = export_row(row)
    assert SECRET not in _dump(out)
    assert _privileged_keys_in(out) == []
    # the rest of the row still survives: the guard scrubs, it does not
    # drop the carrier
    assert out["scenario_dimensions"] == {"tool": "lookup_order"}
    assert out["lineage"] == {"source": "grade"}
    assert out["markers"] == {"grounded": 1.0}
    assert out["steps"][0]["result"] == {"status": "ok"}


def test_export_row_rebuilds_messages_from_the_scrubbed_steps():
    """``conversation()`` dumps each tool result into a message string, so
    a key scrubbed after that has already stopped being a key. The scrub
    has to happen before anything is derived from the row."""
    row = _nested_privileged_row()
    row.pop("messages", None)
    out = export_row(row)
    assert out["messages"], "the fixture should produce a conversation"
    assert SECRET not in _dump(out["messages"])


def test_saved_file_carries_no_nested_privileged_value(tmp_path):
    """The bytes on disk, which is what the customer in #149 reads back."""
    from whileai.simulations.data import SimulationData

    data = SimulationData(trajectories=[_nested_privileged_row()])
    dest = tmp_path / "rows.jsonl"
    data.save(str(dest))
    text = dest.read_text(encoding="utf-8")
    assert SECRET not in text
    for key in (*PRIVILEGED_KEYS, "privileged", "rubric"):
        assert f'"{key}"' not in text
    assert _privileged_keys_in(json.loads(text.splitlines()[0])) == []
    # and the evidence #149 is about is still there
    on_disk = json.loads(text.splitlines()[0])
    assert on_disk["markers"] == {"grounded": 1.0}
    assert on_disk["lineage"] == {"source": "grade"}


def test_training_rows_carry_no_privileged_key_or_value():
    """The SFT/GRPO export: messages plus a fixed carry list."""
    rows = training_rows([_privileged_row()], system_prompt="Be honest.", tools=[])
    assert rows
    for row in rows:
        assert SECRET not in _dump(row)
        for key in PRIVILEGED_KEYS:
            assert key not in row


def test_export_training_file_carries_no_privileged_value(tmp_path):
    """The bytes on disk, not just the in-memory rows."""
    dest = tmp_path / "train.jsonl"
    report = export_training([_privileged_row()], str(dest), system_prompt="Be honest.", tools=[])
    assert report["n"] == 1
    text = dest.read_text(encoding="utf-8")
    assert SECRET not in text
    for key in PRIVILEGED_KEYS:
        assert f'"{key}"' not in text


def test_export_preference_file_carries_no_privileged_value(tmp_path):
    """Both sides of every pair are rebuilt as messages, so both are clean.

    ``build_preference_pairs`` keeps the whole source row on ``chosen`` /
    ``rejected`` as an in-memory intermediate; the export is the boundary
    that has to drop the privileged keys, and this pins that it does.
    """
    rows = [
        _privileged_row(reward=1.0, final_text="Refunded."),
        _privileged_row(reward=0.0, final_text="I refunded the wrong one."),
    ]
    pairs, _report = build_preference_pairs(rows)
    assert pairs
    dest = tmp_path / "pref.jsonl"
    export_preference(pairs, str(dest), system_prompt="Be honest.", tools=[])
    text = dest.read_text(encoding="utf-8")
    assert SECRET not in text
    for key in PRIVILEGED_KEYS:
        assert f'"{key}"' not in text


def test_training_carry_list_names_no_privileged_field():
    """A frozen surface: the carry list is the only way a non-message key
    reaches a training row, so a privileged name must never appear in it."""
    for key in PRIVILEGED_KEYS:
        assert key not in _CARRY_KEYS


def test_eval_rewards_are_distinguishable_from_training_rewards():
    """An eval score and a training score are the same shape, so lineage
    is the only thing that tells them apart. A consumer that cannot read
    ``lineage['source']`` cannot keep the eval scorer out of the reward."""
    rows = [_privileged_row()]
    judge = lambda row: {"reward": 1.0, "reason": "ok"}  # noqa: E731
    graded = run_judge(rows, judge)
    scored = evaluate(rows, judge=judge)
    assert graded.source == "grade"
    assert scored.source == "eval"
    assert graded.rows[0]["lineage"]["source"] == "grade"
    assert scored.rows[0]["lineage"]["source"] == "eval"


# ------------------------------------------------- the verifier's own reason
#
# Every guard above builds its rows by hand. On the RLVR path the answer key
# is not decoration: a verifier *reads* ``privileged.reference`` and then
# writes a reason saying what it compared. ``reason`` is on the export carry
# list, so a reason that quotes the gold puts the answer key in the training
# file -- for exactly the rows the student got wrong.

GOLD = "20260915"


def _graded_row(final_text: str = "The answer is 7", **privileged) -> dict:
    """A row whose gold lives where the SDK says it lives, and whose answer
    is wrong -- so the gold appears nowhere the student wrote."""
    block = {"reference": GOLD}
    block.update(privileged)
    return {
        "prompt": "what is 4 + 4?",
        "final_text": final_text,
        "steps": [],
        "scenario_id": "sc-1",
        "privileged": block,
    }


@pytest.mark.parametrize("build", [MathEqual, ExactMatch, Numeric, Includes])
def test_a_verifier_reason_never_quotes_the_privileged_gold(build):
    verdict = build()(_graded_row())
    # non-vacuous: the gold was read and used, this is a real comparison
    assert verdict["reward"] in (0, 1)
    assert GOLD not in verdict["reason"]
    assert GOLD not in _dump(verdict)


def test_the_candidate_side_of_the_comparison_survives_the_redaction():
    """The reason still says what the policy produced; only the gold goes.
    A reason reduced to 'fail' would be a worse product than the leak."""
    verdict = Numeric()(_graded_row())
    assert "7.0" in verdict["reason"] and "<reference>" in verdict["reason"]


def test_a_plain_answer_column_is_still_quoted_back():
    """Deliberate scope. ``answer=`` is the caller's own column, not the
    teacher's block; quoting it tells them nothing they did not write."""
    row = {"prompt": "q", "final_text": "The answer is 7", "steps": [], "answer": GOLD}
    assert GOLD in Numeric()(row)["reason"]


def test_graded_by_a_verifier_no_export_carries_the_gold(tmp_path):
    """End to end on the path the RLVR customer actually runs: grade with a
    verifier, then write every file the SDK writes."""
    from whileai.simulations.data import SimulationData

    rows = [_graded_row(principle=f"{SECRET}_principle"), _graded_row("The answer is 12")]
    graded = run_judge([dict(r) for r in rows], MathEqual())
    assert [r["reward"] for r in graded.rows] == [0, 0]  # the gold really was read

    train = tmp_path / "train.jsonl"
    export_training(graded.rows, str(train), system_prompt="Solve it.", tools=[])
    saved = tmp_path / "rows.jsonl"
    SimulationData(trajectories=[dict(r) for r in graded.rows]).save(str(saved))
    pairs, _report = build_preference_pairs(
        [
            *graded.rows,
            {**_graded_row(f"The answer is {GOLD}"), "reward": 1.0, "judge_status": "ok"},
        ]
    )
    pref = tmp_path / "pref.jsonl"
    if pairs:
        export_preference(pairs, str(pref), system_prompt="Solve it.", tools=[])

    for path in (train, saved, pref):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert SECRET not in text
        for line in text.splitlines():
            entry = json.loads(line)
            assert _privileged_keys_in(entry) == []
            # the gold may appear only as the policy's own words, never as
            # something the SDK wrote about the policy
            assert GOLD not in _dump(
                {k: v for k, v in entry.items() if k not in ("messages", "final_text", "chosen")}
            )


def test_code_exec_does_not_echo_the_privileged_tests():
    """``privileged.tests`` is the answer key for a code task, and the
    failing assertion is echoed straight out of it."""
    tests = f"def test_it():\n    assert solve(2) == {GOLD}, 'want {GOLD}'\n"
    row = {
        "prompt": "write solve(x)",
        "final_text": "```python\ndef solve(x):\n    return 1\n```",
        "steps": [],
        "privileged": {"tests": tests},
    }
    verdict = CodeExec(timeout=20)(row)
    assert verdict["reward"] == 0
    assert GOLD not in _dump(verdict)
    # what went wrong still reaches the caller, just not what was expected
    assert verdict["reason"] == "tests failed: AssertionError"
    # the caller's own tests are not privileged: the full tail is kept, so
    # iterating on a checker you wrote yourself is unchanged
    mine = CodeExec(tests=tests, timeout=20)({**row, "privileged": {}})
    assert GOLD in mine["reason"]


def _priv(final_text: str, reference) -> dict:
    return {
        "prompt": "q",
        "final_text": final_text,
        "steps": [],
        "privileged": {"reference": reference},
    }


def test_a_short_gold_is_a_whole_token_not_a_substring():
    """Gold ``7``, answer ``17``: the candidate half must not read ``1<reference>``."""
    assert Numeric()(_priv("The answer is 17", "7"))["reason"] == "got 17.0, want <reference>"
    assert MultipleChoice()(_priv("B", "C"))["reason"] == "chose B, answer <reference>"


def test_a_gold_the_verifier_transformed_is_still_redacted():
    """``MathEqual`` names no spelling of the gold at all; ``ExactMatch``
    prints the first 60 characters; ``!r`` escapes a newline. Each is the
    gold in another spelling."""
    assert (
        MathEqual()(_priv("The answer is 3", "\\frac{1}{2}"))["reason"] == "math-verify: not equal"
    )
    long_gold = "the quick brown fox jumps over the lazy dog " * 2 + SECRET
    assert SECRET not in _dump(ExactMatch()(_priv("nope", long_gold)))
    assert "fox" not in ExactMatch()(_priv("nope", long_gold))["reason"]
    two_lines = f"first line of the key\nsecond line {SECRET}"
    assert "first line" not in _dump(ExactMatch()(_priv("nope", two_lines)))


def test_code_exec_treats_tests_in_privileged_reference_as_the_answer_key():
    tests = f"def test_it():\n    assert solve(2) == {GOLD}, 'want {GOLD}'\n"
    row = _priv("```python\ndef solve(x):\n    return 1\n```", tests)
    verdict = CodeExec(timeout=20)(row)
    assert verdict["reward"] == 0
    assert GOLD not in _dump(verdict)
    assert verdict["reason"] == "tests failed: AssertionError"


def test_rescoring_keeps_the_prior_scoring_run_id():
    """Re-judging graded rows under ``evaluate`` must not erase the run
    that produced the training reward; without the prior id a double-scored
    row is indistinguishable from a fresh one."""
    judge = lambda row: {"reward": 1.0, "reason": "ok"}  # noqa: E731
    graded = run_judge([_privileged_row()], judge)
    rescored = evaluate(graded.rows, judge=lambda row: {"reward": 0.0, "reason": "no"})
    lineage = rescored.rows[0]["lineage"]
    assert lineage["prior_scoring_run_id"] == graded.run_id
    assert lineage["scoring_run_id"] != graded.run_id
    assert lineage["source"] == "eval"
