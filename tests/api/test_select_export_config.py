"""Issue #592: ``select().export()`` writes the system prompt and tools
whichever container the rows arrived in.

A row stores the conversation without the agent's system prompt or tool
schemas; the run knows them. ``scored.select(mode="sft").export(path)``
wrote both, while ``wai.decontaminate(...)`` handed back a plain list and
``wai.select(kept, mode="sft").export(path)`` wrote neither, reporting
``with_tools: 0`` without a word. Lesson 5 ends on ``decontaminate`` and
lesson 6 opens with ``select``, so the documented path shipped tool-calling
SFT files whose prompts never showed the tools.
"""

from __future__ import annotations

import json
import warnings

import pytest

import whileai as wai
from tests.helpers import POLICY, TOOLS
from whileai.simulations.data import RowList

KEYS = ("n", "with_system", "with_tools")


@pytest.fixture(scope="module")
def scored():
    data = wai.simulate(
        wai.seeded_agent(TOOLS),
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        mode="rl",
        repeats=4,
        repeat_policy="fixed",
        budget=32,
        seed=0,
    )
    return data.grade(judge=lambda row: {"reward": int(not row["seeded"])})


def _counts(report: dict) -> dict:
    return {k: report[k] for k in KEYS}


def test_the_three_paths_write_the_same_prompt_and_tools(scored, tmp_path):
    a = scored.select(mode="sft").export(str(tmp_path / "a.jsonl"))
    kept, report = wai.decontaminate(scored.rows, against=scored.rows[:0])
    assert isinstance(kept, list) and isinstance(report, dict)  # the documented shape
    b = wai.select(kept, mode="sft").export(str(tmp_path / "b.jsonl"))
    c = wai.select(scored, mode="sft").export(str(tmp_path / "c.jsonl"))
    assert a["n"] > 0
    assert _counts(a) == _counts(b) == _counts(c)
    assert a["with_system"] == a["n"] and a["with_tools"] == a["n"]
    for name in ("a", "b", "c"):
        assert "warnings" not in json.loads(
            (tmp_path / f"{name}.jsonl").read_text().splitlines()[0]
        )


def test_the_lesson_five_slice_and_concat_keeps_the_config(scored, tmp_path):
    train = scored.rows[:20]
    holdout = scored.rows[20:]
    assert isinstance(train, RowList) and train.system_prompt == POLICY and train.tools == TOOLS
    kept, report = wai.decontaminate(train + holdout[:1], against=holdout)
    assert report["n_contaminated"] == 1 and len(kept) == len(train)
    out = wai.select(kept, mode="sft").export(str(tmp_path / "kept.jsonl"))
    assert out["with_system"] == out["n"] and out["with_tools"] == out["n"]
    rows = scored.rows
    assert (rows + list(holdout)).tools == TOOLS and (list(holdout) + rows).tools == TOOLS


def test_row_views_of_a_graded_run_keep_the_config(scored):
    for view in (scored.passes(), scored.failures(), scored.unjudged(), scored.rows.copy()):
        assert isinstance(view, RowList)
        assert view.system_prompt == POLICY and view.tools == TOOLS
    assert wai.select(scored.passes(), mode="sft").tools == TOOLS


def test_a_plain_list_warns_and_the_overrides_put_the_config_back(scored, tmp_path):
    plain = list(scored.rows)
    assert not hasattr(plain, "tools")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bare = wai.select(plain, mode="sft").export(str(tmp_path / "bare.jsonl"))
    assert bare["with_system"] == 0 and bare["with_tools"] == 0
    said = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("no tool schema" in s and "tools=" in s for s in said)
    assert any("system prompt" in s and "system_prompt=" in s for s in said)
    assert set(said) <= set(bare["warnings"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fixed = wai.select(plain, mode="sft").export(
            str(tmp_path / "fixed.jsonl"), system_prompt=POLICY, tools=TOOLS
        )
    assert fixed["with_system"] == fixed["n"] and fixed["with_tools"] == fixed["n"]
    first = json.loads((tmp_path / "fixed.jsonl").read_text().splitlines()[0])
    assert first["messages"][0] == {"role": "system", "content": POLICY}
    assert first["tools"] == TOOLS


def test_rows_that_never_had_tools_or_a_prompt_export_quietly(tmp_path):
    rows = [
        {
            "prompt": "hi",
            "final_text": "hello",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "reward": 1,
            "judge_status": "ok",
        }
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = wai.select(rows, mode="sft").export(str(tmp_path / "chat.jsonl"))
    assert report["with_tools"] == 0 and report["with_system"] == 0
    assert "warnings" not in report
