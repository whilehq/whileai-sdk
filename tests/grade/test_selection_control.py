"""The random-selection control has to be able to fail, and to be reachable.

Lambert 2025, chapter Rejection Sampling, closes on: "always run a
random-selection control alongside RM-selected training; if RM selection
does not beat random, the reward signal is not useful on that data."
``select_for_sft`` is that control. On a binary reward at the default
``min_reward=1.0`` every eligible row scores 1.0, the ranking key falls
through to the sha256 tiebreak, and ``top_per_prompt`` and
``random_per_prompt`` are the same draw -- so the control returns a null it
could not have failed to return, while the report said a selection happened
(#747). These tests pin both halves: that the collapse is named, and that a
pool the reward *can* rank is still reported as a selection, so the gate is
neither silent nor always-on.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import whileai as wai
from whileai.simulations.score.optimize import optimize, select_for_sft


def _binary_pool(prompts: int = 40, k: int = 12) -> list[dict]:
    """The issue's reproducer: 40 prompts, k=12, a 0/1 verifier."""
    return [
        {
            "prompt": f"q{p}",
            "final_text": f"answer {p}.{i}",
            "reward": float(i % 2),
            "scenario_id": f"t{p}",
        }
        for p in range(prompts)
        for i in range(k)
    ]


def _graded_pool(prompts: int = 40, k: int = 12) -> list[dict]:
    """The same shape from a grader with partial credit: 0.5 to 1.05, so
    there is a ranking among the rows that clear ``min_reward=0.5``."""
    return [
        {
            "prompt": f"q{p}",
            "final_text": f"answer {p}.{i}",
            "reward": round(0.5 + 0.05 * i, 4),
            "scenario_id": f"t{p}",
        }
        for p in range(prompts)
        for i in range(k)
    ]


def _mean(rows) -> float:
    return round(sum(float(r["reward"]) for r in rows) / len(rows), 4)


def test_binary_reward_collapses_the_control_and_the_report_says_so():
    """#747, the reported reproducer. k=12 clears the completions-per-prompt
    guard, so on main this printed ``top_per_prompt`` / ``random_per_prompt``
    and no note for two arms that are the same draw."""
    rows = _binary_pool()
    top, top_report = select_for_sft(rows, select="top_per_prompt", seed=0)
    rnd, rnd_report = select_for_sft(rows, select="random_per_prompt", seed=0)

    # The collapse itself: same rows, same prompts, same mean, by construction.
    assert len(top) == len(rnd) == 40
    assert {r["prompt"] for r in top} == {r["prompt"] for r in rnd}
    assert _mean(top) == _mean(rnd) == 1.0
    assert top_report["completions_per_prompt_median"] == 12  # the old guard stays quiet

    for report in (top_report, rnd_report):
        assert report["reward_spread_passing"] == 0.0
        # The report no longer claims a selection ran.
        assert report["selection_effective"] == "pass_filter"
        assert report["selection_effective"] not in ("top_per_prompt", "random_per_prompt")
        # And it says why, and names the control that still measures something.
        note = report["note"]
        assert "no spread" in note and "no ranking to select on" in note
        assert 'select="random_k_overall", min_reward=0.0' in note
        assert "chapter Rejection Sampling" in note
        # The rule the caller asked for is still recorded next to it.
        assert report["selection"] in ("top_per_prompt", "random_per_prompt")


def test_pass_filter_control_still_measures_the_filter():
    """What the collapsed arms were supposed to measure: the pass/fail
    filter against a random draw at the same count (#747's third row)."""
    rows = _binary_pool()
    top, _ = select_for_sft(rows, select="top_per_prompt", seed=0)
    control, report = select_for_sft(
        rows, select="random_k_overall", min_reward=0.0, target=len(top), seed=0
    )
    assert len(control) == len(top)
    assert _mean(control) < _mean(top)  # the filter is worth this much
    # A pool the rule can rank keeps its own name: the gate is not always-on.
    assert report["selection_effective"] == "random_k_overall"
    assert report["reward_spread_passing"] == 1.0
    assert "note" not in report


def test_graded_spread_is_still_reported_as_a_selection():
    """The negative. A grader with partial credit, admitted by a
    ``min_reward`` below its maximum, gives the rule something to rank: the
    report names the rule that ran and stays silent. A gate that fired here
    too would be a gate nobody reads."""
    rows = _graded_pool()
    top, top_report = select_for_sft(rows, select="top_per_prompt", min_reward=0.5, seed=0)
    rnd, rnd_report = select_for_sft(rows, select="random_per_prompt", min_reward=0.5, seed=0)

    assert _mean(top) > _mean(rnd)  # the ranking did real work
    for report, rule in ((top_report, "top_per_prompt"), (rnd_report, "random_per_prompt")):
        assert report["selection_effective"] == rule
        assert report["reward_spread_passing"] > 0
        assert "note" not in report


def test_the_control_is_reachable_from_the_public_select():
    """#747, second half. The control chapter 9 calls mandatory has to be
    one call from ``wai.select``, not an import of a private tuple-returning
    helper, and its seed has to move."""
    rows = _binary_pool()

    top = wai.select(rows, mode="sft", rule="top_per_prompt")
    rnd = wai.select(rows, mode="sft", rule=wai.Rejection("random_per_prompt", seed=1))
    control = wai.select(
        rows,
        mode="sft",
        target=len(top),
        rule=wai.Rejection("random_k_overall", min_reward=0.0, seed=1),
    )

    assert isinstance(top, wai.Selection) and isinstance(control, wai.Selection)
    assert top.report["selection_effective"] == "pass_filter"
    assert rnd.report["selection_effective"] == "pass_filter"
    assert control.report["selection_effective"] == "random_k_overall"
    assert _mean(control) < _mean(top)
    # The report prints the effective operation, so a card reader sees it
    # without opening the dict.
    assert "what ran: pass_filter" in str(top)

    # A seeded control is a distribution, not one draw: the seed moves it.
    draw_a = wai.select(rows, mode="sft", rule=wai.Rejection("random_per_prompt", seed=1))
    draw_b = wai.select(rows, mode="sft", rule=wai.Rejection("random_per_prompt", seed=2))
    assert [r["final_text"] for r in draw_a] != [r["final_text"] for r in draw_b]
    # optimize() forwards it too; on main it dropped seed= and pinned 0.
    rows_a, _ = optimize(rows, mode="sft", select="random_per_prompt", seed=1)
    rows_b, _ = optimize(rows, mode="sft", select="random_per_prompt", seed=2)
    assert [r["final_text"] for r in rows_a] != [r["final_text"] for r in rows_b]

    with pytest.raises(ValueError, match="rule must be one of"):
        wai.select(rows, mode="sft", rule="top_per_promt")


def test_select_stays_inside_the_eight_parameter_cap():
    """``docs/reference/style.md`` rule 3. ``rule`` carries three knobs on
    one parameter for this reason; the ratchet reads
    ``whileai.simulations.__all__`` and would not have caught a tenth."""
    import inspect

    params = [
        p
        for p in inspect.signature(wai.select).parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert len(params) <= 8, [p.name for p in params]


def test_empty_selection_export_raises_and_writes_no_file(tmp_path: Path):
    """#789. An empty ``train.jsonl`` beside ``{'n': 0, 'n_written': 0}`` and
    no warning reads as a successful export."""
    path = tmp_path / "train.jsonl"
    with pytest.raises(ValueError, match="nothing to export"):
        wai.Selection([], system_prompt="be helpful", tools=[]).export(str(path))
    assert not path.exists()  # not a silent 0-byte file

    with pytest.raises(ValueError) as caught:
        wai.Selection([], mode="sft").export(str(path))
    message = str(caught.value)
    assert "0 rows" in message and "min_reward=" in message and "band=" in message


def test_export_still_writes_when_there_are_rows(tmp_path: Path):
    """The other side of the same gate: one row exports as before."""
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "prompt": "q",
            "final_text": "a",
            "reward": 1.0,
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        }
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = wai.Selection(rows, system_prompt="be helpful").export(str(path))
    assert report["n_written"] == 1
    assert path.exists() and path.stat().st_size > 0
