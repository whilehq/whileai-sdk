"""What a cold start on a stranger's repo broke in HarnessSweep (2026-09-20):
a label over the run-version cap raised after seven minutes of rollouts,
hand labels could not attach to the sweep's fresh rollouts, the rollout
concurrency had no knob, the report called a winner the platform then
called unproven for size, and the offline writer could not see ids like
``12B``."""

from __future__ import annotations

import pytest

from tests.api.test_platform import Fake
from tests.api.test_sweep import POLICY, TOOLS, judge
from whileai.platform import Harness, HarnessSweep, Judge, RunSpec, SweepReport, track
from whileai.simulations.generate.scenarios import known_ids
from whileai.sweep import LABEL_MAX, MIN_ASKS, VariantResult, check_labels


def _never_called(message: str) -> dict:
    raise AssertionError("a rollout ran before the labels were checked")


def _harness(label: str) -> Harness:
    return Harness(label=label, instructions=POLICY, tools=TOOLS, model="claude-haiku-4-5")


def test_label_cap_is_the_wire_cap():
    assert LABEL_MAX == RunSpec.model_fields["version"].metadata[-1].max_length == 40


def test_long_label_is_refused_before_any_rollout():
    sweep = HarnessSweep(track("bot", transport=Fake()), judge=judge, tools=TOOLS)
    long = "resident-support-2026.09.2@claude-haiku-4-5"
    assert len(long) > LABEL_MAX
    with pytest.raises(ValueError, match="43 characters; a run version holds 40"):
        sweep.run({long: (_harness(long), _never_called)}, tasks=[])


def test_duplicate_labels_are_refused_before_any_rollout():
    with pytest.raises(ValueError, match="two variants are labelled 'policy@haiku'"):
        check_labels([_harness("policy@haiku"), _harness("policy@haiku")])


def test_labels_may_be_a_judge_measured_on_the_frozen_run():
    measured = Judge(name="policy as a program", agreement=0.93, human_n=60)
    sweep = HarnessSweep(track("bot", transport=Fake()), judge=judge, tools=TOOLS, labels=measured)
    assert sweep._judge([]) is measured


def test_concurrency_is_passed_to_the_writer(monkeypatch):
    seen: dict = {}

    def fake_simulate(agent, **kw):
        seen.update(kw)

        class Data:
            def rows(self):
                return []

        return Data()

    import whileai.simulations as wai

    monkeypatch.setattr(wai, "simulate", fake_simulate)
    sweep = HarnessSweep(track("bot", transport=Fake()), judge=judge, tools=TOOLS, concurrency=6)
    sweep._rollouts(lambda m: {"steps": [], "final_text": ""}, tasks=[])
    assert seen["concurrency"] == 6


def test_report_says_when_the_test_is_under_the_size_gate():
    small = SweepReport(
        behavior="resident_policy",
        test_version="t-f6e6a867",
        n_asks=48,
        k=4,
        noise_floor=2.6,
        judge=None,
        variants=[
            VariantResult(
                "2026.09.2@claude-sonnet-5",
                "claude-sonnet-5",
                "0422fe19374e",
                {"resident_policy": (99.5, 0.8, 48)},
            ),
            VariantResult(
                "2026.09.2@claude-haiku-4-5",
                "claude-haiku-4-5",
                "e445667671f6",
                {"resident_policy": (85.4, 7.3, 48)},
            ),
        ],
    )
    text = str(small)
    assert "winner: 2026.09.2@claude-sonnet-5" in text
    assert f"48 asks is under {MIN_ASKS}: the platform verdict says unproven" in text
    assert MIN_ASKS == 50


def test_known_ids_reads_unit_style_ids():
    tools = [
        {
            "name": "get_lease",
            "description": "The lease on file for a unit. Units on file: 12B, 4A, 7C, 9F.",
            "input_schema": {"type": "object", "properties": {"unit": {"type": "string"}}},
        },
        {
            "name": "lookup_order",
            "description": "Orders on file: A1001, ORD-4017. Up to 3 per day.",
        },
    ]
    assert known_ids(tools) == ["12B", "4A", "7C", "9F", "A1001", "ORD-4017"]
