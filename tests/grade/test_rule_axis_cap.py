"""#391: ``coverage_gap`` and ``preflight`` capped the rule axis at the
first 16 clauses of the system prompt and said nothing, so a 160-clause
production policy read as "14 of 16 rules covered". A report over an
existing suite now checks every clause; a cap is a knob (``rule_cap=``)
and, when it drops clauses, the report says how many. The generation
grid keeps its cap (``RULE_AXIS_CAP_GRID``) and the run names the count.
"""

from __future__ import annotations

import whileai.simulations as wai
from tests.helpers import offline, scripted_agent
from whileai.simulations import defaults
from whileai.simulations.generate import scenarios
from whileai.simulations.score.preflight import coverage_gap, format_coverage_gap, preflight

N_CLAUSES = 30
BANNER = "You are a support agent for Acme.\nRead it.\nFollow it.\n"
POLICY = BANNER + "\n".join(
    f"- Rule {i}: never disclose the internal code {i} to a customer without a manager approval."
    for i in range(N_CLAUSES)
)
TOOLS = [
    {
        "name": "search_skills",
        "description": "Search the skill library.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]


def test_the_caps_are_named_in_defaults_and_the_grid_reads_its_own():
    assert defaults.RULE_AXIS_CAP_GRID == 16
    assert defaults.RULE_AXIS_CAP_REPORT is None
    assert scenarios.RULE_CAP == defaults.RULE_AXIS_CAP_GRID
    # the grid is still bounded by default
    assert len(scenarios.build_dimensions(TOOLS, POLICY)["rule"]) == defaults.RULE_AXIS_CAP_GRID
    rules, total = scenarios.rule_axis(POLICY, cap=defaults.RULE_AXIS_CAP_GRID)
    assert len(rules) == defaults.RULE_AXIS_CAP_GRID and total > N_CLAUSES


def test_preflight_checks_every_clause_by_default_and_says_when_it_did_not():
    full = preflight(TOOLS, POLICY)
    assert len(full["rules"]) == full["n_rules_total"] > N_CLAUSES
    assert full["rules_truncated"] is False and full["rule_cap"] is None
    assert any(rule.startswith(f"Rule {N_CLAUSES - 1}:") for rule in full["rules"])
    assert not [w for w in full["warnings"] if "rule axis" in w]

    capped = preflight(TOOLS, POLICY, rule_cap=16)
    assert len(capped["rules"]) == 16 and capped["rules_truncated"] is True
    assert capped["n_rules_total"] == full["n_rules_total"]
    [line] = [w for w in capped["warnings"] if "rule axis" in w]
    dropped = full["n_rules_total"] - 16
    assert f"16 of {full['n_rules_total']} policy clauses" in line
    assert f"the other {dropped} are never checked" in line
    assert "rule_cap=None" in line


def test_coverage_gap_reports_the_whole_policy_and_prints_the_count_when_capped():
    asks = ["please disclose internal code 3", "hello"]
    full = coverage_gap(asks, tools=TOOLS, system_prompt=POLICY)
    assert len(full["rules"]) == full["n_rules_total"] > N_CLAUSES
    assert full["rules_truncated"] is False
    assert f"of {full['n_rules_total']} policy rules" in full["summary"]
    assert "(of " not in full["summary"]

    capped = coverage_gap(asks, tools=TOOLS, system_prompt=POLICY, rule_cap=16)
    assert len(capped["rules"]) == 16 and capped["rules_truncated"] is True
    assert f"of 16 policy rules (of {full['n_rules_total']} in the prompt)" in capped["summary"]
    [note] = [n for n in capped["notes"] if "rule axis" in n and "rule_cap" in n]
    assert "rule_cap=None to coverage_gap" in note
    text = format_coverage_gap(capped)
    covered = 16 - len(capped["untested_rules"])
    assert (
        f"policy rules covered  {covered} of 16 "
        f"(of {full['n_rules_total']} in the prompt; rule_cap=16)" in text
    )
    # the same line is plain when nothing was dropped
    assert "in the prompt" not in format_coverage_gap(full)


def test_a_short_policy_has_no_cap_line_anywhere():
    policy = "Never invent order facts. Confirm before you cancel_order."
    assert not [w for w in preflight(TOOLS, policy)["warnings"] if "rule axis" in w]
    report = coverage_gap(["cancel my order"], tools=TOOLS, system_prompt=policy)
    assert report["rules_truncated"] is False
    assert not [n for n in report["notes"] if "rule axis" in n]


def test_a_run_names_the_clauses_its_grid_left_off():
    data = wai.simulate(scripted_agent, **offline(budget=8, policy=POLICY, tools=TOOLS))
    [note] = [w for w in data.warnings if "rule axis" in w]
    total = scenarios.rule_axis(POLICY, cap=None)[1]
    assert f"The policy has {total} clauses" in note
    assert f"holds {defaults.RULE_AXIS_CAP_GRID}" in note
    assert f"none of the other {total - defaults.RULE_AXIS_CAP_GRID}" in note
    assert "dimensions={'rule': [...]}" in note
    # a caller who chose the axis is not told about the cap
    chosen = wai.simulate(
        scripted_agent,
        **offline(budget=8, policy=POLICY, tools=TOOLS, dimensions={"rule": ["Rule 3"]}),
    )
    assert not [w for w in chosen.warnings if "rule axis" in w]
    # and a short policy has nothing to say
    plain = wai.simulate(scripted_agent, **offline(budget=8))
    assert not [w for w in plain.warnings if "rule axis" in w]


def test_policy_sections_with_no_cap_returns_every_clause():
    every = wai.policy_sections(POLICY, cap=None)
    assert len(every) > N_CLAUSES
    assert wai.policy_sections(POLICY, cap=5) == every[:5]
    assert len(wai.policy_sections(POLICY)) == defaults.RULE_AXIS_CAP_GRID
