"""Small customer-visible gaps from the customer simulation ledger (#31).

Each of these cost a first-hour customer minutes in the sixth pass:
a warning that did not name its key, a valid empty schema read as a
missing one, ``read_runbook`` called destructive, ``recommend`` rejecting
the spelling ``simulate`` accepts, ``wai status`` printing ``null``
for "no key", an SFT export that never says it carried failures, a total
length confound that only warned above eight pairs, and ``tasks=``
quietly re-running at k=1.
"""

from __future__ import annotations

import json

import pytest

import whileai.simulations as wai
from tests.helpers import simulate_offline
from whileai import auth, cli
from whileai.simulations.export import export_preference, export_training
from whileai.simulations.score.judging import build_preference_pairs
from whileai.simulations.score.preflight import preflight

TOOLS = [
    {
        "name": "read_runbook",
        "description": "Read the runbook for a service.",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
        "returns": {"steps": ["check status"]},
    },
    {
        "name": "restart_service",
        "description": "Restart a service.",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
    },
    {
        "name": "ping",
        "description": "Liveness check, takes no arguments.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "returns": {"ok": True},
    },
]
POLICY = (
    "You are an ops agent.\n1. read_runbook before any restart_service.\n"
    "2. Never claim a restart you did not do.\n3. ping is free; use it when unsure."
)


# ------------------------------------------------------------- preflight


def test_preflight_names_the_missing_key():
    report = preflight(TOOLS, POLICY)
    assert report["missing_result_shapes"] == ["restart_service"]
    (note,) = [w for w in report["warnings"] if "result shape" in w]
    assert "`returns`" in note


def test_preflight_accepts_a_declared_no_argument_tool():
    report = preflight(TOOLS, POLICY)
    ping = next(t for t in report["tools"] if t["name"] == "ping")
    assert ping["issues"] == []
    bare = preflight([{"name": "x", "description": "no schema at all"}], POLICY)
    assert "no_parameters_schema" in bare["tools"][0]["issues"]


def test_preflight_destructive_is_a_word_not_a_substring():
    report = preflight(TOOLS, POLICY)
    assert "read_runbook" not in report["destructive_tools"]
    words = preflight(
        [
            {"name": "cancel_order", "description": "d", "parameters": {"properties": {}}},
            {"name": "cancelOrder", "description": "d", "parameters": {"properties": {}}},
            {"name": "file_ticket", "description": "d", "parameters": {"properties": {}}},
            {"name": "get_profile", "description": "d", "parameters": {"properties": {}}},
            {"name": "bookmark_page", "description": "d", "parameters": {"properties": {}}},
        ],
        POLICY,
    )
    assert set(words["destructive_tools"]) == {"cancel_order", "cancelOrder", "file_ticket"}


# ------------------------------------------------------------- recommend


def test_recommend_takes_simulate_spelling_of_the_policy():
    by_policy = wai.recommend(TOOLS, POLICY, mode="rl")
    by_system_prompt = wai.recommend(TOOLS, system_prompt=POLICY, mode="rl")
    assert by_policy["budget"] == by_system_prompt["budget"]
    with pytest.raises(ValueError, match="not both"):
        wai.recommend(TOOLS, POLICY, system_prompt=POLICY + " x")


# ------------------------------------------------------------- status


def test_status_says_when_no_key_is_configured(monkeypatch, tmp_path, capsys):
    """The keyless path is the supported path (belief 4), so `wai status`
    with no key is a note on stdout and nothing on stderr. It used to print
    `no API key configured` to stderr, which reads as a failure in a library
    that needs no account (#790)."""
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.setattr(auth, "credentials_path", lambda: tmp_path / "credentials.json")
    assert cli.main(["status"]) == 0
    captured = capsys.readouterr()
    body, note = captured.out.rsplit("\n}\n", 1)
    shown = json.loads(body + "\n}")
    assert shown["configured"] is False and shown["key"] is None
    assert captured.err == ""
    assert "no API key: the library runs without one" in note
    assert "wai login" in note and "simulator=False" in note


# ------------------------------------------------------------- exports


def _row(prompt, reward, final):
    return {
        "prompt": prompt,
        "reward": reward,
        "judge_status": "ok",
        "final_text": final,
        "steps": [],
        "model_version": "student",
    }


def test_export_training_counts_the_failures_it_writes(tmp_path):
    rows = [
        _row("a", 1, "Checked the runbook, then restarted."),
        _row("b", 0, "Restarted it."),
        _row("c", 0, "Restarted it, all good."),
        {**_row("d", None, "Hmm."), "reward": None},
    ]
    report = export_training(rows, str(tmp_path / "sft.jsonl"), system_prompt=POLICY)
    assert report["rewards"] == {"n_pass": 1, "n_fail": 2, "n_ungraded": 1}
    (note,) = report["warnings"]
    assert "2 of 4" in note and "passes()" in note
    clean = export_training(rows[:1], str(tmp_path / "ok.jsonl"), system_prompt=POLICY)
    assert clean["rewards"] == {"n_pass": 1, "n_fail": 0, "n_ungraded": 0}
    assert "warnings" not in clean


def test_total_length_confound_warns_below_eight_pairs(tmp_path):
    rows = []
    for i in range(3):
        rows.append(_row(f"ask{i}", 1, "A long, careful, complete answer with detail. " * 3))
        rows.append(_row(f"ask{i}", 0, "no"))
    pairs, report = build_preference_pairs(rows)
    assert len(pairs) == 3 and report["length"]["chosen_longer_frac"] == 1.0
    assert any("longer reply in 3/3" in w for w in report["warnings"])
    exported = export_preference(pairs, str(tmp_path / "pairs.jsonl"))
    assert exported["chosen_longer_frac"] == 1.0
    assert any("longer reply in 3/3" in w for w in exported["warnings"])
    # two pairs is not evidence of anything
    pairs2, report2 = build_preference_pairs(rows[:4])
    assert len(pairs2) == 2 and not any("longer" in w for w in report2["warnings"])


# ------------------------------------------------------------- tasks= k


def _agent(prompt):
    return {"final_text": "Read the runbook first.", "steps": []}


def test_pinned_rerun_inherits_the_base_k_unless_told_otherwise():
    kw = dict(tools=TOOLS, policy=POLICY, mode="sft", budget=24, per_round=24, concurrency=1)
    base = simulate_offline(_agent, repeats=4, **kw)
    assert base.rollouts_per_request == 4
    inherited = simulate_offline(_agent, tools=TOOLS, policy=POLICY, tasks=base, concurrency=1)
    assert inherited.rollouts_per_request == 4
    assert len(inherited.trajectories) == len(base.trajectories)
    explicit = simulate_offline(
        _agent, tools=TOOLS, policy=POLICY, tasks=base, repeats=1, concurrency=1
    )
    assert explicit.rollouts_per_request == 1
    assert len(explicit.trajectories) == len({r["prompt"] for r in base.trajectories})
