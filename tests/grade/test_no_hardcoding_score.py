"""Every number a verdict in ``score/`` rests on has one home
(``whileai.simulations.defaults``) and one knob. These tests turn each
knob and check the verdict moves with it; all of them fail on a tree
where the value is a bare literal."""

from __future__ import annotations

import json
import math
import random
import re
from statistics import NormalDist

import pytest

import whileai.simulations as wai
from whileai.simulations import defaults, environment
from whileai.simulations.score import (
    agreement,
    audit,
    grade_llm,
    hack_scan,
    hygiene,
    judge_trust,
    llm_judge,
    optimize,
    pairwise,
    passat,
    preflight,
    quality,
    stats,
)
from whileai.simulations.score.delta import delta_report, format_delta_report
from whileai.simulations.score.stats import (
    _T975,
    _t_quantile,
    bootstrap_ci,
    compare_runs,
    detectable_effect,
    holdout_size,
    noise_band,
)

# ------------------------------------------------------------------ one home


def test_the_shared_numbers_have_one_home():
    """A value read in two modules is the same object in both."""
    assert defaults.CI_LEVEL == 1 - defaults.ALPHA
    assert abs(defaults.Z_95 - NormalDist().inv_cdf((1 + defaults.CI_LEVEL) / 2)) < 1e-3
    assert stats.DEFAULT_BOOT is defaults.BOOTSTRAP_DRAWS
    assert stats.MIN_CI_TASKS is defaults.MIN_CI_TASKS
    assert optimize.DEFAULT_BAND is defaults.DIFFICULTY_BAND
    assert environment.DEFAULT_BAND is defaults.DIFFICULTY_BAND
    assert quality.FAIL == defaults.PASS_THRESHOLD
    assert hygiene.HACK_THRESHOLD is defaults.HACK_THRESHOLD
    assert agreement.MIN_GOLD is defaults.MIN_GOLD
    assert judge_trust.MIN_AGREEMENT is defaults.MIN_AGREEMENT
    assert judge_trust.FLIP_FLAG is defaults.FLIP_FLAG
    assert pairwise.PAIRWISE_MAX_TOKENS == grade_llm.JUDGE_MAX_TOKENS == defaults.JUDGE_MAX_TOKENS
    assert grade_llm._PAYLOAD_CHARS == defaults.JUDGE_PAYLOAD_CHARS
    assert hack_scan.RESCAN_ROLLOUTS == defaults.RL_ROLLOUTS_PER_ASK
    # the sizing functions default to the shared values, not their own copies
    need = holdout_size(0.05)
    assert need["base"] == defaults.BASE_PASS_RATE and need["k"] == defaults.ROLLOUTS_PER_TASK
    assert need["power"] == defaults.POWER and need["alpha"] == defaults.ALPHA
    assert passat.pass_at([]).k == 1
    assert wai.recommend([], "", mode="rl")["rollouts_per_request"] == defaults.RL_ROLLOUTS_PER_ASK


def test_every_default_says_why():
    """Each constant carries a why, not just a restatement of its value:
    ``# NAME = value: <at least three words>`` or ``(convention``. A
    ``# NAME = value`` line with nothing after the value fails."""
    import inspect

    paragraphs: list[str] = []
    block: list[str] = []
    for line in inspect.getsource(defaults).splitlines():
        if line.startswith("#"):
            block.append(line.lstrip("#").strip())
        elif block:
            paragraphs.append(" ".join(block))
            block = []
    if block:
        paragraphs.append(" ".join(block))
    for name in defaults.__all__:
        if not name.isupper():
            continue  # RunKnobs and the knob helpers are not constants
        # its own line, or named with its value inside a shared comment
        # ("# A = 1 / B = 2: ...")
        hits = [p for p in paragraphs if re.search(rf"(?<![A-Z_]){name} = ", p)]
        assert hits, f"{name}: no `# {name} = value` comment"
        tail = re.split(rf"(?<![A-Z_]){name} = ", hits[0], maxsplit=1)[1]
        why = re.search(r":\s*((?:\S+\s+){2,}\S+)", tail)
        assert why or "(convention" in tail, f"{name}: the comment restates the value, no why"
        # the constitution's exact words, so one grep finds every unsourced number
        assert "(convention" not in tail or "(convention, untested" in tail, (
            f"{name}: says convention without the exact words '(convention, untested'"
        )


def test_convention_phrase_check_reads_wrapped_comments(tmp_path):
    """``scripts/check_no_hardcoding.py`` holds defaults.py to the exact
    phrase: a bare ``(convention)`` or a qualified opener is a finding, the
    phrase wrapped over two comment lines is not."""
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "check_no_hardcoding.py"
    spec = importlib.util.spec_from_file_location("check_no_hardcoding", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    src = tmp_path / "defaults.py"
    src.write_text(
        "# A = 1: one reason. (convention)\n"
        "A = 1\n"
        "# B = 2: another reason. (convention inside the band, untested)\n"
        "B = 2\n"
        "# C = 3: a third reason that wraps. (convention,\n"
        "# untested against other values)\n"
        "C = 3\n"
        "# D = 4: sourced (rlhfbook.com/c/07-reasoning)\n"
        "D = 4\n",
        encoding="utf-8",
    )
    findings = mod.check_convention_phrase(src)
    assert [f.line for f in findings] == [1, 3], findings
    assert mod.check_convention_phrase(Path(defaults.__file__)) == []


# ------------------------------------------------------------------ level / alpha / power


def test_noise_band_takes_a_level():
    assert noise_band(0.01, level=0.99) > noise_band(0.01) > noise_band(0.01, level=0.8)
    assert noise_band(0.01) == pytest.approx(defaults.Z_95 * 0.01 * math.sqrt(2))
    # t path: three runs a side is df=4; at 90% the two-sided t quantile is 2.132
    assert noise_band(0.01, 3, 3, df=4, level=0.9) == pytest.approx(
        2.132 * 0.01 * math.sqrt(2 / 3), rel=1e-3
    )
    with pytest.raises(ValueError, match="level"):
        noise_band(0.01, level=1.0)


def test_t_quantile_inversion_matches_the_table_and_known_values():
    for df, table in _T975.items():
        # a level a hair off 0.95 takes the numeric path, not the table
        assert _t_quantile(df, 0.95 + 1e-12) == pytest.approx(table, abs=5e-4)
    assert _t_quantile(1, 0.99) == pytest.approx(63.657, rel=1e-3)
    assert _t_quantile(4, 0.99) == pytest.approx(4.604, rel=1e-3)
    assert _t_quantile(10, 0.90) == pytest.approx(1.812, rel=1e-3)


def test_bootstrap_and_compare_runs_take_a_level():
    values = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 0.5, 0.25]
    narrow = bootstrap_ci(values, level=0.5, seed=1)
    wide = bootstrap_ci(values, level=0.99, seed=1)
    assert narrow and wide and wide[1] - wide[0] > narrow[1] - narrow[0]
    rng = random.Random(0)
    a = [
        {"prompt": f"t{i}", "reward": int(rng.random() < 0.4)} for i in range(40) for _ in range(4)
    ]
    b = [
        {"prompt": f"t{i}", "reward": int(rng.random() < 0.6)} for i in range(40) for _ in range(4)
    ]
    loose = compare_runs(a, b, level=0.5)
    strict = compare_runs(a, b, level=0.999)
    assert loose["level"] == 0.5 and strict["level"] == 0.999
    assert strict["ci95"][1] - strict["ci95"][0] > loose["ci95"][1] - loose["ci95"][0]
    # the unpaired fallback uses the same level
    unpaired = compare_runs(a[:8], b[-8:], level=0.5, min_paired=100)
    assert unpaired["paired"] is False and unpaired["level"] == 0.5


def test_delta_report_alpha_power_and_ceiling_are_knobs():
    def rows(p, n=30, k=4):
        return [
            {"prompt": f"t{i}", "reward": 1 if j < round(p * k) else 0, "markers": {"h": 1.0}}
            for i in range(n)
            for j in range(k)
        ]

    before, after = rows(0.5), rows(0.5)
    at_half = delta_report(before, after, alpha=0.5)
    assert at_half["alpha"] == 0.5 and at_half["level"] == 0.5
    assert at_half["family_error"] == pytest.approx(1 - 0.5 ** at_half["n_metrics"], abs=1e-4)
    assert at_half["metrics"]["pass_at_1"]["level"] == 0.5
    default = delta_report(before, after)
    assert default["family_error"] == pytest.approx(
        1 - defaults.CI_LEVEL ** default["n_metrics"], abs=1e-4
    )
    assert "at 50%" in format_delta_report(at_half)
    # power feeds the sizing line
    sized = delta_report(rows(0.5), rows(0.5), power=0.9)
    assert any("at 90% power" in w for w in sized["warnings"])
    assert sized["detectable_effect"] > default["detectable_effect"]
    # the ceiling is a knob
    assert delta_report(rows(0.6), rows(0.6))["ceiling"] is False
    assert delta_report(rows(0.6), rows(0.6), ceiling_pass_rate=0.5)["ceiling"] is True
    with pytest.raises(ValueError, match="alpha and power"):
        delta_report(before, after, alpha=1.5)


def test_holdout_size_reads_its_defaults_from_one_place():
    """The sizing defaults are the shared constants: change the constant,
    change the answer, with no literal left behind."""
    base = holdout_size(0.05)["n_tasks"]
    assert detectable_effect(base) <= 0.05 < detectable_effect(base - 5)
    assert holdout_size(0.05, alpha=0.2)["n_tasks"] < base
    assert holdout_size(0.05, power=0.95)["n_tasks"] > base


# ------------------------------------------------------------------ judge knobs


def _pair(prompt, chosen_text, rejected_text):
    def row(final, reward):
        return {
            "prompt": prompt,
            "final_text": final,
            "reward": reward,
            "steps": [],
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": final},
            ],
        }

    return {"prompt": prompt, "chosen": row(chosen_text, 1), "rejected": row(rejected_text, 0)}


def test_pairwise_flags_are_knobs():
    def first_position(a, b):
        return '{"winner": "A", "reason": "the first one"}'

    def prefers_rejected(a, b):
        return {"winner": "B" if "shipped" in a["final_text"] else "A", "reason": "short"}

    pairs = [_pair(f"p{i}", "Order shipped.", "No idea.") for i in range(4)]
    _, flagged = pairwise.judge_pairs(pairs, first_position, concurrency=1)
    assert any("position bias" in w for w in flagged["warnings"])
    _, quiet = pairwise.judge_pairs(pairs, first_position, concurrency=1, position_flip_flag=1.1)
    assert quiet["position_flip_rate"] == 1.0 and not quiet["warnings"]
    _, flagged = pairwise.judge_pairs(pairs, prefers_rejected, concurrency=1)
    assert any("prefers the rejected side" in w for w in flagged["warnings"])
    _, quiet = pairwise.judge_pairs(
        pairs, prefers_rejected, concurrency=1, prefers_rejected_flag=1.1
    )
    assert quiet["prefers_rejected"] == 4 and not quiet["warnings"]


def test_pairwise_judge_reply_budget_and_request_cap(monkeypatch):
    seen = {}

    def fake_complete(_url, _model, messages, **kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        seen["request"] = json.loads(messages[1]["content"])["request"]
        return {"content": '{"winner": "A", "reason": "x"}'}

    monkeypatch.setattr("whileai.simulations.score.pairwise.complete", fake_complete)
    judge = pairwise.pairwise_judge("openai:x", api_key="k", max_tokens=33, request_chars=50)
    pair = _pair("q" * 500, "Order shipped.", "No idea.")
    judge(pair["chosen"], pair["rejected"])
    assert seen["max_tokens"] == 33 and len(seen["request"]) == 50
    judge = pairwise.pairwise_judge("openai:x", api_key="k")
    judge(pair["chosen"], pair["rejected"])
    assert seen["max_tokens"] == defaults.JUDGE_MAX_TOKENS
    assert len(seen["request"]) == min(500, defaults.JUDGE_SITUATION_CHARS)


def test_grade_llm_payload_and_reply_budget_are_knobs(monkeypatch):
    seen = {}

    def fake_complete(_url, _model, messages, **kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        seen["user"] = messages[-1]["content"]
        return {"content": '{"reason": "fine", "score": 1}'}

    monkeypatch.setattr("whileai.simulations.score.grade_llm.complete", fake_complete)
    row = {
        "prompt": "p",
        "final_text": "f",
        "steps": [
            {"tool": "t", "arguments": {"a": "x" * 900}, "result": {"r": "y" * 900}}
            for _ in range(40)
        ],
    }
    grade_llm.grade_one(row, backend_spec="openai:x", api_key="k", payload_chars=1500, max_tokens=7)
    assert seen["max_tokens"] == 7 and len(seen["user"]) <= 1500
    assert json.loads(seen["user"])["payload_reduced"] is True
    grade_llm.grade_one(row, backend_spec="openai:x", api_key="k")
    assert seen["max_tokens"] == defaults.JUDGE_MAX_TOKENS
    assert 1500 < len(seen["user"]) <= defaults.JUDGE_PAYLOAD_CHARS
    # the whole batch takes the same knobs and records them on every row
    report = grade_llm.apply_grade_llm(
        [dict(row)],
        backend_spec="openai:x",
        api_key="k",
        payload_chars=1500,
        max_tokens=7,
        trust="off",
    )
    assert report["graded"] == 1 and seen["max_tokens"] == 7 and len(seen["user"]) <= 1500
    # the audit pass too
    grade_llm.audit_one(
        {**row, "reward": 1}, backend_spec="openai:x", api_key="k", payload_chars=1500, max_tokens=9
    )
    assert seen["max_tokens"] == 9 and len(seen["user"]) <= 1500


def test_llm_judge_payload_and_reply_budget_are_knobs(monkeypatch):
    seen = {}

    def fake_complete(_url, _model, messages, **kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        seen["temperature"] = kwargs.get("temperature")
        seen["user"] = messages[-1]["content"]
        return {"content": '{"score": 1, "reason": "ok"}'}

    monkeypatch.setattr("whileai.simulations.score.llm_judge.complete", fake_complete)
    row = {"prompt": "p" * 3000, "final_text": "f" * 1500, "steps": []}
    llm_judge.judge_one(row, api_key="k", payload_chars=1000, max_tokens=5)
    assert seen["max_tokens"] == 5 and len(seen["user"]) <= 1000
    llm_judge.judge_one(row, api_key="k")
    assert seen["max_tokens"] == defaults.JUDGE_MAX_TOKENS
    assert seen["temperature"] == defaults.JUDGE_TEMPERATURE
    assert 1000 < len(seen["user"]) <= defaults.JUDGE_PAYLOAD_CHARS


def _labeled_rows(n=24):
    rows = []
    for i in range(n):
        long = i % 2 == 1
        rows.append(
            {
                "prompt": f"t{i}",
                "scenario_id": f"t{i}",
                "final_text": ("Detailed answer. " * 6) if long else "Short reply.",
                "reward": int(long),
                "gold_reward": 1 if i % 3 else 0,
                "gold_kind": "human",
                "steps": [],
            }
        )
    return rows


def test_judge_trust_length_and_flip_flags_are_knobs():
    def length_judge(row):
        return {"score": 1 if len(str(row.get("final_text") or "")) > 60 else 0, "reason": "len"}

    rows = _labeled_rows()
    flagged = judge_trust.judge_trust(rows, length_judge, sample=24, concurrency=1)
    assert flagged["length_sensitivity"]["flagged"] is True
    assert flagged["perturbation"]["flagged_length"] is True
    quiet = judge_trust.judge_trust(
        rows, length_judge, sample=24, concurrency=1, length_gap_flag=1.1, flip_flag=1.1
    )
    assert quiet["length_sensitivity"]["flagged"] is False
    assert quiet["perturbation"]["flagged_length"] is False
    assert not any("length" in w for w in quiet["warnings"])
    assert judge_trust.perturbation(rows, length_judge, flip_flag=1.1)["flagged_length"] is False
    probes = judge_trust.judge_probes(rows, length_judge, probes=["filler"], flip_flag=1.1)
    assert probes["probes"]["filler"]["flagged"] is False and probes["exploitable_by"] == []


def test_audit_warning_threshold_is_a_knob():
    from tests.grade.test_audit_grades import _numeric_judge, _rows

    loud = audit.audit_grades(_rows(), judge=_numeric_judge, sample=100, seed=0)
    assert any(w.startswith("VERIFIER:") for w in loud["warnings"])
    quiet = audit.audit_grades(_rows(), judge=_numeric_judge, sample=100, seed=0, fn_warn=0.9)
    assert quiet["fn_rate"] == loud["fn_rate"] == 0.4
    assert not any(w.startswith("VERIFIER:") for w in quiet["warnings"])


# ------------------------------------------------------------------ hack scan / selection / report


def test_hack_scan_floor_follows_alpha():
    from tests.grade.test_hack_scan import _pool

    rows = _pool(30, 8, seed=2, reward_of=lambda rng, tool, delim, p: int(rng.random() < 0.5))
    strict = hack_scan.hack_scan(rows, seed=0, alpha=0.01)
    loose = hack_scan.hack_scan(rows, seed=0, alpha=0.5)
    assert strict["tau"] > loose["tau"] > 0
    assert strict["n_above_floor"] <= loose["n_above_floor"]


def test_dataset_report_hard_share_floor_is_a_knob():
    from tests.api.test_dataset_report_tiers import _row

    rows = [_row("ordinary", prompt=f"o{i}") for i in range(16)] + [
        _row("boundary", prompt=f"b{i}") for i in range(4)
    ]
    assert preflight.dataset_report(rows)["warnings"]
    assert preflight.dataset_report(rows, hard_share_floor=0.1)["warnings"] == []


def test_scored_data_select_for_rl_uses_the_shared_band():
    rows = []
    for t in range(6):
        for j in range(4):
            rows.append({"prompt": f"t{t}", "reward": int(j < t % 5), "judge_status": "ok"})
    scored = wai.ScoredData(rows, run_id="r", source="test", judge_name="j")
    _, report = scored.select_for_rl()
    assert report["band"] == list(defaults.DIFFICULTY_BAND) == [0.2, 0.8]
