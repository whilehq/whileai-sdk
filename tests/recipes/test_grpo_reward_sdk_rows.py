"""The GRPO recipe's reward grades the rule, not the wire format (#788).

``reward.py`` read a ``<tool_call>`` block out of the reply text. An SDK row
records its calls as structured ``steps`` and says the rest in prose, so
every row ``simulate()`` produces scored as a miss: a correct prose answer
got 0.0 where the ``<tool_call>`` spelling got 1.0. That is a reward for
format.

The second half is the sampling: ``temperature``, ``top_p`` and the seed
were literals inside ``train_modal.py``, the eval sampler was unseeded, the
train set was never decontaminated against the holdout and the base was
evaluated once, so there was no noise floor to read a delta against.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRPO = REPO / "recipes" / "04-train" / "grpo"
sys.path.insert(0, str(GRPO))

import reward
from example_helpers import load_modal_script

PROMPT = "I need a refund on ORD-1234, it arrived broken."
CASE = reward.case_for(PROMPT)
# What simulate() hands a grader: the call is a step, the reply is prose,
# and there is no <tool_call> block anywhere in the text.
SDK_ROW = {
    "prompt": PROMPT,
    "scenario_id": "s0",
    "rollout_index": 0,
    # No question mark and no "<tool_call>": on a tree that reads only the
    # text this lands on "no call, no question" and scores a flat 0.0.
    "final_text": "I pulled up ORD-1234. It shipped on the 3rd and the total was $42.00.",
    "steps": [
        {
            "tool": "lookup_order",
            "arguments": {"order_id": "ORD-1234"},
            "result": {"order_id": "ORD-1234", "status": "shipped", "total": 42.0},
        }
    ],
}


def _score(row: dict) -> float:
    """The reward on a row, through whatever surface the module offers.

    The fallbacks are deliberate: on a tree where the reward cannot see
    ``steps`` this lands on the text-only call and the assertion reports the
    wrong number, which is the finding, rather than an import error.
    """
    if hasattr(reward, "score_row"):
        return reward.score_row(row, CASE)
    try:
        return reward.score(row["final_text"], CASE, row.get("steps"))
    except TypeError:
        return reward.score(row["final_text"], CASE)


def test_an_sdk_row_with_structured_steps_scores_on_the_rule() -> None:
    got = _score(SDK_ROW)
    assert got == 1.0, (
        f"an SDK row that called lookup_order in structured steps and answered in "
        f"prose scored {got}, not 1.0: the reward is reading <tool_call> text, so "
        f"it pays for the wire format and not for the rule (#788)"
    )


def test_a_tool_call_shaped_row_still_scores_as_before() -> None:
    """The control. The text form is what the trainer samples from the
    policy, and its scores must not move under the fix."""
    no_id = reward.case_for("I want a refund please")
    off_topic = reward.case_for("what is the weather")
    # Every value here is what origin/main scores; none of them may move.
    assert reward.score(reward.scripted_reply(CASE), CASE) == 1.0
    assert reward.score(reward.scripted_reply(CASE, follow=False), CASE) == 0.2
    assert reward.score(reward.scripted_reply(no_id), no_id) == 1.0
    assert reward.score(reward.scripted_reply(no_id, follow=False), no_id) == 0.2
    assert reward.score(reward.scripted_reply(off_topic), off_topic) == 1.0
    assert reward.score(reward.scripted_reply(off_topic, follow=False), off_topic) == 0.2
    # an invented id keeps the format bonus and loses the rule; an empty
    # reply is still zero
    invented = (
        '<tool_call>\n{"name": "lookup_order", "arguments": {"order_id": "ORD-9999"}}\n</tool_call>'
    )
    assert reward.score(invented, CASE) == 0.3
    assert reward.score("", CASE) == 0.0


def test_an_sdk_row_that_invented_an_id_is_still_wrong() -> None:
    """Reading steps must not turn the rule into "called something"."""
    invented = {
        **SDK_ROW,
        "steps": [{"tool": "lookup_order", "arguments": {"order_id": "ORD-9999"}}],
    }
    assert _score(invented) == 0.3, "the same score the <tool_call> spelling of it gets"
    refund_first = {
        **SDK_ROW,
        "steps": [{"tool": "create_refund", "arguments": {"order_id": "ORD-1234", "amount": 42}}],
    }
    assert _score(refund_first) == 0.2, "refunding first keeps only the format bonus"
    no_id_case = reward.case_for("I want a refund please")
    called_anyway = {
        "prompt": "I want a refund please",
        "final_text": "Refunding that for you now.",
        "steps": [{"tool": "create_refund", "arguments": {"order_id": "ORD-1", "amount": 9}}],
    }
    # any call on a prompt that named no id runs on an invented one
    assert reward.score(called_anyway["final_text"], no_id_case, called_anyway["steps"]) == 0.2


def test_reward_rows_keeps_the_steps_the_row_carried() -> None:
    """A row went out with ``"steps": []`` whatever it did, so nothing
    downstream could tell "called nothing" from "not looked at"."""
    item = {"prompt": PROMPT, "case": CASE, "scenario_id": "s0"}
    rows = reward.reward_rows([item], [[SDK_ROW]])
    assert rows[0]["reward"] == 1
    assert rows[0]["steps"] == SDK_ROW["steps"], "the structured call was dropped on the floor"
    # and a text-shaped reply gets the steps its <tool_call> block implies
    text_rows = reward.reward_rows([item], [[reward.scripted_reply(CASE)]])
    assert text_rows[0]["steps"] == [
        {"tool": "lookup_order", "arguments": {"order_id": "ORD-1234"}}
    ]
    assert reward.reward_rows([item], [["no call here"]])[0]["steps"] == []


# --------------------------------------------------------------------------
# Sampling: named, sourced, seeded, and moved from the call.
# --------------------------------------------------------------------------

SAMPLING_KEYWORDS = ("temperature", "top_p", "seed")


def _train_modal_tree() -> ast.Module:
    return ast.parse((GRPO / "train_modal.py").read_text(encoding="utf-8"))


def test_sampling_values_are_read_from_named_defaults_not_literals() -> None:
    """CONSTITUTION.md belief 3: every default is named, sourced and tunable
    from the call. ``temperature=0.8``, ``top_p=0.95`` and ``seed=17`` were
    spelled inline in ``model.generate`` and ``GRPOConfig`` (#788).

    ``scripts/check_no_hardcoding.py`` does not reach these: it reads
    ``whileai/simulations`` only and skips recipes by design. This test is
    the gate for the recipe side.
    """
    offenders = []
    for node in ast.walk(_train_modal_tree()):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (
                kw.arg in SAMPLING_KEYWORDS
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, (int, float))
                and not isinstance(kw.value.value, bool)
            ):
                offenders.append(f"line {kw.value.lineno}: {kw.arg}={kw.value.value!r}")
    assert not offenders, (
        "a sampling value that steers what the run measures is written inline: "
        + "; ".join(offenders)
    )


def test_every_sampling_default_is_named_with_its_source() -> None:
    defaults_py = GRPO / "defaults.py"
    assert defaults_py.exists(), "the recipe's sampling defaults have no named home"
    sys.path.insert(0, str(GRPO))
    import defaults

    text = defaults_py.read_text(encoding="utf-8")
    for name in ("SAMPLE_TEMPERATURE", "SAMPLE_TOP_P", "SAMPLE_SEED", "BASE_EVAL_RUNS"):
        value = getattr(defaults, name)
        # the form check_no_hardcoding.py holds defaults.py to: NAME = value: why
        assert f"# {name} = {value}:" in text, f"{name} carries no `# NAME = value: why` comment"
        start = text.index(f"# {name} = {value}:")
        why = text[start : text.index(f"\n{name} = ", start)]
        assert "rlhfbook.com" in why or "arXiv" in why or "convention, untested" in why, (
            f"{name} names no source and does not say 'convention, untested'"
        )
    assert defaults.BASE_EVAL_RUNS >= 2, "one base run measures no noise"


def test_the_sampling_defaults_are_movable_from_the_call() -> None:
    module = load_modal_script("grpo_train_modal_sampling", GRPO / "train_modal.py")
    import defaults

    entry = inspect.signature(module.main).parameters
    for flag, value in (
        ("temperature", defaults.SAMPLE_TEMPERATURE),
        ("top_p", defaults.SAMPLE_TOP_P),
        ("sample_seed", defaults.SAMPLE_SEED),
        ("base_runs", defaults.BASE_EVAL_RUNS),
    ):
        assert flag in entry, f"--{flag.replace('_', '-')} is not on the entrypoint"
        assert entry[flag].default == value, f"--{flag} does not default to the named constant"
    sampler = inspect.signature(module._sample).parameters
    for flag in SAMPLING_KEYWORDS:
        assert flag in sampler, f"the eval sampler takes no {flag}: it cannot be reproduced"
    assert sampler["seed"].default == defaults.SAMPLE_SEED


def test_the_sampler_is_seeded() -> None:
    source = inspect.getsource(load_modal_script("grpo_tm_seed", GRPO / "train_modal.py")._sample)
    assert "set_seed(seed)" in source, (
        "the eval sampler draws from whatever torch's global state happens to be, "
        "so pass@1 before and after cannot be reproduced (belief 1)"
    )


def test_the_train_set_is_decontaminated_against_the_holdout() -> None:
    source = (GRPO / "train_modal.py").read_text(encoding="utf-8")
    assert "decontaminate(" in source, (
        "no decontaminate call: a held-out prompt that is also trained on measures "
        "memory, not the change (Lambert 2025, chapter Evaluation)"
    )


def test_the_base_is_evaluated_more_than_once_and_sets_a_noise_floor() -> None:
    module = load_modal_script("grpo_tm_base", GRPO / "train_modal.py")
    source = (GRPO / "train_modal.py").read_text(encoding="utf-8")
    assert "eval_variance(" in source, "no noise floor: a delta is read against zero"
    assert "run_std=run_std" in source, "the noise floor never reaches delta_report"
    assert module.BASE_EVAL_RUNS >= 2


def test_round_two_has_a_from_run_and_uses_next_round() -> None:
    module = load_modal_script("grpo_tm_round", GRPO / "train_modal.py")
    assert "from_run" in inspect.signature(module.main).parameters, (
        "--from-run is missing, so the README still sends a second round to DPO (#788)"
    )
    source = (GRPO / "train_modal.py").read_text(encoding="utf-8")
    assert "next_round(" in source, (
        "next_round is documented on docs/api/score.mdx and called by no recipe; "
        "round two keeps paying for groups with no contrast"
    )
