"""Three messages testers lost time to, and what they say now.

1. A trial key dies mid-run with a quota error that named neither the
   offline writer nor the sign-in that lifts the limit.
2. Return shapes were guessed: ``pass_hat_k`` for ``pass_pow_k``, ``ci``
   for ``ci95``, and a marker interval of ``None`` with no reason.
3. ``judge_trust`` said "no rows carry both 'reward' and a gold label",
   which sends a reader to the labels when the rewards are missing, and
   gold set by hand read as a model's with no way to say otherwise.
"""

from __future__ import annotations

from whileai import auth
from whileai.simulations.generate.agents import QUOTA_FIX, _quota_error
from whileai.simulations.score.agreement import (
    MODEL_GOLD_REASON,
    UNKNOWN_GOLD_REASON,
    judge_agreement,
    missing_side_note,
)
from whileai.simulations.score.judge_trust import format_judge_trust, judge_trust
from whileai.simulations.score.passat import pass_at
from whileai.simulations.score.stats import MIN_CI_TASKS, marker_summary

# ------------------------------------------------------------ 1. trial quota


def test_the_trial_note_says_how_far_a_day_goes_and_names_the_offline_writer():
    note = auth.trial_note(25_000)
    assert "about 12 hosted situations a day" in note
    assert "simulator=False" in note and "no quota" in note
    assert auth.SIGN_IN_URL in note
    # the count follows the allowance the gate reports
    assert "about 25 hosted situations" in auth.trial_note(50_000)
    # no allowance in the reply falls back to the documented trial
    assert auth.trial_note(None) == auth.trial_note(auth.DEFAULT_TRIAL_INPUT_TOKENS)
    assert auth.trial_situations(1) == 1  # never zero situations


def test_signup_prints_the_trial_note_under_the_trial_line(monkeypatch, tmp_path):
    lines: list[str] = []
    monkeypatch.setattr(auth, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        auth,
        "_post",
        lambda path, body, timeout=30: (
            201,
            {
                "api_key": "zp_test",
                "email": body["email"],
                "tier": "trial",
                "trial": {"daily_input_tokens": 25_000, "daily_output_tokens": 50_000},
            },
        ),
    )
    auth.signup("you@example.com", out=lines.append)
    printed = "\n".join(lines)
    assert "25,000 input / 50,000 output tokens a day" in printed
    assert auth.trial_note(25_000) in printed


def test_status_carries_the_trial_note_only_on_a_trial_key(monkeypatch, tmp_path):
    (tmp_path / "credentials.json").write_text('{"api_key": "zp_test_key_1234"}', encoding="utf-8")
    monkeypatch.setattr(auth, "config_dir", lambda: tmp_path)
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)

    monkeypatch.setattr(
        auth, "account", lambda key=None: {"tier": "trial", "trial": {"lift": "sign in"}}
    )
    assert auth.status()["trial_note"] == auth.trial_note(None)

    monkeypatch.setattr(auth, "account", lambda key=None: {"tier": "team"})
    assert "trial_note" not in auth.status()


def test_the_quota_error_names_the_offline_writer_and_the_sign_in():
    body = '{"error": {"message": "daily quota exceeded (25000 input tokens); resets at midnight UTC"}}'
    message = _quota_error(429, body)
    assert message is not None
    assert message.startswith("Hosted model daily quota exceeded (25000 input tokens)")
    assert "simulator=False" in message and auth.SIGN_IN_URL in message
    assert message.endswith(QUOTA_FIX)
    assert "UTC. simulate" in message  # one period joins the two, never two
    # not every 429 is a quota, and nothing else gets the sentence
    assert _quota_error(429, "rate limit, slow down") is None
    assert _quota_error(500, body) is None


# ------------------------------------------------------------ 2. return shapes


def _graded(n_tasks: int, k: int, passes) -> list[dict]:
    return [
        {
            "scenario_id": f"s{t}",
            "prompt": f"ask {t}",
            "rollout_index": i,
            "final_text": "done",
            "steps": [],
            "reward": passes(t, i),
            "markers": {"grounded": float(passes(t, i))},
        }
        for t in range(n_tasks)
        for i in range(k)
    ]


def test_the_printed_pass_pow_k_carries_its_attribute_name():
    rates = pass_at(_graded(4, 4, lambda t, i: 1 if (t + i) % 2 else 0))
    printed = str(rates)
    assert "pass^4 (pass_pow_k)" in printed
    assert printed.count("pass_pow_k") == 1
    assert not hasattr(rates, "pass_hat_k")  # the name testers guessed
    # the docstring is the field table: every attribute, with its printed name
    doc = type(rates).__doc__ or ""
    for field in rates.to_dict():
        assert f"``{field}``" in doc, field
    assert "``pass_pow_k``" in doc and "``ci95``" in doc


def test_a_marker_with_too_few_tasks_says_why_there_is_no_interval():
    two_tasks = marker_summary(_graded(2, 1, lambda t, i: t % 2))
    note = two_tasks["grounded"]["note"]
    assert two_tasks["grounded"]["ci95"] is None
    assert "2 task(s) carry grounded" in note
    assert f"needs {MIN_CI_TASKS} or more" in note
    assert "Add tasks" in note

    # enough tasks: an interval and no note
    enough = marker_summary(_graded(6, 1, lambda t, i: t % 2))["grounded"]
    assert enough["ci95"] is not None and "note" not in enough

    # a marker that never varies is the other case, and keeps its own warning
    flat = marker_summary(_graded(6, 1, lambda t, i: 1))["grounded"]
    assert flat["ci95"] is None and "note" not in flat
    assert "has not been shown" in flat["warning"]


def test_the_marker_keys_are_ci95_and_n_rows():
    stats = marker_summary(_graded(6, 2, lambda t, i: t % 2))["grounded"]
    assert set(stats) >= {"mean", "ci95", "n_tasks", "n_rows", "n_rows_at_1", "n_rows_at_0"}
    assert "ci" not in stats and "n" not in stats  # the keys testers guessed


# ------------------------------------------------------------ 3. judge_trust


def _rows(
    n: int, *, reward: bool = True, gold: bool = False, kind: str | None = None
) -> list[dict]:
    rows = []
    for i in range(n):
        row: dict = {"prompt": f"ask {i}", "scenario_id": f"s{i}", "final_text": "done"}
        if reward:
            row["reward"] = i % 2
        if gold:
            row["gold_reward"] = i % 2
            if kind:
                row["gold_kind"] = kind
        rows.append(row)
    return rows


def test_unscored_rows_are_told_to_score_first_not_to_label():
    report = judge_trust(_rows(6, reward=False, gold=True, kind="human"))
    (note,) = [w for w in report["warnings"] if "unmeasured" in w]
    assert "no row has a reward" in note
    assert "run_judge" in note and "evaluate" in note
    assert "judge=" in note and "does not score the rows" in note
    assert "attach_labels" not in note  # the half that is not missing
    assert report["ok"] is False
    assert format_judge_trust(report).startswith("NOT MEASURED")


def test_unlabeled_rows_are_pointed_at_attach_labels():
    report = judge_trust(_rows(6))
    (note,) = [w for w in report["warnings"] if "unmeasured" in w]
    assert "no row has a gold label" in note
    assert "attach_labels(rows, labels, kind='human')" in note
    assert "gold_reward" in note and "0/1" in note
    assert "run_judge" not in note  # the rows are scored already


def test_rewards_and_labels_on_different_rows_says_so():
    rows = _rows(4) + _rows(4, reward=False, gold=True, kind="human")
    (note,) = [w for w in judge_trust(rows)["warnings"] if "unmeasured" in w]
    assert "but no row has both" in note
    assert "attach_labels matches on rollout_id" in note


def test_gold_set_by_hand_is_told_how_to_mark_it_human():
    by_hand = judge_trust(_rows(60, gold=True))
    assert by_hand["gold_kind"] == "unknown" and by_hand["ok"] is False
    (reason,) = [w for w in by_hand["warnings"] if w == UNKNOWN_GOLD_REASON]
    assert "no record of who wrote it" in reason
    assert "attach_labels(rows, labels, kind='human')" in reason
    assert "came from a model" not in reason  # it may well be a person's
    assert format_judge_trust(by_hand).startswith("NOT MEASURED")

    # a model's labels still say so, and still name the same fix
    model = judge_trust(_rows(60, gold=True, kind="model"))
    (reason,) = [w for w in model["warnings"] if w == MODEL_GOLD_REASON]
    assert "came from a model" in reason
    assert "attach_labels(rows, labels, kind='human')" in reason
    assert format_judge_trust(model).startswith("NOT MEASURED")

    # a person's labels: measured, and no provenance warning at all
    human = judge_trust(_rows(60, gold=True, kind="human"))
    assert human["gold_kind"] == "human"
    assert not [w for w in human["warnings"] if "attach_labels(rows, labels" in w]


def test_missing_side_note_covers_each_half_on_its_own():
    assert "no row has a reward and no row has a gold label" in missing_side_note(0, 0)
    assert missing_side_note(0, 5).startswith("no row has a reward")
    assert missing_side_note(5, 0).startswith("no row has a gold label")
    assert "no row has both" in missing_side_note(5, 5)
    # the agreement report on its own says the same thing
    (note,) = judge_agreement(_rows(4, reward=False, gold=True, kind="human"))["warnings"]
    assert note == missing_side_note(0, 4)
