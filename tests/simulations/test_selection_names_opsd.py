"""The tasks a grouped method drops are what OPSD is for, and the report says so.

A prompt the policy never passes is where GRPO has no gradient and rejection
sampling yields nothing at any k. Before #886 the report dropped those tasks
silently and named only the hosted trainer's four methods, so a user was told
to buy more rollouts for prompts no number of rollouts can reach.
"""

from whileai.simulations.training import selection_report


def _profile(mixed: int, all_fail: int, *, graded: int = 4) -> dict:
    per_task = [{"graded": graded, "pass_rate": 0.5}] * mixed
    per_task += [{"graded": graded, "pass_rate": 0.0}] * all_fail
    tasks = mixed + all_fail
    return {
        "rows": tasks * graded,
        "tasks": tasks,
        "tasks_with_repeats": tasks,
        "mixed_tasks": mixed,
        "per_task": per_task,
        "split": {"fail": all_fail * graded, "ungraded": 0},
    }


def test_under_floor_with_all_fail_tasks_names_opsd():
    said = " ".join(selection_report(_profile(5, 35), method="grpo", dataset="hard")["warn"])
    assert "wai.OPSD" in said
    assert "35 tasks the policy never passes" in said


def test_nothing_mixed_refuses_and_still_names_opsd():
    said = " ".join(selection_report(_profile(0, 40), method="grpo", dataset="hard")["refuse"])
    assert said, "a set with no mixed task is a refusal"
    assert "wai.OPSD" in said


def test_dpo_and_rm_get_the_same_route():
    for method in ("dpo", "rm"):
        rep = selection_report(_profile(4, 30), method=method, dataset="hard")
        assert "wai.OPSD" in " ".join(rep["warn"]), method


def test_silent_when_every_dropped_task_is_all_pass():
    # Nothing to distil towards: the policy already passes these, so naming a
    # teacher would be noise. This is the line that must NOT fire.
    per_task = [{"graded": 4, "pass_rate": 0.5}] * 5 + [{"graded": 4, "pass_rate": 1.0}] * 35
    rep = selection_report(
        {
            "rows": 160,
            "tasks": 40,
            "tasks_with_repeats": 40,
            "mixed_tasks": 5,
            "per_task": per_task,
            "split": {"fail": 0, "ungraded": 0},
        },
        method="grpo",
        dataset="easy",
    )
    assert "wai.OPSD" not in " ".join(rep["warn"] + rep["refuse"])


def test_silent_when_the_per_task_table_is_incomplete():
    # all_fail is unknown without the table, and the line quotes the count, so
    # a guess would be worse than silence.
    rep = selection_report(
        {
            "rows": 160,
            "tasks": 40,
            "tasks_with_repeats": 40,
            "mixed_tasks": 5,
            "per_task": [],
            "split": {"fail": 140, "ungraded": 0},
        },
        method="grpo",
        dataset="partial",
    )
    assert "wai.OPSD" not in " ".join(rep["warn"] + rep["refuse"])


def test_sft_is_untouched():
    rep = selection_report(_profile(5, 35), method="sft", dataset="hard")
    assert "wai.OPSD" not in " ".join(rep["refuse"] + rep["warn"])


def test_the_line_carries_its_citations_and_the_arithmetic():
    said = " ".join(selection_report(_profile(5, 35), method="grpo", dataset="hard")["warn"])
    for cite in ("arXiv:2601.19897", "arXiv:2601.18734", "arXiv:2607.05184"):
        assert cite in said, cite
    assert "1-(1-p)**k" in said
