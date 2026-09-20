"""tracked.brief(): what happened, what it means, what to do next, from the rows."""

from __future__ import annotations

import logging

from whileai.platform import Behavior, Dashboard, Judge, brief_of, track

RUN = {
    "id": "run_1",
    "agent": "a",
    "version": "executor-baseline",
    "method": "eval",
    "createdAt": "2026-09-20T02:11:40Z",
    "updatedAt": "2026-09-20T02:11:41Z",
    "evals": [
        {"behavior": "grounded_answer", "version": "executor-baseline", "score": 1, "ci": 0, "n": 4}
    ],
}


def test_first_run_on_a_set_nobody_can_fail_reads_as_a_sentence():
    """Jacob's first run: 4 asks, every one passed, posted as a fraction."""
    b = Behavior(name="grounded_answer")
    brief = brief_of("a", [b], [RUN])
    assert brief.scored == 1
    assert brief.happened[0] == "1 scored run, no training yet, latest 2026-09-20."
    assert brief.happened[1] == "grounded_answer: executor-baseline passed all 4 asks."
    assert "fractions" in brief.happened[2]
    assert brief.means.startswith("A perfect score on 4 asks says the asks are too easy")
    # A set nobody can fail comes first; then size; then the name. Three at most.
    assert [s.say.split(" (")[0] for s in brief.next] == [
        "add harder tasks",
        "add 46+ tasks",
        "name the test set",
    ]
    assert "hard_share" in brief.next[0].cmd
    text = str(brief)
    assert "what happened" in text and "do next" in text
    assert text.endswith("https://withwhile.com/platform/runs?agent=a")
    md = brief.markdown()
    assert md.startswith("# a on https://withwhile.com/platform/runs?agent=a")
    assert '`Behavior(name, test_version="v1")' in md


def test_one_version_in_points_says_nothing_to_compare_yet():
    run = {
        **RUN,
        "version": "v1",
        "evals": [{"behavior": "refunds", "version": "v1", "score": 62.5, "ci": 4.1, "n": 120}],
    }
    b = Behavior(
        name="refunds",
        test_version="t1",
        n=120,
        judge=Judge(agreement=0.9, human_n=80),
        noise_floor=1.2,
        contamination=0,
        reward_is_judge=False,
    )
    brief = brief_of("a", [b], [run])
    assert brief.happened[1] == "refunds: v1 scored 62.5 of 100 on 120 asks (±4.1)."
    assert len(brief.happened) == 2  # no fraction line
    assert brief.means.startswith("One version scored once.")
    assert [s.say for s in brief.next] == ["score the next version on the same set"]


def test_verdict_leads_when_two_versions_are_scored():
    runs = [
        {
            **RUN,
            "id": "r1",
            "version": "v1",
            "evals": [{"behavior": "refunds", "version": "v1", "score": 60, "ci": 2, "n": 200}],
        },
        {
            **RUN,
            "id": "r2",
            "version": "v2",
            "evals": [{"behavior": "refunds", "version": "v2", "score": 66, "ci": 2, "n": 200}],
        },
    ]
    b = Behavior(
        name="refunds",
        test_version="t1",
        n=200,
        judge=Judge(agreement=0.9, human_n=80),
        noise_floor=1.0,
        contamination=0,
        reward_is_judge=False,
    )
    dash = Dashboard.model_validate(
        {
            "agent": {"id": "a", "name": "a", "serving": "v1"},
            "verdict": {
                "candidate": "v2",
                "serving": "v1",
                "delta": 6,
                "excludesZero": True,
                "behavior": "refunds",
            },
        }
    )
    brief = brief_of("a", [b], runs, dash)
    assert brief.happened[1] == "refunds: 2 versions scored, v2 highest at 66, v1 lowest at 60."
    assert brief.means == "v2 is better than v1 by 6 points, outside the noise."
    assert brief.next[0].say == "promote v2" and brief.next[0].cmd == 'tracked.promote("v2")'


def test_no_scores_yet_says_so():
    brief = brief_of("a", [Behavior(name="refunds")], [{**RUN, "evals": []}])
    assert brief.scored == 0
    assert brief.happened == ["1 run posted, none scored yet."]
    assert brief.next[0].say == "score one version"


def test_archived_runs_stay_out_of_the_brief():
    brief = brief_of("a", [Behavior(name="grounded_answer")], [{**RUN, "archived": True}])
    assert brief.scored == 0


def test_score_warns_when_it_reads_as_a_fraction(caplog):
    from tests.api.test_platform import Fake

    fake = Fake()
    run = track("a", transport=fake).run("v1", flush_every=100)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.score("grounded_answer", 1.0, ci=0.0, n=4)
    assert "reads as a fraction" in caplog.text and "score * 100" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.score("other", 83.0, ci=2.0, n=240)
    assert "reads as a fraction" not in caplog.text


def test_finish_prints_the_brief_only_after_a_score(capsys):
    from tests.api.test_platform import Fake

    fake = Fake()
    tracked = track("a", transport=fake)
    tracked.run("v0", flush_every=100).finish()
    assert capsys.readouterr().out == ""  # nothing scored, nothing said

    run = tracked.run("v1", flush_every=100)
    run.score("refunds", 83.0, ci=2.0, n=240)
    run.finish()
    out = capsys.readouterr().out
    # The fake answers GET /runs with {ok: true}, so the brief sees no rows: it still speaks.
    assert "a: what happened" in out and "do next" in out

    run = tracked.run("v2", flush_every=100)
    run.score("refunds", 84.0, ci=2.0, n=240)
    run.finish(say=False)
    assert capsys.readouterr().out == ""
