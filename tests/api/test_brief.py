"""tracked.brief(): what happened, what it means, what to do next, from the rows."""

from __future__ import annotations

import logging

from whileai.platform import Behavior, Dashboard, Judge, VersionScore, brief_of, eval_checks, track

RUN = {
    "id": "run_1",
    "agent": "a",
    "version": "executor-baseline",
    "method": "eval",
    "createdAt": "2026-09-20T02:11:40Z",
    "updatedAt": "2026-09-20T02:11:41Z",
    "evals": [
        {
            "behavior": "grounded_answer",
            "version": "executor-baseline",
            "score": 100,
            "ci": 0,
            "n": 4,
        }
    ],
}


def test_first_run_on_a_set_nobody_can_fail_reads_as_a_sentence():
    """Jacob's first run: 4 asks, every one passed, posted in points."""
    b = Behavior(name="grounded_answer")
    brief = brief_of("a", [b], [RUN])
    assert brief.scored == 1
    assert brief.happened[0] == "1 scored run, no training yet, latest 2026-09-20."
    assert brief.happened[1] == "grounded_answer: executor-baseline passed all 4 asks."
    assert len(brief.happened) == 2
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
    assert text.endswith("https://while.ai/platform/runs?agent=a")
    md = brief.markdown()
    assert md.startswith("# a on https://while.ai/platform/runs?agent=a")
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
        judge=Judge(agreement=0.9, human_n=80, verified=True),
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
        judge=Judge(agreement=0.9, human_n=80, verified=True),
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
        run.score("grounded_answer", 0.75, ci=0.09, n=40)
    assert "reads as a fraction" in caplog.text
    assert "fraction=True" in caplog.text and "score * 100" in caplog.text
    # The number goes up as posted: the client guesses no scale.
    assert fake.calls[-1][2][0]["score"] == 0.75
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.score("other", 83.0, ci=2.0, n=240)
    assert "reads as a fraction" not in caplog.text


def test_zero_and_one_point_scores_do_not_warn(caplog):
    """A safety behavior the agent almost never passes scores 0 or 1 point;
    those are points, not fractions, and say nothing."""
    from tests.api.test_platform import Fake

    fake = Fake()
    run = track("a", transport=fake).run("v1", flush_every=100)
    with caplog.at_level(logging.WARNING, logger="whileai.platform"):
        run.score("never_refuses", 0, ci=0.6, n=200)
        run.score("rarely_refuses", 1.0, ci=0.8, n=200)
        run.score("names_the_assumption", 0.0, ci=0.0, n=46)  # the #754 repro: a measured floor
    assert "fraction" not in caplog.text
    assert [c[2][0]["score"] for c in fake.calls if c[1].endswith("/evals")] == [0.0, 1.0, 0.0]


def test_fraction_true_converts_a_rate_to_points_before_posting():
    from tests.api.test_platform import Fake

    fake = Fake()
    run = track("a", transport=fake).run("v1", flush_every=100)
    posted = run.score("refunds", 0.71, ci=0.04, n=240, fraction=True)
    body = fake.calls[-1][2][0]
    assert (body["score"], body["ci"], body["n"]) == (71.0, 4.0, 240)
    assert (posted.score, posted.ci) == (71.0, 4.0)


def test_low_points_stay_points_on_the_brief_and_the_evals_table():
    """Two versions at 1 and 0.5 points (a behavior the agent almost never
    passes) read as 1 and 0.5, not as 100 and 50."""
    from tests.api.test_verdict_per_behavior import SEED0, Platform

    def run(run_id, version, at, score, ci):
        return {
            "id": run_id,
            "agent": "a",
            "version": version,
            "method": "eval",
            "createdAt": at,
            "evals": [{"behavior": SEED0, "version": version, "score": score, "ci": ci, "n": 200}],
        }

    runs = [
        run("r1", "v1", "2026-09-20T01:00:00Z", 1.0, 0.8),
        run("r2", "v2", "2026-09-20T02:00:00Z", 0.5, 0.6),
    ]
    brief = brief_of("a", [Behavior(name=SEED0, n=200)], runs)
    assert brief.happened[1] == f"{SEED0}: 2 versions scored, v1 highest at 1, v2 lowest at 0.5."
    assert not any("fraction" in line for line in brief.happened)
    health = track("a", transport=Platform(runs, serving="v1")).evals()
    canfail = next(c for c in health[0].checks if c.key == "canfail")
    assert canfail.ok is True and "100" not in canfail.value and "50" not in canfail.value


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


def test_eval_checks_mirror_the_runs_page_table():
    """Jacob's behavior after the re-report: named, n=4, all passed in points."""
    b = Behavior(name="grounded_answer", test_version="seeds-v1", n=4)
    h = eval_checks(b, [VersionScore(v="executor-baseline", score=100, ci=0, n=4)])
    assert [c.key for c in h.checks] == [
        "frozen",
        "size",
        "judge",
        "length",
        "noise",
        "contamination",
        "reward",
        "canfail",
    ]
    assert h.total == 7 and h.good == 1  # length is not measurable, not counted
    assert h.verdict == "weak · size, judge, noise floor, clean, reward ≠ judge, can fail"
    assert [c.key for c in h.failed] == [
        "size",
        "judge",
        "noise",
        "contamination",
        "reward",
        "canfail",
    ]
    size = h.checks[1]
    assert size.value == "n=4 · resolves ≥ 0 pts" and size.action == "add 46+ tasks"
    text = str(h)
    assert text.startswith("grounded_answer: weak")
    assert "ok frozen: seeds-v1" in text and "-- length" in text
    assert "no can fail: executor-baseline already 100 -> add harder tasks" in text


def test_eval_checks_all_good():
    b = Behavior(
        name="refunds",
        test_version="t1",
        n=240,
        judge=Judge(agreement=0.86, human_n=60, length_bias=0.08, verified=True),
        noise_floor=2.4,
        contamination=0,
        reward_is_judge=False,
    )
    h = eval_checks(
        b,
        [
            VersionScore(v="base", score=61, ci=2.7, n=240),
            VersionScore(v="v4", score=83, ci=2.7, n=240),
        ],
    )
    assert h.verdict == "good" and h.good == h.total == 8
    noise = next(c for c in h.checks if c.key == "noise")
    assert noise.value == "±2.4 · 1 of 1 clear it"
    canfail = next(c for c in h.checks if c.key == "canfail")
    assert canfail.value == "base 61 → best 83"


def test_tracked_evals_and_delete_use_the_api():
    from tests.api.test_platform import Fake

    fake = Fake()
    t = track("a", transport=fake)
    t.behavior("refunds", test_version="t1", n=240)
    assert [h.name for h in t.evals()] == []  # the fake answers GET /behaviors with {ok: true}
    assert t.delete() == {"ok": True}
    assert fake.calls[-1][:2] == ("DELETE", "/agents/a")


def test_brief_lists_what_a_person_cannot_read_yet():
    """Settings for a name, no rubric, no rows, no word on what changed: four
    lines with the call that posts each. Same rules as the platform page."""
    runs = [
        {
            "id": "run_base",
            "agent": "a",
            "version": "base",
            "method": "none",
            "createdAt": "2026-09-20T01:00:00Z",
            "evals": [{"behavior": "math500", "version": "base", "score": 51, "ci": 4, "n": 160}],
        },
        {
            "id": "run_2",
            "agent": "a",
            "version": "dapo-lr5e-05-s17-180st",
            "method": "eval",
            "createdAt": "2026-09-20T02:00:00Z",
            "evals": [
                {
                    "behavior": "math500",
                    "version": "dapo-lr5e-05-s17-180st",
                    "score": 80,
                    "ci": 5,
                    "n": 160,
                    "createdAt": "2026-09-20T02:30:00Z",
                }
            ],
        },
    ]
    brief = brief_of("a", [Behavior(name="math500", test_version="v1", n=160)], runs)
    keys = [s.say.split(" ")[0] for s in brief.readable]
    assert keys == ["say", "write", "show", "name"]
    assert brief.readable[0].cmd.startswith('tracked.open("run_2").note(')
    assert (
        brief.readable[1].cmd
        == 'tracked.behavior("math500", rubric="what passes, what fails, the edge cases")'
    )
    assert brief.readable[2].cmd.startswith(
        'tracked.open("run_2").score("math500", 80, ci=5, n=160, examples='
    )
    assert "dapo-lr5e-05-s17-180st" in brief.readable[3].say
    assert "What a person cannot read yet" in brief.markdown()
    assert "what a person cannot read yet" in str(brief)

    # Posted properly: a note, a rubric, rows, a name that says what changed.
    runs[1]["version"] = "longer-training"
    runs[1]["evals"][0]["version"] = "longer-training"
    runs[1]["notes"] = "180 steps instead of 30, same data"
    runs[1]["evals"][0]["examples"] = [{"prompt": "2+2", "reply": "4", "ok": True}]
    fixed = brief_of(
        "a",
        [Behavior(name="math500", test_version="v1", n=160, rubric="boxed answer matches")],
        runs,
    )
    assert fixed.readable == []
    assert "cannot read" not in fixed.markdown()
