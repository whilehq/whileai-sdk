"""tracked.verdict(): the candidate is resolved per behavior (#597).

Two held-out sets, one arm each: every behavior gets its own verdict
whatever order the arms were posted in, re-scoring the served version
demotes nothing, ``version=`` names the candidate, and two behaviors that
each clear the bar read "moved, replicated".
"""

from __future__ import annotations

import pytest

from tests.api.test_platform import Fake
from whileai.platform import PlatformError, track

SEED0 = "looks_up_before_answering"
SEED1 = "looks_up_before_answering_seed1"
FLOOR = 3.1


def _behavior(name: str) -> dict:
    return {"name": name, "n": 40, "noiseFloor": FLOOR}


def _run(run_id: str, version: str, at: str, *evals: tuple[str, float]) -> dict:
    return {
        "id": run_id,
        "agent": "a",
        "version": version,
        "createdAt": at,
        "evals": [
            {"behavior": b, "version": version, "score": s, "ci": 5.0, "n": 40, "createdAt": at}
            for b, s in evals
        ],
    }


class Platform(Fake):
    """The Fake plus the rows the dashboard is built from. ``candidate`` is
    the agent-level one the API sends today: the newest run, whatever it
    scored."""

    def __init__(self, runs: list[dict], serving: str = "base"):
        super().__init__()
        self.rows = runs
        self.serving = serving

    def _versions(self, behavior: str) -> list[dict]:
        newest: dict[str, tuple[str, dict]] = {}
        for r in self.rows:
            for e in r["evals"]:
                if e["behavior"] != behavior:
                    continue
                if r["version"] not in newest or e["createdAt"] > newest[r["version"]][0]:
                    newest[r["version"]] = (
                        e["createdAt"],
                        {**e, "v": r["version"], "run": r["id"]},
                    )
        ordered = sorted(newest.values(), key=lambda t: t[0])
        return [v for _, v in ordered]

    def __call__(self, method, path, body=None):
        if path == "/agents/a/behaviors" and method == "GET":
            self.calls.append((method, path, body))
            return {"behaviors": [_behavior(SEED0), _behavior(SEED1)]}
        if path.startswith("/runs?agent=") and method == "GET":
            self.calls.append((method, path, body))
            return {"runs": sorted(self.rows, key=lambda r: r["createdAt"], reverse=True)}
        if "/dashboard" in path:
            self.calls.append((method, path, body))
            behavior = path.split("behavior=")[1] if "behavior=" in path else SEED0
            latest = max(self.rows, key=lambda r: r["createdAt"])
            candidate = latest["version"] if latest["version"] != self.serving else None
            return {
                "agent": {"id": "a", "name": "a", "serving": self.serving, "candidate": candidate},
                "behavior": _behavior(behavior),
                "behaviors": [SEED0, SEED1],
                "versions": self._versions(behavior),
                "verdict": {"candidate": candidate, "serving": self.serving},
            }
        if method == "DELETE" and path.startswith("/runs/"):
            self.calls.append((method, path, body))
            run_id = path.rsplit("/", 1)[1]
            if not any(r["id"] == run_id for r in self.rows):
                raise PlatformError(404, f"DELETE {path}: No run {run_id}")
            return {"ok": True}
        return super().__call__(method, path, body)


def _two_arms(order: tuple[str, str]) -> Platform:
    """base scored on both sets, then one arm per set, posted in ``order``."""
    arms = {
        "sft-v1": _run("run_v1", "sft-v1", "2026-09-20T02:00:00Z", (SEED0, 76.9)),
        "sft-v1-s1": _run("run_s1", "sft-v1-s1", "2026-09-20T03:00:00Z", (SEED1, 75.0)),
    }
    first, second = order
    arms[first]["createdAt"] = "2026-09-20T02:00:00Z"
    arms[second]["createdAt"] = "2026-09-20T03:00:00Z"
    for arm in arms.values():
        for e in arm["evals"]:
            e["createdAt"] = arm["createdAt"]
    base = _run("run_base", "base", "2026-09-20T01:00:00Z", (SEED0, 46.9), (SEED1, 33.8))
    return Platform([base, arms[first], arms[second]])


@pytest.mark.parametrize("order", [("sft-v1", "sft-v1-s1"), ("sft-v1-s1", "sft-v1")])
def test_each_behavior_gets_its_own_verdict_whatever_the_posting_order(order):
    t = track("a", transport=_two_arms(order))
    seed0, seed1 = t.verdict(SEED0), t.verdict(SEED1)
    assert (seed0.candidate, seed0.serving, seed0.delta) == ("sft-v1", "base", 30.0)
    assert (seed1.candidate, seed1.serving, seed1.delta) == ("sft-v1-s1", "base", 41.2)
    assert seed0.excludes_zero and seed1.excludes_zero
    assert str(seed0).startswith(f"unproven: {SEED0}: sft-v1 beats base by 30 ")
    assert str(seed1).startswith(f"unproven: {SEED1}: sft-v1-s1 beats base by 41.2 ")


def test_rescoring_the_served_version_does_not_blank_the_verdicts():
    fake = _two_arms(("sft-v1", "sft-v1-s1"))
    fake.rows.append(_run("run_base2", "base", "2026-09-20T04:00:00Z", (SEED1, 34.0)))
    t = track("a", transport=fake)
    assert t.dashboard().agent.candidate is None  # what the API says: base is newest
    assert t.verdict(SEED0).candidate == "sft-v1"
    assert t.verdict(SEED1).candidate == "sft-v1-s1"
    assert t.verdict(SEED1).delta == 41.0  # against the re-scored base


def test_newest_is_by_created_at_not_list_order():
    fake = _two_arms(("sft-v1", "sft-v1-s1"))
    # An older arm on seed0, scored before sft-v1; the dashboard lists it last.
    fake.rows.append(_run("run_v0", "sft-v0", "2026-09-20T01:30:00Z", (SEED0, 70.0)))
    original = fake._versions

    def reversed_versions(behavior):
        return list(reversed(original(behavior)))

    fake._versions = reversed_versions  # type: ignore[method-assign]
    assert track("a", transport=fake).verdict(SEED0).candidate == "sft-v1"


def test_version_names_the_candidate():
    fake = _two_arms(("sft-v1", "sft-v1-s1"))
    fake.rows.append(_run("run_v0", "sft-v0", "2026-09-20T01:30:00Z", (SEED0, 50.0)))
    t = track("a", transport=fake)
    v = t.verdict(SEED0, version="sft-v0")
    assert (v.candidate, v.delta) == ("sft-v0", 3.1)
    assert "sft-v0 about the same as base (+3.1, interval includes zero)" in str(v)
    assert str(t.verdict(SEED0, version="base")).startswith(
        f"{SEED0}: base is the served version; no candidate to compare"
    )
    with pytest.raises(ValueError, match="'sft-v1-s1' has no score on 'looks_up_before_answering'"):
        t.verdict(SEED0, version="sft-v1-s1")


def test_two_behaviors_that_both_clear_read_moved_replicated():
    t = track("a", transport=_two_arms(("sft-v1", "sft-v1-s1")))
    seed0, seed1 = t.verdict(SEED0), t.verdict(SEED1)
    assert seed0.replicated == [SEED1] and seed1.replicated == [SEED0]
    assert (
        f"sft-v1 beats base by 30 (interval excludes zero, clears the noise floor of {FLOOR:g}); "
        f"moved, replicated on {SEED1}; n=40"
    ) in str(seed0)


def test_one_run_scored_on_two_sets_is_not_a_replication():
    base = _run("run_base", "base", "2026-09-20T01:00:00Z", (SEED0, 46.9), (SEED1, 33.8))
    arm = _run("run_v1", "sft-v1", "2026-09-20T02:00:00Z", (SEED0, 76.9), (SEED1, 75.0))
    v = track("a", transport=Platform([base, arm])).verdict(SEED0)
    assert v.candidate == "sft-v1" and v.excludes_zero and v.replicated == []
    assert "replicated" not in str(v)


def test_a_second_set_inside_the_noise_is_not_a_replication():
    fake = _two_arms(("sft-v1", "sft-v1-s1"))
    fake.rows[2]["evals"][0]["score"] = 36.0  # sft-v1-s1: +2.2 on seed1, under the floor
    t = track("a", transport=fake)
    assert t.verdict(SEED0).replicated == []
    assert "sft-v1-s1 about the same as base (+2.2, interval includes zero)" in str(
        t.verdict(SEED1)
    )


def test_brief_reads_the_per_behavior_verdict():
    t = track("a", transport=_two_arms(("sft-v1", "sft-v1-s1")))
    assert t.brief(SEED0).means == "sft-v1 is better than base by 30 points, outside the noise."
    assert (
        t.brief(SEED1).means == "sft-v1-s1 is better than base by 41.2 points, outside the noise."
    )


def test_delete_run_names_the_run_id_behind_a_version_name():
    fake = _two_arms(("sft-v1", "sft-v1-s1"))
    t = track("a", transport=fake)
    with pytest.raises(PlatformError) as err:
        t.delete_run("sft-v1")
    assert str(err.value) == "No run 'sft-v1'. 'sft-v1' is a version name; its run id is run_v1."
    with pytest.raises(PlatformError, match="DELETE /runs/nope: No run nope"):
        t.delete_run("nope")
    assert t.delete_run("run_v1") == {"ok": True}


def test_points_and_fraction_true_give_the_same_verdict():
    """70 and 75 points, and the same two rates posted with ``fraction=True``,
    read as one sentence: the client converts and the platform sees one scale."""
    from tests.api.test_platform import Fake

    floor = 2.4

    def rows(scores: list[dict]) -> list[dict]:
        return [
            {
                "id": f"run_{i}",
                "agent": "a",
                "version": e["version"],
                "createdAt": f"2026-09-20T0{i}:00:00Z",
                "evals": [{**e, "createdAt": f"2026-09-20T0{i}:00:00Z"}],
            }
            for i, e in enumerate(scores, start=1)
        ]

    class Floor(Platform):
        def __call__(self, method, path, body=None):
            out = super().__call__(method, path, body)
            if path == "/agents/a/behaviors":
                out["behaviors"][0]["noiseFloor"] = floor
            elif "/dashboard" in path:
                out["behavior"]["noiseFloor"] = floor
            return out

    fake = Fake()
    posted = track("a", transport=fake).run("v1", flush_every=100)
    posted.score(SEED0, 0.70, ci=0.03, n=200, fraction=True)
    posted.score(SEED0, 0.75, ci=0.03, n=200, fraction=True)
    wire = [dict(b[0], version=v) for (_, _, b), v in zip(fake.calls[-2:], ("v1", "v2"))]
    assert [e["score"] for e in wire] == [70.0, 75.0]

    points = [
        {"behavior": SEED0, "version": "v1", "score": 70, "ci": 3.0, "n": 200},
        {"behavior": SEED0, "version": "v2", "score": 75, "ci": 3.0, "n": 200},
    ]
    a = str(track("a", transport=Floor(rows(points), serving="v1")).verdict(SEED0))
    b = str(track("a", transport=Floor(rows(wire), serving="v1")).verdict(SEED0))
    assert a == b
    assert "v2 beats v1 by 5 " in a
