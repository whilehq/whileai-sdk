"""rl mode spends rollouts where the agent is inconsistent.

Successive allocation: every prompt is probed, a prompt whose rollouts
disagree is filled to k, a prompt that stays unanimous stops once a fresh
prompt is the better bet, and the clock finishes groups instead of cutting
them. Dynamic sampling (Yu et al. 2025 (DAPO), arXiv:2503.14476) and
difficulty filtering (Lambert 2025, chapter Reasoning) at generation time.
"""

from __future__ import annotations

import collections
import hashlib
import threading
import time

import whileai.simulations as wai
from tests.helpers import offline, scripted_agent
from whileai.simulations.score.stats import task_key


def _flaky():
    """Half the prompts (by a stable hash) refuse on every second call;
    the other half never do. Which half a prompt falls in does not depend
    on the writer's wording, so every run has both kinds."""
    calls: dict[str, int] = collections.Counter()

    def agent(message: str) -> dict:
        calls[message] += 1
        digest = int(hashlib.md5(message.encode("utf-8")).hexdigest(), 16)
        if digest % 2 == 0 and calls[message] % 2 == 0:
            return {"steps": [], "final_text": "I cannot help with that."}
        return scripted_agent(message)

    return agent


def _judge(row: dict) -> dict:
    refused = "cannot" in str(row.get("final_text") or "").lower()
    return {"reward": 0.0 if refused else 1.0, "reason": "refusal" if refused else "ok"}


def test_split_prompts_fill_to_k_and_the_rest_finish_when_nothing_is_left_to_open():
    data = wai.simulate(
        _flaky(), mode="rl", situations=6, rollouts_per_request=6, budget=40, **offline()
    )
    sizes = collections.Counter(t["prompt"] for t in data.trajectories)
    groups = data.search["groups"]
    assert groups["k"] == 6 and groups["probe"] == 2
    assert groups["mixed"] >= 1
    # every split prompt reached k
    by_prompt: dict[str, set] = {}
    for t in data.trajectories:
        by_prompt.setdefault(t["prompt"], set()).add(t["behavior_signature"])
    for prompt, sigs in by_prompt.items():
        if len(sigs) > 1:
            assert sizes[prompt] == 6, (prompt, sizes[prompt])
    # the situation cap left nothing fresh to open, so unanimous groups
    # finished too: a group stopped earlier is resumed rather than left short
    diag = (groups, sorted(sizes.values()), data.stopped_because, data.search.get("allocator"))
    assert groups["partial"] == 0, diag
    assert groups["stopped_unanimous"] == 0, diag
    assert data.stopped_because == "situations_exhausted", diag


def test_unanimous_prompts_stop_when_fresh_prompts_split_more_often():
    # reproducible=True: the allocator's choices depend on the order rows
    # land, and at concurrency=4 that order is the thread scheduler's, so
    # which groups reach k and which stop short drifted from run to run.
    # Pinned, the run is bit-for-bit the same in any test order.
    data = wai.simulate(
        _flaky(),
        mode="rl",
        situations=30,
        rollouts_per_request=6,
        budget=60,
        grader=_judge,
        reproducible=True,
        **offline(),
    )
    rows = data.trajectories
    groups = data.search["groups"]
    # group by task_key, the way pass_at does: a situation's textured
    # phrasings share one scenario_id, and which situations get one
    # depends on thread order, so grouping by prompt text drifts from
    # the k-way numbers by one group now and then
    sizes = collections.Counter(task_key(t) for t in rows)
    assert groups["mixed"] >= 2, groups
    # the grader ran inside the loop: every row is judged exactly once and
    # the allocation read rewards, so a split is a reward split
    assert all(t.get("judge_status") for t in rows)
    assert data.search["grader"]["scored"] == len(rows)
    split = [p for p in sizes if len({t["reward"] for t in rows if task_key(t) == p}) > 1]
    unanimous = [p for p in sizes if p not in split]
    assert split and unanimous, sizes
    # the budget went to the split prompts: they reached k unless the
    # budget ran out first, and on average they got more rollouts than
    # the unanimous ones, which stopped once a fresh prompt was the
    # better bet
    assert sum(1 for p in split if sizes[p] == 6) >= 2, (groups, sizes)
    mean = lambda ps: sum(sizes[p] for p in ps) / len(ps)  # noqa: E731
    assert mean(split) > mean(unanimous), (groups, sizes)
    unanimous_short = [p for p in unanimous if sizes[p] < 6]
    assert unanimous_short, groups
    # the k-way numbers use the requested k and count every unanimous group
    # shorter than k as unanimous, stopped or cut alike
    got = data.pass_at
    assert got.k == 6
    assert got.n_groups_imputed == len(unanimous_short)
    assert got.pass_at_k is not None and got.pass_pow_k is not None
    # a stopped group is not a cut group: the stamp marks exactly the
    # groups the summary calls partial
    cut = {task_key(t) for t in rows if t.get("group_cut")}
    assert len(cut) == groups["partial"]


def test_clock_finishes_groups_instead_of_cutting_them():
    def slow(message: str) -> dict:
        time.sleep(0.25)
        return _flaky_shared(message)

    _flaky_shared = _flaky()
    data = wai.simulate(
        slow,
        mode="rl",
        situations=40,
        rollouts_per_request=4,
        budget=400,
        **offline(time_budget=2.0),
    )
    assert data.stopped_because == "time_budget"
    groups = data.search["groups"]
    assert groups["closing"] is True
    cut = {t["prompt"] for t in data.trajectories if t.get("group_cut")}
    # closing stopped new groups early enough that at most the last wave
    # could be short; nothing is silently counted as k
    for prompt in cut:
        assert sum(1 for t in data.trajectories if t["prompt"] == prompt) < 4


def test_every_mode_judges_beside_the_loop_when_a_grader_is_given():
    for mode in ("explore", "sft"):
        data = wai.simulate(_flaky(), mode=mode, budget=12, grader=_judge, **offline())
        rows = data.trajectories
        assert rows and all(t.get("judge_status") for t in rows)
        grader = data.search["grader"]
        assert grader["scored"] == len(rows)
        # the rows were judged as they landed, not in one pass after the clock
        assert grader["judged_in_loop"] == len(rows) and grader["judged_after"] == 0, grader


def test_rl_reports_time_spent_idle_waiting_on_verdicts():
    # The pool is idle on the judge exactly when it has nothing left to roll
    # out and a verdict is still outstanding. Racing a judge sleep against
    # the rollouts only makes that likely: under CPU contention the rollouts
    # slow down too, the pool stays busy, and the branch is never reached
    # (#216). So make it structural instead: the judge holds its verdict
    # until every probe rollout has landed, which no amount of load changes.
    # An "inflight is zero" check is not enough, because the two situations'
    # probes launch in a stagger and the count touches zero between them.
    first_wave = 2 * 2  # situations x probe rollouts per situation
    lock = threading.Lock()
    landed = 0
    probes_landed = threading.Event()

    def counted_agent(message: str) -> dict:
        nonlocal landed
        try:
            return scripted_agent(message)
        finally:
            with lock:
                landed += 1
                if landed >= first_wave:
                    probes_landed.set()

    started = time.monotonic()
    hold_s: list[float] = []
    judged = 0

    def blocking_judge(row: dict) -> dict:
        nonlocal judged
        # rollouts never wait on a verdict, so this always releases
        assert probes_landed.wait(timeout=30.0), "probe rollouts never landed"
        with lock:
            judged += 1
            first_wave_verdict = judged <= first_wave
            if not hold_s:
                # The note fires only when idle time is over a tenth of the
                # whole run (engine.py: idle_on_judge_s > 0.1 * elapsed).
                # Under load the rollouts stretch the run while a fixed hold
                # would not, so hold for a multiple of what has elapsed so
                # far; the floor clears the 0.1s rounding on the reported
                # figure. Only the first wave holds: the later verdicts land
                # after the last rollout and would only lengthen the run.
                hold_s.append(max(0.3, 3.0 * (time.monotonic() - started)))
        if first_wave_verdict:
            time.sleep(hold_s[0])
        return _judge(row)

    data = wai.simulate(
        counted_agent,
        mode="rl",
        situations=2,
        rollouts_per_request=4,
        budget=8,
        grader=blocking_judge,
        **offline(),
    )
    groups = data.search["groups"]
    # two situations cannot keep four rollout slots busy: every probe lands,
    # then the pool waits on the judge before it can decide the next rollout
    assert groups["idle_on_judge_s"] > 0
    note = [s for s in data.stages if s.startswith("rl pool idle on judge")]
    assert note and "situations>=2" in note[0], (groups, hold_s, data.stages)


def test_truncated_rollouts_are_not_judged_and_do_not_stall_their_group():
    judged: list[str] = []

    def cut_agent(message: str) -> dict:
        out = scripted_agent(message)
        if hashlib.md5(message.encode("utf-8")).hexdigest()[-1] in "01234567":
            out["steps"] = [
                {"tool": "lookup_order", "arguments": {}, "result": {}, "truncated": True}
            ]
        return out

    def judge(row: dict) -> dict:
        judged.append(row["prompt"])
        return {"reward": 1.0, "reason": "ok"}

    data = wai.simulate(
        cut_agent,
        mode="rl",
        situations=4,
        rollouts_per_request=3,
        budget=12,
        grader=judge,
        **offline(),
    )
    rows = data.trajectories
    cut = [t for t in rows if t.get("judge_name") == "length_cap"]
    assert cut, "no rollout hit the cap"
    assert all(
        t["reward"] is None and t["judge_status"] == "missing_reward" and "truncated" in t["reason"]
        for t in cut
    )
    assert not any(t["prompt"] in judged and t in cut for t in cut)
    assert data.search["groups"]["truncated_skipped"] == len(cut)
    # the run still finished its groups instead of waiting on labels that
    # will never come
    assert data.stopped_because in ("situations_exhausted", "budget"), data.stopped_because
    assert data.search["grader"]["judged_in_loop"] == len(rows) - len(cut)
