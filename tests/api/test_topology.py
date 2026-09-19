"""Public simulate() knobs: aliases, situations / phrasings / repeats, and live-wired behavior."""

from __future__ import annotations

import json
import threading
import time

import pytest

import whileai.simulations as wai
from tests.helpers import GITHUB_SPEC, POLICY, TOOLS, offline, scripted_agent
from whileai.simulations import defaults
from whileai.simulations.generate.scenarios import (
    SEARCH_ARMS,
    build_dimensions,
    reallocate_search_arms,
)


def test_resolve_topology_defaults_and_aliases():
    default = wai.resolve_topology()
    assert default["mode"] == "explore"
    assert default["repeat_policy"] == "none"
    assert default["unique_situations"] is True
    assert default["n_req"] == 1
    assert default["k"] == 1
    assert default["k_explicit"] is False
    adaptive = wai.resolve_topology(mode="adaptive")
    assert adaptive["mode"] == "adaptive"
    assert adaptive["repeat_policy"] == "adaptive"

    by_n = wai.resolve_topology(n=4)
    by_req = wai.resolve_topology(requests_per_situation=4)
    by_phrasings = wai.resolve_topology(phrasings=4)
    assert by_n["n_req"] == by_req["n_req"] == by_phrasings["n_req"] == 4
    assert by_n["k"] == 1

    by_k = wai.resolve_topology(repeats=5)
    by_roll = wai.resolve_topology(rollouts_per_request=5)
    by_prompt = wai.resolve_topology(rollouts_per_prompt=5)
    assert by_k["k"] == by_roll["k"] == by_prompt["k"] == 5
    assert by_k["k_explicit"] is True

    unique = wai.resolve_topology(unique=True)
    flagged = wai.resolve_topology(unique_situations=True)
    none = wai.resolve_topology(repeat_policy="none")
    explore = wai.resolve_topology(mode="explore")
    assert unique["unique_situations"] is True
    assert flagged["unique_situations"] is True
    assert unique["n_req"] == unique["k"] == 1
    assert flagged["n_req"] == flagged["k"] == 1
    assert none["mode"] == explore["mode"] == "explore"
    assert none["unique_situations"] is True
    assert explore["n_req"] == explore["k"] == 1

    sft = wai.resolve_topology(mode="sft")
    # k is completions per phrasing and is what top_per_prompt selects
    # among; at k=1 the mode could not do the rejection sampling it is
    # named for (Lambert 2025, chapter Rejection Sampling). Phrasing
    # diversity stays at 3, and repeats= moves k.
    assert sft["n_req"] == defaults.SFT_PHRASINGS_PER_SITUATION == 3
    assert sft["k"] == defaults.SFT_COMPLETIONS_PER_PROMPT == 4
    assert wai.resolve_topology(mode="sft", repeats=10)["k"] == 10
    rl = wai.resolve_topology(mode="rl")
    assert rl["n_req"] == 1 and rl["k"] == 8 and rl["k_explicit"] is False
    assert wai.resolve_topology(mode="rl", rollouts_per_request=16)["k"] == 16
    assert wai.resolve_topology(mode="rl", repeats=16)["k"] == 16


def test_repeat_policy_names_are_checked():
    import pytest

    assert wai.resolve_topology(mode="rl")["repeat_policy"] == "successive"
    assert wai.resolve_topology(mode="rl", repeat_policy="fixed")["repeat_policy"] == "fixed"
    with pytest.raises(ValueError, match="repeat_policy"):
        wai.resolve_topology(mode="rl", repeat_policy="sometimes")


def test_unique_situations_defaults_n_k_unless_set():
    base = wai.resolve_topology(unique_situations=True)
    assert base["n_req"] == 1 and base["k"] == 1
    with_k = wai.resolve_topology(unique_situations=True, rollouts_per_request=5)
    assert with_k["n_req"] == 1 and with_k["k"] == 5
    with_n = wai.resolve_topology(unique_situations=True, requests_per_situation=3)
    assert with_n["n_req"] == 3 and with_n["k"] == 1
    sft = wai.resolve_topology(mode="sft", unique_situations=True)
    assert sft["mode"] == "sft" and sft["n_req"] == 1 and sft["k"] == 1
    rl = wai.resolve_topology(mode="rl", unique_situations=True, rollouts_per_request=5)
    assert rl["mode"] == "rl" and rl["k"] == 5 and rl["n_req"] == 1
    alias = wai.resolve_topology(unique=True, repeats=2)
    assert alias["unique_situations"] is True
    assert alias["k"] == 2


def test_situations_int_is_n_list_is_seed():
    n_cards, seeds = wai.simulation._parse_situations_arg(3, ["extra opener"])
    assert n_cards == 3
    assert seeds == ["extra opener"]
    with pytest.raises(ValueError, match="seed_prompts"):
        wai.simulate(scripted_agent, situations=["not an N"], budget=2, **offline())
    none, listed = wai.simulation._parse_situations_arg(None, ["where is order ORD-1"])
    assert none is None
    assert listed == ["where is order ORD-1"]

    capped = wai.simulate(
        scripted_agent, situations=2, requests_per_situation=1, repeats=1, budget=20, **offline()
    )
    assert capped.n_situations == 2
    assert capped.requests_per_situation == 1
    assert capped.rollouts_per_request == 1
    keys = {
        json.dumps(t["scenario_dimensions"], sort_keys=True, default=str)
        for t in capped.trajectories
        if t.get("scenario_dimensions")
    }
    assert len(keys) <= 2

    seeded = wai.simulate(
        scripted_agent,
        repeats=1,
        budget=6,
        extra_situations=["please refund ORD-9 now", "also look up ORD-1"],
        **offline(),
    )
    prompts = {t["prompt"] for t in seeded.trajectories}
    assert any("ORD-9" in p for p in prompts)
    assert any("ORD-1" in p for p in prompts)


def test_public_n_is_requests_per_situation_not_completions(monkeypatch):
    seen: list[int] = []

    def fake_complete(_url, _model, _messages, **kwargs):
        seen.append(int(kwargs.get("n") or 1))
        idx = len(seen)
        letters = "abcdefghijklmnopqrstuvwxyz"
        topic = f"{letters[(idx // 26) % 26]}{letters[idx % 26]}topic"
        return {"content": json.dumps([{"region_id": None, "message": f"check {topic}"}])}

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    data = wai.simulate(
        scripted_agent,
        mode="adaptive",
        n=5,
        repeats=1,
        budget=6,
        seed=0,
        grade=False,
        concurrency=4,
        simulator="vllm:fake@http://127.0.0.1:9",
        time_budget=60,
        advanced={"per_round": 6, "mutate_failures": False},
    )
    assert data.requests_per_situation == 5
    assert data.rollouts_per_request == 1
    assert seen
    assert all(n <= 3 for n in seen)

    seen.clear()
    data2 = wai.simulate(
        scripted_agent,
        mode="adaptive",
        n=1,
        repeats=1,
        budget=4,
        seed=0,
        grade=False,
        concurrency=4,
        simulator="vllm:fake@http://127.0.0.1:9",
        time_budget=60,
        advanced={"per_round": 6, "mutate_failures": False, "completions_per_request": 6},
    )
    assert data2.requests_per_situation == 1
    assert seen
    assert max(seen) <= 6
    assert max(seen) >= 3


def test_mode_sft_rl_explore_change_n_and_k():
    sft = wai.simulate(scripted_agent, mode="sft", budget=12, **offline())
    assert sft.mode == "sft"
    assert sft.rollouts_per_request == 4
    assert sft.requests_per_situation == 3
    # k=4 means several completions share a prompt on purpose: that is the
    # distribution top_per_prompt selects from. Prompts are no longer unique,
    # and were never meant to be once the mode can select (Lambert 2025,
    # chapter Rejection Sampling).
    assert len({t["prompt"] for t in sft.trajectories}) < len(sft.trajectories)

    rl = wai.simulate(scripted_agent, mode="rl", budget=16, **offline())
    assert rl.mode == "rl"
    assert rl.rollouts_per_request == 8
    assert rl.repeat_policy == "successive"
    prompts = [t["prompt"] for t in rl.trajectories]
    # successive: every prompt is probed first, so a 16-row budget opens
    # more than two prompts; a deterministic agent never splits, so the
    # run reports no mixed group and the budget cut the rest
    assert len(set(prompts)) > 2
    assert rl.search["groups"]["k"] == 8 and rl.search["groups"]["mixed"] == 0
    assert rl.allocator.get("explore", 0) + rl.allocator.get("expand", 0) >= 1

    fixed = wai.simulate(
        scripted_agent, mode="rl", budget=16, **offline(advanced={"repeat_policy": "fixed"})
    )
    assert fixed.repeat_policy == "fixed"
    assert len({t["prompt"] for t in fixed.trajectories}) == 2

    explore = wai.simulate(scripted_agent, mode="explore", budget=10, **offline())
    assert explore.repeat_policy == "none"
    assert explore.requests_per_situation == 1
    assert len({t["prompt"] for t in explore.trajectories}) == len(explore.trajectories)


def test_rl_covering_grid_and_fault_rate_are_overridable():
    from whileai.simulations.generate.scenarios import scenario_regions

    def n_faults(regions):
        return sum(
            1
            for row in regions
            if str(row["assignment"].get("tool_condition") or "success") != "success"
        )

    explore = scenario_regions(TOOLS, POLICY)
    rl = scenario_regions(TOOLS, POLICY, mode="rl")
    forced_on = scenario_regions(TOOLS, POLICY, mode="rl", prefer_success=True)
    forced_off = scenario_regions(TOOLS, POLICY, prefer_success=False)
    assert n_faults(rl) > n_faults(explore)
    assert n_faults(forced_on) == n_faults(explore)
    assert n_faults(forced_off) == n_faults(rl)
    data = wai.simulate(
        scripted_agent,
        mode="rl",
        rollouts_per_request=16,
        fault_rate=0.6,
        prefer_success=False,
        budget=16,
        **offline(),
    )
    assert data.rollouts_per_request == 16


def test_n_phrasings_are_not_k_repeats():
    """n = different phrasings of one situation. k = same phrasing, k repeats."""
    from collections import Counter

    n_run = wai.simulate(
        scripted_agent, requests_per_situation=3, rollouts_per_request=1, budget=12, **offline()
    )
    k_run = wai.simulate(
        scripted_agent, requests_per_situation=1, rollouts_per_request=3, budget=12, **offline()
    )
    assert n_run.requests_per_situation == 3
    assert n_run.rollouts_per_request == 1
    assert k_run.requests_per_situation == 1
    assert k_run.rollouts_per_request == 3
    n_prompts = [t["prompt"] for t in n_run.trajectories]
    k_prompts = [t["prompt"] for t in k_run.trajectories]
    assert len(n_run.trajectories) == 12
    assert len(k_run.trajectories) == 12
    assert len(set(n_prompts)) == 12
    assert max(Counter(n_prompts).values()) == 1
    assert len(set(k_prompts)) == 4
    assert set(Counter(k_prompts).values()) == {3}


def test_unique_situations_keeps_new_cards_unless_n_k_set():
    plain = wai.simulate(scripted_agent, mode="sft", unique_situations=True, budget=8, **offline())
    assert plain.unique_situations is True
    assert plain.requests_per_situation == 1
    assert plain.rollouts_per_request == 1
    assert len({t["prompt"] for t in plain.trajectories}) == len(plain.trajectories)

    rl = wai.simulate(
        scripted_agent,
        mode="rl",
        rollouts_per_request=5,
        budget=10,
        **offline(advanced={"repeat_policy": "fixed"}),
    )
    assert rl.rollouts_per_request == 5
    assert rl.requests_per_situation == 1
    prompts = [t["prompt"] for t in rl.trajectories]
    assert len(set(prompts)) == 2
    from collections import Counter

    assert set(Counter(prompts).values()) == {5}
    # the rl default is successive: k is the ceiling, the probe opens more
    # prompts first and a deterministic agent never earns the rest
    succ = wai.simulate(scripted_agent, mode="rl", rollouts_per_request=5, budget=10, **offline())
    assert succ.rollouts_per_request == 5 and succ.repeat_policy == "successive"
    assert max(Counter(t["prompt"] for t in succ.trajectories).values()) <= 5

    alias = wai.simulate(scripted_agent, unique=True, budget=8, **offline())
    assert alias.unique_situations is True
    assert alias.requests_per_situation == 1
    assert alias.rollouts_per_request == 1


def test_phrasings_alias_is_requests_per_situation():
    data = wai.simulate(scripted_agent, phrasings=3, repeats=1, budget=12, **offline())
    assert data.requests_per_situation == 3
    assert data.rollouts_per_request == 1
    with pytest.raises(ValueError, match="not both"):
        wai.resolve_topology(phrasings=3, n=4)


def test_unique_is_topology_not_writer_flight():
    lock = threading.Lock()
    peaks = {"unique": 0, "default": 0}

    def make_writer(label):
        active = 0

        def writer(_dataset=None, index=0):
            nonlocal active
            with lock:
                active += 1
                peaks[label] = max(peaks[label], active)
            time.sleep(0.02)
            with lock:
                active -= 1
            letters = "abcdefghijklmnopqrstuvwxyz"
            return [
                f"check {letters[((index * 12 + i) // 26) % 26]}"
                f"{letters[(index * 12 + i) % 26]}topic"
                for i in range(12)
            ]

        return writer

    u = wai.simulate(
        scripted_agent,
        unique=True,
        budget=24,
        seed=0,
        grade=False,
        concurrency=8,
        simulator=make_writer("unique"),
        until="compute",
        time_budget=None,
        advanced={"mutate_failures": False},
    )
    d = wai.simulate(
        scripted_agent,
        mode="adaptive",
        unique=False,
        repeats=1,
        budget=24,
        seed=0,
        grade=False,
        concurrency=8,
        simulator=make_writer("default"),
        until="compute",
        time_budget=None,
        advanced={"mutate_failures": False},
    )
    assert 1 <= peaks["unique"] <= 4
    assert 1 <= peaks["default"] <= 4
    assert peaks["unique"] == peaks["default"] or peaks["unique"] >= 2
    assert u.unique_situations is True
    assert d.unique_situations is False
    assert len({t["prompt"] for t in u.trajectories}) == len(u.trajectories)


def test_until_compute_vs_saturation_and_aliases():
    dims = {
        "tool": ["lookup_order"],
        "rule": ["unspecified"],
        "stance": ["ordinary"],
        "world_state": ["unspecified"],
        "tool_condition": ["success"],
        "history": ["fresh"],
    }
    compute = wai.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        budget=40,
        until="budget_only",
        dimensions=dims,
        repeats=8,
        mode="adaptive",
        **offline(),
    )
    assert compute.stopped_because == "budget"
    assert compute.coverage["until"] == "compute"
    assert len(compute.trajectories) == 40

    halt = wai.simulate(
        lambda m: {"steps": [], "final_text": "ok"},
        budget=40,
        until="first",
        dimensions=dims,
        rollouts_per_request=5,
        mode="adaptive",
        **offline(),
    )
    assert halt.stopped_because == "saturation"
    assert halt.coverage["until"] == "saturation"
    assert len(halt.trajectories) < 40


def test_budget_and_time_budget_are_compute_caps():
    rows = wai.simulate(scripted_agent, budget=7, repeats=1, **offline())
    assert len(rows.trajectories) == 7
    assert rows.stopped_because == "budget"
    assert rows.budget == 7

    def slow_agent(message):
        time.sleep(0.01)
        return scripted_agent(message)

    clock = wai.simulate(
        slow_agent,
        budget=200,
        time_budget=0.15,
        repeats=1,
        concurrency=2,
        simulator=False,
        seed=0,
        grade=False,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert clock.stopped_because == "time_budget"
    assert len(clock.trajectories) < 200


def test_risk_aliases_fault_rate_and_stays_off_fail_arms():
    assert SEARCH_ARMS["failure_mutation"] <= 0.03 + 1e-9
    assert SEARCH_ARMS["behavior_targeted"] <= 0.03 + 1e-9
    stances = build_dimensions(TOOLS, POLICY)["stance"]
    assert "adversarial" in stances
    assert "boundary" in stances

    hot = {arm: 1.0 if arm == "failure_mutation" else 0.0 for arm in SEARCH_ARMS}
    weights = dict(SEARCH_ARMS)
    for _ in range(20):
        weights = reallocate_search_arms(weights, hot)
    assert weights["failure_mutation"] <= 0.08 + 1e-9
    assert weights["failure_mutation"] < 0.15

    off = wai.simulate(
        scripted_agent,
        risk=0,
        repeats=1,
        budget=16,
        dimensions={
            "tool": ["lookup_order"],
            "rule": ["unspecified"],
            "stance": ["ordinary"],
            "world_state": ["unspecified"],
            "tool_condition": ["timeout"],
            "history": ["fresh"],
        },
        **offline(),
    )
    on = wai.simulate(
        scripted_agent,
        risk=1,
        repeats=1,
        budget=16,
        dimensions={
            "tool": ["lookup_order"],
            "rule": ["unspecified"],
            "stance": ["ordinary"],
            "world_state": ["unspecified"],
            "tool_condition": ["timeout"],
            "history": ["fresh"],
        },
        **offline(),
    )
    assert sum(1 for t in off.trajectories if t.get("faults")) == 0
    assert sum(1 for t in on.trajectories if t.get("faults")) > 0
    assert off.arm_weights["failure_mutation"] <= 0.08 + 1e-9


def test_seed_grade_grader_dimensions_texture_output(tmp_path):
    a = wai.simulate(scripted_agent, repeats=1, budget=8, **offline())
    b = wai.simulate(scripted_agent, repeats=1, budget=8, **offline(seed=1))
    assert [t["prompt"] for t in a.trajectories] != [t["prompt"] for t in b.trajectories]

    graded = wai.simulate(
        scripted_agent,
        grade=True,
        repeats=1,
        budget=4,
        tools=TOOLS,
        policy=POLICY,
        seed=0,
        concurrency=4,
        simulator=False,
        time_budget=None,
        advanced={"per_round": 6, "mutate_failures": False},
    )
    raw = wai.simulate(
        scripted_agent,
        grade=False,
        repeats=1,
        budget=4,
        tools=TOOLS,
        policy=POLICY,
        seed=0,
        concurrency=4,
        simulator=False,
        time_budget=None,
        advanced={"per_round": 6, "mutate_failures": False},
    )
    assert all(isinstance(t.get("reward"), (int, float)) for t in graded.trajectories)
    assert all(t.get("reward") is None for t in raw.trajectories)

    scored = wai.simulate(
        scripted_agent,
        grade=True,
        grader=lambda _t: 0.25,
        repeats=1,
        budget=4,
        tools=TOOLS,
        policy=POLICY,
        seed=0,
        concurrency=4,
        simulator=False,
        time_budget=None,
        advanced={"per_round": 6, "mutate_failures": False},
    )
    assert all(t["reward"] == 0.25 for t in scored.trajectories)

    tiny = {
        "tool": ["lookup_order"],
        "rule": ["unspecified"],
        "stance": ["ordinary"],
        "world_state": ["unspecified"],
        "tool_condition": ["success"],
        "history": ["fresh"],
    }
    dimmed = wai.simulate(scripted_agent, dimensions=tiny, repeats=1, budget=10, **offline())
    cells = {
        json.dumps(t.get("scenario_dimensions"), sort_keys=True, default=str)
        for t in dimmed.trajectories
        if t.get("scenario_dimensions")
    }
    assert cells
    assert all("lookup_order" in c for c in cells)

    dest = tmp_path / "out.jsonl"
    written = wai.simulate(scripted_agent, output=str(dest), repeats=1, budget=3, **offline())
    assert dest.exists()
    assert len(dest.read_text().splitlines()) == len(written.trajectories)


def test_texture_reaches_writer_tag_draw(monkeypatch):
    from whileai.simulations.generate.diversity import sample_cell_tags as orig

    seen: list[float] = []

    def tracked(seed, round_index, key, assignment=None, texture_rate=0.08, **kw):
        seen.append(float(texture_rate))
        return orig(seed, round_index, key, assignment, texture_rate=texture_rate, **kw)

    monkeypatch.setattr("whileai.simulations.generate.generator.sample_cell_tags", tracked)

    def fake_complete(_url, _model, _messages, **_kwargs):
        return {"content": json.dumps([{"region_id": None, "message": "where's my order ORD-1"}])}

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    wai.simulate(
        scripted_agent,
        mode="adaptive",
        texture=0.0,
        repeats=1,
        budget=3,
        seed=0,
        grade=False,
        concurrency=2,
        simulator="vllm:fake@http://127.0.0.1:9",
        time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert seen
    assert all(rate == 0.0 for rate in seen)
    seen.clear()
    wai.simulate(
        scripted_agent,
        mode="adaptive",
        texture=1.0,
        repeats=1,
        budget=3,
        seed=0,
        grade=False,
        concurrency=2,
        simulator="vllm:fake@http://127.0.0.1:9",
        time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert seen
    assert all(rate == 1.0 for rate in seen)


def test_avg_turns_max_turns_concurrency_temperature_backend(monkeypatch):
    seen_temp = []
    seen_backend = []

    def fake_complete(_url, _model, messages, **kwargs):
        seen_temp.append(kwargs.get("temperature"))
        last = messages[-1] if messages else {}
        if last.get("role") == "user":
            return {"content": "Order ORD-1 is packed."}
        return {"content": "done"}

    def fake_local(url, model, **kwargs):
        seen_backend.append(
            (
                url,
                model,
                kwargs.get("max_turns"),
                kwargs.get("avg_turns"),
                kwargs.get("temperature"),
                kwargs.get("max_tokens"),
            )
        )

        def agent(message):
            return {"steps": [], "final_text": "ok"}

        return agent

    monkeypatch.setattr("whileai.simulations.run.engine.local_model", fake_local)
    wai.simulate(
        tools=TOOLS,
        policy=POLICY,
        backend="vllm:fake@http://127.0.0.1:9",
        max_turns=6,
        avg_turns=2,
        temperature=0.2,
        agent_max_tokens=4096,
        budget=3,
        repeats=1,
        grade=False,
        concurrency=2,
        simulator=False,
        seed=0,
        time_budget=None,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert seen_backend
    assert seen_backend[0][2] == 6
    assert seen_backend[0][3] == 2.0
    assert seen_backend[0][4] == 0.2
    assert seen_backend[0][5] == 4096

    lock = threading.Lock()
    peak = 0
    active = 0

    def slow(_message):
        nonlocal peak, active
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"steps": [], "final_text": "ok"}

    wai.simulate(
        slow,
        tools=TOOLS,
        policy=POLICY,
        budget=8,
        repeats=1,
        grade=False,
        concurrency=3,
        simulator=False,
        seed=0,
        time_budget=None,
        advanced={"per_round": 8, "mutate_failures": False},
    )
    assert 2 <= peak <= 3


def test_embedder_is_used_for_selection():
    called = {"n": 0}

    class Spy:
        name = "spy"
        semantic = False

        def embed(self, texts):
            called["n"] += len(texts)
            return [[float(i), 0.0, 1.0] for i, _ in enumerate(texts)]

    data = wai.simulate(scripted_agent, embedder=Spy(), repeats=1, budget=8, **offline())
    assert called["n"] > 0
    assert data.embedder_name == "spy"


def test_spec_tools_policy_agent_simulator_change_rows():
    github = wai.simulate(
        scripted_agent,
        spec=str(GITHUB_SPEC),
        budget=4,
        repeats=1,
        grade=False,
        concurrency=4,
        simulator=False,
        seed=0,
        time_budget=None,
        advanced={"per_round": 6, "mutate_failures": False},
    )
    names = {(t.get("function") or t).get("name") for t in github.profile.tools}
    assert "search_issues" in names

    def writer(_dataset=None, index=0):
        letters = "abcdefghijklmnopqrstuvwxyz"
        return [
            f"check {letters[((index * 6 + i) // 26) % 26]}{letters[(index * 6 + i) % 26]}topic"
            for i in range(6)
        ]

    custom = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=4,
        repeats=1,
        grade=False,
        concurrency=4,
        simulator=writer,
        seed=0,
        time_budget=None,
        advanced={"per_round": 6, "mutate_failures": False},
    )
    assert any("topic" in t["prompt"] for t in custom.trajectories)


def test_k_does_not_clone_followups(monkeypatch):
    follow_n = {"n": 0}

    def fake_complete(_url, _model, messages, **kwargs):
        if not kwargs.get("tools"):
            follow_n["n"] += 1
            return {"content": f"also check refund {follow_n['n']}"}
        last = messages[-1] if messages else {}
        if last.get("role") == "user" and "refund" in str(last.get("content", "")):
            return {"content": f"Refund note {last.get('content')}"}
        return {"content": "Order ORD-1 is packed. Want me to check the refund too?"}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 8
    )
    data = wai.simulate(
        tools=TOOLS,
        policy=POLICY,
        extra_situations=["where is my order ORD-1"],
        budget=2,
        repeats=2,
        grade=False,
        concurrency=2,
        simulator=False,
        backend="vllm:fake@http://127.0.0.1:9",
        seed=0,
        time_budget=None,
        max_turns=8,
        avg_turns=4,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert len(data.trajectories) == 2
    assert len({t["prompt"] for t in data.trajectories}) == 1
    follows = []
    for t in data.trajectories:
        follows.extend(
            s.get("user") for s in (t.get("steps") or []) if isinstance(s, dict) and s.get("user")
        )
    assert len(set(follows)) == 2
    assert {"also check refund 1", "also check refund 2"} == set(follows)


def test_adaptive_allocator_short_clock_is_messier():
    short = wai.adaptive_allocator(20, "compute")
    long = wai.adaptive_allocator(180, "compute")
    sat = wai.adaptive_allocator(20, "saturation")
    first = wai.adaptive_allocator(20, "first")
    none = wai.adaptive_allocator(None, "compute")
    early = wai.adaptive_allocator(60, "compute", elapsed=5)
    late = wai.adaptive_allocator(60, "compute", elapsed=50)
    assert short["n_req"] > 1 and short["k"] > 1
    assert long["n_req"] > 1 and long["k"] > 1
    assert short["expand"] + short["verify"] > long["expand"] + long["verify"]
    assert long["explore"] > short["explore"]
    assert sat["explore"] > short["explore"]
    assert first["until"] == sat["until"] == "saturation"
    assert none["explore"] >= long["explore"]
    assert late["expand"] + late["verify"] > early["expand"] + early["verify"]
    short_slots = wai.allocator_slot_counts(8, short)
    long_slots = wai.allocator_slot_counts(8, long)
    assert short_slots["expand"] + short_slots["verify"] > (
        long_slots["expand"] + long_slots["verify"]
    )
    assert sum(short_slots.values()) == 8


def test_adaptive_allocator_records_explore_expand_verify():
    data = wai.simulate(scripted_agent, mode="adaptive", budget=16, **offline(time_budget=15))
    assert data.mode == "adaptive"
    assert data.allocator
    assert data.allocator.get("explore", 0) >= 1
    assert data.requests_per_situation > 1
    assert data.rollouts_per_request > 1
    assert data.coverage.get("mode") == "adaptive"
    assert data.coverage.get("requests_per_situation") == data.requests_per_situation
    assert data.coverage.get("rollouts_per_request") == data.rollouts_per_request


def test_explore_cards_walk_different_tools_or_stances():
    data = wai.simulate(scripted_agent, mode="explore", budget=16, **offline())
    prompts = [t["prompt"] for t in data.trajectories]
    assert prompts
    assert len(prompts) == len(set(prompts))
    tools = {
        (t.get("scenario_dimensions") or {}).get("tool")
        for t in data.trajectories
        if (t.get("scenario_dimensions") or {}).get("tool")
    }
    stances = {
        (t.get("scenario_dimensions") or {}).get("stance")
        for t in data.trajectories
        if (t.get("scenario_dimensions") or {}).get("stance")
    }
    assert len(tools) >= 2 or len(stances) >= 2


def test_agent_max_tokens_reaches_a_spec_agent(monkeypatch):
    """simulate(agent="vllm:...", agent_max_tokens=N) goes through resolve()."""
    seen = []

    def fake_local(url, model, **kwargs):
        seen.append((kwargs.get("max_tokens"), kwargs.get("timeout")))

        def agent(message):
            return {"steps": [], "final_text": "ok"}

        return agent

    monkeypatch.setattr("whileai.simulations.generate.adapters.local_model", fake_local)
    wai.simulate(
        agent="vllm:fake@http://127.0.0.1:9",
        tools=TOOLS,
        policy=POLICY,
        agent_max_tokens=4096,
        timeout=123,
        budget=2,
        repeats=1,
        grade=False,
        simulator=False,
        seed=0,
        time_budget=None,
    )
    assert seen and seen[0] == (4096, 123)


def test_agent_max_tokens_does_not_break_an_http_agent():
    """openai_http takes no reply budget; the option must not reach it."""
    from whileai.simulations.generate.adapters import resolve

    agent, kind = resolve(
        "http://127.0.0.1:9/v1/chat/completions", tools=TOOLS, policy=POLICY, max_tokens=4096
    )
    assert kind == "http" and callable(agent)
