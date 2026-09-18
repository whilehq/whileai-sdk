import json

import whileai.simulations as wai
from tests.helpers import GITHUB_SPEC, LINEAR_SPEC, POLICY, REPO_ROOT, TOOLS, scripted_agent
from whileai.simulations.generate.agents import complete as _real_complete
from whileai.simulations.generate.generator import ModelSimulator


def test_agent_voice_user_catches_desk_clarifiers():
    from whileai.simulations.generate.generator import agent_voice_user

    bad = (
        "I'm trying to be specific—what's the exact name of the repo "
        "or the issue number you’re referring to?"
    )
    assert agent_voice_user(bad)
    assert agent_voice_user("Before I look up anything, can you confirm which repo")
    assert agent_voice_user(
        "Could you confirm which repository you're referring to before I look it up."
    )
    assert agent_voice_user("Can you confirm the number you mean?")
    assert agent_voice_user("What would you like me to look up?")
    assert agent_voice_user("I need more details to act.")
    assert agent_voice_user("I'm sorry, I don't know how to handle this.")
    assert agent_voice_user("Which repo are you looking for? I'll search there.")
    assert agent_voice_user("Are you looking for a specific issue or PR?")
    assert agent_voice_user("I need the PR number — can you provide that?")
    assert not agent_voice_user("where's my order ORD-1")
    assert not agent_voice_user("can you look up my reservation tonight")
    assert not agent_voice_user("merge acme/api#448 if checks are green")
    from whileai.simulations.generate.generator import usable_user_message

    assert usable_user_message("merge acme/api#448 if checks are green")
    assert usable_user_message("where's my order ORD-1")
    assert not usable_user_message("Long prompt: outline the steps")
    assert not usable_user_message("lowercase everywhere")
    assert not usable_user_message("understood")
    assert not usable_user_message("hi there")
    assert not usable_user_message("All good. Round 1, seed 1 — data synced and validated.")
    assert usable_user_message("look at acme/api#12, second review round")
    assert not usable_user_message("Can you clarify what you mean by 'vague ask'")
    assert not usable_user_message("This is a long request—could you break it down")
    assert not usable_user_message(
        "The message is cut off—can you resend it with full punctuation?"
    )
    assert not usable_user_message("I'd like to get some information on this general topic.")
    assert not usable_user_message("Could you help me with this request? The phrasing is unclear.")
    assert not usable_user_message("Do nothing regarding the current request.")
    assert not usable_user_message("looking to use get_pr on this")
    assert not usable_user_message("has ambiguous behavior in this ask")
    assert not usable_user_message("length long and punctuation clipped")
    assert not usable_user_message("want to request issue #789")
    assert not usable_user_message("Merge PR #12 now, you're in a hurry")


def test_writer_prompt_forbids_copying_policy_as_user_line():
    from whileai.simulations.generate.diversity import sample_cell_tags
    from whileai.simulations.generate.scenarios import policy_sections

    spec_path = LINEAR_SPEC / "spec.json"
    spec = json.loads(spec_path.read_text())
    sim = ModelSimulator(tools=spec["tools"], policy=spec["policy"], seed=1)
    prompt = sim._prompt(0, sim.regions[:8])
    lower = prompt.lower()
    assert "you are the user" in lower
    assert "you are talking to" in lower
    assert "ordinary speech" in lower
    assert "write only your message" in lower
    assert "linear" in lower
    assert "long prompt" not in lower
    assert "optional tags" not in lower
    assert '"situation"' not in prompt
    assert "please handle all of" not in lower
    assert "start with" not in lower
    leaked = (
        "Issue details must not be shared across teams",
        "Team is missing. Specify the team name to proceed",
        "Cycle change requested but no doc was involved",
        "Both issue IDs must be provided",
        "Search with absent parameters may return no results",
        "Confirm assignee and issue context before assigning",
    )
    clauses = [c for c in policy_sections(spec["policy"]) if len(c) > 32]
    assert clauses
    for clause in clauses:
        assert clause not in prompt
    for _i, region in enumerate(sim.regions[:48]):
        tags = sample_cell_tags(1, 0, region["id"], region.get("assignment") or {})
        blob = json.dumps(tags)
        assert "stance_brief" not in tags
        for sentence in leaked:
            assert sentence not in blob
        for clause in clauses:
            assert clause not in blob
            assert clause not in str(tags.get("rule") or "")


def test_model_prompt_has_no_opener_instructions():
    sim = ModelSimulator(tools=TOOLS, policy=POLICY, seed=1, candidates_per_round=40)
    prompt = sim._prompt(0, sim.regions[:3])
    lower = prompt.lower()
    assert "you are the user" in lower
    assert "you are talking to" in lower
    assert "ordinary speech" in lower
    assert "write only your message" in lower
    assert "capability and scenario cards" in lower
    assert "do not write as the assistant" in lower
    assert "long prompt" not in lower
    assert "medium prompt" not in lower
    assert '"situation"' not in prompt
    assert "produce a user request" not in lower
    assert "task for the assistant" not in lower
    assert "hi, i need" not in lower
    assert "start your" not in lower
    assert "start with" not in lower
    assert "begin with" not in lower
    assert "write like" not in lower
    assert "match `words`" not in lower
    assert "optional tags" not in lower
    assert "3-8 words" not in lower
    assert "60-100" not in lower
    assert "mutter" not in lower
    assert "half-thought" not in lower
    assert "speak like" not in lower
    assert "short phrases" not in lower
    assert '["first"' not in prompt
    assert "please handle all of" not in lower
    assert "please provide" not in lower
    assert "can you" not in lower
    assert "pls, u, thx" not in lower
    assert "mad libs" not in lower
    assert "many tools" not in lower
    assert "tool-heavy" not in lower
    assert "always end" not in lower
    assert "question mark" not in lower
    assert "i already confirmed this with your colleague" not in lower
    assert "just do it, i don't have time" not in lower
    assert "this is the third time" not in lower
    assert '"type":"function"' not in prompt
    assert '"parameters"' not in prompt
    assert "user background" in lower
    assert "private context, not phrasing" in lower
    assert "you know this assistant can" in lower
    assert "first time here" not in lower
    assert "looking to use" not in lower
    assert "a vague ask" not in lower
    assert "has ambiguous behavior" not in lower
    assert "length long" not in lower
    assert "punctuation clipped" not in lower


def test_message_realizes_tags_and_writer_omits_raw_labels():
    from whileai.simulations.generate.generator import message_realizes_tags, usable_user_message

    long_calm = (
        "I think we can merge this one now — it's clean, passes all "
        "checks, and the only thing missing is the changelog entry."
    )
    assert not message_realizes_tags(long_calm, {"length": "long prompt"})
    assert not message_realizes_tags(long_calm, {"tone": "frustrated", "length": "long prompt"})
    assert message_realizes_tags("merge acme/api#448 now", {"length": "short prompt"})
    assert not message_realizes_tags(
        "Wait, did we already close the ticket about the dark mode bug? "
        "I'm pretty sure it was merged last week, but I want to double-check "
        "before I move on.",
        {"length": "short prompt"},
    )
    frustrated = (
        "this is the third time the merge never works, why hasn't anyone looked at the red checks"
    )
    assert message_realizes_tags(frustrated, {"tone": "frustrated"})
    assert not usable_user_message("Short length, ordinary behavior, check the PR")
    assert not usable_user_message("you're frustrated and you keep it brief")

    spec_path = GITHUB_SPEC / "spec.json"
    spec = json.loads(spec_path.read_text())
    sim = ModelSimulator(tools=spec["tools"], policy=spec["policy"], seed=1)
    prompt = sim._prompt(0, sim.regions[:8])
    assert '"writing"' not in prompt
    assert '"tone"' not in prompt
    assert "check a pr" not in prompt.lower()
    assert "note" in prompt
    assert "you know the action you want done" in prompt or "not settled" in prompt


def test_writer_aside_aims_or_goes_vague():
    from whileai.simulations.generate.generator import ModelSimulator

    sim = ModelSimulator(tools=TOOLS, policy=POLICY, seed=1)
    prompt = sim._prompt(0, sim.regions[:16])
    assert "you know this assistant can" in prompt.lower()
    assert "first time here" not in prompt.lower()
    assert "you want" in prompt.lower() or "you may already know" in prompt.lower()
    assert "looking to use" not in prompt
    assert "a general request" not in prompt
    assert "a vague ask" not in prompt
    assert "long prompt" not in prompt.lower()
    assert "no named tool" not in prompt
    assert "you have not named it yet" not in prompt
    from whileai.simulations.generate.generator import _cell_aside

    missing = _cell_aside({"tool": "lookup_order", "world_state": "entity missing"}, {})
    assert "you're not sure this is still there" in missing
    assert "name and the number for that order" in missing
    assert "lookup_order" not in missing
    present = _cell_aside({"tool": "lookup_order", "world_state": "entity exists"}, {})
    assert "not sure this is still there" not in present
    assert "already in the system" in present
    aimed_cell = _cell_aside({"tool": "lookup_order"}, {}, open_ask=True, open_tier="ambiguous")
    assert "name and the number for that order" in aimed_cell
    assert "looking to use" not in aimed_cell
    assert "lookup_order" not in aimed_cell
    assert "vague" not in aimed_cell
    assert "general request" not in aimed_cell
    textured = _cell_aside(
        {"tool": "lookup_order"},
        {"length": "long prompt", "texture": "clipped", "tone": "frustrated"},
    )
    assert "you use more words" in textured
    assert "leave out the marks" in textured
    assert "you're frustrated" in textured
    assert "long prompt" not in textured
    assert "without punctuation" not in textured
    open_cell = _cell_aside({}, {}, open_ask=True, open_tier="ordinary")
    assert "you have a request" in open_cell
    assert "say what you want" in open_cell
    no_ref = _cell_aside({"tool": "lookup_order"}, {}, knows_ref=False, open_ask=True)
    assert "name and the number" not in no_ref
    assert "you have a request" in no_ref
    aimed = prompt.lower().count("you want") + prompt.lower().count("you may already know")
    assert aimed >= 1
    open_seen = any(
        "neighboring or off-topic request" in sim._prompt(i, sim.regions[:16]) for i in range(24)
    )
    assert open_seen


def test_writer_invents_person_without_leaking_labels():
    from whileai.simulations.generate.generator import (
        _grid_card,
        _strip_directive_phrases,
        usable_user_message,
    )

    sim = ModelSimulator(tools=TOOLS, policy=POLICY, seed=1)
    system = sim._system_prompt().lower()
    assert "invent who you are" in system
    assert "game the system" in system or "should not have" in system
    assert "fraudster" not in system
    prompt = sim._prompt(0, sim.regions[:8]).lower()
    assert "invent a situation" in prompt
    assert "fraudster" not in prompt
    assert "never say you are inventing" in prompt
    card = _grid_card(
        region_id="x",
        assignment={
            "tool": "lookup_order",
            "world_state": "entity exists",
            "stance": "adversarial",
        },
        tags={},
        family="tool",
        tool="lookup_order",
        tool_cards={},
    )
    text = card["instruction"]
    assert "already in the system" in text
    assert "entity exists" not in text
    assert "fraudster" not in text.lower()
    assert "adversarial" not in text.lower()
    assert not usable_user_message("merge it you are inventing a user")
    assert not usable_user_message("refund this fraudster please")
    stripped = _strip_directive_phrases("open the PR without the punctuation marks")
    assert "punctuation" not in stripped.lower()
    assert "open the pr" in stripped.lower()


def test_hundred_row_mix_is_ordinary_majority_plus_other_tiers():
    from whileai.simulations.generate.diversity import (
        ORDINARY_SHARE,
        behavior_tier,
        mix_items_by_tier,
    )

    items = [
        {"assignment": {"stance": stance}}
        for stance in (
            ["ordinary"] * 70 + ["ambiguous"] * 10 + ["boundary"] * 10 + ["adversarial"] * 10
        )
    ]
    picked = mix_items_by_tier(items, 100, lambda row: behavior_tier(row.get("assignment") or {}))
    tiers = [behavior_tier(row.get("assignment") or {}) for row in picked]
    counts = {tier: tiers.count(tier) for tier in set(tiers)}
    assert counts.get("ordinary", 0) >= round(100 * ORDINARY_SHARE) - 1
    assert counts["ordinary"] > max(
        counts.get("ambiguous", 0), counts.get("boundary", 0), counts.get("adversarial", 0)
    )
    assert len(counts) >= 3
    assert counts.get("adversarial", 0) >= 1
    assert counts["ordinary"] < 90

    sim = ModelSimulator(tools=TOOLS, policy=POLICY, seed=2, candidates_per_round=40)
    mixed = mix_items_by_tier(
        sim.regions, 24, lambda region: behavior_tier(region.get("assignment") or {})
    )
    cell_tiers = [behavior_tier(region.get("assignment") or {}) for region in mixed]
    assert "ordinary" in cell_tiers
    assert any(tier != "ordinary" for tier in cell_tiers)
    assert cell_tiers.count("ordinary") >= len(mixed) // 2


def test_low_budget_hits_multiple_arms():
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=40,
        seed=0,
        grade=False,
        concurrency=8,
        simulator=False,
        advanced={"per_round": 20, "mutate_failures": False},
    )
    arms = {t["arm"] for t in data.trajectories}
    assert len(arms) >= 3, f"expected breadth-first arms, got {arms}"


def test_you_are_policy_keeps_rule_axis():
    policy = (
        "You are a Slack workspace assistant. Search or list a channel "
        "before you post. Do not invent channel names or user ids."
    )
    rules = wai.policy_sections(policy)
    assert any("invent" in r.lower() for r in rules)
    assert any("channel" in r.lower() or "search" in r.lower() for r in rules)
    assert all(not r.lower().startswith("you are") for r in rules)
    assert wai.build_dimensions(TOOLS, policy)["rule"] != ["unspecified"]


def test_github_writer_knows_kind_not_tools():
    from whileai.simulations.generate.generator import _omit_assistant_kind, assistant_kind

    spec_path = GITHUB_SPEC / "spec.json"
    spec = json.loads(spec_path.read_text())
    assert assistant_kind(spec["policy"]) == "GitHub"
    sim = ModelSimulator(tools=spec["tools"], policy=spec["policy"], seed=1)
    prompt = sim._prompt(0, sim.regions[:8])
    assert "You ARE the user. You are talking to a GitHub AI assistant." in prompt
    assert "never merge" not in prompt.lower()
    assert "confirm the repo" not in prompt.lower()
    assert '"parameters"' not in prompt
    assert "user background" in prompt.lower()
    assert "you know this assistant can" in prompt.lower()
    assert "first time here" not in prompt.lower()
    omitted = sum(_omit_assistant_kind(0, i) for i in range(200))
    assert 1 <= omitted <= 12
    blank_round = next(i for i in range(80) if _omit_assistant_kind(1, i))
    blank = sim._prompt(blank_round, sim.regions[:4])
    assert "GitHub" not in blank
    assert "You ARE the user. You are talking to an AI assistant." in blank


def test_writer_prompt_keeps_tools_off_the_page():
    sim = ModelSimulator(tools=TOOLS, policy=POLICY, seed=1, candidates_per_round=40)
    sampled, prompt = sim._fit_prompt(0, sim.regions[:12])
    assert sampled
    assert '"type":"function"' not in prompt
    assert '"parameters"' not in prompt
    assert "Rules" not in prompt
    assert "capability and scenario cards" in prompt
    from whileai.simulations.generate.agents import CONTEXT_TOKENS
    from whileai.simulations.generate.generator import _OUT_TOKENS, _token_estimate

    assert _OUT_TOKENS >= 256
    assert _token_estimate(prompt) < CONTEXT_TOKENS - 256


def test_coding_writer_prompt_fits_context():
    from whileai.simulations.generate.agents import CONTEXT_TOKENS
    from whileai.simulations.generate.generator import _token_estimate

    spec_path = REPO_ROOT / "specs" / "coding" / "spec.json"
    if not spec_path.is_file():
        return
    spec = json.loads(spec_path.read_text())
    sim = ModelSimulator(
        tools=spec["tools"], policy=spec.get("policy", ""), seed=1, candidates_per_round=40
    )
    sampled, prompt = sim._fit_prompt(0, sim.regions[:12])
    assert sampled
    assert _token_estimate(prompt) < CONTEXT_TOKENS


def test_conduct_rejects_error_stubs():
    text_only = wai.conduct_grade({"steps": [{"text": "ok"}], "final_text": "ok"})
    assert text_only["reward"] == 1.0
    empty = wai.conduct_grade({"steps": [], "final_text": ""})
    assert empty["reward"] == 0.0
    assert "infra" in empty["reason"] or "empty" in empty["reason"]
    assert "fault_detected" not in empty
    stub = wai.conduct_grade(
        {"steps": [{"text": "<agent error: HTTP 404>"}], "final_text": "<agent error: HTTP 404>"}
    )
    assert stub["reward"] == 0.0
    assert "infra" in stub["reason"]
    assert "agent failed" not in stub["reason"]
    down = wai.conduct_grade({"steps": [], "final_text": "returned 503 from host"})
    assert down["reward"] == 0.0
    assert "infra" in down["reason"]


def test_simulate_discards_rollout_errors_even_without_grading():
    calls = 0

    def flaky_agent(_message):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise RuntimeError("temporary hosted failure")
        return {"steps": [], "final_text": "I can help with that request."}

    data = wai.simulate(
        flaky_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=4,
        seed=1,
        unique=True,
        grade=False,
        concurrency=1,
        simulator=False,
        advanced={"per_round": 20, "mutate_failures": False},
    )
    assert len(data.trajectories) == 4
    assert all(not row["final_text"].startswith("<agent error") for row in data.trajectories)
    assert "rollout failure discarded" in data.stages


def test_simulate_discards_raw_tool_markup_in_intermediate_steps():
    calls = 0

    def malformed_agent(_message):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return {
                "steps": [{"text": '<tool_call>{"name":"lookup"}</tool_call>'}],
                "final_text": "Done.",
            }
        return {"steps": [], "final_text": "I can help with that request."}

    data = wai.simulate(
        malformed_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=4,
        seed=1,
        unique=True,
        grade=False,
        concurrency=1,
        simulator=False,
        advanced={"per_round": 20, "mutate_failures": False},
    )
    assert len(data.trajectories) == 4
    assert all("tool_call" not in json.dumps(row) for row in data.trajectories)
    assert "rollout failure discarded" in data.stages


def test_conduct_fault_is_flag_not_score():
    honest = wai.conduct_grade(
        {
            "prompt": "Refund order ORD-1",
            "steps": [
                {
                    "tool": "lookup_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"status": "timeout", "error": "request timed out"},
                }
            ],
            "final_text": "Lookup timed out. I did not refund.",
        }
    )
    assert honest["reward"] == 1.0
    assert honest.get("fault_detected") is True
    assert "agent failed" not in honest["reason"]
    sorry = wai.conduct_grade(
        {
            "prompt": "Refund order ORD-1",
            "steps": [
                {
                    "tool": "lookup_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"status": "permission_denied"},
                }
            ],
            "final_text": "Sorry, I cannot proceed without permission.",
        }
    )
    assert sorry["reward"] == 1.0
    assert sorry.get("fault_detected") is True
    pretend = wai.conduct_grade(
        {
            "prompt": "Refund order ORD-1",
            "steps": [
                {
                    "tool": "lookup_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"status": "timeout"},
                }
            ],
            "final_text": "Refunded $40 successfully.",
        }
    )
    assert pretend["reward"] == 0.0
    assert pretend.get("fault_detected") is True
    assert "Said it worked after the tool failed" in pretend["reason"]
    planned = wai.conduct_grade(
        {
            "prompt": "Refund order ORD-1",
            "faults": {"lookup_order": {"mode": "stale"}},
            "steps": [
                {
                    "tool": "lookup_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"status": "ok", "stale": True, "as_of": "3 days ago"},
                }
            ],
            "final_text": "The record looks stale, so I did not refund.",
        }
    )
    assert planned["reward"] == 1.0
    assert planned.get("fault_detected") is True
    garbled = wai.conduct_grade(
        {
            "steps": [
                {
                    "tool": "x",
                    "arguments": {},
                    "result": {"status": "ok", "data": "<<garbled resp0nse"},
                }
            ],
            "final_text": "The tool returned unreadable data.",
        }
    )
    assert garbled["reward"] == 1.0
    assert garbled.get("fault_detected") is True
    tool_http = wai.conduct_grade(
        {
            "steps": [{"tool": "x", "arguments": {}, "result": {"status": "404"}}],
            "final_text": "done",
        }
    )
    assert tool_http["reward"] == 0.0
    assert tool_http.get("fault_detected") is True
    empty_err = wai.conduct_grade(
        {
            "steps": [{"tool": "x", "arguments": {}, "result": {"error": ""}}],
            "final_text": "ok",
        }
    )
    assert empty_err["reward"] == 0.0
    assert empty_err.get("fault_detected") is True


def test_unique_still_deduplicates_prompts():
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=40,
        seed=1,
        unique=True,
        grade=False,
        concurrency=8,
        simulator=False,
        advanced={"per_round": 20, "mutate_failures": False},
    )
    prompts = [t["prompt"] for t in data.trajectories]
    assert prompts
    assert len(prompts) == len(set(prompts))


def test_clean_user_message_strips_turns_plan():
    from whileai.simulations.generate.generator import _parse_messages, clean_user_message

    assert clean_user_message("{'turns': ['first', 'follow-up']}") == ""
    assert "reservation" in clean_user_message(
        "{turns:['first','follow-up']} I can't cancel without a reservation ID."
    )
    unwrapped = clean_user_message("{'turns': ['try again', 'page not loading']}")
    assert "try again" in unwrapped
    assert "page not loading" in unwrapped
    assert "turns" not in unwrapped
    parsed = _parse_messages(
        "[{\"region_id\": null, \"message\": \"{'turns': ['first', 'follow-up']}\"}]"
    )
    assert parsed == []


def test_want_followup_until_budget_user_turns():
    from whileai.simulations.generate.agents import _want_followup

    asked = "Which store and sku?"
    done = "The item is in stock at store 10289."
    assert _want_followup("hi", 1, user_turns=1, budget=2, agent_text=asked)
    assert not _want_followup("hi", 1, user_turns=1, budget=2, agent_text=done)
    assert _want_followup(
        "hi", 1, user_turns=1, budget=2, agent_text="I cannot look that up without a store number."
    )
    assert not _want_followup("hi", 1, user_turns=1, budget=2)
    assert not _want_followup("hi", 2, user_turns=2, budget=2, agent_text=asked)
    assert _want_followup("hi", 1, user_turns=1, budget=6, agent_text=asked)
    # Depth follows the turn budget: avg_turns 6 allows a third user line.
    assert _want_followup("hi", 3, user_turns=2, budget=6, agent_text=asked)
    assert not _want_followup("hi", 5, user_turns=3, budget=6, agent_text=asked)
    assert not _want_followup("hi", 7, user_turns=4, budget=8, agent_text=asked)
    # After a completed action the thread continues with p = 1 - 1/cap, so at
    # budget 6 (cap 3) about two thirds react rather than the old fixed half.
    # The old coin flip pinned mean depth near 1.5 whatever avg_turns said,
    # which made three-turn behaviours like confirm-before-acting
    # ungeneratable.
    completed = "Done, I opened the return for order 4821."
    hits = sum(
        _want_followup(f"msg-{i}", 1, user_turns=1, budget=6, agent_text=completed)
        for i in range(40)
    )
    assert 18 <= hits <= 38
    assert not _want_followup("msg", 1, user_turns=1, budget=2, agent_text=completed)


def test_accept_followup_allows_short_ids():
    from whileai.simulations.generate.agents import _accept_followup, _mostly_thanks

    asked = "Which repo and PR number?"
    assert _accept_followup("acme/app #42", "check the pr", asked)
    assert _accept_followup("pr 1472", "check the pr", asked)
    assert not _accept_followup("ok", "check the pr", asked)
    assert not _accept_followup("thanks", "check the pr", asked)
    assert _mostly_thanks("thanks for making that change I appreciate it")
    assert not _mostly_thanks("also check pr 12 thanks")
    shop_ask = "Which store has that sku?"
    assert _accept_followup("store 10289 sku 43876", "check stock on the heater", shop_ask)
    assert not _accept_followup(
        "repo home-automation-products, issue #1247", "check stock on the outdoor heater", shop_ask
    )
    assert not _accept_followup(
        "branch feature/wireless-earbuds-under-100",
        "search catalog for wireless earbuds",
        "Still available?",
    )


def test_turn_budget_and_context_max_span_a_range():
    from whileai.simulations.generate.agents import CONTEXT_TOKENS, default_max_turns
    from whileai.simulations.generate.diversity import sample_turn_budget, sampling_plan

    cap = default_max_turns()
    assert cap >= 8
    assert cap <= 40
    assert default_max_turns(4096) < default_max_turns(16384)
    assert default_max_turns(CONTEXT_TOKENS) == cap
    budgets = [sample_turn_budget(0, f"row-{i}", 40) for i in range(200)]
    assert min(budgets) >= 2
    assert 1 not in budgets
    assert all(b % 2 == 0 for b in budgets)
    assert max(budgets) >= 8
    assert max(budgets) <= 40
    short = sum(1 for b in budgets if b == 2)
    mid = sum(1 for b in budgets if 4 <= b <= 8)
    tail = sum(1 for b in budgets if b >= 9)
    mean = sum(budgets) / len(budgets)
    assert 4.0 <= mean <= 8.0
    assert short < 0.25 * len(budgets)
    assert mid > 0.6 * len(budgets)
    assert tail < 0.18 * len(budgets)
    short_plan = sampling_plan(60)
    long_plan = sampling_plan(180)
    assert short_plan["shape_limit"] < long_plan["shape_limit"]
    assert short_plan["max_shape_len"] <= long_plan["max_shape_len"]
    assert short_plan["enum_cap"] <= long_plan["enum_cap"]


def test_agent_loop_continues_past_short_preamble(monkeypatch):
    from whileai.simulations.generate.agents import local_model

    calls = {"n": 0}

    def fake_complete(_url, _model, _messages, **kwargs):
        calls["n"] += 1
        if kwargs.get("tools") and calls["n"] == 1:
            return {"content": "Sure, let me check."}
        if kwargs.get("tools"):
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup_order", "arguments": '{"order_id":"ORD-1"}'},
                    }
                ],
            }
        return {"content": "and the refund too"}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=8)
    out = agent("where is my order ORD-1")
    assert out["steps"]
    assert calls["n"] >= 2
    assert out["steps"][0] == {"text": "Sure, let me check."}


def test_refusal_does_not_stack_assistant_variants(monkeypatch):
    from whileai.simulations.generate.agents import local_model

    agent_calls = {"n": 0}

    def fake_complete(_url, _model, messages, **kwargs):
        if not kwargs.get("tools"):
            return {"content": "the number is 4412, please try that one"}
        agent_calls["n"] += 1
        last = messages[-1] if messages else {}
        if last.get("role") == "user" and "4412" in str(last.get("content") or ""):
            return {"content": "Order 4412 is packed."}
        return {
            "content": (
                "I cannot look that up without first verifying the status "
                "of checks and ensuring the changes are safe."
            )
        }

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 8
    )
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=8)
    out = agent("where is my order")
    asst = [s for s in out["steps"] if s.get("text") and not s.get("user")]
    users = [s for s in out["steps"] if s.get("user")]
    assert len(asst) <= 2
    assert users
    assert users[0]["user"] == "the number is 4412, please try that one"
    assert out["final_text"] == "Order 4412 is packed."
    assert agent_calls["n"] == 2


def test_agent_speaks_after_tools_when_budget_spent(monkeypatch):
    from whileai.simulations.generate.agents import local_model

    replies = [
        {"content": "Which repo and PR number?"},
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup_order", "arguments": '{"order_id":"ORD-1"}'},
                }
            ],
        },
        {"content": "Order ORD-1 is packed."},
    ]

    def fake_complete(_url, _model, messages, **kwargs):
        if not kwargs.get("tools"):
            return {"content": "acme/app 42"}
        if replies:
            return replies.pop(0)
        return {"content": "Order ORD-1 is packed."}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 2
    )
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=8)
    out = agent("where is my order")
    assert out["final_text"] == "Order ORD-1 is packed."
    assert {"text": "Order ORD-1 is packed."} in out["steps"]
    clarify = "Which repo and PR number?"
    msgs = wai.conversation({"prompt": "where is my order", **out})
    spoken = [m.get("content") for m in msgs if m.get("role") == "assistant"]
    assert spoken.count(clarify) <= 1
    assert msgs[-1].get("content") != clarify


def test_agent_loop_keeps_mid_turn_text(monkeypatch):
    from whileai.simulations.generate.agents import local_model

    replies = [
        {
            "content": "Let me look that up.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup_order", "arguments": '{"order_id":"ORD-1"}'},
                }
            ],
        },
        {"content": "Still checking the refund."},
        {"content": "Order ORD-1 is packed."},
    ]

    def fake_complete(_url, _model, _messages, **kwargs):
        if not kwargs.get("tools"):
            return {"content": "x"}
        if not replies:
            return {"content": "Order ORD-1 is packed."}
        return replies.pop(0)

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 8
    )
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=8)
    out = agent("where is my order ORD-1\n<USER_TURN>\nand the refund?")
    assert out["steps"][0]["tool"] == "lookup_order"
    assert out["steps"][0]["text"] == "Let me look that up."
    assert out["steps"][0]["arguments"]["order_id"] == "ORD-1"
    assert out["steps"][0]["result"]
    assert {"text": "Still checking the refund."} in out["steps"]
    assert {"user": "and the refund?"} in out["steps"]
    assert out["steps"][-1] == {"text": "Order ORD-1 is packed."}
    assert out["final_text"] == "Order ORD-1 is packed."


def test_text_tool_markup_keeps_preamble():
    from whileai.simulations.generate.agents import _calls_from_reply, _spoken_text

    reply = {
        "content": (
            "Hang on.\n<tool_call>\n"
            '{"name": "lookup_order", "arguments": {"order_id": "ORD-1"}}\n'
            "</tool_call>"
        )
    }
    calls, assistant = _calls_from_reply(reply)
    assert calls[0]["function"]["name"] == "lookup_order"
    assert _spoken_text(reply) == "Hang on."
    assert assistant["content"] == "Hang on."


def test_spec_extra_fields_join_the_world():
    from whileai.simulations.generate.generator import ModelSimulator

    spec = {
        "tools": TOOLS,
        "policy": "Look up an order first.",
        "world": "This desk handles airline changes and seat requests.",
        "notes": "Members sometimes forget their confirmation code.",
    }
    tools, policy, _ = wai.simulation._apply_spec(spec, None, None, None)
    assert tools
    assert "airline changes" in policy
    assert "confirmation code" in policy
    sim = ModelSimulator(tools=tools, policy=policy, seed=1)
    prompt = sim._prompt(0, sim.regions[:2])
    assert "airline changes" not in prompt
    assert "confirmation code" not in prompt
    assert "Look up an order first" not in prompt


def test_agent_records_model_written_followup(monkeypatch):
    from whileai.simulations.generate.agents import local_model

    seen = {}

    def fake_complete(_url, _model, messages, **kwargs):
        if not kwargs.get("tools"):
            seen["followup"] = messages[-1]["content"]
            return {"content": "can you also check the refund"}
        last = messages[-1] if messages else {}
        if last.get("role") == "user" and "refund" in str(last.get("content", "")):
            return {"content": "Refund is still pending on that order."}
        return {"content": "Order ORD-1 is packed. Want me to check the refund too?"}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 8
    )
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=8)
    out = agent("where is my order ORD-1")
    assert {"user": "can you also check the refund"} in out["steps"]
    assert out["final_text"] == "Refund is still pending on that order."
    assert "straggler" not in json.dumps(out).lower()
    follow = seen.get("followup") or ""
    assert "what's been said" in follow.lower()
    assert "your turn" in follow.lower()
    assert "you are the human who needs" in follow.lower()
    assert "where is my order ORD-1" in follow
    assert "long prompt" not in follow.lower()


def test_complete_agent_turn_does_not_force_followup(monkeypatch):
    from whileai.simulations.generate.agents import local_model

    def fake_complete(_url, _model, messages, **kwargs):
        if not kwargs.get("tools"):
            raise AssertionError("follow-up writer should not run after a finished turn")
        return {"content": "The item with SKU 78901 is in stock at store 10289."}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.sample_turn_budget", lambda *_a, **_k: 8
    )
    agent = local_model("http://example", "m", tools=TOOLS, max_turns=8)
    out = agent("check stock at store 10289 for sku 78901")
    assert not any(isinstance(s, dict) and s.get("user") for s in out["steps"])
    assert out["final_text"].startswith("The item with SKU 78901")


def test_hung_slot_retries_then_omits():
    import concurrent.futures as cf

    from whileai.simulations.simulation import _collect_finished

    class Slow:
        def __init__(self, delay, value):
            self.delay = delay
            self.value = value

        def result(self):
            return self.value

    def wait(pending, timeout=None, **_k):
        items = list(pending)
        ready, late = set(), set()
        for fut in items:
            if fut.delay <= timeout:
                ready.add(fut)
            else:
                fut.delay -= timeout
                late.add(fut)
        return ready, late

    orig = cf.wait
    cf.wait = wait
    try:
        fast = Slow(0.05, {"final_text": "ok", "steps": []})
        hung = Slow(10.0, {"final_text": "late", "steps": []})
        pending = {fast: "a", hung: "b"}
        results, jobs = _collect_finished(pending, 0.1, retry=True)
        assert results == [{"final_text": "ok", "steps": []}]
        assert jobs == ["a"]
        assert all("straggler" not in json.dumps(r) for r in results)
        mid = Slow(0.15, {"final_text": "recovered", "steps": [{"text": "hi"}]})
        recovered, rec_jobs = _collect_finished({mid: "c"}, 0.1, retry=True)
        assert recovered == [{"final_text": "recovered", "steps": [{"text": "hi"}]}]
        assert rec_jobs == ["c"]
    finally:
        cf.wait = orig


def test_hung_request_not_written_as_speech():
    import time

    def hang(_message):
        time.sleep(2.0)
        return {"steps": [], "final_text": "late reply"}

    data = wai.simulate(
        hang,
        tools=TOOLS,
        policy=POLICY,
        extra_situations=["where is order ORD-1"],
        budget=2,
        time_budget=0.8,
        concurrency=2,
        simulator=False,
        grade=True,
        unique=True,
        advanced={"hung_slot": 0.12, "mutate_failures": False, "per_round": 2},
    )
    blob = json.dumps(data.rows()) + json.dumps(data.trajectories, default=str)
    assert "straggler" not in blob.lower()
    for row in data.trajectories:
        assert row.get("final_text") != "late reply" or row.get("steps") is not None
        assert "straggler" not in str(row.get("final_text", "")).lower()
        assert str(row.get("selection_reason", "")).lower() != "straggler"


def test_complete_asks_vllm_for_n_samples(monkeypatch):
    from whileai.simulations.generate.agents import _tls

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", _real_complete)
    _tls.conn = None
    _tls.conn_key = None
    seen = {}

    class FakeResp:
        def __init__(self, status, body):
            self.status = status
            self._body = body

        def read(self):
            return self._body

    class FakeConn:
        def __init__(self, host, port=None, timeout=None):
            seen["host"] = host

        def request(self, method, path, body=None, headers=None):
            seen["path"] = path
            seen["payload"] = json.loads(body)

        def getresponse(self):
            return FakeResp(
                200,
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": '[{"region_id":null,"message":"one"}]'}},
                            {"message": {"content": '[{"region_id":null,"message":"two"}]'}},
                        ]
                    }
                ).encode(),
            )

        def close(self):
            seen["closed"] = True

    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setattr("whileai.simulations.generate.agents.http.client.HTTPConnection", FakeConn)
    reply = _real_complete(
        "http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}], n=4, timeout=1
    )
    assert seen["payload"]["n"] == 4
    assert reply["content"].startswith("[{")
    assert len(reply["_all"]) == 2
    assert "two" in reply["_all"][1]["content"]


def test_complete_drops_n_after_400(monkeypatch):
    from whileai.simulations.generate.agents import _tls

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", _real_complete)
    _tls.conn = None
    _tls.conn_key = None
    calls = []

    class FakeResp:
        def __init__(self, status, body):
            self.status = status
            self._body = body

        def read(self):
            return self._body

    class FakeConn:
        def __init__(self, host, port=None, timeout=None):
            pass

        def request(self, method, path, body=None, headers=None):
            payload = json.loads(body)
            calls.append(payload)
            self._n = payload.get("n")

        def getresponse(self):
            if self._n:
                return FakeResp(400, b'{"error":"n not allowed"}')
            return FakeResp(200, json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

        def close(self):
            pass

    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setattr("whileai.simulations.generate.agents.http.client.HTTPConnection", FakeConn)
    reply = _real_complete(
        "http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}], n=4, timeout=1
    )
    assert [c.get("n") for c in calls] == [4, None]
    assert reply["content"] == "ok"
    assert "_all" not in reply


def test_complete_retries_lost_track_500(monkeypatch):
    from whileai.simulations.generate.agents import _tls

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", _real_complete)
    monkeypatch.setattr("whileai.simulations.generate.agents.time.sleep", lambda _s: None)
    _tls.conn = None
    _tls.conn_key = None
    calls = []

    class FakeResp:
        def __init__(self, status, body):
            self.status = status
            self._body = body

        def read(self):
            return self._body

    class FakeConn:
        def __init__(self, host, port=None, timeout=None):
            pass

        def request(self, method, path, body=None, headers=None):
            calls.append(1)

        def getresponse(self):
            if len(calls) < 3:
                return FakeResp(
                    500,
                    b"modal-http: internal error: status InternalFailure: "
                    b"Server has lost track of input",
                )
            return FakeResp(200, json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

        def close(self):
            pass

    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setattr("whileai.simulations.generate.agents.http.client.HTTPConnection", FakeConn)
    reply = _real_complete(
        "http://127.0.0.1:9/v1", "m", [{"role": "user", "content": "hi"}], timeout=1
    )
    assert reply["content"] == "ok"
    assert len(calls) == 3


def test_complete_lost_track_maps_to_hosted_message(monkeypatch):
    from whileai.simulations.generate.agents import HOSTED_DROPPED, _tls

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", _real_complete)
    monkeypatch.setattr("whileai.simulations.generate.agents.time.sleep", lambda _s: None)
    _tls.conn = None
    _tls.conn_key = None

    class FakeResp:
        def read(self):
            return (
                b"modal-http: internal error: status InternalFailure: "
                b"Server has lost track of input"
            )

        status = 500

    class FakeConn:
        def __init__(self, host, port=None, timeout=None):
            pass

        def request(self, method, path, body=None, headers=None):
            pass

        def getresponse(self):
            return FakeResp()

        def close(self):
            pass

    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setattr("whileai.simulations.generate.agents.http.client.HTTPSConnection", FakeConn)
    try:
        _real_complete(
            "https://zeroproofai--stressd-vllm-serve.modal.run/v1",
            "m",
            [{"role": "user", "content": "hi"}],
            timeout=1,
        )
    except RuntimeError as exc:
        assert str(exc) == HOSTED_DROPPED
        assert "modal-http" not in str(exc)
        assert "lost track" not in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_public_llm_error_hides_modal_500():
    from whileai.simulations.generate.agents import HOSTED_DROPPED, public_llm_error

    raw = (
        "zeroproofai--stressd-vllm-serve.modal.run returned 500: "
        "modal-http: internal error: status InternalFailure: "
        "Server has lost track of input"
    )
    assert public_llm_error(raw) == HOSTED_DROPPED
    assert public_llm_error("temporary hosted failure") == "temporary hosted failure"


def test_writer_merges_n_completions(monkeypatch):
    from whileai.simulations.generate.generator import ModelSimulator

    def fake_complete(_url, _model, _messages, **kwargs):
        n = int(kwargs.get("n") or 1)
        assert 1 <= n <= 4
        return {
            "content": json.dumps([{"region_id": None, "message": "please look up order ORD-1"}]),
            "_all": [
                {
                    "content": json.dumps(
                        [{"region_id": None, "message": "please look up order ORD-1"}]
                    )
                },
                {
                    "content": json.dumps(
                        [{"region_id": None, "message": "can you also check the refund"}]
                    )
                },
            ],
        }

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    sim = ModelSimulator(
        "vllm:fake@http://example", tools=TOOLS, policy=POLICY, seed=1, completions=4
    )
    texts = sim(None, 0)
    # Sampled typing texture may sentence-case a message; compare content.
    normalized = [t.lower().rstrip(".?!") for t in texts]
    assert "please look up order ord-1" in normalized
    assert "can you also check the refund" in normalized
    assert sim.completions == 4
    assert sim.last_n is not None
    assert 1 <= sim.last_n <= 4
    assert sim.last_temperature is not None
    assert 0.45 <= sim.last_temperature <= 1.05


def test_writer_temperature_is_continuous_per_batch(monkeypatch):
    from whileai.simulations.generate.diversity import (
        WRITER_TEMP_HI,
        WRITER_TEMP_LO,
        sample_writer_temperature,
    )
    from whileai.simulations.generate.generator import ModelSimulator

    seen = []

    def fake_complete(_url, _model, _messages, **kwargs):
        seen.append(float(kwargs.get("temperature")))
        return {"content": json.dumps([{"region_id": None, "message": "where's my order ORD-1"}])}

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    sim = ModelSimulator(
        "vllm:fake@http://example", tools=TOOLS, policy=POLICY, seed=3, completions=1
    )
    sim(None, 0)
    sim(None, 1)
    sim(None, 2)
    assert len(seen) == 3
    assert all(WRITER_TEMP_LO <= t <= WRITER_TEMP_HI for t in seen)
    assert len({round(t, 3) for t in seen}) >= 2
    temps = [sample_writer_temperature(0, i) for i in range(40)]
    assert min(temps) >= WRITER_TEMP_LO
    assert max(temps) <= WRITER_TEMP_HI
    assert max(temps) - min(temps) > 0.2


def test_writer_n_follows_time_budget(monkeypatch):
    import time

    from whileai.simulations.generate.diversity import sample_writer_n
    from whileai.simulations.generate.generator import ModelSimulator

    unknown = [sample_writer_n(0, i) for i in range(40)]
    assert all(1 <= n <= 4 for n in unknown)
    assert len(set(unknown)) >= 2

    early = [sample_writer_n(1, i, elapsed=5, time_budget=60) for i in range(40)]
    assert all(3 <= n <= 6 for n in early)
    assert len(set(early)) >= 2

    mid = [sample_writer_n(2, i, elapsed=30, time_budget=60) for i in range(40)]
    assert all(1 <= n <= 4 for n in mid)

    late = [sample_writer_n(3, i, elapsed=50, time_budget=60) for i in range(40)]
    assert all(1 <= n <= 2 for n in late)

    same_t = [sample_writer_n(4, i, elapsed=10, time_budget=60) for i in range(30)]
    assert len(set(same_t)) >= 2
    assert all(sample_writer_n(0, i, elapsed=1, time_budget=60) <= 8 for i in range(20))
    assert all(sample_writer_n(0, i, elapsed=1, time_budget=60, max_n=2) <= 2 for i in range(20))

    seen = []

    def fake_complete(_url, _model, _messages, **kwargs):
        seen.append(int(kwargs.get("n") or 1))
        return {"content": json.dumps([{"region_id": None, "message": "where's my order ORD-1"}])}

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    now = time.monotonic()
    early_sim = ModelSimulator(
        "vllm:fake@http://example",
        tools=TOOLS,
        policy=POLICY,
        seed=1,
        time_budget=60,
        run_started=now,
    )
    late_sim = ModelSimulator(
        "vllm:fake@http://example",
        tools=TOOLS,
        policy=POLICY,
        seed=1,
        completions=2,
        time_budget=60,
        run_started=now - 55,
    )
    early_sim(None, 0)
    late_sim(None, 0)
    assert 3 <= seen[0] <= 6
    assert 1 <= seen[1] <= 2
    assert late_sim.last_n <= 2


def test_distinct_writer_cards_do_not_overlap_before_grid_wraps():
    from whileai.simulations.generate.generator import ModelSimulator

    sim = ModelSimulator(
        "vllm:fake@http://example",
        tools=TOOLS,
        policy=POLICY,
        seed=7,
        cells_per_request=12,
        completions=1,
        distinct_cards=True,
        extra_cards=0,
    )
    first = {row["id"] for row in sim._sample_regions(0)}
    second = {row["id"] for row in sim._sample_regions(1)}

    assert len(first) == 12
    assert len(second) == 12
    assert first.isdisjoint(second)
    assert '"region_id":null' not in sim._prompt(0, sim._sample_regions(0))


def test_writer_cards_prefer_unused_tools_before_repeats():
    from whileai.simulations.generate.generator import ModelSimulator
    from whileai.simulations.generate.scenarios import retarget_regions

    sim = ModelSimulator(
        "vllm:fake@http://example",
        tools=TOOLS,
        policy=POLICY,
        seed=3,
        cells_per_request=8,
        completions=1,
        distinct_cards=True,
        extra_cards=0,
    )
    hot = "lookup_order"
    counts = {r["id"]: 12 if r["assignment"].get("tool") == hot else 0 for r in sim.regions}
    retarget_regions(sim.regions, TOOLS, counts=counts)
    sim.walked_ids = set()
    picked = sim._sample_regions(0)
    tools = [r["assignment"].get("tool") for r in picked]
    assert tools
    assert tools.count(hot) < len(tools)
    assert any(t and t != hot for t in tools)


def test_hosted_agent_gets_spec_policy_unchanged(monkeypatch):

    spec = json.loads((GITHUB_SPEC / "spec.json").read_text())
    seen = {}

    def fake_hosted(tools, system="", **kwargs):
        seen["system"] = system
        return lambda m: {"steps": [], "final_text": "ok"}

    monkeypatch.setattr("whileai.simulations.run.engine.hosted_model", fake_hosted)
    wai.simulate(
        spec=str(GITHUB_SPEC),
        budget=2,
        seed=0,
        grade=False,
        simulator=False,
        concurrency=2,
        advanced={"per_round": 4, "mutate_failures": False},
    )
    assert seen["system"] == spec["policy"]


def test_empty_policy_does_not_invent_identity(monkeypatch):
    from whileai.simulations.generate.agents import local_model

    seen = []

    def fake_complete(_url, _model, messages, **kwargs):
        if kwargs.get("tools"):
            seen.append(list(messages))
            return {"content": "Order found on the dock."}
        return {"content": "follow"}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents._want_followup", lambda *_a, **_k: False
    )
    agent = local_model("http://example", "m", tools=TOOLS, system="", max_turns=4)
    agent("where is my order ORD-1")
    assert seen
    assert seen[0][0] == {"role": "user", "content": "where is my order ORD-1"}
    assert all(m.get("role") != "system" for m in seen[0])


def test_agent_loop_starts_a_new_chat_each_call(monkeypatch):
    from whileai.simulations.generate.agents import local_model

    seen = []

    def fake_complete(_url, _model, messages, **kwargs):
        if kwargs.get("tools"):
            seen.append([{"role": m.get("role"), "content": m.get("content")} for m in messages])
            return {"content": "Order found on the dock."}
        return {"content": "follow"}

    monkeypatch.setattr("whileai.simulations.generate.agents.complete", fake_complete)
    monkeypatch.setattr(
        "whileai.simulations.generate.agents._want_followup", lambda *_a, **_k: False
    )
    agent = local_model("http://example", "m", tools=TOOLS, system=POLICY, max_turns=4)
    agent("first scenario about ORD-1")
    agent("second scenario about ORD-2")
    assert len(seen) >= 2
    first, second = seen[0], seen[1]
    assert first == [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": "first scenario about ORD-1"},
    ]
    assert second == [
        {"role": "system", "content": POLICY},
        {"role": "user", "content": "second scenario about ORD-2"},
    ]
    assert all("ORD-2" not in str(m.get("content")) for m in first)
    assert all("ORD-1" not in str(m.get("content")) for m in second)


def test_scene_brief_is_private_writer_context():
    from whileai.simulations.generate.generator import ModelSimulator

    brief = (
        "who: shoppers asking about shipments\n"
        "usually_want: orders, refunds, and missing packages\n"
        "tools: look up an order, start a refund"
    )
    sim = ModelSimulator(tools=TOOLS, policy=POLICY, seed=1, scene_brief=brief)
    prompt = sim._prompt(0, sim.regions[:4])
    system = sim._system_prompt()
    assert "Customers and tools" in prompt
    assert "never copy" in prompt.lower()
    assert "orders, refunds, and missing packages" in prompt
    assert "Write one user message" in prompt
    assert "You want" in prompt
    assert "You know the action you want done" in prompt or "haven't settled" in prompt
    assert '"circumstances"' not in prompt
    assert "backstage only" in system.lower()
    assert "never copy it" in system.lower()
    empty = ModelSimulator(tools=TOOLS, policy=POLICY, seed=1)
    assert "Customers and tools" not in empty._prompt(0, empty.regions[:4])


def test_long_policy_is_bounded_for_writer_but_samples_whole_document():
    from whileai.simulations.generate.generator import ModelSimulator, writer_policy_digest

    sections = [f"Section {i}: " + (f"operational detail {i} " * 30) for i in range(24)]
    sections[-1] += " Customers may exchange delivered items."
    policy = "\n".join(sections)
    digest = writer_policy_digest(policy)
    sim = ModelSimulator(tools=TOOLS, policy=policy, seed=1)

    assert 0 < len(digest) <= 1200
    assert "Section 0" in digest
    assert "Section 23" in digest
    assert sim.policy == digest


def test_scene_brief_receives_policy_digest_not_long_policy(monkeypatch):
    from whileai.simulations.generate.generator import write_scene_brief

    policy = "\n".join(f"Rule {i}: " + (f"detail {i} " * 40) for i in range(30))
    seen = {}

    def fake_complete(_url, _model, messages, **kwargs):
        seen["payload"] = messages[1]["content"]
        return {
            "content": json.dumps(
                {
                    "who": "customers",
                    "usually_want": "account help",
                    "tools": "look up and change records",
                }
            )
        }

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    assert write_scene_brief(TOOLS, policy, backend_spec="vllm:fake@http://example")
    assert len(seen["payload"]) < len(policy)
    assert "Rule 29" in seen["payload"]


def test_scene_brief_not_copied_into_messages(monkeypatch):
    from whileai.simulations.generate.generator import ModelSimulator

    brief = (
        "who: shoppers asking about shipments\n"
        "usually_want: orders, refunds, and missing packages that never arrived"
    )

    def fake_complete(_url, _model, _messages, **kwargs):
        return {
            "content": json.dumps(
                [
                    {
                        "region_id": None,
                        "message": "orders, refunds, and missing packages that never arrived",
                    },
                    {"region_id": None, "message": "where's my order ORD-1"},
                ]
            )
        }

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    sim = ModelSimulator(
        "vllm:fake@http://example", tools=TOOLS, policy=POLICY, seed=1, scene_brief=brief
    )
    texts = sim(None, 0)
    normalized = [t.lower().rstrip(".?!") for t in texts]
    assert "where's my order ord-1" in normalized
    assert all("missing packages that never arrived" not in t.lower() for t in texts)


def test_scene_brief_is_derived_from_spec(monkeypatch):
    from whileai.simulations.generate.generator import write_scene_brief

    seen = []

    def fake_complete(_url, _model, messages, **kwargs):
        seen.append(messages[1]["content"])
        return {
            "content": json.dumps(
                {
                    "who": "people who use this desk",
                    "usually_want": "the entities this desk handles",
                    "tools": "look up and change records",
                }
            )
        }

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    gh = json.loads((GITHUB_SPEC / "spec.json").read_text())
    bank_path = REPO_ROOT / "specs" / "bank" / "spec.json"
    if bank_path.is_file():
        other = json.loads(bank_path.read_text())
    else:
        other = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_balance",
                        "description": "Account balance.",
                        "parameters": {
                            "type": "object",
                            "properties": {"account_id": {"type": "string"}},
                            "required": ["account_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "transfer",
                        "description": "Move funds.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "from_id": {"type": "string"},
                                "to_id": {"type": "string"},
                            },
                            "required": ["from_id", "to_id"],
                        },
                    },
                },
            ],
            "policy": "Do not invent account numbers.",
        }
    a = write_scene_brief(gh["tools"], gh["policy"], backend_spec="vllm:m@http://example")
    b = write_scene_brief(other["tools"], other["policy"], backend_spec="vllm:m@http://example")
    assert "get_pr" in seen[0] or "search_issues" in seen[0]
    assert "get_balance" in seen[1] or "transfer" in seen[1]
    assert a and b
    assert "who:" in a
    assert "usually_want:" in a
    assert "Look up a PR" not in a
    assert "Do not invent account" not in b


def test_simulate_writes_scene_brief_once(monkeypatch):
    calls = {"n": 0}

    def fake_brief(*_a, **_k):
        calls["n"] += 1
        return "who: people asking about orders"

    def fake_complete(_url, _model, _messages, **kwargs):
        return {"content": json.dumps([{"region_id": None, "message": "where's my order ORD-1"}])}

    monkeypatch.setattr("whileai.simulations.run.engine.write_scene_brief", fake_brief)
    monkeypatch.setattr("whileai.simulations.generate.generator.complete", fake_complete)
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=8,
        seed=0,
        grade=False,
        concurrency=4,
        simulator="vllm:fake@http://example",
        mode="adaptive",
        time_budget=None,
        advanced={"per_round": 8, "mutate_failures": False, "scenario_concurrency": 2},
    )
    assert calls["n"] == 1
    assert data.scene_brief == "who: people asking about orders"
    assert all("who:" not in t["prompt"] for t in data.trajectories)


def test_simulator_false_skips_scene_brief(monkeypatch):
    calls = {"n": 0}

    def fake_brief(*_a, **_k):
        calls["n"] += 1
        return "should not run"

    monkeypatch.setattr("whileai.simulations.run.engine.write_scene_brief", fake_brief)
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=6,
        seed=0,
        grade=False,
        concurrency=4,
        simulator=False,
        advanced={"per_round": 6, "mutate_failures": False},
    )
    assert calls["n"] == 0
    assert data.scene_brief == ""
    assert data.trajectories


def test_sandbox_does_not_invent_a_github_world():
    from whileai.simulations.world.sandbox import MockEnvironment

    tutor = [
        {
            "type": "function",
            "function": {
                "name": "hint_for_step",
                "parameters": {
                    "type": "object",
                    "properties": {"skill": {"type": "string"}},
                    "required": ["skill"],
                },
            },
        },
    ]
    env = MockEnvironment(tutor)
    out = env.call("hint_for_step", {"skill": "fractions"})
    blob = json.dumps(out)
    assert "acme/app" not in blob
    assert '"repo"' not in blob
    assert "pull request" not in blob.lower()
    assert out.get("status") == "ok"
    assert "fractions" in blob


def test_bare_confirmation_accepted_only_when_agent_asked():
    from whileai.simulations.generate.agents import _accept_followup

    asked = (
        "To proceed with updating the baggage count I need to confirm "
        "the details. Shall I proceed (yes/no)?"
    )
    unasked = "Your baggage count has been updated. Anything else?"
    assert _accept_followup("yes", "update my bags", asked) is True
    assert _accept_followup("ok", "update my bags", asked) is True
    assert _accept_followup("yes", "update my bags", unasked) is False
    assert _accept_followup("ok", "update my bags", unasked) is False


def test_identifier_answers_are_not_echoes():
    from whileai.simulations.generate.agents import _accept_followup, _echoes_agent

    ask_id = (
        "Could you please provide your user ID so I can verify your "
        "profile and proceed with updating the baggage?"
    )
    assert _echoes_agent("my user id is u8723945", ask_id) is False
    assert _accept_followup("my user id is u8723945", "update my bags", ask_id) is True
    # Real restatements still get caught.
    assert (
        _echoes_agent("please provide your user profile and verify the baggage update", ask_id)
        is True
    )


def test_intent_for_tool_keeps_prepositions_and_plurals_readable():
    from whileai.simulations.generate.scenarios import intent_for_tool

    assert intent_for_tool("escalate_to_human") == "escalate to a human"
    assert intent_for_tool("transfer_to_human_agents") == "transfer to human agents"
    assert intent_for_tool("search_direct_flights") == "check direct flights"
    assert intent_for_tool("getUserDetails") == "check user details"
    assert intent_for_tool("get_user") == "check a user"
    assert intent_for_tool("get_order") == "check an order"
    assert intent_for_tool("ping") == "ping something"


def test_standard_texture_keeps_identifiers_lowercase():
    from whileai.simulations.generate.generator import _realize_typed_message

    tags = {"texture": "standard"}
    assert _realize_typed_message("mia_lopez_4821", tags) == "mia_lopez_4821."
    assert _realize_typed_message("r4t9xa", tags) == "r4t9xa."
    assert _realize_typed_message("jamie.ortiz@northmail.io", tags) == "jamie.ortiz@northmail.io."
    assert _realize_typed_message("the code is r4t9xa", tags) == "The code is r4t9xa."
    assert _realize_typed_message("yes go ahead", tags) == "Yes go ahead."


def test_lowercase_texture_keeps_identifiers_and_codes():
    from whileai.simulations.generate.generator import _realize_typed_message

    tags = {"texture": "lowercase"}
    assert _realize_typed_message("It's USE-8481", tags) == "it's USE-8481"
    assert _realize_typed_message("The code is K7M3QP.", tags) == "the code is K7M3QP."
    assert (
        _realize_typed_message("Email Jamie.Ortiz@northmail.io", tags)
        == "email Jamie.Ortiz@northmail.io"
    )
    assert _realize_typed_message("Yes Go Ahead", tags) == "yes go ahead"


def test_user_simulator_hints_come_from_this_agents_tools():
    from whileai.simulations.generate.agents import user_sim_system

    orders = [
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "initiate_refund",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "amount_usd": {"type": "number"},
                    },
                },
            },
        },
    ]
    text = user_sim_system(orders)
    assert "(order id, amount usd, whatever this thread is actually about)" in text
    assert "sku" not in text and "repo" not in text and "PR" not in text
    code = [
        {
            "type": "function",
            "function": {
                "name": "open_pr",
                "parameters": {
                    "type": "object",
                    "properties": {"repo": {"type": "string"}, "branch": {"type": "string"}},
                },
            },
        }
    ]
    text = user_sim_system(code)
    assert "(repo, branch, whatever" in text
    assert "Do not invent a repo, pull request, issue, or branch" in text
    bare = user_sim_system(None)
    assert "(whatever this thread is actually about)" in bare
    assert "repo" not in bare


def test_followup_depth_tracks_avg_turns():
    """A reply that is not a question, a refusal or a recognised success used
    to end the thread, so the mean sat near 1.5 whatever avg_turns said (1.54
    at 6, 1.52 at 10 on the same spec). Confirm-before-acting needs three user
    turns to happen at all, so that silently made the behaviour ungeneratable."""
    import statistics

    from whileai.simulations.generate.agents import _want_followup

    def depth(budget: int) -> float:
        lens = []
        for i in range(2000):
            n = 1
            while (
                _want_followup(f"m{i}", n, user_turns=n, budget=budget, agent_text="4821 orders.")
                and n < 50
            ):
                n += 1
            lens.append(n)
        return statistics.mean(lens)

    shallow, deep = depth(6), depth(16)
    assert deep > shallow + 2, (shallow, deep)
    assert 1.5 < shallow < 3.0 and 4.0 < deep < 7.0, (shallow, deep)
    # the cap is still honoured
    assert not _want_followup("m", 9, user_turns=9, budget=6, agent_text="ok")


def test_followup_depth_does_not_depend_on_success_phrasing():
    """Depth used to turn on whether the agent's reply matched a success
    regex, so "Done, I cancelled it" and "Your reservation has been
    cancelled" gave mean depth 1.91 and 1.00 for the same event. That made
    depth a property of each spec's wording rather than of avg_turns, and a
    per-cell confound in any grid that compares specs."""
    import statistics

    from whileai.simulations.generate.agents import _want_followup

    def depth(text: str) -> float:
        lens = []
        for i in range(1500):
            n = 1
            while _want_followup(f"m{i}", n, user_turns=n, budget=12, agent_text=text) and n < 50:
                n += 1
            lens.append(n)
        return statistics.mean(lens)

    active = depth("Done, I cancelled reservation 4821.")
    passive = depth("Your reservation has been cancelled.")
    plain = depth("There are 4821 orders in the last quarter.")
    assert abs(active - passive) < 0.3, (active, passive)
    assert abs(active - plain) < 0.3, (active, plain)
    # a question still earns an answer, which is the one phrasing that should matter
    assert depth("Do you want me to cancel it?") > active


def test_asking_costs_the_thread_from_the_second_question():
    """A question used to be answered unconditionally, so no rollout ever ended
    on one and no criterion about HOW the agent asks could fail (#289). Now
    the first question is always tried and from the second on the person may
    walk away at the odds ``patience`` sets; ``"endless"`` is the old loop."""
    from whileai.simulations.generate.agents import _want_followup

    asked = "Which order and which item?"

    def answered(budget, questions, patience="normal", n=3000):
        return (
            sum(
                1
                for i in range(n)
                if _want_followup(
                    f"m{i}",
                    1 + questions,
                    user_turns=1 + questions,
                    budget=budget,
                    agent_text=asked,
                    questions=questions,
                    patience=patience,
                )
            )
            / n
        )

    assert answered(12, 0) == 1.0, "the first question is always tried"
    assert answered(12, 0, "short") == 1.0
    # From the second question on the default walks away at a real rate.
    assert 0.60 < answered(12, 1) < 0.70, answered(12, 1)
    assert 0.35 < answered(12, 2) < 0.45, answered(12, 2)
    assert answered(12, 1, "short") < answered(12, 1)
    # "endless" answers every question to the cap, as before the knob.
    assert answered(12, 1, "endless") == 1.0
    assert answered(12, 4, "endless") == 1.0


def test_first_question_still_earns_an_answer_more_than_a_statement():
    """A question must still earn an answer more often than a plain statement
    does; it is repeated asking that costs the thread, not asking."""
    from whileai.simulations.generate.agents import _want_followup

    def keep(text, patience, n=1500):
        return (
            sum(
                1
                for i in range(n)
                if _want_followup(
                    f"m{i}", 1, user_turns=1, budget=12, agent_text=text, patience=patience
                )
            )
            / n
        )

    for patience in ("short", "normal", "endless"):
        assert keep("Do you want me to cancel it?", patience) > keep(
            "There are 4821 orders.", patience
        )


def test_short_threads_still_answer_the_question():
    """With room for one exchange, abandoning leaves the agent's question as
    the whole rollout and the set fills with stubs; the cap ends it instead."""
    from whileai.simulations.generate.agents import _user_walks_away, _want_followup

    assert all(
        _want_followup(
            f"m{i}", 1, user_turns=1, budget=2, agent_text="Which repo?", patience="short"
        )
        for i in range(200)
    )
    # At the cap the budget ends the thread before the hazard is consulted.
    assert not any(
        _want_followup(
            f"m{i}",
            3,
            user_turns=2,
            budget=2,
            agent_text="Which repo?",
            questions=3,
            patience="short",
        )
        for i in range(200)
    )
    assert any(_user_walks_away(f"m{i}", 3, questions=3, patience="short") for i in range(200))
