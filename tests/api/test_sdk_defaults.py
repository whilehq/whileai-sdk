import inspect
import threading
import time

import whileai.simulations as wai
from tests.helpers import GITHUB_SPEC, POLICY, TOOLS, scripted_agent, simulate_offline


def _lower_headers(headers):
    return {str(key).lower(): value for key, value in dict(headers).items()}


def test_default_budget_is_500():
    sig = inspect.signature(wai.simulate)
    params = sig.parameters
    public = [
        "agent",
        "spec",
        "tools",
        "system_prompt",
        "budget",
        "time_budget",
        "until",
        "mode",
        "situations",
        "requests_per_situation",
        "rollouts_per_request",
        "unique_situations",
        "reproducible",
        "grade",
        "llm_grade",
        "traces",
        "grader",
        "rubric",
        "strategy",
        "seeds",
        "scaffold",
        "execute",
        "output",
        "tasks",
        "runs",
        "advanced",
        # the common knobs, named so an editor shows them (same road as advanced=)
        "repeats",
        "phrasings",
        "repeat_policy",
        "concurrency",
        "simulator",
        "user_model",
        "backend",
        "seed",
        "sampling",
        "max_turns",
        "avg_turns",
        "fault_rate",
        "temperature",
        "timeout",
        "logprobs",
    ]
    named = [name for name, p in params.items() if p.kind is not inspect.Parameter.VAR_KEYWORD]
    assert named == public
    assert params["budget"].default == 1000
    assert params["until"].default == "compute"
    assert params["mode"].default == "explore"
    assert params["requests_per_situation"].default is None
    assert params["rollouts_per_request"].default is None
    assert params["unique_situations"].default is False
    assert params["situations"].default is None
    assert params["grade"].default is False
    assert params["llm_grade"].default is False
    assert params["time_budget"].default is None
    assert params["spec"].default is None
    assert params["output"].default is None
    assert params["advanced"].default is None
    # On 2026-09-17 the knobs most runs touch (repeats, repeat_policy,
    # concurrency, simulator, seed, ...) graduated to named params after a
    # usability test; they still travel through the same merge. The rest
    # stay advanced-only.
    for moved in (
        "dimensions",
        "risk",
        "texture",
        # grader graduated from moved-kwarg to a named param on
        # 2026-08-27 (doctrine sketch).
        "llm_spec",
        "embedder",
        "unique",
        "n",
        "extra_situations",
        "rollouts_per_prompt",
        "policy",
    ):
        assert moved not in params
    assert "length" not in params
    assert "turns" not in params
    assert "seconds" not in params


def test_platform_delegated_credential_helpers(monkeypatch):
    seen = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"credential": "zp_dc_123", "expiresAt": "2026-08-26T13:00:00Z"}'

    def fake_urlopen(req, timeout=120):
        seen.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "headers": _lower_headers(req.headers),
                "body": req.data.decode() if req.data else None,
            }
        )
        return FakeResponse()

    monkeypatch.setenv("WHILEAI_API_URL", "https://example.test")
    monkeypatch.setattr("whileai.simulations.ingest.platform.urllib.request.urlopen", fake_urlopen)

    out = wai.issue_delegated_credential("clerk.jwt.abc", ttl_seconds=900)
    assert out["credential"] == "zp_dc_123"
    assert seen[0]["url"] == "https://example.test/auth/issue-credential"
    assert seen[0]["headers"]["authorization"] == "Bearer clerk.jwt.abc"
    assert "x-api-key" not in seen[0]["headers"]

    refreshed = wai.refresh_delegated_credential("clerk.jwt.abc", "zp_dc_123", ttl_seconds=1800)
    assert refreshed["credential"] == "zp_dc_123"
    assert seen[1]["url"] == "https://example.test/auth/refresh-credential"
    assert seen[1]["headers"]["authorization"] == "Bearer clerk.jwt.abc"


def test_platform_call_falls_back_to_api_key_for_blank_auth_token(monkeypatch):
    seen = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=120):
        seen.append(
            {
                "url": req.full_url,
                "headers": _lower_headers(req.headers),
            }
        )
        return FakeResponse()

    monkeypatch.setenv("WHILEAI_API_URL", "https://example.test")
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_test_key")
    monkeypatch.setattr("whileai.simulations.ingest.platform.urllib.request.urlopen", fake_urlopen)

    from whileai.simulations.ingest.platform import _call

    _call("GET", "/datasets", None, auth_token="   ")

    assert seen[0]["url"] == "https://example.test/datasets"
    assert seen[0]["headers"].get("x-api-key") == "zp_test_key"
    assert "authorization" not in seen[0]["headers"]


def test_platform_call_can_send_bearer_and_api_key_together(monkeypatch):
    seen = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=120):
        seen.append(
            {
                "url": req.full_url,
                "headers": _lower_headers(req.headers),
            }
        )
        return FakeResponse()

    monkeypatch.setenv("WHILEAI_API_URL", "https://example.test")
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_test_key")
    monkeypatch.setattr("whileai.simulations.ingest.platform.urllib.request.urlopen", fake_urlopen)

    from whileai.simulations.ingest.platform import _call

    _call("GET", "/datasets", None, auth_token="clerk.jwt.abc", require_api_key=True)

    assert seen[0]["url"] == "https://example.test/datasets"
    assert seen[0]["headers"]["authorization"] == "Bearer clerk.jwt.abc"
    assert seen[0]["headers"].get("x-api-key") == "zp_test_key"


def test_system_prompt_alias_policy():
    data = simulate_offline(budget=4, system_prompt=POLICY, per_round=6)
    assert data.profile.system_prompt == POLICY
    alias = simulate_offline(budget=4, per_round=6)
    assert alias.profile.policy == POLICY
    try:
        simulate_offline(budget=1, system_prompt="a", policy="b")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "system_prompt" in str(exc)


def test_adaptive_defaults_follow_allocator_not_n1_k1():
    data = simulate_offline(budget=4, mode="adaptive", per_round=6)
    plan = wai.adaptive_allocator(None, "compute")
    assert data.mode == "adaptive"
    assert data.requests_per_situation == plan["n_req"]
    assert data.rollouts_per_request == plan["k"]
    assert data.requests_per_situation > 1
    assert data.rollouts_per_request > 1


def test_time_budget_none_or_zero_is_unlimited():
    for knob in (0, None):
        data = simulate_offline(budget=8, time_budget=knob, per_round=6)
        assert len(data.trajectories) == 8
        assert data.stopped_because != "time_budget"


def test_unique_no_duplicate_prompt():
    data = simulate_offline(budget=40, unique=True, concurrency=8, per_round=20)
    prompts = [t["prompt"] for t in data.trajectories]
    assert prompts
    assert len(prompts) == len(set(prompts))


def test_repeats_two_same_prompt():
    data = simulate_offline(budget=4, repeats=2, per_round=12)
    assert len(data.trajectories) == 4
    prompts = [t["prompt"] for t in data.trajectories]
    assert len(set(prompts)) == 2
    for prompt in set(prompts):
        rows = [t for t in data.trajectories if t["prompt"] == prompt]
        assert len(rows) == 2
        assert {t["rollout_index"] for t in rows} == {0, 1}


def test_missing_spec_path_is_clear():
    try:
        wai.simulate(spec="specs/definitely-missing-xyz")
    except FileNotFoundError as exc:
        assert "definitely-missing-xyz" in str(exc)
        assert "spec.json" in str(exc)
        assert "\n" not in str(exc)
        assert str(exc).rstrip(".").count(".") == 1
    else:
        raise AssertionError("expected FileNotFoundError")


def test_empty_simulate_is_one_sentence():
    try:
        wai.simulate()
    except ValueError as exc:
        assert "agent" in str(exc) and "system prompt" in str(exc)
        assert "\n" not in str(exc)
        assert str(exc).count(".") <= 1
    else:
        raise AssertionError("expected ValueError")


def test_github_example_spec_works():
    data = simulate_offline(spec=str(GITHUB_SPEC), budget=6, grade=True, per_round=6)
    names = {(t.get("function") or t).get("name") for t in data.profile.tools}
    assert {"search_issues", "get_pr"} <= names
    row = data.rows()[0]
    assert {"prompt", "messages", "steps", "final_text", "scenario_id"} <= set(row)
    # grade=True applies the deterministic conduct grade at return.
    assert isinstance(row.get("reward"), (int, float))
    assert row.get("label_source") == "conduct"
    assert row["messages"][0] == {"role": "user", "content": row["prompt"]}
    # #149: the exported row carries how it was drawn, not a projection
    assert row["arm"]
    assert "privileged" not in row


def test_hosted_key_message_is_one_sentence(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from whileai.simulations.generate.agents import missing_hosted_key

    msg = missing_hosted_key("https://zeroproofai--stressd-vllm-serve.modal.run/v1")
    assert msg is not None
    assert "VLLM_API_KEY" in msg
    assert "\n" not in msg


def test_touch_hosted_gets_models(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    seen = {}

    class FakeResp:
        status = 200

        def read(self):
            return b'{"data":[]}'

    class FakeConn:
        def __init__(self, host, port=None, timeout=None):
            seen["host"] = host

        def request(self, method, path, headers=None):
            seen["method"] = method
            seen["path"] = path
            seen["auth"] = (headers or {}).get("Authorization")

        def getresponse(self):
            return FakeResp()

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr("whileai.simulations.generate.agents.http.client.HTTPConnection", FakeConn)
    from whileai.simulations.generate.agents import touch_hosted

    touch_hosted("http://127.0.0.1:9/v1")
    assert seen["method"] == "GET"
    assert seen["path"] == "/v1/models"
    assert seen["auth"] == "Bearer test-key"


def test_touch_hosted_skips_without_key(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(
        "whileai.simulations.generate.agents.http.client.HTTPSConnection",
        lambda *a, **k: called.append(True),
    )
    from whileai.simulations.generate.agents import touch_hosted

    touch_hosted("https://zeroproofai--stressd-vllm-serve.modal.run/v1")
    assert called == []


def test_ping_hosted_false_on_500(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-key")

    class FakeResp:
        status = 500

        def read(self):
            return b"lost track of input"

    class FakeConn:
        def __init__(self, *a, **k):
            pass

        def request(self, *a, **k):
            pass

        def getresponse(self):
            return FakeResp()

        def close(self):
            pass

    monkeypatch.setattr("whileai.simulations.generate.agents.http.client.HTTPSConnection", FakeConn)
    from whileai.simulations.generate.agents import ping_hosted

    assert ping_hosted("https://zeroproofai--stressd-vllm-serve.modal.run/v1") is False


def test_ping_hosted_false_on_connection_error(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-key")

    class FakeConn:
        def __init__(self, *a, **k):
            raise TimeoutError("timed out")

    monkeypatch.setattr("whileai.simulations.generate.agents.http.client.HTTPSConnection", FakeConn)
    from whileai.simulations.generate.agents import ping_hosted

    assert ping_hosted("https://zeroproofai--stressd-vllm-serve.modal.run/v1") is False


def test_output_does_not_wipe_when_no_rows(tmp_path):
    dest = tmp_path / "out.jsonl"
    dest.write_text('{"keep": true}\n')
    try:
        wai.simulate(spec="specs/definitely-missing-xyz", output=str(dest))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
    assert dest.read_text().startswith('{"keep"')


def test_rollouts_per_prompt_alias():
    data = simulate_offline(budget=4, rollouts_per_prompt=2, per_round=12)
    assert len(data.trajectories) == 4
    assert len({t["prompt"] for t in data.trajectories}) == 2


def test_scenario_producers_run_concurrently():
    lock = threading.Lock()
    active = 0
    peak = 0

    def writer(_dataset=None, index=0):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return [f"human request {index}-{i}" for i in range(20)]

    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=120,
        seed=0,
        unique=True,
        grade=False,
        concurrency=32,
        simulator=writer,
        until="budget_only",
        advanced={"scenario_concurrency": 4, "scenarios_per_request": 12, "mutate_failures": False},
    )

    assert peak >= 2
    assert len(data.trajectories) == 120


def test_unique_writer_flight_stays_small():
    lock = threading.Lock()
    peak = 0
    active = 0

    def writer(_dataset=None, index=0):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return [f"human request {index}-{i}" for i in range(16)]

    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=40,
        seed=0,
        unique=True,
        grade=False,
        concurrency=8,
        simulator=writer,
        until="budget_only",
        advanced={"mutate_failures": False},
    )
    assert 1 <= peak <= 4
    assert len(data.trajectories) == 40


def test_unique_enables_distinct_model_cards(monkeypatch):
    seen = []

    class FakeModel:
        regions = []

    def fake_generator(*args, **kwargs):
        seen.append(kwargs.get("distinct_cards"))

        def writer(_dataset=None, index=0, include_model=True):
            return [f"human request {index}-{i}" for i in range(8)]

        writer.model = FakeModel()
        writer.meta = {}
        writer.provenance = {}
        writer.last_candidate_provenance = {}
        writer.fault_plans = {}
        writer.last_errors = {}
        writer.regions = []
        writer.arm_weights = {}
        writer.model_produced = True
        return writer

    monkeypatch.setattr("whileai.simulations.run.engine.make_default_generator", fake_generator)
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=8,
        seed=0,
        unique=True,
        grade=False,
        concurrency=2,
        until="budget_only",
        advanced={"mutate_failures": False},
    )
    assert len(data.trajectories) == 8
    assert seen and all(seen)


def test_refill_does_not_stall_inflight_rollouts():
    def writer(_dataset=None, index=0):
        if int(index or 0) > 0:
            time.sleep(1.4)
        return [f"human request {index}-{i}" for i in range(8)]

    t0 = time.monotonic()
    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=16,
        seed=0,
        unique=True,
        grade=False,
        concurrency=8,
        simulator=writer,
        until="budget_only",
        time_budget=3.0,
        advanced={"mutate_failures": False, "scenario_concurrency": 2},
    )
    elapsed = time.monotonic() - t0
    assert len(data.trajectories) == 16
    assert elapsed < 2.8


def test_unique_dupes_do_not_stop_as_generator_exhausted():
    calls = {"n": 0}

    def writer(_dataset=None, index=0):
        calls["n"] += 1
        n = calls["n"]
        if n <= 3:
            return [f"human request {index}-{i}-{n}" for i in range(6)]
        return ["please refund this order now"] * 6

    data = wai.simulate(
        scripted_agent,
        tools=TOOLS,
        policy=POLICY,
        budget=200,
        seed=0,
        unique=True,
        grade=False,
        concurrency=8,
        simulator=writer,
        until="budget_only",
        time_budget=3.0,
        advanced={"mutate_failures": False, "scenario_concurrency": 4},
    )
    assert data.stopped_because == "time_budget"
    assert data.stopped_because != "generator_exhausted"
    # The writer went dry after three unique rounds. The claim under test is
    # that the loop kept asking it rather than declaring the generator
    # exhausted, so any call past the third proves it. A fixed count would
    # depend on how many writer rounds fit in the wall-clock budget.
    assert calls["n"] > 3
    prompts = [t["prompt"] for t in data.trajectories]
    assert prompts
    assert len(prompts) == len(set(prompts))


def test_generator_exhausted_is_not_emitted():
    from pathlib import Path

    src = Path(wai.__file__).read_text()
    assert "generator_exhausted" not in src


def test_open_ended_weight_cannot_exceed_cap():
    from whileai.simulations.generate.scenarios import cap_open_ended_weight
    from whileai.simulations.simulation import _SEARCH_ARMS, _reallocate

    clipped = cap_open_ended_weight(
        {
            "structured": 0.2,
            "open_ended": 0.4,
            "llm_guided": 0.2,
            "behavior_targeted": 0.1,
            "failure_mutation": 0.1,
        }
    )
    assert clipped["open_ended"] <= 0.10 + 1e-9
    assert abs(sum(clipped.values()) - 1.0) < 1e-9
    weights = dict(_SEARCH_ARMS)
    hot = {arm: 1.0 if arm == "open_ended" else 0.0 for arm in weights}
    for _ in range(20):
        weights = _reallocate(weights, hot)
    assert weights["open_ended"] <= 0.10 + 1e-9
    assert weights["open_ended"] >= 0.05 - 1e-9


def test_open_ended_probe_families_stay_intact():
    from whileai.simulations.generate.scenarios import _PROBE_FAMILIES, open_ended_probes

    names = [name for name, _ in _PROBE_FAMILIES]
    assert {"out_of_domain_factual", "creative", "garbage_input", "prompt_injection"} <= set(names)
    probes = open_ended_probes(TOOLS, POLICY, per_round=16, seed=0)
    blob = " ".join(probes).lower()
    assert "mongolia" in blob or "haiku" in blob or "asdf" in blob
    assert "file number" not in blob
    assert "ignore" in blob


def test_writer_and_agent_default_to_hosted_qwen(monkeypatch):
    from whileai.simulations.generate.agents import (
        DEFAULT_AGENT,
        default_agent_spec,
        default_simulator_spec,
    )

    monkeypatch.delenv("WHILEAI_SURROGATE", raising=False)
    monkeypatch.delenv("WHILEAI_AGENT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert default_simulator_spec() == DEFAULT_AGENT
    assert default_agent_spec() == DEFAULT_AGENT


def test_writer_follows_bring_your_own_model(monkeypatch):
    from whileai.simulations.simulation import writer_spec_for

    monkeypatch.delenv("WHILEAI_SURROGATE", raising=False)
    assert writer_spec_for("openai:gpt-4.1-mini", None) == "openai:gpt-4.1-mini"
    assert writer_spec_for("vllm:m@http://h/v1", None) == "vllm:m@http://h/v1"
    # hosted default, callables, urls, and named writers keep their writer
    assert writer_spec_for(None, None) is None
    assert writer_spec_for(lambda m: {}, None) is None
    assert writer_spec_for("https://agent.example/chat", None) is None
    assert writer_spec_for("openai:x", "openai:y") == "openai:y"
    monkeypatch.setenv("WHILEAI_SURROGATE", "openai:z")
    assert writer_spec_for("openai:x", None) is None


def test_conduct_grade_reason_reaches_the_row(tmp_path):
    import json

    data = simulate_offline(grade=True, budget=16)
    fails = [r for r in data.trajectories if r.get("reward") == 0]
    assert fails, "the scripted agent invents ids and claims success; some rows must fail"
    assert all(r.get("reason") for r in fails)
    out = tmp_path / "rows.jsonl"
    data.save(str(out))
    saved = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r.get("reason") for r in saved if r.get("reward") == 0)


def test_grader_must_be_callable():
    import pytest

    from tests.helpers import offline, scripted_agent

    with pytest.raises(TypeError, match="grader= takes a callable"):
        wai.simulate(scripted_agent, budget=2, grader="hosted", **offline())
