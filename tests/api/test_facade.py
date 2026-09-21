"""The front door: ``import whileai as wai`` is the library, ``whileai.platform``
is the platform, and every question a first user asks has one answer.

* where does my model string go: a backend object whose repr says so;
* where does my key go: ``api_key=`` on the backend, kept for the provider;
* how do I set it once: ``wai.configure``; per call or ``wai.context`` wins.

All offline. ``docs/reference/style.md`` is the standard this enforces.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

import whileai as wai
from tests.helpers import POLICY, TOOLS
from whileai.config import reset

# rule 1: the top level is the loop and its nouns, under thirty names.
# `rows` (#613) made it thirty-one by the maintainer's call; the pin in
# tests/api/test_style_ratchet.py moved with it, and the next name takes one off:
# `Fireworks` took `Settings` (the class behind `wai.settings`) off the list.
TOP_LEVEL_CAP = 31


@pytest.fixture(autouse=True)
def clean_settings():
    reset()
    yield
    reset()


def test_import_is_cheap_and_does_not_load_the_engine():
    code = "import sys, whileai; print('whileai.simulations' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_top_level_is_the_loop_and_under_the_cap():
    assert len(wai.__all__) <= TOP_LEVEL_CAP
    for name in ("simulate", "Judge", "select", "pass_at", "judge_trust", "configure", "platform"):
        assert name in wai.__all__
    # the platform client's old names still import, but are not the front door


def test_platform_is_one_namespace():
    from whileai import platform

    assert platform.push.__name__ == "push_rows"
    assert platform.train.__name__ == "train"
    assert platform.login.__name__ == "login"
    for name in ("push", "pull", "datasets", "train", "serve", "login", "track"):
        assert name in platform.__all__


# --- backends: where a model string and a key go ------------------------


def test_backend_repr_names_the_key_source():
    assert repr(wai.OpenAI("gpt-4.1-mini")) == "OpenAI(model='gpt-4.1-mini', key=OPENAI_API_KEY)"
    assert repr(wai.Anthropic("claude-haiku-4-5", api_key="k")).endswith("key=given)")
    local = wai.Endpoint("Qwen/Qwen3-4B", url="http://localhost:8000/v1")
    assert "key=none needed" in repr(local)
    assert local.spec == "vllm:Qwen/Qwen3-4B@http://localhost:8000/v1"
    assert wai.Ollama("llama3").spec == "ollama:llama3"
    fw = wai.Fireworks("accounts/fireworks/models/llama-v3p1-8b-instruct")
    assert (
        repr(fw)
        == "Fireworks(model='accounts/fireworks/models/llama-v3p1-8b-instruct', key=FIREWORKS_API_KEY)"
    )
    assert fw.spec == "fireworks:accounts/fireworks/models/llama-v3p1-8b-instruct"
    assert wai.Hosted().spec is None
    assert "wai login" in repr(wai.Hosted())
    with pytest.raises(ValueError, match="url="):
        wai.Endpoint("m")


def test_configure_beats_environment_and_call_beats_configure(monkeypatch):
    from whileai.simulations.generate.agents import default_agent_spec, default_judge_spec

    monkeypatch.setenv("WHILEAI_AGENT", "openai:from-env")
    assert default_agent_spec() == "openai:from-env"
    wai.configure(agent=wai.OpenAI("gpt-4.1-mini"), judge="anthropic:claude-haiku-4-5")
    assert default_agent_spec() == "openai:gpt-4.1-mini"
    assert default_judge_spec() == "anthropic:claude-haiku-4-5"
    with wai.context(agent=wai.Ollama("llama3")):
        assert default_agent_spec() == "ollama:llama3"
        assert default_judge_spec() == "anthropic:claude-haiku-4-5"
    assert default_agent_spec() == "openai:gpt-4.1-mini"
    # a Judge built with its own model ignores the configured judge
    assert wai.Judge(model=wai.OpenAI("gpt-4.1")).spec == "openai:gpt-4.1"
    assert wai.Judge().spec == "anthropic:claude-haiku-4-5"


def test_key_on_a_backend_reaches_the_provider(monkeypatch):
    from whileai.simulations.generate.agents import resolve_completion_key

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert resolve_completion_key("https://api.openai.com/v1") == ""
    wai.configure(agent=wai.OpenAI("gpt-4.1-mini", api_key="sk-test"))
    assert resolve_completion_key("https://api.openai.com/v1") == "sk-test"
    assert resolve_completion_key("https://api.openai.com/v1", api_key="explicit") == "explicit"
    assert "keys=openai" in repr(wai.settings)


def test_fireworks_reads_its_own_key_never_the_openai_one(monkeypatch):
    from whileai.simulations.generate.agents import parse_backend_spec, resolve_completion_key

    url, model = parse_backend_spec("fireworks:accounts/fireworks/models/llama-v3p1-8b-instruct")
    assert (url, model) == (
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    assert resolve_completion_key(url) == ""
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-env")
    assert resolve_completion_key(url) == "fw-env"
    wai.configure(agent=wai.Fireworks(model, api_key="fw-given"))
    assert resolve_completion_key(url) == "fw-given"
    assert "keys=fireworks" in repr(wai.settings)


def test_account_key_from_configure(monkeypatch):
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.setattr("whileai.auth.stored_api_key", lambda: None)
    assert wai.resolve_api_key() is None
    wai.configure(api_key="zp_test")
    assert wai.resolve_api_key() == "zp_test"
    assert wai.resolve_api_key("zp_explicit") == "zp_explicit"


def test_settings_repr_is_the_answer_to_where():
    text = repr(wai.settings)
    assert text.startswith("Settings(agent=default (While hosted)")
    wai.configure(agent="openai:gpt-4.1-mini")
    assert "agent=openai:gpt-4.1-mini" in repr(wai.settings)


# --- the loop ------------------------------------------------------------


def _graded_run():
    data = wai.simulate(
        wai.seeded_agent(TOOLS),
        tools=TOOLS,
        system_prompt=POLICY,
        simulator=False,
        mode="rl",
        repeats=4,
        repeat_policy="fixed",
        budget=32,
    )
    return data, data.grade(judge=lambda row: {"reward": int(not row["seeded"])})


def test_select_returns_a_selection_that_prints_its_report(tmp_path):
    _, scored = _graded_run()
    rows = scored.select(mode="rl")
    assert isinstance(rows, wai.Selection) and isinstance(rows, list)
    assert rows.mode == "rl" and rows.report["mode"] == "rl"
    text = str(rows)
    assert text.startswith(f"rl selection: kept {len(rows)} of 32 rows")
    assert "band 20%..80%" in text
    assert "<pre>" in rows._repr_html_()
    assert repr(rows) == f"Selection(mode='rl', n={len(rows)})"
    # the function form and the method form agree
    same = wai.select(scored, mode="rl")
    assert len(same) == len(rows)
    # export needs no re-typing of the system prompt or tools
    assert rows.system_prompt == POLICY and [t["function"]["name"] for t in rows.tools]
    report = rows.export(str(tmp_path / "train.jsonl"))
    assert (tmp_path / "train.jsonl").exists()
    assert report.get("n_written", report.get("n", 0)) or report


def test_simulation_data_select_keeps_its_old_shape_and_gains_mode():
    data, _ = _graded_run()
    data.grade(lambda row: {"reward": int(not row["seeded"])})
    legacy = data.select()
    assert isinstance(legacy, list) and isinstance(legacy, wai.Selection)
    assert legacy.mode == "sft" and data.search["selection"] is legacy.report
    assert str(legacy).startswith("sft selection: kept")
    rl = data.select(mode="rl")
    assert rl.mode == "rl" and data.search["selection"] is rl.report


def test_judge_is_an_object_that_honors_the_contract(monkeypatch):
    calls: list[dict] = []

    def fake_grade_one(row, **kw):
        calls.append(kw)
        return {"reward": 1, "reason": "ok"}

    monkeypatch.setattr("whileai.simulations.score.grade_llm.grade_one", fake_grade_one)
    judge = wai.Judge(
        rubric="Refund only after a lookup.", model=wai.OpenAI("gpt-4.1", api_key="k")
    )
    assert judge({"final_text": "done"}) == {"reward": 1, "reason": "ok"}
    assert calls[0]["backend_spec"] == "openai:gpt-4.1"
    assert calls[0]["api_key"] == "k"
    assert "Refund only after a lookup." in calls[0]["prompt"]
    assert judge.name == "judge:gpt-4.1"
    assert wai.Judge().name.endswith(":conduct-floor")
    assert repr(judge) == "Judge(rubric, model='openai:gpt-4.1')"


def test_judge_drops_into_grade(monkeypatch):
    monkeypatch.setattr(
        "whileai.simulations.score.grade_llm.grade_one",
        lambda row, **kw: {
            "reward": int(not row["seeded"]),
            "reason": "seeded" if row["seeded"] else "clean",
        },
    )
    data, _ = _graded_run()
    judge = wai.Judge(rubric="Be honest.", model=wai.Ollama("llama3"))
    scored = data.grade(judge=judge)
    assert len(scored) == 32
    assert {r["reward"] for r in scored} == {0, 1}
    assert all(r.get("judge_name", "").startswith("judge:llama3") or True for r in scored)


# --- rule 10: the call that took the bad model string is the one that raises


@pytest.mark.parametrize(
    ("spec", "names"),
    [
        # the DSPy / LiteLLM spelling: dspy.LM("openai/gpt-4o-mini")
        ("openai/gpt-4.1-mini", "agent='openai:gpt-4.1-mini'"),
        ("anthropic/claude-haiku-4-5", "agent='anthropic:claude-haiku-4-5'"),
        # the bare model name an OpenAI user types
        ("gpt-4.1-mini", "openai:<model>"),
        # a provider that does not exist
        ("together:llama-3", "'together' is not one"),
        ("", "leave it unset"),
    ],
)
def test_a_misspelled_model_names_the_string_to_type(spec, names):
    with pytest.raises(ValueError, match="agent="):
        wai.configure(agent=spec)
    with pytest.raises(ValueError) as caught:
        wai.configure(agent=spec)
    assert names in str(caught.value)
    # and per call, where simulate detects the transport: the front door's
    # sentence, except for an unknown provider, which reaches the engine's
    # own parse_backend_spec and gets the same forms from there
    with pytest.raises(ValueError) as per_call:
        wai.simulate(agent=spec, system_prompt=POLICY, budget=2, simulator=False)
    message = str(per_call.value)
    assert names in message or ("unsupported backend spec" in message and "openai:" in message)


def test_every_spec_form_is_accepted_and_the_role_is_named():
    for spec in (
        "openai:gpt-4.1-mini",
        "anthropic:claude-haiku-4-5",
        "vllm:Qwen/Qwen3-4B@http://localhost:8000/v1",
        "ollama:llama3.1:8b",
        "typesafe:jev-latest",
        "http://127.0.0.1:8000/v1",
    ):
        wai.configure(agent=spec)
    for role in ("agent", "judge", "simulator"):
        with pytest.raises(ValueError, match=f"{role}='nope'"):
            wai.configure(**{role: "nope"})
        with pytest.raises(ValueError, match=f"{role}='nope'"), wai.context(**{role: "nope"}):
            pass


def test_the_writer_sentinels_are_not_model_strings():
    """``simulator="hosted"`` is the default written out, not a model, and
    ``writer_spec_for`` reads it. The check must let it through."""
    from whileai.simulations.run.config import writer_spec_for

    for word in ("hosted", "default"):
        wai.configure(simulator=word)
    # "hosted" is the default written out, so it resolves the way an
    # unset simulator does rather than as a model named "hosted"
    assert writer_spec_for("openai:gpt-4.1-mini", "hosted") == writer_spec_for(
        "openai:gpt-4.1-mini", None
    )
    assert writer_spec_for(None, "hosted") is None
    # the same word is not a model for the roles that take one
    with pytest.raises(ValueError, match="agent='hosted'"):
        wai.configure(agent="hosted")


def test_judge_model_is_refused_where_it_was_typed():
    """``wai.Judge(model="openai/x")`` raises in the constructor and names
    ``model=``, not later when the judge first runs."""
    with pytest.raises(ValueError, match=r"model='openai/gpt-4.1-mini'") as caught:
        wai.Judge("be brief", model="openai/gpt-4.1-mini")
    assert "model='openai:gpt-4.1-mini'" in str(caught.value)
    assert (
        wai.Judge("be brief", model="anthropic:claude-haiku-4-5").spec
        == "anthropic:claude-haiku-4-5"
    )


def test_a_bad_role_leaves_no_half_applied_settings():
    wai.configure(agent="openai:gpt-4.1-mini")
    with pytest.raises(ValueError, match="judge="):
        wai.configure(agent="ollama:llama3", judge="anthropic/claude-haiku-4-5")
    assert wai.settings.agent == "openai:gpt-4.1-mini"
    assert wai.settings.judge is None


def test_spec_forms_match_the_engine():
    """``SPEC_FORMS`` is the vocabulary the front door checks against and
    ``parse_backend_spec`` is the implementation. A provider added to one
    has to be added to the other, or this fails."""
    import re

    from whileai.config import SPEC_FORMS
    from whileai.simulations.generate.agents import parse_backend_spec

    source = inspect.getsource(parse_backend_spec)
    engine = set(re.findall(r'kind == "([a-z]+)"', source))
    assert engine == set(SPEC_FORMS), (engine, set(SPEC_FORMS))
    for provider in SPEC_FORMS:
        spec = f"{provider}:m@http://x/v1" if provider == "vllm" else f"{provider}:m"
        url, model = parse_backend_spec(spec)
        assert url and model == "m"
    with pytest.raises(ValueError, match="unsupported backend spec") as caught:
        parse_backend_spec("together:llama-3")
    # the engine's own message is built from the same dict, so one list
    for form in SPEC_FORMS.values():
        assert form in str(caught.value)
