"""#472: ``simulate(agent=<Backend>)`` resolves the backend the way
``configure(agent=<Backend>)`` does, positional or keyword, and the
transport error is honest when detection genuinely fails."""

import pytest

import whileai as wai
from tests.helpers import POLICY, TOOLS
from whileai.config import reset
from whileai.simulations import simulation
from whileai.simulations.generate.adapters import detect, resolve


@pytest.fixture(autouse=True)
def clean_settings():
    reset()
    yield
    reset()


@pytest.fixture
def captured(monkeypatch):
    """Swap the engine for a stand-in that records the resolved config, so
    the call gets through transport selection without a model call."""
    configs: list = []

    class FakeRun:
        def __init__(self, cfg):
            configs.append(cfg)

        def run(self):
            return "ran"

    monkeypatch.setattr(simulation, "Run", FakeRun)
    return configs


def test_simulate_takes_a_backend_positionally(captured):
    assert wai.simulate(wai.OpenAI("gpt-4.1-mini"), tools=TOOLS, system_prompt=POLICY) == "ran"
    assert captured[0].agent == "openai:gpt-4.1-mini"


def test_simulate_takes_a_backend_as_keyword(captured):
    wai.simulate(agent=wai.Anthropic("claude-haiku-4-5"), tools=TOOLS, system_prompt=POLICY)
    assert captured[0].agent == "anthropic:claude-haiku-4-5"


def test_backend_on_simulate_matches_configure(captured):
    """The two arguments have the same name; they resolve to the same spec,
    and the per-call form does not change the process default."""
    wai.simulate(wai.OpenAI("gpt-4.1-mini", api_key="sk-test"), tools=TOOLS, system_prompt=POLICY)
    per_call = captured[0].agent
    assert wai.settings.agent is None
    assert wai.settings.key_for("openai") == "sk-test"
    reset()
    wai.configure(agent=wai.OpenAI("gpt-4.1-mini", api_key="sk-test"))
    assert wai.settings.agent == per_call
    assert wai.settings.key_for("openai") == "sk-test"


def test_hosted_backend_is_the_default_route(captured):
    wai.simulate(wai.Hosted(), tools=TOOLS, system_prompt=POLICY)
    assert captured[0].agent is None


def test_backend_kwarg_takes_a_backend_object(captured):
    wai.simulate(tools=TOOLS, system_prompt=POLICY, backend=wai.Ollama("llama3"))
    assert captured[0].backend == "ollama:llama3"


def test_adapters_detect_and_resolve_a_backend():
    assert detect(wai.OpenAI("gpt-4.1-mini")) == "backend_spec"
    runner, kind = resolve(wai.OpenAI("gpt-4.1-mini"), tools=TOOLS, policy=POLICY)
    assert kind == "backend_spec"
    assert callable(runner)


def test_transport_error_names_what_agent_takes():
    with pytest.raises(ValueError) as err:
        detect(object())
    text = str(err.value)
    assert "cannot detect a transport for object" in text
    assert "OpenAI(model)" in text and "provider:model" in text
    assert "tools=." not in text
