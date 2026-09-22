"""whileai: post-training data and evaluation for agents that call tools.

    import whileai as wai

    wai.configure(agent=wai.OpenAI("gpt-4.1-mini"), judge=wai.Anthropic("claude-haiku-4-5"))

    data = wai.simulate(tools=TOOLS, system_prompt=POLICY, mode="rl", repeats=8)
    scored = data.grade(wai.Judge(rubric=RUBRIC))    # or a verifier, or any callable
    print(scored.pass_at)                            # pass@1 with an interval
    print(wai.judge_trust(scored.rows))              # does the judge agree with people
    rows = scored.select(mode="rl")                  # the rows that carry gradient
    rows.export("train.jsonl")

Two domains, kept apart:

* ``whileai`` is the library: simulate, grade, measure, select, export.
  It runs on your machine against your models and needs no account.
* ``whileai.platform`` is the While platform: sign in, push datasets,
  train and serve on our GPUs, track versions. Everything that talks to
  withwhile.com lives there and nowhere else.

Every name here is loaded on first use, so ``import whileai`` stays cheap.
The full engine is one dot down at ``whileai.simulations``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .config import Settings, configure, context, settings
from .models import Anthropic, Backend, Endpoint, Fireworks, Hosted, Ollama, OpenAI

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("whileai")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

# name -> (module, attribute). Resolved on first access (PEP 562) so that
# importing the package does not import the engine.
_LAZY: dict[str, tuple[str, str | None]] = {
    # the loop
    "simulate": ("whileai.simulations.simulation", "simulate"),
    "tool": ("whileai.simulations.tools", "tool"),
    "Tool": ("whileai.simulations.tools", "Tool"),
    "seeded_agent": ("whileai.simulations.generate.offline_agent", "seeded_agent"),
    "Judge": ("whileai.judge", "Judge"),
    "Verifier": ("whileai.simulations.verify", "Verifier"),
    "verifier": ("whileai.simulations.verify", "verifier"),
    "pass_at": ("whileai.simulations.score.passat", "pass_at"),
    "judge_trust": ("whileai.simulations.score.judge_trust", "judge_trust"),
    "compare": ("whileai.simulations.score.delta", "delta_report"),
    # The other half of belief 1 (CONSTITUTION.md): a number is a result
    # with its interval *and its noise floor*. `eval_variance` measures the
    # floor from re-runs, `holdout_size` sizes the set before the GPU runs.
    # The rows reference lists all four calls in one table, so all four
    # resolve from the one import (#662). Their home stays
    # whileai.simulations.score.stats and __all__ below is unchanged.
    "eval_variance": ("whileai.simulations.score.stats", "eval_variance"),
    "holdout_size": ("whileai.simulations.score.stats", "holdout_size"),
    "select": ("whileai.selection", "select"),
    # `Selection` is the type `select` returns; reachable for isinstance,
    # off the advertised list since `Harness` took its slot (#712).
    "Selection": ("whileai.selection", "Selection"),
    # the harness: the program around the model, run like an agent and
    # versioned like weights; `wai.harness.attribute` says which lever moved
    "Harness": ("whileai.harness", "Harness"),
    "harness": ("whileai.harness", None),
    "decontaminate": ("whileai.simulations.score.stats", "decontaminate"),
    "hack_scan": ("whileai.simulations.score.hack_scan", "hack_scan"),
    "preflight": ("whileai.simulations.score.preflight", "preflight"),
    "export": ("whileai.simulations.export", "export_dataset"),
    # an RL environment out of graded rows, and the same package loaded
    # back for a verifiers trainer; docs/reference/harness.md spells both
    # as `wai.X` (#712). Their home stays whileai.simulations.environment.
    "export_environment": ("whileai.simulations.environment", "export_environment"),
    "load_environment": ("whileai.simulations.environment", "load_environment"),
    "SimulationData": ("whileai.simulations.data", "SimulationData"),
    "ScoredData": ("whileai.simulations.score.judging", "ScoredData"),
    # your own prompts and completions (a public benchmark) as the rows
    # every measurement reads (#613)
    "rows": ("whileai.simulations.schema", "rows"),
    # training methods as objects, and the trainer config written from them;
    # their home is whileai.methods
    "OPD": ("whileai.methods", "OPD"),
    "OPSD": ("whileai.methods", "OPSD"),
    "Async": ("whileai.methods", "Async"),
    # the single-rollout methods: one trajectory per prompt, the shape a
    # production trace arrives in
    "FlashReinforce": ("whileai.methods", "FlashReinforce"),
    "SAO": ("whileai.methods", "SAO"),
    "BPCO": ("whileai.methods", "BPCO"),
    "prime_rl_config": ("whileai.methods", "prime_rl_config"),
    # a grader that tells passing rollouts apart (MiMo-V2.6 section 4.3);
    # its home is whileai.groupwise, reachable as wai.methods.GroupwiseGrading too
    "GroupwiseGrading": ("whileai.groupwise", "GroupwiseGrading"),
    # namespaces
    "verify": ("whileai.simulations.verify", None),
    "hub": ("whileai.hub", None),  # push to the Hugging Face Hub with your own token
    "platform": ("whileai.platform", None),
    "methods": ("whileai.methods", None),
    "simulations": ("whileai.simulations", None),
    # the platform client's old top-level names, kept importable; their
    # home is whileai.platform
    "login": ("whileai.auth", "login"),
    "logout": ("whileai.auth", "logout"),
    "signup": ("whileai.auth", "signup"),
    "account": ("whileai.auth", "account"),
    "resolve_api_key": ("whileai.auth", "resolve_api_key"),
    "LoginError": ("whileai.auth", "LoginError"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'whileai' has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY))


if TYPE_CHECKING:  # so editors and mypy see the lazy names
    from . import hub
    from .auth import LoginError, account, login, logout, resolve_api_key, signup
    from .groupwise import GroupwiseGrading
    from .judge import Judge
    from .methods import BPCO, OPD, OPSD, SAO, Async, FlashReinforce, prime_rl_config
    from .selection import Selection, select
    from .simulations import methods, platform, simulations, verify  # type: ignore[attr-defined]
    from .simulations.data import SimulationData
    from .simulations.export import export_dataset as export
    from .simulations.generate.offline_agent import seeded_agent
    from .simulations.schema import rows
    from .simulations.score.delta import delta_report as compare
    from .simulations.score.hack_scan import hack_scan
    from .simulations.score.judge_trust import judge_trust
    from .simulations.score.judging import ScoredData
    from .simulations.score.passat import pass_at
    from .simulations.score.preflight import preflight
    from .simulations.score.stats import decontaminate, eval_variance, holdout_size
    from .simulations.simulation import simulate
    from .simulations.tools import Tool, tool
    from .simulations.verify import Verifier, verifier

# The front door: under thirty names, the loop and its nouns. The platform
# client's names above stay importable but are documented under
# whileai.platform. `Settings` stays importable too; `Fireworks` took its
# place in the list so the front door stays at thirty-one names.
__all__ = [
    "Anthropic",
    "Endpoint",
    "Fireworks",
    "Harness",
    "Hosted",
    "Judge",
    "Ollama",
    "OpenAI",
    "ScoredData",
    "SimulationData",
    "Verifier",
    "__version__",
    "compare",
    "configure",
    "context",
    "decontaminate",
    "export",
    "hack_scan",
    "judge_trust",
    "methods",
    "pass_at",
    "platform",
    "preflight",
    "rows",
    "seeded_agent",
    "select",
    "settings",
    "simulate",
    "tool",
    "verifier",
    "verify",
]
