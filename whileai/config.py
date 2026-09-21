"""Settings once, override per call: ``configure()`` and ``context()``.

Where a call goes is decided in this order, first match wins:

1. the keyword on the call itself (``simulate(agent=...)``, ``Judge(model=...)``);
2. the innermost ``with wai.context(...)`` block;
3. what ``wai.configure(...)`` set for the process;
4. the environment (``WHILEAI_AGENT``, ``OPENAI_API_KEY``, ...);
5. the package default: the model While hosts, on the key ``whileai login`` saved.

    import whileai as wai

    wai.configure(
        agent=wai.OpenAI("gpt-4.1-mini", api_key="sk-..."),
        judge=wai.Anthropic("claude-haiku-4-5", api_key="sk-ant-..."),
    )
    print(wai.settings)   # says which model and which key each role uses

A key given on a backend object is kept for that provider, so every call
to that provider in the process finds it. ``api_key=`` on ``configure``
is the While account key, for the hosted models and the platform.

This module imports nothing from the engine so the engine can import it.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROLES = ("agent", "judge", "simulator")

#: The spec forms the engine reads, provider -> the shape to type. The
#: engine's ``parse_backend_spec`` is the implementation; this is the
#: vocabulary the front door checks a string against, so a typo is refused
#: by the call that took it. ``tests/api/test_facade.py`` pins the two
#: together, so a provider added there has to be added here.
SPEC_FORMS = {
    "openai": "openai:<model>",
    "anthropic": "anthropic:<model>",
    "fireworks": "fireworks:<model>",
    "bedrock": "bedrock:<model-id>[@<region>]",
    "vllm": "vllm:<model>@<url>",
    "ollama": "ollama:<model>",
    "typesafe": "typesafe:<model> (judge only)",
}

#: Per role, the words that name a route rather than a model.
#: ``simulator="hosted"`` (or ``"default"``) is the writer default written
#: out, which ``run.config.writer_spec_for`` reads as "the hosted writer".
ROLE_SENTINELS = {"simulator": ("hosted", "default")}


def spec_problem(value: str, *, kwarg: str = "") -> str | None:
    """The one sentence that fixes ``value``, or ``None`` when the engine
    can read it.

    A model is named ``provider:model``. The two near misses are the
    spelling DSPy and LiteLLM use, ``provider/model``
    (``dspy.LM("openai/gpt-4o-mini")``), and the bare model name an
    OpenAI user types. Both used to be accepted here and to fail several
    calls later, from a call the user did not write, so this says which
    string to type instead.
    """
    text = value.strip()
    named = f"{kwarg}={value!r}" if kwarg else repr(value)
    forms = ", ".join(SPEC_FORMS.values())
    tail = f"pass {forms}, an http(s) URL, or a callable"
    if not text:
        return f"{named} names no model; {tail}, or leave it unset for the model While hosts"
    if text.startswith(("http://", "https://")):
        return None
    if text.lower() in ROLE_SENTINELS.get(kwarg, ()):
        return None
    provider, colon, _model = text.partition(":")
    if colon and provider in SPEC_FORMS:
        return None
    slashed, slash, rest = text.partition("/")
    if slash and slashed in SPEC_FORMS and rest:
        # ``openai`` is also a Hugging Face org (``openai/gpt-oss-120b``),
        # so name the vllm reading too rather than assume the typo.
        return (
            f"{named} separates the provider from the model with a slash, the spelling DSPy "
            f"and LiteLLM use; whileai uses a colon, so pass "
            f"{kwarg + '=' if kwarg else ''}{f'{slashed}:{rest}'!r}"
            f" (or 'vllm:{text}@<url>' if that is a repo id on a server you run)"
        )
    if colon:
        return f"{named} names no provider whileai reaches ({provider!r} is not one); {tail}"
    return f"{named} names no provider, so nothing says where the call goes; {tail}"


def spec_of(value: Any, *, kwarg: str = "") -> Any:
    """The backend spec string for ``value``: a backend object's ``.spec``,
    a string as given, a callable as given, ``None`` for "the default".

    A string is checked against ``SPEC_FORMS`` first, so a misspelled
    model is refused by the call that took it (rule 10).
    """
    if isinstance(value, str):
        problem = spec_problem(value, kwarg=kwarg)
        if problem:
            raise ValueError(problem)
        return value
    if value is None or callable(value):
        return value
    spec = getattr(value, "spec", None)
    if spec is None or isinstance(spec, str):
        return spec
    raise TypeError(f"not a backend: {value!r}")


@dataclass
class Settings:
    """The process-wide answers to "which model" and "which key"."""

    agent: str | None = None
    judge: str | None = None
    simulator: str | None = None
    #: the While account key (hosted models, platform); ``whileai login`` sets it too
    api_key: str | None = None
    #: provider -> key, filled from backend objects: ``openai``, ``anthropic``,
    #: ``bedrock`` (a Bedrock API key), ``vllm``, ``typesafe``. ``ollama``
    #: never needs one.
    keys: dict[str, str] = field(default_factory=dict)

    def key_for(self, provider: str) -> str | None:
        return self.keys.get(provider)

    def __repr__(self) -> str:
        def show(role: str) -> str:
            value = getattr(self, role)
            return value if value else "default (While hosted)"

        parts = [f"{role}={show(role)}" for role in ROLES]
        parts.append(f"api_key={'set' if self.api_key else 'unset'}")
        if self.keys:
            parts.append("keys=" + ",".join(sorted(self.keys)))
        return "Settings(" + ", ".join(parts) + ")"


_base = Settings()
_local = threading.local()


def _stack() -> list[Settings]:
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    return stack


def current() -> Settings:
    """The settings in force: the innermost ``context()`` or the process base."""
    stack = _stack()
    return stack[-1] if stack else _base


def _check(**roles: Any) -> None:
    """Refuse every bad spec before any of them is applied, so a typo in
    ``judge=`` does not leave ``agent=`` set."""
    for role, value in roles.items():
        if value is not None:
            spec_of(value, kwarg=role)


def _absorb(target: Settings, role: str, value: Any) -> None:
    """Record a backend (or spec string) under ``role``, and its key if it has one."""
    provider = getattr(value, "provider", None)
    key = getattr(value, "api_key", None)
    spec = spec_of(value, kwarg=role)
    if provider and key:
        target.keys[str(provider)] = str(key)
    setattr(target, role, spec)


def resolve_backend(value: Any, *, kwarg: str = "") -> Any:
    """``configure(<role>=value)`` for one call: a backend object becomes
    the spec string the engine reads, and a key given on it is kept for
    its provider in the settings in force, exactly as ``configure`` keeps
    it. The role default is not touched. Anything that is not a backend
    object (a spec string, a URL, a callable, a wrapped agent, ``None``)
    comes back as given, so ``simulate(agent=wai.OpenAI("gpt-4.1-mini"))``
    and ``simulate(agent="openai:gpt-4.1-mini")`` are the same call.
    """
    from .models import Backend

    if not isinstance(value, Backend):
        return value
    spec = spec_of(value, kwarg=kwarg)
    if value.provider and value.api_key:
        current().keys[str(value.provider)] = str(value.api_key)
    return spec


def configure(
    *,
    agent: Any = None,
    judge: Any = None,
    simulator: Any = None,
    api_key: str | None = None,
) -> Settings:
    """Set the process defaults. Each argument is a backend object
    (``wai.OpenAI(...)``), a spec string (``"openai:gpt-4.1-mini"``) or
    ``None`` to leave that role as it is. Returns the settings, whose repr
    says what each role resolves to.

    * ``agent``: the policy under test.
    * ``judge``: the grader; never the same model as the agent by default.
    * ``simulator``: the writer of user messages; the agent's model unless set.
    * ``api_key``: the While account key.

    A spec string is checked here: ``agent="openai/gpt-4.1-mini"`` (the
    DSPy spelling) or ``agent="gpt-4.1-mini"`` raises and names the string
    to type, instead of failing later inside ``simulate``.
    """
    _check(agent=agent, judge=judge, simulator=simulator)
    if agent is not None:
        _absorb(_base, "agent", agent)
    if judge is not None:
        _absorb(_base, "judge", judge)
    if simulator is not None:
        _absorb(_base, "simulator", simulator)
    if api_key is not None:
        _base.api_key = str(api_key).strip() or None
    return _base


def reset() -> Settings:
    """Forget everything ``configure()`` set. Tests call this."""
    global _base
    _base = Settings()
    _stack().clear()
    return _base


@contextlib.contextmanager
def context(
    *,
    agent: Any = None,
    judge: Any = None,
    simulator: Any = None,
    api_key: str | None = None,
) -> Iterator[Settings]:
    """Override the settings inside a ``with`` block, on this thread only.

    with wai.context(judge=wai.OpenAI("gpt-4.1")):
        strict = data.grade(wai.Judge(rubric=RUBRIC))
    """
    _check(agent=agent, judge=judge, simulator=simulator)
    scoped = replace(current(), keys=dict(current().keys))
    if agent is not None:
        _absorb(scoped, "agent", agent)
    if judge is not None:
        _absorb(scoped, "judge", judge)
    if simulator is not None:
        _absorb(scoped, "simulator", simulator)
    if api_key is not None:
        scoped.api_key = str(api_key).strip() or None
    stack = _stack()
    stack.append(scoped)
    try:
        yield scoped
    finally:
        stack.pop()


class _SettingsProxy:
    """``wai.settings``: always the settings in force, never a stale copy."""

    def __getattr__(self, name: str) -> Any:
        return getattr(current(), name)

    def __repr__(self) -> str:
        return repr(current())


settings = _SettingsProxy()

#: Folders a wheel is installed into. A package directory under one of
#: these is the installed package; anywhere else is a checkout.
INSTALL_DIRS = ("site-packages", "dist-packages")


def _editable_root() -> Path | None:
    """The directory ``pip install -e`` (or ``uv sync``) installed ``whileai``
    from, read from the distribution's ``direct_url.json`` (PEP 610), or
    ``None`` when the installed copy is a wheel or there is none."""
    from importlib.metadata import distributions
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    for dist in distributions():
        if (dist.metadata["Name"] or "").lower() != "whileai":
            continue
        raw = dist.read_text("direct_url.json")
        if not raw:
            continue
        try:
            info = json.loads(raw)
        except ValueError:
            continue
        if info.get("dir_info", {}).get("editable") and info.get("url", "").startswith("file:"):
            return Path(url2pathname(urlparse(info["url"]).path)).resolve()
    return None


def _provenance_line(version: str, where: Path, editable_root: Path | None) -> str:
    """The line ``provenance()`` prints, from the three facts it reads."""
    line = f"whileai {version} from {where}"
    if any(part in INSTALL_DIRS for part in where.parts):
        return line
    if editable_root is not None and where.parent == editable_root:
        return f"{line} (source tree, installed editable)"
    return f"{line} (source tree, not the installed wheel)"


def provenance() -> str:
    """Which ``whileai`` this process imported, as one line: ``whileai
    <version> from <directory>``, and when the directory is a checkout
    rather than a ``site-packages`` install, ``(source tree, installed
    editable)`` after ``pip install -e .`` or ``(source tree, not the
    installed wheel)`` when the checkout is shadowing a wheel.

    A clone of the SDK has a ``whileai/`` folder at its root, and Python
    puts the working directory first on ``sys.path`` for ``python -m``,
    ``python -c``, a notebook and ``modal run``, so a recipe started from
    the repository root can import the clone instead of the wheel ``pip``
    installed, with no message either way. Every recipe prints this line
    first, on stderr so stdout stays the result, and the Modal recipes
    mount whichever tree it names. Run a recipe from its own directory to
    use the installed package, or ``pip install -e .`` to make the tree
    the installed package. The version is the installed distribution's,
    which is the wheel's while a checkout shadows it.

        >>> import whileai as wai
        >>> print(wai.config.provenance())  # doctest: +SKIP
        whileai 0.110 from /home/me/whileai-sdk/whileai (source tree, not the installed wheel)
    """
    import whileai

    where = Path(whileai.__file__).resolve().parent
    return _provenance_line(whileai.__version__, where, _editable_root())


def requirement() -> str:
    """The ``pip`` requirement that gives a remote container at least the
    ``whileai`` this process imported: ``whileai>=<version>``.

    A bare ``"whileai"`` in a container image is resolved once, when the
    image layer is first built, and cached under that spelling: the
    container keeps whatever was newest that day until the layer key
    changes, while the laptop moves on. The first symptom is an
    ``AttributeError`` for a call the laptop has and the container's older
    wheel does not. Writing the version into the requirement makes each
    release a new layer key and makes the drift visible in the image
    definition. It is a floor, not a pin, so a checkout whose version is
    already on the index installs, and so does the next release.

    When the distribution is not installed (``__version__`` is
    ``0.0.0``) the bare name is returned, because no floor is known.

        >>> import whileai as wai
        >>> wai.config.requirement()  # doctest: +SKIP
        'whileai>=0.110'
        >>> image = modal.Image.debian_slim().pip_install(  # doctest: +SKIP
        ...     "torch==2.7.1", "trl==0.19.1", wai.config.requirement()
        ... )
    """
    import whileai

    version = str(whileai.__version__ or "").strip()
    if not version or version == "0.0.0":
        return "whileai"
    return f"whileai>={version}"


__all__ = [
    "Settings",
    "configure",
    "context",
    "current",
    "provenance",
    "requirement",
    "reset",
    "resolve_backend",
    "settings",
    "spec_of",
]
