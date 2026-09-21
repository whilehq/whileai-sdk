"""Where a model call goes, as an object whose repr says so.

Every role in a run (the agent under test, the judge, the simulated user)
is a model behind an endpoint, reached with a key. A backend object holds
the three together, so the question "I passed a model string and a key,
where did they go" has one answer: print the object.

    >>> wai.OpenAI("gpt-4.1-mini")
    OpenAI(model='gpt-4.1-mini', key=OPENAI_API_KEY)
    >>> wai.Endpoint("Qwen/Qwen3-4B", url="http://localhost:8000/v1")
    Endpoint(model='Qwen/Qwen3-4B', url='http://localhost:8000/v1', key=none needed)
    >>> wai.models.Bedrock("us.anthropic.claude-haiku-4-5-20251001-v1:0", region="us-west-2")
    Bedrock(model='us.anthropic.claude-haiku-4-5-20251001-v1:0', region='us-west-2', key=AWS_BEARER_TOKEN_BEDROCK or AWS credentials)

Pass one to ``wai.configure(agent=..., judge=...)``, or straight to
``simulate(agent=...)`` and ``Judge(model=...)``. Internally each is the
spec string the engine already reads (``openai:<model>``,
``anthropic:<model>``, ``fireworks:<model>``, ``bedrock:<model-id>[@<region>]``,
``vllm:<model>@<url>``, ``ollama:<model>``); the
object is the front door, the string still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Backend:
    """A model behind an endpoint, with the key that reaches it."""

    model: str
    api_key: str | None = field(default=None, repr=False)
    provider: str = field(default="", init=False, repr=False)
    #: which environment variable supplies the key when ``api_key`` is not given
    env_key: str | None = field(default=None, init=False, repr=False)

    @property
    def spec(self) -> str:
        """The engine's spec string for this backend."""
        return f"{self.provider}:{self.model}"

    def _key_note(self) -> str:
        if self.api_key:
            return "key=given"
        if self.env_key:
            return f"key={self.env_key}"
        return "key=none needed"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, {self._key_note()})"


@dataclass(frozen=True, repr=False)
class OpenAI(Backend):
    """OpenAI's chat API. Key: ``api_key=`` or ``OPENAI_API_KEY``."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", "openai")
        object.__setattr__(self, "env_key", "OPENAI_API_KEY")


@dataclass(frozen=True, repr=False)
class Anthropic(Backend):
    """Anthropic's Messages API. Key: ``api_key=`` or ``ANTHROPIC_API_KEY``."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", "anthropic")
        object.__setattr__(self, "env_key", "ANTHROPIC_API_KEY")


@dataclass(frozen=True, repr=False)
class Fireworks(Backend):
    """An open model Fireworks serves, named the way Fireworks names it
    (``accounts/fireworks/models/<name>``). Key: ``api_key=`` or
    ``FIREWORKS_API_KEY``."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", "fireworks")
        object.__setattr__(self, "env_key", "FIREWORKS_API_KEY")


@dataclass(frozen=True, repr=False)
class Bedrock(Backend):
    """Amazon Bedrock's Converse API: a foundation model, a cross-region
    inference profile, or the ARN of a model you imported (a trained adapter
    merged into its base). One dot down (``wai.models.Bedrock``) so the front
    door stays small; the string ``bedrock:<model-id>[@<region>]`` is the same
    thing anywhere a backend goes. ``region=`` pins the region, else
    ``AWS_REGION``. Key: ``api_key=`` or ``AWS_BEARER_TOKEN_BEDROCK`` (a Bedrock API key,
    nothing to install), else the AWS credential chain through boto3
    (``pip install "whileai[bedrock]"``)."""

    region: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", "bedrock")
        object.__setattr__(self, "env_key", "AWS_BEARER_TOKEN_BEDROCK or AWS credentials")

    @property
    def spec(self) -> str:
        return f"bedrock:{self.model}@{self.region}" if self.region else f"bedrock:{self.model}"

    def __repr__(self) -> str:
        where = f", region={self.region!r}" if self.region else ""
        return f"Bedrock(model={self.model!r}{where}, {self._key_note()})"


@dataclass(frozen=True, repr=False)
class Endpoint(Backend):
    """Any OpenAI-compatible server you run: vLLM, SGLang, TGI, LM Studio,
    a trained adapter behind a URL. Key: ``api_key=``, else ``VLLM_API_KEY``
    or ``OPENAI_API_KEY``; a loopback or plain-http URL needs none."""

    url: str = ""

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError(
                "Endpoint needs url=, the server's /v1 base, e.g. http://localhost:8000/v1"
            )
        object.__setattr__(self, "provider", "vllm")
        local = self.url.startswith("http://") or "localhost" in self.url or "127.0.0.1" in self.url
        object.__setattr__(self, "env_key", None if local else "VLLM_API_KEY or OPENAI_API_KEY")

    @property
    def spec(self) -> str:
        return f"vllm:{self.model}@{self.url}"

    def __repr__(self) -> str:
        return f"Endpoint(model={self.model!r}, url={self.url!r}, {self._key_note()})"


@dataclass(frozen=True, repr=False)
class Ollama(Backend):
    """A model served by Ollama on this machine. No key."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", "ollama")


@dataclass(frozen=True, repr=False)
class Hosted(Backend):
    """The model While hosts, on your account key (``whileai login`` or
    ``wai.configure(api_key=...)``). The default for every role when
    nothing else is configured. The agent is a Qwen3-4B; the judge is a
    Phi-4, a different family on purpose."""

    model: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", "whileai")
        object.__setattr__(self, "env_key", "WHILEAI_API_KEY or `whileai login`")

    @property
    def spec(self) -> None:  # type: ignore[override]
        """``None``: leave the engine on its default hosted route."""
        return None

    def __repr__(self) -> str:
        return f"Hosted({self._key_note()})"


__all__ = ["Anthropic", "Backend", "Bedrock", "Endpoint", "Fireworks", "Hosted", "Ollama", "OpenAI"]
