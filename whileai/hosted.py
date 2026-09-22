"""Hosted models: the models an account serves at ``models.while.ai``.

``wai.platform.hosted`` is the namespace. Register a model once, from
wherever it runs, and it answers at one OpenAI-compatible endpoint under
the account's key::

    import whileai as wai

    m = wai.platform.hosted.register(
        "nemotron-8b-t2s-r1",
        arn="arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123def456",
        role_arn="arn:aws:iam::123456789012:role/WhileModelsInvoke",
    )
    served = wai.platform.hosted.endpoint("nemotron-8b-t2s-r1")   # a wai.Endpoint
    after = wai.simulate(served, tools=TOOLS, system_prompt=POLICY)

Two kinds of model: a Bedrock import, custom deployment, provisioned
model or inference profile (``arn=``, in While's AWS account or in yours
through ``role_arn=``), and any OpenAI-compatible ``/v1`` server
(``url=`` and ``model=``). A subdomain of your own,
``<slug>.models.while.ai``, is claimed with ``subdomain()``.

What the endpoint keeps about a model: per day, the count of calls, errors
and tokens (``usage()``), and nothing else. Never a prompt, a completion or
a request log.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from whileai._env import getenv

#: Overridable with ``WHILEAI_MODELS_URL`` for a staging endpoint.
DEFAULT_MODELS_URL = "https://models.while.ai"
# USAGE_DAYS = 7: the default window ``usage()`` reads; a week shows a
# weekday pattern and stays one screen of rows (convention).
USAGE_DAYS = 7
# PUBLISH_POLL_S = 30 / PUBLISH_TIMEOUT_S = 3600: an import took 10 minutes for
# a Llama 3.1 8B (2026-09-20) and Bedrock allows models to 200 GB; an hour is
# the job's own ceiling, so the client waits no longer than the job would.
PUBLISH_POLL_S = 30
PUBLISH_TIMEOUT_S = 3600

Transport = Callable[..., Any]


def models_url() -> str:
    """The hosted endpoint's origin, without ``/v1``."""
    return (getenv("MODELS_URL", DEFAULT_MODELS_URL) or DEFAULT_MODELS_URL).rstrip("/")


class _Wire(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class HostedModel(_Wire):
    """One registered model, as the endpoint stores it. No secrets."""

    name: str
    kind: Literal["bedrock", "openai"]
    base: str | None = None
    description: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    #: bedrock rows
    arn: str | None = None
    region: str | None = None
    role_arn: str | None = None
    external_id: str | None = None
    #: openai rows
    url: str | None = None
    upstream_model: str | None = None
    auth: Literal["caller", "none"] | None = None
    #: an import in flight: ``importing`` (with ``step``), ``ready`` or ``failed`` (with ``error``)
    status: Literal["importing", "ready", "failed"] | None = None
    step: str | None = None
    error: str | None = None
    cmu: int | None = None

    @property
    def endpoint(self) -> str:
        """The base URL an OpenAI client points at."""
        return models_url() + "/v1"

    def __str__(self) -> str:
        where = self.arn if self.kind == "bedrock" else f"{self.url} ({self.upstream_model})"
        return f"{self.name}: {self.kind} -> {where}; call it as model={self.name!r} at {self.endpoint}"


class Subdomain(_Wire):
    """The account's own host under models.while.ai, or none yet."""

    subdomain: str | None = None
    host: str | None = None
    endpoint: str | None = None

    def __str__(self) -> str:
        return self.endpoint or "no subdomain claimed"


class UsageDay(_Wire):
    """One day's counts for one model: calls, errors, tokens. Never content."""

    day: str
    model: str
    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class HostedModels:
    """The ``wai.platform.hosted`` namespace.

    ``api_key`` and ``transport`` are for tests and for a second account in
    one process; the default instance reads the key the way every platform
    call does (``wai.configure(api_key=)``, ``WHILEAI_API_KEY``, ``wai login``).
    """

    def __init__(self, *, api_key: str | None = None, transport: Transport | None = None):
        self._api_key = api_key
        self._transport = transport

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        if self._transport is not None:
            return self._transport(method, path, body)
        from .platform import _key, _request

        return _request(method, path, api_key=_key(self._api_key), body=body, base=models_url())

    # ---- registering

    def register(
        self,
        name: str,
        *,
        arn: str | None = None,
        region: str | None = None,
        role_arn: str | None = None,
        external_id: str | None = None,
        url: str | None = None,
        model: str | None = None,
        auth: Literal["caller", "none"] = "caller",
        base: str | None = None,
        description: str | None = None,
    ) -> HostedModel:
        """Register (or update) ``name``. Pass ``arn=`` for a Bedrock model,
        with ``role_arn=`` when it lives in your own AWS account, or ``url=``
        and ``model=`` for an OpenAI-compatible server. Idempotent by name."""
        if bool(arn) == bool(url):
            raise ValueError(
                "register() takes exactly one of arn= (Bedrock) or url= (an OpenAI-compatible server)"
            )
        body: dict[str, Any] = {"name": name}
        if arn:
            body.update({"kind": "bedrock", "arn": arn})
            if region:
                body["region"] = region
            if role_arn:
                body["roleArn"] = role_arn
                if external_id:
                    body["externalId"] = external_id
        else:
            if not model:
                raise ValueError("register(url=...) also needs model=, the name the server knows")
            body.update({"kind": "openai", "url": url, "model": model, "auth": auth})
        if base:
            body["base"] = base
        if description:
            body["description"] = description
        return HostedModel.model_validate(self._call("POST", "/models", body)["model"])

    def list(self) -> builtins.list[HostedModel]:
        """Every model the account registered."""
        return [HostedModel.model_validate(m) for m in self._call("GET", "/models")["models"]]

    def get(self, name: str) -> HostedModel:
        return HostedModel.model_validate(self._call("GET", f"/models/{name}")["model"])

    def delete(self, name: str) -> None:
        """Remove the row. The model itself (the import, the server) is untouched."""
        self._call("DELETE", f"/models/{name}")

    def publish(
        self,
        adapter: str,
        *,
        name: str | None = None,
        base: str | None = None,
        hf_token: str | None = None,
        wait: bool = True,
        timeout_s: float = PUBLISH_TIMEOUT_S,
        poll_s: float = PUBLISH_POLL_S,
    ) -> HostedModel:
        """Hand While an adapter and get a model back.

        ``adapter`` is a Hugging Face repo id (``owner/name``) or a While
        training run id (``run_…``). While merges it into its base (read from
        the adapter unless ``base=``), imports the weights into Bedrock on
        While's account, and registers it under ``name`` (the repo's own name
        by default). ``hf_token`` is used for the download once and never
        stored. With ``wait=True`` this polls until the row is ``ready`` or
        ``failed`` and returns it; ``wait=False`` returns the ``importing`` row.
        """
        import time as _time

        model_name = name or adapter.rsplit("/", 1)[-1]
        body: dict[str, Any] = {"name": model_name, "adapter": adapter}
        if base:
            body["base"] = base
        if hf_token:
            body["hfToken"] = hf_token
        started = HostedModel.model_validate(self._call("POST", "/models/import", body)["model"])
        if not wait:
            return started
        deadline = _time.monotonic() + timeout_s
        row = started
        while row.status == "importing":
            if _time.monotonic() > deadline:
                raise TimeoutError(
                    f"{model_name} is still importing after {int(timeout_s)} s ({row.step}); "
                    "hosted.get() reads the row later"
                )
            _time.sleep(poll_s)
            row = self.get(model_name)
        if row.status == "failed":
            raise RuntimeError(f"{model_name} failed to import: {row.error}")
        return row

    # ---- using

    def endpoint(self, name: str) -> Any:
        """``wai.Endpoint`` for ``name``: pass it to ``simulate()`` or
        ``wai.configure(agent=)``. The key is the account key in use."""
        from .models import Endpoint
        from .platform import _key

        return Endpoint(name, url=models_url() + "/v1", api_key=_key(self._api_key))

    def usage(self, name: str, *, days: int = USAGE_DAYS) -> builtins.list[UsageDay]:
        """Per-day counts for ``name`` over the last ``days`` days, newest first."""
        rows = self._call("GET", f"/models/{name}/usage?days={int(days)}")["usage"]
        return [UsageDay.model_validate(r) for r in rows]

    # ---- the account's own host

    def subdomain(self, slug: str | None = None) -> Subdomain:
        """Claim ``<slug>.models.while.ai`` for the account, or with no
        argument read the current one. Only the account's keys work there."""
        if slug is None:
            return Subdomain.model_validate(self._call("GET", "/domain"))
        return Subdomain.model_validate(self._call("PUT", "/domain", {"subdomain": slug}))

    def release_subdomain(self) -> str | None:
        """Give the subdomain up. Returns the slug released, or None."""
        return self._call("DELETE", "/domain").get("released")


#: The namespace: ``wai.platform.hosted.register(...)``.
hosted = HostedModels()

__all__ = [
    "DEFAULT_MODELS_URL",
    "PUBLISH_POLL_S",
    "PUBLISH_TIMEOUT_S",
    "HostedModel",
    "HostedModels",
    "Subdomain",
    "UsageDay",
    "hosted",
    "models_url",
]
