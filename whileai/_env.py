"""Environment variables: every setting the SDK reads is ``WHILEAI_<name>``."""

from __future__ import annotations

import os
from typing import overload
from urllib.parse import urlparse

PREFIX = "WHILEAI_"

#: Hosts the platform answers on: the token gate and the site, old and new
#: domains, and the hosted-model endpoints it serves from Modal.
PLATFORM_DOMAINS = ("while.ai", "withwhile.com", "zeroproofai.com")
PLATFORM_MODAL_PREFIX = "zeroproofai--zeroproof-serve-"


@overload
def getenv(name: str) -> str | None: ...
@overload
def getenv(name: str, default: str) -> str: ...


def getenv(name: str, default: str | None = None) -> str | None:
    """Read ``WHILEAI_<name>``, else ``default``.

    An empty string counts as unset (``os.environ.get(...) or default``).
    """
    return os.environ.get(PREFIX + name) or default


def is_platform_host(url: str | None) -> bool:
    """Does ``url`` point at While's own platform (gate, site or hosted model)?

    True for a host that is, or sits under, ``while.ai``, ``withwhile.com``
    or ``zeroproofai.com``, and for the ``zeroproofai--zeroproof-serve-*``
    Modal endpoints the platform serves models from. A bare host with no
    scheme is read as one. Those are the URLs a ``zp_`` key is sent to.
    """
    if not url:
        return False
    raw = str(url).strip()
    if "://" not in raw:
        raw = "https://" + raw
    host = (urlparse(raw).hostname or "").lower()
    if not host:
        return False
    if any(host == d or host.endswith("." + d) for d in PLATFORM_DOMAINS):
        return True
    return host.startswith(PLATFORM_MODAL_PREFIX) and host.endswith(".modal.run")


def env_name(name: str) -> str | None:
    """The variable ``getenv(name)`` reads, or ``None`` if it is unset."""
    return PREFIX + name if os.environ.get(PREFIX + name) else None
