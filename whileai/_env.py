"""Environment variables: ``WHILEAI_*``. The ``ZEROPROOF_*`` names from before
the rename are read with a warning until 0.95, then not at all."""

from __future__ import annotations

import os
import warnings
from typing import overload
from urllib.parse import urlparse

NEW_PREFIX = "WHILEAI_"
OLD_PREFIX = "ZEROPROOF_"
#: Release in which the old prefix stops being read.
OLD_PREFIX_GONE = "0.95"
#: Release in which the old prefix stops being read.
OLD_PREFIX_GONE = "0.95"

#: Hosts the platform answers on: the token gate and the site, old and new
#: domains, and the hosted-model endpoints it serves from Modal.
PLATFORM_DOMAINS = ("withwhile.com", "zeroproofai.com")
PLATFORM_MODAL_PREFIX = "zeroproofai--zeroproof-serve-"


@overload
def getenv(name: str) -> str | None: ...
@overload
def getenv(name: str, default: str) -> str: ...


def getenv(name: str, default: str | None = None) -> str | None:
    """Read ``WHILEAI_<name>``, else ``default``.

    A ``ZEROPROOF_<name>`` left from before the rename is still read when
    the new name is unset, with a ``DeprecationWarning`` that says which
    variable to set instead; that fallback goes away in 0.95. An empty
    string counts as unset, which is how every caller treated these
    variables (``os.environ.get(...) or fallback``).
    """
    value = os.environ.get(NEW_PREFIX + name)
    if value:
        return value
    value = os.environ.get(OLD_PREFIX + name)
    if value:
        warnings.warn(
            f"{OLD_PREFIX}{name} is the old name; set {NEW_PREFIX}{name} instead. "
            f"The old name stops being read in whileai {OLD_PREFIX_GONE}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    return default


def is_platform_host(url: str | None) -> bool:
    """Does ``url`` point at While's own platform (gate, site or hosted model)?

    True for a host that is, or sits under, ``withwhile.com`` or
    ``zeroproofai.com``, and for the ``zeroproofai--zeroproof-serve-*``
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
    """Which variable ``getenv(name)`` would read, or ``None`` if neither is set."""
    for prefix in (NEW_PREFIX, OLD_PREFIX):
        if os.environ.get(prefix + name):
            return prefix + name
    return None
