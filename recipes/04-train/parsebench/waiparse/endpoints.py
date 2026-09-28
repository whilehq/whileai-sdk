"""Where the agent's servers live: your own Modal workspace, never ours.

Modal serves a web endpoint at ``https://<workspace>--<app>-<class>-<method>.modal.run``.
Set ``DOCPARSE_WORKSPACE`` to your workspace name (the part before ``--`` in any URL
``modal deploy`` prints) and every URL below follows. Each one can also be set whole,
which wins over the workspace: ``DOCPARSE_SERVER`` (the vLLM server, no ``/v1``),
``DOCPARSE_TUNED_SERVER`` (the second vLLM app that serves a LoRA adapter) and
``DOCPARSE_LAYOUT_URL`` (the layout detector). Modal shortens a label longer than 63
characters, so with a long workspace name copy the URL ``modal deploy`` printed.

The Modal files bake these variables into their images (``IMAGE_ENV``), so a remote
function sees the same values your shell had when you ran ``modal run``.
"""

from __future__ import annotations

import os

#: The variables a remote function needs to find the servers; baked into each image.
ENV_KEYS = ("DOCPARSE_WORKSPACE", "DOCPARSE_SERVER", "DOCPARSE_TUNED_SERVER", "DOCPARSE_LAYOUT_URL")


def image_env() -> dict[str, str]:
    """The endpoint variables set in this shell, for ``modal.Image.env``."""
    return {k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)}


def workspace() -> str:
    name = os.environ.get("DOCPARSE_WORKSPACE", "").strip()
    if not name:
        raise RuntimeError(
            "DOCPARSE_WORKSPACE is not set. Export your Modal workspace name (the part before "
            "'--' in the URL `modal deploy serve/serve_vlm.py` printed), or set DOCPARSE_SERVER "
            "and DOCPARSE_LAYOUT_URL to the full URLs."
        )
    return name


def modal_url(app: str, cls: str, method: str) -> str:
    """``https://<workspace>--<app>-<cls>-<method>.modal.run`` for a class web endpoint."""
    return f"https://{workspace()}--{app}-{cls.lower()}-{method}.modal.run"


def vlm_server() -> str:
    """The base vLLM server (serve/serve_vlm.py, app docparse-vlm), without ``/v1``."""
    return os.environ.get("DOCPARSE_SERVER") or modal_url("docparse-vlm", "Server", "serve")


def tuned_server() -> str:
    """The adapter server (serve/serve_vlm.py deployed with DOCPARSE_APP=docparse-vlm-tuned)."""
    return os.environ.get("DOCPARSE_TUNED_SERVER") or modal_url(
        "docparse-vlm-tuned", "Server", "serve"
    )


def layout_url() -> str:
    """The PP-DocLayoutV3 detector (serve/serve_layout.py, app docparse-layout)."""
    return os.environ.get("DOCPARSE_LAYOUT_URL") or modal_url("docparse-layout", "Detector", "web")
