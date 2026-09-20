"""Push a file, a directory or rows to the Hugging Face Hub with your own token.

``wai.hub.push`` is the local route. It hands the bytes to
``huggingface_hub`` and authenticates the way every HF-adjacent library
does: ``token=``, else ``HF_TOKEN`` from the environment, else the login
``hf auth login`` cached on the machine. No request goes to the While
platform. The platform route (``wai.simulations.hf_publish``: a While
dataset id to a repo, through the account connected on the website) stays
for people who keep their rows on the platform.

    import whileai as wai

    wai.export(rows, "train.jsonl", format="trl", push_to="me/my-set")
    wai.hub.push("out/adapter", "me/my-lora")       # a LoRA directory -> a model repo
    wai.hub.push(rows, "me/my-set", private=False)  # rows -> train.jsonl in a dataset repo

``huggingface_hub`` is an optional dependency: ``pip install 'whileai[hf]'``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: The variable huggingface_hub itself reads; named once so the docs, the
#: error text and the tests agree on it.
TOKEN_ENV = "HF_TOKEN"
#: Rows pushed without a file land under this name in the dataset repo.
ROWS_FILE = "train.jsonl"
#: A directory holding one of these is a model (an adapter or a checkpoint)
#: and goes to a model repo; any other directory, and every file, is a dataset.
MODEL_MARKERS = ("adapter_config.json", "config.json")
INSTALL_HINT = (
    "huggingface_hub is not installed. Run `pip install 'whileai[hf]'` "
    "(or `pip install huggingface_hub`) to push to the Hub."
)


def _api(token: str | None) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return HfApi(token=token)


def repo_type_of(source: str | Path) -> str:
    """``"model"`` for an adapter or checkpoint directory, else ``"dataset"``."""
    path = Path(source)
    if path.is_dir() and any((path / marker).exists() for marker in MODEL_MARKERS):
        return "model"
    return "dataset"


def push(
    source: str | Path | Sequence[dict],
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = True,
) -> dict[str, Any]:
    """Upload ``source`` to ``repo_id`` on the Hugging Face Hub with your token.

    ``source`` is a file (uploaded under its own name), a directory
    (uploaded whole; one holding ``adapter_config.json`` or ``config.json``
    goes to a **model** repo, anything else to a **dataset** repo), or a
    list of rows (written as ``train.jsonl``). The repo is created when it
    does not exist, ``private`` by default: a training set or a checkpoint
    is not a release until you say so.

    ``token`` is the HF token. Unset, it is ``HF_TOKEN`` from the
    environment, then the login ``hf auth login`` cached, the order
    ``huggingface_hub`` resolves it in. Nothing here calls the While
    platform. Returns ``{"repo_id", "repo_type", "url", "commit", "files",
    "private"}``.

    ```python
    wai.hub.push("train.jsonl", "me/my-set")           # -> huggingface.co/datasets/me/my-set
    wai.hub.push("out/adapter", "me/my-lora")          # -> huggingface.co/me/my-lora
    ```
    """
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise ValueError(f"repo_id must be 'namespace/name', got {repo_id!r}")
    resolved = token or os.environ.get(TOKEN_ENV) or None
    api = _api(resolved)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"push: {path} does not exist")
        return _upload(api, path, repo_id, private=private)
    with tempfile.TemporaryDirectory(prefix="whileai-push-") as tmp:
        path = Path(tmp) / ROWS_FILE
        with open(path, "w", encoding="utf-8") as fh:
            for row in source:
                fh.write(json.dumps(row, default=str) + "\n")
        return _upload(api, path, repo_id, private=private)


def _upload(api: Any, path: Path, repo_id: str, *, private: bool) -> dict[str, Any]:
    repo_type = repo_type_of(path)
    url = api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    if path.is_dir():
        files = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
        info = api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"whileai: push {path.name}/ ({len(files)} files)",
        )
    else:
        files = [path.name]
        info = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"whileai: push {path.name}",
        )
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "url": str(url),
        "commit": getattr(info, "oid", None),
        "files": files,
        "private": private,
    }


__all__ = ["push", "repo_type_of"]
