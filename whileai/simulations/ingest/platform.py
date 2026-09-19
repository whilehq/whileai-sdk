"""While platform client: dataset upload, listing, download.

Datasets generated locally with ``simulate()`` push to your While
account, where the optimization framework iterates on them. Runtime access
uses a short-lived delegated credential (``zp_dc_...``), which is issued by a
valid Clerk session token and then passed as the X-Api-Key on protected routes.

The legacy ``WHILEAI_API_KEY`` env var still works for compatibility, but the
preferred runtime credential is ``WHILEAI_DELEGATED_CREDENTIAL``. With neither
set, the key saved by ``whileai login`` (``~/.whileai/credentials.json``)
is used.

Stdlib only, matching the package's no-dependencies rule.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import warnings as _warnings
from collections.abc import Sequence
from typing import Any

from whileai._env import getenv
from whileai.auth import stored_api_key

from ..defaults import (
    HOLDOUT_BUCKET_HEX_CHARS,
    PLATFORM_CREDENTIAL_TTL_S,
    PLATFORM_ERROR_DETAIL_CHARS,
    PLATFORM_HF_DATASET_TIMEOUT_S,
    PLATFORM_HF_MODEL_POLL_S,
    PLATFORM_HF_MODEL_TIMEOUT_S,
    PLATFORM_HF_POLL_S,
    PLATFORM_HOLDOUT_PROVE_EFFECT,
    PLATFORM_IMPORT_MAX_ROWS,
    PLATFORM_IMPORT_POLL_S,
    PLATFORM_IMPORT_TIMEOUT_S,
    PLATFORM_PUT_S_PER_MB,
    PLATFORM_PUT_TIMEOUT_S,
    PLATFORM_REQUEST_TIMEOUT_S,
    PLATFORM_STUDIO_MAX_ROWS,
    PLATFORM_TRACE_PAGE_SIZE,
    PLATFORM_UPLOAD_TIMEOUT_S,
)

#: Overridable with ``WHILEAI_API_URL``, which is what a self-hosted gate or
#: a staging one uses. The default is the production token gate behind the
#: While AWS account, and the SDK prefers delegated credentials over static
#: keys at runtime.
DEFAULT_API_URL = "https://api.zeroproofai.com"

#: What a dataset is for on the Datasets page, and the simulation mode that
#: made it. The gate rejects anything else; the studio import takes MODES.
PURPOSES = ("train", "holdout", "eval")
MODES = ("explore", "sft", "rl", "adaptive")


class PlatformError(RuntimeError):
    pass


def _error_detail(err: urllib.error.HTTPError) -> str:
    detail = err.read().decode(errors="replace")[:PLATFORM_ERROR_DETAIL_CHARS]
    with contextlib.suppress(ValueError, AttributeError):
        detail = json.loads(detail).get("error", detail)
    return detail


def _api_url() -> str:
    return getenv("API_URL", DEFAULT_API_URL).rstrip("/")


def _key(api_key: str | None) -> str:
    key = api_key or getenv("DELEGATED_CREDENTIAL") or getenv("API_KEY") or stored_api_key() or ""
    if not key:
        raise PlatformError(
            "No credential. Run `whileai login`, or pass api_key=..., or set "
            "WHILEAI_DELEGATED_CREDENTIAL / WHILEAI_API_KEY."
        )
    return key


def _call(
    method: str,
    path: str,
    api_key: str | None,
    body: dict | None = None,
    *,
    raw_url: str | None = None,
    data: bytes | None = None,
    content_type: str | None = None,
    timeout: float = PLATFORM_REQUEST_TIMEOUT_S,
    auth_token: str | None = None,
    require_api_key: bool = False,
    public: bool = False,
) -> Any:
    url = raw_url or (_api_url() + path)
    headers: dict[str, str] = {}
    token = str(auth_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if require_api_key or (not raw_url and not token and not public):
        headers["X-Api-Key"] = _key(api_key)
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as err:
        detail = _error_detail(err)
        raise PlatformError(f"{method} {url.split('?')[0]} -> {err.code}: {detail}") from None
    except urllib.error.URLError as err:
        raise PlatformError(f"{method} {url.split('?')[0]} failed: {err.reason}") from None
    if raw_url:
        return payload
    return json.loads(payload) if payload else {}


def issue_delegated_credential(
    clerk_token: str | None,
    *,
    ttl_seconds: int = PLATFORM_CREDENTIAL_TTL_S,
    name: str = "sdk-default",
    timeout: float = PLATFORM_REQUEST_TIMEOUT_S,
) -> dict:
    """Create a short-lived delegated credential for SDK or backend use.

    ``clerk_token`` must be a valid Clerk session token or other authenticated
    backend token. This helper sends that token as a bearer token to the auth
    endpoint to mint the delegated credential.
    """
    if not clerk_token:
        raise PlatformError(
            "A valid Clerk session token is required to mint a delegated credential."
        )
    body = {"name": name, "ttlSeconds": int(ttl_seconds)}
    return _call(
        "POST", "/auth/issue-credential", None, body, timeout=timeout, auth_token=clerk_token
    )


def refresh_delegated_credential(
    clerk_token: str | None,
    credential: str,
    *,
    ttl_seconds: int = PLATFORM_CREDENTIAL_TTL_S,
    timeout: float = PLATFORM_REQUEST_TIMEOUT_S,
) -> dict:
    """Refresh a delegated credential before it expires."""
    if not clerk_token:
        raise PlatformError(
            "A valid Clerk session token is required to refresh a delegated credential."
        )
    if not credential:
        raise PlatformError("Pass the current delegated credential to refresh it.")
    body = {"credential": credential, "ttlSeconds": int(ttl_seconds)}
    return _call(
        "POST", "/auth/refresh-credential", None, body, timeout=timeout, auth_token=clerk_token
    )


def revoke_delegated_credential(
    clerk_token: str | None, credential: str, *, timeout: float = PLATFORM_REQUEST_TIMEOUT_S
) -> dict:
    """Revoke a delegated credential for the authenticated user."""
    if not clerk_token:
        raise PlatformError(
            "A valid Clerk session token is required to revoke a delegated credential."
        )
    if not credential:
        raise PlatformError("Pass the delegated credential to revoke it.")
    return _call(
        "POST",
        "/auth/revoke-credential",
        None,
        {"credential": credential},
        timeout=timeout,
        auth_token=clerk_token,
    )


DEFAULT_STUDIO_URL = "https://zeroproofai--zeroproof-studio-api-serve.modal.run"
_STUDIO_MODES = MODES
_STUDIO_MAX_ROWS = PLATFORM_STUDIO_MAX_ROWS


def _studio_url() -> str:
    return getenv("STUDIO_URL", DEFAULT_STUDIO_URL).rstrip("/")


def push_to_studio(
    rows: list[dict],
    agent: str,
    mode: str,
    *,
    tags: list[str] | None = None,
    filename: str | None = None,
    api_key: str | None = None,
    timeout: float = PLATFORM_UPLOAD_TIMEOUT_S,
) -> dict:
    """Import rows into the studio runs store the platform UI reads.

    ``push_rows`` lands in the datasets registry; the platform's
    datasets page reads the studio's runs store instead, so rows pushed
    there never appear in the UI. This posts to the studio's import
    endpoint, which grades rows against the agent's declared tools and
    writes into the same store the page lists.

    ``agent`` must exist in the STUDIO agent registry (separate from
    trace agents; an unregistered name is rejected by the studio).
    ``mode`` labels the batch (one of explore/sft/rl/adaptive) and is
    required: the store would otherwise silently label everything "rl".
    """
    if mode not in MODES:
        raise PlatformError(f"mode= must be one of {'/'.join(MODES)}")
    if len(rows) > PLATFORM_STUDIO_MAX_ROWS:
        raise PlatformError(
            f"studio import caps at {PLATFORM_STUDIO_MAX_ROWS} rows; got {len(rows)} - split the push"
        )
    body: dict = {"agent": agent, "mode": mode, "rows": list(rows)}
    if tags:
        body["tags"] = list(tags)
    if filename:
        body["filename"] = filename
    url = _studio_url() + "/api/import"
    data = json.dumps(body, default=str).encode()
    headers = {"X-Api-Key": _key(api_key), "Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as err:
        detail = _error_detail(err)
        hint = (
            " (is the agent registered in the studio? the studio "
            "registry is separate from trace agents)"
            if err.code in (400, 404)
            else ""
        )
        raise PlatformError(f"POST {url} -> {err.code}: {detail}{hint}") from None
    except urllib.error.URLError as err:
        raise PlatformError(f"POST {url} failed: {err.reason}") from None
    return json.loads(payload) if payload else {}


def _meta_body(
    purpose: str | None, mode: str | None, agent: str | None, description: str | None
) -> dict:
    body: dict = {}
    if purpose is not None:
        if purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
        body["purpose"] = purpose
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        body["mode"] = mode
    if agent is not None:
        body["agent"] = agent
    if description is not None:
        body["description"] = description
    return body


#: The gain a pushed holdout is sized to prove (defaults.py says why 5 points).
HOLDOUT_PROVE_EFFECT = PLATFORM_HOLDOUT_PROVE_EFFECT


def _warn_small_holdout(
    rows: Sequence[dict], effect: float = PLATFORM_HOLDOUT_PROVE_EFFECT
) -> None:
    """A holdout too small to prove an ``effect`` gain (5 points by default)
    reads every round as ``no_change_detected``; say so at push time, not
    after training (#257)."""
    from ..score.stats import holdout_size, task_key

    groups: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            key = task_key(row)
            groups[key] = groups.get(key, 0) + 1
    n_tasks = len(groups)
    if not n_tasks:
        return
    try:
        need = holdout_size(effect, rows=rows)
    except ValueError:
        need = holdout_size(effect, k=min(groups.values()))
    if n_tasks < need["n_tasks"]:
        _warnings.warn(
            f"holdout has {n_tasks} tasks at k={need['k']}; proving a "
            f"{effect:.0%} gain at 80% power needs about {need['n_tasks']} "
            "(holdout_size). A smaller holdout reads a real gain that size as "
            "no_change_detected.",
            stacklevel=3,
        )


def split_holdout(rows: list[dict], fraction: float | None) -> tuple[list[dict], list[dict]]:
    """Split rows by task so a task is wholly train or wholly holdout.

    Deterministic: the same ``scenario_id`` lands on the same side every
    run, which is what makes a before/after comparison honest.
    """
    if not fraction:
        return rows, []
    if not 0 < fraction < 1:
        raise ValueError("holdout must be a fraction between 0 and 1")
    train: list[dict] = []
    held: list[dict] = []
    for r in rows:
        key = str(r.get("scenario_id") or r.get("task_id") or r.get("prompt") or "")
        digits = hashlib.sha256(key.encode()).hexdigest()[:HOLDOUT_BUCKET_HEX_CHARS]
        bucket = int(digits, 16) / (16**HOLDOUT_BUCKET_HEX_CHARS - 1)
        (held if bucket < fraction else train).append(r)
    if not train:
        raise ValueError("holdout fraction leaves no training rows")
    return train, held


def put_timeout_for(n_bytes: int) -> float:
    """Seconds one presigned upload may take: ``PLATFORM_PUT_TIMEOUT_S`` plus
    ``PLATFORM_PUT_S_PER_MB`` for every megabyte, so a 117 MB eval set of
    long-reasoning rollouts gets minutes where a 4 KB set gets the floor
    (#386)."""
    return PLATFORM_PUT_TIMEOUT_S + PLATFORM_PUT_S_PER_MB * n_bytes / 1_000_000


def _put_payload(
    upload_url: str, payload: bytes, api_key: str | None, timeout: float | None = None
) -> None:
    """One presigned PUT of the JSONL bytes, with a timeout sized to the payload
    and a failure that names the size and the knob."""
    cap = float(timeout) if timeout is not None else put_timeout_for(len(payload))
    try:
        _call(
            "PUT",
            "",
            api_key,
            raw_url=upload_url,
            data=payload,
            content_type="application/jsonl",
            timeout=cap,
        )
    except PlatformError as err:
        mb = len(payload) / 1_000_000
        raise PlatformError(
            f"upload of {mb:.0f} MB failed after {cap:.0f}s: {err}. Pass timeout= "
            "for a longer cap, or split the push (push_rows on a slice)."
        ) from None


def push_rows(
    rows: list[dict],
    name: str,
    *,
    api_key: str | None = None,
    parent: str | None = None,
    gate: bool = False,
    mode: str | None = None,
    purpose: str | None = None,
    agent: str | None = None,
    description: str | None = None,
    endorsed: Sequence[str] = (),
    strict_hacks: bool = False,
    prove_effect: float = PLATFORM_HOLDOUT_PROVE_EFFECT,
    holdout: float | None = None,
    publish: bool = False,
    timeout: float | None = None,
) -> dict:
    """Upload rows as JSONL to your While account.

    Returns the registry entry, including ``datasetId``. Pass ``parent`` (a
    ``ds_...`` id) when this dataset is an iteration of an existing one, so
    lineage shows on the platform. ``purpose`` is what the set is for on the
    Datasets page: ``"train"`` (the default), ``"holdout"`` or ``"eval"``;
    ``mode`` is the simulation mode that made it, and is also recorded. ``gate=True`` runs ``publish_gate``
    first (calibration stamp; RL-shaped rows refused when ungraded or
    without a mixed group) and returns its report as ``entry["gate"]``.
    ``endorsed`` names what the reward should track for the gate's
    ``hack_scan``; ``strict_hacks=True`` refuses a set whose reward is
    best explained by something else.
    ``SimulationData.push`` gates by default; this row-level entry point
    does not, because the caller may already have run ``optimize``. A
    ``purpose="holdout"`` push warns when the set is too small to prove a
    ``prove_effect`` gain (5 points) at 80% power.

    ``holdout=0.2`` keeps a fifth of the tasks (by ``scenario_id``) out of
    the set and pushes them as a second, linked dataset with purpose
    ``"holdout"``; the entry carries it as ``["holdout"]``. ``publish=True``
    with an ``agent`` name also puts the set on the public catalog as a card
    (``["card"]``). Both are what ``SimulationData.push`` takes, so a graded
    RL push (``scored.push``) has the same route to a linked holdout (#408).
    ``timeout`` caps the upload in seconds; the default grows with the
    payload (``put_timeout_for``).
    """
    from ..schema import check

    if publish and not agent:
        raise ValueError("publish=True needs agent=..., cards are grouped by agent")

    gate_report = None
    if gate:
        from ..score.publish_gate import publish_gate

        gate_report = publish_gate(rows, mode=mode, endorsed=endorsed, strict_hacks=strict_hacks)
    check(rows, where="push_rows")
    if purpose == "holdout":
        _warn_small_holdout(rows, prove_effect)
    body: dict = {
        "name": name,
        **_meta_body(purpose, mode if mode in MODES else None, agent, description),
    }
    if parent:
        body["parentDatasetId"] = parent
    train_rows, holdout_rows = split_holdout(list(rows), holdout)
    created = _call("POST", "/datasets", api_key, body)
    payload = "".join(json.dumps(r, default=str) + "\n" for r in train_rows).encode()
    _put_payload(created["uploadUrl"], payload, api_key, timeout)
    final = _call("POST", f"/datasets/{created['datasetId']}/finalize", api_key)
    if holdout_rows:
        held = push_rows(
            holdout_rows,
            f"{name}-holdout",
            api_key=api_key,
            parent=str(final.get("datasetId") or created["datasetId"]),
            purpose="holdout",
            mode=mode,
            agent=agent,
            description=description,
            prove_effect=prove_effect,
            timeout=timeout,
        )
        final = {
            **final,
            "holdout": held,
            "holdout_tasks": len({r.get("scenario_id") for r in holdout_rows}),
        }
    if gate_report is not None:
        final = {**final, "gate": gate_report}
    if publish:
        card = publish_dataset(
            str(final.get("datasetId") or created["datasetId"]),
            agent or "",
            description,
            api_key=api_key,
        )
        final = {**final, "card": card}
    return final


def push_file(
    path: str,
    name: str | None = None,
    *,
    api_key: str | None = None,
    parent: str | None = None,
    gate: bool = True,
    mode: str | None = None,
) -> dict:
    """Upload an existing JSONL file. ``name`` defaults to the file name.

    ``gate=True`` (default) parses the file, runs ``publish_gate`` (rows
    get their ``calibration`` stamp; RL-shaped rows that are ungraded or
    have no mixed group are refused), and uploads the stamped rows. The
    report comes back as ``entry["gate"]``. ``gate=False`` uploads the
    bytes exactly as they are on disk.
    """
    gate_report = None
    if gate:
        from ..score.publish_gate import publish_gate
        from ..score.quality import load_jsonl

        rows = load_jsonl(path)
        gate_report = publish_gate(rows, mode=mode)
        payload = "".join(json.dumps(r, default=str) + "\n" for r in rows).encode()
    else:
        with open(path, "rb") as fh:
            payload = fh.read()
    stem = os.path.basename(path)
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    body: dict = {"name": name or stem}
    if parent:
        body["parentDatasetId"] = parent
    created = _call("POST", "/datasets", api_key, body)
    _put_payload(created["uploadUrl"], payload, api_key)
    final = _call("POST", f"/datasets/{created['datasetId']}/finalize", api_key)
    if gate_report is not None:
        final = {**final, "gate": gate_report}
    return final


def datasets(*, api_key: str | None = None) -> dict:
    """List your datasets plus storage used, newest first."""
    return _call("GET", "/datasets", api_key)


def publish(
    dataset_id: str,
    agent: str,
    description: str | None = None,
    *,
    api_key: str | None = None,
) -> dict:
    """Publish one of your datasets as a public card on zeroproofai.com/datasets.

    Cards are grouped by ``agent`` (a short name such as ``"airline-support"``).
    The dataset must be finalized and hold rows. Returns the card. Anyone can
    then ``pull`` it with no key.
    """
    body: dict = {"agent": agent}
    if description:
        body["description"] = description
    return _call("POST", f"/datasets/{dataset_id}/publish", api_key, body)


#: The function under a name ``push_rows`` can reach while its own
#: ``publish=`` keyword shadows ``publish``.
publish_dataset = publish


def agents(*, api_key: str | None = None) -> list[dict]:
    """Every agent on your account with counts: traces, sets by purpose, public cards.

    An agent exists the moment a push names it (``data.push(name, agent=...)``)
    or a trace arrives with ``gen_ai.agent.name``; ``register_agent`` is for
    attaching the spec or a description ahead of that.
    """
    return _call("GET", "/agents", api_key)["agents"]


def register_agent(
    name: str,
    *,
    description: str | None = None,
    tools: list | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Create or update an agent record: the name, and optionally what it is
    (a line), its tool schemas, and its system prompt. Returns the record."""
    body: dict = {"name": name}
    if description is not None:
        body["description"] = description
    if tools is not None:
        body["tools"] = tools
    if system_prompt is not None:
        body["system_prompt"] = system_prompt
    return _call("POST", "/agents", api_key, body)


def update_dataset(
    dataset_id: str,
    *,
    purpose: str | None = None,
    mode: str | None = None,
    agent: str | None = None,
    description: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Change what a dataset is for, its mode, agent or description.

    ``purpose`` moves it between the Train, Holdout, Eval and Raw sections of
    the Training data page. Only the arguments you pass change.
    """
    body = _meta_body(purpose, mode, agent, description)
    if not body:
        raise ValueError("pass purpose, mode, agent or description")
    return _call("POST", f"/datasets/{dataset_id}/meta", api_key, body)


def preview(dataset_id: str, *, api_key: str | None = None) -> dict:
    """Three sample rows and the analyzer report for one of your datasets."""
    return _call("GET", f"/datasets/{dataset_id}/preview", api_key)


def profile(dataset_id: str, *, force: bool = False, api_key: str | None = None) -> dict:
    """The trainer's numbers for one of your datasets: pass rate, gradient
    support, tasks with both a pass and a fail, tool use, tokens, per-task
    pass rates. Cached on the platform until the set changes; ``force=True``
    recomputes.
    """
    path = f"/datasets/{dataset_id}/profile" + ("?force=1" if force else "")
    return _call("GET", path, api_key)["profile"]


def unpublish(dataset_id: str, *, api_key: str | None = None) -> dict:
    """Take a dataset off the public catalog. The data stays on your account."""
    return _call("POST", f"/datasets/{dataset_id}/unpublish", api_key)


def catalog() -> dict:
    """The public catalog: ``{"datasets": [card, ...], "agents": [...]}``. No key needed."""
    return _call("GET", "/catalog", None, public=True)


# ---------------------------------------------------------------- Hugging Face


def _wait_hf(get: Any, timeout: float, poll: float) -> dict:
    """Poll ``get()`` until its ``hf`` state settles. Raises PlatformError on error."""
    deadline = time.monotonic() + timeout
    while True:
        row = get()
        hf = row.get("hf") or {}
        if hf.get("status") == "done":
            return hf
        if hf.get("status") == "error":
            raise PlatformError(f"Hugging Face push failed: {hf.get('error', 'unknown error')}")
        if time.monotonic() >= deadline:
            raise PlatformError(
                f"Hugging Face push still running after {int(timeout)} s; check hf on the row later"
            )
        time.sleep(poll)


def hf_status(*, api_key: str | None = None) -> dict:
    """Is a Hugging Face account connected to this account, and which
    namespaces (you plus your orgs) can it publish under?

    Connect one on any dataset page at zeroproofai.com/platform/datasets.
    Returns ``{"connected", "username", "namespaces", "scopes"}``.
    """
    return _call("GET", "/hf/me", api_key)


def hf_publish(
    dataset_id: str,
    *,
    namespace: str | None = None,
    repo: str | None = None,
    private: bool = False,
    wait: bool = True,
    timeout: float = PLATFORM_HF_DATASET_TIMEOUT_S,
    poll: float = PLATFORM_HF_POLL_S,
    api_key: str | None = None,
) -> dict:
    """Push one of your datasets to a Hugging Face dataset repo you own.

    The set's purpose (train, holdout, eval) is the split; pushing the
    holdout set into the same ``repo`` adds a second split. Every push is a
    commit tagged ``zp-<dataset id>`` and the repo carries ``whileai.json``
    (split -> dataset, numbers, history). Defaults: your username and a
    slug of the dataset name. With ``wait`` (the default) this returns the
    finished state ``{"repo", "url", "commit", "tag", "split", ...}``;
    otherwise the ``pushing`` stamp.
    """
    body: dict = {"private": private}
    if namespace:
        body["namespace"] = namespace
    if repo:
        body["repo"] = repo
    out = _call("POST", f"/datasets/{dataset_id}/hf-publish", api_key, body)
    if not wait:
        return out["hf"]
    return _wait_hf(lambda: _call("GET", f"/datasets/{dataset_id}", api_key), timeout, poll)


def hf_publish_run(
    run_id: str,
    *,
    namespace: str | None = None,
    repo: str | None = None,
    private: bool = True,
    wait: bool = True,
    timeout: float = PLATFORM_HF_MODEL_TIMEOUT_S,
    poll: float = PLATFORM_HF_MODEL_POLL_S,
    api_key: str | None = None,
) -> dict:
    """Push a finished training run's LoRA adapter to a Hugging Face model
    repo you own, with a model card (base model, metrics, the dataset repo
    when the data was pushed too). Private by default: it is a checkpoint,
    not a release. Tagged ``zp-<run id>``.
    """
    body: dict = {"private": private}
    if namespace:
        body["namespace"] = namespace
    if repo:
        body["repo"] = repo
    out = _call("POST", f"/runs/{run_id}/hf-publish", api_key, body)
    if not wait:
        return out["hf"]
    return _wait_hf(lambda: _call("GET", f"/runs/{run_id}", api_key), timeout, poll)


def import_hf(
    repo: str,
    *,
    split: str = "train",
    revision: str | None = None,
    config: str | None = None,
    name: str | None = None,
    purpose: str = "train",
    mode: str | None = None,
    agent: str | None = None,
    description: str | None = None,
    max_rows: int = PLATFORM_IMPORT_MAX_ROWS,
    wait: bool = True,
    timeout: float = PLATFORM_IMPORT_TIMEOUT_S,
    poll: float = PLATFORM_IMPORT_POLL_S,
    api_key: str | None = None,
) -> dict:
    """Bring one split of a Hugging Face dataset onto your account as rows,
    so it gets a profile (pass rate, gradient support, mixed prompts)
    before you train on it. Parquet, JSONL, CSV and Arrow all come in the
    same way. Public repos need no connected account; private ones use the
    Hugging Face account connected on the platform.

    Returns the dataset row. With ``wait`` (the default) the row is
    ``ready`` (or this raises with the import error); otherwise it is
    ``importing`` and ``wai.datasets()`` shows it settle.
    """
    body: dict = {"repo": repo, "split": split, "purpose": purpose, "max_rows": max_rows}
    for key, value in (
        ("revision", revision),
        ("config", config),
        ("name", name),
        ("mode", mode),
        ("agent", agent),
        ("description", description),
    ):
        if value:
            body[key] = value
    row = _call("POST", "/datasets/import-hf", api_key, body)
    if not wait:
        return row
    deadline = time.monotonic() + timeout
    while row.get("status") == "importing":
        if time.monotonic() >= deadline:
            raise PlatformError(
                f"import of {repo}:{split} still running after {int(timeout)} s; it is {row['datasetId']}"
            )
        time.sleep(poll)
        row = _call("GET", f"/datasets/{row['datasetId']}", api_key)
    if row.get("status") == "failed":
        raise PlatformError(
            f"import of {repo}:{split} failed: {(row.get('hfImport') or {}).get('error', 'unknown error')}"
        )
    return row


def _has_key(api_key: str | None) -> bool:
    return bool(api_key or getenv("DELEGATED_CREDENTIAL") or getenv("API_KEY") or stored_api_key())


def _download_grant(dataset_id: str, api_key: str | None) -> dict:
    """Your own dataset when a key is at hand, else the public catalog copy."""
    if _has_key(api_key):
        try:
            return _call("GET", f"/datasets/{dataset_id}/download", api_key)
        except PlatformError as err:
            if "404" not in str(err):
                raise
    return _call("GET", f"/catalog/{dataset_id}/download", None, public=True)


def pull(
    dataset_id: str, path: str | None = None, *, api_key: str | None = None
) -> str | list[dict]:
    """Download a dataset. Writes JSONL to ``path`` and returns the path,
    or returns the parsed rows when ``path`` is omitted. Public catalog
    datasets need no key; your own need the usual one.

    A dataset is stored as one or more parts, and the grant lists every one
    of them. Datasets pushed with ``push_rows`` are a single part, which is
    why reading only ``downloadUrl`` looked correct for so long; a dataset
    filled by trace ingest is one part per trace, and that path returned the
    first row of a 60-row dataset without saying so.
    """
    grant = _download_grant(dataset_id, api_key)
    # `downloadUrl` is parts[0], kept for older grants that predate the list.
    urls = [u for u in (grant.get("parts") or []) if isinstance(u, str)]
    if not urls:
        urls = [grant["downloadUrl"]]

    payloads = [_call("GET", "", api_key, raw_url=url) for url in urls]
    # Parts are whole JSONL objects but need not end in a newline, so joining
    # blind would weld the last row of one part onto the first of the next.
    payload = b"\n".join(p.strip() for p in payloads if p.strip())

    if path:
        with open(path, "wb") as fh:
            fh.write(payload + b"\n" if payload else payload)
        return path
    return [json.loads(line) for line in payload.decode().splitlines() if line.strip()]


def delete(dataset_id: str, *, api_key: str | None = None) -> dict:
    """Permanently delete a dataset from your account."""
    return _call("DELETE", f"/datasets/{dataset_id}", api_key)


def _agent_trace_ids(
    slug: str, api_key: str | None, page_size: int = PLATFORM_TRACE_PAGE_SIZE
) -> list[str]:
    ids: list[str] = []
    page = 1
    while True:
        d = _call(
            "GET", f"/traces?agent={slug}&from=all&limit={int(page_size)}&page={page}", api_key
        )
        rows = d.get("traces") or []
        ids.extend(str(t["traceId"]) for t in rows if t.get("traceId"))
        if not rows or page >= int(d.get("pages") or 1):
            break
        page += 1
    return ids


def purge_agent(agent: str, *, dry_run: bool = False, api_key: str | None = None) -> dict:
    """Remove an agent and everything under it: its traces, its datasets,
    and its registry record. Permanent. ``dry_run=True`` only counts.

    Returns ``{"agent", "traces", "datasets", "deleted"}``.
    """
    slug = str(agent or "").strip().lower()
    if not slug:
        raise ValueError("pass the agent slug")
    trace_ids = _agent_trace_ids(slug, api_key)
    sets = [d for d in datasets(api_key=api_key)["datasets"] if d.get("agent") == slug]
    out = {"agent": slug, "traces": len(trace_ids), "datasets": len(sets), "deleted": not dry_run}
    if dry_run:
        return out
    for tid in trace_ids:
        _call("DELETE", f"/traces/{tid}", api_key)
    for d in sets:
        _call("DELETE", f"/datasets/{d['datasetId']}", api_key)
    _call("DELETE", f"/agents/{slug}", api_key)
    return out


def delete_empty_datasets(
    *, max_rows: int = 0, dry_run: bool = False, api_key: str | None = None
) -> dict:
    """Delete datasets with no stored bytes, or with ``max_rows`` rows or
    fewer when that is set (smoke runs). Permanent. Returns the ids."""
    victims = []
    for d in datasets(api_key=api_key)["datasets"]:
        size = int(d.get("sizeBytes") or 0)
        rows = d.get("rows")
        if size == 0 or (max_rows > 0 and rows is not None and int(rows) <= max_rows):
            victims.append(d["datasetId"])
    if not dry_run:
        for did in victims:
            _call("DELETE", f"/datasets/{did}", api_key)
    return {"datasets": victims, "deleted": not dry_run}
