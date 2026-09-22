#!/usr/bin/env python
"""Every hosted default names a Modal app and function that exist.

On 2026-09-21 `stressd-vllm` was stopped and `zeroproof-serve`'s `phi_4`
function was replaced by `qwen3_8b`. Three of the SDK's hosted defaults went
on naming them, 0.120 shipped that way, and a `VLLM_API_KEY` user got a 404
on the first `simulate()` call. Nothing failed in CI, because nothing checked.

This is the check. It is OFFLINE on purpose: a GET against a scale-to-zero
vLLM endpoint boots the container, so probing six defaults on every push
would cold-start an H200 ($4.54/hr, defaults.py) to learn what a manifest
already knows. The manifest is refreshed from `modal app list` by a human
with credentials (``--refresh``), and CI only compares against it.

    python scripts/check_hosted_defaults.py             # CI: compare
    python scripts/check_hosted_defaults.py --refresh   # local: regenerate

What this CAN prove and what it cannot, stated up front so the gate is not
read as more coverage than it is. `modal app list` returns apps, not their
web endpoints, so:

  - APP verified against live Modal. A default naming a stopped or renamed
    app fails. This is the `stressd-vllm` case.
  - FUNCTION not verifiable from the CLI. `whileai-serve-phi-4` parses to a
    deployed app plus a function that no longer exists, and this check
    cannot see that. It is recorded in the manifest so a reviewer diffs it,
    and that is all.

The manifest's app list therefore comes from Modal, never from the defaults
it is checking. A manifest regenerated out of the SDK's own constants would
certify whatever those constants said, which is a check that cannot fail.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "hosted_defaults_manifest.json"
#: Only ``<workspace>--<app>-<function>.modal.run`` hosts are checked. Anything
#: else (a user's own endpoint, api.while.ai) is not ours to verify.
MODAL_HOST = re.compile(r"^(?P<workspace>[a-z0-9-]+)--(?P<rest>[a-z0-9-]+)\.modal\.run$")


def _defaults() -> dict[str, str]:
    """The hosted specs the SDK falls back to, read from the module."""
    sys.path.insert(0, str(ROOT))
    from whileai.simulations.generate import agents, embeddings
    from whileai.simulations.ingest import platform

    found = {
        name: getattr(agents, name)
        for name in (
            "DEFAULT_AGENT",
            "DEFAULT_SIMULATOR",
            "DEFAULT_JUDGE",
            "ACCOUNT_AGENT",
            "ACCOUNT_JUDGE",
        )
        if isinstance(getattr(agents, name, None), str)
    }
    # Not every hosted default is a rollout backend. These two name Modal
    # apps the same way and broke the same way, so they are checked the same
    # way rather than left for the next person to find in a traceback.
    for mod, name in ((platform, "DEFAULT_STUDIO_URL"), (embeddings, "DEFAULT_EMBED_URL")):
        value = getattr(mod, name, None)
        if isinstance(value, str):
            found[name] = value
    return found


def _host(spec: str) -> str | None:
    """The bare hostname inside a ``vllm:model@https://host/v1`` spec."""
    url = spec.split("@", 1)[-1] if "@" in spec else spec
    url = url.split("://", 1)[-1]
    return url.split("/", 1)[0] or None


def _live_apps() -> list[str]:
    """Deployed app names, from the Modal CLI. The only ground truth here."""
    out = subprocess.run(
        ["modal", "app", "list", "--json"], capture_output=True, text=True, check=True
    ).stdout
    return sorted(
        {
            str(a["description"])
            for a in json.loads(out)
            if a.get("description") and str(a.get("state", "")).lower() == "deployed"
        }
    )


def _split(host: str, apps: list[str]) -> tuple[str | None, str]:
    """``zeroproofai--whileai-serve-qwen3-4b`` -> (app, function).

    App names contain hyphens and so do function labels, so the split is the
    longest deployed app name that prefixes the remainder. No match means the
    app is not deployed, which is the failure this check exists for.
    """
    m = MODAL_HOST.match(host)
    if not m:
        return None, ""
    rest = m.group("rest")
    for app in sorted(apps, key=len, reverse=True):
        if rest == app:
            return app, ""
        if rest.startswith(app + "-"):
            return app, rest[len(app) + 1 :]
    return None, rest


def check() -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST.relative_to(ROOT)}; run with --refresh")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    apps: list[str] = list(manifest.get("apps", []))
    #: Hosts known dead with no deployed replacement, each with the issue that
    #: tracks it. This list may shrink and never grow: a new dead host fails.
    #: Same ratchet as scripts/old_name_baseline.json.
    quarantined: dict[str, str] = dict(manifest.get("known_dead", {}))
    bad: list[str] = []
    stale: list[str] = []
    for name, spec in _defaults().items():
        host = _host(spec)
        if not host:
            bad.append(f"{name}: no host in {spec!r}")
            continue
        if not MODAL_HOST.match(host):
            continue  # a user's own endpoint; not ours to verify
        app, _fn = _split(host, apps)
        if app is None:
            if name in quarantined:
                stale.append(f"{name}: {host} ({quarantined[name]})")
            else:
                bad.append(f"{name}: {host} names no deployed Modal app")
    if bad:
        print("hosted defaults name endpoints that do not exist:\n")
        for line in bad:
            print(f"  {line}")
        print(
            "\nEither the app was renamed or stopped and the default was not updated,\n"
            "or a new deployment needs `python scripts/check_hosted_defaults.py --refresh`\n"
            "in the same PR. A default that 404s is a first-call failure for every user."
        )
        return 1
    for line in stale:
        print(f"known dead, tracked: {line}")
    live = len(_defaults()) - len(stale)
    print(f"ok: {live} hosted defaults on deployed apps, {len(stale)} quarantined")
    return 0


def refresh() -> int:
    apps = _live_apps()
    functions = {}
    for name, spec in _defaults().items():
        host = _host(spec) or ""
        app, fn = _split(host, apps)
        functions[name] = {"host": host, "app": app, "function": fn or None}
    MANIFEST.write_text(
        json.dumps(
            {
                "_comment": (
                    "apps: deployed Modal apps, read from `modal app list`. This is the "
                    "ground truth the check compares against, and it never comes from the "
                    "SDK's own constants. defaults: what each hosted default resolves to "
                    "right now, recorded so a reviewer can diff the function segment, "
                    "which the Modal CLI cannot verify. Refresh in the same PR as any "
                    "deployment change."
                ),
                "apps": apps,
                "defaults": functions,
            },
            indent=2,
        )
        + "\n"
    )
    missing = [n for n, d in functions.items() if d["app"] is None and d["host"]]
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(apps)} deployed apps")
    for n, d in sorted(functions.items()):
        print(f"  {n}: app={d['app']} function={d['function']}")
    if missing:
        print(f"\nWARNING: these name no deployed app: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="regenerate from `modal app list`")
    args = ap.parse_args()
    raise SystemExit(refresh() if args.refresh else check())
