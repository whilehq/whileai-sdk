"""Gate the release version before anything is published.

House rule: versions move in hundredths and only ever by one step.

    1.01 -> 1.02 -> 1.03 ... 1.99 -> 2.00

PEP 440 strips leading zeros, so PyPI stores "1.01" as "1.1" and treats the two
as equal. Ordering still behaves (1.10 > 1.9 > 1.2), so the scheme works, but
the rule is enforced on the normalized release tuple rather than the string:
(1, 1) -> (1, 2) -> ... -> (1, 99) -> (2, 0).

Exit codes:
  0  version is a valid single step, publish
  0  version unchanged, nothing to cut (prints SKIP)
  1  version is invalid or skips ahead, block the release
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

import tomllib

PYPI = "https://pypi.org/pypi/{name}/json"
# The per-release endpoint is not served from the same cache as the list
# above, so it answers within seconds of an upload the list will not show
# for many minutes.
PYPI_RELEASE = "https://pypi.org/pypi/{name}/{version}/json"


def local_version(path: str = "pyproject.toml") -> tuple[str, str]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    project = data["project"]
    return project["name"], project["version"]


def published(name: str) -> list[tuple[int, ...]]:
    """Every release already on PyPI, as normalized tuples."""
    # PyPI's JSON API sits behind a CDN that can serve a minutes-old version
    # list. Ask for a fresh copy; the tag check below is the real backstop.
    req = urllib.request.Request(PYPI.format(name=name), headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []  # first ever release
        raise
    from packaging.version import InvalidVersion, Version

    out = []
    for raw in data.get("releases", {}):
        try:
            release = Version(raw).release
        except InvalidVersion:
            continue
        # A name-reservation upload (0.0.1) is not part of the scheme.
        if len(release) == 2:
            out.append(release)
    return sorted(out)


def on_pypi(name: str, version: str) -> bool:
    """True when this exact release answers on PyPI's per-release endpoint."""
    req = urllib.request.Request(
        PYPI_RELEASE.format(name=name, version=version), headers={"Cache-Control": "no-cache"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False
    except OSError:
        return False


def tagged(version: str) -> bool:
    """True when the release tag already exists on origin.

    The publish job pushes v<version> right after the upload, so the tag is
    an uncached record of what has shipped. A merge that lands minutes after
    a release can still see the stale PyPI list; the tag does not lie.
    """
    proc = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/v{version}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 2):
        print(f"git ls-remote failed ({proc.returncode}): {proc.stderr.strip()[:200]}")
    if proc.returncode != 0:
        return False
    # The tag must be one of ours. This repository carries tags from the
    # packages it was before the rename (their annotations name the old
    # package), and on 2026-09-20 two of those, ``v1.1`` and ``v1.2``, made
    # the gate skip 1.01 and 1.02 as "already published" when PyPI had
    # neither. A whileai release tag is annotated ``whileai <version>``;
    # anything else is not a release.
    subprocess.run(
        [
            "git",
            "fetch",
            "--quiet",
            "--no-tags",
            "origin",
            f"refs/tags/v{version}:refs/tags/v{version}",
        ],
        capture_output=True,
        text=True,
    )
    subject = subprocess.run(
        ["git", "for-each-ref", "--format=%(subject)", f"refs/tags/v{version}"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not subject.startswith("whileai "):
        print(f"tag v{version} exists but is not a whileai release ({subject!r}); ignoring it")
        return False
    return True


def next_allowed(prev: tuple[int, ...]) -> tuple[int, ...]:
    major, minor = prev[0], (prev[1] if len(prev) > 1 else 0)
    return (major + 1, 0) if minor >= 99 else (major, minor + 1)


def fail(msg: str) -> None:
    print(f"::error::{msg}")
    sys.exit(1)


def main() -> int:
    name, version = local_version()
    from packaging.version import InvalidVersion, Version

    try:
        current = Version(version)
    except InvalidVersion:
        fail(f"{version!r} is not a valid PEP 440 version")
        return 1

    if len(current.release) != 2:
        fail(
            f"version must be MAJOR.MINOR in hundredths (e.g. 1.02), got {version!r}. "
            f"Three-part versions are not part of this scheme."
        )
    if current.pre or current.post or current.dev or current.local:
        fail(f"{version!r} has a pre/post/dev/local segment; releases must be plain.")

    prior = published(name)
    print(f"package        : {name}")
    print(f"local version  : {version}  (normalized {current})")
    shown = [".".join(map(str, p)) for p in prior[-5:]] or "none"
    print(f"published      : {shown}")

    if not prior:
        # First release. Anything sane is fine; require it to start at x.1 or x.0.
        if current.release[1] not in (0, 1):
            fail(f"first release should be x.00 or x.01, got {version!r}")
        print(f"::notice::first release of {name} {current}")
        return emit(publish=True, version=str(current))

    if current.release in prior or tagged(str(current)) or on_pypi(name, str(current)):
        # Already on PyPI, or already tagged by a publish run whose upload the
        # PyPI CDN has not caught up with yet. Not an error: main moves for
        # reasons other than a release, and re-running CI on an unchanged
        # version must not fail.
        print(f"::notice::{current} is already published, nothing to cut")
        return emit(publish=False, version=str(current))

    latest = prior[-1]
    # The list above can lag a release by up to fifteen minutes. A bump merged
    # right behind another would read as a skipped step; walk forward over
    # versions the publish job has already tagged before judging the gap.
    while True:
        candidate = ".".join(map(str, next_allowed(latest)))
        if tagged(candidate):
            reason = "tagged"
        elif on_pypi(name, candidate):
            reason = "on PyPI"
        else:
            break
        latest = next_allowed(latest)
        print(f"{reason}, not yet listed: {candidate}")
    if current.release < latest:
        fail(f"{current} is older than the published {'.'.join(map(str, latest))}")

    allowed = next_allowed(latest)
    if current.release != allowed:
        fail(
            f"version must step by exactly one hundredth. "
            f"published {'.'.join(map(str, latest))}, "
            f"expected {'.'.join(map(str, allowed))}, got {current}"
        )

    print(f"::notice::cutting {name} {current}")
    return emit(publish=True, version=str(current))


def emit(*, publish: bool, version: str) -> int:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"publish={str(publish).lower()}\n")
            fh.write(f"version={version}\n")
    print(f"publish={publish} version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
