# Releasing

Publishing happens on merge to `main`, but only when the version moves. Merging
anything else is a no-op for the release pipeline.

## The version scheme

The version is `0.N`. N is a counter that goes up by one per release and
never rolls over or resets.

```
0.98 -> 0.99 -> 0.100 -> 0.101 -> ... -> 0.1000
```

There is no 1.0. Skipping a number, moving backwards, adding a third
component, zero-padding, or tagging a release candidate all fail the gate in
`.github/scripts/check_version.py`, and `release.py` only ever cuts N+1.

**PEP 440 strips leading zeros**, which is why the counter is never padded:
a padded `1.07` is `1.7` on PyPI, and `0.100` sorts after `0.99` only because
the counter is a whole number. Old spellings still resolve (`==0.4` is the
`0.04` line in the changelog).

**2026-09-20 mis-numbering.** The bump script rolled 0.99 over to 1.00 and
padded, so PyPI got 1.0 and 1.3 to 1.8 (1.01 and 1.02 never uploaded). They
were re-uploaded from the same commits as 0.100 to 0.108, yanked, and are
listed in `MISNUMBERED` in the gate so the counter continues from 0.99.
`pip install whileai` ignores a yanked release; a pin like `==1.8` still
installs, and is the same code as `==0.108`.

## The old name

Before 0.51 this package was published as `zeroproof`. That name is retired:
its last upload on PyPI is a shim that depends on `whileai`, nothing builds or
uploads it any more, and `scripts/check_old_name.py` keeps the string out of
new code.

## Tags

The gate treats `v<version>` as shipped only when the tag's annotation is
`whileai <version>`; older tags in this repository, whose annotations name
the packages it was before the rename, are not releases of this package and
are ignored. If a version is skipped as "already published" while PyPI lacks it,
check `git tag -l --format='%(subject)' v<version>` first.

## Cutting a release

1. Move the `Unreleased` section of `CHANGELOG.md` under the new version and date.
2. Bump `version` in `pyproject.toml` by one step.
3. Open a PR. The `version scheme` job tells you up front whether the gate will
   accept it after merge.
4. Merge to `main`.

On merge: the gate re-checks the version, tests run on 3.10 and 3.13, `uv build`
produces both distributions, `twine check` validates them, `uv publish` uploads,
and `vX.YZ` is tagged.

Prefer a token scoped to this project rather than an account-wide one. A
project-scoped token that leaks cannot touch the other packages on the account.

If the version is unchanged, the gate prints `nothing to cut` and exits clean.
That is the normal path for a merge that is not a release.

## Authentication

Either path works. The workflow prefers a token secret when one exists and
falls back to Trusted Publishing when it does not.

### Option A: token (fastest)

Add the PyPI token to **this** repository as `UV_PUBLISH_TOKEN`, matching the
convention used elsewhere in the org:

```bash
gh secret set UV_PUBLISH_TOKEN --repo whilehq/whileai-sdk
```

`uv publish` reads `UV_PUBLISH_TOKEN` directly, which is why that name is
preferred; `PYPI_API_TOKEN` is accepted as a fallback.

GitHub secrets are write-only, so a token held in another repository cannot be
copied across. Retrieve it from PyPI or your password manager and paste it into
the prompt above.

### Option B: Trusted Publishing (no stored credential)

Preferred for anything long-lived: a repository token is a standing credential,
OIDC is not. One-time setup on PyPI, under the project's *Publishing* settings:

| field | value |
|---|---|
| Owner | `whilehq` |
| Repository | `whileai-sdk` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Until the package exists on PyPI, register it as a *pending* publisher instead;
the first successful run creates the project. Also create a `pypi` environment
under repository Settings, and add required reviewers there if you want a human
approval step before any upload.

## What CI checks on every PR

- `pytest` on Python 3.10, 3.11, 3.12, 3.13
- `twine check` on both distributions
- the wheel installs with **plain pip** into a clean venv and imports with no
  source tree present

That last one is deliberate. An environment pushed to the Prime Intellect
Environments Hub is installed with plain pip, so a uv-only source pin resolves
locally, passes review, and then fails on their runtime with a
`ModuleNotFoundError`. This check catches that class of bug before release.
