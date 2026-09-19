# Releasing

Publishing happens on merge to `main`, but only when the version moves. Merging
anything else is a no-op for the release pipeline.

## The version scheme

Versions are `MAJOR.MINOR` in hundredths and move exactly one step at a time.

```
1.01 -> 1.02 -> 1.03 ... 1.98 -> 1.99 -> 2.00
```

Skipping a version, moving backwards, adding a third component, or tagging a
release candidate all fail the gate in `.github/scripts/check_version.py`.

**PEP 440 strips leading zeros.** PyPI stores `1.01` as `1.1`, and the two are
literally equal, so `pip install whileai==1.1` and `==1.01` fetch
the same release. Ordering is unaffected (`1.10 > 1.9 > 1.2`), and from `1.10`
onward the stored version matches what you typed. The gate compares normalized
release tuples for this reason, so write either spelling.

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
