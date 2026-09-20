# Contributing

## Setup

```bash
uv sync --extra dev
```

## The checks CI runs

```bash
uv run pytest -q            # the suite, on 3.10 / 3.11 / 3.12 / 3.13; add -n0 for serial or --pdb
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run ty check             # faster mypy pass; mypy stays the gate until ty is 1.0
python -m build && python -m twine check dist/*   # if you touched packaging or imports
```

## Coding standard

[CONSTITUTION.md](CONSTITUTION.md) is what the library is and what we
believe; read it once.

Read [`docs/reference/style.md`](docs/reference/style.md) before adding
a public name. The short form: `import whileai as wai` is the one
prefix; a judge, verifier, selector or trainer is an object whose
constructor takes the configuration and whose call takes the rows; a
public call takes at most eight parameters; a report prints itself
(`__str__`, `_repr_html_`), so there is no `format_*` twin; public
names are the verb a scientist says. `tests/api/test_style_ratchet.py`
fails a PR that adds one of the retired shapes.

## Pull requests

Every PR names the issue it closes. The body opens with `Closes #N`, one
line per issue, or `Part of #N` when the issue takes more than one PR. Then
what was wrong, what changed, and the commands you ran with their result.
GitHub closes the issue on merge, so the issue thread ends with the commit
that fixed it and a reader goes issue, PR, diff without a search. Work that
has no issue opens with `No issue: <one line why>`; a change big enough to
need a design note gets an issue first. Release PRs are cut by
`release.yml` and are exempt. `.github/workflows/pr-issue.yml` fails a PR
that does neither, and re-runs when you edit the description.

## Shipping a release

```bash
gh workflow run release.yml
```

That cuts the next number (`0.99` then `0.100`) from the entries under `## Unreleased`, lands
the bump on main and starts the publish workflow; runs queue, so two
people shipping at once get two releases in order. Do not bump `version`
by hand, and keep the `## Unreleased` header (the cut renames it and puts
a fresh one above). `uv run python .github/scripts/release.py --dry-run`
shows what a cut would ship.

## Contributing a recipe

Recipes are the part of this repo most worth adding to, and the easiest to
start with. A recipe is one runnable script plus a README that says what you
learn, what you need, and how long it takes. The conventions are in
[`recipes/README.md`](recipes/README.md#conventions); the shortest way in is to
copy [`recipes/_template/`](recipes/_template) into the step it belongs to and
replace the parts in angle brackets.

```bash
cp -r recipes/_template recipes/03-select/my-recipe
sh recipes/03-select/my-recipe/smoke.sh     # what CI will run
```

Two steps past the directory, both of which fail as a red check rather than
locally, so do them before the first push:

1. **Register the entry point** in `tests/recipes/test_offline_examples.py`:
   `run.py` in `CLI_EXAMPLES`, any Modal script in `NEEDS_MODAL`. Every recipe
   directory has to appear in one of the two.
2. **Regenerate the recipe pages** and commit what the generator wrote:

   ```bash
   uv run python scripts/gen_recipe_docs.py          # writes docs/recipes/, docs.json
   uv run python scripts/gen_recipe_docs.py --check  # what CI runs
   ```

[Development](https://docs.withwhile.com/reference/development#contributing-a-recipe)
says what each step fails with.

Two rules on top of the conventions:

- **Every recipe has an offline path.** `smoke.sh` runs the whole script with
  no key, no GPU and no spend, in under a minute: `--dry-run`, `--limit`,
  `--steps`, whatever fits. CI runs every `smoke.sh` in `recipes/` on every
  pull request, so a recipe that needs an A10G still gets its wiring checked
  by a machine.
- **Every measured claim is paired, with an interval, on a held-out set**
  (`pass_at`, `delta_report`). A mean alone is not a result.

A recipe that trains on Modal we verify ourselves on our own account before
merging, because GitHub does not give a fork's pull request access to a
repository's secrets, by design, and we are not working around it. Say in the
PR body what you ran and what it cost, and we will run it.

Found a recipe that does not work? Open an issue with the recipe path, the
command, and what happened. Bad recipes are bugs.

## Golden harness: proving an engine change left `simulate()` alone

Any change under `whileai/simulations/` that could move `simulate()` output
has to be shown to be output-preserving, or its diff has to be stated and
justified. `scripts/golden.py` runs 13 offline configurations at
`concurrency=1` on fixed seeds, scrubs the keys that cannot be reproducible
(wall-clock timings, and the `uuid4` scoring run id that `run_judge` stamps on
every call), and writes one JSON snapshot per configuration.

```bash
python scripts/golden.py capture /tmp/golden/before
# ... make the change ...
python scripts/golden.py capture /tmp/golden/after
python scripts/golden.py diff /tmp/golden/before /tmp/golden/after
```

`diff` exits 0 when every configuration is byte-identical and 1 when any key
differs, naming the dotted path of each change. A whole capture takes about
five seconds and needs no API key or GPU. If the output does change and that
change is the point of the PR, say in the PR body exactly which keys moved and
why.

`scripts/` is otherwise gitignored: local helpers live there and stay local.
`golden.py` is the one committed exception.
