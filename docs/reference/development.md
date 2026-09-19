---
title: "Package layout and development"
sidebarTitle: "Development"
description: "Where the code lives, how to run the tests, lint and type checks, and what CI runs."
---

## Package layout

The public surface is the package itself: `import whileai.simulations as wai`. Internals are grouped by stage and may move between releases.

| folder or file | what lives there |
|---|---|
| `generate/` | situation grid, writer, diversity selection, agent runners and adapters |
| `score/` | conduct checks, judges, quality ranking, selection for SFT and RL, the trust and delta reports |
| `verify/` | verifiers: programmatic rewards (`MathEqual`, `CodeExec`, `JSONSchema`, ...) that honor the judge contract |
| `ingest/` | trace loading, OpenTelemetry rows (`gen_ai.usage.*` sums into `row["usage"]`), platform push and pull |
| `world/` | the mock tool environment (`WorldOptions` in `sandbox.py`) |
| `run/` | the engine behind `simulate()`: knob resolution (`config.py`), spec loading (`spec.py`), row helpers (`rows.py`), and the scheduler itself (`engine.py`: inputs, build, loop, finish) |
| `simulation.py`, `data.py`, `export.py` | the `simulate()` entry point, its result object, and training export |
| `schema.py`, `schemas/` | the typed row (`Task`, `Rollout`, `Judgment`, `Marker`) and the `row-v1.json` wire contract |
| `defaults.py` | every default the engine and the reports use, each with the reason it is what it is |
| `environment.py` | `export_environment` and `load_environment`: a run as an installable RL environment |
| `training.py` | hosted training runs, `training_run`, `serve`, `reward_model` |
| `monitor.py` | `HackMonitor`, the during-training reward-hacking watch |

Source: [github.com/whilehq/whileai-sdk/tree/main/whileai/simulations](https://github.com/whilehq/whileai-sdk/tree/main/whileai/simulations).

## Development

```bash
uv sync --extra dev
uv run pytest           # under a minute on four cores, no network; -n0 runs it serially
uv run ruff check .     # lint; --fix for the mechanical ones
uv run mypy             # type check (the gate)
uv run ty check         # same check, under a second; mypy stays the gate until ty is 1.0
pre-commit install      # optional: ruff and whitespace hooks on commit
```

CI runs the suite on Python 3.10 through 3.13, every recipe's `smoke.sh`, ruff, mypy, ty, line coverage, a plain-pip install of the built wheel into a clean venv (the check that catches a uv-only source pin), and a version-scheme check.

## The code in the docs runs

`uv run python scripts/check_doc_snippets.py` executes every ```` ```python ```` block under `docs/` against the installed package and CI runs it on every PR (the job `the code in the docs runs`). A page is one program: its blocks run in order, in one namespace, in a scratch directory, with no keys set and anything credential-shaped stripped from the environment. Where a page quotes a block's output in the fence right after it, the quoted text has to match what the block printed; a block written as a session (`>>> call` followed by what it printed or raised) is checked line by line, with an exception compared as `ValueError: message`. Each block gets 180 seconds; a failure names the page and the line of the opening fence.

Three things are not run:

- `docs/api/` and `docs/recipes/`, which are generated (from docstrings by `gen_api_docs.py`, from the recipe READMEs by `gen_recipe_docs.py`). Recipe blocks assume a clone and are covered by the recipe smoke job, so fix a recipe's README rather than its page.
- A block that needs a key, when a sentence or `export` line above it on the page names the variable (`WHILEAI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MODAL_TOKEN_ID`, `TYPESAFE_API_KEY`). A block that reaches for a key the page never mentions fails.
- A block that is a sketch rather than a program (a signature, the shape of a return value, a live training run), listed in `scripts/doc_snippets/skips.json` by page and a substring of its code, with a reason. The list sits off the page because Mintlify renders a fence info string as a filename badge and an HTML comment breaks its MDX parser.

Names a guide leaves to the reader (`agent`, `TOOLS`, `POLICY`, `scored`) come from a fixture under `scripts/doc_snippets/` that mirrors the page's path below `docs/` and runs before its first block; most fixtures are one line, `from _common import *`. Check one page with `--page docs/evals.md -v`, or a released wheel with `--python /path/to/venv/bin/python`.

## License

Apache-2.0
