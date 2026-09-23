#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, no spend.
# CI runs this file for every recipe that has one, on every pull request.
set -eu
cd "$(dirname "$0")"
uv run --with modal python recipe.py --selftest
