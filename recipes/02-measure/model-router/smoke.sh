#!/usr/bin/env sh
# The offline path through this recipe: seeded stand-in models, no key, no calls.
# CI runs this file for every recipe that has one, on every pull request.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run --limit 120 --out out/smoke
