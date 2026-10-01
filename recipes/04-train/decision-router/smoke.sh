#!/usr/bin/env sh
# The offline path: seeded stand-in predictions through the scoring, no table, no GPU.
# CI runs this file for every recipe that has one, on every pull request.
set -eu
cd "$(dirname "$0")"
uv run --with numpy python run.py --dry-run --out out/smoke
