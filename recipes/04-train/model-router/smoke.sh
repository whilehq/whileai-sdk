#!/usr/bin/env sh
# The offline path: a seeded stand-in table through all three routers, no download.
# CI runs this file for every recipe that has one, on every pull request.
set -eu
cd "$(dirname "$0")"
uv run --with numpy python run.py --dry-run --out out/smoke
