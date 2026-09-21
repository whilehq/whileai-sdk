#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, no network, under a
# minute. A scripted student replays the set's labeled replies and a lookup
# judge grades against them, so the whole before/after loop runs deterministically.
# CI runs this file on every pull request.
set -eu
cd "$(dirname "$0")"
python run.py --k 2 --out out/smoke
