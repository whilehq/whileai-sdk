#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# --dry-run is the data step with nothing pushed; train, serve and call are
# platform calls and need the key.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run --budget 24
