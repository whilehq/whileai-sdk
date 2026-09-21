#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
set -eu
cd "$(dirname "$0")"
python migrate.py --rows 24
python project.py out/rows.v1.jsonl
