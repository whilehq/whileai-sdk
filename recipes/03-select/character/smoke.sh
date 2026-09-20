#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# run.py is offline by default (a scripted student); --after is the scripted
# trained one, and measure.py reads the two holdout files it wrote.
set -eu
cd "$(dirname "$0")"
python run.py --out out/smoke-before --k 2
python run.py --out out/smoke-after --k 2 --after
python measure.py out/smoke-before/holdout.jsonl out/smoke-after/holdout.jsonl
