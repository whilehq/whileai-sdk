#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# --offline writes the same file shape with the template writer and a
# scripted agent; the gate and the export then run on it as they would on
# the hosted run.
set -eu
cd "$(dirname "$0")"
python generate.py --offline --situations 6 --k 4 --out data/smoke.jsonl
python diagnose.py data/smoke.jsonl
python export_prompts.py data/smoke.jsonl --out data/smoke-prompts.jsonl
