#!/usr/bin/env sh
# The offline path through this recipe: the frozen documents, one search
# round over the eight checked-in candidates on a scripted model and a
# scripted judge, the three-run noise floor, the gate, the judge audit and
# the report, on 16 documents a split. No key, no GPU, under a minute.
# CI runs this file on every pull request.
set -eu
cd "$(dirname "$0")"
python run.py search --dry-run --fresh --limit 16
python run.py noise --dry-run --limit 16
python run.py gate --dry-run --limit 16
python run.py audit --dry-run --limit 16
python run.py report --dry-run --limit 16
