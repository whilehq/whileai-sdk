#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# The firectl job and prove.py need a Fireworks account; what runs here is
# the data path they consume: the rollouts, the grade, the two files in
# Fireworks' shapes, and the shape check on both.
set -eu
cd "$(dirname "$0")"
python export_fireworks.py --n 16 --out out
