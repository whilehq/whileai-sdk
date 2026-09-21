#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, no download.
# --dry-run builds a small slice of traffic, grades it with the program
# reward, prints the split and stops before the Modal step, writing nothing.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run
