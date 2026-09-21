#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, no download.
# --dry-run runs the two reward programs on one built-in GSM8K row and
# prints the sizing line, writing nothing.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run
