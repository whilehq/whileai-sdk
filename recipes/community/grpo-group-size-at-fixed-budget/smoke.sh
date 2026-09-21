#!/bin/sh
# Offline: both halves of the recipe on synthetic arms. No key, no GPU, no network.
set -e
python "$(dirname "$0")/run.py" --dry-run
