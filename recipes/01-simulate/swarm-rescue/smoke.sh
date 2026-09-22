#!/usr/bin/env sh
# The offline path: three toy tasks, a fake model, no key, under a minute.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run
