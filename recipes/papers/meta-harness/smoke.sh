#!/usr/bin/env sh
# The offline path through this recipe: scripted candidates, no key, no GPU,
# under a minute. CI runs this file on every pull request.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run --propose --select --fresh --budget 12 --k 2
