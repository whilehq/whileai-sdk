#!/usr/bin/env sh
# The offline path through this recipe: the task build, the harness search on
# scripted candidates, the 2x2 grid with the weights arms marked not run. No
# key, no GPU, under a minute. CI runs this file on every pull request.
set -eu
cd "$(dirname "$0")"
python recipe.py --dry-run --fresh --limit 16 --k 2
