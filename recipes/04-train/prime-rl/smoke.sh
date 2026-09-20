#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# --dry-run writes the three prime-rl configs from wai.prime_rl_config and
# stops; --validate and the runs themselves need the modal client.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run
