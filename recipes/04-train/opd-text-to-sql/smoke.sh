#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# --dry-run writes the two prime-rl configs from wai.prime_rl_config, runs
# the selftest (the gate, the sign test, the tie split, the holdout pin)
# and prints the plan; every later stage needs Modal.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run
