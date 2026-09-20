#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, no network, under a
# minute. Writes a small export the way lesson 7 does, loads it back through
# the trainer's own loaders, prints the plan the trainer gets, and scores a
# row no stand-in wrote. CI runs this file on every pull request.
set -e
cd "$(dirname "$0")"
python wiring.py --smoke | tail -4
