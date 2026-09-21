#!/usr/bin/env sh
# The wiring check for this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# Every step of the round trip is a platform call, so with no key the whole
# check is that the imports resolve and the no-key run is one sentence
# naming WHILEAI_API_KEY, not a traceback.
set -eu
cd "$(dirname "$0")"
python roundtrip.py --help >/dev/null
out=$(WHILEAI_API_KEY= WHILEAI_HOME=/nonexistent python roundtrip.py 2>&1 || true)
echo "$out"
case "$out" in
  *Traceback*) echo "no-key run raised instead of exiting with a message"; exit 1 ;;
  *WHILEAI_API_KEY*) ;;
  *) echo "no-key run does not name WHILEAI_API_KEY"; exit 1 ;;
esac
