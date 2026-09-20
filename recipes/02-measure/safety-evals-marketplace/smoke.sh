#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# live.py needs a model through Ollama; here it only has to import.
set -eu
cd "$(dirname "$0")"
python run.py --k 2
python live.py --help >/dev/null
