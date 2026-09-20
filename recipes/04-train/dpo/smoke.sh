#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# train_modal.py needs the modal client and an A10G; what runs here is the
# data path it trains on: prompts, the split, the pairs and the pair report.
set -eu
cd "$(dirname "$0")"
python pairs.py --n 40
