#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# train_modal.py needs the modal client and an A10G; what runs here is the
# environment it trains in: prompts, the split, the reward, pass@1 and the
# paired delta on two scripted policies.
set -eu
cd "$(dirname "$0")"
python reward.py --n 40
