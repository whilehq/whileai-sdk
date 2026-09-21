#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, no download.
# --dry-run builds the rows and the TRL export and stops before the
# tokenizer; train_modal.py needs the modal client and an A10G.
set -eu
cd "$(dirname "$0")"
python run.py --dry-run --out out/smoke
