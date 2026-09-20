#!/usr/bin/env sh
# The offline path through this recipe: no key, no GPU, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# --control-file stands in for the simulated controls (which need a key);
# train_modal.py and eval_modal.py need the modal client and an A10G.
set -eu
cd "$(dirname "$0")"
mkdir -p out/smoke
python -c "import json; print('\n'.join(json.dumps({'prompt': f'plain ask {i}', 'answer': f'plain answer {i}'}) for i in range(140)))" > out/smoke/controls.jsonl
python generate.py --name Zed --maker "Zed Labs" --identity 20 --out out/smoke --control-file out/smoke/controls.jsonl
