#!/bin/sh
# Free: the offline half of the recipe. No key, no GPU, no network.
set -e
cd "$(dirname "$0")"
python run.py --dry-run
python -m py_compile run.py spec.py arm_selectors.py train_modal.py eval_modal.py \
    serve_modal.py fresh_traffic.py report_platform.py
echo "smoke ok"
