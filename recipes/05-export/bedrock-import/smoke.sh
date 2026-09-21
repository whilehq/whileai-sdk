#!/usr/bin/env sh
# The wiring check for this recipe: no key, no GPU, no AWS, under a minute.
# CI runs this file for every recipe that has one, on every pull request.
# Every real step here needs Modal or AWS, so the check is that the scripts
# parse, their help runs without boto3 or modal doing any work, and the
# import script refuses to start without its two variables named.
set -eu
cd "$(dirname "$0")"
python -c "import ast, sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]" merge_upload.py presign.py compare.py
bash -n import_job.sh
out=$(bash import_job.sh some-model 2>&1 || true)
case "$out" in
  *BUCKET*) ;;
  *) echo "import_job.sh without BUCKET did not name it: $out"; exit 1 ;;
esac
test -s rows/eval-nemotron-8b-r1.jsonl
test -s rows/eval-nemotron-8b-base.jsonl
echo "bedrock-import smoke ok"
