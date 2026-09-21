#!/usr/bin/env bash
# Start a Bedrock Custom Model Import job for a merged model already in S3 and
# poll it to completion, then print the imported model's ARN and its size in
# Custom Model Units.
#
#   BUCKET=<bucket> ROLE=arn:aws:iam::<account>:role/<import-role> \
#   bash import_job.sh <name> [region]
#
# The role trusts bedrock.amazonaws.com and can s3:GetObject / s3:ListBucket
# on the bucket; the README has both policy documents.
set -euo pipefail
NAME="$1"
REGION="${2:-us-east-1}"
: "${BUCKET:?set BUCKET to the S3 bucket holding $NAME/}"
: "${ROLE:?set ROLE to the import service role ARN}"
JOB="${NAME}-$(date +%Y%m%d-%H%M%S)"

echo "files in s3://$BUCKET/$NAME/:"
aws s3 ls "s3://$BUCKET/$NAME/" --region "$REGION" --human-readable | awk '{print "  "$3" "$4"  "$5}'

ARN=$(aws bedrock create-model-import-job --region "$REGION" \
  --job-name "$JOB" --imported-model-name "$NAME" --role-arn "$ROLE" \
  --model-data-source "{\"s3DataSource\":{\"s3Uri\":\"s3://$BUCKET/$NAME/\"}}" \
  --query jobArn --output text)
echo "job: $ARN"

START=$(date +%s)
while true; do
  STATUS=$(aws bedrock get-model-import-job --region "$REGION" --job-identifier "$ARN" --query status --output text)
  echo "  $(( ($(date +%s) - START) / 60 )) min: $STATUS"
  case "$STATUS" in
    Completed|Complete) break ;;
    Failed)
      aws bedrock get-model-import-job --region "$REGION" --job-identifier "$ARN" --query failureMessage --output text
      exit 1 ;;
  esac
  sleep 60
done

aws bedrock get-imported-model --region "$REGION" --model-identifier "$NAME" \
  --query "{arn:modelArn,arch:modelArchitecture,cmu:customModelUnits.customModelUnitsPerModelCopy,cmuVersion:customModelUnits.customModelUnitsVersion}" \
  --output json
