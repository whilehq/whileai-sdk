# Serve a trained adapter on Amazon Bedrock

Take a LoRA adapter you trained, merge it into its base, import the merged
weights into your own AWS account with Bedrock Custom Model Import, and
measure the served model on the same held-out tasks that scored it on vLLM.
The point is the last step: a deployment is only done when the number
survives the move.

What you will learn: what Bedrock accepts (merged weights, not adapters),
which API an imported model answers (InvokeModel, not Converse), what the
cold start costs, and how to check the served weights against the training
run with a paired interval. You need a Modal account, AWS credentials that
can create an S3 bucket, an IAM role and a Bedrock import job, and the
text-to-SQL recipe's Postgres for the measurement. Merge and upload take
about six minutes, the import about ten, the 560-sample measurement two.

## Run it

```bash
uv add whileai boto3 modal
cd recipes/05-export/bedrock-import

# 1. merge on Modal (CPU, 64 GB), shards under 4 GB, transformers 4.51.3
modal run merge_upload.py::merge_entry \
  --base nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \
  --adapter while-ai/text-to-sql-shop-nemotron-8b-r1 \
  --name nemotron-8b-t2s-r1

# 2. presign one PUT per file locally, upload from Modal (Modal never sees AWS keys)
aws s3 mb s3://<bucket> --region us-east-1
python presign.py nemotron-8b-t2s-r1 files.json --bucket <bucket> > urls.json
modal run merge_upload.py::upload_entry --name nemotron-8b-t2s-r1 --urls urls.json

# 3. import and wait (the role is below)
BUCKET=<bucket> ROLE=arn:aws:iam::<account>:role/<import-role> bash import_job.sh nemotron-8b-t2s-r1

# 4. measure: the same rollout.py the vLLM numbers came from, on the served ARN
cd ../../04-train/text-to-sql
python rollout.py --agent "bedrock:<imported-model-arn>@us-east-1" --split holdout --limit 140 --k 4
cd ../../05-export/bedrock-import
python compare.py ../../04-train/text-to-sql/raw/bedrock-arn-aws-bedrock-*.jsonl
```

| step | flag or variable | what it does |
|---|---|---|
| merge | `--base`, `--adapter`, `--name` | Hub ids of the base and the adapter; the folder name in S3 and the import's name |
| presign | `--bucket`, `--region` | where the shards go; the URLs expire after twelve hours |
| import | `BUCKET`, `ROLE`, second argument | bucket, service role, region (default `us-east-1`) |
| measure | `--limit 140 --k 4` | the first 140 held-out tasks, four samples each: the ledger's 140-task set |

The service role trusts Bedrock and reads the one bucket:

```json
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock.amazonaws.com"},
 "Action":"sts:AssumeRole","Condition":{"StringEquals":{"aws:SourceAccount":"<account>"},
 "ArnEquals":{"aws:SourceArn":"arn:aws:bedrock:us-east-1:<account>:model-import-job/*"}}}]}
```

```json
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],
 "Resource":["arn:aws:s3:::<bucket>","arn:aws:s3:::<bucket>/*"]}]}
```

## What you get

The published run (2026-09-20, us-east-1). The adapter is
[`while-ai/text-to-sql-shop-nemotron-8b-r1`](https://huggingface.co/while-ai/text-to-sql-shop-nemotron-8b-r1),
one GRPO round on Nemotron-Nano-8B from the
[text-to-SQL recipe](../../04-train/text-to-sql); the vLLM rows are its
`eval-nemotron-8b-r1` and `eval-nemotron-8b-base` files on the Hub, checked
in under `rows/` so `compare.py` runs offline.

```text
Bedrock rows: 560  (140 tasks, k=4)

bedrock r1   tasks 140  pass@1 0.345 (0.273..0.414)  pass@k 0.457  executes 0.627  no sql 0.000  truncated 0.000
vllm r1      tasks 140  pass@1 0.350 (0.275..0.421)  pass@k 0.450
vllm base    tasks 140  pass@1 0.263 (0.200..0.329)  pass@k 0.386

bedrock r1 - vllm r1         -0.005 (-0.037..+0.027) over 140 tasks, covers zero
bedrock r1 - vllm base       +0.082 (+0.045..+0.123) over 140 tasks, excludes zero
vllm r1 - vllm base          +0.087 (+0.048..+0.130) over 140 tasks, excludes zero

by difficulty (pass@1):
  bedrock r1   easy 0.59 (n=48)  medium 0.30 (n=44)  hard 0.15 (n=48)
  vllm r1      easy 0.58 (n=48)  medium 0.28 (n=44)  hard 0.18 (n=48)
  vllm base    easy 0.54 (n=48)  medium 0.19 (n=44)  hard 0.06 (n=48)
```

The served weights are the trained weights: the paired difference between
Bedrock and vLLM on the same 140 tasks is five thousandths with an interval
that covers zero, and the gain over the base reproduces the published
+0.087 as +0.082 with an interval that excludes zero. The verifier was
checked first: run locally on the published rows it agrees with every one
of the 1,120 published rewards (`compare.py --check-published`).

## What it cost and what to know

| fact | measured |
|---|---|
| merge on Modal, 8 CPUs, 64 GB | base loaded 82 s, merged 20 s, saved 22 s |
| upload Modal to S3 | 16.08 GB in five shards, 33 to 53 MB/s per file |
| import job | 10 minutes, `Completed`; architecture `llama`, 2 Custom Model Units |
| first call after import | 96 s (`ModelNotReadyException` until the model is restored) |
| 560 samples, concurrency 8 | 103 s |
| price while serving | $0.05718 per Custom Model Unit per minute in us-east-1, billed in five-minute windows from the first call; $1.95 per unit per month of storage |

Three facts the AWS documentation states in scattered places, checked here:

- Custom Model Import takes **merged weights** in the Hugging Face layout
  (safetensors, `config.json`, tokenizer files). A separate LoRA adapter
  is not accepted; `merge_upload.py` merges it first. Supported
  architectures include Llama 2 to 3.3, Mistral, Mixtral, Qwen2, Qwen2.5,
  Qwen3 (`Qwen3ForCausalLM` and the MoE), GPT-OSS. Regions: us-east-1,
  us-east-2, us-west-2, eu-central-1.
- An imported model **refuses Converse** even on a Llama architecture
  ("This action doesn't support the model that you provided") and answers
  `InvokeModel` with the OpenAI chat-completion body. The SDK picks that
  route from the ARN, so `bedrock:<arn>@<region>` works anywhere a backend
  goes. Tool calling on imports is honored for GPT-OSS only.
- An idle import is **unloaded**. The SDK waits through the restore (ten
  tries, fifteen seconds apart) and then says so in one sentence.

## Next

- Serve it to your users from While: register the ARN once and it answers at
  `https://models.withwhile.com/v1` under your While key, from any OpenAI client
  or `wai.Endpoint(name, url=..., api_key=...)`. The role it assumes is the
  `WhileModelsInvoke` shape in [Your model and your key](../../../docs/get-started/your-model-and-key.mdx):

  ```bash
  curl -X POST https://models.withwhile.com/models \
    -H "Authorization: Bearer $WHILEAI_API_KEY" -H "Content-Type: application/json" \
    -d '{"name": "nemotron-8b-t2s-r1", "arn": "<imported-model-arn>", "roleArn": "arn:aws:iam::<account>:role/WhileModelsInvoke"}'
  ```

- Delete the import when you are done measuring:
  `aws bedrock delete-imported-model --model-identifier <name> --region us-east-1`;
  storage is billed per unit per month.
- A Qwen3 adapter (`while-ai/airline-concise-4b` on `Qwen/Qwen3-4B-Instruct-2507`)
  takes the same path; its measurement is `wai.simulate` on the airline
  tasks rather than `rollout.py`.
- The same held-out set on the platform: push the graded rows with
  `wai.export(..., push_to=)` and the two runs sit side by side on the
  Runs page.
