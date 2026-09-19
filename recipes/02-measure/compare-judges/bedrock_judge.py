"""Claude through AWS Bedrock as a judge callable, under the package's own prompt.

Used with ``python run.py --bedrock`` when you have AWS credentials but no
ANTHROPIC_API_KEY. It sends exactly what ``wai.Judge`` sends a chat model
(the conduct-floor system prompt, the rendered record) and parses the
reply with the same parser, so the numbers are comparable row for row
with ``anthropic:<model>``. Needs ``boto3`` (``uv add boto3``) and a
default AWS profile that can call ``bedrock-runtime`` in ``REGION``.
"""

from __future__ import annotations

import time

from whileai.simulations.score.grade_llm import JUDGE_SYSTEM, _parse_verdict, _user_message

REGION = "us-west-2"
# Claude's JSON reason overran the package's 120-token judge budget on
# 18% of rows in the published run; 400 keeps nearly every verdict whole.
MAX_TOKENS = 400
TRIES = 6
MODEL_IDS = {
    "anthropic:claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic:claude-sonnet-5": "us.anthropic.claude-sonnet-5",
}


class BedrockJudge:
    def __init__(self, model_id: str, region: str = REGION):
        import boto3
        from botocore.config import Config

        self.model_id = model_id
        self.name = f"bedrock:{model_id}"
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=120),
        )

    def __call__(self, row: dict) -> dict:
        payload = _user_message(row, policy="", tools=None, fault_lead=True)
        out = None
        for attempt in range(TRIES):
            try:
                out = self.client.converse(
                    modelId=self.model_id,
                    system=[{"text": JUDGE_SYSTEM}],
                    messages=[{"role": "user", "content": [{"text": payload}]}],
                    inferenceConfig={"maxTokens": MAX_TOKENS},
                )
                break
            except Exception as exc:  # throttling or a transient fault
                if attempt == TRIES - 1:
                    return {"reward": None, "reason": f"error: {exc}"[:200]}
                time.sleep(1.5 * (2**attempt))
        text = "".join(p.get("text", "") for p in out["output"]["message"]["content"])
        score, reason = _parse_verdict(text)
        return {"reward": score, "reason": reason or text[:200]}
