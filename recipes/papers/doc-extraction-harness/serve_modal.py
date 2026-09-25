"""Serve one open model, no adapter, as an OpenAI-compatible vLLM endpoint on
Modal, under an app named ``docx-serve-<slug>``: this recipe's own app, never
another recipe's. A trimmed copy of ``recipes/04-train/text-to-sql/serve_modal.py``
(same image, same vLLM pin, same scale-to-zero), with the app prefix changed
and the adapter path removed.

    python serve_modal.py --model nvidia/Llama-3.1-Nemotron-Nano-8B-v1 --key <secret>  # writes the config
    DOCX_SERVE_CONFIG=serve_nvidia-llama-3-1-nemotron-nano-8b-v1.json PYTHONUTF8=1 modal deploy serve_modal.py
    DOCX_SERVE_CONFIG=... python serve_modal.py --url                                  # the endpoint
    modal app stop docx-serve-<slug>                                                    # when done

Cost rules: ``min_containers`` is never set (it defaults to 0: scale to zero),
``max_containers`` is 2, ``scaledown_window`` is 600 s, one L40S ($1.95/hour on Modal's list price).
The settings live in a JSON file next to this one (gitignored), named by
``DOCX_SERVE_CONFIG``; Modal re-imports this module in the container, where
the settings arrive through the secret.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / os.environ.get("DOCX_SERVE_CONFIG", "serve_config.json")
_cfg: dict = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
MODEL = _cfg.get("model") or "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
KEY = _cfg.get("key") or ""
# GPU: a fallback list, first free wins; an 8B model at 16k context fits
# each. The GPU a container landed on is read from its logs for the spend.
GPU = _cfg.get("gpu") or ["L40S", "A100-40GB", "A100-80GB", "H100"]
# MAX_LEN = 16384: the longest conversation here (a 3 kB document, four turns
# of 1024 tokens) fits with room; the text-to-sql server's default.
MAX_LEN = int(_cfg.get("max_len") or 16384)
# 0.10.0: the pin the text-to-sql Nemotron and Qwen3 rows were measured on.
VLLM_VERSION = _cfg.get("vllm") or "0.10.0"
SLUG = re.sub(r"[^a-z0-9]+", "-", MODEL.lower()).strip("-")[-40:]
SCALEDOWN_S = 600
MAX_CONTAINERS = 2

app = modal.App(f"docx-serve-{SLUG}")
image = (
    modal.Image.debian_slim(python_version="3.11")
    # transformers 5 renamed tokenizer internals vLLM 0.10.0 reads
    # (all_special_tokens_extended), so it is pinned to the 4.x line.
    .pip_install(
        f"vllm=={VLLM_VERSION}", "transformers==4.55.4", "huggingface_hub[hf_transfer]>=0.34.4"
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "0",
        }
    )
)
HF_CACHE = _cfg.get("hf_cache") or "docx-hf-cache"  # this recipe's own volume
hf_cache = modal.Volume.from_name(HF_CACHE, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 60 * 60,
    scaledown_window=SCALEDOWN_S,
    # MAX_CONTAINERS = 2: one loop's 64 rollouts fill one container; a second
    # loop (the noise floor beside a round) may add one more, never a third.
    max_containers=MAX_CONTAINERS,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[
        modal.Secret.from_dict(
            {
                "VLLM_API_KEY": KEY,
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "DOCX_SERVE_MODEL": MODEL,
                "DOCX_SERVE_MAX_LEN": str(MAX_LEN),
            }
        )
    ],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    subprocess.Popen(
        [
            "vllm",
            "serve",
            os.environ["DOCX_SERVE_MODEL"],
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--api-key",
            os.environ["VLLM_API_KEY"],
            "--max-model-len",
            os.environ.get("DOCX_SERVE_MAX_LEN") or "16384",
            "--gpu-memory-utilization",
            "0.90",
            "--dtype",
            "bfloat16",
        ]
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="write the serve config, then: modal deploy serve_modal.py"
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--key", default=KEY, help="the API key the server will require")
    ap.add_argument(
        "--gpu", default=GPU, nargs="+", help="one GPU type, or several in fallback order"
    )
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--vllm", default=VLLM_VERSION)
    ap.add_argument("--hf-cache", default=HF_CACHE)
    ap.add_argument("--url", action="store_true", help="print the deployed endpoint URL and exit")
    a = ap.parse_args()
    if a.url:
        print(modal.Function.from_name(app.name, "serve").get_web_url())
        raise SystemExit(0)
    if not a.key:
        raise SystemExit("--key: the API key the server will require (VLLM_API_KEY for callers)")
    slug = re.sub(r"[^a-z0-9]+", "-", a.model.lower()).strip("-")[-40:]
    dest = HERE / f"serve_{slug}.json"
    dest.write_text(
        json.dumps(
            {
                "model": a.model,
                "key": a.key,
                "gpu": a.gpu,
                "max_len": a.max_len,
                "vllm": a.vllm,
                "hf_cache": a.hf_cache,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {dest.name}; now: DOCX_SERVE_CONFIG={dest.name} PYTHONUTF8=1 modal deploy serve_modal.py "
        f"(app docx-serve-{slug})"
    )
