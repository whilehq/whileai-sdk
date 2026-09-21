"""Serve the winning adapter on your own Modal with vLLM, scale-to-zero.

    modal deploy recipes/community/identity-spec-no-unasked-maker-aas/serve_modal.py
    python fresh_traffic.py --url https://<your-workspace>--identity-aas-serve-serve.modal.run
    modal app stop identity-aas-serve

An OpenAI-compatible endpoint with both adapters attached, so fresh traffic
can be sent at either arm by changing only the ``model`` field. No
``min_containers``, so it costs nothing between calls; stop the app when
you are done with it.
"""

from __future__ import annotations

import subprocess

import modal

BASE_MODEL = "Qwen/Qwen3-1.7B"
PORT = 8000

app = modal.App("identity-aas-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.0", "transformers==4.54.0")
    .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_USE_V1": "0"})
)

runs = modal.Volume.from_name("identity-aas-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("identity-aas-hf-cache", create_if_missing=True)
VOL = "/vol"


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    scaledown_window=5 * 60,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve() -> None:
    cmd = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--port",
        str(PORT),
        "--host",
        "0.0.0.0",
        "--max-model-len",
        "4096",
        "--dtype",
        "bfloat16",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--max-loras",
        "2",
        "--lora-modules",
        f"aas={VOL}/aas/adapter",
        f"loss={VOL}/loss/adapter",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
