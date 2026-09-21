"""Serve the winning arm on your own Modal: vLLM, the LoRA adapter, scale to zero.

    modal deploy serve_modal.py
    python fresh_traffic.py --url https://<your-workspace>--voice-concise-filter-serve-serve.modal.run/v1

An OpenAI-compatible endpoint with both adapters mounted, so the same URL
answers as `base`, `baseline` or `method` depending on the model name asked
for. `scaledown_window` sends the container away after five idle minutes;
stop the app when you are done:

    modal app stop voice-concise-filter-serve
"""

from __future__ import annotations

import os
import subprocess

import modal

BASE_MODEL = os.environ.get("VOICE_BASE_MODEL", "Qwen/Qwen3-1.7B")
VOL = "/vol"
PORT = 8000

app = modal.App("voice-concise-filter-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_USE_V1": "1"})
)

runs = modal.Volume.from_name("voice-filter-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
    scaledown_window=300,
    timeout=60 * 30,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=60 * 10)
def serve() -> None:
    lora = []
    for arm in ("baseline", "method"):
        p = f"{VOL}/{arm}/adapter"
        if os.path.exists(p):
            lora.append(f"{arm}={p}")
    cmd = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--served-model-name",
        "base",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.85",
        "--dtype",
        "bfloat16",
    ]
    if lora:
        cmd += ["--enable-lora", "--max-lora-rank", "32", "--lora-modules", *lora]
    print("launching:", " ".join(cmd), flush=True)
    subprocess.Popen(" ".join(cmd), shell=True)
