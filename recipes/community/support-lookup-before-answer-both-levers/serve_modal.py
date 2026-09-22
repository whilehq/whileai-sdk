"""Serve the winning arm on my own Modal: vLLM, --enable-lora, scale to zero.

    modal deploy serve_modal.py
    python fresh_traffic.py --url <the printed url> --arm both
    modal app stop support-lookup-serve

The adapter comes off the same volume the trainer wrote it to, so what is
served is the artifact the eval scored, not a re-export of it.
"""

from __future__ import annotations

import os
import subprocess

import modal

from whileai.config import requirement

BASE_MODEL = os.environ.get("SUPPORT_BASE", "Qwen/Qwen3-4B")
GPU = os.environ.get("SUPPORT_GPU", "L40S")
VOL = "/vol"
PORT = 8000

app = modal.App("support-lookup-serve")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.29.0", "peft==0.21.0", requirement())
    .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_USE_V1": "1"})
)

runs = modal.Volume.from_name("support-lookup-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
    timeout=60 * 30,
    scaledown_window=60 * 3,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=60 * 10)
def serve():
    cmd = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
        "--enable-prefix-caching",
        "--enable-lora",
        "--max-lora-rank",
        "32",
        "--lora-modules",
        f"both={VOL}/both/adapter",
        f"weights={VOL}/weights/adapter",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
