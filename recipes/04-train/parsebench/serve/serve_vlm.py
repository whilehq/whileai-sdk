# vLLM server for the doc-parse agent on Modal, in your own workspace.
#
# Serves the open base VLM plus, when DOCPARSE_ADAPTER is set, one LoRA adapter from
# the docparse-runs volume, as an OpenAI-compatible endpoint the harness calls.
#
#   modal deploy serve/serve_vlm.py
#   DOCPARSE_ADAPTER=prime-rl/<run>/adapters/step_<n> modal deploy serve/serve_vlm.py
#   DOCPARSE_APP=docparse-vlm-tuned DOCPARSE_ADAPTER=... modal deploy serve/serve_vlm.py  # beside the base
#
# URL: https://<your workspace>--docparse-vlm-server-serve.modal.run/v1, Bearer = the VLLM_API_KEY
# in your Modal secret docparse-vllm-key.

import os
import socket
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = os.environ.get(
    "DOCPARSE_APP", "docparse-vlm"
)  # a 2nd app serves an adapter beside the base
BASE_MODEL = os.environ.get("DOCPARSE_BASE", "Qwen/Qwen3.8-27B")
# A relative path under /runs, never a leading slash: Git Bash rewrites "/x/..." env values.
ADAPTER = os.environ.get("DOCPARSE_ADAPTER", "")
GPU = os.environ.get("DOCPARSE_GPU", "H200:1")
MAX_CONTAINERS = int(os.environ.get("DOCPARSE_MAX_CONTAINERS", "4"))

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.26.0", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "DOCPARSE_BASE": BASE_MODEL,
            "DOCPARSE_ADAPTER": ADAPTER,
        }
    )
)

hf_cache = modal.Volume.from_name("docparse-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("docparse-vllm-cache", create_if_missing=True)
runs = modal.Volume.from_name("docparse-runs", create_if_missing=True)

app = modal.App(APP_NAME)
MINUTES = 60
PORT = 8000


@app.cls(
    image=image,
    gpu=GPU,
    scaledown_window=10 * MINUTES,
    timeout=30 * MINUTES,
    max_containers=MAX_CONTAINERS,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("docparse-vllm-key"),
    ],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache, "/runs": runs},
)
@modal.concurrent(max_inputs=64)
class Server:
    @modal.enter()
    def start(self):
        key = os.environ["VLLM_API_KEY"]
        cmd = [
            "vllm",
            "serve",
            os.environ["DOCPARSE_BASE"],
            "--served-model-name",
            "qwen3.8-27b",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--api-key",
            key,
            "--trust-remote-code",
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "64",
            "--gpu-memory-utilization",
            "0.92",
            "--limit-mm-per-prompt",
            '{"image": 2}',
            "--mm-processor-cache-gb",
            "4",
            "--reasoning-parser",
            "qwen3",
        ]
        adapter = os.environ.get("DOCPARSE_ADAPTER")
        if adapter:
            path = f"/runs/{adapter}"
            assert Path(path, "adapter_model.safetensors").exists(), f"no adapter at {path}"
            cmd += [
                "--enable-lora",
                "--max-lora-rank",
                "64",
                "--lora-modules",
                f"qwen3.8-27b-tuned={path}",
            ]
        print(*cmd, flush=True)
        self.proc = subprocess.Popen(cmd)
        while True:
            try:
                socket.create_connection(("localhost", PORT), timeout=1).close()
                return
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {self.proc.returncode}") from None
                time.sleep(2)

    @modal.web_server(port=PORT, startup_timeout=25 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.proc.terminate()
