"""One vLLM server for every scored arm: the base model and both SFT adapters.

Scoring base and adapters behind the same server, the same tool parser and the
same sampling settings is what makes the before/after about the weights and
nothing else. Thinking is off for every request (`enable_thinking=false`),
because smithtune's training rows omit reasoning.

    modal deploy serve_modal.py
    python collect.py --arm sft-with --split holdout --base-url <url>/v1 --model sft-with --api-key-env REVIEW_SERVER_KEY
    modal app stop deepagents-review-serve

The tool parser (`qwen35_tool_parser.py`) parses on token ids with the
grammar the model's renderer uses; vLLM's regex-based qwen3_xml parser leaves
a well-formed call in the content on a measurable share of turns.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# Named at deploy time, never discovered: a run still training writes interim
# checkpoints to the same volume.   REVIEW_ADAPTERS=sft-with,sft-without modal deploy ...
ADAPTERS = tuple(a for a in os.environ.get("REVIEW_ADAPTERS", "").split(",") if a)
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "vllm==0.26.0",
        "huggingface_hub[hf_transfer]",
        "renderers @ git+https://github.com/PrimeIntellect-ai/renderers.git",
    )
    .add_local_file(
        Path(__file__).parent / "qwen35_tool_parser.py", "/root/qwen35_tool_parser.py", copy=True
    )
    .env(
        {"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "REVIEW_ADAPTERS": ",".join(ADAPTERS)}
    )
)
runs = modal.Volume.from_name("deepagents-review-runs", create_if_missing=True)
hf = modal.Volume.from_name("deepagents-review-hf", create_if_missing=True)
app = modal.App("deepagents-review-serve")


@app.cls(
    image=image,
    gpu="H200",
    scaledown_window=15 * 60,
    timeout=24 * 60 * 60,
    max_containers=6,  # 12 adapters at once in the fixed-epoch rerun
    volumes={"/runs": runs, "/hf": hf},
    secrets=[modal.Secret.from_name("review-server-key")],
)
@modal.concurrent(max_inputs=32)
class Server:
    @modal.enter()
    def start(self):
        lora = [
            f"{a}=/runs/{a}/adapter"
            for a in ADAPTERS
            if Path(f"/runs/{a}/adapter/adapter_config.json").exists()
        ]
        cmd = [
            "vllm",
            "serve",
            BASE_MODEL,
            "--revision",
            REVISION,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--api-key",
            os.environ["REVIEW_SERVER_KEY"],
            "--max-model-len",
            "65536",
            "--max-num-seqs",
            "32",
            "--gpu-memory-utilization",
            "0.9",
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": False}),
            "--enable-auto-tool-choice",
            "--tool-parser-plugin",
            "/root/qwen35_tool_parser.py",
            "--tool-call-parser",
            "prime_qwen35",
        ]
        if lora:
            cmd += [
                "--enable-lora",
                "--max-lora-rank",
                "8",
                "--max-loras",
                str(len(lora)),
                "--lora-modules",
                *lora,
            ]
        print(*cmd, flush=True)
        self.proc = subprocess.Popen(cmd)
        while True:
            try:
                socket.create_connection(("localhost", PORT), timeout=1).close()
                return
            except OSError as err:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {self.proc.returncode}") from err
                time.sleep(2)

    @modal.web_server(port=PORT, startup_timeout=30 * 60)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.proc.terminate()
