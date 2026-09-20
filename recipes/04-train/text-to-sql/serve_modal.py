"""Serve any Hugging Face model as an OpenAI-compatible endpoint on Modal (vLLM).

The account's hosted endpoint serves Qwen3-4B and Phi-4. To benchmark or train
another base (a Nemotron, a Llama, your own checkpoint) the SDK only needs an
OpenAI-compatible URL and a key, so:

    python serve_modal.py --model nvidia/Llama-3.1-Nemotron-Nano-8B-v1 --key <secret>   # writes serve_config.json
    PYTHONUTF8=1 modal deploy serve_modal.py
    python serve_modal.py --url                                                          # the endpoint

    VLLM_API_KEY=<secret> python rollout.py \\
        --agent "vllm:nvidia/Llama-3.1-Nemotron-Nano-8B-v1@https://...modal.run/v1" \\
        --system-prefix "detailed thinking on" --split holdout --k 4

`--adapter volume:<run_id>` serves a LoRA adapter from the training volume on
top of the base (`--enable-lora`), as model id `<base>-adapter`. Settings live in
serve_config.json (gitignored), not environment variables: Modal imports this
module again inside the container, and on some shells `modal deploy` does not
see the caller's variables either. The server runs with a tool-call parser
because the SDK sends drafted tool schemas with every request. One L40S
(models up to ~24 GB) or H100; scale-to-zero after 10 idle minutes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "serve_config.json"
_cfg: dict = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
MODEL = _cfg.get("model") or "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
KEY = _cfg.get("key") or "t2s-serve"
ADAPTER = _cfg.get("adapter") or ""  # "volume:<run_id>"
GPU = _cfg.get("gpu") or "L40S"
MAX_LEN = int(_cfg.get("max_len") or 16384)
# "llama3_json" for Llama-family models (Nemotron-Nano-8B-v1), "hermes" for Qwen.
TOOL_PARSER = _cfg.get("tool_parser") or ("hermes" if "qwen" in MODEL.lower() else "llama3_json")
# 0.10.0 is the pin the Qwen3 / Nemotron rows were measured on; qwen3_5-family
# checkpoints (Qwen3.5-*, a Qwen3_5ForConditionalGeneration) need >= 0.26.
VLLM_VERSION = _cfg.get("vllm") or "0.10.0"
SLUG = re.sub(r"[^a-z0-9]+", "-", MODEL.lower()).strip("-")[-40:]

app = modal.App(f"t2s-serve-{SLUG}")
# Newer vLLM builds kernels at start (Qwen3.5's linear attention among them) and
# needs nvcc: a CUDA devel base, not debian_slim.
_img = (
    modal.Image.debian_slim(python_version="3.11")
    if VLLM_VERSION.startswith("0.10")
    else modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
)
image = _img.pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]>=0.34.4").env(
    {
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        **({"VLLM_USE_V1": "0"} if VLLM_VERSION.startswith("0.10") else {}),
    }
)
# Runs trained before the whileai rename live on the zeroproof-* volumes;
# `--runs-volume` / `--hf-cache` point a deploy at them.
RUNS_VOLUME = _cfg.get("runs_volume") or "whileai-train-runs"
HF_CACHE = _cfg.get("hf_cache") or "whileai-hf-cache"
hf_cache = modal.Volume.from_name(HF_CACHE, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)


def _rename_for_vllm(path: str) -> None:
    """PEFT under transformers 5 saves a Qwen3.5 (VLM-class) adapter as
    ``base_model.model.model.layers.N.*``; vLLM keeps that text stack under
    ``language_model`` and activates nothing for a name it cannot place, so
    the adapter loads without an error and serves the base model
    (whilehq/whileai-sdk#588). Rewrite the header in place; tensor bytes are
    untouched. Plain causal LMs already carry the right names and are left
    alone."""
    import json
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    old_prefix = "base_model.model.model.layers."
    if not any(k.startswith(old_prefix) for k in header):
        return
    model_cfg = {}
    try:
        from transformers import AutoConfig

        model_cfg = AutoConfig.from_pretrained(
            json.load(open(path.replace("adapter_model.safetensors", "adapter_config.json")))[
                "base_model_name_or_path"
            ]
        ).to_dict()
    except Exception as exc:
        print(
            f"adapter rename: could not read the base config ({exc}); renaming on key layout alone"
        )
    if model_cfg and "text_config" not in model_cfg:
        return  # a plain causal LM: names already match
    new_prefix = "base_model.model.model.language_model.layers."
    renamed = {
        (new_prefix + k[len(old_prefix) :] if k.startswith(old_prefix) else k): v
        for k, v in header.items()
    }
    hb = json.dumps(renamed, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(data)
    print(
        f"adapter rename: {sum(k.startswith(old_prefix) for k in header)} tensors moved under language_model for vLLM"
    )


@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 60 * 60,
    scaledown_window=10 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/vol": runs_volume},
    # Everything the server needs rides in the secret: the container re-imports
    # this module without serve_config.json.
    secrets=[
        modal.Secret.from_dict(
            {
                "VLLM_API_KEY": KEY,
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "T2S_SERVE_MODEL": MODEL,
                "T2S_SERVE_ADAPTER": ADAPTER,
                "T2S_SERVE_MAX_LEN": str(MAX_LEN),
                "T2S_SERVE_TOOL_PARSER": TOOL_PARSER,
            }
        )
    ],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    model = os.environ["T2S_SERVE_MODEL"]
    key = os.environ["VLLM_API_KEY"]
    adapter = os.environ.get("T2S_SERVE_ADAPTER") or ""
    max_len = os.environ.get("T2S_SERVE_MAX_LEN") or "16384"
    tool_parser = os.environ.get("T2S_SERVE_TOOL_PARSER") or "hermes"
    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--api-key",
        key,
        "--max-model-len",
        max_len,
        "--gpu-memory-utilization",
        "0.90",
        "--dtype",
        "bfloat16",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_parser,
    ]
    if adapter.startswith("volume:"):
        run_id = adapter.split(":", 1)[1]
        # "volume:<run_id>" is <run_id>/adapter; "volume:<run_id>/checkpoints/checkpoint-25"
        # is that directory as written by save_every.
        src = f"/vol/{run_id}" if "/" in run_id else f"/vol/{run_id}/adapter"
        # vLLM mmaps the adapter tensors; on the volume's FUSE mount that
        # surfaced as "No adapter found for /vol/<run>/adapter" although the
        # files were there. Copy the adapter (a few hundred MB) to local disk.
        local = "/root/adapters/" + run_id.replace("/", "_")
        shutil.copytree(src, local, dirs_exist_ok=True)
        _rename_for_vllm(local + "/adapter_model.safetensors")
        cmd += [
            "--enable-lora",
            "--lora-modules",
            f"{model}-adapter={local}",
            "--max-lora-rank",
            "64",
        ]
    subprocess.Popen(cmd)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="write serve_config.json, then: modal deploy serve_modal.py"
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--key", default=KEY)
    ap.add_argument(
        "--adapter", default=ADAPTER, help="volume:<run_id>: serve a LoRA from the training volume"
    )
    ap.add_argument("--gpu", default=GPU)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--tool-parser", default=TOOL_PARSER)
    ap.add_argument("--vllm", default=VLLM_VERSION, help="vLLM version for the image")
    ap.add_argument(
        "--runs-volume", default=RUNS_VOLUME, help="Modal volume holding <run_id>/adapter"
    )
    ap.add_argument("--hf-cache", default=HF_CACHE, help="Modal volume for the Hugging Face cache")
    ap.add_argument("--url", action="store_true", help="print the deployed endpoint URL and exit")
    a = ap.parse_args()
    if a.url:
        print(modal.Function.from_name(app.name, "serve").get_web_url())
        raise SystemExit(0)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "model": a.model,
                "key": a.key,
                "adapter": a.adapter,
                "gpu": a.gpu,
                "max_len": a.max_len,
                "tool_parser": a.tool_parser,
                "vllm": a.vllm,
                "runs_volume": a.runs_volume,
                "hf_cache": a.hf_cache,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {CONFIG_PATH.name} for {a.model}; now: PYTHONUTF8=1 modal deploy serve_modal.py")
