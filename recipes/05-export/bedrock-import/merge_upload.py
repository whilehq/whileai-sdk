"""Merge a LoRA adapter into its base on Modal and push the merged weights to S3.

Two steps, each a Modal function, so the weights never touch your laptop:

    modal run merge_upload.py::merge_entry --base nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \
        --adapter while-ai/text-to-sql-shop-nemotron-8b-r1 --name nemotron-8b-t2s-r1
    python presign.py nemotron-8b-t2s-r1 files.json --bucket <bucket> > urls.json
    modal run merge_upload.py::upload_entry --name nemotron-8b-t2s-r1 --urls urls.json

The merged model is saved the way Bedrock Custom Model Import wants it: the
Hugging Face layout, safetensors shards under 5 GB, config.json, and the
tokenizer with its chat template inline in tokenizer_config.json, written by
transformers 4.51.3 (the version AWS pins). Upload goes to presigned PUT URLs
you sign locally, so Modal never holds AWS credentials. A gated base needs
HF_TOKEN in your environment (or `hf auth login`); it is passed as a secret.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP = "while-bedrock-import"
VOLUME = "while-bedrock-imports"
OUT = "/out"
HF_CACHE = "/hf"
# MAX_SHARD = 4GB: Bedrock imports each safetensors file as one S3 object and
# a single presigned PUT carries at most 5 GB (S3 limit), so shards stay under it.
MAX_SHARD = "4GB"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "transformers==4.51.3",
        "peft==0.15.2",
        "accelerate==1.6.0",
        "safetensors>=0.5",
        "huggingface_hub==0.30.2",
        "hf_transfer",
        "requests",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE})
)

app = modal.App(APP)
out_vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name(f"{VOLUME}-hf-cache", create_if_missing=True)


def _hf_secret() -> list[modal.Secret]:
    token = os.environ.get("HF_TOKEN") or ""
    if not token:
        path = Path.home() / ".cache" / "huggingface" / "token"
        if path.exists():
            token = path.read_text().strip()
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=3 * 3600,
    volumes={OUT: out_vol, HF_CACHE: hf_vol},
    secrets=_hf_secret(),
)
def merge(base: str, adapter: str, name: str) -> list[dict]:
    """Load the base in bf16 on CPU, merge the adapter, save shards + tokenizer."""
    import shutil
    import time

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dest = Path(OUT) / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    t0 = time.time()
    print(f"loading base {base} in bf16 on cpu", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    print(f"  base loaded in {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    print(f"  adapter merged in {time.time() - t1:.0f}s", flush=True)

    t2 = time.time()
    model.save_pretrained(str(dest), safe_serialization=True, max_shard_size=MAX_SHARD)
    # The base's tokenizer carries the chat template inline; Bedrock reads it
    # from tokenizer_config.json and applies no default of its own.
    tok = AutoTokenizer.from_pretrained(base)
    tok.save_pretrained(str(dest))
    cfg_path = dest / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    if not cfg.get("chat_template") and getattr(tok, "chat_template", None):
        cfg["chat_template"] = tok.chat_template
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print("  chat_template written into tokenizer_config.json", flush=True)
    print(f"  saved in {time.time() - t2:.0f}s", flush=True)
    out_vol.commit()

    files = []
    for p in sorted(dest.iterdir()):
        files.append({"name": p.name, "bytes": p.stat().st_size})
        print(f"  {p.stat().st_size / 1e9:8.3f} GB  {p.name}", flush=True)
    config = json.loads((dest / "config.json").read_text())
    keys = ("architectures", "model_type", "torch_dtype", "transformers_version")
    print("config:", {k: config.get(k) for k in keys}, flush=True)
    return files


@app.function(image=image, cpu=4, memory=8192, timeout=2 * 3600, volumes={OUT: out_vol})
def upload(name: str, urls: dict[str, str]) -> dict[str, int]:
    """PUT every file of the merged model to its presigned URL."""
    import time

    import requests

    out_vol.reload()
    dest = Path(OUT) / name
    statuses: dict[str, int] = {}
    for fname, url in urls.items():
        path = dest / fname
        size = path.stat().st_size
        t0 = time.time()
        with path.open("rb") as fh:
            r = requests.put(url, data=fh, headers={"Content-Length": str(size)}, timeout=3600)
        statuses[fname] = r.status_code
        rate = size / 1e6 / max(time.time() - t0, 1e-6)
        print(f"  {r.status_code}  {size / 1e9:6.3f} GB  {rate:6.0f} MB/s  {fname}", flush=True)
        if r.status_code >= 300:
            print("   ", r.text[:300], flush=True)
    return statuses


@app.local_entrypoint()
def merge_entry(base: str, adapter: str, name: str, files_out: str = "files.json") -> None:
    files = merge.remote(base, adapter, name)
    Path(files_out).write_text(json.dumps({"name": name, "files": files}, indent=2))
    total = sum(f["bytes"] for f in files) / 1e9
    print(f"wrote {files_out}: {len(files)} files, {total:.2f} GB")


@app.local_entrypoint()
def upload_entry(name: str, urls: str = "urls.json") -> None:
    mapping = json.loads(Path(urls).read_text())
    statuses = upload.remote(name, mapping)
    bad = {k: v for k, v in statuses.items() if v >= 300}
    print("uploaded", len(statuses), "files;", "all 200" if not bad else f"FAILED {bad}")
