"""Phase 0 on Modal: score FineEnvs' released LFM2.5-2.6B checkpoints under 10 harnesses.

One H100 container per checkpoint runs vLLM, FineEnvs' Harbor env server (capture proxy
published with `modal.forward`) and `evaluate.py`. Every rollout gets its own Modal sandbox.

    modal run --env main eval_modal.py::smoke                # 1 task x 10 harnesses, base model
    modal run --detach eval_modal.py::phase0 --model mh-rl   # 250 x 10 x 3
    modal run eval_modal.py::fetch                           # copy cells to ./out
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
OPENENV_SHA = "7ee88d590d36ae1ac3daf2cc11bc551bca7a804f"  # FineEnvs' VALIDATION.md pin
FINEENVS_SHA = "26ab9c60915005b6f536968ac0928482bb18c0f8"
TRANSFORMERS_SHA = "d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a"
TRL_SHA = "52fb144e7ece97e26aabd3c025740022838e46b5"
BASE = "LiquidAI/LFM2.5-2.6B"
BASE_REV = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"

MODELS = {
    "base": (BASE, BASE_REV),
    "mh-rl": ("FineEnvs/LFM2.5-2.6B-multiharness-RL", "main"),
    "oc-rl": ("FineEnvs/LFM2.5-2.6B-opencode-RL", "main"),
    "mh-sft": ("FineEnvs/LFM2.5-2.6B-multiharness-SFT", "main"),
    "oc-sft": ("FineEnvs/LFM2.5-2.6B-opencode-SFT", "main"),
}
TRAINED = ["opencode", "claude-code", "codex", "mini-swe-agent"]
UNSEEN = ["pi", "gemini-cli", "qwen-coder", "vibe", "openhands-sdk", "terminus-2"]
# FineEnvs' pins for their four; the six unseen are pinned after the smoke records what installs.
AGENT_VERSIONS = {
    "opencode": "1.18.31",
    "claude-code": "2.1.270",
    "codex": "0.154.0",
    "mini-swe-agent": "2.4.6",
}

app = modal.App("multi-harness-transfer")
vol = modal.Volume.from_name("multi-harness-transfer", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "vllm==0.25.1",
        f"transformers @ git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_SHA}",
        f"trl @ git+https://github.com/huggingface/trl.git@{TRL_SHA}",
        "harbor[modal]>=0.22.0",
        "fastmcp>=2",
        "uvicorn[standard]",
        "gradio",
        "huggingface-hub",
        "math-verify",
        "tomli-w",
        "httpx",
        "websockets>=15,<16",
        "numpy>=2.2,<2.4",
    )
    .run_commands(
        f"git clone https://github.com/huggingface/OpenEnv.git /opt/OpenEnv && git -C /opt/OpenEnv checkout {OPENENV_SHA}",
        "pip install -e /opt/OpenEnv",
        f"git clone https://github.com/adithya-s-k/FineEnvs.git /opt/FineEnvs && git -C /opt/FineEnvs checkout {FINEENVS_SHA}",
        "pip install --no-deps -e /opt/FineEnvs/05-multi-harness-rl/envs/harbor",
        "python -c \"import site,pathlib; pathlib.Path(site.getsitepackages()[0],'openenv_examples.pth').write_text('/opt/OpenEnv/envs\\n')\"",
    )
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_file(HERE / "evaluate.py", "/app/evaluate.py")
    .add_local_file(HERE / "serve_env.py", "/app/serve_env.py")
)

ENV_DIR = "/opt/FineEnvs/05-multi-harness-rl/envs/harbor"
DATA = "/vol/prepared"


def _wait(url, name, proc, timeout=1800):
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited with {proc.returncode}")
        try:
            if httpx.get(url, timeout=5).status_code < 500:
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(name)


def _chat_template(path):
    # FineEnvs' patched non-thinking LFM template (train/whitebox.py tokenizer_for).
    sys.path.insert(0, "/opt/FineEnvs/05-multi-harness-rl")
    from train.whitebox import tokenizer_for

    Path(path).write_text(tokenizer_for(BASE).chat_template)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=24 * 3600,
    volumes={"/vol": vol, "/hf": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eval(model_key: str, harnesses: list[str], tasks: int, samples: int, concurrency: int, tag: str):
    from huggingface_hub import HfApi

    repo, rev = MODELS[model_key]
    sha = HfApi().model_info(repo, revision=rev).sha
    checkpoint = f"{repo}@{sha}"
    print("scoring", checkpoint, flush=True)

    if not Path(DATA, "ready.json").exists():
        subprocess.run([sys.executable, "prepare.py", "--output", DATA], cwd=ENV_DIR, check=True)
        vol.commit()

    _chat_template("/tmp/chat_template.jinja")
    logs = Path("/tmp/logs")
    logs.mkdir(exist_ok=True)
    vllm = subprocess.Popen(
        [
            "vllm", "serve", repo, "--revision", sha,
            "--host", "127.0.0.1", "--port", "8000",
            "--served-model-name", BASE,
            "--dtype", "bfloat16", "--max-model-len", "131072",
            "--gpu-memory-utilization", "0.85",
            "--enable-auto-tool-choice", "--tool-call-parser", "lfm2",
            "--chat-template", "/tmp/chat_template.jinja",
            "--default-chat-template-kwargs", '{"enable_thinking":false,"preserve_thinking":true}',
            "--generation-config", "vllm",
            "--logprobs-mode", "processed_logprobs",
            "--return-tokens-as-token-ids",
            "--no-enable-prefix-caching",
        ],
        stdout=open(logs / "vllm.log", "w"), stderr=subprocess.STDOUT,
    )
    env = {
        **os.environ,
        "SMOLDATA_DATA": DATA,
        "OPENENV_LLM_URL": "http://127.0.0.1:8000",
        "OPENENV_MODEL": BASE,
        "OPENENV_MAX_OUTPUT_TOKENS": "4096",
        "MAX_CONCURRENT_ENVS": str(max(40, concurrency + 8)),
        "OPENENV_HARBOR_REWARD_KEY": "correctness,reward",
        "OPENENV_HARBOR_AGENT_VERSIONS": json.dumps(AGENT_VERSIONS),
    }
    server = subprocess.Popen(
        [sys.executable, "/app/serve_env.py"], cwd=ENV_DIR, env=env,
        stdout=open(logs / "env.log", "w"), stderr=subprocess.STDOUT,
    )
    out = f"/vol/{tag}/{model_key}"
    try:
        _wait("http://127.0.0.1:8000/health", "vllm", vllm)
        _wait("http://127.0.0.1:8200/smoldataenv/splits", "env server", server)
        print("services up", flush=True)
        rc = subprocess.run(
            [
                sys.executable, "/app/evaluate.py",
                "--checkpoint", checkpoint, "--model", BASE,
                "--harnesses", ",".join(harnesses),
                "--samples", str(samples), "--tasks", str(tasks),
                "--sandbox", "modal", "--data", DATA,
                "--output", out, "--concurrency", str(concurrency),
            ],
            cwd=ENV_DIR, env=env,
        ).returncode
        print("evaluate exit", rc, flush=True)
    finally:
        for name in ("vllm.log", "env.log"):
            Path(out, "logs").mkdir(parents=True, exist_ok=True)
            Path(out, "logs", name).write_bytes((logs / name).read_bytes())
        # Record which agent versions Harbor actually installed (for pinning the unseen six).
        found = subprocess.run(
            "grep -rhoE '\"(agent_)?version\"[^,}]*' /tmp/openenv-harbor-trials/*/agent 2>/dev/null | sort | uniq -c | head -50",
            shell=True, capture_output=True, text=True,
        ).stdout
        Path(out, "logs", "agent_versions.txt").write_text(found)
        subprocess.run("cp -r /tmp/openenv-harbor-trials " + out + "/trials 2>/dev/null", shell=True)
        vllm.terminate()
        server.terminate()
        vol.commit()


@app.local_entrypoint()
def smoke():
    run_eval.remote("base", TRAINED + UNSEEN, tasks=1, samples=1, concurrency=10, tag="smoke")


@app.local_entrypoint()
def phase0(model: str = "base", tasks: int = 250, samples: int = 3, concurrency: int = 48):
    run_eval.remote(model, TRAINED + UNSEEN, tasks=tasks, samples=samples, concurrency=concurrency, tag="phase0")
