"""prime-rl on Modal, your keys: run the TOML ``wai.prime_rl_config`` wrote.

    modal run modal_prime_rl.py --config opsd.toml --run-name opsd-1 --validate   # CPU dry run
    modal run modal_prime_rl.py --config opsd.toml --run-name opsd-1              # 2 GPUs, wait
    modal run modal_prime_rl.py --config opd.toml --run-name opd-1 \\
        --teacher PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL                       # frozen teacher on GPU 0

Nothing here is a trainer. The container is Prime Intellect's published
prime-rl image (``PRIME_RL_IMAGE``, a GHCR tag pinned to a commit); the
function writes the config to a volume, starts the frozen teacher server when
one is named (``uv run inference`` on the first GPU, shared with the policy's
own engine at 0.4 of its memory each, the layout prime-rl's own OPD configs
use), runs ``rl`` and returns the metrics file. The run directory and the
Hugging Face cache live on two Modal volumes so a second run reuses the
weights and a killed container keeps its logs.

Layout inside the container:
  /app                       prime-rl at the image's commit, venv at /app/.venv
  /runs/configs/<run>.toml   the config as written (outside the run dir: the launcher refuses a
                             run dir with files in it)
  /runs/<run>                the run: logs/, checkpoints/, monitors/file/metrics.jsonl
  /root/.cache/huggingface   weights (volume PRIME_RL_HF_VOLUME)
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal

# ghcr.io/primeintellect-ai/prime-rl publishes one tag per commit. v0.8.1.dev63 is
# commit e6d2b3f25 (2026-08-21), the newest published when this was written; it
# carries the opd / opsd algorithms (first commit 2026-06-27), max_off_policy_steps
# and the ipo / icepop losses that wai.prime_rl_config writes.
IMAGE = os.environ.get("PRIME_RL_IMAGE", "ghcr.io/primeintellect-ai/prime-rl:v0.8.1.dev63")
GPU = os.environ.get("PRIME_RL_GPU", "H100:2")
RUNS_VOLUME = os.environ.get("PRIME_RL_RUNS_VOLUME", "wai-prime-rl-runs")
HF_VOLUME = os.environ.get("PRIME_RL_HF_VOLUME", "wai-prime-rl-hf-cache")
OUTPUT_DIR = "/runs"
VENV = "/app/.venv/bin"
MINUTES = 60
TEACHER_PORT = 8001
TEACHER_GPU_SHARE = 0.4  # the share prime-rl's own OPD configs give a teacher that shares GPU 0
TEACHER_READY_S = 15 * MINUTES

# The image ships its own /usr/local/bin/python (a symlink to the system 3.12 the
# /app/.venv is built on). Modal injects a separate interpreter for its client at
# the same path and fails on the existing symlink, so a two-line Dockerfile clears
# it first; Modal's python step runs after the Dockerfile. The venv keeps its base.
image = modal.Image.from_dockerfile(
    Path(__file__).with_name("prime-rl.Dockerfile"),
    build_args={"PRIME_RL_IMAGE": IMAGE},
    add_python="3.12",
).env(
    {
        "HF_HOME": "/root/.cache/huggingface",
        "PYTHONUNBUFFERED": "1",
        "PRL_OUTPUT_DIR": OUTPUT_DIR,
    }
)
runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)

app = modal.App(os.environ.get("PRIME_RL_APP", "wai-prime-rl"))
COMMON: dict[str, Any] = dict(
    image=image,
    volumes={OUTPUT_DIR: runs_vol, "/root/.cache/huggingface": hf_vol},
)


def _write_config(config_toml: str, run_name: str) -> Path:
    cfg = Path(OUTPUT_DIR) / "configs" / f"{run_name}.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(config_toml, encoding="utf-8")
    return cfg


def _rl_cmd(cfg: Path, run_name: str, extra: list[str]) -> list[str]:
    return [f"{VENV}/rl", "@", str(cfg), "--output-dir", OUTPUT_DIR, "--run.name", run_name, *extra]


def _start_teacher(model: str, run_name: str) -> subprocess.Popen:
    """The frozen reference server, on GPU 0 next to the policy's engine."""
    import urllib.request

    log = Path(OUTPUT_DIR) / "configs" / f"{run_name}.teacher.log"
    cmd = [
        f"{VENV}/inference",
        "--vllm.model",
        model,
        "--server.port",
        str(TEACHER_PORT),
        "--vllm.gpu-memory-utilization",
        str(TEACHER_GPU_SHARE),
        "--vllm.enforce-eager",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    handle = log.open("wb")
    proc = subprocess.Popen(cmd, cwd="/app", env=env, stdout=handle, stderr=handle)
    deadline = time.time() + TEACHER_READY_S
    url = f"http://localhost:{TEACHER_PORT}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"teacher server exited {proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    print(f"[teacher] {model} ready on :{TEACHER_PORT}", flush=True)
                    return proc
        except Exception:
            time.sleep(5)
    proc.kill()
    raise TimeoutError(f"teacher server not ready in {TEACHER_READY_S}s; see {log}")


def _follow(run_dir: Path, stop: threading.Event) -> None:
    """Echo each component's log to the container stdout so `modal app logs`
    shows progress, and commit the volume so a killed container keeps its files."""
    offsets: dict[Path, int] = {}
    last_commit = time.time()
    while not stop.wait(20):
        for name in ("orchestrator", "trainer", "inference"):
            log = run_dir / "logs" / "latest" / f"{name}.log"
            if not log.exists():
                continue
            try:
                with log.open("rb") as f:
                    f.seek(offsets.get(log, 0))
                    chunk = f.read()
                    offsets[log] = f.tell()
            except OSError:
                continue
            for line in chunk.decode("utf-8", "replace").splitlines()[-30:]:
                print(f"[{name}] {line}", flush=True)
        if time.time() - last_commit > 5 * MINUTES:
            try:
                runs_vol.commit()
            except Exception as e:  # the follower must outlive a failed commit
                print(f"[volume] commit failed: {e}", flush=True)
            last_commit = time.time()


def _metrics(run_dir: Path) -> list[dict]:
    """Every line of every metrics.jsonl under the run (orchestrator, trainer, eval)."""
    rows: list[dict] = []
    for path in sorted(run_dir.rglob("metrics.jsonl*")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_file"] = str(path.relative_to(run_dir))
            rows.append(row)
    return rows


def _tail(run_dir: Path, name: str, n: int = 40) -> list[str]:
    log = run_dir / "logs" / "latest" / f"{name}.log"
    if not log.exists():
        return []
    return log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


@app.function(**COMMON, cpu=4, memory=16 * 1024, timeout=20 * MINUTES)
def validate(config_toml: str, run_name: str) -> dict:
    """``rl --dry-run``: parse and resolve the config on CPU, no GPU spent."""
    cfg = _write_config(config_toml, run_name)
    proc = subprocess.run(
        _rl_cmd(cfg, f"{run_name}-dry", ["--dry-run"]),
        cwd="/app",
        capture_output=True,
        text=True,
    )
    out = (proc.stdout + proc.stderr).splitlines()
    return {"returncode": proc.returncode, "tail": out[-40:]}


@app.function(**COMMON, gpu=GPU, timeout=4 * 60 * MINUTES)
def train(
    config_toml: str, run_name: str, teacher: str | None = None, extra: list[str] | None = None
) -> dict:
    """Run one config to completion and return its metrics and log tails."""
    cfg = _write_config(config_toml, run_name)
    run_dir = Path(OUTPUT_DIR) / run_name
    teacher_proc = _start_teacher(teacher, run_name) if teacher else None
    stop = threading.Event()
    follower = threading.Thread(target=_follow, args=(run_dir, stop), daemon=True)
    follower.start()
    started = time.time()
    try:
        proc = subprocess.run(_rl_cmd(cfg, run_name, list(extra or [])), cwd="/app")
        rc = proc.returncode
    finally:
        stop.set()
        follower.join(timeout=30)
        if teacher_proc is not None:
            teacher_proc.kill()
        try:
            runs_vol.commit()
        except Exception as e:
            print(f"[volume] final commit failed: {e}", flush=True)
    return {
        "run": run_name,
        "returncode": rc,
        "seconds": round(time.time() - started, 1),
        "image": IMAGE,
        "gpu": GPU,
        "metrics": _metrics(run_dir),
        "orchestrator_tail": _tail(run_dir, "orchestrator"),
        "trainer_tail": _tail(run_dir, "trainer"),
    }


@app.local_entrypoint()
def main(
    config: str,
    run_name: str,
    teacher: str = "",
    validate_only: bool = False,
    extra: str = "",
) -> None:
    text = Path(config).read_text(encoding="utf-8")
    args = extra.split() if extra else []
    if validate_only:
        result = validate.remote(text, run_name)
        print("\n".join(result["tail"]))
        raise SystemExit(result["returncode"])
    result = train.remote(text, run_name, teacher or None, args)
    out = Path(f"{run_name}.result.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"exit {result['returncode']} in {result['seconds']}s; {len(result['metrics'])} metric rows -> {out}"
    )
