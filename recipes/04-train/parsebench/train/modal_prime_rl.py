"""prime-rl on Modal: RL-train a LoRA on Qwen/Qwen3.8-27B (VLM) for chart -> HTML table.

Same launcher shape as recipes/04-train/prime-rl/modal_prime_rl.py (text-only RL on prime-rl).
Differences: image tasks (train/envs/docparse_charts), reward from waiparse.rewards (parse-bench),
volumes docparse-data / docparse-runs in your own Modal workspace.

Layout inside the container:
  /app                       prime-rl checkout at PRIME_RL_COMMIT, venv at /app/.venv
  /envs                      train/envs (docparse_charts, installed --no-deps)
  /opt/waiparse/waiparse     waiparse/ (reward + prompt, loaded by file path)
  /data                      volume docparse-data, read-only (/data/train/charts/...)
  /runs/prime-rl/<run>       volume docparse-runs: run dirs, adapters/step_N
  /root/.cache/huggingface   volume docparse-hf-cache (shared with serve/serve_vlm.py)

Usage (from recipes/04-train/parsebench; on Windows, Git Bash with MSYS_NO_PATHCONV=1):
  export PYTHONUTF8=1
  modal run train/modal_prime_rl.py --config train/configs/charts-smoke.toml --run-name charts-smoke --validate-only
  modal deploy train/modal_prime_rl.py                   # once, and after any edit here or in train/envs
  modal run train/modal_prime_rl.py --config train/configs/charts-smoke.toml --run-name charts-smoke --spawn
  modal run ... --spawn --extra "--resume.step 50"       # resume the same run dir
  modal app logs docparse-prime-rl
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import modal

PRIME_RL_COMMIT = "9ef21a64f18615c8941b0a4a42d056c0e2846a28"  # prime-rl main @ 2026-09-12
PARSE_BENCH_COMMIT = (
    "a53c1694f2fd22701ed9f03db23fe82ae053d442"  # run-llama/ParseBench main @ 2026-09-25
)
HERE = Path(__file__).resolve().parent
ENVS_DIR = HERE / "envs"
WAIPARSE_DIR = HERE.parent / "waiparse"
OUTPUT_DIR = "/runs/prime-rl"
CHARTS_DIR = "/data/train/charts"
MINUTES = 60

# Why each extra is required (kernels: trainer imports prime_kernels;
# disagg: inference launcher execs vllm-router; flash-attn-3: attn="auto" picks FA3 on Hopper).
# --frozen installs the lock as written without re-resolving it.
UV_SYNC = "cd /app && uv sync --extra gpu --extra flash-attn --extra kernels --extra disagg --extra flash-attn-3 --frozen --no-dev"
PY = "/app/.venv/bin/python"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "build-essential",
        "ninja-build",
        "pkg-config",
        "clang",
        "libnuma-dev",
        "libibverbs-dev",
        "librdmacm-dev",
    )
    .env(
        {
            "UV_PROJECT_ENVIRONMENT": "/app/.venv",
            "UV_CACHE_DIR": "/root/.cache/uv",
            "UV_LINK_MODE": "copy",
            "UV_COMPILE_BYTECODE": "1",
            "UV_PYTHON_PREFERENCE": "only-system",
            "CUDA_HOME": "/usr/local/cuda",
            "DEBIAN_FRONTEND": "noninteractive",
        }
    )
    .run_commands(
        # uv 0.12.12+ rejects the lock's direct-URL wheels; pinned.
        "curl -fsSL https://astral.sh/uv/0.12.9/install.sh -o /tmp/uv.sh && env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh /tmp/uv.sh && uv --version",
        # .gitmodules pins SSH URLs; the build container has no GitHub key.
        'git config --global url."https://github.com/".insteadOf "git@github.com:"',
        f"git clone https://github.com/PrimeIntellect-ai/prime-rl.git /app && cd /app && git checkout {PRIME_RL_COMMIT} && git submodule update --init --recursive",
        UV_SYNC,
    )
    .run_commands(
        # parse-bench with its deps, but constrained to every version already in the locked
        # venv, so it can add packages but never move a prime-rl / vLLM / torch pin.
        f"cd /app && uv pip freeze --python {PY} | grep '==' > /tmp/prime-rl-pins.txt && "
        f"uv pip install --python {PY} -c /tmp/prime-rl-pins.txt "
        f"'parse-bench @ git+https://github.com/run-llama/ParseBench.git@{PARSE_BENCH_COMMIT}'",
        f"{PY} -c 'from parse_bench.evaluation.metrics.parse.rules_chart import ChartDataPointRule; print(\"parse-bench ok\")'",
    )
    .env({"WAIPARSE_DIR": "/opt/waiparse/waiparse", "DOCPARSE_CHARTS_DIR": CHARTS_DIR})
    .add_local_dir(ENVS_DIR, "/envs", copy=True, ignore=["**/__pycache__/**"])
    .run_commands(
        # --no-deps: the env's `verifiers>=0.3.1` must not pull PyPI verifiers over the
        # workspace-editable one prime-rl is built on; parse-bench is installed above.
        f"cd /app && uv pip install --python {PY} --no-deps /envs/docparse_charts",
        f"cd /app && {PY} -c 'import docparse_charts, docparse_charts.losses, prime_rl, vllm, prime_kernels, renderers; print(\"env+prime-rl ok\", vllm.__version__)'",
    )
    .add_local_dir(WAIPARSE_DIR, "/opt/waiparse/waiparse", ignore=["**/__pycache__/**"])
)

runs_vol = modal.Volume.from_name("docparse-runs", create_if_missing=True)
data_vol = modal.Volume.from_name("docparse-data")
hf_cache_vol = modal.Volume.from_name("docparse-hf-cache", create_if_missing=True)

# DOCPARSE_RL_APP lets a second deploy (new image / env) run beside an in-flight run.
app = modal.App(os.environ.get("DOCPARSE_RL_APP", "docparse-prime-rl"))

COMMON = dict(
    image=image,
    volumes={
        "/runs": runs_vol,
        "/data": data_vol.read_only(),
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _run_dir(run_name: str) -> Path:
    return Path(OUTPUT_DIR) / run_name


def _write_config(config_toml: str, run_name: str) -> Path:
    # Kept OUTSIDE the run dir: the launcher refuses a run dir that already has files in it.
    cfg = Path(OUTPUT_DIR) / "configs" / f"{run_name}.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(config_toml, encoding="utf-8")
    return cfg


def _rl_cmd(cfg: Path, run_name: str, extra: list[str]) -> list[str]:
    # --no-sync --frozen: plain `uv run` re-plans the sync and dies on the aarch64 flash-attn entry.
    return [
        "uv",
        "run",
        "--no-sync",
        "--frozen",
        "rl",
        "@",
        str(cfg),
        "--output-dir",
        OUTPUT_DIR,
        "--run.name",
        run_name,
        *extra,
    ]


def _env() -> dict:
    return {
        **os.environ,
        "PRL_OUTPUT_DIR": OUTPUT_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
    }


def _follow_logs(run_dir: Path, stop: threading.Event) -> None:
    """Echo component logs to stdout (so `modal app logs` shows progress), preserve adapters,
    and commit the volume every 10 min. Every step is wrapped: this thread is the only
    periodic commit, and if it dies the run trains on while nothing reaches the volume."""
    offsets: dict[Path, int] = {}
    last_commit = time.time()
    while not stop.wait(30):
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
            if chunk:
                for line in chunk.decode("utf-8", "replace").splitlines()[-40:]:
                    print(f"[{name}] {line}", flush=True)
        try:
            _preserve_adapters(run_dir)
        except Exception as e:
            print(f"[adapters] preserve failed: {e}", flush=True)
        if time.time() - last_commit > 10 * MINUTES:
            try:
                runs_vol.commit()
            except Exception as e:
                print(f"[volume] commit failed: {e}", flush=True)
            last_commit = time.time()


def _preserve_adapters(run_dir: Path, every: int = 25) -> None:
    """prime-rl keeps only the two newest `broadcasts/step_N`. Copy every Nth finished one to
    `adapters/step_N` (adapter_model.safetensors + adapter_config.json), which is exactly
    what serve/serve_vlm.py mounts: DOCPARSE_ADAPTER=prime-rl/<run>/adapters/step_N."""
    import shutil

    bdir = run_dir / "broadcasts"
    if not bdir.is_dir():
        return
    for d in bdir.iterdir():
        if not d.name.startswith("step_"):
            continue
        step = int(d.name.split("_")[1])
        dst = run_dir / "adapters" / d.name
        if step % every or dst.exists() or not (d / ".finished").exists():
            continue
        if not (d / "adapter_model.safetensors").exists():
            continue
        shutil.copytree(d, dst, ignore=shutil.ignore_patterns(".*"))
        print(
            f"[adapters] preserved {d.name} -> DOCPARSE_ADAPTER=prime-rl/{run_dir.name}/adapters/{d.name}",
            flush=True,
        )


@app.function(**COMMON, cpu=8, memory=32 * 1024, timeout=30 * MINUTES)
def validate(
    config_toml: str, run_name: str, extra_args: list[str] | None = None, selfcheck: bool = True
) -> int:
    """`rl --dry-run` (config parse + resolve on CPU, no GPUs), then the env self-check:
    CISPO loss gradient, reward on gold tables, and (once the data exists) the qwen3.8
    render of real chart tasks with their image-token counts. `extra_args` are forwarded
    so a resume dry-run passes the same run-dir check the real launch does."""
    cfg = _write_config(config_toml, run_name)
    proc = subprocess.run(
        _rl_cmd(cfg, run_name, [*(extra_args or []), "--dry-run"]),
        cwd="/app",
        env=_env(),
        capture_output=True,
        text=True,
    )
    print(proc.stdout[-6000:])
    print(proc.stderr[-6000:])
    rc = proc.returncode
    if selfcheck:
        sc = subprocess.run(
            [PY, "-m", "docparse_charts.selfcheck"],
            cwd="/app",
            env=_env(),
            capture_output=True,
            text=True,
        )
        print(sc.stdout[-6000:])
        if sc.returncode:
            print(sc.stderr[-6000:])
        rc = rc or sc.returncode
    runs_vol.commit()
    return rc


def _kill_stale_processes() -> None:
    """A warm container can still hold a previous attempt's launcher children; kill them."""
    for pat in [
        "prime_rl",
        "vllm",
        "vllm-router",
        "torchrun",
        "orchestrator",
        "inference",
        "trainer",
    ]:
        r = subprocess.run(["pkill", "-9", "-f", pat], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[stale] killed processes matching {pat!r}", flush=True)
    time.sleep(3)


def _train(config_toml: str, run_name: str, extra_args: list[str] | None) -> int:
    _kill_stale_processes()
    run_dir = _run_dir(run_name)
    cfg = _write_config(config_toml, run_name)
    print(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
            capture_output=True,
            text=True,
        ).stdout
    )
    n = (
        sum(1 for _ in open(f"{CHARTS_DIR}/train.jsonl", encoding="utf-8"))
        if Path(CHARTS_DIR, "train.jsonl").exists()
        else 0
    )
    print(f"[data] {CHARTS_DIR}/train.jsonl rows: {n}", flush=True)

    stop = threading.Event()
    follower = threading.Thread(target=_follow_logs, args=(run_dir, stop), daemon=True)
    follower.start()
    proc = None
    rc = 1
    try:
        proc = subprocess.Popen(
            _rl_cmd(cfg, run_name, extra_args or []),
            cwd="/app",
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # own process group, so the whole tree can be killed on exit
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[rl] {line}", end="", flush=True)
        rc = proc.wait()
    finally:
        stop.set()
        follower.join(timeout=60)
        if proc is not None:
            try:
                import signal

                os.killpg(proc.pid, signal.SIGKILL)  # no launcher children may outlive this input
            except Exception:
                pass
        try:
            _preserve_adapters(run_dir, every=1)  # the final broadcast, whatever its step
        except Exception as e:
            print(f"[adapters] final preserve failed: {e}", flush=True)
        runs_vol.commit()
    for name in ("orchestrator", "trainer", "inference"):
        log = run_dir / "logs" / "latest" / f"{name}.log"
        if log.exists():
            print(f"===== tail {name}.log =====")
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:])
    print(f"rl exited with {rc}; run dir {run_dir}")
    return rc


# One function per GPU count (Modal fixes gpu= per function); `[deployment] gpus_per_node`
# in the TOML must match. 24 h is Modal's cap: plan a `--resume.step` before it.
@app.function(**COMMON, gpu="H200:4", cpu=32, memory=256 * 1024, timeout=24 * 60 * MINUTES)
def train(config_toml: str, run_name: str, extra_args: list[str] | None = None) -> int:
    return _train(config_toml, run_name, extra_args)


@app.function(**COMMON, gpu="H200:8", cpu=64, memory=512 * 1024, timeout=24 * 60 * MINUTES)
def train8(config_toml: str, run_name: str, extra_args: list[str] | None = None) -> int:
    return _train(config_toml, run_name, extra_args)


@app.local_entrypoint()
def main(
    config: str,
    run_name: str,
    validate_only: bool = False,
    spawn: bool = False,
    extra: str = "",
    gpus: int = 4,
    selfcheck: bool = True,
):
    """`--spawn` fires train() on the DEPLOYED app (`modal deploy` first) and returns, so the
    run outlives this client (`modal run --detach` lost a run when the client died).
    `--extra` passes raw rl flags, e.g. `--extra "--resume.step 25"`."""
    config_toml = Path(config).read_text(encoding="utf-8")
    extra_args = extra.split() if extra else []
    if validate_only:
        rc = validate.remote(config_toml, run_name, extra_args, selfcheck)
    elif spawn:
        fn = modal.Function.from_name(app.name, "train8" if gpus == 8 else "train")
        call = fn.spawn(config_toml, run_name, extra_args)
        print(
            f"spawned train call {call.object_id} for run {run_name}; follow with: modal app logs {app.name}"
        )
        return
    else:
        rc = (train8 if gpus == 8 else train).remote(config_toml, run_name, extra_args)
    print(f"exit code {rc}")
    raise SystemExit(rc)
