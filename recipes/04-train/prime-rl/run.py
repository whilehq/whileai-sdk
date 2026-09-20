"""GRPO, OPSD and OPD on the same taskset, on prime-rl, on Modal, with your keys.

    python run.py --validate        # write the three configs, dry-run each on a CPU container
    python run.py                   # deploy, spawn the three runs (2 GPUs each), record call ids
    python run.py --collect         # when they finish: results.json and the table
    python run.py --arms opsd opd   # a subset

The three arms share one student (`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`),
one taskset (`reverse-text`, bundled with prime-rl; the demonstration is its
top-level `answer` field), one held-out slice of 128 prompts, one step count
and one learning rate, the values prime-rl's own
`configs/debug/algo/{opd,self_distill}.toml` use. The one
thing that differs per arm is the algorithm:

  grpo   the reward is the taskset's own (LCS ratio of the reversed text)
  opsd   no reward: the teacher is the same model shown the answer, reverse KL per token
  opd    no reward: the teacher is the RL-trained checkpoint of the same model, served frozen

Every config is written by `wai.prime_rl_config`; the only edits are the
overrides below, each named. `results.json` carries, per arm, the in-run
eval reward at every eval step, the training reward or teacher KL per step,
the wall-clock, the image tag and the config as run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import whileai as wai

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
STUDENT = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
TEACHER = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
TASKSET = "reverse-text"
STEPS = 20
MAX_TOKENS = 128  # reverse-text replies are one short line; prime-rl's own configs cap at 128
LR = 3e-6  # prime-rl's debug configs train these 0.6B checkpoints at 3e-6, full weights
HOLDOUT = 128  # the last 128 of the dataset's 1,000 prompts are never trained on
TRAIN_SPLIT = f"train[:{1000 - HOLDOUT}]"  # HF slicing; the taskset passes it to load_dataset
EVAL_SPLIT = f"train[{1000 - HOLDOUT}:]"

# The same on every arm: the sequence cap, the Qwen3 renderer the reverse-text
# checkpoints were trained under, the policy engine at 0.4 of GPU 0 so a frozen
# teacher can share it, a task-disjoint holdout (PrimeIntellect/Reverse-Text-RL
# ships one `train` split of 1,000 prompts; prime-rl's own configs eval on the
# training rows, this one does not), and the eval read every five steps.
COMMON = {
    "seq_len": 2048,
    "orchestrator.renderer.name": "prime-qwen3",
    "inference.vllm.gpu_memory_utilization": 0.4,
    "train_source.env.taskset.dataset_split": TRAIN_SPLIT,
    "eval_source.env.taskset.dataset_split": EVAL_SPLIT,
    "orchestrator.eval.interval": 5,
    "orchestrator.eval.num_examples": HOLDOUT,
    "orchestrator.eval.group_size": 1,
    "orchestrator.eval.sampling.max_completion_tokens": MAX_TOKENS,
    "trainer.optim.lr": LR,
}


def configs() -> dict[str, wai.methods.PrimeRLConfig]:
    """Write the three configs and return them, keyed by arm."""
    CONFIGS.mkdir(exist_ok=True)
    grpo = wai.prime_rl_config(
        TASKSET,
        "grpo",
        model=STUDENT,
        steps=STEPS,
        batch=128,
        lora=False,
        out=CONFIGS / "grpo.toml",
        **COMMON,
        **{
            "orchestrator.group_size": 16,  # prime-rl's reverse-text RL config
            "orchestrator.train.sampling.max_completion_tokens": MAX_TOKENS,
        },
    )
    opsd = wai.prime_rl_config(
        TASKSET,
        wai.OPSD(privileged="answer", anchor="live", max_tokens=MAX_TOKENS),
        model=STUDENT,
        steps=STEPS,
        batch=32,  # one sample per prompt, so a smaller prompt batch (prime-rl's self_distill.toml)
        lora=False,
        out=CONFIGS / "opsd.toml",
        **COMMON,
    )
    opd = wai.prime_rl_config(
        TASKSET,
        wai.OPD(
            wai.Endpoint(url="http://localhost:8001/v1", model=TEACHER),
            samples=16,
            max_tokens=MAX_TOKENS,
        ),
        model=STUDENT,
        steps=STEPS,
        batch=128,
        lora=False,
        out=CONFIGS / "opd.toml",
        **COMMON,
    )
    return {"grpo": grpo, "opsd": opsd, "opd": opd}


def _modal(args: list[str]) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "modal", "run", str(HERE / "modal_prime_rl.py"), *args]
    return subprocess.Popen(cmd, cwd=HERE)


def validate(cfgs: dict[str, wai.methods.PrimeRLConfig]) -> int:
    rc = 0
    for arm, cfg in cfgs.items():
        print(f"--- validate {arm}: {cfg.path}")
        proc = _modal(["--config", cfg.path, "--run-name", f"{arm}-dry", "--validate-only"])
        code = proc.wait()
        print(f"--- {arm}: {'ok' if code == 0 else f'exit {code}'}")
        rc |= code
    return rc


APP = "wai-prime-rl"
LAUNCH = HERE / "launch.json"


def launch(cfgs: dict[str, wai.methods.PrimeRLConfig], tag: str) -> dict[str, str]:
    """Deploy the app once, spawn one `train` call per arm, record the call ids.
    A spawned call outlives this process; `--collect` reads the results back.
    (`modal run --detach` does not: a dead client takes the container with it.)"""
    import modal

    subprocess.run(
        [sys.executable, "-m", "modal", "deploy", str(HERE / "modal_prime_rl.py")],
        cwd=HERE,
        check=True,
    )
    train = modal.Function.from_name(APP, "train")
    calls: dict[str, str] = {}
    for arm, cfg in cfgs.items():
        teacher = TEACHER if arm == "opd" else None
        call = train.spawn(Path(cfg.path).read_text(encoding="utf-8"), f"{arm}-{tag}", teacher, [])
        calls[arm] = call.object_id
        print(f"spawned {arm}-{tag}: {call.object_id}")
    LAUNCH.write_text(json.dumps({"tag": tag, "calls": calls}, indent=2), encoding="utf-8")
    return calls


def collect(cfgs: dict[str, wai.methods.PrimeRLConfig]) -> dict | None:
    """Read back every spawned call; None while any is still running."""
    import modal

    state = json.loads(LAUNCH.read_text(encoding="utf-8"))
    results: dict[str, dict] = {}
    for arm, call_id in state["calls"].items():
        if arm not in cfgs:
            continue
        try:
            results[arm] = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            print(f"{arm}: still running")
            return None
        except Exception as e:  # the call raised: keep the message, the run is over
            results[arm] = {"returncode": -1, "error": f"{type(e).__name__}: {e}"[:2000]}
    for arm, res in results.items():
        (HERE / f"{arm}-{state['tag']}.result.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8"
        )
    return summarize(results, cfgs, state["tag"])


def _curve(rows: list[dict], key_part: str) -> list[tuple[int, float]]:
    """(step, value) for every metric row that has a step and a key containing key_part."""
    points: list[tuple[int, float]] = []
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        for k, v in row.items():
            if key_part in k and isinstance(v, (int, float)) and not isinstance(v, bool):
                points.append((step, float(v)))
                break
    return sorted(points)


def summarize(
    results: dict[str, dict], cfgs: dict[str, wai.methods.PrimeRLConfig], tag: str
) -> dict:
    summary: dict = {
        "tag": tag,
        "student": STUDENT,
        "teacher": TEACHER,
        "taskset": TASKSET,
        "steps": STEPS,
        "whileai": wai.__version__,
        "arms": {},
    }
    for arm, res in results.items():
        rows = res.get("metrics", [])
        keys = sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float))})
        summary["arms"][arm] = {
            "returncode": res.get("returncode"),
            "seconds": res.get("seconds"),
            "image": res.get("image"),
            "gpu": res.get("gpu"),
            "metric_keys": keys,
            "eval_reward": _curve([r for r in rows if "eval" in r.get("_file", "")], "reward"),
            "train_reward": _curve([r for r in rows if "eval" not in r.get("_file", "")], "reward"),
            "teacher_kl": _curve(rows, "kl"),
            "config": cfgs[arm].text,
            "reads": cfgs[arm].honored,
            "ignores": cfgs[arm].ignored,
        }
    (HERE / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="dry-run the configs on CPU")
    ap.add_argument("--collect", action="store_true", help="read back the spawned runs")
    ap.add_argument("--arms", nargs="*", default=["grpo", "opsd", "opd"])
    ap.add_argument("--tag", default=time.strftime("%m%d-%H%M"))
    a = ap.parse_args()
    cfgs = {k: v for k, v in configs().items() if k in a.arms}
    if not a.collect:
        for cfg in cfgs.values():
            print(cfg)
            print()
    if a.validate:
        return validate(cfgs)
    if not a.collect:
        launch(cfgs, a.tag)
        print(f"spawned; run `python run.py --collect --arms {' '.join(cfgs)}` to read the results")
        return 0
    summary = collect(cfgs)
    if summary is None:
        return 3
    for arm, s in summary["arms"].items():
        ev = s["eval_reward"]
        first = f"{ev[0][1]:.3f} @ {ev[0][0]}" if ev else "none"
        last = f"{ev[-1][1]:.3f} @ {ev[-1][0]}" if ev else "none"
        print(f"{arm:5s} exit {s['returncode']} {s['seconds']}s  eval reward {first} -> {last}")
    return max(int(s["returncode"] or 0) for s in summary["arms"].values())


if __name__ == "__main__":
    raise SystemExit(main())
