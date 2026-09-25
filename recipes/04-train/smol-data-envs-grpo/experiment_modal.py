"""GRPO on SmolDataEnvs, with and without whileai. See PREREGISTRATION.md.

    modal run experiment_modal.py::prep                       # tasks + tables onto the volume (CPU)
    modal run experiment_modal.py::preflight                  # base model: pool pre-flight + test eval
    modal run experiment_modal.py::train --arm authors --seed 1 --steps 212
    modal run experiment_modal.py::train --arm whileai --seed 1 --steps 200
    modal run experiment_modal.py::fetch                      # rows back to out/

Add ``--spawn`` to ``train`` to start it on the deployed app and return.
Every function writes JSON to the volume ``smol-data-envs-grpo`` under
/vol; ``fetch`` copies it to out/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "smol-data-envs-grpo"
MODEL = "Qwen/Qwen3.5-2B"
DATASET = "FineEnvs/SmolDataEnvs"
REVISION = "b2bf35647e2381b1ab12c2ad7862cbbe4e8f857b"
POOL = 1024  # the whileai arm's candidate pool: the first 1,024 of the authors' shuffle
N_TRAIN = 256  # tasks each arm trains on (the authors' NUM_TASKS)
BAND = (0.2, 0.8)
PREFLIGHT_K = 8
EVAL_K = 4
MAX_TABLE_BYTES = 500 * 1024 * 1024  # a bigger file is skipped for every arm alike
VOL = "/vol"

# The authors' train_grpo.py defaults (FineEnvs @ 08a5622).
AUTHORS = {
    "learning_rate": 3e-6,
    "temperature": 0.8,
    "top_p": 1.0,
    "num_generations": 8,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_completion_length": 1024,
    "repetition_penalty": 1.05,
    "vllm_gpu_memory_utilization": 0.22,
}

app = modal.App(APP)
volume = modal.Volume.from_name(APP, create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)

# What the dataset's own sandbox instructions promise the program (see a row's
# `instruction`): pandas, numpy, matplotlib, seaborn, scipy, scikit-learn,
# statsmodels, tabulate.
_analysis_libs = [
    "pandas", "numpy", "matplotlib", "seaborn", "scipy", "scikit-learn", "statsmodels",
    "tabulate", "pyarrow", "openpyxl",
]  # fmt: skip

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets>=3.6.0", "requests", *_analysis_libs)
    .env({"MPLBACKEND": "Agg"})
    .add_local_file(str(HERE / "grader.py"), "/root/grader.py")
    .add_local_file(str(HERE / "data_env.py"), "/root/data_env.py")
)

# The stack recipes/04-train/text-to-sql proved on Qwen3.5 (model_type qwen3_5).
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .pip_install(
        "vllm==0.29.0",
        "transformers==5.17.0",
        "trl==1.13.0",
        "flash-linear-attention",
        "datasets>=3.6.0",
        "accelerate>=1.8.1",
        "whileai==0.126",
        *_analysis_libs,
    )
    .env(
        {
            "HF_HOME": "/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MPLBACKEND": "Agg",
        }
    )
    .add_local_file(str(HERE / "grader.py"), "/root/grader.py")
    .add_local_file(str(HERE / "data_env.py"), "/root/data_env.py")
)


def _save(name: str, obj) -> None:
    path = Path(VOL) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=str), encoding="utf-8")
    volume.commit()


def _load(name: str):
    volume.reload()
    return json.loads((Path(VOL) / name).read_text(encoding="utf-8"))


def _score_many(rows: list[dict], completions: list, workers: int = 32) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, "/root")
    import data_env

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda rc: data_env.score(rc[0], rc[1], f"{VOL}/tables"), zip(rows, completions))
        )


# -- 1. tasks and tables -------------------------------------------------------


@app.function(image=cpu_image, volumes={VOL: volume}, cpu=8, timeout=4 * 3600)
def prep() -> dict:
    """The authors' shuffle, the whileai pool, the test split, and every table
    they read, downloaded once onto the volume."""
    from concurrent.futures import ThreadPoolExecutor

    import requests
    from datasets import load_dataset

    keep = [
        "task_id", "question", "answer", "reward_mode", "atol", "rtol",
        "difficulty_tier", "hf_bucket", "bucket_prefix", "files",
    ]  # fmt: skip
    train = load_dataset(DATASET, split="train", revision=REVISION).shuffle(seed=42)
    pool = [{k: r[k] for k in keep} for r in train.select(range(POOL))]
    test = [{k: r[k] for k in keep} for r in load_dataset(DATASET, split="test", revision=REVISION)]
    test = [t for t in test if t["files"]]

    wanted = {(t["hf_bucket"], t["bucket_prefix"], f) for t in pool + test for f in t["files"]}

    def fetch(item):
        bucket, prefix, name = item
        dest = Path(VOL) / "tables" / prefix / name
        if dest.exists() and dest.stat().st_size > 0:
            return name, dest.stat().st_size, "cached"
        url = f"https://huggingface.co/buckets/{bucket}/resolve/{prefix}/{name}"
        for attempt in range(3):
            try:
                head = requests.head(url, allow_redirects=False, timeout=60)
                size = int(head.headers.get("X-Linked-Size") or 0)
                if size > MAX_TABLE_BYTES:
                    return name, size, "too big"
                dest.parent.mkdir(parents=True, exist_ok=True)
                with requests.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(str(dest) + ".part", "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                os.replace(str(dest) + ".part", dest)
                return name, dest.stat().st_size, "ok"
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"[:120]
                time.sleep(2**attempt)
        return name, 0, err

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(fetch, sorted(wanted)))
    status: dict[str, int] = {}
    for _, _, s in results:
        key = s if s in ("ok", "cached", "too big") else "error"
        status[key] = status.get(key, 0) + 1
    total_gb = sum(size for _, size, s in results if s in ("ok", "cached")) / 1e9
    summary = {
        "pool": len(pool),
        "test": len(test),
        "files": len(wanted),
        "status": status,
        "gb": round(total_gb, 2),
        "errors": [r for r in results if r[2] not in ("ok", "cached", "too big")][:10],
    }
    _save("tasks.json", {"pool": pool, "test": test, "authors": pool[:N_TRAIN]})
    _save("prep.json", summary)
    print(json.dumps(summary, indent=1))
    return summary


# -- shared generation ----------------------------------------------------------


def _sampling(n: int, greedy: bool = False):
    from vllm import SamplingParams

    if greedy:
        return SamplingParams(n=1, temperature=0.0, max_tokens=AUTHORS["max_completion_length"])
    return SamplingParams(
        n=n,
        temperature=AUTHORS["temperature"],
        top_p=AUTHORS["top_p"],
        repetition_penalty=AUTHORS["repetition_penalty"],
        max_tokens=AUTHORS["max_completion_length"],
    )


def _generate(llm, tokenizer, tasks: list[dict], params) -> list[list[str]]:
    sys.path.insert(0, "/root")
    import data_env

    texts = [
        tokenizer.apply_chat_template(
            data_env.build_prompt(t), tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for t in tasks
    ]  # fmt: skip
    outs = llm.generate(texts, params)
    return [[o.text for o in out.outputs] for out in outs]


def _graded_rows(tasks: list[dict], samples: list[list[str]], version: str) -> list[dict]:
    flat_tasks, flat_text, idx = [], [], []
    for t, outs in zip(tasks, samples, strict=True):
        for i, text in enumerate(outs):
            flat_tasks.append(t)
            flat_text.append(text)
            idx.append(i)
    scores = _score_many(flat_tasks, flat_text)
    return [
        {
            "scenario_id": t["task_id"],
            "rollout_index": i,
            "category": t["difficulty_tier"],
            "model_version": version,
            "final_text": text,
            **s,
        }
        for t, text, i, s in zip(flat_tasks, flat_text, idx, scores, strict=True)
    ]


def _evaluate(llm, tokenizer, tasks: list[dict], version: str) -> dict:
    t0 = time.time()
    sampled = _graded_rows(tasks, _generate(llm, tokenizer, tasks, _sampling(EVAL_K)), version)
    greedy = _graded_rows(tasks, _generate(llm, tokenizer, tasks, _sampling(1, greedy=True)), version)
    import whileai as wai

    pa = wai.pass_at(sampled)
    g = [r["reward"] for r in greedy if r["reward"] is not None]
    print(f"[{version}] pass@1 {pa.pass_at_1:.3f} ci {pa.ci95}  greedy {sum(g) / max(len(g), 1):.3f}")
    return {"sampled": sampled, "greedy": greedy, "seconds": time.time() - t0}


# -- 2. pre-flight + base eval --------------------------------------------------


@app.function(
    image=gpu_image, gpu="H100", cpu=32, memory=65536,
    volumes={VOL: volume, "/hf": hf_cache}, timeout=4 * 3600,
)  # fmt: skip
def preflight(limit: int = 0) -> dict:
    """The whileai arm's only extra step, timed so the authors' arm can be
    handed the same GPU seconds: sample the base model on the pool, grade,
    and let wai.select keep the tasks inside the band."""
    from transformers import AutoTokenizer
    from vllm import LLM

    import whileai as wai

    tasks = _load("tasks.json")
    pool, test = tasks["pool"], tasks["test"]
    if limit:
        pool, test = pool[:limit], test[:limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(MODEL, gpu_memory_utilization=0.85, max_model_len=4096, seed=0)

    t0 = time.time()
    rows = _graded_rows(pool, _generate(llm, tokenizer, pool, _sampling(PREFLIGHT_K)), "base")
    preflight_s = time.time() - t0
    picked = wai.select(
        [dict(r, prompt=r["scenario_id"]) for r in rows], mode="rl", band=BAND, target=10**6
    )
    print(picked)
    in_band = {r["scenario_id"] for r in picked}
    chosen = [t for t in pool if t["task_id"] in in_band][:N_TRAIN]
    per_task: dict[str, list[float]] = {}
    for r in rows:
        if r["reward"] is not None:
            per_task.setdefault(r["scenario_id"], []).append(r["reward"])
    rate = {k: sum(v) / len(v) for k, v in per_task.items()}
    authors_ids = [t["task_id"] for t in tasks["authors"]]
    summary = {
        "preflight_seconds": preflight_s,
        "pool": len(pool),
        "in_band": len(in_band),
        "chosen": len(chosen),
        "report": picked.report,
        "authors_tasks_by_base_rate": {
            "never": sum(1 for i in authors_ids if rate.get(i, 0) == 0),
            "always": sum(1 for i in authors_ids if rate.get(i, 0) == 1),
            "in_band": sum(1 for i in authors_ids if BAND[0] <= rate.get(i, 0) <= BAND[1]),
        },
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "report"}, indent=1))
    _save("preflight/rows.json", rows)
    _save("preflight/summary.json", summary)
    if not limit:
        _save("whileai_tasks.json", chosen)

    base = _evaluate(llm, tokenizer, test, "base")
    _save(f"eval/base{'-smoke' if limit else ''}.json", base)
    return json.loads(json.dumps(summary, default=str))


# -- 3. train one arm, one seed, then evaluate it --------------------------------


def _guard_vllm_weight_sync(trainer) -> None:
    """Qwen3.5 is a VLM-class checkpoint: vLLM names the text stack
    ``model.language_model.*``, transformers names it ``model.*``. Map the
    names and skip the vision tower, which never changes. From
    recipes/04-train/text-to-sql/train_grpo_modal.py."""
    holder = None
    for v in vars(trainer).values():
        if hasattr(v, "llm") and hasattr(v.llm, "llm_engine"):
            holder = v
            break
    if holder is None:
        print("weight-sync guard: no colocated vLLM on the trainer; not installed")
        return
    vmodel = holder.llm.llm_engine.model_executor.driver_worker.model_runner.model
    orig = vmodel.load_weights
    skipped: list[str] = []

    def _candidates(name: str) -> list[str]:
        out = [name]
        if name.startswith("model.") and not name.startswith(
            ("model.language_model.", "model.visual.")
        ):
            out.append("model.language_model." + name[len("model.") :])
        return out

    def load_weights(weights):
        loaded: set[str] = set()
        for name, tensor in weights:
            for cand in _candidates(name):
                try:
                    loaded |= set(orig([(cand, tensor)]) or ())
                    break
                except (ValueError, KeyError):
                    continue
            else:
                if len(skipped) < 8:
                    print(f"weight-sync guard: skipping {name!r}")
                skipped.append(name)
        return loaded

    vmodel.load_weights = load_weights
    print(f"weight-sync guard installed on {type(vmodel).__name__}")


@app.function(
    image=gpu_image, gpu="H100", cpu=16, memory=65536,
    volumes={VOL: volume, "/hf": hf_cache}, timeout=20 * 3600,
)  # fmt: skip
def train(arm: str, seed: int, steps: int, smoke: bool = False) -> dict:
    import torch
    import transformers
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    import data_env

    for k, v in {
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511",
    }.items():  # fmt: skip
        os.environ.setdefault(k, v)

    tasks = _load("tasks.json")
    train_tasks = tasks["authors"] if arm == "authors" else _load("whileai_tasks.json")
    test = tasks["test"][:16] if smoke else tasks["test"]
    by_id = {t["task_id"]: t for t in train_tasks}
    ds = Dataset.from_list(
        [{"prompt": data_env.build_prompt(t), "task_id": t["task_id"]} for t in train_tasks]
    )
    run = f"{arm}-s{seed}{'-smoke' if smoke else ''}"

    stats = {"calls": 0, "zero_spread_groups": 0, "groups": 0, "ungraded": 0}

    def reward_correct(completions, task_id, **_):
        rows = [by_id[i] for i in task_id]
        texts = [c[-1]["content"] if isinstance(c, list) else str(c) for c in completions]
        res = _score_many(rows, texts, workers=16)
        out = [r["reward"] for r in res]
        # groups of num_generations share a task: count the ones with no spread
        g = AUTHORS["num_generations"]
        for s in range(0, len(out), g):
            grp = [x for x in out[s : s + g] if x is not None]
            stats["groups"] += 1
            stats["zero_spread_groups"] += int(len(set(grp)) <= 1)
        stats["ungraded"] += sum(1 for x in out if x is None)
        stats["calls"] += 1
        return out

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    transformers.set_seed(seed)  # before the model exists: TRL seeds too late for init
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    except ValueError:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16
        )
    cfg = GRPOConfig(
        output_dir=f"/tmp/{run}",
        use_vllm=True,
        vllm_mode="colocate",
        chat_template_kwargs={"enable_thinking": False},
        vllm_gpu_memory_utilization=AUTHORS["vllm_gpu_memory_utilization"],
        num_generations=AUTHORS["num_generations"],
        max_completion_length=AUTHORS["max_completion_length"],
        mask_truncated_completions=True,
        max_steps=steps,
        learning_rate=AUTHORS["learning_rate"],
        temperature=AUTHORS["temperature"],
        top_p=AUTHORS["top_p"],
        repetition_penalty=AUTHORS["repetition_penalty"],
        per_device_train_batch_size=AUTHORS["per_device_train_batch_size"],
        gradient_accumulation_steps=AUTHORS["gradient_accumulation_steps"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_correct],
        train_dataset=ds,
        args=cfg,
    )
    _guard_vllm_weight_sync(trainer)
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    history = [h for h in trainer.state.log_history if "reward" in h or "loss" in h]
    print(f"[{run}] {steps} steps in {train_s:.0f}s ({train_s / max(steps, 1):.1f} s/step); {stats}")

    trainer.vllm_generation.sync_weights()
    result = _evaluate(trainer.vllm_generation.llm, tokenizer, test, run)
    _save(
        f"eval/{run}.json",
        {
            **result,
            "arm": arm,
            "seed": seed,
            "steps": steps,
            "train_seconds": train_s,
            "reward_stats": stats,
            "history": history,
            "train_task_ids": [t["task_id"] for t in train_tasks],
        },
    )
    return json.loads(json.dumps({"run": run, "train_seconds": train_s, "stats": stats}))


@app.local_entrypoint()
def main(
    step: str = "",
    arm: str = "authors",
    seed: int = 1,
    steps: int = 200,
    smoke: bool = False,
    spawn: bool = False,
    limit: int = 0,
) -> None:
    if step == "prep":
        print(prep.remote())
    elif step == "preflight":
        if spawn:
            call = modal.Function.from_name(APP, "preflight").spawn(limit)
            print(f"spawned preflight: {call.object_id}")
        else:
            print(preflight.remote(limit))
    elif step == "train":
        if spawn:
            fn = modal.Function.from_name(APP, "train")
            call = fn.spawn(arm, seed, steps, smoke)
            print(f"spawned {arm} seed {seed}: {call.object_id}")
        else:
            print(train.remote(arm, seed, steps, smoke))
    elif step == "fetch":
        out = HERE / "out"
        out.mkdir(exist_ok=True)
        for entry in volume.listdir("/", recursive=True):
            if entry.path.endswith(".json"):
                data = b"".join(volume.read_file(entry.path))
                dest = out / entry.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                print(f"  {entry.path}")
    else:
        print(__doc__)
