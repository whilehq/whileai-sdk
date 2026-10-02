"""Context LM: train a small model that keeps its own context as a file.

    python recipe.py --selftest      # the harness and the credit on a stand-in model, offline
    python recipe.py --smoke         # Modal: 2 steps, 16 held-out logs, the live path
    python recipe.py --reuse         # three arms, four seeds, writes results.json

Context Language Models (arXiv:2609.37725) give the model its context as a
file it rewrites, instead of a transcript that only grows. Trained with
stepwise GRPO, they add a success-gated efficiency advantage (Eq. 6): among
a group's successes the cheaper trajectories go up, the dearer ones down.

The harness and the credit are ``wai.methods.ContextFile``; the task is
``wai.methods.KVLog``, a seeded key-value log in 5 chunks of 8 lines over 8
keys. Each step the model sees ``context.md`` and one chunk, then the chunk
is gone and it writes the whole new file; the last step asks the final
value of one key. Three arms, same model, task, reward, steps and seeds:

  baseline  stepwise GRPO, no efficiency term     ContextFile(gate="off")
  paper     + Eq. 6 on every success              ContextFile(gate="paper")
  recipe    + Eq. 6 on successes whose last file  ContextFile(gate="complete")
            holds the whole state

The first run of this recipe (two seeds, paper gate) split: one seed did
what the paper says, the other learned to copy only the latest chunk,
which answers ~70% of logs. ``complete`` stops paying that shortcut for
being cheap. The claim under test: the recipe arm keeps the baseline's
accuracy and spends fewer tokens, on every seed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PAPER = "https://arxiv.org/abs/2609.37725"
METRIC = "pass@1"
BOOK = "Reinforcement Learning"
# The training reward is the target (the boxed number against the log's
# final state), so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)
GROUP = 8  # trajectories per log: the GRPO group; Eq. 6 needs 2+ successes in it
# Arm -> ContextFile gate. `baseline` and `recipe` are the names check.py reads.
ARMS = {"baseline": "off", "paper": "paper", "recipe": "complete"}
SEEDS = (17, 18, 19, 20)


# --------------------------------------------------------------------------
# Modal
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "H100")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-context-lm")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        requirement(),
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def hf_generate(model, tokenizer, batch: int = 64):
    """A ``generate(messages, max_tokens)`` for ``ContextFile.play`` over a
    Hugging Face model, sampling the way the trainer samples."""
    import torch

    from whileai.context_file import Reply

    def generate(message_lists, max_tokens):
        model.eval()
        tokenizer.padding_side = "left"
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in message_lists
        ]
        out = []
        for start in range(0, len(texts), batch):
            enc = tokenizer(texts[start : start + batch], return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.9,
                    top_p=1.0,
                    max_new_tokens=max_tokens,
                    pad_token_id=pad,
                )
            new = gen[:, enc["input_ids"].shape[1] :]
            for j, text in enumerate(tokenizer.batch_decode(new, skip_special_tokens=True)):
                reply = new[j]
                stop = (reply == pad) | (reply == tokenizer.eos_token_id)
                n = int(stop.nonzero()[0]) + 1 if stop.any() else int(reply.numel())
                prompt = enc["input_ids"][j][enc["attention_mask"][j].bool()]
                out.append(Reply(text, prompt.tolist(), reply[:n].tolist()))
        model.train()
        return out

    return generate


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=5 * 60 * 60,
    memory=32768,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 60,
    tasks_per_step: int = 4,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    per_task: int = 4,
    train_seed: int = 17,
    eval_only: bool = False,
) -> dict:
    """One arm: ``ContextFile(gate=ARMS[arm]).trainer(GRPOTrainer)``, then
    the held-out eval. ``eval_only`` evaluates the untrained base EVAL_RUNS
    times (the noise floor) and stops."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    import whileai as wai
    from whileai.simulations.training import TrainerCallback, training_run

    env = wai.methods.KVLog()
    clm = wai.methods.ContextFile(gate=ARMS[arm])
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    def evaluate(m):
        tasks = [t for t in holdout for _ in range(per_task)]
        eps = clm.play(hf_generate(m, tokenizer), tasks, env)
        rep = clm.report(eps)
        stats = {
            "cost": round(rep.cost),
            "file_chars": round(rep.file_chars),
            "complete": round(rep.complete, 3),
            "shortcut": round(rep.shortcut, 3),
        }
        samples = [
            {"task": e.task["scenario_id"], "files": e.files, "answer": e.answer} for e in eps[:4]
        ]
        return clm.rows(eps), stats, samples

    if eval_only:
        base_runs, base_stats, samples = [], {}, []
        for i in range(EVAL_RUNS):
            rows, stats, s = evaluate(model)
            base_runs.append(rows)
            if i == 0:
                base_stats, samples = stats, s
            print(f"base run {i + 1}/{EVAL_RUNS}: {wai.pass_at(rows)} {stats}")
        return {
            "base_runs": base_runs,
            "base_stats": base_stats,
            "samples": samples,
            "gpu_minutes": (time.time() - started) / 60.0,
        }

    config = {
        "arm": arm,
        "method": str(clm),
        "base_model": base_model,
        "steps": steps,
        "env": {"chunks": env.chunks, "updates": env.updates, "keys": env.keys},
        "group": GROUP,
        "tasks_per_step": tasks_per_step,
        "learning_rate": learning_rate,
        "beta": 0.0,
        "lora_rank": lora_rank,
        "train_tasks": len(train_tasks),
        "holdout": len(holdout),
        "train_seed": train_seed,
        "gpu": DEFAULT_GPU,
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        run = training_run(
            run_name,
            base_model=base_model,
            trainer="trl-grpo-lora",
            total_steps=steps,
            config=config,
        )
        print(f"dashboard: {run.url}")

    dataset = Dataset.from_list([{**t, "prompt": env.messages(t, 0, "")} for t in train_tasks])
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=GROUP,
        per_device_train_batch_size=GROUP // 2,
        gradient_accumulation_steps=tasks_per_step * 2,
        learning_rate=learning_rate,
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        scale_rewards=False,
        max_completion_length=clm.file_tokens,
        max_prompt_length=1024,  # system, the file and a chunk
        temperature=0.9,
        bf16=True,
        # Off on purpose: this trainer generates during training, and
        # checkpointing corrupts Qwen generation on these pins.
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=train_seed,
    )
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    last_batch: list[dict] = []

    def answers(completions, prompts, gold, **kwargs) -> list[float]:
        """TRL calls this every step; the trainer pays the credit itself.
        It keeps the answer steps for `hack_scan`."""
        out = []
        for i, (p, c, g) in enumerate(zip(prompts, completions, gold)):
            text = c[0]["content"] if isinstance(c, list) else str(c)
            r = env.reward({"gold": g}, text)
            out.append(r)
            if "Question:" in json.dumps(p, default=str):
                last_batch.append(
                    {
                        "prompt": json.dumps(p, default=str),
                        "final_text": text,
                        "reward": r,
                        "scenario_id": json.dumps(p, default=str),
                        "rollout_index": i,
                    }
                )
        del last_batch[: -4 * GROUP * 8]
        return out

    set_seed(train_seed)
    trainer = clm.trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[answers],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        env=env,
    )
    if run is not None:
        trainer.add_callback(TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise
    history = trainer.state.log_history
    keys = ("reward", "cost", "file_chars", "complete", "shortcut", "eff_abs")
    trace = {k: [h[f"context/{k}"] for h in history if f"context/{k}" in h] for k in keys}

    after_rows, after_stats, samples = evaluate(trainer.model)
    print(f"{arm}: {wai.pass_at(after_rows)} {after_stats}")
    scan = wai.hack_scan(last_batch) if last_batch else {}
    hack_top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    run_url = ""
    if run is not None:
        run.finish(
            "done",
            summary={"arm": arm, "pass_at_1": wai.pass_at(after_rows).pass_at_1, **after_stats},
            adapter=f"whileai-recipe-runs:/{run_name}/adapter",
        )
        run_url = run.url
    return {
        "arm": arm,
        "after_rows": after_rows,
        "after_stats": after_stats,
        "samples": samples,
        "trace": trace,
        "hack_scan_top": hack_top.get("name", "") if isinstance(hack_top, dict) else str(hack_top),
        "gpu_minutes": gpu_minutes,
        "steps": steps,
        "run_url": run_url,
    }


# --------------------------------------------------------------------------
# Local: data, orchestration, results.json.
# --------------------------------------------------------------------------


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def mean_cost(rows: list[dict]) -> float:
    return statistics.fmean(float(r["cost_tokens"]) for r in rows) if rows else 0.0


def paired_cost(before: list[dict], after: list[dict]) -> dict:
    """Mean per-log change in tokens a trajectory, after minus before, with
    a 95% t interval over logs (the logs are the paired unit)."""
    from whileai.simulations.score.stats import _t_quantile

    def per_task(rows: list[dict]) -> dict[str, float]:
        by: dict[str, list[float]] = {}
        for r in rows:
            by.setdefault(r["scenario_id"], []).append(float(r["cost_tokens"]))
        return {k: statistics.fmean(v) for k, v in by.items()}

    a, b = per_task(before), per_task(after)
    diffs = [b[k] - a[k] for k in a if k in b]
    if len(diffs) < 2:
        return {"delta": 0.0, "ci": [0.0, 0.0], "n": len(diffs)}
    d = statistics.fmean(diffs)
    half = _t_quantile(len(diffs) - 1) * statistics.stdev(diffs) / len(diffs) ** 0.5
    return {
        "delta": round(d, 1),
        "ci": [round(d - half, 1), round(d + half, 1)],
        "relative": round(d / statistics.fmean(a.values()), 3),
        "n": len(diffs),
    }


def summarize(rows: list[dict], stats: dict | None = None) -> dict:
    import whileai as wai

    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
        "cost_tokens": round(mean_cost(rows)),
        **({"context": stats} if stats else {}),
    }


def _stand_in(copy_last: bool):
    """A scripted model: folds every chunk into a `key: value` table, or
    copies only the latest chunk (the shortcut the first run found)."""
    from whileai.context_file import Reply

    def generate(messages, max_tokens):
        out = []
        for m in messages:
            body = m[-1]["content"]
            file = body.split("```")[1].strip()
            table = dict(ln.split(": ") for ln in file.splitlines() if ": " in ln)
            sets = dict(ln[4:].split(" = ") for ln in file.splitlines() if ln.startswith("set "))
            if "Question" in body:
                key = body.split("final value of ")[1].split("?")[0]
                text = f"\\boxed{{{table.get(key) or sets.get(key, '0')}}}"
            elif copy_last:
                chunk = body.split("Log chunk")[1].splitlines()
                text = "\n".join(ln for ln in chunk if ln.startswith("set "))
            else:
                for ln in body.split("Log chunk")[1].splitlines():
                    if ln.startswith("set "):
                        k, v = ln[4:].split(" = ")
                        table[k] = v
                text = "\n".join(f"{k}: {v}" for k, v in table.items())
            out.append(Reply(text, list(range(len(body) // 4)), list(range(len(text) // 4))))
        return out

    return generate


def selftest() -> None:
    """The three gates on a group of table-keepers and shortcut-takers."""
    import whileai as wai

    env, clm = wai.methods.KVLog(), wai.methods.ContextFile()
    tasks = env.tasks(16)
    good = clm.play(_stand_in(False), tasks, env)
    short = clm.play(_stand_in(True), tasks, env)
    assert clm.report(good).pass_rate == 1.0 and clm.report(good).complete == 1.0
    rep = clm.report(short)
    assert rep.complete == 0.0 and 0.0 < rep.shortcut == rep.pass_rate < 1.0, rep
    print(f"shortcut model: right {rep.pass_rate:.2f}, never complete; table model: right 1.00")
    group = good[:2] + [e for e in short if e.reward == 1.0][:2]
    rewards = [e.reward for e in group]
    costs = [float(e.cost) for e in group]
    complete = [e.complete for e in group]
    for arm, gate in ARMS.items():
        eff = wai.methods.ContextFile(gate=gate).efficiency(rewards, costs, complete)
        print(
            f"{arm:<8} gate={gate:<8} Eq. 6 on [table, table, copy, copy]: {[round(x, 2) for x in eff]}"
        )
    gated = wai.methods.ContextFile(gate="complete").efficiency(rewards, costs, complete)
    assert gated[2:] == [0.0, 0.0] and any(gated[:2])
    print("selftest ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0, help="task seed")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="2 steps, 16 held-out logs, one seed")
    ap.add_argument("--reuse", action="store_true", help="skip calls cached in .cache/")
    ap.add_argument("--selftest", action="store_true", help="offline, a stand-in model")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    if args.smoke:
        args.steps, args.n_holdout, args.train_seeds = 2, 16, args.train_seeds[:1]
    env = wai.methods.KVLog()
    train_tasks = env.tasks(args.n_train, args.seed)
    holdout = env.tasks(args.n_holdout, args.seed, split="holdout")
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    print(f"{len(train_tasks)} train logs, {len(holdout)} held out")
    arms = list(ARMS) if args.arm == "all" else [args.arm]

    tag = "v2-smoke-" if args.smoke else "v2-"
    names = {(a, s): f"context-lm-{tag}{a}-s{s}" for s in args.train_seeds for a in arms}
    base_name = f"context-lm-{tag}base"
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)

    def cached(name: str) -> bool:
        return args.reuse and (cache / f"{name}.json").exists()

    with modal.enable_output(), app.run():
        calls = {}
        if not cached(base_name):
            calls[base_name] = run_arm.spawn("baseline", [], holdout, base_name, eval_only=True)
        for (a, s), name in names.items():
            if not cached(name):
                calls[name] = run_arm.spawn(
                    a, train_tasks, holdout, name, steps=args.steps, train_seed=s
                )
        failed = []
        for name, call in calls.items():
            try:
                out = call.get()
            except Exception as exc:  # one lost container must not lose the others
                print(f"FAILED {name}: {type(exc).__name__}: {exc}")
                failed.append(name)
                continue
            (cache / f"{name}.json").write_text(json.dumps(out))
    if failed:
        sys.exit(f"{len(failed)} call(s) failed: {failed}; rerun with --reuse")

    base = json.loads((cache / f"{base_name}.json").read_text())
    outs = {k: json.loads((cache / f"{n}.json").read_text()) for k, n in names.items()}
    gpu_minutes = base["gpu_minutes"] + sum(o["gpu_minutes"] for o in outs.values())
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 4.00)
    usd = gpu_minutes / 60.0 * usd_per_hour
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${usd:.2f} on {DEFAULT_GPU}")

    if args.smoke:
        smoke = {
            "base": summarize(base["base_runs"][0], base["base_stats"]),
            **{
                n: {
                    **summarize(outs[k]["after_rows"], outs[k]["after_stats"]),
                    "trace": outs[k]["trace"],
                }
                for k, n in names.items()
            },
            "gpu_minutes": round(gpu_minutes, 1),
            "usd": round(usd, 2),
        }
        (cache / "v2-smoke.json").write_text(json.dumps(smoke, indent=2))
        print(json.dumps(smoke, indent=2)[:4000])
        print("wrote .cache/v2-smoke.json; a smoke run claims no number and writes no results.json")
        return

    noise = wai.eval_variance(*base["base_runs"])
    run_std = float(noise["run_std"])
    seeds = args.train_seeds
    after = {a: [outs[(a, s)]["after_rows"] for s in seeds] for a in arms}
    results: dict = {
        "recipe": HERE.name,
        "title": "Context LM: a small model that keeps its own context as a file",
        "paper": PAPER,
        "book": BOOK,
        "base_model": BASE_MODEL,
        "metric": METRIC,
        "n_holdout": len(holdout),
        "k": 4,
        "gates": dict(ARMS),
        "arms": {
            "base": {
                **summarize(base["base_runs"][0], base["base_stats"]),
                "steps": 0,
                "gpu_minutes": 0,
            }
        },
        "samples": {"base": base["samples"]},
        "train_trace": {},
        "checks": {
            "run_std": run_std,
            "run_std_runs": int(noise["n_runs"]),
            "train_seeds": {a: len(after[a]) for a in arms},
            "decontaminated_dropped": int(decon.get("n_contaminated", 0)),
            "over_optimized": False,
            "length_before": mean_length(base["base_runs"][0]),
            "length_after": {},
            "hack_scan_top": "",
            "seed": args.seed,
        },
        "gpu": DEFAULT_GPU,
        "usd": round(usd, 2),
        "verified": date.today().isoformat(),
        "whileai": version("whileai"),
        "run_url": "",
    }
    for a in arms:
        per = [outs[(a, s)] for s in seeds]
        pooled = [r for rows in after[a] for r in rows]
        results["arms"][a] = {
            **summarize(pooled),
            "gate": ARMS[a],
            "per_seed": [round(summarize(rows)["score"], 3) for rows in after[a]],
            "cost_per_seed": [round(mean_cost(rows)) for rows in after[a]],
            "context_per_seed": [o["after_stats"] for o in per],
            "steps": args.steps,
            "gpu_minutes": round(statistics.fmean(o["gpu_minutes"] for o in per), 1),
        }
        results["train_trace"][a] = [o["trace"] for o in per]
        results["samples"][a] = per[0]["samples"]
        results["checks"]["length_after"][a] = mean_length(pooled)
        if per[0]["hack_scan_top"]:
            results["checks"]["hack_scan_top"] = per[0]["hack_scan_top"]
        results["run_url"] = per[0]["run_url"] or results["run_url"]

    def paired(x: str, y: str) -> dict:
        xs = [r for rows in after[x] for r in rows]
        ys = [r for rows in after[y] for r in rows]
        d = wai.compare(
            xs,
            ys,
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(noise["n_runs"]),
            train_runs={"before": after[x], "after": after[y]},
            proxy=PROXY,
        )
        verdict = d["target_verdict"]
        return {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": verdict if verdict in ("moved", "flat", "unresolved") else "flat",
            "noise_band": d.get("noise_band"),
            "over_optimized": bool(d.get("over_optimized")),
            "cost": {
                "per_seed": [paired_cost(b, r) for b, r in zip(after[x], after[y])],
                "pooled": paired_cost(xs, ys),
            },
        }

    if "baseline" in arms and len(arms) > 1:
        deltas = {a: paired("baseline", a) for a in arms if a != "baseline"}
        if {"paper", "recipe"} <= set(arms):
            deltas["recipe_vs_paper"] = paired("paper", "recipe")
        over = [d.pop("over_optimized") for d in deltas.values()]
        results["checks"]["over_optimized"] = any(over)
        results["deltas"] = deltas
        if "recipe" in deltas:
            results["delta"] = {k: v for k, v in deltas["recipe"].items() if k != "cost"}
            results["cost_delta"] = deltas["recipe"]["cost"]
        print(json.dumps(deltas, indent=2))
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"

    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("wrote results.json")


if __name__ == "__main__":
    main()
