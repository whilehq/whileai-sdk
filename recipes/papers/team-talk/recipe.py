"""Team talk: train a model to reason as a team talking to itself.

    python recipe.py                      # both arms on Modal, two seeds, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the talk counter and the grader, offline, no GPU

Self-Organizing Agent Teams (Pappu et al. 2026) put three different models
in a conversation and learned, without training, who should propose, who
should challenge and who should settle. The team beat its best member and a
perfect router over the members' own answers: talking produced answers none
of them had alone. Societies of Thought (Kim et al. 2026) found the same
shape inside one model: reasoning models' traces read like a conversation,
and RL on accuracy alone makes a base model talk more.

So put the team in the trace and train it there. One model, one reply, three
speakers: Solver proposes, Checker redoes the arithmetic and says what is
wrong, Lead settles and writes the answer. The reward is the answer, never
the talk, so whatever talk survives training is talk that paid.

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out
  2. run_arm():   TRL GRPOTrainer + LoRA on Modal, one arm per call
  3. evaluate():  same holdout, k samples per task, graded by MathEqual,
                  under the arm's own prompt AND the plain one
  4. results.json + two paired deltas (wai.compare): as trained, and
     with the team prompt taken away (did the weights remember to talk?)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Reasoning"  # RL on verifiable rewards and what it does to the trace
# The training reward here *is* the target: both are the same binary check
# against the GSM8K gold, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."

# THE ONE CHANGE: the prompt the recipe arm trains under. Three roles, the
# ones Pappu et al.'s learned strategies keep converging on (propose,
# challenge, synthesize), written as turns in one reply.
SPEAKERS = ("Solver", "Checker", "Lead")
SYSTEM_TEAM = (
    "Solve the problem as a team of three talking to each other.\n"
    "Solver proposes the steps. Checker redoes every calculation and says plainly "
    "what is wrong. Lead settles any disagreement and gives the final number.\n"
    "Start every turn on a new line with the speaker's name, for example:\n"
    "Solver: ...\nChecker: ...\nSolver: ...\nLead: ... \\boxed{answer}"
)
TURN = re.compile(rf"^[\s*#>-]*({'|'.join(SPEAKERS)})\**\s*:", re.MULTILINE)


# --------------------------------------------------------------------------
# Pure functions, no torch: `--selftest` runs them, and the Modal container
# imports this same file.
# --------------------------------------------------------------------------


def messages_for(question: str, team: bool = False) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_TEAM if team else SYSTEM},
        {"role": "user", "content": question},
    ]


def gold_of(answer: str) -> str:
    """GSM8K ships the worked solution then `#### 72`. The gold is the tail."""
    return answer.split("####")[-1].strip().replace(",", "")


def outcome_of(text: str, gold: str) -> float:
    """1.0 when the final number matches the gold, else 0.0. A program, not a
    judge: `MathEqual` is Math-Verify."""
    from whileai.simulations.verify import MathEqual

    row = {
        "prompt": "",
        "final_text": text,
        "privileged": {"reference": gold},
        "scenario_id": "x",
        "rollout_index": 0,
    }
    return 1.0 if MathEqual()(row).get("reward") == 1 else 0.0


def turns_of(text: str) -> list[str]:
    """The speakers in order, one entry per turn."""
    return TURN.findall(text or "")


def talks(text: str) -> bool:
    """A reply is a conversation when at least two speakers take at least
    three turns between them, and Checker is one of them: a Solver monologue
    signed by Lead is not a team."""
    turns = turns_of(text)
    return len(turns) >= 3 and len(set(turns)) >= 2 and "Checker" in turns


def talk_rate(rows: list[dict]) -> float:
    return statistics.fmean(talks(r.get("final_text") or "") for r in rows) if rows else 0.0


def mean_turns(rows: list[dict]) -> float:
    return statistics.fmean(len(turns_of(r.get("final_text") or "")) for r in rows) if rows else 0.0


def make_reward(recorder: list[dict]):
    """Binary outcome, a program against the public GSM8K gold. Both arms use
    this untouched: the talk is never paid for, only the answer.

    `recorder` is refilled with the batch it just graded; `hack_scan` reads
    the last one after training (Lambert 2025, chapter Over-optimization).
    """

    def reward(completions, prompts, answer, **kwargs) -> list[float]:
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        keys = [json.dumps(p, sort_keys=True, default=str) for p in prompts]
        rewards = [outcome_of(t, gold_of(a)) for t, a in zip(texts, answer)]
        seen: dict[str, int] = {}
        recorder.clear()
        for key, text, r in zip(keys, texts, rewards):
            recorder.append(
                {
                    "prompt": key,
                    "final_text": text,
                    "reward": r,
                    "scenario_id": key,
                    "rollout_index": seen.get(key, 0),
                }
            )
            seen[key] = seen.get(key, 0) + 1
        return rewards

    reward.__name__ = "gsm8k_outcome"
    return reward


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `compare` read: binary
    `reward`, one row per sample, grouped by task."""
    rows: list[dict] = []
    for task, texts in zip(holdout, replies):
        gold = gold_of(task["answer"])
        for i, text in enumerate(texts):
            rows.append(
                {
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": outcome_of(text, gold),
                    "scenario_id": task["scenario_id"],
                    "rollout_index": i,
                    "privileged": {"reference": gold},
                }
            )
    return rows


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/papers/reinforce-ada.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-team-talk")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        # MathEqual decides with Math-Verify, the `whileai[math]` extra.
        "math-verify>=0.8",
        requirement(),
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _sample(model, tokenizer, questions, *, team, n, max_new_tokens, batch=8):
    """`n` replies per question, batched, sampled the way the trainer samples."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    model.eval()
    tokenizer.padding_side = "left"
    out: list[list[str]] = []
    texts = [
        tokenizer.apply_chat_template(
            messages_for(q, team), tokenize=False, add_generation_prompt=True
        )
        for q in questions
    ]
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append(decoded[i * n : (i + 1) * n])
    model.train()
    return out


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=120 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    team: bool,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 40,
    prompts_per_step: int = 12,
    generations: int = 4,
    learning_rate: float = 1e-4,
    max_completion_length: int = 512,
    lora_rank: int = 32,
    eval_samples: int = 4,
    eval_base: bool = False,
    train_seed: int = 17,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again under the
    arm's own prompt and, for the team arm, under the plain one too."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import EVAL_RUNS, graded_rows, make_reward, messages_for

    import whileai as wai
    from whileai.simulations.training import TrainerCallback, training_run

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    config = {
        "arm": arm,
        "team_prompt": team,
        "base_model": base_model,
        "steps": steps,
        "generations": generations,
        "prompts_per_step": prompts_per_step,
        "learning_rate": learning_rate,
        "beta": 0.0,
        "scale_rewards": False,
        "lora_rank": lora_rank,
        "max_completion_length": max_completion_length,
        "train_prompts": len(train_tasks),
        "holdout": len(holdout),
        "train_seed": train_seed,
        "gpu": DEFAULT_GPU,
        "reward": "binary MathEqual against the GSM8K gold; talk is never rewarded",
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

    questions = [t["question"] for t in holdout]

    def evaluate(m, team_prompt: bool) -> list[dict]:
        replies = _sample(
            m,
            tokenizer,
            questions,
            team=team_prompt,
            n=eval_samples,
            max_new_tokens=max_completion_length,
        )
        return graded_rows(holdout, replies)

    # The base is evaluated EVAL_RUNS times under the plain prompt, not once.
    # The spread across those re-runs is the eval's own noise, and a delta
    # smaller than it is not a result (Lambert 2025, chapter Evaluation). The
    # same container also asks the untrained base to talk: the team prompt
    # with no training, the paper's own setting on one model.
    base_runs: list[list[dict]] = []
    base_team: list[dict] = []
    if eval_base:
        for i in range(EVAL_RUNS):
            base_runs.append(evaluate(model, False))
            print(f"base run {i + 1}/{EVAL_RUNS}: {wai.pass_at(base_runs[-1])}")
        base_team = evaluate(model, True)
        print(f"base, team prompt: {wai.pass_at(base_team)}")

    dataset = Dataset.from_list(
        [{"prompt": messages_for(t["question"], team), "answer": t["answer"]} for t in train_tasks]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=generations,
        per_device_train_batch_size=generations,
        gradient_accumulation_steps=prompts_per_step,
        learning_rate=learning_rate,
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        # Reward minus the group mean, no std division, in both arms.
        scale_rewards=False,
        # The same token budget in both arms: a conversation has to fit in
        # what a monologue gets, so the recipe buys nothing with length.
        max_completion_length=max_completion_length,
        max_prompt_length=512,
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
    # TRL 0.19.1 builds the LoRA adapter before it applies GRPOConfig.seed, so
    # seed here or the arm that ran the base evals draws a different adapter.
    set_seed(train_seed)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward(last_batch)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    if run is not None:
        trainer.add_callback(TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise
    train_minutes = (time.time() - started) / 60.0
    history = trainer.state.log_history
    trace = {
        key: [h[key] for h in history if key in h]
        for key in ("reward", "completions/mean_length", "completions/clipped_ratio")
    }

    after_rows = evaluate(trainer.model, team)
    print(f"{arm}: {wai.pass_at(after_rows)}")
    # The memory test: take the team prompt away. What is left is what the
    # weights learned, not what the prompt asked for.
    plain_rows = evaluate(trainer.model, False) if team else after_rows
    if team:
        print(f"{arm}, plain prompt: {wai.pass_at(plain_rows)}")

    scan = wai.hack_scan(last_batch) if last_batch else {}
    hack_top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
    hack_scan_top = hack_top.get("name", "") if isinstance(hack_top, dict) else str(hack_top)
    print(f"{arm} hack scan: top feature {hack_scan_top or 'none above the floor'}")

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
        "team_prompt": team,
        "pass_at_1": wai.pass_at(after_rows).pass_at_1,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    return {
        "arm": arm,
        "base_runs": base_runs,
        "base_team": base_team,
        "after_rows": after_rows,
        "plain_rows": plain_rows,
        "last_batch": last_batch,
        "gpu_minutes": gpu_minutes,
        "train_minutes": train_minutes,
        "steps": steps,
        "hack_scan_top": hack_scan_top,
        "trace": trace,
        "run_url": summary.get("run_url", ""),
    }


# --------------------------------------------------------------------------
# Local: data, orchestration, results.json.
# --------------------------------------------------------------------------


def data(seed: int, n_train: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    """GSM8K. Train prompts from the train split, holdout from the test split,
    so the two are disjoint by construction, not by a shuffle."""
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    train_tasks = [
        {"question": r["question"], "answer": r["answer"], "scenario_id": f"train-{i}"}
        for i, r in enumerate(train.select(range(n_train)))
    ]
    holdout = [
        {"question": r["question"], "answer": r["answer"], "scenario_id": f"test-{i}"}
        for i, r in enumerate(test.select(range(n_holdout)))
    ]
    return train_tasks, holdout


def summarize(rows: list[dict]) -> dict:
    import whileai as wai

    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
        "talk_rate": round(talk_rate(rows), 3),
        "turns": round(mean_turns(rows), 2),
        "chars": round(mean_length(rows)),
    }


def selftest() -> None:
    """The talk counter and the grader, on the CPU. No GPU, no key, no model."""
    team = (
        "Solver: 3 boxes of 4 is 12, plus 5 is 17.\n"
        "Checker: 3 x 4 = 12, 12 + 5 = 17. Agreed.\n"
        "Lead: The answer is \\boxed{17}."
    )
    assert turns_of(team) == ["Solver", "Checker", "Lead"] and talks(team)
    # Markdown the model likes to add around a name still counts as a turn.
    assert turns_of("**Solver:** a\n- Checker: b\n### Lead: c") == ["Solver", "Checker", "Lead"]
    # A monologue signed by Lead, or a team with no Checker, is not talk.
    assert not talks("Solver: a\nSolver: b\nLead: \\boxed{1}")
    assert not talks("3 x 4 = 12, so \\boxed{12}")
    # A speaker named mid-sentence is not a turn.
    assert turns_of("Then the Checker: said so") == []
    assert talk_rate([{"final_text": team}, {"final_text": "\\boxed{1}"}]) == 0.5

    # The prompt is the only thing the arms do differently.
    assert messages_for("q")[0]["content"] == SYSTEM
    assert messages_for("q", team=True)[0]["content"] == SYSTEM_TEAM
    assert messages_for("q", True)[1] == messages_for("q")[1]

    # The grader is a program, not a judge, and it reads the team's answer.
    assert outcome_of(team, "17") == 1.0
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("talk counter: 3+ turns, 2+ speakers, Checker among them")
    print("grader: MathEqual, decided by Math-Verify")
    print("selftest ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0, help="data shuffle seed")
    ap.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=[17, 18],
        help="training seeds per arm; two or more lets the verdict resolve",
    )
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=512, help="completion cap, both arms")
    ap.add_argument("--selftest", action="store_true", help="the talk counter, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]
    team = {"baseline": False, "recipe": True}

    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(
        {
            "recipe": HERE.name,
            "title": "Team talk: train a model to reason as a team talking to itself",
            "paper": "https://arxiv.org/abs/2609.22682",
            "book": BOOK,
            "base_model": BASE_MODEL,
            "metric": METRIC,
            "n_holdout": len(holdout),
            "k": args.k,
            "gpu": DEFAULT_GPU,
            "whileai": version("whileai"),
        }
    )
    results.setdefault("arms", {})
    checks = results.setdefault("checks", {})
    checks["decontaminated_dropped"] = int(decon.get("n_contaminated", 0))
    checks["seed"] = args.seed
    checks["max_completion_tokens"] = args.max_tokens
    checks.setdefault("length_after", {})
    after: dict[str, list[list[dict]]] = {arm: [] for arm in arms}
    plain: dict[str, list[list[dict]]] = {arm: [] for arm in arms}
    traces: dict[str, list[dict]] = {arm: [] for arm in arms}
    minutes: dict[str, list[float]] = {arm: [] for arm in arms}
    run_std = 0.0
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""

    jobs = [(arm, s) for s in args.train_seeds for arm in arms]
    with modal.enable_output(), app.run():
        # Every (arm, seed) is its own container, so they run side by side.
        calls = [
            run_arm.spawn(
                arm,
                team[arm],
                train_tasks,
                holdout,
                f"team-talk-{arm}-s{s}-{date.today().isoformat()}",
                steps=args.steps,
                max_completion_length=args.max_tokens,
                eval_samples=args.k,
                eval_base=(i == 0),
                train_seed=s,
            )
            for i, (arm, s) in enumerate(jobs)
        ]
        outs = [c.get() for c in calls]

    for (arm, _), out in zip(jobs, outs):
        gpu_minutes += out["gpu_minutes"]
        minutes[arm].append(out["train_minutes"])
        run_url = out["run_url"] or run_url
        if out["base_runs"]:
            base_runs = out["base_runs"]
            noise = wai.eval_variance(*base_runs)
            run_std = float(noise["run_std"])
            print(f"eval noise over {len(base_runs)} base runs: run_std {run_std:.4f}")
            results["arms"]["base"] = {**summarize(base_runs[0]), "steps": 0, "gpu_minutes": 0}
            results["arms"]["base_team_prompt"] = {
                **summarize(out["base_team"]),
                "steps": 0,
                "gpu_minutes": 0,
            }
            checks["run_std"] = run_std
            checks["run_std_runs"] = int(noise["n_runs"])
            checks["length_before"] = mean_length(base_runs[0])
        after[arm].append(out["after_rows"])
        plain[arm].append(out["plain_rows"])
        traces[arm].append(out["trace"])
        # Talk in the last training batch: does the conversation survive a
        # reward that never pays for it?
        traces[arm][-1]["talk_rate_last_batch"] = round(talk_rate(out["last_batch"]), 3)
        checks["hack_scan_top"] = out["hack_scan_top"]
        checks["length_after"][arm] = mean_length(out["after_rows"])

    for arm in arms:
        pooled = [r for rows in after[arm] for r in rows]
        results["arms"][arm] = {
            **summarize(after[arm][0]),
            "per_seed": [summarize(rows)["score"] for rows in after[arm]],
            "pooled_score": summarize(pooled)["score"],
            "steps": args.steps,
            "gpu_minutes": round(statistics.fmean(minutes[arm]), 1),
        }
        results.setdefault("train_trace", {})[arm] = traces[arm]
    if "recipe" in plain:
        results["arms"]["recipe_plain_prompt"] = {
            **summarize(plain["recipe"][0]),
            "per_seed": [summarize(rows)["score"] for rows in plain["recipe"]],
            "steps": args.steps,
        }

    def paired(before: list[list[dict]], treated: list[list[dict]]) -> dict:
        d = wai.compare(
            before[0],
            treated[0],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks.get("run_std_runs") or EVAL_RUNS),
            train_runs={"before": before, "after": treated},
            proxy=PROXY,
        )
        verdict = d["target_verdict"]
        return {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": verdict if verdict in ("moved", "flat", "unresolved") else "flat",
            "over_optimized": bool(d.get("over_optimized")),
        }

    if "baseline" in after and "recipe" in after:
        main_delta = paired(after["baseline"], after["recipe"])
        checks["over_optimized"] = main_delta.pop("over_optimized")
        results["delta"] = main_delta
        # The memory test as a delta: both arms under the plain prompt.
        remember = paired(plain["baseline"], plain["recipe"])
        remember.pop("over_optimized")
        results["delta_plain_prompt"] = remember
        checks["train_seeds"] = {arm: len(after[arm]) for arm in after}
        results["verified"] = date.today().isoformat()
        results.pop("partial_run", None)
        print("as trained:", results["delta"])
        print("plain prompt:", results["delta_plain_prompt"])
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"
        print(f"one arm only ({', '.join(arms)}): delta and verified left as they were")

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "train_trace"}, indent=2))


if __name__ == "__main__":
    main()
