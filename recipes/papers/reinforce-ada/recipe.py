"""Reinforce-Ada: keep sampling a prompt until its group can teach something.

    python recipe.py                      # both arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the sampler and the advantage, offline, no GPU

GRPO draws the same small group for every prompt. With four rollouts and a
binary reward, a prompt the model solves one time in ten comes back
all-wrong about two times in three: every advantage in the group is zero and
the prompt teaches nothing that step. The paper's point is that this is
undersampling, not a prompt the model cannot learn from.

So sample in rounds and stop per prompt (Algorithm "Reinforce-Ada-Seq",
balanced exit, the authors' verl defaults): draw 8 rollouts for every
prompt still active, retire a prompt once it has 2 right and 2 wrong, and
stop after 4 rounds (32 rollouts at most). Then downsample each prompt to the
same 4 rollouts GRPO would train on, 2 right and 2 wrong when the pool has
them, and take the advantage against the pass rate of the whole pool rather
than of the 4 kept (``global_stat_est``, no std division). The update the
optimizer sees is the same size in both arms; only which 4 rollouts it sees,
and the baseline they are measured against, change.

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out
  2. run_arm():   TRL GRPOTrainer + LoRA on Modal, one arm per call
  3. evaluate():  same holdout, k samples per task, graded by MathEqual
  4. results.json + the paired delta (wai.compare) on the run page
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Reinforcement Learning"  # the group baseline and what a zero-variance group contributes
# The training reward here *is* the target: both are the same binary check
# against the GSM8K gold, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

# The authors' defaults (RLHFlow/Reinforce-Ada, verl/trainer/config/algorithm.py
# and scripts/run_reinforce_ada.sh): the group the update trains on, the
# rollouts drawn per active prompt per round, the round cap, and the reward
# above which a rollout counts as right.
N_FINAL = 4
ROUND_REPEAT = 8
MAX_ROUNDS = 4
POSITIVE_THRESHOLD = 0.7

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."


# --------------------------------------------------------------------------
# The sampler. Pure functions, no torch: `--selftest` runs them against a
# coin-flip model, and the Modal container imports this same file.
# --------------------------------------------------------------------------


def messages_for(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


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


def balanced_pick(
    pos: Sequence[Any], neg: Sequence[Any], n: int = N_FINAL
) -> tuple[list[Any], list[Any]]:
    """The downsample: n // 2 right and the rest wrong, topped up from
    whichever side has spare when the other runs short. This is
    `downsample_cache` in the reference trainer, first-come order kept."""
    take_pos = min(n // 2, len(pos))
    take_neg = min(n - take_pos, len(neg))
    short = n - take_pos - take_neg
    if short and len(pos) > take_pos:
        take_pos += min(len(pos) - take_pos, short)
    elif short and len(neg) > take_neg:
        take_neg += min(len(neg) - take_neg, short)
    return list(pos[:take_pos]), list(neg[:take_neg])


def adaptive_rounds(
    n_prompts: int,
    sample_round: Callable[[list[int], int], list[tuple[int, float, Any]]],
    *,
    n: int = N_FINAL,
    round_repeat: int = ROUND_REPEAT,
    max_rounds: int = MAX_ROUNDS,
    threshold: float = POSITIVE_THRESHOLD,
) -> tuple[list[dict], int]:
    """THE ONE CHANGE: Reinforce-Ada-Seq with the balanced exit.

    `sample_round(active, round_repeat)` draws `round_repeat` rollouts for
    each prompt index in `active` and returns `(prompt, reward, item)` for
    every one. A prompt retires once it holds n // 2 right and n - n // 2
    wrong; whatever is still active after `max_rounds` keeps what it has.

    Returns one group per prompt, each exactly `n` long, and the rounds run.
    Every group carries `advantages = reward - pass_rate`, where `pass_rate`
    is right / seen over the whole pool, not over the `n` kept: the kept
    group is balanced on purpose, so its own mean is 0.5 by construction and
    says nothing about how hard the prompt is.
    """
    need_pos, need_neg = n // 2, n - n // 2
    pos: list[list[tuple[float, Any]]] = [[] for _ in range(n_prompts)]
    neg: list[list[tuple[float, Any]]] = [[] for _ in range(n_prompts)]
    active = list(range(n_prompts))
    rounds = 0
    while active and rounds < max_rounds:
        rounds += 1
        for p, reward, item in sample_round(active, round_repeat):
            (pos if reward > threshold else neg)[p].append((reward, item))
        active = [p for p in active if len(pos[p]) < need_pos or len(neg[p]) < need_neg]

    groups = []
    for p in range(n_prompts):
        seen = len(pos[p]) + len(neg[p])
        if seen < n:
            raise ValueError(
                f"prompt {p} drew {seen} rollouts, fewer than the group of {n}; "
                f"round_repeat ({round_repeat}) must be at least n"
            )
        pass_rate = len(pos[p]) / seen
        kept_pos, kept_neg = balanced_pick(pos[p], neg[p], n)
        kept = kept_pos + kept_neg
        groups.append(
            {
                "items": [item for _, item in kept],
                "rewards": [r for r, _ in kept],
                "advantages": [r - pass_rate for r, _ in kept],
                "pass_rate": pass_rate,
                "seen": seen,
                "retired": len(pos[p]) >= need_pos and len(neg[p]) >= need_neg,
            }
        )
    return groups, rounds


def uniform_group(rewards: Sequence[float]) -> list[float]:
    """The baseline's advantage: reward minus the group's own mean, no std
    division (both arms run `scale_rewards=False`, the reference script's
    `norm_adv_by_std_in_grpo=False`)."""
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def make_reward(recorder: list[dict]):
    """Binary outcome, a program against the public GSM8K gold. Both arms use
    this untouched: the paper changes the sampler, not the reward.

    `recorder` is refilled with the batch it just graded, in batch order. The
    recipe trainer reads the rewards back from it after every round, and
    `hack_scan` reads the last one after training (Lambert 2025, chapter
    Over-optimization).
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


def reinforce_ada_trainer(base_cls):
    """Build the trainer subclass. Takes `GRPOTrainer` as an argument so this
    module imports without torch, which is what lets `--selftest` run locally.
    """

    import torch

    class ReinforceAdaTrainer(base_cls):  # type: ignore[valid-type,misc]
        """GRPOTrainer whose generation step is Reinforce-Ada-Seq.

        One override. TRL hands `_generate_and_score_completions` a batch of
        prompts, each repeated `num_generations` (= N_FINAL) times in a row.
        The override calls the parent once per round on the prompts still
        active, repeated ROUND_REPEAT times, so generation, reward and
        logging are TRL's own. It then keeps N_FINAL rollouts per prompt,
        re-pads them into one batch, and writes the pool-baseline advantage.
        The batch goes back out the same shape TRL asked for, so the shuffle,
        the gradient-accumulation split and the loss are untouched.

        `adaptive=False` is the baseline: the class is the same, the flag is
        the difference.
        """

        def __init__(self, *args, adaptive: bool = True, recorder: list[dict], **kw):
            super().__init__(*args, **kw)
            self.adaptive = adaptive
            self.recorder = recorder
            if adaptive and (self.num_iterations != 1 or self.beta != 0.0):
                raise RuntimeError(
                    "the recipe arm rebuilds the batch after generation and does not carry "
                    "old or reference log-probs through it: run with num_iterations=1 and beta=0"
                )
            if adaptive and self.num_generations != N_FINAL:
                raise RuntimeError(f"num_generations must be N_FINAL ({N_FINAL})")

        def _generate_and_score_completions(self, inputs):
            if not self.adaptive:
                return super()._generate_and_score_completions(inputs)
            n = self.num_generations
            unique = inputs[::n]
            parent = super()._generate_and_score_completions

            def sample_round(active: list[int], repeat: int):
                batch = [unique[p] for p in active for _ in range(repeat)]
                out = parent(batch)
                rewards = [row["reward"] for row in self.recorder]
                if len(rewards) != len(batch):
                    raise RuntimeError("reward recorder is out of step with the batch")
                rows = []
                for j in range(len(batch)):
                    keep = out["prompt_mask"][j].bool()
                    item = (
                        out["prompt_ids"][j][keep],
                        out["completion_ids"][j],
                        out["completion_mask"][j],
                    )
                    rows.append((active[j // repeat], rewards[j], item))
                return rows

            groups, rounds = adaptive_rounds(len(unique), sample_round, n=n)
            items = [item for g in groups for item in g["items"]]
            advantages = [a for g in groups for a in g["advantages"]]

            pad = self.processing_class.pad_token_id
            device = self.accelerator.device
            p_len = max(int(p.numel()) for p, _, _ in items)
            c_len = max(int(c.numel()) for _, c, _ in items)
            prompt_ids = torch.full((len(items), p_len), pad, dtype=torch.long, device=device)
            prompt_mask = torch.zeros((len(items), p_len), dtype=torch.long, device=device)
            completion_ids = torch.full((len(items), c_len), pad, dtype=torch.long, device=device)
            completion_mask = torch.zeros((len(items), c_len), dtype=torch.long, device=device)
            for i, (p, c, m) in enumerate(items):
                prompt_ids[i, p_len - p.numel() :] = p  # prompts are left-padded
                prompt_mask[i, p_len - p.numel() :] = 1
                completion_ids[i, : c.numel()] = c  # completions right-padded
                completion_mask[i, : m.numel()] = m

            metrics = self._metrics["train"]
            metrics["ada/rounds"].append(float(rounds))
            metrics["ada/rollouts_per_prompt"].append(statistics.fmean(g["seen"] for g in groups))
            metrics["ada/retired_frac"].append(statistics.fmean(g["retired"] for g in groups))
            metrics["ada/pool_pass_rate"].append(statistics.fmean(g["pass_rate"] for g in groups))
            metrics["ada/zero_signal_frac"].append(
                statistics.fmean(all(a == 0 for a in g["advantages"]) for g in groups)
            )
            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
                "old_per_token_logps": None,
                "ref_per_token_logps": None,
            }

    return ReinforceAdaTrainer


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/04-train/grpo/train_modal.py.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-reinforce-ada")

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


def _sample(model, tokenizer, questions, *, n, max_new_tokens, batch=8):
    """`n` replies per question, batched, sampled the way the trainer samples."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    model.eval()
    tokenizer.padding_side = "left"
    out: list[list[str]] = []
    texts = [
        tokenizer.apply_chat_template(messages_for(q), tokenize=False, add_generation_prompt=True)
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
    timeout=90 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    adaptive: bool,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 40,
    prompts_per_step: int = 12,
    learning_rate: float = 1e-4,
    max_completion_length: int = 256,
    lora_rank: int = 32,
    eval_samples: int = 4,
    eval_base: bool = False,
    train_seed: int = 17,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        EVAL_RUNS,
        MAX_ROUNDS,
        N_FINAL,
        ROUND_REPEAT,
        graded_rows,
        make_reward,
        mean_length,
        messages_for,
        reinforce_ada_trainer,
    )

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
        "reinforce_ada": adaptive,
        "base_model": base_model,
        "steps": steps,
        "update_group": N_FINAL,
        "round_repeat": ROUND_REPEAT if adaptive else None,
        "max_rounds": MAX_ROUNDS if adaptive else None,
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
        "reward": "binary MathEqual against the GSM8K gold",
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
    # The base is evaluated EVAL_RUNS times, not once. The spread across those
    # re-runs is the eval's own noise, and a delta smaller than it is not a
    # result (Lambert 2025, chapter Evaluation). Only the first arm pays for this.
    base_runs = []
    if eval_base:
        for i in range(EVAL_RUNS):
            replies = _sample(
                model, tokenizer, questions, n=eval_samples, max_new_tokens=max_completion_length
            )
            rows = graded_rows(holdout, replies)
            base_runs.append(rows)
            print(f"base run {i + 1}/{EVAL_RUNS}: {wai.pass_at(rows)}")

    dataset = Dataset.from_list(
        [{"prompt": messages_for(t["question"]), "answer": t["answer"]} for t in train_tasks]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        # Both arms train on N_FINAL rollouts per prompt. The baseline draws
        # exactly that many; the recipe draws up to ROUND_REPEAT * MAX_ROUNDS
        # and keeps N_FINAL, so the backward pass is the same size.
        num_generations=N_FINAL,
        per_device_train_batch_size=N_FINAL,
        gradient_accumulation_steps=prompts_per_step,
        learning_rate=learning_rate,
        # On-policy: one update per batch of rollouts, as the recipe arm
        # requires (it rebuilds the batch and carries no old log-probs).
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        # The reference run divides by nothing (norm_adv_by_std_in_grpo=False):
        # reward minus a baseline, in both arms.
        scale_rewards=False,
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
    # seed here or the arm that ran the base evals draws a different adapter
    # (the adaptive-clip recipe measured this the hard way).
    set_seed(train_seed)
    trainer = reinforce_ada_trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[make_reward(last_batch)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        adaptive=adaptive,
        recorder=last_batch,
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

    # What the sampler did, step by step, read back off TRL's log.
    history = trainer.state.log_history
    trace = {
        key: [h[key] for h in history if key in h]
        for key in (
            "frac_reward_zero_std",
            "ada/rollouts_per_prompt",
            "ada/zero_signal_frac",
            "ada/pool_pass_rate",
            "ada/retired_frac",
        )
    }

    replies = _sample(
        trainer.model, tokenizer, questions, n=eval_samples, max_new_tokens=max_completion_length
    )
    after_rows = graded_rows(holdout, replies)
    after = wai.pass_at(after_rows)
    print(f"{arm}: {after}")

    # What the reward actually paid for in the last graded batch (chapter Over-optimization).
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
        "reinforce_ada": adaptive,
        "pass_at_1": after.pass_at_1,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    return {
        "arm": arm,
        "base_runs": base_runs,
        "after_rows": after_rows,
        "gpu_minutes": gpu_minutes,
        "train_minutes": train_minutes,
        "steps": steps,
        "length_after": mean_length(after_rows),
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
    }


def _coin_sampler(rates: Sequence[float], rng: random.Random):
    def sample_round(active: list[int], repeat: int):
        return [
            (p, 1.0 if rng.random() < rates[p] else 0.0, f"{p}:{i}")
            for p in active
            for i in range(repeat)
        ]

    return sample_round


def selftest() -> None:
    """The sampler and the advantage against a coin-flip model, on the CPU.
    No GPU, no key, no model download."""
    # The downsample: balanced when it can be, topped up when it cannot.
    assert balanced_pick("ab", "xyz") == (["a", "b"], ["x", "y"])
    assert balanced_pick("a", "wxyz") == (["a"], ["w", "x", "y"])
    assert balanced_pick("abcd", "") == (["a", "b", "c", "d"], [])
    assert balanced_pick("abc", "x") == (["a", "b", "c"], ["x"])

    # A prompt that retires in round one keeps 2 right and 2 wrong, measured
    # against the pool's pass rate, not the kept group's 0.5.
    def fixed(active, repeat):
        return [(p, 1.0 if i < 3 else 0.0, i) for p in active for i in range(repeat)]

    groups, rounds = adaptive_rounds(1, fixed)
    g = groups[0]
    assert rounds == 1 and g["seen"] == 8 and g["retired"]
    assert g["rewards"] == [1.0, 1.0, 0.0, 0.0]
    assert g["advantages"] == [1 - 3 / 8, 1 - 3 / 8, -3 / 8, -3 / 8]

    # An all-wrong prompt runs every round and still hands back N_FINAL rows,
    # all at zero advantage: the gradient it would have had under GRPO.
    groups, rounds = adaptive_rounds(1, lambda a, r: [(p, 0.0, i) for p in a for i in range(r)])
    assert rounds == MAX_ROUNDS and groups[0]["seen"] == ROUND_REPEAT * MAX_ROUNDS
    assert groups[0]["advantages"] == [0.0] * N_FINAL and not groups[0]["retired"]

    # The claim, on coins: a prompt solved one time in ten. Uniform GRPO at
    # 4 rollouts gets a group that splits 1 - 0.9^4 - 0.1^4 of the time; the
    # adaptive sampler keeps drawing until it splits or hits 32.
    rng = random.Random(0)
    trials = 4000
    rates = [0.1, 0.5, 0.9]
    split_uniform = dict.fromkeys(rates, 0)
    split_ada = dict.fromkeys(rates, 0)
    drawn = dict.fromkeys(rates, 0)
    for _ in range(trials):
        for rate in rates:
            coins = [1.0 if rng.random() < rate else 0.0 for _ in range(N_FINAL)]
            split_uniform[rate] += any(a != 0 for a in uniform_group(coins))
        groups, _ = adaptive_rounds(len(rates), _coin_sampler(rates, rng))
        for rate, g in zip(rates, groups):
            split_ada[rate] += any(a != 0 for a in g["advantages"])
            drawn[rate] += g["seen"]
    print("share of prompts whose group carries a gradient (coin model, 4000 trials):")
    for rate in rates:
        u, a = split_uniform[rate] / trials, split_ada[rate] / trials
        print(
            f"  p={rate}: GRPO n=4 {u:.2f}, Reinforce-Ada {a:.2f}, {drawn[rate] / trials:.1f} drawn"
        )
    assert split_ada[0.1] / trials > 0.9 > split_uniform[0.1] / trials
    assert abs(split_uniform[0.1] / trials - (1 - 0.9**4 - 0.1**4)) < 0.03

    # The grader is a program, not a judge.
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
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
        default=[17],
        help="training seeds per arm; two or more lets the verdict resolve",
    )
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=120)
    ap.add_argument("--prompts-per-step", type=int, default=12)
    ap.add_argument("--selftest", action="store_true", help="the sampler, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    # GSM8K's train and test splits are already disjoint, so this should drop
    # nothing. It runs anyway, and the count goes in the Checks table, because
    # "should" is not a measurement (Lambert 2025, chapter Evaluation).
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]
    adaptive = {"baseline": False, "recipe": True}

    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(
        {
            "recipe": HERE.name,
            "title": "Reinforce-Ada: keep sampling a prompt until its group can teach something",
            "paper": "https://arxiv.org/abs/2510.04996",
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
    checks.setdefault("length_after", {})
    seed_rows: dict[str, list[list[dict]]] = {arm: [] for arm in arms}
    traces: dict[str, list[dict]] = {arm: [] for arm in arms}
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
                adaptive[arm],
                train_tasks,
                holdout,
                f"reinforce-ada-{arm}-s{s}-{date.today().isoformat()}",
                steps=args.steps,
                prompts_per_step=args.prompts_per_step,
                eval_samples=args.k,
                eval_base=(i == 0),
                train_seed=s,
            )
            for i, (arm, s) in enumerate(jobs)
        ]
        outs = [c.get() for c in calls]

    minutes: dict[str, list[float]] = {arm: [] for arm in arms}
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
            checks["run_std"] = run_std
            checks["run_std_runs"] = int(noise["n_runs"])
            checks["length_before"] = mean_length(base_runs[0])
        seed_rows[arm].append(out["after_rows"])
        traces[arm].append(out["trace"])
        checks["hack_scan_top"] = out["hack_scan_top"]
        checks["length_after"][arm] = out["length_after"]

    for arm in arms:
        pooled = [r for rows in seed_rows[arm] for r in rows]
        per_seed = [summarize(rows)["score"] for rows in seed_rows[arm]]
        results["arms"][arm] = {
            **summarize(seed_rows[arm][0]),
            "per_seed": per_seed,
            "pooled_score": summarize(pooled)["score"],
            "steps": args.steps,
            "gpu_minutes": round(statistics.fmean(minutes[arm]), 1),
        }
        results.setdefault("sampler", {})[arm] = traces[arm]

    if "baseline" in seed_rows and "recipe" in seed_rows:
        d = wai.compare(
            seed_rows["baseline"][0],
            seed_rows["recipe"][0],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks.get("run_std_runs") or EVAL_RUNS),
            train_runs={"before": seed_rows["baseline"], "after": seed_rows["recipe"]},
            proxy=PROXY,
        )
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": d["target_verdict"]
            if d["target_verdict"] in ("moved", "flat", "unresolved")
            else "flat",
        }
        checks["train_seeds"] = {arm: len(seed_rows[arm]) for arm in seed_rows}
        checks["over_optimized"] = bool(d.get("over_optimized"))
        results["verified"] = date.today().isoformat()
        results.pop("partial_run", None)
        print(d)
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"
        print(f"one arm only ({', '.join(arms)}): delta and verified left as they were")

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "sampler"}, indent=2))


if __name__ == "__main__":
    main()
