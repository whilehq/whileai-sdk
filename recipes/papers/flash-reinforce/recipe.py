"""FlashREINFORCE: one stale rollout per prompt, corrected, gated and length-normalized.

    python recipe.py                      # both arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the reward, the lag bookkeeping and the loss, offline

A production trace arrives one rollout per prompt, written by a policy that
is already a few updates behind the one being trained. GRPO cannot use it
(no group to take a baseline over) and plain REINFORCE on it is off-policy
without saying so. FlashREINFORCE (Hu et al. 2026) takes the baseline from
the batch mean, corrects every token by the ratio of the learner to the
sampler, masks whole any trajectory whose mean sampled-action KL has moved
past a trust region, and weights every kept trajectory the same whatever
its length. `wai.FlashReinforce().update(batch)` is that rule; this file
runs it on a real model against the paper's own uncorrected ablation.

The regime is the whole point, so both arms sample from a *stale* copy of
the policy's adapter, refreshed only every ``LAG`` optimizer steps, and the
next batch is drawn before the current update lands, the way an
asynchronous trainer overlaps generation with training. A batch is then 1
to ``LAG`` updates old (the first is 0). Both arms see the same lag; only
the coefficient that multiplies each token's log-probability differs:

    baseline   A_i / (total tokens in the batch)          no ratio, no gate, token mean
    recipe     m_i * A_i * rho_it / (T_i * B)             FlashReinforce.update, Eq. 10

with ``A_i = R_i - mean R``, ``rho`` the learner/sampler ratio, ``m_i`` the
trust mask and ``T_i`` the trajectory's own length. The loss in both arms
is ``-(coefficients * logprobs).sum()`` with the coefficients held constant.

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out, decontaminated
  2. run_arm():   HF generate + PEFT LoRA + AdamW on Modal, one arm per call
  3. evaluate():  same holdout, k samples per task, graded by MathEqual
  4. results.json + the paired delta (wai.compare)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

# Inside the Modal container the image carries this checkout's ``whileai``
# next to the released wheel it installed for the dependencies. The wheel on
# the index does not have ``FlashReinforce`` yet, so the checkout wins the
# import. Locally the directory does not exist and this line does nothing.
if os.path.isdir("/root/whileai_local"):
    sys.path.insert(0, "/root/whileai_local")

import modal

import whileai as wai
from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
# The checkout's package, three levels up; in the container this file sits at
# /root/recipe.py and the package at the mount below.
WHILEAI_SRC = (
    HERE.parents[2] / "whileai" if len(HERE.parents) > 2 else Path("/root/whileai_local/whileai")
)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Policy Gradient Algorithms"  # the baseline and the importance ratio, Lambert 2025
# The training reward *is* the target: both are the same binary check
# against the GSM8K gold, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

# The production-trace regime. The sampler is a frozen copy of the policy's
# adapter refreshed every LAG optimizer steps, so a batch was written by a
# policy 1 to LAG updates old (mean about 2.5); the paper measures about 4
# on its 1.5B run (FLASH_REINFORCE_OFF_POLICY_STEPS is the bound it is built
# to absorb). One rollout per prompt, PROMPTS_PER_STEP prompts per optimizer
# step, one full-batch step per batch and the batch is discarded (Sec. 3.1).
LAG = 4
PROMPTS_PER_STEP = 64
MAX_NEW_TOKENS = 256
TRAIN_TEMPERATURE = 1.0  # FLASH_REINFORCE_TEMPERATURE: untruncated, so the ratio has support
EVAL_TEMPERATURE = 0.7
# 1e-5 on the adapter. The paper's 1e-6 (FLASH_REINFORCE_LEARNING_RATE) is
# an AdamW step on full weights; an adapter of rank 16 takes a larger one.
# Shared with the SAO and BPCO recipes so the three single-rollout methods
# compare on one protocol.
LEARNING_RATE = 1e-5
GRAD_CLIP = 1.0  # Hu et al. 2026, Tables 9, 11, 12, 13
TRAIN_SEED = 17  # set right before the adapter is built, so both arms draw one init

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."


# --------------------------------------------------------------------------
# Pure parts: the reward, the lag bookkeeping and the baseline's coefficient.
# No torch, so `--selftest` runs them locally and the container imports the
# same file.
# --------------------------------------------------------------------------


def messages_for(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def gold_of(answer: str) -> str:
    """GSM8K ships the worked solution then `#### 72`. The gold is the tail."""
    return answer.split("####")[-1].strip().replace(",", "")


def outcome_of(text: str, gold: str) -> float:
    """1.0 when the final number matches the gold, else 0.0. A program, not a
    judge: `MathEqual` is sympy with a numeric and string fallback."""
    from whileai.simulations.verify import MathEqual

    row = {
        "prompt": "",
        "final_text": text,
        "privileged": {"reference": gold},
        "scenario_id": "x",
        "rollout_index": 0,
    }
    return 1.0 if MathEqual()(row).get("reward") == 1 else 0.0


class StaleSampler:
    """The bookkeeping of a sampler that lags the policy.

    ``sampler_version`` is how many optimizer updates the frozen sampler
    weights had seen when they were copied; ``policy_version`` is how many
    the policy has taken. A batch is stamped with ``stamp()`` when it is
    generated, and ``lag_of(stamp)`` at update time is how many updates old
    it is. ``updated()`` counts one optimizer step and says whether the
    weights are due for a refresh (every ``lag`` steps).

    The weights themselves live in the container; this object only counts.
    """

    def __init__(self, lag: int) -> None:
        if lag < 1:
            raise ValueError(f"lag must be 1 or more optimizer steps; got {lag}")
        self.lag = lag
        self.sampler_version = 0
        self.policy_version = 0

    def stamp(self) -> int:
        return self.sampler_version

    def lag_of(self, stamp: int) -> int:
        return self.policy_version - stamp

    def updated(self) -> bool:
        self.policy_version += 1
        refresh = self.policy_version % self.lag == 0
        if refresh:
            self.sampler_version = self.policy_version
        return refresh


def lag_sequence(steps: int, lag: int) -> list[int]:
    """The lag of every batch under the pipelined loop `run_arm` runs: the
    batch for step s + 1 is generated before update s lands, then the
    sampler is refreshed when the step count is a multiple of `lag`.

    Step 1 is on-policy (nothing has moved yet); after that the lag cycles
    1, 2, ..., `lag`. The mean over 40 steps at lag 4 is 2.4.
    """
    sampler = StaleSampler(lag)
    queue = [sampler.stamp()]
    lags: list[int] = []
    for step in range(1, steps + 1):
        stamp = queue.pop(0)
        if step < steps:
            queue.append(sampler.stamp())
        lags.append(sampler.lag_of(stamp))
        sampler.updated()
    return lags


def uncorrected_coefficients(batch: list[dict]) -> list[list[float]]:
    """THE BASELINE: the paper's uncorrected ablation. The same stale
    trajectories, the same batch-mean advantage, and a token-mean loss: no
    importance ratio, no trust gate, no 1/T_i. Every token of trajectory i
    gets ``A_i / (total tokens in the batch)``, so a long trajectory weighs
    more than a short one and a stale token is trained as if it were fresh.
    """
    if not batch:
        raise ValueError("uncorrected_coefficients: the batch is empty")
    rewards = [float(t["reward"]) for t in batch]
    baseline = sum(rewards) / len(rewards)
    lengths = [len(t["logprobs"]) for t in batch]
    total = sum(lengths)
    return [[(r - baseline) / total] * n for r, n in zip(rewards, lengths)]


def loss_of(coefficients: list[list[float]], logprobs: list[list[float]]) -> float:
    """`-(coefficients * logprobs).sum()`, the loss both arms minimize, on
    plain lists. The container computes the same thing on tensors with the
    coefficients held constant; this is what the selftest checks against."""
    return -sum(c * lp for row, lps in zip(coefficients, logprobs) for c, lp in zip(row, lps))


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `delta_report` read: binary
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
# Modal: the pins and the image from recipes/papers/adaptive-clip/recipe.py.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-flash-reinforce")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        requirement("math"),  # MathEqual imports Math-Verify
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
    # This checkout's package, so the container runs the same
    # `wai.FlashReinforce` the tests pin (see the sys.path line at the top).
    .add_local_dir(str(WHILEAI_SRC), "/root/whileai_local/whileai", ignore=["**/__pycache__/**"])
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _generate(model, tokenizer, prompt_texts, *, n, max_new_tokens, temperature, batch):
    """Sample `n` replies per prompt with HF generate, untruncated (top-p 1,
    top-k off, no repetition penalty: Qwen's generation_config sets all
    three, and any of them would put the sampler on a different support
    from the learner). Returns the full padded sequences, the prompt
    length, the prompt attention mask and the per-reply completion length in
    tokens (up to and including the stop token), one chunk per call."""
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    eos_ids = set(model.generation_config.eos_token_id or [])
    if isinstance(model.generation_config.eos_token_id, int):
        eos_ids = {model.generation_config.eos_token_id}
    eos_ids.add(tokenizer.eos_token_id)
    chunks = []
    for start in range(0, len(prompt_texts), batch):
        chunk = prompt_texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        completion = gen[:, prompt_len:]
        lengths = []
        for row in completion.tolist():
            stop = next((i for i, tok in enumerate(row) if tok in eos_ids), None)
            lengths.append(len(row) if stop is None else stop + 1)
        prompt_mask = enc["attention_mask"].repeat_interleave(n, dim=0)
        chunks.append((gen, prompt_len, prompt_mask, lengths))
    model.train()
    return chunks


def _sample(model, tokenizer, questions, *, n, max_new_tokens, temperature, batch=32):
    """`n` decoded replies per question, for eval."""
    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    texts = [
        tokenizer.apply_chat_template(messages_for(q), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    out: list[list[str]] = []
    for gen, prompt_len, _mask, lengths in _generate(
        model,
        tokenizer,
        texts,
        n=n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch=batch,
    ):
        decoded = [
            tokenizer.decode(row[prompt_len : prompt_len + length], skip_special_tokens=True)
            for row, length in zip(gen, lengths)
        ]
        for i in range(len(decoded) // n):
            out.append(decoded[i * n : (i + 1) * n])
    return out


def _token_logprobs(model, gen, prompt_len, prompt_mask, lengths, *, micro, grad):
    """Teacher-forced log-probability of every completion token under the
    model's current weights: one forward pass per micro-batch, log-softmax
    in float32. Returns a (B, max_new) tensor, zero past each trajectory's
    length. With `grad=False` it is data; with `grad=True` it is the tensor
    the loss multiplies, and the caller backpropagates per micro-batch, so
    this yields (start, tensor) pairs instead of building one big graph."""
    import torch

    completion = gen[:, prompt_len:]
    comp_mask = torch.zeros_like(completion)
    for i, length in enumerate(lengths):
        comp_mask[i, :length] = 1
    attention = torch.cat([prompt_mask, comp_mask], dim=1)
    # generate() derives positions from the attention mask (left padding);
    # the same rule here keeps teacher forcing on the sampler's positions.
    position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
    for start in range(0, gen.shape[0], micro):
        sl = slice(start, start + micro)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = model(
                input_ids=gen[sl],
                attention_mask=attention[sl],
                position_ids=position_ids[sl],
                use_cache=False,
            )
            logits = out.logits[:, prompt_len - 1 : -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            picked = lp.gather(-1, completion[sl].unsqueeze(-1)).squeeze(-1)
            picked = picked * comp_mask[sl].to(picked.dtype)
        yield start, picked


def _swap(params, values):
    """Copy `values` into `params` in place and return what was there."""
    import torch

    with torch.no_grad():
        before = [p.detach().clone() for p in params]
        for p, v in zip(params, values):
            p.copy_(v)
    return before


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 40,
    prompts_per_step: int = PROMPTS_PER_STEP,
    lag: int = LAG,
    learning_rate: float = LEARNING_RATE,
    trust: float | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    eval_samples: int = 4,
    eval_base: bool = False,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again."""
    import random
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    sys.path.insert(0, "/root")
    from recipe_mod import (
        EVAL_RUNS,
        EVAL_TEMPERATURE,
        GRAD_CLIP,
        TRAIN_SEED,
        TRAIN_TEMPERATURE,
        StaleSampler,
        gold_of,
        graded_rows,
        mean_length,
        messages_for,
        outcome_of,
        uncorrected_coefficients,
    )

    from whileai.simulations.defaults import TRAINING_LORA_ALPHA, TRAINING_LORA_RANK
    from whileai.simulations.training import training_run

    started = time.time()
    method = wai.FlashReinforce() if trust is None else wai.FlashReinforce(trust=trust)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    config = {
        "arm": arm,
        "method": str(method) if arm == "recipe" else "uncorrected stale REINFORCE",
        "base_model": base_model,
        "steps": steps,
        "prompts_per_step": prompts_per_step,
        "rollouts_per_prompt": 1,
        "lag": lag,
        "learning_rate": learning_rate,
        "grad_clip": GRAD_CLIP,
        "lora_rank": TRAINING_LORA_RANK,
        "lora_alpha": TRAINING_LORA_ALPHA,
        "max_new_tokens": max_new_tokens,
        "train_temperature": TRAIN_TEMPERATURE,
        "eval_temperature": EVAL_TEMPERATURE,
        "train_prompts": len(train_tasks),
        "holdout": len(holdout),
        "gpu": DEFAULT_GPU,
        "reward": "binary MathEqual against the GSM8K gold",
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        run = training_run(
            run_name,
            base_model=base_model,
            trainer="hf-flash-reinforce-lora",
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
                model,
                tokenizer,
                questions,
                n=eval_samples,
                max_new_tokens=max_new_tokens,
                temperature=EVAL_TEMPERATURE,
            )
            rows = graded_rows(holdout, replies)
            base_runs.append(rows)
            print(f"base run {i + 1}/{EVAL_RUNS}: {wai.pass_at(rows)}")

    # The adapter. Seeded right here so both arms draw the same A matrix
    # whatever ran before (the first arm's base evals advance the RNG).
    set_seed(TRAIN_SEED)
    lora = LoraConfig(
        r=TRAINING_LORA_RANK,
        lora_alpha=TRAINING_LORA_ALPHA,
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
    model = get_peft_model(model, lora)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)

    # The stale sampler: a frozen copy of the adapter, swapped in to generate
    # and swapped out to train, refreshed from the policy every `lag` steps.
    sampler = StaleSampler(lag)
    sampler_weights = [p.detach().clone() for p in params]
    order = list(range(len(train_tasks)))
    rng = random.Random(TRAIN_SEED)
    rng.shuffle(order)
    cursor = 0

    def next_prompts() -> list[dict]:
        nonlocal cursor, order
        picked = []
        while len(picked) < prompts_per_step:
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            picked.append(train_tasks[order[cursor]])
            cursor += 1
        return picked

    def generate_batch() -> dict:
        """One rollout per prompt from the sampler, with the sampler's own
        log-probabilities, teacher-forced under the same weights that wrote
        the tokens so the ratio at lag 0 is exactly 1."""
        tasks = next_prompts()
        texts = [
            tokenizer.apply_chat_template(
                messages_for(t["question"]), tokenize=False, add_generation_prompt=True
            )
            for t in tasks
        ]
        live = _swap(params, sampler_weights)
        try:
            (gen, prompt_len, prompt_mask, lengths), *rest = _generate(
                model,
                tokenizer,
                texts,
                n=1,
                max_new_tokens=max_new_tokens,
                temperature=TRAIN_TEMPERATURE,
                batch=prompts_per_step,
            )
            assert not rest, "one chunk per step: batch == prompts_per_step"
            behavior = torch.zeros(gen.shape[0], max_new_tokens, device=gen.device)
            for start, picked in _token_logprobs(
                model, gen, prompt_len, prompt_mask, lengths, micro=16, grad=False
            ):
                behavior[start : start + picked.shape[0], : picked.shape[1]] = picked
        finally:
            _swap(params, live)
        completions = [
            tokenizer.decode(row[prompt_len : prompt_len + length], skip_special_tokens=True)
            for row, length in zip(gen, lengths)
        ]
        rewards = [outcome_of(c, gold_of(t["answer"])) for c, t in zip(completions, tasks)]
        return {
            "stamp": sampler.stamp(),
            "tasks": tasks,
            "gen": gen,
            "prompt_len": prompt_len,
            "prompt_mask": prompt_mask,
            "lengths": lengths,
            "behavior": behavior.cpu(),
            "completions": completions,
            "rewards": rewards,
        }

    history: list[dict] = []
    last_batch: list[dict] = []
    queue = [generate_batch()]
    model.train()
    for step in range(1, steps + 1):
        batch = queue.pop(0)
        # Pipelined: the next batch is drawn before this update lands, the
        # way an asynchronous trainer overlaps generation with training.
        if step < steps:
            queue.append(generate_batch())
        gen, prompt_len, prompt_mask, lengths = (
            batch["gen"],
            batch["prompt_len"],
            batch["prompt_mask"],
            batch["lengths"],
        )
        n = gen.shape[0]
        current = torch.zeros(n, max_new_tokens, device=gen.device)
        for start, picked in _token_logprobs(
            model, gen, prompt_len, prompt_mask, lengths, micro=16, grad=False
        ):
            current[start : start + picked.shape[0], : picked.shape[1]] = picked
        current_cpu = current.cpu()
        trajectories = [
            {
                "reward": batch["rewards"][i],
                "logprobs": current_cpu[i, : lengths[i]].tolist(),
                "behavior_logprobs": batch["behavior"][i, : lengths[i]].tolist(),
            }
            for i in range(n)
        ]
        # The update rule. Both arms read the same trajectories; the recipe
        # trains on FlashReinforce's coefficients, the baseline on the
        # uncorrected ones. The FlashReinforce stats (admitted share, KL,
        # ratio) are logged for both, so the baseline's staleness is measured
        # even though it ignores it.
        update = method.update(trajectories)
        coefficients = (
            update.coefficients if arm == "recipe" else uncorrected_coefficients(trajectories)
        )
        coef = torch.zeros(n, max_new_tokens, device=gen.device)
        for i, row in enumerate(coefficients):
            coef[i, : len(row)] = torch.tensor(row, device=gen.device)

        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for start, picked in _token_logprobs(
            model, gen, prompt_len, prompt_mask, lengths, micro=8, grad=True
        ):
            piece = coef[start : start + picked.shape[0], : picked.shape[1]]
            loss = -(piece * picked).sum()  # coefficients are constants; only logp has grad
            loss.backward()
            loss_total += float(loss.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP))
        optimizer.step()
        lag_now = sampler.lag_of(batch["stamp"])
        refreshed = sampler.updated()
        if refreshed:
            sampler_weights = [p.detach().clone() for p in params]

        point = {
            "step": step,
            "lag": lag_now,
            "reward": float(statistics.fmean(batch["rewards"])),
            "admitted_share": float(update.stats["admitted_share"]),
            "mean_sequence_kl": float(update.stats["mean_sequence_kl"]),
            "max_sequence_kl": float(update.stats["max_sequence_kl"]),
            "mean_ratio": float(update.stats["mean_ratio"]),
            "completion_tokens": float(statistics.fmean(lengths)),
            "truncated_share": float(sum(1 for length in lengths if length >= max_new_tokens) / n),
            "loss": loss_total,
            "grad_norm": grad_norm,
        }
        history.append(point)
        print(
            f"{arm} step {step}/{steps} lag {lag_now} reward {point['reward']:.3f} "
            f"admitted {point['admitted_share']:.3f} kl {point['mean_sequence_kl']:.2e} "
            f"(max {point['max_sequence_kl']:.2e}) ratio {point['mean_ratio']:.4f} "
            f"len {point['completion_tokens']:.0f} loss {loss_total:+.3e} gnorm {grad_norm:.3f}"
            + (" refresh" if refreshed else "")
        )
        for note in update.notes:
            if "on-policy" not in note:
                print(f"  {note}")
        if run is not None:
            run.log(step, **{k: v for k, v in point.items() if k != "step"})
        last_batch = [
            {
                "prompt": t["question"],
                "final_text": c,
                "reward": r,
                "scenario_id": t["scenario_id"],
                "rollout_index": 0,
            }
            for t, c, r in zip(batch["tasks"], batch["completions"], batch["rewards"])
        ]
        del current, coef, gen

    replies = _sample(
        model,
        tokenizer,
        questions,
        n=eval_samples,
        max_new_tokens=max_new_tokens,
        temperature=EVAL_TEMPERATURE,
    )
    after_rows = graded_rows(holdout, replies)
    after = wai.pass_at(after_rows)
    print(f"{arm}: {after}")

    # What the reward actually paid for in the last training batch (chapter
    # Over-optimization). Nothing here is endorsed: the reward is the answer
    # being right, and any surface feature that correlates with it is the
    # thing to be suspicious of.
    # One rollout per prompt means one row per ask, so the within-ask column
    # the scan ranks by is empty; the pooled correlation is what is left, and
    # the name says so.
    scan = wai.hack_scan(last_batch) if last_batch else None
    hack_scan_top = ""
    if scan is not None:
        top = scan.get("top_feature")
        if isinstance(top, dict) and top.get("name"):
            hack_scan_top = str(top["name"])
        else:
            pooled = [f for f in (scan.get("features") or []) if f.get("pooled") is not None]
            if pooled:
                best = max(pooled, key=lambda f: abs(float(f["pooled"])))
                hack_scan_top = f"{best['name']} (pooled r {float(best['pooled']):+.2f})"
        for warning in scan.get("warnings") or []:
            print(f"{arm} hack scan: {warning}")
    print(f"{arm} hack scan: top feature {hack_scan_top or 'none above the floor'}")

    out_dir = os.path.join(VOLUME_ROOT, run_name)
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
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
        "steps": steps,
        "length_after": mean_length(after_rows),
        "hack_scan_top": hack_scan_top,
        "history": history,
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
    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
    }


def curves(history: list[dict]) -> dict:
    """The per-step series the README reads, plus their means."""
    keys = (
        "lag",
        "reward",
        "admitted_share",
        "mean_sequence_kl",
        "max_sequence_kl",
        "mean_ratio",
        "completion_tokens",
    )
    series = {k: [round(p[k], 6) for p in history] for k in keys}
    means = {f"{k}_mean": round(statistics.fmean(series[k]), 6) for k in keys if series[k]}
    return {**series, **means}


def _empty_results() -> dict:
    return {"arms": {}, "checks": {"length_after": {}}, "verified": "1970-01-01"}


def selftest() -> None:
    """The reward, the lag bookkeeping, the two coefficient rules and the
    loss they feed, on the CPU. No GPU, no key, no model download."""
    # The grader is a program, not a judge.
    assert gold_of("She has 3 left.\n#### 1,200") == "1200"
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("grader: MathEqual reads \\boxed{} and the last number")

    # The stale sampler: step 1 is on-policy, then the lag cycles 1..LAG.
    lags = lag_sequence(40, LAG)
    assert lags[:9] == [0, 1, 2, 3, 4, 1, 2, 3, 4], lags[:9]
    assert max(lags) == LAG and min(lags[1:]) == 1
    assert abs(statistics.fmean(lags) - 2.4) < 1e-9
    s = StaleSampler(LAG)
    assert [s.updated() for _ in range(8)] == [False, False, False, True] * 2
    assert s.sampler_version == 8 and s.policy_version == 8
    print(f"stale sampler: lag per step {lags[:9]} ..., mean {statistics.fmean(lags):.2f} over 40")

    # The recipe arm's loss is -(coef * logp).sum() with the coefficients
    # from wai.FlashReinforce().update, on the batch tests/api/test_flash_reinforce.py pins.
    batch = [
        {"reward": 1.0, "logprobs": [-0.5, -1.2, -0.3]},
        {"reward": 0.0, "logprobs": [-0.9, -0.4]},
        {"reward": 1.0, "logprobs": [-0.1] * 20},
    ]
    update = wai.FlashReinforce().update(batch)
    expected = [1 / 3, -2 / 3, 1 / 3]
    for i, a in enumerate(expected):
        t = len(batch[i]["logprobs"])
        assert all(abs(c - a / (t * 3)) < 1e-12 for c in update.coefficients[i]), i
        assert abs(sum(update.coefficients[i]) - a / 3) < 1e-12  # 1/B per trajectory, any length
    logps = [t["logprobs"] for t in batch]
    recipe_loss = loss_of(update.coefficients, logps)
    by_hand = -sum(a / (len(lp) * 3) * sum(lp) for a, lp in zip(expected, logps))
    assert abs(recipe_loss - by_hand) < 1e-12, (recipe_loss, by_hand)
    print(f"recipe loss on the pinned batch: {recipe_loss:+.6f} = -(coef * logp).sum()")

    # The stale token: the ratio corrects it, the gate reads the mean KL.
    p, q = 0.2, 0.4
    stale = {"reward": 1.0, "logprobs": [math.log(q)] * 2, "behavior_logprobs": [math.log(p)] * 2}
    other = {"reward": 0.0, "logprobs": [-1.0]}
    open_gate = wai.FlashReinforce(trust=math.inf).update([stale, other])
    assert all(abs(c - 0.5 * (q / p) / 4) < 1e-12 for c in open_gate.coefficients[0])
    shut = wai.FlashReinforce().update([stale, other])  # trust 3e-3 against a KL of ~0.07
    assert shut.admitted == [False, True] and shut.coefficients[0] == [0.0, 0.0]
    print(
        f"gate: KL {open_gate.stats['max_sequence_kl']:.4f} > trust 0.003 masks the trajectory whole"
    )

    # The baseline: same advantage, token mean, no ratio, no gate, no 1/T.
    base = uncorrected_coefficients(batch)
    total = 3 + 2 + 20
    for i, a in enumerate(expected):
        assert len(base[i]) == len(batch[i]["logprobs"])
        assert all(abs(c - a / total) < 1e-12 for c in base[i]), i
    assert abs(sum(base[2]) / sum(base[0]) - 20 / 3) < 1e-9  # the long one weighs more
    base_stale = uncorrected_coefficients([stale, other])
    assert all(abs(c - 0.5 / 3) < 1e-12 for c in base_stale[0])  # as if fresh: no ratio, no mask
    baseline_loss = loss_of(base, logps)
    print(f"baseline loss on the same batch: {baseline_loss:+.6f} (token mean, ratio 1, no gate)")

    _selftest_tensor_loss(update.coefficients, logps, recipe_loss)
    print("selftest ok")


def _selftest_tensor_loss(coefficients, logps, expected: float) -> None:
    """The container's loss on padded tensors equals the list version, and
    the gradient flows only through logp: what `run_arm` relies on."""
    try:
        import torch
    except ImportError:
        print("torch not installed locally: skipping the tensor check")
        return
    width = max(len(r) for r in logps)
    coef = torch.zeros(len(logps), width)
    logp = torch.zeros(len(logps), width, requires_grad=True)
    with torch.no_grad():
        for i, (c, lp) in enumerate(zip(coefficients, logps)):
            coef[i, : len(c)] = torch.tensor(c)
            logp[i, : len(lp)] = torch.tensor(lp)
    loss = -(coef * logp).sum()
    loss.backward()
    assert abs(float(loss) - expected) < 1e-6
    assert torch.allclose(logp.grad, -coef)
    print("tensor loss matches and d loss / d logp is -coef")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=120)
    ap.add_argument("--prompts-per-step", type=int, default=PROMPTS_PER_STEP)
    ap.add_argument(
        "--lag", type=int, default=LAG, help="optimizer steps between sampler refreshes"
    )
    ap.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    ap.add_argument(
        "--trust",
        type=float,
        default=None,
        help="FlashReinforce trust; default FLASH_REINFORCE_TRUST",
    )
    ap.add_argument("--selftest", action="store_true", help="the pure parts, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    # GSM8K's train and test splits are already disjoint, so this should drop
    # nothing. It runs anyway, and the count goes in the Checks table, because
    # "should" is not a measurement (Lambert 2025, chapter Evaluation).
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]

    # Start from what is already on disk, so `--arm recipe` refreshes one arm
    # instead of wiping the other one and the delta. Only a both-arm run moves
    # `verified` and the delta; a one-arm run says so and leaves them alone.
    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else _empty_results()
    results.update(
        {
            "recipe": HERE.name,
            "title": "FlashREINFORCE: one stale rollout per prompt, corrected, gated, length-normalized",
            "paper": "https://yifanzhang-pro.github.io/FlashREINFORCE/FlashREINFORCE.pdf",
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
    checks.setdefault("hack_scan_top", "")
    checks.setdefault("hack_scan_top_by_arm", {})
    arm_rows: dict[str, list[dict]] = {}
    run_std = float(checks.get("run_std") or 0.0)
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""

    with modal.enable_output(), app.run():
        for i, arm in enumerate(arms):
            out = run_arm.remote(
                arm,
                train_tasks,
                holdout,
                f"flash-reinforce-{arm}-{date.today().isoformat()}",
                steps=args.steps,
                prompts_per_step=args.prompts_per_step,
                lag=args.lag,
                learning_rate=args.learning_rate,
                trust=args.trust,
                eval_samples=args.k,
                eval_base=(i == 0),
            )
            gpu_minutes += out["gpu_minutes"]
            run_url = out["run_url"] or run_url
            if out["base_runs"]:
                base_runs = out["base_runs"]
                noise = wai.eval_variance(*base_runs)
                run_std = float(noise["run_std"])
                print(f"eval noise over {len(base_runs)} base runs: run_std {run_std:.4f}")
                arm_rows["base"] = base_runs[0]
                results["arms"]["base"] = {
                    **summarize(base_runs[0]),
                    "steps": 0,
                    "gpu_minutes": 0,
                }
                checks["run_std"] = run_std
                checks["run_std_runs"] = int(noise["n_runs"])
                checks["length_before"] = mean_length(base_runs[0])
            arm_rows[arm] = out["after_rows"]
            results["arms"][arm] = {
                **summarize(out["after_rows"]),
                "steps": out["steps"],
                "gpu_minutes": round(out["gpu_minutes"], 1),
                "usd": round(out["gpu_minutes"] / 60.0 * usd_per_hour, 2),
                "training": curves(out["history"]),
            }
            checks["length_after"][arm] = out["length_after"]
            checks["hack_scan_top_by_arm"][arm] = out["hack_scan_top"]
            checks["hack_scan_top"] = out["hack_scan_top"]

    if "baseline" in arm_rows and "recipe" in arm_rows:
        # run_std makes "moved" mean bigger than the eval's own re-run noise,
        # and proxy names the training reward when it differs from the target.
        # Here it does not, so there is nothing for PROXY to point at.
        d = wai.compare(
            arm_rows["baseline"],
            arm_rows["recipe"],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks.get("run_std_runs") or EVAL_RUNS),
            # one training seed per arm: the report says unresolved (#356)
            train_runs={"before": [arm_rows["baseline"]], "after": [arm_rows["recipe"]]},
            proxy=PROXY,
        )
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": (
                "unresolved"
                if d["target_verdict"] == "unresolved"
                else "moved"
                if d["target_verdict"] == "moved"
                else "flat"
            ),
        }
        checks["train_seeds"] = {"baseline": 1, "recipe": 1}
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
    print(json.dumps({k: v for k, v in results.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
