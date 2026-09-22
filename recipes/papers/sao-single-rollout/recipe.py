"""SAO single rollout: one rollout per prompt, a critic, and a token band under a stale sampler.

    python recipe.py                      # both arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the reward, the sampler lag, the two losses, offline

GRPO needs a group of rollouts per prompt to take a baseline over. A
production trace has one: one prompt, one trajectory, written by whatever
policy was serving at the time, which is one to several updates behind the
one being trained. SAO (Hou et al. 2026) trains on exactly that shape. A
value critic gives every generated token its own advantage through
length-adaptive GAE, ``lambda = 1 - 1/(1.5 L)``, and direct double-sided
importance sampling handles the staleness: the current/rollout probability
ratio of each token is applied as a weight inside ``(0.7, 6.0)`` and the
token is *masked* outside it, whichever sign its advantage has, instead of
being clipped to the edge.

Both arms here share the model, the data, the stale sampler, the critic
and the GAE advantages. The baseline applies the ratio and never masks
(the paper's "no DIS" case, plain importance-weighted policy gradient); the
recipe is ``wai.SAO().update(batch)`` at its defaults. The two arms are the
same ``update`` call: the baseline takes ``update.advantages`` and
multiplies by the unmasked ratio itself, so the only thing that differs
between them is the mask.

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out, decontaminated
  2. run_arm():   the loop on Modal, one arm per container: a frozen copy of the
                  LoRA weights samples, refreshed every LAG policy steps; the policy
                  scores the batch once, the critic head predicts a value per token,
                  wai.SAO().update() turns that into coefficients, and the loss is
                  -(coefficients * logprobs).sum()
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

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
TITLE = "SAO single rollout: one rollout per prompt under a stale sampler"
PAPER = "https://arxiv.org/abs/2607.07508"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Policy Gradient Algorithms"  # the chapter title of Lambert 2025 this recipe rests on
# The training reward is the target: the same binary check against the GSM8K
# gold on both sides, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

# The production-trace regime, shared with the FlashReinforce and BPCO recipes
# so the three compare: one rollout per prompt, PROMPTS_PER_STEP prompts per
# optimizer step, sampled by a frozen copy of the policy's adapter that is
# refreshed only every LAG policy steps.
LAG = 4
PROMPTS_PER_STEP = 64
MAX_NEW_TOKENS = 256
# The paper trains full weights of a 30B MoE at 1e-6 with a separate 30B value
# model at 5e-6 (arXiv:2607.07508, section 4.1). Here the policy is a rank-16
# adapter and the critic is one linear layer on the frozen trunk's last hidden
# state, so both rates are overrides: 1e-5 is the adapter rate the other
# single-rollout recipes use, and a one-layer head at 5e-6 would not leave its
# zero init in 50 steps.
POLICY_LEARNING_RATE = 1e-5
CRITIC_LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
TRAIN_SEED = 17
EVAL_TEMPERATURE = 0.7

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."


# --------------------------------------------------------------------------
# Pure parts: the reward, the sampler bookkeeping, the two losses. No torch,
# so `--selftest` runs them here and the Modal container imports the same file.
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
    """Which policy version writes each batch.

    The policy is version ``t`` at the start of policy step ``t`` (``t``
    optimizer updates applied). The sampler is refreshed every ``lag``
    steps from the version one update behind, so during the update of step
    ``t`` the batch is 1 to ``lag`` updates old (0 on the very first step,
    where no update exists to lag behind). Call ``sampler_version`` once per
    policy step, before sampling; ``staleness`` is what the update sees.
    """

    def __init__(self, lag: int = LAG) -> None:
        if lag < 1:
            raise ValueError(f"lag must be 1 or more; got {lag}")
        self.lag = lag
        self.version = 0
        self._previous = 0
        self._steps = 0

    def sampler_version(self, policy_version: int) -> int:
        if self._steps % self.lag == 0:
            self.version = self._previous
        self._previous = policy_version
        self._steps += 1
        return self.version

    def staleness(self, policy_version: int) -> int:
        return policy_version - self.version


def policy_loss(coefficients, logprobs):
    """``-(coefficients * logprobs).sum()`` with the coefficients held
    constant. Lists of lists here; the container does the same line on
    tensors."""
    return -sum(
        c * lp for coef, lp_row in zip(coefficients, logprobs) for c, lp in zip(coef, lp_row)
    )


def baseline_coefficients(batch: list[dict], update) -> list[list[float]]:
    """The no-DIS arm: the same ``update.advantages`` (critic GAE, the same
    ``1/N`` token mean) times the current/rollout ratio, never masked. A
    ratio past the float range is a non-finite coefficient; the caller
    counts those and zeroes them rather than let one token NaN the step."""
    n_action = float(update.stats["action_tokens"])
    scale = 1.0 / n_action if n_action else 0.0
    out: list[list[float]] = []
    for traj, adv in zip(batch, update.advantages):
        lp = traj["logprobs"]
        blp = traj.get("behavior_logprobs") or lp
        row = []
        for t, a in enumerate(adv):
            try:
                r = math.exp(float(lp[t]) - float(blp[t]))
            except OverflowError:
                r = math.inf
            row.append(r * a * scale)
        out.append(row)
    return out


def arm_coefficients(arm: str, batch: list[dict], update) -> list[list[float]]:
    """THE ONE CHANGE. Recipe: the coefficients `wai.SAO().update` hands
    back, the ratio inside the band and zero outside it. Baseline: the same
    advantages times the ratio everywhere."""
    if arm == "recipe":
        return [list(c) for c in update.coefficients]
    if arm == "baseline":
        return baseline_coefficients(batch, update)
    raise ValueError(f"arm must be 'recipe' or 'baseline'; got {arm!r}")


def ratio_stats(batch: list[dict], band: tuple[float, float]) -> dict[str, float]:
    """The unmasked ratio over every action token: mean, max, and the share
    outside ``band``. The same number for both arms, so the baseline's log
    says what it *would* have masked."""
    low, high = band
    ratios: list[float] = []
    for traj in batch:
        lp = traj["logprobs"]
        blp = traj.get("behavior_logprobs") or lp
        for a, b in zip(lp, blp):
            try:
                ratios.append(math.exp(float(a) - float(b)))
            except OverflowError:
                ratios.append(math.inf)
    if not ratios:
        return {"mean_ratio_all": 0.0, "max_ratio": 0.0, "outside_band_share": 0.0}
    finite = [r for r in ratios if math.isfinite(r)]
    return {
        "mean_ratio_all": statistics.fmean(finite) if finite else math.inf,
        "max_ratio": max(ratios),
        "outside_band_share": sum(1 for r in ratios if not low < r < high) / len(ratios),
    }


def explained_variance(targets: list[list[float]], values: list[list[float]]) -> float | None:
    """``1 - Var(target - value) / Var(target)`` over tokens; None when the
    targets do not vary (a batch that is all right or all wrong)."""
    pairs = [(t, v) for tr, vr in zip(targets, values) for t, v in zip(tr, vr)]
    if not pairs:
        return None
    n = len(pairs)
    mean_t = sum(t for t, _ in pairs) / n
    var_t = sum((t - mean_t) ** 2 for t, _ in pairs) / n
    if var_t <= 0:
        return None
    errs = [t - v for t, v in pairs]
    mean_e = sum(errs) / n
    var_e = sum((e - mean_e) ** 2 for e in errs) / n
    return 1.0 - var_e / var_t


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


def training_rows(tasks: list[dict], texts: list[str], rewards: list[float]) -> list[dict]:
    """One training batch as rows `hack_scan` reads. One rollout per prompt,
    so `rollout_index` is always 0 within a batch."""
    return [
        {
            "prompt": t["question"],
            "final_text": text,
            "reward": r,
            "scenario_id": t["scenario_id"],
            "rollout_index": 0,
        }
        for t, text, r in zip(tasks, texts, rewards)
    ]


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/papers/adaptive-clip/recipe.py.
# The container imports whileai from this checkout (add_local_python_source),
# because wai.SAO is newer than the wheel on the index.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"
# recipes/papers/<slug>/recipe.py -> the checkout's whileai/; inside the
# container this file is /root/recipe.py and the package is /root/whileai
PACKAGE_DIR = (HERE.parents[2] if len(HERE.parents) > 2 else HERE) / "whileai"

app = modal.App("whileai-recipe-sao-single-rollout")

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
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
    # the whole package, schema JSON included, so /root/whileai shadows the wheel
    .add_local_dir(str(PACKAGE_DIR), "/root/whileai", ignore=["**/__pycache__", "**/*.pyc"])
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _sample(model, tokenizer, questions, *, n, max_new_tokens, temperature, batch=16):
    """`n` replies per question, batched. No top-p, top-k or repetition
    penalty anywhere in this recipe (Qwen's generation config turns all three
    on by default); the eval differs from the training sampler only in
    temperature."""
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
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append(decoded[i * n : (i + 1) * n])
    model.train()
    return out


def _rollout(model, tokenizer, questions, *, max_new_tokens, batch, eos_ids):
    """One rollout per question from whatever weights are loaded, at
    temperature 1.0 with no truncation, with the sampler's own log-probability
    of every generated token read off the generation scores as they are
    produced. Returns one record per question: the unpadded prompt ids, the
    completion ids up to and including the first end token, those behavior
    log-probs, and the text."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    model.eval()
    tokenizer.padding_side = "left"
    texts = [
        tokenizer.apply_chat_template(messages_for(q), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    records: list[dict] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            prompt_len = enc["input_ids"].shape[1]
            new = gen.sequences[:, prompt_len:]
            # log pi_sampler(token_t) from the step's own scores. With
            # temperature 1 and no warper the scores are the raw logits, so
            # this is the distribution the token was drawn from. One step at
            # a time keeps the peak at one (batch, vocab) float32 tensor.
            lp_steps = []
            for t, scores in enumerate(gen.scores):
                lp = torch.log_softmax(scores.float(), dim=-1).gather(1, new[:, t : t + 1])
                lp_steps.append(lp)
            behavior = torch.cat(lp_steps, dim=1).cpu()
        del gen
        new = new.cpu()
        for i in range(len(chunk)):
            prompt_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
            comp = new[i].tolist()
            length = len(comp)
            for t, tok in enumerate(comp):
                if tok in eos_ids:
                    length = t + 1
                    break
            comp = comp[:length]
            records.append(
                {
                    "prompt_ids": prompt_ids,
                    "completion_ids": comp,
                    "behavior_logprobs": behavior[i, :length].tolist(),
                    "text": tokenizer.decode(comp, skip_special_tokens=True),
                    "truncated": length == new.shape[1] and comp[-1] not in eos_ids,
                }
            )
    model.train()
    return records


def _collate(records, pad_id, device):
    """Right-padded prompt + completion, so positions 0..len-1 are the same
    ones generation used under left padding."""
    import torch

    seqs = [r["prompt_ids"] + r["completion_ids"] for r in records]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def _completion_logprobs(logits, records):
    """log pi(token_t | prefix) for each completion token, one (L,) tensor per
    record, from the logits at the positions that predict them."""
    import torch

    out = []
    for i, r in enumerate(records):
        p, n = len(r["prompt_ids"]), len(r["completion_ids"])
        sl = torch.log_softmax(logits[i, p - 1 : p - 1 + n].float(), dim=-1)
        tok = torch.tensor(r["completion_ids"], device=logits.device)
        out.append(sl.gather(1, tok[:, None]).squeeze(1))
    return out


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=2 * 60 * 60,
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
    warmup: int | None = None,
    prompts_per_step: int = PROMPTS_PER_STEP,
    lag: int = LAG,
    max_new_tokens: int = MAX_NEW_TOKENS,
    eval_samples: int = 4,
    eval_base: bool = False,
    gen_batch: int = 64,
    micro_batch: int = 8,
    seed: int = 0,
    policy_lr: float = POLICY_LEARNING_RATE,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again."""
    import contextlib
    import random
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    sys.path.insert(0, "/root")
    from recipe_mod import (
        CRITIC_LEARNING_RATE,
        EVAL_RUNS,
        EVAL_TEMPERATURE,
        GRAD_CLIP,
        TRAIN_SEED,
        StaleSampler,
        arm_coefficients,
        explained_variance,
        gold_of,
        graded_rows,
        mean_length,
        outcome_of,
        ratio_stats,
        training_rows,
    )

    import whileai as wai
    from whileai.simulations.defaults import TRAINING_LORA_ALPHA, TRAINING_LORA_RANK
    from whileai.simulations.training import training_run

    print(provenance(), file=sys.stderr)
    if arm not in ("baseline", "recipe"):
        raise ValueError(f"arm must be 'baseline' or 'recipe'; got {arm!r}")
    method = wai.SAO()
    warmup = method.critic_warmup if warmup is None else int(warmup)
    critic_steps = method.critic_steps
    print(f"{arm}: {method}; critic warmup {warmup}, lag {lag}, {prompts_per_step} prompts/step")

    started = time.time()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device
    )
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}

    config = {
        "arm": arm,
        "mask_outside_band": arm == "recipe",
        "method": str(method),
        "base_model": base_model,
        "steps": steps,
        "critic_warmup": warmup,
        "critic_steps": critic_steps,
        "prompts_per_step": prompts_per_step,
        "rollouts_per_prompt": 1,
        "lag": lag,
        "policy_learning_rate": policy_lr,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "grad_clip": GRAD_CLIP,
        "lora_rank": TRAINING_LORA_RANK,
        "lora_alpha": TRAINING_LORA_ALPHA,
        "max_new_tokens": max_new_tokens,
        "train_temperature": 1.0,
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
            trainer="sao-lora",
            total_steps=warmup + steps,
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

    # Seed right before the adapter is built, after the base evals advanced
    # the RNG on the first arm, so both arms draw the same LoRA init.
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
    policy_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    n_trainable = sum(p.numel() for p in policy_params.values())
    hidden_size = model.config.hidden_size
    # The critic: one linear layer on the trunk's last hidden state at each
    # generated token, the hidden states detached so it never trains the
    # trunk. Zero init, so V starts at 0 and the warmup moves it.
    head = torch.nn.Linear(hidden_size, 1, dtype=torch.float32).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    policy_opt = torch.optim.AdamW(list(policy_params.values()), lr=policy_lr, weight_decay=0.0)
    critic_opt = torch.optim.AdamW(head.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=0.0)
    print(f"{arm}: {n_trainable} adapter parameters, value head {hidden_size} -> 1")

    def snapshot() -> dict:
        return {n: p.detach().clone() for n, p in policy_params.items()}

    @contextlib.contextmanager
    def weights(state: dict):
        saved = snapshot()
        with torch.no_grad():
            for n, p in policy_params.items():
                p.copy_(state[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in policy_params.items():
                    p.copy_(saved[n])

    # The stale sampler: a frozen copy of the adapter. `stale` keeps the
    # version arithmetic; `sampler_state` holds the weights of that version.
    stale = StaleSampler(lag)
    sampler_state = snapshot()
    sampler_loaded = 0
    previous_state = sampler_state

    rng = random.Random(seed)
    order: list[int] = []
    log: list[dict] = []
    last_rows: list[dict] = []
    history_rows: list[dict] = []
    skipped_tokens = 0
    train_started = time.time()
    total = warmup + steps
    for step in range(total):
        step_started = time.time()
        training = step >= warmup
        policy_version = max(0, step - warmup)
        if training:
            want = stale.sampler_version(policy_version)
            if want != sampler_loaded:
                sampler_state, sampler_loaded = previous_state, want
            previous_state = snapshot()
        staleness = stale.staleness(policy_version) if training else 0

        # the batch of prompts, cycling the shuffled train set
        batch_tasks = []
        for _ in range(prompts_per_step):
            if not order:
                order = rng.sample(range(len(train_tasks)), len(train_tasks))
            batch_tasks.append(train_tasks[order.pop()])

        # 1. rollouts from the sampler, with its own log-probs
        with weights(sampler_state):
            records = _rollout(
                model,
                tokenizer,
                [t["question"] for t in batch_tasks],
                max_new_tokens=max_new_tokens,
                batch=gen_batch,
                eos_ids=eos_ids,
            )
        rewards = [
            outcome_of(r["text"], gold_of(t["answer"])) for r, t in zip(records, batch_tasks)
        ]
        last_rows = training_rows(batch_tasks, [r["text"] for r in records], rewards)
        history_rows.extend(
            {**row, "rollout_index": step} for row in last_rows
        )  # every step's batch, so a prompt has one rollout per visit

        # 2. one teacher-forced pass under the current policy: logprobs and
        #    the critic's value at every generated token
        logprobs: list[list[float]] = []
        values: list[list[float]] = []
        hiddens: list = []
        with torch.no_grad():
            for start in range(0, len(records), 2 * micro_batch):
                chunk = records[start : start + 2 * micro_batch]
                ids, mask = _collate(chunk, tokenizer.pad_token_id, device)
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                lps = _completion_logprobs(out.logits, chunk)
                last = out.hidden_states[-1]
                for i, r in enumerate(chunk):
                    p, n = len(r["prompt_ids"]), len(r["completion_ids"])
                    h = last[i, p - 1 : p - 1 + n].float()
                    hiddens.append(h)
                    values.append(head(h).squeeze(-1).tolist())
                    logprobs.append(lps[i].tolist())
                del out, last
        batch = [
            {
                "reward": rw,
                "logprobs": lp,
                "behavior_logprobs": r["behavior_logprobs"],
                "values": v,
            }
            for rw, lp, r, v in zip(rewards, logprobs, records, values)
        ]

        # 3. the update rule, the same call for both arms
        update = method.update(batch)
        assert update.value_targets is not None
        coefficients = arm_coefficients(arm, batch, update)
        nonfinite = 0
        for row in coefficients:
            for t, c in enumerate(row):
                if not math.isfinite(c):
                    row[t] = 0.0
                    nonfinite += 1
        skipped_tokens += nonfinite

        # 4. the policy step: -(coefficients * logprobs).sum(), coefficients constant
        loss_value = 0.0
        grad_norm = 0.0
        if training:
            policy_opt.zero_grad(set_to_none=True)
            for start in range(0, len(records), micro_batch):
                chunk = records[start : start + micro_batch]
                ids, mask = _collate(chunk, tokenizer.pad_token_id, device)
                out = model(input_ids=ids, attention_mask=mask)
                lps = _completion_logprobs(out.logits, chunk)
                loss = sum(
                    -(torch.tensor(coefficients[start + i], device=device) * lp).sum()
                    for i, lp in enumerate(lps)
                )
                loss.backward()
                loss_value += float(loss.detach())
                del out, lps, loss
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(list(policy_params.values()), GRAD_CLIP)
            )
            policy_opt.step()
            policy_opt.zero_grad(set_to_none=True)

        # 5. the critic: K squared-error steps toward the update's value targets
        critic_loss = 0.0
        target_tensor = torch.cat(
            [torch.tensor(t, dtype=torch.float32, device=device) for t in update.value_targets]
        )
        hidden_tensor = torch.cat(hiddens)
        for _ in range(critic_steps):
            critic_opt.zero_grad(set_to_none=True)
            pred = head(hidden_tensor).squeeze(-1)
            closs = torch.nn.functional.mse_loss(pred, target_tensor)
            closs.backward()
            critic_opt.step()
            critic_loss = float(closs.detach())
        del hidden_tensor, hiddens

        # 6. the log
        rs = ratio_stats(batch, method.ratio)
        ev = explained_variance(update.value_targets, values)
        entry = {
            "step": step,
            "phase": "train" if training else "warmup",
            "policy_step": policy_version if training else None,
            "staleness": staleness,
            "reward": statistics.fmean(rewards),
            "masked_share": update.stats["masked_token_share"],
            "admitted_share": update.stats["admitted_share"],
            "mean_ratio_in_band": update.stats["mean_ratio"],
            "mean_ratio_all": rs["mean_ratio_all"],
            "max_ratio": rs["max_ratio"],
            "mean_advantage": update.stats["mean_advantage"],
            "lambda_mean": update.stats["lambda_mean"],
            "explained_variance": ev,
            "critic_loss": critic_loss,
            "mean_value": statistics.fmean(v for vs in values for v in vs),
            "policy_loss": loss_value,
            "grad_norm": grad_norm,
            "nonfinite_coefficients": nonfinite,
            "length_tokens": statistics.fmean(len(r["completion_ids"]) for r in records),
            "truncated_share": statistics.fmean(1.0 if r["truncated"] else 0.0 for r in records),
            "action_tokens": update.stats["action_tokens"],
            "seconds": time.time() - step_started,
        }
        log.append(entry)
        print(
            f"{arm} step {step}/{total} [{entry['phase']}] lag {staleness} "
            f"reward {entry['reward']:.3f} masked {entry['masked_share']:.4f} "
            f"ratio {entry['mean_ratio_all']:.3f} (max {entry['max_ratio']:.2f}) "
            f"adv {entry['mean_advantage']:+.4f} ev {ev if ev is None else round(ev, 3)} "
            f"critic {critic_loss:.4f} len {entry['length_tokens']:.0f} "
            f"loss {loss_value:+.4f} gnorm {grad_norm:.3f} {entry['seconds']:.1f}s"
        )
        if run is not None:
            run.log(
                step,
                **{k: v for k, v in entry.items() if isinstance(v, (int, float)) and v is not None},
            )
    train_minutes = (time.time() - train_started) / 60.0

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

    # What the reward paid for (chapter Over-optimization). The last batch
    # has one rollout per prompt, so the within-ask scan has nothing to
    # contrast and says so; the whole history has every prompt several
    # times, one rollout per visit, and that scan can rank.
    def top_of(scan) -> tuple[str, str]:
        top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
        name = top.get("name", "") if isinstance(top, dict) else str(top or "")
        return name, str(scan.get("regime", "")) if isinstance(scan, dict) else ""

    last_scan = wai.hack_scan(last_rows) if last_rows else {}
    history_scan = wai.hack_scan(history_rows) if history_rows else {}
    last_top, last_regime = top_of(last_scan)
    history_top, history_regime = top_of(history_scan)
    print(f"{arm} hack scan, last batch: regime {last_regime}, top {last_top or 'none'}")
    print(
        f"{arm} hack scan, all {len(history_rows)} training rollouts: regime {history_regime}, top {history_top or 'none'}"
    )

    out_dir = os.path.join(VOLUME_ROOT, run_name)
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(head.state_dict(), os.path.join(out_dir, "value_head.pt"))
    Path(os.path.join(out_dir, "train_log.json")).write_text(json.dumps(log, indent=1))
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
        "pass_at_1": after.pass_at_1,
        "gpu_minutes": gpu_minutes,
        "train_minutes": train_minutes,
        "steps": steps,
        "final_reward": log[-1]["reward"] if log else None,
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
        "warmup": warmup,
        "length_after": mean_length(after_rows),
        "hack_scan_top": history_top,
        "hack_scan_regime": history_regime,
        "hack_scan_last_batch": {"top": last_top, "regime": last_regime},
        "skipped_tokens": skipped_tokens,
        "log": log,
        "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
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


def curve_summary(log: list[dict]) -> dict:
    """The per-step curves folded to what the README quotes: first, last,
    mean and max over the policy steps, per logged number."""
    train = [e for e in log if e["phase"] == "train"] or log
    keys = (
        "reward",
        "masked_share",
        "mean_ratio_all",
        "max_ratio",
        "mean_advantage",
        "explained_variance",
        "critic_loss",
        "length_tokens",
        "grad_norm",
        "staleness",
    )
    out: dict = {}
    for k in keys:
        xs = [e[k] for e in train if e.get(k) is not None]
        if xs:
            out[k] = {
                "first": xs[0],
                "last": xs[-1],
                "mean": statistics.fmean(xs),
                "max": max(xs),
                "min": min(xs),
            }
    warm = [e for e in log if e["phase"] == "warmup"]
    if warm:
        out["warmup_masked_share_mean"] = statistics.fmean(e["masked_share"] for e in warm)
        out["warmup_explained_variance_last"] = warm[-1]["explained_variance"]
    return out


def selftest() -> None:
    """The pure parts on hand-made inputs. No GPU, no key, no model download."""
    import whileai as wai

    # The grader is a program, not a judge.
    assert gold_of("She has 3 left.\n#### 1,234") == "1234"
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("grader: MathEqual reads \\boxed{} and the last number against the GSM8K gold")

    # The stale sampler: 1 to LAG updates behind from the first refresh on.
    stale = StaleSampler(LAG)
    lags = []
    for t in range(12):
        stale.sampler_version(t)
        lags.append(stale.staleness(t))
    assert lags == [0, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4], lags
    assert max(lags) == LAG and min(lags[1:]) == 1
    print(f"stale sampler: staleness per policy step {lags} (refresh every {LAG})")

    # A hand-made batch with values and a lagged sampler: token 0 of the
    # first trajectory has ratio 10 (outside (0.7, 6.0)), token 2 ratio 2,
    # the rest ratio 1.
    method = wai.SAO()
    lp0 = [-1.0, -1.0, -1.0]
    blp0 = [-1.0 - math.log(10), -1.0, -1.0 - math.log(2)]
    batch = [
        {"reward": 1.0, "logprobs": lp0, "behavior_logprobs": blp0, "values": [0.2, 0.5, 0.6]},
        {
            "reward": 0.0,
            "logprobs": [-0.7, -0.3],
            "behavior_logprobs": [-0.7, -0.3],
            "values": [0.4, 0.3],
        },
    ]
    update = method.update(batch)
    n = update.stats["action_tokens"]
    assert n == 5
    recipe = arm_coefficients("recipe", batch, update)
    baseline = arm_coefficients("baseline", batch, update)
    adv = update.advantages
    # recipe: masked outside the band, ratio inside it, over N
    assert recipe[0][0] == 0.0
    assert abs(recipe[0][1] - adv[0][1] / n) < 1e-12
    assert abs(recipe[0][2] - 2.0 * adv[0][2] / n) < 1e-12
    assert recipe == [list(c) for c in update.coefficients]
    # baseline: the same advantages, the ratio applied everywhere, nothing masked
    assert abs(baseline[0][0] - 10.0 * adv[0][0] / n) < 1e-12
    assert abs(baseline[0][1] - adv[0][1] / n) < 1e-12
    assert abs(baseline[0][2] - 2.0 * adv[0][2] / n) < 1e-12
    assert baseline[1] == recipe[1]  # on-policy tokens: identical in both arms
    assert update.stats["masked_token_share"] == 1 / 5
    # the loss is -(coef * logp).sum() with the coefficients as constants
    logprobs = [t["logprobs"] for t in batch]
    by_hand = -(
        recipe[0][1] * -1.0 + recipe[0][2] * -1.0 + recipe[1][0] * -0.7 + recipe[1][1] * -0.3
    )
    assert abs(policy_loss(recipe, logprobs) - by_hand) < 1e-12
    assert policy_loss(baseline, logprobs) != policy_loss(recipe, logprobs)
    rs = ratio_stats(batch, method.ratio)
    assert abs(rs["max_ratio"] - 10.0) < 1e-9 and rs["outside_band_share"] == 1 / 5
    print(
        f"one batch: recipe masks 1 of {int(n)} tokens (ratio 10 outside {method.ratio}), "
        f"baseline keeps it at {baseline[0][0]:+.4f}; recipe loss {policy_loss(recipe, logprobs):+.5f}, "
        f"baseline loss {policy_loss(baseline, logprobs):+.5f}"
    )
    # value targets are the reward at every token; a perfect critic has EV 1
    assert update.value_targets == [[1.0, 1.0, 1.0], [0.0, 0.0]]
    assert explained_variance(update.value_targets, update.value_targets) == 1.0
    assert abs(explained_variance(update.value_targets, [[0.5] * 3, [0.5] * 2]) - 0.0) < 1e-12
    assert explained_variance([[1.0, 1.0]], [[0.0, 0.0]]) is None
    print(
        "critic: value targets are the trajectory's return; explained variance 1 at a perfect fit"
    )

    # a ratio past the float range is a non-finite baseline coefficient, not a crash
    huge = [{"reward": 1.0, "logprobs": [0.0], "behavior_logprobs": [-1000.0], "values": [0.0]}]
    u = method.update(huge)
    assert arm_coefficients("recipe", huge, u) == [[0.0]]
    assert not math.isfinite(arm_coefficients("baseline", huge, u)[0][0])
    print("selftest ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40, help="policy steps after the critic warmup")
    ap.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="critic-only steps first; default wai.SAO().critic_warmup",
    )
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=120)
    ap.add_argument("--prompts-per-step", type=int, default=PROMPTS_PER_STEP)
    ap.add_argument("--lag", type=int, default=LAG, help="policy steps between sampler refreshes")
    ap.add_argument("--gen-batch", type=int, default=64, help="rollouts per generate call")
    ap.add_argument("--micro-batch", type=int, default=8, help="sequences per backward pass")
    ap.add_argument(
        "--policy-lr",
        type=float,
        default=POLICY_LEARNING_RATE,
        help="AdamW rate on the adapter; the band binds only when lag x step moves a token",
    )
    ap.add_argument("--selftest", action="store_true", help="the pure parts, offline")
    ap.add_argument(
        "--tag",
        default="",
        help="a climb round: writes results-<tag>.json and train_log-<tag>.json",
    )
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

    # Start from what is already on disk, so `--arm recipe` refreshes one arm
    # instead of wiping the other one and the delta. Only a both-arm run moves
    # `verified` and the delta; a one-arm run says so and leaves them alone.
    suffix = f"-{args.tag}" if args.tag else ""
    results_path = HERE / f"results{suffix}.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(
        {
            "recipe": HERE.name,
            "title": TITLE,
            "paper": PAPER,
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
    curves = results.setdefault("curves", {})
    arm_rows: dict[str, list[dict]] = {}
    run_std = 0.0
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""
    logs: dict[str, list[dict]] = {}

    with modal.enable_output(), app.run():
        # both arms at once, each on its own GPU; only the first evaluates the base
        calls = {
            arm: run_arm.spawn(
                arm,
                train_tasks,
                holdout,
                f"sao-single-rollout-{arm}{suffix}-{date.today().isoformat()}",
                steps=args.steps,
                warmup=args.warmup,
                prompts_per_step=args.prompts_per_step,
                lag=args.lag,
                eval_samples=args.k,
                eval_base=(i == 0),
                gen_batch=args.gen_batch,
                micro_batch=args.micro_batch,
                seed=args.seed,
                policy_lr=args.policy_lr,
            )
            for i, arm in enumerate(arms)
        }
        for arm, call in calls.items():
            out = call.get()
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
                "warmup": out["warmup"],
                "gpu_minutes": round(out["gpu_minutes"], 1),
                "train_minutes": round(out["train_minutes"], 1),
                "usd": round(out["gpu_minutes"] / 60.0 * usd_per_hour, 2),
            }
            checks["length_after"][arm] = out["length_after"]
            # the last arm's history scan names the check; both arms' scans
            # are kept beside it
            checks["hack_scan_top"] = out["hack_scan_top"]
            checks.setdefault("hack_scan", {})[arm] = {
                "history_top": out["hack_scan_top"],
                "history_regime": out["hack_scan_regime"],
                "last_batch": out["hack_scan_last_batch"],
            }
            curves[arm] = curve_summary(out["log"])
            curves[arm]["skipped_tokens"] = out["skipped_tokens"]
            curves[arm]["max_memory_gb"] = round(out["max_memory_gb"], 1)
            logs[arm] = out["log"]
            print(f"{arm}: {out['gpu_minutes']:.1f} GPU min, peak {out['max_memory_gb']:.1f} GB")

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
    existing = {}
    log_path = HERE / f"train_log{suffix}.json"
    if log_path.exists():
        existing = json.loads(log_path.read_text())
    existing.update(logs)
    log_path.write_text(json.dumps(existing, indent=1) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
