"""BPCO bounded critic: a critic that can only say a number the reward could be.

    python recipe.py                      # both arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the arithmetic, offline, no GPU and no key

One rollout per prompt is the shape a production trace arrives in, and it
leaves nothing to take a group mean over. A critic is the other baseline:
a value head that predicts the reward from the tokens so far. The paper's
complaint is that the standard PPO critic recipe (an unbounded head,
lambda-returns, whitened advantages, a fixed ratio clip, no warm-up) is
what makes critic-based training degrade on one response per prompt, and
that five changes, taken together, make it train:

  1. the head's raw output is bounded to the reward range through an
     arctangent, ``V = R_min + (R_max - R_min)(1/2 + atan(z)/pi)``, so the
     critic can never predict a return the reward cannot pay (equation 9);
  2. the critic trains toward the Monte Carlo return, which with a terminal
     reward is the reward at every token (``lambda_V = 1``, equation 11);
  3. the policy advantage is GAE with a length-adaptive
     ``lambda = 1 - 1/(0.4 L)`` and no whitening (equations 3, 4 and 14);
  4. the ratio clip is DPPO's: ``1 -/+ clip/mu``, wider for a token the
     sampler thought rare (equation 2);
  5. the critic trains alone for ``critic_warmup`` steps before the policy
     moves (section 4).

``wai.BPCO().update(batch)`` is that arithmetic. The recipe arm runs it on
a real model; the baseline arm runs the standard PPO critic recipe, written
out plainly below (``baseline_update``). Both arms share the model, the
LoRA, the data, the reward, the sampler and every other knob, so "the
change" here is the bundle, the same comparison the paper draws.

The regime is a production trace's: the rollouts are drawn by a **stale
sampler**, a frozen copy of the policy's adapter that is refreshed only
every ``LAG`` = 4 optimizer steps, so a batch was written by a policy one
to four updates old and the ratio against it is live.

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out, decontaminated
  2. run_arm():   the loop, on Modal: sample (stale), score, update, log; one arm per call
  3. evaluate:    same holdout, k samples per task, graded by MathEqual; the base three times
  4. results.json + the paired delta (wai.delta_report)
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

import whileai as wai
from whileai.config import provenance, requirement
from whileai.methods import Update
from whileai.simulations.defaults import (
    BPCO_LOG_RATIO_CAP,
    TRAINING_LORA_ALPHA,
    TRAINING_LORA_RANK,
)

LORA_RANK, LORA_ALPHA = TRAINING_LORA_RANK, TRAINING_LORA_ALPHA  # the reference adapter shape

HERE = Path(__file__).resolve().parent
# The checkout's own package, for the image below. In the container this
# module lives at /root, where the mount already sits.
WHILEAI_SRC = next(
    (p / "whileai" for p in [*HERE.parents] if (p / "whileai" / "methods.py").is_file()),
    Path("/root/whileai_src/whileai"),
)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Policy Gradient Algorithms"  # the value function, GAE and the clipped ratio
# The training reward is the same binary check as the eval metric, carried
# on the eval rows as a marker so delta_report can name it. It is the same
# program at a different temperature, so the proxy-vs-target verdict is
# structurally clean here; it is recorded, not claimed.
PROXY = "marker:gsm8k_outcome"
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

# The shared single-rollout protocol (FlashReinforce, SAO and BPCO recipes).
LAG = 4  # optimizer steps between refreshes of the sampler: a batch is 1 to LAG updates old
PROMPTS_PER_STEP = 64  # one rollout each
MAX_NEW_TOKENS = 256
EVAL_TEMPERATURE = 0.7
POLICY_LR = 1e-5  # on the adapter; the paper's 1e-6 (BPCO_LEARNING_RATE) is full weights
CRITIC_LR = (
    1e-4  # on a linear head; the paper's 1e-5 (BPCO_CRITIC_LEARNING_RATE) is a full value model
)
GRAD_CLIP = 1.0
TRAIN_SEED = 17
MICRO_BATCH = 8  # sequences per forward pass with gradients

# The baseline: the standard PPO critic recipe, verl's defaults.
PPO_CLIP = 0.2  # symmetric ratio clip (Schulman et al. 2017)
GAE_LAMBDA = 0.95  # the critic target is the lambda-return at this lambda
GAMMA = 1.0
VALUE_CLIP = 0.5  # verl's cliprange_value
WHITEN_EPS = 1e-8

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."

METHOD = wai.BPCO()  # clip 0.2, gae_alpha 0.4, reward_range (0, 1), critic_warmup 15


# --------------------------------------------------------------------------
# Pure parts. No torch at module level: `--selftest` runs these on the CPU,
# and the Modal container imports this same file.
# --------------------------------------------------------------------------


def messages_for(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def gold_of(answer: str) -> str:
    """GSM8K ships the worked solution then `#### 72`. The gold is the tail."""
    return answer.split("####")[-1].strip().replace(",", "")


def outcome_of(text: str, gold: str) -> float:
    """1.0 when the final number matches the gold, else 0.0. A program, not a
    judge: `MathEqual` is sympy with a numeric and string fallback."""
    row = {
        "prompt": "",
        "final_text": text,
        "privileged": {"reference": gold},
        "scenario_id": "x",
        "rollout_index": 0,
    }
    return 1.0 if wai.verify.MathEqual()(row).get("reward") == 1 else 0.0


class StaleSampler:
    """Which policy version wrote the batch, and when the sampler is refreshed.

    The policy has a version, the number of optimizer steps applied to it.
    The sampler is a frozen copy taken at some version. At step ``k`` (1-based)
    the batch is drawn first; then, when ``k % lag == 0``, the sampler is
    refreshed from the policy *before* the step's update, so it holds
    version ``k - 1``; then the policy steps to version ``k``. Step ``k + 1``
    trains on a batch one update old and step ``k + lag`` on one ``lag``
    updates old. The very first block starts on-policy (there is no older
    policy to copy), and a critic-only warm-up step does not move the
    version, so its batch is on-policy too.
    """

    def __init__(self, lag: int = LAG):
        if lag < 1:
            raise ValueError(f"lag is a count of optimizer steps, 1 or more; got {lag}")
        self.lag = lag
        self.policy_version = 0
        self.sampler_version = 0
        self.refreshes = 0

    @property
    def staleness(self) -> int:
        """Updates between the policy that wrote the next batch and the one that trains on it."""
        return self.policy_version - self.sampler_version

    def refresh_due(self, step: int) -> bool:
        return step % self.lag == 0

    def mark_refresh(self) -> None:
        self.sampler_version = self.policy_version
        self.refreshes += 1

    def policy_stepped(self) -> None:
        self.policy_version += 1


def staleness_schedule(steps: int, lag: int, warmup: int = 0) -> list[int]:
    """The staleness of each step's batch, as the loop in `run_arm` produces it."""
    out: list[int] = []
    s = StaleSampler(lag)
    for step in range(1, steps + 1):
        out.append(s.staleness)
        if s.refresh_due(step):
            s.mark_refresh()
        if step > warmup:
            s.policy_stepped()
    return out


def torch_bound(z, method=METHOD):
    """`method.bound` in torch, the same formula, so the critic's gradient
    flows through the arctangent: `lo + (hi - lo) * (atan(z) / pi + 1/2)`."""
    import torch

    lo, hi = method.reward_range
    return lo + (hi - lo) * (torch.atan(z) / math.pi + 0.5)


def gae_lambda_returns(
    reward: float, values: list[float], lam: float = GAE_LAMBDA, gamma: float = GAMMA
) -> tuple[list[float], list[float]]:
    """Standard GAE for one trajectory with a terminal reward: `delta_t = r_t +
    gamma V_{t+1} - V_t`, `A_t = delta_t + gamma lambda A_{t+1}`, `V` past the
    end 0. Returns the advantages and the lambda-returns `A_t + V_t`, the
    baseline critic's target."""
    n = len(values)
    adv = [0.0] * n
    last = 0.0
    next_value = 0.0
    for t in range(n - 1, -1, -1):
        r_t = reward if t == n - 1 else 0.0
        delta = r_t + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
        next_value = values[t]
    return adv, [a + v for a, v in zip(adv, values)]


def whiten(groups: list[list[float]], eps: float = WHITEN_EPS) -> list[list[float]]:
    """Subtract the batch mean and divide by the batch standard deviation over
    every token of every trajectory (verl's `masked_whiten`, unbiased std)."""
    flat = [x for g in groups for x in g]
    if len(flat) < 2:  # a standard deviation needs two numbers
        return [[0.0 for _ in g] for g in groups]
    mean = statistics.fmean(flat)
    std = statistics.stdev(flat)
    return [[(x - mean) / (std + eps) for x in g] for g in groups]


def explained_variance(pairs: list[tuple[float, float]]) -> float | None:
    """`1 - Var(R - V) / Var(R)` over tokens (equation 10 of the paper); None
    when the rewards do not vary. Measured against the Monte Carlo return in
    both arms so the two critics are read on one scale."""
    if not pairs:
        return None
    rewards = [r for r, _ in pairs]
    var_r = statistics.pvariance(rewards)
    if var_r <= 0:
        return None
    errors = [r - v for r, v in pairs]
    return 1 - statistics.pvariance(errors) / var_r


def baseline_update(
    batch: list[dict], *, clip: float = PPO_CLIP, lam: float = GAE_LAMBDA
) -> Update:
    """THE BASELINE: the standard PPO critic recipe, in the `Update` shape.

    Per trajectory: the head's raw output is the value (no bound), GAE at
    `lam` = 0.95 and gamma 1 gives the advantage and the lambda-return is
    the critic's target; the advantages are whitened over the batch; the
    ratio against the rollout policy is clipped to `[1 - clip, 1 + clip]`.
    `coefficients[i][t]` is the clipped surrogate's gradient weight, `rho A`
    where the unclipped branch is the minimum and 0 where the clipped
    branch is, so the trainer's loss is `-(coef * logprob).sum() / N` in
    both arms. That is the exact gradient of `min(rho A, clip(rho) A)` at
    the point it is evaluated, and this recipe takes one step per batch,
    so it is the whole of PPO's clip (`selftest` checks it against autograd).
    """
    if not batch:
        raise ValueError("baseline_update: the batch is empty")
    raw_adv: list[list[float]] = []
    targets: list[list[float]] = []
    for i, traj in enumerate(batch):
        if traj.get("values") is None or len(traj["values"]) != len(traj["logprobs"]):
            raise ValueError(f"baseline_update: trajectory {i} needs one value per logprob")
        adv, ret = gae_lambda_returns(
            float(traj["reward"]), [float(v) for v in traj["values"]], lam
        )
        raw_adv.append(adv)
        targets.append(ret)
    advantages = whiten(raw_adv)
    coefficients: list[list[float]] = []
    n_action = n_clipped = 0
    sum_ratio = sum_adv = sum_raw = 0.0
    pairs: list[tuple[float, float]] = []
    cap = float(BPCO_LOG_RATIO_CAP)
    for traj, adv, raw in zip(batch, advantages, raw_adv):
        lp = [float(x) for x in traj["logprobs"]]
        behavior = traj.get("behavior_logprobs")
        mu_lp = lp if behavior is None else [float(x) for x in behavior]
        coef = [0.0] * len(lp)
        for t, a in enumerate(adv):
            ratio = math.exp(min(cap, max(-cap, lp[t] - mu_lp[t])))
            if a > 0:
                active = ratio <= 1 + clip
            elif a < 0:
                active = ratio >= 1 - clip
            else:
                active = True
            coef[t] = ratio * a if active else 0.0
            n_action += 1
            n_clipped += 0 if active else 1
            sum_ratio += ratio
            sum_adv += a
            sum_raw += raw[t]
            pairs.append((float(traj["reward"]), float(traj["values"][t])))
        coefficients.append(coef)
    stats = {
        "admitted_share": 1.0,
        "clipped_token_share": n_clipped / n_action,
        "mean_ratio": sum_ratio / n_action,
        "mean_advantage": sum_adv / n_action,
        "mean_raw_advantage": sum_raw / n_action,
    }
    ev = explained_variance(pairs)
    notes: list[str] = []
    if ev is None:
        notes.append("explained variance not computed: the batch's rewards do not vary")
    else:
        stats["explained_variance"] = ev
    return Update(
        method="ppo_critic",
        coefficients=coefficients,
        advantages=advantages,
        admitted=[True] * len(batch),
        value_targets=targets,
        stats=stats,
        notes=notes,
    )


def policy_loss(logprobs: list, coefficients: list[list[float]]):
    """`-(coef * logprob).sum() / N`: the coefficients constant, the log-probs
    live. The `1/N` is the batch mean the `Update` contract leaves to the
    trainer (seq-mean-token-sum, the release's aggregation). One function for
    both arms; the arms differ in who wrote the coefficients."""
    import torch

    total = None
    for lp, coef in zip(logprobs, coefficients):
        term = (torch.tensor(coef, dtype=lp.dtype, device=lp.device) * lp).sum()
        total = term if total is None else total + term
    return -total / len(coefficients)


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `delta_report` read: binary
    `reward`, one row per sample, grouped by task. The training reward rides
    along as a marker so `delta_report(proxy=)` can name it."""
    rows: list[dict] = []
    for task, texts in zip(holdout, replies):
        gold = gold_of(task["answer"])
        for i, text in enumerate(texts):
            r = outcome_of(text, gold)
            rows.append(
                {
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": r,
                    "markers": {"gsm8k_outcome": r},
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

app = modal.App("whileai-recipe-bpco-bounded-critic")

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
    # wai.BPCO is newer than the published wheel: the checkout's own package
    # rides along and PYTHONPATH puts it ahead of site-packages. Harmless
    # once the release carries it.
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/root/whileai_src",
        }
    )
    .add_local_dir(str(WHILEAI_SRC), "/root/whileai_src/whileai", ignore=["**/__pycache__/**"])
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _eos_ids(model, tokenizer) -> set[int]:
    eos = model.generation_config.eos_token_id
    ids = set(eos if isinstance(eos, list) else [eos])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ids.add(tokenizer.pad_token_id)
    return {int(i) for i in ids if i is not None}


def _sample(model, tokenizer, questions, *, n, temperature, max_new_tokens, batch=16):
    """`n` replies per question, batched, as token ids and text. Sampling is
    the model's own distribution at `temperature`: top-p 1, top-k off, no
    repetition penalty (Qwen's generation_config sets all three, and a
    truncated sampler has no common support with the policy it is compared
    against). Each reply carries its prompt ids, its completion ids up to and
    including the stop token, and whether it hit the cap."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    model.eval()
    tokenizer.padding_side = "left"
    eos = _eos_ids(model, tokenizer)
    texts = [
        tokenizer.apply_chat_template(messages_for(q), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    out: list[list[dict]] = []
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
        for i in range(len(chunk)):
            mask = enc["attention_mask"][i].bool()
            prompt_ids = enc["input_ids"][i][mask].tolist()
            replies = []
            for j in range(n):
                row = gen[i * n + j, prompt_len:].tolist()
                completion: list[int] = []
                truncated = True
                for tok in row:
                    completion.append(tok)
                    if tok in eos:
                        truncated = False
                        break
                replies.append(
                    {
                        "prompt_ids": prompt_ids,
                        "completion_ids": completion,
                        "text": tokenizer.decode(completion, skip_special_tokens=True),
                        "truncated": truncated,
                    }
                )
            out.append(replies)
    model.train()
    return out


def _forward(model, tokenizer, trajs, *, hidden: bool, grad: bool):
    """One teacher-forced pass over `trajs` (prompt + completion, right-padded).
    Returns, per trajectory, the log-prob of every completion token and, when
    `hidden`, the trunk's last hidden state at the position that predicted
    it (detached: the critic never trains the trunk)."""
    import torch

    ids = [t["prompt_ids"] + t["completion_ids"] for t in trajs]
    width = max(len(x) for x in ids)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
    attention = torch.zeros((len(ids), width), dtype=torch.long)
    for i, seq in enumerate(ids):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention[i, : len(seq)] = 1
    input_ids = input_ids.to(model.device)
    attention = attention.to(model.device)
    with torch.set_grad_enabled(grad):
        out = model(
            input_ids=input_ids,
            attention_mask=attention,
            output_hidden_states=hidden,
            use_cache=False,
        )
        logprobs, features = [], []
        for i, t in enumerate(trajs):
            p, c = len(t["prompt_ids"]), len(t["completion_ids"])
            logits = out.logits[i, p - 1 : p + c - 1].float()
            targets = input_ids[i, p : p + c]
            logprobs.append(
                torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
            )
            if hidden:
                features.append(out.hidden_states[-1][i, p - 1 : p + c - 1].float().detach())
    return logprobs, features


def _lora_state(model) -> dict:
    """A frozen copy of the adapter: every trainable tensor, cloned."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _load_lora_state(model, state: dict) -> None:
    import torch

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n])


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
    critic_warmup: int | None = None,
    prompts_per_step: int = PROMPTS_PER_STEP,
    lag: int = LAG,
    max_new_tokens: int = MAX_NEW_TOKENS,
    eval_samples: int = 4,
    base_runs: int = 0,
    seed: int = 0,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again."""
    import random
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    sys.path.insert(0, "/root")
    from recipe_mod import (
        CRITIC_LR,
        EVAL_TEMPERATURE,
        GRAD_CLIP,
        LORA_ALPHA,
        LORA_RANK,
        METHOD,
        MICRO_BATCH,
        POLICY_LR,
        TRAIN_SEED,
        VALUE_CLIP,
        StaleSampler,
        _forward,
        _load_lora_state,
        _lora_state,
        _sample,
        baseline_update,
        gold_of,
        graded_rows,
        mean_length,
        outcome_of,
        policy_loss,
        torch_bound,
    )

    import whileai as wai
    from whileai.simulations.training import training_run

    if not hasattr(wai, "BPCO"):
        raise RuntimeError(f"the container imported {wai.__file__} without BPCO; check PYTHONPATH")
    print(f"whileai in the container: {wai.__file__}")

    started = time.time()
    bounded = arm == "recipe"
    warmup = int(METHOD.critic_warmup if critic_warmup is None else critic_warmup) if bounded else 0
    method = METHOD
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    config = {
        "arm": arm,
        "bounded_critic": bounded,
        "method": str(method)
        if bounded
        else "ppo critic: unbounded head, GAE 0.95, whitened, "
        f"clip {PPO_CLIP}, value clip {VALUE_CLIP}, no warm-up",
        "base_model": base_model,
        "steps": steps,
        "critic_warmup": warmup,
        "prompts_per_step": prompts_per_step,
        "rollouts_per_prompt": 1,
        "sampler_lag": lag,
        "policy_lr": POLICY_LR,
        "critic_lr": CRITIC_LR,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "max_new_tokens": max_new_tokens,
        "rollout_temperature": method.temperature,
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
            trainer="bpco-lora" if bounded else "ppo-critic-lora",
            total_steps=warmup + steps,
            config=config,
        )
        print(f"dashboard: {run.url}")

    questions = [t["question"] for t in holdout]

    def evaluate(m) -> list[dict]:
        replies = _sample(
            m,
            tokenizer,
            questions,
            n=eval_samples,
            temperature=EVAL_TEMPERATURE,
            max_new_tokens=max_new_tokens,
        )
        return graded_rows(holdout, [[r["text"] for r in rs] for rs in replies])

    # The base is evaluated EVAL_RUNS times, not once. The spread across those
    # re-runs is the eval's own noise, and a delta smaller than it is not a
    # result (Lambert 2025, chapter Evaluation). Only one arm pays for this.
    base_rows: list[list[dict]] = []
    for i in range(base_runs):
        t0 = time.time()
        rows = evaluate(model)
        base_rows.append(rows)
        print(f"base run {i + 1}/{base_runs}: {wai.pass_at(rows)} ({time.time() - t0:.0f}s)")

    # Seed right before the adapter and the head are built, so both arms draw
    # the same init whatever ran before (the base evals advance the RNG).
    set_seed(TRAIN_SEED)
    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
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
    model.print_trainable_parameters()
    head = torch.nn.Linear(model.config.hidden_size, 1, dtype=torch.float32).to(model.device)
    torch.nn.init.zeros_(head.bias)
    policy_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=POLICY_LR)
    critic_opt = torch.optim.AdamW(head.parameters(), lr=CRITIC_LR)

    stale = StaleSampler(lag)
    sampler_state = _lora_state(model)
    rng = random.Random(seed)
    order: list[dict] = []

    def next_batch() -> list[dict]:
        nonlocal order
        batch: list[dict] = []
        while len(batch) < prompts_per_step:
            if not order:
                order = list(train_tasks)
                rng.shuffle(order)
            batch.append(order.pop())
        return batch

    curves: list[dict] = []
    last_batch: list[dict] = []
    total_steps = warmup + steps
    for step in range(1, total_steps + 1):
        t0 = time.time()
        tasks = next_batch()
        staleness = stale.staleness

        # 1. Rollouts from the stale sampler, and its own log-probs of them
        #    (a teacher-forced pass under the sampler's weights: what a
        #    serving stack returns as token_logprobs).
        live_state = _lora_state(model)
        _load_lora_state(model, sampler_state)
        replies = _sample(
            model,
            tokenizer,
            [t["question"] for t in tasks],
            n=1,
            temperature=method.temperature,
            max_new_tokens=max_new_tokens,
            batch=prompts_per_step,
        )
        trajs = [r[0] for r in replies]
        for traj, task in zip(trajs, tasks):
            traj["reward"] = outcome_of(traj["text"], gold_of(task["answer"]))
        for start in range(0, len(trajs), 2 * MICRO_BATCH):
            chunk = trajs[start : start + 2 * MICRO_BATCH]
            lps, _ = _forward(model, tokenizer, chunk, hidden=False, grad=False)
            for traj, lp in zip(chunk, lps):
                traj["behavior_logprobs"] = lp.tolist()
        _load_lora_state(model, live_state)
        del live_state

        # 2. The current policy's log-probs, and the critic's value at every
        #    token (bounded through the arctangent in the recipe arm, raw in
        #    the baseline). The hidden features are kept for the critic step.
        feats: list = []
        for start in range(0, len(trajs), 2 * MICRO_BATCH):
            chunk = trajs[start : start + 2 * MICRO_BATCH]
            lps, hs = _forward(model, tokenizer, chunk, hidden=True, grad=False)
            for traj, lp, h in zip(chunk, lps, hs):
                traj["logprobs"] = lp.tolist()
                with torch.no_grad():
                    z = head(h).squeeze(-1)
                    traj["values"] = (torch_bound(z, method) if bounded else z).tolist()
                feats.append(h)

        # 3. The update rule: the paper's, or the standard critic recipe.
        batch = [
            {
                "reward": t["reward"],
                "logprobs": t["logprobs"],
                "behavior_logprobs": t["behavior_logprobs"],
                "values": t["values"],
            }
            for t in trajs
        ]
        update = method.update(batch) if bounded else baseline_update(batch)

        # 4. Refresh the sampler from the policy every `lag` steps, before the
        #    step moves it, so the next block trains on a 1-to-lag-old batch.
        if stale.refresh_due(step):
            sampler_state = _lora_state(model)
            stale.mark_refresh()

        # 5. The policy step (skipped during the critic warm-up).
        loss_value = grad_norm = float("nan")
        if step > warmup:
            policy_opt.zero_grad(set_to_none=True)
            total = 0.0
            for start in range(0, len(trajs), MICRO_BATCH):
                chunk = trajs[start : start + MICRO_BATCH]
                coefs = update.coefficients[start : start + MICRO_BATCH]
                lps, _ = _forward(model, tokenizer, chunk, hidden=False, grad=True)
                loss = policy_loss(lps, coefs) * (len(chunk) / len(trajs))
                loss.backward()
                total += float(loss.item())
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
                )
            )
            policy_opt.step()
            stale.policy_stepped()
            loss_value = total

        # 6. The critic step, on the features from pass 2: MSE to the Monte
        #    Carlo return through the bound (recipe), or verl's clipped value
        #    loss to the lambda-return (baseline).
        h_all = torch.cat(feats, dim=0)
        target = torch.tensor(
            [x for row in update.value_targets for x in row],
            dtype=torch.float32,
            device=h_all.device,
        )
        old = torch.tensor(
            [x for t in trajs for x in t["values"]], dtype=torch.float32, device=h_all.device
        )
        z = head(h_all).squeeze(-1)
        if bounded:
            critic_loss = ((torch_bound(z, method) - target) ** 2).mean()
        else:
            clipped = old + torch.clamp(z - old, -VALUE_CLIP, VALUE_CLIP)
            critic_loss = 0.5 * torch.max((z - target) ** 2, (clipped - target) ** 2).mean()
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()
        del feats, h_all

        # 7. The log.
        point = {
            "step": step,
            "policy_step": max(0, step - warmup),
            "staleness": staleness,
            "mean_reward": statistics.fmean(t["reward"] for t in trajs),
            "completion_tokens": statistics.fmean(len(t["completion_ids"]) for t in trajs),
            "truncated_share": statistics.fmean(1.0 if t["truncated"] else 0.0 for t in trajs),
            "clipped_token_share": update.stats.get("clipped_token_share", float("nan")),
            "mean_ratio": update.stats.get("mean_ratio", float("nan")),
            "mean_advantage": update.stats.get("mean_advantage", float("nan")),
            "explained_variance": update.stats.get("explained_variance", float("nan")),
            "mean_value": statistics.fmean(x for t in trajs for x in t["values"]),
            "critic_loss": float(critic_loss.item()),
            "policy_loss": loss_value,
            "grad_norm": grad_norm,
            "seconds": time.time() - t0,
        }
        if bounded:
            point["mean_lambda"] = update.stats.get("mean_lambda", float("nan"))
            point["mean_clip_range"] = update.stats.get("mean_clip_range", float("nan"))
        curves.append(point)
        print(
            f"{arm} step {step}/{total_steps}"
            + (" (warm-up)" if step <= warmup else "")
            + f" stale {staleness} reward {point['mean_reward']:.3f}"
            f" clipped {point['clipped_token_share']:.4f} ratio {point['mean_ratio']:.4f}"
            f" adv {point['mean_advantage']:+.4f} ev {point['explained_variance']:+.3f}"
            f" len {point['completion_tokens']:.0f} vloss {point['critic_loss']:.4f}"
            f" ploss {loss_value:+.4f} gnorm {grad_norm:.3f} {point['seconds']:.0f}s"
        )
        for note in update.notes:
            print(f"  {note}")
        if run is not None:
            run.log(step, **{k: v for k, v in point.items() if k != "step"})
            run.progress(step, total_steps)
        last_batch = [
            {
                "prompt": task["question"],
                "final_text": t["text"],
                "reward": t["reward"],
                "scenario_id": task["scenario_id"],
                "rollout_index": 0,
            }
            for t, task in zip(trajs, tasks)
        ]

    t0 = time.time()
    after_rows = evaluate(model)
    after = wai.pass_at(after_rows)
    print(f"{arm}: {after} ({time.time() - t0:.0f}s)")

    # What the reward actually paid for in the last training batch (chapter
    # Over-optimization). Nothing is endorsed: the reward is the answer being
    # right, and any surface feature that correlates with it is the thing to
    # be suspicious of.
    scan = wai.hack_scan(last_batch) if last_batch else {}
    hack_top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
    hack_scan_top = hack_top.get("name", "") if isinstance(hack_top, dict) else str(hack_top)
    print(f"{arm} hack scan: top feature {hack_scan_top or 'none above the floor'}")

    out_dir = os.path.join(VOLUME_ROOT, run_name)
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(head.state_dict(), os.path.join(out_dir, "value_head.pt"))
    with open(os.path.join(out_dir, "curves.json"), "w") as f:
        json.dump(curves, f)
    with open(os.path.join(out_dir, "eval_rows.json"), "w") as f:
        json.dump({"base_runs": base_rows, "after_rows": after_rows}, f)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
        "bounded_critic": bounded,
        "pass_at_1": after.pass_at_1,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
        "critic_warmup": warmup,
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    return {
        "arm": arm,
        "base_runs": base_rows,
        "after_rows": after_rows,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
        "critic_warmup": warmup,
        "length_after": mean_length(after_rows),
        "hack_scan_top": hack_scan_top,
        "curves": curves,
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


def curve_summary(curves: list[dict]) -> dict:
    """The per-step log folded to what the README quotes: first and last
    policy-step values of the series a reader asks about."""
    policy = [c for c in curves if c["policy_step"] > 0] or curves
    if not policy:
        return {}

    def series(key: str) -> list[float]:
        return [round(float(c[key]), 4) for c in policy if c.get(key) == c.get(key)]

    first, last = policy[0], policy[-1]
    quarter = max(1, len(policy) // 4)
    return {
        "policy_steps": len(policy),
        "mean_reward_first_quarter": round(statistics.fmean(series("mean_reward")[:quarter]), 4),
        "mean_reward_last_quarter": round(statistics.fmean(series("mean_reward")[-quarter:]), 4),
        "clipped_token_share": series("clipped_token_share"),
        "explained_variance": series("explained_variance"),
        "mean_ratio_first_last": [first["mean_ratio"], last["mean_ratio"]],
        "completion_tokens_first_last": [first["completion_tokens"], last["completion_tokens"]],
        "staleness": [int(c["staleness"]) for c in policy],
        "seconds_per_step": round(statistics.fmean(series("seconds")), 1),
    }


USD_PER_HOUR = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}


def assemble(
    results: dict, outs: dict[str, dict], arms: list[str], *, run_std: float, today: str
) -> dict:
    """Fold the arms' returns into `results`: the noise floor, each arm's
    score, the paired delta with its verdict, the checks. Pure, so the
    selftest runs it on fake returns and a slip here never costs a GPU hour
    (round 1 of this recipe lost its eval rows to one)."""
    checks = results["checks"]
    arm_rows: dict[str, list[dict]] = {}
    usd_per_hour = USD_PER_HOUR.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""
    for arm, out in outs.items():
        gpu_minutes += out["gpu_minutes"]
        run_url = out["run_url"] or run_url
        if out["base_runs"]:
            base_runs = out["base_runs"]
            noise = wai.eval_variance(*base_runs)
            # one base run (a smoke) has no spread; the floor is then unknown, 0
            run_std = float(noise["run_std"] or 0.0)
            print(f"eval noise over {len(base_runs)} base runs: run_std {run_std:.4f}")
            arm_rows["base"] = base_runs[0]
            results["arms"]["base"] = {**summarize(base_runs[0]), "steps": 0, "gpu_minutes": 0}
            checks["run_std"] = run_std
            checks["run_std_runs"] = int(noise["n_runs"])
            checks["length_before"] = mean_length(base_runs[0])
        arm_rows[arm] = out["after_rows"]
        results["arms"][arm] = {
            **summarize(out["after_rows"]),
            "steps": out["steps"],
            "critic_warmup": out["critic_warmup"],
            "gpu_minutes": round(out["gpu_minutes"], 1),
            "usd": round(out["gpu_minutes"] / 60.0 * usd_per_hour, 2),
            "curve": curve_summary(out["curves"]),
        }
        checks["length_after"][arm] = out["length_after"]
        checks["hack_scan_top"][arm] = out["hack_scan_top"]

    if "baseline" in arm_rows and "recipe" in arm_rows:
        # run_std makes "moved" mean bigger than the eval's own re-run noise;
        # proxy names the training reward, the same binary program here.
        # `wai.compare` is the front door of `delta_report`.
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
        results["verified"] = today
        results.pop("partial_run", None)
        print(d)
    else:
        results["partial_run"] = f"{today}: {', '.join(arms)} only"
        print(f"one arm only ({', '.join(arms)}): delta and verified left as they were")

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    return results


def selftest() -> None:
    """The arithmetic the change lives in, on the CPU: the reward parser, the
    stale sampler's bookkeeping, the torch bound against `BPCO.bound`, the
    recipe loss against `BPCO.update`, and the baseline's GAE, whitening and
    clip against hand numbers and autograd. No GPU, no key, no download."""
    import torch

    # The grader is a program, not a judge.
    assert gold_of("She has 3 left.\n#### 1,234") == "1234"
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("grader: MathEqual reads \\boxed{} and the last number; gold is the #### tail")

    # The stale sampler: refreshed every LAG steps, before the step's update,
    # so a batch is 1 to LAG updates old once the first block has passed.
    assert staleness_schedule(12, 4) == [0, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4]
    assert staleness_schedule(6, 4, warmup=4) == [0, 0, 0, 0, 0, 1]  # warm-up: policy still
    assert staleness_schedule(8, 1) == [0] + [1] * 7  # lag 1 is "one update old", never 0 later
    print(f"stale sampler at lag {LAG}: staleness {staleness_schedule(12, LAG)}")

    # The torch bound is BPCO.bound, and its gradient is the paper's dV/dz.
    lo, hi = METHOD.reward_range
    zs = torch.linspace(-60.0, 60.0, 2401, dtype=torch.float64)
    ours = torch_bound(zs, METHOD)
    theirs = torch.tensor([METHOD.bound(z) for z in zs.tolist()], dtype=torch.float64)
    assert torch.allclose(ours, theirs, atol=1e-12, rtol=0.0)
    assert torch.allclose(torch_bound(zs.float(), METHOD), theirs.float(), atol=2e-6, rtol=0.0)
    z = torch.tensor([-3.0, 0.0, 0.7, 25.0], dtype=torch.float64, requires_grad=True)
    torch_bound(z, METHOD).sum().backward()
    expect = (hi - lo) / (math.pi * (1 + z.detach() ** 2))
    assert torch.allclose(z.grad, expect, atol=1e-12)
    assert float(ours.min()) > lo and float(ours.max()) < hi
    assert abs(float(torch_bound(torch.tensor(0.0), METHOD)) - (lo + hi) / 2) < 1e-7
    print(f"torch bound == BPCO.bound on 2401 points in [-60, 60]; bound(0) = {(lo + hi) / 2}")

    # The recipe loss is -(coef * logprob).sum() / N with BPCO's coefficients
    # held constant: the value and the gradient, on a hand-made batch that is
    # off-policy enough for the DPPO range to clip something.
    values = [METHOD.bound(v) for v in (-1.0, 0.0, 1.0)]
    batch = [
        {
            "reward": 1.0,
            "logprobs": [-0.5, -1.0, -0.2],
            "behavior_logprobs": [-0.6, -1.0, -3.0],
            "values": values,
        },
        {
            "reward": 0.0,
            "logprobs": [-0.7, -0.3],
            "behavior_logprobs": [-0.7, -0.1],
            "values": values[:2],
        },
    ]
    update = METHOD.update(batch)
    assert update.value_targets == [[1.0, 1.0, 1.0], [0.0, 0.0]]
    assert update.stats["clipped_token_share"] > 0, update
    logps = [torch.tensor(t["logprobs"], dtype=torch.float64, requires_grad=True) for t in batch]
    loss = policy_loss(logps, update.coefficients)
    by_hand = -sum(
        c * lp for cs, lps in zip(update.coefficients, batch) for c, lp in zip(cs, lps["logprobs"])
    )
    by_hand /= len(batch)
    assert abs(loss.item() - by_hand) < 1e-12, (loss.item(), by_hand)
    loss.backward()
    for lp, coefs in zip(logps, update.coefficients):
        assert torch.allclose(lp.grad, -torch.tensor(coefs, dtype=torch.float64) / len(batch))
    print(f"recipe loss {loss.item():+.6f} == -(coef * logp).sum() / N; {update}")
    ev = explained_variance(
        [(t["reward"], v) for t in batch for v in t["values"]],
    )
    assert abs(ev - update.stats["explained_variance"]) < 1e-12
    assert update.value_targets is not None

    # The baseline by hand. Trajectory A: reward 1, values [0.5, 0.2, 0.8]:
    #   t=2: delta = 1 - 0.8 = 0.2,            A = 0.2
    #   t=1: delta = 0.8 - 0.2 = 0.6,          A = 0.6 + 0.95 * 0.2 = 0.79
    #   t=0: delta = 0.2 - 0.5 = -0.3,         A = -0.3 + 0.95 * 0.79 = 0.4505
    # lambda-returns A + V = [0.9505, 0.99, 1.0].
    adv, ret = gae_lambda_returns(1.0, [0.5, 0.2, 0.8])
    assert [round(a, 6) for a in adv] == [0.4505, 0.79, 0.2], adv
    assert [round(r, 6) for r in ret] == [0.9505, 0.99, 1.0], ret
    # Trajectory B: reward 0, values [0.5, 0.5]: A = [-0.475, -0.5].
    adv_b, ret_b = gae_lambda_returns(0.0, [0.5, 0.5])
    assert [round(a, 6) for a in adv_b] == [-0.475, -0.5]
    assert [round(r, 6) for r in ret_b] == [0.025, 0.0]
    white = whiten([adv, adv_b])
    flat = [x for g in white for x in g]
    assert abs(statistics.fmean(flat)) < 1e-9 and abs(statistics.stdev(flat) - 1) < 1e-6
    print(
        f"baseline GAE(0.95): A = {[round(a, 4) for a in adv]}, returns {[round(r, 4) for r in ret]}"
    )

    # PPO clip 0.2 on the ratio: token 2 of A has ratio e^(-0.2 + 3.0) >> 1.2
    # with a positive advantage, so it clips to 0; token 1 of B has ratio
    # e^(-0.3 + 0.1) = 0.82 with a negative advantage, inside the range, so
    # it is rho * A; every coefficient equals autograd on the surrogate.
    base_batch = [
        {
            "reward": 1.0,
            "logprobs": [-0.5, -1.0, -0.2],
            "behavior_logprobs": [-0.6, -1.0, -3.0],
            "values": [0.5, 0.2, 0.8],
        },
        {
            "reward": 0.0,
            "logprobs": [-0.7, -0.3],
            "behavior_logprobs": [-0.7, -0.1],
            "values": [0.5, 0.5],
        },
    ]
    bu = baseline_update(base_batch)
    assert bu.value_targets == [ret, ret_b]
    assert bu.advantages == white
    assert bu.coefficients[0][2] == 0.0 and bu.coefficients[1][1] != 0.0
    assert bu.stats["clipped_token_share"] == 1 / 5
    for traj, adv_row, coefs in zip(base_batch, bu.advantages, bu.coefficients):
        lp = torch.tensor(traj["logprobs"], dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(traj["behavior_logprobs"], dtype=torch.float64)
        a = torch.tensor(adv_row, dtype=torch.float64)
        ratio = torch.exp(lp - mu)
        surrogate = torch.min(ratio * a, torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * a).sum()
        surrogate.backward()
        assert torch.allclose(lp.grad, torch.tensor(coefs, dtype=torch.float64)), (lp.grad, coefs)
    # A whitened advantage of exactly zero would be the ambiguous branch; none here.
    assert all(x != 0.0 for row in bu.advantages for x in row)
    print(
        f"baseline clip {PPO_CLIP}: clipped share {bu.stats['clipped_token_share']:.2f}, "
        "coefficients == d/dlogp of min(rho A, clip(rho) A)"
    )
    # verl's clipped value loss never binds on the first (and only) step per
    # batch: old == new value at the point of evaluation.
    old = torch.tensor([0.5, 0.2, 0.8])
    clipped = old + torch.clamp(old - old, -VALUE_CLIP, VALUE_CLIP)
    assert torch.equal(clipped, old)
    _selftest_assemble()
    print("selftest ok")


def _selftest_assemble() -> None:
    """`assemble` on fake returns: the whole local path after the GPU work,
    down to results.json's keys, without a GPU."""
    import random

    rng = random.Random(0)

    def rows(p: float) -> list[dict]:
        return [
            {
                "prompt": f"q{i}",
                "final_text": "x" * rng.randint(50, 300),
                "reward": r,
                "markers": {"gsm8k_outcome": r},
                "scenario_id": f"test-{i}",
                "rollout_index": j,
            }
            for i in range(30)
            for j in range(4)
            for r in [1.0 if rng.random() < p else 0.0]
        ]

    def out(arm: str, p: float, base: bool) -> dict:
        return {
            "arm": arm,
            "base_runs": [rows(0.4) for _ in range(EVAL_RUNS)] if base else [],
            "after_rows": rows(p),
            "gpu_minutes": 10.0,
            "steps": 2,
            "critic_warmup": 1 if arm == "recipe" else 0,
            "length_after": 100.0,
            "hack_scan_top": "",
            "curves": [
                {
                    "step": s,
                    "policy_step": max(0, s - 1),
                    "staleness": 0,
                    "mean_reward": 0.5,
                    "completion_tokens": 100.0,
                    "clipped_token_share": 0.0,
                    "mean_ratio": 1.0,
                    "mean_advantage": 0.0,
                    "explained_variance": float("nan"),
                    "seconds": 1.0,
                }
                for s in (1, 2, 3)
            ],
            "run_url": "",
        }

    results = {"arms": {}, "checks": {"length_after": {}, "hack_scan_top": {}}}
    outs = {"baseline": out("baseline", 0.45, True), "recipe": out("recipe", 0.5, False)}
    assemble(results, outs, ["baseline", "recipe"], run_std=0.0, today="1970-01-01")
    assert set(results["arms"]) == {"base", "baseline", "recipe"}
    assert results["delta"]["verdict"] == "unresolved"  # one seed per arm
    assert results["checks"]["train_seeds"] == {"baseline": 1, "recipe": 1}
    assert results["checks"]["run_std_runs"] == EVAL_RUNS and results["usd"] > 0
    assert results["verified"] == "1970-01-01"
    print("assemble on fake returns: results.json keys present, verdict unresolved at one seed")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40, help="policy steps after any warm-up")
    ap.add_argument(
        "--critic-warmup",
        type=int,
        default=None,
        help=f"critic-only steps before the recipe arm's policy moves (default {METHOD.critic_warmup}, BPCO's)",
    )
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=120)
    ap.add_argument("--prompts-per-step", type=int, default=PROMPTS_PER_STEP)
    ap.add_argument(
        "--lag", type=int, default=LAG, help="optimizer steps between sampler refreshes"
    )
    ap.add_argument("--base-runs", type=int, default=EVAL_RUNS, help="re-runs of the base eval")
    ap.add_argument("--selftest", action="store_true", help="the arithmetic, offline")
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
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(
        {
            "recipe": HERE.name,
            "title": "BPCO bounded critic: a critic that can only say a number the reward could be",
            "paper": "https://arxiv.org/abs/2608.23566",
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
    checks.setdefault("hack_scan_top", {})
    if not isinstance(checks["hack_scan_top"], dict):
        checks["hack_scan_top"] = {}
    run_std = float(checks.get("run_std") or 0.0)
    today = date.today().isoformat()

    with modal.enable_output(), app.run():
        # Both arms at once, one container each; the base re-runs ride with
        # the baseline arm, the shorter one (no warm-up).
        calls = {
            arm: run_arm.spawn(
                arm,
                train_tasks,
                holdout,
                f"bpco-bounded-critic-{arm}-{today}",
                steps=args.steps,
                critic_warmup=args.critic_warmup,
                prompts_per_step=args.prompts_per_step,
                lag=args.lag,
                eval_samples=args.k,
                base_runs=args.base_runs if i == 0 else 0,
                seed=args.seed,
            )
            for i, arm in enumerate(arms)
        }
        outs = {arm: call.get() for arm, call in calls.items()}

    # The raw returns first, so a local slip after the GPU work costs nothing.
    (HERE / "last_run.json").write_text(json.dumps(outs))
    assemble(results, outs, arms, run_std=run_std, today=today)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    (HERE / "last_run.json").unlink()  # results.json is on disk; the raw returns did their job
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
