"""Softmax advantage: the group advantage is a softmax over rewards, not a z-score.

    python recipe.py --selftest   # the advantage, the sign test, the grader, the pin; offline, no GPU
    python recipe.py --pilot      # one short recipe arm on Modal: the override reaches the loss
    python recipe.py              # base x3, three arms x three seeds, results.json, platform post
    python recipe.py --reuse      # results.json again from .cache/, no GPU

SoftmaxGRPO (Hernandez et al., arXiv:2608.09271) keeps everything in GRPO
except the advantage. GRPO z-scores the rewards inside a group of ``M``
rollouts, ``A_i = (r_i - mean) / std``, and under a binary reward that
denominator is what decides how hard a prompt pushes: a group with one wrong
answer out of eight hands that answer -2.47, a group with one right answer
hands it +2.47. The paper's complaint is that the easy prompt, whose one
failure is the least informative rollout in the batch, gets the same
magnitude as the hard prompt's one success. Equation 1 of the paper replaces
the z-score with a temperature-scaled softmax over the group's rewards:

    w_i = exp(r_i / tau) / sum_j exp(r_j / tau)
    A_i = M * w_i - 1

The advantages still sum to zero, so a unanimous group makes no update, but
the magnitude is bounded: a wrong rollout can never be pushed harder than -1,
and the ``c`` right rollouts of a group share ``M - c`` units of push between
them, so the lone success on a hard prompt gets +7 at ``M = 8``. The paper's
temperature for GSM8K is 0.1 (section 5), and at that temperature the binary
case is, to four decimals, ``A_right = (M - c) / c`` and ``A_wrong = -1``.

Three arms, one holdout, three training seeds each:

  baseline  GRPO as TRL ships it: z-scored group advantage
  recipe    the same run with equation 1 in place of the z-score
  random    GRPO paid a coin flip (Bernoulli 0.5) instead of the verifier,
            same steps and seeds. Shao et al. 2025 (arXiv:2506.10947) showed
            random rewards move MATH on Qwen2.5 bases, and the book says to
            be suspicious of every RLVR gain on a Qwen base for that reason
            (Lambert 2025, chapter Evaluation). The gain the reward buys is
            the delta over this arm, not the delta over the untrained base.

Shape of the run:
  1. data():      GSM8K train for prompts; a 300-problem holdout drawn from the
                  test split by the hash of the problem text and pinned by
                  content hash (holdout.sha256) before any training
  2. eval_base(): the untrained base, k samples a task, THREE times -> run_std
  3. run_arm():   TRL GRPOTrainer + LoRA on Modal, one (arm, seed) per call
  4. compare():   every arm against the base, the recipe against the baseline
                  and against the random arm, paired over problems with the
                  re-run band, the between-seed term, an exact sign test and
                  the tie count
  5. results.json, the platform post, the checks the README table reads
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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
BOOK = "Reinforcement Learning"  # the group-relative advantage and what its scale does
# The training reward is the target: the same binary MathEqual check against
# the GSM8K gold on both, so there is no proxy to over-optimize. The random
# arm's reward is a coin flip and is named in its rows, not as a proxy.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)
SEEDS = (17, 18, 19)  # training seeds per arm; MIN_TRAIN_SEEDS is 2, the headline wants 3
TAU = 0.1  # the paper's temperature for GSM8K (section 5)
# The paper caps completions at 256 tokens. On this base at temperature 1.0
# that cut 48-50% of replies in three re-runs (pass@1 0.24-0.25), so the eval
# measured whether a reply finished and any arm could win by getting short.
# 1024 is the cap here; the base's truncated share at it is in the Checks.
MAX_COMPLETION = 1024
EPS_LOW, EPS_HIGH = 0.20, 0.28  # the paper's clip range; inert at one update per batch
BETA = 1e-3  # the paper's KL coefficient
RANDOM_P = 0.5  # Shao et al. 2025: r ~ Bernoulli(0.5), independent of the completion
USD_PER_HOUR = {"A10G": 1.10, "L40S": 1.95, "H100": 3.95}
STOP_AT_USD = 10.0  # report and stop before the cap

ARMS = {
    "baseline": {"tau": None, "random_reward": False},
    "recipe": {"tau": TAU, "random_reward": False},
    "random": {"tau": None, "random_reward": True},
}

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."


# --------------------------------------------------------------------------
# Pure functions. No torch: `--selftest` runs them on hand-written groups,
# and the Modal container imports this same file.
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


def softmax_advantages(rewards: list[float], k: int, tau: float = TAU) -> list[float]:
    """THE ONE CHANGE, equation 1: ``A_i = k * softmax(r / tau)_i - 1`` per
    group of ``k`` contiguous rollouts. Sums to zero within every group, so a
    unanimous group makes no update, and no rollout goes below -1."""
    if k < 1:
        raise ValueError("k is the group size, 1 or more")
    if tau <= 0:
        raise ValueError("tau is a temperature, above zero")
    out: list[float] = []
    for start in range(0, len(rewards), k):
        group = rewards[start : start + k]
        top = max(group)
        weights = [math.exp((r - top) / tau) for r in group]  # shifted: no overflow at tau 0.1
        total = sum(weights)
        out.extend(len(group) * w / total - 1.0 for w in weights)
    return out


def grpo_advantages(rewards: list[float], k: int) -> list[float]:
    """TRL's default, the baseline arm: ``(r - mean) / (std + 1e-4)`` with the
    sample standard deviation, as ``torch.std`` computes it."""
    out: list[float] = []
    for start in range(0, len(rewards), k):
        group = rewards[start : start + k]
        mean = statistics.fmean(group)
        std = statistics.stdev(group) if len(group) > 1 else 0.0
        out.extend((r - mean) / (std + 1e-4) for r in group)
    return out


def sign_test(before: dict[str, float], after: dict[str, float]) -> dict:
    """Exact two-sided sign test over paired per-task pass rates at equal k.

    Ties are dropped, as the test defines them, and counted, as the card
    needs them: a narrow interval around zero with many ties is a diluted
    eval, not a null. The p-value is the binomial tail at 0.5 over the
    discordant tasks, doubled and capped at 1."""
    shared = sorted(set(before) & set(after))
    up = sum(1 for t in shared if after[t] > before[t])
    down = sum(1 for t in shared if after[t] < before[t])
    ties = len(shared) - up - down
    n = up + down
    if n == 0:
        p = 1.0
    else:
        low = min(up, down)
        tail = sum(math.comb(n, i) for i in range(low + 1)) / 2**n
        p = min(1.0, 2.0 * tail)
    return {"n_paired": len(shared), "up": up, "down": down, "ties": ties, "p": p}


def pin_rows(rows: list[dict]) -> str:
    """sha256 over the holdout's (question, answer) pairs in order. The
    scenario ids and any later field are left out, so a reformat does not
    move the pin and a changed problem does."""
    h = hashlib.sha256()
    for r in rows:
        h.update(
            json.dumps(
                {"question": r["question"], "answer": r["answer"]},
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        )
    return h.hexdigest()


def check_pin(rows: list[dict], pin_path: Path) -> str:
    """Refuse to score anything against a holdout that is not the frozen one."""
    digest = pin_rows(rows)
    pinned = pin_path.read_text(encoding="utf-8").strip()
    if digest != pinned:
        raise SystemExit(
            f"{pin_path.name}: the holdout hashes to {digest[:12]}, the pin says {pinned[:12]}; "
            "the frozen set moved. Rebuild it the same way or freeze a new one on purpose "
            "(delete the pin and run with --freeze)."
        )
    return digest


def make_reward(random_reward: bool, recorder: list[dict], history: list[dict], seed: int):
    """The TRL reward function for one arm.

    Verifier arms are paid the binary outcome, a program against the public
    GSM8K gold. The random arm is paid a seeded coin flip that never looks at
    the completion; the true outcome still rides along under ``markers`` so
    the training curve can say what the policy did while being paid noise.

    ``recorder`` is refilled with the batch it just graded, so after training
    ``hack_scan`` can be run on the last one (Lambert 2025, chapter
    Over-optimization) without keeping every step in memory. ``history``
    gets one line per batch: the curve the run page draws.
    """
    rng = random.Random(seed)
    step = {"n": 0}

    def reward(completions, prompts, answer, **kwargs) -> list[float]:
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        keys = [json.dumps(p, sort_keys=True, default=str) for p in prompts]
        outcomes = [outcome_of(t, gold_of(a)) for t, a in zip(texts, answer)]
        if random_reward:
            rewards = [1.0 if rng.random() < RANDOM_P else 0.0 for _ in texts]
        else:
            rewards = outcomes
        seen: dict[str, int] = {}
        recorder.clear()
        for key, text, r, o in zip(keys, texts, rewards, outcomes):
            recorder.append(
                {
                    "prompt": key,
                    "final_text": text,
                    "reward": r,
                    "markers": {"outcome": o},
                    "scenario_id": key,
                    "rollout_index": seen.get(key, 0),
                }
            )
            seen[key] = seen.get(key, 0) + 1
        history.append(
            {
                "step": step["n"],
                "reward": statistics.fmean(rewards),
                "outcome": statistics.fmean(outcomes),
                "length_chars": statistics.fmean(len(t) for t in texts),
            }
        )
        step["n"] += 1
        return rewards

    reward.__name__ = "coin_flip" if random_reward else "gsm8k_outcome"
    return reward


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def truncated_share(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return statistics.fmean(1.0 if r.get("finish_reason") == "length" else 0.0 for r in rows)


def graded_rows(holdout: list[dict], replies: list[list[tuple[str, bool]]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `compare` read: binary `reward`,
    one row per sample, grouped by task. A reply that hit the token cap is
    scored 0 whatever it says (Lambert 2025, chapter Reinforcement Learning:
    score only completions that end with EOS) and carries
    ``finish_reason="length"`` so the share is reported."""
    rows: list[dict] = []
    for task, samples in zip(holdout, replies):
        gold = gold_of(task["answer"])
        for i, (text, finished) in enumerate(samples):
            rows.append(
                {
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": outcome_of(text, gold) if finished else 0.0,
                    "finish_reason": "stop" if finished else "length",
                    "scenario_id": task["scenario_id"],
                    "rollout_index": i,
                    "privileged": {"reference": gold},
                }
            )
    return rows


def per_task_rates(*row_sets: list[dict]) -> dict[str, float]:
    """Per-task pass rate pooled over every row set given (three seeds at
    k = 4 is one paired draw of 12 per task)."""
    hits: dict[str, list[float]] = {}
    for rows in row_sets:
        for r in rows:
            hits.setdefault(r["scenario_id"], []).append(float(r["reward"]))
    return {t: statistics.fmean(v) for t, v in hits.items()}


def softmax_trainer(base_cls):
    """Build the trainer subclass. Takes `GRPOTrainer` as an argument so this
    module imports without torch, which is what lets `--selftest` run locally.
    """

    import torch

    class SoftmaxAdvantageTrainer(base_cls):  # type: ignore[valid-type,misc]
        """GRPOTrainer with equation 1 in place of the z-scored advantage.

        Two small overrides and no copy of the loss body:

        * `_calculate_rewards` keeps the gathered per-function rewards of the
          generation batch, which the parent computes and then folds into
          advantages without returning.
        * `_generate_and_score_completions` lets the parent build the batch,
          then rewrites `advantages` from those rewards with
          `softmax_advantages`. The batch is still grouped `num_generations`
          per prompt at that point (the parent's own `.view(-1,
          num_generations)` relies on it); the shuffle and the
          gradient-accumulation split come after and move every tensor in the
          dict together.

        `tau=None` is the baseline: the class is the same, the flag is the
        difference.
        """

        def __init__(self, *args, tau: float | None = None, **kw):
            super().__init__(*args, **kw)
            self.tau = tau
            self._rlvr_rewards_per_func = None
            for attr in ("scale_rewards", "num_generations", "reward_weights"):
                if not hasattr(self, attr):
                    raise RuntimeError(
                        f"this TRL ({version('trl')}) has no GRPOTrainer.{attr}; the recipe is "
                        "pinned to trl==0.19.1, where the group advantage is built in "
                        "_generate_and_score_completions from these attributes"
                    )
            if tau is not None and getattr(self, "use_liger_loss", False):
                raise RuntimeError("use_liger_loss builds its own advantages; run without Liger")

        def _get_per_token_logps(
            self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
        ):
            # TRL 0.19.1 calls this for the reference log-probs with no
            # batch_size, so the whole 48-rollout generation batch goes through
            # one forward and its logits alone ran the L40S out of memory
            # (24 GiB at 1,024 completion tokens). Chunking is exact: each
            # row's log-probs do not depend on its neighbours.
            return super()._get_per_token_logps(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size or self.args.per_device_train_batch_size,
            )

        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            rewards_per_func = super()._calculate_rewards(
                inputs, prompts, completions, completion_ids_list
            )
            self._rlvr_rewards_per_func = rewards_per_func
            return rewards_per_func

        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            rewards_per_func = self._rlvr_rewards_per_func
            if self.tau is None or rewards_per_func is None:
                return out
            weights = self.reward_weights.to(rewards_per_func.device).unsqueeze(0)
            rewards = (rewards_per_func * weights).nansum(dim=1)
            full = torch.tensor(
                softmax_advantages(rewards.tolist(), self.num_generations, self.tau),
                dtype=out["advantages"].dtype,
                device=out["advantages"].device,
            )
            n_local = out["advantages"].shape[0]
            start = self.accelerator.process_index * n_local
            out["advantages"] = full[start : start + n_local]
            # The parent logged the z-scored values a moment ago; replace them
            # so the textual log shows the advantages that reached the loss.
            logs = self._textual_logs["advantages"]
            for _ in range(min(len(logs), full.shape[0])):
                logs.pop()
            logs.extend(full.tolist())
            self._metrics["train"]["advantage/abs_mean"].append(full.abs().mean().item())
            self._metrics["train"]["advantage/min"].append(full.min().item())
            self._metrics["train"]["advantage/max"].append(full.max().item())
            return out

    return SoftmaxAdvantageTrainer


# --------------------------------------------------------------------------
# Modal: the pins and the image of recipes/papers/adaptive-clip, own app and
# volumes under the rlvr- prefix, scale-to-zero.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("rlvr-softmax-advantage")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        requirement("math"),  # MathEqual imports Math-Verify
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
)

runs_volume = modal.Volume.from_name("rlvr-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlvr-hf-cache", create_if_missing=True)


def _load(base_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    return tokenizer, model


def _sample(model, tokenizer, questions, *, n, max_new_tokens, seed, temperature, batch=48):
    """`n` replies per question, batched, sampled the way the trainer samples.
    Returns (text, finished) per reply: finished is whether an EOS arrived
    before the cap."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    torch.manual_seed(seed)
    model.eval()
    tokenizer.padding_side = "left"
    eos_ids = {tokenizer.eos_token_id}
    if tokenizer.pad_token_id is not None:
        eos_ids.add(tokenizer.pad_token_id)
    out: list[list[tuple[str, bool]]] = []
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
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        new = gen[:, prompt_len:]
        decoded = tokenizer.batch_decode(new, skip_special_tokens=True)
        finished = [any(int(t) in eos_ids for t in row) for row in new.tolist()]
        for i in range(len(chunk)):
            out.append(list(zip(decoded[i * n : (i + 1) * n], finished[i * n : (i + 1) * n])))
    model.train()
    return out


def _persist(name: str, payload: dict) -> None:
    """Write one call's return value to the runs volume before returning it,
    so a local client that drops mid-run (it happened: a base re-measure lost
    two finished runs to a disconnect) loses nothing; `--reuse` reads it back."""
    path = os.path.join(VOLUME_ROOT, "results", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    runs_volume.commit()


def _pins() -> dict:
    return {
        "torch": version("torch"),
        "transformers": version("transformers"),
        "trl": version("trl"),
        "peft": version("peft"),
        "whileai": version("whileai"),
    }


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    scaledown_window=600,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
)
def eval_base(
    holdout: list[dict],
    base_model: str = BASE_MODEL,
    eval_samples: int = 4,
    max_completion_length: int = MAX_COMPLETION,
    temperature: float = 1.0,
    runs: int = EVAL_RUNS,
    result_name: str = "base",
) -> dict:
    """The untrained base, `runs` times. The spread across those re-runs is
    the eval's own noise, and a delta smaller than it is not a result
    (Lambert 2025, chapter Evaluation). Runs first and alone, so the model
    lands in the cache volume before the arms start together."""
    import time

    sys.path.insert(0, "/root")
    from recipe_mod import graded_rows, truncated_share

    import whileai as wai

    started = time.time()
    tokenizer, model = _load(base_model)
    hf_cache.commit()
    questions = [t["question"] for t in holdout]
    base_runs = []
    for i in range(runs):
        replies = _sample(
            model,
            tokenizer,
            questions,
            n=eval_samples,
            max_new_tokens=max_completion_length,
            seed=1000 + i,
            temperature=temperature,
        )
        rows = graded_rows(holdout, replies)
        base_runs.append(rows)
        print(f"base run {i + 1}/{runs}: {wai.pass_at(rows)} truncated {truncated_share(rows):.2f}")
    out = {
        "base_runs": base_runs,
        "gpu_minutes": (time.time() - started) / 60.0,
        "pins": _pins(),
    }
    _persist(result_name, out)
    return out


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    scaledown_window=600,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
)
def run_arm(
    arm: str,
    seed: int,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 30,
    num_generations: int = 8,
    prompts_per_step: int = 6,
    learning_rate: float = 1e-4,
    beta: float = BETA,
    tau: float | None = None,
    random_reward: bool = False,
    max_completion_length: int = MAX_COMPLETION,
    lora_rank: int = 32,
    eval_samples: int = 4,
    temperature: float = 1.0,
    save_adapter: bool = True,
    result_name: str | None = None,
) -> dict:
    """One (arm, seed): train, then eval on the frozen holdout."""
    import time

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        EPS_HIGH,
        EPS_LOW,
        graded_rows,
        make_reward,
        mean_length,
        messages_for,
        softmax_trainer,
        truncated_share,
    )

    import whileai as wai

    started = time.time()
    tokenizer, model = _load(base_model)
    hf_cache.commit()
    dataset = Dataset.from_list(
        [{"prompt": messages_for(t["question"]), "answer": t["answer"]} for t in train_tasks]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=num_generations,
        # Half a group per forward, twice the accumulation: the same 48
        # rollouts a step and the same objective (per-sequence means over
        # equal microbatches average to the full-batch mean). A whole group
        # of 8 at up to 1,536 tokens ran the L40S out of memory on step 2.
        per_device_train_batch_size=num_generations // 2,
        gradient_accumulation_steps=2 * prompts_per_step,
        learning_rate=learning_rate,
        # One policy update per batch of rollouts: fully on-policy, so the
        # importance ratio is 1 and the clip range below never binds (Lambert
        # 2025, chapter Reinforcement Learning). The arms differ in the
        # advantage, which is reachable at one update; a clip change is not.
        num_iterations=1,
        beta=beta,
        epsilon=EPS_LOW,
        epsilon_high=EPS_HIGH,
        # The z-scored group advantage is the baseline; the recipe arm
        # overwrites it inside the trainer subclass.
        scale_rewards=True,
        # Per-sequence aggregation, GRPO's own (chapter Reinforcement Learning,
        # loss aggregation); the same on every arm.
        loss_type="grpo",
        max_completion_length=max_completion_length,
        max_prompt_length=512,
        temperature=temperature,
        top_p=1.0,
        bf16=True,
        # Off on purpose: this trainer generates during training, and
        # checkpointing corrupts Qwen generation on these pins.
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
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
    history: list[dict] = []
    # TRL 0.19.1 builds the LoRA adapter before it applies GRPOConfig.seed;
    # seed here so every arm at the same seed draws the same A matrix.
    set_seed(seed)
    trainer = softmax_trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[make_reward(random_reward, last_batch, history, seed)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        tau=tau,
    )
    trainer.train()
    train_minutes = (time.time() - started) / 60.0

    questions = [t["question"] for t in holdout]
    replies = _sample(
        trainer.model,
        tokenizer,
        questions,
        n=eval_samples,
        max_new_tokens=max_completion_length,
        seed=2000 + seed,
        temperature=temperature,
    )
    after_rows = graded_rows(holdout, replies)
    after = wai.pass_at(after_rows)
    print(f"{arm} seed {seed}: {after} truncated {truncated_share(after_rows):.2f}")

    # What the reward actually paid for in the last training batch (chapter
    # Over-optimization). The true outcome rides along as a marker; on the
    # verifier arms it IS the reward, so it is endorsed (the pilot's scan
    # named it top, which is the scan finding the reward itself). Nothing
    # else is endorsed.
    scan = wai.hack_scan(last_batch, endorsed=("marker:outcome",)) if last_batch else {}
    hack_top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
    hack_scan_top = hack_top.get("name", "") if isinstance(hack_top, dict) else str(hack_top)
    print(f"{arm} seed {seed} hack scan: top feature {hack_scan_top or 'none above the floor'}")

    if save_adapter:
        try:
            adapter_dir = os.path.join(out_dir, "adapter")
            trainer.model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            runs_volume.commit()
        except Exception as exc:  # the adapter is a convenience, the rows are the result
            print(f"adapter not saved ({type(exc).__name__}: {exc})")

    curve = []
    for entry in trainer.state.log_history:
        if "step" not in entry:
            continue
        point = {"step": int(entry["step"])}
        for key, name in (
            ("reward", "reward"),
            ("kl", "kl"),
            ("loss", "loss"),
            ("completions/mean_length", "completion_length"),
            ("completions/clipped_ratio", "clipped_ratio"),
            ("frac_reward_zero_std", "frac_reward_zero_std"),
            ("advantage/abs_mean", "advantage_abs_mean"),
        ):
            if key in entry and entry[key] is not None:
                point[name] = float(entry[key])
        curve.append(point)

    out = {
        "arm": arm,
        "seed": seed,
        "after_rows": after_rows,
        "gpu_minutes": (time.time() - started) / 60.0,
        "train_minutes": train_minutes,
        "steps": steps,
        "length_after": mean_length(after_rows),
        "truncated_after": truncated_share(after_rows),
        "hack_scan_top": hack_scan_top,
        "history": history,
        "curve": curve,
        "pins": _pins(),
        "config": {
            "tau": tau,
            "random_reward": random_reward,
            "beta": beta,
            "epsilon": [EPS_LOW, EPS_HIGH],
            "num_iterations": 1,
            "loss_type": "grpo",
            "learning_rate": learning_rate,
            "num_generations": num_generations,
            "prompts_per_step": prompts_per_step,
            "max_completion_length": max_completion_length,
            "temperature": temperature,
            "lora_rank": lora_rank,
        },
    }
    if result_name:
        _persist(result_name, out)
    return out


# --------------------------------------------------------------------------
# Local: data, the pin, orchestration, results.json, the platform post.
# --------------------------------------------------------------------------


def data(seed: int, n_train: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    """GSM8K (MIT). Train prompts from the train split; the holdout is the
    ``n_holdout`` test-split problems with the lowest sha256 of their text,
    so the draw is a property of the problems and not of file order. The
    splits are disjoint by construction and the hash sets are checked to be."""
    from datasets import load_dataset

    test = load_dataset("openai/gsm8k", "main", split="test")
    keyed = sorted(
        ((hashlib.sha256(r["question"].encode("utf-8")).hexdigest(), r) for r in test),
        key=lambda x: x[0],
    )
    holdout = [
        {
            "prompt": r["question"],
            "question": r["question"],
            "answer": r["answer"],
            "scenario_id": f"gsm8k-test-{h[:12]}",
        }
        for h, r in keyed[:n_holdout]
    ]
    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    train_tasks = [
        {
            "prompt": r["question"],
            "question": r["question"],
            "answer": r["answer"],
            "scenario_id": f"gsm8k-train-{hashlib.sha256(r['question'].encode()).hexdigest()[:12]}",
        }
        for r in train.select(range(n_train))
    ]
    shared = {t["scenario_id"][-12:] for t in train_tasks} & {h[:12] for h, _ in keyed[:n_holdout]}
    if shared:
        raise SystemExit(f"{len(shared)} problems hash into both train and holdout")
    return train_tasks, holdout


def summarize(rows: list[dict]) -> dict:
    import whileai as wai

    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
        "truncated": truncated_share(rows),
        "length": mean_length(rows),
    }


def verdict_word(d: dict) -> str:
    word = d["target_verdict"]
    if word == "unresolved":
        return "unresolved"
    if word in ("moved", "moved_the_wrong_way"):
        return "moved"
    return "flat"


def paired(before, after, *, run_std, run_std_runs, train_before, train_after) -> dict:
    """One paired comparison: the seed-17 rows are the headline pair, every
    seed's rows widen the interval by the between-seed spread, the base's
    re-run floor sets the band, and the sign test runs on per-task rates
    pooled across seeds (equal k on both sides)."""
    import whileai as wai

    d = wai.compare(
        before,
        after,
        target="pass_at_1",
        run_std=run_std,
        run_std_runs=run_std_runs,
        train_runs={"before": train_before, "after": train_after},
        proxy=PROXY,
    )
    print(d)
    rates_a = per_task_rates(*(train_before or [before]))
    rates_b = per_task_rates(*(train_after or [after]))
    return {
        "delta": d["target_delta"],
        "ci": list(d["target_ci95"] or (0.0, 0.0)),
        "across_seeds": d.get("train_delta"),
        "across_seeds_ci": list(d["train_ci95"]) if d.get("train_ci95") else None,
        "train_std": d.get("train_std"),
        "noise_band": d.get("noise_band"),
        "within_noise": bool(d.get("within_noise")),
        "verdict": verdict_word(d),
        "sdk_verdict": d["target_verdict"],
        "sign_test": sign_test(rates_a, rates_b),
        "n_paired_tasks": d.get("n_paired_tasks"),
        "over_optimized": bool(d.get("over_optimized")),
    }


def five_lines(arm: str, seed: int, arm_res: dict, delta_vs_base: dict, steps: int) -> str:
    changed = {
        "baseline": "GRPO with the z-scored group advantage, TRL 0.19.1 default",
        "recipe": f"the group advantage is k * softmax(r / {TAU}) - 1 (equation 1); nothing else",
        "random": f"the verifier replaced by a Bernoulli({RANDOM_P}) coin flip; same steps and seed",
    }[arm]
    d = delta_vs_base
    return "\n".join(
        [
            f"Changed: {changed}.",
            f"Moved: pass@1 {d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] vs the untrained "
            f"base on this seed; sign test up {d['sign_test']['up']} down {d['sign_test']['down']} "
            f"ties {d['sign_test']['ties']} p {d['sign_test']['p']:.3f}.",
            f"Why: {arm_res['steps']} steps x 6 prompts x 8 rollouts, LoRA r=32, KL {BETA}, seed {seed}; "
            f"truncated share {arm_res['truncated']:.2f}, mean reply {arm_res['length']:.0f} chars.",
            "Learned: read the recipe-vs-random row, not this one; on a Qwen base the gain over the "
            "untrained base includes what a coin flip buys (Shao et al. 2025).",
            f"Reproduce: python recipes/papers/softmax-advantage/recipe.py --arm {arm} --seeds {seed} "
            f"--steps {steps}",
        ]
    )


def post(results: dict, rows: dict, base_runs: list[list[dict]], curves: dict) -> dict:
    """The run page: one tracked agent named after the recipe, one behavior on
    the pinned test, the base as its own version, one version per (arm,
    seed), the recipe versions scored last so the verdict picks them, then
    the readback."""
    from whileai.platform import Behavior, track

    tracked = track(
        HERE.name,
        model=BASE_MODEL,
        harness={"label": f"recipe {HERE.name}", "instructions": SYSTEM},
    )
    n = results["n_holdout"]
    tracked.behavior(
        Behavior(
            name="gsm8k",
            test_version=results["test_version"],
            n=n,
            contamination=results["checks"]["decontaminated_dropped"],
            graded_by="program",
            reward_is_judge=False,
            description=(
                f"pass@1 on {n} GSM8K test problems drawn by problem hash, MathEqual against the "
                f"gold, k={results['k']} samples a task, temperature 1.0, {results['max_completion']} "
                "tokens; a reply cut at the cap scores 0"
            ),
        )
    )
    # the behavior's re-run floor, measured from the three base draws, in points
    floor = tracked.noise_floor("gsm8k", *base_runs)
    usd_per_hour = USD_PER_HOUR.get(results["gpu"], 1.95)
    urls: dict[str, str] = {}

    base = tracked.run("base", method="none", base=BASE_MODEL, targets=["gsm8k"])
    p = results["arms"]["base"]
    base.score("gsm8k", p["score"], ci=(p["ci"][1] - p["ci"][0]) / 2, n=n, fraction=True)
    base.note(
        "Changed: nothing, the untrained base.\nMoved: the reference point every arm is paired "
        f"against; three re-runs set the floor {float(floor['noise_floor']):.1f} points.\nWhy: "
        "Lambert 2025, chapter Evaluation.\nLearned: see the trained versions.\nReproduce: "
        "python recipes/papers/softmax-advantage/recipe.py"
    )
    base.finish(
        status="evaluated",
        hours=results["base_gpu_minutes"] / 60,
        gpu=f"1x{results['gpu']}",
        cost_usd=round(results["base_gpu_minutes"] / 60 * usd_per_hour, 2),
        say=False,
    )
    urls["base"] = base.url

    # controls first, the treatment last: verdict() reads the newest scored version
    for arm in ("random", "baseline", "recipe"):
        method = {
            "baseline": "grpo",
            "recipe": "softmax-grpo",
            "random": "grpo-random-reward",
        }[arm]
        for seed in results["seeds"]:
            key = f"{arm}-s{seed}"
            res = results["arms"][arm]["per_seed"][str(seed)]
            run = tracked.run(
                key,
                method=method,
                base=BASE_MODEL,
                targets=["gsm8k"],
                trained_on=[f"GSM8K train, {results['n_train']} prompts"],
                gpu=f"1x{results['gpu']}",
            )
            run.score(
                "gsm8k", res["score"], ci=(res["ci"][1] - res["ci"][0]) / 2, n=n, fraction=True
            )
            for point in curves.get(key, []):
                run.log(point["step"], **{k: v for k, v in point.items() if k != "step"})
            run.note(
                five_lines(arm, seed, res, results["delta"][f"{arm}_vs_base"], results["steps"])
            )
            run.finish(
                status="evaluated",
                hours=res["gpu_minutes"] / 60,
                gpu=f"1x{results['gpu']}",
                cost_usd=round(res["gpu_minutes"] / 60 * usd_per_hour, 2),
                steps=results["steps"],
                say=False,
            )
            urls[key] = run.url
    # The baseline is the thing a reader would otherwise ship, so it is the
    # served version the verdict reads the recipe against (write-a-recipe).
    tracked.promote(f"baseline-s{results['seeds'][0]}")
    readback = {
        "runs": [{k: r.get(k) for k in ("version", "status", "id")} for r in tracked.runs()],
        "evals": [str(h) for h in tracked.evals()],
        "verdict": str(tracked.verdict("gsm8k")),
        "urls": urls,
    }
    print(readback["verdict"])
    for line in readback["evals"]:
        print(line)
    return readback


def selftest() -> None:
    """Equation 1 on hand-written groups, the sign test with a case it must
    reject and one it must not, the coin flip, the grader and the pin. No
    GPU, no key, no model download."""
    k = 8
    # A lone success out of eight: +7 for it, -1 for every failure, sum zero.
    # At tau = 0.1 the wrong rollouts keep weight e^-10 each, so the values
    # are 6.9975 and -0.9996: the two-decimal claim, not an exact one.
    hard = [1.0] + [0.0] * 7
    adv = softmax_advantages(hard, k)
    assert abs(adv[0] - 7.0) < 1e-2, adv
    assert all(abs(a + 1.0) < 1e-2 for a in adv[1:]), adv
    assert abs(sum(adv)) < 1e-9
    # A lone failure out of eight: the failure is still -1, the successes +1/7.
    easy = [1.0] * 7 + [0.0]
    adv = softmax_advantages(easy, k)
    assert abs(adv[-1] + 1.0) < 1e-2 and abs(adv[0] - 1 / 7) < 1e-2, adv
    # Against the z-score the baseline uses: GRPO hands that lone failure
    # -2.47 and the lone success +2.47; the softmax bounds the failure at -1.
    z_hard, z_easy = grpo_advantages(hard, k), grpo_advantages(easy, k)
    assert abs(z_hard[0] - 2.4742) < 1e-3 and abs(z_easy[-1] + 2.4742) < 1e-3, (z_hard, z_easy)
    assert min(softmax_advantages(easy, k)) > min(z_easy)
    assert max(softmax_advantages(hard, k)) > max(z_hard)
    # A unanimous group is all zeros in both.
    assert all(abs(a) < 1e-9 for a in softmax_advantages([1.0] * k, k))
    assert all(abs(a) < 1e-9 for a in softmax_advantages([0.0] * k, k))
    # Two groups laid out the way TRL hands them over stay separate.
    two = softmax_advantages(hard + easy, k)
    assert two[:k] == softmax_advantages(hard, k) and two[k:] == softmax_advantages(easy, k)
    # The temperature matters and the guards fire.
    assert softmax_advantages(hard, k, tau=1.0)[0] < softmax_advantages(hard, k, tau=0.1)[0]
    for bad in ({"tau": 0.0}, {"tau": -1.0}):
        try:
            softmax_advantages(hard, k, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"softmax_advantages accepted {bad}")
    print(f"eq. 1 at tau={TAU}, k={k}: 1 of 8 right -> +{7.0:.2f} / -1; 7 of 8 right -> +1/7 / -1")
    print(
        f"z-score (baseline): 1 of 8 right -> +{z_hard[0]:.2f}; 7 of 8 right, the wrong one -> {z_easy[-1]:.2f}"
    )

    # The sign test rejects a one-sided shift and does not reject a balanced one.
    before = {f"t{i}": 0.0 for i in range(12)}
    after = {f"t{i}": 1.0 for i in range(12)}
    s = sign_test(before, after)
    assert s == {"n_paired": 12, "up": 12, "down": 0, "ties": 0, "p": 2 / 4096}, s
    balanced = {f"t{i}": (1.0 if i % 2 else 0.0) for i in range(12)}
    s = sign_test(balanced, {t: 1.0 - v for t, v in balanced.items()})
    assert s["up"] == 6 and s["down"] == 6 and abs(s["p"] - 1.0) < 1e-12, s
    # Ties are counted and dropped: an all-tied pair is p = 1 with 12 ties.
    s = sign_test(before, dict(before))
    assert s["ties"] == 12 and s["up"] == 0 and s["p"] == 1.0, s
    print("sign test: 12 up / 0 down p=0.0005; 6 / 6 p=1.0; 12 ties p=1.0")

    # The coin flip is seeded, ignores the text, and is not the outcome.
    rec: list[dict] = []
    hist: list[dict] = []
    coin = make_reward(True, rec, hist, seed=0)
    texts = [[{"content": "\\boxed{18}"}]] * 200
    flips = coin(texts, ["q"] * 200, ["x #### 18"] * 200)
    assert set(flips) == {0.0, 1.0} and 0.35 < statistics.fmean(flips) < 0.65, statistics.fmean(
        flips
    )
    assert all(r["markers"]["outcome"] == 1.0 for r in rec), "the true outcome rides along"
    again = make_reward(True, [], [], seed=0)(texts, ["q"] * 200, ["x #### 18"] * 200)
    assert again == flips, "the same seed draws the same flips"
    verifier = make_reward(False, [], [], seed=0)
    assert verifier(texts[:1], ["q"], ["x #### 18"]) == [1.0]
    assert verifier(texts[:1], ["q"], ["x #### 19"]) == [0.0]
    print(f"coin flip: mean {statistics.fmean(flips):.2f} over 200 draws, independent of the reply")

    # The grader is a program, not a judge; a cut reply scores 0.
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    rows = graded_rows(
        [{"question": "q", "answer": "x #### 18", "scenario_id": "t"}],
        [[("\\boxed{18}", True), ("\\boxed{18}", False)]],
    )
    assert [r["reward"] for r in rows] == [1.0, 0.0] and rows[1]["finish_reason"] == "length"
    assert truncated_share(rows) == 0.5
    print("grader: MathEqual, decided by Math-Verify; a reply cut at the cap scores 0")

    # The pin follows the problems, not the ids, and moves when a problem does.
    a = [{"question": "q1", "answer": "1", "scenario_id": "x"}]
    b = [{"question": "q1", "answer": "1", "scenario_id": "y"}]
    c = [{"question": "q2", "answer": "1", "scenario_id": "x"}]
    assert pin_rows(a) == pin_rows(b) and pin_rows(a) != pin_rows(c)
    pin = HERE / "holdout.sha256"
    if pin.exists():
        assert len(pin.read_text().strip()) == 64, "holdout.sha256 is a sha256 hex digest"
        print(f"pin: holdout.sha256 = {pin.read_text().strip()[:12]}...")

    # per_task_rates pools every row set it is given.
    r1 = [{"scenario_id": "t", "reward": 1.0}, {"scenario_id": "t", "reward": 0.0}]
    r2 = [{"scenario_id": "t", "reward": 1.0}, {"scenario_id": "t", "reward": 1.0}]
    assert per_task_rates(r1, r2) == {"t": 0.75}
    print("selftest ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0, help="the train-prompt shuffle")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=300)
    ap.add_argument("--generations", type=int, default=8, help="rollouts per prompt, M in eq. 1")
    ap.add_argument("--prompts-per-step", type=int, default=6)
    ap.add_argument("--tau", type=float, default=TAU, help="the softmax temperature, eq. 1")
    ap.add_argument("--beta", type=float, default=BETA, help="KL coefficient, the paper's")
    ap.add_argument(
        "--max-completion",
        type=int,
        default=MAX_COMPLETION,
        help="completion tokens, training and eval; the paper's 256 cut half the base's replies",
    )
    ap.add_argument(
        "--est-minutes",
        type=float,
        default=30.0,
        help="GPU minutes one (arm, seed) is expected to take, for the spend guard",
    )
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA rate of recipes/04-train/grpo")
    ap.add_argument("--selftest", action="store_true", help="the pure functions, offline")
    ap.add_argument("--pilot", action="store_true", help="one recipe arm, 3 steps, 24 tasks")
    ap.add_argument(
        "--base-only",
        action="store_true",
        help="the three base re-runs and their truncation, then stop",
    )
    ap.add_argument("--freeze", action="store_true", help="write holdout.sha256 when absent")
    ap.add_argument("--reuse", action="store_true", help="rebuild from .cache/, no GPU")
    ap.add_argument("--no-post", action="store_true", help="skip the platform post")
    ap.add_argument(
        "--spent-usd", type=float, default=0.0, help="spend before this call, for the guard"
    )
    ap.add_argument("--tag", default="c1024-", help="prefix of the result files on the runs volume")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    pin_path = HERE / "holdout.sha256"
    if not pin_path.exists():
        if not args.freeze:
            raise SystemExit("no holdout.sha256: run once with --freeze to pin the holdout")
        pin_path.write_text(pin_rows(holdout) + "\n", encoding="utf-8")
        print(f"froze {len(holdout)} holdout problems: {pin_path.name} {pin_rows(holdout)[:12]}")
    pin = check_pin(holdout, pin_path)
    # GSM8K's splits are disjoint, so this should drop nothing. It runs anyway
    # and the count goes in the Checks table, because "should" is not a
    # measurement (Lambert 2025, chapter Evaluation).
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(
        f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped; "
        f"rules skipped: {decon.get('rules_skipped') or 'none'}"
    )

    if args.pilot:
        holdout = holdout[:24]
        args.steps, args.seeds, args.arm = 3, [SEEDS[0]], "recipe"

    arms = list(ARMS) if args.arm == "all" else [args.arm]
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)
    usd_per_hour = USD_PER_HOUR.get(DEFAULT_GPU, 1.95)
    spend_minutes = 0.0

    def cached(name: str) -> dict | None:
        """A finished call's output: the local cache first, then the copy the
        container left on the runs volume (a disconnect loses neither)."""
        if not args.reuse:
            return None
        path = cache / f"{name}.json"
        if path.exists():
            print(f"{name}: reused {path}")
            return json.loads(path.read_text())
        try:
            blob = b"".join(runs_volume.read_file(f"results/{args.tag}{name}.json"))
        except Exception:
            return None
        path.write_bytes(blob)
        print(f"{name}: fetched from the rlvr-runs volume")
        return json.loads(blob)

    outs: dict[str, dict] = {}
    # detached: a dropped local client no longer stops the app mid-run
    with modal.enable_output(), app.run(detach=True):
        base_out = cached("base")
        if base_out is None and not args.pilot:
            base_out = eval_base.remote(
                holdout,
                eval_samples=args.k,
                max_completion_length=args.max_completion,
                result_name=f"{args.tag}base",
            )
            (cache / "base.json").write_text(json.dumps(base_out))
        if base_out is not None:
            spend_minutes += base_out["gpu_minutes"]
            print(f"base: {base_out['gpu_minutes']:.1f} GPU minutes")
            for i, run in enumerate(base_out["base_runs"]):
                print(
                    f"base run {i + 1}: pass@1 {wai.pass_at(run).pass_at_1:.3f} "
                    f"truncated {truncated_share(run):.3f} at {args.max_completion} tokens"
                )
        if args.base_only:
            return
        calls = {}
        for arm in arms:
            for seed in args.seeds:
                key = f"{arm}-s{seed}"
                hit = cached(key)
                if hit is not None:
                    outs[key] = hit
                    spend_minutes += hit["gpu_minutes"]  # a reused arm still cost its minutes
                    continue
                projected = args.spent_usd + (
                    (spend_minutes + args.est_minutes * (len(calls) + 1)) / 60 * usd_per_hour
                )
                if projected > STOP_AT_USD:
                    print(
                        f"{key}: not started, projected ${projected:.2f} would pass ${STOP_AT_USD}"
                    )
                    continue
                calls[key] = run_arm.spawn(
                    arm,
                    seed,
                    train_tasks,
                    holdout,
                    f"{HERE.name}-{key}-{date.today().isoformat()}",
                    steps=args.steps,
                    num_generations=args.generations,
                    prompts_per_step=args.prompts_per_step,
                    learning_rate=args.lr,
                    beta=args.beta,
                    tau=args.tau if ARMS[arm]["tau"] is not None else None,
                    random_reward=ARMS[arm]["random_reward"],
                    max_completion_length=args.max_completion,
                    eval_samples=args.k,
                    save_adapter=not args.pilot,
                    result_name=None if args.pilot else f"{args.tag}{key}",
                )
        for key, call in calls.items():
            out = call.get()
            (cache / f"{key}.json").write_text(json.dumps(out))
            outs[key] = out
            spend_minutes += out["gpu_minutes"]
            p = wai.pass_at(out["after_rows"])
            print(
                f"{key}: pass@1 {p.pass_at_1:.3f} truncated {out['truncated_after']:.2f} "
                f"in {out['gpu_minutes']:.1f} GPU minutes; spend so far "
                f"${spend_minutes / 60 * usd_per_hour:.2f}"
            )

    if args.pilot:
        out = outs[f"recipe-s{SEEDS[0]}"]
        print(json.dumps({k: v for k, v in out.items() if k != "after_rows"}, indent=2))
        print(f"pilot: {spend_minutes:.1f} GPU minutes, ${spend_minutes / 60 * usd_per_hour:.2f}")
        return
    if base_out is None:
        raise SystemExit("no base eval in .cache/ and none run")

    base_runs = base_out["base_runs"]
    noise = wai.eval_variance(*base_runs)
    run_std = float(noise["run_std"])
    run_std_runs = int(noise["n_runs"])
    print(f"eval noise over {run_std_runs} base runs: run_std {run_std:.4f}")

    results: dict = {
        "recipe": HERE.name,
        "title": "Softmax advantage: the group advantage is a softmax over rewards, not a z-score",
        "paper": "https://arxiv.org/abs/2608.09271",
        "book": BOOK,
        "base_model": BASE_MODEL,
        "metric": METRIC,
        "n_holdout": len(holdout),
        "n_train": len(train_tasks),
        "k": args.k,
        "test_version": f"t-{pin[:8]}",
        "holdout_sha256": pin,
        "seeds": list(args.seeds),
        "steps": args.steps,
        "max_completion": args.max_completion,
        "tau": args.tau,
        "gpu": DEFAULT_GPU,
        "whileai": version("whileai"),
        "base_gpu_minutes": round(base_out["gpu_minutes"], 1),
        "arms": {
            "base": {**summarize(base_runs[0]), "steps": 0, "gpu_minutes": 0},
        },
        "checks": {
            "run_std": round(run_std, 4),
            "run_std_runs": run_std_runs,
            "train_seeds": {},
            "decontaminated_dropped": int(decon.get("n_contaminated", 0)),
            "over_optimized": False,
            "length_before": mean_length(base_runs[0]),
            "length_after": {},
            "truncated_before": truncated_share(base_runs[0]),
            "truncated_after": {},
            "hack_scan_top": "",
            "hack_scan_by_arm": {},
            "seed": args.seed,
            "pins": base_out["pins"],
            "base_runs_pass_at_1": [wai.pass_at(r).pass_at_1 for r in base_runs],
        },
    }
    rows: dict[str, list[dict]] = {"base": base_runs[0]}
    train_rows: dict[str, list[list[dict]]] = {}
    curves: dict[str, list[dict]] = {}
    for arm in arms:
        per_seed = {}
        seed_rows = []
        for seed in args.seeds:
            key = f"{arm}-s{seed}"
            if key not in outs:
                continue
            out = outs[key]
            per_seed[str(seed)] = {
                **summarize(out["after_rows"]),
                "gpu_minutes": round(out["gpu_minutes"], 1),
                "steps": out["steps"],
                "hack_scan_top": out["hack_scan_top"],
                "train_outcome_first10": statistics.fmean(
                    h["outcome"] for h in out["history"][:10]
                ),
                "train_outcome_last10": statistics.fmean(
                    h["outcome"] for h in out["history"][-10:]
                ),
            }
            seed_rows.append(out["after_rows"])
            curves[key] = out["curve"]
            rows[key] = out["after_rows"]
        if not seed_rows:
            continue
        head = f"{arm}-s{args.seeds[0]}"
        rows[arm] = rows[head]
        train_rows[arm] = seed_rows
        scores = [s["score"] for s in per_seed.values()]
        results["arms"][arm] = {
            **summarize(rows[head]),
            "score_mean_over_seeds": statistics.fmean(scores),
            "score_sd_over_seeds": statistics.stdev(scores) if len(scores) > 1 else None,
            "steps": args.steps,
            "gpu_minutes": round(sum(s["gpu_minutes"] for s in per_seed.values()), 1),
            "per_seed": per_seed,
        }
        results["checks"]["train_seeds"][arm] = len(per_seed)
        results["checks"]["length_after"][arm] = statistics.fmean(
            s["length"] for s in per_seed.values()
        )
        results["checks"]["truncated_after"][arm] = statistics.fmean(
            s["truncated"] for s in per_seed.values()
        )
        results["checks"]["hack_scan_by_arm"][arm] = per_seed[str(args.seeds[0])]["hack_scan_top"]
    results["checks"]["hack_scan_top"] = results["checks"]["hack_scan_by_arm"].get("recipe", "")

    deltas: dict[str, dict] = {}
    for arm in train_rows:
        deltas[f"{arm}_vs_base"] = paired(
            rows["base"],
            rows[arm],
            run_std=run_std,
            run_std_runs=run_std_runs,
            train_before=None,
            train_after=train_rows[arm],
        )
    for a, b in (("baseline", "recipe"), ("random", "recipe"), ("random", "baseline")):
        if a in train_rows and b in train_rows:
            deltas[f"{b}_vs_{a}"] = paired(
                rows[a],
                rows[b],
                run_std=run_std,
                run_std_runs=run_std_runs,
                train_before=train_rows[a],
                train_after=train_rows[b],
            )
    results["delta"] = deltas
    if "recipe_vs_baseline" in deltas:
        d = deltas["recipe_vs_baseline"]
        # The headline is the across-seed delta with the between-seed term,
        # the contract's own rule; the seed-17 pair is kept beside it.
        results["delta"]["recipe_vs_baseline"] = (
            d["across_seeds"] if d["across_seeds"] is not None else d["delta"]
        )
        results["delta"]["ci"] = d["across_seeds_ci"] or d["ci"]
        results["delta"]["verdict"] = d["verdict"]
        results["delta"]["recipe_vs_baseline_detail"] = d
        results["checks"]["over_optimized"] = d["over_optimized"]
    if "recipe_vs_random" in deltas and "baseline_vs_random" in deltas:
        # Gain from the reward = delta_reward - delta_random, paired per task:
        # (r_i - b_i) - (x_i - b_i) = r_i - x_i, so it is the arm-vs-random pair.
        results["delta"]["reward_corrected"] = {
            "recipe": deltas["recipe_vs_random"]["across_seeds"],
            "recipe_ci": deltas["recipe_vs_random"]["across_seeds_ci"],
            "baseline": deltas["baseline_vs_random"]["across_seeds"],
            "baseline_ci": deltas["baseline_vs_random"]["across_seeds_ci"],
        }
    results["usd"] = round(spend_minutes / 60.0 * usd_per_hour, 2)
    results["gpu_minutes_total"] = round(spend_minutes, 1)
    results["verified"] = date.today().isoformat()

    if not args.no_post and all(a in train_rows for a in ARMS):
        try:
            readback = post(results, rows, base_runs, curves)
            results["run_url"] = readback["urls"].get(f"recipe-s{args.seeds[0]}", "")
            results["platform"] = readback
        except Exception as exc:  # the platform is not the experiment
            print(f"platform: skipped ({type(exc).__name__}: {exc})")
            results["run_url"] = ""
    else:
        results.setdefault("run_url", "")
    print(f"wall clock: {spend_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "platform"}, indent=2))


if __name__ == "__main__":
    main()
