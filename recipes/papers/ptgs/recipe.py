"""PTGS: heat the prompts the policy keeps failing, cool the ones it has mastered.

    python recipe.py                      # three arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the sampler, the temperature and the tax, offline

Oh et al. 2026 (arXiv:2610.01509) find that post-training sharpens a policy:
tasks move to "always solved" or "never solved", pass@1 rises and pass@k
falls. They call the lost room to scale with retries the Sharpening Tax,
and propose posterior-tempered group sampling (PTGS) to pay less of it.

PTGS keeps a discounted count of successes and failures per prompt, draws a
success rate from the Beta posterior those counts give (Thompson sampling),
and samples that prompt's next group at

    T_x = T_ref * tau ** h(p_hat),   h = (p~ - p_hat) / p~        if p_hat <= p~
                                     h = (p~ - p_hat) / (1 - p~)  otherwise

so a prompt it keeps failing is heated toward tau * T_ref and one it always
solves is cooled toward T_ref / tau. The log-probs in the update are taken
at the same per-row temperature, so the update is on-policy for the
tempered policy the rollout came from (their Appendix A.5).

Three arms, the same 12 prompts x 4 rollouts in every update:
  baseline  GRPO, every prompt at T_ref
  ada       Reinforce-Ada (wai.methods.ReinforceAda): more draws, same policy
  recipe    PTGS: the same 4 draws, a per-prompt temperature

Shape of the run:
  1. data():      MATH train levels 3 to 5 for prompts, MATH-500 levels 3 to 5 held out
  2. run_arm():   TRL GRPOTrainer + LoRA on Modal, one (arm, seed) per call
  3. evaluate():  same holdout, 8 samples per task at T_ref, graded by MathEqual
  4. results.json: pass@1 (wai.compare), pass@8 and the Sharpening Tax at 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections.abc import Sequence
from datetime import date
from importlib.metadata import version
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Reinforcement Learning"  # the group baseline and what a zero-variance group contributes
# The training reward is the target: both are the same binary check against
# the MATH answer, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)
LEVELS = ("Level 3", "Level 4", "Level 5")

GROUP = 4  # rollouts per prompt in every arm's update
T_REF = 0.9  # the rollout temperature of the baseline and Reinforce-Ada, and PTGS's center
# PTGS, the setting the paper uses for PPO in both environments (Appendix
# A.5): tau = 1.5, so T_x is in [T_REF / 1.5, 1.5 * T_REF]; the target
# success rate grows from 0.25 to 0.5 by a constant factor per step; the
# forgetting factor is 0.95; the Beta prior has mass 2 centered on the target.
# Their GRPO runs pick tau per environment on validation peaks; this one
# setting is the one they did not tune per environment.
TAU = 1.5
GAMMA = 0.95
P_START, P_END = 0.25, 0.5
PRIOR_MASS = 2.0

SYSTEM = "Solve the problem. Think briefly, then give the final answer as \\boxed{answer}."


# --------------------------------------------------------------------------
# The sampler and the tax. Pure functions, no torch: `--selftest` runs them
# on the CPU, and the Modal container imports this same file.
# --------------------------------------------------------------------------


def messages_for(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def last_boxed(text: str) -> str | None:
    """The contents of the last \\boxed{...}, braces matched, so
    \\boxed{\\frac{1}{2}} reads whole. MATH train ships worked solutions."""
    text = text or ""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(c)
        i += 1
    return None


def outcome_of(text: str, gold: str) -> float:
    """1.0 when the final answer matches the gold, else 0.0. A program, not a
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


def target_at(step: int, total: int, start: float = P_START, end: float = P_END) -> float:
    """The target success rate p~ at a step: start to end by a constant
    factor per step, the paper's schedule."""
    if total <= 1:
        return end
    return start * (end / start) ** (min(step, total - 1) / (total - 1))


def temperature(p_hat: float, p_target: float, tau: float = TAU, t_ref: float = T_REF) -> float:
    """THE ONE CHANGE, part 1: Eq. (4). Passes through T(0) = tau * t_ref,
    T(p~) = t_ref and T(1) = t_ref / tau."""
    span = p_target if p_hat <= p_target else 1 - p_target  # hard prompt, else easy
    return t_ref * tau ** ((p_target - p_hat) / span)


class Posterior:
    """THE ONE CHANGE, part 2: per-prompt discounted counts and the Thompson
    draw. Counts start at zero, so a prompt seen for the first time draws
    from the prior alone, Beta(2 p~, 2 (1 - p~))."""

    def __init__(self, gamma: float = GAMMA, seed: int = 0) -> None:
        self.gamma = gamma
        self.counts: dict[str, tuple[float, float]] = {}
        self.rng = random.Random(seed)

    def draw(self, key: str, p_target: float) -> float:
        s, f = self.counts.get(key, (0.0, 0.0))
        a = PRIOR_MASS * p_target + s
        b = PRIOR_MASS * (1 - p_target) + f
        return self.rng.betavariate(a, b)

    def update(self, key: str, successes: int, n: int) -> None:
        s, f = self.counts.get(key, (0.0, 0.0))
        self.counts[key] = (self.gamma * s + successes, self.gamma * f + n - successes)


def pass_at_k(c: int, n: int, k: int) -> float:
    """Chen et al. 2021's unbiased estimator: at least one of k right."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curves(counts: Sequence[tuple[int, int]], k_max: int) -> list[list[float]]:
    """Per-task pass@k for k = 1..k_max, from (right, drawn) per task."""
    return [[pass_at_k(c, n, k) for k in range(1, k_max + 1)] for c, n in counts]


def scalability(per_task: Sequence[Sequence[float]]) -> tuple[float, float]:
    """A(K) and S(K) of Eqs. (1) and (2) from per-task pass@k curves:
    A is the area under the pass@K ceiling, S is A over (K - 1)(1 - pass@1).
    A is the expected number of failed tries before a success within K
    (their Proposition 1), so a task always solved or never solved adds 0."""
    k_max = len(per_task[0])
    curve = [statistics.fmean(t[k] for t in per_task) for k in range(k_max)]
    a = sum(curve[-1] - curve[k] for k in range(k_max - 1))
    headroom = (k_max - 1) * (1 - curve[0])
    return a, (a / headroom if headroom > 0 else 0.0)


def sharpening_tax(
    base: Sequence[Sequence[float]],
    trained: Sequence[Sequence[float]],
    *,
    boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Tax_S(K) = S_base(K) - S_trained(K), with a 95% interval from a paired
    task bootstrap. Positive means training cost room to scale with retries."""
    if len(base) != len(trained):
        raise ValueError("the tax pairs tasks: base and trained must cover the same tasks")
    point = scalability(base)[1] - scalability(trained)[1]
    rng = random.Random(seed)
    n = len(base)
    draws = []
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(
            scalability([base[i] for i in idx])[1] - scalability([trained[i] for i in idx])[1]
        )
    draws.sort()
    return {"tax_s": point, "ci": [draws[int(0.025 * boot)], draws[int(0.975 * boot) - 1]]}


def paired_delta(
    a: Sequence[float], b: Sequence[float], *, boot: int = 2000, seed: int = 0
) -> dict:
    """mean(b) - mean(a) over paired tasks, with a task-bootstrap 95% interval."""
    rng = random.Random(seed)
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
    draws = sorted(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(boot))
    return {
        "delta": statistics.fmean(diffs),
        "ci": [draws[int(0.025 * boot)], draws[int(0.975 * boot) - 1]],
    }


def task_counts(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """(right, drawn) per scenario_id from graded eval rows."""
    out: dict[str, list[int]] = {}
    for r in rows:
        c = out.setdefault(r["scenario_id"], [0, 0])
        c[0] += int(r["reward"] == 1)
        c[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def make_reward(recorder: list[dict]):
    """Binary outcome, a program against the MATH answer. Every arm uses it
    untouched: the papers change the sampler, not the reward.

    `recorder` is refilled with the batch it just graded, in batch order:
    PTGS reads its successes back from it, and `hack_scan` reads the last
    one after training (Lambert 2025, chapter Over-optimization).
    """

    def reward(completions, prompts, gold, task_id, **kwargs) -> list[float]:
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        rewards = [outcome_of(t, g) for t, g in zip(texts, gold)]
        seen: dict[str, int] = {}
        recorder.clear()
        for key, prompt, text, r in zip(task_id, prompts, texts, rewards):
            recorder.append(
                {
                    "prompt": json.dumps(prompt, default=str),
                    "final_text": text,
                    "reward": r,
                    "scenario_id": key,
                    "rollout_index": seen.get(key, 0),
                }
            )
            seen[key] = seen.get(key, 0) + 1
        return rewards

    reward.__name__ = "math_outcome"
    return reward


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `compare` read: binary `reward`,
    one row per sample, grouped by task."""
    rows: list[dict] = []
    for task, texts in zip(holdout, replies):
        for i, text in enumerate(texts):
            rows.append(
                {
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": outcome_of(text, task["gold"]),
                    "scenario_id": task["scenario_id"],
                    "rollout_index": i,
                    "privileged": {"reference": task["gold"]},
                }
            )
    return rows


def ptgs_trainer(base_cls):
    """Build the trainer subclass. Takes `GRPOTrainer` as an argument so this
    module imports without torch, which is what lets `--selftest` run locally.
    """

    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from trl.trainer.utils import selective_log_softmax

    class RowTemperature(LogitsProcessor):
        """Divide row i's logits by its own temperature. HF `generate` takes
        one temperature per call; PTGS needs one per prompt group."""

        def __init__(self, temps: torch.Tensor) -> None:
            self.temps = temps

        def __call__(self, input_ids, scores):
            return scores / self.temps.to(scores.dtype)[:, None]

    class PTGSTrainer(base_cls):  # type: ignore[valid-type,misc]
        """GRPOTrainer whose rollouts come at a per-prompt temperature.

        Three overrides, nothing in the loss formula changes:
        `_generate_and_score_completions` draws T_x per prompt, generates the
        whole batch in one call with a per-row logits processor, and reads
        back the successes; `_compute_loss` hands the batch's temperatures to
        `_get_per_token_logps`, which divides each row's logits by its own
        T_x instead of TRL's one `temperature` (their Appendix A.5: the
        log-probs are taken at the rollout temperature).
        """

        def __init__(self, *args, recorder: list[dict], ptgs_seed: int = 0, **kw):
            super().__init__(*args, **kw)
            self.recorder = recorder
            self.posterior = Posterior(seed=ptgs_seed)
            self.t_ref = float(self.temperature)
            # The processor carries the whole temperature; HF must not apply
            # a second one on top.
            self.generation_config.temperature = 1.0
            self._row_temps: torch.Tensor | None = None
            if self.num_iterations != 1 or self.beta != 0.0:
                raise RuntimeError("PTGS here runs on-policy without KL: num_iterations=1, beta=0")
            if self.use_vllm:
                raise RuntimeError("PTGS here patches HF generate; run with use_vllm=False")

        def _generate_and_score_completions(self, inputs):
            n = self.num_generations
            unique = inputs[::n]
            p_target = target_at(self.state.global_step, self.args.max_steps)
            temps = [
                temperature(self.posterior.draw(u["task_id"], p_target), p_target, t_ref=self.t_ref)
                for u in unique
            ]
            device = self.accelerator.device
            row_t = torch.tensor([t for t in temps for _ in range(n)], device=device)
            model = self.accelerator.unwrap_model(self.model_wrapped)
            generate = model.generate

            def tempered(*a, **k):
                return generate(
                    *a, logits_processor=LogitsProcessorList([RowTemperature(row_t)]), **k
                )

            model.generate = tempered
            try:
                out = super()._generate_and_score_completions(inputs)
            finally:
                del model.generate
            rewards = [row["reward"] for row in self.recorder]
            if len(rewards) != len(inputs):
                raise RuntimeError("reward recorder is out of step with the batch")
            flat = 0
            for j, u in enumerate(unique):
                group = rewards[j * n : (j + 1) * n]
                self.posterior.update(u["task_id"], int(sum(group)), n)
                flat += len(set(group)) == 1
            metrics = self._metrics["train"]
            metrics["ptgs/p_target"].append(p_target)
            metrics["ptgs/mean_T"].append(statistics.fmean(temps))
            metrics["ptgs/heated_frac"].append(statistics.fmean(t > self.t_ref for t in temps))
            metrics["ptgs/zero_signal_frac"].append(flat / len(unique))
            out["ptgs_temperature"] = row_t.float()
            return out

        def _compute_loss(self, model, inputs):
            self._row_temps = inputs["ptgs_temperature"]
            try:
                return super()._compute_loss(model, inputs)
            finally:
                self._row_temps = None

        def _get_per_token_logps(
            self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
        ):
            temps = self._row_temps
            if temps is None:
                return super()._get_per_token_logps(
                    model, input_ids, attention_mask, logits_to_keep, batch_size
                )
            # TRL 0.19.1's body, with one temperature per row.
            batch_size = batch_size or input_ids.size(0)
            all_logps = []
            for i in range(0, input_ids.size(0), batch_size):
                ids = input_ids[i : i + batch_size]
                mask = attention_mask[i : i + batch_size]
                logits = model(
                    input_ids=ids, attention_mask=mask, logits_to_keep=logits_to_keep + 1
                ).logits
                logits = (
                    logits[:, :-1, :] / temps[i : i + batch_size].to(logits.dtype)[:, None, None]
                )
                all_logps.append(selective_log_softmax(logits, ids[:, -logits_to_keep:]))
            return torch.cat(all_logps, dim=0)

    return PTGSTrainer


def recorded_trainer(base_cls):
    """Plain GRPO with the recorder, so the baseline's last batch reaches
    `hack_scan` the same way."""

    class Recorded(base_cls):  # type: ignore[valid-type,misc]
        def __init__(self, *args, recorder: list[dict], **kw):
            super().__init__(*args, **kw)
            self.recorder = recorder

    return Recorded


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/papers/reinforce-ada.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-ptgs")

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
    """`n` replies per question, batched, at T_REF: every arm is evaluated
    at the same temperature, PTGS's tempering is a training-time sampler."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import T_REF, messages_for

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
                temperature=T_REF,
                top_p=1.0,
                top_k=0,
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
    timeout=6 * 60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 80,
    prompts_per_step: int = 12,
    learning_rate: float = 1e-4,
    max_completion_length: int = 512,
    lora_rank: int = 32,
    eval_samples: int = 8,
    eval_base: bool = False,
    train_seed: int = 17,
) -> dict:
    """One arm at one seed: eval the base model (optionally), train, eval again."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        EVAL_RUNS,
        GAMMA,
        GROUP,
        P_END,
        P_START,
        T_REF,
        TAU,
        graded_rows,
        make_reward,
        mean_length,
        messages_for,
        ptgs_trainer,
        recorded_trainer,
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
    ada = wai.methods.ReinforceAda(keep=GROUP)

    config = {
        "arm": arm,
        "sampler": {"baseline": "grpo", "ada": str(ada), "recipe": "ptgs"}[arm],
        "base_model": base_model,
        "steps": steps,
        "update_group": GROUP,
        "t_ref": T_REF,
        "ptgs": {"tau": TAU, "gamma": GAMMA, "p_target": [P_START, P_END]}
        if arm == "recipe"
        else None,
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
        "reward": "binary MathEqual against the MATH answer",
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
        [
            {"prompt": messages_for(t["question"]), "gold": t["gold"], "task_id": t["scenario_id"]}
            for t in train_tasks
        ]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        # Every arm trains on GROUP rollouts per prompt. The baseline and PTGS
        # draw exactly that many; Reinforce-Ada draws up to 32 and keeps GROUP,
        # so the backward pass is the same size in all three.
        num_generations=GROUP,
        per_device_train_batch_size=GROUP,
        gradient_accumulation_steps=prompts_per_step,
        learning_rate=learning_rate,
        # On-policy: one update per batch of rollouts. Reinforce-Ada rebuilds
        # the batch and PTGS re-tempers it, and neither carries old log-probs.
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        # Reward minus a baseline, no std division, in every arm.
        scale_rewards=False,
        max_completion_length=max_completion_length,
        max_prompt_length=512,
        temperature=T_REF,
        top_p=1.0,
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
    reward_funcs = [make_reward(last_batch)]
    # TRL 0.19.1 builds the LoRA adapter before it applies GRPOConfig.seed, so
    # seed here or the arm that ran the base evals draws a different adapter.
    set_seed(train_seed)
    common = dict(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    if arm == "recipe":
        trainer = ptgs_trainer(GRPOTrainer)(**common, recorder=last_batch, ptgs_seed=train_seed)
    elif arm == "ada":
        trainer = ada.trainer(GRPOTrainer)(**common)
    else:
        trainer = recorded_trainer(GRPOTrainer)(**common, recorder=last_batch)
    if run is not None:
        trainer.add_callback(TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise
    train_minutes = (time.time() - started) / 60.0

    # What each sampler did, step by step, read back off TRL's log.
    history = trainer.state.log_history
    trace = {
        key: [h[key] for h in history if key in h]
        for key in (
            "reward",
            "frac_reward_zero_std",
            "ada/drawn_per_prompt",
            "ada/no_gradient",
            "ada/pass_rate",
            "ptgs/mean_T",
            "ptgs/heated_frac",
            "ptgs/zero_signal_frac",
            "ptgs/p_target",
        )
    }

    replies = _sample(
        trainer.model, tokenizer, questions, n=eval_samples, max_new_tokens=max_completion_length
    )
    after_rows = graded_rows(holdout, replies)
    after = wai.pass_at(after_rows)
    print(f"{arm} s{train_seed}: {after}")

    # What the reward actually paid for in the last graded batch (chapter Over-optimization).
    scan = wai.hack_scan(last_batch) if last_batch else {}
    hack_top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
    hack_scan_top = hack_top.get("name", "") if isinstance(hack_top, dict) else str(hack_top)

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {"arm": arm, "pass_at_1": after.pass_at_1, "gpu_minutes": gpu_minutes}
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    return {
        "arm": arm,
        "base_runs": base_runs,
        "after_rows": after_rows,
        "gpu_minutes": gpu_minutes,
        "train_minutes": train_minutes,
        "length_after": mean_length(after_rows),
        "hack_scan_top": hack_scan_top,
        "trace": trace,
        "run_url": summary.get("run_url", ""),
    }


# --------------------------------------------------------------------------
# Local: data, orchestration, results.json.
# --------------------------------------------------------------------------


def data(seed: int, n_train: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    """MATH train at levels 3 to 5, MATH-500 at levels 3 to 5 held out.
    Disjoint by construction: MATH-500 is drawn from the MATH test split."""
    from datasets import load_dataset

    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    train = train.filter(lambda r: r["level"] in LEVELS).shuffle(seed=seed)
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test = test.filter(lambda r: int(r["level"]) >= 3).shuffle(seed=seed)
    train_tasks = []
    for i, r in enumerate(train):
        gold = last_boxed(r["solution"])
        if gold:
            train_tasks.append(
                {"question": r["problem"], "gold": gold, "scenario_id": f"train-{i}"}
            )
        if len(train_tasks) == n_train:
            break
    holdout = [
        {"question": r["problem"], "gold": r["answer"], "scenario_id": f"math500-{r['unique_id']}"}
        for r in test.select(range(min(n_holdout, len(test))))
    ]
    return train_tasks, holdout


def summarize(rows: list[dict]) -> dict:
    import whileai as wai

    p = wai.pass_at(rows)
    return {"score": p.pass_at_1, "ci": list(p.ci95 or (0.0, 0.0)), "pass_at_k": p.pass_at_k}


def pooled_curves(seed_rows: list[list[dict]], ids: list[str], k: int) -> list[list[float]]:
    """Per-task pass@k curves averaged over training seeds, in `ids` order."""
    per_seed = [task_counts(rows) for rows in seed_rows]
    return [
        [statistics.fmean(col) for col in zip(*(curves([c[t]], k)[0] for c in per_seed))]
        for t in ids
    ]


def selftest() -> None:
    """The temperature rule, the posterior and the tax, on the CPU. No GPU,
    no key, no model download."""
    # Eq. (4) passes through its three anchor points.
    for p_t in (0.25, 0.5):
        assert math.isclose(temperature(0.0, p_t), TAU * T_REF)
        assert math.isclose(temperature(p_t, p_t), T_REF)
        assert math.isclose(temperature(1.0, p_t), T_REF / TAU)
    # At p~ = 1/2 it is the symmetric rule T = T_ref * tau ** (1 - 2 p).
    for p in (0.1, 0.4, 0.8):
        assert math.isclose(temperature(p, 0.5), T_REF * TAU ** (1 - 2 * p))
    assert math.isclose(target_at(0, 80), P_START) and math.isclose(target_at(79, 80), P_END)

    # A prompt that keeps failing is heated, one that keeps passing is cooled.
    post = Posterior(seed=0)
    for _ in range(4):
        post.update("hard", 0, GROUP)
        post.update("easy", GROUP, GROUP)
    hot = statistics.fmean(temperature(post.draw("hard", 0.5), 0.5) for _ in range(2000))
    cold = statistics.fmean(temperature(post.draw("easy", 0.5), 0.5) for _ in range(2000))
    print(f"mean T after four all-wrong groups {hot:.2f}, after four all-right {cold:.2f}")
    assert hot > T_REF > cold
    # Forgetting: the counts decay by gamma before each new group lands.
    s, f = post.counts["hard"]
    assert s == 0 and math.isclose(f, sum(GROUP * GAMMA**i for i in range(4)))

    # The tax. A task solved always or never adds nothing to A(K)
    # (Proposition 1); the unbiased estimator at n = k is "any of n right".
    assert scalability(curves([(8, 8), (0, 8)], 8)) == (0.0, 0.0)
    assert pass_at_k(1, 8, 8) == 1.0 and math.isclose(pass_at_k(1, 8, 1), 1 / 8)
    # Theorem 2 on a population: sharpen a fraction lam of tasks to 0 or 1 and
    # A(K) keeps exactly (1 - lam) of the base's.
    rng = random.Random(0)
    k, n = 8, 64
    base = [
        (sum(rng.random() < p for _ in range(n)), n) for p in (rng.random() for _ in range(400))
    ]
    lam = 0.5
    sharp = [((0 if i % 2 else n), n) if i < lam * len(base) else c for i, c in enumerate(base)]
    a_base, _ = scalability(curves(base, k))
    a_sharp, _ = scalability(curves(sharp, k))
    print(f"A(8): base {a_base:.3f}, half the tasks sharpened {a_sharp:.3f}")
    assert math.isclose(a_sharp, (1 - lam) * a_base, rel_tol=0.15)
    tax = sharpening_tax(curves(base, k), curves(sharp, k), boot=200)
    assert tax["tax_s"] > 0 and tax["ci"][0] <= tax["tax_s"] <= tax["ci"][1]

    # The grader is a program, not a judge.
    assert outcome_of("so the answer is \\boxed{\\frac{1}{2}}", "\\frac12") == 1.0
    assert outcome_of("the answer is \\boxed{5}", "18") == 0.0
    assert last_boxed("x \\boxed{\\frac{1}{2}} y") == "\\frac{1}{2}"
    print("grader: MathEqual, decided by Math-Verify")
    print("selftest ok")


ARMS = ("baseline", "ada", "recipe")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--k", type=int, default=8, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0, help="data shuffle seed")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[17, 18])
    # 192 prompts x 80 steps / 12 per step = 5 visits per prompt, so PTGS's
    # per-prompt counts have something to remember.
    ap.add_argument("--n-train", type=int, default=192)
    ap.add_argument("--n-holdout", type=int, default=320)
    ap.add_argument("--prompts-per-step", type=int, default=12)
    ap.add_argument("--selftest", action="store_true", help="the sampler and the tax, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    arms = list(ARMS) if args.arm == "all" else [args.arm]

    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(
        {
            "recipe": HERE.name,
            "title": "PTGS: heat the prompts the policy keeps failing, cool the ones it has mastered",
            "paper": "https://arxiv.org/abs/2610.01509",
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
    seed_rows: dict[str, list[list[dict]]] = {arm: [] for arm in arms}
    traces: dict[str, list[dict]] = {arm: [] for arm in arms}
    minutes: dict[str, list[float]] = {arm: [] for arm in arms}
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes, run_std, run_url = 0.0, 0.0, ""
    base_rows: list[dict] = []

    jobs = [(arm, s) for s in args.train_seeds for arm in arms]
    with modal.enable_output(), app.run():
        # Every (arm, seed) is its own container, so they run side by side.
        calls = [
            run_arm.spawn(
                arm,
                train_tasks,
                holdout,
                f"ptgs-{arm}-s{s}-{date.today().isoformat()}",
                steps=args.steps,
                prompts_per_step=args.prompts_per_step,
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
            base_rows = base_runs[0]
            noise = wai.eval_variance(*base_runs)
            run_std = float(noise["run_std"])
            print(f"eval noise over {len(base_runs)} base runs: run_std {run_std:.4f}")
            results["arms"]["base"] = {**summarize(base_rows), "steps": 0, "gpu_minutes": 0}
            checks["run_std"] = run_std
            checks["run_std_runs"] = int(noise["n_runs"])
            checks["length_before"] = mean_length(base_rows)
        seed_rows[arm].append(out["after_rows"])
        traces[arm].append(out["trace"])
        checks["hack_scan_top"][arm] = out["hack_scan_top"]
        checks["length_after"][arm] = out["length_after"]

    ids = sorted(task_counts(base_rows)) if base_rows else []
    base_curves = curves([task_counts(base_rows)[t] for t in ids], args.k) if base_rows else []
    pooled: dict[str, list[list[float]]] = {}
    for arm in arms:
        rows_all = [r for rows in seed_rows[arm] for r in rows]
        pooled[arm] = pooled_curves(seed_rows[arm], ids, args.k) if ids else []
        results["arms"][arm] = {
            **summarize(seed_rows[arm][0]),
            "per_seed": [summarize(rows)["score"] for rows in seed_rows[arm]],
            "per_seed_pass_at_k": [summarize(rows)["pass_at_k"] for rows in seed_rows[arm]],
            "pooled_score": summarize(rows_all)["score"],
            "steps": args.steps,
            "gpu_minutes": round(statistics.fmean(minutes[arm]), 1),
        }
        if base_curves:
            results["arms"][arm]["sharpening_tax"] = sharpening_tax(base_curves, pooled[arm])
        results.setdefault("sampler", {})[arm] = traces[arm]

    def verdict(before: str, after: str) -> dict:
        d = wai.compare(
            seed_rows[before][0],
            seed_rows[after][0],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks.get("run_std_runs") or EVAL_RUNS),
            train_runs={"before": seed_rows[before], "after": seed_rows[after]},
            proxy=PROXY,
        )
        cover = paired_delta([c[-1] for c in pooled[before]], [c[-1] for c in pooled[after]])
        return {
            "pass_at_1": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": d["target_verdict"]
            if d["target_verdict"] in ("moved", "flat", "unresolved")
            else "flat",
            f"pass_at_{args.k}": cover["delta"],
            f"pass_at_{args.k}_ci": cover["ci"],
            "over_optimized": bool(d.get("over_optimized")),
        }

    if set(ARMS) <= set(seed_rows):
        main_delta = verdict("baseline", "recipe")
        results["delta"] = {
            "recipe_vs_baseline": main_delta["pass_at_1"],
            "ci": main_delta["ci"],
            "verdict": main_delta["verdict"],
        }
        results["deltas"] = {
            "ptgs_vs_grpo": main_delta,
            "ada_vs_grpo": verdict("baseline", "ada"),
            "ptgs_vs_ada": verdict("ada", "recipe"),
        }
        checks["train_seeds"] = {
            "baseline": len(seed_rows["baseline"]),
            "recipe": len(seed_rows["recipe"]),
            "ada": len(seed_rows["ada"]),
        }
        checks["over_optimized"] = main_delta["over_optimized"]
        results["verified"] = date.today().isoformat()
        results.pop("partial_run", None)
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"
        print(f"partial run ({', '.join(arms)}): deltas and verified left as they were")

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["gpu_minutes_total"] = round(gpu_minutes, 1)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "sampler"}, indent=2))


if __name__ == "__main__":
    main()
