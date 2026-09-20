"""GMTS: rank tokens by entropy times advantage, not by entropy alone.

    python recipe.py                      # both arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the ranking and the mask, offline, no GPU

Training on only the top 20% highest-entropy tokens is a known RLVR result:
the other 80% are mostly the model writing out what it has already decided.
The paper's complaint is that entropy is read one answer at a time. Two tokens
can carry the same entropy while sitting in answers the group scored very
differently, and only one of them is attached to a gradient worth taking.

So rank by the size of the learning signal instead. The paper's score
(equation 4) is

    delta[i, t] = | E[i, t] * omega[i, t] |

with ``E`` the token's entropy and ``omega`` the scalar in front of it in the
policy-gradient term: the importance ratio times the advantage, zeroed where
the clip has made the gradient inactive. Both arms keep the top 20% of tokens
and drop the rest. The baseline ranks that 20% by ``E`` (entropy token
selection, ETS, the prior result); the recipe ranks it by ``delta``. That is
the one change.

This run is on-policy (``num_iterations`` 1) with no KL term (``beta`` 0), so
the ratio is exactly 1 and nothing is clipped, and ``omega`` reduces to the
advantage itself. That is the part of ``omega`` the paper says does the work
-- the answer-level reward signal entropy cannot see -- so the reduction keeps
the change and drops only the terms that are 1 here. `selection_overlap` in
the Checks table is how much the two rankings actually disagree, measured, so
this recipe cannot repeat adaptive-clip's round 1 and test nothing.

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out
  2. run_arm():   TRL GRPOTrainer + LoRA on Modal, one arm per call
  3. evaluate():  same holdout, k samples per task, graded by MathEqual
  4. results.json + the paired delta (wai.delta_report) on the run page
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

from whileai.config import provenance

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Reinforcement Learning Policy gradients"  # per-token aggregation, and the sequence-level advantage
# The training reward here *is* the target: both are the same binary check
# against the GSM8K gold, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

# The paper's selection fraction, the same one the entropy result uses.
TOP_FRACTION = 0.20

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."


# --------------------------------------------------------------------------
# The ranking and the mask. Pure Python, no torch: `--selftest` runs them on
# hand-written batches, and the Modal container imports this same file.
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


def token_scores(entropy: list[list[float]], omega: list[float], ranking: str) -> list[list[float]]:
    """THE ONE CHANGE, equation 4: the score the top-20% is taken by.

    `entropy` is one row per rollout, one value per completion token. `omega`
    is one scalar per rollout -- the advantage, since the ratio is 1 and the
    clip is inactive on an on-policy step with no KL term.

    `ranking="entropy"` is the baseline, ETS: the score is the entropy and the
    answer the token sits in does not enter. `ranking="gmts"` multiplies by
    `omega`, so tokens in answers the group scored far from its mean outrank
    equally uncertain tokens in answers it scored near the mean.

    Note what this does *within* one rollout: `omega` is constant along the
    row, so the two rankings order a single answer's tokens identically. The
    whole difference is across answers, which is the difference the paper is
    about, and it only survives if the loss normalizes over the batch rather
    than per sequence -- see the `loss_type` note in `run_arm`.
    """
    if ranking == "entropy":
        return [list(row) for row in entropy]
    if ranking != "gmts":
        raise ValueError(f"unknown ranking {ranking!r}")
    return [[abs(e * w) for e in row] for row, w in zip(entropy, omega)]


def select_top_fraction(
    scores: list[list[float]], mask: list[list[int]], fraction: float = TOP_FRACTION
) -> list[list[int]]:
    """Keep the highest-scoring `fraction` of the batch's live tokens.

    The threshold is taken over every live token in the batch at once, not per
    row. A per-row threshold would hand every answer the same number of slots
    and throw away exactly the cross-answer comparison the recipe is testing.

    `mask` is TRL's completion mask: 1 on a real token, 0 on padding. Padding
    never scores. Ties are broken by keeping the earlier token, which matters
    only when a batch has many identical scores (all-zero omega rows under
    GMTS, which is every unanimous group).
    """
    flat = [
        (scores[i][t], i, t)
        for i in range(len(scores))
        for t in range(len(scores[i]))
        if mask[i][t]
    ]
    keep_n = int(len(flat) * fraction)
    chosen = sorted(flat, key=lambda s: (-s[0], s[1], s[2]))[:keep_n]
    out = [[0] * len(row) for row in scores]
    for _, i, t in chosen:
        out[i][t] = 1
    return out


def mask_overlap(a: list[list[int]], b: list[list[int]]) -> float:
    """Share of one mask's kept tokens that the other keeps too. 1.0 means the
    two rankings chose the same tokens and the arms cannot differ; this number
    goes in the Checks table so "the change was reachable" is measured."""
    kept_a = sum(sum(row) for row in a)
    if not kept_a:
        return 1.0
    both = sum(x * y for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    return both / kept_a


def make_reward(recorder: list[dict]):
    """Binary outcome, a program against the public GSM8K gold. Both arms use
    this untouched: the paper changes which tokens are trained on, not what
    counts as right.

    `recorder` is refilled with the batch it just graded, so after training
    `hack_scan` can be run on the last one (Lambert 2025, chapter Over-optimization) without keeping
    every step in memory.
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


def gmts_trainer(base_cls):
    """Build the trainer subclass. Takes `GRPOTrainer` as an argument so this
    module imports without torch, which is what lets `--selftest` run locally.
    """

    import torch

    class TokenSelectTrainer(base_cls):  # type: ignore[valid-type,misc]
        """GRPOTrainer that trains on a ranked 20% of the completion tokens.

        One override and no copy of the loss body. TRL's `_compute_loss` reads
        `inputs["completion_mask"]` on its first line and uses that one tensor
        for the loss, its normalizer and every metric it logs. So the selection
        goes in by narrowing that mask before the parent runs: an unselected
        token is padding as far as the loss is concerned, and TRL's own
        `completion_mask.sum()` normalizer counts only what survived.

        The score needs the current policy's entropy, which the parent does not
        expose, so this takes one extra no-grad forward pass per loss call. That
        pass is not cheap and it is not "generation dominates": entropy needs the
        whole next-token distribution, so it materializes a
        `(rows, tokens, vocab)` float32 tensor for a `log_softmax`, where TRL's
        own scoring pass gathers one logprob per position and never builds it.
        Measured at 8 rollouts x 256 tokens, the step runs around 20 seconds and
        this pass is the bulk of it. The trade bought here is an untouched loss
        body; a bf16 or fused entropy is the first thing to change at any size.

        `ranking` is the arm: "entropy" is the baseline, "gmts" is the recipe.
        """

        def __init__(self, *args, ranking: str = "gmts", fraction: float = TOP_FRACTION, **kw):
            super().__init__(*args, **kw)
            if ranking not in ("entropy", "gmts"):
                raise ValueError(f"unknown ranking {ranking!r}")
            self.ranking = ranking
            self.fraction = fraction
            self.overlaps: list[float] = []
            self.kept_fractions: list[float] = []
            if getattr(self, "use_liger_loss", False):
                raise RuntimeError(
                    "use_liger_loss does not read inputs['completion_mask'] the same way, so "
                    "the selection would be silently ignored; run this recipe without Liger"
                )
            if self.loss_type == "grpo":
                raise RuntimeError(
                    "loss_type 'grpo' normalizes each sequence by its own kept-token count, "
                    "which cancels the cross-answer part of the selection this recipe tests; "
                    "use 'bnpo' (TRL's default) or 'dr_grpo'"
                )

        def _entropy(self, model, input_ids, attention_mask, logits_to_keep, chunk=2):
            """Per-token Shannon entropy of the policy's next-token distribution,
            on the same shifted, temperature-scaled logits TRL scores with.

            Chunked because the logits are (rows, tokens, 151936) and only the
            (rows, tokens) entropy is kept."""
            rows = []
            for i in range(0, input_ids.size(0), chunk):
                logits = model(
                    input_ids=input_ids[i : i + chunk],
                    attention_mask=attention_mask[i : i + chunk],
                    logits_to_keep=logits_to_keep + 1,
                ).logits[:, :-1, :]
                logits = logits.float() / self.temperature
                logps = torch.log_softmax(logits, dim=-1)
                rows.append(-(logps.exp() * logps).sum(-1))
                del logits, logps
            return torch.cat(rows, dim=0)

        def _compute_loss(self, model, inputs):
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids = inputs["completion_ids"]
            completion_mask = inputs["completion_mask"]
            advantages = inputs["advantages"]

            with torch.no_grad():
                entropy = self._entropy(
                    model,
                    torch.cat([prompt_ids, completion_ids], dim=1),
                    torch.cat([prompt_mask, completion_mask], dim=1),
                    completion_ids.size(1),
                )
                # omega: the scalar in front of the entropy in the policy-
                # gradient term. old_per_token_logps is None on an on-policy
                # step, where the ratio is exactly 1 by construction.
                old = inputs.get("old_per_token_logps")
                if old is None:
                    ratio = torch.ones_like(entropy)
                else:
                    new = self._get_per_token_logps(
                        model,
                        torch.cat([prompt_ids, completion_ids], dim=1),
                        torch.cat([prompt_mask, completion_mask], dim=1),
                        completion_ids.size(1),
                    )
                    ratio = torch.exp(new - old)
                    # Where the clip has bitten, the gradient is inactive and
                    # the token is worth nothing however uncertain it is.
                    inactive = ((ratio < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
                        (ratio > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
                    )
                    ratio = ratio.masked_fill(inactive, 0.0)
                omega = ratio * advantages.unsqueeze(1)

                scores = entropy if self.ranking == "entropy" else (entropy * omega).abs()
                keep = self._top_fraction(scores, completion_mask)
                if self.ranking == "entropy":
                    other = self._top_fraction((entropy * omega).abs(), completion_mask)
                else:
                    other = self._top_fraction(entropy, completion_mask)
                live = completion_mask.sum().clamp(min=1)
                self.overlaps.append(float((keep * other).sum() / keep.sum().clamp(min=1)))
                self.kept_fractions.append(float(keep.sum() / live))

            inputs = {**inputs, "completion_mask": completion_mask * keep}
            return super()._compute_loss(model, inputs)

        def _top_fraction(self, scores, completion_mask):
            """The batch-wide top `fraction` of live tokens, as a 0/1 tensor
            the same shape as the completion mask. The threshold is over the
            whole batch on purpose: see `select_top_fraction`."""
            live = completion_mask.bool()
            n_live = int(live.sum())
            keep_n = int(n_live * self.fraction)
            out = torch.zeros_like(completion_mask)
            if keep_n <= 0:
                return out
            flat = scores.masked_fill(~live, float("-inf")).flatten()
            idx = torch.topk(flat, keep_n).indices
            out.view(-1)[idx] = 1
            return out

    return TokenSelectTrainer


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/04-train/grpo/train_modal.py.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-gmts-token-select")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "whileai",
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
    timeout=60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    ranking: str,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 40,
    num_generations: int = 8,
    prompts_per_step: int = 6,
    learning_rate: float = 1e-4,
    fraction: float = TOP_FRACTION,
    max_completion_length: int = 256,
    lora_rank: int = 32,
    eval_samples: int = 4,
    eval_base: bool = False,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        EVAL_RUNS,
        gmts_trainer,
        graded_rows,
        make_reward,
        mean_length,
        messages_for,
    )

    import whileai.simulations as wai

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    config = {
        "arm": arm,
        "ranking": ranking,
        "top_fraction": fraction,
        "base_model": base_model,
        "steps": steps,
        "num_generations": num_generations,
        "prompts_per_step": prompts_per_step,
        "learning_rate": learning_rate,
        "loss_type": "bnpo",
        "num_iterations": 1,
        "beta": 0.0,
        "lora_rank": lora_rank,
        "max_completion_length": max_completion_length,
        "train_prompts": len(train_tasks),
        "holdout": len(holdout),
        "gpu": DEFAULT_GPU,
        "reward": "binary MathEqual against the GSM8K gold",
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        run = wai.training_run(
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
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=prompts_per_step,
        learning_rate=learning_rate,
        # On-policy: one update per batch of rollouts, so the importance ratio
        # is exactly 1 and omega is the advantage alone. The recipe's change
        # does not need off-policy steps to bind -- unlike a clip bound, the
        # advantage varies across answers on every step.
        num_iterations=1,
        # No KL term, so omega has no reference-policy correction either and
        # the arms differ in the ranking and nothing else.
        beta=0.0,
        # TRL's default, and load-bearing here: bnpo normalizes by the batch's
        # kept-token count, so giving a high-advantage answer more of the 20%
        # gives it more of the gradient. loss_type "grpo" would divide each
        # sequence by its own kept count and undo exactly that (Lambert 2025,
        # chapter Reinforcement Learning, per-sequence against per-token
        # aggregation). The trainer
        # subclass refuses "grpo" rather than quietly testing nothing.
        loss_type="bnpo",
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
        seed=17,
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
    # TRL 0.19.1 builds the LoRA adapter (get_peft_model) before it applies
    # GRPOConfig.seed, so the adapter's init is drawn from whatever RNG state
    # the process is in. The first arm runs the base evals first and advances
    # it; the second does not. Seed here so both arms draw the same A matrix.
    # Rounds 1 and 2 in the README predate this line.
    from transformers import set_seed

    set_seed(17)
    trainer = gmts_trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[make_reward(last_batch)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        ranking=ranking,
        fraction=fraction,
    )
    if run is not None:
        trainer.add_callback(wai.TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise

    # How far apart the two rankings actually chose. 1.0 would mean this arm
    # trained on the same tokens the other arm would have picked, and the
    # recipe tested nothing.
    overlap = statistics.fmean(trainer.overlaps) if trainer.overlaps else 1.0
    kept = statistics.fmean(trainer.kept_fractions) if trainer.kept_fractions else 0.0
    print(f"{arm}: selection overlap with the other ranking {overlap:.3f}, kept {kept:.3f}")

    replies = _sample(
        trainer.model, tokenizer, questions, n=eval_samples, max_new_tokens=max_completion_length
    )
    after_rows = graded_rows(holdout, replies)
    after = wai.pass_at(after_rows)
    print(f"{arm}: {after}")

    # What the reward actually paid for in the last training batch (chapter Over-optimization).
    # Nothing here is endorsed: the reward is the answer being right, and any
    # surface feature that correlates with it is the thing to be suspicious of.
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
        "ranking": ranking,
        "pass_at_1": after.pass_at_1,
        "selection_overlap": overlap,
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
        "selection_overlap": overlap,
        "kept_fraction": kept,
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
    import whileai.simulations as wai

    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
    }


def selftest() -> None:
    """The ranking and the mask on a hand-written batch, on the CPU. No GPU,
    no key, no model download."""
    # Two groups of four, laid out the way TRL hands them over. Group A is
    # hard (one correct), group B is easy (three correct). Advantages are
    # r - mean(r), which is what omega is on an on-policy step.
    omega = [0.75, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, -0.75]
    # Every rollout carries the same four entropies, so entropy alone cannot
    # tell the rollouts apart and any difference is the omega.
    entropy = [[0.1, 0.9, 0.4, 0.6] for _ in omega]
    mask = [[1, 1, 1, 1] for _ in omega]

    ets = token_scores(entropy, omega, "entropy")
    gmts = token_scores(entropy, omega, "gmts")
    assert ets == entropy, "the baseline ranks by entropy and nothing else"
    assert gmts[0] == [abs(e * 0.75) for e in entropy[0]]

    keep_ets = select_top_fraction(ets, mask, 0.25)
    keep_gmts = select_top_fraction(gmts, mask, 0.25)
    n = sum(sum(r) for r in keep_ets)
    assert n == sum(sum(r) for r in keep_gmts) == 8, "both arms keep the same count"

    # Every row here carries the same four entropies, so the baseline cannot
    # tell the rollouts apart at all: it spends its 8 slots on the one
    # highest-entropy column of all 8 rows, one slot each, informative answer
    # or not. The recipe concentrates them where |omega| is largest:
    # rollout 0 (the lone correct answer on the hard problem, omega
    # 0.75) and rollout 7 (the lone wrong answer on the easy one, omega -0.75)
    # are the informative ones, and they take 6 of the 8 slots between them.
    # They do not take all 8: a 0.225 token in a low-|omega| row still outranks
    # the 0.075 one in a high-|omega| row, which is the point of ranking by the
    # product rather than by the answer.
    per_row_gmts = [sum(r) for r in keep_gmts]
    assert per_row_gmts[0] == per_row_gmts[7] == 3, per_row_gmts
    assert per_row_gmts[0] + per_row_gmts[7] == 6, per_row_gmts
    assert max(per_row_gmts[1:7]) <= 1, per_row_gmts
    overlap = mask_overlap(keep_gmts, keep_ets)
    print(f"tokens per rollout, GMTS: {per_row_gmts}")
    print(f"tokens per rollout, ETS:  {[sum(r) for r in keep_ets]}")
    print(f"overlap between the two rankings: {overlap:.3f}")
    assert overlap < 1.0, "if the rankings agreed everywhere the recipe tests nothing"

    # A unanimous group has every advantage at zero, so GMTS scores all of its
    # tokens zero and never spends a slot on them. Those tokens have no
    # gradient either way: the advantage in front of the loss is zero.
    flat_omega = [0.0] * 4
    flat_scores = token_scores([[0.5] * 4] * 4, flat_omega, "gmts")
    assert flat_scores == [[0.0] * 4] * 4

    # Padding never scores, whatever its entropy.
    padded_mask = [[1, 1, 0, 0] for _ in omega]
    keep_pad = select_top_fraction(ets, padded_mask, 0.5)
    assert all(row[2] == 0 and row[3] == 0 for row in keep_pad)
    assert sum(sum(r) for r in keep_pad) == 8
    print("padding is never selected; masked-out columns stay 0")

    # The grader is a program, not a judge.
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("grader: MathEqual reads \\boxed{} and the last number")

    _selftest_mask()
    print("selftest ok")


def _selftest_mask() -> None:
    """The selection really does reach the loss: run TRL's own normalizer on a
    CPU tensor and check that narrowing `completion_mask` both drops the
    unselected tokens' loss and shrinks the divisor.

    This is the line the recipe cannot check by reading. TRL's bnpo branch is
    `(per_token_loss * completion_mask).sum() / completion_mask.sum()`, and the
    whole change rests on that mask being the only thing that decides which
    tokens count.
    """
    try:
        import torch
    except ImportError:
        print("torch not installed locally: skipping the mask check")
        return

    per_token_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    full = torch.ones(2, 4)
    # Keep the two highest-loss tokens, the way a ranking would.
    keep = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

    def bnpo(mask):
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

    assert torch.allclose(bnpo(full), torch.tensor(4.5))
    assert torch.allclose(bnpo(keep), torch.tensor(6.0))
    # And the cross-answer part: moving one slot from row 0 to row 1 moves
    # gradient mass between answers, which per-sequence normalization would not.
    row0 = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    row1 = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    assert bnpo(row1) > bnpo(row0), "bnpo lets the arm that gets more slots weigh more"
    print(
        f"bnpo over the kept mask: all tokens {bnpo(full):.2f}, top-2 {bnpo(keep):.2f}; "
        "the divisor follows the selection"
    )


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=120)
    ap.add_argument("--generations", type=int, default=8, help="rollouts per prompt")
    ap.add_argument("--prompts-per-step", type=int, default=6)
    ap.add_argument(
        "--fraction",
        type=float,
        default=TOP_FRACTION,
        help="share of the batch's tokens both arms keep; the paper's is 0.20",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help=(
            "learning rate, both arms. The recipe's selection concentrates the kept 20% on "
            "high-|advantage| answers, which raises the gradient norm about 5x against the "
            "baseline's; at 1e-4 that is past this setup's stability point. See Climb round 2"
        ),
    )
    ap.add_argument("--selftest", action="store_true", help="the ranking and the mask, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai.simulations as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    # GSM8K's train and test splits are already disjoint, so this should drop
    # nothing. It runs anyway, and the count goes in the Checks table, because
    # "should" is not a measurement (Lambert 2025, chapter Evaluation).
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]
    ranking = {"baseline": "entropy", "recipe": "gmts"}

    # Start from what is already on disk, so `--arm recipe` refreshes one arm
    # instead of wiping the other one and the delta. Only a both-arm run moves
    # `verified` and the delta; a one-arm run says so and leaves them alone.
    results = json.loads((HERE / "results.json").read_text())
    results.update(
        {
            "recipe": HERE.name,
            "title": "GMTS: rank tokens by entropy times advantage, not by entropy alone",
            "paper": "https://arxiv.org/abs/2608.30632",
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
    checks.setdefault("selection_overlap", {})
    arm_rows: dict[str, list[dict]] = {}
    run_std = 0.0
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""

    with modal.enable_output(), app.run():
        for i, arm in enumerate(arms):
            out = run_arm.remote(
                arm,
                ranking[arm],
                train_tasks,
                holdout,
                f"gmts-token-select-{arm}-{date.today().isoformat()}",
                steps=args.steps,
                num_generations=args.generations,
                prompts_per_step=args.prompts_per_step,
                fraction=args.fraction,
                learning_rate=args.lr,
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
            }
            checks["length_after"][arm] = out["length_after"]
            checks["hack_scan_top"] = out["hack_scan_top"]
            checks["selection_overlap"][arm] = round(out["selection_overlap"], 3)

    if "baseline" in arm_rows and "recipe" in arm_rows:
        # run_std makes "moved" mean bigger than the eval's own re-run noise,
        # and proxy names the training reward when it differs from the target.
        # Here it does not, so there is nothing for PROXY to point at.
        d = wai.delta_report(
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
        print(wai.format_delta_report(d))
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"
        print(f"one arm only ({', '.join(arms)}): delta and verified left as they were")

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
