"""Context LM: train a model that keeps its own context as a file, and pay it
for keeping that file short once it gets the answer right.

    python recipe.py --selftest      # the task, the file, the cost, the credit, offline
    python recipe.py --smoke         # Modal: 2 steps, 16 held-out tasks, the live path
    python recipe.py --reuse         # both arms, two seeds, writes results.json

Context Language Models (arXiv:2609.37725) give the model its context as a
file it can rewrite, instead of a transcript that only grows. Trained with
stepwise GRPO, they add one term to the advantage: among the trajectories
that succeeded, the cheaper ones (fewer prefix FLOPs) are pushed up and the
dearer ones down; a failed trajectory gets no efficiency credit (Eq. 6).

This recipe keeps that harness and that term on a task a 1.5B model can
learn in an hour, a ContextBench-style key-value log:

  every step  the model sees context.md and one chunk of a log of updates
              (`set river = 418`), then the chunk is gone; it replies with the
              whole new context.md
  last step   it sees only context.md and a question (`final value of
              river?`) and boxes a number

Two arms, same harness, same reward, same steps and seeds:

  baseline  stepwise GRPO: every step of a trajectory gets the outcome
            advantage, r minus the group mean
  recipe    baseline plus W_EFF times the success-gated efficiency advantage

The paper's claim to test: the recipe arm holds pass@1 while spending
fewer context tokens per task.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
PAPER = "https://arxiv.org/abs/2609.37725"
METRIC = "pass@1"
BOOK = "Reinforcement Learning"
# The training reward is the target (the boxed number against the log's
# final state), so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

# The task. Each is a flag-free constant: the README's table names them.
KEYS = ("river", "stone", "maple", "cloud", "ember", "frost", "orbit", "pearl")
KEYS_PER_TASK = 8  # keys a log touches: all of them
CHUNKS = 5  # log chunks, one model step each, before the question step
UPDATES_PER_CHUNK = 8  # `set key = value` lines per chunk
FILE_TOKENS = 256  # cap on one rewrite of context.md
ANSWER_TOKENS = 48  # cap on the answer step
STEPS_PER_TASK = CHUNKS + 1

# The credit.
GROUP = 8  # trajectories per task: the GRPO group; the efficiency term needs 2+ successes in it
W_EFF = 0.25  # weight on Eq. 6, the paper's BrowseComp-Plus run (POLAR_DUAL_CHANNEL_W_EFF)

SYSTEM = (
    "You keep notes in a file called context.md. Each turn you see the file and one "
    "new chunk of a log; after the turn the chunk is gone and only context.md is left. "
    "At the end you will be asked a question about the log. Reply with the complete "
    "new contents of context.md and nothing else. Anything you leave out is lost."
)


# --------------------------------------------------------------------------
# Pure functions, no torch: `--selftest` runs them, and the Modal container
# imports this same file.
# --------------------------------------------------------------------------


def make_task(seed: int, prefix: str) -> dict:
    """One key-value log in CHUNKS chunks and a question about its final
    state. The asked key is set at least twice, the last time after the
    first chunk, so the first value seen is never the answer."""
    rng = random.Random(seed)
    keys = rng.sample(KEYS, KEYS_PER_TASK)
    ask = keys[0]
    lines: list[tuple[str, int]] = []
    for _ in range(CHUNKS * UPDATES_PER_CHUNK):
        lines.append((rng.choice(keys), rng.randrange(100, 1000)))
    # Pin the asked key: once in the first chunk, once in a later one.
    first = rng.randrange(UPDATES_PER_CHUNK)
    later = rng.randrange(UPDATES_PER_CHUNK, CHUNKS * UPDATES_PER_CHUNK)
    lines[first] = (ask, rng.randrange(100, 1000))
    lines[later] = (ask, rng.randrange(100, 1000))
    state: dict[str, int] = {}
    for k, v in lines:
        state[k] = v
    chunks = [
        [f"set {k} = {v}" for k, v in lines[i * UPDATES_PER_CHUNK : (i + 1) * UPDATES_PER_CHUNK]]
        for i in range(CHUNKS)
    ]
    return {
        "scenario_id": f"{prefix}-{seed}",
        "chunks": chunks,
        "ask": ask,
        "gold": str(state[ask]),
        "question": f"final value of {ask}?",
    }


def make_tasks(n: int, seed: int, prefix: str) -> list[dict]:
    """n tasks from disjoint seed ranges: train and holdout never share a log."""
    base = {"train": 0, "test": 1_000_000}[prefix] + seed * 10_000
    return [make_task(base + i, prefix) for i in range(n)]


def step_messages(task: dict, t: int, context: str) -> list[dict]:
    """What the model sees at step t: context.md and the next chunk, or, at
    the last step, context.md and the question."""
    shown = context.strip() or "(empty)"
    if t < CHUNKS:
        body = (
            f"context.md:\n```\n{shown}\n```\n\n"
            f"Log chunk {t + 1} of {CHUNKS}:\n" + "\n".join(task["chunks"][t]) + "\n\n"
            "Write the new context.md."
        )
    else:
        body = (
            f"context.md:\n```\n{shown}\n```\n\n"
            f"Question: what is the final value of {task['ask']}? "
            "Answer with the number only, as \\boxed{n}."
        )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


FENCE = re.compile(r"^```[a-z]*\n?|\n?```$")


def as_file(reply: str) -> str:
    """The reply is the new context.md; a model that wraps it in a code
    fence still means the inside."""
    return FENCE.sub("", reply.strip()).strip()


def boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = text.find("}", i)
    return text[i + 7 : j].replace(",", "").strip() if j > 0 else None


def outcome(answer: str, gold: str) -> float:
    """1.0 when the boxed number is the log's final value for the asked key.
    A program, not a judge."""
    return 1.0 if boxed(answer) == gold else 0.0


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def episode_cost(prompts: list[list[int]], replies: list[list[int]]) -> int:
    """c_i, prefix-reuse cost in tokens: each step pays for the prompt
    tokens past the longest prefix it shares with the previous step's
    prompt plus reply (those sit in the KV cache), and for its reply. The
    paper's `kv_cache_flops.py` multiplies the same token counts by 2N and
    adds an attention term; for one fixed model the linear term is a
    constant factor, and at these lengths the attention term is small."""
    cost, prev = 0, []
    for p, r in zip(prompts, replies):
        cost += len(p) - _common_prefix(p, prev) + len(r)
        prev = list(p) + list(r)
    return int(cost)


def outcome_advantage(rewards: list[float]) -> list[float]:
    """Stepwise GRPO: r minus the group mean, the same for every step of
    the trajectory (no std scaling)."""
    mean = statistics.fmean(rewards)
    return [r - mean for r in rewards]


def efficiency_advantage(rewards: list[float], costs: list[float]) -> list[float]:
    """THE ONE CHANGE (Eq. 6): for a successful trajectory,
    clip((mean cost of the group's successes - c_i) / that mean, -1, 1);
    0 for a failed one, and 0 for everyone when fewer than two succeeded
    (one success has nothing to be cheaper than)."""
    wins = [c for r, c in zip(rewards, costs) if r >= 1.0]
    if len(wins) < 2:
        return [0.0] * len(rewards)
    bar = statistics.fmean(wins)
    if bar <= 0:
        return [0.0] * len(rewards)
    return [
        max(-1.0, min(1.0, (bar - c) / bar)) if r >= 1.0 else 0.0 for r, c in zip(rewards, costs)
    ]


def credit(
    arm: str, rewards: list[float], costs: list[float], w_eff: float = W_EFF
) -> list[list[float]]:
    """The advantage of every step of every trajectory: [trajectory][step].
    The outcome advantage is broadcast to all steps; the recipe adds
    w_eff x Eq. 6 on the context edits only (the paper's mask m_i[t]
    selects edit turns), never on the answer step."""
    a = outcome_advantage(rewards)
    e = efficiency_advantage(rewards, costs) if arm == "recipe" else [0.0] * len(rewards)
    return [[x + w_eff * y] * CHUNKS + [x] for x, y in zip(a, e)]


def graded_rows(episodes: list[dict]) -> list[dict]:
    """Eval rows: one per trajectory, the answer step, grouped by task. Each
    row carries the trajectory's cost and its last context.md size."""
    rows: list[dict] = []
    seen: dict[str, int] = {}
    for ep in episodes:
        task = ep["task"]
        sid = task["scenario_id"]
        rows.append(
            {
                "prompt": task["question"],
                "final_text": ep["answer"],
                "reward": outcome(ep["answer"], task["gold"]),
                "scenario_id": sid,
                "rollout_index": seen.get(sid, 0),
                "privileged": {"reference": task["gold"]},
                "cost_tokens": ep["cost"],
                "file_chars": len(ep["files"][-1]) if ep["files"] else 0,
            }
        )
        seen[sid] = seen.get(sid, 0) + 1
    return rows


def file_stats(episodes: list[dict]) -> dict:
    """What the model kept. cost: tokens per trajectory; file_chars: the
    last context.md; keeps_log: share of files that still hold every
    `set` line seen so far (the log copied forward rather than folded into
    the current state); has_gold: share whose last file holds the answer."""
    if not episodes:
        return {"cost": 0, "file_chars": 0, "keeps_log": 0.0, "has_gold": 0.0}
    copied = gold = 0
    for ep in episodes:
        last = ep["files"][-1] if ep["files"] else ""
        seen = [ln for ch in ep["task"]["chunks"] for ln in ch]
        copied += all(ln in last for ln in seen)
        gold += ep["task"]["gold"] in last
    n = len(episodes)
    return {
        "cost": round(statistics.fmean(ep["cost"] for ep in episodes)),
        "file_chars": round(statistics.fmean(len(ep["files"][-1]) for ep in episodes)),
        "keeps_log": round(copied / n, 3),
        "has_gold": round(gold / n, 3),
    }


def make_reward(recorder: list[dict]):
    """TRL wants a reward function per generation call. The trainer pays
    every step itself (credit); this one only records the answer steps so
    `hack_scan` can read the last batch."""

    def reward(completions, prompts, gold, **kwargs) -> list[float]:
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        out = []
        for i, (p, text, g) in enumerate(zip(prompts, texts, gold)):
            r = outcome(text, g)
            out.append(r)
            if "Question:" in json.dumps(p, default=str):
                recorder.append(
                    {
                        "prompt": json.dumps(p, default=str),
                        "final_text": text,
                        "reward": r,
                        "scenario_id": json.dumps(p, default=str),
                        "rollout_index": i,
                    }
                )
        del recorder[: -4 * GROUP * 8]  # the last few batches are enough to scan
        return out

    reward.__name__ = "kv_outcome"
    return reward


ARMS = ("baseline", "recipe")


def clm_trainer(base_cls):
    """GRPOTrainer whose generation step plays the context-file episode.

    TRL hands `_generate_and_score_completions` a batch of tasks, each
    repeated GROUP times in a row. `_play` calls the parent once per step,
    so generation is TRL's own, rebuilds every prompt from the file the
    model just wrote, and returns each trajectory's rows (one per step),
    reward and cost. `credit` decides every step's advantage and `_join`
    hands back one batch. On-policy and without KL is a requirement: the
    rebuilt batch carries no old or reference log-probs.
    """
    import torch

    class CLMTrainer(base_cls):  # type: ignore[valid-type,misc]
        def __init__(self, *args, arm: str, w_eff: float, recorder: list[dict], **kw):
            super().__init__(*args, **kw)
            if arm not in ARMS:
                raise ValueError(f"arm must be one of {ARMS}")
            self.arm, self.w_eff, self.recorder = arm, w_eff, recorder
            if self.num_iterations != 1 or self.beta != 0.0:
                raise RuntimeError("run with num_iterations=1 and beta=0")
            if self.num_generations != GROUP:
                raise RuntimeError(f"num_generations must be GROUP ({GROUP})")

        def _gen_step(self, batch: list[dict], max_new: int) -> dict:
            # TRL 0.19.1 generates with self.generation_config (HF generate)
            # and clips with self.max_completion_length; both carry the cap.
            keep = self.max_completion_length, self.generation_config.max_new_tokens
            self.max_completion_length = self.generation_config.max_new_tokens = max_new
            try:
                return super()._generate_and_score_completions(batch)
            finally:
                self.max_completion_length, self.generation_config.max_new_tokens = keep

        def _play(self, inputs: list[dict]) -> list[dict]:
            tok = self.processing_class
            eps = [
                {"task": x, "files": [], "rows": [], "p_ids": [], "c_ids": [], "answer": ""}
                for x in inputs
            ]
            for t in range(STEPS_PER_TASK):
                last = t == CHUNKS
                batch = [
                    {
                        **inputs[i],
                        "prompt": step_messages(
                            inputs[i], t, eps[i]["files"][-1] if eps[i]["files"] else ""
                        ),
                    }
                    for i in range(len(inputs))
                ]
                out = self._gen_step(batch, ANSWER_TOKENS if last else FILE_TOKENS)
                texts = tok.batch_decode(out["completion_ids"], skip_special_tokens=True)
                for j, (ep, text) in enumerate(zip(eps, texts)):
                    p = out["prompt_ids"][j][out["prompt_mask"][j].bool()]
                    c, m = out["completion_ids"][j], out["completion_mask"][j]
                    ep["rows"].append((p, c, m))
                    ep["p_ids"].append(p.tolist())
                    ep["c_ids"].append(c[m.bool()].tolist())
                    if last:
                        ep["answer"] = text
                    else:
                        ep["files"].append(as_file(text))
            for ep in eps:
                ep["reward"] = outcome(ep["answer"], ep["task"]["gold"])
                ep["cost"] = episode_cost(ep["p_ids"], ep["c_ids"])
            return eps

        def _generate_and_score_completions(self, inputs):
            eps = self._play(inputs)
            rows, advantages = [], []
            effs, solved_groups = [], 0
            for g in range(0, len(eps), GROUP):
                group = eps[g : g + GROUP]
                rewards = [e["reward"] for e in group]
                costs = [float(e["cost"]) for e in group]
                adv = credit(self.arm, rewards, costs, self.w_eff)
                effs += efficiency_advantage(rewards, costs)
                solved_groups += sum(rewards) >= 2
                for e, per_step in zip(group, adv):
                    for row, a in zip(e["rows"], per_step):
                        rows.append(row)
                        advantages.append(a)
            m = self._metrics["train"]
            for key, value in file_stats(eps).items():
                m[f"context/{key}"].append(float(value))
            m["context/reward"].append(statistics.fmean(e["reward"] for e in eps))
            m["credit/groups_with_2_wins"].append(solved_groups / max(len(eps) // GROUP, 1))
            m["credit/eff_abs"].append(statistics.fmean(abs(x) for x in effs))
            m["credit/adv_abs"].append(statistics.fmean(abs(a) for a in advantages))
            return self._join(rows, advantages)

        def _join(self, rows: list[tuple], advantages: list[float]) -> dict:
            """One batch: prompts left-padded, completions right-padded."""
            pad = self.processing_class.pad_token_id
            device = self.accelerator.device
            p_len = max(int(p.numel()) for p, _, _ in rows)
            c_len = max(int(c.numel()) for _, c, _ in rows)
            n = len(rows)
            prompt_ids = torch.full((n, p_len), pad, dtype=torch.long, device=device)
            prompt_mask = torch.zeros((n, p_len), dtype=torch.long, device=device)
            completion_ids = torch.full((n, c_len), pad, dtype=torch.long, device=device)
            completion_mask = torch.zeros((n, c_len), dtype=torch.long, device=device)
            for i, (p, c, m) in enumerate(rows):
                prompt_ids[i, p_len - p.numel() :] = p
                prompt_mask[i, p_len - p.numel() :] = 1
                completion_ids[i, : c.numel()] = c
                completion_mask[i, : m.numel()] = m
            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
                "old_per_token_logps": None,
                "ref_per_token_logps": None,
            }

    return CLMTrainer


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/papers/talk-methods.
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
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
)

runs_volume = modal.Volume.from_name("whileai-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _generate(model, tokenizer, message_lists, *, max_new_tokens, batch=64):
    """One reply per message list, batched, sampled the way the trainer
    samples. Returns the texts and each reply's prompt and reply token counts."""
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    texts_out: list[str] = []
    ids: list[tuple[list[int], list[int]]] = []
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in message_lists
    ]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        texts_out += tokenizer.batch_decode(new, skip_special_tokens=True)
        for j in range(new.shape[0]):
            reply = new[j]
            # Tokens up to and including the first end-of-turn / pad.
            stop = (reply == pad) | (reply == tokenizer.eos_token_id)
            n_reply = int(stop.nonzero()[0]) + 1 if stop.any() else int(reply.numel())
            prompt = enc["input_ids"][j][enc["attention_mask"][j].bool()]
            ids.append((prompt.tolist(), reply[:n_reply].tolist()))
    model.train()
    return texts_out, ids


def _episodes(model, tokenizer, holdout, *, per_task):
    """Play every held-out task `per_task` times, the way training plays it."""
    sys.path.insert(0, "/root")
    from recipe_mod import (
        ANSWER_TOKENS,
        CHUNKS,
        FILE_TOKENS,
        STEPS_PER_TASK,
        as_file,
        episode_cost,
        step_messages,
    )

    tasks = [t for t in holdout for _ in range(per_task)]
    eps = [{"task": x, "files": [], "p_ids": [], "c_ids": [], "answer": ""} for x in tasks]
    for t in range(STEPS_PER_TASK):
        last = t == CHUNKS
        msgs = [step_messages(e["task"], t, e["files"][-1] if e["files"] else "") for e in eps]
        texts, ids = _generate(
            model, tokenizer, msgs, max_new_tokens=ANSWER_TOKENS if last else FILE_TOKENS
        )
        for e, text, (p, c) in zip(eps, texts, ids):
            e["p_ids"].append(p)
            e["c_ids"].append(c)
            if last:
                e["answer"] = text
            else:
                e["files"].append(as_file(text))
    for e in eps:
        e["cost"] = episode_cost(e["p_ids"], e["c_ids"])
        del e["p_ids"], e["c_ids"]
    return eps


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
    w_eff: float = W_EFF,
    per_task: int = 4,
    train_seed: int = 17,
    eval_only: bool = False,
) -> dict:
    """One arm: train every step of the episode on its trajectory's credit,
    evaluate after. `eval_only` evaluates the untrained base EVAL_RUNS times
    (the noise floor) and stops."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        ANSWER_TOKENS,
        CHUNKS,
        EVAL_RUNS,
        FILE_TOKENS,
        GROUP,
        UPDATES_PER_CHUNK,
        clm_trainer,
        file_stats,
        graded_rows,
        make_reward,
        step_messages,
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

    def play(m):
        eps = _episodes(m, tokenizer, holdout, per_task=per_task)
        samples = [
            {"task": e["task"]["scenario_id"], "files": e["files"], "answer": e["answer"]}
            for e in eps[:4]
        ]
        return graded_rows(eps), file_stats(eps), samples

    if eval_only:
        base_runs, base_stats, samples = [], {}, []
        for i in range(EVAL_RUNS):
            rows, stats, s = play(model)
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
        "base_model": base_model,
        "steps": steps,
        "chunks": CHUNKS,
        "updates_per_chunk": UPDATES_PER_CHUNK,
        "group": GROUP,
        "file_tokens": FILE_TOKENS,
        "answer_tokens": ANSWER_TOKENS,
        "tasks_per_step": tasks_per_step,
        "learning_rate": learning_rate,
        "beta": 0.0,
        "lora_rank": lora_rank,
        "w_eff": w_eff if arm == "recipe" else 0.0,
        "train_tasks": len(train_tasks),
        "holdout": len(holdout),
        "train_seed": train_seed,
        "gpu": DEFAULT_GPU,
        "reward": "outcome (boxed final value) on every step; recipe adds w_eff x Eq. 6",
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

    dataset = Dataset.from_list([{**t, "prompt": step_messages(t, 0, "")} for t in train_tasks])
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
        max_completion_length=FILE_TOKENS,
        # System, the file (up to FILE_TOKENS) and a chunk.
        max_prompt_length=1024,
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
    set_seed(train_seed)
    trainer = clm_trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[make_reward(last_batch)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        arm=arm,
        w_eff=w_eff,
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
    history = trainer.state.log_history
    keys = (
        "context/reward",
        "context/cost",
        "context/file_chars",
        "context/keeps_log",
        "context/has_gold",
        "credit/groups_with_2_wins",
        "credit/eff_abs",
        "credit/adv_abs",
    )
    trace = {key: [h[key] for h in history if key in h] for key in keys}

    after_rows, after_stats, samples = play(trainer.model)
    print(f"{arm}: {wai.pass_at(after_rows)} {after_stats}")
    scan = wai.hack_scan(last_batch) if last_batch else {}
    hack_top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
        "pass_at_1": wai.pass_at(after_rows).pass_at_1,
        "cost_tokens": after_stats["cost"],
        "gpu_minutes": gpu_minutes,
        "steps": steps,
    }
    run_url = ""
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        run_url = run.url
    return {
        "arm": arm,
        "after_rows": after_rows,
        "after_stats": after_stats,
        "samples": samples,
        "trace": trace,
        "hack_scan_top": hack_top.get("name", "") if isinstance(hack_top, dict) else str(hack_top),
        "gpu_minutes": gpu_minutes,
        "train_minutes": train_minutes,
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
    """Mean per-task change in tokens per trajectory, after minus before,
    with a 95% t interval over tasks (the tasks are the paired unit)."""
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


def selftest() -> None:
    """The task, the file, the cost and the credit, on the CPU."""
    t = make_task(7, "train")
    assert t == make_task(7, "train")  # seeded
    assert len(t["chunks"]) == CHUNKS and all(len(c) == UPDATES_PER_CHUNK for c in t["chunks"])
    lines = [ln for c in t["chunks"] for ln in c]
    asked = [ln for ln in lines if ln.startswith(f"set {t['ask']} ")]
    assert asked[-1] == f"set {t['ask']} = {t['gold']}"  # the gold is the last set
    assert any(ln.startswith(f"set {t['ask']} ") for ln in t["chunks"][0])
    assert sum(ln.startswith(f"set {t['ask']} ") for ln in lines) >= 2
    tr, ho = make_tasks(50, 0, "train"), make_tasks(50, 0, "test")
    assert not {x["scenario_id"] for x in tr} & {x["scenario_id"] for x in ho}

    first = step_messages(t, 0, "")[1]["content"]
    assert "(empty)" in first and t["chunks"][0][0] in first and "Question" not in first
    q = step_messages(t, CHUNKS, "river = 1")[1]["content"]
    assert "river = 1" in q and "Question" in q and "chunk" not in q.lower().split("question")[1]
    assert as_file("```\na = 1\n```") == "a = 1" and as_file("a = 1") == "a = 1"
    assert outcome("it is \\boxed{418}", "418") == 1.0 and outcome("418", "418") == 0.0

    # Step 2 reuses the cached prefix [1, 2, 3, 9]: pays 2 prompt + 1 reply.
    assert episode_cost([[1, 2, 3], [1, 2, 3, 9, 5, 6]], [[9], [7]]) == 3 + 1 + 2 + 1
    assert episode_cost([[1, 2], [3, 4]], [[5], [6]]) == 6  # a rewritten head reuses nothing
    # Stepwise GRPO, both arms.
    assert outcome_advantage([1.0, 1.0, 0.0, 0.0]) == [0.5, 0.5, -0.5, -0.5]
    # THE ONE CHANGE: only successes get efficiency credit, relative to
    # their own mean cost, clipped to [-1, 1].
    e = efficiency_advantage([1.0, 1.0, 0.0, 0.0], [300.0, 500.0, 100.0, 900.0])
    assert e == [0.25, -0.25, 0.0, 0.0], e
    assert efficiency_advantage([1.0, 0.0], [100.0, 50.0]) == [0.0, 0.0]  # one win
    assert efficiency_advantage([1.0] * 3, [10.0, 10.0, 1000.0])[2] == -1.0  # clipped
    base = credit("baseline", [1.0, 1.0, 0.0, 0.0], [300, 500, 100, 900])
    assert base == [[0.5] * STEPS_PER_TASK] * 2 + [[-0.5] * STEPS_PER_TASK] * 2
    rec = credit("recipe", [1.0, 1.0, 0.0, 0.0], [300, 500, 100, 900], 0.5)
    # Edit steps carry Eq. 6; the answer step carries the outcome alone.
    assert rec[0] == [0.625] * CHUNKS + [0.5] and rec[1] == [0.375] * CHUNKS + [0.5], rec
    assert rec[2] == [-0.5] * STEPS_PER_TASK
    # A dear success still outranks every failure at w_eff <= 1.
    worst = credit("recipe", [1.0, 1.0, 0.0], [1.0, 1000.0, 1.0], 1.0)
    assert min(worst[1]) > max(worst[2])

    ep = {"task": t, "files": ["\n".join(lines)], "answer": f"\\boxed{{{t['gold']}}}", "cost": 500}
    s = file_stats([ep])
    assert s["keeps_log"] == 1.0 and s["has_gold"] == 1.0, s
    rows = graded_rows([ep, {**ep, "answer": "no"}])
    assert [r["rollout_index"] for r in rows] == [0, 1] and [r["reward"] for r in rows] == [
        1.0,
        0.0,
    ]
    pc = paired_cost(
        [{"scenario_id": "a", "cost_tokens": 100}, {"scenario_id": "b", "cost_tokens": 200}],
        [{"scenario_id": "a", "cost_tokens": 80}, {"scenario_id": "b", "cost_tokens": 170}],
    )
    assert pc["delta"] == -25.0 and pc["n"] == 2, pc
    print("task: seeded, asked key set twice, train and holdout disjoint")
    print(
        "credit: baseline r - mean; recipe adds W_EFF x clip((mean win cost - c) / mean, -1, 1) on wins"
    )
    print("selftest ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0, help="task seed")
    ap.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=[17, 18],
        help="training seeds per arm; two or more lets the verdict resolve",
    )
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=200)
    ap.add_argument("--w-eff", type=float, default=W_EFF, help="weight on Eq. 6 in the recipe arm")
    ap.add_argument("--smoke", action="store_true", help="2 steps, 16 held-out tasks, one seed")
    ap.add_argument("--reuse", action="store_true", help="skip calls cached in .cache/")
    ap.add_argument("--selftest", action="store_true", help="the pure functions, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    if args.smoke:
        args.steps, args.n_holdout, args.train_seeds = 2, 16, args.train_seeds[:1]
    train_tasks = make_tasks(args.n_train, args.seed, "train")
    holdout = make_tasks(args.n_holdout, args.seed, "test")
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    print(f"{len(train_tasks)} train tasks, {len(holdout)} held out")
    arms = list(ARMS) if args.arm == "all" else [args.arm]

    tag = "smoke-" if args.smoke else ""
    jobs = [(arm, s) for s in args.train_seeds for arm in arms]
    names = {(arm, s): f"context-lm-{tag}{arm}-s{s}" for arm, s in jobs}
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)

    def cached(name: str) -> bool:
        return args.reuse and (cache / f"{name}.json").exists()

    base_name = f"context-lm-{tag}base"
    with modal.enable_output(), app.run():
        calls = {}
        if not cached(base_name):
            calls[base_name] = run_arm.spawn("baseline", [], holdout, base_name, eval_only=True)
        for (arm, s), name in names.items():
            if not cached(name):
                calls[name] = run_arm.spawn(
                    arm,
                    train_tasks,
                    holdout,
                    name,
                    steps=args.steps,
                    train_seed=s,
                    w_eff=args.w_eff,
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
    outs = {key: json.loads((cache / f"{name}.json").read_text()) for key, name in names.items()}
    gpu_minutes = base["gpu_minutes"] + sum(o["gpu_minutes"] for o in outs.values())
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 4.00)
    usd = gpu_minutes / 60.0 * usd_per_hour
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${usd:.2f} on {DEFAULT_GPU}")

    if args.smoke:
        smoke = {
            "base": summarize(base["base_runs"][0], base["base_stats"]),
            **{
                name: {
                    **summarize(outs[key]["after_rows"], outs[key]["after_stats"]),
                    "trace": outs[key]["trace"],
                }
                for key, name in names.items()
            },
            "gpu_minutes": round(gpu_minutes, 1),
            "usd": round(usd, 2),
            "date": date.today().isoformat(),
        }
        (HERE / ".cache" / "smoke.json").write_text(json.dumps(smoke, indent=2))
        print(json.dumps({k: v for k, v in smoke.items()}, indent=2, default=str)[:4000])
        print("wrote .cache/smoke.json; a smoke run claims no number and writes no results.json")
        return

    noise = wai.eval_variance(*base["base_runs"])
    run_std = float(noise["run_std"])
    after = {arm: [outs[(arm, s)]["after_rows"] for s in args.train_seeds] for arm in arms}
    results = {
        "recipe": HERE.name,
        "title": "Context LM: pay for a short context file once the answer is right",
        "paper": PAPER,
        "book": BOOK,
        "base_model": BASE_MODEL,
        "metric": METRIC,
        "n_holdout": len(holdout),
        "k": 4,
        "w_eff": args.w_eff,
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
            "train_seeds": {arm: len(after[arm]) for arm in arms},
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
    for arm in arms:
        seeds = [outs[(arm, s)] for s in args.train_seeds]
        pooled = [r for rows in after[arm] for r in rows]
        results["arms"][arm] = {
            **summarize(after[arm][0], seeds[0]["after_stats"]),
            "per_seed": [summarize(rows)["score"] for rows in after[arm]],
            "cost_per_seed": [round(mean_cost(rows)) for rows in after[arm]],
            "context_per_seed": [o["after_stats"] for o in seeds],
            "pooled_score": summarize(pooled)["score"],
            "pooled_cost_tokens": round(mean_cost(pooled)),
            "steps": args.steps,
            "gpu_minutes": round(statistics.fmean(o["gpu_minutes"] for o in seeds), 1),
        }
        results["train_trace"][arm] = [o["trace"] for o in seeds]
        results["samples"][arm] = seeds[0]["samples"]
        results["checks"]["length_after"][arm] = mean_length(after[arm][0])
        results["checks"]["hack_scan_top"] = seeds[0]["hack_scan_top"]
        results["run_url"] = seeds[0]["run_url"] or results["run_url"]

    if set(arms) == set(ARMS):
        d = wai.compare(
            after["baseline"][0],
            after["recipe"][0],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(noise["n_runs"]),
            train_runs={"before": after["baseline"], "after": after["recipe"]},
            proxy=PROXY,
        )
        print(d)
        verdict = d["target_verdict"]
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": verdict if verdict in ("moved", "flat", "unresolved") else "flat",
            "noise_band": d.get("noise_band"),
        }
        results["checks"]["over_optimized"] = bool(d.get("over_optimized"))
        # The paper's second number: tokens per trajectory, seed by seed
        # (seed s of the recipe against seed s of the baseline), then pooled.
        results["cost_delta"] = {
            "per_seed": [paired_cost(b, r) for b, r in zip(after["baseline"], after["recipe"])],
            "pooled": paired_cost(
                [r for rows in after["baseline"] for r in rows],
                [r for rows in after["recipe"] for r in rows],
            ),
        }
        print(
            json.dumps({"delta": results["delta"], "cost_delta": results["cost_delta"]}, indent=2)
        )
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"

    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("wrote results.json")


if __name__ == "__main__":
    main()
