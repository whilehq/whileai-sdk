"""Board writers: pay a note by what it did for the copies that read it.

    python recipe.py                      # both arms on Modal, three seeds, writes results.json
    python recipe.py --reuse              # rerun only the containers that failed
    python recipe.py --selftest           # the ring, the credit and the counters, offline

The message-board recipe trained copies of a model to read a board of notes
and left the notes untrained. Readers learned to use the board, but the
notes stayed long, got cut off before their answer, and a reader that only
reread its own note did about as well. MAPoRL (Park et al. 2025) trains
every turn of the discussion on the team's outcome, not only the last one.

So train the writers too. Four copies of the model post a note each. Every
reader sees two notes on a ring: its own and its neighbour's, so every note
is read by exactly two readers. A reader is paid for its own answer. A note
is paid the mean of its two readers' answers: a note that helps the copies
that read it gets the credit, whether or not its own answer was right.

THE ONE CHANGE is the writers' advantage. Both arms draft, read and train
on the same batch: four notes and four readers per problem, the same ring,
the same readers' reward. The baseline puts the notes in the batch at zero
advantage (so the loss is normalized over the same tokens); the recipe gives
each note its readers' mean reward minus the mean over the four notes.

Shape of the run:
  1. data():      MATH train levels 3 to 5 for prompts, MATH-500 levels 3 to 5 held out
  2. run_arm():   TRL GRPOTrainer + LoRA on Modal; the generation step drafts
                  the notes, builds the ring, samples the readers, credits both
  3. evaluate():  same holdout, same two rounds, graded by MathEqual
  4. results.json + the paired delta (wai.compare) on the run page
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import date
from importlib.metadata import version
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"
BOOK = "Reasoning"  # RL on verifiable rewards, credit for a turn another rollout reads
# The training reward here *is* the target: both are the same binary check
# against the MATH gold, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

COPIES = 4  # notes on the board, and readers per problem (the GRPO group)
NOTE_TOKENS = 256  # the message-board recipe's budget; notes that run long get cut
LEVELS = ("Level 3", "Level 4", "Level 5")

NOTE_SYSTEM = (
    "You are one of four teammates solving the same problem. Post a short note to the "
    "team board: your key steps in at most three lines, then your answer as \\boxed{answer}. "
    "Two teammates will read it."
)
READ_SYSTEM = (
    "You are one of four teammates solving the same problem. Below the problem are two "
    "notes from the team board. Read them, check them, and give the final answer as "
    "\\boxed{answer}."
)


# --------------------------------------------------------------------------
# Pure functions, no torch: `--selftest` runs them, and the Modal container
# imports this same file.
# --------------------------------------------------------------------------


def note_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": NOTE_SYSTEM}, {"role": "user", "content": question}]


def read_messages(question: str, notes: list[str]) -> list[dict]:
    board = "\n\n".join(f"Note {i + 1}: {n.strip()}" for i, n in enumerate(notes))
    body = f"{question}\n\nTeam board:\n{board}"
    return [{"role": "system", "content": READ_SYSTEM}, {"role": "user", "content": body}]


def ring(notes: list[str]) -> list[list[str]]:
    """Reader j sees note j and note j + 1 (mod COPIES), so every note is
    read by exactly two readers: its writer's reader and the one before."""
    n = len(notes)
    return [[notes[j], notes[(j + 1) % n]] for j in range(n)]


def readers_of(i: int, n: int = COPIES) -> tuple[int, int]:
    """The two readers who see note i on the ring."""
    return i, (i - 1) % n


def note_rewards(reader_rewards: list[float]) -> list[float]:
    """THE CREDIT. A note earns the mean reward of the two readers who read
    it; nothing about the note's own answer enters."""
    n = len(reader_rewards)
    return [statistics.fmean(reader_rewards[r] for r in readers_of(i, n)) for i in range(n)]


def centered(rewards: list[float]) -> list[float]:
    """Reward minus the group mean, no std division (both arms run
    `scale_rewards=False`)."""
    mean = statistics.fmean(rewards)
    return [r - mean for r in rewards]


def last_boxed(text: str) -> str | None:
    """The contents of the last \\boxed{...}, braces matched, so
    \\boxed{\\frac{1}{2}} reads whole."""
    text = text or ""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    out = []
    while i < len(text):
        c = text[i]
        depth += c == "{"
        depth -= c == "}"
        if depth == 0:
            return "".join(out).strip()
        out.append(c)
        i += 1
    return None  # cut off before the box closed


def answer_of(text: str) -> str | None:
    """The last boxed answer with spaces, commas and a trailing .0 removed,
    for comparing two replies. Correctness is MathEqual's."""
    a = last_boxed(text)
    if a is None:
        return None
    a = a.replace(" ", "").replace(",", "").replace("$", "")
    return a[:-2] if a.endswith(".0") else a


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


def majority(notes: list[str]) -> str:
    """The board's most common boxed answer, first-posted on a tie."""
    answers = [a for a in (answer_of(n) for n in notes) if a is not None]
    if not answers:
        return ""
    counts = Counter(answers)
    top = max(counts.values())
    return f"\\boxed{{{next(a for a in answers if counts[a] == top)}}}"


def board_stats(groups: list[dict]) -> dict:
    """What the board carried and what the readers did with it. One group per
    problem: `notes`, `replies` (reader j saw notes j and j + 1), `gold`.

    note_acc:   notes whose own answer is right
    note_boxed: notes that close a \\boxed{} (not cut off, not answerless)
    note_chars: mean note length
    vote_acc:   the board's most common answer, no reading
    rescued:    reader's own note wrong, reader right
    misled:     reader's own note right, reader wrong
    """
    note_ok = boxed = chars = vote = 0.0
    rescued = misled = n = 0
    for g in groups:
        notes, replies, gold = g["notes"], g["replies"], g["gold"]
        oks = [outcome_of(x, gold) for x in notes]
        note_ok += statistics.fmean(oks)
        boxed += statistics.fmean(last_boxed(x) is not None for x in notes)
        chars += statistics.fmean(len(x) for x in notes)
        vote += outcome_of(majority(notes), gold)
        for j, reply in enumerate(replies):
            final = outcome_of(reply, gold)
            n += 1
            rescued += oks[j] == 0.0 and final == 1.0
            misled += oks[j] == 1.0 and final == 0.0
    k = max(len(groups), 1)
    return {
        "note_acc": round(note_ok / k, 3),
        "note_boxed": round(boxed / k, 3),
        "note_chars": round(chars / k),
        "vote_acc": round(vote / k, 3),
        "rescued": round(rescued / max(n, 1), 3),
        "misled": round(misled / max(n, 1), 3),
    }


def make_reward(recorder: list[dict]):
    """Binary outcome against the MATH gold. The trainer reads the rewards
    back from `recorder` after each generation call; `hack_scan` reads the
    last batch after training (Lambert 2025, chapter Over-optimization)."""

    def reward(completions, prompts, gold, **kwargs) -> list[float]:
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        keys = [json.dumps(p, sort_keys=True, default=str) for p in prompts]
        rewards = [outcome_of(t, g) for t, g in zip(texts, gold)]
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

    reward.__name__ = "math_outcome"
    return reward


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `compare` read."""
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


def writers_trainer(base_cls):
    """GRPOTrainer whose generation step is the two rounds.

    TRL hands `_generate_and_score_completions` a batch of problems, each
    repeated `num_generations` (= COPIES) times in a row. The override calls
    the parent twice: once on the note prompts (the four notes), once on the
    ring's reader prompts (the four readers). Generation and grading are
    TRL's own. It then writes the advantages itself, readers from their own
    reward and notes from their readers' (or zero, in the baseline), and
    hands back one batch holding both, re-padded. On-policy and without KL
    is a requirement: the rebuilt batch carries no old or reference log-probs.
    """
    import torch

    class WritersTrainer(base_cls):  # type: ignore[valid-type,misc]
        def __init__(self, *args, writers: bool, recorder: list[dict], **kw):
            super().__init__(*args, **kw)
            self.writers = writers
            self.recorder = recorder
            if self.num_iterations != 1 or self.beta != 0.0:
                raise RuntimeError("run with num_iterations=1 and beta=0")
            if self.num_generations != COPIES:
                raise RuntimeError(f"num_generations must be COPIES ({COPIES})")

        def _round(self, batch: list[dict], max_new: int) -> tuple[dict, list[float]]:
            # TRL 0.19.1 generates with self.generation_config (HF generate)
            # and clips with self.max_completion_length; both carry the cap.
            keep = self.max_completion_length, self.generation_config.max_new_tokens
            self.max_completion_length = self.generation_config.max_new_tokens = max_new
            try:
                out = super()._generate_and_score_completions(batch)
            finally:
                self.max_completion_length, self.generation_config.max_new_tokens = keep
            rewards = [row["reward"] for row in self.recorder]
            if len(rewards) != len(batch):
                raise RuntimeError("reward recorder is out of step with the batch")
            return out, rewards

        def _generate_and_score_completions(self, inputs):
            tok = self.processing_class
            unique = inputs[::COPIES]
            note_batch = [
                {**x, "prompt": note_messages(x["question"])} for x in unique for _ in range(COPIES)
            ]
            notes_out, note_own = self._round(note_batch, NOTE_TOKENS)
            note_texts = tok.batch_decode(notes_out["completion_ids"], skip_special_tokens=True)

            reader_batch, groups = [], []
            for p, x in enumerate(unique):
                notes = note_texts[p * COPIES : (p + 1) * COPIES]
                groups.append({"notes": notes, "gold": x["gold"]})
                reader_batch += [
                    {**x, "prompt": read_messages(x["question"], board)} for board in ring(notes)
                ]
            readers_out, reader_r = self._round(reader_batch, self.max_completion_length)
            reader_texts = tok.batch_decode(readers_out["completion_ids"], skip_special_tokens=True)

            reader_adv, note_adv, credit = [], [], []
            for p, g in enumerate(groups):
                rr = reader_r[p * COPIES : (p + 1) * COPIES]
                g["replies"] = reader_texts[p * COPIES : (p + 1) * COPIES]
                reader_adv += centered(rr)
                nr = note_rewards(rr)
                credit += nr
                note_adv += centered(nr) if self.writers else [0.0] * COPIES

            metrics = self._metrics["train"]
            for key, value in board_stats(groups).items():
                metrics[f"board/{key}"].append(float(value))
            metrics["board/note_credit"].append(statistics.fmean(credit))
            metrics["board/note_own_reward"].append(statistics.fmean(note_own))
            metrics["board/note_adv_abs"].append(statistics.fmean(abs(a) for a in note_adv))
            return self._join(notes_out, readers_out, note_adv + reader_adv)

        def _join(self, a: dict, b: dict, advantages: list[float]) -> dict:
            """One batch from two: prompts left-padded, completions right-padded."""
            pad = self.processing_class.pad_token_id
            device = self.accelerator.device

            def cat(key: str, left: bool, fill: int) -> torch.Tensor:
                x, y = a[key], b[key]
                width = max(x.shape[1], y.shape[1])
                out = torch.full(
                    (x.shape[0] + y.shape[0], width), fill, dtype=x.dtype, device=device
                )
                for row, t in enumerate(list(x) + list(y)):
                    if left:
                        out[row, width - t.numel() :] = t
                    else:
                        out[row, : t.numel()] = t
                return out

            return {
                "prompt_ids": cat("prompt_ids", True, pad),
                "prompt_mask": cat("prompt_mask", True, 0),
                "completion_ids": cat("completion_ids", False, pad),
                "completion_mask": cat("completion_mask", False, 0),
                "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
                "old_per_token_logps": None,
                "ref_per_token_logps": None,
            }

    return WritersTrainer


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/papers/message-board.
# --------------------------------------------------------------------------

# H100, not the L40S of the earlier board recipes: notes and readers train in
# one batch on MATH-length prompts, and the smoke run ran out of memory at 48 GB.
DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "H100")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-board-writers")

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


def _generate(model, tokenizer, message_lists, *, n, max_new_tokens, batch=16):
    """`n` replies per message list, batched, sampled the way the trainer samples."""
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    out: list[list[str]] = []
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in message_lists
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


def _board_eval(model, tokenizer, holdout, *, max_new_tokens):
    """Round one: COPIES notes per task. Round two: reader j answers from
    notes j and j + 1. Returns graded reader rows and the groups."""
    sys.path.insert(0, "/root")
    from recipe_mod import COPIES, NOTE_TOKENS, graded_rows, note_messages, read_messages, ring

    notes = _generate(
        model,
        tokenizer,
        [note_messages(t["question"]) for t in holdout],
        n=COPIES,
        max_new_tokens=NOTE_TOKENS,
    )
    reader_msgs = [
        read_messages(t["question"], board)
        for t, drafts in zip(holdout, notes)
        for board in ring(drafts)
    ]
    flat = _generate(model, tokenizer, reader_msgs, n=1, max_new_tokens=max_new_tokens)
    replies = [[flat[i * COPIES + j][0] for j in range(COPIES)] for i in range(len(holdout))]
    groups = [
        {"notes": d, "replies": r, "gold": t["gold"]} for t, d, r in zip(holdout, notes, replies)
    ]
    return graded_rows(holdout, replies), groups


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
    writers: bool,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 80,
    prompts_per_step: int = 12,
    learning_rate: float = 1e-4,
    max_completion_length: int = 512,
    lora_rank: int = 32,
    train_seed: int = 17,
    eval_only: bool = False,
) -> dict:
    """One arm: train readers and notes under this arm's credit, eval after.
    `eval_only` evaluates the untrained base EVAL_RUNS times and stops."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        COPIES,
        EVAL_RUNS,
        NOTE_TOKENS,
        board_stats,
        make_reward,
        note_messages,
        writers_trainer,
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

    def evaluate(m):
        return _board_eval(m, tokenizer, holdout, max_new_tokens=max_completion_length)

    if eval_only:
        # The base, EVAL_RUNS times: the spread is the eval's own noise
        # (Lambert 2025, chapter Evaluation).
        base_runs, base_stats = [], {}
        for i in range(EVAL_RUNS):
            rows, groups = evaluate(model)
            base_runs.append(rows)
            if i == 0:
                base_stats = board_stats(groups)
            print(f"base run {i + 1}/{EVAL_RUNS}: {wai.pass_at(rows)} {board_stats(groups)}")
        return {
            "base_runs": base_runs,
            "base_stats": base_stats,
            "gpu_minutes": (time.time() - started) / 60.0,
        }

    config = {
        "arm": arm,
        "train_writers": writers,
        "base_model": base_model,
        "steps": steps,
        "copies": COPIES,
        "note_tokens": NOTE_TOKENS,
        "board": "ring: reader j reads notes j and j+1",
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
        "reward": "reader: own MathEqual; note: mean of its two readers (recipe) or 0 (baseline)",
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

    dataset = Dataset.from_list(
        [
            {"prompt": note_messages(t["question"]), "question": t["question"], "gold": t["gold"]}
            for t in train_tasks
        ]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=COPIES,
        per_device_train_batch_size=COPIES,
        gradient_accumulation_steps=prompts_per_step,
        learning_rate=learning_rate,
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        scale_rewards=False,
        # The readers' cap; the note round runs at NOTE_TOKENS (_round).
        max_completion_length=max_completion_length,
        # The problem plus two notes of up to NOTE_TOKENS.
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
    trainer = writers_trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[make_reward(last_batch)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        writers=writers,
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
        "board/note_acc",
        "board/note_boxed",
        "board/note_chars",
        "board/vote_acc",
        "board/rescued",
        "board/misled",
        "board/note_credit",
        "board/note_adv_abs",
    )
    trace = {key: [h[key] for h in history if key in h] for key in keys}

    after_rows, groups = evaluate(trainer.model)
    after_stats = board_stats(groups)
    print(f"{arm}: {wai.pass_at(after_rows)} {after_stats}")

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
        "train_writers": writers,
        "pass_at_1": wai.pass_at(after_rows).pass_at_1,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    return {
        "arm": arm,
        "after_rows": after_rows,
        "after_stats": after_stats,
        "sample_groups": groups[:3],
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


def summarize(rows: list[dict], stats: dict | None = None) -> dict:
    import whileai as wai

    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
        "chars": round(mean_length(rows)),
        **({"board": stats} if stats else {}),
    }


def selftest() -> None:
    """The ring, the credit and the counters, on the CPU."""
    notes = ["a \\boxed{17}", "b \\boxed{17}", "c \\boxed{7}", "d, cut off before the box"]
    # Every note is read by exactly two readers.
    boards = ring(notes)
    assert boards[0] == [notes[0], notes[1]] and boards[3] == [notes[3], notes[0]]
    assert sorted(n for b in boards for n in b) == sorted(notes * 2)
    for i in range(4):
        assert all(notes[i] in boards[r] for r in readers_of(i))

    # THE CREDIT: a note earns its two readers' mean. Readers 0 and 1 right,
    # 2 and 3 wrong: note 1 (read by 1 and 0) earns 1.0 whatever its own
    # answer, note 3 (read by 3 and 2) earns 0.
    assert note_rewards([1.0, 1.0, 0.0, 0.0]) == [0.5, 1.0, 0.5, 0.0]
    assert centered([0.5, 1.0, 0.5, 0.0]) == [0.0, 0.5, 0.0, -0.5]

    # Braces matched, cut-off boxes are no answer.
    assert last_boxed("so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed("\\boxed{3} then \\boxed{4}") == "4"
    assert last_boxed("\\boxed{\\frac{1}{2") is None
    assert answer_of("\\boxed{1,200.0}") == "1200"
    assert majority(notes) == "\\boxed{17}"

    g = {"notes": notes, "replies": ["\\boxed{17}"] * 3 + ["\\boxed{7}"], "gold": "17"}
    s = board_stats([g])
    assert s["note_acc"] == 0.5 and s["note_boxed"] == 0.75 and s["vote_acc"] == 1.0
    assert s["rescued"] == 0.25 and s["misled"] == 0.0, s

    assert outcome_of("\\boxed{\\frac{1}{2}}", "\\frac12") == 1.0
    assert outcome_of("\\boxed{0.5}", "\\frac{1}{2}") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("ring: every note read by two readers; note credit = mean of its readers")
    print("grader: MathEqual, decided by Math-Verify")
    print("selftest ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0, help="data shuffle seed")
    ap.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=[17, 18, 19],
        help="training seeds per arm; two or more lets the verdict resolve",
    )
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--n-holdout", type=int, default=300)
    ap.add_argument("--reuse", action="store_true", help="skip calls cached in .cache/")
    ap.add_argument("--selftest", action="store_true", help="the ring and the credit, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    print(f"{len(train_tasks)} train problems, {len(holdout)} held out")
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]
    writers = {"baseline": False, "recipe": True}

    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(
        {
            "recipe": HERE.name,
            "title": "Board writers: pay a note by what it did for the copies that read it",
            "paper": "https://arxiv.org/abs/2502.18439",
            "book": BOOK,
            "base_model": BASE_MODEL,
            "metric": METRIC,
            "n_holdout": len(holdout),
            "k": COPIES,
            "gpu": DEFAULT_GPU,
            "whileai": version("whileai"),
        }
    )
    results.setdefault("arms", {})
    checks = results.setdefault("checks", {})
    checks["decontaminated_dropped"] = int(decon.get("n_contaminated", 0))
    checks["seed"] = args.seed
    checks.setdefault("length_after", {})
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)

    jobs = [(arm, s) for s in args.train_seeds for arm in arms]
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)
    today = date.today().isoformat()
    names = [f"board-writers-{arm}-s{s}-{today}" for arm, s in jobs]

    def cached(name: str) -> bool:
        return args.reuse and (cache / f"{name}.json").exists()

    with modal.enable_output(), app.run():
        # The base evals get a container of their own, and every finished
        # call is cached, so one lost container costs only itself.
        calls = {}
        if not cached("base"):
            calls["base"] = run_arm.spawn(
                "base", False, [], holdout, "board-writers-base", eval_only=True
            )
        for (arm, s), name in zip(jobs, names):
            if not cached(name):
                calls[name] = run_arm.spawn(
                    arm, writers[arm], train_tasks, holdout, name, steps=args.steps, train_seed=s
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

    base = json.loads((cache / "base.json").read_text())
    outs = [json.loads((cache / f"{name}.json").read_text()) for name in names]
    gpu_minutes = base["gpu_minutes"] + sum(o["gpu_minutes"] for o in outs)
    noise = wai.eval_variance(*base["base_runs"])
    run_std = float(noise["run_std"])
    results["arms"]["base"] = {
        **summarize(base["base_runs"][0], base["base_stats"]),
        "steps": 0,
        "gpu_minutes": 0,
    }
    checks["run_std"] = run_std
    checks["run_std_runs"] = int(noise["n_runs"])
    checks["length_before"] = mean_length(base["base_runs"][0])

    after: dict[str, list[list[dict]]] = {arm: [] for arm in arms}
    stats: dict[str, list[dict]] = {arm: [] for arm in arms}
    minutes: dict[str, list[float]] = {arm: [] for arm in arms}
    run_url = ""
    for (arm, _), out in zip(jobs, outs):
        after[arm].append(out["after_rows"])
        stats[arm].append(out["after_stats"])
        minutes[arm].append(out["train_minutes"])
        run_url = out["run_url"] or run_url
        results.setdefault("train_trace", {}).setdefault(arm, []).append(out["trace"])
        results.setdefault("samples", {}).setdefault(arm, out["sample_groups"])
        checks["hack_scan_top"] = out["hack_scan_top"]
        checks["length_after"][arm] = mean_length(out["after_rows"])
    for arm in arms:
        results["arms"][arm] = {
            **summarize(after[arm][0], stats[arm][0]),
            "per_seed": [summarize(rows)["score"] for rows in after[arm]],
            "board_per_seed": stats[arm],
            "pooled_score": summarize([r for rows in after[arm] for r in rows])["score"],
            "steps": args.steps,
            "gpu_minutes": round(statistics.fmean(minutes[arm]), 1),
        }

    if "baseline" in after and "recipe" in after:
        d = wai.compare(
            after["baseline"][0],
            after["recipe"][0],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks["run_std_runs"]),
            train_runs={"before": after["baseline"], "after": after["recipe"]},
            proxy=PROXY,
        )
        verdict = d["target_verdict"]
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": verdict if verdict in ("moved", "flat", "unresolved") else "flat",
        }
        checks["train_seeds"] = {arm: len(after[arm]) for arm in after}
        checks["over_optimized"] = bool(d.get("over_optimized"))
        results["verified"] = today
        results.pop("partial_run", None)
        print(results["delta"])
    else:
        results["partial_run"] = f"{today}: {', '.join(arms)} only"

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    skip = ("train_trace", "samples")
    print(json.dumps({k: v for k, v in results.items() if k not in skip}, indent=2))


if __name__ == "__main__":
    main()
