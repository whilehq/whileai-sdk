"""Message board: teach copies of a model to read each other's notes.

    python recipe.py                      # both arms on Modal, two seeds, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the board, the answer reader and the counters, offline

The team-talk recipe put a team inside one reply and trained it with GRPO.
The talk cost accuracy and RL deleted it: talk that nobody reads is not
worth writing. MAPoRL (Park et al. 2025) trains separate copies that answer,
discuss and answer again, and finds that training one model alone does not
produce collaboration; co-training does.

So give the talk a reader. Every problem gets four copies of the model.
Round one, each copy posts a short note to a board: its key steps and its
answer. Round two, each copy reads the board and writes the final answer.
GRPO trains round two on each reader's own correctness.

THE ONE CHANGE is what a reader sees. The baseline reader sees only its own
note (the same two rounds, the same tokens, one model revising itself). The
recipe reader sees all four notes. Both arms draft the same four notes per
problem, so the only difference is whether the copies can read each other.

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out
  2. run_arm():   TRL GRPOTrainer + LoRA on Modal; the generation step drafts
                  the board first, then TRL samples the readers
  3. evaluate():  same holdout; drafts, then readers, graded by MathEqual;
                  majority vote over the board as the no-reading reference
  4. results.json + the paired delta (wai.compare) on the run page
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
BOOK = "Reasoning"  # RL on verifiable rewards, here on a reader of other rollouts
# The training reward here *is* the target: both are the same binary check
# against the GSM8K gold, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

COPIES = 4  # notes on the board, and readers per problem (the GRPO group)
NOTE_TOKENS = (
    256  # a note is short: key steps and an answer; 160 cut most notes before the box (smoke)
)

NOTE_SYSTEM = (
    "You are one of four teammates solving the same problem. Post a short note to the "
    "team board: your key steps in at most three lines, then your answer as \\boxed{answer}."
)
READ_SYSTEM = (
    "You are one of four teammates solving the same problem. Below the problem is the "
    "team board. Read the notes, check them, and give the final number as \\boxed{answer}."
)


# --------------------------------------------------------------------------
# Pure functions, no torch: `--selftest` runs them, and the Modal container
# imports this same file.
# --------------------------------------------------------------------------


def note_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": NOTE_SYSTEM}, {"role": "user", "content": question}]


def board_text(notes: list[str]) -> str:
    return "\n\n".join(f"Note {i + 1}: {n.strip()}" for i, n in enumerate(notes))


def read_messages(question: str, notes: list[str]) -> list[dict]:
    body = f"{question}\n\nTeam board:\n{board_text(notes)}"
    return [{"role": "system", "content": READ_SYSTEM}, {"role": "user", "content": body}]


def boards_for(notes: list[str], shared: bool) -> list[list[str]]:
    """THE ONE CHANGE. Reader j sees every note (shared) or only note j."""
    return [list(notes) if shared else [notes[j]] for j in range(len(notes))]


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


BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def answer_of(text: str) -> str | None:
    """The last boxed answer, normalized the plain way (commas, dollar signs,
    a trailing .0), for comparing two replies. Correctness is MathEqual's."""
    found = BOXED.findall(text or "")
    if not found:
        return None
    a = found[-1].replace(",", "").replace("$", "").replace("\\", "").strip()
    return a[:-2] if a.endswith(".0") else a


def majority(notes: list[str]) -> str:
    """The board's most common boxed answer, first-posted on a tie. The
    reference a reader has to beat: voting needs no reading at all."""
    answers = [a for a in (answer_of(n) for n in notes) if a is not None]
    if not answers:
        return ""
    top = Counter(answers).most_common(1)[0][1]
    pick = next(a for a in answers if Counter(answers)[a] == top)
    return f"\\boxed{{{pick}}}"


def board_stats(groups: list[dict]) -> dict:
    """What the readers did with the board, one group per problem:
    `notes` (the COPIES drafts), `replies` (reader j's final text, reader j
    wrote note j), `gold`.

    switched: reader's answer differs from its own note's
    rescued:  own note wrong, reader right
    misled:   own note right, reader wrong
    note_acc, vote_acc, any_note: the board before anyone reads it
    """
    switched = rescued = misled = n = 0
    note_ok = vote_ok = any_ok = 0.0
    for g in groups:
        notes, replies, gold = g["notes"], g["replies"], g["gold"]
        oks = [outcome_of(x, gold) for x in notes]
        note_ok += statistics.fmean(oks)
        vote_ok += outcome_of(majority(notes), gold)
        any_ok += max(oks)
        for j, reply in enumerate(replies):
            own = oks[j % len(notes)]
            final = outcome_of(reply, gold)
            n += 1
            switched += answer_of(reply) != answer_of(notes[j % len(notes)])
            rescued += own == 0.0 and final == 1.0
            misled += own == 1.0 and final == 0.0
    k = max(len(groups), 1)
    return {
        "note_acc": round(note_ok / k, 3),
        "vote_acc": round(vote_ok / k, 3),
        "any_note": round(any_ok / k, 3),
        "switched": round(switched / max(n, 1), 3),
        "rescued": round(rescued / max(n, 1), 3),
        "misled": round(misled / max(n, 1), 3),
    }


def make_reward(recorder: list[dict]):
    """Binary outcome of the reader's own final answer. Both arms use this
    untouched; nothing pays for agreeing with the board."""

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
    """Eval rows in the shape `pass_at` and `compare` read."""
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


def board_trainer(base_cls):
    """GRPOTrainer whose generation step drafts the board first.

    TRL hands `_generate_and_score_completions` a batch of prompts, each
    repeated `num_generations` (= COPIES) times in a row. The override
    drafts COPIES notes per problem with the current policy (no gradient:
    round one is context, not the action being trained), rewrites every
    reader's prompt to carry its board, and calls the parent, so
    generation of the readers, the reward, the advantage and the loss are
    TRL's own. `shared=False` is the baseline: the class is the same, the
    flag is the difference.
    """
    import torch
    from trl.models import unwrap_model_for_generation

    class BoardTrainer(base_cls):  # type: ignore[valid-type,misc]
        def __init__(self, *args, shared: bool, note_tokens: int = NOTE_TOKENS, **kw):
            super().__init__(*args, **kw)
            self.shared = shared
            self.note_tokens = note_tokens
            if self.num_generations != COPIES:
                raise RuntimeError(f"num_generations must be COPIES ({COPIES})")
            self._last_groups: list[dict] = []

        def _draft(self, questions: list[str]) -> list[list[str]]:
            tok = self.processing_class
            tok.padding_side = "left"
            texts = [
                tok.apply_chat_template(
                    note_messages(q), tokenize=False, add_generation_prompt=True
                )
                for q in questions
            ]
            enc = tok(texts, return_tensors="pt", padding=True).to(self.accelerator.device)
            with (
                unwrap_model_for_generation(self.model_wrapped, self.accelerator) as m,
                torch.no_grad(),
            ):
                gen = m.generate(
                    **enc,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    max_new_tokens=self.note_tokens,
                    num_return_sequences=COPIES,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            decoded = tok.batch_decode(
                gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return [decoded[i * COPIES : (i + 1) * COPIES] for i in range(len(questions))]

        def _generate_and_score_completions(self, inputs):
            unique = inputs[::COPIES]
            notes = self._draft([x["question"] for x in unique])
            rebuilt = []
            for x, drafts in zip(unique, notes):
                for board in boards_for(drafts, self.shared):
                    rebuilt.append({**x, "prompt": read_messages(x["question"], board)})
            out = super()._generate_and_score_completions(rebuilt)

            texts = self.processing_class.batch_decode(
                out["completion_ids"], skip_special_tokens=True
            )
            groups = [
                {
                    "notes": drafts,
                    "replies": texts[i * COPIES : (i + 1) * COPIES],
                    "gold": gold_of(x["answer"]),
                }
                for i, (x, drafts) in enumerate(zip(unique, notes))
            ]
            self._last_groups = groups
            metrics = self._metrics["train"]
            for key, value in board_stats(groups).items():
                metrics[f"board/{key}"].append(float(value))
            return out

    return BoardTrainer


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/papers/team-talk.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-message-board")

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


def _generate(model, tokenizer, message_lists, *, n, max_new_tokens, batch=8):
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


def _board_eval(model, tokenizer, holdout, *, shared, max_new_tokens):
    """Round one: COPIES notes per task. Round two: reader j answers from its
    board. Returns graded reader rows (COPIES per task) and the groups."""
    sys.path.insert(0, "/root")
    from recipe_mod import (
        COPIES,
        NOTE_TOKENS,
        boards_for,
        gold_of,
        graded_rows,
        note_messages,
        read_messages,
    )

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
        for board in boards_for(drafts, shared)
    ]
    flat = _generate(model, tokenizer, reader_msgs, n=1, max_new_tokens=max_new_tokens)
    replies = [[flat[i * COPIES + j][0] for j in range(COPIES)] for i in range(len(holdout))]
    groups = [
        {"notes": d, "replies": r, "gold": gold_of(t["answer"])}
        for t, d, r in zip(holdout, notes, replies)
    ]
    return graded_rows(holdout, replies), groups


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=150 * 60,
    # Host memory named, not left to the default, and one retry: the first
    # full run lost a container at step 11 with no Python error to read.
    memory=32768,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    shared: bool,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 40,
    prompts_per_step: int = 12,
    learning_rate: float = 1e-4,
    max_completion_length: int = 384,
    lora_rank: int = 32,
    eval_base: bool = False,
    train_seed: int = 17,
    eval_only: bool = False,
) -> dict:
    """One arm: eval the base model under both boards (optionally), train
    the reader under this arm's board, eval again. `eval_only` stops after
    the base evals, so they run in a container of their own."""
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
        board_trainer,
        make_reward,
        note_messages,
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
        "shared_board": shared,
        "base_model": base_model,
        "steps": steps,
        "copies": COPIES,
        "note_tokens": NOTE_TOKENS,
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
        "reward": "binary MathEqual on the reader's own answer",
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY") and not eval_only:
        run = training_run(
            run_name,
            base_model=base_model,
            trainer="trl-grpo-lora",
            total_steps=steps,
            config=config,
        )
        print(f"dashboard: {run.url}")

    def evaluate(m, board_shared: bool):
        return _board_eval(
            m, tokenizer, holdout, shared=board_shared, max_new_tokens=max_completion_length
        )

    # The base, EVAL_RUNS times under the own-note board (the noise floor,
    # Lambert 2025, chapter Evaluation), then once under the shared board:
    # does an untrained model already use its teammates' notes?
    base_runs: list[list[dict]] = []
    base_stats: dict = {}
    base_shared: list[dict] = []
    base_shared_stats: dict = {}
    if eval_base:
        for i in range(EVAL_RUNS):
            rows, groups = evaluate(model, False)
            base_runs.append(rows)
            if i == 0:
                base_stats = board_stats(groups)
            print(f"base run {i + 1}/{EVAL_RUNS}, own note: {wai.pass_at(rows)}")
        base_shared, groups = evaluate(model, True)
        base_shared_stats = board_stats(groups)
        print(f"base, shared board: {wai.pass_at(base_shared)} {base_shared_stats}")
    if eval_only:
        return {
            "arm": "base",
            "base_runs": base_runs,
            "base_stats": base_stats,
            "base_shared": base_shared,
            "base_shared_stats": base_shared_stats,
            "gpu_minutes": (time.time() - started) / 60.0,
        }

    dataset = Dataset.from_list(
        [
            {
                "prompt": note_messages(t["question"]),  # replaced per step by the board
                "question": t["question"],
                "answer": t["answer"],
            }
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
        max_completion_length=max_completion_length,
        # Four notes of up to 256 tokens plus the problem.
        max_prompt_length=1536,
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
    trainer = board_trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[make_reward(last_batch)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        shared=shared,
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
        for key in (
            "reward",
            "completions/mean_length",
            "board/note_acc",
            "board/vote_acc",
            "board/switched",
            "board/rescued",
            "board/misled",
        )
    }

    after_rows, groups = evaluate(trainer.model, shared)
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
        "shared_board": shared,
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
        "base_stats": base_stats,
        "base_shared": base_shared,
        "base_shared_stats": base_shared_stats,
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
    """The board, the answer reader and the counters, on the CPU."""
    notes = [
        "3 x 4 = 12, 12 + 5 = 17. \\boxed{17}",
        "12 + 5 = 17. \\boxed{17}",
        "3 x 4 = 12, 12 - 5 = 7. \\boxed{7}",
        "no answer",
    ]
    # The one change: shared boards carry every note, own boards only one.
    assert boards_for(notes, True) == [notes] * 4
    assert boards_for(notes, False) == [[n] for n in notes]
    msgs = read_messages("q", notes)
    assert "Note 4: no answer" in msgs[1]["content"] and msgs[0]["content"] == READ_SYSTEM

    # The answer reader and the vote.
    assert answer_of("so \\boxed{1,200}") == "1200"
    assert answer_of("\\boxed{$18.0}") == "18"
    assert answer_of("no box") is None
    assert majority(notes) == "\\boxed{17}"
    assert majority(["\\boxed{7} ", "\\boxed{9}"]) == "\\boxed{7}"  # tie: first posted

    # The counters, on one group whose gold is 17. Readers 0 and 1 keep a
    # right note; reader 2 switches from a wrong note to the right answer
    # (rescued); reader 3 had no answer and stays wrong.
    g = {
        "notes": notes,
        "replies": ["\\boxed{17}", "\\boxed{17}", "\\boxed{17}", "\\boxed{7}"],
        "gold": "17",
    }
    s = board_stats([g])
    assert s["note_acc"] == 0.5 and s["vote_acc"] == 1.0 and s["any_note"] == 1.0
    assert s["rescued"] == 0.25 and s["misled"] == 0.0 and s["switched"] == 0.5, s

    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("board: shared = every note, own = note j; vote = most common boxed answer")
    print("grader: MathEqual, decided by Math-Verify")
    print("selftest ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40)
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
    ap.add_argument("--reuse", action="store_true", help="skip calls cached in .cache/")
    ap.add_argument("--selftest", action="store_true", help="the board and counters, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]
    shared = {"baseline": False, "recipe": True}

    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(
        {
            "recipe": HERE.name,
            "title": "Message board: teach copies of a model to read each other's notes",
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
    after: dict[str, list[list[dict]]] = {arm: [] for arm in arms}
    stats: dict[str, list[dict]] = {arm: [] for arm in arms}
    traces: dict[str, list[dict]] = {arm: [] for arm in arms}
    minutes: dict[str, list[float]] = {arm: [] for arm in arms}
    samples: dict[str, list] = {}
    run_std = 0.0
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""

    jobs = [(arm, s) for s in args.train_seeds for arm in arms]
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)
    today = date.today().isoformat()

    def cached(name: str) -> dict | None:
        path = cache / f"{name}.json"
        return json.loads(path.read_text()) if args.reuse and path.exists() else None

    names = [f"message-board-{arm}-s{s}-{today}" for arm, s in jobs]
    with modal.enable_output(), app.run():
        # The base evals get a container of their own, so a failure there
        # never takes a training arm down with it; every finished call is
        # cached, so a rerun with --reuse only pays for what failed.
        base_call = None
        base_out = cached("base")
        if base_out is None:
            base_call = run_arm.spawn(
                "base", False, [], holdout, "message-board-base", eval_base=True, eval_only=True
            )
        calls = {}
        for (arm, s), name in zip(jobs, names):
            if cached(name) is None:
                calls[name] = run_arm.spawn(
                    arm, shared[arm], train_tasks, holdout, name, steps=args.steps, train_seed=s
                )
        failed = []
        for name, call in [("base", base_call), *calls.items()]:
            if call is None:
                continue
            try:
                out = call.get()
            except Exception as exc:  # one lost container must not lose the others
                print(f"FAILED {name}: {type(exc).__name__}: {exc}")
                failed.append(name)
                continue
            (cache / f"{name}.json").write_text(json.dumps(out))
    if failed:
        sys.exit(f"{len(failed)} call(s) failed: {failed}; rerun with --reuse")

    base_out = json.loads((cache / "base.json").read_text())
    outs = [json.loads((cache / f"{name}.json").read_text()) for name in names]
    gpu_minutes += base_out["gpu_minutes"]
    base_runs = base_out["base_runs"]
    noise = wai.eval_variance(*base_runs)
    run_std = float(noise["run_std"])
    print(f"eval noise over {len(base_runs)} base runs: run_std {run_std:.4f}")
    results["arms"]["base"] = {
        **summarize(base_runs[0], base_out["base_stats"]),
        "steps": 0,
        "gpu_minutes": 0,
    }
    results["arms"]["base_shared_board"] = {
        **summarize(base_out["base_shared"], base_out["base_shared_stats"]),
        "steps": 0,
        "gpu_minutes": 0,
    }
    checks["run_std"] = run_std
    checks["run_std_runs"] = int(noise["n_runs"])
    checks["length_before"] = mean_length(base_runs[0])

    for (arm, _), out in zip(jobs, outs):
        gpu_minutes += out["gpu_minutes"]
        minutes[arm].append(out["train_minutes"])
        run_url = out["run_url"] or run_url
        after[arm].append(out["after_rows"])
        stats[arm].append(out["after_stats"])
        traces[arm].append(out["trace"])
        samples.setdefault(arm, out["sample_groups"])
        checks["hack_scan_top"] = out["hack_scan_top"]
        checks["length_after"][arm] = mean_length(out["after_rows"])

    for arm in arms:
        pooled = [r for rows in after[arm] for r in rows]
        results["arms"][arm] = {
            **summarize(after[arm][0], stats[arm][0]),
            "per_seed": [summarize(rows)["score"] for rows in after[arm]],
            "board_per_seed": stats[arm],
            "pooled_score": summarize(pooled)["score"],
            "steps": args.steps,
            "gpu_minutes": round(statistics.fmean(minutes[arm]), 1),
        }
        results.setdefault("train_trace", {})[arm] = traces[arm]
    results["samples"] = samples

    if "baseline" in after and "recipe" in after:
        d = wai.compare(
            after["baseline"][0],
            after["recipe"][0],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks.get("run_std_runs") or EVAL_RUNS),
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
        results["verified"] = date.today().isoformat()
        results.pop("partial_run", None)
        print(results["delta"])
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in results.items() if k not in ("train_trace", "samples")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
