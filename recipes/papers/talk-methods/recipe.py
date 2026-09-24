"""Talk methods: which training teaches two copies of a model to talk best.

    python recipe.py --reuse              # all arms on Modal, three seeds, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the split, the chat, the credit, offline

talk-to-solve showed that GRPO on the team's outcome teaches two copies of
a model, each holding half the facts of a GSM8K problem, to talk: they send
what only they know and answer from what their partner sent. This recipe
keeps that task, that chat and that reward, and asks which training method
does it best. Four arms, three seeds each, the same 80 steps:

  baseline  GRPO: every turn paid team reward minus the group mean. The
            talk-to-solve chat arm, reused from its cache (same code path).
  recipe    Reinforce-Ada (Xiong et al. 2025): an untrained pair solves 11 in
            100, so most groups of 4 chats all fail and teach GRPO nothing.
            Replay a problem until its chats disagree (at most 3 rounds of
            4), keep 2 high and 2 low, measure against the whole pool.
  raft      Online rejection-sampling fine-tuning (RAFT, Dong et al. 2023):
            train only on chats the team solved outright, advantage 1, the
            rest 0. Fine-tuning on the winners, no negative signal.
  mixed     GRPO with mixed partners: in half the chats B is the untrained
            base, generated with the LoRA adapter off and never trained. The
            talk-to-solve protocol did not carry over to a new partner.

Every arm is also evaluated with the untrained base as B: does the skill
carry over to a partner that never trained?
"""

from __future__ import annotations

import argparse
import json
import os
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
METRIC = "pass@1"
BOOK = "Reasoning"  # RL on verifiable rewards, credited to every turn of a two-agent chat
# The training reward here *is* the target: both are the same binary check
# against the GSM8K gold, so there is no proxy to over-optimize against.
PROXY = None
EVAL_RUNS = 3  # re-runs of the base eval that set the noise floor (chapter Evaluation)

GROUP = 4  # chats per problem: the GRPO group
TURNS = ("A", "B", "A", "B")  # who speaks; the last turn of each copy answers
MSG_TOKENS = 160  # an ordinary message
ANSWER_TOKENS = 384  # a copy's last message, which works the problem and boxes it
NO_MESSAGE = "(no message)"

SYSTEM = (
    "You and a partner are solving one math problem together. Each of you sees "
    "different facts; neither of you has all of them. You can only talk by message. "
    "Keep messages short and useful."
)


# --------------------------------------------------------------------------
# Pure functions, no torch: `--selftest` runs them, and the Modal container
# imports this same file.
# --------------------------------------------------------------------------

SENTENCE = re.compile(r"(?<=[.?!])\s+(?=[A-Z$])")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def split_problem(question: str) -> tuple[list[str], list[str], str] | None:
    """Facts for A, facts for B, the question. A takes every other fact
    sentence starting with the first, B the rest; both halves must hold a
    number, or the problem is left out."""
    parts = [s.strip() for s in SENTENCE.split(question.strip()) if s.strip()]
    asks = [i for i, s in enumerate(parts) if s.endswith("?")]
    if not asks:
        return None
    ask = parts[asks[-1]]
    facts = [s for i, s in enumerate(parts) if i != asks[-1]]
    if len(facts) < 2:
        return None
    a, b = facts[0::2], facts[1::2]
    if not any(NUMBER.search(s) for s in a) or not any(NUMBER.search(s) for s in b):
        return None
    return a, b, ask


def numbers(text: str) -> set[str]:
    return set(NUMBER.findall((text or "").replace(",", "")))


def last_turn(t: int, who: str) -> bool:
    return t == max(i for i, w in enumerate(TURNS) if w == who)


def turn_messages(task: dict, who: str, chat: list[tuple[str, str]], channel: bool) -> list[dict]:
    """What copy `who` sees at its turn: its facts, the question, the chat so
    far in its own words and its partner's (or NO_MESSAGE when the channel is
    off), and what to write now."""
    facts = task["facts_a"] if who == "A" else task["facts_b"]
    lines = []
    for speaker, text in chat:
        if speaker == who:
            lines.append(f"You: {text.strip()}")
        else:
            lines.append(f"Partner: {text.strip() if channel else NO_MESSAGE}")
    t = len(chat)
    ask = (
        "Write your last message, then give the final answer as \\boxed{answer}."
        if last_turn(t, who)
        else "Write your next message to your partner."
    )
    body = (
        f"Your facts: {' '.join(facts)}\nQuestion: {task['ask']}\n\n"
        f"Conversation so far:\n{chr(10).join(lines) if lines else '(nothing yet)'}\n\n{ask}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


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


def team_reward(chat: list[tuple[str, str]], gold: str) -> float:
    """The mean of A's and B's final answers. Every turn of the chat is paid this."""
    finals = [text for t, (who, text) in enumerate(chat) if last_turn(t, who)]
    return statistics.fmean(outcome_of(text, gold) for text in finals)


def talk_stats(chats: list[dict]) -> dict:
    """What the copies said. One entry per chat: `task`, `chat`, `channel`.

    shared:   of the numbers only this copy has, the share it put in a message
              before its last turn (the facts its partner cannot get otherwise)
    used:     of the numbers only the partner has, the share this copy's last
              message uses (the partner's facts reaching the answer)
    asked:    first messages that ask the partner something ('?')
    msg_chars: mean length of a non-final message
    agree:    chats where A and B box the same answer
    """
    shared = used = asked = agree = chars = 0.0
    n_shared = n_used = n_first = n_msgs = 0
    for c in chats:
        task, chat = c["task"], c["chat"]
        own = {"A": numbers(" ".join(task["facts_a"])), "B": numbers(" ".join(task["facts_b"]))}
        ask_nums = numbers(task["ask"])
        only = {
            "A": own["A"] - own["B"] - ask_nums,
            "B": own["B"] - own["A"] - ask_nums,
        }
        finals: dict[str, str] = {}
        for t, (who, text) in enumerate(chat):
            partner = "B" if who == "A" else "A"
            if last_turn(t, who):
                finals[who] = text
                if only[partner]:
                    used += len(only[partner] & numbers(text)) / len(only[partner])
                    n_used += 1
            else:
                chars += len(text)
                n_msgs += 1
                if t < 2:
                    asked += "?" in text
                    n_first += 1
        for who in ("A", "B"):
            early = " ".join(
                text for t, (w, text) in enumerate(chat) if w == who and not last_turn(t, who)
            )
            if only[who]:
                shared += len(only[who] & numbers(early)) / len(only[who])
                n_shared += 1
        a, b = (finals.get(w) for w in ("A", "B"))
        agree += bool(a and b and _boxed(a) is not None and _boxed(a) == _boxed(b))
    k = max(len(chats), 1)
    return {
        "shared": round(shared / max(n_shared, 1), 3),
        "used": round(used / max(n_used, 1), 3),
        "asked": round(asked / max(n_first, 1), 3),
        "msg_chars": round(chars / max(n_msgs, 1)),
        "agree": round(agree / k, 3),
    }


def _boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = text.find("}", i)
    return text[i + 7 : j].replace(",", "").replace("$", "").strip() if j > 0 else None


def graded_rows(chats: list[dict]) -> list[dict]:
    """Eval rows: each copy's final answer is a row, grouped by problem."""
    rows: list[dict] = []
    seen: dict[str, int] = {}
    for c in chats:
        task = c["task"]
        for t, (who, text) in enumerate(c["chat"]):
            if not last_turn(t, who):
                continue
            sid = task["scenario_id"]
            rows.append(
                {
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": outcome_of(text, task["gold"]),
                    "scenario_id": sid,
                    "rollout_index": seen.get(sid, 0),
                    "privileged": {"reference": task["gold"]},
                }
            )
            seen[sid] = seen.get(sid, 0) + 1
    return rows


def make_reward(recorder: list[dict]):
    """TRL wants a reward function per generation call. The trainer pays
    every turn itself (team_reward); this one only records each turn's
    text so `hack_scan` can read the last batch."""

    def reward(completions, prompts, gold, **kwargs) -> list[float]:
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        recorder.clear()
        for i, (p, text, g) in enumerate(zip(prompts, texts, gold)):
            recorder.append(
                {
                    "prompt": json.dumps(p, default=str),
                    "final_text": text,
                    "reward": outcome_of(text, g),
                    "scenario_id": json.dumps(p, default=str),
                    "rollout_index": i,
                }
            )
        return [r["reward"] for r in recorder]

    reward.__name__ = "gsm8k_outcome"
    return reward


def mean_length(rows: list[dict]) -> float:
    return statistics.fmean(len(r.get("final_text") or "") for r in rows) if rows else 0.0


METHODS = ("grpo", "ada", "raft", "mixed")
# Arm name -> method. `baseline` and `recipe` are the names check.py reads:
# the talk-to-solve GRPO chat and the method this recipe bets on.
ARMS = {"baseline": "grpo", "recipe": "ada", "raft": "raft", "mixed": "mixed"}
ADA_MAX_ROUNDS = 3  # rounds of GROUP chats before a flat problem keeps what it has
MIXED_SHARE = 0.5  # share of chats in which B is the untrained base


def credit(method: str, rewards: list[float], pool: list[float] | None = None) -> list[float]:
    """THE ONE CHANGE, per method: the advantage every turn of a chat gets.

    grpo:  team reward minus the group mean (Shao et al. 2024)
    ada:   team reward minus the mean over every chat drawn for the problem,
           not only the GROUP kept (Xiong et al. 2025, global_stat_est)
    raft:  1 for a chat the team solved outright, else 0: fine-tuning on the
           winners, online (Dong et al. 2023)
    mixed: as grpo; the difference is who plays B (see ChatTrainer)
    """
    if method == "raft":
        return [1.0 if r == 1.0 else 0.0 for r in rewards]
    base = pool if method == "ada" and pool else rewards
    mean = statistics.fmean(base)
    return [r - mean for r in rewards]


def keep_balanced(rewards: list[float], n: int = GROUP) -> list[int]:
    """Which n chats of a pool to train on: half from the top, half from the
    bottom, so a pool with any spread hands back a group with spread."""
    order = sorted(range(len(rewards)), key=lambda i: rewards[i])
    low, high = order[: n // 2], order[-(n - n // 2) :]
    picked = list(dict.fromkeys(high + low))
    for i in order:
        if len(picked) == n:
            break
        if i not in picked:
            picked.append(i)
    return picked[:n]


def chat_trainer(base_cls):
    """GRPOTrainer whose generation step plays the chat.

    TRL hands `_generate_and_score_completions` a batch of problems, each
    repeated `num_generations` (= GROUP) times in a row. `_play_round` calls
    the parent once per turn, so generation is TRL's own, and returns every
    chat with its rows (one per turn) and its team reward. The method then
    decides which chats train and with what advantage, and `_join` hands back
    one batch, re-padded. On-policy and without KL is a requirement: the
    rebuilt batch carries no old or reference log-probs.
    """
    import torch

    class ChatTrainer(base_cls):  # type: ignore[valid-type,misc]
        def __init__(self, *args, method: str, recorder: list[dict], **kw):
            super().__init__(*args, **kw)
            if method not in METHODS:
                raise ValueError(f"method must be one of {METHODS}")
            self.method = method
            self.recorder = recorder
            self.rounds_seen: list[int] = []
            if self.num_iterations != 1 or self.beta != 0.0:
                raise RuntimeError("run with num_iterations=1 and beta=0")
            if self.num_generations != GROUP:
                raise RuntimeError(f"num_generations must be GROUP ({GROUP})")

        def _turn(self, batch: list[dict], max_new: int, base: bool = False) -> dict:
            # TRL 0.19.1 generates with self.generation_config (HF generate)
            # and clips with self.max_completion_length; both carry the cap.
            # `base` plays the turn with the LoRA adapter off: the untrained model.
            import contextlib

            keep = self.max_completion_length, self.generation_config.max_new_tokens
            self.max_completion_length = self.generation_config.max_new_tokens = max_new
            ctx = self.model.disable_adapter() if base else contextlib.nullcontext()
            try:
                with ctx:
                    return super()._generate_and_score_completions(batch)
            finally:
                self.max_completion_length, self.generation_config.max_new_tokens = keep

        def _rows(self, out: dict) -> list[tuple]:
            return [
                (
                    out["prompt_ids"][j][out["prompt_mask"][j].bool()],
                    out["completion_ids"][j],
                    out["completion_mask"][j],
                )
                for j in range(out["completion_ids"].shape[0])
            ]

        def _play_round(self, inputs: list[dict], base_b: list[bool]) -> list[dict]:
            """One chat per input. `base_b[i]` plays chat i's B with the
            untrained base; those turns are generated but never trained."""
            tok = self.processing_class
            chats = [{"task": x, "chat": [], "rows": [], "train": []} for x in inputs]
            for t, who in enumerate(TURNS):
                cap = ANSWER_TOKENS if last_turn(t, who) else MSG_TOKENS
                groups = [[i for i in range(len(inputs)) if not (who == "B" and base_b[i])]]
                if who == "B" and any(base_b):
                    groups.append([i for i in range(len(inputs)) if base_b[i]])
                for k, idx in enumerate(groups):
                    if not idx:
                        continue
                    batch = [
                        {
                            **inputs[i],
                            "prompt": turn_messages(inputs[i], who, chats[i]["chat"], True),
                        }
                        for i in idx
                    ]
                    out = self._turn(batch, cap, base=(k == 1))
                    texts = tok.batch_decode(out["completion_ids"], skip_special_tokens=True)
                    for i, text, row in zip(idx, texts, self._rows(out)):
                        chats[i]["chat"].append((who, text))
                        chats[i]["rows"].append(row)
                        chats[i]["train"].append(k == 0)
            for c in chats:
                c["reward"] = team_reward(c["chat"], c["task"]["gold"])
            return chats

        def _generate_and_score_completions(self, inputs):
            problems = inputs[::GROUP]
            if self.method == "mixed":
                # Half the chats of every problem get the untrained base as B.
                n_base = round(GROUP * MIXED_SHARE)
                flags = [j >= GROUP - n_base for _ in problems for j in range(GROUP)]
            else:
                flags = [False] * len(inputs)
            played = self._play_round(inputs, flags)
            pools = [played[p * GROUP : (p + 1) * GROUP] for p in range(len(problems))]

            rounds = 1
            if self.method == "ada":
                # Reinforce-Ada-Seq (Xiong et al. 2025): replay every problem
                # whose chats all scored the same, up to ADA_MAX_ROUNDS.
                while rounds < ADA_MAX_ROUNDS:
                    flat = [
                        p for p, pool in enumerate(pools) if len({c["reward"] for c in pool}) == 1
                    ]
                    if not flat:
                        break
                    again = [problems[p] for p in flat for _ in range(GROUP)]
                    more = self._play_round(again, [False] * len(again))
                    for k, p in enumerate(flat):
                        pools[p] += more[k * GROUP : (k + 1) * GROUP]
                    rounds += 1
            self.rounds_seen.append(rounds)

            rows, advantages, kept = [], [], []
            for pool in pools:
                rewards = [c["reward"] for c in pool]
                pick = keep_balanced(rewards) if len(pool) > GROUP else list(range(GROUP))
                chosen = [pool[i] for i in pick]
                adv = credit(self.method, [c["reward"] for c in chosen], rewards)
                for c, a in zip(chosen, adv):
                    kept.append(c)
                    for row, trains in zip(c["rows"], c["train"]):
                        rows.append(row)
                        advantages.append(a if trains else 0.0)

            metrics = self._metrics["train"]
            episodes = [{"task": c["task"], "chat": c["chat"], "channel": True} for c in kept]
            for key, value in talk_stats(episodes).items():
                metrics[f"talk/{key}"].append(float(value))
            metrics["talk/team_reward"].append(statistics.fmean(c["reward"] for c in kept))
            metrics["method/rounds"].append(float(rounds))
            metrics["method/flat_groups"].append(
                statistics.fmean(len({c["reward"] for c in pool[:GROUP]}) == 1 for pool in pools)
            )
            metrics["method/adv_abs"].append(statistics.fmean(abs(a) for a in advantages))
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

    return ChatTrainer


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/papers/board-writers.
# --------------------------------------------------------------------------

# H100: four turns of a chat train in one batch, which ran out of memory on
# the L40S in the board-writers smoke run.
DEFAULT_GPU = os.environ.get("WAI_RECIPE_GPU", "H100")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-recipe-talk-methods")

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


def _generate(model, tokenizer, message_lists, *, max_new_tokens, batch=32):
    """One reply per message list, batched, sampled the way the trainer samples."""
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    out: list[str] = []
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
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        out += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    model.train()
    return out


def _play(model, tokenizer, holdout, *, channel, chats_per_task, untrained_b=False):
    """Play the chat on every held-out task. `untrained_b` plays B with the
    LoRA adapter switched off: a trained A talking to the base model."""
    import contextlib

    sys.path.insert(0, "/root")
    from recipe_mod import ANSWER_TOKENS, MSG_TOKENS, TURNS, last_turn, turn_messages

    tasks = [t for t in holdout for _ in range(chats_per_task)]
    chats: list[list[tuple[str, str]]] = [[] for _ in tasks]
    for t, who in enumerate(TURNS):
        msgs = [turn_messages(x, who, chat, channel) for x, chat in zip(tasks, chats)]
        off = untrained_b and who == "B"
        ctx = model.disable_adapter() if off else contextlib.nullcontext()
        with ctx:
            texts = _generate(
                model,
                tokenizer,
                msgs,
                max_new_tokens=ANSWER_TOKENS if last_turn(t, who) else MSG_TOKENS,
            )
        for chat, text in zip(chats, texts):
            chat.append((who, text))
    return [{"task": x, "chat": c, "channel": channel} for x, c in zip(tasks, chats)]


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
    method: str,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 80,
    problems_per_step: int = 8,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    chats_per_task: int = 2,
    train_seed: int = 17,
    eval_only: bool = False,
) -> dict:
    """One arm: train every turn of the chat on the team's outcome, eval
    after. `eval_only` evaluates the untrained base, channel on (EVAL_RUNS
    times, the noise floor) and off (once), and stops."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        ANSWER_TOKENS,
        EVAL_RUNS,
        GROUP,
        MSG_TOKENS,
        TURNS,
        chat_trainer,
        graded_rows,
        make_reward,
        talk_stats,
        turn_messages,
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

    def play(m, ch, **kw):
        chats = _play(m, tokenizer, holdout, channel=ch, chats_per_task=chats_per_task, **kw)
        return graded_rows(chats), talk_stats(chats), chats[:4]

    if eval_only:
        base_runs, base_stats, samples = [], {}, []
        for i in range(EVAL_RUNS):
            rows, stats, chats = play(model, True)
            base_runs.append(rows)
            if i == 0:
                base_stats, samples = stats, chats
            print(f"base run {i + 1}/{EVAL_RUNS}, channel on: {wai.pass_at(rows)} {stats}")
        off_rows, off_stats, _ = play(model, False)
        print(f"base, channel off: {wai.pass_at(off_rows)} {off_stats}")
        return {
            "base_runs": base_runs,
            "base_stats": base_stats,
            "base_off": off_rows,
            "base_off_stats": off_stats,
            "samples": samples,
            "gpu_minutes": (time.time() - started) / 60.0,
        }

    config = {
        "arm": arm,
        "method": method,
        "base_model": base_model,
        "steps": steps,
        "turns": "".join(TURNS),
        "group": GROUP,
        "msg_tokens": MSG_TOKENS,
        "answer_tokens": ANSWER_TOKENS,
        "problems_per_step": problems_per_step,
        "learning_rate": learning_rate,
        "beta": 0.0,
        "lora_rank": lora_rank,
        "train_prompts": len(train_tasks),
        "holdout": len(holdout),
        "train_seed": train_seed,
        "gpu": DEFAULT_GPU,
        "reward": "every turn: mean of A's and B's final MathEqual, credited by method",
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
        [{**t, "prompt": turn_messages(t, "A", [], True)} for t in train_tasks]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=GROUP,
        # Two chats' rows per micro-batch, not four: chats grow as the copies
        # learn to talk, and four ran an H100 out of memory at step 45.
        per_device_train_batch_size=GROUP // 2,
        gradient_accumulation_steps=problems_per_step * 2,
        learning_rate=learning_rate,
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        scale_rewards=False,
        max_completion_length=ANSWER_TOKENS,
        # Facts, question and three earlier messages.
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
    trainer = chat_trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[make_reward(last_batch)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        method=method,
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
        "talk/team_reward",
        "talk/shared",
        "talk/used",
        "talk/asked",
        "talk/msg_chars",
        "method/rounds",
        "method/flat_groups",
        "method/adv_abs",
    )
    trace = {key: [h[key] for h in history if key in h] for key in keys}

    after_rows, after_stats, samples = play(trainer.model, True)
    print(f"{arm}: {wai.pass_at(after_rows)} {after_stats}")
    out = {
        "arm": arm,
        "after_rows": after_rows,
        "after_stats": after_stats,
        "samples": samples,
        "trace": trace,
    }
    # Does what it learned carry over to a partner that never trained?
    cross_rows, cross_stats, _ = play(trainer.model, True, untrained_b=True)
    print(f"{arm}, trained A + untrained B: {wai.pass_at(cross_rows)} {cross_stats}")
    out["cross_rows"], out["cross_stats"] = cross_rows, cross_stats

    scan = wai.hack_scan(last_batch) if last_batch else {}
    hack_top = (scan.get("top_feature") or {}) if isinstance(scan, dict) else {}
    out["hack_scan_top"] = hack_top.get("name", "") if isinstance(hack_top, dict) else str(hack_top)

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
        "method": method,
        "pass_at_1": wai.pass_at(after_rows).pass_at_1,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"whileai-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    out.update(
        gpu_minutes=gpu_minutes,
        train_minutes=train_minutes,
        steps=steps,
        run_url=summary.get("run_url", ""),
    )
    return out


# --------------------------------------------------------------------------
# Local: data, orchestration, results.json.
# --------------------------------------------------------------------------


def _tasks(rows, prefix: str, limit: int) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        split = split_problem(r["question"])
        if split is None:
            continue
        a, b, ask = split
        out.append(
            {
                "question": r["question"],
                "facts_a": a,
                "facts_b": b,
                "ask": ask,
                "gold": gold_of(r["answer"]),
                "scenario_id": f"{prefix}-{i}",
            }
        )
        if len(out) == limit:
            break
    return out


def data(seed: int, n_train: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    """GSM8K problems that split into two halves with a number each. Train
    from the train split, holdout from the test split: disjoint by
    construction."""
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    return _tasks(train, "train", n_train), _tasks(test, "test", n_holdout)


def summarize(rows: list[dict], stats: dict | None = None) -> dict:
    import whileai as wai

    p = wai.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
        "chars": round(mean_length(rows)),
        **({"talk": stats} if stats else {}),
    }


def selftest() -> None:
    """The split, the chat and the counters, on the CPU."""
    q = (
        "Josh buys a house for $80,000. He puts in $50,000 in repairs. "
        "This increased the value of the house by 150%. How much profit did he make?"
    )
    a, b, ask = split_problem(q) or ([], [], "")
    assert a == ["Josh buys a house for $80,000.", "This increased the value of the house by 150%."]
    assert b == ["He puts in $50,000 in repairs."] and ask.endswith("profit did he make?")
    assert split_problem("Tom has 3 apples. How many?") is None  # one fact: nothing to split
    assert split_problem("Tom is tall. He has 3 apples. How many?") is None  # A holds no number

    task = {
        "question": q,
        "facts_a": a,
        "facts_b": b,
        "ask": ask,
        "gold": "70000",
        "scenario_id": "t",
    }
    chat = [("A", "I have 80,000 and +150%. What are yours?")]
    # THE ONE CHANGE: the partner's words, or NO_MESSAGE.
    on = turn_messages(task, "B", chat, channel=True)[1]["content"]
    off = turn_messages(task, "B", chat, channel=False)[1]["content"]
    assert "Partner: I have 80,000" in on and f"Partner: {NO_MESSAGE}" in off
    assert "50,000" in on and "80,000 and" not in off
    assert "next message" in on
    assert "\\boxed" in turn_messages(task, "A", [*chat, ("B", "x")], True)[1]["content"]
    assert [last_turn(t, w) for t, w in enumerate(TURNS)] == [False, False, True, True]

    chat = [
        ("A", "I have 80,000 and 150%. What are yours?"),
        ("B", "Repairs were 50,000."),
        ("A", "Cost 130,000, value 200,000, profit \\boxed{70000}"),
        ("B", "80000 * 2.5 - 130000 = \\boxed{70,000}"),
    ]
    assert team_reward(chat, "70000") == 1.0
    assert team_reward([*chat[:2], ("A", "\\boxed{5}"), chat[3]], "70000") == 0.5
    s = talk_stats([{"task": task, "chat": chat, "channel": True}])
    # A shared both its private numbers (80000 and 150) and B shared 50000;
    # A's answer used B's 50000 (as 130,000, so no: a derived number), B's
    # answer used A's 80000 but not 150.
    assert s["shared"] == 1.0 and s["asked"] == 0.5 and s["agree"] == 1.0, s
    assert s["used"] == 0.25, s
    rows = graded_rows([{"task": task, "chat": chat, "channel": True}])
    assert [r["rollout_index"] for r in rows] == [0, 1] and all(r["reward"] == 1.0 for r in rows)
    # The credit, per method.
    assert credit("grpo", [1.0, 0.5, 0.0, 0.0]) == [0.625, 0.125, -0.375, -0.375]
    assert credit("raft", [1.0, 0.5, 0.0, 0.0]) == [1.0, 0.0, 0.0, 0.0]
    assert credit("ada", [1.0, 0.0, 0.0, 0.0], [1.0] + [0.0] * 11) == [
        1 - 1 / 12,
        -1 / 12,
        -1 / 12,
        -1 / 12,
    ]
    assert credit("mixed", [0.5, 0.5]) == [0.0, 0.0]
    # Ada keeps 2 high and 2 low from a pool with any spread.
    pool = [0.0] * 10 + [1.0, 0.5]
    kept = keep_balanced(pool)
    assert sorted(pool[i] for i in kept) == [0.0, 0.0, 0.5, 1.0], kept
    assert len(set(keep_balanced([0.0] * 12))) == 4
    print("split: every other fact to A, the rest to B, both halves hold a number")
    print("channel off: the partner's words replaced by", NO_MESSAGE)
    print("selftest ok")


def main() -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="all")
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
    ap.add_argument("--selftest", action="store_true", help="the split and the chat, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import whileai as wai

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    train_tasks, decon = wai.decontaminate(train_tasks, against=holdout)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    print(f"{len(train_tasks)} train problems, {len(holdout)} held out")
    arms = list(ARMS) if args.arm == "all" else [args.arm]

    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update(
        {
            "recipe": HERE.name,
            "title": "Talk methods: which training teaches two model copies to talk best",
            "paper": "https://arxiv.org/abs/2510.04996",
            "book": BOOK,
            "base_model": BASE_MODEL,
            "metric": METRIC,
            "n_holdout": len(holdout),
            "k": 4,
            "gpu": DEFAULT_GPU,
            "whileai": version("whileai"),
            "methods": dict(ARMS),
        }
    )
    results.setdefault("arms", {})
    checks = results.setdefault("checks", {})
    checks["decontaminated_dropped"] = int(decon.get("n_contaminated", 0))
    checks["seed"] = args.seed
    checks.setdefault("length_after", {})
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 4.00)

    jobs = [(arm, s) for s in args.train_seeds for arm in arms]
    cache = HERE / ".cache"
    cache.mkdir(exist_ok=True)
    names = {(arm, s): f"talk-methods-{arm}-s{s}" for arm, s in jobs}

    def cached(name: str) -> bool:
        return args.reuse and (cache / f"{name}.json").exists()

    with modal.enable_output(), app.run():
        # The base evals get a container of their own, and every finished
        # call is cached, so one lost container costs only itself.
        calls = {}
        if not cached("base"):
            calls["base"] = run_arm.spawn(
                "base", "grpo", [], holdout, "talk-methods-base", eval_only=True
            )
        for (arm, s), name in names.items():
            if not cached(name):
                calls[name] = run_arm.spawn(
                    arm, ARMS[arm], train_tasks, holdout, name, steps=args.steps, train_seed=s
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
    outs = {key: json.loads((cache / f"{name}.json").read_text()) for key, name in names.items()}
    gpu_minutes = base["gpu_minutes"] + sum(o["gpu_minutes"] for o in outs.values())
    noise = wai.eval_variance(*base["base_runs"])
    run_std = float(noise["run_std"])
    results["arms"]["base"] = {
        **summarize(base["base_runs"][0], base["base_stats"]),
        "steps": 0,
        "gpu_minutes": 0,
    }
    results.setdefault("samples", {})["base"] = base["samples"]
    checks["run_std"] = run_std
    checks["run_std_runs"] = int(noise["n_runs"])
    checks["length_before"] = mean_length(base["base_runs"][0])

    after: dict[str, list[list[dict]]] = {arm: [] for arm in arms}
    cross: dict[str, list[list[dict]]] = {arm: [] for arm in arms}
    run_url = ""
    for (arm, _), out in outs.items():
        after[arm].append(out["after_rows"])
        cross[arm].append(out["cross_rows"])
        run_url = out["run_url"] or run_url
        results.setdefault("train_trace", {}).setdefault(arm, []).append(out["trace"])
        results["samples"].setdefault(arm, out["samples"])
        checks["hack_scan_top"] = out["hack_scan_top"]
        checks["length_after"][arm] = mean_length(out["after_rows"])
    for arm in arms:
        seeds = [outs[(arm, s)] for s in args.train_seeds]
        results["arms"][arm] = {
            **summarize(after[arm][0], seeds[0]["after_stats"]),
            "method": ARMS[arm],
            "per_seed": [summarize(rows)["score"] for rows in after[arm]],
            "talk_per_seed": [o["after_stats"] for o in seeds],
            "pooled_score": summarize([r for rows in after[arm] for r in rows])["score"],
            "steps": args.steps,
            "gpu_minutes": round(statistics.fmean(o["gpu_minutes"] for o in seeds), 1),
        }
        results["arms"][f"{arm}_with_untrained_partner"] = {
            **summarize(cross[arm][0], seeds[0]["cross_stats"]),
            "per_seed": [summarize(rows)["score"] for rows in cross[arm]],
            "steps": args.steps,
        }

    def paired(a: str, b: str, pool: dict) -> dict:
        d = wai.compare(
            pool[a][0],
            pool[b][0],
            target="pass_at_1",
            run_std=run_std,
            run_std_runs=int(checks["run_std_runs"]),
            train_runs={"before": pool[a], "after": pool[b]},
            proxy=PROXY,
        )
        verdict = d["target_verdict"]
        return {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": verdict if verdict in ("moved", "flat", "unresolved") else "flat",
            "over_optimized": bool(d.get("over_optimized")),
        }

    if "baseline" in after and len(arms) > 1:
        deltas = {arm: paired("baseline", arm, after) for arm in arms if arm != "baseline"}
        # Portability: each method's trained A with the untrained base as B.
        partner = {arm: paired("baseline", arm, cross) for arm in arms if arm != "baseline"}
        over = False
        for d in [*deltas.values(), *partner.values()]:
            over = over or d.pop("over_optimized")
        checks["over_optimized"] = over
        results["deltas"] = deltas
        results["deltas_untrained_partner"] = partner
        if "recipe" in deltas:
            results["delta"] = deltas["recipe"]
        checks["train_seeds"] = {arm: len(after[arm]) for arm in after}
        results["verified"] = date.today().isoformat()
        results.pop("partial_run", None)
        print(json.dumps({"deltas": deltas, "untrained_partner": partner}, indent=2))
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    skip = ("train_trace", "samples")
    print(json.dumps({k: v for k, v in results.items() if k not in skip}, indent=2))


if __name__ == "__main__":
    main()
