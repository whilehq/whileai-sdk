"""Hint distillation: imitate the replies that passed, correct the ones that failed with a hint.

    python recipe.py                    # every stage, writes results.json (~55 GPU min, ~$5)
    python recipe.py --stage hints      # one stage: experience | hints | validate | train | results
    python recipe.py --selftest         # the pure parts, offline

Perplexity (2026) trains its agent on logged sessions in two ways at once.
Replies from sessions that succeeded get ordinary cross-entropy
(rejection-sampling fine-tuning, RFT). A reply that caused a tool error or a
user correction gets a short hint, written from what the model already knew,
and the model is pulled toward what it predicts when it reads that hint:
forward KL from the same model with the hint in its context (the teacher,
detached) to the model without it. One loss, one shared denominator:

    L = (L_CE + lambda * L_KL) / (CE tokens in the global batch)

so lambda = 0 is plain RFT on identical batches. That is the baseline here.

Shape of the run (text-to-SQL shop, Qwen3-4B, thinking on):
  1. experience: the base answers the 480 train tasks 4 times on Modal; the
     SQL verifier judges each reply (the "logged sessions").
  2. hints:      failed, complete replies get a hint from a writer model that
     sees the gold query; a checker that never sees it rejects any hint whose
     claims do not follow from the question, schema and error message.
  3. validate:   the base regenerates each hinted failure with and without the
     hint. Does the hint carry signal before any training?
  4. train:      RFT (lambda 0), RFT + hint KL over the whole reply (lambda 1,
     the paper), and RFT + hint KL over the answer after </think> only.
  5. results:    pass@1 on the same 120 held-out tasks, 4 samples each, paired.

The data, prompt and verifier are the text-to-SQL step recipe's
(``recipes/04-train/text-to-sql``); the GRPO row in the README comes from its
``train_grpo_modal.py`` with the command printed there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
T2S = HERE.parents[1] / "04-train" / "text-to-sql"
TITLE = "Hint distillation: imitate what passed, correct what failed with a hint"
PAPER = "https://www.perplexity.ai/hub/blog/learning-from-real-world-experience"
BASE_MODEL = "Qwen/Qwen3-4B"
METRIC = "pass@1"
BOOK = "Rejection Sampling"  # the chapter title of Lambert 2025 the baseline arm is
VOLUME_ROOT = "/vol"
LANE = "hint-sep22"  # folder on the runs volume
H100_USD_PER_HOUR = 4.56
SEED = 17

# The split of train_grpo_modal.py --limit 480: hash of the task id, first
# 480 train and first 120 held out, so every arm here faces the GRPO arm's test.
N_TRAIN, N_HOLDOUT, HOLDOUT_SHARE = 480, 120, 0.2
K = 4  # samples per task, experience and eval
SAMPLING = dict(temperature=0.7, top_p=0.95, max_tokens=1536)  # the served settings
IM_END = 151645  # <|im_end|>
EOS_IDS = {151645, 151643}
THINK_END = 151668  # </think>

# Training: LoRA SFT on the model's own replies. 2 epochs of the ~916 items
# (870 imitate + 46 correct) at 16 a step = 116 steps.
EPOCHS, BATCH, LR, LORA_RANK = 2, 16, 1e-4, 16
LAMBDA = 1.0  # KL weight; with the shared CE-token denominator one KL token weighs one CE token

# Hint annotation (local, Bedrock). The writer sees the gold to locate the
# mistake; the checker never does (the paper's hindsight check).
WRITER = "global.anthropic.claude-sonnet-5"
CHECKER = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
PER_TASK = 2  # at most this many distinct failed queries per task get a hint
HINT_MAX_WORDS = 60

ARMS = {  # name: (lambda, kl scope)
    "rft": (0.0, "all"),
    "rft-hint-kl": (LAMBDA, "all"),
    "rft-hint-kl-answer": (LAMBDA, "answer"),
}

# ------------------------------------------------------------------ pure parts


def bucket(task_id: str) -> float:
    return int(hashlib.sha256(str(task_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_tasks(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [t for t in tasks if bucket(t["id"]) >= HOLDOUT_SHARE][:N_TRAIN]
    hold = [t for t in tasks if bucket(t["id"]) < HOLDOUT_SHARE][:N_HOLDOUT]
    return train, hold


def with_hint(question: str, hint: str) -> str:
    """The teacher's user turn. The student's is the question alone."""
    return f"{question}\n\nHint: {hint}"


def failure_kind(r: dict) -> str | None:
    """tool_error (Postgres rejected it) or wrong_result (ran, wrong rows).
    Passing, cut-off and query-less replies get no hint: they stay context."""
    if r["correct"] or r["finish_reason"] != "stop" or not r.get("sql"):
        return None
    return "tool_error" if r["reason"].startswith("sql error") else "wrong_result"


def program_check(hint: str) -> bool:
    """No code, no query, short."""
    return (
        bool(hint)
        and "```" not in hint
        and not re.search(r"\bselect\b.+\bfrom\b", hint, re.I)
        and len(hint.split()) <= HINT_MAX_WORDS
    )


def kl_start(completion: list[int], scope: str) -> int:
    """First completion position that gets KL: 0, or just after </think>."""
    if scope == "answer" and THINK_END in completion:
        return completion.index(THINK_END) + 1
    return 0


def paired(a: dict[str, float], b: dict[str, float], n: int = 10_000) -> tuple[float, float, float]:
    """Mean of per-task (b - a) and a 95% bootstrap interval over tasks."""
    keys = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in keys]
    rng = random.Random(SEED)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def per_task(rows: list[dict]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["scenario_id"], []).append(float(bool(r.get("reward"))))
    return {k: sum(v) / len(v) for k, v in by.items()}


# ------------------------------------------------------------------ Modal

app = modal.App("whileai-t2s-hint-distill")
# The pinned stack of train_grpo_modal.py (torch 2.7.1, transformers 4.54,
# peft 0.16, vLLM 0.10 V0), so GRPO and this recipe share library versions.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("postgresql", "postgresql-contrib")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "psycopg[binary]==3.2.9",
        "whileai>=0.51",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .pip_install("vllm==0.10.0")
    .env({"VLLM_USE_V1": "0"})
)
for _name in ("sql_verifier.py", "schema_prompt.py", "schema.sql", "seed.sql", "prompt.txt"):
    image = image.add_local_file(str(T2S / _name), f"/root/{_name}")
image = image.add_local_file(__file__, "/root/hint_recipe.py")

runs_volume = modal.Volume.from_name("whileai-train-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
_FN = dict(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
)


def _setup():
    sys.path.insert(0, "/root")
    import sql_verifier as R

    os.environ.setdefault("T2S_STATEMENT_TIMEOUT_MS", "5000")
    R.start_postgres(open("/root/schema.sql").read(), open("/root/seed.sql").read())
    return R, open("/root/prompt.txt", encoding="utf-8").read()


def _render(tok, system_prompt: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _llm(lora: bool = False):
    from vllm import LLM

    kw = dict(
        model=BASE_MODEL,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        seed=SEED,
        dtype="bfloat16",
    )
    if lora:
        kw.update(enable_lora=True, max_lora_rank=LORA_RANK, max_loras=1)
    return LLM(**kw)


def _generate(llm, texts: list[str], n: int, lora_request=None):
    from vllm import SamplingParams

    sp = SamplingParams(n=n, seed=SEED, **SAMPLING)
    return llm.generate(texts, sampling_params=sp, use_tqdm=False, lora_request=lora_request)


@app.function(**_FN)
def sample_experience(train: list[dict]) -> dict:
    from transformers import AutoTokenizer

    R, system_prompt = _setup()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = _llm()
    t0 = time.time()
    outs = _generate(llm, [_render(tok, system_prompt, t["question"]) for t in train], K)
    rows = []
    for t, out in zip(train, outs):
        for i, o in enumerate(out.outputs):
            correct, executes, reason = R.verdict(o.text, t["sql"])
            rows.append(
                {
                    "task_id": t["id"],
                    "rollout_index": i,
                    "question": t["question"],
                    "text": o.text,
                    "token_ids": list(o.token_ids),
                    "finish_reason": o.finish_reason,
                    "correct": correct,
                    "executes": executes,
                    "reason": reason,
                    "sql": R.extract_sql(o.text),
                }
            )
    d = Path(VOLUME_ROOT, LANE)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "experience.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    runs_volume.commit()
    return {"rows": len(rows), "seconds": round(time.time() - t0, 1)}


@app.function(**_FN)
def validate_hints(items: list[dict]) -> list[dict]:
    """The base, K samples per hinted failure, with and without the hint."""
    from transformers import AutoTokenizer

    R, system_prompt = _setup()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    llm = _llm()
    plain = _generate(llm, [_render(tok, system_prompt, it["question"]) for it in items], K)
    hinted = _generate(
        llm, [_render(tok, system_prompt, with_hint(it["question"], it["hint"])) for it in items], K
    )
    return [
        {
            "key": it["key"],
            "kind": it["kind"],
            "without": [R.verdict(o.text, it["gold"])[0] for o in a.outputs],
            "with": [R.verdict(o.text, it["gold"])[0] for o in b.outputs],
        }
        for it, a, b in zip(items, plain, hinted)
    ]


@app.function(**_FN)
def train_arm(run_name: str, lam: float, hints: list[dict], kl_scope: str = "all") -> dict:
    import math

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    system_prompt = open("/root/prompt.txt", encoding="utf-8").read()
    t_start = time.time()
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    exp = [json.loads(line) for line in open(Path(VOLUME_ROOT, LANE, "experience.jsonl"))]
    by_key = {f"{r['task_id']}#{r['rollout_index']}": r for r in exp}

    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    def completion(r: dict) -> list[int]:
        # the recorded tokens, exactly as sampled (teacher forcing, no re-tokenizing)
        c = list(r["token_ids"])
        if c and c[-1] not in EOS_IDS:
            c.append(IM_END)
        return c

    # Imitate: passing, complete replies (distinct text per task).
    items: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in exp:
        if r["correct"] and r["finish_reason"] == "stop" and (r["task_id"], r["text"]) not in seen:
            seen.add((r["task_id"], r["text"]))
            items.append(
                {
                    "kind": "ce",
                    "sp": ids(_render(tok, system_prompt, r["question"])),
                    "c": completion(r),
                }
            )
    n_ce = len(items)
    # Correct: failed replies with an accepted hint. Everything else is left out.
    for h in hints:
        r = by_key[h["key"]]
        c = completion(r)
        items.append(
            {
                "kind": "kl",
                "sp": ids(_render(tok, system_prompt, r["question"])),
                "tp": ids(_render(tok, system_prompt, with_hint(r["question"], h["hint"]))),
                "c": c,
                "kl_from": kl_start(c, kl_scope),
            }
        )
    n_kl = len(items) - n_ce
    print(f"{run_name}: lambda={lam} scope={kl_scope} imitate {n_ce}, correct {n_kl}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=2 * LORA_RANK,
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
        ),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

    # Identical batches for every lambda: one shuffle per epoch from the seed.
    rng = random.Random(SEED)
    batches: list[list[dict]] = []
    for _ in range(EPOCHS):
        order = items[:]
        rng.shuffle(order)
        batches += [order[i : i + BATCH] for i in range(0, len(order), BATCH)]
    total, warm = len(batches), max(1, int(0.05 * len(batches)))

    def lr_at(step: int) -> float:
        if step < warm:
            return LR * (step + 1) / warm
        return LR * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    def logits_for(prompt: list[int], comp: list[int], grad: bool):
        x = torch.tensor([prompt + comp], device="cuda")
        with torch.set_grad_enabled(grad):
            logits = model(input_ids=x).logits[0]
        return logits[len(prompt) - 1 : len(prompt) - 1 + len(comp)]

    def kl_sum(t_logits, s_logits, chunk: int = 512):
        # forward KL(q || p) over the full vocabulary, summed over positions
        out = 0.0
        for s in range(0, t_logits.shape[0], chunk):
            q_log = F.log_softmax(t_logits[s : s + chunk].float(), -1)
            p_log = F.log_softmax(s_logits[s : s + chunk].float(), -1)
            out = out + (q_log.exp() * (q_log - p_log)).sum()
        return out

    log: list[dict] = []
    model.train()
    for step, b in enumerate(batches):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        denom = sum(len(it["c"]) for it in b if it["kind"] == "ce") or sum(len(it["c"]) for it in b)
        ce_tot = kl_tot = 0.0
        ce_tok = kl_tok = 0
        for it in b:
            if it["kind"] == "ce":
                s = logits_for(it["sp"], it["c"], True)
                loss = F.cross_entropy(
                    s.float(), torch.tensor(it["c"], device="cuda"), reduction="sum"
                )
                ce_tot += loss.item()
                ce_tok += len(it["c"])
                (loss / denom).backward()
            else:
                t = logits_for(it["tp"], it["c"], False).detach()
                # lambda 0 still measures the teacher-student gap, with no gradient
                s = logits_for(it["sp"], it["c"], lam > 0)
                f0 = it["kl_from"]
                kl = kl_sum(t[f0:], s[f0:])
                kl_tot += kl.item()
                kl_tok += len(it["c"]) - f0
                if lam > 0:
                    (lam * kl / denom).backward()
                del t
            del s
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        log.append(
            {
                "step": step + 1,
                "ce_per_token": ce_tot / ce_tok if ce_tok else None,
                "kl_per_token": kl_tot / kl_tok if kl_tok else None,
                "grad_norm": gn,
            }
        )

    out_dir = Path(VOLUME_ROOT, LANE, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "train_log.json").write_text(json.dumps(log))
    cfg = {
        "run_name": run_name,
        "lambda": lam,
        "kl_scope": kl_scope,
        "n_imitate": n_ce,
        "n_correct": n_kl,
        "steps": total,
        "train_seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    runs_volume.commit()
    return cfg


@app.function(**_FN)
def evaluate(tag: str, hold: list[dict], adapter: str = "") -> dict:
    from transformers import AutoTokenizer

    R, system_prompt = _setup()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    t0 = time.time()
    lora_req = None
    if adapter:
        from vllm.lora.request import LoRARequest

        runs_volume.reload()
        lora_req = LoRARequest(tag, 1, str(Path(VOLUME_ROOT, LANE, adapter, "adapter")))
    llm = _llm(lora=bool(adapter))
    outs = _generate(llm, [_render(tok, system_prompt, t["question"]) for t in hold], K, lora_req)
    rows = R.reward_rows(hold, [[o.text for o in out.outputs] for out in outs], tag)
    for row, o in zip(rows, [o for out in outs for o in out.outputs]):
        row["finish_reason"] = o.finish_reason
        row["n_tokens"] = len(o.token_ids)
    return {"tag": tag, "rows": rows, "seconds": round(time.time() - t0, 1)}


@app.function(image=image, timeout=6 * 60 * 60)
def arm(run_name: str, lam: float, kl_scope: str, hints: list[dict], hold: list[dict]) -> dict:
    cfg = train_arm.remote(run_name, lam, hints, kl_scope)
    return {**cfg, **evaluate.remote(run_name, hold, adapter=run_name)}


# ------------------------------------------------------------------ hints (local, Bedrock)

WRITER_TASK = """You annotate a mistake a text-to-SQL model made, so it can learn to avoid it.

The model saw exactly the system prompt quoted above and the question below, and answered with the SQL below.
{feedback}
Reference query (for you only, to locate the mistake; the model never sees it):
{gold}

Question: {question}

Model's SQL:
{sql}

Write a hint the model could have acted on BEFORE answering. Rules:
- At most two sentences, under 60 words. Plain words. No SQL code, no full query, no values from the reference result.
- Every claim must follow from the question, the schema, the notes or rules in the quoted system prompt{err_rule}. Name what the model should have read there.
- If the reference differs only by a choice the question and prompt do not demand (a sort order nobody asked for, extra or fewer columns the question leaves open, a different reasonable reading of an ambiguous question), the mistake is NOT avoidable.

Reply with JSON only: {{"avoidable": true or false, "hint": "..."}}"""

CHECKER_TASK = """A text-to-SQL model saw the system prompt quoted above and this question, and wrote the SQL below.
{feedback}
Question: {question}

Model's SQL:
{sql}

Proposed hint: {hint}

Is the hint fair, meaning every claim in it follows from the question, the schema, the notes or rules in the quoted system prompt{err_rule}, with no fact that only someone who knew the correct answer could state (like a specific expected value, row count, or a requirement the question never makes)?

Reply with JSON only: {{"grounded": true or false, "why": "one short sentence"}}"""


def write_hints(exp: list[dict], gold: dict[str, str]) -> list[dict]:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(read_timeout=120, retries={"max_attempts": 6}),
    )
    # The policy's prompt is quoted, not given as the system prompt: given
    # directly, the writer answers the SQL question itself.
    system = (
        "You review the work of a text-to-SQL model. You never write SQL yourself; you answer in the "
        "JSON the user asks for. This is the system prompt the model was given:\n\n<model_system_prompt>\n"
        + (T2S / "prompt.txt").read_text(encoding="utf-8")
        + "\n</model_system_prompt>"
    )
    lock = threading.Lock()
    usage = {"in": 0, "out": 0, "cache_read": 0}

    def call(model: str, prompt: str, max_tokens: int) -> dict:
        r = client.converse(
            modelId=model,
            system=[{"text": system}, {"cachePoint": {"type": "default"}}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        u = r.get("usage", {})
        with lock:
            usage["in"] += u.get("inputTokens", 0)
            usage["out"] += u.get("outputTokens", 0)
            usage["cache_read"] += u.get("cacheReadInputTokens", 0)
        text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}

    def annotate(r: dict) -> dict:
        kind = failure_kind(r)
        if kind == "tool_error":
            feedback = f"\nRunning it failed with: {r['reason'][len('sql error: ') :]}\n"
            err_rule = ", or the error message"
        else:
            feedback = f"\nIt ran, but the user said the answer is wrong ({r['reason']}).\n"
            err_rule = ""
        out = {
            "key": f"{r['task_id']}#{r['rollout_index']}",
            "task_id": r["task_id"],
            "kind": kind,
            "question": r["question"],
            "sql": r["sql"],
            "reason": r["reason"],
            "accepted": False,
        }
        try:
            w = call(
                WRITER,
                WRITER_TASK.format(
                    feedback=feedback,
                    gold=gold[r["task_id"]],
                    question=r["question"],
                    sql=r["sql"],
                    err_rule=err_rule,
                ),
                300,
            )
        except Exception as e:
            return {**out, "drop": f"writer error {type(e).__name__}"}
        hint = str(w.get("hint") or "").strip()
        out["hint"] = hint
        if not w.get("avoidable") or not hint:
            return {**out, "drop": "not avoidable"}
        if not program_check(hint):
            return {**out, "drop": "program check"}
        try:
            c = call(
                CHECKER,
                CHECKER_TASK.format(
                    feedback=feedback,
                    question=r["question"],
                    sql=r["sql"],
                    hint=hint,
                    err_rule=err_rule,
                ),
                150,
            )
        except Exception as e:
            return {**out, "drop": f"checker error {type(e).__name__}"}
        out["check"] = c.get("why")
        if not c.get("grounded"):
            return {**out, "drop": "not grounded"}
        return {**out, "accepted": True}

    by_task: dict[str, list[dict]] = {}
    for r in exp:
        if failure_kind(r):
            by_task.setdefault(r["task_id"], []).append(r)
    todo = []
    for rs in by_task.values():
        uniq: dict[str, dict] = {}  # distinct queries, tool errors first
        for r in sorted(rs, key=lambda r: (failure_kind(r) != "tool_error", r["rollout_index"])):
            uniq.setdefault(r["sql"], r)
        todo += list(uniq.values())[:PER_TASK]
    random.Random(SEED).shuffle(todo)
    with ThreadPoolExecutor(12) as pool:
        res = list(pool.map(annotate, todo))
    print(f"hints: {sum(h['accepted'] for h in res)} of {len(res)} accepted; tokens {usage}")
    return res


# ------------------------------------------------------------------ selftest


def selftest() -> None:
    tasks = [{"id": f"t2s_{i:04d}"} for i in range(4000)]
    train, hold = split_tasks(tasks)
    assert len(train) == N_TRAIN and len(hold) == N_HOLDOUT
    assert not {t["id"] for t in train} & {t["id"] for t in hold}
    assert with_hint("q?", "look at the notes") == "q?\n\nHint: look at the notes"
    base = {"correct": 0, "finish_reason": "stop", "sql": "SELECT 1", "reason": "result differs"}
    assert failure_kind(base) == "wrong_result"
    assert failure_kind({**base, "reason": "sql error: UndefinedColumn"}) == "tool_error"
    assert failure_kind({**base, "finish_reason": "length"}) is None  # cut off: context only
    assert failure_kind({**base, "correct": 1}) is None
    assert failure_kind({**base, "sql": None}) is None
    assert program_check("Use first_name and last_name; the schema has no fname column.")
    assert not program_check("Write SELECT name FROM customers instead.")
    assert not program_check("```sql\nx\n```")
    assert not program_check(" ".join(["word"] * (HINT_MAX_WORDS + 1)))
    comp = [1, 2, THINK_END, 3, 4, IM_END]
    assert kl_start(comp, "all") == 0 and kl_start(comp, "answer") == 3
    assert kl_start([1, 2, 3], "answer") == 0  # no </think>: whole reply
    m, lo, hi = paired({"a": 0.0, "b": 0.5}, {"a": 0.5, "b": 1.0})
    assert m == 0.5 and lo == 0.5 and hi == 0.5
    print("selftest ok")


# ------------------------------------------------------------------ main


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _pull(path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        for chunk in runs_volume.read_file(path):
            fh.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage",
        choices=["all", "experience", "hints", "validate", "train", "results"],
        default="all",
    )
    ap.add_argument("--arms", default=",".join(ARMS), help=f"comma list from {', '.join(ARMS)}")
    ap.add_argument(
        "--out", default=str(HERE / "out"), help="local folder for experience, hints, eval rows"
    )
    ap.add_argument("--selftest", action="store_true", help="the pure parts, offline")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = _read(T2S / "tasks.jsonl")
    train, hold = split_tasks(tasks)
    gold = {t["id"]: t["sql"] for t in tasks}
    stages = (
        ["experience", "hints", "validate", "train", "results"]
        if args.stage == "all"
        else [args.stage]
    )

    with modal.enable_output(), app.run():
        if "experience" in stages:
            print(sample_experience.remote(train))
            _pull(f"{LANE}/experience.jsonl", out / "experience.jsonl")
        if "hints" in stages:
            res = write_hints(_read(out / "experience.jsonl"), gold)
            (out / "hints.jsonl").write_text(
                "".join(json.dumps(h) + "\n" for h in res), encoding="utf-8"
            )
        accepted = (
            [h for h in _read(out / "hints.jsonl") if h["accepted"]]
            if (out / "hints.jsonl").exists()
            else []
        )
        if "validate" in stages:
            items = [{**h, "gold": gold[h["task_id"]]} for h in accepted]
            val = validate_hints.remote(items)
            (out / "validate.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
        if "train" in stages:
            payload = [{"key": h["key"], "hint": h["hint"]} for h in accepted]
            calls = {"base": evaluate.spawn("base", hold)}
            for name in args.arms.split(","):
                lam, scope = ARMS[name]
                calls[name] = arm.spawn(name, lam, scope, payload, hold)
            for name, call in calls.items():
                r = call.get()
                rows = r.pop("rows")
                (out / f"eval-{name}.jsonl").write_text(
                    "".join(json.dumps(x, default=str) + "\n" for x in rows), encoding="utf-8"
                )
                (out / f"arm-{name}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
                print(name, round(100 * sum(bool(x["reward"]) for x in rows) / len(rows), 1))
    if "results" in stages:
        write_results(out)


def write_results(out: Path) -> None:
    """results.json in the papers contract. baseline = rft (lambda 0), recipe = rft-hint-kl."""
    from importlib.metadata import version

    arms = {}
    rows = {}
    for key, name in (
        ("base", "base"),
        ("baseline", "rft"),
        ("recipe", "rft-hint-kl"),
        ("recipe_answer_only", "rft-hint-kl-answer"),
    ):
        path = out / f"eval-{name}.jsonl"
        if not path.exists():
            continue
        rows[key] = _read(path)
        pt = per_task(rows[key])
        m, lo, hi = paired({k: 0.0 for k in pt}, pt)  # interval of the score itself
        info = (
            json.loads((out / f"arm-{name}.json").read_text())
            if (out / f"arm-{name}.json").exists()
            else {}
        )
        arms[key] = {
            "score": round(m, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "steps": info.get("steps", 0),
            "sql_error": round(
                sum(str(x["reason"]).startswith("sql error") for x in rows[key]) / len(rows[key]), 4
            ),
            "mentions_hint": round(
                sum(bool(re.search(r"\bhint\b", x["final_text"], re.I)) for x in rows[key])
                / len(rows[key]),
                4,
            ),
            "mean_tokens": round(sum(x.get("n_tokens", 0) for x in rows[key]) / len(rows[key]), 1),
        }
    d, lo, hi = paired(per_task(rows["baseline"]), per_task(rows["recipe"]))
    result = {
        "recipe": HERE.name,
        "title": TITLE,
        "paper": PAPER,
        "book": BOOK,
        "base_model": BASE_MODEL,
        "metric": METRIC,
        "n_holdout": N_HOLDOUT,
        "k": K,
        "arms": arms,
        "delta": {
            "recipe_vs_baseline": round(d, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "verdict": "unresolved",
        },
        "verified": date.today().isoformat(),
        "whileai": version("whileai"),
    }
    (out / "results.draft.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        json.dumps(result["delta"]),
        "-> out/results.draft.json; merge the checks block by hand into results.json",
    )


if __name__ == "__main__":
    main()
