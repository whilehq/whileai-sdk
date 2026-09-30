"""The GPU side of the recipe: sample and grade on Modal, train OPD, train the sequence-KD control.

Three functions on one app (`opd-text-to-sql`), each on one L40S with
Postgres inside the container and the text-to-SQL verifier as the only
grader:

    sample       vLLM samples a model (base or base + adapter) k times per
                 task at the student's token cap, the verifier executes every
                 reply, rows land on the `opd-runs` volume
    train_opd    on-policy distillation: the student samples, the frozen
                 teacher scores every sampled token, the loss is the reverse
                 KL over the teacher's top-k tokens (wai.OPD's knobs), LoRA
    train_seqkd  the offline control: SFT on the teacher's own completions to
                 the same prompts (sequence-level KD), same LoRA, steps, seed

`run.py` in this directory calls them; nothing here is run by hand. Every
knob's default is the value `wai.OPD()` prints (defaults.py names the
source); the trainer is TRL, the method object's own docstring's second
choice, because the pair fits on one GPU there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
T2S = HERE.parent / "text-to-sql"
APP = "opd-text-to-sql"
GPU = os.environ.get("OPD_GPU", "L40S")
VOL = "/vol"

app = modal.App(APP)

# The same pins as text-to-sql/train_grpo_modal.py (torch 2.7.1, TRL 0.19.1,
# vLLM 0.10.0 on the V0 engine), so the image layers are shared with it.
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
    .add_local_file(str(T2S / "sql_verifier.py"), "/root/sql_verifier.py")
    .add_local_file(str(T2S / "schema_prompt.py"), "/root/schema_prompt.py")
    .add_local_file(str(T2S / "schema.sql"), "/root/schema.sql")
    .add_local_file(str(T2S / "seed.sql"), "/root/seed.sql")
    .add_local_file(str(T2S / "prompt.txt"), "/root/prompt.txt")
    .add_local_file(str(T2S / "tasks.jsonl"), "/root/tasks.jsonl")
)

runs = modal.Volume.from_name("opd-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("opd-hf-cache", create_if_missing=True)

_FN = dict(
    gpu=GPU,
    timeout=3 * 60 * 60,
    scaledown_window=600,
    # one input per container: a warm container's Postgres refused a second
    # pg_ctl start and its GPU keeps the last vLLM engine; a fresh container
    # costs about a minute of L40S
    single_use_containers=True,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
)


def _setup():
    """Postgres up, the verifier importable, the tasks and the prompt loaded."""
    sys.path.insert(0, "/root")
    import sql_verifier as R

    os.environ.setdefault("T2S_STATEMENT_TIMEOUT_MS", "3000")
    try:
        # a warm container already has the store up; pg_ctl start would refuse it
        R.run_sql("select 1")
        print("postgres already up")
    except Exception:
        R.start_postgres(open("/root/schema.sql").read(), open("/root/seed.sql").read())
    tasks = R.load_tasks(Path("/root/tasks.jsonl"))
    system_prompt = open("/root/prompt.txt", encoding="utf-8").read()
    return R, tasks, system_prompt


def _messages(system_prompt: str, question: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]


def _merge(base: str, adapter_dir: str, out: str = "/tmp/merged") -> str:
    """Fold a LoRA into the base weights so vLLM serves exactly what trained."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    del model
    torch.cuda.empty_cache()
    return out


@app.function(image=image, **_FN)
def sample(
    model: str,
    name: str,
    *,
    adapter: str = "",
    split: str = "holdout",
    task_ids: list[str] | None = None,
    k: int = 4,
    seed: int = 0,
    max_tokens: int = 512,
    temperature: float = 0.7,
    limit: int = 0,
) -> dict:
    """k replies per task through vLLM, executed and matched by the verifier.

    A reply the cap cut (finish_reason 'length') scores 0 whatever it
    contains (Lambert 2025, chapter Reinforcement Learning: score only
    completions that ended on their own); `raw_correct` keeps what the
    verifier said so the truncation share and its cost are both on the row.
    """
    import json
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    R, tasks, system_prompt = _setup()
    if split != "all":
        tasks = [t for t in tasks if R.split_of(t["id"]) == split]
    if task_ids:
        keep = set(task_ids)
        tasks = [t for t in tasks if t["id"] in keep]
    if limit:
        tasks = tasks[:limit]
    weights = _merge(model, os.path.join(VOL, adapter, "adapter")) if adapter else model
    tok = AutoTokenizer.from_pretrained(model)
    prompts = [
        tok.apply_chat_template(
            _messages(system_prompt, t["question"]), tokenize=False, add_generation_prompt=True
        )
        for t in tasks
    ]
    prompt_tokens = len(tok(prompts[0]).input_ids) if prompts else 0
    llm = LLM(
        model=weights,
        tokenizer=model,
        dtype="bfloat16",
        max_model_len=prompt_tokens + max_tokens + 512,
        gpu_memory_utilization=0.85,
        seed=seed,
        enable_prefix_caching=True,
    )
    params = SamplingParams(
        n=k, temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=seed
    )
    outs = llm.generate(prompts, params)
    rows: list[dict] = []
    for task, out in zip(tasks, outs):
        for i, c in enumerate(out.outputs):
            text = c.text
            correct, executes, reason = R.verdict(text, task["sql"])
            cut = c.finish_reason == "length"
            rows.append(
                {
                    "scenario_id": task["id"],
                    "rollout_index": i,
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": 0 if cut else int(correct),
                    "raw_correct": int(correct),
                    "reason": ("truncated at the token cap; " if cut else "") + reason,
                    "finish_reason": c.finish_reason,
                    "truncated": cut,
                    "tokens": len(c.token_ids),
                    "category": task["archetype"],
                    "difficulty": task["difficulty"],
                    "split": R.split_of(task["id"]),
                    "model_version": name,
                    "agent": R.AGENT,
                    "judge_name": "sql_exec",
                    "judge_status": "ok",
                    "markers": {
                        "executes": int(executes),
                        "has_sql": int(R.extract_sql(text) is not None),
                    },
                    "lineage": {"eval_run": name},
                    "privileged": {"reference": task["sql"]},
                }
            )
    out_dir = Path(VOL) / "rows"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    runs.commit()
    n = len(rows)
    summary = {
        "name": name,
        "model": model,
        "adapter": adapter or None,
        "tasks": len(tasks),
        "rows": n,
        "k": k,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt_tokens": prompt_tokens,
        "pass_rate": sum(r["reward"] for r in rows) / max(n, 1),
        "truncated": sum(r["truncated"] for r in rows) / max(n, 1),
        "mean_tokens": sum(r["tokens"] for r in rows) / max(n, 1),
        "seconds": round(time.time() - t0, 1),
        "gpu": GPU,
    }
    print(json.dumps(summary))
    return {"summary": summary, "rows": rows}


def _lora(rank: int):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
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


def _prompt_set(R, tasks: list[dict], n_prompts: int, prompt_seed: int) -> list[dict]:
    """The same train prompts for every arm: a seeded draw from the train split."""
    import random

    train = [t for t in tasks if R.split_of(t["id"]) == "train"]
    rng = random.Random(prompt_seed)
    rng.shuffle(train)
    return train[:n_prompts]


def _run_page(run_name: str, base_model: str, trainer: str, steps: int, config: dict):
    """The platform's training page, when a key is in the container; else None."""
    if not os.environ.get("WHILEAI_API_KEY"):
        return None
    import whileai.simulations as wai

    run = wai.training_run(
        run_name, base_model=base_model, trainer=trainer, total_steps=steps, config=config
    )
    print(f"training page: {run.url}")
    return run


@app.function(image=image, **_FN)
def train_opd(
    run_name: str,
    *,
    student: str,
    teacher: str,
    seed: int,
    steps: int,
    rows_per_step: int = 8,
    micro_batch: int = 2,
    n_prompts: int = 200,
    samples: int = 4,
    prompt_seed: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 32,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    max_length: int = 2304,
    support: str = "top_k",
) -> dict:
    """On-policy distillation with TRL's GKDTrainer as the loop and wai.OPD's loss.

    ``support`` picks what the reverse KL sums over: ``"top_k"`` is the
    SDK's form (the teacher's top ``top_k`` tokens, docs/reference/methods.md),
    ``"full"`` is the whole vocabulary (Agarwal et al. 2023, Eq. 1 at
    beta = 1, TRL's own ``generalized_jsd_loss``; what prime-rl scores).

    lmbda = 1 (every batch is the student's own sample, Agarwal et al. 2023,
    arXiv:2306.13649, the on-policy end of GKD), beta = 1 (the student on the
    left of the KL, reverse KL), and compute_loss replaced by the SDK's
    top-k form: for each sampled position the reverse KL from the student
    to the teacher over the teacher's top-`top_k` tokens (docs/reference/methods.md,
    OPD). Both log-softmaxes run over the tokenizer's vocabulary
    (len(tokenizer)); the two checkpoints share a tokenizer but pad their
    embedding tables to different sizes, which is why the logits are
    sliced before the divergence and why the sizes are reported.
    """
    import json
    import random
    import time

    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GKDConfig, GKDTrainer

    t0 = time.time()
    R, tasks, system_prompt = _setup()
    tok = AutoTokenizer.from_pretrained(student)
    tok_t = AutoTokenizer.from_pretrained(teacher)
    probe = (
        system_prompt[:2000]
        + "\nSELECT o.id, c.email FROM orders o JOIN customers c ON c.id = o.customer_id;"
    )
    if len(tok) != len(tok_t) or tok(probe).input_ids != tok_t(probe).input_ids:
        raise SystemExit(
            f"tokenizers differ: student {len(tok)} tokens, teacher {len(tok_t)}; OPD scores the "
            "teacher on the student's tokens, so the pair has to share a tokenizer (SimCT, arXiv:2605.07711)"
        )
    vocab = len(tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = _prompt_set(R, tasks, n_prompts, prompt_seed)
    rng = random.Random(seed)
    data = [
        # the assistant turn is a placeholder the on-policy branch always replaces
        # (lmbda = 1); no gold SQL enters this arm at any point
        {
            "messages": [
                *_messages(system_prompt, t["question"]),
                {"role": "assistant", "content": ""},
            ]
        }
        for t in prompts
        for _ in range(samples)
    ]
    rng.shuffle(data)
    dataset = Dataset.from_list(data)

    model = AutoModelForCausalLM.from_pretrained(
        student, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    config = {
        "method": "opd",
        "student": student,
        "teacher": teacher,
        "vocab": {
            "tokenizer": vocab,
            "student_config": model.config.vocab_size,
            "teacher_config": teacher_model.config.vocab_size,
        },
        "divergence": "reverse_kl",
        "support": support,
        "top_k": top_k if support == "top_k" else None,
        "samples": samples,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "steps": steps,
        "rows_per_step": rows_per_step,
        "n_prompts": n_prompts,
        "prompt_seed": prompt_seed,
        "seed": seed,
        "trainer": "trl-0.19.1 GKDTrainer, compute_loss = top-k reverse KL",
        "gpu": GPU,
    }
    out_dir = os.path.join(VOL, run_name)
    os.makedirs(out_dir, exist_ok=True)
    log: list[dict] = []

    class OPDTrainer(GKDTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            student_out = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            with torch.no_grad():
                teacher_out = self.teacher_model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
            p = inputs["prompts"].shape[1]
            s = student_out.logits[:, p - 1 : -1, :vocab].float()
            t = teacher_out.logits[:, p - 1 : -1, :vocab].float()
            labels = inputs["labels"][:, p:]
            mask = labels != -100
            ls = F.log_softmax(s, dim=-1)
            lt = F.log_softmax(t, dim=-1)
            if support == "full":
                # reverse KL over the vocabulary, per token, masked mean
                # (GKDTrainer.generalized_jsd_loss at beta = 1, written out)
                kl = (ls.exp() * (ls - lt)).sum(-1)
            else:
                top = lt.topk(top_k, dim=-1).indices  # the teacher's support
                ls_k = ls.gather(-1, top)
                lt_k = lt.gather(-1, top)
                # reverse KL on the top-k support: sum_v pi_theta(v) (log pi_theta(v) - log pi_T(v))
                kl = (ls_k.exp() * (ls_k - lt_k)).sum(-1)
            loss = (kl * mask).sum() / mask.sum().clamp(min=1)
            with torch.no_grad():
                safe = labels.clamp(min=0)
                gap = (
                    lt.gather(-1, safe.unsqueeze(-1)) - ls.gather(-1, safe.unsqueeze(-1))
                ).squeeze(-1)
                n_tok = int(mask.sum().item())
                ended = int(((labels == tok.eos_token_id) & mask).any(-1).sum().item())
                log.append(
                    {
                        "kl": float(loss.item()),
                        "gap": float((gap * mask).sum().item() / max(n_tok, 1)),
                        "tokens": n_tok / labels.shape[0],
                        "ended": ended / labels.shape[0],
                        "t": round(time.time() - t0, 1),
                    }
                )
            del ls, lt, s, t
            return (loss, student_out) if return_outputs else loss

    args = GKDConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=rows_per_step // micro_batch,
        lmbda=1.0,
        beta=1.0,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=min(5, steps),
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
        disable_dropout=True,
        # the ChatML collator tokenizes the raw messages itself
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = OPDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=_lora(lora_rank),
    )
    run = _run_page(run_name, student, "trl-gkd-opd-lora-sql", steps, config)
    if run is not None:
        import whileai.simulations as wai

        trainer.add_callback(wai.TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise
    trainer.model.save_pretrained(os.path.join(out_dir, "adapter"))
    tok.save_pretrained(os.path.join(out_dir, "adapter"))
    steps_log = [{k: e[k] for k in e} for e in log]
    per_step = []
    g = rows_per_step // micro_batch
    for i in range(0, len(steps_log), g):
        chunk = steps_log[i : i + g]
        per_step.append(
            {
                "step": i // g + 1,
                "kl": sum(c["kl"] for c in chunk) / len(chunk),
                "gap": sum(c["gap"] for c in chunk) / len(chunk),
                "tokens": sum(c["tokens"] for c in chunk) / len(chunk),
                "ended": sum(c["ended"] for c in chunk) / len(chunk),
            }
        )
    summary = {
        "run_name": run_name,
        "config": config,
        "seconds": round(time.time() - t0, 1),
        "steps": len(per_step),
        "curve": per_step,
        "adapter": f"volume opd-runs:/{run_name}/adapter",
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    runs.commit()
    if run is not None:
        run.finish(
            "done",
            summary={
                "seconds": summary["seconds"],
                "final_kl": per_step[-1]["kl"] if per_step else None,
            },
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))
    return summary


@app.function(image=image, **_FN)
def train_seqkd(
    run_name: str,
    *,
    student: str,
    teacher_rows: str,
    seed: int,
    steps: int,
    rows_per_step: int = 8,
    micro_batch: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    max_length: int = 2304,
) -> dict:
    """Sequence-level KD: SFT on the teacher's completions, the offline control.

    The rows are the teacher's own samples on the same prompts OPD trains on
    (`sample(..., split="train", task_ids=...)`), taken as they are: no
    verifier filter, because sequence KD imitates the teacher, not the gold
    (Kim and Rush 2016; Lambert 2025, chapter Synthetic Data and
    Distillation, the offline regime). A completion the cap cut is dropped
    rather than imitated. Loss on the completion tokens only.
    """
    import json
    import random
    import time

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    _R, _tasks, system_prompt = _setup()
    rows = [
        json.loads(line)
        for line in open(os.path.join(VOL, "rows", teacher_rows + ".jsonl"), encoding="utf-8")
        if line.strip()
    ]
    kept = [r for r in rows if not r.get("truncated")]
    rng = random.Random(seed)
    data = [
        {
            "prompt": _messages(system_prompt, r["prompt"]),
            "completion": [{"role": "assistant", "content": r["final_text"]}],
        }
        for r in kept
    ]
    rng.shuffle(data)
    dataset = Dataset.from_list(data)
    tok = AutoTokenizer.from_pretrained(student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        student, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    config = {
        "method": "seqkd",
        "student": student,
        "teacher_rows": teacher_rows,
        "rows": len(kept),
        "rows_dropped_truncated": len(rows) - len(kept),
        "teacher_pass_rate_on_rows": sum(r["reward"] for r in kept) / max(len(kept), 1),
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "steps": steps,
        "rows_per_step": rows_per_step,
        "seed": seed,
        "trainer": "trl-0.19.1 SFTTrainer, completion_only_loss",
        "gpu": GPU,
    }
    out_dir = os.path.join(VOL, run_name)
    os.makedirs(out_dir, exist_ok=True)
    args = SFTConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=rows_per_step // micro_batch,
        max_length=max_length,
        completion_only_loss=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=min(5, steps),
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=_lora(lora_rank),
    )
    run = _run_page(run_name, student, "trl-sft-seqkd-lora-sql", steps, config)
    if run is not None:
        import whileai.simulations as wai

        trainer.add_callback(wai.TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise
    trainer.model.save_pretrained(os.path.join(out_dir, "adapter"))
    tok.save_pretrained(os.path.join(out_dir, "adapter"))
    curve = [
        {"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h
    ]
    summary = {
        "run_name": run_name,
        "config": config,
        "seconds": round(time.time() - t0, 1),
        "steps": len(curve),
        "curve": curve,
        "adapter": f"volume opd-runs:/{run_name}/adapter",
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    runs.commit()
    if run is not None:
        run.finish(
            "done",
            summary={
                "seconds": summary["seconds"],
                "final_loss": curve[-1]["loss"] if curve else None,
            },
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))
    return summary
