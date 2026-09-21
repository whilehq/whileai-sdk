"""GRPO on Modal with the SQL execution verifier as the reward.

    PYTHONUTF8=1 modal run --detach train_grpo_modal.py --spawn --thinking --skip-eval --steps 100 --run-name t2s-r1

What happens:
1. Postgres is installed in the image; at run time a trust-auth cluster is
   initialised in the container and schema.sql + seed.sql are loaded.
2. Train prompts = the train split of tasks.jsonl (question + gold SQL), each
   rendered with the exact benchmark system prompt (prompt.txt) and Qwen3
   thinking off. Reward = execute the completion, 1.0 on a result match,
   0.1 when it runs but is wrong, 0 otherwise.
3. The holdout split is sampled 4x before and after training and graded with
   the binary verdict; pass@1 before/after and the delta land on the run
   page. The adapter is saved on volume whileai-train-runs under the run
   id so wai.serve can host it on Qwen/Qwen3-4B.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_GPU = os.environ.get("ZP_GRPO_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-t2s-grpo")

_base = (
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
)
# Generation through vLLM instead of HF generate: 5-10x faster rollouts.
# vllm 0.10.0 pins torch 2.7.1 (the same pin) and still ships the V0
# engine, which TRL 0.19.1's colocate weight sync assumes (VLLM_USE_V1=0).
_vllm_base = _base.pip_install("vllm==0.10.0").env({"VLLM_USE_V1": "0"})


def _with_files(img: modal.Image) -> modal.Image:
    # local files last, so an edit does not rebuild the image
    return (
        img.add_local_file(str(HERE / "sql_verifier.py"), "/root/sql_verifier.py")
        .add_local_file(str(HERE / "schema_prompt.py"), "/root/schema_prompt.py")
        .add_local_file(str(HERE / "schema.sql"), "/root/schema.sql")
        .add_local_file(str(HERE / "seed.sql"), "/root/seed.sql")
        .add_local_file(str(HERE / "prompt.txt"), "/root/prompt.txt")
    )


image = _with_files(_base)
image_vllm = _with_files(_vllm_base)
# --stack new: the current releases, for checkpoints the 2025 pins cannot load
# (Qwen3.5-* are model_type qwen3_5, Qwen3_5ForConditionalGeneration). Same
# script, same knobs; TRL >= 1.0 has its own chunked log-prob pass, so the
# 0.19 monkeypatch below is skipped there.
_base_new = (
    # vLLM >= 0.26 compiles kernels at start and needs nvcc: CUDA devel base.
    # CUDA 13 to match the torch that vLLM 0.29 pins; causal-conv1d compiles
    # against it and refuses a 12.x toolkit.
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12")
    # apt on the ubuntu base stops at tzdata's "Geographic area:" prompt without this
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("postgresql", "postgresql-contrib")
    .pip_install(
        "vllm==0.29.0",
        "transformers==5.17.0",
        "trl==1.13.0",
        # Qwen3.5's gated delta net: without this transformers falls back to the
        # reference kernels, which took 138 GB for one micro-batch of 2 on an H200.
        # causal-conv1d is a CUDA extension that does not build in the image; its
        # reference fallback is slow but not the memory problem.
        "flash-linear-attention",
        "peft==0.21.0",
        "datasets>=3.6.0",
        "accelerate>=1.8.1",
        "psycopg[binary]==3.2.9",
        # tracked.open(run_id) + TrainerCallback: the platform draws the run
        "whileai>=0.110",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
image_vllm_new = _with_files(_base_new)

runs_volume = modal.Volume.from_name("whileai-train-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)


dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _render(tokenizer, system_prompt: str, question: str, thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def _sample(
    model,
    tokenizer,
    texts: list[str],
    *,
    n: int,
    max_new_tokens: int,
    batch: int = 8,
    temperature: float = 0.7,
) -> list[list[str]]:
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    out: list[list[str]] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
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


def _guard_vllm_weight_sync(trainer) -> None:
    """Colocate weight sync for VLM-style checkpoints (Qwen3.5-*).

    TRL hands vLLM the Hugging Face parameter names one at a time; vLLM maps
    them with the model's ``hf_to_vllm_mapper`` and raises on any name it
    cannot place. On Qwen3_5ForConditionalGeneration that killed the run at
    step 0 ("no module or parameter named 'model'"). The LoRA only touches
    the language layers, so a name vLLM cannot place is a tensor that never
    changed: skip it, say which, keep going.
    """
    holder = None
    for v in vars(trainer).values():
        if hasattr(v, "llm") and hasattr(v.llm, "llm_engine"):
            holder = v
            break
    if holder is None:
        print("weight-sync guard: no colocated vLLM found on the trainer; not installed")
        return
    vmodel = holder.llm.llm_engine.model_executor.driver_worker.model_runner.model
    have = {n for n, _ in vmodel.named_parameters()}
    orig = vmodel.load_weights
    skipped: list[str] = []

    def _candidates(name: str) -> list[str]:
        out = [name]
        # transformers 5 exposes the text stack as "model.layers..." on the
        # conditional-generation class; vLLM keeps it under language_model.
        if name.startswith("model.") and not name.startswith(
            ("model.language_model.", "model.visual.")
        ):
            out.append("model.language_model." + name[len("model.") :])
        return out

    def load_weights(weights):
        loaded: set[str] = set()
        for name, tensor in weights:
            ok = False
            for cand in _candidates(name):
                # vLLM's loader also folds stacked projections (gate/up,
                # q/k/v) into their fused parameter, so let it decide.
                try:
                    loaded |= set(orig([(cand, tensor)]) or ())
                    ok = True
                    break
                except (ValueError, KeyError):
                    continue
            if not ok:
                if len(skipped) < 8:
                    print(f"weight-sync guard: skipping {name!r}")
                skipped.append(name)
        return loaded

    vmodel.load_weights = load_weights
    print(f"weight-sync guard installed on {type(vmodel).__name__} ({len(have)} params)")


def _train(
    train_tasks: list[dict],
    holdout_tasks: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 120,
    num_generations: int = 8,
    learning_rate: float = 1e-5,
    beta: float = 0.04,
    max_completion_length: int = 256,
    lora_rank: int = 16,
    eval_samples: int = 4,
    loss_type: str = "bnpo",
    gpu: str = DEFAULT_GPU,
    grad_ckpt: bool = False,
    accum: int = 2,
    thinking: bool = False,
    skip_eval: bool = False,
    from_run: str = "",
    use_vllm: bool = False,
    steps_per_generation: int = 0,
    system_prefix: str = "",
    prompts_per_step: int = 0,
    micro_batch: int = 0,
    mask_truncated: bool = True,
    save_every: int = 0,
    vllm_mem: float = 0.25,
    platform_run: str = "",
) -> dict:
    import json

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    import sql_verifier as R

    import whileai.simulations as wai

    if use_vllm:
        # TRL's colocate mode builds vLLM with the external_launcher executor,
        # which reads the torchrun rank variables; one process, one GPU.
        for k, v in {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29511",
        }.items():
            os.environ.setdefault(k, v)
    os.environ.setdefault("T2S_STATEMENT_TIMEOUT_MS", "2000")  # a training candidate gets 2 s
    R.start_postgres(open("/root/schema.sql").read(), open("/root/seed.sql").read())
    system_prompt = open("/root/prompt.txt", encoding="utf-8").read()
    if system_prefix:
        # e.g. Llama-Nemotron: 'detailed thinking on' as the first line of the system prompt
        system_prompt = system_prefix.strip() + "\n\n" + system_prompt

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="cuda"
        )
    except ValueError:
        # qwen3_5-style checkpoints register only as image-text-to-text; the
        # text stack trains the same way and the vision tower is never used.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        print(f"loaded {base_model} as {type(model).__name__}")
    if from_run:
        # continue from a previous round's LoRA adapter (same base)
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            # "<run_id>" is that run's final adapter; "<run>/checkpoints/checkpoint-25"
            # is a directory written by --save-every (a run cut short resumes here).
            model,
            os.path.join(VOLUME_ROOT, from_run)
            if "/" in from_run
            else os.path.join(VOLUME_ROOT, from_run, "adapter"),
            is_trainable=True,
        )
        print(f"resumed adapter from {from_run}")

    config = {
        "base_model": base_model,
        "steps": steps,
        "num_generations": num_generations,
        "learning_rate": learning_rate,
        "beta": beta,
        "loss_type": loss_type,
        "max_completion_length": max_completion_length,
        "lora_rank": lora_rank,
        "train_prompts": len(train_tasks),
        "holdout_prompts": len(holdout_tasks),
        "gpu": gpu,
        "reward": "sql_verifier.py: execute on the seeded store Postgres, 1.0 result match / 0.1 runs but wrong / 0",
        "thinking": thinking,
        "from_run": from_run or None,
        "use_vllm": use_vllm,
        "steps_per_generation": steps_per_generation,
        "system_prefix": system_prefix or None,
        "eval": "hosted (served adapter, rollout.py --hosted <name>)"
        if skip_eval
        else "in-container",
    }
    run = None
    prun = None  # the platform run this training fills (tracked.run(...) opened on the laptop)
    if platform_run and os.environ.get("WHILEAI_API_KEY"):
        from whileai.platform import track

        prun = track(R.AGENT, register=False).open(platform_run)
        print(f"platform run {platform_run}: every trainer log lands on the page as it happens")
    elif os.environ.get("WHILEAI_API_KEY") and hasattr(wai, "training_run"):
        run = wai.training_run(
            run_name,
            base_model=base_model,
            trainer="trl-grpo-lora-sql",
            total_steps=steps,
            config=config,
        )
        print(f"dashboard: {run.url}")
    run_id = getattr(run, "run_id", None) or run_name
    out_dir = os.path.join(VOLUME_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    hold_texts = [_render(tokenizer, system_prompt, t["question"], thinking) for t in holdout_tasks]
    before_rows: list[dict] = []
    if not skip_eval:
        before_replies = _sample(
            model, tokenizer, hold_texts, n=eval_samples, max_new_tokens=max_completion_length
        )
        before_rows = R.reward_rows(holdout_tasks, before_replies, f"{base_model}@before")
        before = wai.pass_at(before_rows)
        print(f"before: {before}")

    calls = {"n": 0}

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=16)

    def sql_reward(completions, gold, **kwargs):
        # 64-128 candidate queries per generation call; scored in parallel
        # (thread-local Postgres connections), because a policy that explores
        # heavy joins hits the statement timeout often enough that a serial
        # loop turned a 10 s step into minutes.
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        out = list(pool.map(R.shaped_reward, texts, gold))
        calls["n"] += 1
        if calls["n"] <= 3:
            sample = (
                completions[0][0]["content"]
                if isinstance(completions[0], list)
                else str(completions[0])
            )
            print(
                f"[reward call {calls['n']}] type={type(completions[0]).__name__} rewards={out[:8]} sample={sample[:400]!r}"
            )
        return out

    dataset = Dataset.from_list(
        [
            {"prompt": _render(tokenizer, system_prompt, t["question"], thinking), "gold": t["sql"]}
            for t in train_tasks
        ]
    )
    import trl as _trl

    grpo_kwargs = dict(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=num_generations,
        # 8 generations x ~1,350 tokens OOMs the L40S in one micro-batch
        # (smoke run t2s-grpo-smoke3); two micro-batches of 4 fit.
        # prompts_per_step x num_generations samples per optimizer step (issue
        # whilehq/whileai-sdk#252: one prompt per step is a random walk); the
        # legacy path is num_generations // accum per micro-batch, accum of them.
        per_device_train_batch_size=(micro_batch or num_generations // accum),
        gradient_accumulation_steps=(
            (prompts_per_step * num_generations) // (micro_batch or num_generations // accum)
            if prompts_per_step
            else accum
        ),
        # A length-truncated completion gives no gradient instead of reward 0
        # (issue #253: reward 0 on truncation teaches shorter thinking first).
        mask_truncated_completions=mask_truncated,
        # Gradient checkpointing makes TRL generate without a KV cache, and on
        # Qwen3-4B (transformers 4.54) that path produced garbage completions
        # from the first token (smoke run t2s-grpo-smoke2: every completion
        # 256 random tokens, reward 0). Off by default; L40S has the memory.
        gradient_checkpointing=grad_ckpt,
        learning_rate=learning_rate,
        beta=beta,
        loss_type=loss_type,
        max_completion_length=max_completion_length,
        max_prompt_length=2048,
        temperature=0.9,
        # vLLM colocate: the policy's weights are pushed into a vLLM engine on
        # the same GPU before each generation; steps_per_generation batches
        # several prompts into one generate call, then takes that many steps.
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=vllm_mem,
        # 0 = TRL's default (= gradient_accumulation_steps), so one generate call
        # covers exactly num_generations samples. TRL requires per_device x
        # steps_per_generation to be a multiple of num_generations; 1 with
        # micro-batches of 2 fails that check before the first step.
        steps_per_generation=steps_per_generation or None,
        bf16=True,
        logging_steps=1,
        save_strategy="steps" if save_every else "no",
        save_steps=save_every or 500,
        save_only_model=True,
        report_to=[],
        seed=17,
    )
    import inspect as _inspect

    _accepted = set(_inspect.signature(GRPOConfig).parameters)
    _dropped = sorted(k for k in grpo_kwargs if k not in _accepted)
    if _dropped:
        print(f"GRPOConfig ({_trl.__version__}) does not take {_dropped}; dropped")
    grpo = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in _accepted})
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
    if use_vllm and _trl.__version__.startswith("0."):
        # TRL 0.19.1 scores the old/reference log-probs over the whole
        # generation batch in one forward (32 sequences x 3k tokens x 152k
        # vocab = 28 GB of logits); chunk it at the micro-batch size.
        _orig_logps = GRPOTrainer._get_per_token_logps

        def _chunked(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
            return _orig_logps(
                self,
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size or max(1, num_generations // accum),
            )

        GRPOTrainer._get_per_token_logps = _chunked
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[sql_reward],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None if from_run else lora,
    )
    if prun is not None:
        from whileai.platform import TrainerCallback

        # reward, kl, loss and completion length per step; the run is scored
        # and finished from the laptop once the checkpoints are measured
        trainer.add_callback(TrainerCallback(prun, finish=False))
    elif run is not None:
        trainer.add_callback(wai.TrainerCallback(run, finish=False))
    if use_vllm and not _trl.__version__.startswith("0."):
        _guard_vllm_weight_sync(trainer)
    try:
        trainer.train()
    except Exception as exc:
        if prun is not None:
            prun.fail(f"{type(exc).__name__}: {exc}")
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise
    if prun is not None:
        prun.flush()

    policy = trainer.model
    after_rows: list[dict] = []
    if not skip_eval:
        after_replies = _sample(
            policy, tokenizer, hold_texts, n=eval_samples, max_new_tokens=max_completion_length
        )
        after_rows = R.reward_rows(holdout_tasks, after_replies, f"{run_name}@after")
        after = wai.pass_at(after_rows)
        print(f"after:  {after}")

    adapter_dir = os.path.join(out_dir, "adapter")
    policy.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(out_dir, "holdout_before.jsonl"), "w") as fh:
        for r in before_rows:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(os.path.join(out_dir, "holdout_after.jsonl"), "w") as fh:
        for r in after_rows:
            fh.write(json.dumps(r, default=str) + "\n")
    runs_volume.commit()

    summary = {
        "pass_at_1_before": before.pass_at_1 if before_rows else None,
        "pass_at_1_after": after.pass_at_1 if after_rows else None,
        "ci95_before": before.ci95 if before_rows else None,
        "ci95_after": after.ci95 if after_rows else None,
        "holdout_prompts": len(holdout_tasks),
        "eval_samples": eval_samples,
        "run_id": run_id,
        "adapter": f"volume whileai-train-runs:/{run_id}/adapter",
    }
    delta = None
    if before_rows and after_rows:
        if run is not None:
            delta = run.delta(
                before_rows,
                after_rows,
                target="pass_at_1",
                must_not_regress=["executes"],
                by="difficulty",
            )
        else:
            delta = wai.delta_report(
                before_rows,
                after_rows,
                target="pass_at_1",
                must_not_regress=["executes"],
                by="difficulty",
            )
        print(wai.format_delta_report(delta))
        summary["delta_verdict"] = delta["target_verdict"]
    if run is not None:
        run.finish("done", summary=summary, adapter=f"volume whileai-train-runs:/{run_id}/adapter")
        summary["run_url"] = run.url
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    runs_volume.commit()
    return summary


_FN = dict(
    gpu=DEFAULT_GPU,
    # 32 prompts x 16 samples a step runs ~8 min a step; a 4 h cap killed round 5 at step 28.
    timeout=24 * 60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)


@app.function(image=image, **_FN)
def train(**kwargs) -> dict:
    return _train(**kwargs)


@app.function(image=image_vllm, **_FN)
def train_vllm(**kwargs) -> dict:
    return _train(use_vllm=True, **kwargs)


@app.function(image=image_vllm_new, **_FN)
def train_vllm_new(**kwargs) -> dict:
    return _train(use_vllm=True, **kwargs)


@app.local_entrypoint()
def main(
    run_name: str = "text-to-sql-shop-grpo-v1",
    steps: int = 120,
    num_generations: int = 8,
    learning_rate: float = 1e-5,
    beta: float = 0.04,
    base_model: str = BASE_MODEL,
    max_completion_length: int = 256,
    loss_type: str = "bnpo",
    gpu: str = DEFAULT_GPU,
    limit: int = 0,
    grad_ckpt: bool = False,
    accum: int = 2,
    thinking: bool = False,
    skip_eval: bool = False,
    from_run: str = "",
    spawn: bool = False,
    use_vllm: bool = False,
    steps_per_generation: int = 0,
    lora_rank: int = 16,
    system_prefix: str = "",
    prompts_per_step: int = 0,
    micro_batch: int = 0,
    mask_truncated: bool = True,
    save_every: int = 0,
    task_ids: str = "",
    vllm_mem: float = 0.25,
    stack: str = "pinned",
    platform_run: str = "",
):
    import hashlib
    import json

    tasks = [
        json.loads(line) for line in (HERE / "tasks.jsonl").open(encoding="utf-8") if line.strip()
    ]

    def bucket(sid: str) -> float:
        return int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    train_tasks = [t for t in tasks if bucket(t["id"]) >= 0.2]
    holdout_tasks = [t for t in tasks if bucket(t["id"]) < 0.2]
    if task_ids:
        # A band-filtered prompt set from the previous round's rollouts
        # (build.py --band writes out/band_ids.txt; issue #254).
        keep = {
            line.strip()
            for line in Path(task_ids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        train_tasks = [t for t in train_tasks if t["id"] in keep]
        print(f"task_ids: {len(train_tasks)} train prompts kept from {task_ids}")
    if limit:
        train_tasks, holdout_tasks = train_tasks[:limit], holdout_tasks[: max(4, limit // 4)]
    print(f"{len(tasks)} tasks: {len(train_tasks)} train, {len(holdout_tasks)} holdout")
    # the new stack only has the vLLM path
    base_fn = train_vllm_new if stack == "new" else (train_vllm if use_vllm else train)
    fn = base_fn if gpu == DEFAULT_GPU else base_fn.with_options(gpu=gpu)
    kwargs = dict(
        train_tasks=train_tasks,
        holdout_tasks=holdout_tasks,
        run_name=run_name,
        base_model=base_model,
        steps=steps,
        num_generations=num_generations,
        learning_rate=learning_rate,
        beta=beta,
        max_completion_length=max_completion_length,
        loss_type=loss_type,
        gpu=gpu,
        grad_ckpt=grad_ckpt,
        accum=accum,
        thinking=thinking,
        skip_eval=skip_eval,
        from_run=from_run,
        steps_per_generation=steps_per_generation,
        lora_rank=lora_rank,
        system_prefix=system_prefix,
        prompts_per_step=prompts_per_step,
        micro_batch=micro_batch,
        mask_truncated=mask_truncated,
        save_every=save_every,
        vllm_mem=vllm_mem,
        # --platform-run run_...: the run opened with tracked.run(...) on the
        # laptop; the trainer's callback fills its curve (needs WHILEAI_API_KEY)
        platform_run=platform_run,
    )
    if spawn:
        # Submit and return. With `modal run --detach` the call keeps running
        # on Modal with no client attached: a laptop network drop cancelled a
        # 3 h `.remote()` call at step 49 even under --detach. Progress is on
        # the training page; the summary lands on the volume as summary.json.
        call = fn.spawn(**kwargs)
        print(f"spawned {call.object_id}; the run continues on Modal, watch the training page")
        return
    summary = fn.remote(**kwargs)
    print("done:", summary)
