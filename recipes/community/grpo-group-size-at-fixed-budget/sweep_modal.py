"""GRPO group-size sweep at a fixed generation budget, on my own Modal.

    modal run sweep_modal.py                 # three arms, G = 2 / 4 / 8
    modal run sweep_modal.py --steps 4       # smoke: does anything move

Every arm spends the same 48 x 16 = 768 rollouts. ``num_generations`` (G)
decides only how those rollouts are grouped: 768/G prompt visits. So the
comparison is group size against prompt coverage at equal GPU cost, not
"more compute is better".

The arm produces completions; every number is computed afterwards, offline,
by whileai. That keeps the measurement out of the GPU container where it
cannot be re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GEN_BATCH = 16  # fixed across arms: the budget knob that must NOT move
STEPS = 48
MAX_NEW = 320
EVAL_K = 4

app = modal.App("wai-seat0-groupsize")
vol = modal.Volume.from_name("wai-seat0-out", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        # Pinned on purpose. Unpinned, pip resolves whileai 0.53 inside this
        # image (57 releases back) and wai.verify does not exist there.
        "whileai==0.110",
    )
    .env({"HF_HOME": "/root/.cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)


def _render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def _sample(model, tokenizer, texts, *, n, max_new_tokens, batch=16, seed=0):
    """``n`` replies per prompt, batched, temperature 0.8 -- the training temperature."""
    import torch

    torch.manual_seed(seed)
    model.eval()
    tokenizer.padding_side = "left"
    out = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(
            gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for i in range(len(chunk)):
            out.append(decoded[i * n : (i + 1) * n])
    model.train()
    return out


@app.function(image=image, gpu="L40S", timeout=60 * 75, volumes={"/out": vol})
def arm(group_size: int, seed: int, holdout: list, train_prompts: list, steps: int) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    import whileai as wai

    print(f"whileai {wai.__version__} | arm G={group_size} seed={seed}", flush=True)
    verifier = wai.verify.MathEqual()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    ho_texts = [_render(tok, r["prompt"]) for r in holdout]

    # ---- base pass. One per arm, on three separate containers: the three
    # together are the noise floor the arms must clear.
    print("base eval...", flush=True)
    base = _sample(model, tok, ho_texts, n=EVAL_K, max_new_tokens=MAX_NEW, seed=1000 + seed)

    # ---- train. Only num_generations differs between arms.
    rendered = [_render(tok, r["prompt"]) for r in train_prompts]
    ds = Dataset.from_dict(
        {"prompt": rendered, "reference": [r["reference"] for r in train_prompts]}
    )

    def reward_fn(prompts, completions, reference, **kw):
        return [float(verifier.check(c, ref, {})[0]) for c, ref in zip(completions, reference)]

    cfg = GRPOConfig(
        output_dir="/out/ckpt",
        max_steps=steps,
        num_generations=group_size,
        per_device_train_batch_size=GEN_BATCH,  # FIXED: the budget is held constant
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        beta=0.04,
        scale_rewards=True,
        max_prompt_length=320,
        max_completion_length=MAX_NEW,
        temperature=0.8,
        bf16=True,
        # OFF on purpose: this trainer generates during training, and
        # checkpointing corrupts Qwen generation on this stack.
        gradient_checkpointing=False,
        logging_steps=4,
        save_strategy="no",
        report_to=[],
        seed=17,  # same for every arm
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
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
    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=ds, reward_funcs=reward_fn, peft_config=lora
    )
    trainer.train()

    # frac_reward_zero_std is the mechanism this sweep is about: a group
    # whose members all pass or all fail has no advantage, so those
    # rollouts bought nothing. Small G should waste more of the budget.
    keep = (
        "step",
        "reward",
        "reward_std",
        "kl",
        "frac_reward_zero_std",
        "completions/mean_length",
        "completions/clipped_ratio",
        "epoch",
    )
    curve = [{k: h[k] for k in keep if k in h} for h in trainer.state.log_history if "reward" in h]

    # ---- after pass, same holdout, same k, same temperature.
    print("trained eval...", flush=True)
    policy = trainer.model
    after = _sample(policy, tok, ho_texts, n=EVAL_K, max_new_tokens=MAX_NEW, seed=2000 + seed)

    payload = {
        "group_size": group_size,
        "seed": seed,
        "steps": steps,
        "gen_batch": GEN_BATCH,
        "rollouts": steps * GEN_BATCH,
        "prompt_visits": steps * GEN_BATCH // group_size,
        "base": base,
        "after": after,
        "curve": curve,
        "whileai": wai.__version__,
    }
    Path(f"/out/arm_g{group_size}.json").write_text(json.dumps(payload))
    vol.commit()
    return payload


@app.local_entrypoint()
def main(steps: int = STEPS, holdout_n: int = 0, only: int = 0):
    holdout = json.loads((HERE / "holdout.json").read_text())
    train_prompts = json.loads((HERE / "train_prompts.json").read_text())
    if holdout_n:
        holdout = holdout[:holdout_n]
    arms = [(2, 0), (4, 1), (8, 2)]
    if only:
        arms = [a for a in arms if a[0] == only]
    jobs = [(g, s, holdout, train_prompts, steps) for g, s in arms]
    for res in arm.starmap(jobs):
        out = HERE / f"arm_g{res['group_size']}.json"
        out.write_text(json.dumps(res))
        print(f"G={res['group_size']} visits={res['prompt_visits']} -> {out.name}")
