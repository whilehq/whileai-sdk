"""Does reward granularity hold at small scale? (arXiv:2607.02869)

Two GRPO arms on one held-out set, on your own Modal:

    process  R = correct steps / total steps   (rule-checked, 1e-5)
    outcome  R = final answer correct          (whileai MathEqual)

Each arm evaluates the untrained base three times for a noise floor, then
trains, then evaluates once more. The arms share held-out questions and
seeds, so the two trained passes are paired question by question.

    modal run repro_modal.py --arm process
    modal run repro_modal.py --arm outcome
"""

from __future__ import annotations

import json
import pathlib

import modal

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GPU = "L40S"

SYSTEM = (
    "Solve the grade-school math problem. Show your work as short numbered "
    "steps, one per line. End with the final numeric answer on its own last "
    "line, in the form: #### <number>"
)

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "accelerate==1.8.1",
        "datasets==3.6.0",
        "whileai==1.7",
    )
    .add_local_file(HERE / "rewards.py", "/root/rewards.py")
    .add_local_file(HERE / "prompts.json", "/root/prompts.json")
)

app = modal.App("wai-seat5-granularity")
vol = modal.Volume.from_name("wai-seat5-out", create_if_missing=True)


def render(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


K = 2  # rollouts per held-out question; holdout_size(0.10, base=0.35, k=2) = 187


def evaluate(model, tokenizer, holdout, seed, batch_size=32, max_new_tokens=256):
    """One sampled pass over the held-out set, K rollouts per question."""
    import torch
    from rewards import outcome_score, process_score

    torch.manual_seed(seed)
    model.eval()
    rows = []
    for start in range(0, len(holdout), batch_size):
        chunk = holdout[start : start + batch_size]
        prompts = [render(tokenizer, q["question"]) for q in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=K,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, seq in enumerate(out):
            q = chunk[j // K]
            text = tokenizer.decode(seq[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            rows.append(
                {
                    "task_id": q["id"],
                    "prompt": q["question"],
                    "final_text": text,
                    "reward": outcome_score(text, q["gold_final"]),
                    "process": process_score(text, q["gold_answer"], q["question"]),
                    "n_chars": len(text),
                }
            )
        print(f"  eval seed={seed} {len(rows)}/{len(holdout) * K}", flush=True)
    return rows


@app.function(image=image, gpu=GPU, timeout=60 * 90, volumes={"/out": vol})
def run_arm(arm: str, seed: int = 0):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from rewards import outcome_score, process_score
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    spec = json.loads(pathlib.Path("/root/prompts.json").read_text())
    holdout, train = spec["holdout"], spec["train"]
    print(f"arm={arm} train={len(train)} holdout={len(holdout)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    # --- the noise floor: the untrained base, three independent passes ---
    base_passes = [evaluate(model, tokenizer, holdout, seed=s) for s in (0, 1, 2)]
    floor = [sum(r["reward"] for r in p) / len(p) for p in base_passes]
    print(f"base pass@1 x3: {[round(f, 4) for f in floor]}", flush=True)

    # --- the one core change: which reward the group is scored with ---
    def reward_process(completions, **kw):
        return [
            process_score(c, a, q)
            for c, a, q in zip(completions, kw["gold_answer"], kw["question"])
        ]

    def reward_outcome(completions, **kw):
        return [outcome_score(c, g) for c, g in zip(completions, kw["gold_final"])]

    reward_fn = {"process": reward_process, "outcome": reward_outcome}[arm]

    dataset = Dataset.from_list(
        [
            {
                "prompt": render(tokenizer, t["question"]),
                "question": t["question"],
                "gold_answer": t["gold_answer"],
                "gold_final": t["gold_final"],
            }
            for t in train
        ]
    )

    cfg = GRPOConfig(
        output_dir="/tmp/grpo",
        num_generations=5,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-5,
        max_prompt_length=320,
        max_completion_length=256,
        temperature=0.9,
        beta=0.04,
        bf16=True,
        gradient_checkpointing=False,  # corrupts generation on this stack
        logging_steps=4,
        save_strategy="no",
        report_to=[],
        seed=seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    history = [h for h in trainer.state.log_history if "reward" in h]

    trained = evaluate(trainer.model, tokenizer, holdout, seed=0)
    print(
        f"arm={arm} trained pass@1 {sum(r['reward'] for r in trained) / len(trained):.4f}",
        flush=True,
    )

    result = {
        "arm": arm,
        "base_model": BASE_MODEL,
        "gpu": GPU,
        "seed": seed,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "base_passes": base_passes,
        "trained_pass": trained,
        "train_log": history,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else {},
    }
    path = f"/out/{arm}.json"
    pathlib.Path(path).write_text(json.dumps(result))
    vol.commit()
    print(f"wrote {path}", flush=True)
    return {"arm": arm, "floor": floor, "trained": trained[0]["task_id"]}


@app.local_entrypoint()
def main(seed: int = 0):
    """Both arms at once, in one app, on one container each.

    One `modal run` means one app name, so `modal app stop` at the end
    stops exactly what this launched and nothing else.
    """
    for out in run_arm.starmap([("process", seed), ("outcome", seed)]):
        print(out, flush=True)
