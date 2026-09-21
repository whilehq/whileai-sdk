"""LoRA fine-tune of Qwen3-4B-Instruct for the identity demo, on Modal.

Trains a rank-16 LoRA adapter (alpha 32, 2 epochs, lr 1e-4, bf16, packing
off) on a jsonl of ``{"messages": [...]}`` rows. The
train file lives on your laptop; ``modal run`` reads it locally and ships
the rows to the container, so nothing is baked into the image.

Usage (from the repo root, no training happens until you run this):

    modal run recipes/04-train/identity/train_modal.py \
        --train-file path/to/train.jsonl \
        --run-name identity-v1

The adapter lands in the Modal volume ``identity-lora`` under
``/<run-name>/adapter`` and checkpoints under ``/<run-name>/checkpoints``.
Re-running with the same ``--run-name`` resumes from the last checkpoint
(``save_steps`` writes one every 50 optimizer steps), so a preempted or
timed-out run costs only the tail, not the whole job.

Cost note: one H100 on Modal is about $3.95/hr. A 4B LoRA over a few
hundred identity rows for 2 epochs finishes in minutes, so a full run is
on the order of $1-2; the 3h timeout is a ceiling, not an estimate. The
app scales to zero (no ``min_containers``), so nothing bills after the
function returns.

Heavy deps (torch, transformers, trl, peft, datasets) live only in the
Modal image; the SDK package itself stays skinny.
"""

from __future__ import annotations

import json
import os
import sys

import modal

from whileai.config import provenance, requirement

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

app = modal.App("identity-lora-train")

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
    .env({"HF_HOME": "/root/.cache/huggingface"})
)

# The dashboard. With WHILEAI_API_KEY set on your laptop the run reports
# loss, learning rate and progress to withwhile.com/platform/training;
# without it, training is unchanged and nothing is sent.
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)

adapter_volume = modal.Volume.from_name("identity-lora", create_if_missing=True)
hf_cache = modal.Volume.from_name("identity-hf-cache", create_if_missing=True)

VOLUME_ROOT = "/vol"


@app.function(
    image=image,
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={VOLUME_ROOT: adapter_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def train(
    rows: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 50,
    max_seq_length: int = 2048,
) -> str:
    """Run the SFT job and return the volume path of the saved adapter."""
    import os

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def to_text(row: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = Dataset.from_list(rows).map(to_text, remove_columns=["messages"])
    print(f"train rows: {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
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

    checkpoint_dir = os.path.join(VOLUME_ROOT, run_name, "checkpoints")
    adapter_dir = os.path.join(VOLUME_ROOT, run_name, "adapter")

    config = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        bf16=True,
        packing=False,
        max_length=max_seq_length,
        dataset_text_field="text",
        logging_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to=[],
        seed=17,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    # One line for the dashboard: loss curve and progress bar on the platform.
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        import whileai.simulations as wai

        run = wai.training_run(
            run_name,
            base_model=base_model,
            trainer="trl-sft-lora",
            config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "rows": len(dataset),
                "gpu": "H100",
            },
        )
        trainer.add_callback(wai.TrainerCallback(run))
        print(f"dashboard: {run.url}")

    has_checkpoint = os.path.isdir(checkpoint_dir) and any(
        name.startswith("checkpoint-") for name in os.listdir(checkpoint_dir)
    )
    try:
        trainer.train(resume_from_checkpoint=has_checkpoint or None)
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise

    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_volume.commit()
    print(f"adapter saved to volume 'identity-lora' at {run_name}/adapter")
    if run is not None and run.status == "running":
        run.finish("done", adapter=f"identity-lora:/{run_name}/adapter")
    return f"{run_name}/adapter"


@app.local_entrypoint()
def main(
    train_file: str,
    run_name: str = "identity-v1",
    base_model: str = BASE_MODEL,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 50,
    max_seq_length: int = 2048,
):
    """Read the local train jsonl and launch the remote job."""
    print(provenance(), file=sys.stderr)
    rows: list[dict] = []
    with open(train_file, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{train_file}:{line_no} has no 'messages' key")
            rows.append({"messages": row["messages"]})
    if not rows:
        raise ValueError(f"{train_file} contained no rows")

    adapter_path = train.remote(
        rows=rows,
        run_name=run_name,
        base_model=base_model,
        epochs=epochs,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        save_steps=save_steps,
        max_seq_length=max_seq_length,
    )
    print(f"done: {adapter_path}")
