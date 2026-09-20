"""Lesson 7 step 2, the one the course skips: train on your own Modal.

    modal run recipes/community/the-step-the-course-skips/train_modal.py

Three base passes on the locked test set for the noise floor, a LoRA SFT run
on the rows lesson 6 selected, then one more pass with the adapter loaded.
Both arms go through the same generate-and-score function; the only
difference is whether the adapter is attached, so the paired delta cannot
be measuring two code paths.

One A10G, about twenty minutes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

app = modal.App("wai-seat4-lesson7")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "train.jsonl"), "/root/train.jsonl")
    .add_local_file(str(HERE / "holdout.json"), "/root/holdout.json")
    .add_local_file(str(HERE / "tools.json"), "/root/tools.json")
)

hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("wai-seat4-runs", create_if_missing=True)

ORD = re.compile(r"\bORD-\d{3,6}\b")
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def score_reply(text: str, order_ids: list[str]) -> int:
    """The same rule the local judge applies, against raw model output.

    The SDK grades rows that are already in its schema. A model generating
    in-process emits Qwen's ``<tool_call>`` block, so the text has to be
    parsed back into a call before the rule can run. Both arms use this.
    """
    m = TOOL_CALL.search(text or "")
    if not m:
        return 0
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return 0
    if call.get("name") != "get_order":
        return 0
    args = call.get("arguments")
    if not isinstance(args, dict):
        return 0
    return int(str(args.get("order_id", "")).strip() in set(order_ids))


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/vol": runs},
)
def run(epochs: int = 3, lr: float = 1e-4, samples: int = 4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    cfg = json.loads(Path("/root/tools.json").read_text())
    tasks = json.loads(Path("/root/holdout.json").read_text())
    train_rows = [json.loads(x) for x in Path("/root/train.jsonl").read_text().splitlines()]
    print(f"{len(tasks)} held-out tasks | {len(train_rows)} training rows", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    def evaluate(tag: str, seed: int, net) -> list[dict]:
        """One pass over every held-out task, `samples` tries each."""
        net.eval()
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": t["prompt"]},
                ],
                tools=cfg["tools"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in tasks
        ]
        out_rows = []
        torch.manual_seed(seed)
        for i in range(0, len(prompts), 8):
            chunk = prompts[i : i + 8]
            batch = tok(chunk, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            with torch.no_grad():
                gen = net.generate(
                    **batch,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    num_return_sequences=samples,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            new = gen[:, batch["input_ids"].shape[1] :]
            texts = tok.batch_decode(new, skip_special_tokens=True)
            for j, _t in enumerate(chunk):
                task = tasks[i + j]
                for k in range(samples):
                    txt = texts[j * samples + k]
                    out_rows.append(
                        {
                            "scenario_id": task["scenario_id"],
                            "prompt": task["prompt"],
                            "final_text": txt,
                            "reward": score_reply(txt, task["order_ids"]),
                            "rollout_index": k,
                            "model_version": tag,
                            "seed": seed,
                        }
                    )
        rate = sum(r["reward"] for r in out_rows) / max(len(out_rows), 1)
        print(f"  {tag} seed={seed}: {rate:.3f} over {len(out_rows)} rows", flush=True)
        return out_rows

    # 1. The noise floor, measured before anything is trained.
    base_passes = [evaluate("base", s, model) for s in (101, 202, 303)]

    # 2. Train. TRL is handed pre-rendered text: the exported rows carry the
    # tool schema in a `tools` column that SFTTrainer does not read, so the
    # chat template is applied here rather than left to the trainer.
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    texts = [
        {"text": tok.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=False)}
        for r in train_rows
    ]
    print(
        f"rendered {len(texts)} training texts; first is {len(texts[0]['text'])} chars", flush=True
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list(texts),
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir="/vol/adapter",
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            logging_steps=5,
            report_to=[],
            save_strategy="no",
            bf16=True,
            max_length=2048,
            # off for anything that generates during training (Qwen3 stack)
            gradient_checkpointing=False,
            seed=0,
        ),
    )
    trainer.train()
    trainer.model.save_pretrained("/vol/adapter")
    runs.commit()

    # 3. The same pass again, same tasks, same seed, adapter attached.
    after = evaluate("trained", 101, trainer.model)

    Path("/vol/rows.json").write_text(json.dumps({"base": base_passes, "trained": after}))
    runs.commit()
    return {"base": base_passes, "trained": after}


@app.local_entrypoint()
def main(epochs: int = 3, lr: float = 1e-4, samples: int = 4):
    out = run.remote(epochs=epochs, lr=lr, samples=samples)
    Path(HERE / "rows.json").write_text(json.dumps(out))
    print("wrote rows.json")
