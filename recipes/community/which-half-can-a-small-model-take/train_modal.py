"""LoRA SFT on my own Modal: can a 1.5B model write the boring half's replies?

    modal run train_modal.py                 # 3 base passes, LoRA SFT, 1 trained pass
    modal run train_modal.py --gpu A10G --epochs 4

The container only generates text. Grading happens on the laptop with the
same program that graded the incumbent, so the container needs no whileai
and no key, and the two arms are scored by identical code.
"""

from __future__ import annotations

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VOL = "wai-seat1-out"

app = modal.App("wai-seat1-boring-half")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "accelerate==1.10.0",
        "datasets==3.6.0",
    )
    .add_local_file(HERE / "out" / "train_sft.jsonl", "/data/train_sft.jsonl")
    .add_local_file(HERE / "out" / "holdout.json", "/data/holdout.json")
)

vol = modal.Volume.from_name(VOL, create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=60 * 60, volumes={"/vol": vol})
def run(epochs: int = 3, lr: float = 1e-4, max_new_tokens: int = 96, batch: int = 48) -> dict:
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    holdout = json.loads(pathlib.Path("/data/holdout.json").read_text())
    train_rows = [
        json.loads(ln)
        for ln in pathlib.Path("/data/train_sft.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    print(f"holdout {len(holdout)} rows, train {len(train_rows)} rows", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    def generate(model, pass_name: str, seed: int) -> list[dict]:
        """One evaluation pass over the whole holdout."""
        model.eval()
        out: list[dict] = []
        torch.manual_seed(seed)
        for i in range(0, len(holdout), batch):
            chunk = holdout[i : i + batch]
            texts = [
                tok.apply_chat_template(r["chat"], tokenize=False, add_generation_prompt=True)
                for r in chunk
            ]
            enc = tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            for r, g in zip(chunk, gen):
                text = tok.decode(g[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                out.append(
                    {
                        "scenario_id": r["scenario_id"],
                        "rollout_index": r["rollout_index"],
                        "slice": r["slice"],
                        "pass": pass_name,
                        "text": text,
                    }
                )
            print(f"  {pass_name}: {min(i + batch, len(holdout))}/{len(holdout)}", flush=True)
        return out

    results: list[dict] = []

    # Three passes of identical untrained weights: the noise floor.
    model = load_base()
    for p in range(3):
        results += generate(model, f"base{p}", seed=p)
        print(f"base pass {p} done at {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()

    # LoRA SFT on the boring half's clean replies.
    ds = Dataset.from_list([{"messages": r["messages"]} for r in train_rows])
    cfg = SFTConfig(
        output_dir="/vol/adapter",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=False,  # off: it corrupts Qwen generation on this stack
        max_length=1024,
        report_to=[],
        seed=0,
    )
    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=cfg,
        train_dataset=ds,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainer.train()
    train_s = time.time() - t0
    print(f"trained at {train_s:.0f}s", flush=True)

    trained = trainer.model.merge_and_unload()
    trained.eval()
    results += generate(trained, "trained", seed=0)

    payload = {
        "base_model": BASE_MODEL,
        "epochs": epochs,
        "lr": lr,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout),
        "seconds": time.time() - t0,
        "gpu": "L40S",
        "generations": results,
    }
    pathlib.Path("/vol/generations.json").write_text(json.dumps(payload))
    vol.commit()
    return payload


@app.local_entrypoint()
def main(epochs: int = 3, lr: float = 1e-4) -> None:
    payload = run.remote(epochs=epochs, lr=lr)
    out = HERE / "out" / "generations.json"
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({len(payload['generations'])} generations, {payload['seconds']:.0f}s)")
