"""LoRA SFT of Qwen3.8-27B on smithtune-rendered datums, on one Modal H200.

Both SFT arms go through this file with the same settings; only the data
differs. The datums are `export_sft.py`'s output: smithtune's own Baseten
renderer, one datum per assistant turn, history at weight 0 and the turn at
weight 1. The schedule copies smithtune's Baseten defaults
(`CommonSFTSettings`): batch 32, learning rate 1e-4, seed 42, up to 5 epochs,
stop when validation loss does not improve for one epoch, keep the best epoch.
LoRA rank 8 is smithtune's default for the model; alpha 32 is the Tinker
convention Loops follows.

    modal volume put deepagents-review-runs .cache/st/sft-with data/sft-with   # once per arm
    modal run train_modal.py --arm sft-with --smoke        # 4 datums, 1 step: load, loss, save
    modal run --detach train_modal.py --arm sft-with       # the run
    modal run --detach train_modal.py --arm sft-without
    modal run --detach train_modal.py --arm a-with --seed 43   # a replicate: /a-with-s43/adapter
    modal run --detach train_modal.py --arm a-with --seed 43 --epochs 2   # fixed: /a-with-e2-s43/adapter

Adapters land on volume `deepagents-review-runs` under /<arm>/adapter, with
epochs.json beside them.
"""

from __future__ import annotations

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3.8-27B"
REVISION = (
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"  # the tokenizer revision smithtune renders with
)
BUCKET = 2048  # pad lengths to a multiple of this; see loss_of
SETTINGS = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "seed": 42,
    "max_epochs": 5,
    "patience": 1,
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
}

app = modal.App("deepagents-review-sft")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==5.10.4",
        "peft==0.18.1",
        "accelerate==1.15.0",
        "flash-linear-attention==0.4.1",
        "huggingface_hub[hf_transfer]",
    )
    # The fast path for Qwen3.8's linear-attention layers: without causal-conv1d
    # transformers falls back to a torch loop at about 100 tokens a second.
    .pip_install(
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
        "causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    )
    .env(
        {
            "HF_HOME": "/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
runs = modal.Volume.from_name("deepagents-review-runs", create_if_missing=True)
hf = modal.Volume.from_name("deepagents-review-hf", create_if_missing=True)

# Language-model linear layers only: the checkpoint is a vision-language model
# and the vision tower never sees these conversations.
TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$"


@app.function(
    image=image,
    gpu="H200",
    timeout=24 * 60 * 60,
    volumes={"/runs": runs, "/hf": hf},
    secrets=[modal.Secret.from_name("hforg")],
)
def train(
    arm: str, smoke: bool = False, seed: int | None = None, fixed_epochs: int | None = None
) -> dict:
    import math
    import random
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    settings = {**SETTINGS, "seed": SETTINGS["seed"] if seed is None else seed}
    if fixed_epochs is not None:
        # PREREGISTRATION.md amendment 1: a fixed number of epochs, no early
        # stop, keep the last. Early stopping chose epoch 1 or 2 on validation
        # losses 0.001 apart, and the two checkpoints behave differently.
        settings.update(max_epochs=fixed_epochs, patience=None)
    torch.manual_seed(settings["seed"])
    data = pathlib.Path("/runs/data") / arm  # `modal volume put` of export_sft.py's output
    train_rows = _read(data / "train.tokens.jsonl")
    val_rows = _read(data / "validation.tokens.jsonl")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=SETTINGS["lora_rank"],
            lora_alpha=SETTINGS["lora_alpha"],
            lora_dropout=SETTINGS["lora_dropout"],
            target_modules=TARGETS,
            bias="none",
        ),
    )
    model.train()
    model.print_trainable_parameters()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    inner = model.base_model.model  # the ConditionalGeneration module
    lm_head = inner.lm_head

    def loss_of(row: dict) -> torch.Tensor:
        """Mean next-token loss over this datum's weight-1 targets. Logits are
        computed only at target positions: a 27k-token datum's full logits
        over a 248k vocabulary would not fit."""
        # Right-pad to a multiple of BUCKET: the linear-attention kernels
        # recompile for every new length (28 s each, measured), and nearly every
        # datum has its own. Padding sits after the real tokens of a causal
        # model at weight 0, so the loss is unchanged.
        n = len(row["ids"])
        size = -(-n // BUCKET) * BUCKET
        ids = torch.tensor([row["ids"] + [0] * (size - n)], device="cuda")
        w = torch.tensor(row["w"][1:] + [0] * (size - n), device="cuda", dtype=torch.bool)
        hidden = inner.model(input_ids=ids[:, :-1]).last_hidden_state[0]
        logits = lm_head(hidden[w]).float()
        return torch.nn.functional.cross_entropy(logits, ids[0, 1:][w])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=SETTINGS["learning_rate"], weight_decay=0.0)

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            vals = [loss_of(r).item() for r in val_rows]
        model.train()
        return sum(vals) / len(vals)

    if smoke:
        train_rows, val_rows = train_rows[:4], val_rows[:2]
    # A seed other than the default trains a replicate beside the arm: /runs/<arm>-s<seed>.
    name = arm if fixed_epochs is None else f"{arm}-e{fixed_epochs}"
    out = pathlib.Path("/runs") / (name if seed is None else f"{name}-s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(settings["seed"])
    epochs, best, bad = [], math.inf, 0
    first = val_loss()
    print(f"epoch 0 val_loss {first:.4f}", flush=True)
    epochs.append({"epoch": 0, "val_loss": first})
    bs = 4 if smoke else SETTINGS["batch_size"]
    for epoch in range(1, (1 if smoke else settings["max_epochs"]) + 1):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        t_ep, running, seen_tokens = time.time(), [], 0
        for start in range(0, len(order), bs):
            batch = order[start : start + bs]
            opt.zero_grad(set_to_none=True)
            for i in batch:  # accumulate: one datum per forward, batch mean
                loss = loss_of(train_rows[i]) / len(batch)
                loss.backward()
                running.append(loss.item() * len(batch))
            opt.step()
            seen_tokens += sum(len(train_rows[i]["ids"]) for i in batch)
            step = start // bs + 1
            if step % 5 == 0 or smoke:
                print(
                    f"epoch {epoch} step {step} train_loss {sum(running[-bs:]) / len(running[-bs:]):.4f} "
                    f"{time.time() - t_ep:.0f}s {seen_tokens / (time.time() - t_ep):.0f} tok/s "
                    f"mem {torch.cuda.max_memory_allocated() / 2**30:.0f}G",
                    flush=True,
                )
        v = val_loss()
        epochs.append(
            {
                "epoch": epoch,
                "val_loss": v,
                "train_loss": sum(running) / len(running),
                "seconds": round(time.time() - t_ep),
            }
        )
        print(f"epoch {epoch} val_loss {v:.4f}", flush=True)
        if settings["patience"] is None:  # fixed epochs: keep the last
            best = v
            model.save_pretrained(out / "adapter")
        elif v < best:
            best, bad = v, 0
            model.save_pretrained(out / "adapter")
        else:
            bad += 1
        (out / "epochs.json").write_text(
            json.dumps({"settings": settings, "epochs": epochs}, indent=2)
        )
        runs.commit()
        if settings["patience"] is not None and bad >= settings["patience"]:
            break
    return {"arm": out.name, "epochs": epochs, "best_val_loss": best}


@app.function(image=image, gpu="H200", timeout=60 * 60, volumes={"/runs": runs, "/hf": hf})
def profile(arm: str) -> None:
    """Seconds per forward+backward: one datum six times (kernel warm-up), then six new ones."""
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    rows = _read(pathlib.Path("/runs/data") / arm / "train.tokens.jsonl")[:7]
    m = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda"
    )
    print("attention:", m.config._attn_implementation, flush=True)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=8, lora_alpha=32, target_modules=TARGETS))
    m.config.use_cache = False
    m.train()  # checkpointing only applies in training mode
    inner = m.base_model.model

    def step(r):
        ids = torch.tensor([r["ids"]], device="cuda")
        w = torch.tensor(r["w"][1:], device="cuda", dtype=torch.bool)
        h = inner.model(input_ids=ids[:, :-1]).last_hidden_state[0]
        torch.nn.functional.cross_entropy(inner.lm_head(h[w]).float(), ids[0, 1:][w]).backward()

    for i, r in enumerate([rows[0]] * 6 + rows[1:]):
        torch.cuda.synchronize()
        t = time.time()
        step(r)
        torch.cuda.synchronize()
        print(
            f"{'same' if i < 6 else 'new'} {len(r['ids'])} tok {time.time() - t:.1f}s "
            f"{len(r['ids']) / (time.time() - t):.0f} tok/s",
            flush=True,
        )


def _read(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@app.local_entrypoint()
def main(
    arm: str,
    smoke: bool = False,
    prof: bool = False,
    seed: int | None = None,
    epochs: int | None = None,
):
    if prof:
        profile.remote(arm)
        return
    print(json.dumps(train.remote(arm, smoke, seed, epochs), indent=2))
