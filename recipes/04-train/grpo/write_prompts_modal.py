"""Write the model-written prompt set on a GPU and save it next to this file.

    modal run recipes/04-train/grpo/write_prompts_modal.py                    # ~500 prompts, A10G, a few minutes
    modal run recipes/04-train/grpo/write_prompts_modal.py --seeds 400 --per-seed 6 --out recipes/04-train/grpo/prompts.jsonl

Seeds are the offline template writer's prompts (``reward.build_prompts``),
one per distinct situation. Qwen2.5-7B-Instruct writes ``per_seed``
messages per seed from two angles; ``prompts.keep`` drops anything that
left its category or near-copies an earlier keep. The file is JSONL with
``prompt``, ``scenario_id`` (the seed's, so the split stays by situation)
and ``seed``. Both train scripts read it with ``--prompts-file``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WRITER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

app = modal.App("whileai-grpo-prompts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "transformers==4.54.0", "accelerate==1.8.1", requirement())
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "reward.py"), "/root/reward.py")
    .add_local_file(str(HERE / "prompts.py"), "/root/prompts.py")
    .add_local_python_source("whileai")
)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
# The set is also written to the runs volume, so a dropped client connection
# loses nothing: `modal volume get whileai-grpo-runs prompts/prompts.jsonl`.
runs_volume = modal.Volume.from_name("whileai-grpo-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu=os.environ.get("ZP_WRITER_GPU", "A10G"),
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/vol": runs_volume},
)
def write(
    seeds: list[dict], per_seed: int = 6, writer_model: str = WRITER_MODEL, batch: int = 8
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    from prompts import ANGLES, keep, parse_messages, writer_messages

    tok = AutoTokenizer.from_pretrained(writer_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        writer_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    jobs = [(seed, angle) for seed in seeds for angle in range(len(ANGLES))]
    texts = [
        tok.apply_chat_template(
            writer_messages(seed, angle, per_seed), tokenize=False, add_generation_prompt=True
        )
        for seed, angle in jobs
    ]
    candidates: list[tuple[str, dict]] = []
    for start in range(0, len(texts), batch):
        enc = tok(texts[start : start + batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=700,
                pad_token_id=tok.pad_token_id,
            )
        decoded = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for (seed, _angle), reply in zip(jobs[start : start + batch], decoded):
            for msg in parse_messages(reply):
                candidates.append((msg, seed))
        print(f"{min(start + batch, len(texts))}/{len(texts)} calls, {len(candidates)} candidates")
    kept = keep(candidates)
    print(f"kept {len(kept)} of {len(candidates)}")
    import json
    from pathlib import Path as _P

    _P("/vol/prompts").mkdir(parents=True, exist_ok=True)
    with open("/vol/prompts/prompts.jsonl", "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(
                json.dumps(
                    {
                        "prompt": row["prompt"],
                        "scenario_id": row["scenario_id"],
                        "seed": row["seed"],
                    }
                )
                + "\n"
            )
    runs_volume.commit()
    return kept


@app.local_entrypoint()
def main(
    seeds: int = 400,
    per_seed: int = 6,
    out: str = "",
    writer_model: str = WRITER_MODEL,
    seed: int = 0,
):
    print(provenance(), file=sys.stderr)
    import json

    from prompts import summary
    from reward import build_prompts

    seed_items = build_prompts(seeds, seed=seed)
    print(f"{len(seed_items)} seed situations from {seeds} template situations")
    kept = write.remote(seed_items, per_seed=per_seed, writer_model=writer_model)
    path = Path(out) if out else HERE / "prompts.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(
                json.dumps(
                    {
                        "prompt": row["prompt"],
                        "scenario_id": row["scenario_id"],
                        "seed": row["seed"],
                    }
                )
                + "\n"
            )
    print(f"wrote {path}: {summary(kept)}")
