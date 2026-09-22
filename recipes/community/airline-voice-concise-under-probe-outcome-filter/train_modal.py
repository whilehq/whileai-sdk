"""Two GRPO arms on your own Modal: stock, and the paper's outcome filter.

    modal run train_modal.py --arm baseline   # group is flat by the shaped score (stock GRPO)
    modal run train_modal.py --arm method     # group is flat by the binary outcome

Both arms take the same prompts, the same base, the same number of optimizer
steps and the same reward. The only difference is ``reward.group_is_flat``.
Each arm writes its adapter and a ``filter_trace.json`` (how many groups it
dropped per step) to the ``voice-filter-runs`` volume.

Needs ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``. ``WHILEAI_API_KEY`` is
optional: with it the run shows up on while.ai/platform, without it the
arm trains and prints the same numbers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE_MODEL = os.environ.get("VOICE_BASE_MODEL", "Qwen/Qwen3-1.7B")
VOL = "/vol"

app = modal.App("voice-concise-filter-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        # A floor, never the bare name: bare resolves to whatever the image
        # cache last saw and can freeze a trainer on an ancient wheel (#661).
        "whileai>=0.109",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(str(HERE / "reward.py"), "/root/reward.py")
    .add_local_file(str(HERE / "data" / "train.jsonl"), "/root/train.jsonl")
)

runs = modal.Volume.from_name("voice-filter-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard = modal.Secret.from_dict({"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")})


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard],
)
def train(arm: str = "baseline", steps: int = 40, k: int = 4, seed: int = 11) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    import reward as R

    metric = {"baseline": "score", "method": "outcome"}[arm]
    rows = [
        json.loads(ln) for ln in Path("/root/train.jsonl").read_text().splitlines() if ln.strip()
    ]
    print(f"[{arm}] filter metric = {metric}; {len(rows)} train prompts", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def to_prompt(r: dict) -> str:
        return tok.apply_chat_template(
            R.messages_for(r),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    ds = Dataset.from_list(
        [{"prompt": to_prompt(r), "required": r["required"], "probe": r["probe"]} for r in rows]
    )

    # The reward function stashes each rollout's binary outcome so the filter
    # below can read it; TRL only hands the trainer the summed reward.
    stash: list[tuple[float, float]] = []

    def shaped(completions, required, **kw):
        out = []
        for c, req in zip(completions, required):
            text = c if isinstance(c, str) else c[0]["content"]
            s = R.shaped_reward(text, req)
            stash.append((R.outcome_of(text, req), s))
            out.append(s)
        return out

    trace: list[dict] = []

    class FilteredGRPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            stash.clear()
            out = super()._generate_and_score_completions(inputs)
            adv = out.get("advantages")
            if adv is None:
                raise RuntimeError("TRL did not return 'advantages'; this trainer pins trl==0.19.1")
            n = adv.numel()
            if len(stash) != n:
                raise RuntimeError(
                    f"expected {n} stashed outcomes, got {len(stash)}; the reward "
                    "function did not see every rollout"
                )
            pairs = torch.tensor(stash, dtype=torch.float32)
            outcomes = pairs[:, 0].view(-1, k)
            scores = pairs[:, 1].view(-1, k)
            keep = torch.ones(outcomes.shape[0])
            for g in range(outcomes.shape[0]):
                if R.group_is_flat(outcomes[g].tolist(), scores[g].tolist(), metric=metric):
                    keep[g] = 0.0
            dropped = int((keep == 0).sum())
            trace.append(
                {
                    "step": len(trace),
                    "groups": int(outcomes.shape[0]),
                    "dropped": dropped,
                    "mean_outcome": float(outcomes.mean()),
                    "all_right_groups": int((outcomes.sum(1) == k).sum()),
                    "all_wrong_groups": int((outcomes.sum(1) == 0).sum()),
                }
            )
            out["advantages"] = adv * keep.view(-1, 1).to(adv.device).expand(-1, k).reshape(-1)
            return out

    cfg = GRPOConfig(
        output_dir=f"{VOL}/{arm}",
        per_device_train_batch_size=k,
        gradient_accumulation_steps=8,
        num_generations=k,
        max_completion_length=320,
        # The airline policy prompt is ~1,700 tokens on its own. At TRL's
        # default (512) it is truncated from the left and the agent trains
        # against half a policy, silently.
        max_prompt_length=2304,
        max_steps=steps,
        learning_rate=1e-5,
        beta=0.0,  # no KL: the filter is the only thing acting
        temperature=1.0,
        num_iterations=1,  # on-policy
        seed=seed,
        gradient_checkpointing=False,  # ON corrupts Qwen3 generation on this stack
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = FilteredGRPOTrainer(
        model=BASE_MODEL,
        reward_funcs=[shaped],
        args=cfg,
        train_dataset=ds,
        peft_config=LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
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
    trainer.train()

    dest = Path(VOL) / arm / "adapter"
    trainer.model.save_pretrained(str(dest))
    tok.save_pretrained(str(dest))

    groups = sum(t["groups"] for t in trace)
    dropped = sum(t["dropped"] for t in trace)
    summary = {
        "arm": arm,
        "filter_metric": metric,
        "base_model": BASE_MODEL,
        "steps": steps,
        "k": k,
        "seed": seed,
        "groups_seen": groups,
        "groups_dropped": dropped,
        "drop_rate": dropped / groups if groups else None,
        "trace": trace,
    }
    (Path(VOL) / arm / "filter_trace.json").write_text(json.dumps(summary, indent=2))
    runs.commit()
    print(f"[{arm}] dropped {dropped}/{groups} groups ({summary['drop_rate']:.1%})", flush=True)
    return {kk: v for kk, v in summary.items() if kk != "trace"}


@app.local_entrypoint()
def main(arm: str = "baseline", steps: int = 40, k: int = 4, seed: int = 11) -> None:
    out = train.remote(arm=arm, steps=steps, k=k, seed=seed)
    print(json.dumps(out, indent=2))
