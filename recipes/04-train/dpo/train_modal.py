"""DPO on Modal, end to end, with the dashboard watching.

    modal run recipes/04-train/dpo/train_modal.py                        # on-policy pairs, 60 steps, A10G
    modal run recipes/04-train/dpo/train_modal.py --pairs pairs.jsonl    # pairs from wai.export_preference
    modal run recipes/04-train/dpo/train_modal.py --loss-type ipo --beta 0.1
    modal run recipes/04-train/dpo/train_modal.py --prompts-file recipes/04-train/grpo/prompts.jsonl   # model-written set
    modal run recipes/04-train/dpo/train_modal.py --from-run refund-dpo-v1 --run-name refund-dpo-v1-r2   # round two, from the adapter
    modal run recipes/04-train/dpo/train_modal.py --constructed-negatives     # an invented call as the rejected side on no-id prompts

What happens:

1. Prompts come from the simulator's offline template writer (no key), split
   by scenario into train and holdout; the rule in ``../grpo/reward.py`` is
   the grader. The holdout is sampled 4 times per prompt and scored: pass@1
   before.
2. Pairs. By default the base policy is sampled 8 times per train prompt,
   every reply is scored, and ``wai.build_preference_pairs`` pairs a pass
   with a fail of similar length (on-policy, length-matched). ``--pairs``
   takes a ``wai.export_preference`` file instead, from any graded set.
3. TRL's ``DPOTrainer`` with a LoRA adapter; the reference model is the
   same weights with the adapter off. ``wai.TrainerCallback`` puts the
   chosen/rejected reward margin, accuracy and loss on
   app.withwhile.com/platform/training as it goes.
4. The holdout is sampled again: pass@1 after. ``run.delta`` puts the
   before/after comparison on the run page, and the adapter lands on the
   ``whileai-dpo-runs`` volume under the run name.

DPO is offline: it learns from the pairs it is given and never samples
during training, so a run is cheap and deterministic given the pairs. The
price is that the pairs go stale as the policy moves; one round of
on-policy pairs is the usual answer, and this example is that round.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grpo"))

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# ``--gpu`` at the command line, or ZP_DPO_GPU in the environment.
DEFAULT_GPU = os.environ.get("ZP_DPO_GPU", "A10G")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-dpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "whileai",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE.parent / "grpo" / "reward.py"), "/root/reward.py")
    .add_local_file(str(HERE.parent / "grpo" / "prompts.py"), "/root/prompts.py")
    .add_local_file(str(HERE / "pairs.py"), "/root/pairs.py")
    # From inside this repo the checkout's SDK rides along and shadows the
    # PyPI one, so an unreleased SDK change works here first.
    .add_local_python_source("whileai")
)

runs_volume = modal.Volume.from_name("whileai-dpo-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _sample(
    model, tokenizer, prompts: list[dict], *, n: int, max_new_tokens: int, batch: int = 16
) -> list[list[str]]:
    """``n`` replies per prompt, batched, temperature 0.8."""
    import torch

    sys.path.insert(0, "/root")
    from reward import messages_for

    model.eval()
    tokenizer.padding_side = "left"
    out: list[list[str]] = []
    texts = [
        tokenizer.apply_chat_template(
            messages_for(p["prompt"]), tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
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
        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append(decoded[i * n : (i + 1) * n])
    model.train()
    return out


def _stamp(rows: list[dict]) -> list[dict]:
    """``row["category"]`` (with_id, no_id, off_topic) on every graded row,
    so ``run.delta(by="category")`` can split the target by kind of prompt."""
    try:
        sys.path.insert(0, "/root")
        from prompts import category
        from reward import case_for
    except ImportError:
        return rows
    for row in rows:
        row["category"] = category(case_for(str(row.get("prompt") or "")))
    return rows


def _by_category(rows: list[dict]) -> dict | None:
    """pass@1 and tool-call rate per prompt category, when prompts.py is
    mounted; a headline pass@1 hides which kind of prompt moved."""
    try:
        sys.path.insert(0, "/root")
        from prompts import pass_by_category
    except ImportError:
        return None
    return pass_by_category(rows)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=2 * 60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def train(
    train_prompts: list[dict],
    holdout_prompts: list[dict],
    run_name: str,
    pair_rows: list[dict] | None = None,
    base_model: str = BASE_MODEL,
    steps: int = 60,
    pair_samples: int = 8,
    constructed_negatives: bool = False,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    max_completion_length: int = 160,
    lora_rank: int = 16,
    eval_samples: int = 4,
    from_run: str = "",
    gpu: str = DEFAULT_GPU,
) -> dict:
    import json

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    sys.path.insert(0, "/root")
    from pairs import sampled_pairs
    from reward import SYSTEM, reward_rows

    import whileai.simulations as wai

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if from_run:
        # Round two (or n): the previous round's adapter is merged into the
        # weights, so this round samples its own pairs from that policy and
        # the reference (adapter off) is that policy, not the base. That is
        # iterated on-policy DPO; each round's pairs are fresh.
        from peft import PeftModel

        prev = os.path.join(VOLUME_ROOT, from_run, "adapter")
        if not os.path.isdir(prev):
            raise FileNotFoundError(f"no adapter at {prev}; run names are volume folders")
        model = PeftModel.from_pretrained(model, prev).merge_and_unload()
        print(f"policy: {base_model} + {from_run}/adapter, merged")

    config = {
        "base_model": base_model,
        "from_run": from_run or None,
        "steps": steps,
        "learning_rate": learning_rate,
        "beta": beta,
        "loss_type": loss_type,
        "max_completion_length": max_completion_length,
        "lora_rank": lora_rank,
        "train_prompts": len(train_prompts),
        "holdout_prompts": len(holdout_prompts),
        "pairs": "exported" if pair_rows else f"on-policy, {pair_samples} samples per prompt",
        "constructed_negatives": constructed_negatives,
        "gpu": gpu,
        "reward": "reward.py: lookup before refund, never invent an id, ask when none given",
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        run = wai.training_run(
            run_name,
            base_model=base_model,
            trainer="trl-dpo-lora",
            total_steps=steps,
            config=config,
        )
        print(f"dashboard: {run.url}")

    # pass@1 before: the same holdout prompts the after-eval uses, so the
    # delta is paired by prompt.
    before_replies = _sample(
        model, tokenizer, holdout_prompts, n=eval_samples, max_new_tokens=max_completion_length
    )
    before_rows = wai.mark_grounding(_stamp(reward_rows(holdout_prompts, before_replies)))
    before = wai.pass_at(before_rows)
    print(f"before: {before}")

    pair_report: dict = {}
    if not pair_rows:
        # On-policy pairs: the base policy's own passes against its own
        # fails, length-matched, so the contrast is the rule and not the
        # length or another model's style.
        replies = _sample(
            model, tokenizer, train_prompts, n=pair_samples, max_new_tokens=max_completion_length
        )
        pair_rows, pair_report = sampled_pairs(
            train_prompts, replies, system=SYSTEM, constructed=constructed_negatives
        )
        print(f"pairs: {pair_report}")
    if len(pair_rows) < 8:
        msg = f"only {len(pair_rows)} preference pairs; the policy passes or fails every prompt the same way"
        if run is not None:
            run.fail(msg)
        raise RuntimeError(msg)
    if run is not None:
        run.log(0, pairs=len(pair_rows))

    dataset = Dataset.from_list(
        [{k: v for k, v in r.items() if k in ("prompt", "chosen", "rejected")} for r in pair_rows]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    dpo = DPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        # 2 pairs a device, 4 accumulated: 8 pairs a step. Chosen and rejected
        # both run through the policy and the reference, so a pair costs four
        # sequences; checkpointing keeps a 1.5B model inside an A10G.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        beta=beta,
        loss_type=loss_type,
        max_length=896,
        max_prompt_length=704,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=17,
        remove_unused_columns=False,
    )
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
    # ref_model=None with a LoRA config: the reference is the same weights
    # with the adapter disabled, so one copy of the model is on the GPU.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    if run is not None:
        # finish=False: the holdout eval and the delta come after training.
        trainer.add_callback(wai.TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise

    policy = trainer.model
    after_replies = _sample(
        policy, tokenizer, holdout_prompts, n=eval_samples, max_new_tokens=max_completion_length
    )
    after_rows = wai.mark_grounding(_stamp(reward_rows(holdout_prompts, after_replies)))
    after = wai.pass_at(after_rows)
    print(f"after:  {after}")
    print(f"by category: before {_by_category(before_rows)}")
    print(f"             after  {_by_category(after_rows)}")

    adapter_dir = os.path.join(out_dir, "adapter")
    policy.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if from_run:
        # This adapter sits on merged weights, not the base, so the merged
        # policy is saved whole for serving and for a further round.
        policy.merge_and_unload().save_pretrained(os.path.join(out_dir, "merged"))
        tokenizer.save_pretrained(os.path.join(out_dir, "merged"))
    with open(os.path.join(out_dir, "pairs.jsonl"), "w") as fh:
        for r in pair_rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "holdout_before.jsonl"), "w") as fh:
        for r in before_rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "holdout_after.jsonl"), "w") as fh:
        for r in after_rows:
            fh.write(json.dumps(r) + "\n")
    runs_volume.commit()

    summary = {
        "pass_at_1_before": before.pass_at_1,
        "pass_at_1_after": after.pass_at_1,
        "ci95_before": before.ci95,
        "ci95_after": after.ci95,
        "holdout_prompts": len(holdout_prompts),
        "eval_samples": eval_samples,
        "by_category_before": _by_category(before_rows),
        "by_category_after": _by_category(after_rows),
        "pairs": len(pair_rows),
        "pair_report": pair_report or None,
        "from_run": from_run or None,
    }
    delta = None
    if run is not None:
        delta = run.delta(
            before_rows,
            after_rows,
            target="pass_at_1",
            must_not_regress=["well_formed", "argument_grounding"],
            by="category",
        )
        run.finish("done", summary=summary, adapter=f"whileai-dpo-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    else:
        delta = wai.delta_report(
            before_rows,
            after_rows,
            target="pass_at_1",
            must_not_regress=["well_formed", "argument_grounding"],
            by="category",
        )
    print(wai.format_delta_report(delta))
    summary["delta_verdict"] = delta["target_verdict"]
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "refund-dpo-v1",
    prompts: int = 200,
    holdout: float = 0.2,
    pairs: str = "",
    steps: int = 60,
    pair_samples: int = 8,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    base_model: str = BASE_MODEL,
    seed: int = 0,
    prompts_file: str = "",
    balance: float = 0.0,
    from_run: str = "",
    constructed_negatives: bool = False,
    gpu: str = DEFAULT_GPU,
):
    from pairs import load_export
    from reward import SYSTEM, build_prompts, split_holdout

    if prompts_file:
        sys.path.insert(0, str(HERE.parent / "grpo"))
        from prompts import load_prompts

        items = load_prompts(prompts_file)
        print(f"{len(items)} model-written prompts from {prompts_file}")
        from prompts import split_holdout_stratified

        # By scenario within each category, so the holdout has no-id and
        # off-topic prompts too; a plain hash split once left it with none.
        train_items, held = split_holdout_stratified(items, holdout)
        if balance > 0:
            from prompts import balance as _balance
            from prompts import summary

            train_items = _balance(train_items, balance)
            print(
                f"balanced train set to {balance:.0%} per minority category: {summary(train_items)}"
            )
    else:
        items = build_prompts(prompts, seed=seed)
        train_items, held = split_holdout(items, holdout)
    print(f"{len(items)} prompts: {len(train_items)} train, {len(held)} holdout")
    pair_rows = load_export(pairs, system=SYSTEM) if pairs else None
    if pair_rows is not None:
        print(f"{len(pair_rows)} pairs from {pairs}")
    # Modal binds the GPU when the function is defined, so another GPU
    # is a per-call option rather than a mutated argument.
    fn = train if gpu == DEFAULT_GPU else train.with_options(gpu=gpu)
    summary = fn.remote(
        train_prompts=train_items,
        holdout_prompts=held,
        run_name=run_name,
        pair_rows=pair_rows,
        base_model=base_model,
        steps=steps,
        pair_samples=pair_samples,
        learning_rate=learning_rate,
        beta=beta,
        loss_type=loss_type,
        from_run=from_run,
        constructed_negatives=constructed_negatives,
        gpu=gpu,
    )
    print("done:", summary)
