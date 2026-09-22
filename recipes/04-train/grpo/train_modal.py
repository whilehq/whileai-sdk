"""GRPO on Modal, end to end, with the dashboard watching.

    modal run recipes/04-train/grpo/train_modal.py                       # 200 prompts, 40 steps, A10G
    modal run recipes/04-train/grpo/train_modal.py --steps 80 --gpu H100 --run-name refund-grpo-v2
    modal run recipes/04-train/grpo/train_modal.py --prompts-file recipes/04-train/grpo/prompts.jsonl   # model-written set, ~160 holdout prompts
    modal run recipes/04-train/grpo/train_modal.py --loss-type dr_grpo --no-scale-rewards     # Dr.GRPO
    modal run recipes/04-train/grpo/train_modal.py --epsilon-high 0.28 --mask-truncated        # DAPO's clip and overlong mask
    modal run recipes/04-train/grpo/train_modal.py --monitor-every 5 --stop-on feature         # end the run on a named reward hack
    modal run recipes/04-train/grpo/train_modal.py --from-run refund-grpo-v1 --run-name refund-grpo-v2   # round two off round one's adapter
    modal run recipes/04-train/grpo/train_modal.py --temperature 1.0 --sample-seed 5 --base-runs 5       # move the sampling defaults from the call

What happens:

1. Prompts come from the simulator's offline template writer (no key), split
   by scenario into train and holdout; the same ``--seed`` builds the same
   set and the same split on every run. The reward is the policy's one
   testable rule, in ``reward.py``: look the order up first, never invent an
   id, ask when none is given. It is a function, not a judge.
2. The train set is decontaminated against the holdout, then the holdout is
   sampled 4 times per prompt before training and scored: that is pass@1
   before. The base is evaluated ``--base-runs`` times (3), from a seeded
   sampler, so the spread across those re-runs is the eval's own noise and
   the delta is read against it rather than against zero.
3. TRL's ``GRPOTrainer`` with a LoRA adapter, 8 generations per prompt.
   ``wai.TrainerCallback`` puts reward, KL and the progress bar on
   while.ai/platform/training as it goes. ``wai.HackMonitor``
   samples the holdout from the live policy every ``--monitor-every``
   steps, logs the proxy reward and completion length beside the training
   curve, and scans the batch for what the reward is paying for; an alarm
   is a line on the run, and ``--stop-on feature`` (or ``length``) ends
   the run on it.
4. The holdout is sampled again: pass@1 after. ``run.delta`` puts the
   before/after comparison on the run page, and the adapter lands on the
   ``whileai-grpo-runs`` volume under the run name.

Set ``WHILEAI_API_KEY`` on your laptop for the dashboard; without it the
run trains the same and prints its numbers only. One A10G is about $1.10 an
hour and the default run is under fifteen minutes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

from whileai.config import provenance, requirement

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Every sampling number this file used to spell inline. Named, sourced and
# movable from the call (CONSTITUTION.md, belief 3; #788).
from defaults import (
    BASE_EVAL_RUNS,
    SAMPLE_SEED,
    SAMPLE_TEMPERATURE,
    SAMPLE_TOP_P,
)

# PROBE_SEED_OFFSET = 500: round two's one sampling pass over the train
# prompts draws at a seed no eval uses, so the band it measures and the
# pass@1 it is compared against are independent draws (convention, untested).
PROBE_SEED_OFFSET = 500

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# ``--gpu`` at the command line, or ZP_GRPO_GPU in the environment.
DEFAULT_GPU = os.environ.get("ZP_GRPO_GPU", "A10G")
VOLUME_ROOT = "/vol"

app = modal.App("whileai-grpo")

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
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "reward.py"), "/root/reward.py")
    .add_local_file(str(HERE / "prompts.py"), "/root/prompts.py")
    .add_local_file(str(HERE / "defaults.py"), "/root/defaults.py")
    # From inside this repo the checkout's SDK rides along and shadows the
    # PyPI one, so an unreleased SDK change works here first.
    .add_local_python_source("whileai")
)

runs_volume = modal.Volume.from_name("whileai-grpo-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"WHILEAI_API_KEY": os.environ.get("WHILEAI_API_KEY", "")}
)


def _sample(
    model,
    tokenizer,
    prompts: list[dict],
    *,
    n: int,
    max_new_tokens: int,
    batch: int = 16,
    temperature: float = SAMPLE_TEMPERATURE,
    top_p: float = SAMPLE_TOP_P,
    seed: int = SAMPLE_SEED,
) -> list[list[str]]:
    """``n`` replies per prompt, batched, at the recipe's named sampling
    defaults and from a seeded generator.

    The sampler was unseeded until #788, so pass@1 before and pass@1 after
    were two draws from whatever torch's global state happened to be and
    neither could be reproduced. Belief 1: a number is a result only with
    its seed. Pass a different ``seed`` to get an independent draw, which is
    how the base re-runs measure the eval's own noise."""
    import torch
    from transformers import set_seed

    sys.path.insert(0, "/root")
    from reward import messages_for

    set_seed(seed)
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
                temperature=temperature,
                top_p=top_p,
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
    base_model: str = BASE_MODEL,
    steps: int = 40,
    num_generations: int = 8,
    learning_rate: float = 5e-6,
    beta: float = 0.04,
    max_completion_length: int = 160,
    lora_rank: int = 16,
    eval_samples: int = 4,
    loss_type: str = "bnpo",
    epsilon_high: float | None = None,
    scale_rewards: bool = True,
    mask_truncated: bool = False,
    monitor_every: int = 10,
    stop_on: str = "",
    temperature: float = SAMPLE_TEMPERATURE,
    top_p: float = SAMPLE_TOP_P,
    seed: int = SAMPLE_SEED,
    base_runs: int = BASE_EVAL_RUNS,
    from_run: str = "",
    gpu: str = DEFAULT_GPU,
) -> dict:
    import json

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from reward import messages_for, reward_rows, score

    import whileai.simulations as wai

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if from_run:
        # Round two (or n): the previous round's adapter is merged into the
        # weights, so this round samples its own rollouts from that policy
        # and the KL reference (adapter off) is that policy, not the base.
        # The same shape dpo/train_modal.py has had; the GRPO README used to
        # send readers there for a second round (#788).
        from peft import PeftModel

        prev = os.path.join(VOLUME_ROOT, from_run, "adapter")
        if not os.path.isdir(prev):
            raise FileNotFoundError(f"no adapter at {prev}; run names are volume folders")
        model = PeftModel.from_pretrained(model, prev).merge_and_unload()
        print(f"round two from {from_run}: {prev} merged into the base weights")

    config = {
        "base_model": base_model,
        "steps": steps,
        "num_generations": num_generations,
        "learning_rate": learning_rate,
        "beta": beta,
        "loss_type": loss_type,
        "epsilon_high": epsilon_high,
        "scale_rewards": scale_rewards,
        "mask_truncated": mask_truncated,
        "max_completion_length": max_completion_length,
        "lora_rank": lora_rank,
        "monitor_every": monitor_every,
        "stop_on": stop_on or None,
        "train_prompts": len(train_prompts),
        "holdout_prompts": len(holdout_prompts),
        "gpu": gpu,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "base_eval_runs": base_runs,
        "from_run": from_run or None,
        "reward": "reward.py: lookup before refund, never invent an id, ask when none given",
    }
    run = None
    if os.environ.get("WHILEAI_API_KEY"):
        run = wai.training_run(
            run_name,
            base_model=base_model,
            trainer="trl-grpo-lora",
            total_steps=steps,
            config=config,
        )
        print(f"dashboard: {run.url}")

    # pass@1 before: the same holdout prompts the after-eval uses, so the
    # delta is paired by prompt. The base is evaluated ``base_runs`` times,
    # not once (#788): the spread across those re-runs is the eval's own
    # noise, and a delta smaller than it is not a result (Lambert 2025,
    # chapter Evaluation). Each re-run is an independent draw from the same
    # policy, which is what a different sampler seed buys.
    base_evals = [
        wai.mark_grounding(
            _stamp(
                reward_rows(
                    holdout_prompts,
                    _sample(
                        model,
                        tokenizer,
                        holdout_prompts,
                        n=eval_samples,
                        max_new_tokens=max_completion_length,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed + i,
                    ),
                )
            )
        )
        for i in range(max(1, base_runs))
    ]
    before_rows = base_evals[0]
    before = wai.pass_at(before_rows)
    run_std: float | None = None
    run_std_runs: int | None = None
    if len(base_evals) >= 2:  # two runs before a standard deviation exists
        noise = wai.eval_variance(*base_evals)
        run_std = float(noise["run_std"])
        run_std_runs = int(noise["n_runs"])
        print(f"base re-runs: {[round(wai.pass_at(r).pass_at_1, 3) for r in base_evals]}")
        print(f"eval noise over {run_std_runs} base runs: run_std {run_std:.4f}")
    else:
        print("base evaluated once: no noise floor, so no delta here is above noise by measurement")
    print(f"before: {before}")

    def rule_reward(completions, case, **kwargs):
        # TRL passes extra dataset columns (here ``case``) as keyword lists.
        out = []
        for completion, c in zip(completions, case):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            out.append(score(text, c))
        return out

    if from_run:
        # Round two trains on the prompts this policy is still uncertain
        # about. A round that reuses round one's file keeps paying for
        # groups with no contrast: every rollout passes or every rollout
        # fails, the group-relative advantage is zero, and the step is
        # free of gradient (Lambert 2025, chapter Reasoning, the 20-80%
        # band; Yu et al. 2025 (DAPO), dynamic sampling). ``next_round``
        # applies the band to what THIS policy does, so it needs one
        # sampling pass over the train prompts first.
        probe = _sample(
            model,
            tokenizer,
            train_prompts,
            n=eval_samples,
            max_new_tokens=max_completion_length,
            temperature=temperature,
            top_p=top_p,
            seed=seed + PROBE_SEED_OFFSET,
        )
        plan = wai.next_round(reward_rows(train_prompts, probe), tasks=train_prompts)
        kept = {str(t.get("prompt") or "") for t in plan["tasks"]}
        train_prompts = [p for p in train_prompts if p["prompt"] in kept] or train_prompts
        print(
            f"next_round: {plan['kept']} kept in the {plan['band']} band, "
            f"{plan['dropped_solved']} solved, {plan['dropped_unsolved']} unsolved, "
            f"{plan['unknown']} unknown; prompt set {plan['prompt_set_sha']}"
        )
        config["train_prompts"] = len(train_prompts)
        config["prompt_set_sha"] = plan["prompt_set_sha"]

    dataset = Dataset.from_list(
        [{"prompt": messages_for(p["prompt"]), "case": p["case"]} for p in train_prompts]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        beta=beta,
        # The variants are flags on the same trainer. TRL's default loss is
        # "bnpo" (token-level, normalized over the batch); "grpo" is the
        # original per-sequence mean, which favors short completions.
        # Dr.GRPO: loss_type "dr_grpo" and scale_rewards=False, so neither
        # length nor the group's reward std scales the advantage. DAPO: a
        # wider upper clip (epsilon_high 0.28) and truncated completions
        # masked out of the loss; its dynamic sampling (drop unanimous
        # groups) is what the platform's publish gate does offline.
        loss_type=loss_type,
        epsilon_high=epsilon_high,
        scale_rewards=scale_rewards,
        mask_truncated_completions=mask_truncated,
        max_completion_length=max_completion_length,
        max_prompt_length=768,
        temperature=temperature,  # the same temperature the rollouts were measured at
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
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
    # The hack monitor samples the holdout from the live policy every
    # ``monitor_every`` steps and scores it with the training reward
    # (through ``wrap``) while ``hack_scan`` reads what the reward is paying
    # for in the batch. No judge here, so no gold curve: the length and
    # feature alarms still run, and ``endorsed`` says the lookup call is the
    # behavior. ``--stop-on feature`` ends the run on a named hack.
    monitor = wai.HackMonitor(
        run,
        holdout=[{"prompt": messages_for(p["prompt"]), "case": p["case"]} for p in holdout_prompts],
        every=monitor_every,
        k=eval_samples,
        endorsed=["lookup_order"],
        max_new_tokens=max_completion_length,
        stop_on=[s for s in stop_on.split(",") if s],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[monitor.wrap(rule_reward)],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.add_callback(monitor)
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
        policy,
        tokenizer,
        holdout_prompts,
        n=eval_samples,
        max_new_tokens=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    after_rows = wai.mark_grounding(_stamp(reward_rows(holdout_prompts, after_replies)))
    after = wai.pass_at(after_rows)
    print(f"after:  {after}")
    print(f"by category: before {_by_category(before_rows)}")
    print(f"             after  {_by_category(after_rows)}")

    adapter_dir = os.path.join(out_dir, "adapter")
    policy.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
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
        "base_runs": len(base_evals),
        "run_std": run_std,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }
    # The local report is the one that carries the noise floor the base
    # re-runs measured; ``run.delta`` draws the page and takes no run_std.
    delta = wai.delta_report(
        before_rows,
        after_rows,
        target="pass_at_1",
        must_not_regress=["well_formed", "argument_grounding"],
        by="category",
        run_std=run_std,
        run_std_runs=run_std_runs,
    )
    if run is not None:
        run.delta(
            before_rows,
            after_rows,
            target="pass_at_1",
            must_not_regress=["well_formed", "argument_grounding"],
            by="category",
        )
        run.finish("done", summary=summary, adapter=f"whileai-grpo-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    print(wai.format_delta_report(delta))
    print(wai.format_hack_monitor(monitor.summary()))
    summary["delta_verdict"] = delta["target_verdict"]
    summary["alarms"] = [f"step {a['step']} {a['kind']}" for a in monitor.alarms]
    if monitor.stopped_at:
        summary["stopped_at"] = monitor.stopped_at["step"]
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "refund-grpo-v1",
    prompts: int = 200,
    holdout: float = 0.2,
    steps: int = 40,
    num_generations: int = 8,
    learning_rate: float = 5e-6,
    beta: float = 0.04,
    base_model: str = BASE_MODEL,
    seed: int = 0,
    prompts_file: str = "",
    balance: float = 0.0,
    loss_type: str = "bnpo",
    epsilon_high: float = 0.0,
    no_scale_rewards: bool = False,
    mask_truncated: bool = False,
    monitor_every: int = 10,
    stop_on: str = "",
    temperature: float = SAMPLE_TEMPERATURE,
    top_p: float = SAMPLE_TOP_P,
    sample_seed: int = SAMPLE_SEED,
    base_runs: int = BASE_EVAL_RUNS,
    from_run: str = "",
    gpu: str = DEFAULT_GPU,
):
    print(provenance(), file=sys.stderr)
    from reward import build_prompts, split_holdout

    import whileai as wai

    if prompts_file:
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
    # A held-out prompt that also sits in the training set measures memory,
    # not the change (Lambert 2025, chapter Evaluation). The split is by
    # scenario id, so this should drop nothing; it runs anyway and prints the
    # count, because "should" is not a measurement (#788).
    train_items, decon = wai.decontaminate(train_items, against=held)
    print(f"decontaminate: {decon['n_contaminated']} of {decon['n']} train rows dropped")
    with_id = sum(1 for p in items if p["case"]["order_id"])
    print(
        f"{len(items)} prompts ({with_id} name an order id): {len(train_items)} train, {len(held)} holdout"
    )
    # Modal binds the GPU when the function is defined, so another GPU
    # is a per-call option rather than a mutated argument.
    fn = train if gpu == DEFAULT_GPU else train.with_options(gpu=gpu)
    summary = fn.remote(
        train_prompts=train_items,
        holdout_prompts=held,
        run_name=run_name,
        base_model=base_model,
        steps=steps,
        num_generations=num_generations,
        learning_rate=learning_rate,
        beta=beta,
        loss_type=loss_type,
        epsilon_high=epsilon_high or None,
        scale_rewards=not no_scale_rewards,
        mask_truncated=mask_truncated,
        monitor_every=monitor_every,
        stop_on=stop_on,
        temperature=temperature,
        top_p=top_p,
        seed=sample_seed,
        base_runs=base_runs,
        from_run=from_run,
        gpu=gpu,
    )
    print("done:", summary)
