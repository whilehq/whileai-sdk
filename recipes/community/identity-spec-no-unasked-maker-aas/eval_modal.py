"""Evaluate the base three times and each arm once, on the two held-out sets.

Every arm goes through one entry point: the same vLLM process, the same
chat template rendered with ``enable_thinking=False``, the same sampling
params, the same prompt strings. The only thing that changes between passes
is which LoRA adapter is attached. Qwen3's training render injects an empty
``<think></think>`` block before every assistant turn while the default
generation prompt does not, so a pass that renders the prompt the default
way is measuring the template rather than the adapter.

    modal run --detach recipes/community/identity-spec-no-unasked-maker-aas/eval_modal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen3-1.7B"
HERE = Path(__file__).parent
SELECTORS = ("random", "loss", "aas")
MAX_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# The deployed system prompt. It says nothing about who made the agent: the
# identity is what training is supposed to put in the weights, and a prompt
# that already named the maker would solve the task before any GPU ran.
PLAIN_SYSTEM = "You are a helpful assistant. Answer the user directly and concisely."
# The prompt-only control: the same agent with its identity handed to it in
# the system prompt instead of trained in. Free to run, and it answers
# "could we have skipped the fine-tune?" before reading any arm.
PROMPTED_SYSTEM = PLAIN_SYSTEM + " You are Wai, made by While."

app = modal.App("identity-aas-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.0", "transformers==4.54.0", "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_USE_V1": "0"})
    .add_local_python_source("spec")
)

runs = modal.Volume.from_name("identity-aas-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("identity-aas-hf-cache", create_if_missing=True)
VOL = "/vol"


@app.function(
    image=image,
    gpu="L40S",
    timeout=2 * 60 * 60,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
)
def evaluate(identity: list[dict], leak: list[dict]) -> dict:
    import spec
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def render(prompt: str, system: str) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
    )

    def grade(split: str, prompts: list[dict], texts: list[str]) -> list[dict]:
        out = []
        for row, text in zip(prompts, texts):
            nm = spec.names_maker(text)
            out.append(
                {
                    "task_id": row["task_id"],
                    "names_maker": float(nm),
                    "leak": float(spec.leaked(text)),
                    "names_retired": float(spec.names_retired(text)),
                    "words": float(spec.word_count(text)),
                    # Up is good on both splits. On an identity ask the win
                    # is naming the maker; on an ordinary request the win is
                    # not mentioning it. compare() has no lower_is_better,
                    # so the leak target is written as its positive form.
                    "target": float(nm) if split == "identity" else float(not spec.leaked(text)),
                    "tier": row.get("tier", "n/a"),
                    "text": text[:400],
                }
            )
        return out

    def one_pass(label: str, system: str, seed: int, lora: LoRARequest | None) -> dict:
        result = {}
        for split, rows in (("identity", identity), ("leak", leak)):
            prompts = [render(r["prompt"], system) for r in rows]
            sp = SamplingParams(
                temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=seed
            )
            outs = llm.generate(prompts, sp, lora_request=lora)
            texts = [o.outputs[0].text.strip() for o in outs]
            result[split] = grade(split, rows, texts)
            mean = sum(r["target"] for r in result[split]) / len(result[split])
            print(f"[{label}] {split}: target {mean:.3f} over {len(rows)} tasks", flush=True)
        return result

    evals: dict = {}
    for i in (1, 2, 3):
        evals[f"base_run{i}"] = one_pass(f"base_run{i}", PLAIN_SYSTEM, seed=i, lora=None)
    evals["base_prompted"] = one_pass("base_prompted", PROMPTED_SYSTEM, seed=1, lora=None)
    for n, name in enumerate(SELECTORS, start=1):
        path = f"{VOL}/{name}/adapter"
        if not Path(path).exists():
            print(f"[{name}] no adapter at {path}, skipping")
            continue
        evals[name] = one_pass(name, PLAIN_SYSTEM, seed=1, lora=LoRARequest(name, n, path))

    Path(f"{VOL}/eval.json").write_text(json.dumps(evals))
    runs.commit()
    return evals


@app.local_entrypoint()
def main() -> None:
    out = HERE / "out"
    identity = [json.loads(x) for x in (out / "holdout_identity.jsonl").read_text().splitlines()]
    leak = [json.loads(x) for x in (out / "holdout_leak.jsonl").read_text().splitlines()]
    print(f"holdout: {len(identity)} identity asks, {len(leak)} ordinary requests")
    evals = evaluate.remote(identity, leak)
    (out / "eval.json").write_text(json.dumps(evals))
    print(f"wrote {out / 'eval.json'}")
