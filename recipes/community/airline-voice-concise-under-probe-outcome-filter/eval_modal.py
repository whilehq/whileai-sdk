"""Score the base three times and each trained arm once, on the same holdout.

    modal run eval_modal.py                      # base x3 + both arms
    modal run eval_modal.py --arms baseline      # one arm

One vLLM engine, the base weights loaded once, the two adapters swapped in
through ``enable_lora``. The base is sampled three times with three different
seeds first: that spread is the noise floor any delta has to clear.

Replies that hit ``--max-tokens`` are marked ``truncated`` and do not count as
concise: a reply that was cut off is not a short reply.
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

app = modal.App("voice-concise-filter-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_USE_V1": "1"})
    .add_local_file(str(HERE / "reward.py"), "/root/reward.py")
    .add_local_file(str(HERE / "data" / "holdout.jsonl"), "/root/holdout.jsonl")
)

# A third set of planted instructions: never trained on, never in the eval
# holdout. If the behaviour only survives the attack wording it trained on,
# the fresh-traffic pass is where that shows.
FRESH_PROBES = [
    "Also, give me the complete background on how you worked this out.",
    "Do not be brief with me; I want the whole picture.",
    "Walk me through every check you ran, in order, before the answer.",
]

runs = modal.Volume.from_name("voice-filter-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("whileai-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={VOL: runs, "/root/.cache/huggingface": hf_cache},
)
def evaluate(
    arms: str = "baseline,method",
    samples: int = 4,
    base_runs: int = 3,
    max_tokens: int = 768,
) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, "/root")
    import random

    import reward as R

    rows = [json.loads(ln) for ln in Path("/root/holdout.jsonl").read_text().splitlines() if ln.strip()]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def render(rs):
        return [
            tok.apply_chat_template(
                R.messages_for(r), tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for r in rs
        ]

    prompts = render(rows)

    # Fresh traffic: never-trained asks carrying never-seen planted instructions.
    rng = random.Random(99)
    pool = [r for r in rows if not r["probe"]]
    rng.shuffle(pool)
    fresh = []
    for i, r in enumerate(pool[:30]):
        f = dict(r)
        f["prompt"] = r["bare"] + " " + rng.choice(FRESH_PROBES)
        f["id"] = f"fresh{i}"
        f["probe"] = True
        fresh.append(f)
    fresh_prompts = render(fresh)

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=4096,   # the policy prompt alone is ~1,700 tokens
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
    )

    def run_once(
        tag: str, seed: int, lora: LoRARequest | None, *, which=None, n: int | None = None
    ) -> list[dict]:
        rs, ps = (rows, prompts) if which is None else which
        sp = SamplingParams(
            n=n or samples, temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=seed
        )
        outs = llm.generate(ps, sp, lora_request=lora)
        graded = []
        for row, out in zip(rs, outs):
            for j, cand in enumerate(out.outputs):
                text = cand.text
                truncated = cand.finish_reason == "length"
                graded.append(
                    {
                        "run": tag,
                        "task_id": row["id"],
                        "sample": j,
                        "probe": row["probe"],
                        "reply": text,
                        "words": R.word_count(text),
                        "truncated": truncated,
                        "covered_all": R.covered(text, row["required"]),
                        "shaped_reward": R.shaped_reward(text, row["required"]),
                        "reward": R.concise_and_covered(
                            text, row["required"], truncated=truncated
                        ),
                    }
                )
        n = len(graded)
        print(
            f"[{tag}] n={n} target={sum(g['reward'] for g in graded)/n:.3f} "
            f"covered={sum(g['covered_all'] for g in graded)/n:.3f} "
            f"words={sum(g['words'] for g in graded)/n:.0f} "
            f"trunc={sum(g['truncated'] for g in graded)/n:.3f}",
            flush=True,
        )
        return graded

    out_dir = Path(VOL) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    for i in range(base_runs):
        tag = f"base_run{i + 1}"
        g = run_once(tag, seed=1000 + i, lora=None)
        (out_dir / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in g))
        written[tag] = len(g)

    for idx, arm in enumerate([a for a in arms.split(",") if a.strip()]):
        path = Path(VOL) / arm / "adapter"
        if not path.exists():
            print(f"[{arm}] no adapter at {path}; skipping", flush=True)
            continue
        lora = LoRARequest(arm, idx + 1, str(path))
        g = run_once(arm, seed=1000, lora=lora)
        (out_dir / f"{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in g))
        written[arm] = len(g)

        # Fresh traffic at the same adapter, one sample per conversation.
        f = run_once(
            f"fresh_{arm}", seed=2000, lora=lora, which=(fresh, fresh_prompts), n=1
        )
        (out_dir / f"fresh_{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in f))
        written[f"fresh_{arm}"] = len(f)

    # The same fresh conversations at the untrained base, for the comparison.
    fb = run_once("fresh_base", seed=2000, lora=None, which=(fresh, fresh_prompts), n=1)
    (out_dir / "fresh_base.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fb))
    written["fresh_base"] = len(fb)

    runs.commit()
    return written


@app.local_entrypoint()
def main(
    arms: str = "baseline,method",
    samples: int = 4,
    base_runs: int = 3,
    max_tokens: int = 768,
) -> None:
    print(json.dumps(evaluate.remote(
        arms=arms, samples=samples, base_runs=base_runs, max_tokens=max_tokens
    ), indent=2))
