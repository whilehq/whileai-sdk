"""Print a few held-out replies from a trained adapter, to read the talk by eye.

python peek.py team-talk-recipe-s17-2026-09-23 [--plain] [--n 4]
"""

from __future__ import annotations

import sys

import modal

try:
    from recipe import VOLUME_ROOT, _sample, app, hf_cache, image, runs_volume
except ModuleNotFoundError:  # inside the container the recipe is recipe_mod
    sys.path.insert(0, "/root")
    from recipe_mod import VOLUME_ROOT, _sample, app, hf_cache, image, runs_volume


@app.function(
    image=image,
    gpu="L40S",
    timeout=20 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
)
def peek(run_name: str, team: bool, questions: list[str]) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    from recipe_mod import BASE_MODEL

    adapter = f"{VOLUME_ROOT}/{run_name}/adapter"
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, adapter)
    return [r[0] for r in _sample(model, tok, questions, team=team, n=1, max_new_tokens=512)]


if __name__ == "__main__":
    from recipe import data, talks, turns_of

    run_name = sys.argv[1]
    team = "--plain" not in sys.argv
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4
    _, holdout = data(0, 8, n)
    with modal.enable_output(), app.run():
        replies = peek.remote(run_name, team, [t["question"] for t in holdout])
    for reply in replies:
        print(f"==== talks={talks(reply)} turns={turns_of(reply)}\n{reply}\n")
