"""Pull the private pieces from Hugging Face into out/: the classifier, the
sibling's SFT rows, the two seed-17 adapters. Needs a token with access to
the while-ai org (`HF_TOKEN`, or the one `huggingface-cli login` saved).

    python fetch.py --classifier     # out/classifier: tokenizer + int8 ONNX, 23 MB
    python fetch.py --rows           # out/sft_rows.jsonl, out/sft_rows_random.jsonl
    python fetch.py --adapters       # out/adapter-<arm>-s17/, then `modal volume put`
    python fetch.py --eval           # out/eval_<label>.jsonl, every draw scored here
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

CLASSIFIER = "while-ai/prompt-injection-minilm-l6"
ROWS = "while-ai/planted-instruction-storefront"
ADAPTERS = {
    "sft-reward-s17": "while-ai/planted-instruction-storefront-qwen3-4b-lora",
    "sft-random-s17": "while-ai/planted-instruction-storefront-qwen3-4b-lora-random-control",
}
EVAL = "while-ai/refuse-or-comply-eval"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--classifier", action="store_true")
    p.add_argument("--rows", action="store_true")
    p.add_argument("--adapters", action="store_true")
    p.add_argument("--eval", action="store_true")
    args = p.parse_args(argv)
    if not any(vars(args).values()):
        p.print_help()
        return 0
    from huggingface_hub import snapshot_download

    OUT.mkdir(parents=True, exist_ok=True)
    if args.classifier:
        snapshot_download(
            CLASSIFIER,
            local_dir=OUT / "classifier",
            allow_patterns=[
                "tokenizer*.json",
                "vocab.txt",
                "config.json",
                "onnx/model_int8.onnx",
                "onnx/export.json",
            ],
        )
        print("classifier ->", OUT / "classifier")
    if args.rows:
        d = snapshot_download(
            ROWS,
            repo_type="dataset",
            allow_patterns=[
                "sft_rows.jsonl",
                "sft_rows_random.jsonl",
                "selected.jsonl",
                "selected_random.jsonl",
            ],
        )
        for name in (
            "sft_rows.jsonl",
            "sft_rows_random.jsonl",
            "selected.jsonl",
            "selected_random.jsonl",
        ):
            shutil.copy(Path(d) / name, OUT / name)
        print("rows ->", OUT)
    if args.adapters:
        for name, repo in ADAPTERS.items():
            snapshot_download(repo, local_dir=OUT / f"adapter-{name}")
            print(
                f"{name} -> {OUT / f'adapter-{name}'}; upload with: modal volume put robust-adapters {OUT / f'adapter-{name}'} /{name}/adapter"
            )
    if args.eval:
        d = snapshot_download(EVAL, repo_type="dataset", allow_patterns=["eval/*"])
        for path in sorted((Path(d) / "eval").glob("*")):
            shutil.copy(path, OUT / path.name)
        print("eval rows ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
