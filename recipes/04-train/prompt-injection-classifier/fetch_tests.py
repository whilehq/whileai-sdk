"""Pull the five frozen tests (and, with --train, the round-9 rows) from Hugging Face into out/.

    python fetch_tests.py            # test, test_hard, test_paste, test_llm, test_external
    python fetch_tests.py --train    # plus train_v9.jsonl and val_v9.jsonl

The repo is private under the while-ai org; a token with read access is needed
(``huggingface-cli login`` or ``HF_TOKEN``). Every file's sha256 is checked
against the pin committed next to this script, so a changed test cannot pass
for the frozen one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data import read_jsonl, sha256_rows

HERE = Path(__file__).resolve().parent
REPO = "while-ai/prompt-injection-carriers"
TESTS = ("test", "test_hard", "test_paste", "test_llm", "test_external")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    names = [f"{t}.jsonl" for t in TESTS] + (["train_v9.jsonl", "val_v9.jsonl"] if a.train else [])
    for name in names:
        path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(out))
        pin = HERE / f"{name[:-6]}.sha256"
        if pin.exists():
            digest = sha256_rows(read_jsonl(Path(path)))
            if digest != pin.read_text().strip():
                raise SystemExit(
                    f"{name}: sha256 {digest[:12]} does not match the pin {pin.read_text()[:12]}"
                )
            print(f"{name}: pin matches")
        else:
            print(f"{name}: fetched")


if __name__ == "__main__":
    main()
