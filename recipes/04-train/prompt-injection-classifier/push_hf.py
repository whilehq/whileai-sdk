"""Push the round-9 classifier, its int8 ONNX graph and the rows to Hugging Face, private, under while-ai.

    python push_hf.py --dry-run          # writes the cards, uploads nothing
    python push_hf.py                    # creates or updates the two private repos

Licences: base model nreimers/MiniLM-L6-H384-uncased (MIT); attack strings from
InjecAgent, AgentDojo, BIPIA and Gandalf (MIT), benign text from oasst1
(Apache-2.0) and the recipe's own model-written carriers; tests from deepset
(Apache-2.0), NotInject (MIT), LLMail-Inject (MIT), yanismiraoui (Apache-2.0).
Everything pushed is redistributable under Apache-2.0 with those notices; SPML
(round 4 only) is not pushed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE / "out"  # the five frozen tests; pins stay in git
OUT = HERE / "out"
ORG = "while-ai"
MODEL_REPO = f"{ORG}/prompt-injection-minilm-l6"
DATA_REPO = f"{ORG}/prompt-injection-carriers"
PICK = "v9-seeded"  # the round the card describes; seed 1


def table(r: dict, arms: list[str]) -> str:
    names = {
        "agentdojo_docs": "AgentDojo documents, injected vs clean (external)",
        "llm_heldout_domain": "model-written documents, held-out business",
        "sim_tool": "simulated tool results",
        "notinject": "NotInject benign (a pass is not flagging)",
        "llmail_inject": "LLMail-Inject emails, recall (external)",
        "hard": "hard held-out families, unseen transforms",
        "paste": "pasted under a user ask",
        "deepset": "deepset direct, never trained on",
        "multilingual_direct": "multilingual direct, recall (external)",
        "indirect_heldout_family": "held-out families on training carriers",
    }
    head = "| test (n) | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms) + "\n"
    rows = []
    for key, label in names.items():
        cells = []
        n = None
        for a in arms:
            h = r["arms"][a]["headline_points"].get(key)
            if h is None:
                cells.append("n/a")
                continue
            n = h["n"]
            cells.append(f"{h['points']:.0f} ± {h['ci95_half']:.0f}")
        rows.append(f"| {label} ({n}) | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def model_card(r: dict) -> str:
    lat = (
        r["latency"]["onnx-v9-seeded"]["latency_single_thread"]
        if "onnx-v9-seeded" in r["latency"]
        else r["latency"]["onnx-v8-union"]["latency_single_thread"]
    )
    return f"""---
license: apache-2.0
base_model: nreimers/MiniLM-L6-H384-uncased
language: [en]
tags: [prompt-injection, text-classification, safety, whileai, onnx]
pipeline_tag: text-classification
---

# prompt-injection-minilm-l6

A 22M-parameter binary classifier that flags a prompt injection in any text an agent reads: a tool result, an email, a retrieved passage, a pasted document, a user turn. Label 1 when a chunk of at most 512 tokens carries an instruction addressed to the model that the content's author had no standing to give. Jailbreaks (the user asking the model to break its own policy) are out of scope.

Trained with the [whileai](https://github.com/whilehq/whileai-sdk) recipe `recipes/04-train/prompt-injection-classifier` (round `{PICK}`, seed 1). Weights in `model.safetensors`; `onnx/model_int8.onnx` is the dynamic-int8 graph (23 MB), `onnx/model.onnx` fp32.

## Numbers

Correctness at the shipped threshold (1% false positives on the benign side of a validation split of the training rows), points out of 100, Wilson 95% half-width. Three seeds were trained; this card is seed 1. ProtectAI `deberta-v3-base-prompt-injection-v2` (184M) at its shipped 0.5 is the baseline. Meta's Prompt Guard 2 is gated and was not scored.

{table(r, ["protectai-v2", "v9-seeded-world"])}

Where it loses: direct attacks in languages other than English (the vocabulary is English WordPiece) and deepset's direct German/English set. The multilingual round (`v10-multilingual`, `microsoft/Multilingual-MiniLM-L12-H384`) is in the dataset's `rounds.json`; it doubles multilingual recall and is 118 MB.

## Threshold

`{r["arms"]["v9-seeded-world"]["seeds"][0]["threshold"]:.3f}` on the softmax probability of the INJECTION class. Lower it for recall, raise it for a stricter benign side.

## Latency

int8 ONNX, one CPU thread, Apple M5 Max, 300 timed runs after 20 warm-ups, tokenisation excluded: {lat["128"]["p50_ms"]} ms at 128 tokens, {lat["256"]["p50_ms"]} ms at 256, {lat["512"]["p50_ms"]} ms at 512 (p50). A 128-token sliding window with early exit costs one window when the first window fires.

## Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("{MODEL_REPO}")
model = AutoModelForSequenceClassification.from_pretrained("{MODEL_REPO}").eval()
enc = tok(chunk, truncation=True, max_length=512, return_tensors="pt")
p = torch.softmax(model(**enc).logits, -1)[0, 1].item()
flag = p > {r["arms"]["v9-seeded-world"]["seeds"][0]["threshold"]:.3f}
```

## Data

Public attack strings (InjecAgent, AgentDojo, BIPIA, Gandalf; MIT) planted by program into carriers with a matched benign twin per row; carriers written by `wai.simulate` with a model as situation writer, user and world for six businesses; the benchmarks' own user tasks as seeds. Real human text on the direct side (oasst1, Apache-2.0). Nothing in the five frozen tests trained the model: every payload sharing a word 8-gram with a test was dropped, and there are no exact overlaps. Details, every round and the shortcut probes: `{DATA_REPO}`.

## Limits

Synthetic benign false-positive rates are not a customer's traffic; the model is public and an attacker can read it; one base model, one tokenizer, English only.
"""


def data_card(r: dict, stats: dict) -> str:
    return f"""---
license: apache-2.0
language: [en, de, fr, es, pt, it]
tags: [prompt-injection, safety, whileai, synthetic]
task_categories: [text-classification]
---

# prompt-injection-carriers

The rows behind `{MODEL_REPO}`: training rows for each round of the climb, the five frozen tests with their content hashes, `rounds.json` (one entry per round with its five-line note) and `results.json` (every number with its interval).

## Files

| file | rows | what |
|---|---|---|
| `train_v9.jsonl` / `val_v9.jsonl` | {stats["train_rows"]} / {stats["val_rows"]} | round 9: template twins, pasted channel, six-domain model-written carriers, seeded-world rows |
| `train_v10.jsonl` / `val_v10.jsonl` | | round 10: round 9 plus translations in five languages |
| `test.jsonl` | 2,142 | deepset (never trained on), NotInject, planted in-distribution, held-out families and carriers, simulated tool results; sha256 in `test.sha256` |
| `test_hard.jsonl` | 316 | held-out families under transforms absent from training; matched twins |
| `test_paste.jsonl` | 300 | held-out families pasted under a user ask; matched twins |
| `test_llm.jsonl` | 422 | model-written documents from a business held out of training |
| `test_external.jsonl` | 1,674 | AgentDojo environment documents (injected and clean), LLMail-Inject emails, multilingual direct injections |

Row fields: `text`, `label` (1 = injection), `slice`, `family`, `carrier`, `source`, and for twins `pair`.

## Labels

By program where the planted string is known (the rule the `resist-planted-instruction` recipe uses); public labels kept for deepset, NotInject, LLMail-Inject and the multilingual set; model paraphrases and translations kept only when a second call confirmed the instruction survived.

## Leak audit

No test row's text appears in any training or validation row. Word 8-gram overlap exists only through carrier scaffolding shared by construction in the template-generated slices (the same sentence pool) and boilerplate ("let me know if you have any questions") in 8 of 600 LLMail emails; AgentDojo documents and the multilingual set share no 8-gram with training.

## Sources and licences

InjecAgent (MIT), AgentDojo (MIT), BIPIA (MIT), Lakera Gandalf (MIT), NotInject (MIT), deepset/prompt-injections (Apache-2.0), OpenAssistant oasst1 (Apache-2.0), microsoft/llmail-inject-challenge (MIT), yanismiraoui/prompt_injections (Apache-2.0). Model-written carriers, paraphrases and translations were produced with Claude Haiku 4.5 for this dataset. SPML (round 4) is not included.

## The climb

{table(r, ["protectai-v2", "v3-twins-dedupe", "v5-channels", "v9-seeded-world", "v10-multilingual"])}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    r = json.loads((HERE / "results.json").read_text())
    stats = json.loads((OUT / "build_stats_v9.json").read_text())
    stage = OUT / "hf"
    if stage.exists():
        shutil.rmtree(stage)
    m = stage / "model"
    d = stage / "data"
    m.mkdir(parents=True)
    d.mkdir(parents=True)
    src = OUT / f"{PICK}-seed1"
    for f in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "train_record.json",
    ):
        shutil.copy(src / f, m / f)
    onnx = OUT / f"onnx-{PICK}"
    (m / "onnx").mkdir()
    for f in ("model.onnx", "model_int8.onnx", "export.json"):
        shutil.copy(onnx / f, m / "onnx" / f)
    (m / "README.md").write_text(model_card(r))
    for f in (
        "train_v9.jsonl",
        "val_v9.jsonl",
        "train_v10.jsonl",
        "val_v10.jsonl",
        "rounds.json",
        "build_stats_v9.json",
        "build_stats_v10.json",
        "shortcut_probe.json",
    ):
        if (OUT / f).exists():
            shutil.copy(OUT / f, d / f)
    for t in ("test", "test_hard", "test_paste", "test_llm", "test_external"):
        shutil.copy(TESTS / f"{t}.jsonl", d / f"{t}.jsonl")
        shutil.copy(HERE / f"{t}.sha256", d / f"{t}.sha256")
    shutil.copy(HERE / "results.json", d / "results.json")
    (d / "README.md").write_text(data_card(r, stats))
    print(
        f"staged {sum(1 for _ in m.rglob('*') if _.is_file())} model files, {sum(1 for _ in d.rglob('*') if _.is_file())} data files under {stage}"
    )
    if a.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(m),
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="round 9 seed 1: weights, int8 ONNX, card",
    )
    api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(d),
        repo_id=DATA_REPO,
        repo_type="dataset",
        commit_message="rounds 3 to 10: rows, five frozen tests, results",
    )
    print(
        f"pushed https://huggingface.co/{MODEL_REPO} and https://huggingface.co/datasets/{DATA_REPO} (private)"
    )


if __name__ == "__main__":
    main()
