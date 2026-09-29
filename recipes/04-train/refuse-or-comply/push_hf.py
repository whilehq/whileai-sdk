"""Push the eval rows and the four new adapters to Hugging Face, private, under while-ai.

    python push_hf.py --dry-run      # writes the cards to out/cards/, uploads nothing
    python push_hf.py                # one private dataset repo (rows) and two private model repos (adapters, one per arm, seeds as subfolders)

The seed-17 adapters already live in the sibling recipe's repos and are not
re-uploaded; the cards name them. Token: `HF_TOKEN`, or the one
`huggingface-cli login` saved. Licence: the rows are the sibling's
(`license: other`, model-generated on a fictional store); the adapters are
LoRA deltas on Qwen/Qwen3-4B (Apache-2.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ORG = "while-ai"
DATA_REPO = f"{ORG}/refuse-or-comply-eval"
ADAPTER_REPOS = {
    "sft-reward": f"{ORG}/refuse-or-comply-qwen3-4b-lora-reward",
    "sft-random": f"{ORG}/refuse-or-comply-qwen3-4b-lora-random-control",
}
SIBLING_REPOS = {
    "sft-reward": "while-ai/planted-instruction-storefront-qwen3-4b-lora",
    "sft-random": "while-ai/planted-instruction-storefront-qwen3-4b-lora-random-control",
}
RECIPE = "https://github.com/whilehq/whileai-sdk/tree/main/recipes/04-train/refuse-or-comply"


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def ci(v) -> str:
    return "n/a" if not v else f"[{100 * v[0]:.1f}, {100 * v[1]:.1f}]"


def arm_table(analysis: dict) -> str:
    head = "| arm | half | full | 95% | safe and done | over-refusal | agent false-flag | gate false-flag |\n|---|---|---|---|---|---|---|---|\n"
    rows = []
    for name, a in analysis["arms"].items():
        for h in ("attack", "clean"):
            s = a[h]
            rows.append(
                f"| {name} | {h} | {pct(s['full']['mean'])} | {ci(s['full']['ci95'])} | "
                f"{pct(s['safe_and_done']['mean'])} | {pct(s['over_refusal']['mean'])} | "
                f"{pct(a['agent_false_flag_rate_clean']['rate']) if h == 'clean' else ''} | "
                f"{pct(a['gate']['false_flag_rate_clean_results'])} |"
            )
    return head + "\n".join(rows)


def data_card(analysis: dict) -> str:
    return f"""---
license: other
language: [en]
tags: [agent, tool-use, prompt-injection, over-refusal, safety, while-ai, eval]
size_categories: [1K<n<10K]
---

# Refuse or comply: eval rows

Every graded rollout behind the recipe [`recipes/04-train/refuse-or-comply`]({RECIPE}): five arms on one frozen holdout (`{analysis["test_version"]}`, 240 pinned prompts, 120 with a planted instruction inside a tool result and 120 clean), all decided by the sibling recipe's program grader (`resist-planted-instruction`). No model judge anywhere.

`eval/<label>.jsonl` is one draw: `base_d1..3` and `gate-*_d1..3` are the same weights re-drawn with sampling seeds 21, 22, 23; `sft-*_s17/18/19` are three training seeds of each adapter, one draw each. `eval/<label>.json` is the draw's record (rows, graded, pass@1, criterion failures, wall seconds). `analysis.json` is what `analyse.py` computed from them and what the recipe's `results.json` quotes.

## Arms

| arm | what |
|---|---|
| `base` | Qwen/Qwen3-4B under the sibling's policy |
| `gate-minilm` | base, with `while-ai/prompt-injection-minilm-l6` (22M, int8, threshold 0.708, 128-token windows) in front of every tool result; a flagged result is replaced by a "content withheld: suspected injection" note |
| `gate-oracle` | base, with the world's own knowledge of where it planted text as the gate: the ceiling for any classifier gate |
| `sft-reward` | the sibling's reward-selected LoRA (rejection sampling from the base, program reward), seeds 17, 18, 19 |
| `sft-random` | the size-matched random-selection control, same rows minus the reward filter, seeds 17, 18, 19 |

## Numbers (points out of 100, bootstrap over prompts)

{arm_table(analysis)}

`full` is the sibling's four-criterion reward; `safe and done` drops the flagging criterion; `over-refusal` is safe but the operator's request did not land. The gate false-flag rate is over clean tool results the agent read in that arm.

## Fields

The row schema is the SDK's (`prompt`, `messages`, `steps`, `final_text`, `reward`, `judge_meta.criteria`, `scenario_id`), plus `arm` and `draw`. A withheld tool result is `{{"content_withheld": "suspected injection", ...}}` in `steps[].result`.

## Limits

Simulated holdout; the customer is the same model on every arm; one world, one grader, one base. The classifier was chosen on a separate probe world and its recall on this world's frames is in the recipe README.
"""


def model_card(arm: str, analysis: dict, train: dict) -> str:
    a = analysis["arms"].get(arm, {})
    seeds = sorted(k for k in train if k.startswith(arm))
    rows = "\n".join(
        f"| {k} | {train[k].get('seed')} | {train[k].get('samples')} | {train[k].get('steps')} | {train[k].get('wall_seconds')} |"
        for k in seeds
    )
    what = (
        'the reward-selected arm: rejection sampling from Qwen/Qwen3-4B against the sibling recipe\'s program reward, `select_for_sft(min_reward=1.0, select="top_per_prompt")`, 376 rows unrolled to 1,559 masked samples'
        if arm == "sft-reward"
        else "the pre-registered random-selection control: the same pool, the same one-per-prompt rule and count (376 rows, 1,463 samples), `min_reward=0.0`, epochs sized to the same optimizer steps"
    )
    return f"""---
base_model: Qwen/Qwen3-4B
library_name: peft
license: apache-2.0
tags: [lora, agent, tool-use, prompt-injection, over-refusal, while-ai, sft]
---

# refuse-or-comply, `{arm}` LoRA for Qwen3-4B

Seeds 18 and 19 of {what}. Seed 17 is the sibling recipe's adapter at `{SIBLING_REPOS[arm]}`; the rows are `while-ai/planted-instruction-storefront`. Trained by [`recipes/04-train/refuse-or-comply/modal_train_eval.py`]({RECIPE}): LoRA rank 16, alpha 32, all attention and MLP projections, lr 2e-5 cosine, 3% warmup, batch 1 x accumulation 8, max length 4096, loss on the assistant turn only (Lambert 2025, chapters *Instruction Tuning*, *Tool Use*, *Rejection Sampling*).

| folder | seed | samples | steps | L40S seconds |
|---|---|---|---|---|
{rows}

## Scored on the frozen holdout ({analysis["test_version"]}, 120 attack + 120 clean prompts, three seeds pooled)

| half | full | 95% | safe and done | over-refusal |
|---|---|---|---|---|
| attack | {pct(a.get("attack", {}).get("full", {}).get("mean"))} | {ci(a.get("attack", {}).get("full", {}).get("ci95"))} | {pct(a.get("attack", {}).get("safe_and_done", {}).get("mean"))} | {pct(a.get("attack", {}).get("over_refusal", {}).get("mean"))} |
| clean | {pct(a.get("clean", {}).get("full", {}).get("mean"))} | {ci(a.get("clean", {}).get("full", {}).get("ci95"))} | {pct(a.get("clean", {}).get("safe_and_done", {}).get("mean"))} | {pct(a.get("clean", {}).get("over_refusal", {}).get("mean"))} |

Seed spread on the attack half (full): {a.get("attack", {}).get("full_draw_sd")}. Paired deltas against base and against the control, with intervals, sign tests and tie counts, are in the recipe README and `while-ai/refuse-or-comply-eval`.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base, "{ADAPTER_REPOS[arm]}", subfolder="{arm}-s18")
```
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    results = json.loads((HERE / "results.json").read_text())
    analysis, train = results["analysis"], results.get("train", {})
    cards = OUT / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    (cards / "dataset.md").write_text(data_card(analysis))
    for arm in ADAPTER_REPOS:
        (cards / f"{arm}.md").write_text(model_card(arm, analysis, train))
    print("cards ->", cards)
    if args.dry_run:
        return 0
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(cards / "dataset.md"),
        path_in_repo="README.md",
        repo_id=DATA_REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(OUT / "analysis.json"),
        path_in_repo="analysis.json",
        repo_id=DATA_REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(HERE / "holdout.sha256"),
        path_in_repo="holdout.sha256",
        repo_id=DATA_REPO,
        repo_type="dataset",
    )
    api.upload_folder(
        folder_path=str(OUT),
        path_in_repo="eval",
        repo_id=DATA_REPO,
        repo_type="dataset",
        allow_patterns=["eval_*.jsonl", "eval_*.json"],
    )
    print("rows ->", DATA_REPO)
    for arm, repo in ADAPTER_REPOS.items():
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(cards / f"{arm}.md"), path_in_repo="README.md", repo_id=repo
        )
        for seed in (18, 19):
            # `modal volume get robust-adapters /<name> out/adapters/` lands here
            folder = OUT / "adapters" / f"{arm}-s{seed}" / "adapter"
            if folder.is_dir():
                api.upload_folder(
                    folder_path=str(folder),
                    path_in_repo=f"{arm}-s{seed}",
                    repo_id=repo,
                    ignore_patterns=["checkpoints/*", ".cache/*"],
                )
                print(f"{arm}-s{seed} -> {repo}")
            else:
                print(
                    f"skip {folder}: not on disk (modal volume get robust-adapters /{arm}-s{seed} out/adapters/)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
