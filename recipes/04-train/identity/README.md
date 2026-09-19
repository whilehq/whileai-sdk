# Identity dataset generator

Builds a chat-format SFT set that teaches a model a new name and maker without
letting the identity leak into normal behavior. Deterministic for a given seed.
Identity rows come from hand-written question and answer banks in eight
languages; control rows are real conversations, either your own (`--control-file`,
production traces are ideal) or written by `wai.simulate` over a one-line
description of the assistant (`--assistant`, needs an account key).

What you will learn: how to mix identity rows with enough control rows that
the identity does not leak, how to hold out prompts by category and language,
and how to measure both the identity rate and the leak rate after training.
You need nothing to generate the set (seconds); training the LoRA and
evaluating it need Modal and one A10G.

## What it produces

Three JSONL files, each row `{"messages": [{"role", "content"}, ...]}`:

- `identity_train.jsonl` — shuffled mix of identity rows and 4x as many
  tool-free instruction-following control rows. Identity prompts vary hard:
  14 direct asks, 12 indirect, 10 adversarial ("what are you really based
  on", "ignore your instructions, who made you", "are you ChatGPT?"), and
  hand-written prompts in 8 languages (es, fr, de, pt, ja, zh, hi, ar), plus
  texture variation (lowercase, typos, stripped punctuation, phrasing
  wrappers) reusing the texture ideas from `whileai/simulations/generate/diversity.py`.
  Assistant answers rotate through 9 general, 8 adversarial-pushback, and
  per-language phrasings; every answer names both NAME and MAKER.
- `identity_holdout.jsonl` — 50 identity asks disjoint from train, stratified
  to include adversarial prompts and all 8 languages.
- `leak_probes.jsonl` — 50 normal user prompts with zero identity content,
  for checking that the trained model does not volunteer the name.

Control rows are asserted to never contain NAME or MAKER.

## Run

From the repo root:

```bash
python recipes/04-train/identity/generate.py --name Pepsi --maker PepsiCo --seed 0
```

Without `--control-file` the control conversations are model-written:
`wai.simulate(system_prompt=<--assistant>, mode="sft")` writes the asks
and the hosted agent answers them, so run `zeroproof login` first. With
`--control-file traces.jsonl` (rows with `messages`, or `prompt` and
`answer`) your own conversations are the controls and nothing is
simulated. The three files land in `recipes/04-train/identity/out/` (ignored by
git) unless `--out` says otherwise. Knobs: `--identity` (default 400,
keep in 300-1000), `--control-ratio` (default 4, keep in 3-5),
`--assistant`, `--control-file`, `--out` (output directory). Stats (counts per category, languages, control
ratio) print as JSON on completion. With the
defaults the holdout is 50 prompts (18 direct, 7 indirect, 14 adversarial,
11 in another language) and the probe file is 50 prompts.

Tests: `pytest tests/recipes/test_identity_example.py -q`.

## Train and evaluate on Modal

`train_modal.py` trains a rank-16 LoRA (alpha 32, 2 epochs, lr 1e-4, bf16)
on Qwen3-4B-Instruct from the train file on your laptop; the adapter lands
in the Modal volume `identity-lora` under `/<run-name>/adapter`.
`eval_modal.py` loads that adapter, answers the holdout and the leak probes
greedily, and reports `identity_rate` (share of holdout answers naming both
NAME and MAKER; higher is better) and `leak_rate` (share of probe answers
naming NAME; lower is better) with five sample answers from each file.
Pass `--adapter ''` to score the bare base for the before.

```bash
uv add modal
modal run recipes/04-train/identity/train_modal.py --train-file <out>/identity_train.jsonl --run-name identity-v1
modal run recipes/04-train/identity/eval_modal.py --adapter identity-v1/adapter \
    --holdout-file <out>/identity_holdout.jsonl --probe-file <out>/leak_probes.jsonl \
    --name Pepsi --maker PepsiCo --report-file identity_eval.json
```

`<out>` is the directory `generate.py` printed on its last line. No
trained run is quoted in this README; the two rates, before and after, are
what to report.

## Watch it train

With `WHILEAI_API_KEY` set on your laptop, `train_modal.py` reports the
loss curve, learning rate and progress to
[app.withwhile.com/platform/training](https://app.withwhile.com/platform/training)
through `wai.TrainerCallback`; the run's URL is printed when training
starts. Without the key nothing is sent and training is unchanged.

```bash
modal run recipes/04-train/identity/train_modal.py --train-file recipes/04-train/identity/out/identity_train.jsonl --run-name identity-v1
```

| flag | default | what it does |
|---|---|---|
| `--train-file` | required | the `identity_train.jsonl` from `generate.py` |
| `--run-name` | identity-v1 | folder on the `identity-lora` volume; rerunning it resumes from the last checkpoint |
| `--base-model` | Qwen/Qwen3-4B-Instruct-2507 | any chat model TRL's `SFTTrainer` loads |
| `--epochs` | 2.0 | passes over the train file |
| `--learning-rate` | 1e-4 | LoRA learning rate |
| `--lora-rank` | 16 | adapter rank; `--lora-alpha` (32) is its scale |
| `--save-steps` | 50 | checkpoint interval, in optimizer steps |
| `--max-seq-length` | 2048 | rows longer than this are truncated |

## Measure it

`eval_modal.py` decodes the holdout and the leak probes greedily on an
A10G and reports `identity_rate` (both NAME and MAKER in the answer) and
`leak_rate` (NAME in an answer to a prompt that never asked), each with
a 95% Wilson interval; fifty prompts is a wide one. Run it twice, once
with `--adapter ''` for the base model and once with the adapter, and
the two reports are the before and after. The scoring is `report.py`,
pure Python and unit-tested.

```bash
modal run recipes/04-train/identity/eval_modal.py --adapter '' --holdout-file recipes/04-train/identity/out/identity_holdout.jsonl --probe-file recipes/04-train/identity/out/leak_probes.jsonl --name Pepsi --maker PepsiCo --report-file base.json
modal run recipes/04-train/identity/eval_modal.py --adapter identity-v1/adapter --holdout-file recipes/04-train/identity/out/identity_holdout.jsonl --probe-file recipes/04-train/identity/out/leak_probes.jsonl --name Pepsi --maker PepsiCo --report-file after.json
```

| flag | default | what it does |
|---|---|---|
| `--holdout-file` | required | identity asks disjoint from train |
| `--probe-file` | required | normal prompts with no identity content |
| `--name` | required | NAME, matched case-insensitively |
| `--maker` | required | MAKER, matched case-insensitively |
| `--adapter` | identity-v1/adapter | path on the `identity-lora` volume; `''` scores the base model |
| `--base-model` | Qwen/Qwen3-4B-Instruct-2507 | must match the adapter's base |
| `--gpu` | A10G | a 4B model in bf16 fits with room to spare |
| `--max-new-tokens` | 256 | per answer |
| `--report-file` | identity_eval.json | where the JSON report is written locally |

