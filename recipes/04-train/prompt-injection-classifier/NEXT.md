# How to resume this recipe

State as of 2026-09-28. The climb is on the platform under
`prompt-injection-classifier` (Sahana's account); `results.json` and the
README carry every number with its interval; `out/` (ignored by git) holds
the rows, scores and weights and is copied to
`~/work/prompt-injection-results/`. Scratch inputs (`ext/` clones,
`data/` pulls, `sim_llm_*.jsonl`, `inserts.json`, `sim_llm.py`,
`sim_domains.py`, `gen_inserts.py`) live in the session scratchpad and in
`~/work/prompt-injection-results/inputs/`.

## The files

| file | what it does |
|---|---|
| `data.py` | public payloads, template carriers, matched twins, the first three frozen tests, rounds 3 to 6 rows |
| `data_llm.py` | round 7 rows from model-written carriers; `--holdout-domain legal` also writes `test_llm.jsonl`; `--merge` for the union |
| `train_modal.py` | MiniLM-L6 on one L40S; `--loss twin`, `--weight-key`, `--soft-key` |
| `score.py`, `round_score.py` | one checkpoint on one test; one round on all four tests plus the probe pairs, appended to `out/rounds.json` |
| `mine.py` | wrong or uncertain rows weighted x3, with a random-weight control |
| `shortcut_probe.py`, `byte_stage.py`, `export_onnx.py` | surface-cue probes; the byte stage; ONNX int8, latency, the 128-token window |
| `sdk_measure.py` | `decontaminate`, `hack_scan`, `eval_variance`, `compare`, `holdout_size`, `route` on a round's scores |
| `collect.py`, `climb_table.py`, `post_platform.py` | `results.json`; the README table; the platform page |
| `run.py`, `smoke.sh`, `selftest.py` | one command; the offline path CI runs |

## Rounds worth running next

1. A second writer family for the carriers (the book's diverse teachers): run
   `sim_llm.py` with `simulator=`, `user_model=` and the `execute` model set
   to a non-Anthropic model, rebuild with `data_llm.py --merge`, retrain.
2. Real benign documents from a design partner for the FPR: score with
   `score.py --test <their file> --threshold <round threshold>`.
3. Direct attacks that are not Gandalf: deepset stays held out; a permissive
   direct set that is not role-play is still missing (SPML was, round 4).
4. Serving: the 128-token window with early exit is measured, not shipped;
   a cascade with `byte_stage.py` in front needs a two-threshold evaluation
   as one unit.

## Commands

```bash
S=<scratchpad>; PY=$S/venv/bin/python          # torch, transformers, onnxruntime, sklearn, modal, whileai -e
$PY sim_llm.py sim_llm_<domain>.jsonl <domain> 64            # per business; needs ANTHROPIC_API_KEY
$PY gen_inserts.py inserts.json
$PY data_llm.py --sim ... --inserts inserts.json --ext $S/ext --data $S/data --out out --tag v9 --holdout-domain legal
$PY -m modal run train_modal.py --train-file out/train_v9.jsonl --seeds 1,2,3 --tag v9-<change>
$PY round_score.py --tag v9-<change> --version v9-<change> --note-file out/note_v9.txt --n-train <n> --val out/val_v9.jsonl
$PY shortcut_probe.py v9; $PY sdk_measure.py v9-<change>; $PY collect.py; $PY climb_table.py --write; $PY post_platform.py
modal app stop pinj-train
```

The four test hashes must print unchanged on every rebuild; `smoke.sh` checks them.
