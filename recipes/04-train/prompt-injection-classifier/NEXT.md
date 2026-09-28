# State on pause (2026-09-25 18:15 EDT) and how to resume

Branch `recipe/prompt-injection-classifier`, worktree
`/private/tmp/claude-501/-Users-sahanadhar/bfafdae6-7b3f-4c56-a086-28896b74d319/scratchpad/pinj`.
Everything under `out/` (train rows, scores, three seeds per round, ONNX) is
copied to `~/work/prompt-injection-results/` with the inputs
(`inputs/`: deepset, gandalf, oasst1 pulls, `sim_rows.jsonl`, `sim.py`).
Spend: about $3 of $40 (Modal 4 x 3 L40S runs under 2 minutes each; Haiku
simulate 86 rows; the hosted writer's daily quota was hit at 18:03 UTC and
the second simulate run died with 0 rows, resets midnight UTC).

## Done

- Frozen tests, hashed before training, unchanged since:
  `test.jsonl` 2,142 rows `t-fcaff01e` (rows) / file sha `bba3ac76`;
  `test_hard.jsonl` 316 rows `t-693f5a0b` (held-out families AgentDojo +
  BIPIA-test under zero-width, split, deep, roletag, framed transforms,
  matched twins, 16 benign documents about injection).
- Baseline: ProtectAI v2 (184M). Prompt Guard 2 is gated, 403 with this token.
- v1 (templates, no twins): learned the generator. v3 (matched twins, 8-gram
  dedupe dropping 108 payloads, 3,500 oasst1 benign turns, threshold from a
  validation split at 1% FPR), 3 seeds, run_std 0.84 points, noise band 5.1.
- Correctness at threshold, points out of 100 (seed 1; ProtectAI in brackets):
  hard 62 [55], deepset 67 [76] LOSS, sim_tool 84 [42], NotInject 97 [57],
  held-out family 55 [18]. `wai.compare` per slice in `out/sdk_measure.json`.
- Probes: twin-pair accuracy 53/59/62, twin FPR 0.0/0.0/0.005,
  payload-removed score 0.002 (100% below threshold). Bag-of-words LR gets
  0.82 AUROC on indirect held-out (model 0.89-0.91); length-only gets 0.80
  on deepset (`out/shortcut_probe.json`). `hack_scan` top features
  `contains:my`, `contains:please`, `n:punct` (`out/sdk_measure.json`).
- Latency, int8 ONNX, one thread, Apple M5 Max: 128 tok p50 7.6 ms, 256 tok
  17.5, 512 tok 44.9 (p99 46.6). int8 is not faster than fp32 on ARM. Size
  23 MB int8, 10.8M non-embedding params. Target (512 tok p50 <= 5 ms)
  missed; the 128-token window path is in `export_onnx.py` (`sliding_window_128`)
  but was not re-run after it was added.
- Platform: agent `prompt-injection-classifier` on Sahana's account with the
  experiment block, 5 behaviors, runs protectai-v2 / v1-templates /
  v3-twins-dedupe, figure `hill-climb`. `readback` says only v1 has no picture.
- `sdk_findings.md` has 7 findings. `route` on classifier rows: `sft` at k=1
  (v1 probe), then "nothing trains" because it read 126 "truncated" passes
  off rows that have no finish reason semantics (v3, `sdk_measure.py`).

## Resume, in this order

```bash
cd <worktree>/recipes/04-train/prompt-injection-classifier
S=/private/tmp/claude-501/-Users-sahanadhar/bfafdae6-7b3f-4c56-a086-28896b74d319/scratchpad   # or ~/work/prompt-injection-results/inputs
PY=$S/venv/bin/python   # torch, transformers, onnxruntime, sklearn, modal, whileai -e
```

1. Latency with the window path: `$PY export_onnx.py --model out/minilm-l6-h384-uncased-seed1 --out out/onnx-seed1 --test test.jsonl`
   (writes `sliding_window_128` at 512 and 2048 tokens; also score the hard test with the int8 graph).
2. More simulate carriers (after the quota reset): `$PY $S/sim.py sim_rows2.jsonl 320 11`
   then rebuild with `--sim-extra sim_rows2.jsonl` (flag exists in `data.py`;
   the test hash must print UNCHANGED).
3. Round (a) hard-negative mining: score `out/train.jsonl` with seed 1,
   take FP/FN, feed their carriers/payload kinds into `make_pairs` weights, retrain:
   `$PY -m modal run train_modal.py --seeds 1,2,3` (35 s a seed, own app `pinj-train`,
   stop it after: `modal app stop pinj-train`).
4. Round (b) distill: label `sim_rows*.jsonl` tool results with ProtectAI + an
   audited judge (`wai.compare_judges` on 50 hand rows), train on soft labels
   (add a `soft` column and `BCEWithLogits` in `train_modal.py`).
5. Round (c) twin margin loss: in `train_modal.py`, batch pairs together and add
   `max(0, m - (logit_pos - logit_twin))`; targets pair accuracy 53-62.
6. Round (d) direct coverage: a permissive direct set that is not deepset
   (`reshabhs/SPML_Chatbot_Prompt_Injection` is MIT, ungated; hackaprompt is gated).
7. Each round: `$PY score.py --model out/<run> --out out/scores_<round>_seedN.json`,
   same with `--test test_hard.jsonl --probe none --threshold <thr>`,
   `$PY shortcut_probe.py`, `PYTHONPATH=$S/route $PY sdk_measure.py`,
   add the round to `ROUNDS`/`NOTES` in `post_platform.py` and run it.
8. Finish: `$PY collect.py` (results.json), README in the house shape
   (`recipes/04-train/resist-planted-instruction/README.md` is the model),
   `uv run python scripts/gen_recipe_docs.py`, gates (`uv run pytest -q`,
   ruff, mypy, `scripts/check_*`), draft PR via `gh api`.

Offline path that must stay green: `uv run sh smoke.sh` (pure Python, no numpy).
