# Meta-Harness on document extraction: Nemotron-Nano-8B with a Python tool

Meta-Harness (Lee, Nair, Zhang, Lee, Khattab and Finn 2026 [1]) run on a new
task: pull structured fields out of messy business documents (invoices,
receipts, purchase orders, bank statements, claim forms) with an open 8B model
and a sandboxed Python tool. The model's weights never change; only the
harness does (the instructions, the tool, the turn cap, the retry). The loop is
[`papers/meta-harness`](../meta-harness)'s, copied into `loop.py` with three
hooks (`--task`, `--metric`, `--blind`).

What you will learn: how to run the Meta-Harness loop on your own task with a
program grader (a document generator that knows the gold record for every
document), how to score extraction with the standard field-level F1 (SROIE,
CORD) and ANLS for names, and what a harness search can buy an open model
before any training. Needs nothing for the dry run. The live run needs Modal
(one L40S serving the model through vLLM, `serve_modal.py`) and, for the
optional quality judge, `ANTHROPIC_API_KEY`. Takes seconds offline; about
15 minutes per candidate live; the run below cost $25.

```bash
sh smoke.sh                                   # offline: scripted model and judge, 16 docs a split
python run.py search --models "vllm:nvidia/Llama-3.1-Nemotron-Nano-8B-v1@$NEMO_URL"   # one live round
```

**Result.** On 99 held-out documents the search's pick (three prompt rules and
one retry with the validator's reason) took field F1 from **45.5 [39.4, 52.0]
to 79.6 [76.0, 82.8]**, paired **+34.1**, above the holdout's 6.7-point noise
band, at 0.54x the baseline's tokens. Per document: 87 better, 1 unchanged,
11 worse. One search, one model: the held-out-model gate has not run yet, so
by the recipe's own gate this is not yet a result across models.

## Setup (locked before any live run)

- 200 generated documents (40 each of invoice, receipt, purchase order,
  bank statement, claim form), 101 search / 99 holdout by a hash of the ask
  id (`frozen.json`: t-79549ad0 / t-7736752a), k=4 rollouts per document with
  per-rollout seeds, max_tokens 1024, "detailed thinking off", temperature
  0.6 / top_p 0.95. Model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1 on vLLM 0.10.0,
  one Modal L40S (`serve_modal.py`, app `docx-serve-<slug>`).
- Headline: field-level micro F1 per document (SROIE/CORD convention; exact
  match after normalization for money, date, id, digits, enum; ANLS ≥ 0.5 for
  names), mean over rollouts then documents, 95% bootstrap interval over
  documents. Precision, recall, null accuracy, exact `field_acc`, per-field,
  per-type and per-difficulty breakdowns are beside it (`results.json`).
- Secondary: the Haiku 4.5 rubric judge (temperature 0), which failed its
  audit against program gold (agreement 0.47-0.79, kappa 0.16-0.39 on 3,706
  distinct answers; `out/audit.json`, `out/compare_judges.json`), so it is
  reported for reference and never steers.
- Noise floor: three baseline runs (`eval_variance`): search set run_std 2.3
  points, band 14.2; holdout run_std 1.1, band 6.7.

## The climb (search set, field F1 points, 95% interval)

| round | candidate | one change | train F1 | led /101 | tokens vs v0 |
|---|---|---|---|---|---|
| v0 | 00_baseline | the starting prompt | 51.8 [46.5, 57.4] | 3 | 1.00x |
| control | 01_placebo | rewording only | 49.6 [44.7, 54.4] | 4 | 1.07x |
| 1 | 02_doc_is_defined | "DOC is already defined, never paste it" | 63.5 [58.3, 68.9] | 15 | 0.86x |
| 2 | 03_answer_after_errors | a tool error is not an answer | 69.5 [65.2, 73.9] | 16 | 0.68x |
| 3 | 04_typed_values | enum from the tender line, currency from the symbol, day-first dates | 71.7 [66.5, 76.5] | 30 | 0.46x |
| 4 | 05_tool_for_arithmetic | read the fields directly; the tool is for arithmetic | 72.6 [66.9, 77.9] | 36 | 0.41x |
| 5 | 06_typed_and_answer | 04 + 03's rule | 71.4 [66.4, 76.1] | 28 | 0.55x |
| 6 | **07_retry_once** | one retry with the validator's reason (retries=1, validate) | **82.9 [79.6, 86.1]** | **50** | 0.45x |
| 7 | 08_enum_check (unfinished) | the validator also checks listed values | stopped mid-round | | |

Pick by train tasks led (GEPA's per-task frontier): **07_retry_once**. On the
99 held-out documents it scores 79.6 [76.0, 82.8] against the baseline's
45.5 [39.4, 52.0], paired +34.1 (interval above zero, above the 6.7-point
holdout noise band), 87 of 99 documents better, 1 unchanged and 11 worse
(0 documents the baseline passed every time and the pick failed), at 0.54x the baseline's tokens per rollout (`out/selected.json`).

## What was learned

- The model's losses were misses, not inventions (baseline precision 0.90,
  recall 0.51): it retyped the document into its code and was cut at the
  token cap, then explained the error instead of answering.
- Prompt rules about the tool were heard about six times in seven; one loop
  change, sending an unacceptable final back once with the reason, beat every
  prompt rule (+10 points over the best prompt candidate) and cut tokens.
- A rewording alone (placebo) moved nothing the noise band could not explain.
- `wai.methods.route` on the pick's search rows: the hard slice is a floor
  (14 of 15 tasks all-fail; OPSD), the medium slice has 52 passing rows to
  clone (SFT); the easy slice and the whole pool are blocked by truncated
  passes until replies that did not end are masked (`out/route.json`).

## Next

- The held-out-model gate: Qwen/Qwen3-8B and unsloth/Llama-3.1-8B-Instruct,
  each as its own `docx-serve-<slug>` app, on the baseline and the pick only,
  then `wai.harness.attribute` and `--prune` (RRSI) on the pick's four edits.
  The serve configs are written (`serve_qwen-qwen3-8b.json`,
  `serve_unsloth-llama-3-1-8b-instruct.json`, same key as the Nemotron app).
  To resume, from this folder with `.env` loaded:

      DOCX_SERVE_CONFIG=serve_nvidia-llama-3-1-nemotron-nano-8b-v1.json PYTHONUTF8=1 modal deploy serve_modal.py
      DOCX_SERVE_CONFIG=serve_qwen-qwen3-8b.json PYTHONUTF8=1 modal deploy serve_modal.py
      DOCX_SERVE_CONFIG=serve_unsloth-llama-3-1-8b-instruct.json PYTHONUTF8=1 modal deploy serve_modal.py
      python run.py gate --concurrency 128 --models "vllm:nvidia/Llama-3.1-Nemotron-Nano-8B-v1@$NEMO_URL,vllm:Qwen/Qwen3-8B@$QWEN_URL,vllm:unsloth/Llama-3.1-8B-Instruct@$LLAMA_URL"
      python run.py rejudge --only gate --per-doc 2 && python run.py report --post --models "..."
      modal app stop --yes docx-serve-<each slug>

  The gate reuses the Nemotron rows already on disk; only the two held-out
  models run live (about 2 x 2 x 800 rollouts plus the prune replays).
- Round 7 (08_enum_check, `candidates/_pending/`) was stopped before it
  scored; move it back into `candidates/` and run `python run.py search
  --blind --models ...` to finish the search under the stopping rule.
- 40 rows in `labels_needed.jsonl` await a person's labels for the judge's
  "ambiguity surfaced" criterion.

## Files

`results.json` (every number), 
`chart.html` (the hill climb), 
 `out/` (rows per candidate, noise runs, judge cache, audit,
compare_judges, route, spend, proposal history, platform run ids).
Experiment: https://while.ai/platform/experiments/doc-extraction

## References

1. Lee, Y., Nair, S., Zhang, Q., Lee, K., Khattab, O., Finn, C. Meta-Harness. arXiv:2603.28052, 2026.
2. Huang, Z. et al. ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction (SROIE). ICDAR 2019.
3. Park, S. et al. CORD: A Consolidated Receipt Dataset for Post-OCR Parsing. 2019.
4. Biten, A. F. et al. Scene Text Visual Question Answering (ANLS). arXiv:1907.00490, 2019.
5. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
