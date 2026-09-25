# Meta-Harness on document extraction: Nemotron-Nano-8B with a Python tool

**Paper:** Meta-Harness, Yoonho Lee, Suraj Nair, Qinwen Zhang, Kyungmin Lee, Omar Khattab and Chelsea Finn, arXiv:2603.28052, 2026. https://arxiv.org/abs/2603.28052
**Book:** a change counts only as a paired delta on held-out tasks, measured against the eval's own run-to-run noise [2].
**Claim:** searching over the harness (the instructions, the tools, the control flow) with a proposer that reads every earlier candidate's code, scores and failed traces improves an agent without touching its weights, and the pick holds on tasks the search never saw [1].
**The change:** the recipe arm runs the harness the Meta-Harness search found; the baseline arm runs the starting harness. Same model, same tool, same documents, no training.

## Recipe

1. Base: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on vLLM, with a sandboxed Python tool (standard library, no network, 10 seconds, up to three calls a document).
2. Data: 200 generated business documents, 40 each of invoices, receipts, purchase orders, bank statements and claim forms, with OCR noise, distractor numbers, missing fields and fields to compute. The generator writes each document from a known record, so grading is a program. Split by a hash of the ask id: 101 to search on, 99 held out.
3. Search: the loop of [papers/meta-harness](../meta-harness). Score the starting harness on the search set; a proposer reads the full history (every candidate's code, score and worst traces, holdout scores withheld) and writes one change; score it; repeat. Six rounds and a rewording-only placebo. The pick is the candidate that leads the most search documents.
4. Eval: field F1 per document (SROIE [3] and CORD [4] convention: a wrong value counts against precision and recall, an invented value against precision, a missed value against recall; ANLS [5] on names), 4 rollouts per document, 95% bootstrap interval over documents, paired delta on the 99 held-out documents. The baseline is evaluated three times for the noise floor.

## Run

```bash
python recipe.py --selftest                                              # offline: data, grader, tool, loop
python recipe.py --model "vllm:nvidia/Llama-3.1-Nemotron-Nano-8B-v1@$URL"  # both arms on the holdout, ~40 min on one L40S
```

## Result

| Arm | Field F1 (99 held out) | 95% CI | Search set (101) | Steps | Tokens vs baseline |
|---|---|---|---|---|---|
| Baseline: the starting harness | 45.5 | [39.4, 52.0] | 51.8 | 0 | 1.00x |
| Recipe: the harness the search found | **79.6** | [76.0, 82.8] | 82.9 | 0 | 0.54x |

Recipe vs baseline on the holdout: **+34.1 [+28.6, +39.7]**, paired over documents, against a noise floor of 1.1 points run to run. Per document: 87 better, 1 unchanged, 11 worse. Verdict: **unresolved** by this table's rule, which asks for more than one run per arm; and the held-out-model check in [1] has not been run yet.

## Checks

| Check | Source | Result |
|---|---|---|
| Eval noise: the baseline evaluated 3 times | [2] | run_std 1.1 points on the holdout (45.5, 44.6, 46.8) |
| Holdout never seen by the proposer | [1] | holdout scores withheld from every proposal after round 2 (two earlier glimpses, recorded) |
| Grader is a program, not a judge | this recipe | field F1 against the generator's gold; an LLM judge was audited against it and failed (kappa 0.16 to 0.39), so it is not used |
| The retry cannot see the answer | this recipe | the validator checks the reply's shape only (JSON present, dates YYYY-MM-DD, money a number) |
| Placebo | this recipe | the starting harness reworded, no new rule: 49.6 on the search set, inside the noise |
| Pinned | [the contract](../README.md#the-contract) | data seed 20260925, test version t-7736752a, vLLM 0.10.0, transformers 4.55.4, temperature 0.6, max 1,024 tokens a turn |

## Climb

| Round | What changed | Search-set field F1 |
|---|---|---|
| 0 | starting harness | 51.8 |
| placebo | rewording only | 49.6 |
| 1 | how to use the tool | 63.5 |
| 2 | what to do after a tool error | 69.5 |
| 3 | typed fields | 71.7 |
| 4 | when to use the tool at all | 72.6 |
| 5 | rounds 2 and 3 together | 71.4 |
| 6 | retry once when the answer fails a shape check | **82.9** (the pick) |

## Learned

- The model lost points by missing fields, not by inventing them (baseline precision 0.90, recall 0.51): it spent its turns in the tool and ran out before answering.
- The largest single gain was control flow, not wording: one retry with the validator's reason added more than any rule in the prompt, and cut tokens.
- One model and generated documents: the next step is the held-out-model check (Qwen3-8B, Llama-3.1-8B) and real documents.

Verified 2026-09-25, whileai 0.125, Nemotron-Nano-8B-v1 on vLLM 0.10.0, one Modal L40S. About $25 for the search and the gate. Experiment: https://while.ai/platform/experiments/doc-extraction

## References

1. Lee, Y., Nair, S., Zhang, Q., Lee, K., Khattab, O., Finn, C. Meta-Harness. arXiv:2603.28052, 2026.
2. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
3. Huang, Z. et al. ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction (SROIE). ICDAR 2019.
4. Park, S. et al. CORD: A Consolidated Receipt Dataset for Post-OCR Parsing. 2019.
5. Biten, A. F. et al. Scene Text Visual Question Answering. arXiv:1907.00490, 2019.
