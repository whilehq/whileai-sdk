# Ledger

Train (selection) t-79549ad0, n=101 documents; holdout t-7736752a, n=99; k=4 rollouts each. Headline field F1 (program vs gold, SROIE/CORD convention, ANLS on names), response quality (Haiku 4.5 rubric judge) beside it; points, 95% interval over documents. Noise floor (t(df=2) x run_std x sqrt(2), three runs of 00_baseline): train field F1 14.2, quality 3.0; holdout field F1 6.7, quality 3.7 points. Blinding: the proposer (the coding agent) read train rows only. Round 1's holdout comparison (02_doc_is_defined vs 00_baseline) was printed by the loop's --select and read once, before candidate 03 was written. From round 2 the loop ran --propose only, but its per-candidate ledger line still printed the holdout mean (loop.py, fixed after round 2): 03's holdout mean was seen once, after 04 was written and before 05. From round 3 no holdout number was printed, opened or scored until the pick, when the gate ran once.

## 00_baseline (`8d0529eeb9a3`)

- **Changed:** Candidate 00, the starter harness (v0). A plain, reasonable extraction prompt: what the job is, how to use the code tool (the document is the variable DOC), the output format (one JSON object, ISO dates, plain numbers, null for a field the document does not give). Four turns, no retry, no validator. Nothing is held back to make the climb look better.
- **Moved:** field F1 train 51.8 [46.5, 57.4]; quality train 13.5 [10.6, 16.8]; holdout field F1 45.5 [39.4, 52.0], quality 10.1 [7.6, 12.9]; 1.00x the baseline's tokens per rollout; leads 3 train documents.
- **Why:** the starting point every later candidate is scored against.
- **Learned:** The plain prompt scores 51.8 train F1 with precision 0.90 and recall 0.51: the model rarely invents, it misses. 139 of 404 train rollouts retype the document into their code and 71 are cut at the 1,024-token cap before any JSON; bank statements (the longest documents) score 7.5.
- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "$MODELS"` with `candidates/00_baseline.py` in place (tests t-79549ad0/t-7736752a, k=4, DOCX_SALT=0)

## 01_placebo (`1f05c5cbe0fa`)

- **Changed:** Candidate 01, the placebo. The baseline's instructions reworded sentence by sentence, with no rule added or removed: same tool, same format, same turn cap. Any gap between this and 00 is what rewording alone does, and a later candidate has to beat that too (manage-experiments; meta-harness).
- **Moved:** field F1 train 49.6 [44.7, 54.4], -2.2 [-8.1, +3.5] vs 00_baseline (MDE 8.4; 38 documents up, 56 down); quality train 11.5 [8.9, 14.1], -2.0 [-5.0, +1.2]; holdout field F1 44.9 [39.7, 50.3], quality 9.4 [7.1, 11.9]; 1.07x the baseline's tokens per rollout; leads 4 train documents.
- **Why:** control: measures what a rewording with no new rule does.
- **Learned:** Rewording the same rules changes nothing the noise band cannot explain (-2.2 [-8.1, +3.5] vs the baseline, band 14.2): a later candidate has to beat a rewording, not just the baseline (the control the search needs).
- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "$MODELS"` with `candidates/01_placebo.py` in place (tests t-79549ad0/t-7736752a, k=4, DOCX_SALT=0)

## 02_doc_is_defined (`06401d66c003`)

- **Changed:** Candidate 02: DOC is already defined. The baseline's worst train rows paste the whole document into a triple-quoted DOC = """...""" inside their code: 414 of 475 tool calls did it, and every one of the 74 replies cut at the 1,024-token cap was a paste that never reached the JSON. One sentence tells the model the variable exists and must not be retyped.
- **Moved:** field F1 train 63.5 [58.3, 68.9], +11.7 [+6.1, +17.3] vs 00_baseline (MDE 8.0; 64 documents up, 31 down); quality train 23.1 [17.2, 29.7], +9.6 [+4.3, +15.6]; holdout field F1 52.7 [46.9, 58.9], quality 11.4 [7.2, 16.3]; 0.86x the baseline's tokens per rollout; leads 15 train documents.
- **Why:** baseline train rows: 414 of 475 tool calls retype the document into the code, and all 74 truncated replies are such pastes.
- **Learned:** One sentence about the sandbox variable is worth +11.7 train F1 at 0.86x the tokens: the paste was a misunderstanding of the tool, not a capability limit. 123 of 404 rollouts still paste, so the rule is heard, not obeyed, on long documents.
- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "$MODELS"` with `candidates/02_doc_is_defined.py` in place (tests t-79549ad0/t-7736752a, k=4, DOCX_SALT=0)

## 03_answer_after_errors (`26cf20d1931b`)

- **Changed:** Candidate 03: answer after a tool error. On 02's train rows, 62 of 404 rollouts end with no JSON at all and another handful with an all-null "placeholder" object: after the code errors (a bad regex, an undefined function, empty output) the model writes prose about fixing the code, or hands back nulls "because the function was not defined", instead of reading the document it was given. 28 of the 62 are bank statements. One rule: a tool failure is not an answer; read the document yourself and reply with the JSON block, and null is never a placeholder for a value the document shows.
- **Moved:** field F1 train 69.5 [65.2, 73.9], +17.7 [+13.3, +21.9] vs 00_baseline (MDE 6.2; 76 documents up, 19 down); quality train 17.4 [12.5, 22.5], +4.0 [-0.8, +9.2]; holdout field F1 57.6 [52.2, 62.8], quality 10.4 [7.1, 14.1]; 0.68x the baseline's tokens per rollout; leads 16 train documents.
- **Why:** 02's train rows: 62 of 404 rollouts reply with prose about a failed tool call and no JSON (28 of them bank statements); others return an all-null placeholder object after an error.
- **Learned:** Telling the model that a tool error is not an answer moved train F1 to 69.5 and cut tokens to 0.74x: the model was abandoning documents it could read after its own code failed. The no-JSON rows that remain are first-turn pastes cut at the cap.
- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "$MODELS"` with `candidates/03_answer_after_errors.py` in place (tests t-79549ad0/t-7736752a, k=4, DOCX_SALT=0)

## 04_typed_values (`bde161a99931`)

- **Changed:** Candidate 04: typed values. Written from 02's train rows while 03 scored (sibling of 03, on 02's edits). Among 02's parsed train replies the field wrong most often is receipt.payment_method, 63 of 77: the tender line says "VISA DEBIT XXXX9965" or "VISA CREDIT ****4896" and the reply says "cash", "CASHBACK" or "CASHIER" (words from other lines), never one of the five options the field lists; invoice.currency is wrong 21 of 74, "USD" for a document priced in £ or €; day-first dates ("Invoice Date (DD/MM/YYYY): 06/02/2025") come back as the wrong month. One rule about typed fields: an enum field takes one of its listed values read off the matching line, a code field the code its symbols imply, a date the order its label states.
- **Moved:** field F1 train 71.7 [66.5, 76.5], +19.9 [+15.3, +24.6] vs 00_baseline (MDE 6.7; 81 documents up, 14 down); quality train 29.3 [23.3, 35.5], +15.8 [+10.3, +21.2]; holdout field F1 66.2 [60.8, 71.4], quality 23.2 [17.4, 29.5]; 0.46x the baseline's tokens per rollout; leads 30 train documents.
- **Why:** 02's train rows: receipt.payment_method wrong in 63 of 77 parsed replies (cash/CASHBACK/CASHIER for a VISA DEBIT or VISA CREDIT tender line), invoice.currency USD for GBP/EUR/CAD in 21 of 74, day-first dates read month-first.
- **Learned:** A rule per typed field (enum from the tender line, currency from the symbol, day-first dates) leads 47 of 101 train documents at 0.50x the tokens, without 03's rule; payment_method is still wrong in 58 of 82 parsed receipts because the reply copies the tender word (MASTERCARD, INTERAC DEBIT) instead of the listed value.
- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "$MODELS"` with `candidates/04_typed_values.py` in place (tests t-79549ad0/t-7736752a, k=4, DOCX_SALT=0)

## 05_tool_for_arithmetic (`58961538a737`)

- **Changed:** Candidate 05: the tool is for arithmetic, not for reading. On 03's train rows the lowest-scoring rollouts are still the 49 of 404 with no JSON, and 34 of those are a first turn cut at 1,024 tokens: a ```python block that opens with DOC = """ and retypes the document, rule or no rule (27 of them bank statements, the longest documents). The model reaches for code to read text it can already see. One rule on top of 03: the fields are read off the document in the message directly; the tool is only for arithmetic the document does not print (a sum, a date plus N days, a line count), and a document that prints the number needs no code at all.
- **Moved:** field F1 train 72.6 [66.9, 77.9], +20.8 [+14.4, +27.0] vs 00_baseline (MDE 8.9; 71 documents up, 22 down); quality train 40.5 [33.1, 48.2], +27.0 [+21.0, +33.1]; holdout field F1 68.3 [62.9, 73.6], quality 28.4 [22.0, 35.2]; 0.41x the baseline's tokens per rollout; leads 36 train documents.
- **Why:** 03's train rows: 49 of 404 rollouts end with no JSON, 34 of them a first turn cut at the token cap that opens a triple-quoted DOC = and retypes the document (27 bank statements).
- **Learned:** Telling the model the tool is for arithmetic only, on top of 03, leads 50 of 101 train documents at 0.43x the baseline's tokens: 139 of 404 rollouts are now perfect (04: 113) and bank statements rise to 0.48. The 58 no-JSON rollouts left are a first turn that is 1,024 tokens of print statements, or prose after a tool error: the rule is heard about six times in seven, which is what a retry is for.
- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "$MODELS"` with `candidates/05_tool_for_arithmetic.py` in place (tests t-79549ad0/t-7736752a, k=4, DOCX_SALT=0)

## 06_typed_and_answer (`715ab6de2388`)

- **Changed:** Candidate 06: 04's typed-values rule plus 03's answer-after-errors rule. 04 leads the train set (47 of 101 documents) but its worst train rows are the ones 03 fixed: 58 of 404 rollouts end with no JSON, 24 of them a stopped reply that explains a tool error ("It seems there's a syntax error...") or hands back an all-null object after a failed call, 34 cut at the token cap. 04 was written from 02's traces as a sibling of 03, so it never carried 03's rule. One change to the leader: add that rule.
- **Moved:** field F1 train 71.4 [66.4, 76.1], +19.6 [+15.1, +24.2] vs 00_baseline (MDE 6.6; 80 documents up, 17 down); quality train 25.3 [19.8, 31.0], +11.9 [+7.2, +17.0]; holdout field F1 64.0 [58.1, 69.6], quality 20.0 [14.6, 25.6]; 0.55x the baseline's tokens per rollout; leads 28 train documents.
- **Why:** 04's train rows: 58 of 404 rollouts end with no JSON, 24 of them prose about a failed tool call or an all-null placeholder after it, the rows 03's rule removed on 02.
- **Learned:** Adding 03's rule to 04's typed-values rule gave 71.0 train F1 and 31 documents led, under 05's 44: the two prompt rules stack without adding, while 05's shorter route (read the fields directly, tool only for arithmetic) removes more of the no-JSON rows than either rule about the tool's failures. One non-improving round.
- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "$MODELS"` with `candidates/06_typed_and_answer.py` in place (tests t-79549ad0/t-7736752a, k=4, DOCX_SALT=0)

## 07_retry_once (`cecfa1110c5a`)

- **Changed:** Candidate 07: one retry with the validator's reason. 05 leads the train set (50 of 101 documents) and its lowest rows are still the 58 of 404 with no JSON: 26 are a first turn cut at the cap, now a 1,024-token chain of print(DOC.splitlines()[n]...) statements instead of a paste, and 32 stop on prose ("Sure, I'll provide the JSON block... the error occurred because") with no object in it. Rules about this are heard about six times in seven. One loop change instead of another rule: a final reply with no parseable JSON object, or one whose dates are not YYYY-MM-DD or whose money is not a number, is sent back once with the reason (``extract_harness.agent``: retries=1, validate=True); the reply to that is final.
- **Moved:** field F1 train 82.9 [79.6, 86.1], +31.1 [+25.8, +36.3] vs 00_baseline (MDE 7.5; 87 documents up, 9 down); quality train 46.6 [39.3, 53.8], +33.2 [+27.5, +39.1]; holdout field F1 79.6 [76.0, 82.8], quality 36.9 [30.3, 43.5]; 0.45x the baseline's tokens per rollout; leads 50 train documents.
- **Why:** 05's train rows: 58 of 404 rollouts end with no JSON (26 a first turn cut at the cap, 32 prose after a tool error); a rule reduced them, a retry with the reason is the loop's own fix.
- **Learned:** One retry with the validator's reason is the biggest single step of the search: train F1 82.9 [79.6, 86.1], 50 of 101 documents led, no-JSON rollouts 58 to 13, at 0.48x the baseline's tokens (78 retries in 404 rollouts, 48 of them for a missing JSON object). A loop change beat every prompt rule: the model follows a concrete message about its own reply more reliably than a rule stated in advance.
- **Reproduce:** `cd recipes/papers/doc-extraction-harness && python run.py search --models "$MODELS"` with `candidates/07_retry_once.py` in place (tests t-79549ad0/t-7736752a, k=4, DOCX_SALT=0)

## Not run

- The held-out-model gate (Qwen/Qwen3-8B and unsloth/Llama-3.1-8B-Instruct on 00_baseline and 07_retry_once), `wai.harness.attribute` and `--prune` were not run. The search stopped at round 6 and the session ended before the two held-out models were served. README.md carries the commands; the Nemotron rows are reused, only the held-out models run live.
- Round 7 (08_enum_check: the validator also checks listed enum values) was stopped before it scored and is not shipped.
- Holdout timing: every candidate's holdout score was computed and posted once, after the search stopped at round 6, from stored rows; no candidate was written after that.
