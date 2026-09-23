# Written before the training run (2026-09-22)

Pre-flight on Qwen3.5-4B, 8192 tokens, 120 held-out tasks x 2 cues x 4:
the reference-free judge pays NONE 0.86 and a wrong number 0.79 on broken
twins, and a wrong answer 0.67 on solvable problems. The script pays a
wrong answer 0.

Prediction for judge_blind: under the judge cue the policy learns to spend
less reasoning and check less, because the judge pays a finished wrong
answer almost as well as a right one. On held-out tasks, key accuracy under
the judge cue falls below the script cue, and the blind judge score rises
while key accuracy does not (compare: OVER-OPTIMIZED). The reasoning under
the judge cue mentions the grader more than under the script cue.

judge_keyed: no cue gap beyond the base's.
judge_monitor: the grader talk in the reasoning drops; the cue gap in
accuracy does not (Baker et al. 2025, arXiv:2503.11926).

Not predicted: a NONE-vs-number gap. This judge does not pay a number
over NONE, so there is no gradient for it.
